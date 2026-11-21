"""Unified diff parser, in-memory patcher, and before/after AST diff analyzer."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass(frozen=True, slots=True)
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[str, ...]
    deleted_lines: tuple[tuple[int, str], ...]
    added_lines: tuple[tuple[int, str], ...]
    insertion_anchors: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RawFilePatch:
    old_path: str
    new_path: str
    hunks: tuple[Hunk, ...]
    is_new: bool = False
    is_deleted: bool = False
    is_renamed: bool = False
    is_binary: bool = False


@dataclass(frozen=True, slots=True)
class AttrChange:
    class_qn: str | None
    attr_name: str
    changed_dict_keys: frozenset[str]
    keys_membership_changed: bool
    whole_attr_changed: bool


@dataclass(frozen=True, slots=True)
class FileAnalysis:
    old_path: str
    new_path: str
    rel_path: str
    is_new: bool
    is_deleted: bool
    is_renamed: bool
    is_binary: bool
    is_python: bool
    syntax_broken: bool = False
    is_ast_noop: bool = False
    docstring_changed: bool = False
    raw_text_changed: bool = False
    changed_old_lines: frozenset[int] = frozenset()
    expanded_old_lines: frozenset[int] = frozenset()
    changed_qualnames: frozenset[str] = frozenset()
    added_methods: tuple[tuple[str, str], ...] = ()
    imports_changed: bool = False
    attr_changes: tuple[AttrChange, ...] = ()
    top_level_code_changed: bool = False
    class_header_changed: frozenset[str] = frozenset()
    hunk_count: int = 0


@dataclass(slots=True)
class CommitAnalysis:
    commit_id: str
    files: list[FileAnalysis] = field(default_factory=list)
    patched_contents: dict[str, str | None] = field(default_factory=dict)


def _unquote_git_path(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        raw = raw[1:-1].encode("utf-8").decode("unicode_escape")
    if raw.startswith(("a/", "b/")):
        return raw[2:]
    return raw


def _parse_diff_git_header(line: str) -> tuple[str, str]:
    rest = line[len("diff --git ") :]
    if rest.startswith('"'):
        m = re.match(r'^"((?:[^"\\]|\\.)*)"\s+"((?:[^"\\]|\\.)*)"$', rest)
        if m:
            return _unquote_git_path(f'"{m.group(1)}"'), _unquote_git_path(f'"{m.group(2)}"')
    parts = rest.split(" ")
    if len(parts) == 2:
        return _unquote_git_path(parts[0]), _unquote_git_path(parts[1])
    mid = len(parts) // 2
    return _unquote_git_path(" ".join(parts[:mid])), _unquote_git_path(" ".join(parts[mid:]))


def parse_unified_diff(patch_text: str) -> list[RawFilePatch]:
    """Parse a git/unified diff into structured file patches."""
    if not patch_text or not patch_text.strip():
        return []

    raw_files: list[RawFilePatch] = []
    cur_old = ""
    cur_new = ""
    cur_hunks: list[Hunk] = []
    is_new = False
    is_deleted = False
    is_renamed = False
    is_binary = False
    in_file = False

    hunk_hdr: tuple[int, int, int, int] | None = None
    hunk_lines: list[str] = []
    deleted_lines: list[tuple[int, str]] = []
    added_lines: list[tuple[int, str]] = []
    anchors: list[int] = []
    old_ln = 0
    new_ln = 0
    last_old_ctx = 0
    pending_add_anchor = False

    def flush_hunk() -> None:
        nonlocal hunk_hdr, hunk_lines, deleted_lines, added_lines, anchors, pending_add_anchor
        if hunk_hdr is not None:
            cur_hunks.append(
                Hunk(
                    old_start=hunk_hdr[0],
                    old_count=hunk_hdr[1],
                    new_start=hunk_hdr[2],
                    new_count=hunk_hdr[3],
                    lines=tuple(hunk_lines),
                    deleted_lines=tuple(deleted_lines),
                    added_lines=tuple(added_lines),
                    insertion_anchors=tuple(sorted(set(a for a in anchors if a > 0))),
                )
            )
        hunk_hdr = None
        hunk_lines = []
        deleted_lines = []
        added_lines = []
        anchors = []
        pending_add_anchor = False

    def flush_file() -> None:
        nonlocal in_file, cur_old, cur_new, cur_hunks, is_new, is_deleted, is_renamed, is_binary
        flush_hunk()
        if in_file:
            raw_files.append(
                RawFilePatch(
                    old_path=cur_old,
                    new_path=cur_new,
                    hunks=tuple(cur_hunks),
                    is_new=is_new,
                    is_deleted=is_deleted,
                    is_renamed=is_renamed,
                    is_binary=is_binary,
                )
            )
        in_file = False
        cur_old = ""
        cur_new = ""
        cur_hunks = []
        is_new = False
        is_deleted = False
        is_renamed = False
        is_binary = False

    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            flush_file()
            in_file = True
            cur_old, cur_new = _parse_diff_git_header(line)
            continue

        if not in_file and line.startswith("--- "):
            in_file = True
            p = line[4:].split("\t")[0].strip()
            if p == "/dev/null":
                is_new = True
                cur_old = ""
            else:
                cur_old = _unquote_git_path(p)
            continue

        if not in_file:
            continue

        if hunk_hdr is None:
            if line.startswith("new file mode"):
                is_new = True
            elif line.startswith("deleted file mode"):
                is_deleted = True
            elif line.startswith("rename from "):
                is_renamed = True
                cur_old = _unquote_git_path(line[len("rename from ") :])
            elif line.startswith("rename to "):
                is_renamed = True
                cur_new = _unquote_git_path(line[len("rename to ") :])
            elif line.startswith("Binary files ") or line == "GIT binary patch":
                is_binary = True
            elif line.startswith("--- "):
                p = line[4:].split("\t")[0].strip()
                if p == "/dev/null":
                    is_new = True
                else:
                    cur_old = _unquote_git_path(p)
            elif line.startswith("+++ "):
                p = line[4:].split("\t")[0].strip()
                if p == "/dev/null":
                    is_deleted = True
                else:
                    cur_new = _unquote_git_path(p)
            elif line.startswith("@@ "):
                m = _HUNK_RE.match(line)
                if m:
                    old_start = int(m.group(1))
                    old_count = int(m.group(2)) if m.group(2) is not None else 1
                    new_start = int(m.group(3))
                    new_count = int(m.group(4)) if m.group(4) is not None else 1
                    hunk_hdr = (old_start, old_count, new_start, new_count)
                    old_ln = old_start
                    new_ln = new_start
                    last_old_ctx = old_start if old_count == 0 else old_start - 1
                    pending_add_anchor = False
            continue

        if line.startswith("@@ "):
            flush_hunk()
            m = _HUNK_RE.match(line)
            if m:
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                new_start = int(m.group(3))
                new_count = int(m.group(4)) if m.group(4) is not None else 1
                hunk_hdr = (old_start, old_count, new_start, new_count)
                old_ln = old_start
                new_ln = new_start
                last_old_ctx = old_start if old_count == 0 else old_start - 1
                pending_add_anchor = False
            continue

        if line.startswith("--- ") and not hunk_lines:
            continue

        if line.startswith("\\"):
            hunk_lines.append(line)
        elif line.startswith("-"):
            hunk_lines.append(line)
            deleted_lines.append((old_ln, line[1:]))
            old_ln += 1
            pending_add_anchor = False
        elif line.startswith("+"):
            hunk_lines.append(line)
            added_lines.append((new_ln, line[1:]))
            if last_old_ctx > 0:
                anchors.append(last_old_ctx)
            pending_add_anchor = True
            new_ln += 1
        elif line.startswith(" ") or line == "":
            hunk_lines.append(line if line.startswith(" ") else " ")
            if pending_add_anchor:
                anchors.append(old_ln)
                pending_add_anchor = False
            last_old_ctx = old_ln
            old_ln += 1
            new_ln += 1

    flush_file()
    return raw_files


def apply_hunks(base_text: str, hunks: tuple[Hunk, ...]) -> str:
    """Apply parsed unified diff hunks in memory against base_text."""
    if not hunks:
        return base_text
    src = base_text.splitlines(keepends=True)
    out: list[str] = []
    idx = 0
    for h in hunks:
        start = h.old_start if h.old_count == 0 else max(0, h.old_start - 1)
        while idx < start and idx < len(src):
            out.append(src[idx])
            idx += 1
        last_kind = ""
        for line in h.lines:
            if line.startswith("\\"):
                if last_kind == "+" and out and out[-1].endswith("\n"):
                    out[-1] = out[-1][:-1]
                continue
            if line.startswith("-"):
                idx += 1
                last_kind = "-"
            elif line.startswith("+"):
                out.append(line[1:] + "\n")
                last_kind = "+"
            else:
                if idx < len(src):
                    out.append(src[idx])
                    idx += 1
                else:
                    out.append(line[1:] + "\n")
                last_kind = " "
    while idx < len(src):
        out.append(src[idx])
        idx += 1
    return "".join(out)


def _is_docstring_stmt(stmt: ast.AST) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _body_without_doc(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and _is_docstring_stmt(body[0]):
        return body[1:]
    return body


_ENUM_BASES = frozenset({"Enum", "IntEnum", "StrEnum", "AutoName"})
_UNSAFE_EQ_CLASSES = frozenset({"Expr", "Expression", "Dialect", "_Dialect", "reverse_key"})


def _resolve_imported_file(
    tree: ast.Module, rel_path: str, symbol: str, sqlglot_root: Path
) -> tuple[str, str, ast.Module] | None:
    pkg_parts = rel_path.split("/")[:-1]
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            asname = alias.asname or alias.name
            if asname != symbol:
                continue
            base_mod = ""
            if node.level > 0:
                up = node.level - 1
                if up <= len(pkg_parts):
                    prefix = ".".join(pkg_parts[: len(pkg_parts) - up])
                    base_mod = (
                        f"{prefix}.{node.module}"
                        if (prefix and node.module)
                        else (prefix or node.module or "")
                    )
            elif node.module:
                base_mod = node.module
            if not base_mod:
                continue
            for cand_rel in (
                f"{base_mod.replace('.', '/')}.py",
                f"{base_mod.replace('.', '/')}/__init__.py",
                f"{base_mod.replace('.', '/')}/{alias.name}.py",
            ):
                loaded = _load_base_python(sqlglot_root, cand_rel)
                if loaded is not None:
                    return cand_rel, alias.name, loaded[1]
    return None


def _find_class_node(
    tree: ast.Module, rel_path: str, cls_name: str, sqlglot_root: Path
) -> tuple[str, ast.ClassDef, ast.Module] | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            return rel_path, node, tree
    imp = _resolve_imported_file(tree, rel_path, cls_name, sqlglot_root)
    if imp is not None:
        t_rel, orig_name, t_tree = imp
        for node in ast.walk(t_tree):
            if isinstance(node, ast.ClassDef) and node.name == orig_name:
                return t_rel, node, t_tree
    return None


def _class_defines_custom_eq(cnode: ast.ClassDef) -> bool:
    return any(
        isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name in ("__eq__", "__ne__")
        for s in cnode.body
    )


def _is_safe_eq_class(
    cls_name: str,
    tree: ast.Module,
    rel_path: str,
    sqlglot_root: Path,
    patched_trees: dict[str, ast.Module],
    depth: int = 0,
) -> bool:
    if depth > 4 or cls_name in _UNSAFE_EQ_CLASSES:
        return False
    found = _find_class_node(tree, rel_path, cls_name, sqlglot_root)
    if found is None:
        return False
    c_rel, cnode, c_tree = found
    if c_rel in patched_trees and patched_trees[c_rel] is not c_tree:
        p_found = _find_class_node(patched_trees[c_rel], c_rel, cnode.name, sqlglot_root)
        if p_found is None or _class_defines_custom_eq(p_found[1]) or p_found[1].keywords:
            return False
    if cnode.name in _UNSAFE_EQ_CLASSES or cnode.keywords or _class_defines_custom_eq(cnode):
        return False
    for base in cnode.bases:
        bname = (
            base.id
            if isinstance(base, ast.Name)
            else (base.attr if isinstance(base, ast.Attribute) else "")
        )
        if bname == "object":
            continue
        if not bname or not _is_safe_eq_class(
            bname, c_tree, c_rel, sqlglot_root, patched_trees, depth + 1
        ):
            return False
    return True


def _get_safe_enum_members(
    enum_name: str,
    tree: ast.Module,
    rel_path: str,
    sqlglot_root: Path,
    patched_trees: dict[str, ast.Module],
) -> set[str] | None:
    found = _find_class_node(tree, rel_path, enum_name, sqlglot_root)
    if found is None:
        return None
    c_rel, cnode, _ = found
    if c_rel in patched_trees:
        p_found = _find_class_node(patched_trees[c_rel], c_rel, cnode.name, sqlglot_root)
        if p_found is None or not _ast_equal(cnode, p_found[1]):
            return None
    if _class_defines_custom_eq(cnode):
        return None
    bnames = {
        b.id if isinstance(b, ast.Name) else (b.attr if isinstance(b, ast.Attribute) else "")
        for b in cnode.bases
    }
    if not (bnames & _ENUM_BASES):
        return None
    members = {
        t.id
        for s in cnode.body
        if isinstance(s, ast.Assign)
        for t in s.targets
        if isinstance(t, ast.Name)
    }
    return members if members else None


@dataclass
class _EquivContext:
    rel_path: str
    sqlglot_root: Path
    before_tree: ast.Module
    after_tree: ast.Module
    b_cls: ast.ClassDef | None
    a_cls: ast.ClassDef | None
    b_func: ast.FunctionDef | ast.AsyncFunctionDef | None
    a_func: ast.FunctionDef | ast.AsyncFunctionDef | None
    patched_trees: dict[str, ast.Module]
    ext_mutated_attrs: set[str]


def _resolve_self_attr_name(
    expr: ast.AST,
    b_func: ast.FunctionDef | ast.AsyncFunctionDef | None,
    a_func: ast.FunctionDef | ast.AsyncFunctionDef | None,
) -> str | None:
    if (
        isinstance(expr, ast.Attribute)
        and isinstance(expr.value, ast.Name)
        and expr.value.id == "self"
    ):
        return expr.attr
    if isinstance(expr, ast.Name) and b_func is not None and a_func is not None:
        var = expr.id
        for fn in (b_func, a_func):
            param_names = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
            if fn.args.vararg:
                param_names.add(fn.args.vararg.arg)
            if fn.args.kwarg:
                param_names.add(fn.args.kwarg.arg)
            if var in param_names:
                return None
            assign_vals: list[ast.AST] = []
            for sub in ast.walk(fn):
                if isinstance(sub, ast.Assign):
                    for t in sub.targets:
                        if isinstance(t, ast.Name) and t.id == var:
                            assign_vals.append(sub.value)
                        elif any(
                            isinstance(n, ast.Name) and n.id == var and isinstance(n.ctx, ast.Store)
                            for n in ast.walk(t)
                        ):
                            return None
                elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
                    if any(
                        isinstance(n, ast.Name) and n.id == var and isinstance(n.ctx, ast.Store)
                        for n in ast.walk(sub.target)
                    ):
                        return None
                elif isinstance(sub, ast.NamedExpr) and sub.target.id == var:
                    return None
            if len(assign_vals) != 1:
                return None
            v0 = assign_vals[0]
            if not (
                isinstance(v0, ast.Attribute)
                and isinstance(v0.value, ast.Name)
                and v0.value.id == "self"
            ):
                return None
        assert isinstance(assign_vals[0], ast.Attribute)
        return assign_vals[0].attr
    return None


def _collect_self_attr_stores(
    cnode: ast.ClassDef, attr_name: str
) -> tuple[list[ast.AST], list[ast.AST]]:
    anns: list[ast.AST] = []
    vals: list[ast.AST] = []
    for sub in ast.walk(cnode):
        if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Attribute):
            if (
                isinstance(sub.target.value, ast.Name)
                and sub.target.value.id == "self"
                and sub.target.attr == attr_name
            ):
                anns.append(sub.annotation)
                if sub.value is not None:
                    vals.append(sub.value)
        elif isinstance(sub, ast.Assign):
            for t in sub.targets:
                if (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                    and t.attr == attr_name
                ):
                    vals.append(sub.value)
    return anns, vals


def _extract_optional_type_name(ann: ast.AST) -> str | None:
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        if (
            isinstance(ann.right, ast.Constant)
            and ann.right.value is None
            and isinstance(ann.left, ast.Name)
        ):
            return ann.left.id
        if (
            isinstance(ann.left, ast.Constant)
            and ann.left.value is None
            and isinstance(ann.right, ast.Name)
        ):
            return ann.right.id
    if isinstance(ann, ast.Subscript) and isinstance(ann.slice, ast.Name):
        fn = ann.value
        if (isinstance(fn, ast.Name) and fn.id == "Optional") or (
            isinstance(fn, ast.Attribute) and fn.attr == "Optional"
        ):
            return ann.slice.id
    return None


def _is_sound_compare_equiv(ca: ast.Compare, cb: ast.Compare, ctx: _EquivContext) -> bool:
    if (
        ctx.b_cls is None
        or ctx.a_cls is None
        or len(ca.ops) != 1
        or len(cb.ops) != 1
        or len(ca.comparators) != 1
        or len(cb.comparators) != 1
    ):
        return False
    op_pair = {type(ca.ops[0]), type(cb.ops[0])}
    if op_pair not in ({ast.Is, ast.Eq}, {ast.IsNot, ast.NotEq}):
        return False
    if not _ast_equal(ca.left, cb.left) or not _ast_equal(ca.comparators[0], cb.comparators[0]):
        return False

    lhs, rhs = ca.left, ca.comparators[0]
    if isinstance(lhs, ast.Constant) and lhs.value is None:
        lhs, rhs = rhs, lhs

    attr_name = _resolve_self_attr_name(lhs, ctx.b_func, ctx.a_func)
    if attr_name is None or attr_name in ctx.ext_mutated_attrs:
        return False

    b_anns, b_vals = _collect_self_attr_stores(ctx.b_cls, attr_name)
    a_anns, a_vals = _collect_self_attr_stores(ctx.a_cls, attr_name)
    if not b_vals or not _ast_equal(b_anns, a_anns) or not _ast_equal(b_vals, a_vals):
        return False

    # Case 1: self.<attr> is None <-> self.<attr> == None
    if isinstance(rhs, ast.Constant) and rhs.value is None:
        if len(b_anns) != 1:
            return False
        t_name = _extract_optional_type_name(b_anns[0])
        if not t_name:
            return False
        # Reject if any assignment assigns a direct bare Name parameter that could be untyped
        if any(isinstance(v, ast.Name) for v in b_vals):
            return False
        return _is_safe_eq_class(
            t_name,
            ctx.before_tree,
            ctx.rel_path,
            ctx.sqlglot_root,
            ctx.patched_trees,
        )

    # Case 2: self.<attr> is EnumCls.MEMBER <-> self.<attr> == EnumCls.MEMBER
    if isinstance(rhs, ast.Attribute) and isinstance(rhs.value, ast.Name):
        enum_name = rhs.value.id
        member_name = rhs.attr
        members = _get_safe_enum_members(
            enum_name,
            ctx.before_tree,
            ctx.rel_path,
            ctx.sqlglot_root,
            ctx.patched_trees,
        )
        if members is None or member_name not in members:
            return False
        cls_attrs: dict[str, ast.AST] = {}
        for s in ctx.b_cls.body:
            if isinstance(s, ast.Assign) and s.value is not None:
                for t in s.targets:
                    if isinstance(t, ast.Name):
                        cls_attrs[t.id] = s.value
            elif (
                isinstance(s, ast.AnnAssign)
                and isinstance(s.target, ast.Name)
                and s.value is not None
            ):
                cls_attrs[s.target.id] = s.value
        for v in b_vals:
            if (
                isinstance(v, ast.Call)
                and isinstance(v.func, ast.Name)
                and v.func.id == enum_name
            ):
                continue
            if (
                isinstance(v, ast.Attribute)
                and isinstance(v.value, ast.Name)
                and v.value.id == enum_name
                and v.attr in members
            ):
                continue
            if (
                isinstance(v, ast.Attribute)
                and isinstance(v.value, ast.Name)
                and v.value.id == "self"
            ):
                cv = cls_attrs.get(v.attr)
                if (
                    isinstance(cv, ast.Attribute)
                    and isinstance(cv.value, ast.Name)
                    and cv.value.id == enum_name
                    and cv.attr in members
                ):
                    continue
            return False
        return True

    return False


def _ast_equal(
    a: object,
    b: object,
    *,
    ignore_docstrings: bool = False,
    equiv_ctx: _EquivContext | None = None,
) -> bool:
    """Fast structural AST equality that short-circuits on the first mismatch."""
    if type(a) is not type(b):
        return False
    if isinstance(a, ast.AST):
        assert isinstance(b, ast.AST)
        if (
            equiv_ctx is not None
            and isinstance(a, ast.Compare)
            and isinstance(b, ast.Compare)
            and _is_sound_compare_equiv(a, b, equiv_ctx)
        ):
            return True
        strip_doc = ignore_docstrings and isinstance(
            a, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        for field_name in a._fields:
            va = getattr(a, field_name)
            vb = getattr(b, field_name)
            if strip_doc and field_name == "body":
                va = _body_without_doc(va)
                vb = _body_without_doc(vb)
            if not _ast_equal(
                va, vb, ignore_docstrings=ignore_docstrings, equiv_ctx=equiv_ctx
            ):
                return False
        return True
    if isinstance(a, list):
        assert isinstance(b, list)
        if len(a) != len(b):
            return False
        return all(
            _ast_equal(x, y, ignore_docstrings=ignore_docstrings, equiv_ctx=equiv_ctx)
            for x, y in zip(a, b)
        )
    return a == b


def _statement_header_lines(tree: ast.Module, target_lines: set[int]) -> set[int]:
    """Expand changed lines across enclosing statement and condition header spans."""
    if not target_lines:
        return set()
    expanded = set(target_lines)
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While)):
            test_start = node.test.lineno
            test_end = node.test.end_lineno or test_start
            if any(test_start <= ln <= test_end for ln in target_lines):
                expanded.update(range(node.lineno, test_end + 1))
        elif isinstance(node, ast.For):
            iter_end = node.iter.end_lineno or node.lineno
            if any(node.lineno <= ln <= iter_end for ln in target_lines):
                expanded.update(range(node.lineno, iter_end + 1))
        elif isinstance(
            node,
            (
                ast.Return,
                ast.Assign,
                ast.AnnAssign,
                ast.AugAssign,
                ast.Expr,
                ast.Assert,
                ast.Raise,
            ),
        ):
            end = node.end_lineno or node.lineno
            if any(node.lineno <= ln <= end for ln in target_lines):
                expanded.update(range(node.lineno, end + 1))
    return expanded


def _expr_str_key(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _node_start_line(node: ast.AST) -> int:
    decs = getattr(node, "decorator_list", None)
    if decs:
        return min([node.lineno, *(d.lineno for d in decs)])  # type: ignore[attr-defined]
    return node.lineno  # type: ignore[attr-defined]


def _extract_dict_nodes(value: ast.AST) -> tuple[dict[str, ast.AST], list[tuple[ast.AST | None, ast.AST]]] | None:
    if not isinstance(value, ast.Dict):
        return None
    literal_keys: dict[str, ast.AST] = {}
    other_items: list[tuple[ast.AST | None, ast.AST]] = []
    for k_node, v_node in zip(value.keys, value.values):
        k_str = _expr_str_key(k_node)
        if k_str is not None:
            literal_keys[k_str] = v_node
        else:
            other_items.append((k_node, v_node))
    return literal_keys, other_items


@dataclass
class _TreeSummary:
    funcs: dict[str, tuple[int, int, ast.AST]]
    classes: dict[str, ast.ClassDef]
    class_methods: dict[str, set[str]]
    class_headers: dict[str, tuple[list[ast.expr], list[ast.keyword], list[ast.expr]]]
    attrs: dict[tuple[str | None, str], tuple[int, int, ast.AST]]
    imports: list[ast.stmt]
    other_top_level: list[ast.stmt]
    other_class_stmts: dict[str, list[ast.stmt]]


def _summarize_tree(tree: ast.Module) -> _TreeSummary:
    funcs: dict[str, tuple[int, int, ast.AST]] = {}
    classes: dict[str, ast.ClassDef] = {}
    class_methods: dict[str, set[str]] = {}
    class_headers: dict[str, tuple[list[ast.expr], list[ast.keyword], list[ast.expr]]] = {}
    attrs: dict[tuple[str | None, str], tuple[int, int, ast.AST]] = {}
    imports: list[ast.stmt] = []
    other_top_level: list[ast.stmt] = []
    other_class_stmts: dict[str, list[ast.stmt]] = {}

    def record_assign(stmt: ast.Assign | ast.AnnAssign, class_qn: str | None) -> bool:
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        if stmt.value is None or not all(isinstance(t, ast.Name) for t in targets):
            return False
        start = stmt.lineno
        end = stmt.end_lineno or stmt.lineno
        for tgt in targets:
            assert isinstance(tgt, ast.Name)
            attrs[(class_qn, tgt.id)] = (start, end, stmt.value)
        return True

    def visit_block(stmts: list[ast.stmt], scope: list[tuple[str, str]]) -> None:
        is_module = not scope
        is_class = bool(scope and scope[-1][1] == "class")
        class_qn = ".".join(s[0] for s in scope) if is_class else None

        for stmt in stmts:
            if isinstance(stmt, ast.ClassDef):
                qn = ".".join([*(s[0] for s in scope), stmt.name])
                classes[qn] = stmt
                class_methods.setdefault(qn, set())
                class_headers[qn] = (stmt.bases, stmt.keywords, stmt.decorator_list)
                other_class_stmts.setdefault(qn, [])
                visit_block(stmt.body, [*scope, (stmt.name, "class")])
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qn = ".".join([*(s[0] for s in scope), stmt.name])
                start = _node_start_line(stmt)
                end = stmt.end_lineno or stmt.lineno
                funcs[qn] = (start, end, stmt)
                if class_qn is not None:
                    class_methods.setdefault(class_qn, set()).add(stmt.name)
                visit_block(stmt.body, [*scope, (stmt.name, "func")])
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                imports.append(stmt)
            elif (is_module or is_class) and isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                if not record_assign(stmt, class_qn):
                    if is_module:
                        other_top_level.append(stmt)
                    elif class_qn is not None:
                        other_class_stmts.setdefault(class_qn, []).append(stmt)
            elif is_module:
                if not _is_docstring_stmt(stmt):
                    other_top_level.append(stmt)
            elif is_class and class_qn is not None:
                if not _is_docstring_stmt(stmt):
                    other_class_stmts.setdefault(class_qn, []).append(stmt)

    visit_block(tree.body, [])
    return _TreeSummary(
        funcs=funcs,
        classes=classes,
        class_methods=class_methods,
        class_headers=class_headers,
        attrs=attrs,
        imports=imports,
        other_top_level=other_top_level,
        other_class_stmts=other_class_stmts,
    )


def _overlaps_ranges(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    for rs, re_ in ranges:
        if start <= re_ and rs <= end:
            return True
    return False


def _diff_attr_map(
    before_map: dict[tuple[str | None, str], tuple[int, int, ast.AST]],
    after_map: dict[tuple[str | None, str], tuple[int, int, ast.AST]],
    old_ranges: list[tuple[int, int]],
    new_ranges: list[tuple[int, int]],
) -> list[AttrChange]:
    changes: list[AttrChange] = []
    for key in sorted(set(before_map) | set(after_map), key=lambda k: (k[0] or "", k[1])):
        b = before_map.get(key)
        a = after_map.get(key)
        if b is not None and a is not None:
            if not _overlaps_ranges(b[0], b[1], old_ranges) and not _overlaps_ranges(
                a[0], a[1], new_ranges
            ):
                continue
            if _ast_equal(b[2], a[2]):
                continue
        cls_qn, attr_name = key
        b_dict = _extract_dict_nodes(b[2]) if b is not None else None
        a_dict = _extract_dict_nodes(a[2]) if a is not None else None
        if b_dict is None or a_dict is None:
            changes.append(
                AttrChange(
                    class_qn=cls_qn,
                    attr_name=attr_name,
                    changed_dict_keys=frozenset(),
                    keys_membership_changed=True,
                    whole_attr_changed=True,
                )
            )
            continue
        b_lit, b_other = b_dict
        a_lit, a_other = a_dict
        if len(b_other) != len(a_other) or any(
            not (_ast_equal(bk, ak) and _ast_equal(bv, av))
            for (bk, bv), (ak, av) in zip(b_other, a_other)
        ):
            changes.append(
                AttrChange(
                    class_qn=cls_qn,
                    attr_name=attr_name,
                    changed_dict_keys=frozenset(),
                    keys_membership_changed=True,
                    whole_attr_changed=True,
                )
            )
            continue
        b_keys = set(b_lit)
        a_keys = set(a_lit)
        membership_changed = b_keys != a_keys
        diff_keys = (b_keys ^ a_keys) | {
            k for k in (b_keys & a_keys) if not _ast_equal(b_lit[k], a_lit[k])
        }
        changes.append(
            AttrChange(
                class_qn=cls_qn,
                attr_name=attr_name,
                changed_dict_keys=frozenset(diff_keys),
                keys_membership_changed=membership_changed,
                whole_attr_changed=not diff_keys,
            )
        )
    return changes


def _enclosing_func_for_line(
    funcs: dict[str, tuple[int, int, ast.AST]], ln: int
) -> str | None:
    best_qn: str | None = None
    best_span = 10**9
    for qn, (start, end, _) in funcs.items():
        if start <= ln <= end:
            span = end - start
            if span < best_span:
                best_span = span
                best_qn = qn
    return best_qn


_BASE_PARSE_CACHE: dict[tuple[str, str], tuple[str, ast.Module, _TreeSummary]] = {}


def _load_base_python(sqlglot_root: Path, rel_path: str) -> tuple[str, ast.Module, _TreeSummary] | None:
    key = (str(sqlglot_root), rel_path)
    cached = _BASE_PARSE_CACHE.get(key)
    if cached is not None:
        return cached
    path = sqlglot_root / rel_path
    if not path.is_file():
        return None
    src = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src, filename=rel_path)
    summary = _summarize_tree(tree)
    entry = (src, tree, summary)
    _BASE_PARSE_CACHE[key] = entry
    return entry


def _make_file_analysis(
    rf: RawFilePatch,
    rel_path: str,
    is_python: bool,
    **overrides: object,
) -> FileAnalysis:
    kwargs: dict[str, object] = {
        "old_path": rf.old_path,
        "new_path": rf.new_path,
        "rel_path": rel_path,
        "is_new": rf.is_new,
        "is_deleted": rf.is_deleted,
        "is_renamed": rf.is_renamed,
        "is_binary": rf.is_binary,
        "is_python": is_python,
        "hunk_count": len(rf.hunks),
    }
    kwargs.update(overrides)
    return FileAnalysis(**kwargs)  # type: ignore[arg-type]


def _collect_ext_mutated_attrs(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                for sub in ast.walk(tgt):
                    if isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Store):
                        if not (
                            isinstance(sub.value, ast.Name)
                            and sub.value.id in ("self", "cls", "klass")
                        ):
                            out.add(sub.attr)
        elif isinstance(node, ast.Delete):
            for tgt in node.targets:
                if isinstance(tgt, ast.Attribute):
                    out.add(tgt.attr)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("setattr", "delattr")
        ):
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                out.add(node.args[1].value)
    return out


def analyze_patch(commit_id: str, patch_text: str, sqlglot_root: Path) -> CommitAnalysis:
    """Parse a commit patch, apply hunks in memory, and analyze AST changes per file."""
    result = CommitAnalysis(commit_id=commit_id)
    raw_files = parse_unified_diff(patch_text)

    patched_trees: dict[str, ast.Module] = {}
    ext_mutated_attrs: set[str] = set()
    for rf in raw_files:
        if rf.is_binary or rf.is_deleted:
            continue
        rel_p = rf.new_path or rf.old_path
        if not (rel_p.endswith(".py") or rf.old_path.endswith(".py")):
            continue
        b_loaded = None if rf.is_new or not rf.old_path else _load_base_python(sqlglot_root, rf.old_path)
        b_src = b_loaded[0] if b_loaded is not None else ""
        a_src = apply_hunks(b_src, rf.hunks)
        try:
            a_tree = ast.parse(a_src, filename=rel_p)
            patched_trees[rel_p] = a_tree
            ext_mutated_attrs.update(_collect_ext_mutated_attrs(a_tree))
            if b_loaded is not None:
                ext_mutated_attrs.update(_collect_ext_mutated_attrs(b_loaded[1]))
        except (SyntaxError, ValueError):
            pass

    for rf in raw_files:
        rel_path = rf.old_path if rf.is_deleted else (rf.new_path or rf.old_path)
        is_python = rel_path.endswith(".py") or rf.old_path.endswith(".py")

        if rf.is_binary:
            result.files.append(_make_file_analysis(rf, rel_path, is_python, raw_text_changed=True))
            continue

        if rf.is_deleted:
            result.patched_contents[rf.old_path] = None
            result.files.append(
                _make_file_analysis(rf, rel_path, is_python, is_new=False, raw_text_changed=True)
            )
            continue

        if not is_python:
            before_path = sqlglot_root / rf.old_path if rf.old_path else None
            before_src = (
                before_path.read_text(encoding="utf-8", errors="replace")
                if before_path and not rf.is_new and before_path.is_file()
                else ""
            )
            after_src = apply_hunks(before_src, rf.hunks)
            if rf.is_renamed and rf.old_path and rf.old_path != rf.new_path:
                result.patched_contents[rf.old_path] = None
            result.patched_contents[rel_path] = after_src
            result.files.append(
                _make_file_analysis(
                    rf,
                    rel_path,
                    False,
                    raw_text_changed=(before_src != after_src or rf.is_new or rf.is_renamed),
                )
            )
            continue

        base_loaded = None if rf.is_new or not rf.old_path else _load_base_python(sqlglot_root, rf.old_path)
        before_src = base_loaded[0] if base_loaded is not None else ""
        after_src = apply_hunks(before_src, rf.hunks)
        if rf.is_renamed and rf.old_path and rf.old_path != rf.new_path:
            result.patched_contents[rf.old_path] = None
        result.patched_contents[rel_path] = after_src

        try:
            after_tree = ast.parse(after_src, filename=rel_path)
        except (SyntaxError, ValueError):
            result.files.append(
                _make_file_analysis(rf, rel_path, True, syntax_broken=True, raw_text_changed=True)
            )
            continue

        if rf.is_new or base_loaded is None:
            result.files.append(
                _make_file_analysis(rf, rel_path, True, is_new=True, raw_text_changed=True)
            )
            continue

        _, before_tree, before_sum = base_loaded
        raw_changed = before_src != after_src

        if not rf.is_renamed:
            if _ast_equal(before_tree, after_tree, ignore_docstrings=False):
                result.files.append(
                    _make_file_analysis(
                        rf, rel_path, True, is_ast_noop=True, raw_text_changed=raw_changed
                    )
                )
                continue

            if _ast_equal(before_tree, after_tree, ignore_docstrings=True):
                result.files.append(
                    _make_file_analysis(
                        rf,
                        rel_path,
                        True,
                        is_ast_noop=True,
                        docstring_changed=True,
                        raw_text_changed=raw_changed,
                    )
                )
                continue

        after_sum = _summarize_tree(after_tree)
        old_ranges = [
            (max(1, h.old_start), max(1, h.old_start + max(1, h.old_count) - 1)) for h in rf.hunks
        ]
        new_ranges = [
            (max(1, h.new_start), max(1, h.new_start + max(1, h.new_count) - 1)) for h in rf.hunks
        ]

        changed_qualnames: set[str] = set()
        for qn, (bs, be, b_node) in before_sum.funcs.items():
            a_info = after_sum.funcs.get(qn)
            if a_info is None:
                changed_qualnames.add(qn)
                continue
            as_, ae, a_node = a_info
            if not _overlaps_ranges(bs, be, old_ranges) and not _overlaps_ranges(
                as_, ae, new_ranges
            ):
                continue
            cls_qn = qn.rsplit(".", 1)[0] if "." in qn else None
            b_cls = before_sum.classes.get(cls_qn) if cls_qn else None
            a_cls = after_sum.classes.get(cls_qn) if cls_qn else None
            equiv_ctx = (
                _EquivContext(
                    rel_path=rel_path,
                    sqlglot_root=sqlglot_root,
                    before_tree=before_tree,
                    after_tree=after_tree,
                    b_cls=b_cls,
                    a_cls=a_cls,
                    b_func=b_node if isinstance(b_node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None,
                    a_func=a_node if isinstance(a_node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None,
                    patched_trees=patched_trees,
                    ext_mutated_attrs=ext_mutated_attrs,
                )
                if b_cls is not None and a_cls is not None and not rf.is_renamed
                else None
            )
            if not _ast_equal(b_node, a_node, equiv_ctx=equiv_ctx):
                changed_qualnames.add(qn)

        added_methods: list[tuple[str, str]] = []
        for cls_qn, a_meths in after_sum.class_methods.items():
            b_meths = before_sum.class_methods.get(cls_qn, set())
            for meth in sorted(a_meths - b_meths):
                added_methods.append((cls_qn, meth))

        class_header_changed: set[str] = set()
        for cls_qn in set(before_sum.class_headers) | set(after_sum.class_headers):
            bh = before_sum.class_headers.get(cls_qn)
            ah = after_sum.class_headers.get(cls_qn)
            if bh is None or ah is None or any(
                not _ast_equal(x, y) for x, y in zip(bh, ah)
            ):
                class_header_changed.add(cls_qn)
                continue
            bo = before_sum.other_class_stmts.get(cls_qn, [])
            ao = after_sum.other_class_stmts.get(cls_qn, [])
            if not _ast_equal(bo, ao):
                class_header_changed.add(cls_qn)

        attr_changes = tuple(
            _diff_attr_map(before_sum.attrs, after_sum.attrs, old_ranges, new_ranges)
        )
        imports_changed = not _ast_equal(before_sum.imports, after_sum.imports)
        deleted_top_symbols = any(
            "." not in qn for qn in (set(before_sum.funcs) - set(after_sum.funcs))
        ) or bool(set(before_sum.classes) - set(after_sum.classes))
        top_level_changed = deleted_top_symbols or not _ast_equal(
            before_sum.other_top_level, after_sum.other_top_level
        )

        if (
            not rf.is_renamed
            and not changed_qualnames
            and not added_methods
            and not class_header_changed
            and not attr_changes
            and not imports_changed
            and not top_level_changed
        ):
            result.files.append(
                _make_file_analysis(
                    rf, rel_path, True, is_ast_noop=True, raw_text_changed=raw_changed
                )
            )
            continue

        changed_old_lines: set[int] = set()
        for h in rf.hunks:
            if h.deleted_lines:
                for ln, _ in h.deleted_lines:
                    changed_old_lines.add(ln)
            elif h.added_lines:
                matched_anchor = False
                for anchor_ln in h.insertion_anchors:
                    enc = _enclosing_func_for_line(before_sum.funcs, anchor_ln)
                    if enc is not None and enc in changed_qualnames:
                        changed_old_lines.add(anchor_ln)
                        matched_anchor = True
                if (
                    not matched_anchor
                    and not added_methods
                    and not attr_changes
                    and not imports_changed
                ):
                    if h.insertion_anchors:
                        changed_old_lines.update(h.insertion_anchors)
                    elif h.old_start > 0:
                        changed_old_lines.add(h.old_start)

        expanded_old_lines = _statement_header_lines(before_tree, changed_old_lines)

        result.files.append(
            _make_file_analysis(
                rf,
                rel_path,
                True,
                raw_text_changed=raw_changed,
                changed_old_lines=frozenset(changed_old_lines),
                expanded_old_lines=frozenset(expanded_old_lines),
                changed_qualnames=frozenset(changed_qualnames),
                added_methods=tuple(added_methods),
                imports_changed=imports_changed,
                attr_changes=attr_changes,
                top_level_code_changed=top_level_changed,
                class_header_changed=frozenset(class_header_changed),
            )
        )

    return result
