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
    semantic_equiv_used: bool = False
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


_PURE_ENUM_BASES = frozenset({"Enum", "Flag"})
_VALUE_MIXIN_BASES = frozenset(
    {"str", "int", "float", "bool", "bytes", "StrEnum", "IntEnum", "IntFlag", "ReprEnum"}
)
_STDLIB_ENUM_NAMES = frozenset({"Enum", "Flag", "StrEnum", "IntEnum", "IntFlag", "ReprEnum"})


def _get_file_tree(
    rel_path: str,
    sqlglot_root: Path,
    patched_trees: dict[str, ast.Module],
) -> ast.Module | None:
    if rel_path in patched_trees:
        return patched_trees[rel_path]
    loaded = _load_base_python(sqlglot_root, rel_path)
    return loaded[1] if loaded is not None else None


def _resolve_imported_file(
    tree: ast.Module,
    rel_path: str,
    symbol: str,
    sqlglot_root: Path,
    patched_trees: dict[str, ast.Module] | None = None,
) -> tuple[str, str, ast.Module] | None:
    pkg_parts = rel_path.split("/")[:-1]
    pt = patched_trees or {}
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
                loaded_tree = _get_file_tree(cand_rel, sqlglot_root, pt)
                if loaded_tree is not None:
                    return cand_rel, alias.name, loaded_tree
    return None


def _is_imported_from_enum(tree: ast.Module, symbol: str) -> bool:
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and stmt.module == "enum":
            for alias in stmt.names:
                if (alias.asname or alias.name) == symbol and alias.name in _STDLIB_ENUM_NAMES:
                    return True
    return False


def _local_top_class_names(tree: ast.Module) -> set[str]:
    return {s.name for s in tree.body if isinstance(s, ast.ClassDef)}


def _find_class_node(
    tree: ast.Module,
    rel_path: str,
    cls_name: str,
    sqlglot_root: Path,
    patched_trees: dict[str, ast.Module] | None = None,
) -> tuple[str, ast.ClassDef, ast.Module] | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            return rel_path, node, tree
    imp = _resolve_imported_file(tree, rel_path, cls_name, sqlglot_root, patched_trees)
    if imp is not None:
        t_rel, orig_name, t_tree = imp
        for node in t_tree.body:
            if isinstance(node, ast.ClassDef) and node.name == orig_name:
                return t_rel, node, t_tree
    return None


def _class_defines_custom_eq(cnode: ast.ClassDef) -> bool:
    def scan(stmts: list[ast.stmt]) -> bool:
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if stmt.name in ("__eq__", "__ne__"):
                    return True
                continue
            if isinstance(stmt, ast.ClassDef):
                continue
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    for sub in ast.walk(tgt):
                        if isinstance(sub, ast.Name) and sub.id in ("__eq__", "__ne__"):
                            return True
            elif isinstance(stmt, ast.AnnAssign):
                for sub in ast.walk(stmt.target):
                    if isinstance(sub, ast.Name) and sub.id in ("__eq__", "__ne__"):
                        return True
            for child_body_attr in ("body", "orelse", "finalbody"):
                child = getattr(stmt, child_body_attr, None)
                if isinstance(child, list) and child and scan(child):
                    return True
            for handler in getattr(stmt, "handlers", ()):
                if scan(handler.body):
                    return True
        return False

    return scan(cnode.body)


def _base_expr_name(base: ast.expr, tree: ast.Module) -> tuple[str, bool]:
    """Return (base_name, is_enum_module_attr) for Name or dotted Attribute chains."""
    if isinstance(base, ast.Name):
        return base.id, False
    if isinstance(base, ast.Attribute):
        cur: ast.expr = base
        while isinstance(cur, ast.Attribute):
            cur = cur.value
        if not isinstance(cur, ast.Name):
            return "", False
        if isinstance(base.value, ast.Name):
            mod_alias = base.value.id
            for stmt in tree.body:
                if isinstance(stmt, ast.Import):
                    for alias in stmt.names:
                        if (alias.asname or alias.name) == mod_alias and alias.name == "enum":
                            return base.attr, True
        return base.attr, False
    return "", False


_REPO_AST_CACHE: dict[tuple[str, str], ast.Module] = {}


def _parse_repo_tree(sqlglot_root: Path, rel: str, raw: str) -> ast.Module | None:
    key = (str(sqlglot_root), rel)
    cached = _REPO_AST_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        tree = ast.parse(raw, filename=rel)
    except (SyntaxError, ValueError):
        return None
    _REPO_AST_CACHE[key] = tree
    return tree


def _iter_repo_trees_containing(
    needles: tuple[str, ...],
    sqlglot_root: Path,
    patched_trees: dict[str, ast.Module],
    prefilter: re.Pattern[str] | None = None,
) -> list[tuple[str, ast.Module]]:
    out: list[tuple[str, ast.Module]] = []
    seen: set[str] = set()
    for rel, p_tree in patched_trees.items():
        seen.add(rel)
        out.append((rel, p_tree))
    word_re = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in needles) + r")\b")
    for path in sorted(sqlglot_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(sqlglot_root).as_posix()
        if rel in seen:
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        if not word_re.search(raw):
            continue
        if prefilter is not None and not prefilter.search(raw):
            continue
        tree = _parse_repo_tree(sqlglot_root, rel, raw)
        if tree is not None:
            out.append((rel, tree))
    return out


def _has_unsafe_subclass_in_repo(
    target_cls_name: str,
    target_rel: str,
    sqlglot_root: Path,
    patched_trees: dict[str, ast.Module],
) -> bool:
    """Check transitively across repo + patched trees for unsafe subclasses or post-class __eq__ mutation."""
    frontier: set[str] = {target_cls_name}
    checked: set[str] = set()
    while frontier:
        batch = tuple(sorted(frontier - checked))
        if not batch:
            break
        checked.update(batch)
        pat = "|".join(re.escape(n) for n in batch)
        sub_prefilter = re.compile(
            rf"(?:\bclass\s+\w+\s*\([^:]{{0,300}}\b(?:{pat})\b"
            rf"|\b(?:{pat})\s+as\s+\w+"
            rf"|\b(?:{pat})\s*\.\s*__(?:eq|ne)__"
            rf"|\bsetattr\s*\([^,\n]{{0,120}}\b(?:{pat})\b)"
        )
        next_frontier: set[str] = set()
        for rel, tree in _iter_repo_trees_containing(
            batch, sqlglot_root, patched_trees, prefilter=sub_prefilter
        ):
            aliases = set(batch)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for a in node.names:
                        if a.name in batch:
                            aliases.add(a.asname or a.name)

            def _matches_cls_ref(expr: ast.AST) -> bool:
                return (isinstance(expr, ast.Name) and expr.id in aliases) or (
                    isinstance(expr, ast.Attribute) and expr.attr in aliases
                )

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    bnames = {_base_expr_name(b, tree)[0] for b in node.bases}
                    if bnames & aliases:
                        if node.keywords or node.decorator_list or _class_defines_custom_eq(node):
                            return True
                        for b in node.bases:
                            bn, _ = _base_expr_name(b, tree)
                            if bn in aliases or bn == "object":
                                continue
                            if not bn or not _is_safe_eq_class(
                                bn, tree, rel, sqlglot_root, patched_trees
                            ):
                                return True
                        if node.name not in checked:
                            next_frontier.add(node.name)
                elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for tgt in targets:
                        for sub in ast.walk(tgt):
                            if (
                                isinstance(sub, ast.Attribute)
                                and sub.attr in ("__eq__", "__ne__")
                                and _matches_cls_ref(sub.value)
                            ):
                                return True
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "setattr"
                    and len(node.args) >= 2
                    and _matches_cls_ref(node.args[0])
                    and (
                        not isinstance(node.args[1], ast.Constant)
                        or node.args[1].value in ("__eq__", "__ne__")
                    )
                ):
                    return True
        frontier = next_frontier
    return False


def _check_single_class_mro_object_eq(
    cnode: ast.ClassDef,
    c_tree: ast.Module,
    c_rel: str,
    sqlglot_root: Path,
    patched_trees: dict[str, ast.Module],
    visited: frozenset[tuple[str, str]],
) -> bool:
    if cnode.keywords or cnode.decorator_list or _class_defines_custom_eq(cnode):
        return False
    local_classes = _local_top_class_names(c_tree)
    for base in cnode.bases:
        if not isinstance(base, ast.Name):
            bname, _ = _base_expr_name(base, c_tree)
            if not bname or bname == "object":
                return False
        else:
            bname = base.id
        if bname == "object" and "object" not in local_classes:
            continue
        if not bname or not _is_safe_eq_class(
            bname, c_tree, c_rel, sqlglot_root, patched_trees, visited
        ):
            return False
    return True


def _is_safe_eq_class(
    cls_name: str,
    tree: ast.Module,
    rel_path: str,
    sqlglot_root: Path,
    patched_trees: dict[str, ast.Module],
    visited: frozenset[tuple[str, str]] = frozenset(),
) -> bool:
    """Return True iff every class in cls_name's MRO inherits __eq__ and __ne__ strictly from object."""
    if len(visited) > 8 or cls_name in _VALUE_MIXIN_BASES or cls_name in _PURE_ENUM_BASES:
        return False
    found = _find_class_node(tree, rel_path, cls_name, sqlglot_root, patched_trees=None)
    if found is None:
        return False
    c_rel, cnode, c_tree = found
    key = (c_rel, cnode.name)
    if key in visited:
        return False
    next_visited = visited | {key}
    if not _check_single_class_mro_object_eq(
        cnode, c_tree, c_rel, sqlglot_root, patched_trees, next_visited
    ):
        return False
    if c_rel in patched_trees and patched_trees[c_rel] is not c_tree:
        p_tree = patched_trees[c_rel]
        p_found = _find_class_node(p_tree, c_rel, cnode.name, sqlglot_root, patched_trees)
        if p_found is None or not _check_single_class_mro_object_eq(
            p_found[1], p_tree, c_rel, sqlglot_root, patched_trees, next_visited
        ):
            return False
    return True


def _analyze_enum_class(
    enum_name: str,
    tree: ast.Module,
    rel_path: str,
    sqlglot_root: Path,
    patched_trees: dict[str, ast.Module],
) -> tuple[set[str], bool] | None:
    """Resolve enum_name's MRO and return (members, has_value_mixin) or None if unsafe."""
    found = _find_class_node(tree, rel_path, enum_name, sqlglot_root, patched_trees=None)
    if found is None:
        return None
    c_rel, cnode, c_tree = found
    if c_rel in patched_trees:
        p_found = _find_class_node(patched_trees[c_rel], c_rel, cnode.name, sqlglot_root, patched_trees)
        if p_found is None or not _ast_equal(cnode, p_found[1]):
            return None
    if cnode.keywords or cnode.decorator_list or _class_defines_custom_eq(cnode):
        return None

    saw_enum_base = False
    has_value_mixin = False
    visited: set[tuple[str, str]] = {(c_rel, cnode.name)}

    def walk_bases(node: ast.ClassDef, n_tree: ast.Module, n_rel: str, depth: int) -> bool:
        nonlocal saw_enum_base, has_value_mixin
        if depth > 8:
            return False
        local_classes = _local_top_class_names(n_tree)
        for base in node.bases:
            bname, is_enum_mod = _base_expr_name(base, n_tree)
            if not bname:
                return False
            if isinstance(base, ast.Name) and bname == "object" and "object" not in local_classes:
                continue
            if bname in _PURE_ENUM_BASES and (is_enum_mod or _is_imported_from_enum(n_tree, bname)):
                saw_enum_base = True
                continue
            if bname in _VALUE_MIXIN_BASES:
                if bname in ("StrEnum", "IntEnum", "IntFlag", "ReprEnum"):
                    if not (is_enum_mod or _is_imported_from_enum(n_tree, bname)):
                        return False
                    saw_enum_base = True
                elif not isinstance(base, ast.Name) or bname in local_classes:
                    return False
                has_value_mixin = True
                continue
            b_found = _find_class_node(n_tree, n_rel, bname, sqlglot_root, patched_trees=None)
            if b_found is None:
                return False
            b_rel, bnode, b_tree = b_found
            bkey = (b_rel, bnode.name)
            if bkey in visited:
                return False
            visited.add(bkey)
            if bnode.keywords or bnode.decorator_list or _class_defines_custom_eq(bnode):
                return False
            if b_rel in patched_trees and patched_trees[b_rel] is not b_tree:
                pb_tree = patched_trees[b_rel]
                pb_found = _find_class_node(pb_tree, b_rel, bnode.name, sqlglot_root, patched_trees)
                if (
                    pb_found is None
                    or pb_found[1].keywords
                    or pb_found[1].decorator_list
                    or _class_defines_custom_eq(pb_found[1])
                ):
                    return False
                if not walk_bases(pb_found[1], pb_tree, b_rel, depth + 1):
                    return False
            if not walk_bases(bnode, b_tree, b_rel, depth + 1):
                return False
        return True

    if not walk_bases(cnode, c_tree, c_rel, 0) or not saw_enum_base:
        return None

    members: set[str] = set()
    for s in cnode.body:
        if isinstance(s, ast.Assign):
            if (
                isinstance(s.value, ast.Call)
                and (
                    (isinstance(s.value.func, ast.Name) and s.value.func.id == "nonmember")
                    or (isinstance(s.value.func, ast.Attribute) and s.value.func.attr == "nonmember")
                )
            ):
                continue
            for t in s.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    members.add(t.id)
    return (members, has_value_mixin) if members else None


def _is_enum_member_or_ctor(val: ast.AST, enum_name: str, members: set[str]) -> bool:
    if (
        isinstance(val, ast.Call)
        and isinstance(val.func, ast.Name)
        and val.func.id == enum_name
        and len(val.args) == 1
        and not isinstance(val.args[0], ast.Starred)
        and not val.keywords
    ):
        return True
    if (
        isinstance(val, ast.Attribute)
        and isinstance(val.value, ast.Name)
        and val.value.id == enum_name
        and val.attr in members
    ):
        return True
    return False


def _verify_repo_wide_attr_closed(
    attr_name: str,
    cls_attr_names: set[str],
    owner_rel: str,
    owner_cls_name: str,
    sqlglot_root: Path,
    patched_trees: dict[str, ast.Module],
    *,
    enum_check: tuple[str, set[str]] | None = None,
) -> bool:
    """Verify across repo + patched trees that attr_name (and cls_attr_names) cannot hold unverified values."""
    all_names = {attr_name, *cls_attr_names}
    needles = tuple(sorted(all_names))

    class _Visitor(ast.NodeVisitor):
        def __init__(self, rel: str) -> None:
            self.rel = rel
            self.cls_stack: list[str] = []
            self.ok = True

        def _is_owner_class(self) -> bool:
            return self.rel == owner_rel and self.cls_stack == [owner_cls_name]

        def _scan_class_body(self, stmts: list[ast.stmt], *, direct: bool) -> None:
            for stmt in stmts:
                if not self.ok:
                    return
                if isinstance(stmt, ast.ClassDef):
                    continue
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if stmt.name in all_names:
                        self.ok = False
                        return
                    continue
                if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                    targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                    for tgt in targets:
                        for sub in ast.walk(tgt):
                            if isinstance(sub, ast.Name) and sub.id in all_names:
                                if (
                                    direct
                                    and enum_check is not None
                                    and isinstance(tgt, ast.Name)
                                    and stmt.value is not None
                                    and _is_enum_member_or_ctor(stmt.value, enum_check[0], enum_check[1])
                                ):
                                    continue
                                self.ok = False
                                return
                elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                    if any(
                        isinstance(sub, ast.Name) and sub.id in all_names
                        for sub in ast.walk(stmt.target)
                    ):
                        self.ok = False
                        return
                elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                    for item in stmt.items:
                        if item.optional_vars is not None and any(
                            isinstance(sub, ast.Name) and sub.id in all_names
                            for sub in ast.walk(item.optional_vars)
                        ):
                            self.ok = False
                            return
                for attr_b in ("body", "orelse", "finalbody"):
                    child = getattr(stmt, attr_b, None)
                    if isinstance(child, list) and child:
                        self._scan_class_body(child, direct=False)
                for handler in getattr(stmt, "handlers", ()):
                    if handler.name in all_names:
                        self.ok = False
                        return
                    self._scan_class_body(handler.body, direct=False)
                for case in getattr(stmt, "cases", ()):
                    self._scan_class_body(case.body, direct=False)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            if not self.ok:
                return
            self.cls_stack.append(node.name)
            self._scan_class_body(node.body, direct=True)
            if self.ok:
                self.generic_visit(node)
            self.cls_stack.pop()

        def visit_Assign(self, node: ast.Assign) -> None:
            if not self.ok:
                return
            for tgt in node.targets:
                self._check_store_target(tgt, node.value, is_direct=isinstance(tgt, ast.Attribute))
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if not self.ok:
                return
            self._check_store_target(
                node.target, node.value, is_direct=isinstance(node.target, ast.Attribute)
            )
            self.generic_visit(node)

        def visit_For(self, node: ast.For) -> None:
            if not self.ok:
                return
            self._check_store_target(node.target, None, is_direct=False)
            self.generic_visit(node)

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
            if not self.ok:
                return
            self._check_store_target(node.target, None, is_direct=False)
            self.generic_visit(node)

        def visit_With(self, node: ast.With) -> None:
            if not self.ok:
                return
            for item in node.items:
                if item.optional_vars is not None:
                    self._check_store_target(item.optional_vars, None, is_direct=False)
            self.generic_visit(node)

        def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
            if not self.ok:
                return
            for item in node.items:
                if item.optional_vars is not None:
                    self._check_store_target(item.optional_vars, None, is_direct=False)
            self.generic_visit(node)

        def visit_comprehension(self, node: ast.comprehension) -> None:
            if not self.ok:
                return
            self._check_store_target(node.target, None, is_direct=False)
            self.generic_visit(node)

        def _check_store_target(self, tgt: ast.AST, value: ast.AST | None, *, is_direct: bool) -> None:
            for sub in ast.walk(tgt):
                if isinstance(sub, ast.Subscript):
                    if isinstance(sub.value, ast.Attribute) and sub.value.attr == "__dict__":
                        if (
                            isinstance(sub.slice, ast.Constant)
                            and isinstance(sub.slice.value, str)
                            and sub.slice.value in all_names
                        ):
                            self.ok = False
                            return
                if not (isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Store)):
                    continue
                if sub.attr == attr_name:
                    if (
                        is_direct
                        and sub is tgt
                        and self._is_owner_class()
                        and isinstance(sub.value, ast.Name)
                        and sub.value.id == "self"
                    ):
                        continue
                    if (
                        enum_check is not None
                        and is_direct
                        and sub is tgt
                        and value is not None
                        and _is_enum_member_or_ctor(value, enum_check[0], enum_check[1])
                    ):
                        continue
                    self.ok = False
                    return
                if sub.attr in cls_attr_names:
                    if (
                        enum_check is not None
                        and is_direct
                        and sub is tgt
                        and value is not None
                        and _is_enum_member_or_ctor(value, enum_check[0], enum_check[1])
                    ):
                        continue
                    self.ok = False
                    return

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            if isinstance(node.target, ast.Attribute) and node.target.attr in all_names:
                self.ok = False
            elif isinstance(node.target, ast.Name) and node.target.id in all_names:
                self.ok = False
            self.generic_visit(node)

        def visit_Delete(self, node: ast.Delete) -> None:
            for tgt in node.targets:
                for sub in ast.walk(tgt):
                    if isinstance(sub, ast.Attribute) and sub.attr in all_names:
                        self.ok = False
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if not self.ok:
                return
            fn = node.func
            fn_name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
            if fn_name in ("setattr", "delattr") and len(node.args) >= 2:
                arg1 = node.args[1]
                if not isinstance(arg1, ast.Constant) or not isinstance(arg1.value, str):
                    self.ok = False
                elif arg1.value in all_names:
                    self.ok = False
            elif fn_name == "object" and isinstance(fn, ast.Attribute) and len(node.args) >= 2:
                arg1 = node.args[1]
                if isinstance(arg1, ast.Constant) and arg1.value in all_names:
                    self.ok = False
            elif fn_name == "update" and isinstance(fn, ast.Attribute):
                if isinstance(fn.value, ast.Attribute) and fn.value.attr == "__dict__":
                    self.ok = False
            self.generic_visit(node)

    for rel, tree in _iter_repo_trees_containing(needles, sqlglot_root, patched_trees):
        v = _Visitor(rel)
        v.visit(tree)
        if not v.ok:
            return False
    return True


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
    used_equiv: bool = False


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
        resolved_attrs: list[str] = []
        for fn in (b_func, a_func):
            param_names = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
            if fn.args.vararg:
                param_names.add(fn.args.vararg.arg)
            if fn.args.kwarg:
                param_names.add(fn.args.kwarg.arg)
            if var in param_names:
                return None

            # Reject any binding of var outside a single top-level simple Assign in fn.body.
            assign_vals: list[ast.AST] = []
            unsafe = False

            def scan_stmt_list(stmts: list[ast.stmt], *, top_level: bool) -> None:
                nonlocal unsafe
                for stmt in stmts:
                    if unsafe:
                        return
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        if stmt.name == var or any(
                            isinstance(sub, (ast.Nonlocal, ast.Global)) and var in sub.names
                            for sub in ast.walk(stmt)
                        ):
                            unsafe = True
                            return
                        continue
                    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                        for alias in stmt.names:
                            if (alias.asname or alias.name.split(".")[0]) == var:
                                unsafe = True
                                return
                        continue
                    if isinstance(stmt, ast.Assign):
                        for t in stmt.targets:
                            if top_level and isinstance(t, ast.Name) and t.id == var:
                                assign_vals.append(stmt.value)
                            elif any(
                                isinstance(n, ast.Name)
                                and n.id == var
                                and isinstance(n.ctx, ast.Store)
                                for n in ast.walk(t)
                            ):
                                unsafe = True
                                return
                    elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
                        if any(
                            isinstance(n, ast.Name)
                            and n.id == var
                            and isinstance(n.ctx, ast.Store)
                            for n in ast.walk(stmt.target)
                        ):
                            unsafe = True
                            return
                    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                        if any(
                            isinstance(n, ast.Name)
                            and n.id == var
                            and isinstance(n.ctx, ast.Store)
                            for n in ast.walk(stmt.target)
                        ):
                            unsafe = True
                            return
                    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                        for item in stmt.items:
                            if item.optional_vars is not None and any(
                                isinstance(n, ast.Name)
                                and n.id == var
                                and isinstance(n.ctx, ast.Store)
                                for n in ast.walk(item.optional_vars)
                            ):
                                unsafe = True
                                return
                    elif isinstance(stmt, ast.Delete):
                        if any(
                            isinstance(n, ast.Name) and n.id == var
                            for tgt in stmt.targets
                            for n in ast.walk(tgt)
                        ):
                            unsafe = True
                            return

                    for sub in ast.walk(stmt):
                        if isinstance(sub, ast.NamedExpr) and sub.target.id == var:
                            unsafe = True
                            return
                        if isinstance(sub, ast.ExceptHandler) and sub.name == var:
                            unsafe = True
                            return
                        if isinstance(sub, ast.MatchAs) and sub.name == var:
                            unsafe = True
                            return
                        if isinstance(sub, ast.comprehension) and any(
                            isinstance(n, ast.Name) and n.id == var and isinstance(n.ctx, ast.Store)
                            for n in ast.walk(sub.target)
                        ):
                            unsafe = True
                            return

                    for attr_b in ("body", "orelse", "finalbody"):
                        child = getattr(stmt, attr_b, None)
                        if isinstance(child, list) and child:
                            scan_stmt_list(child, top_level=False)
                    for handler in getattr(stmt, "handlers", ()):
                        scan_stmt_list(handler.body, top_level=False)
                    for case in getattr(stmt, "cases", ()):
                        scan_stmt_list(case.body, top_level=False)

            scan_stmt_list(fn.body, top_level=True)
            if unsafe or len(assign_vals) != 1:
                return None
            v0 = assign_vals[0]
            if not (
                isinstance(v0, ast.Attribute)
                and isinstance(v0.value, ast.Name)
                and v0.value.id == "self"
            ):
                return None
            resolved_attrs.append(v0.attr)
        if len(resolved_attrs) == 2 and resolved_attrs[0] == resolved_attrs[1]:
            return resolved_attrs[0]
    return None


def _collect_self_attr_stores(
    cnode: ast.ClassDef, attr_name: str
) -> tuple[list[ast.AST], list[ast.AST]] | None:
    """Collect direct self.<attr> and class-body <attr> stores; return None on unsafe patterns."""
    anns: list[ast.AST] = []
    vals: list[ast.AST] = []
    allowed_store_ids: set[int] = set()

    def scan_class_body(stmts: list[ast.stmt], *, direct: bool) -> bool:
        for stmt in stmts:
            if isinstance(stmt, ast.ClassDef):
                continue
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if stmt.name == attr_name:
                    return False
                continue
            if isinstance(stmt, ast.AnnAssign):
                if direct and isinstance(stmt.target, ast.Name) and stmt.target.id == attr_name:
                    anns.append(stmt.annotation)
                    if stmt.value is not None:
                        vals.append(stmt.value)
                elif any(isinstance(n, ast.Name) and n.id == attr_name for n in ast.walk(stmt.target)):
                    return False
            elif isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if direct and isinstance(tgt, ast.Name) and tgt.id == attr_name:
                        vals.append(stmt.value)
                    elif any(isinstance(n, ast.Name) and n.id == attr_name for n in ast.walk(tgt)):
                        return False
            elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                if any(isinstance(n, ast.Name) and n.id == attr_name for n in ast.walk(stmt.target)):
                    return False
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                for item in stmt.items:
                    if item.optional_vars is not None and any(
                        isinstance(n, ast.Name) and n.id == attr_name
                        for n in ast.walk(item.optional_vars)
                    ):
                        return False
            for attr_b in ("body", "orelse", "finalbody"):
                child = getattr(stmt, attr_b, None)
                if isinstance(child, list) and child and not scan_class_body(child, direct=False):
                    return False
            for handler in getattr(stmt, "handlers", ()):
                if handler.name == attr_name or not scan_class_body(handler.body, direct=False):
                    return False
            for case in getattr(stmt, "cases", ()):
                if not scan_class_body(case.body, direct=False):
                    return False
        return True

    if not scan_class_body(cnode.body, direct=True):
        return None

    for sub in ast.walk(cnode):
        if isinstance(sub, ast.AnnAssign):
            if (
                isinstance(sub.target, ast.Attribute)
                and isinstance(sub.target.value, ast.Name)
                and sub.target.value.id == "self"
                and sub.target.attr == attr_name
            ):
                allowed_store_ids.add(id(sub.target))
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
                    allowed_store_ids.add(id(t))
                    vals.append(sub.value)
        elif isinstance(sub, ast.AugAssign):
            if isinstance(sub.target, ast.Attribute) and sub.target.attr == attr_name:
                return None
        elif isinstance(sub, ast.Delete):
            if any(
                isinstance(n, ast.Attribute) and n.attr == attr_name
                for tgt in sub.targets
                for n in ast.walk(tgt)
            ):
                return None

    for sub in ast.walk(cnode):
        if (
            isinstance(sub, ast.Attribute)
            and isinstance(sub.ctx, ast.Store)
            and sub.attr == attr_name
            and id(sub) not in allowed_store_ids
        ):
            return None
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


def _extract_mapping_value_type(ann: ast.AST | None) -> str | None:
    if not isinstance(ann, ast.Subscript):
        return None
    fn = ann.value
    fn_name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
    if fn_name not in ("dict", "Dict", "Mapping", "MutableMapping"):
        return None
    if isinstance(ann.slice, ast.Tuple) and len(ann.slice.elts) == 2:
        val_t = ann.slice.elts[1]
        if isinstance(val_t, ast.Name):
            return val_t.id
    return None


_MUTATING_DICT_METHODS = frozenset({"update", "setdefault", "pop", "popitem", "clear"})


def _has_mapping_attr_mutation(node: ast.AST, map_attr: str) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
            for tgt in targets:
                if (
                    isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.value, ast.Attribute)
                    and tgt.value.attr == map_attr
                ):
                    return True
                if (
                    isinstance(sub, ast.AugAssign)
                    and isinstance(tgt, ast.Attribute)
                    and tgt.attr == map_attr
                ):
                    return True
        elif isinstance(sub, ast.Delete):
            for tgt in sub.targets:
                if (
                    isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.value, ast.Attribute)
                    and tgt.value.attr == map_attr
                ):
                    return True
        elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            if (
                sub.func.attr in _MUTATING_DICT_METHODS
                and isinstance(sub.func.value, ast.Attribute)
                and sub.func.value.attr == map_attr
            ):
                return True
    return False


def _verify_repo_wide_mapping_closed(
    map_attr: str,
    owner_cls_name: str,
    param_name: str,
    param_pos_idx: int | None,
    t_name: str,
    owner_rel: str,
    sqlglot_root: Path,
    patched_trees: dict[str, ast.Module],
) -> bool:
    """Verify no repo/patched file mutates .<map_attr>, reassigns it externally, or passes non-T dict literals."""
    ma_esc = re.escape(map_attr)
    oc_esc = re.escape(owner_cls_name)
    prefilter = re.compile(
        rf"(?:\.{ma_esc}\s*(?:\[|\.\s*(?:update|setdefault|pop|popitem|clear)\b|[|]?=)"
        rf"|\b(?:setattr|delattr)\s*\("
        rf"|\b{oc_esc}\b)"
    )
    for rel, tree in _iter_repo_trees_containing(
        (map_attr, owner_cls_name), sqlglot_root, patched_trees, prefilter=prefilter
    ):
        if _has_mapping_attr_mutation(tree, map_attr):
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bnames = {_base_expr_name(b, tree)[0] for b in node.bases}
                if owner_cls_name in bnames:
                    for sub in ast.walk(node):
                        if (
                            isinstance(sub, ast.Attribute)
                            and isinstance(sub.ctx, ast.Store)
                            and sub.attr == map_attr
                        ):
                            return False
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store) and node.attr == map_attr:
                if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
                    return False
            elif isinstance(node, ast.Delete):
                for tgt in node.targets:
                    for sub in ast.walk(tgt):
                        if isinstance(sub, ast.Attribute) and sub.attr == map_attr:
                            return False
            elif isinstance(node, ast.Call):
                fn = node.func
                fn_name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
                if fn_name in ("setattr", "delattr") and len(node.args) >= 2:
                    arg1 = node.args[1]
                    if not isinstance(arg1, ast.Constant) or arg1.value == map_attr:
                        return False
                elif fn_name == owner_cls_name:
                    arg_val: ast.AST | None = None
                    if param_pos_idx is not None and len(node.args) > param_pos_idx:
                        arg_val = node.args[param_pos_idx]
                    for kw in node.keywords:
                        if kw.arg == param_name:
                            arg_val = kw.value
                    if isinstance(arg_val, ast.Dict):
                        for k_node, v_node in zip(arg_val.keys, arg_val.values):
                            if k_node is None:
                                return False
                            if not (
                                isinstance(v_node, ast.Name)
                                or (
                                    isinstance(v_node, ast.Call)
                                    and isinstance(v_node.func, ast.Name)
                                    and v_node.func.id == t_name
                                )
                            ):
                                return False
    return True


def _is_typed_mapping_value_extract(
    val: ast.AST, cnode: ast.ClassDef, t_name: str, ctx: _EquivContext
) -> bool:
    """Check `list(self.<m>.values())[int_const]` where self.<m> is exclusively an unmutated dict[_, T] __init__ arg."""
    if not (
        isinstance(val, ast.Subscript)
        and isinstance(val.slice, ast.Constant)
        and isinstance(val.slice.value, int)
        and not isinstance(val.slice.value, bool)
        and isinstance(val.value, ast.Call)
        and isinstance(val.value.func, ast.Name)
        and val.value.func.id == "list"
        and len(val.value.args) == 1
    ):
        return False
    inner = val.value.args[0]
    if not (
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Attribute)
        and inner.func.attr == "values"
        and not inner.args
        and isinstance(inner.func.value, ast.Attribute)
        and isinstance(inner.func.value.value, ast.Name)
        and inner.func.value.value.id == "self"
    ):
        return False
    map_attr = inner.func.value.attr
    stores = _collect_self_attr_stores(cnode, map_attr)
    if stores is None or len(stores[1]) != 1:
        return False
    if _has_mapping_attr_mutation(cnode, map_attr):
        return False
    rhs = stores[1][0]
    if not isinstance(rhs, ast.Name):
        return False
    param_name = rhs.id
    matched_pos_idx: int | None = None
    matched_init = False
    for stmt in cnode.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
            pos_args = stmt.args.posonlyargs + stmt.args.args
            for idx, arg in enumerate(pos_args):
                if arg.arg == param_name and _extract_mapping_value_type(arg.annotation) == t_name:
                    matched_init = True
                    matched_pos_idx = idx - 1 if idx > 0 else None
                    break
            if not matched_init:
                for arg in stmt.args.kwonlyargs:
                    if arg.arg == param_name and _extract_mapping_value_type(arg.annotation) == t_name:
                        matched_init = True
                        break
    if not matched_init:
        return False
    return _verify_repo_wide_mapping_closed(
        map_attr=map_attr,
        owner_cls_name=cnode.name,
        param_name=param_name,
        param_pos_idx=matched_pos_idx,
        t_name=t_name,
        owner_rel=ctx.rel_path,
        sqlglot_root=ctx.sqlglot_root,
        patched_trees=ctx.patched_trees,
    )


def _is_provable_case1_rhs(
    val: ast.AST, cnode: ast.ClassDef, t_name: str, ctx: _EquivContext
) -> bool:
    if isinstance(val, ast.Constant) and val.value is None:
        return True
    if isinstance(val, ast.Call) and isinstance(val.func, ast.Name) and val.func.id == t_name:
        return True
    return _is_typed_mapping_value_extract(val, cnode, t_name, ctx)


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
    if attr_name is None:
        return False

    b_stores = _collect_self_attr_stores(ctx.b_cls, attr_name)
    a_stores = _collect_self_attr_stores(ctx.a_cls, attr_name)
    if b_stores is None or a_stores is None:
        return False
    b_anns, b_vals = b_stores
    a_anns, a_vals = a_stores
    if not b_vals or not _ast_equal(b_anns, a_anns) or not _ast_equal(b_vals, a_vals):
        return False

    # Case 1: self.<attr> is None <-> self.<attr> == None
    if isinstance(rhs, ast.Constant) and rhs.value is None:
        if len(b_anns) != 1:
            return False
        t_name = _extract_optional_type_name(b_anns[0])
        if not t_name:
            return False
        if not all(_is_provable_case1_rhs(v, ctx.b_cls, t_name, ctx) for v in b_vals):
            return False
        if not _is_safe_eq_class(
            t_name,
            ctx.before_tree,
            ctx.rel_path,
            ctx.sqlglot_root,
            ctx.patched_trees,
        ):
            return False
        if _has_unsafe_subclass_in_repo(t_name, ctx.rel_path, ctx.sqlglot_root, ctx.patched_trees):
            return False
        if not _verify_repo_wide_attr_closed(
            attr_name=attr_name,
            cls_attr_names=set(),
            owner_rel=ctx.rel_path,
            owner_cls_name=ctx.b_cls.name,
            sqlglot_root=ctx.sqlglot_root,
            patched_trees=ctx.patched_trees,
            enum_check=None,
        ):
            return False
        ctx.used_equiv = True
        return True

    # Case 2: self.<attr> is EnumCls.MEMBER <-> self.<attr> == EnumCls.MEMBER
    if isinstance(rhs, ast.Attribute) and isinstance(rhs.value, ast.Name):
        enum_name = rhs.value.id
        member_name = rhs.attr
        enum_info = _analyze_enum_class(
            enum_name,
            ctx.before_tree,
            ctx.rel_path,
            ctx.sqlglot_root,
            ctx.patched_trees,
        )
        if enum_info is None:
            return False
        members, _has_value_mixin = enum_info
        if member_name not in members:
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
        used_cls_attrs: set[str] = set()
        for v in b_vals:
            if _is_enum_member_or_ctor(v, enum_name, members):
                continue
            if (
                isinstance(v, ast.Attribute)
                and isinstance(v.value, ast.Name)
                and v.value.id == "self"
            ):
                cv = cls_attrs.get(v.attr)
                if cv is not None and _is_enum_member_or_ctor(cv, enum_name, members):
                    used_cls_attrs.add(v.attr)
                    continue
            return False
        if not _verify_repo_wide_attr_closed(
            attr_name=attr_name,
            cls_attr_names=used_cls_attrs,
            owner_rel=ctx.rel_path,
            owner_cls_name=ctx.b_cls.name,
            sqlglot_root=ctx.sqlglot_root,
            patched_trees=ctx.patched_trees,
            enum_check=(enum_name, members),
        ):
            return False
        ctx.used_equiv = True
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
        child_equiv = equiv_ctx
        if (
            equiv_ctx is not None
            and isinstance(a, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and a is not equiv_ctx.b_func
        ):
            child_equiv = None
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
                va, vb, ignore_docstrings=ignore_docstrings, equiv_ctx=child_equiv
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
    funcs: dict[str, list[tuple[int, int, ast.AST]]]
    classes: dict[str, ast.ClassDef]
    duplicate_classes: set[str]
    class_methods: dict[str, set[str]]
    class_headers: dict[str, tuple[list[ast.expr], list[ast.keyword], list[ast.expr]]]
    attrs: dict[tuple[str | None, str], list[tuple[int, int, ast.AST, ast.AST | None]]]
    imports: list[ast.stmt]
    other_top_level: list[ast.stmt]
    other_class_stmts: dict[str, list[ast.stmt]]
    top_order: list[str]
    class_order: dict[str, list[str]]
    docstrings: dict[str | None, str | None]


def _summarize_tree(tree: ast.Module) -> _TreeSummary:
    funcs: dict[str, list[tuple[int, int, ast.AST]]] = {}
    classes: dict[str, ast.ClassDef] = {}
    duplicate_classes: set[str] = set()
    class_methods: dict[str, set[str]] = {}
    class_headers: dict[str, tuple[list[ast.expr], list[ast.keyword], list[ast.expr]]] = {}
    attrs: dict[tuple[str | None, str], list[tuple[int, int, ast.AST, ast.AST | None]]] = {}
    imports: list[ast.stmt] = []
    other_top_level: list[ast.stmt] = []
    other_class_stmts: dict[str, list[ast.stmt]] = {}
    top_order: list[str] = []
    class_order: dict[str, list[str]] = {}
    docstrings: dict[str | None, str | None] = {}

    def record_assign(stmt: ast.Assign | ast.AnnAssign, class_qn: str | None) -> list[str] | None:
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        if stmt.value is None or not all(isinstance(t, ast.Name) for t in targets):
            return None
        ann = stmt.annotation if isinstance(stmt, ast.AnnAssign) else None
        start = stmt.lineno
        end = stmt.end_lineno or stmt.lineno
        names: list[str] = []
        for tgt in targets:
            assert isinstance(tgt, ast.Name)
            attrs.setdefault((class_qn, tgt.id), []).append((start, end, stmt.value, ann))
            names.append(tgt.id)
        return names

    def visit_block(stmts: list[ast.stmt], scope: list[tuple[str, str]]) -> None:
        is_module = not scope
        is_class = bool(scope and scope[-1][1] == "class")
        class_qn = ".".join(s[0] for s in scope) if is_class else None
        order = top_order if is_module else (class_order.setdefault(class_qn, []) if class_qn is not None else None)
        if is_module or is_class:
            first_doc = (
                stmts[0].value.value  # type: ignore[union-attr]
                if stmts and _is_docstring_stmt(stmts[0])
                else None
            )
            docstrings[class_qn] = first_doc

        for idx, stmt in enumerate(stmts):
            if isinstance(stmt, ast.ClassDef):
                qn = ".".join([*(s[0] for s in scope), stmt.name])
                if qn in classes:
                    duplicate_classes.add(qn)
                classes[qn] = stmt
                class_methods.setdefault(qn, set())
                class_headers[qn] = (stmt.bases, stmt.keywords, stmt.decorator_list)
                other_class_stmts.setdefault(qn, [])
                if order is not None:
                    order.append(f"cls:{stmt.name}")
                visit_block(stmt.body, [*scope, (stmt.name, "class")])
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qn = ".".join([*(s[0] for s in scope), stmt.name])
                start = _node_start_line(stmt)
                end = stmt.end_lineno or stmt.lineno
                funcs.setdefault(qn, []).append((start, end, stmt))
                if class_qn is not None:
                    class_methods.setdefault(class_qn, set()).add(stmt.name)
                if order is not None:
                    order.append(f"fn:{stmt.name}")
                visit_block(stmt.body, [*scope, (stmt.name, "func")])
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                imports.append(stmt)
                if order is not None:
                    order.append("imp")
            elif (is_module or is_class) and isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                names = record_assign(stmt, class_qn)
                if names is not None:
                    if order is not None:
                        order.append("attr:" + ",".join(names))
                else:
                    if is_module:
                        other_top_level.append(stmt)
                    elif class_qn is not None:
                        other_class_stmts.setdefault(class_qn, []).append(stmt)
                    if order is not None:
                        order.append("other")
            elif is_module:
                if not (idx == 0 and _is_docstring_stmt(stmt)):
                    other_top_level.append(stmt)
                    if order is not None:
                        order.append("other")
            elif is_class and class_qn is not None:
                if not (idx == 0 and _is_docstring_stmt(stmt)):
                    other_class_stmts.setdefault(class_qn, []).append(stmt)
                    if order is not None:
                        order.append("other")

    visit_block(tree.body, [])
    return _TreeSummary(
        funcs=funcs,
        classes=classes,
        duplicate_classes=duplicate_classes,
        class_methods=class_methods,
        class_headers=class_headers,
        attrs=attrs,
        imports=imports,
        other_top_level=other_top_level,
        other_class_stmts=other_class_stmts,
        top_order=top_order,
        class_order=class_order,
        docstrings=docstrings,
    )


def _overlaps_ranges(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    for rs, re_ in ranges:
        if start <= re_ and rs <= end:
            return True
    return False


def _diff_attr_map(
    before_map: dict[tuple[str | None, str], list[tuple[int, int, ast.AST, ast.AST | None]]],
    after_map: dict[tuple[str | None, str], list[tuple[int, int, ast.AST, ast.AST | None]]],
    old_ranges: list[tuple[int, int]],
    new_ranges: list[tuple[int, int]],
) -> list[AttrChange]:
    changes: list[AttrChange] = []
    for key in sorted(set(before_map) | set(after_map), key=lambda k: (k[0] or "", k[1])):
        b_list = before_map.get(key)
        a_list = after_map.get(key)
        cls_qn, attr_name = key
        if b_list is not None and a_list is not None and len(b_list) == len(a_list):
            overlaps = any(_overlaps_ranges(s, e, old_ranges) for s, e, _, _ in b_list) or any(
                _overlaps_ranges(s, e, new_ranges) for s, e, _, _ in a_list
            )
            if not overlaps:
                continue
            if all(
                _ast_equal(bv, av) and _ast_equal(bann, aann)
                for (_, _, bv, bann), (_, _, av, aann) in zip(b_list, a_list)
            ):
                continue
        if (
            b_list is None
            or a_list is None
            or len(b_list) != 1
            or len(a_list) != 1
            or not _ast_equal(b_list[0][3], a_list[0][3])
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
        b_dict = _extract_dict_nodes(b_list[0][2])
        a_dict = _extract_dict_nodes(a_list[0][2])
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
    funcs: dict[str, list[tuple[int, int, ast.AST]]], ln: int
) -> str | None:
    best_qn: str | None = None
    best_span = 10**9
    for qn, defs in funcs.items():
        for start, end, _ in defs:
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


def analyze_patch(commit_id: str, patch_text: str, sqlglot_root: Path) -> CommitAnalysis:
    """Parse a commit patch, apply hunks in memory, and analyze AST changes per file."""
    result = CommitAnalysis(commit_id=commit_id)
    raw_files = parse_unified_diff(patch_text)

    patched_trees: dict[str, ast.Module] = {}
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
        semantic_equiv_used = False
        for qn in sorted(set(before_sum.funcs) | set(after_sum.funcs)):
            b_list = before_sum.funcs.get(qn)
            a_list = after_sum.funcs.get(qn)
            if b_list is None:
                if "." not in qn:
                    changed_qualnames.add(qn)
                continue
            if a_list is None or len(b_list) != len(a_list):
                changed_qualnames.add(qn)
                continue
            cls_qn = qn.rsplit(".", 1)[0] if "." in qn else None
            b_cls = before_sum.classes.get(cls_qn) if cls_qn else None
            a_cls = after_sum.classes.get(cls_qn) if cls_qn else None
            for (bs, be, b_node), (as_, ae, a_node) in zip(b_list, a_list):
                if not _overlaps_ranges(bs, be, old_ranges) and not _overlaps_ranges(
                    as_, ae, new_ranges
                ):
                    continue
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
                    )
                    if b_cls is not None and a_cls is not None and not rf.is_renamed
                    else None
                )
                if not _ast_equal(b_node, a_node, equiv_ctx=equiv_ctx):
                    changed_qualnames.add(qn)
                    break
                if equiv_ctx is not None and equiv_ctx.used_equiv:
                    semantic_equiv_used = True

        added_methods: list[tuple[str, str]] = []
        for cls_qn, a_meths in after_sum.class_methods.items():
            b_meths = before_sum.class_methods.get(cls_qn, set())
            for meth in sorted(a_meths - b_meths):
                added_methods.append((cls_qn, meth))

        class_header_changed: set[str] = set(
            before_sum.duplicate_classes | after_sum.duplicate_classes
        )
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
                continue
            if before_sum.class_order.get(cls_qn) != after_sum.class_order.get(cls_qn):
                class_header_changed.add(cls_qn)

        attr_changes = tuple(
            _diff_attr_map(before_sum.attrs, after_sum.attrs, old_ranges, new_ranges)
        )
        imports_changed = not _ast_equal(before_sum.imports, after_sum.imports)
        deleted_top_symbols = any(
            "." not in qn for qn in (set(before_sum.funcs) - set(after_sum.funcs))
        ) or bool(set(before_sum.classes) - set(after_sum.classes))
        added_top_symbols = any(
            "." not in qn for qn in (set(after_sum.funcs) - set(before_sum.funcs))
        ) or bool(set(after_sum.classes) - set(before_sum.classes))
        top_order_changed = before_sum.top_order != after_sum.top_order
        top_level_changed = (
            deleted_top_symbols
            or added_top_symbols
            or top_order_changed
            or not _ast_equal(before_sum.other_top_level, after_sum.other_top_level)
        )
        docs_changed = before_sum.docstrings != after_sum.docstrings

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
                    rf,
                    rel_path,
                    True,
                    is_ast_noop=True,
                    docstring_changed=docs_changed,
                    semantic_equiv_used=semantic_equiv_used,
                    raw_text_changed=raw_changed,
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
