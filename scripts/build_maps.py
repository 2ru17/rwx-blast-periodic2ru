#!/usr/bin/env python3
"""Generate deterministic offline maps from pinned sqlglot@eb11aec."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import pickle
import sqlite3
import subprocess
import sys
import tempfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

PINNED_COMMIT = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
SQLGLOT_REPO_URL = "https://github.com/tobymao/sqlglot.git"


class _InlineExecutor:
    """Synchronous replacement for ProcessPoolExecutor during coverage tracing."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        if initializer is not None:
            initializer(*initargs)

    def submit(self, fn, /, *args, **kwargs):
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        try:
            p_fn, p_args, p_kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            res = p_fn(*p_args, **p_kwargs)
            future.set_result(pickle.loads(pickle.dumps(res)))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return [fn(*args) for args in zip(*iterables)]

    def shutdown(self, wait=True, *, cancel_futures=False):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


def encode_test_set(sorted_nums: list[int]) -> list[int | list[int]]:
    """Encode sorted test numbers by collapsing contiguous runs >= 3 into [start, end]."""
    if not sorted_nums:
        return []
    out: list[int | list[int]] = []
    start = prev = sorted_nums[0]
    for num in sorted_nums[1:]:
        if num == prev + 1:
            prev = num
            continue
        _flush_run(out, start, prev)
        start = prev = num
    _flush_run(out, start, prev)
    return out


def _flush_run(out: list[int | list[int]], start: int, end: int) -> None:
    if end - start >= 2:
        out.append([start, end])
    elif end == start + 1:
        out.extend([start, end])
    else:
        out.append(start)


class SetInterner:
    """Deduplicate sorted test-number tuples into compact integer IDs."""

    def __init__(self) -> None:
        self._index: dict[tuple[int, ...], int] = {(): 0}
        self._sets: list[list[int | list[int]]] = [[]]

    def intern(self, nums: set[int] | list[int] | tuple[int, ...]) -> int:
        key = tuple(sorted(set(nums)))
        existing = self._index.get(key)
        if existing is not None:
            return existing
        set_id = len(self._sets)
        self._index[key] = set_id
        self._sets.append(encode_test_set(list(key)))
        return set_id

    @property
    def sets(self) -> list[list[int | list[int]]]:
        return self._sets


def ensure_sqlglot_checkout(sqlglot_dir: Path, commit: str) -> Path:
    """Verify or clone pinned sqlglot checkout."""
    sqlglot_dir = sqlglot_dir.resolve()
    if not sqlglot_dir.exists():
        sqlglot_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", SQLGLOT_REPO_URL, str(sqlglot_dir)], check=True)
    head = subprocess.check_output(
        ["git", "-C", str(sqlglot_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != commit:
        subprocess.run(["git", "-C", str(sqlglot_dir), "checkout", commit], check=True)
        head = subprocess.check_output(
            ["git", "-C", str(sqlglot_dir), "rev-parse", "HEAD"], text=True
        ).strip()
    if head != commit:
        raise RuntimeError(f"Expected sqlglot commit {commit}, found {head}")
    return sqlglot_dir


def load_test_metadata(tests_json_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load canonical test numbers and IDs from tests.json."""
    raw = json.loads(tests_json_path.read_text(encoding="utf-8"))
    tests: list[dict[str, Any]] = []
    id_to_num: dict[str, int] = {}
    for entry in raw["tests"]:
        ms = int(entry["seconds"] * 1000 + 0.5)
        tests.append(
            {
                "number": entry["number"],
                "id": entry["id"],
                "file": entry["file"],
                "ms": ms,
            }
        )
        id_to_num[entry["id"]] = entry["number"]
    tests.sort(key=lambda item: item["number"])
    return tests, id_to_num


def collect_runtime_coverage_and_fixtures(
    sqlglot_dir: Path,
    id_to_num: dict[str, int],
) -> tuple[
    dict[str, dict[int, set[int]]],
    dict[str, set[int]],
    dict[str, set[int]],
    set[int],
]:
    """Run sqlglot pytest suite under ctrace dynamic contexts and audit file opens."""
    import coverage
    from coverage.numbits import numbits_to_nums
    import pytest

    os.environ.pop("COVERAGE_CORE", None)
    os.environ["SKIP_INTEGRATION"] = "0"
    os.environ["PYTHONPATH"] = str(sqlglot_dir)

    sqlglot_root_str = str(sqlglot_dir)
    if sys.path[0] != sqlglot_root_str:
        sys.path.insert(0, sqlglot_root_str)
    prev_cwd = os.getcwd()
    os.chdir(sqlglot_root_str)

    fixture_to_tests: dict[str, set[int]] = defaultdict(set)
    source_inspecting_tests: set[int] = set()
    active_state: dict[str, int | None] = {"test_num": None}

    def audit_hook(event: str, args: tuple[Any, ...]) -> None:
        if event != "open" or not args:
            return
        test_num = active_state["test_num"]
        if test_num is None:
            return
        path_arg = args[0]
        if not isinstance(path_arg, (str, os.PathLike)):
            return
        raw_path = os.fspath(path_arg)
        if not os.path.isabs(raw_path):
            abs_path = os.path.normpath(os.path.join(sqlglot_root_str, raw_path))
        else:
            abs_path = os.path.normpath(raw_path)
        prefix = sqlglot_root_str + os.sep
        if not abs_path.startswith(prefix):
            return
        rel = abs_path[len(prefix) :].replace(os.sep, "/")
        if rel.startswith("tests/fixtures/") or rel.endswith((".sql", ".csv", ".csv.gz", ".json")):
            if rel.startswith("tests/"):
                fixture_to_tests[rel].add(test_num)
        elif rel.startswith("sqlglot/") and rel.endswith(".py"):
            source_inspecting_tests.add(test_num)

    with tempfile.TemporaryDirectory() as td:
        rc_path = os.path.join(td, ".coveragerc")
        data_path = os.path.join(td, ".coverage")
        Path(rc_path).write_text("[run]\ncore = ctrace\nbranch = False\n", encoding="utf-8")

        cov = coverage.Coverage(
            config_file=rc_path,
            data_file=data_path,
            include=[f"{sqlglot_root_str}/sqlglot/*", f"{sqlglot_root_str}/tests/*"],
        )

        orig_ppe = concurrent.futures.ProcessPoolExecutor
        concurrent.futures.ProcessPoolExecutor = _InlineExecutor  # type: ignore[misc]
        sys.addaudithook(audit_hook)

        cov.start()
        collector_repr = repr(cov._collector)
        if "CTracer" not in collector_repr:
            cov.stop()
            raise RuntimeError(f"Expected CTracer coverage core, got: {collector_repr}")

        executed_tests = 0

        class _ContextPlugin:
            @pytest.hookimpl(hookwrapper=True)
            def pytest_runtest_protocol(self, item: Any, nextitem: Any) -> Any:
                nodeid = item.nodeid
                test_num = id_to_num.get(nodeid)
                active_state["test_num"] = test_num
                cov.switch_context(nodeid if test_num is not None else "")
                try:
                    yield
                finally:
                    active_state["test_num"] = None
                    cov.switch_context("")

            def pytest_runtest_logreport(self, report: Any) -> None:
                nonlocal executed_tests
                if report.when == "call":
                    executed_tests += 1

        try:
            exit_code = pytest.main(
                ["tests", "-q", "--disable-warnings"],
                plugins=[_ContextPlugin()],
            )
            if exit_code not in (0, 5):
                raise RuntimeError(f"pytest exited with unexpected code {exit_code}")
            if executed_tests < 1600:
                raise RuntimeError(f"Expected >= 1600 executed tests, got {executed_tests}")
        finally:
            cov.stop()
            cov.save()
            concurrent.futures.ProcessPoolExecutor = orig_ppe  # type: ignore[misc]
            os.chdir(prev_cwd)

        line_to_tests: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
        import_time_lines: dict[str, set[int]] = defaultdict(set)

        conn = sqlite3.connect(data_path)
        try:
            file_map = {
                row[0]: _normalize_rel_path(row[1], sqlglot_root_str)
                for row in conn.execute("SELECT id, path FROM file")
            }
            ctx_map = {row[0]: row[1] for row in conn.execute("SELECT id, context FROM context")}
            for file_id, ctx_id, numbits in conn.execute(
                "SELECT file_id, context_id, numbits FROM line_bits"
            ):
                rel_file = file_map.get(file_id)
                if not rel_file:
                    continue
                ctx_name = ctx_map.get(ctx_id, "")
                lines = numbits_to_nums(numbits)
                if ctx_name == "":
                    import_time_lines[rel_file].update(lines)
                else:
                    test_num = id_to_num.get(ctx_name)
                    if test_num is not None:
                        file_lines = line_to_tests[rel_file]
                        for ln in lines:
                            file_lines[ln].add(test_num)
        finally:
            conn.close()

    return line_to_tests, import_time_lines, fixture_to_tests, source_inspecting_tests


def _normalize_rel_path(path_str: str, root_str: str) -> str | None:
    norm = os.path.normpath(path_str)
    prefix = root_str + os.sep
    if not norm.startswith(prefix):
        return None
    rel = norm[len(prefix) :].replace(os.sep, "/")
    if rel.startswith(("sqlglot/", "tests/")):
        return rel
    return None


def iter_python_files(sqlglot_dir: Path) -> list[tuple[str, Path]]:
    """Return sorted (relative_path, absolute_path) for all sqlglot/ and tests/ Python files."""
    files: list[tuple[str, Path]] = []
    for base in ("sqlglot", "tests"):
        root = sqlglot_dir / base
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            rel = p.relative_to(sqlglot_dir).as_posix()
            files.append((rel, p))
    files.sort(key=lambda item: item[0])
    return files


def resolve_module_index(py_files: list[tuple[str, Path]]) -> dict[str, str]:
    """Map dotted module names (e.g. sqlglot.parser) to relative file paths."""
    mod_to_file: dict[str, str] = {}
    for rel, _ in py_files:
        if rel.endswith("/__init__.py"):
            mod = rel[: -len("/__init__.py")].replace("/", ".")
        else:
            mod = rel[: -len(".py")].replace("/", ".")
        mod_to_file[mod] = rel
    return mod_to_file


def extract_imports(tree: ast.Module, rel_file: str, mod_to_file: dict[str, str]) -> list[str]:
    """Extract internal repo file dependencies imported by a Python module."""
    is_pkg = rel_file.endswith("/__init__.py")
    pkg_parts = rel_file.split("/")[:-1]
    if not is_pkg:
        pkg_parts = rel_file.split("/")[:-1]

    deps: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = _resolve_dotted(alias.name, mod_to_file)
                if target:
                    deps.add(target)
        elif isinstance(node, ast.ImportFrom):
            base_mod = ""
            if node.level > 0:
                up = node.level - 1
                if up <= len(pkg_parts):
                    prefix_parts = pkg_parts[: len(pkg_parts) - up]
                    base_mod = ".".join(prefix_parts)
                    if node.module:
                        base_mod = f"{base_mod}.{node.module}" if base_mod else node.module
            elif node.module:
                base_mod = node.module

            if base_mod:
                target = _resolve_dotted(base_mod, mod_to_file)
                if target:
                    deps.add(target)
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    sub_mod = f"{base_mod}.{alias.name}"
                    sub_target = _resolve_dotted(sub_mod, mod_to_file)
                    if sub_target:
                        deps.add(sub_target)
                    elif base_mod == "sqlglot.dialects":
                        dialect_file = f"sqlglot/dialects/{alias.name.lower()}.py"
                        if dialect_file in mod_to_file.values():
                            deps.add(dialect_file)
    deps.discard(rel_file)
    return sorted(deps)


def _resolve_dotted(mod_name: str, mod_to_file: dict[str, str]) -> str | None:
    if mod_name in mod_to_file:
        return mod_to_file[mod_name]
    parts = mod_name.split(".")
    while len(parts) > 1:
        parts.pop()
        candidate = ".".join(parts)
        if candidate in mod_to_file:
            return mod_to_file[candidate]
    return None


def dialect_family_for_file(rel_file: str) -> str | None:
    """Return dialect family stem for dialect/parser/generator/typing files."""
    parts = rel_file.split("/")
    if len(parts) == 3 and parts[0] == "sqlglot" and parts[1] in (
        "dialects",
        "parsers",
        "generators",
        "typing",
    ):
        stem = Path(parts[2]).stem
        if stem not in ("__init__", "dialect"):
            return stem
    return None


def _expr_str_key(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _node_start_line(node: ast.AST) -> int:
    decs = getattr(node, "decorator_list", None)
    if decs:
        return min([node.lineno, *(d.lineno for d in decs)])  # type: ignore[attr-defined]
    return node.lineno  # type: ignore[attr-defined]


def _const_or_attr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_forwarding_helper(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if node.name in ("__init__", "__new__", "__init_subclass__"):
        return True
    if len(node.body) == 1 and isinstance(node.body[0], ast.Return):
        val = node.body[0].value
        if isinstance(val, ast.Call) and len(val.keywords) >= 3:
            attr_kws = sum(
                1
                for kw in val.keywords
                if kw.arg and isinstance(kw.value, ast.Attribute)
            )
            return attr_kws >= 3
    return False


def _root_attr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        cur: ast.AST = node.value
        while isinstance(cur, ast.Attribute):
            cur = cur.value
        if isinstance(cur, ast.Name) and cur.id in ("self", "cls", "klass"):
            return node.attr
    return None


def _collect_truthy_or_membership_ids(node: ast.AST) -> set[int]:
    """Return AST node ids of expressions used purely for membership (in) or truthiness (bool)."""
    ids: set[int] = set()

    def mark_truthy(expr: ast.AST) -> None:
        if isinstance(expr, ast.BoolOp):
            for v in expr.values:
                mark_truthy(v)
        elif isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
            mark_truthy(expr.operand)
        elif isinstance(expr, (ast.Attribute, ast.Name)):
            ids.add(id(expr))

    for sub in ast.walk(node):
        if isinstance(sub, ast.Compare):
            for op, comp in zip(sub.ops, sub.comparators):
                if isinstance(op, (ast.In, ast.NotIn)) and isinstance(comp, (ast.Attribute, ast.Name)):
                    ids.add(id(comp))
        elif isinstance(sub, (ast.If, ast.While, ast.IfExp)):
            mark_truthy(sub.test)
        elif (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "bool"
            and len(sub.args) == 1
        ):
            mark_truthy(sub.args[0])
    return ids


class _SymbolCollector(ast.NodeVisitor):
    """Collect symbols, class/module attrs, attribute reads, and name references."""

    def __init__(self, rel_file: str) -> None:
        self.rel_file = rel_file
        self.scope_stack: list[tuple[str, str]] = []
        self.symbols: list[list[Any]] = []
        self.class_attrs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        self.module_attrs: dict[str, dict[str, Any]] = {}
        self.attr_reads: dict[str, set[int]] = defaultdict(set)
        self.runtime_attr_reads: dict[str, set[int]] = defaultdict(set)
        self.runtime_value_reads: dict[str, set[int]] = defaultdict(set)
        self.forwarded_attrs: list[tuple[str, str]] = []
        self.name_refs: dict[str, set[int]] = defaultdict(set)
        self.dialect_bases: list[tuple[str, list[str]]] = []
        self._skip_hoist_lines: set[int] = set()
        self._in_forwarding: bool = False
        self._non_value_ids: set[int] = set()

    def _cur_qualname(self, name: str) -> str:
        if not self.scope_stack:
            return name
        return ".".join([*(s[0] for s in self.scope_stack), name])

    def _in_func(self) -> bool:
        return any(s[1] == "func" for s in self.scope_stack)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qn = self._cur_qualname(node.name)
        start = _node_start_line(node)
        end = node.end_lineno or node.lineno
        self.symbols.append([qn, "class", start, end])
        bases = [b for base in node.bases if (b := _const_or_attr_name(base))]
        self.dialect_bases.append((node.name, bases))

        self.scope_stack.append((node.name, "class"))
        for stmt in node.body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                self._record_attr_assign(stmt, class_qn=qn)
            self.visit(stmt)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func(node)

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qn = self._cur_qualname(node.name)
        kind = "method" if self.scope_stack and self.scope_stack[-1][1] == "class" else "function"
        start = _node_start_line(node)
        end = node.end_lineno or node.lineno
        self.symbols.append([qn, kind, start, end])
        self.scope_stack.append((node.name, "func"))

        prev_fwd = self._in_forwarding
        prev_skip = self._skip_hoist_lines
        prev_nv = self._non_value_ids

        if _is_forwarding_helper(node):
            self._in_forwarding = True
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    for kw in sub.keywords:
                        if kw.arg:
                            src_attr = _root_attr_name(kw.value)
                            if src_attr:
                                self.forwarded_attrs.append((src_attr, kw.arg))
            self.generic_visit(node)
        else:
            self._in_forwarding = False
            non_val_ids = _collect_truthy_or_membership_ids(node)
            self._non_value_ids = non_val_ids

            hoisted: dict[str, tuple[str, int]] = {}
            hoist_lines: set[int] = set()
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Attribute)
                    and isinstance(stmt.value.value, ast.Name)
                    and stmt.value.value.id in ("self", "cls")
                ):
                    hoisted[stmt.targets[0].id] = (stmt.value.attr, stmt.lineno)
                    hoist_lines.add(stmt.lineno)

            hoist_uses: dict[str, set[int]] = defaultdict(set)
            hoist_val_uses: dict[str, set[int]] = defaultdict(set)
            if hoisted:
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Name)
                        and isinstance(sub.ctx, ast.Load)
                        and sub.id in hoisted
                        and sub.lineno not in hoist_lines
                    ):
                        hoist_uses[sub.id].add(sub.lineno)
                        if id(sub) not in non_val_ids:
                            hoist_val_uses[sub.id].add(sub.lineno)

            for local_name, (attr_name, def_ln) in hoisted.items():
                use_lines = hoist_uses.get(local_name)
                if use_lines:
                    self.runtime_attr_reads[attr_name].update(use_lines)
                else:
                    self.runtime_attr_reads[attr_name].add(def_ln)
                val_lines = hoist_val_uses.get(local_name)
                if val_lines:
                    self.runtime_value_reads[attr_name].update(val_lines)

            self._skip_hoist_lines = hoist_lines
            self.generic_visit(node)

        self._in_forwarding = prev_fwd
        self._skip_hoist_lines = prev_skip
        self._non_value_ids = prev_nv
        self.scope_stack.pop()

    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                self._record_attr_assign(stmt, class_qn=None)
            self.visit(stmt)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load) and self._in_func():
            self.attr_reads[node.attr].add(node.lineno)
            if not self._in_forwarding and node.lineno not in self._skip_hoist_lines:
                self.runtime_attr_reads[node.attr].add(node.lineno)
                if id(node) not in self._non_value_ids:
                    self.runtime_value_reads[node.attr].add(node.lineno)
        self.name_refs[node.attr].add(node.lineno)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if any(s[1] == "func" for s in self.scope_stack):
            self.name_refs[node.id].add(node.lineno)

    def _record_attr_assign(self, stmt: ast.Assign | ast.AnnAssign, class_qn: str | None) -> None:
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        value = stmt.value
        if value is None:
            return
        start = stmt.lineno
        end = stmt.end_lineno or stmt.lineno
        dict_keys: dict[str, tuple[int, int, str | None, bool]] = {}
        if isinstance(value, ast.Dict):
            for k_node, v_node in zip(value.keys, value.values):
                k_str = _expr_str_key(k_node)
                if k_str is not None and k_node is not None:
                    k_start = k_node.lineno
                    k_end = v_node.end_lineno or v_node.lineno
                    callee = _call_name(v_node)
                    k_callable = _contains_callable(v_node)
                    dict_keys[k_str] = (k_start, k_end, callee, k_callable)
        attr_callable = _contains_callable(value)
        for tgt in targets:
            if isinstance(tgt, ast.Name):
                entry = {
                    "start": start,
                    "end": end,
                    "dict_keys": dict_keys,
                    "callee": _call_name(value),
                    "has_callable": attr_callable,
                }
                if class_qn is not None:
                    self.class_attrs[class_qn][tgt.id] = entry
                else:
                    self.module_attrs[tgt.id] = entry


def _contains_callable(node: ast.AST) -> bool:
    return any(isinstance(n, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(node))


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        return _const_or_attr_name(node.func)
    return None


def _static_fixture_scan(
    sqlglot_dir: Path,
    py_files: list[tuple[str, Path]],
    trees: dict[str, ast.Module],
    id_to_num: dict[str, int],
    fixture_to_tests: dict[str, set[int]],
) -> None:
    """Supplement runtime audit hooks with static AST fixture dependencies."""
    all_fixtures = [
        p.relative_to(sqlglot_dir).as_posix()
        for p in sorted((sqlglot_dir / "tests" / "fixtures").rglob("*"))
        if p.is_file()
    ]
    tpch_files = [f for f in all_fixtures if f.startswith("tests/fixtures/optimizer/tpc-h/")]
    tpcds_files = [f for f in all_fixtures if f.startswith("tests/fixtures/optimizer/tpc-ds/")]

    for rel, tree in trees.items():
        if not rel.startswith("tests/"):
            continue
        for cls_node in ast.walk(tree):
            if not isinstance(cls_node, ast.ClassDef):
                continue
            for item in cls_node.body:
                if not isinstance(item, ast.FunctionDef) or not item.name.startswith("test_"):
                    continue
                nodeid = f"{rel}::{cls_node.name}::{item.name}"
                test_num = id_to_num.get(nodeid)
                if test_num is None:
                    continue
                for sub in ast.walk(item):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                        if sub.func.attr == "check_file" and sub.args:
                            arg0 = _expr_str_key(sub.args[0])
                            if arg0:
                                fpath = f"tests/fixtures/optimizer/{arg0}.sql"
                                if fpath in all_fixtures:
                                    fixture_to_tests[fpath].add(test_num)
                        elif sub.func.attr in ("load_sql_fixtures", "load_sql_fixture_pairs"):
                            pass
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                        if sub.func.id in ("load_sql_fixtures", "load_sql_fixture_pairs") and sub.args:
                            arg0 = _expr_str_key(sub.args[0])
                            if arg0:
                                fpath = f"tests/fixtures/{arg0}"
                                if fpath in all_fixtures:
                                    fixture_to_tests[fpath].add(test_num)
                    if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "self":
                        if "tpch" in sub.attr:
                            for f in tpch_files:
                                fixture_to_tests[f].add(test_num)
                        elif "tpcds" in sub.attr:
                            for f in tpcds_files:
                                fixture_to_tests[f].add(test_num)
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        if sub.value == "cts.json":
                            fixture_to_tests["tests/fixtures/jsonpath/cts.json"].add(test_num)


def load_reused_coverage(
    reuse_dir: Path,
    sqlglot_dir: Path,
) -> tuple[
    dict[str, dict[int, set[int]]],
    dict[str, set[int]],
    dict[str, set[int]],
    set[int],
]:
    """Load existing coverage maps and supplement ProcessPoolExecutor pickle paths."""
    import coverage
    from coverage.numbits import numbits_to_nums

    cov_raw = json.loads((reuse_dir / "coverage_map.json").read_text(encoding="utf-8"))
    ti_raw = json.loads((reuse_dir / "test_index.json").read_text(encoding="utf-8"))

    decoded_sets: list[set[int]] = []
    for enc in cov_raw["sets"]:
        s: set[int] = set()
        for item in enc:
            if isinstance(item, list):
                s.update(range(item[0], item[1] + 1))
            else:
                s.add(item)
        decoded_sets.append(s)

    line_to_tests: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for rel, ln_dict in cov_raw["line_to_set"].items():
        for ln_str, sid in ln_dict.items():
            line_to_tests[rel][int(ln_str)].update(decoded_sets[sid])

    import_time_lines: dict[str, set[int]] = defaultdict(set)
    for rel, lns in cov_raw.get("import_time_lines", {}).items():
        import_time_lines[rel].update(lns)

    fixture_to_tests: dict[str, set[int]] = defaultdict(set)
    for fpath, nums in ti_raw.get("fixture_to_test_numbers", {}).items():
        fixture_to_tests[fpath].update(nums)

    source_inspecting_tests = set(ti_raw.get("source_inspecting_tests", []))

    pool_tests: set[int] = set()
    pool_tests.update(line_to_tests.get("tests/test_optimizer.py", {}).get(259, set()))
    pool_tests.update(line_to_tests.get("tests/test_executor.py", {}).get(181, set()))

    if pool_tests:
        os.environ.pop("COVERAGE_CORE", None)
        sqlglot_root_str = str(sqlglot_dir)
        if sys.path[0] != sqlglot_root_str:
            sys.path.insert(0, sqlglot_root_str)
        import sqlglot
        from sqlglot.optimizer.annotate_types import annotate_types

        sample_expr = annotate_types(sqlglot.parse_one("SELECT CAST(1 AS INT) AS a, x FROM y /* c */"))
        sample_expr.meta["k"] = 1

        with tempfile.TemporaryDirectory() as td:
            rc_path = os.path.join(td, ".coveragerc")
            data_path = os.path.join(td, ".coverage")
            Path(rc_path).write_text("[run]\ncore = ctrace\nbranch = False\n", encoding="utf-8")
            cov = coverage.Coverage(
                config_file=rc_path,
                data_file=data_path,
                include=[f"{sqlglot_root_str}/sqlglot/*"],
            )
            cov.start()
            cov.switch_context("pickle_roundtrip")
            _ = pickle.loads(pickle.dumps(sample_expr))
            cov.stop()
            cov.save()

            conn = sqlite3.connect(data_path)
            try:
                fmap = {
                    row[0]: _normalize_rel_path(row[1], sqlglot_root_str)
                    for row in conn.execute("SELECT id, path FROM file")
                }
                for fid, numbits in conn.execute("SELECT file_id, numbits FROM line_bits"):
                    rel_f = fmap.get(fid)
                    if not rel_f:
                        continue
                    for ln in numbits_to_nums(numbits):
                        line_to_tests[rel_f][ln].update(pool_tests)
            finally:
                conn.close()

    return line_to_tests, import_time_lines, fixture_to_tests, source_inspecting_tests


def _compile_plan_signature(pe: Any, plan: Any, exp_mod: Any) -> tuple[Any, ...]:
    memo: dict[int, tuple[Any, ...]] = {}

    def gen_str(expr: Any) -> str | None:
        if not expr:
            return None
        replaced = pe._replace_subqueries(expr)
        return pe.generator.generate(replaced)

    def gen_tuple(exprs: Any) -> tuple[str | None, ...]:
        if not exprs:
            return ()
        return tuple(gen_str(e) for e in exprs)

    def alias_tuple(exprs: Any) -> tuple[str, ...]:
        if not exprs:
            return ()
        return tuple(e.alias_or_name if isinstance(e, exp_mod.Expr) else str(e) for e in exprs)

    def visit_step(node: Any) -> tuple[Any, ...]:
        nid = id(node)
        if nid in memo:
            return memo[nid]
        ntype = type(node).__name__
        cond = gen_str(getattr(node, "condition", None))
        projs_raw = getattr(node, "projections", None)
        projs = gen_tuple(projs_raw)
        proj_names = alias_tuple(projs_raw)
        src = getattr(node, "source", None)
        src_alias = src.alias_or_name if isinstance(src, exp_mod.Expr) else str(src)
        extra: list[Any] = []
        if ntype == "Join":
            for jname, jinfo in sorted(node.joins.items()):
                extra.append(
                    (
                        jname,
                        jinfo.get("side"),
                        gen_str(jinfo.get("condition")),
                        gen_tuple(jinfo.get("source_key")),
                        gen_tuple(jinfo.get("join_key")),
                    )
                )
        elif ntype == "Aggregate":
            ops_raw = getattr(node, "operands", None)
            extra.append(
                (
                    tuple((k, gen_str(v)) for k, v in node.group.items()),
                    gen_tuple(node.aggregations),
                    alias_tuple(node.aggregations),
                    gen_tuple(ops_raw),
                    alias_tuple(ops_raw),
                )
            )
        elif ntype == "Sort":
            extra.append(gen_tuple(node.key))
        elif ntype == "SetOperation":
            extra.append(
                (
                    getattr(node.op, "__name__", str(node.op)),
                    node.left,
                    node.right,
                    node.distinct,
                )
            )
        deps = tuple(sorted(visit_step(d) for d in node.dependencies))
        res = (
            ntype,
            node.name,
            str(src),
            src_alias,
            getattr(node, "limit", None),
            getattr(node, "offset", None),
            cond,
            projs,
            proj_names,
            tuple(extra),
            deps,
        )
        memo[nid] = res
        return res

    root_sig = visit_step(plan.root)
    sub_sigs = []
    for sq_name in sorted(list(pe._subquery_plans.keys())):
        sq_plan, _ = pe._subquery_plans[sq_name]
        sub_sigs.append((sq_name, visit_step(sq_plan.root)))
    return (root_sig, tuple(sub_sigs))


def compute_tpcds_plan_hashes(sqlglot_dir: Path) -> list[str]:
    """Precompute 128-bit SHA-256 plan signatures for TPC-DS queries from pinned sqlglot@eb11aec."""
    sqlglot_root_str = str(sqlglot_dir)
    if sys.path[0] != sqlglot_root_str:
        sys.path.insert(0, sqlglot_root_str)
    prev_cwd = os.getcwd()
    os.chdir(sqlglot_root_str)
    try:
        from sqlglot import exp, find_tables, parse_one, transpile
        from sqlglot.executor.python import PythonExecutor
        from sqlglot.optimizer import optimize
        from sqlglot.planner import Plan
        from sqlglot.schema import ensure_schema
        from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs

        sqls = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
        schema_ = ensure_schema(TPCDS_SCHEMA, dialect=None)
        sigs: list[str] = []
        for meta, sql, _ in sqls:
            expr = parse_one(sql)
            p_expr, p_meta = pickle.loads(pickle.dumps((expr, meta)))
            if not p_meta.get("execute"):
                sigs.append("")
                continue
            tables = sorted(t.name for t in find_tables(p_expr))
            opt_expr = optimize(p_expr, schema_, leave_tables_isolated=True, dialect=None)
            plan = Plan(opt_expr)
            pe = PythonExecutor()
            plan_sig = _compile_plan_signature(pe, plan, exp)
            duck_sql = transpile(sql, write="duckdb")[0]
            payload = repr((sql, tables, opt_expr.sql(), plan_sig, duck_sql)).encode("utf-8")
            sigs.append(hashlib.sha256(payload).hexdigest()[:32])
        return sigs
    finally:
        os.chdir(prev_cwd)


def build_maps(
    sqlglot_commit: str,
    out_dir: Path,
    sqlglot_dir: Path,
    tests_json_path: Path,
    reuse_coverage_from: Path | None = None,
) -> dict[str, Any]:
    """Build and write all offline maps to out_dir."""
    sqlglot_dir = ensure_sqlglot_checkout(sqlglot_dir, sqlglot_commit)
    tests, id_to_num = load_test_metadata(tests_json_path)

    if reuse_coverage_from is not None:
        line_to_tests, import_time_lines, fixture_to_tests, source_inspecting_tests = (
            load_reused_coverage(reuse_coverage_from, sqlglot_dir)
        )
    else:
        line_to_tests, import_time_lines, fixture_to_tests, source_inspecting_tests = (
            collect_runtime_coverage_and_fixtures(sqlglot_dir, id_to_num)
        )

    py_files = iter_python_files(sqlglot_dir)
    mod_to_file = resolve_module_index(py_files)
    trees: dict[str, ast.Module] = {}
    for rel, path in py_files:
        trees[rel] = ast.parse(path.read_text(encoding="utf-8"), filename=rel)

    _static_fixture_scan(sqlglot_dir, py_files, trees, id_to_num, fixture_to_tests)

    interner = SetInterner()

    file_to_tests: dict[str, set[int]] = defaultdict(set)
    for rel, ln_map in line_to_tests.items():
        agg: set[int] = set()
        for tset in ln_map.values():
            agg.update(tset)
        file_to_tests[rel] = agg

    collectors: dict[str, _SymbolCollector] = {}
    global_attr_reads: dict[str, set[int]] = defaultdict(set)
    global_runtime_attr_reads: dict[str, set[int]] = defaultdict(set)
    global_runtime_value_reads: dict[str, set[int]] = defaultdict(set)
    forwarded_edges: set[tuple[str, str]] = set()
    global_name_refs: dict[str, set[int]] = defaultdict(set)
    class_to_dialect_family: dict[str, str] = {}
    dialect_edges: list[tuple[str, str]] = []

    for rel, tree in trees.items():
        col = _SymbolCollector(rel)
        col.visit(tree)
        collectors[rel] = col
        ln_map = line_to_tests.get(rel, {})
        for attr, lines in col.attr_reads.items():
            for ln in lines:
                global_attr_reads[attr].update(ln_map.get(ln, ()))
        for attr, lines in col.runtime_attr_reads.items():
            for ln in lines:
                global_runtime_attr_reads[attr].update(ln_map.get(ln, ()))
        for attr, lines in col.runtime_value_reads.items():
            for ln in lines:
                global_runtime_value_reads[attr].update(ln_map.get(ln, ()))
        forwarded_edges.update(col.forwarded_attrs)
        for name, lines in col.name_refs.items():
            if rel.startswith("sqlglot/") or rel.startswith("tests/"):
                for ln in lines:
                    global_name_refs[name].update(ln_map.get(ln, ()))
        fam = dialect_family_for_file(rel)
        if fam and rel.startswith("sqlglot/dialects/"):
            for cls_name, bases in col.dialect_bases:
                class_to_dialect_family[cls_name] = fam
                for base in bases:
                    dialect_edges.append((base, cls_name))

    effective_runtime_reads: dict[str, set[int]] = {
        k: set(v) for k, v in global_runtime_attr_reads.items()
    }
    effective_value_reads: dict[str, set[int]] = {
        k: set(v) for k, v in global_runtime_value_reads.items()
    }
    for src_attr, dst_attr in forwarded_edges:
        dst_reads = global_runtime_attr_reads.get(dst_attr)
        if dst_reads:
            effective_runtime_reads.setdefault(src_attr, set()).update(dst_reads)
        dst_val_reads = global_runtime_value_reads.get(dst_attr)
        if dst_val_reads:
            effective_value_reads.setdefault(src_attr, set()).update(dst_val_reads)

    file_to_test_numbers: dict[str, list[int]] = defaultdict(list)
    for t in tests:
        file_to_test_numbers[t["file"]].append(t["number"])
    for tf in file_to_test_numbers:
        file_to_test_numbers[tf].sort()

    family_subclasses: dict[str, set[str]] = defaultdict(set)
    direct_children: dict[str, set[str]] = defaultdict(set)
    for base, child in dialect_edges:
        b_fam = class_to_dialect_family.get(base)
        c_fam = class_to_dialect_family.get(child)
        if b_fam and c_fam and b_fam != c_fam:
            direct_children[b_fam].add(c_fam)

    for fam in set(class_to_dialect_family.values()):
        seen: set[str] = {fam}
        q = deque([fam])
        while q:
            cur = q.popleft()
            for nxt in direct_children.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        seen.discard(fam)
        family_subclasses[fam] = seen

    dialect_family_tests: dict[str, set[int]] = defaultdict(set)
    for rel, _ in py_files:
        fam = dialect_family_for_file(rel)
        if fam:
            dialect_family_tests[fam].update(file_to_tests.get(rel, ()))
            dialect_family_tests[fam].update(
                file_to_test_numbers.get(f"tests/dialects/test_{fam}.py", ())
            )

    dialect_family_with_subs: dict[str, set[int]] = {}
    for fam in set(dialect_family_tests.keys()) | set(family_subclasses.keys()):
        agg = set(dialect_family_tests.get(fam, ()))
        for sub in family_subclasses.get(fam, ()):
            agg.update(dialect_family_tests.get(sub, ()))
        dialect_family_with_subs[fam] = agg

    imports_map: dict[str, list[str]] = {}
    reverse_imports_map: dict[str, list[str]] = defaultdict(list)
    for rel, tree in trees.items():
        imps = extract_imports(tree, rel, mod_to_file)
        imports_map[rel] = imps
        for dep in imps:
            reverse_imports_map[dep].append(rel)

    test_files_set = {
        rel for rel, _ in py_files if rel.startswith("tests/") and Path(rel).name.startswith("test_")
    }

    import_graph_modules: dict[str, Any] = {}
    for rel, _ in py_files:
        rev = sorted(reverse_imports_map.get(rel, ()))
        visited = {rel}
        q = deque([rel])
        t_files: set[str] = set()
        if rel in test_files_set:
            t_files.add(rel)
        while q:
            cur = q.popleft()
            for parent in reverse_imports_map.get(cur, ()):
                if parent not in visited:
                    visited.add(parent)
                    q.append(parent)
                    if parent in test_files_set:
                        t_files.add(parent)
        import_graph_modules[rel] = {
            "imports": imports_map[rel],
            "reverse_imports": rev,
            "transitive_test_files": sorted(t_files),
            "dialect_family": dialect_family_for_file(rel),
        }

    # Map every symbol and line in coverage_map.json
    line_to_set: dict[str, dict[str, int]] = {}
    for rel in sorted(line_to_tests):
        ln_dict: dict[str, int] = {}
        for ln, tset in sorted(line_to_tests[rel].items()):
            sid = interner.intern(tset)
            if sid != 0:
                ln_dict[str(ln)] = sid
        if ln_dict:
            line_to_set[rel] = ln_dict

    qualname_to_set: dict[str, dict[str, int]] = {}
    file_to_set: dict[str, int] = {}
    ast_index_out: dict[str, Any] = {}

    for rel, _ in py_files:
        col = collectors[rel]
        ln_map = line_to_tests.get(rel, {})
        qn_sets: dict[str, int] = {}
        for qn, _kind, start, end in col.symbols:
            tset: set[int] = set()
            for ln in range(start, end + 1):
                tset.update(ln_map.get(ln, ()))
            sid = interner.intern(tset)
            if sid != 0:
                qn_sets[qn] = sid
        if qn_sets:
            qualname_to_set[rel] = qn_sets

        fsid = interner.intern(file_to_tests.get(rel, ()))
        if fsid != 0:
            file_to_set[rel] = fsid

        fam = dialect_family_for_file(rel)
        fam_tests = dialect_family_with_subs.get(fam, set()) if fam else set()
        trans_tests: set[int] = set()
        for tf in import_graph_modules[rel]["transitive_test_files"]:
            trans_tests.update(file_to_test_numbers.get(tf, ()))

        def resolve_attr_tests(
            attr_name: str,
            start: int,
            end: int,
            cls_leaf: str | None,
            callee: str | None,
            has_callable: bool,
        ) -> set[int]:
            if has_callable:
                span_tests: set[int] = set()
                for ln in range(start, end + 1):
                    span_tests.update(ln_map.get(ln, ()))
                if span_tests:
                    return span_tests
            reader_tests = effective_runtime_reads.get(attr_name) or global_attr_reads.get(
                attr_name, set()
            )
            if fam_tests:
                if reader_tests:
                    inter = reader_tests & fam_tests
                    if inter:
                        return inter
                return fam_tests
            if reader_tests and len(reader_tests) <= 100:
                return set(reader_tests)
            if cls_leaf:
                cls_refs = global_name_refs.get(cls_leaf, set())
                if cls_refs:
                    return cls_refs
            if reader_tests and file_to_tests.get(rel):
                inter = reader_tests & file_to_tests[rel]
                if inter:
                    return inter
            return file_to_tests.get(rel) or trans_tests

        def build_attr_entry(
            attr_name: str, info: dict[str, Any], cls_leaf: str | None
        ) -> dict[str, Any]:
            attr_tests = resolve_attr_tests(
                attr_name,
                info["start"],
                info["end"],
                cls_leaf,
                info["callee"],
                info["has_callable"],
            )
            val_readers = effective_value_reads.get(attr_name)
            if val_readers:
                if fam_tests:
                    inter = val_readers & fam_tests
                    dict_val_tests = inter if inter else attr_tests
                elif len(val_readers) <= 100:
                    dict_val_tests = set(val_readers)
                else:
                    dict_val_tests = val_readers & attr_tests or attr_tests
            else:
                dict_val_tests = attr_tests
            dk_out: dict[str, list[int]] = {}
            for k_str, (ks, ke, _k_callee, k_callable) in sorted(info["dict_keys"].items()):
                k_span: set[int] = set()
                if k_callable:
                    for ln in range(ks, ke + 1):
                        k_span.update(ln_map.get(ln, ()))
                k_tests = k_span if k_span else dict_val_tests
                dk_out[k_str] = [ks, ke, interner.intern(k_tests)]
            return {
                "start": info["start"],
                "end": info["end"],
                "set_id": interner.intern(attr_tests),
                "dict_keys": dk_out,
            }

        class_attrs_out: dict[str, Any] = {
            cls_qn: {
                attr_name: build_attr_entry(attr_name, info, cls_qn.split(".")[-1])
                for attr_name, info in sorted(attrs.items())
            }
            for cls_qn, attrs in sorted(col.class_attrs.items())
        }
        module_attrs_out: dict[str, Any] = {
            attr_name: build_attr_entry(attr_name, info, None)
            for attr_name, info in sorted(col.module_attrs.items())
        }

        ast_index_out[rel] = {
            "symbols": col.symbols,
            "class_attrs": class_attrs_out,
            "module_attrs": module_attrs_out,
        }

    dialect_families_out = {
        fam: interner.intern(tset) for fam, tset in sorted(dialect_family_with_subs.items())
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    tpcds_plan_hashes = compute_tpcds_plan_hashes(sqlglot_dir)

    test_index_data = {
        "sqlglot_commit": sqlglot_commit,
        "tests": tests,
        "file_to_test_numbers": dict(sorted(file_to_test_numbers.items())),
        "fixture_to_test_numbers": {
            k: sorted(v) for k, v in sorted(fixture_to_tests.items())
        },
        "source_inspecting_tests": sorted(source_inspecting_tests),
        "tpcds_plan_hashes": tpcds_plan_hashes,
    }

    coverage_map_data = {
        "sets": interner.sets,
        "line_to_set": line_to_set,
        "import_time_lines": {
            k: sorted(v) for k, v in sorted(import_time_lines.items()) if v
        },
        "qualname_to_set": qualname_to_set,
        "file_to_set": file_to_set,
    }

    import_graph_data = {
        "modules": import_graph_modules,
        "dialect_families": dialect_families_out,
        "dialect_subclasses": {
            k: sorted(v) for k, v in sorted(family_subclasses.items()) if v
        },
    }

    _write_compact_json(out_dir / "test_index.json", test_index_data)
    _write_compact_json(out_dir / "coverage_map.json", coverage_map_data)
    _write_compact_json(out_dir / "ast_index.json", ast_index_out)
    _write_compact_json(out_dir / "import_graph.json", import_graph_data)

    return {
        "sets": len(interner.sets),
        "covered_files": len(line_to_set),
        "fixtures": len(fixture_to_tests),
        "source_inspecting": len(source_inspecting_tests),
    }


def _write_compact_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build offline blast-radius maps for sqlglot.")
    parser.add_argument("--sqlglot-commit", default=PINNED_COMMIT)
    parser.add_argument("--out-dir", type=Path, default=Path("maps"))
    parser.add_argument("--sqlglot-dir", type=Path, default=Path("../ai-notes/sqlglot"))
    parser.add_argument(
        "--tests-json",
        type=Path,
        default=Path("../sample-kit/sample/tests.json"),
    )
    parser.add_argument("--reuse-coverage-from", type=Path, default=None)
    args = parser.parse_args()
    summary = build_maps(
        sqlglot_commit=args.sqlglot_commit,
        out_dir=args.out_dir,
        sqlglot_dir=args.sqlglot_dir,
        tests_json_path=args.tests_json,
        reuse_coverage_from=args.reuse_coverage_from,
    )
    print(f"Built maps: {summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
