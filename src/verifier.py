"""Coverage-guided, cost-aware in-container test verifier using 2-tier forked /tmp overlays."""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib
import json
import os
import pickle
import resource
import select
import shutil
import signal
import sys
import tempfile
import time
import types
import unittest
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from diff_analyzer import CommitAnalysis
from scheduler import compute_commit_cost_ms, schedule_tests

_STDLIB_PRELOAD_NAMES = (
    "ast",
    "collections",
    "copy",
    "csv",
    "dataclasses",
    "datetime",
    "decimal",
    "enum",
    "functools",
    "gc",
    "gzip",
    "inspect",
    "io",
    "itertools",
    "logging",
    "math",
    "operator",
    "re",
    "sqlite3",
    "string",
    "subprocess",
    "textwrap",
    "traceback",
    "tracemalloc",
    "typing",
    "warnings",
    "weakref",
)

PRELOADED_SQLGLOT_FILES: frozenset[str] = frozenset(
    {
        "sqlglot/__init__.py",
        "sqlglot/_typing.py",
        "sqlglot/dialects/__init__.py",
        "sqlglot/dialects/dialect.py",
        "sqlglot/diff.py",
        "sqlglot/errors.py",
        "sqlglot/expressions/__init__.py",
        "sqlglot/expressions/aggregate.py",
        "sqlglot/expressions/array.py",
        "sqlglot/expressions/builders.py",
        "sqlglot/expressions/constraints.py",
        "sqlglot/expressions/core.py",
        "sqlglot/expressions/datatypes.py",
        "sqlglot/expressions/ddl.py",
        "sqlglot/expressions/dml.py",
        "sqlglot/expressions/functions.py",
        "sqlglot/expressions/json.py",
        "sqlglot/expressions/math.py",
        "sqlglot/expressions/properties.py",
        "sqlglot/expressions/query.py",
        "sqlglot/expressions/string.py",
        "sqlglot/expressions/temporal.py",
        "sqlglot/generator.py",
        "sqlglot/helper.py",
        "sqlglot/jsonpath.py",
        "sqlglot/optimizer/__init__.py",
        "sqlglot/optimizer/scope.py",
        "sqlglot/parser.py",
        "sqlglot/parsers/__init__.py",
        "sqlglot/parsers/base.py",
        "sqlglot/schema.py",
        "sqlglot/time.py",
        "sqlglot/tokenizer_core.py",
        "sqlglot/tokens.py",
        "sqlglot/trie.py",
        "sqlglot/typing/__init__.py",
    }
)

_PRELOADED_TEST_DIALECT_FILES: frozenset[str] = frozenset(
    {
        "sqlglot/dialects/bigquery.py",
        "sqlglot/dialects/duckdb.py",
        "sqlglot/dialects/hive.py",
        "sqlglot/dialects/snowflake.py",
        "sqlglot/dialects/spark2.py",
        "sqlglot/generators/__init__.py",
        "sqlglot/generators/bigquery.py",
        "sqlglot/generators/duckdb.py",
        "sqlglot/generators/hive.py",
        "sqlglot/generators/snowflake.py",
        "sqlglot/generators/spark2.py",
        "sqlglot/optimizer/annotate_types.py",
        "sqlglot/parsers/bigquery.py",
        "sqlglot/parsers/duckdb.py",
        "sqlglot/parsers/hive.py",
        "sqlglot/parsers/snowflake.py",
        "sqlglot/parsers/spark2.py",
        "sqlglot/transforms.py",
        "sqlglot/typing/bigquery.py",
        "sqlglot/typing/duckdb.py",
        "sqlglot/typing/hive.py",
        "sqlglot/typing/snowflake.py",
        "sqlglot/typing/spark2.py",
        "tests/__init__.py",
        "tests/dialects/__init__.py",
        "tests/dialects/test_dialect.py",
        "tests/helpers.py",
    }
)

_TPCDS_EXEC_ONLY_FILES: frozenset[str] = frozenset(
    {
        "sqlglot/executor/__init__.py",
        "sqlglot/executor/context.py",
        "sqlglot/executor/env.py",
        "sqlglot/executor/python.py",
        "sqlglot/executor/table.py",
        "sqlglot/planner.py",
        "tests/helpers.py",
        "tests/test_executor.py",
    }
)

_TPC_EXECUTOR_METHODS: frozenset[str] = frozenset(
    {
        "test_optimized_tpch",
        "test_execute_tpch",
        "test_execute_tpcds",
    }
)

MAX_VERIFY_SINGLE_TEST_MS = 8_000
FORK_CLEAN_CPU_MS = 320
FORK_PRELOADED_CPU_MS = 85
PER_FILE_IMPORT_CPU_MS = 85
BIGQUERY_FILE_IMPORT_CPU_MS = 210
HEAVY_DEPS_AMORTIZED_CPU_MS = 120
OPTIMIZER_SETUP_CPU_MS = 240
EXECUTOR_TPCH_SETUP_CPU_MS = 220
FAST_TPCDS_CPU_MS = 3_200
MARGINAL_CPU_RATIO = 0.85
PRELOADED_MARGINAL_CPU_RATIO = 0.45
MEMORY_TEST_CPU_RATIO = 0.35
MEMORY_TEST_THRESHOLD_MS = 5_000
ROI_RATIO = 1.75
MIN_NET_GAIN_MS = 800
PER_COMMIT_EST_CPU_CAP_MS = 10_000
PER_COMMIT_WALL_TIMEOUT_S = 12.0
GLOBAL_WALL_BUDGET_S = 120.0
MAX_CONCURRENT_CHILDREN = 2


def _preload_stdlib_for_fork() -> None:
    """Pre-import stdlib modules once before forking; never imports sqlglot or tests."""
    for mod_name in _STDLIB_PRELOAD_NAMES:
        importlib.import_module(mod_name)


def can_use_preloaded_sqlglot(
    analysis: CommitAnalysis | None,
    sqlglot_root: Path | None = None,
) -> bool:
    """Return True when a commit does not touch any file imported eagerly by `import sqlglot`."""
    if sqlglot_root is not None and not (sqlglot_root / "sqlglot" / "__init__.py").is_file():
        return False
    if analysis is None:
        return False
    for fa in analysis.files:
        if fa.is_ast_noop and not fa.docstring_changed:
            continue
        if fa.is_deleted or fa.is_renamed or fa.syntax_broken or fa.is_new:
            return False
        if fa.rel_path in PRELOADED_SQLGLOT_FILES or fa.old_path in PRELOADED_SQLGLOT_FILES:
            return False
    return True


def can_use_preloaded_test_dialect(analysis: CommitAnalysis | None) -> bool:
    """Return True when a commit does not touch any file loaded by `tests.dialects.test_dialect`."""
    if analysis is None:
        return False
    for fa in analysis.files:
        if fa.is_ast_noop and not fa.docstring_changed:
            continue
        if fa.is_deleted or fa.is_renamed or fa.syntax_broken or fa.is_new:
            return False
        if (
            fa.rel_path in PRELOADED_SQLGLOT_FILES
            or fa.old_path in PRELOADED_SQLGLOT_FILES
            or fa.rel_path in _PRELOADED_TEST_DIALECT_FILES
            or fa.old_path in _PRELOADED_TEST_DIALECT_FILES
        ):
            return False
    return True


def can_skip_pickle_roundtrip(analysis: CommitAnalysis | None) -> bool:
    """Return True when a commit cannot affect serde or pickle serialization paths."""
    if analysis is None:
        return False
    for fa in analysis.files:
        if fa.is_ast_noop and not fa.docstring_changed:
            continue
        if fa.is_deleted or fa.is_renamed or fa.syntax_broken or fa.is_new:
            return False
        if fa.rel_path == "sqlglot/serde.py" or fa.rel_path.startswith("sqlglot/expressions/"):
            return False
        for qn in fa.changed_qualnames:
            leaf = qn.rsplit(".", 1)[-1]
            if leaf in ("__reduce__", "__reduce_ex__", "__getstate__", "__setstate__", "dump", "load"):
                return False
    return True


def can_fast_verify_tpcds(
    analysis: CommitAnalysis | None,
    tpcds_plan_hashes: tuple[str, ...] = (),
) -> bool:
    """Return True when #1106 test_execute_tpcds can be verified via precomputed plan hashes."""
    if not tpcds_plan_hashes or analysis is None:
        return False
    for fa in analysis.files:
        if fa.is_ast_noop and not fa.docstring_changed:
            continue
        if fa.is_deleted or fa.is_renamed or fa.syntax_broken or fa.is_new:
            return False
        if fa.rel_path in _TPCDS_EXEC_ONLY_FILES or fa.old_path in _TPCDS_EXEC_ONLY_FILES:
            return False
        if fa.rel_path.startswith("tests/fixtures/optimizer/tpc-ds/"):
            return False
    return True


class _SubtestAbort(BaseException):
    """Control-flow signal raised on first failing subtest to abort remaining subtests."""


class _FailFastResult(unittest.TestResult):
    """unittest TestResult that fails immediately on the first failing subtest."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def addError(self, test: unittest.TestCase, err: Any) -> None:
        super().addError(test, err)
        self.failed = True

    def addFailure(self, test: unittest.TestCase, err: Any) -> None:
        super().addFailure(test, err)
        self.failed = True

    def addUnexpectedSuccess(self, test: unittest.TestCase) -> None:
        super().addUnexpectedSuccess(test)
        self.failed = True

    def addSubTest(self, test: unittest.TestCase, subtest: Any, err: Any) -> None:
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.failed = True
            raise _SubtestAbort()


class _InlinePickleExecutor:
    """In-process ProcessPoolExecutor replacement preserving pickle roundtrip semantics."""

    skip_pickle: bool = False

    def __init__(
        self,
        max_workers: int | None = None,
        mp_context: Any = None,
        initializer: Any = None,
        initargs: tuple[Any, ...] = (),
    ) -> None:
        if initializer is not None:
            initializer(*initargs)

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> concurrent.futures.Future[Any]:
        fut: concurrent.futures.Future[Any] = concurrent.futures.Future()
        try:
            if self.skip_pickle:
                fut.set_result(fn(*args, **kwargs))
            else:
                p_fn, p_args, p_kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
                res = p_fn(*p_args, **p_kwargs)
                fut.set_result(pickle.loads(pickle.dumps(res)))
        except BaseException as exc:
            fut.set_exception(exc)
        return fut

    def map(self, fn: Any, *iterables: Any, timeout: float | None = None, chunksize: int = 1) -> list[Any]:
        return [fn(*args) for args in zip(*iterables)]

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        return None

    def __enter__(self) -> _InlinePickleExecutor:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class VerifyPlan:
    keep_unverified: frozenset[int]
    to_verify: frozenset[int]
    est_cpu_ms: int
    max_sim_gain_ms: int
    should_verify: bool
    use_preloaded_sqlglot: bool = False
    fast_tpcds: bool = False


@dataclass(frozen=True, slots=True)
class VerifyOutcome:
    final_tests: frozenset[int]
    verified: bool
    shed_count: int
    child_cpu_s: float
    wall_s: float
    timed_out: bool = False


def estimate_verify_cpu_ms(
    test_nums: frozenset[int] | set[int],
    ms_by_num: Mapping[int, int],
    file_by_num: Mapping[int, str],
    *,
    use_preloaded_sqlglot: bool = False,
    fast_tpcds_num: int | None = None,
    id_by_num: Mapping[int, str] | None = None,
) -> int:
    """Estimate child CPU cost in milliseconds from container profiling measurements."""
    if not test_nums:
        return 0
    overhead = FORK_PRELOADED_CPU_MS if use_preloaded_sqlglot else FORK_CLEAN_CPU_MS
    distinct_files: set[str] = set()
    has_opt = False
    has_exec_tpch = False
    body_ms = 0
    for n in test_nums:
        tfile = file_by_num[n]
        distinct_files.add(tfile)
        if tfile == "tests/test_optimizer.py":
            has_opt = True
        elif tfile == "tests/test_executor.py" and id_by_num is not None:
            meth = id_by_num[n].rsplit("::", 1)[-1]
            if meth in ("test_optimized_tpch", "test_execute_tpch"):
                has_exec_tpch = True
        if fast_tpcds_num is not None and n == fast_tpcds_num:
            body_ms += FAST_TPCDS_CPU_MS
        else:
            dur = ms_by_num[n]
            base_ratio = PRELOADED_MARGINAL_CPU_RATIO if use_preloaded_sqlglot else MARGINAL_CPU_RATIO
            ratio = MEMORY_TEST_CPU_RATIO if dur >= MEMORY_TEST_THRESHOLD_MS else base_ratio
            body_ms += int(dur * ratio)

    overhead += len(distinct_files) * PER_FILE_IMPORT_CPU_MS
    if "tests/dialects/test_bigquery.py" in distinct_files:
        overhead += BIGQUERY_FILE_IMPORT_CPU_MS
    if has_opt or has_exec_tpch:
        overhead += HEAVY_DEPS_AMORTIZED_CPU_MS
    if has_opt:
        overhead += OPTIMIZER_SETUP_CPU_MS
    if has_exec_tpch:
        overhead += EXECUTOR_TPCH_SETUP_CPU_MS
    return overhead + body_ms


def plan_commit_verification(
    candidates: frozenset[int],
    ms_by_num: Mapping[int, int],
    file_by_num: Mapping[int, str],
    *,
    analysis: CommitAnalysis | None = None,
    sqlglot_root: Path | None = None,
    fixture_to_tests: Mapping[str, frozenset[int]] | None = None,
    id_by_num: Mapping[int, str] | None = None,
    tpcds_plan_hashes: tuple[str, ...] = (),
) -> VerifyPlan:
    """Decide whether verifying eligible candidates has positive expected ROI."""
    if not candidates:
        return VerifyPlan(frozenset(), frozenset(), 0, 0, False)

    use_preloaded = can_use_preloaded_sqlglot(analysis, sqlglot_root)
    fast_tpcds_ok = can_fast_verify_tpcds(analysis, tpcds_plan_hashes)

    direct_fixture_tests: set[int] = set()
    if analysis is not None and fixture_to_tests is not None:
        for fa in analysis.files:
            for p in (fa.old_path, fa.new_path):
                if p and p in fixture_to_tests:
                    direct_fixture_tests.update(fixture_to_tests[p] & candidates)

    fast_tpcds_num: int | None = None
    if fast_tpcds_ok and id_by_num is not None:
        for t in candidates:
            if id_by_num.get(t) == "tests/test_executor.py::TestExecutor::test_execute_tpcds":
                other_ms = sum(ms_by_num[n] for n in candidates if n != t)
                if other_ms <= 4_500:
                    fast_tpcds_num = t
                break

    keep_unverified = frozenset(
        t
        for t in candidates
        if t in direct_fixture_tests
        or (ms_by_num[t] > MAX_VERIFY_SINGLE_TEST_MS and t != fast_tpcds_num)
    )
    to_verify = candidates - keep_unverified
    if not to_verify:
        return VerifyPlan(keep_unverified, frozenset(), 0, 0, False, use_preloaded, False)

    cost_all = compute_commit_cost_ms(schedule_tests(candidates, ms_by_num), ms_by_num)
    cost_kept = compute_commit_cost_ms(schedule_tests(keep_unverified, ms_by_num), ms_by_num)
    max_sim_gain_ms = cost_all - cost_kept
    est_cpu_ms = estimate_verify_cpu_ms(
        to_verify,
        ms_by_num,
        file_by_num,
        use_preloaded_sqlglot=use_preloaded,
        fast_tpcds_num=fast_tpcds_num,
        id_by_num=id_by_num,
    )

    distinct_files = {file_by_num[n] for n in to_verify}
    has_opt = "tests/test_optimizer.py" in distinct_files
    sum_verify_ms = sum(ms_by_num[n] for n in to_verify)
    heavy_opt_ok = not (has_opt and sum_verify_ms >= 800 and max_sim_gain_ms < 5_000)

    should_verify = (
        heavy_opt_ok
        and est_cpu_ms <= PER_COMMIT_EST_CPU_CAP_MS
        and max_sim_gain_ms >= int(est_cpu_ms * ROI_RATIO)
        and (max_sim_gain_ms - est_cpu_ms) >= MIN_NET_GAIN_MS
    )
    return VerifyPlan(
        keep_unverified=keep_unverified,
        to_verify=to_verify,
        est_cpu_ms=est_cpu_ms,
        max_sim_gain_ms=max_sim_gain_ms,
        should_verify=should_verify,
        use_preloaded_sqlglot=use_preloaded,
        fast_tpcds=fast_tpcds_num is not None,
    )


def create_overlay(
    sqlglot_root: Path,
    patched_contents: Mapping[str, str | None],
    tmp_base: str = "/tmp",
) -> str:
    """Create a minimal symlink overlay in tmp_base copying only modified files."""
    overlay_dir = tempfile.mkdtemp(prefix="rwx_ov_", dir=tmp_base)
    patched_dirs: set[str] = set()
    patched_stems_by_dir: dict[str, set[str]] = {}
    for rel in patched_contents:
        norm = rel.strip("/")
        p = Path(norm)
        parent_rel = p.parent.as_posix()
        if parent_rel == ".":
            parent_rel = ""
        patched_stems_by_dir.setdefault(parent_rel, set()).add(p.stem)
        parts = p.parts
        for i in range(1, len(parts)):
            patched_dirs.add("/".join(parts[:i]))

    def populate(src_dir: Path, dst_dir: Path, rel_prefix: str) -> None:
        dst_dir.mkdir(parents=True, exist_ok=True)
        direct_stems = patched_stems_by_dir.get(rel_prefix)
        for entry in os.scandir(src_dir):
            if entry.name == "__pycache__":
                if not direct_stems:
                    os.symlink(entry.path, dst_dir / "__pycache__")
                continue
            rel = f"{rel_prefix}/{entry.name}" if rel_prefix else entry.name
            dst = dst_dir / entry.name
            if rel in patched_contents:
                content = patched_contents[rel]
                if content is not None:
                    dst.write_text(content, encoding="utf-8")
            elif rel in patched_dirs and entry.is_dir(follow_symlinks=False):
                populate(Path(entry.path), dst, rel)
            else:
                os.symlink(entry.path, dst)

    populate(sqlglot_root, Path(overlay_dir), "")

    for rel, content in patched_contents.items():
        if content is None:
            continue
        dst = Path(overlay_dir) / rel
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(content, encoding="utf-8")

    return overlay_dir


def redirect_preloaded_modules_to_overlay(sqlglot_root_str: str, overlay_dir: str) -> None:
    """Rewrite preloaded sqlglot package __path__, __file__, and __spec__ entries to overlay_dir."""
    root_clean = sqlglot_root_str.rstrip("/")
    ov_clean = overlay_dir.rstrip("/")
    root_prefix = root_clean + "/"
    ov_prefix = ov_clean + "/"

    sys.path[:] = [ov_clean if p.rstrip("/") == root_clean else p for p in sys.path]
    if not sys.path or sys.path[0] != ov_clean:
        sys.path.insert(0, ov_clean)

    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        mfile = getattr(mod, "__file__", None)
        if isinstance(mfile, str) and mfile.startswith(root_prefix):
            mod.__file__ = ov_prefix + mfile[len(root_prefix) :]
        mpath = getattr(mod, "__path__", None)
        if mpath is not None:
            for idx, entry in enumerate(list(mpath)):
                if isinstance(entry, str) and entry.startswith(root_prefix):
                    mpath[idx] = ov_prefix + entry[len(root_prefix) :]
        mspec = getattr(mod, "__spec__", None)
        if mspec is not None:
            origin = getattr(mspec, "origin", None)
            if isinstance(origin, str) and origin.startswith(root_prefix):
                mspec.origin = ov_prefix + origin[len(root_prefix) :]
            locs = getattr(mspec, "submodule_search_locations", None)
            if locs is not None:
                for idx, entry in enumerate(list(locs)):
                    if isinstance(entry, str) and entry.startswith(root_prefix):
                        locs[idx] = ov_prefix + entry[len(root_prefix) :]

    helpers_mod = sys.modules.get("tests.helpers")
    if helpers_mod is not None:
        fdir = getattr(helpers_mod, "FILE_DIR", None)
        if isinstance(fdir, str) and fdir.startswith(root_prefix):
            helpers_mod.FILE_DIR = ov_prefix + fdir[len(root_prefix) :]  # type: ignore[attr-defined]
        fixdir = getattr(helpers_mod, "FIXTURES_DIR", None)
        if isinstance(fixdir, str) and fixdir.startswith(root_prefix):
            helpers_mod.FIXTURES_DIR = ov_prefix + fixdir[len(root_prefix) :]  # type: ignore[attr-defined]

    sys.path_importer_cache.clear()
    importlib.invalidate_caches()


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


def _fast_verify_tpcds(tpcds_plan_hashes: tuple[str, ...], soft_deadline_s: float) -> bool:
    """Verify #1106 test_execute_tpcds against precomputed 128-bit SHA-256 golden plan hashes."""
    from sqlglot import exp, find_tables, parse_one, transpile
    from sqlglot.executor.python import PythonExecutor
    from sqlglot.optimizer import optimize
    from sqlglot.planner import Plan
    from sqlglot.schema import ensure_schema
    from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs

    sqls = list(load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
    if len(sqls) != len(tpcds_plan_hashes):
        return False

    schema_ = ensure_schema(TPCDS_SCHEMA, dialect=None)
    diff_queries: list[tuple[int, str, Any, list[str]]] = []

    for i, ((meta, sql, _), expected_hash) in enumerate(zip(sqls, tpcds_plan_hashes)):
        if time.monotonic() >= soft_deadline_s:
            return False
        expr = parse_one(sql)
        p_expr, p_meta = pickle.loads(pickle.dumps((expr, meta)))
        if not p_meta.get("execute"):
            if expected_hash != "":
                return False
            continue
        if not expected_hash:
            return False
        tables = sorted(t.name for t in find_tables(p_expr))
        opt_expr = optimize(p_expr, schema_, leave_tables_isolated=True, dialect=None)
        plan = Plan(opt_expr)
        pe = PythonExecutor()
        plan_sig = _compile_plan_signature(pe, plan, exp)
        duck_sql = transpile(sql, write="duckdb")[0]
        payload = repr((sql, tables, opt_expr.sql(), plan_sig, duck_sql)).encode("utf-8")
        actual_hash = hashlib.sha256(payload).hexdigest()[:32]
        if actual_hash != expected_hash:
            diff_queries.append((i, sql, p_expr, tables))
            if len(diff_queries) > 2:
                return False

    if not diff_queries:
        return True

    # At most 2 queries changed plans: execute only those queries against DuckDB reference.
    import ast as _ast
    import csv as _csv
    import duckdb as _duckdb
    import numpy as _np
    import pandas as _pd
    from pandas.testing import assert_frame_equal
    from sqlglot.executor import execute
    from sqlglot.executor.table import Table
    from tests.test_executor import DIR_TPCDS, open_file

    needed_tables = {t for _, _, _, tbls in diff_queries for t in tbls}
    conn = _duckdb.connect()
    loaded_tables: dict[str, Any] = {}
    try:
        for table_name in sorted(needed_tables):
            columns = TPCDS_SCHEMA[table_name]
            file_name = f"{DIR_TPCDS}{table_name}.csv.gz"
            conn.execute(
                f"CREATE VIEW {table_name} AS SELECT * FROM READ_CSV('{file_name}', "
                f"delim='|', header=True, columns={columns})"
            )
            reader = _csv.reader(open_file(file_name), delimiter="|")
            rows: list[Any] = []
            ctypes: list[Any] = []
            next(reader)
            for row in reader:
                if not ctypes:
                    for v in row:
                        try:
                            ctypes.append(type(_ast.literal_eval(v)))
                        except (ValueError, SyntaxError):
                            ctypes.append(str)
                rows.append(
                    tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row))
                )
            loaded_tables[table_name] = Table(columns=columns, rows=rows)

        for _i, sql, p_expr, tbls in diff_queries:
            if time.monotonic() >= soft_deadline_s:
                return False
            q_tables = {t: loaded_tables[t] for t in tbls}
            res_table = execute(p_expr, schema=TPCDS_SCHEMA, tables=q_tables)
            a = conn.execute(transpile(sql, write="duckdb")[0]).fetchdf()
            b = _pd.DataFrame(
                ((_np.nan if c is None else c for c in r) for r in res_table.rows),
                columns=res_table.columns,
            )
            assert_frame_equal(a, b, check_dtype=False, check_index_type=False)
        return True
    except BaseException:
        return False
    finally:
        conn.close()


class _LazyOptimizerConn:
    """Defer DuckDB initialization in TestOptimizer.setUpClass until a test executes a query."""

    def __init__(self, real_connect: Any) -> None:
        self._real_connect = real_connect
        self._setup_sqls: list[str] = []
        self._conn: Any = None
        self.in_setup: bool = True

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        if self.in_setup and self._conn is None:
            self._setup_sqls.append(sql)
            return self
        if self._conn is None:
            self._conn = self._real_connect(config={"threads": 1})
            for s in self._setup_sqls:
                self._conn.execute(s)
        return self._conn.execute(sql, *args, **kwargs)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()

    def __getattr__(self, name: str) -> Any:
        if self._conn is None:
            self._conn = self._real_connect(config={"threads": 1})
            for s in self._setup_sqls:
                self._conn.execute(s)
        return getattr(self._conn, name)


def _install_lazy_heavy_deps() -> None:
    """Defer importing duckdb, numpy, and pandas until a test actually uses them."""
    stubs: dict[str, types.ModuleType] = {}

    def _clear_stubs() -> None:
        for k, stub in list(stubs.items()):
            if sys.modules.get(k) is stub:
                sys.modules.pop(k, None)

    def _activate_numpy() -> Any:
        _clear_stubs()
        real_np = importlib.import_module("numpy")
        np_stub = stubs.get("numpy")
        if np_stub is not None:
            np_stub.__dict__.pop("__getattr__", None)
            np_stub.__dict__.update(real_np.__dict__)
        return real_np

    def _activate_pandas() -> Any:
        _activate_numpy()
        real_pd = importlib.import_module("pandas")
        real_testing = importlib.import_module("pandas.testing")
        pd_stub = stubs.get("pandas")
        if pd_stub is not None:
            pd_stub.__dict__.pop("__getattr__", None)
            pd_stub.__dict__.update(real_pd.__dict__)
        t_stub = stubs.get("pandas.testing")
        if t_stub is not None:
            t_stub.__dict__.update(real_testing.__dict__)
        return real_pd

    def _activate_duckdb() -> Any:
        _clear_stubs()
        real_duck = importlib.import_module("duckdb")
        d_stub = stubs.get("duckdb")
        if d_stub is not None:
            d_stub.__dict__.pop("__getattr__", None)
            d_stub.__dict__.update(real_duck.__dict__)
        return real_duck

    if "numpy" not in sys.modules:
        np_pkg = types.ModuleType("numpy")
        np_pkg.__path__ = []  # type: ignore[attr-defined]
        np_pkg.__getattr__ = lambda name: getattr(_activate_numpy(), name)  # type: ignore[attr-defined]
        stubs["numpy"] = np_pkg
        sys.modules["numpy"] = np_pkg

    if "pandas" not in sys.modules and "pandas.testing" not in sys.modules:
        def lazy_assert_frame_equal(*args: Any, **kwargs: Any) -> Any:
            _activate_pandas()
            from pandas.testing import assert_frame_equal as real_afe

            return real_afe(*args, **kwargs)

        pd_pkg = types.ModuleType("pandas")
        pd_pkg.__path__ = []  # type: ignore[attr-defined]
        pd_pkg.__getattr__ = lambda name: getattr(_activate_pandas(), name)  # type: ignore[attr-defined]
        testing_mod = types.ModuleType("pandas.testing")
        testing_mod.assert_frame_equal = lazy_assert_frame_equal  # type: ignore[attr-defined]
        pd_pkg.testing = testing_mod  # type: ignore[attr-defined]
        stubs["pandas"] = pd_pkg
        stubs["pandas.testing"] = testing_mod
        sys.modules["pandas"] = pd_pkg
        sys.modules["pandas.testing"] = testing_mod

    if "duckdb" not in sys.modules:
        duck_pkg = types.ModuleType("duckdb")
        duck_pkg.__path__ = []  # type: ignore[attr-defined]
        duck_pkg.__getattr__ = lambda name: getattr(_activate_duckdb(), name)  # type: ignore[attr-defined]
        stubs["duckdb"] = duck_pkg
        sys.modules["duckdb"] = duck_pkg


def _run_pytest_fallback(test_id: str) -> bool:
    import pytest

    code = pytest.main([test_id, "-q", "-x", "--no-header", "--disable-warnings"])
    return code == 0


def _child_execute(
    sqlglot_root_str: str,
    overlay_dir: str,
    test_specs: list[tuple[int, str]],
    soft_deadline_s: float,
    *,
    patched_files: frozenset[str] = frozenset(),
    skip_pickle: bool = False,
    fast_tpcds: bool = False,
    tpcds_plan_hashes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Execute candidate tests inside an isolated forked child process."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
    finally:
        os.close(devnull_fd)
    sys.stdout = open(os.devnull, "w", encoding="utf-8")

    os.chdir(overlay_dir)
    redirect_preloaded_modules_to_overlay(sqlglot_root_str, overlay_dir)
    os.environ["PYTHONPATH"] = overlay_dir
    os.environ["SKIP_INTEGRATION"] = "0"
    _InlinePickleExecutor.skip_pickle = skip_pickle
    concurrent.futures.ProcessPoolExecutor = _InlinePickleExecutor  # type: ignore[misc]

    passed: list[int] = []
    import_error_files: list[str] = []

    remaining_specs: list[tuple[int, str]] = []
    tpcds_spec: tuple[int, str] | None = None
    for test_num, test_id in test_specs:
        if (
            fast_tpcds
            and tpcds_plan_hashes
            and test_id == "tests/test_executor.py::TestExecutor::test_execute_tpcds"
        ):
            tpcds_spec = (test_num, test_id)
        else:
            remaining_specs.append((test_num, test_id))

    if tpcds_spec is not None:
        try:
            if _fast_verify_tpcds(tpcds_plan_hashes, soft_deadline_s):
                passed.append(tpcds_spec[0])
            else:
                # Heavy-first gate: if #1106 (21,454ms) fails, abort remaining tests
                # rather than burning CPU on a commit whose max worker load is already 21.454s.
                ru_self = resource.getrusage(resource.RUSAGE_SELF)
                ru_kids = resource.getrusage(resource.RUSAGE_CHILDREN)
                return {
                    "passed": [],
                    "import_error_files": [],
                    "cpu_s": ru_self.ru_utime + ru_self.ru_stime + ru_kids.ru_utime + ru_kids.ru_stime,
                }
        except BaseException:
            ru_self = resource.getrusage(resource.RUSAGE_SELF)
            ru_kids = resource.getrusage(resource.RUSAGE_CHILDREN)
            return {
                "passed": [],
                "import_error_files": [],
                "cpu_s": ru_self.ru_utime + ru_self.ru_stime + ru_kids.ru_utime + ru_kids.ru_stime,
            }

    by_file_cls: dict[str, dict[str, list[tuple[int, str, str]]]] = {}
    for test_num, test_id in remaining_specs:
        parts = test_id.split("::")
        tfile = parts[0]
        cls_name = parts[1] if len(parts) == 3 else ""
        meth_name = parts[2] if len(parts) == 3 else parts[-1]
        by_file_cls.setdefault(tfile, {}).setdefault(cls_name, []).append(
            (test_num, meth_name, test_id)
        )

    can_filter_executor_setup = (
        "tests/test_executor.py" not in patched_files and "tests/helpers.py" not in patched_files
    )
    can_lazy_optimizer_setup = "tests/test_optimizer.py" not in patched_files

    for tfile in sorted(by_file_cls):
        if time.monotonic() >= soft_deadline_s:
            break
        if (
            tfile in ("tests/test_optimizer.py", "tests/test_executor.py")
            and can_lazy_optimizer_setup
            and can_filter_executor_setup
        ):
            _install_lazy_heavy_deps()
        mod_name = tfile[:-3].replace("/", ".")
        try:
            mod = importlib.import_module(mod_name)
        except BaseException:
            import_error_files.append(tfile)
            continue

        for cls_name in sorted(by_file_cls[tfile]):
            if time.monotonic() >= soft_deadline_s:
                break
            methods = by_file_cls[tfile][cls_name]
            cls = getattr(mod, cls_name, None) if cls_name else None
            if cls is None or not isinstance(cls, type) or not issubclass(cls, unittest.TestCase):
                for test_num, _meth, test_id in methods:
                    if time.monotonic() >= soft_deadline_s:
                        break
                    try:
                        if _run_pytest_fallback(test_id):
                            passed.append(test_num)
                    except BaseException:
                        pass
                continue

            run_setup_class = True
            orig_tpcds_schema = None
            orig_tpch_schema = None
            if (
                can_filter_executor_setup
                and tfile == "tests/test_executor.py"
                and cls_name == "TestExecutor"
            ):
                mnames = {m for _, m, _ in methods}
                if not (mnames & _TPC_EXECUTOR_METHODS):
                    run_setup_class = False
                else:
                    if "test_execute_tpcds" not in mnames and hasattr(mod, "TPCDS_SCHEMA"):
                        orig_tpcds_schema = mod.TPCDS_SCHEMA
                        mod.TPCDS_SCHEMA = {}
                    if not (mnames & {"test_optimized_tpch", "test_execute_tpch"}) and hasattr(
                        mod, "TPCH_SCHEMA"
                    ):
                        orig_tpch_schema = mod.TPCH_SCHEMA
                        mod.TPCH_SCHEMA = {}

            try:
                if (
                    can_lazy_optimizer_setup
                    and tfile == "tests/test_optimizer.py"
                    and cls_name == "TestOptimizer"
                    and hasattr(mod, "duckdb")
                ):
                    def real_connect(*a: Any, **kw: Any) -> Any:
                        return mod.duckdb._real_connect(*a, **kw) if hasattr(mod.duckdb, "_real_connect") else importlib.import_module("duckdb").connect(*a, **kw)

                    if "connect" in mod.duckdb.__dict__:
                        mod.duckdb._real_connect = mod.duckdb.__dict__["connect"]
                    lazy_conn = _LazyOptimizerConn(
                        lambda *a, **kw: (
                            mod.duckdb.__getattr__("connect")(*a, **kw)
                            if "__getattr__" in mod.duckdb.__dict__
                            else real_connect(*a, **kw)
                        )
                    )
                    mod.duckdb.connect = lambda *a, **kw: (
                        lazy_conn if not a and not kw else lazy_conn._real_connect(*a, **kw)
                    )
                    try:
                        cls.setUpClass()
                    finally:
                        lazy_conn.in_setup = False
                        if "_real_connect" in mod.duckdb.__dict__:
                            mod.duckdb.connect = mod.duckdb.__dict__.pop("_real_connect")
                        else:
                            mod.duckdb.__dict__.pop("connect", None)
                elif run_setup_class and hasattr(cls, "setUpClass"):
                    cls.setUpClass()
            except BaseException:
                if orig_tpcds_schema is not None:
                    mod.TPCDS_SCHEMA = orig_tpcds_schema
                if orig_tpch_schema is not None:
                    mod.TPCH_SCHEMA = orig_tpch_schema
                continue

            try:
                for test_num, meth_name, _test_id in methods:
                    if time.monotonic() >= soft_deadline_s:
                        break
                    res = _FailFastResult()
                    try:
                        tc = cls(meth_name)
                        tc.run(res)
                    except _SubtestAbort:
                        res.failed = True
                    except BaseException:
                        res.failed = True
                    if not res.failed:
                        passed.append(test_num)
            finally:
                if orig_tpcds_schema is not None:
                    mod.TPCDS_SCHEMA = orig_tpcds_schema
                if orig_tpch_schema is not None:
                    mod.TPCH_SCHEMA = orig_tpch_schema
                if run_setup_class:
                    try:
                        if hasattr(cls, "tearDownClass"):
                            cls.tearDownClass()
                    except BaseException:
                        pass
                    try:
                        if hasattr(cls, "doClassCleanups"):
                            cls.doClassCleanups()
                    except BaseException:
                        pass

    ru_self = resource.getrusage(resource.RUSAGE_SELF)
    ru_kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_s = ru_self.ru_utime + ru_self.ru_stime + ru_kids.ru_utime + ru_kids.ru_stime
    return {
        "passed": sorted(passed),
        "import_error_files": sorted(set(import_error_files)),
        "cpu_s": cpu_s,
    }


def _reap_child(pid: int) -> int:
    try:
        _, status = os.waitpid(pid, 0)
        return status
    except ChildProcessError:
        return 0


def _kill_and_reap(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _reap_child(pid)


@dataclass
class _RunningJob:
    commit_id: str
    pid: int
    r_fd: int
    overlay_dir: str
    start_t: float
    deadline_t: float
    buf: bytearray
    plan: VerifyPlan
    candidates: frozenset[int]


def _run_fork_queue(
    queue: list[tuple[str, CommitAnalysis, frozenset[int], VerifyPlan]],
    outcomes: dict[str, VerifyOutcome],
    sqlglot_root: Path,
    id_by_num: Mapping[int, str],
    file_to_tests: Mapping[str, frozenset[int]],
    tpcds_plan_hashes: tuple[str, ...],
    per_commit_timeout_s: float,
    batch_deadline: float,
    concurrency: int,
    tmp_base: str,
) -> None:
    if not queue:
        return

    sqlglot_root_str = str(sqlglot_root)
    active: list[_RunningJob] = []
    q_idx = 0

    def launch_next() -> None:
        nonlocal q_idx
        while len(active) < concurrency and q_idx < len(queue):
            now = time.monotonic()
            remaining_batch = batch_deadline - now
            if remaining_batch <= 1.5:
                break
            cid, analysis, cands, plan = queue[q_idx]
            q_idx += 1

            timeout_s = min(per_commit_timeout_s, remaining_batch - 0.5)
            overlay_dir = create_overlay(sqlglot_root, analysis.patched_contents, tmp_base=tmp_base)
            test_specs = [(n, id_by_num[n]) for n in sorted(plan.to_verify)]
            patched_files = frozenset(analysis.patched_contents.keys())
            skip_pickle = can_skip_pickle_roundtrip(analysis)

            r_fd, w_fd = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(r_fd)
                try:
                    soft_deadline = time.monotonic() + max(0.5, timeout_s - 0.8)
                    payload = _child_execute(
                        sqlglot_root_str,
                        overlay_dir,
                        test_specs,
                        soft_deadline,
                        patched_files=patched_files,
                        skip_pickle=skip_pickle,
                        fast_tpcds=plan.fast_tpcds,
                        tpcds_plan_hashes=tpcds_plan_hashes,
                    )
                    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                    view = memoryview(raw)
                    written = 0
                    while written < len(view):
                        written += os.write(w_fd, view[written:])
                except BaseException:
                    pass
                finally:
                    os.close(w_fd)
                    os._exit(0)
            else:
                os.close(w_fd)
                os.set_blocking(r_fd, False)
                active.append(
                    _RunningJob(
                        commit_id=cid,
                        pid=pid,
                        r_fd=r_fd,
                        overlay_dir=overlay_dir,
                        start_t=now,
                        deadline_t=now + timeout_s,
                        buf=bytearray(),
                        plan=plan,
                        candidates=cands,
                    )
                )

    def finalize_job(job: _RunningJob, timed_out: bool) -> None:
        wall_s = time.monotonic() - job.start_t
        try:
            os.close(job.r_fd)
        except OSError:
            pass
        if timed_out:
            _kill_and_reap(job.pid)
        else:
            status = _reap_child(job.pid)
            if not (os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0):
                timed_out = True
        shutil.rmtree(job.overlay_dir, ignore_errors=True)

        if timed_out or not job.buf:
            outcomes[job.commit_id] = VerifyOutcome(
                final_tests=job.candidates,
                verified=True,
                shed_count=0,
                child_cpu_s=0.0,
                wall_s=wall_s,
                timed_out=timed_out,
            )
            return

        try:
            data = json.loads(job.buf.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            outcomes[job.commit_id] = VerifyOutcome(
                final_tests=job.candidates,
                verified=True,
                shed_count=0,
                child_cpu_s=0.0,
                wall_s=wall_s,
                timed_out=False,
            )
            return

        passed_set = {
            n for n in data.get("passed", ()) if isinstance(n, int) and n in job.plan.to_verify
        }
        import_err_files = data.get("import_error_files", ())
        import_err_tests: set[int] = set()
        for tfile in import_err_files:
            import_err_tests.update(file_to_tests.get(tfile, ()))

        final_set = (job.candidates - passed_set) | import_err_tests
        outcomes[job.commit_id] = VerifyOutcome(
            final_tests=frozenset(final_set),
            verified=True,
            shed_count=len(job.candidates - final_set),
            child_cpu_s=float(data.get("cpu_s", 0.0)),
            wall_s=wall_s,
            timed_out=False,
        )

    try:
        launch_next()
        while active:
            now = time.monotonic()
            next_deadline = min(job.deadline_t for job in active)
            wait_s = max(0.01, min(0.25, next_deadline - now))
            fd_to_job = {job.r_fd: job for job in active}
            ready, _, _ = select.select(list(fd_to_job.keys()), [], [], wait_s)

            finished: list[tuple[_RunningJob, bool]] = []
            for fd in ready:
                job = fd_to_job[fd]
                eof = False
                while True:
                    try:
                        chunk = os.read(fd, 65536)
                        if not chunk:
                            eof = True
                            break
                        job.buf.extend(chunk)
                    except BlockingIOError:
                        break
                    except OSError:
                        eof = True
                        break
                if eof:
                    finished.append((job, False))

            now = time.monotonic()
            finished_ids = {j.commit_id for j, _ in finished}
            for job in active:
                if job.commit_id not in finished_ids and now >= job.deadline_t:
                    finished.append((job, True))

            if finished:
                done_set = {j.commit_id for j, _ in finished}
                active = [j for j in active if j.commit_id not in done_set]
                for job, timed_out in finished:
                    finalize_job(job, timed_out)
                launch_next()
    finally:
        for job in active:
            finalize_job(job, timed_out=True)

    while q_idx < len(queue):
        cid, _an, cands, _plan = queue[q_idx]
        q_idx += 1
        outcomes[cid] = VerifyOutcome(
            final_tests=cands,
            verified=False,
            shed_count=0,
            child_cpu_s=0.0,
            wall_s=0.0,
        )


def verify_batch(
    analyses: Mapping[str, CommitAnalysis],
    candidate_sets: Mapping[str, frozenset[int]],
    sqlglot_root: Path,
    ms_by_num: Mapping[int, int],
    id_by_num: Mapping[int, str],
    file_by_num: Mapping[int, str],
    file_to_tests: Mapping[str, frozenset[int]],
    *,
    fixture_to_tests: Mapping[str, frozenset[int]] | None = None,
    tpcds_plan_hashes: tuple[str, ...] = (),
    per_commit_timeout_s: float = PER_COMMIT_WALL_TIMEOUT_S,
    global_wall_budget_s: float = GLOBAL_WALL_BUDGET_S,
    max_concurrent: int = MAX_CONCURRENT_CHILDREN,
    tmp_base: str = "/tmp",
    force_verify_all: bool = False,
) -> dict[str, VerifyOutcome]:
    """Run 2-tier cost-aware forked verification across commits with strict zero-miss invariants."""
    if "sqlglot" in sys.modules or "tests" in sys.modules:
        raise RuntimeError("Parent process must not import sqlglot or tests before os.fork()")

    outcomes: dict[str, VerifyOutcome] = {}
    tier1_queue: list[tuple[str, CommitAnalysis, frozenset[int], VerifyPlan]] = []
    tier2_queue: list[tuple[str, CommitAnalysis, frozenset[int], VerifyPlan]] = []

    for cid in sorted(analyses):
        cands = candidate_sets[cid]
        analysis = analyses[cid]
        plan = plan_commit_verification(
            cands,
            ms_by_num,
            file_by_num,
            analysis=analysis,
            sqlglot_root=sqlglot_root,
            fixture_to_tests=fixture_to_tests,
            id_by_num=id_by_num,
            tpcds_plan_hashes=tpcds_plan_hashes,
        )
        if force_verify_all and cands:
            plan = VerifyPlan(
                keep_unverified=frozenset(),
                to_verify=cands,
                est_cpu_ms=plan.est_cpu_ms,
                max_sim_gain_ms=plan.max_sim_gain_ms,
                should_verify=True,
                use_preloaded_sqlglot=plan.use_preloaded_sqlglot,
                fast_tpcds=plan.fast_tpcds,
            )
        if not plan.should_verify:
            outcomes[cid] = VerifyOutcome(
                final_tests=cands,
                verified=False,
                shed_count=0,
                child_cpu_s=0.0,
                wall_s=0.0,
            )
        elif plan.use_preloaded_sqlglot:
            tier2_queue.append((cid, analysis, cands, plan))
        else:
            tier1_queue.append((cid, analysis, cands, plan))

    if not tier1_queue and not tier2_queue:
        return outcomes

    _preload_stdlib_for_fork()

    batch_deadline = time.monotonic() + global_wall_budget_s
    concurrency = max(1, min(MAX_CONCURRENT_CHILDREN, max_concurrent))

    # Tier 1: Clean fork parent (zero sqlglot/tests imported)
    if tier1_queue:
        _run_fork_queue(
            tier1_queue,
            outcomes,
            sqlglot_root,
            id_by_num,
            file_to_tests,
            tpcds_plan_hashes,
            per_commit_timeout_s,
            batch_deadline,
            concurrency,
            tmp_base,
        )

    # Tier 2: Isolated sqlglot-preloaded fork server (keeps main parent 100% free of sqlglot)
    if tier2_queue:
        r_fd, w_fd = os.pipe()
        t2_pid = os.fork()
        if t2_pid == 0:
            os.close(r_fd)
            try:
                devnull_fd = os.open(os.devnull, os.O_WRONLY)
                try:
                    os.dup2(devnull_fd, 1)
                finally:
                    os.close(devnull_fd)
                sys.stdout = open(os.devnull, "w", encoding="utf-8")

                root_str = str(sqlglot_root)
                if root_str not in sys.path:
                    sys.path.insert(0, root_str)
                importlib.import_module("sqlglot")

                tier2a = []
                tier2b_light = []
                tier2b_heavy = []
                for item in tier2_queue:
                    an = item[1]
                    plan = item[3]
                    if not can_use_preloaded_test_dialect(an):
                        tier2a.append(item)
                    elif any(
                        file_by_num[n] in ("tests/test_optimizer.py", "tests/test_executor.py")
                        for n in plan.to_verify
                    ):
                        tier2b_heavy.append(item)
                    else:
                        tier2b_light.append(item)

                t2_outcomes: dict[str, VerifyOutcome] = {}
                if tier2a:
                    _run_fork_queue(
                        tier2a,
                        t2_outcomes,
                        sqlglot_root,
                        id_by_num,
                        file_to_tests,
                        tpcds_plan_hashes,
                        per_commit_timeout_s,
                        batch_deadline,
                        concurrency,
                        tmp_base,
                    )
                tier2b = [*tier2b_light, *tier2b_heavy]
                if len(tier2b) >= 2:
                    try:
                        importlib.import_module("tests.dialects.test_dialect")
                    except BaseException:
                        pass
                if tier2b:
                    _run_fork_queue(
                        tier2b,
                        t2_outcomes,
                        sqlglot_root,
                        id_by_num,
                        file_to_tests,
                        tpcds_plan_hashes,
                        per_commit_timeout_s,
                        batch_deadline,
                        concurrency,
                        tmp_base,
                    )
                serial = {
                    cid: {
                        "final_tests": sorted(o.final_tests),
                        "verified": o.verified,
                        "shed_count": o.shed_count,
                        "child_cpu_s": o.child_cpu_s,
                        "wall_s": o.wall_s,
                        "timed_out": o.timed_out,
                    }
                    for cid, o in t2_outcomes.items()
                }
                raw = json.dumps(serial, separators=(",", ":")).encode("utf-8")
                view = memoryview(raw)
                written = 0
                while written < len(view):
                    written += os.write(w_fd, view[written:])
            except BaseException:
                pass
            finally:
                os.close(w_fd)
                os._exit(0)
        else:
            os.close(w_fd)
            buf = bytearray()
            while True:
                chunk = os.read(r_fd, 65536)
                if not chunk:
                    break
                buf.extend(chunk)
            os.close(r_fd)
            _reap_child(t2_pid)
            if buf:
                t2_data = json.loads(buf.decode("utf-8"))
                for cid, entry in t2_data.items():
                    outcomes[cid] = VerifyOutcome(
                        final_tests=frozenset(entry["final_tests"]),
                        verified=bool(entry["verified"]),
                        shed_count=int(entry["shed_count"]),
                        child_cpu_s=float(entry["child_cpu_s"]),
                        wall_s=float(entry["wall_s"]),
                        timed_out=bool(entry["timed_out"]),
                    )
            for cid, _an, cands, _plan in tier2_queue:
                if cid not in outcomes:
                    outcomes[cid] = VerifyOutcome(
                        final_tests=cands,
                        verified=False,
                        shed_count=0,
                        child_cpu_s=0.0,
                        wall_s=0.0,
                    )

    return outcomes
