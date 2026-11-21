"""Generate 32 unseen adversarial synthetic commits against sqlglot@eb11aec and compute ground truth."""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import importlib
import json
import os
import resource
import shutil
import signal
import sys
import time
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diff_analyzer import analyze_patch  # noqa: E402
from selector import BlastSelector  # noqa: E402
from verifier import (  # noqa: E402
    _FailFastResult,
    _fast_verify_tpcds,
    _InlinePickleExecutor,
    _install_lazy_heavy_deps,
    _LazyOptimizerConn,
    _SubtestAbort,
    _TPC_EXECUTOR_METHODS,
    create_overlay,
    redirect_preloaded_modules_to_overlay,
)


def make_file_diff(rel_path: str, old_text: str, new_text: str) -> str:
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            n=3,
        )
    )
    if not diff:
        raise ValueError(f"Empty diff generated for {rel_path}")
    return f"diff --git a/{rel_path} b/{rel_path}\n" + "".join(diff)


def replace_once(rel_path: str, old_text: str, target: str, replacement: str) -> str:
    if old_text.count(target) != 1:
        raise ValueError(
            f"Expected exactly 1 occurrence of {target!r} in {rel_path}, found {old_text.count(target)}"
        )
    new_text = old_text.replace(target, replacement, 1)
    return make_file_diff(rel_path, old_text, new_text)


def build_synthetic_patches(sqlglot_root: Path) -> dict[str, str]:
    def read(rel: str) -> str:
        return (sqlglot_root / rel).read_text(encoding="utf-8")

    patches: dict[str, str] = {}

    # 1. Core syntax error
    patches["s01_core_syntax_err"] = replace_once(
        "sqlglot/tokens.py",
        read("sqlglot/tokens.py"),
        'SQLGLOTC_INSTALLED = not getattr(tokenizer_core, "__file__", "py").endswith(".py")',
        "SQLGLOTC_INSTALLED = (not getattr(tokenizer_core,",
    )

    # 2. Lazy dialect syntax error
    patches["s02_lazy_dialect_syntax_err"] = replace_once(
        "sqlglot/dialects/postgres.py",
        read("sqlglot/dialects/postgres.py"),
        "    INDEX_OFFSET = 1",
        "    INDEX_OFFSET = ???",
    )

    # 3. Lazy parser syntax error (touches _PRELOADED_TEST_DIALECT_FILES)
    patches["s03_lazy_parser_syntax_err"] = replace_once(
        "sqlglot/parsers/duckdb.py",
        read("sqlglot/parsers/duckdb.py"),
        "from sqlglot.helper import seq_get",
        "from sqlglot.helper import (",
    )

    # 4. Core broken import
    patches["s04_core_broken_import"] = replace_once(
        "sqlglot/schema.py",
        read("sqlglot/schema.py"),
        "from sqlglot.errors import SchemaError",
        "from sqlglot.errors import SchemaError, NonExistentSchemaSymbolXYZ",
    )

    # 5. Lazy dialect broken import
    patches["s05_lazy_dialect_broken_import"] = replace_once(
        "sqlglot/dialects/clickhouse.py",
        read("sqlglot/dialects/clickhouse.py"),
        "    INDEX_OFFSET = 1",
        "    import nonexistent_clickhouse_dependency_xyz\n    INDEX_OFFSET = 1",
    )

    # 6. Test file broken import
    patches["s06_test_file_broken_import"] = replace_once(
        "tests/test_serde.py",
        read("tests/test_serde.py"),
        "import unittest",
        "import unittest\nimport nonexistent_test_serde_pkg_xyz",
    )

    # 7. Single subTest failure inside #1106 test_execute_tpcds (and test_canonicalize)
    patches["s07_subtest_fail_tpcds_1106"] = replace_once(
        "sqlglot/optimizer/canonicalize.py",
        read("sqlglot/optimizer/canonicalize.py"),
        '        expression.set("desc", None)',
        '        expression.set("desc", True)',
    )

    # 8. Harmless dead-branch edit in canonicalize.py that selects #1106 and sheds it via golden plans
    patches["s08_subtest_pass_tpcds_1106"] = replace_once(
        "sqlglot/optimizer/canonicalize.py",
        read("sqlglot/optimizer/canonicalize.py"),
        '        expression.set("desc", None)',
        '        expression.set("desc", None if expression is not None else None)',
    )

    # 9. Single subTest failure in TestOptimizer.test_simplify
    patches["s09_subtest_fail_optimizer"] = replace_once(
        "sqlglot/optimizer/simplify.py",
        read("sqlglot/optimizer/simplify.py"),
        "            if isinstance(child, expression.__class__):\n                node.replace(child)",
        "            if isinstance(child, expression.__class__) and False:\n                node.replace(child)",
    )

    # 10. Single subTest failure in dialect Validator (test_mysql)
    patches["s10_subtest_fail_dialect_validator"] = replace_once(
        "sqlglot/generators/mysql.py",
        read("sqlglot/generators/mysql.py"),
        "def _date_trunc_sql(self: MySQLGenerator, expression: exp.DateTrunc) -> str:\n    expr = self.sql(expression, \"this\")",
        "def _date_trunc_sql(self: MySQLGenerator, expression: exp.DateTrunc) -> str:\n    return \"BROKEN_DATE_TRUNC\"\n    expr = self.sql(expression, \"this\")",
    )

    # 11. Infinite loop regression inside a verified function (triggers child timeout without hanging)
    patches["s11_infinite_loop_timeout"] = replace_once(
        "sqlglot/generators/duckdb.py",
        read("sqlglot/generators/duckdb.py"),
        "    def _array_bag_sql(self, condition: exp.Expr, arr1: exp.Expr, arr2: exp.Expr) -> str:\n        cond = exp.Paren(this=exp.replace_placeholders(condition, arr1=arr1, arr2=arr2))",
        "    def _array_bag_sql(self, condition: exp.Expr, arr1: exp.Expr, arr2: exp.Expr) -> str:\n        while True:\n            pass\n        cond = exp.Paren(this=exp.replace_placeholders(condition, arr1=arr1, arr2=arr2))",
    )

    # 12. Slow sleep regression inside a verified function (triggers child timeout without hanging)
    patches["s12_slow_sleep_timeout"] = replace_once(
        "sqlglot/parsers/postgres.py",
        read("sqlglot/parsers/postgres.py"),
        "        return self.expression(\n            exp.InOutColumnConstraint(",
        "        import time as _t; _t.sleep(30)\n        return self.expression(\n            exp.InOutColumnConstraint(",
    )

    # 13. Adversarial unsafe `is None -> == None` when target class defines custom `__eq__`
    p13_ctx = replace_once(
        "sqlglot/executor/context.py",
        read("sqlglot/executor/context.py"),
        "        if self._table is None:",
        "        if self._table == None:",
    )
    p13_tbl = replace_once(
        "sqlglot/executor/table.py",
        read("sqlglot/executor/table.py"),
        "        self.range_reader = RangeReader(self)",
        "        self.range_reader = RangeReader(self)\n\n    def __eq__(self, other: object) -> bool:\n        return self.columns == other.columns",  # type: ignore[attr-defined]
    )
    patches["s13_unsafe_is_to_eq_custom_eq"] = p13_ctx + p13_tbl

    # 14. Adversarial unsafe `is -> ==` on metaclass `_Dialect.__eq__`
    patches["s14_unsafe_is_to_eq_dialect_meta"] = replace_once(
        "sqlglot/dialects/dialect.py",
        read("sqlglot/dialects/dialect.py"),
        "    def __eq__(cls, other: object) -> bool:\n        if cls is other:",
        "    def __eq__(cls, other: object) -> bool:\n        if cls == other:",
    )

    # 15. Sound closed-world `self._table is None -> self._table == None` when Table inherits object.__eq__
    patches["s15_safe_is_none_equiv_noop"] = replace_once(
        "sqlglot/executor/context.py",
        read("sqlglot/executor/context.py"),
        "        if self._table is None:",
        "        if self._table == None:",
    )

    # 16. Sound closed-world `self.normalization_strategy is NormalizationStrategy.UPPERCASE -> ==`
    patches["s16_safe_enum_is_equiv_noop"] = replace_once(
        "sqlglot/dialects/dialect.py",
        read("sqlglot/dialects/dialect.py"),
        "self.normalization_strategy is NormalizationStrategy.UPPERCASE",
        "self.normalization_strategy == NormalizationStrategy.UPPERCASE",
    )

    # 17. Harmless comment edit in core parser.py
    patches["s17_harmless_comment_core"] = replace_once(
        "sqlglot/parser.py",
        read("sqlglot/parser.py"),
        "class Parser:",
        "# synthetic harmless parser comment\nclass Parser:",
    )

    # 18. Harmless comment edit in lazy dialect postgres.py
    patches["s18_harmless_comment_dialect"] = replace_once(
        "sqlglot/dialects/postgres.py",
        read("sqlglot/dialects/postgres.py"),
        "class Postgres(Dialect):",
        "# synthetic harmless postgres comment\nclass Postgres(Dialect):",
    )

    # 19. Comment edit in sqlglot/typing/duckdb.py (triggers typing_meta_test #1764, passes and sheds)
    patches["s19_typing_comment_triggers_meta_test"] = replace_once(
        "sqlglot/typing/duckdb.py",
        read("sqlglot/typing/duckdb.py"),
        "DATETIME_WITHOUT_TZ = {",
        "# synthetic typing comment\nDATETIME_WITHOUT_TZ = {",
    )

    # 20. Redundant typing metadata edit in sqlglot/typing/spark2.py that FAILS #1764
    patches["s20_typing_redundant_metadata_fails"] = replace_once(
        "sqlglot/typing/spark2.py",
        read("sqlglot/typing/spark2.py"),
        "    **HIVE_EXPRESSION_METADATA,",
        '    **HIVE_EXPRESSION_METADATA,\n    exp.Abs: {"annotator": lambda self, e: self._annotate_by_args(e, "this")},',
    )

    # 21. Docstring edit in sqlglot/generator.py
    patches["s21_docstring_edit_harmless"] = replace_once(
        "sqlglot/generator.py",
        read("sqlglot/generator.py"),
        "    \"\"\"\n    Generator converts a given syntax tree to the corresponding SQL string.",
        "    \"\"\"\n    Generator converts a syntax tree into the corresponding SQL string.",
    )

    # 22. Whitespace-only formatting edit in sqlglot/expressions/core.py
    patches["s22_whitespace_only_edit"] = replace_once(
        "sqlglot/expressions/core.py",
        read("sqlglot/expressions/core.py"),
        "class Expr:",
        "\n\nclass Expr:",
    )

    # 23. Deleted function in sqlglot/helper.py
    old_helper = read("sqlglot/helper.py")
    target_fn = (
        "def split_num_words(\n"
        "    value: str, sep: str, min_num_words: int, fill_from_start: bool = True\n"
        ") -> list[str | None]:\n"
        '    """\n'
        "    Perform a split on a value and return N words as a result with `None` used for words that don't exist.\n"
        "\n"
        "    Args:\n"
        "        value: The value to be split.\n"
        "        sep: The value to use to split on.\n"
        "        min_num_words: The minimum number of words that are going to be in the result.\n"
        "        fill_from_start: Indicates that if `None` values should be inserted at the start or end of the list.\n"
        "\n"
        "    Examples:\n"
        '        >>> split_num_words("db.table", ".", 3)\n'
        "        [None, 'db', 'table']\n"
        '        >>> split_num_words("db.table", ".", 3, fill_from_start=False)\n'
        "        ['db', 'table', None]\n"
        '        >>> split_num_words("db.table", ".", 1)\n'
        "        ['db', 'table']\n"
        "\n"
        "    Returns:\n"
        "        The list of words returned by `split`, possibly augmented by a number of `None` values.\n"
        '    """\n'
        "    words = value.split(sep)\n"
        "    if fill_from_start:\n"
        "        return [None] * (min_num_words - len(words)) + words\n"
        "    return words + [None] * (min_num_words - len(words))\n"
    )
    patches["s23_deleted_function_in_helper"] = replace_once(
        "sqlglot/helper.py",
        old_helper,
        target_fn,
        "",
    )

    # 24. Newly added dialect subclass method override (added-only hunk)
    patches["s24_added_dialect_method_override"] = replace_once(
        "sqlglot/generators/sqlite.py",
        read("sqlglot/generators/sqlite.py"),
        "class SQLiteGenerator(generator.Generator):",
        'class SQLiteGenerator(generator.Generator):\n    def select_sql(self, expression: exp.Select) -> str:\n        return "BROKEN_SQLITE_SELECT"\n',
    )

    # 25. Class/module dict key value mutation in tsql generator
    patches["s25_class_attr_dict_key_mutation"] = replace_once(
        "sqlglot/generators/tsql.py",
        read("sqlglot/generators/tsql.py"),
        '    "WEEKISO": "ISO_WEEK",',
        '    "WEEKISO": "BROKEN_ISO_WEEK",',
    )

    # 26. Dialect class attribute mutation in doris.py
    patches["s26_class_attr_dialect_format"] = replace_once(
        "sqlglot/dialects/doris.py",
        read("sqlglot/dialects/doris.py"),
        "    DATE_FORMAT = \"'yyyy-MM-dd'\"",
        "    DATE_FORMAT = \"'BROKEN-DATE'\"",
    )

    # 27. Edit in tests/helpers.py
    patches["s27_tests_helpers_edit"] = replace_once(
        "tests/helpers.py",
        read("tests/helpers.py"),
        "def assert_logger_contains(message, logger, level=\"error\"):",
        "def assert_logger_contains(message, logger, level=\"error\"):\n    raise AssertionError(\"synthetic helper failure\")",
    )

    # 28. Edit in TestSerde setUpClass / helper method in tests/test_serde.py
    patches["s28_test_setup_class_edit"] = replace_once(
        "tests/test_serde.py",
        read("tests/test_serde.py"),
        "class TestSerde(unittest.TestCase):\n    def dump_load(self, expression):",
        "class TestSerde(unittest.TestCase):\n    @classmethod\n    def setUpClass(cls):\n        raise RuntimeError(\"broken setUpClass\")\n\n    def dump_load(self, expression):",
    )

    # 29. Direct test method edit in tests/test_build.py
    patches["s29_direct_test_method_edit"] = replace_once(
        "tests/test_build.py",
        read("tests/test_build.py"),
        "    def test_build(self):\n        x = condition(\"x\")",
        "    def test_build(self):\n        self.fail(\"synthetic direct test failure\")\n        x = condition(\"x\")",
    )

    # 30. Optimizer SQL fixture edit
    patches["s30_optimizer_fixture_edit"] = replace_once(
        "tests/fixtures/optimizer/pushdown_predicates.sql",
        read("tests/fixtures/optimizer/pushdown_predicates.sql"),
        "SELECT x.a AS a FROM (SELECT x.a FROM x AS x WHERE x.a = 1 AND x.b = 1) AS x JOIN y ON y.a = 1 WHERE TRUE AND TRUE AND TRUE;",
        "SELECT x.a AS a FROM (SELECT x.a FROM x AS x WHERE x.a = 999) AS x JOIN y ON y.a = 1;",
    )

    # 31. Pure non-code edit (README.md)
    patches["s31_pure_non_code_readme"] = replace_once(
        "README.md",
        read("README.md"),
        "![SQLGlot logo](sqlglot.png)",
        "![SQLGlot logo](sqlglot.png)\n<!-- synthetic non-code edit -->",
    )

    # 32. Serde dump mutation (verifies pickle roundtrip / serde test detection)
    patches["s32_serde_expression_mutation"] = replace_once(
        "sqlglot/serde.py",
        read("sqlglot/serde.py"),
        "def dump(expression: exp.Expr) -> list[dict[str, t.Any]]:",
        "def dump(expression: exp.Expr) -> list[dict[str, t.Any]]:\n    return []",
    )

    return patches


def _eval_tests_in_fork(
    sqlglot_root: Path,
    patched_contents: dict[str, str | None],
    test_specs: list[tuple[int, str]],
    tpcds_plan_hashes: tuple[str, ...],
    timeout_s: float = 4.0,
) -> tuple[set[int], set[str], bool]:
    """Run test_specs in an isolated forked child without #1106 heavy-first abort shortcut."""
    if not test_specs:
        return set(), set(), False
    overlay_dir = create_overlay(sqlglot_root, patched_contents, tmp_base="/tmp")
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(devnull_fd, 1)
                os.dup2(devnull_fd, 2)
            finally:
                os.close(devnull_fd)
            sys.stdout = open(os.devnull, "w", encoding="utf-8")
            sys.stderr = open(os.devnull, "w", encoding="utf-8")

            os.chdir(overlay_dir)
            redirect_preloaded_modules_to_overlay(str(sqlglot_root), overlay_dir)
            os.environ["PYTHONPATH"] = overlay_dir
            os.environ["SKIP_INTEGRATION"] = "0"
            concurrent.futures.ProcessPoolExecutor = _InlinePickleExecutor  # type: ignore[misc]

            soft_deadline = time.monotonic() + timeout_s
            passed: list[int] = []
            import_error_files: list[str] = []

            by_file_cls: dict[str, dict[str, list[tuple[int, str, str]]]] = {}
            for tnum, tid in test_specs:
                if tid == "tests/test_executor.py::TestExecutor::test_execute_tpcds":
                    try:
                        if _fast_verify_tpcds(tpcds_plan_hashes, soft_deadline):
                            passed.append(tnum)
                    except BaseException:
                        pass
                    continue
                parts = tid.split("::")
                tfile = parts[0]
                cls_name = parts[1] if len(parts) == 3 else ""
                meth_name = parts[2] if len(parts) == 3 else parts[-1]
                by_file_cls.setdefault(tfile, {}).setdefault(cls_name, []).append(
                    (tnum, meth_name, tid)
                )

            patched_files = frozenset(patched_contents.keys())
            can_filter_exec = (
                "tests/test_executor.py" not in patched_files
                and "tests/helpers.py" not in patched_files
            )
            can_lazy_opt = "tests/test_optimizer.py" not in patched_files

            for tfile in sorted(by_file_cls):
                if time.monotonic() >= soft_deadline:
                    break
                if (
                    tfile in ("tests/test_optimizer.py", "tests/test_executor.py")
                    and can_lazy_opt
                    and can_filter_exec
                ):
                    _install_lazy_heavy_deps()
                mod_name = tfile[:-3].replace("/", ".")
                try:
                    mod = importlib.import_module(mod_name)
                except BaseException:
                    import_error_files.append(tfile)
                    continue

                for cls_name in sorted(by_file_cls[tfile]):
                    if time.monotonic() >= soft_deadline:
                        break
                    methods = by_file_cls[tfile][cls_name]
                    cls = getattr(mod, cls_name, None) if cls_name else None
                    if cls is None or not isinstance(cls, type) or not issubclass(cls, unittest.TestCase):
                        continue

                    run_setup = True
                    if can_filter_exec and tfile == "tests/test_executor.py" and cls_name == "TestExecutor":
                        mnames = {m for _, m, _ in methods}
                        if not (mnames & _TPC_EXECUTOR_METHODS):
                            run_setup = False

                    try:
                        if (
                            can_lazy_opt
                            and tfile == "tests/test_optimizer.py"
                            and cls_name == "TestOptimizer"
                            and hasattr(mod, "duckdb")
                        ):
                            def real_connect(*a: Any, **kw: Any) -> Any:
                                return (
                                    mod.duckdb._real_connect(*a, **kw)
                                    if hasattr(mod.duckdb, "_real_connect")
                                    else importlib.import_module("duckdb").connect(*a, **kw)
                                )

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
                        elif run_setup and hasattr(cls, "setUpClass"):
                            cls.setUpClass()
                    except BaseException:
                        continue

                    try:
                        for tnum, meth_name, _tid in methods:
                            if time.monotonic() >= soft_deadline:
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
                                passed.append(tnum)
                    finally:
                        if run_setup:
                            for cleanup_attr in ("tearDownClass", "doClassCleanups"):
                                try:
                                    if hasattr(cls, cleanup_attr):
                                        getattr(cls, cleanup_attr)()
                                except BaseException:
                                    pass

            raw = json.dumps({"passed": passed, "import_error_files": import_error_files}).encode(
                "utf-8"
            )
            os.write(w_fd, raw)
        except BaseException:
            pass
        finally:
            os.close(w_fd)
            os._exit(0)
    else:
        os.close(w_fd)
        deadline = time.monotonic() + timeout_s + 0.5
        buf = bytearray()
        os.set_blocking(r_fd, False)
        timed_out = False
        import select as _select

        while True:
            rem = deadline - time.monotonic()
            if rem <= 0:
                timed_out = True
                break
            ready, _, _ = _select.select([r_fd], [], [], min(0.1, rem))
            if ready:
                try:
                    chunk = os.read(r_fd, 65536)
                    if not chunk:
                        break
                    buf.extend(chunk)
                except BlockingIOError:
                    pass
                except OSError:
                    break
        os.close(r_fd)
        if timed_out:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        shutil.rmtree(overlay_dir, ignore_errors=True)
        if timed_out or not buf:
            return set(), set(), True
        data = json.loads(buf.decode("utf-8"))
        return set(data.get("passed", ())), set(data.get("import_error_files", ())), False


def compute_ground_truth(
    patches: dict[str, str],
    sqlglot_root: Path,
    maps_dir: Path,
    tests_raw: list[dict[str, Any]],
) -> dict[str, list[int]]:
    selector = BlastSelector(maps_dir, runtime_tests=tests_raw)
    id_by_num = {t["number"]: t["id"] for t in tests_raw}
    answers: dict[str, list[int]] = {}

    for cid, patch_text in sorted(patches.items()):
        analysis = analyze_patch(cid, patch_text, sqlglot_root)
        sel = selector.select(analysis)
        probe_nums: set[int] = set(sel.candidates)

        # For sound equivalence no-op commits (s15, s16), independently run the targeted
        # test suite to verify 0 tests actually fail.
        if cid == "s15_safe_is_none_equiv_noop":
            probe_nums.update(
                n
                for n in selector.file_to_tests.get("tests/test_executor.py", ())
                if id_by_num[n] != "tests/test_executor.py::TestExecutor::test_execute_tpcds"
            )
        elif cid == "s16_safe_enum_is_equiv_noop":
            probe_nums.update(selector.file_to_tests.get("tests/test_tokens.py", ()))
            probe_nums.update(
                n
                for n in selector.file_to_tests.get("tests/test_parser.py", ())
                if id_by_num[n] != "tests/test_parser.py::TestParser::test_max_nodes"
            )

        # Group by test file so a single slow/broken file doesn't block other files
        by_file: dict[str, list[tuple[int, str]]] = {}
        for n in sorted(probe_nums):
            tid = id_by_num[n]
            tfile = tid.split("::", 1)[0]
            by_file.setdefault(tfile, []).append((n, tid))

        # Core syntax/import errors (s01, s04, s23) fail all 1,615 tests;
        # evaluate 1 representative test per file first to detect file-wide import failures fast.
        if len(by_file) > 10:
            rep_specs = [specs[0] for _tfile, specs in sorted(by_file.items())]
            rep_passed, rep_imp_errs, rep_to = _eval_tests_in_fork(
                sqlglot_root,
                analysis.patched_contents,
                rep_specs,
                selector.tpcds_plan_hashes,
                timeout_s=8.0,
            )
            if len(rep_imp_errs) == len(by_file):
                answers[cid] = sorted(probe_nums)
                print(
                    f"[synth] {cid}: all {len(by_file)} test files failed import -> {len(answers[cid])} failing tests"
                )
                continue

        failed_nums: set[int] = set()
        for tfile, specs in sorted(by_file.items()):
            passed_nums, imp_errs, _timed_out = _eval_tests_in_fork(
                sqlglot_root,
                analysis.patched_contents,
                specs,
                selector.tpcds_plan_hashes,
                timeout_s=2.0 if "timeout" in cid else 45.0,
            )
            if tfile in imp_errs:
                failed_nums.update(selector.file_to_tests.get(tfile, ()))
                for n, _tid in specs:
                    failed_nums.add(n)
            else:
                for n, _tid in specs:
                    if n not in passed_nums:
                        failed_nums.add(n)

        answers[cid] = sorted(failed_nums)
        print(
            f"[synth] {cid}: candidates={len(sel.candidates)}, ground_truth_failing={len(answers[cid])}"
        )

    return answers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic adversarial commit suite.")
    parser.add_argument(
        "--sqlglot-root",
        type=Path,
        default=REPO_ROOT.parent / "ai-notes" / "sqlglot",
    )
    parser.add_argument(
        "--sample-tests-json",
        type=Path,
        default=REPO_ROOT.parent / "sample-kit" / "sample" / "tests.json",
    )
    parser.add_argument(
        "--maps-dir",
        type=Path,
        default=REPO_ROOT / "maps",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT.parent / "ai-notes" / "synth",
    )
    args = parser.parse_args(argv)

    patches = build_synthetic_patches(args.sqlglot_root)
    if len(patches) < 30:
        raise RuntimeError(f"Expected >= 30 synthetic commits, built {len(patches)}")

    commits_dir = args.out_dir / "commits"
    if commits_dir.exists():
        shutil.rmtree(commits_dir)
    commits_dir.mkdir(parents=True, exist_ok=True)

    for cid, patch_text in sorted(patches.items()):
        (commits_dir / f"{cid}.patch").write_text(patch_text, encoding="utf-8")

    tests_payload = json.loads(args.sample_tests_json.read_text(encoding="utf-8"))
    (args.out_dir / "tests.json").write_text(
        json.dumps(tests_payload, indent=2) + "\n", encoding="utf-8"
    )

    answers = compute_ground_truth(
        patches=patches,
        sqlglot_root=args.sqlglot_root,
        maps_dir=args.maps_dir,
        tests_raw=tests_payload["tests"],
    )
    answers_payload = {
        "commits": {
            cid: {"failing": failing_ids, "kind": "synthetic"}
            for cid, failing_ids in sorted(answers.items())
        }
    }
    (args.out_dir / "answers.json").write_text(
        json.dumps(answers_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ru = resource.getrusage(resource.RUSAGE_SELF)
    print(
        f"Generated {len(patches)} synthetic commits in {args.out_dir} (host CPU {ru.ru_utime + ru.ru_stime:.2f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
