"""Contract, scheduler, diff-analyzer, selector, and verifier unit tests."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
MAPS_DIR = REPO_ROOT / "maps"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diff_analyzer import CommitAnalysis, analyze_patch  # noqa: E402
from main import format_and_validate_output  # noqa: E402
from scheduler import compute_commit_cost_ms, duration_ms, schedule_tests  # noqa: E402
from selector import BlastSelector  # noqa: E402
from verifier import (  # noqa: E402
    _FailFastResult,
    _SubtestAbort,
    can_skip_pickle_roundtrip,
    can_use_preloaded_sqlglot,
    can_use_preloaded_test_dialect,
    create_overlay,
    redirect_preloaded_modules_to_overlay,
    verify_batch,
)


def test_duration_ms_rounds_halves_up() -> None:
    assert duration_ms(0.0) == 0
    assert duration_ms(0.0005) == 1
    assert duration_ms(0.0015) == 2
    assert duration_ms(0.0025) == 3
    assert duration_ms(21.454) == 21454
    assert duration_ms(28.817) == 28817


def test_scheduler_empty_single_and_refinement() -> None:
    assert schedule_tests([], {}) == []

    # Single-worker beats 2 workers when worker overhead (2000ms) outweighs parallelism
    ms = {0: 400, 1: 400, 2: 400}
    workers = schedule_tests([2, 0, 1], ms)
    assert workers == [[0, 1, 2]]
    assert compute_commit_cost_ms(workers, ms) == 3200

    # Pairwise swap / local refinement reaches exact makespan on classic 2-worker instance
    # Durations [3000, 3000, 2000, 2000, 2000]: LPT gives [3000, 2000, 2000]=7000 vs [3000, 2000]=5000,
    # swap of 3000 <-> 2000+2000 or optimal balancing yields 6000 vs 6000!
    ms2 = {10: 3000, 11: 3000, 12: 2000, 13: 2000, 14: 2000}
    w2 = schedule_tests([14, 13, 12, 11, 10], ms2)
    assert len(w2) == 2
    assert compute_commit_cost_ms(w2, ms2) == 10000  # max(6000, 6000) + 4000
    # Deterministic ordering: ascending test IDs per worker, descending load then first test ID
    for w in w2:
        assert w == sorted(w)
    assert schedule_tests([10, 11, 12, 13, 14], ms2) == w2

    # Never exceeds 8 workers
    ms_many = {i: 5000 for i in range(20)}
    w_many = schedule_tests(range(20), ms_many)
    assert 1 <= len(w_many) <= 8
    assert sorted(t for w in w_many for t in w) == list(range(20))


def test_format_and_validate_output_enforces_all_8_rules() -> None:
    valid_tests = {0, 1, 2, 3}
    commits = {"c1", "c2"}

    raw = format_and_validate_output({"c2": [], "c1": [[0, 2], [1]]}, commits, valid_tests)
    assert not raw.startswith(b"\xef\xbb\xbf")
    lines = raw.decode("utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first == {"commit": "c1", "workers": [[0, 2], [1]]}
    assert json.loads(lines[1]) == {"commit": "c2", "workers": []}

    # Rule 3/missing: missing or extra commit
    with pytest.raises(ValueError, match="Commit mismatch"):
        format_and_validate_output({"c1": []}, commits, valid_tests)

    # Rule 5: bool is rejected even though isinstance(True, int)
    with pytest.raises(ValueError, match="Invalid test number"):
        format_and_validate_output({"c1": [[True]], "c2": []}, commits, valid_tests)  # type: ignore[list-item]

    # Rule 5: non-int float / string rejected
    with pytest.raises(ValueError, match="Invalid test number"):
        format_and_validate_output({"c1": [[1.5]], "c2": []}, commits, valid_tests)  # type: ignore[list-item]

    # Rule 6: unknown test number
    with pytest.raises(ValueError, match="Unknown test number"):
        format_and_validate_output({"c1": [[99]], "c2": []}, commits, valid_tests)

    # Rule 7: duplicate test across workers
    with pytest.raises(ValueError, match="Duplicate test number"):
        format_and_validate_output({"c1": [[0, 1], [1, 2]], "c2": []}, commits, valid_tests)

    # Rule 8: > 8 non-empty workers
    many_tests = set(range(10))
    with pytest.raises(ValueError, match="Too many workers"):
        format_and_validate_output(
            {"c1": [[i] for i in range(9)], "c2": []},
            commits,
            many_tests,
        )


def test_diff_analyzer_edge_cases(tmp_path: Path) -> None:
    base = tmp_path / "repo"
    (base / "pkg").mkdir(parents=True)
    orig = textwrap.dedent(
        '''\
        """Module doc."""

        class Base:
            VAL = {"a": 1, "b": 2}

            def foo(self, x):
                if (
                    x > 0
                    and x < 10
                ):
                    return x + 1
                return 0
        '''
    )
    (base / "pkg" / "mod.py").write_text(orig, encoding="utf-8")

    # 1. Empty patch
    empty = analyze_patch("empty", "", base)
    assert empty.files == []

    # 2. Pure comment edit (AST no-op, docstring unchanged)
    comment_patch = textwrap.dedent(
        """\
        diff --git a/pkg/mod.py b/pkg/mod.py
        --- a/pkg/mod.py
        +++ b/pkg/mod.py
        @@ -5,3 +5,4 @@ class Base:
             def foo(self, x):
        +        # pure comment
                 if (
                     x > 0
        """
    )
    c_an = analyze_patch("comment", comment_patch, base)
    assert len(c_an.files) == 1
    assert c_an.files[0].is_ast_noop is True
    assert c_an.files[0].docstring_changed is False

    # 3. Docstring-only edit
    doc_patch = textwrap.dedent(
        '''\
        diff --git a/pkg/mod.py b/pkg/mod.py
        --- a/pkg/mod.py
        +++ b/pkg/mod.py
        @@ -1,3 +1,3 @@
        -"""Module doc."""
        +"""Updated module doc."""
         
         class Base:
        '''
    )
    d_an = analyze_patch("doc", doc_patch, base)
    assert d_an.files[0].is_ast_noop is True
    assert d_an.files[0].docstring_changed is True

    # 4. Syntax-breaking patch
    syn_patch = textwrap.dedent(
        """\
        diff --git a/pkg/mod.py b/pkg/mod.py
        --- a/pkg/mod.py
        +++ b/pkg/mod.py
        @@ -7,3 +7,3 @@ class Base:
                 if (
        -            x > 0
        +            x >>> 0 ???
                     and x < 10
        """
    )
    s_an = analyze_patch("syn", syn_patch, base)
    assert s_an.files[0].syntax_broken is True

    # 5. Added-only hunk (old_count == 0) adding a new method override
    add_method_patch = textwrap.dedent(
        """\
        diff --git a/pkg/mod.py b/pkg/mod.py
        --- a/pkg/mod.py
        +++ b/pkg/mod.py
        @@ -12,0 +13,3 @@ class Base:
        +
        +    def bar(self):
        +        return 42
        """
    )
    m_an = analyze_patch("add_meth", add_method_patch, base)
    assert m_an.files[0].is_ast_noop is False
    assert ("Base", "bar") in m_an.files[0].added_methods
    assert "Base.foo" not in m_an.files[0].changed_qualnames

    # 6. Dict value-only edit vs key-membership edit
    dict_val_patch = textwrap.dedent(
        """\
        diff --git a/pkg/mod.py b/pkg/mod.py
        --- a/pkg/mod.py
        +++ b/pkg/mod.py
        @@ -4,1 +4,1 @@ class Base:
        -    VAL = {"a": 1, "b": 2}
        +    VAL = {"a": 1, "b": 99}
        """
    )
    dv_an = analyze_patch("dict_val", dict_val_patch, base)
    assert len(dv_an.files[0].attr_changes) == 1
    ac = dv_an.files[0].attr_changes[0]
    assert ac.class_qn == "Base"
    assert ac.attr_name == "VAL"
    assert ac.changed_dict_keys == frozenset({"b"})
    assert ac.keys_membership_changed is False

    # 7. Multi-line condition expansion (If.test)
    cond_patch = textwrap.dedent(
        """\
        diff --git a/pkg/mod.py b/pkg/mod.py
        --- a/pkg/mod.py
        +++ b/pkg/mod.py
        @@ -8,3 +8,3 @@ class Base:
                     x > 0
        -            and x < 10
        +            or x < 10
                 ):
        """
    )
    cond_an = analyze_patch("cond", cond_patch, base)
    assert {7, 8, 9}.issubset(cond_an.files[0].expanded_old_lines)

    # 8. Binary patch & deleted file
    bin_patch = "diff --git a/img.png b/img.png\nBinary files a/img.png and b/img.png differ\n"
    b_an = analyze_patch("bin", bin_patch, base)
    assert b_an.files[0].is_binary is True
    assert b_an.files[0].is_python is False


def test_selector_typing_comment_and_unknown_fixture(tmp_path: Path) -> None:
    selector = BlastSelector(MAPS_DIR)
    base = tmp_path / "sqlglot"
    (base / "sqlglot" / "typing").mkdir(parents=True)
    (base / "sqlglot" / "typing" / "snowflake.py").write_text("X = 1\n", encoding="utf-8")

    # Comment edit in sqlglot/typing/*.py still checks source-inspecting typing metadata test
    patch = textwrap.dedent(
        """\
        diff --git a/sqlglot/typing/snowflake.py b/sqlglot/typing/snowflake.py
        --- a/sqlglot/typing/snowflake.py
        +++ b/sqlglot/typing/snowflake.py
        @@ -1,1 +1,2 @@
         X = 1
        +# comment
        """
    )
    an = analyze_patch("typ_c", patch, base)
    sel = selector.select(an)
    assert selector.typing_meta_test is not None
    assert selector.typing_meta_test in sel.candidates

    # Unknown fixture under tests/fixtures/optimizer/ falls back conservatively
    unk_fix = textwrap.dedent(
        """\
        diff --git a/tests/fixtures/optimizer/brand_new.sql b/tests/fixtures/optimizer/brand_new.sql
        new file mode 100644
        --- /dev/null
        +++ b/tests/fixtures/optimizer/brand_new.sql
        @@ -0,0 +1,1 @@
        +SELECT 1;
        """
    )
    f_an = analyze_patch("unk_fix", unk_fix, base)
    f_sel = selector.select(f_an)
    assert selector.file_to_tests["tests/test_optimizer.py"].issubset(f_sel.candidates)

    # Direct tests/*.py method edit selects only that test; setUpClass edit selects whole class
    (base / "tests").mkdir(parents=True, exist_ok=True)
    (base / "tests" / "test_serde.py").write_text(
        textwrap.dedent(
            """\
            import unittest

            class TestSerde(unittest.TestCase):
                @classmethod
                def setUpClass(cls):
                    cls.x = 1

                def test_serde(self):
                    self.assertEqual(self.x, 1)
            """
        ),
        encoding="utf-8",
    )
    meth_patch = textwrap.dedent(
        """\
        diff --git a/tests/test_serde.py b/tests/test_serde.py
        --- a/tests/test_serde.py
        +++ b/tests/test_serde.py
        @@ -8,2 +8,2 @@ class TestSerde(unittest.TestCase):
             def test_serde(self):
        -        self.assertEqual(self.x, 1)
        +        self.assertEqual(self.x, 2)
        """
    )
    m_sel = selector.select(analyze_patch("t_meth", meth_patch, base))
    assert m_sel.candidates == frozenset(
        {selector.num_by_id["tests/test_serde.py::TestSerde::test_serde"]}
    )

    setup_patch = textwrap.dedent(
        """\
        diff --git a/tests/test_serde.py b/tests/test_serde.py
        --- a/tests/test_serde.py
        +++ b/tests/test_serde.py
        @@ -5,2 +5,2 @@ class TestSerde(unittest.TestCase):
             def setUpClass(cls):
        -        cls.x = 1
        +        cls.x = 2
        """
    )
    s_sel = selector.select(analyze_patch("t_setup", setup_patch, base))
    assert selector.file_to_tests["tests/test_serde.py"].issubset(s_sel.candidates)


def test_subtest_fail_fast_aborts_on_first_failure() -> None:
    import unittest

    executed: list[int] = []

    class Dummy(unittest.TestCase):
        def test_many_subtests(self) -> None:
            for i in range(50):
                with self.subTest(i=i):
                    executed.append(i)
                    if i == 2:
                        self.fail("boom")

    res = _FailFastResult()
    tc = Dummy("test_many_subtests")
    try:
        tc.run(res)
    except _SubtestAbort:
        pass
    assert res.failed is True
    assert executed == [0, 1, 2]


def test_verifier_isolation_import_error_timeout_and_overlay_cleanup(tmp_path: Path) -> None:
    repo = tmp_path / "fake_repo"
    (repo / "fake_pkg").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "fake_pkg" / "__init__.py").write_text("VALUE = 10\n", encoding="utf-8")
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "test_fake.py").write_text(
        textwrap.dedent(
            """\
            import time
            import unittest
            import fake_pkg

            class TestFake(unittest.TestCase):
                def test_val(self):
                    self.assertEqual(fake_pkg.VALUE, 10)

                def test_sub(self):
                    for i in range(10):
                        with self.subTest(i=i):
                            self.assertEqual(fake_pkg.VALUE, 10)

                def test_slow(self):
                    time.sleep(5)
                    self.assertTrue(True)
            """
        ),
        encoding="utf-8",
    )

    tmp_ov = tmp_path / "ov_tmp"
    tmp_ov.mkdir()

    ms_by_num = {0: 5, 1: 5, 2: 5}
    id_by_num = {
        0: "tests/test_fake.py::TestFake::test_val",
        1: "tests/test_fake.py::TestFake::test_sub",
        2: "tests/test_fake.py::TestFake::test_slow",
    }
    file_by_num = {0: "tests/test_fake.py", 1: "tests/test_fake.py", 2: "tests/test_fake.py"}
    file_to_tests = {"tests/test_fake.py": frozenset({0, 1, 2})}

    # Commit A breaks VALUE (fails both test_val and test_sub via subtest fail-fast)
    # Commit B keeps VALUE = 10 (passes both test_val and test_sub -> zero stale leakage!)
    # Commit C breaks syntax of fake_pkg -> ImportError marks ALL 3 tests in test_fake.py!
    # Commit D triggers timeout on test_slow -> keeps test_slow!
    analyses = {
        "c_fail": CommitAnalysis("c_fail", patched_contents={"fake_pkg/__init__.py": "VALUE = 99\n"}),
        "c_pass": CommitAnalysis("c_pass", patched_contents={"fake_pkg/__init__.py": "VALUE = 10\n"}),
        "c_imp": CommitAnalysis("c_imp", patched_contents={"fake_pkg/__init__.py": "def broken(:\n"}),
        "c_time": CommitAnalysis("c_time", patched_contents={"fake_pkg/__init__.py": "VALUE = 10\n"}),
    }
    cands = {
        "c_fail": frozenset({0, 1}),
        "c_pass": frozenset({0, 1}),
        "c_imp": frozenset({0}),
        "c_time": frozenset({2}),
    }

    outcomes = verify_batch(
        analyses=analyses,
        candidate_sets=cands,
        sqlglot_root=repo,
        ms_by_num=ms_by_num,
        id_by_num=id_by_num,
        file_by_num=file_by_num,
        file_to_tests=file_to_tests,
        per_commit_timeout_s=1.0,
        tmp_base=str(tmp_ov),
        force_verify_all=True,
    )

    assert outcomes["c_fail"].final_tests == frozenset({0, 1})
    assert outcomes["c_pass"].final_tests == frozenset()
    assert outcomes["c_imp"].final_tests == frozenset({0, 1, 2})
    assert outcomes["c_time"].final_tests == frozenset({2})
    assert outcomes["c_time"].timed_out is True

    # Parent must never have imported fake_pkg or sqlglot or tests
    assert "fake_pkg" not in sys.modules
    assert "sqlglot" not in sys.modules

    # Overlay directory must be completely cleaned up
    assert list(tmp_ov.iterdir()) == []


def test_sound_static_semantic_equivalence_and_adversarial_guards(tmp_path: Path) -> None:
    repo = tmp_path / "equiv_repo"
    (repo / "pkg").mkdir(parents=True)
    src = textwrap.dedent(
        """\
        from enum import Enum, auto

        class Mode(Enum):
            FAST = auto()
            SAFE = auto()

        class BadEnum(Enum):
            A = auto()
            def __eq__(self, other):
                return True

        class SafeBox:
            pass

        class CustomEqBox:
            def __eq__(self, other):
                return True

        class Worker:
            mode = Mode.FAST
            bad_mode = BadEnum.A

            def __init__(self):
                self._box: SafeBox | None = None
                self._bad: CustomEqBox | None = None
                self._mutated: SafeBox | None = None
                self.mode = Mode(self.mode)
                self.bad_mode = BadEnum(self.bad_mode)

            def build(self) -> None:
                self._box = SafeBox()
                self._bad = CustomEqBox()
                self._mutated = SafeBox()

            def check_safe_none(self) -> bool:
                return self._box is None

            def check_unsafe_custom_eq_none(self) -> bool:
                return self._bad is None

            def check_safe_enum(self) -> bool:
                return self.mode is Mode.FAST

            def check_unsafe_bad_enum(self) -> bool:
                return self.bad_mode is BadEnum.A

            def check_untyped_param(self, x) -> bool:
                return x is None

            def check_externally_mutated(self) -> bool:
                return self._mutated is None

        def external_mutator(w: Worker, other) -> None:
            w._mutated = other
        """
    )
    (repo / "pkg" / "mod.py").write_text(src, encoding="utf-8")

    def make_patch(old_line: str, new_line: str) -> str:
        old_lines = src.splitlines()
        idx = next(i for i, line in enumerate(old_lines, start=1) if old_line in line)
        return (
            f"diff --git a/pkg/mod.py b/pkg/mod.py\n"
            f"--- a/pkg/mod.py\n"
            f"+++ b/pkg/mod.py\n"
            f"@@ -{idx},1 +{idx},1 @@\n"
            f"-{old_lines[idx - 1]}\n"
            f"+{old_lines[idx - 1].replace(old_line, new_line)}\n"
        )

    # 1. Closed SafeBox | None attribute: `is None` -> `== None` is sound
    p_safe_none = make_patch("return self._box is None", "return self._box == None")
    an_safe_none = analyze_patch("safe_none", p_safe_none, repo)
    assert an_safe_none.files[0].is_ast_noop is True

    # 2. Closed stdlib Enum attribute without custom __eq__: `is Mode.FAST` -> `== Mode.FAST` is sound
    p_safe_enum = make_patch("return self.mode is Mode.FAST", "return self.mode == Mode.FAST")
    an_safe_enum = analyze_patch("safe_enum", p_safe_enum, repo)
    assert an_safe_enum.files[0].is_ast_noop is True

    # 3. Adversarial: CustomEqBox overrides __eq__: `is None` -> `== None` MUST be rejected
    p_bad_eq = make_patch("return self._bad is None", "return self._bad == None")
    an_bad_eq = analyze_patch("bad_eq", p_bad_eq, repo)
    assert an_bad_eq.files[0].is_ast_noop is False

    # 4. Adversarial: BadEnum overrides __eq__: `is BadEnum.A` -> `== BadEnum.A` MUST be rejected
    p_bad_enum = make_patch("return self.bad_mode is BadEnum.A", "return self.bad_mode == BadEnum.A")
    an_bad_enum = analyze_patch("bad_enum", p_bad_enum, repo)
    assert an_bad_enum.files[0].is_ast_noop is False

    # 5. Adversarial: Untyped parameter `x is None` -> `x == None` MUST be rejected
    p_untyped = make_patch("return x is None", "return x == None")
    an_untyped = analyze_patch("untyped", p_untyped, repo)
    assert an_untyped.files[0].is_ast_noop is False

    # 6. Adversarial: Attribute mutated outside class `self._mutated is None` -> `== None` MUST be rejected
    p_ext = make_patch("return self._mutated is None", "return self._mutated == None")
    an_ext = analyze_patch("ext", p_ext, repo)
    assert an_ext.files[0].is_ast_noop is False


def test_two_tier_fork_server_and_overlay_redirection(tmp_path: Path) -> None:
    import shutil

    repo = tmp_path / "sqlglot_repo"
    (repo / "sqlglot" / "dialects").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "sqlglot" / "__init__.py").write_text("CORE_VER = 1\n", encoding="utf-8")
    (repo / "sqlglot" / "dialects" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "sqlglot" / "dialects" / "leaf.py").write_text("LEAF_VAL = 'orig'\n", encoding="utf-8")
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "helpers.py").write_text(
        "import os\nFILE_DIR = os.path.dirname(__file__)\nFIXTURES_DIR = os.path.join(FILE_DIR, 'fixtures')\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_leaf.py").write_text(
        textwrap.dedent(
            """\
            import unittest
            from sqlglot.dialects import leaf

            class TestLeaf(unittest.TestCase):
                def test_leaf_val(self):
                    self.assertEqual(leaf.LEAF_VAL, "orig")
            """
        ),
        encoding="utf-8",
    )

    # Tier classification checks
    an_core = analyze_patch(
        "c_core",
        "diff --git a/sqlglot/__init__.py b/sqlglot/__init__.py\n"
        "--- a/sqlglot/__init__.py\n"
        "+++ b/sqlglot/__init__.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-CORE_VER = 1\n"
        "+CORE_VER = 2\n",
        repo,
    )
    assert can_use_preloaded_sqlglot(an_core, repo) is False
    assert can_use_preloaded_test_dialect(an_core) is False

    an_leaf_fail = analyze_patch(
        "c_leaf_fail",
        "diff --git a/sqlglot/dialects/leaf.py b/sqlglot/dialects/leaf.py\n"
        "--- a/sqlglot/dialects/leaf.py\n"
        "+++ b/sqlglot/dialects/leaf.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-LEAF_VAL = 'orig'\n"
        "+LEAF_VAL = 'patched'\n",
        repo,
    )
    assert can_use_preloaded_sqlglot(an_leaf_fail, repo) is True
    assert can_use_preloaded_test_dialect(an_leaf_fail) is True
    assert can_skip_pickle_roundtrip(an_leaf_fail) is True

    an_leaf_pass = analyze_patch(
        "c_leaf_pass",
        "diff --git a/sqlglot/dialects/leaf.py b/sqlglot/dialects/leaf.py\n"
        "--- a/sqlglot/dialects/leaf.py\n"
        "+++ b/sqlglot/dialects/leaf.py\n"
        "@@ -1,1 +1,2 @@\n"
        " LEAF_VAL = 'orig'\n"
        "+EXTRA = 1\n",
        repo,
    )

    tmp_ov = tmp_path / "t2_ov"
    tmp_ov.mkdir()
    ov_dir = create_overlay(repo, an_leaf_fail.patched_contents, tmp_base=str(tmp_ov))
    try:
        dummy_mod = type(sys)("dummy_sqlglot_pkg")
        dummy_mod.__file__ = f"{repo}/sqlglot/__init__.py"
        dummy_mod.__path__ = [f"{repo}/sqlglot"]  # type: ignore[attr-defined]
        sys.modules["_test_dummy_sqlglot_pkg"] = dummy_mod
        try:
            redirect_preloaded_modules_to_overlay(str(repo), ov_dir)
            assert dummy_mod.__file__ == f"{ov_dir}/sqlglot/__init__.py"
            assert dummy_mod.__path__ == [f"{ov_dir}/sqlglot"]  # type: ignore[attr-defined]
        finally:
            sys.modules.pop("_test_dummy_sqlglot_pkg", None)
            if ov_dir in sys.path:
                sys.path.remove(ov_dir)
    finally:
        shutil.rmtree(ov_dir, ignore_errors=True)

    # Full Tier-2 preloaded fork execution must isolate c_leaf_fail vs c_leaf_pass
    outcomes = verify_batch(
        analyses={"c_leaf_fail": an_leaf_fail, "c_leaf_pass": an_leaf_pass},
        candidate_sets={"c_leaf_fail": frozenset({0}), "c_leaf_pass": frozenset({0})},
        sqlglot_root=repo,
        ms_by_num={0: 10},
        id_by_num={0: "tests/test_leaf.py::TestLeaf::test_leaf_val"},
        file_by_num={0: "tests/test_leaf.py"},
        file_to_tests={"tests/test_leaf.py": frozenset({0})},
        tmp_base=str(tmp_ov),
        force_verify_all=True,
    )
    assert outcomes["c_leaf_fail"].final_tests == frozenset({0})
    assert outcomes["c_leaf_pass"].final_tests == frozenset()
    assert "sqlglot" not in sys.modules
    assert list(tmp_ov.iterdir()) == []
