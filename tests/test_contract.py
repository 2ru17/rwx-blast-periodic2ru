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
SQLGLOT_ROOT = REPO_ROOT.parent / "ai-notes" / "sqlglot"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diff_analyzer import CommitAnalysis, analyze_patch  # noqa: E402
from main import format_and_validate_output  # noqa: E402
from scheduler import compute_commit_cost_ms, duration_ms, schedule_tests  # noqa: E402
from selector import BlastSelector  # noqa: E402
from verifier import (  # noqa: E402
    _FailFastResult,
    _InlinePickleExecutor,
    _SubtestAbort,
    can_fast_verify_tpcds,
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

    # 7. Adversarial: @dataclass synthesizes __eq__, metaclass may override __eq__, subclass overrides __eq__
    extra_src = textwrap.dedent(
        """\
        from dataclasses import dataclass
        from enum import Enum, auto

        class StrMode(str, Enum):
            FAST = "FAST"
            SAFE = "SAFE"

        @dataclass
        class DataBox:
            x: int = 1

        class MetaEq(type):
            def __eq__(cls, other):
                return True

        class MetaBox(metaclass=MetaEq):
            pass

        class ParentBox:
            pass

        class ChildBadBox(ParentBox):
            def __eq__(self, other):
                return True

        class ExtraWorker:
            MODE = StrMode.FAST

            def __init__(self):
                self._dc: DataBox | None = None
                self._mb: MetaBox | None = None
                self._pb: ParentBox | None = None
                self.sm = self.MODE

            def set_all(self):
                self._dc = DataBox()
                self._mb = MetaBox()
                self._pb = ParentBox()

            def check_dc(self) -> bool:
                return self._dc is None

            def check_mb(self) -> bool:
                return self._mb is None

            def check_pb(self) -> bool:
                return self._pb is None

            def check_sm(self) -> bool:
                return self.sm is StrMode.FAST
        """
    )
    (repo / "pkg" / "extra.py").write_text(extra_src, encoding="utf-8")

    def make_extra_patch(old_line: str, new_line: str) -> str:
        lines = extra_src.splitlines()
        idx = next(i for i, line in enumerate(lines, start=1) if old_line in line)
        return (
            f"diff --git a/pkg/extra.py b/pkg/extra.py\n"
            f"--- a/pkg/extra.py\n"
            f"+++ b/pkg/extra.py\n"
            f"@@ -{idx},1 +{idx},1 @@\n"
            f"-{lines[idx - 1]}\n"
            f"+{lines[idx - 1].replace(old_line, new_line)}\n"
        )

    assert analyze_patch("dc", make_extra_patch("return self._dc is None", "return self._dc == None"), repo).files[0].is_ast_noop is False
    assert analyze_patch("mb", make_extra_patch("return self._mb is None", "return self._mb == None"), repo).files[0].is_ast_noop is False
    assert analyze_patch("pb", make_extra_patch("return self._pb is None", "return self._pb == None"), repo).files[0].is_ast_noop is False

    # Closed-world (str, Enum): safe only while NO file assigns raw str to MODE or .sm
    p_sm = make_extra_patch("return self.sm is StrMode.FAST", "return self.sm == StrMode.FAST")
    assert analyze_patch("sm_closed", p_sm, repo).files[0].is_ast_noop is True

    # Adding a subclass in another file setting MODE = "FAST" (raw str) MUST reject equivalence
    (repo / "pkg" / "sub.py").write_text(
        'from pkg.extra import ExtraWorker\nclass SubWorker(ExtraWorker):\n    MODE = "FAST"\n',
        encoding="utf-8",
    )
    assert analyze_patch("sm_raw_str", p_sm, repo).files[0].is_ast_noop is False

    # 8. Source-inspecting test preservation under AST/comment/semantic-equiv no-op
    selector = BlastSelector(MAPS_DIR)
    (repo / "sqlglot" / "typing").mkdir(parents=True)
    typ_src = (
        "from enum import Enum\n"
        "class M(Enum):\n"
        "    A = 1\n"
        "class W:\n"
        "    def __init__(self):\n"
        "        self.m = M.A\n"
        "    def f(self):\n"
        "        return self.m is M.A\n"
    )
    (repo / "sqlglot" / "typing" / "bigquery.py").write_text(typ_src, encoding="utf-8")
    typ_patch = (
        "diff --git a/sqlglot/typing/bigquery.py b/sqlglot/typing/bigquery.py\n"
        "--- a/sqlglot/typing/bigquery.py\n"
        "+++ b/sqlglot/typing/bigquery.py\n"
        "@@ -8,1 +8,1 @@\n"
        "-        return self.m is M.A\n"
        "+        return self.m == M.A\n"
    )
    an_typ = analyze_patch("typ_equiv", typ_patch, repo)
    assert an_typ.files[0].is_ast_noop is True
    assert an_typ.files[0].raw_text_changed is True
    sel_typ = selector.select(an_typ)
    assert selector.typing_meta_test is not None
    assert selector.typing_meta_test in sel_typ.candidates


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


def test_fast_verify_tpcds_and_setup_and_pickle_guards(tmp_path: Path) -> None:
    repo = tmp_path / "tpcds_guard_repo"
    (repo / "sqlglot" / "optimizer").mkdir(parents=True)
    (repo / "tests" / "fixtures" / "optimizer").mkdir(parents=True)
    (repo / "sqlglot" / "helper.py").write_text(
        "def unused_fn(x):\n    return x + 1\n\ndef dict_depth(d):\n    try:\n        return 1\n    except AttributeError:\n        return 0\n",
        encoding="utf-8",
    )
    (repo / "sqlglot" / "optimizer" / "canonicalize.py").write_text("def c(x):\n    return x\n", encoding="utf-8")
    (repo / "tests" / "fixtures" / "optimizer" / "other.sql").write_text("SELECT 1;\n", encoding="utf-8")
    (repo / "tests" / "helpers.py").write_text("X = 1\n", encoding="utf-8")

    hashes = ("abc123",)
    exec_lines = {"sqlglot/helper.py": frozenset({8})}
    exec_qns = {"sqlglot/helper.py": frozenset({"dict_depth"})}

    # Edit in executed line of helper outside sqlglot/executor/ MUST disable fast TPC-DS
    an_exec_line = analyze_patch(
        "exec_line",
        "diff --git a/sqlglot/helper.py b/sqlglot/helper.py\n"
        "--- a/sqlglot/helper.py\n"
        "+++ b/sqlglot/helper.py\n"
        "@@ -8,1 +8,1 @@\n"
        "-        return 0\n"
        "+        return 1\n",
        repo,
    )
    assert can_fast_verify_tpcds(an_exec_line, hashes, exec_lines, exec_qns) is False

    # Insertion in executed function header (outside exec_lines) MUST disable via exec_qualnames
    an_exec_qn = analyze_patch(
        "exec_qn",
        "diff --git a/sqlglot/helper.py b/sqlglot/helper.py\n"
        "--- a/sqlglot/helper.py\n"
        "+++ b/sqlglot/helper.py\n"
        "@@ -4,1 +4,2 @@\n"
        " def dict_depth(d):\n"
        "+    return 99\n",
        repo,
    )
    assert can_fast_verify_tpcds(an_exec_qn, hashes, exec_lines, exec_qns) is False

    # Any fixture or tests/helpers.py edit MUST disable fast TPC-DS
    an_fix = analyze_patch(
        "fix",
        "diff --git a/tests/fixtures/optimizer/other.sql b/tests/fixtures/optimizer/other.sql\n"
        "--- a/tests/fixtures/optimizer/other.sql\n"
        "+++ b/tests/fixtures/optimizer/other.sql\n"
        "@@ -1,1 +1,1 @@\n"
        "-SELECT 1;\n"
        "+SELECT 2;\n",
        repo,
    )
    assert can_fast_verify_tpcds(an_fix, hashes, exec_lines, exec_qns) is False

    an_helpers = analyze_patch(
        "helpers",
        "diff --git a/tests/helpers.py b/tests/helpers.py\n"
        "--- a/tests/helpers.py\n"
        "+++ b/tests/helpers.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-X = 1\n"
        "+X = 2\n",
        repo,
    )
    assert can_fast_verify_tpcds(an_helpers, hashes, exec_lines, exec_qns) is False

    # Missing exec_lines fails closed
    an_ok = analyze_patch(
        "ok",
        "diff --git a/sqlglot/optimizer/canonicalize.py b/sqlglot/optimizer/canonicalize.py\n"
        "--- a/sqlglot/optimizer/canonicalize.py\n"
        "+++ b/sqlglot/optimizer/canonicalize.py\n"
        "@@ -2,1 +2,1 @@\n"
        "-    return x\n"
        "+    return x\n",
        repo,
    )
    an_ok2 = analyze_patch(
        "ok2",
        "diff --git a/sqlglot/helper.py b/sqlglot/helper.py\n"
        "--- a/sqlglot/helper.py\n"
        "+++ b/sqlglot/helper.py\n"
        "@@ -2,1 +2,1 @@\n"
        "-    return x + 1\n"
        "+    return x + 2\n",
        repo,
    )
    assert can_fast_verify_tpcds(an_ok2, hashes, {}, exec_qns) is False
    assert can_fast_verify_tpcds(an_ok2, hashes, exec_lines, exec_qns) is True
    assert an_ok.files[0].is_ast_noop is True

    # _InlinePickleExecutor always roundtrips through pickle (rejects unpicklable local function)
    with _InlinePickleExecutor() as pool:
        fut = pool.submit(lambda x: x + 1, 5)
        with pytest.raises(Exception):
            fut.result()

    # Fixture modification disables TestExecutor.setUpClass skipping and _LazyOptimizerConn
    setup_repo = tmp_path / "setup_guard_repo"
    (setup_repo / "tests" / "fixtures").mkdir(parents=True)
    (setup_repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (setup_repo / "tests" / "fixtures" / "a.sql").write_text("SELECT 1;\n", encoding="utf-8")
    (setup_repo / "tests" / "test_executor.py").write_text(
        textwrap.dedent(
            """\
            import unittest
            class TestExecutor(unittest.TestCase):
                @classmethod
                def setUpClass(cls):
                    raise RuntimeError("setup ran and failed")
                def test_simple(self):
                    self.assertTrue(True)
            """
        ),
        encoding="utf-8",
    )
    tmp_ov = tmp_path / "setup_ov"
    tmp_ov.mkdir()
    an_fix_setup = CommitAnalysis(
        "c_fix_setup",
        patched_contents={"tests/fixtures/a.sql": "SELECT 2;\n"},
    )
    out_setup = verify_batch(
        analyses={"c_fix_setup": an_fix_setup},
        candidate_sets={"c_fix_setup": frozenset({0})},
        sqlglot_root=setup_repo,
        ms_by_num={0: 10},
        id_by_num={0: "tests/test_executor.py::TestExecutor::test_simple"},
        file_by_num={0: "tests/test_executor.py"},
        file_to_tests={"tests/test_executor.py": frozenset({0})},
        tmp_base=str(tmp_ov),
        force_verify_all=True,
    )
    # setUpClass MUST NOT be skipped when a fixture is modified -> test_simple fails (kept!)
    assert out_setup["c_fix_setup"].final_tests == frozenset({0})

    # tearDownClass failure MUST NOT mark tests in that class as passed
    td_repo = tmp_path / "teardown_repo"
    (td_repo / "tests").mkdir(parents=True)
    (td_repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (td_repo / "tests" / "test_td.py").write_text(
        textwrap.dedent(
            """\
            import unittest
            class TestTD(unittest.TestCase):
                @classmethod
                def tearDownClass(cls):
                    raise RuntimeError("teardown boom")
                def test_ok(self):
                    self.assertTrue(True)
            """
        ),
        encoding="utf-8",
    )
    out_td = verify_batch(
        analyses={"c_td": CommitAnalysis("c_td", patched_contents={})},
        candidate_sets={"c_td": frozenset({0})},
        sqlglot_root=td_repo,
        ms_by_num={0: 10},
        id_by_num={0: "tests/test_td.py::TestTD::test_ok"},
        file_by_num={0: "tests/test_td.py"},
        file_to_tests={"tests/test_td.py": frozenset({0})},
        tmp_base=str(tmp_ov),
        force_verify_all=True,
    )
    assert out_td["c_td"].final_tests == frozenset({0})

    # _InlinePickleExecutor.__init__ and map() also enforce pickle roundtrip
    with pytest.raises(Exception):
        _InlinePickleExecutor(initializer=lambda: None)
    with _InlinePickleExecutor() as pool:
        with pytest.raises(Exception):
            list(pool.map(lambda x: x + 1, [1, 2]))


def test_diff_analyzer_soundness_regressions(tmp_path: Path) -> None:
    repo = tmp_path / "reg_repo"
    (repo / "pkg").mkdir(parents=True)
    base_src = textwrap.dedent(
        """\
        A: int = 1
        B = A

        class SafeBox:
            pass

        class W:
            def __init__(self, p):
                self._b: SafeBox | None = p or None
                self._ok: SafeBox | None = None

            def make(self):
                self._ok = SafeBox()

            def check_untrusted_rhs(self) -> bool:
                return self._b is None

            def check_lambda_shadow(self):
                x = self._ok
                return (lambda x: x is None)(None)
        """
    )
    (repo / "pkg" / "m.py").write_text(base_src, encoding="utf-8")

    # 1. Added top-level function MUST NOT be treated as AST no-op
    p_add_fn = (
        "diff --git a/pkg/m.py b/pkg/m.py\n"
        "--- a/pkg/m.py\n"
        "+++ b/pkg/m.py\n"
        "@@ -20,1 +20,4 @@\n"
        "         return (lambda x: x is None)(None)\n"
        "+\n"
        "+def newly_added_top():\n"
        "+    return 1 / 0\n"
    )
    an_add = analyze_patch("add_top_fn", p_add_fn, repo)
    assert an_add.files[0].is_ast_noop is False
    assert an_add.files[0].top_level_code_changed is True

    # 2. Reordering top-level statements MUST NOT be treated as AST no-op
    p_reorder = (
        "diff --git a/pkg/m.py b/pkg/m.py\n"
        "--- a/pkg/m.py\n"
        "+++ b/pkg/m.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-A: int = 1\n"
        "-B = A\n"
        "+B = A\n"
        "+A: int = 1\n"
    )
    an_reord = analyze_patch("reorder", p_reorder, repo)
    assert an_reord.files[0].is_ast_noop is False
    assert an_reord.files[0].top_level_code_changed is True

    # 3. Changing AnnAssign annotation MUST NOT be treated as AST no-op
    p_ann = (
        "diff --git a/pkg/m.py b/pkg/m.py\n"
        "--- a/pkg/m.py\n"
        "+++ b/pkg/m.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-A: int = 1\n"
        "+A: UndefinedType = 1\n"
    )
    an_ann = analyze_patch("ann_chg", p_ann, repo)
    assert an_ann.files[0].is_ast_noop is False

    # 4. Untrusted RHS `p or None` MUST NOT allow `is None` -> `== None`
    p_rhs = (
        "diff --git a/pkg/m.py b/pkg/m.py\n"
        "--- a/pkg/m.py\n"
        "+++ b/pkg/m.py\n"
        "@@ -16,1 +16,1 @@\n"
        "-        return self._b is None\n"
        "+        return self._b == None\n"
    )
    assert analyze_patch("untrusted_rhs", p_rhs, repo).files[0].is_ast_noop is False

    # 5. Lambda parameter shadowing `x` MUST NOT allow `is None` -> `== None`
    p_lam = (
        "diff --git a/pkg/m.py b/pkg/m.py\n"
        "--- a/pkg/m.py\n"
        "+++ b/pkg/m.py\n"
        "@@ -20,1 +20,1 @@\n"
        "-        return (lambda x: x is None)(None)\n"
        "+        return (lambda x: x == None)(None)\n"
    )
    assert analyze_patch("lambda_shadow", p_lam, repo).files[0].is_ast_noop is False


def test_duplicate_property_and_import_time_qualname_selection(tmp_path: Path) -> None:
    repo = tmp_path / "dup_repo"
    (repo / "pkg").mkdir(parents=True)
    src = textwrap.dedent(
        """\
        class Item:
            def __init__(self, v):
                self._v = v

            @property
            def val(self):
                return self._v

            @val.setter
            def val(self, v):
                self._v = v
        """
    )
    (repo / "pkg" / "item.py").write_text(src, encoding="utf-8")
    p_getter = (
        "diff --git a/pkg/item.py b/pkg/item.py\n"
        "--- a/pkg/item.py\n"
        "+++ b/pkg/item.py\n"
        "@@ -7,1 +7,1 @@\n"
        "-        return self._v\n"
        "+        return None\n"
    )
    an = analyze_patch("dup_getter", p_getter, repo)
    assert an.files[0].is_ast_noop is False
    assert "Item.val" in an.files[0].changed_qualnames

    # Live sqlglot Expression.type getter edit and Expr.__init_subclass__ edit
    selector = BlastSelector(MAPS_DIR)
    if SQLGLOT_ROOT.is_dir():
        core_lines = (SQLGLOT_ROOT / "sqlglot" / "expressions" / "core.py").read_text(
            encoding="utf-8"
        ).splitlines()
        idx_type = next(
            i
            for i, ln in enumerate(core_lines, start=1)
            if ln == "        return self._type" and i > 960
        )
        p_expr_type = (
            "diff --git a/sqlglot/expressions/core.py b/sqlglot/expressions/core.py\n"
            "--- a/sqlglot/expressions/core.py\n"
            "+++ b/sqlglot/expressions/core.py\n"
            f"@@ -{idx_type},1 +{idx_type},1 @@\n"
            "-        return self._type\n"
            "+        return None\n"
        )
        an_type = analyze_patch("expr_type_getter", p_expr_type, SQLGLOT_ROOT)
        assert an_type.files[0].is_ast_noop is False
        assert "Expression.type" in an_type.files[0].changed_qualnames
        assert len(selector.select(an_type).candidates) > 50

        idx_init_sub = next(
            i
            for i, ln in enumerate(core_lines, start=1)
            if ln == "        cls.key = cls.__name__.lower()"
        )
        p_init_sub = (
            "diff --git a/sqlglot/expressions/core.py b/sqlglot/expressions/core.py\n"
            "--- a/sqlglot/expressions/core.py\n"
            "+++ b/sqlglot/expressions/core.py\n"
            f"@@ -{idx_init_sub},1 +{idx_init_sub},1 @@\n"
            "-        cls.key = cls.__name__.lower()\n"
            '+        cls.key = "x" + cls.__name__.lower()\n'
        )
        an_sub = analyze_patch("expr_init_sub", p_init_sub, SQLGLOT_ROOT)
        assert an_sub.files[0].is_ast_noop is False
        assert "Expr.__init_subclass__" in an_sub.files[0].changed_qualnames
        sel_sub = selector.select(an_sub)
        assert len(sel_sub.candidates) > 1000
        assert not can_fast_verify_tpcds(
            an_sub,
            tpcds_plan_hashes=selector.tpcds_plan_hashes,
            tpcds_exec_lines=selector.tpcds_exec_lines,
            tpcds_exec_qualnames=selector.tpcds_exec_qualnames,
            import_time_qualnames=selector.import_time_qualnames,
        )


def test_closed_world_is_eq_hardening_edge_cases(tmp_path: Path) -> None:
    def run_case(name: str, files: dict[str, str], target_file: str, old_ln: str, new_ln: str) -> bool:
        repo = tmp_path / name
        for rel, content in files.items():
            p = repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(textwrap.dedent(content), encoding="utf-8")
        lines = (repo / target_file).read_text(encoding="utf-8").splitlines()
        idx = next(i for i, ln in enumerate(lines, start=1) if old_ln in ln)
        patch = (
            f"diff --git a/{target_file} b/{target_file}\n"
            f"--- a/{target_file}\n"
            f"+++ b/{target_file}\n"
            f"@@ -{idx},1 +{idx},1 @@\n"
            f"-{lines[idx - 1]}\n"
            f"+{lines[idx - 1].replace(old_ln, new_ln)}\n"
        )
        return analyze_patch(name, patch, repo).files[0].is_ast_noop

    # 1. For-loop attribute store `for self.kind in ["a"]:` MUST reject
    assert not run_case(
        "for_attr_store",
        {
            "pkg/m.py": """\
                from enum import Enum
                class K(str, Enum):
                    A = "a"
                class W:
                    def __init__(self):
                        self.kind = K.A
                    def poison(self):
                        for self.kind in ["a"]:
                            pass
                    def check(self):
                        return self.kind is K.A
            """
        },
        "pkg/m.py",
        "return self.kind is K.A",
        "return self.kind == K.A",
    )

    # 2. Nested `nonlocal` rebinding MUST reject
    assert not run_case(
        "nonlocal_rebind",
        {
            "pkg/m.py": """\
                from enum import Enum
                class K(str, Enum):
                    A = "a"
                class W:
                    def __init__(self):
                        self.kind = K.A
                    def check(self):
                        x = self.kind
                        def inner():
                            nonlocal x
                            x = "a"
                        inner()
                        return x is K.A
            """
        },
        "pkg/m.py",
        "return x is K.A",
        "return x == K.A",
    )

    # 3. Conditional class-body store `if True: KIND = "a"` MUST reject
    assert not run_case(
        "cond_cls_body",
        {
            "pkg/m.py": """\
                from enum import Enum
                class K(str, Enum):
                    A = "a"
                class Base:
                    KIND = K.A
                    def __init__(self):
                        self.kind = self.KIND
                    def check(self):
                        return self.kind is K.A
                class Sub(Base):
                    if True:
                        KIND = "a"
            """
        },
        "pkg/m.py",
        "return self.kind is K.A",
        "return self.kind == K.A",
    )

    # 4. Slice `[:]` on typed mapping extraction MUST reject
    assert not run_case(
        "slice_mapping_extract",
        {
            "pkg/m.py": """\
                class SafeBox:
                    pass
                class Holder:
                    def __init__(self, m: dict[str, SafeBox]):
                        self.m = m
                        self._b: SafeBox | None = None
                    def get(self):
                        if self._b is None:
                            self._b = list(self.m.values())[:]
                        return self._b
            """
        },
        "pkg/m.py",
        "if self._b is None:",
        "if self._b == None:",
    )

    # 5. Mapping mutation `self.m.update(...)` MUST reject
    assert not run_case(
        "mapping_update_mut",
        {
            "pkg/m.py": """\
                class SafeBox:
                    pass
                class Holder:
                    def __init__(self, m: dict[str, SafeBox]):
                        self.m = m
                        self._b: SafeBox | None = None
                    def poison(self, bad):
                        self.m.update({"x": bad})
                    def get(self):
                        if self._b is None:
                            self._b = list(self.m.values())[0]
                        return self._b
            """
        },
        "pkg/m.py",
        "if self._b is None:",
        "if self._b == None:",
    )

    # 6. Multi-dot subclass `class Sub(pkg.m.SafeBox): def __eq__` MUST reject
    assert not run_case(
        "multidot_sub_eq",
        {
            "pkg/m.py": """\
                class SafeBox:
                    pass
                class Holder:
                    def __init__(self):
                        self._b: SafeBox | None = None
                    def set_b(self):
                        self._b = SafeBox()
                    def check(self):
                        return self._b is None
            """,
            "pkg/evil.py": """\
                import pkg.m
                class EvilSub(pkg.m.SafeBox):
                    def __eq__(self, other):
                        return True
            """,
        },
        "pkg/m.py",
        "return self._b is None",
        "return self._b == None",
    )

    # 7. Qualified monkeypatch `mod.SafeBox.__eq__ = ...` MUST reject
    assert not run_case(
        "qualified_eq_patch",
        {
            "pkg/m.py": """\
                class SafeBox:
                    pass
                class Holder:
                    def __init__(self):
                        self._b: SafeBox | None = None
                    def set_b(self):
                        self._b = SafeBox()
                    def check(self):
                        return self._b is None
            """,
            "pkg/patcher.py": """\
                import pkg.m as mod
                mod.SafeBox.__eq__ = lambda self, o: True
            """,
        },
        "pkg/m.py",
        "return self._b is None",
        "return self._b == None",
    )

    # 8. 2-arg functional Enum creation `K("Dyn", ["A"])` MUST reject
    assert not run_case(
        "two_arg_enum_ctor",
        {
            "pkg/m.py": """\
                from enum import Enum
                class K(str, Enum):
                    A = "a"
                class W:
                    def __init__(self):
                        self.kind = K("Dyn", ["A"])
                    def check(self):
                        return self.kind is K.A
            """
        },
        "pkg/m.py",
        "return self.kind is K.A",
        "return self.kind == K.A",
    )


def test_class_cleanup_failure_eager_map_and_lazy_heavy_submodules(tmp_path: Path) -> None:
    import subprocess

    # 1. doClassCleanups() exception stored in tearDown_exceptions MUST fail the test class
    repo = tmp_path / "cleanup_repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "test_cleanup.py").write_text(
        textwrap.dedent(
            """\
            import unittest

            class TestCleanup(unittest.TestCase):
                @classmethod
                def setUpClass(cls):
                    def boom():
                        raise RuntimeError("cleanup boom")
                    cls.addClassCleanup(boom)

                def test_ok(self):
                    self.assertEqual(1, 1)
            """
        ),
        encoding="utf-8",
    )
    an_cl = analyze_patch("c_cl", "", repo)
    out_cl = verify_batch(
        analyses={"c_cl": an_cl},
        candidate_sets={"c_cl": frozenset({0})},
        sqlglot_root=repo,
        ms_by_num={0: 20},
        id_by_num={0: "tests/test_cleanup.py::TestCleanup::test_ok"},
        file_by_num={0: "tests/test_cleanup.py"},
        file_to_tests={"tests/test_cleanup.py": frozenset({0})},
        force_verify_all=True,
    )
    assert out_cl["c_cl"].final_tests == frozenset({0})

    # 2. _InlinePickleExecutor.map eagerly checks pickling even if consumer breaks early
    with _InlinePickleExecutor() as pool:
        with pytest.raises(Exception):
            pool.map(abs, [1, lambda x: x])

    # 3. _install_lazy_heavy_deps supports submodule imports and pandas.testing attributes
    probe_code = (
        f"import sys; sys.path.insert(0, {str(SRC_DIR)!r}); "
        "import verifier; verifier._install_lazy_heavy_deps(); "
        "import numpy.typing; "
        "from pandas.errors import EmptyDataError; "
        "from pandas.testing import assert_series_equal; "
        "assert hasattr(numpy.typing, 'NDArray') and EmptyDataError and callable(assert_series_equal)"
    )
    res = subprocess.run([sys.executable, "-c", probe_code], capture_output=True, text=True)
    assert res.returncode == 0, f"lazy heavy submodule probe failed: {res.stderr}"


def test_verifier_early_exit_preserves_unverified_and_failed_tests(tmp_path: Path) -> None:
    repo = tmp_path / "early_exit_repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    marker = tmp_path / "ran.log"
    (repo / "tests" / "test_early.py").write_text(
        textwrap.dedent(
            f"""\
            import unittest
            from pathlib import Path

            MARKER = Path({str(marker)!r})

            class TestEarly(unittest.TestCase):
                def test_fast_fail(self):
                    with MARKER.open("a", encoding="utf-8") as fh:
                        fh.write("fast_fail\\n")
                    self.fail("intentional")

                def test_slow_unrun_fail(self):
                    with MARKER.open("a", encoding="utf-8") as fh:
                        fh.write("slow_unrun_fail\\n")
                    self.fail("must remain kept even when skipped")

                def test_slow_unrun_pass(self):
                    with MARKER.open("a", encoding="utf-8") as fh:
                        fh.write("slow_unrun_pass\\n")
                    self.assertTrue(True)
            """
        ),
        encoding="utf-8",
    )
    tmp_ov = tmp_path / "ee_ov"
    tmp_ov.mkdir()

    # Failure-triggered early exit: test 0 (10ms) runs first and fails; remaining tests 1+2 (100ms)
    # yield residual gain 100ms < MIN_RESIDUAL_GAIN_MS, so verification stops immediately and keeps
    # both the failed test AND all unverified tests.
    ms_by_num = {0: 10, 1: 50, 2: 50, 3: 15_000}
    id_by_num = {
        0: "tests/test_early.py::TestEarly::test_fast_fail",
        1: "tests/test_early.py::TestEarly::test_slow_unrun_fail",
        2: "tests/test_early.py::TestEarly::test_slow_unrun_pass",
        3: "tests/test_early.py::TestEarly::test_kept_huge",
    }
    file_by_num = {n: "tests/test_early.py" for n in ms_by_num}
    file_to_tests = {"tests/test_early.py": frozenset(ms_by_num)}

    an = CommitAnalysis("c_fail_ee", patched_contents={})
    outcomes = verify_batch(
        analyses={"c_fail_ee": an},
        candidate_sets={"c_fail_ee": frozenset({0, 1, 2})},
        sqlglot_root=repo,
        ms_by_num=ms_by_num,
        id_by_num=id_by_num,
        file_by_num=file_by_num,
        file_to_tests=file_to_tests,
        tmp_base=str(tmp_ov),
        force_verify_all=True,
    )
    assert outcomes["c_fail_ee"].final_tests == frozenset({0, 1, 2})
    assert marker.read_text(encoding="utf-8").splitlines() == ["fast_fail"]

    # keep_unverified-triggered early exit: test 3 (15_000ms) is in keep_unverified, so remaining
    # ordinary tests 1+2 (300ms) have residual gain 300ms < MIN_RESIDUAL_GAIN_MS; child exits
    # before running any test and preserves 100% of keep_unverified + unverified candidates.
    marker.unlink()
    from verifier import VerifyPlan, _run_fork_queue

    plan = VerifyPlan(
        keep_unverified=frozenset({3}),
        to_verify=frozenset({1, 2}),
        est_cpu_ms=400,
        max_sim_gain_ms=300,
        should_verify=True,
        use_preloaded_sqlglot=False,
        fast_tpcds=False,
    )
    ku_outcomes: dict = {}
    _run_fork_queue(
        [("c_ku_ee", an, frozenset({1, 2, 3}), plan)],
        ku_outcomes,
        repo,
        id_by_num,
        file_to_tests,
        (),
        5.0,
        10**9,
        1,
        str(tmp_ov),
        ms_by_num,
    )
    assert ku_outcomes["c_ku_ee"].final_tests == frozenset({1, 2, 3})
    assert not marker.exists()
    assert list(tmp_ov.iterdir()) == []


def test_ordinary_failure_skips_remaining_ordinary_but_still_runs_fast_tpcds(
    tmp_path: Path, monkeypatch
) -> None:
    import verifier

    repo = tmp_path / "tpcds_jump_repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    marker = tmp_path / "jump.log"
    (repo / "tests" / "test_jump.py").write_text(
        textwrap.dedent(
            f"""\
            import unittest
            from pathlib import Path

            MARKER = Path({str(marker)!r})

            class TestJump(unittest.TestCase):
                def test_fail_first(self):
                    with MARKER.open("a", encoding="utf-8") as fh:
                        fh.write("fail_first\\n")
                    self.fail("boom")

                def test_skip_second(self):
                    with MARKER.open("a", encoding="utf-8") as fh:
                        fh.write("skip_second\\n")
                    self.assertTrue(True)
            """
        ),
        encoding="utf-8",
    )
    tpcds_called = tmp_path / "tpcds_called.txt"

    def fake_fast_tpcds(hashes, soft_deadline_s):
        tpcds_called.write_text("ran", encoding="utf-8")
        return True

    monkeypatch.setattr(verifier, "_fast_verify_tpcds", fake_fast_tpcds)

    ms_by_num = {0: 10, 1: 100, 1106: 21_454}
    id_by_num = {
        0: "tests/test_jump.py::TestJump::test_fail_first",
        1: "tests/test_jump.py::TestJump::test_skip_second",
        1106: "tests/test_executor.py::TestExecutor::test_execute_tpcds",
    }
    file_to_tests = {
        "tests/test_jump.py": frozenset({0, 1}),
        "tests/test_executor.py": frozenset({1106}),
    }
    tmp_ov = tmp_path / "ov"
    tmp_ov.mkdir()

    an = CommitAnalysis("c_jump", patched_contents={})
    plan = verifier.VerifyPlan(
        keep_unverified=frozenset(),
        to_verify=frozenset({0, 1, 1106}),
        est_cpu_ms=3_500,
        max_sim_gain_ms=23_564,
        should_verify=True,
        use_preloaded_sqlglot=False,
        fast_tpcds=True,
    )
    outcomes: dict = {}
    verifier._run_fork_queue(
        [("c_jump", an, frozenset({0, 1, 1106}), plan)],
        outcomes,
        repo,
        id_by_num,
        file_to_tests,
        ("dummy_hash",),
        5.0,
        10**9,
        1,
        str(tmp_ov),
        ms_by_num,
    )
    # fast_tpcds runs first and sheds #1106 -> test 0 fails -> test 1 skipped (kept unverified!)
    assert outcomes["c_jump"].final_tests == frozenset({0, 1})
    assert marker.read_text(encoding="utf-8").splitlines() == ["fail_first"]
    assert tpcds_called.read_text(encoding="utf-8") == "ran"

    # When fast_tpcds FAILS, kept={1106} triggers early exit BEFORE running unworthwhile ordinary tests!
    marker.unlink()
    tpcds_called.unlink()

    def fake_fast_tpcds_fail(hashes, soft_deadline_s):
        tpcds_called.write_text("failed", encoding="utf-8")
        return False

    monkeypatch.setattr(verifier, "_fast_verify_tpcds", fake_fast_tpcds_fail)
    outcomes_fail: dict = {}
    verifier._run_fork_queue(
        [("c_jump_fail", an, frozenset({0, 1, 1106}), plan)],
        outcomes_fail,
        repo,
        id_by_num,
        file_to_tests,
        ("dummy_hash",),
        5.0,
        10**9,
        1,
        str(tmp_ov),
        ms_by_num,
    )
    assert outcomes_fail["c_jump_fail"].final_tests == frozenset({0, 1, 1106})
    assert not marker.exists()
    assert tpcds_called.read_text(encoding="utf-8") == "failed"
    assert list(tmp_ov.iterdir()) == []


def test_deferred_heavy_conditional_execution_and_timeout_checkpoint(tmp_path: Path) -> None:
    import verifier

    repo = tmp_path / "dh_repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    marker = tmp_path / "dh.log"
    (repo / "tests" / "test_parser.py").write_text(
        textwrap.dedent(
            f"""\
            import time
            import unittest
            from pathlib import Path

            MARKER = Path({str(marker)!r})

            class TestParser(unittest.TestCase):
                def test_ord_pass(self):
                    with MARKER.open("a", encoding="utf-8") as fh:
                        fh.write("ord_pass\\n")
                    self.assertTrue(True)

                def test_ord_fail_0(self):
                    with MARKER.open("a", encoding="utf-8") as fh:
                        fh.write("ord_fail_0\\n")
                    self.fail("boom")

                def test_ord_fail_1(self):
                    self.fail("boom")

                def test_ord_fail_2(self):
                    self.fail("boom")

                def test_ord_fail_3(self):
                    self.fail("boom")

                def test_ord_fail_4(self):
                    self.fail("boom")

                def test_ord_fail_5(self):
                    self.fail("boom")

                def test_max_nodes(self):
                    with MARKER.open("a", encoding="utf-8") as fh:
                        fh.write("max_nodes\\n")
                    if Path("tests/sleep_heavy.flag").exists():
                        time.sleep(5)
                    self.assertTrue(True)
            """
        ),
        encoding="utf-8",
    )

    tmp_ov = tmp_path / "dh_ov"
    tmp_ov.mkdir()

    ms_by_num = {
        0: 500,
        1: 500,
        2: 500,
        3: 500,
        4: 500,
        5: 500,
        6: 500,
        1490: 28_817,
    }
    id_by_num = {
        0: "tests/test_parser.py::TestParser::test_ord_pass",
        1: "tests/test_parser.py::TestParser::test_ord_fail_0",
        2: "tests/test_parser.py::TestParser::test_ord_fail_1",
        3: "tests/test_parser.py::TestParser::test_ord_fail_2",
        4: "tests/test_parser.py::TestParser::test_ord_fail_3",
        5: "tests/test_parser.py::TestParser::test_ord_fail_4",
        6: "tests/test_parser.py::TestParser::test_ord_fail_5",
        1490: "tests/test_parser.py::TestParser::test_max_nodes",
    }
    file_by_num = {n: "tests/test_parser.py" for n in ms_by_num}
    file_to_tests = {"tests/test_parser.py": frozenset(ms_by_num)}

    an = CommitAnalysis("c_dh", patched_contents={})

    # 1. Ordinary tests fail broadly (6 failures > DEFERRED_HEAVY_MAX_KEPT_TESTS) -> skip #1490
    broad_cands = frozenset({1, 2, 3, 4, 5, 6, 1490})
    broad_plan = verifier.plan_commit_verification(
        broad_cands, ms_by_num, file_by_num, analysis=an, id_by_num=id_by_num
    )
    assert broad_plan.should_verify is True
    assert broad_plan.deferred_heavy == frozenset({1490})
    broad_out: dict = {}
    verifier._run_fork_queue(
        [("c_broad", an, broad_cands, broad_plan)],
        broad_out,
        repo,
        id_by_num,
        file_to_tests,
        (),
        10.0,
        10**9,
        1,
        str(tmp_ov),
        ms_by_num,
    )
    assert broad_out["c_broad"].final_tests == broad_cands
    assert "max_nodes" not in marker.read_text(encoding="utf-8").splitlines()

    # 2. Ordinary tests pass -> #1490 runs and sheds cleanly
    marker.unlink()
    pass_cands = frozenset({0, 1490})
    pass_plan = verifier.plan_commit_verification(
        pass_cands, ms_by_num, file_by_num, analysis=an, id_by_num=id_by_num
    )
    assert pass_plan.should_verify is True
    assert pass_plan.deferred_heavy == frozenset({1490})
    pass_out: dict = {}
    verifier._run_fork_queue(
        [("c_pass", an, pass_cands, pass_plan)],
        pass_out,
        repo,
        id_by_num,
        file_to_tests,
        (),
        10.0,
        10**9,
        1,
        str(tmp_ov),
        ms_by_num,
    )
    assert pass_out["c_pass"].final_tests == frozenset()
    assert marker.read_text(encoding="utf-8").splitlines() == ["ord_pass", "max_nodes"]

    # 3. Timeout during deferred heavy test still preserves checkpointed passed ordinary tests
    marker.unlink()
    an_sleep = CommitAnalysis("c_sleep", patched_contents={"tests/sleep_heavy.flag": "1\n"})
    sleep_out: dict = {}
    verifier._run_fork_queue(
        [("c_sleep", an_sleep, pass_cands, pass_plan)],
        sleep_out,
        repo,
        id_by_num,
        file_to_tests,
        (),
        1.5,
        10**9,
        1,
        str(tmp_ov),
        ms_by_num,
    )
    assert sleep_out["c_sleep"].timed_out is True
    assert sleep_out["c_sleep"].final_tests == frozenset({1490})
    assert list(tmp_ov.iterdir()) == []


def test_partial_heavy_outlier_subset_plan_keeps_cheap_tests_conservatively(
    tmp_path: Path, monkeypatch
) -> None:
    import verifier

    # 600 cheap tests (200ms each, different files) would blow PER_COMMIT_EST_CPU_CAP_MS / ROI
    # under full verification, so plan_commit_verification selects the Heavy-Outlier Subset Plan
    # (fast #1106 + heavy #1224) and keeps all 600 cheap tests unverified (zero misses).
    ms_by_num: dict[int, int] = {1106: 21_454, 1224: 6_686}
    id_by_num: dict[int, str] = {
        1106: "tests/test_executor.py::TestExecutor::test_execute_tpcds",
        1224: "tests/test_generator.py::TestGenerator::test_generator_memory",
    }
    file_by_num: dict[int, str] = {
        1106: "tests/test_executor.py",
        1224: "tests/test_generator.py",
    }
    for i in range(600):
        ms_by_num[i] = 10
        tfile = f"tests/test_cheap_{i}.py"
        id_by_num[i] = f"{tfile}::TestCheap::test_{i}"
        file_by_num[i] = tfile

    cands = frozenset(ms_by_num)
    an = CommitAnalysis("c_giant", patched_contents={})
    plan = verifier.plan_commit_verification(
        cands,
        ms_by_num,
        file_by_num,
        analysis=an,
        id_by_num=id_by_num,
        tpcds_plan_hashes=("dummy_hash",),
        tpcds_exec_lines={"sqlglot/other.py": frozenset({1})},
    )
    assert plan.should_verify is True
    assert plan.fast_tpcds is True
    assert {1106, 1224}.issubset(plan.to_verify)
    assert plan.keep_unverified == frozenset(range(600))
    assert plan.to_verify | plan.keep_unverified == cands

    # Execute the partial subset plan: #1106 and #1224 pass and shed; all 600 cheap tests remain kept
    repo = tmp_path / "subset_repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "test_generator.py").write_text(
        textwrap.dedent(
            """\
            import unittest

            class TestGenerator(unittest.TestCase):
                def test_generator_memory(self):
                    self.assertTrue(True)
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(verifier, "_fast_verify_tpcds", lambda hashes, deadline: True)
    tmp_ov = tmp_path / "sub_ov"
    tmp_ov.mkdir()
    out: dict = {}
    verifier._run_fork_queue(
        [("c_giant", an, cands, plan)],
        out,
        repo,
        id_by_num,
        {"tests/test_generator.py": frozenset({1224})},
        ("dummy_hash",),
        10.0,
        10**9,
        1,
        str(tmp_ov),
        ms_by_num,
    )
    assert out["c_giant"].final_tests == frozenset(range(600))
    assert out["c_giant"].shed_count == 2
    assert list(tmp_ov.iterdir()) == []
