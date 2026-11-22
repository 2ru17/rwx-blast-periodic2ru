"""Safety tests for verifier dynamic module tracking, sibling .pyc symlinks, and lazy heavy deps."""

from __future__ import annotations

import compileall
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diff_analyzer import CommitAnalysis, analyze_patch  # noqa: E402
from verifier import create_overlay, verify_batch  # noqa: E402


def test_create_overlay_symlinks_only_unpatched_sibling_pyc(tmp_path: Path) -> None:
    repo = tmp_path / "pyc_repo"
    pkg = repo / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "patched.py").write_text("X = 1\n", encoding="utf-8")
    (pkg / "sibling.py").write_text("Y = 2\n", encoding="utf-8")
    assert compileall.compile_dir(pkg, quiet=1)

    tmp_ov = tmp_path / "ov"
    tmp_ov.mkdir()
    ov_dir = create_overlay(repo, {"pkg/patched.py": "X = 99\n"}, tmp_base=str(tmp_ov))
    try:
        cache = Path(ov_dir) / "pkg" / "__pycache__"
        assert cache.is_dir() and not cache.is_symlink()
        names = [p.name for p in cache.iterdir()]
        assert any(n.startswith("sibling.") and n.endswith(".pyc") for n in names)
        assert any(n.startswith("__init__.") and n.endswith(".pyc") for n in names)
        assert not any(n.startswith("patched.") for n in names)
        for p in cache.iterdir():
            assert p.is_symlink()
    finally:
        shutil.rmtree(ov_dir, ignore_errors=True)


def test_dynamic_loaded_module_bounce_to_clean_tier1(tmp_path: Path) -> None:
    """If importing sqlglot eagerly loads an unlisted submodule, verifier bounces that commit to Tier-1."""
    repo = tmp_path / "eager_repo"
    (repo / "sqlglot" / "extra").mkdir(parents=True)
    (repo / "tests").mkdir(parents=True)
    (repo / "sqlglot" / "__init__.py").write_text("from sqlglot.extra import hidden\n", encoding="utf-8")
    (repo / "sqlglot" / "extra" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "sqlglot" / "extra" / "hidden.py").write_text("VAL = 1\n", encoding="utf-8")
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "test_hidden.py").write_text(
        textwrap.dedent(
            """\
            import unittest
            from sqlglot.extra import hidden

            class TestHidden(unittest.TestCase):
                def test_val(self):
                    self.assertEqual(hidden.VAL, 1)
            """
        ),
        encoding="utf-8",
    )

    # sqlglot/extra/hidden.py is NOT in static PRELOADED_SQLGLOT_FILES, so static Tier-2 accepts it,
    # but runtime _loaded_python_files detects it and bounces it cleanly to Tier-1!
    patch = (
        "diff --git a/sqlglot/extra/hidden.py b/sqlglot/extra/hidden.py\n"
        "--- a/sqlglot/extra/hidden.py\n"
        "+++ b/sqlglot/extra/hidden.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-VAL = 1\n"
        "+VAL = 2\n"
    )
    an_fail = analyze_patch("c_hidden_fail", patch, repo)
    an_pass = CommitAnalysis("c_hidden_pass", patched_contents={})
    tmp_ov = tmp_path / "ov"
    tmp_ov.mkdir()

    outcomes = verify_batch(
        analyses={"c_hidden_fail": an_fail, "c_hidden_pass": an_pass},
        candidate_sets={"c_hidden_fail": frozenset({0}), "c_hidden_pass": frozenset({0})},
        sqlglot_root=repo,
        ms_by_num={0: 10},
        id_by_num={0: "tests/test_hidden.py::TestHidden::test_val"},
        file_by_num={0: "tests/test_hidden.py"},
        file_to_tests={"tests/test_hidden.py": frozenset({0})},
        tmp_base=str(tmp_ov),
        force_verify_all=True,
    )
    assert outcomes["c_hidden_fail"].final_tests == frozenset({0})
    assert outcomes["c_hidden_pass"].final_tests == frozenset()
    assert "sqlglot" not in sys.modules
    assert list(tmp_ov.iterdir()) == []


def test_lazy_pandas_testing_from_import_does_not_activate_numpy_or_pandas() -> None:
    probe = (
        f"import sys; sys.path.insert(0, {str(SRC_DIR)!r}); "
        "import verifier; verifier._install_lazy_heavy_deps(); "
        "import numpy as np; import pandas as pd; from pandas.testing import assert_frame_equal; "
        "assert '__version__' not in sys.modules['numpy'].__dict__; "
        "assert '__version__' not in sys.modules['pandas'].__dict__; "
        "assert callable(assert_frame_equal)"
    )
    res = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert res.returncode == 0, f"from pandas.testing import activated numpy/pandas: {res.stderr}"


def test_conditional_test_dialect_preload_and_heavy_submaster_isolation(tmp_path: Path) -> None:
    """Verify <2 dialect commits skips test_dialect preload and >=2 heavy commits share numpy+pandas safely."""
    repo = tmp_path / "t2_cond_repo"
    (repo / "sqlglot").mkdir(parents=True)
    (repo / "tests" / "dialects").mkdir(parents=True)
    td_marker = tmp_path / "td_imported.log"
    (repo / "sqlglot" / "__init__.py").write_text("CORE = 1\n", encoding="utf-8")
    (repo / "sqlglot" / "leaf.py").write_text("VAL = 1\n", encoding="utf-8")
    (repo / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "dialects" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "tests" / "dialects" / "test_dialect.py").write_text(
        f"from pathlib import Path\nPath({str(td_marker)!r}).write_text('imported', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_non_dialect.py").write_text(
        textwrap.dedent(
            """\
            import unittest
            from sqlglot import leaf

            class TestNonDialect(unittest.TestCase):
                def test_ok(self):
                    self.assertEqual(leaf.VAL, 1)
            """
        ),
        encoding="utf-8",
    )
    (repo / "tests" / "test_optimizer.py").write_text(
        textwrap.dedent(
            """\
            import sys
            import unittest
            from sqlglot import leaf

            class TestOptimizer(unittest.TestCase):
                def test_optimize(self):
                    assert "numpy" in sys.modules and "__version__" in sys.modules["numpy"].__dict__
                    assert "pandas" in sys.modules and "__version__" in sys.modules["pandas"].__dict__
                    self.assertEqual(leaf.VAL, 1)
            """
        ),
        encoding="utf-8",
    )

    tmp_ov = tmp_path / "ov"
    tmp_ov.mkdir()

    # 1. Two non-dialect commits in Tier-2 MUST NOT preload tests.dialects.test_dialect
    an1 = CommitAnalysis("c1", patched_contents={"sqlglot/leaf.py": "VAL = 1\n"})
    an2 = CommitAnalysis("c2", patched_contents={"sqlglot/leaf.py": "VAL = 2\n"})
    out = verify_batch(
        analyses={"c1": an1, "c2": an2},
        candidate_sets={"c1": frozenset({0}), "c2": frozenset({0})},
        sqlglot_root=repo,
        ms_by_num={0: 10},
        id_by_num={0: "tests/test_non_dialect.py::TestNonDialect::test_ok"},
        file_by_num={0: "tests/test_non_dialect.py"},
        file_to_tests={"tests/test_non_dialect.py": frozenset({0})},
        tmp_base=str(tmp_ov),
        force_verify_all=True,
    )
    assert out["c1"].final_tests == frozenset()
    assert out["c2"].final_tests == frozenset({0})
    assert not td_marker.exists()

    # 2. Two heavy optimizer commits share preloaded numpy+pandas while staying isolated on leaf.VAL
    out_h = verify_batch(
        analyses={"h1": an1, "h2": an2},
        candidate_sets={"h1": frozenset({1}), "h2": frozenset({1})},
        sqlglot_root=repo,
        ms_by_num={1: 100},
        id_by_num={1: "tests/test_optimizer.py::TestOptimizer::test_optimize"},
        file_by_num={1: "tests/test_optimizer.py"},
        file_to_tests={"tests/test_optimizer.py": frozenset({1})},
        tmp_base=str(tmp_ov),
        force_verify_all=True,
    )
    assert out_h["h1"].final_tests == frozenset()
    assert out_h["h2"].final_tests == frozenset({1})
    assert "numpy" not in sys.modules or "sqlglot" not in sys.modules
    assert list(tmp_ov.iterdir()) == []

