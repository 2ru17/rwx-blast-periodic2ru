"""Soundness and precision regressions for BlastSelector and diff_analyzer."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
MAPS_DIR = REPO_ROOT / "maps"
SQLGLOT_ROOT = REPO_ROOT.parent / "ai-notes" / "sqlglot"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diff_analyzer import analyze_patch  # noqa: E402
from selector import BlastSelector  # noqa: E402


def test_dict_keys_narrowing_and_expression_arg_types_precision() -> None:
    if not SQLGLOT_ROOT.is_dir():
        return
    sel = BlastSelector(MAPS_DIR)

    # 1. Snowflake NUMERIC_ESCAPES single-key value change narrows to dict_keys['u'], not whole class attr
    p_4d63 = (
        "diff --git a/sqlglot/dialects/snowflake.py b/sqlglot/dialects/snowflake.py\n"
        "--- a/sqlglot/dialects/snowflake.py\n"
        "+++ b/sqlglot/dialects/snowflake.py\n"
        "@@ -132,3 +132,3 @@ class Snowflake(Dialect):\n"
        "         STRING_ESCAPES = [\"\\\\\", \"'\"]\n"
        '-        NUMERIC_ESCAPES = {"x": (16, 2, 2, 0xFF), "u": (16, 4, 4, 0xFFFF), "0": (8, 1, 3, 0xFF)}\n'
        '+        NUMERIC_ESCAPES = {"x": (16, 2, 2, 0xFF), "u": (17, 4, 4, 0xFFFF), "0": (8, 1, 3, 0xFF)}\n'
        "         DROP_UNKNOWN_ESCAPES = True\n"
    )
    res_4d63 = sel.select(analyze_patch("p_4d63", p_4d63, SQLGLOT_ROOT))
    assert sel.num_by_id["tests/dialects/test_snowflake.py::TestSnowflake::test_snowflake"] in res_4d63.candidates
    assert len(res_4d63.candidates) <= 40, f"4d63 candidate explosion: {len(res_4d63.candidates)}"
    assert 1490 not in res_4d63.candidates

    # 2. TriggerProperties.arg_types stays tight to TriggerProperties tests without pulling #1490
    p_ktz3 = (
        "diff --git a/sqlglot/expressions/ddl.py b/sqlglot/expressions/ddl.py\n"
        "--- a/sqlglot/expressions/ddl.py\n"
        "+++ b/sqlglot/expressions/ddl.py\n"
        "@@ -80,3 +80,3 @@ class TriggerProperties(Expression):\n"
        '         "execute": True,\n'
        '-        "constraint": False,\n'
        '+        "constraint": True,\n'
        '         "referenced_table": False,\n'
    )
    res_ktz3 = sel.select(analyze_patch("p_ktz3", p_ktz3, SQLGLOT_ROOT))
    assert sel.num_by_id["tests/dialects/test_postgres.py::TestPostgres::test_ddl"] in res_ktz3.candidates
    assert len(res_ktz3.candidates) <= 20, f"ktz3 candidate explosion: {len(res_ktz3.candidates)}"
    assert 1490 not in res_ktz3.candidates

    # 3. DremioParser.FUNCTIONS['REGEXP_SPLIT'] lambda edit narrows cleanly without unrelated dialect_imports
    p_6ze8 = (
        "diff --git a/sqlglot/parsers/dremio.py b/sqlglot/parsers/dremio.py\n"
        "--- a/sqlglot/parsers/dremio.py\n"
        "+++ b/sqlglot/parsers/dremio.py\n"
        "@@ -110,3 +110,3 @@ class DremioParser(parser.Parser):\n"
        '         "REGEXP_SPLIT": lambda args: exp.RegexpSplit(\n'
        "-            this=seq_get(args, 0),\n"
        "+            this=seq_get(args, 1),\n"
        "             expression=seq_get(args, 1),\n"
    )
    res_6ze8 = sel.select(analyze_patch("p_6ze8", p_6ze8, SQLGLOT_ROOT))
    assert sel.num_by_id["tests/dialects/test_dremio.py::TestDremio::test_regexp_split"] in res_6ze8.candidates
    assert len(res_6ze8.candidates) <= 10, f"6ze8 candidate explosion: {len(res_6ze8.candidates)}"
    assert not (res_6ze8.candidates & sel.dialect_import_tests)


def test_dialect_import_tests_narrowed_and_sound() -> None:
    if not SQLGLOT_ROOT.is_dir():
        return
    sel = BlastSelector(MAPS_DIR)

    # Pure runtime method edit in unrelated file does NOT pull test_dialect_imports.py
    ck_lines = (SQLGLOT_ROOT / "sqlglot" / "generators" / "clickhouse.py").read_text(encoding="utf-8").splitlines()
    idx_ck = next(i for i, ln in enumerate(ck_lines, start=1) if "values_as_table = any(isinstance(value, exp.Tuple)" in ln)
    p_ck = (
        "diff --git a/sqlglot/generators/clickhouse.py b/sqlglot/generators/clickhouse.py\n"
        "--- a/sqlglot/generators/clickhouse.py\n"
        "+++ b/sqlglot/generators/clickhouse.py\n"
        f"@@ -{idx_ck},1 +{idx_ck},1 @@\n"
        f"-{ck_lines[idx_ck - 1]}\n"
        f"+{ck_lines[idx_ck - 1].replace('any(', 'all(')}\n"
    )
    an_ck = analyze_patch("p_ck", p_ck, SQLGLOT_ROOT)
    assert an_ck.files[0].syntax_broken is False
    res_ck = sel.select(an_ck)
    assert not (res_ck.candidates & sel.dialect_import_tests)

    # Edit inside sqlglot/dialects/__init__.py MUST pull test_dialect_imports.py
    init_lines = (SQLGLOT_ROOT / "sqlglot" / "dialects" / "__init__.py").read_text(encoding="utf-8").splitlines()
    p_init = (
        "diff --git a/sqlglot/dialects/__init__.py b/sqlglot/dialects/__init__.py\n"
        "--- a/sqlglot/dialects/__init__.py\n"
        "+++ b/sqlglot/dialects/__init__.py\n"
        "@@ -1,1 +1,2 @@\n"
        f" {init_lines[0]}\n"
        "+_EXTRA = 1\n"
    )
    res_init = sel.select(analyze_patch("p_init", p_init, SQLGLOT_ROOT))
    assert sel.dialect_import_tests.issubset(res_init.candidates)


def test_statement_header_expansions_and_encoding_bytes(tmp_path: Path) -> None:
    repo = tmp_path / "hdr_repo"
    (repo / "pkg").mkdir(parents=True)
    src = textwrap.dedent(
        """\
        def demo(x):
            with (
                open(x)
            ) as fh:
                try:
                    return fh.read()
                except (
                    OSError,
                    ValueError,
                ):
                    return ""
        """
    )
    (repo / "pkg" / "m.py").write_text(src, encoding="utf-8")
    p_except = (
        "diff --git a/pkg/m.py b/pkg/m.py\n"
        "--- a/pkg/m.py\n"
        "+++ b/pkg/m.py\n"
        "@@ -8,2 +8,2 @@ def demo(x):\n"
        "             OSError,\n"
        "-            ValueError,\n"
        "+            KeyError,\n"
    )
    an = analyze_patch("hdr_except", p_except, repo)
    assert {7, 8, 9, 10}.issubset(an.files[0].expanded_old_lines)

    # Non-UTF8 base file with replacement character must fail closed as syntax_broken
    (repo / "pkg" / "bad_enc.py").write_bytes(b"x = '\xff'\n")
    p_enc = (
        "diff --git a/pkg/bad_enc.py b/pkg/bad_enc.py\n"
        "--- a/pkg/bad_enc.py\n"
        "+++ b/pkg/bad_enc.py\n"
        "@@ -1,1 +1,2 @@\n"
        " x = '\ufffd'\n"
        "+y = 1\n"
    )
    an_enc = analyze_patch("bad_enc", p_enc, repo)
    assert an_enc.files[0].syntax_broken is True
