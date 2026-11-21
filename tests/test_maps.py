"""Verification tests for offline sqlglot@eb11aec maps."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAPS_DIR = REPO_ROOT / "maps"
REQUIRED_FILES = (
    "test_index.json",
    "coverage_map.json",
    "ast_index.json",
    "import_graph.json",
)


def _decode_set(encoded: list[int | list[int]]) -> set[int]:
    out: set[int] = set()
    for item in encoded:
        if isinstance(item, list):
            out.update(range(item[0], item[1] + 1))
        else:
            out.add(item)
    return out


def _load_maps() -> tuple[dict, dict, dict, dict]:
    return tuple(
        json.loads((MAPS_DIR / name).read_text(encoding="utf-8"))
        for name in REQUIRED_FILES
    )


def test_maps_exist_and_within_size_budget() -> None:
    total_bytes = 0
    for name in REQUIRED_FILES:
        path = MAPS_DIR / name
        assert path.is_file(), f"Missing map file: {path}"
        total_bytes += path.stat().st_size
    assert total_bytes < 10 * 1024 * 1024, f"Maps exceed 10 MB: {total_bytes} bytes"


def test_index_matches_pinned_sqlglot() -> None:
    test_index, _, _, _ = _load_maps()
    assert test_index["sqlglot_commit"] == "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"
    tests = test_index["tests"]
    assert len(tests) == 1620
    assert [t["number"] for t in tests] == list(range(1620))


def test_dynamic_contexts_preserve_shared_core_attribution() -> None:
    test_index, cov_map, _, _ = _load_maps()
    id_to_num = {t["id"]: t["number"] for t in test_index["tests"]}
    sets = cov_map["sets"]
    line_to_set = cov_map["line_to_set"]

    serde_line = _decode_set(sets[line_to_set["sqlglot/serde.py"]["53"]])
    assert len(serde_line) >= 35
    assert id_to_num["tests/test_serde.py::TestSerde::test_serde"] in serde_line
    assert id_to_num["tests/test_optimizer.py::TestOptimizer::test_optimize"] in serde_line
    assert id_to_num["tests/test_executor.py::TestExecutor::test_execute_tpcds"] in serde_line

    elim_line = _decode_set(sets[line_to_set["sqlglot/optimizer/eliminate_joins.py"]["71"]])
    assert len(elim_line) >= 8
    assert id_to_num["tests/test_optimizer.py::TestOptimizer::test_eliminate_joins"] in elim_line
    assert id_to_num["tests/test_optimizer.py::TestOptimizer::test_tpcds"] in elim_line
    assert id_to_num["tests/test_executor.py::TestExecutor::test_execute_tpcds"] in elim_line

    for core_file in ("sqlglot/parser.py", "sqlglot/generator.py", "sqlglot/dialects/dialect.py"):
        max_fanout = max(
            len(_decode_set(sets[sid])) for sid in line_to_set[core_file].values()
        )
        assert max_fanout > 500, f"Unexpectedly low fanout in {core_file}: {max_fanout}"


def test_fixture_mapping_completeness() -> None:
    test_index, _, _, _ = _load_maps()
    id_to_num = {t["id"]: t["number"] for t in test_index["tests"]}
    fixtures = test_index["fixture_to_test_numbers"]

    expected_pairs = [
        (
            "tests/fixtures/optimizer/qualify_columns_ddl.sql",
            "tests/test_optimizer.py::TestOptimizer::test_qualify_columns",
        ),
        (
            "tests/fixtures/optimizer/pushdown_projections.sql",
            "tests/test_optimizer.py::TestOptimizer::test_pushdown_projection",
        ),
        (
            "tests/fixtures/optimizer/pushdown_predicates.sql",
            "tests/test_optimizer.py::TestOptimizer::test_pushdown_predicates",
        ),
        (
            "tests/fixtures/optimizer/annotate_types.sql",
            "tests/test_optimizer.py::TestOptimizer::test_annotate_types",
        ),
        (
            "tests/fixtures/optimizer/tpc-ds/tpc-ds.sql",
            "tests/test_executor.py::TestExecutor::test_execute_tpcds",
        ),
        (
            "tests/fixtures/optimizer/tpc-h/tpc-h.sql",
            "tests/test_executor.py::TestExecutor::test_execute_tpch",
        ),
    ]
    for fixture_path, test_id in expected_pairs:
        assert id_to_num[test_id] in fixtures[fixture_path]


def test_class_attribute_and_lambda_mapping() -> None:
    test_index, cov_map, ast_index, _ = _load_maps()
    id_to_num = {t["id"]: t["number"] for t in test_index["tests"]}
    sets = cov_map["sets"]

    tsql_week = _decode_set(
        sets[ast_index["sqlglot/dialects/tsql.py"]["class_attrs"]["TSQL"]["WEEK_OFFSET"]["set_id"]]
    )
    assert id_to_num["tests/test_optimizer.py::TestOptimizer::test_simplify"] in tsql_week

    dremio_regexp = _decode_set(
        sets[cov_map["line_to_set"]["sqlglot/parsers/dremio.py"]["111"]]
    )
    assert id_to_num["tests/dialects/test_dremio.py::TestDremio::test_regexp_split"] in dremio_regexp
    dremio_dk = _decode_set(
        sets[
            ast_index["sqlglot/parsers/dremio.py"]["class_attrs"]["DremioParser"]["FUNCTIONS"][
                "dict_keys"
            ]["REGEXP_SPLIT"][2]
        ]
    )
    assert id_to_num["tests/dialects/test_dremio.py::TestDremio::test_regexp_split"] in dremio_dk
