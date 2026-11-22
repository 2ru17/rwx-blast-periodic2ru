"""Tests for Phase 1 runtime caches, word index, prefilters, and comment fast-path."""

from __future__ import annotations

import ast
import os
import re
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
MAPS_DIR = REPO_ROOT / "maps"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from diff_analyzer import (  # noqa: E402
    _bounded_start,
    _comment_only_change,
    _iter_repo_trees_containing,
    analyze_patch,
    build_repo_word_index,
    write_repo_word_index,
)
from selector import (  # noqa: E402
    BlastSelector,
    CoverageSets,
    decode_static_maps,
    load_static_maps,
    write_maps_cache,
)


def test_maps_cache_roundtrip_lazy_sets_and_stamp_invalidation(tmp_path: Path) -> None:
    cache_path = tmp_path / "maps_cache.marshal"
    write_maps_cache(MAPS_DIR, cache_path)
    assert cache_path.is_file()

    eager = decode_static_maps(MAPS_DIR)
    cached = load_static_maps(MAPS_DIR, cache_path)
    assert set(eager.keys()) == set(cached.keys())
    for k in eager:
        assert eager[k] == cached[k], f"Cache mismatch on key {k}"

    sel_json = BlastSelector(MAPS_DIR)
    sel_cache = BlastSelector(MAPS_DIR, cache_path=cache_path)
    assert isinstance(sel_cache.sets, CoverageSets)
    assert len(sel_cache.sets) == len(sel_json.sets)
    for sid in (0, 1, len(sel_json.sets) // 2, len(sel_json.sets) - 1):
        assert sel_cache.sets[sid] == sel_json.sets[sid]

    # Corrupt cache falls back cleanly to JSON
    corrupt = tmp_path / "corrupt.marshal"
    corrupt.write_bytes(b"not-valid-marshal")
    fallback = load_static_maps(MAPS_DIR, corrupt)
    assert fallback["sets"] == eager["sets"]

    # Stamp invalidation when a map file changes size
    alt_maps = tmp_path / "alt_maps"
    alt_maps.mkdir()
    for name in ("test_index.json", "coverage_map.json", "ast_index.json", "import_graph.json"):
        (alt_maps / name).write_bytes((MAPS_DIR / name).read_bytes())
    alt_cache = tmp_path / "alt_cache.marshal"
    write_maps_cache(alt_maps, alt_cache)
    (alt_maps / "import_graph.json").write_text(
        (alt_maps / "import_graph.json").read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    reloaded = load_static_maps(alt_maps, alt_cache)
    assert reloaded["file_to_set"] == eager["file_to_set"]


def test_repo_word_index_and_bounded_start_match_regex_scan(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "mini_repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "a.py").write_text("class Alpha:\n    x = 1\n", encoding="utf-8")
    (repo / "pkg" / "b.py").write_text("class Beta(Alpha):\n    pass\n", encoding="utf-8")
    (repo / "pkg" / "c.py").write_text("AlphaExtra = 2\n", encoding="utf-8")

    idx_path = tmp_path / "word_index.marshal"
    write_repo_word_index(repo, idx_path)
    raw_idx = build_repo_word_index(repo)
    assert "Alpha" in raw_idx["words"]
    assert "AlphaExtra" in raw_idx["words"]

    monkeypatch.delenv("REPO_WORD_INDEX", raising=False)
    import diff_analyzer

    diff_analyzer._REPO_WORD_INDEX.clear()
    no_idx = [rel for rel, _ in _iter_repo_trees_containing(("Alpha",), repo, {})]

    monkeypatch.setenv("REPO_WORD_INDEX", str(idx_path))
    diff_analyzer._REPO_WORD_INDEX.clear()
    with_idx = [rel for rel, _ in _iter_repo_trees_containing(("Alpha",), repo, {})]
    assert with_idx == no_idx == ["pkg/a.py", "pkg/b.py"]

    # _bounded_start must match identical spans as \b<word>
    pat_orig = re.compile(r"\bAlpha\b")
    pat_fast = re.compile(_bounded_start("Alpha") + r"\b")
    for text in ("Alpha", "xAlpha", "Alpha_1", "foo(Alpha)", "1Alpha", " Alpha "):
        assert bool(pat_orig.search(text)) == bool(pat_fast.search(text))


def test_comment_only_fast_path_soundness(tmp_path: Path) -> None:
    base = textwrap.dedent(
        '''\
        x = """
        inside string
        """
        y = 1 + \\
            2
        z = 3
        '''
    )
    tree = ast.parse(base)

    # Pure comment addition outside strings/continuations is accepted
    pure = base + "# trailing note\n"
    assert _comment_only_change(base, tree, pure) is True

    # Comment-looking line inside multi-line string MUST be rejected
    in_str = base.replace("inside string", "inside string\n# not a comment")
    assert _comment_only_change(base, tree, in_str) is False

    # Comment line after backslash continuation MUST be rejected
    after_bs = base.replace("y = 1 + \\\n    2", "y = 1 + \\\n# bad\n    2")
    assert _comment_only_change(base, tree, after_bs) is False

    # Encoding cookie edit on line 1-2 MUST be rejected
    enc = "# coding: latin-1\n" + base
    assert _comment_only_change(base, tree, enc) is False

    # Analyze end-to-end ensures AST noop and raw_text_changed preserved
    repo = tmp_path / "c_repo"
    repo.mkdir()
    (repo / "m.py").write_text("x = 1\n", encoding="utf-8")
    patch = (
        "diff --git a/m.py b/m.py\n"
        "--- a/m.py\n"
        "+++ b/m.py\n"
        "@@ -1,1 +1,2 @@\n"
        " x = 1\n"
        "+# comment\n"
    )
    an = analyze_patch("c1", patch, repo)
    assert an.files[0].is_ast_noop is True
    assert an.files[0].raw_text_changed is True
    assert os.environ.get("REPO_WORD_INDEX") != "__invalid__"
