"""Coverage- and AST-guided candidate test selector."""

from __future__ import annotations

import json
import marshal
import os
from array import array
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from diff_analyzer import CommitAnalysis, FileAnalysis

_NON_EXEC_PREFIXES = (
    "README",
    "CHANGELOG",
    "CONTRIBUTING",
    "LICENSE",
    "posts/",
    "pdoc/",
    "benchmarks/",
    ".github/",
    ".gitignore",
    ".pre-commit",
    "Makefile",
)
_NON_EXEC_SUFFIXES = (".md", ".png", ".svg", ".jpg", ".jpeg", ".gif", ".txt", ".rst")
_BUILD_CONFIG_FILES = frozenset({"pyproject.toml", "setup.py", "setup.cfg", "MANIFEST.in"})
_CORE_BASE_FILES = (
    "sqlglot/parser.py",
    "sqlglot/generator.py",
    "sqlglot/dialects/dialect.py",
    "sqlglot/tokens.py",
    "sqlglot/tokenizer_core.py",
)
_LIFECYCLE_METHODS = frozenset({"setUp", "tearDown", "setUpClass", "tearDownClass"})
_MAP_FILES = ("test_index.json", "coverage_map.json", "ast_index.json", "import_graph.json")


def _decode_set(encoded: list[int | list[int]]) -> frozenset[int]:
    out: list[int] = []
    for item in encoded:
        if isinstance(item, list):
            out.extend(range(item[0], item[1] + 1))
        else:
            out.append(item)
    return frozenset(out)


def _pack_set(encoded: list[int | list[int]]) -> bytes:
    """Pack a JSON-encoded test set as native uint32 (start, end) pairs, one pair per item."""
    pairs = array("I")
    for item in encoded:
        if isinstance(item, list):
            pairs.extend((item[0], item[1]))
        else:
            pairs.extend((item, item))
    return pairs.tobytes()


def _unpack_set(packed: bytes) -> frozenset[int]:
    pairs = array("I")
    pairs.frombytes(packed)
    out: list[int] = []
    for i in range(0, len(pairs), 2):
        out.extend(range(pairs[i], pairs[i + 1] + 1))
    return frozenset(out)


class CoverageSets:
    """The coverage map's test sets, each decoded the first time it is indexed.

    A run reads a few hundred of the ~10k sets, so decoding all of them up front costs more than
    the whole selection.
    """

    __slots__ = ("_packed", "_decoded")

    def __init__(self, packed: tuple[bytes, ...]) -> None:
        self._packed = packed
        self._decoded: list[frozenset[int] | None] = [None] * len(packed)

    def __len__(self) -> int:
        return len(self._packed)

    def __getitem__(self, sid: int) -> frozenset[int]:
        found = self._decoded[sid]
        if found is None:
            found = self._decoded[sid] = _unpack_set(self._packed[sid])
        return found


def _dialect_family_from_path(rel_path: str) -> str | None:
    parts = rel_path.split("/")
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


@dataclass(frozen=True, slots=True)
class SelectionResult:
    candidates: frozenset[int]
    is_provable_noop: bool
    has_syntax_or_import_break: bool


def decode_static_maps(maps_dir: Path) -> dict[str, Any]:
    """Decode the four plain JSON maps into every lookup structure that does not depend on /data."""
    test_index = json.loads((maps_dir / "test_index.json").read_text(encoding="utf-8"))
    cov_map = json.loads((maps_dir / "coverage_map.json").read_text(encoding="utf-8"))
    ast_index = json.loads((maps_dir / "ast_index.json").read_text(encoding="utf-8"))
    import_graph = json.loads((maps_dir / "import_graph.json").read_text(encoding="utf-8"))

    encoded_sets = cov_map["sets"]
    file_to_tests = {f: frozenset(nums) for f, nums in test_index["file_to_test_numbers"].items()}
    source_inspecting_tests = frozenset(test_index.get("source_inspecting_tests", ()))
    source_file_to_inspecting_tests = {
        rel: frozenset(nums)
        for rel, nums in test_index.get("source_file_to_inspecting_tests", {}).items()
    }
    mapped_inspecting: set[int] = set()
    for nums in source_file_to_inspecting_tests.values():
        mapped_inspecting.update(nums)
    tpcds_exec_lines = {
        rel: _decode_set(enc) for rel, enc in test_index.get("tpcds_exec_lines", {}).items()
    }
    tpcds_exec_qualnames: dict[str, frozenset[str]] = {}
    for rel, exec_lines in tpcds_exec_lines.items():
        f_ast = ast_index.get(rel, {})
        tpcds_exec_qualnames[rel] = frozenset(
            qn
            for qn, kind, start, end in f_ast.get("symbols", ())
            if kind in ("function", "method") and any(start <= ln <= end for ln in exec_lines)
        )

    qualname_to_set: dict[str, dict[str, int]] = cov_map["qualname_to_set"]
    # Base-class method coverage by leaf method name, for subclass override propagation.
    base_method_sids: dict[str, list[int]] = {}
    for rel in _CORE_BASE_FILES:
        for qn, sid in qualname_to_set.get(rel, {}).items():
            if "." in qn:
                meth = qn.rsplit(".", 1)[1]
                base_method_sids.setdefault(meth, []).append(sid)
    # Class/module attributes across core base files (non-empty sets only), for newly added subclass attributes.
    base_attr_sids: dict[str, list[int]] = {}
    for rel in _CORE_BASE_FILES:
        f_ast = ast_index.get(rel, {})
        for attrs in f_ast.get("class_attrs", {}).values():
            for attr_name, info in attrs.items():
                if _decode_set(encoded_sets[info["set_id"]]):
                    base_attr_sids.setdefault(attr_name, []).append(info["set_id"])

    return {
        "sets": tuple(_pack_set(s) for s in encoded_sets),
        "line_to_set": {
            rel: {int(k): v for k, v in ln_dict.items()}
            for rel, ln_dict in cov_map["line_to_set"].items()
        },
        "import_time_lines": {
            rel: frozenset(lns) for rel, lns in cov_map.get("import_time_lines", {}).items()
        },
        "import_time_qualnames": {
            rel: frozenset(qns) for rel, qns in cov_map.get("import_time_qualnames", {}).items()
        },
        "qualname_to_set": qualname_to_set,
        "file_to_set": cov_map["file_to_set"],
        "ast_index": ast_index,
        "modules": import_graph["modules"],
        "dialect_family_sids": dict(import_graph["dialect_families"]),
        "dialect_import_closure": frozenset(import_graph.get("dialect_import_closure", ())),
        "file_to_tests": file_to_tests,
        "fixture_to_tests": {
            f: frozenset(nums) for f, nums in test_index["fixture_to_test_numbers"].items()
        },
        "source_inspecting_tests": source_inspecting_tests,
        "source_file_to_inspecting_tests": source_file_to_inspecting_tests,
        "unmapped_source_inspecting_tests": source_inspecting_tests - mapped_inspecting,
        "tpcds_plan_hashes": tuple(test_index.get("tpcds_plan_hashes", ())),
        "tpcds_exec_lines": tpcds_exec_lines,
        "tpcds_exec_qualnames": tpcds_exec_qualnames,
        "base_method_sids": base_method_sids,
        "base_attr_sids": base_attr_sids,
        "tests": test_index["tests"],
    }


def _maps_stamp(maps_dir: Path) -> tuple[str, tuple[tuple[str, int, int], ...]]:
    # Whole seconds: image layers keep mtimes only to the second, so nanoseconds differ after export.
    stats = []
    for name in _MAP_FILES:
        st = os.stat(maps_dir / name)
        stats.append((name, st.st_size, int(st.st_mtime)))
    return str(maps_dir.resolve()), tuple(stats)


def write_maps_cache(maps_dir: Path, out_path: Path) -> None:
    """Build-time only: marshal decode_static_maps() with a stamp of the JSON files it was decoded from."""
    out_path.write_bytes(marshal.dumps({"stamp": _maps_stamp(maps_dir), "maps": decode_static_maps(maps_dir)}))


def load_static_maps(maps_dir: Path, cache_path: Path | None = None) -> dict[str, Any]:
    """Return the build-time cache when it was made from these exact JSON files, else decode the JSON."""
    if cache_path is not None:
        try:
            payload = marshal.loads(cache_path.read_bytes())
            if payload["stamp"] == _maps_stamp(maps_dir):
                return payload["maps"]
        except (OSError, EOFError, ValueError, TypeError, KeyError):
            pass
    return decode_static_maps(maps_dir)


class BlastSelector:
    """Selects candidate failing tests for each commit using precomputed sqlglot@eb11aec maps."""

    def __init__(
        self,
        maps_dir: Path,
        runtime_tests: list[dict[str, Any]] | None = None,
        cache_path: Path | None = None,
    ) -> None:
        m = load_static_maps(maps_dir, cache_path)
        self.sets = CoverageSets(m["sets"])
        self.line_to_set: dict[str, dict[int, int]] = m["line_to_set"]
        self.import_time_lines: dict[str, frozenset[int]] = m["import_time_lines"]
        self.import_time_qualnames: dict[str, frozenset[str]] = m["import_time_qualnames"]
        self.qualname_to_set: dict[str, dict[str, int]] = m["qualname_to_set"]
        self.file_to_set: dict[str, int] = m["file_to_set"]
        self.ast_index: dict[str, Any] = m["ast_index"]
        self.modules: dict[str, Any] = m["modules"]
        self.dialect_families: dict[str, frozenset[int]] = {
            fam: self.sets[sid] for fam, sid in m["dialect_family_sids"].items()
        }
        raw_closure = m.get("dialect_import_closure")
        if raw_closure:
            self.dialect_import_closure: frozenset[str] = frozenset(raw_closure)
        else:
            closure_set: set[str] = {
                "sqlglot/dialects/__init__.py",
                "sqlglot/dialects/athena.py",
                "sqlglot/dialects/snowflake.py",
            }
            queue = list(closure_set)
            while queue:
                cur = queue.pop()
                for dep in self.modules.get(cur, {}).get("imports", ()):
                    if dep not in closure_set:
                        closure_set.add(dep)
                        queue.append(dep)
            self.dialect_import_closure = frozenset(closure_set)
        self.file_to_tests: dict[str, frozenset[int]] = m["file_to_tests"]
        self.fixture_to_tests: dict[str, frozenset[int]] = m["fixture_to_tests"]
        self.source_inspecting_tests: frozenset[int] = m["source_inspecting_tests"]
        self.source_file_to_inspecting_tests: dict[str, frozenset[int]] = m[
            "source_file_to_inspecting_tests"
        ]
        self.unmapped_source_inspecting_tests: frozenset[int] = m["unmapped_source_inspecting_tests"]
        self.tpcds_plan_hashes: tuple[str, ...] = m["tpcds_plan_hashes"]
        self.tpcds_exec_lines: dict[str, frozenset[int]] = m["tpcds_exec_lines"]
        self.tpcds_exec_qualnames: dict[str, frozenset[str]] = m["tpcds_exec_qualnames"]
        self.base_method_sids: dict[str, list[int]] = m["base_method_sids"]
        self.base_attr_sids: dict[str, list[int]] = m["base_attr_sids"]

        self.id_by_num: dict[int, str] = {}
        self.num_by_id: dict[str, int] = {}
        self.class_to_tests: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.method_to_test: dict[tuple[str, str], int] = {}

        tests_src = runtime_tests if runtime_tests is not None else m["tests"]
        self.all_test_nums: frozenset[int] = frozenset(t["number"] for t in tests_src)
        for t in tests_src:
            num = t["number"]
            tid = t["id"]
            tfile = t["file"]
            self.id_by_num[num] = tid
            self.num_by_id[tid] = num
            parts = tid.split("::")
            if len(parts) == 3:
                cls_name, meth_name = parts[1], parts[2]
                self.class_to_tests[(tfile, cls_name)].append(num)
                self.method_to_test[(tfile, f"{cls_name}.{meth_name}")] = num

        self.typing_meta_test: int | None = self.num_by_id.get(
            "tests/test_typing.py::TestTyping::test_no_redundant_expression_metadata"
        )
        self.dialect_import_tests: frozenset[int] = self.file_to_tests.get(
            "tests/test_dialect_imports.py", frozenset()
        )

    def _transitive_tests(self, rel_path: str) -> set[int]:
        out: set[int] = set()
        mod = self.modules.get(rel_path)
        if mod:
            for tf in mod.get("transitive_test_files", ()):
                out.update(self.file_to_tests.get(tf, ()))
        if rel_path in self.file_to_tests:
            out.update(self.file_to_tests[rel_path])
        return out

    def _file_runtime_tests(self, rel_path: str) -> frozenset[int]:
        sid = self.file_to_set.get(rel_path, 0)
        return self.sets[sid]

    def _dialect_family_tests(self, rel_path: str) -> frozenset[int]:
        fam = None
        mod = self.modules.get(rel_path)
        if mod:
            fam = mod.get("dialect_family")
        if fam is None:
            fam = _dialect_family_from_path(rel_path)
        if fam:
            return self.dialect_families.get(fam, frozenset())
        return frozenset()

    def _select_non_python(self, fa: FileAnalysis) -> set[int]:
        out: set[int] = set()
        for path in (fa.old_path, fa.new_path):
            if path and path in self.fixture_to_tests:
                out.update(self.fixture_to_tests[path])
        if out:
            return out

        rel = fa.rel_path
        if rel.startswith("tests/fixtures/"):
            if rel.startswith("tests/fixtures/optimizer/tpc-ds/"):
                for fpath, nums in self.fixture_to_tests.items():
                    if fpath.startswith("tests/fixtures/optimizer/tpc-ds/"):
                        out.update(nums)
            elif rel.startswith("tests/fixtures/optimizer/tpc-h/"):
                for fpath, nums in self.fixture_to_tests.items():
                    if fpath.startswith("tests/fixtures/optimizer/tpc-h/"):
                        out.update(nums)
            elif rel.startswith("tests/fixtures/optimizer/"):
                out.update(self.file_to_tests.get("tests/test_optimizer.py", ()))
                out.update(self.file_to_tests.get("tests/test_executor.py", ()))
            elif rel.startswith("tests/fixtures/jsonpath/"):
                out.update(self.file_to_tests.get("tests/test_jsonpath.py", ()))
            return out if out else set(self.all_test_nums)

        if rel in _BUILD_CONFIG_FILES or rel.startswith("sqlglotc/"):
            out.update(self.file_to_tests.get("tests/test_sqlglotc.py", ()))
            out.update(self.file_to_tests.get("tests/test_build.py", ()))
            out.update(self.file_to_tests.get("tests/test_dialect_entry_points.py", ()))
            return out if out else set(self.all_test_nums)

        if rel.startswith(_NON_EXEC_PREFIXES) or rel.endswith(_NON_EXEC_SUFFIXES):
            return set()

        return set(self.all_test_nums)

    def _select_broken_or_removed(self, fa: FileAnalysis) -> set[int]:
        out: set[int] = set()
        paths = [p for p in (fa.old_path, fa.new_path) if p]
        for p in paths:
            out.update(self._transitive_tests(p))
            out.update(self._file_runtime_tests(p))
            out.update(self._dialect_family_tests(p))
            if p.startswith("sqlglot/"):
                out.update(self.dialect_import_tests)
            if p.startswith("sqlglot/typing/") and self.typing_meta_test is not None:
                out.add(self.typing_meta_test)
            if p.startswith("sqlglot/dialects/"):
                out.update(self.file_to_tests.get("tests/dialects/test_dialect.py", ()))
            if p in (
                "sqlglot/__init__.py",
                "sqlglot/dialects/__init__.py",
                "sqlglot/expressions/__init__.py",
                "sqlglot/optimizer/__init__.py",
            ):
                return set(self.all_test_nums)
        return out if out else set(self.all_test_nums)

    def _select_new_python(self, fa: FileAnalysis) -> set[int]:
        rel = fa.rel_path
        out: set[int] = set()
        if rel.startswith("sqlglot/typing/"):
            if self.typing_meta_test is not None:
                out.add(self.typing_meta_test)
            out.update(self.dialect_import_tests)
            out.update(self.file_to_tests.get("tests/test_typing.py", ()))
            out.update(self._dialect_family_tests(rel))
            return out
        if rel.startswith("sqlglot/dialects/"):
            out.update(self.dialect_import_tests)
            out.update(self.file_to_tests.get("tests/dialects/test_dialect.py", ()))
            out.update(self._dialect_family_tests(rel))
            return out
        if rel.startswith("tests/"):
            return set(self.file_to_tests.get(rel, self.all_test_nums))
        return set(self.all_test_nums)

    def _select_ast_noop(self, fa: FileAnalysis) -> set[int]:
        rel = fa.rel_path
        out: set[int] = set()
        if rel.startswith("sqlglot/typing/") and self.typing_meta_test is not None:
            out.add(self.typing_meta_test)
        if fa.raw_text_changed:
            out.update(self.source_file_to_inspecting_tests.get(rel, ()))
            if self.unmapped_source_inspecting_tests:
                out.update(self.unmapped_source_inspecting_tests & self._file_runtime_tests(rel))
        if fa.docstring_changed:
            file_tests = self._file_runtime_tests(rel)
            out.update(self.source_inspecting_tests & file_tests)
            if rel.startswith("sqlglot/dialects/"):
                out.update(self.dialect_import_tests)
            if rel in self.file_to_tests:
                out.update(self.file_to_tests[rel])
        return out

    def _select_test_file(self, fa: FileAnalysis) -> set[int]:
        rel = fa.rel_path
        if rel not in self.file_to_tests:
            if rel.endswith("conftest.py"):
                return set(self.all_test_nums)
            out = self._select_source_file(fa)
            if fa.imports_changed or fa.top_level_code_changed or not out:
                out.update(self._transitive_tests(rel))
            return out if out else set(self.all_test_nums)

        all_in_file = self.file_to_tests[rel]
        if fa.imports_changed or fa.top_level_code_changed or fa.class_header_changed:
            return set(all_in_file)

        out: set[int] = set()
        ln_map = self.line_to_set.get(rel, {})
        qn_map = self.qualname_to_set.get(rel, {})

        for qn in fa.changed_qualnames:
            tnum = self.method_to_test.get((rel, qn))
            if tnum is not None:
                out.add(tnum)
                continue
            parts = qn.split(".")
            if len(parts) >= 2 and parts[-1] in _LIFECYCLE_METHODS:
                out.update(self.class_to_tests.get((rel, parts[0]), all_in_file))
                continue
            sid = qn_map.get(qn, 0)
            if sid:
                out.update(self.sets[sid] & all_in_file)
            else:
                out.update(all_in_file)

        for cls_qn, _meth in fa.added_methods:
            out.update(self.class_to_tests.get((rel, cls_qn.split(".")[0]), all_in_file))

        for ac in fa.attr_changes:
            if ac.class_qn:
                out.update(self.class_to_tests.get((rel, ac.class_qn.split(".")[0]), all_in_file))
            else:
                out.update(all_in_file)

        for ln in fa.expanded_old_lines:
            sid = ln_map.get(ln, 0)
            if sid:
                out.update(self.sets[sid] & all_in_file)

        return out if out else set(all_in_file)

    def _select_source_file(self, fa: FileAnalysis) -> set[int]:
        rel = fa.rel_path
        out: set[int] = set()
        ln_map = self.line_to_set.get(rel, {})
        qn_map = self.qualname_to_set.get(rel, {})
        imp_lines = self.import_time_lines.get(rel, frozenset())
        imp_qns = self.import_time_qualnames.get(rel, frozenset())
        f_ast = self.ast_index.get(rel, {})
        fam_tests = self._dialect_family_tests(rel)
        if rel == "sqlglot/dialects/__init__.py" or (
            rel in self.dialect_import_closure
            and (
                fa.imports_changed
                or fa.top_level_code_changed
                or fa.class_header_changed
                or fa.attr_changes
                or fa.added_methods
                or any(qn in imp_qns for qn in fa.changed_qualnames)
                or any(ln in imp_lines for ln in (fa.changed_old_lines | fa.expanded_old_lines))
            )
        ):
            # Lazy dialect imports execute import-time code whose coverage is not attributed to tests.
            out.update(self.dialect_import_tests)

        sym_spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for qn, _k, s, e in f_ast.get("symbols", ()):
            sym_spans[qn].append((s, e))
        changed_func_spans = [
            span for qn in fa.changed_qualnames for span in sym_spans.get(qn, ())
        ]
        mapped_attr_spans: list[tuple[int, int]] = []
        for ac in fa.attr_changes:
            info = (
                f_ast.get("class_attrs", {}).get(ac.class_qn, {}).get(ac.attr_name)
                if ac.class_qn is not None
                else f_ast.get("module_attrs", {}).get(ac.attr_name)
            )
            if info is not None:
                mapped_attr_spans.append((info["start"], info["end"]))

        for ln in fa.expanded_old_lines:
            if (
                ln in imp_lines
                and mapped_attr_spans
                and any(s <= ln <= e for s, e in mapped_attr_spans)
                and not any(s <= ln <= e for s, e in changed_func_spans)
            ):
                continue
            sid = ln_map.get(ln, 0)
            if sid:
                out.update(self.sets[sid])

        if fa.imports_changed:
            out.update(self._transitive_tests(rel))
            out.update(self._file_runtime_tests(rel))
            out.update(fam_tests)
            if rel.startswith("sqlglot/dialects/"):
                out.update(self.dialect_import_tests)
            if rel.startswith("sqlglot/typing/") and self.typing_meta_test is not None:
                out.add(self.typing_meta_test)

        if fa.top_level_code_changed or fa.class_header_changed:
            out.update(self._file_runtime_tests(rel))
            out.update(fam_tests)
            out.update(self._transitive_tests(rel))

        for ac in fa.attr_changes:
            info = (
                f_ast.get("class_attrs", {}).get(ac.class_qn, {}).get(ac.attr_name)
                if ac.class_qn is not None
                else f_ast.get("module_attrs", {}).get(ac.attr_name)
            )
            if info is not None:
                is_expression_arg_types = (
                    rel.startswith("sqlglot/expressions/")
                    and ac.class_qn is not None
                    and ac.attr_name == "arg_types"
                )
                if (
                    ac.changed_dict_keys
                    and not ac.whole_attr_changed
                    and not ac.keys_membership_changed
                    and not is_expression_arg_types
                ):
                    dk_map = info.get("dict_keys", {})
                    for key in ac.changed_dict_keys:
                        dk_entry = dk_map.get(key)
                        if dk_entry is not None:
                            out.update(self.sets[dk_entry[2]])
                        else:
                            out.update(self.sets[info["set_id"]])
                else:
                    out.update(self.sets[info["set_id"]])
                if is_expression_arg_types:
                    out.update(self.sets[qn_map.get(ac.class_qn, 0)])
            else:
                base_sids = self.base_attr_sids.get(ac.attr_name)
                if base_sids:
                    for sid in base_sids:
                        out.update(self.sets[sid])
                    out.update(fam_tests)
                elif fam_tests:
                    out.update(fam_tests)
                else:
                    out.update(self._file_runtime_tests(rel))

            if (
                ac.class_qn is not None
                and ac.attr_name in {"KEYWORDS", "TOKENS", "SINGLE_TOKENS"}
                and (rel == "sqlglot/tokens.py" or rel.startswith("sqlglot/dialects/"))
            ):
                # Tokenizer tables can be copied into per-dialect objects before test coverage starts.
                out.update(self._file_runtime_tests("sqlglot/tokenizer_core.py"))
                out.update(self._file_runtime_tests("sqlglot/tokens.py"))
                out.update(self._transitive_tests(rel))
                out.update(self.dialect_import_tests)
            if rel.startswith("sqlglot/typing/") and self.typing_meta_test is not None:
                out.add(self.typing_meta_test)

        # Newly added subclass methods can affect both base-class and dialect-family callers.
        for _cls_qn, meth in fa.added_methods:
            base_sids = self.base_method_sids.get(meth)
            if base_sids:
                base_union = set()
                for sid in base_sids:
                    base_union.update(self.sets[sid])
                out.update(base_union)
                out.update(fam_tests)
            elif fam_tests:
                out.update(fam_tests)
            else:
                out.update(self._file_runtime_tests(rel))

        # Fall back to enclosing symbol coverage when edits hit signatures, decorators, or uncovered lines,
        # and expand to transitive importers when a function whose body runs at import time changes.
        for qn in fa.changed_qualnames:
            spans = sym_spans.get(qn, ())
            if qn in imp_qns:
                child_non_imp_spans = [
                    sp
                    for other in fa.changed_qualnames
                    if other != qn and other.startswith(qn + ".") and other not in imp_qns
                    for sp in sym_spans.get(other, ())
                ]
                direct_lines = [
                    ln
                    for ln in (fa.changed_old_lines or fa.expanded_old_lines)
                    if any(s_ln <= ln <= e_ln for s_ln, e_ln in spans)
                ]
                only_inside_non_imp_child = bool(direct_lines) and bool(child_non_imp_spans) and all(
                    any(cs <= ln <= ce for cs, ce in child_non_imp_spans) for ln in direct_lines
                )
                if not only_inside_non_imp_child:
                    out.update(self._transitive_tests(rel))
                    out.update(self._file_runtime_tests(rel))
                    out.update(fam_tests)

            sid = qn_map.get(qn, 0)
            if not sid:
                if any(
                    ln in imp_lines
                    for ln in fa.changed_old_lines
                    for s_ln, e_ln in spans
                    if s_ln <= ln <= e_ln
                ):
                    out.update(self._transitive_tests(rel))
                    out.update(self._file_runtime_tests(rel))
                    out.update(fam_tests)
                continue
            if not spans:
                out.update(self.sets[sid])
                continue
            for s_ln, e_ln in spans:
                in_sym_changed = [ln for ln in fa.expanded_old_lines if s_ln <= ln <= e_ln]
                if not in_sym_changed:
                    continue
                hit_uncovered_or_import = any(
                    ln not in ln_map or ln in imp_lines for ln in in_sym_changed
                )
                hit_header_or_imp = any(
                    ln == s_ln or ln in imp_lines
                    for ln in fa.changed_old_lines
                    if s_ln <= ln <= e_ln
                )
                if hit_uncovered_or_import or hit_header_or_imp:
                    out.update(self.sets[sid])
                    break

        if fa.raw_text_changed:
            out.update(self.source_file_to_inspecting_tests.get(rel, ()))
            if self.unmapped_source_inspecting_tests:
                out.update(self.unmapped_source_inspecting_tests & self._file_runtime_tests(rel))
            if rel.startswith("sqlglot/typing/") and self.typing_meta_test is not None:
                out.add(self.typing_meta_test)

        return out

    def select(self, analysis: CommitAnalysis) -> SelectionResult:
        """Return candidate test numbers for a parsed commit."""
        if not analysis.files:
            return SelectionResult(frozenset(), is_provable_noop=True, has_syntax_or_import_break=False)

        candidates: set[int] = set()
        all_noop = True
        has_break = False

        for fa in analysis.files:
            if not fa.is_python:
                non_py = self._select_non_python(fa)
                if non_py:
                    all_noop = False
                    candidates.update(non_py)
                continue

            if fa.syntax_broken or fa.is_deleted or fa.is_renamed:
                all_noop = False
                has_break = True
                candidates.update(self._select_broken_or_removed(fa))
                continue

            if fa.is_new:
                all_noop = False
                has_break = True
                candidates.update(self._select_new_python(fa))
                continue

            if fa.is_ast_noop:
                noop_hits = self._select_ast_noop(fa)
                if noop_hits:
                    all_noop = False
                    candidates.update(noop_hits)
                continue

            all_noop = False
            if fa.rel_path.startswith("tests/"):
                candidates.update(self._select_test_file(fa))
            else:
                candidates.update(self._select_source_file(fa))

        return SelectionResult(
            candidates=frozenset(candidates & self.all_test_nums),
            is_provable_noop=all_noop and not candidates,
            has_syntax_or_import_break=has_break,
        )
