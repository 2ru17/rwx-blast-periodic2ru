"""Coverage- and AST-guided candidate test selector."""

from __future__ import annotations

import json
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


def _decode_set(encoded: list[int | list[int]]) -> frozenset[int]:
    out: list[int] = []
    for item in encoded:
        if isinstance(item, list):
            out.extend(range(item[0], item[1] + 1))
        else:
            out.append(item)
    return frozenset(out)


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


class BlastSelector:
    """Selects candidate failing tests for each commit using precomputed sqlglot@eb11aec maps."""

    def __init__(self, maps_dir: Path, runtime_tests: list[dict[str, Any]] | None = None) -> None:
        test_index = json.loads((maps_dir / "test_index.json").read_text(encoding="utf-8"))
        cov_map = json.loads((maps_dir / "coverage_map.json").read_text(encoding="utf-8"))
        ast_index = json.loads((maps_dir / "ast_index.json").read_text(encoding="utf-8"))
        import_graph = json.loads((maps_dir / "import_graph.json").read_text(encoding="utf-8"))

        self.sets: tuple[frozenset[int], ...] = tuple(_decode_set(s) for s in cov_map["sets"])
        self.line_to_set: dict[str, dict[int, int]] = {
            rel: {int(k): v for k, v in ln_dict.items()}
            for rel, ln_dict in cov_map["line_to_set"].items()
        }
        self.import_time_lines: dict[str, frozenset[int]] = {
            rel: frozenset(lns) for rel, lns in cov_map.get("import_time_lines", {}).items()
        }
        self.import_time_qualnames: dict[str, frozenset[str]] = {
            rel: frozenset(qns) for rel, qns in cov_map.get("import_time_qualnames", {}).items()
        }
        self.qualname_to_set: dict[str, dict[str, int]] = cov_map["qualname_to_set"]
        self.file_to_set: dict[str, int] = cov_map["file_to_set"]
        self.ast_index: dict[str, Any] = ast_index
        self.modules: dict[str, Any] = import_graph["modules"]
        self.dialect_families: dict[str, frozenset[int]] = {
            fam: self.sets[sid] for fam, sid in import_graph["dialect_families"].items()
        }

        self.file_to_tests: dict[str, frozenset[int]] = {
            f: frozenset(nums) for f, nums in test_index["file_to_test_numbers"].items()
        }
        self.fixture_to_tests: dict[str, frozenset[int]] = {
            f: frozenset(nums) for f, nums in test_index["fixture_to_test_numbers"].items()
        }
        self.source_inspecting_tests: frozenset[int] = frozenset(
            test_index.get("source_inspecting_tests", ())
        )
        self.source_file_to_inspecting_tests: dict[str, frozenset[int]] = {
            rel: frozenset(nums)
            for rel, nums in test_index.get("source_file_to_inspecting_tests", {}).items()
        }
        mapped_inspecting: set[int] = set()
        for nums in self.source_file_to_inspecting_tests.values():
            mapped_inspecting.update(nums)
        self.unmapped_source_inspecting_tests: frozenset[int] = (
            self.source_inspecting_tests - mapped_inspecting
        )
        self.tpcds_plan_hashes: tuple[str, ...] = tuple(
            test_index.get("tpcds_plan_hashes", ())
        )
        self.tpcds_exec_lines: dict[str, frozenset[int]] = {
            rel: _decode_set(enc)
            for rel, enc in test_index.get("tpcds_exec_lines", {}).items()
        }
        self.tpcds_exec_qualnames: dict[str, frozenset[str]] = {}
        for rel, exec_lines in self.tpcds_exec_lines.items():
            f_ast = self.ast_index.get(rel, {})
            self.tpcds_exec_qualnames[rel] = frozenset(
                qn
                for qn, kind, start, end in f_ast.get("symbols", ())
                if kind in ("function", "method") and any(start <= ln <= end for ln in exec_lines)
            )

        self.id_by_num: dict[int, str] = {}
        self.num_by_id: dict[str, int] = {}
        self.class_to_tests: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.method_to_test: dict[tuple[str, str], int] = {}

        tests_src = runtime_tests if runtime_tests is not None else test_index["tests"]
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

        # Index base-class method coverage by leaf method name for subclass override propagation.
        self.base_method_sets: dict[str, list[frozenset[int]]] = defaultdict(list)
        for rel in _CORE_BASE_FILES:
            for qn, sid in self.qualname_to_set.get(rel, {}).items():
                if "." in qn:
                    meth = qn.rsplit(".", 1)[1]
                    self.base_method_sets[meth].append(self.sets[sid])

        # Index class/module attributes across core base files for newly added subclass attributes.
        self.base_attr_sets: dict[str, list[frozenset[int]]] = defaultdict(list)
        for rel in _CORE_BASE_FILES:
            f_ast = self.ast_index.get(rel, {})
            for attrs in f_ast.get("class_attrs", {}).values():
                for attr_name, info in attrs.items():
                    tset = self.sets[info["set_id"]]
                    if tset:
                        self.base_attr_sets[attr_name].append(tset)

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
            if p.startswith("sqlglot/typing/") and self.typing_meta_test is not None:
                out.add(self.typing_meta_test)
            if p.startswith("sqlglot/dialects/"):
                out.update(self.dialect_import_tests)
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
                mapped_attr_spans
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
                if (
                    ac.changed_dict_keys
                    and not ac.whole_attr_changed
                    and not ac.keys_membership_changed
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
            else:
                base_sets = self.base_attr_sets.get(ac.attr_name)
                if base_sets and fam_tests:
                    base_union: set[int] = set()
                    for s in base_sets:
                        base_union.update(s)
                    inter = base_union & fam_tests
                    out.update(inter if inter else fam_tests)
                elif fam_tests:
                    out.update(fam_tests)
                else:
                    out.update(self._file_runtime_tests(rel))

            if rel.startswith("sqlglot/typing/") and self.typing_meta_test is not None:
                out.add(self.typing_meta_test)

        # Newly added subclass methods propagate to base-class coverage intersected with dialect family.
        for _cls_qn, meth in fa.added_methods:
            base_sets = self.base_method_sets.get(meth)
            if base_sets:
                base_union = set()
                for s in base_sets:
                    base_union.update(s)
                if fam_tests:
                    inter = base_union & fam_tests
                    out.update(inter if inter else fam_tests)
                else:
                    out.update(base_union)
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
                hit_covered_runtime = any(ln in ln_map and ln not in imp_lines for ln in in_sym_changed)
                hit_header_or_imp = any(
                    ln == s_ln or ln in imp_lines
                    for ln in fa.changed_old_lines
                    if s_ln <= ln <= e_ln
                )
                if not hit_covered_runtime or hit_header_or_imp:
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
