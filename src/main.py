"""Entrypoint for RWX Blast Radius hybrid test selector."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

from diff_analyzer import CommitAnalysis, analyze_patch
from scheduler import MAX_WORKERS, compute_commit_cost_ms, duration_ms, schedule_tests
from selector import BlastSelector
from verifier import verify_batch

MAX_STDOUT_BYTES = 2 * 1024 * 1024


def format_and_validate_output(
    commit_workers: dict[str, list[list[int]]],
    expected_commits: set[str],
    valid_test_nums: set[int],
) -> bytes:
    """Serialize commit schedules to UTF-8 JSONL while enforcing all 8 SCORING.md rules."""
    if set(commit_workers.keys()) != expected_commits:
        missing = sorted(expected_commits - set(commit_workers.keys()))
        extra = sorted(set(commit_workers.keys()) - expected_commits)
        raise ValueError(f"Commit mismatch: missing={missing[:5]}, extra={extra[:5]}")

    lines: list[str] = []
    for cid in sorted(expected_commits):
        workers = commit_workers[cid]
        seen_tests: set[int] = set()
        clean_workers: list[list[int]] = []
        for worker in workers:
            if not worker:
                continue
            clean_w: list[int] = []
            for item in worker:
                if isinstance(item, bool) or not isinstance(item, int):
                    raise ValueError(f"Invalid test number {item!r} on commit {cid}")
                if item not in valid_test_nums:
                    raise ValueError(f"Unknown test number {item} on commit {cid}")
                if item in seen_tests:
                    raise ValueError(f"Duplicate test number {item} on commit {cid}")
                seen_tests.add(item)
                clean_w.append(item)
            clean_workers.append(clean_w)
        if len(clean_workers) > MAX_WORKERS:
            raise ValueError(f"Too many workers ({len(clean_workers)}) on commit {cid}")

        line = json.dumps(
            {"commit": cid, "workers": clean_workers},
            allow_nan=False,
            separators=(",", ":"),
        )
        lines.append(line)

    raw = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    if len(raw) > MAX_STDOUT_BYTES:
        raise ValueError(f"stdout exceeds 2 MB limit: {len(raw)} bytes")
    return raw


def run_pipeline(
    data_dir: Path,
    maps_dir: Path,
    sqlglot_root: Path,
    *,
    enable_verify: bool = True,
    tmp_base: str = "/tmp",
) -> tuple[bytes, dict[str, Any]]:
    """Run diff analysis, coverage selection, cost-aware verification, and scheduling."""
    t0 = time.perf_counter()
    tests_raw = json.loads((data_dir / "tests.json").read_text(encoding="utf-8"))["tests"]
    ms_by_num = {t["number"]: duration_ms(t["seconds"]) for t in tests_raw}
    id_by_num = {t["number"]: t["id"] for t in tests_raw}
    file_by_num = {t["number"]: t["file"] for t in tests_raw}
    valid_test_nums = set(ms_by_num.keys())

    selector = BlastSelector(maps_dir, runtime_tests=tests_raw)

    commits_dir = data_dir / "commits"
    patch_paths = sorted(commits_dir.glob("*.patch"), key=lambda p: p.name)
    analyses: dict[str, CommitAnalysis] = {}
    candidate_sets: dict[str, frozenset[int]] = {}
    for patch_path in patch_paths:
        cid = patch_path.name.removesuffix(".patch")
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
        analysis = analyze_patch(cid, patch_text, sqlglot_root)
        sel = selector.select(analysis)
        analyses[cid] = analysis
        candidate_sets[cid] = sel.candidates

    if enable_verify:
        verify_outcomes = verify_batch(
            analyses=analyses,
            candidate_sets=candidate_sets,
            sqlglot_root=sqlglot_root,
            ms_by_num=ms_by_num,
            id_by_num=id_by_num,
            file_by_num=file_by_num,
            file_to_tests=selector.file_to_tests,
            fixture_to_tests=selector.fixture_to_tests,
            tpcds_plan_hashes=selector.tpcds_plan_hashes,
            tmp_base=tmp_base,
        )
    else:
        verify_outcomes = {}

    commit_workers: dict[str, list[list[int]]] = {}
    total_sim_ms = 0
    verified_commits = 0
    total_shed = 0
    total_child_cpu_s = 0.0

    for cid in sorted(analyses):
        outcome = verify_outcomes.get(cid)
        if outcome is not None:
            final_tests = outcome.final_tests
            if outcome.verified:
                verified_commits += 1
                total_shed += outcome.shed_count
                total_child_cpu_s += outcome.child_cpu_s
        else:
            final_tests = candidate_sets[cid]
        workers = schedule_tests(final_tests, ms_by_num)
        commit_workers[cid] = workers
        total_sim_ms += compute_commit_cost_ms(workers, ms_by_num)

    payload = format_and_validate_output(
        commit_workers=commit_workers,
        expected_commits=set(analyses.keys()),
        valid_test_nums=valid_test_nums,
    )
    ru_self = resource.getrusage(resource.RUSAGE_SELF)
    ru_kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    total_cpu_s = ru_self.ru_utime + ru_self.ru_stime + ru_kids.ru_utime + ru_kids.ru_stime
    stats = {
        "commits": len(analyses),
        "verified_commits": verified_commits,
        "shed_tests": total_shed,
        "child_cpu_s": round(total_child_cpu_s, 3),
        "total_cpu_s": round(total_cpu_s, 4),
        "simulated_s": round(total_sim_ms / 1000.0, 3),
        "wall_s": round(time.perf_counter() - t0, 3),
    }
    return payload, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RWX Blast Radius hybrid test selector.")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", "/data")))
    parser.add_argument("--maps-dir", type=Path, default=Path(os.environ.get("MAPS_DIR", "/app/maps")))
    parser.add_argument(
        "--sqlglot-root",
        type=Path,
        default=Path(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot")),
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        default=os.environ.get("DISABLE_VERIFY", "0") == "1",
    )
    args = parser.parse_args(argv)

    payload, stats = run_pipeline(
        data_dir=args.data_dir,
        maps_dir=args.maps_dir,
        sqlglot_root=args.sqlglot_root,
        enable_verify=not args.no_verify,
    )
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
    print(f"rwx-blast summary: {json.dumps(stats, sort_keys=True)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
