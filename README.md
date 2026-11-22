# rwx-blast-periodic2ru

Hybrid coverage-guided test selector, cost-aware in-container verifier, and exact worker scheduler for the **RWX Blast Radius** track targeting `sqlglot` (`eb11aec4be8cb2122afe713a4816662c3f0fb2fa`).

## Leaderboard Results (40-Commit Practice Board)

Scoring formula (lower is better, median of 3 runs):
`simulated_seconds + container_CPU_seconds`, where each commit costs `max(worker_load_ms) + 2000 * num_workers` (`<= 8` workers, `0` if empty) plus a `+300 s` penalty for any commit missing at least one failing test.

| Submission | Commit | Simulated (s) | CPU Runs 1 / 2 / 3 (s) | Misses | Total Score | Notes |
| :--- | :--- | ---: | :--- | ---: | ---: | :--- |
| `BB-9624B233` | `778740e` | `471.395` | `8.44 / 10.03 / 10.40` | `1` | `481.4 s` | Initial baseline |
| **`BB-B6AB2C38`** | **`27819a7`** | **`184.057`** | **`6.19 / 6.30 / 6.27`** | **`0`** | **`190.3 s`** | **Entered in the final (best score)** |
| `BB-E960E3B4` | `ba11c25` | `140.357` | `93.01 / 92.66 / 93.67` | `0` | `233.4 s` | `-43.700 s` simulated (`0` misses), higher verifier CPU |

**Takeaway:**
- Hardening static analysis and offline dynamic-context coverage eliminated all misses (`1 -> 0`) while keeping container CPU near `6.27 s` (`BB-B6AB2C38`, **`190.3 s`**, entered in the final).
- Expanding verification into heavy long-running tests (`BB-E960E3B4`) proved zero-miss determinism (`140.357 s` simulated across all 3 runs, `-43.7 s` simulated vs `BB-B6AB2C38`), showing clearly where the simulated-time vs container-CPU breakeven sits under RWX's scoring function.

## Architecture

```text
src/
  main.py            Entrypoint, pipeline orchestration, and strict 8-rule JSONL output validator
  diff_analyzer.py   Unified-diff patch applier, AST summary, comment-only fast path, and closed-world equivalence checks
  selector.py        Offline map loader (marshal cache + lazy CoverageSets) and conservative candidate selector
  verifier.py        2-tier forked /tmp symlink-overlay verifier with golden-hash TPC-DS verification, ROI gating, and checkpoints
  scheduler.py       Exact integer-millisecond LPT + local-refinement scheduler (<= 8 workers)
maps/                Compact precomputed maps derived strictly from pinned sqlglot@eb11aec (< 4.5 MB total)
scripts/
  build_maps.py      Deterministic offline map generator (coverage ctrace dynamic contexts + AST/import/fixture/TPC-DS maps)
tests/               Unit, soundness, cache, module-safety, and output-contract regression tests
Dockerfile           Multi-stage python:3.13-slim image with precompiled stdlib/sqlglot bytecode and build-time marshal caches
```

### Pipeline Stages
1. **Patch & AST Analysis (`diff_analyzer.py`):** Applies each commit patch in memory against `/opt/sqlglot`, detects comment-only / AST no-ops and closed-world `is`/`==` equivalences, and extracts changed lines, qualified names, class attributes, dict keys, and added method overrides.
2. **Candidate Selection (`selector.py`):** Maps each diff against precomputed `ctrace` per-test dynamic coverage, AST/MRO inheritance, dialect import closures, fixtures, and source-inspecting tests (`0`-miss conservative recall).
3. **Cost-Aware Verification (`verifier.py`):** Runs profitable candidate subsets inside isolated forked children over `/tmp` symlink overlays (clean Tier-1 fork + preloaded `sqlglot` Tier-2 fork server with dynamic `sys.modules` safety bounce), fail-fast subtests, golden-hash TPC-DS verification, and incremental pipe checkpoints. Tests are only dropped on verified `PASS`.
4. **Scheduling (`scheduler.py`):** Partitions remaining tests across up to 8 workers using exact `round(seconds * 1000)` durations to minimize `max_load + 2000 * k` with deterministic output ordering.

## Local Build, Sandbox Run & Tests

### Run Unit & Contract Tests
```bash
pytest tests/ -v
ruff check src tests scripts
```

### Build Container Image
```bash
docker build --no-cache --platform linux/amd64 -t localhost/rwx-blast-periodic2ru:local .
```

### Run Under RWX Sandbox Parity
```bash
make sample SAMPLE=/path/to/sample SCORER=/path/to/score.py
```
Equivalent manual command:
```bash
docker run --rm --platform linux/amd64 --network none --cpus 2 \
  --memory 4096m --memory-swap 4096m --pids-limit 256 --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m,mode=1777 --cap-drop ALL \
  --security-opt no-new-privileges --user 65534:65534 --env HOME=/tmp \
  -v /path/to/data/commits:/data/commits:ro \
  -v /path/to/data/tests.json:/data/tests.json:ro \
  localhost/rwx-blast-periodic2ru:local > output.jsonl
```

## Contributors

- [2ru17](https://github.com/2ru17)
- [PeriodicMango](https://github.com/PeriodicMango)
