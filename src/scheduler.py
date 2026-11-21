"""Exact integer-millisecond worker scheduler matching SCORING.md."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

WORKER_COST_MS = 2_000
MAX_WORKERS = 8


def duration_ms(seconds: float) -> int:
    """Round seconds to whole milliseconds with halves rounded up."""
    return math.floor(seconds * 1000 + 0.5)


def compute_commit_cost_ms(workers: list[list[int]], ms_by_num: Mapping[int, int]) -> int:
    """Compute exact simulated millisecond cost for one commit's worker layout."""
    occupied = [w for w in workers if w]
    if not occupied:
        return 0
    max_load = max(sum(ms_by_num[t] for t in w) for w in occupied)
    return max_load + WORKER_COST_MS * len(occupied)


def _refine_workers(
    workers: list[list[int]],
    loads: list[int],
    ms_by_num: Mapping[int, int],
    lower_bound_load: int,
) -> None:
    """Greedily reduce makespan via single-item moves and pairwise swaps."""
    k = len(workers)
    for _ in range(64):
        hi = max(range(k), key=lambda i: loads[i])
        hi_load = loads[hi]
        if hi_load <= lower_bound_load:
            break

        best_move: tuple[int, int, int] | None = None
        best_new_max = hi_load
        for idx_hi, t_hi in enumerate(workers[hi]):
            d_hi = ms_by_num[t_hi]
            if d_hi == 0:
                continue
            for j in range(k):
                if j == hi:
                    continue
                new_pair_max = max(hi_load - d_hi, loads[j] + d_hi)
                if new_pair_max < best_new_max:
                    best_new_max = new_pair_max
                    best_move = (idx_hi, j, d_hi)

        if best_move is not None:
            idx_hi, j, d_hi = best_move
            t_hi = workers[hi].pop(idx_hi)
            workers[j].append(t_hi)
            loads[hi] -= d_hi
            loads[j] += d_hi
            continue

        best_swap: tuple[int, int, int, int] | None = None
        best_swap_max = hi_load
        for idx_hi, t_hi in enumerate(workers[hi]):
            d_hi = ms_by_num[t_hi]
            for j in range(k):
                if j == hi:
                    continue
                j_load = loads[j]
                for idx_j, t_j in enumerate(workers[j]):
                    d_j = ms_by_num[t_j]
                    diff = d_hi - d_j
                    if diff <= 0:
                        continue
                    new_pair_max = max(hi_load - diff, j_load + diff)
                    if new_pair_max < best_swap_max:
                        best_swap_max = new_pair_max
                        best_swap = (idx_hi, j, idx_j, diff)

        if best_swap is None:
            break
        idx_hi, j, idx_j, diff = best_swap
        workers[hi][idx_hi], workers[j][idx_j] = workers[j][idx_j], workers[hi][idx_hi]
        loads[hi] -= diff
        loads[j] += diff


def schedule_tests(test_nums: Iterable[int], ms_by_num: Mapping[int, int]) -> list[list[int]]:
    """Partition tests across up to 8 workers minimizing max_load + 2000 * k."""
    unique = sorted(set(test_nums))
    if not unique:
        return []

    ordered = sorted(unique, key=lambda n: (-ms_by_num[n], n))
    total_ms = sum(ms_by_num[n] for n in ordered)
    max_single_ms = ms_by_num[ordered[0]]

    best_cost = total_ms + WORKER_COST_MS
    best_workers: list[list[int]] = [list(ordered)]

    max_k = min(MAX_WORKERS, len(ordered))
    for k in range(2, max_k + 1):
        worker_overhead = WORKER_COST_MS * k
        if max_single_ms + worker_overhead >= best_cost:
            break
        lb_load = max(max_single_ms, (total_ms + k - 1) // k)
        if lb_load + worker_overhead >= best_cost:
            continue

        workers: list[list[int]] = [[] for _ in range(k)]
        loads = [0] * k
        for t in ordered:
            i = min(range(k), key=lambda idx: (loads[idx], idx))
            workers[i].append(t)
            loads[i] += ms_by_num[t]

        _refine_workers(workers, loads, ms_by_num, lb_load)
        occupied = [(w, ld) for w, ld in zip(workers, loads) if w]
        cost = max(ld for _, ld in occupied) + WORKER_COST_MS * len(occupied)
        if cost < best_cost:
            best_cost = cost
            best_workers = [w for w, _ in occupied]

    canonical = [sorted(w) for w in best_workers if w]
    canonical.sort(key=lambda w: (-sum(ms_by_num[t] for t in w), w[0]))
    return canonical
