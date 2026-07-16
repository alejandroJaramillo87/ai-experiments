"""CCD/L3 core pinning for the 9950X (Phase 0 D9).

The 9950X is a single NUMA node with two CCDs (physical cores 0-7 and 8-15,
separate 32MB L3 each; SMT siblings are 16-31). Pinning the main training
process to one CCD and dataloader workers to the other keeps the GPU-feeding
thread's working set out of the workers' L3. This is CCD/L3 pinning, not NUMA
pinning — there is only one memory node.

Driven by the recipe's affinity block (off by default). The wrapper existing
and being measured is the D9 deliverable; if throughput is not better than
affinity-off, the feature stays parked off.
"""

from __future__ import annotations

import os


def parse_cpu_list(spec: str) -> set[int]:
    """Parse a taskset-style cpu list ("0-7", "0,2,4", "0-3,8-11") into a set."""
    cpus: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, _, hi_s = part.partition("-")
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"cpu range {part!r}: start exceeds end")
            cpus.update(range(lo, hi + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError(f"cpu list {spec!r} is empty")
    if min(cpus) < 0:
        raise ValueError(f"cpu list {spec!r} contains a negative cpu id")
    return cpus


def with_smt_siblings(cpus: set[int], logical_cpus: int | None = None) -> set[int]:
    """Extend a physical-core mask with its SMT siblings.

    Linux enumerates the 9950X (and Zen generally) with siblings at
    physical_id + physical_count: core 3's sibling is 19 on a 16-core/32-thread
    part. logical_cpus defaults to os.cpu_count(); masks already containing
    sibling ids pass through unchanged.
    """
    total = logical_cpus if logical_cpus is not None else (os.cpu_count() or 1)
    half = total // 2
    if half == 0:
        return set(cpus)
    return cpus | {c + half for c in cpus if c < half}


def resolve_masks(affinity) -> tuple[set[int], set[int]]:
    """Recipe affinity block -> (main_mask, workers_mask), SMT applied.

    Raises ValueError when either list fails to parse or the resolved masks
    overlap — disjointness is the point of the feature.
    """
    main = parse_cpu_list(affinity.main)
    workers = parse_cpu_list(affinity.workers)
    if affinity.include_smt:
        main = with_smt_siblings(main)
        workers = with_smt_siblings(workers)
    overlap = main & workers
    if overlap:
        raise ValueError(
            f"affinity: main ({affinity.main}) and workers ({affinity.workers}) "
            f"masks overlap on cpus {sorted(overlap)}"
        )
    return main, workers


def apply_mask(cpus: set[int], allowed: set[int] | None = None) -> None:
    """Pin the calling process (and its future threads) to the given cpus.

    Cpus outside the allowance (container cpusets) are dropped rather than
    failing the run; pinning degrades, never aborts. allowed overrides the
    current scheduler allowance: a dataloader worker forks from a main
    process already pinned to the disjoint main mask, so its pin must check
    against the pre-pin container allowance, not the inherited one (which
    would always intersect to the empty set).
    """
    allowed = set(allowed) if allowed is not None else os.sched_getaffinity(0)
    effective = cpus & allowed
    if not effective:
        raise ValueError(
            f"affinity: none of cpus {sorted(cpus)} are available "
            f"(allowed: {sorted(allowed)})"
        )
    os.sched_setaffinity(0, effective)


def worker_init_fn(cpus: set[int], allowed: set[int] | None = None):
    """DataLoader worker_init_fn pinning every worker to the workers mask.

    Pass the main process's pre-pin allowance as allowed; without it every
    worker inherits the main mask and the workers-mask intersection is empty.
    """

    def _init(worker_id: int) -> None:
        apply_mask(cpus, allowed=allowed)

    return _init
