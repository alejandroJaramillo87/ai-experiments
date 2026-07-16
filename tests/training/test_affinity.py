"""D9 core-pinning helpers: cpu-list parsing, SMT expansion, disjointness,
and the sched_setaffinity round trip."""

import os

import pytest
from omegaconf import OmegaConf

from training.affinity import (
    apply_mask,
    parse_cpu_list,
    resolve_masks,
    with_smt_siblings,
    worker_init_fn,
)


def cfg(main="0-7", workers="8-15", include_smt=True):
    return OmegaConf.create(
        {"enabled": True, "main": main, "workers": workers, "include_smt": include_smt}
    )


def test_parse_ranges_and_singles():
    assert parse_cpu_list("0-3") == {0, 1, 2, 3}
    assert parse_cpu_list("0,2,4") == {0, 2, 4}
    assert parse_cpu_list("0-1,8-9") == {0, 1, 8, 9}


@pytest.mark.parametrize("bad", ["", "7-0", "a-b", "-3"])
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_cpu_list(bad)


def test_smt_siblings_offset_by_half():
    assert with_smt_siblings({0, 7}, logical_cpus=32) == {0, 7, 16, 23}
    # ids at or above half pass through unchanged
    assert with_smt_siblings({20}, logical_cpus=32) == {20}


def test_resolve_masks_disjoint(monkeypatch):
    # Pin the topology: with the real os.cpu_count() this test fails on any
    # host with 10-31 logical CPUs (SMT folding maps workers onto main).
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    main, workers = resolve_masks(cfg())
    assert not main & workers


def test_resolve_masks_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        resolve_masks(cfg(main="0-8", workers="8-15", include_smt=False))


def test_apply_mask_round_trip():
    original = os.sched_getaffinity(0)
    try:
        target = {sorted(original)[0]}
        apply_mask(target)
        assert os.sched_getaffinity(0) == target
    finally:
        os.sched_setaffinity(0, original)


def test_apply_mask_intersects_with_allowed():
    original = os.sched_getaffinity(0)
    try:
        cpu = sorted(original)[0]
        # 4096 is beyond any real allowance; the available cpu must survive
        apply_mask({cpu, 4096})
        assert os.sched_getaffinity(0) == {cpu}
    finally:
        os.sched_setaffinity(0, original)


def test_apply_mask_with_no_available_cpu_raises():
    with pytest.raises(ValueError, match="none of cpus"):
        apply_mask({4096})


def test_worker_init_fn_pins(monkeypatch):
    applied = []
    monkeypatch.setattr("training.affinity.apply_mask", applied.append)
    worker_init_fn({1, 2})(worker_id=0)
    assert applied == [{1, 2}]
