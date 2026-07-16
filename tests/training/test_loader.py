"""D9 prefetch dataloader: exact sequential window order, wrap-around,
index-based resume equivalence with 0 and >0 workers, legacy cursor
conversion, and the multi-worker deterministic loop resume — the guarantee
the index-based design exists to protect."""

import glob
import importlib.util
import os

import numpy as np
import pytest
from support import make_shard_dir

from training.data.loader import (
    CudaPrefetcher,
    build_dataloader,
    resume_index,
)
from training.data.shards import ShardedTokenDataset

# resume_index is pure index arithmetic; only the DataLoader/prefetcher tests
# need torch, so the gate is per-test rather than module-level
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch required for DataLoader and prefetcher tests",
)

VOCAB = 512
WINDOW = 64


def data_cfg(micro_batch=2, workers=0, prefetch_factor=2, pin_memory=False):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "micro_batch": micro_batch,
            "workers": workers,
            "prefetch_factor": prefetch_factor,
            "pin_memory": pin_memory,
        }
    )


@pytest.fixture()
def shard_dir(tmp_path) -> str:
    rng = np.random.default_rng(11)
    tokens = rng.integers(0, VOCAB, size=6000).tolist()
    return make_shard_dir(tmp_path, tokens, shard_tokens=2000)


def ground_truth(dataset: ShardedTokenDataset, start: int, count: int):
    n = len(dataset)
    return [dataset[(start + i) % n].astype(np.int64) for i in range(count)]


def take_windows(loader, batches: int, micro_batch: int):
    import torch

    out = []
    it = iter(loader)
    for _ in range(batches):
        batch = next(it)
        assert batch.dtype == torch.int64
        out.extend(batch[i].numpy() for i in range(micro_batch))
    return out


@requires_torch
@pytest.mark.parametrize("workers", [0, 2])
def test_sequential_order_matches_dataset(shard_dir, workers):
    dataset = ShardedTokenDataset(shard_dir, window=WINDOW)
    loader = build_dataloader(dataset, data_cfg(workers=workers), start_index=0)
    got = take_windows(loader, batches=5, micro_batch=2)
    expected = ground_truth(dataset, start=0, count=10)
    for g, e in zip(got, expected, strict=True):
        np.testing.assert_array_equal(g, e)


@requires_torch
def test_wraps_at_corpus_end(shard_dir):
    dataset = ShardedTokenDataset(shard_dir, window=WINDOW)
    n = len(dataset)
    loader = build_dataloader(dataset, data_cfg(), start_index=n - 1)
    got = take_windows(loader, batches=2, micro_batch=2)
    expected = ground_truth(dataset, start=n - 1, count=4)  # wraps to 0, 1, 2
    for g, e in zip(got, expected, strict=True):
        np.testing.assert_array_equal(g, e)


@requires_torch
@pytest.mark.parametrize("workers", [0, 2])
def test_resume_from_index_is_exact(shard_dir, workers):
    dataset = ShardedTokenDataset(shard_dir, window=WINDOW)
    straight = take_windows(
        build_dataloader(dataset, data_cfg(workers=workers), 0), 6, 2
    )
    head = take_windows(build_dataloader(dataset, data_cfg(workers=workers), 0), 3, 2)
    tail = take_windows(build_dataloader(dataset, data_cfg(workers=workers), 6), 3, 2)
    for g, e in zip(head + tail, straight, strict=True):
        np.testing.assert_array_equal(g, e)


def test_resume_index_formats(shard_dir):
    dataset = ShardedTokenDataset(shard_dir, window=WINDOW)
    assert resume_index(dataset, None) == 0
    assert (
        resume_index(dataset, {"format": "index", "next_index": 7, "window": WINDOW})
        == 7
    )
    # index format wraps modulo the corpus
    n = len(dataset)
    assert (
        resume_index(
            dataset, {"format": "index", "next_index": n + 3, "window": WINDOW}
        )
        == 3
    )
    # legacy pre-D9 cursor: shard 1 begins after shard 0's windows
    per_shard0 = 2000 // WINDOW
    assert (
        resume_index(dataset, {"shard": 1, "offset": 4, "window": WINDOW})
        == per_shard0 + 4
    )


def test_resume_index_rejects_window_mismatch(shard_dir):
    dataset = ShardedTokenDataset(shard_dir, window=WINDOW)
    with pytest.raises(ValueError, match="window"):
        resume_index(dataset, {"format": "index", "next_index": 0, "window": 128})


@requires_torch
def test_prefetcher_cpu_passthrough_measures_wait(shard_dir):
    import torch

    dataset = ShardedTokenDataset(shard_dir, window=WINDOW)
    loader = build_dataloader(dataset, data_cfg(), start_index=0)
    prefetcher = CudaPrefetcher(loader, torch.device("cpu"))
    try:
        first = prefetcher.next()
        assert first.shape == (2, WINDOW)
        assert prefetcher.last_wait_s >= 0.0
        second = prefetcher.next()
        np.testing.assert_array_equal(second[0].numpy(), dataset[2].astype(np.int64))
    finally:
        prefetcher.close()


@requires_torch
def test_multiworker_deterministic_loop_resume(shard_dir, tmp_path):
    """The D9 acceptance guarantee: with num_workers > 0 and prefetch on, an
    interrupted-and-resumed run reproduces the straight run's losses exactly."""
    pytest.importorskip("transformers")
    import torch
    from support import TINY_RECIPE

    from training.recipes import load_recipe
    from training.runtime.loop import train

    def recipe(root, *extra):
        return load_recipe(
            TINY_RECIPE,
            [
                f"data.source={shard_dir}",
                f"run.output_root={root}",
                f"data.seq_len={WINDOW}",
                "data.workers=2",
                "data.pin_memory=false",
                "schedule.total_steps=8",
                "schedule.warmup_steps=2",
                "checkpoint.interval_steps=4",
                *extra,
            ],
        )

    torch.use_deterministic_algorithms(True)
    try:
        straight = train(recipe(str(tmp_path / "straight")))
        root = str(tmp_path / "resumable")
        first = train(recipe(root), stop_after=4)
        second = train(recipe(root, "checkpoint.resume=latest"))
    finally:
        torch.use_deterministic_algorithms(False)
    assert first.losses + second.losses == straight.losses
    # dataloader.json carries the index format
    ckpts = sorted(glob.glob(os.path.join(first.checkpoint_root, "step_*")))
    import json

    with open(os.path.join(ckpts[0], "dataloader.json")) as f:
        state = json.load(f)
    assert state["format"] == "index"
    assert state["next_index"] == 4 * 2  # 4 steps x micro_batch 2 windows consumed
