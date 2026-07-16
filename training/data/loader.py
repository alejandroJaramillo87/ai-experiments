"""Pinned-memory prefetch dataloader (Phase 0 D9).

Adopted primitives, thin glue: a standard torch.utils.data.DataLoader over
ShardedTokenDataset (already map-style: __len__/__getitem__) with pin_memory,
num_workers, and prefetch_factor from the recipe's data block, wrapped by a
CudaPrefetcher that keeps one batch ahead on a dedicated CUDA stream with
non_blocking copies. On CPU (unit tests, sandbox) the prefetcher degrades to a
timed passthrough.

Resume model (resolved D9 decision 2): the loop records the flat index of the
next window to consume in dataloader.json ({"format": "index", "next_index":
N, "window": W}); the sampler here replays from that index, wrapping modulo
the corpus (sequential multi-epoch reading, as before). Multi-worker prefetch
therefore never desynchronizes resume state — the index counts consumed
batches, not fetched ones. Pre-D9 cursor checkpoints ({"shard", "offset",
"window"}) convert via ShardedTokenDataset.cursor_index.

data_wait_ms: CudaPrefetcher.next() measures the time the training loop is
blocked before the batch is usable (worker-queue wait plus H2D stream sync).
The loop sums it per step — the in-band overlap metric D9's acceptance uses.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np

from training.data.mixture import MixtureDataset
from training.data.shards import ShardedTokenDataset


def resume_index(dataset, state: dict | None) -> int:
    """dataloader.json contents -> flat window (or draw) index to resume from.

    Accepts the D9 index format, the pre-D9 cursor format, the P2-D2 mixture
    format ({"format": "mixture", "next_draw": N, "window": W}), and None
    (fresh start). The state format must match the dataset type: a mixture
    checkpoint cannot resume a single-source run or vice versa. Validates the
    window size either way: a checkpoint written with a different seq_len
    cannot resume.
    """
    if state is None:
        return 0
    if state["window"] != dataset.window:
        raise ValueError(
            f"checkpoint was saved with window {state['window']}, "
            f"dataset uses {dataset.window}")
    mixing = isinstance(dataset, MixtureDataset)
    if state.get("format") == "mixture":
        if not mixing:
            raise ValueError(
                "checkpoint was saved from a mixture run; this recipe trains "
                "on a single source")
        return state["next_draw"]  # draw index: unbounded, never wraps
    if mixing:
        raise ValueError(
            "checkpoint was saved from a single-source run; this recipe "
            "trains on a mixture")
    if state.get("format") == "index":
        return state["next_index"] % len(dataset)
    return dataset.cursor_index(state["shard"], state["offset"]) % len(dataset)


def _wrapping_indices(num_windows: int, start_index: int) -> Iterator[int]:
    i = start_index
    while True:
        yield i % num_windows
        i += 1


def _counting_indices(start_index: int) -> Iterator[int]:
    # Mixture draw indices never wrap: the draw sequence is unbounded and the
    # per-source wrap happens inside MixtureDataset.
    i = start_index
    while True:
        yield i
        i += 1


class _WindowTensorDataset:
    """int64 tensor view over ShardedTokenDataset windows (torch map-style)."""

    def __init__(self, shards: ShardedTokenDataset) -> None:
        self.shards = shards

    def __len__(self) -> int:
        return len(self.shards)

    def __getitem__(self, index: int):
        import torch

        return torch.from_numpy(self.shards[index].astype(np.int64))


class _TaggedWindowTensorDataset:
    """(int64 window tensor, source tag) view over MixtureDataset draws.

    The default collate stacks the pairs into ((micro_batch, seq_len) tokens,
    (micro_batch,) tags); tags stay on CPU end to end — only accounting and
    the P2-D4 replay buffer read them. __len__ satisfies the DataLoader's
    map-style contract only; the sampler never wraps on it."""

    def __init__(self, mixture: MixtureDataset) -> None:
        self.mixture = mixture

    def __len__(self) -> int:
        return len(self.mixture)

    def __getitem__(self, index: int):
        import torch

        window, source = self.mixture[index]
        return torch.from_numpy(window.astype(np.int64)), source


def build_dataloader(dataset, data_cfg, start_index: int = 0,
                     worker_init_fn=None):
    """DataLoader over a ShardedTokenDataset or MixtureDataset, forever.

    Single source: (micro_batch, seq_len) int64 batches, sequential windows
    from start_index wrapping at corpus end. Mixture: (tokens, tags) batch
    pairs from an unbounded draw index. The caller counts consumed windows;
    the loader holds no resume state of its own."""
    from torch.utils.data import DataLoader

    torch_dataset: _TaggedWindowTensorDataset | _WindowTensorDataset
    if isinstance(dataset, MixtureDataset):
        torch_dataset = _TaggedWindowTensorDataset(dataset)
        sampler = _counting_indices(start_index)
    else:
        torch_dataset = _WindowTensorDataset(dataset)
        sampler = _wrapping_indices(len(dataset), start_index)
    kwargs = {}
    if data_cfg.workers > 0:
        kwargs["prefetch_factor"] = data_cfg.prefetch_factor
        kwargs["persistent_workers"] = True
    return DataLoader(
        torch_dataset,
        batch_size=data_cfg.micro_batch,
        sampler=sampler,
        num_workers=data_cfg.workers,
        pin_memory=data_cfg.pin_memory,
        worker_init_fn=worker_init_fn,
        **kwargs)


class CudaPrefetcher:
    """Keeps one batch ahead on a dedicated CUDA stream.

    next() returns the current device batch and immediately starts the H2D
    copy of the following one, so the copy overlaps the step's compute.
    last_wait_s is the time next() spent blocked (queue wait + stream sync) —
    zero-ish when the pipeline keeps up. Without CUDA it is a timed
    passthrough: batches stay on CPU and last_wait_s is the fetch time.
    """

    def __init__(self, loader, device) -> None:
        import torch

        self._it = iter(loader)
        self.device = device
        self.use_cuda = device.type == "cuda"
        self.stream = torch.cuda.Stream(device) if self.use_cuda else None
        self.last_wait_s = 0.0
        self._next = None
        self._preload()

    def _preload(self) -> None:
        import torch

        batch = next(self._it)  # infinite sampler: never StopIteration
        if self.use_cuda:
            with torch.cuda.stream(self.stream):
                # Mixture batches are (tokens, tags); only the tokens move —
                # tags are read by CPU-side accounting.
                if isinstance(batch, (tuple, list)):
                    tokens, *rest = batch
                    batch = (tokens.to(self.device, non_blocking=True), *rest)
                else:
                    batch = batch.to(self.device, non_blocking=True)
        self._next = batch

    def next(self):
        import torch

        start = time.perf_counter()
        if self.use_cuda:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self._next
        if self.use_cuda:
            tokens = batch[0] if isinstance(batch, (tuple, list)) else batch
            tokens.record_stream(torch.cuda.current_stream(self.device))
        self._preload()
        self.last_wait_s = time.perf_counter() - start
        return batch

    def close(self) -> None:
        """Drop the loader iterator so persistent workers shut down."""
        self._it = None
        self._next = None
