"""Weighted multi-source token mixing (Phase 2 P2-D2).

MixtureDataset wraps one ShardedTokenDataset per source and yields
(window, source_tag) pairs, sampling the source per draw from the configured
weights. Weights are normalized proportions of tokens drawn, not of files.

Determinism and resume: the draw sequence is a pure function of the seed —
draw i always selects the same source and the same window within it — so the
only resume state is the flat count of consumed draws (dataloader.json format
"mixture", saved by the loop). Random access is served from a lazily grown
cache of the source sequence: every instance regenerates the identical
sequence in fixed-size chunks from one seeded generator, so DataLoader worker
processes stay consistent with the main process and with each other.

Within a source, reading is sequential windows wrapping modulo the source
(the D9 convention; no shuffling), with the epoch implied by
draw_count // len(source). strict_single_epoch aborts on wrap instead, for
runs where repeating domain data would confound a measurement.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from training.data.shards import ShardedTokenDataset, TokenizerMismatchError

# Draws generated per cache-extension step; sizes the amortized cost of
# random access (1 byte source id + 8 byte position per draw).
_CHUNK = 65_536


class SourceExhaustedError(RuntimeError):
    """A source wrapped past one epoch under strict_single_epoch."""


class MixtureDataset:
    """Map-style dataset over draw indices mixing several shard directories.

    sources is a sequence of (name, source, weight) triples where source is
    either a shard-directory path or a pre-built dataset satisfying the
    ShardedTokenDataset duck-type (__len__, __getitem__, manifest with
    tokenizer_hash, window) — the P2-D4 ReplayBuffer enters this way. All
    sources must share one tokenizer hash (refused here and, earlier, at
    recipe validate()). __getitem__(i) returns (window ndarray, source index).
    """

    def __init__(self, sources: Sequence[tuple[str, object, float]], window: int,
                 seed: int, strict_single_epoch: bool = False,
                 verify: bool = True) -> None:
        if not sources:
            raise ValueError("mixture requires at least one source")
        self.names = [name for name, _, _ in sources]
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"mixture source names must be unique: {self.names}")
        weights = np.array([w for _, _, w in sources], dtype=np.float64)
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError(f"mixture weights must be positive and finite: {weights.tolist()}")
        self.weights = weights / weights.sum()
        self.window = window
        self.seed = seed
        self.strict_single_epoch = strict_single_epoch

        self.datasets = []
        for name, src, _ in sources:
            if isinstance(src, str):
                self.datasets.append(ShardedTokenDataset(src, window, verify=verify))
                continue
            if src.window != window:
                raise ValueError(
                    f"mixture source {name!r} was built with window {src.window}, "
                    f"mixture uses {window}")
            self.datasets.append(src)
        hashes = {ds.manifest["tokenizer_hash"] for ds in self.datasets}
        if len(hashes) != 1:
            raise TokenizerMismatchError(
                f"mixture sources carry {len(hashes)} distinct tokenizer hashes: "
                f"{sorted(hashes)}")
        self.tokenizer_hash = hashes.pop()
        for name, ds in zip(self.names, self.datasets, strict=True):
            if len(ds) == 0:
                raise ValueError(
                    f"mixture source {name!r} has no complete window of size {window}")

        self._rng = np.random.default_rng(seed)
        self._source_seq = np.empty(0, dtype=np.uint8)
        self._pos_seq = np.empty(0, dtype=np.int64)
        self._counts = np.zeros(len(self.datasets), dtype=np.int64)

    def __len__(self) -> int:
        """Total windows across sources (one epoch of everything). The draw
        index is unbounded; this length is informational and never wraps it."""
        return sum(len(ds) for ds in self.datasets)

    def _ensure(self, index: int) -> None:
        """Extend the cached draw sequence to cover index. Chunked sequential
        generation from one stream keeps the sequence identical across
        instances regardless of access order."""
        while len(self._source_seq) <= index:
            draws = self._rng.choice(
                len(self.datasets), size=_CHUNK, p=self.weights).astype(np.uint8)
            pos = np.empty(_CHUNK, dtype=np.int64)
            for s in range(len(self.datasets)):
                mask = draws == s
                n = int(mask.sum())
                pos[mask] = self._counts[s] + np.arange(n, dtype=np.int64)
                self._counts[s] += n
            self._source_seq = np.concatenate([self._source_seq, draws])
            self._pos_seq = np.concatenate([self._pos_seq, pos])

    def source_at(self, index: int) -> int:
        """Source index chosen at a draw index."""
        self._ensure(index)
        return int(self._source_seq[index])

    def epoch_of(self, index: int) -> int:
        """Epoch of the chosen source at a draw index (0 on first pass)."""
        self._ensure(index)
        source = int(self._source_seq[index])
        return int(self._pos_seq[index]) // len(self.datasets[source])

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        if index < 0:
            raise IndexError(index)
        self._ensure(index)
        source = int(self._source_seq[index])
        position = int(self._pos_seq[index])
        ds = self.datasets[source]
        if self.strict_single_epoch and position >= len(ds):
            raise SourceExhaustedError(
                f"source {self.names[source]!r} exhausted after {len(ds)} windows "
                f"at draw {index} (strict_single_epoch)")
        return ds[position % len(ds)], source
