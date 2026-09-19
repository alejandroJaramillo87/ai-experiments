"""RAM-resident replay buffer (Phase 2 P2-D4).

The anti-forgetting mechanism for continual pre-training: a configurable
fraction of every effective batch comes from a proxy of the base model's
training distribution. ReplayBuffer loads a bounded token budget from a
prepared shard directory (P2-D1 output) into a host array at construction and
serves fixed windows from RAM thereafter, so replay never competes with the
domain stream for NVMe bandwidth.

Replay is a mixture source, not a parallel code path: ReplayBuffer satisfies
the ShardedTokenDataset duck-type MixtureDataset consumes (__len__,
__getitem__, manifest with tokenizer_hash, window), so sampling, source-tag
accounting, and the single-draw-index resume story are all inherited from
P2-D2 unchanged.

Within-source uniformity: P2-D2 positions advance sequentially per source, so
the buffer maps position to window through a fixed seeded permutation —
uniform-without-replacement order per epoch, still a pure function of
(seed, position), no extra resume state.

Budget and residency: if the shard set exceeds max_bytes, a uniform window
subsample is taken with a seeded RNG and the loaded window set is recorded in
the run directory (replay_manifest_<name>.json plus a .npy index file) so a
resume or audit can reconstruct exactly what was resident. Before allocating,
the buffer checks psutil available RAM and refuses to start if loading would
cross the D10 RAM floor; the watchdog remains the runtime backstop. Machines
without psutil skip the check (fail-soft, same policy as the watchdog).
"""

from __future__ import annotations

import json
import os

import numpy as np

from training.data.shards import load_manifest, verify_shards

GIB = 1 << 30


class ReplayBudgetError(ValueError):
    """max_bytes cannot hold even one window of the replay shard set."""


class ReplayRamFloorError(RuntimeError):
    """Loading the buffer would push available RAM below the watchdog floor."""


def _available_ram_bytes() -> int | None:
    """psutil available RAM, or None when psutil is unusable (fail-soft).

    Catches Exception, not just ImportError, to match watchdog.sample_host: a
    psutil that imports but fails at call time must degrade the RAM-floor check
    to "unknown", never crash the buffer.
    """
    try:
        import psutil

        return psutil.virtual_memory().available
    except Exception:
        return None


def resolve_max_bytes(recipe) -> int:
    """Total replay byte budget for a recipe (resolved P2-D4 decision 2).

    An explicit replay.max_gb is authoritative. When unset, the budget is the
    D4 calculator's remaining host headroom: 128GB minus offloaded optimizer
    state minus the watchdog RAM floor.
    """
    if recipe.replay.max_gb is not None:
        return int(recipe.replay.max_gb * GIB)
    from training.calculator.components import host_ram_bytes, spec_from_recipe
    from training.calculator.estimate import HOST_RAM_TOTAL_BYTES

    floor = int(recipe.watchdog.ram_floor_gb * GIB)
    headroom = (
        HOST_RAM_TOTAL_BYTES - host_ram_bytes(spec_from_recipe(recipe), recipe) - floor
    )
    if headroom <= 0:
        raise ReplayBudgetError(
            f"no host-RAM headroom for replay: offloaded optimizer state plus "
            f"the {recipe.watchdog.ram_floor_gb}GB watchdog floor consume the "
            f"full host budget; set replay.max_gb explicitly or shrink the run"
        )
    return headroom


class ReplayBuffer:
    """RAM-resident window store over one prepared shard directory.

    Duck-type of ShardedTokenDataset for MixtureDataset consumption: __len__
    counts loaded windows, __getitem__(i) returns a window ndarray, manifest
    carries the source tokenizer_hash, window is the fixed window size.
    """

    def __init__(
        self,
        shard_dir: str,
        window: int,
        max_bytes: int | None = None,
        seed: int = 0,
        run_dir: str | None = None,
        name: str = "replay",
        ram_floor_gb: float = 0.0,
        verify: bool = True,
    ) -> None:
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        self.shard_dir = shard_dir
        self.window = window
        self.name = name
        self.manifest = load_manifest(shard_dir)
        if verify:
            verify_shards(shard_dir, self.manifest)
        dtype = np.dtype(self.manifest["dtype"])

        # Flat window index space, per-shard tails dropped — identical
        # windowing to ShardedTokenDataset so the two views never disagree.
        windows_per_shard = [rec["tokens"] // window for rec in self.manifest["shards"]]
        total_windows = sum(windows_per_shard)
        if total_windows == 0:
            raise ValueError(
                f"replay source {shard_dir} has no complete window of size {window}"
            )

        window_bytes = window * dtype.itemsize
        subsample_seed = None
        if max_bytes is not None and total_windows * window_bytes > max_bytes:
            n_keep = int(max_bytes // window_bytes)
            if n_keep == 0:
                raise ReplayBudgetError(
                    f"replay.max_gb budget ({max_bytes} bytes) holds no complete "
                    f"window ({window_bytes} bytes each) of {shard_dir}"
                )
            subsample_seed = seed
            keep = np.random.default_rng([seed, 0]).choice(
                total_windows, size=n_keep, replace=False
            )
            keep.sort()
        else:
            keep = np.arange(total_windows, dtype=np.int64)

        need = len(keep) * window_bytes
        available = _available_ram_bytes()
        floor = int(ram_floor_gb * GIB)
        if available is not None and available - need < floor:
            raise ReplayRamFloorError(
                f"loading {need / GIB:.2f}GB of replay windows from {shard_dir} "
                f"would leave {(available - need) / GIB:.2f}GB available, below "
                f"the {ram_floor_gb}GB watchdog RAM floor; lower replay.max_gb "
                f"or free host memory"
            )

        self._tokens = np.empty((len(keep), window), dtype=dtype)
        self._load_windows(keep, windows_per_shard, dtype)
        self._perm = np.random.default_rng([seed, 1]).permutation(len(keep))
        self._write_run_manifest(
            run_dir, keep, total_windows, subsample_seed, max_bytes
        )

    def _load_windows(
        self, keep: np.ndarray, windows_per_shard: list[int], dtype: np.dtype
    ) -> None:
        """Copy the kept windows shard by shard; each mmap closes on scope
        exit so no file handles outlive the load."""
        out_pos = 0
        base = 0
        for rec, n_windows in zip(
            self.manifest["shards"], windows_per_shard, strict=True
        ):
            in_shard = keep[(keep >= base) & (keep < base + n_windows)] - base
            if len(in_shard):
                mm = np.memmap(
                    os.path.join(self.shard_dir, rec["file"]),
                    dtype=dtype,
                    mode="r",
                    shape=(rec["tokens"],),
                )
                view = mm[: n_windows * self.window].reshape(n_windows, self.window)
                self._tokens[out_pos : out_pos + len(in_shard)] = view[in_shard]
                out_pos += len(in_shard)
            base += n_windows
        assert out_pos == len(keep)

    def _write_run_manifest(
        self,
        run_dir: str | None,
        keep: np.ndarray,
        total_windows: int,
        subsample_seed: int | None,
        max_bytes: int | None,
    ) -> None:
        """Record what was loaded so a resume or audit can reconstruct the
        resident set; the exact window indices go to a sidecar .npy when the
        set is a subsample (the full set is implied by the source manifest)."""
        if run_dir is None:
            return
        record = {
            "source_dir": self.shard_dir,
            "tokenizer_hash": self.manifest["tokenizer_hash"],
            "window": self.window,
            "source_windows": int(total_windows),
            "loaded_windows": int(len(keep)),
            "loaded_bytes": int(self._tokens.nbytes),
            "max_bytes": max_bytes,
            "subsample_seed": subsample_seed,
            "windows_file": None,
        }
        if subsample_seed is not None:
            windows_file = f"replay_windows_{self.name}.npy"
            np.save(os.path.join(run_dir, windows_file), keep)
            record["windows_file"] = windows_file
        path = os.path.join(run_dir, f"replay_manifest_{self.name}.json")
        with open(path, "w") as f:
            json.dump(record, f, indent=2, sort_keys=True)
            f.write("\n")

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, index: int) -> np.ndarray:
        if index < 0 or index >= len(self._tokens):
            raise IndexError(index)
        return np.array(self._tokens[self._perm[index]])
