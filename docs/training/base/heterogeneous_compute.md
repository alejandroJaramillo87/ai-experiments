# Heterogeneous-compute skeleton

Phase 0 D9: keep the GPU training step from waiting on the CPU within a single
training job. This document describes the implemented pieces, their recipe
knobs, and the host-side acceptance checks. Design rationale and
repository-versus-roadmap deviations are recorded in
[d9_heterogeneous_compute_investigation.md](../investigations/d9_heterogeneous_compute_investigation.md).

## Table of Contents

<!-- toc:start -->

1. [Pieces](#pieces)
2. [Recipe configuration](#recipe-configuration)
3. [Resume model](#resume-model)
4. [Run lock](#run-lock)
5. [Overlap metric](#overlap-metric)
6. [Acceptance checks](#acceptance-checks)

<!-- toc:end -->

## Pieces

| Piece | Module | Default |
|-------|--------|---------|
| Prefetch dataloader | `/training/data/loader.py` | on (`data.workers: 2`, `data.pin_memory: true`) |
| RAM-disk shard staging | `/training/data/staging.py` | off (`staging.enabled: false`) |
| CCD core pinning | `/training/runtime/affinity.py`, `/scripts/training/pin_run.sh` | off (`affinity.enabled: false`) |
| LD_PRELOAD interposer | `/docker/training/entrypoint.sh` (`TRAIN_LD_PRELOAD`) | empty |
| Run lock | `/training/runtime/lock.py` | always on during `train()` |

The dataloader is a standard `torch.utils.data.DataLoader` over
`ShardedTokenDataset` wrapped by a `CudaPrefetcher` that keeps one batch ahead
on a dedicated CUDA stream with non-blocking host-to-device copies. Without
CUDA the prefetcher degrades to a timed passthrough, so unit tests run on CPU.

Staging copies the shard set to the tmpfs mount `/staging` (added to the
training service in `docker-compose.training.yaml`, sized by
`TRAIN_STAGING_GB`, default 32G) when it fits under `staging.max_gb`, and
falls back to NVMe mmap otherwise. With 128GB of page cache and inference
down, NVMe reads are usually already RAM-speed; staging exists for controlled
experiments and multi-epoch small corpora.

Core pinning is CCD/L3 pinning, not NUMA pinning: the 9950X is one NUMA node
with two CCDs (physical cores 0-7 and 8-15, SMT siblings 16-31). The main
training process is pinned to one CCD and dataloader workers to the other.
`scripts/training/pin_run.sh <cpu-list> <command>` is the host-side `taskset`
wrapper for jobs that cannot host `training/runtime/affinity.py`.

## Recipe configuration

```yaml
data:
  workers: 4          # dataloader worker processes
  pin_memory: true    # page-locked host buffers for async H2D copies
  prefetch_factor: 2  # batches queued per worker

staging:
  enabled: false
  dir: /staging       # tmpfs mount in docker-compose.training.yaml
  max_gb: 32.0        # refuse to stage beyond this; fall back to NVMe

affinity:
  enabled: false
  main: "0-7"         # cpu list for the main training process
  workers: "8-15"     # cpu list for dataloader workers
  include_smt: true   # extend each mask with its SMT siblings
```

Validation (`training/recipes/validate.py`) rejects overlapping affinity
masks, malformed cpu lists, `prefetch_factor < 1`, and a non-positive
`staging.max_gb` at recipe load.

## Resume model

The authoritative resume state in `dataloader.json` is the flat index of the
next window to consume:

```json
{"format": "index", "next_index": 8, "window": 128}
```

The loop counts windows as batches are trained on, never by the loader's
read-ahead, so multi-worker prefetch cannot desynchronize resume state.
Pre-D9 cursor checkpoints (`{"shard", "offset", "window"}`) convert on load
via `ShardedTokenDataset.cursor_index`. The deterministic-resume acceptance
test passes with `workers: 0` and `workers: 2`
(`tests/training/test_loader.py`).

## Run lock

`train()` holds `/mnt/ai-data/training/.train.lock` (JSON: pid, run name,
creation time) for its whole run. A second job, or a host serving tool that
checks it (`training/serving.py::ensure_training_idle`), refuses to start
while the lock exists. A stale
lock (holder pid not alive) is reported with removal instructions and is never
auto-removed: automatic removal races two simultaneous starters.
`TRAIN_LOCK_PATH` overrides the path for tests.

## Overlap metric

The loop logs `data_wait_ms` per step: the time it was blocked before the next
batch was usable (worker-queue wait plus CUDA stream synchronization). The run
summary carries the mean (`TrainResult.data_wait_ms_mean`). nsys remains the
diagnostic tool; acceptance uses the in-band metric.

## Acceptance checks

Host-side, on the workstation:

```bash
# Unit suite (also runs in the sandbox). Beside the loader, staging, affinity
# and lock tests it drives pin_run.sh and the entrypoint's TRAIN_LD_PRELOAD
# slot as subprocesses (tests/training/test_pin_run.py, test_entrypoint.py).
make test

# Overlap: prefetcher on (recipe defaults) versus off, compare mean data_wait_ms
make train CONFIG=training/recipes/reference/pretrain-150m.yaml
make train CONFIG=training/recipes/reference/pretrain-150m.yaml \
    ARGS="data.workers=0 data.pin_memory=false" 

# Affinity: disjoint masks visible from the host while a pinned run is active
taskset -pc <main-pid>
taskset -pc <worker-pid>

# Lock: a second run and a host serving tool must refuse while a run is active
make train CONFIG=...   # while another run holds the lock
```

Pass criteria: mean `data_wait_ms` under 5 percent of mean step time on the
150M reference workload and lower than the workers-off baseline; staging
respects `max_gb` with a logged fallback; affinity masks disjoint and
throughput not worse than affinity-off; the lock blocks the second run and the
serving swap.
