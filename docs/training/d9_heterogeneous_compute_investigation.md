# D9 heterogeneous-compute skeleton: implementation investigation

Comparison of the Phase 0 D9 design (roadmap document, "D9: Heterogeneous-compute
skeleton") against the current repository state, with a work breakdown, acceptance
checks, deviations, and open questions. Per operating rules, the repository is the
source of truth; where the roadmap document disagrees with the repository, this
investigation records the repository state and treats the document as advisory.

## Table of contents

- [Design summary](#design-summary)
- [Current repository state](#current-repository-state)
- [Deviations from the document](#deviations-from-the-document)
- [Work breakdown](#work-breakdown)
- [Acceptance checks](#acceptance-checks)
- [Risks](#risks)
- [Resolved decisions](#resolved-decisions)
- [Open questions](#open-questions)

## Design summary

D9 is the whole-machine pillar inside a single training job: keep the GPU step from
waiting on the CPU. It explicitly does not mean coexisting with the inference
services, and it does not involve cloud or multi-node compute. Tag: orchestrate.
Depends on D1 (container, complete), D2 (shards, complete), and D6 (loop, complete).
The document specifies:

1. **Prefetch dataloader** `training/data/loader.py`: a standard
   `torch.utils.data.DataLoader` over `ShardedTokenDataset` with `pin_memory=True`,
   configurable `num_workers` (default 4) and `prefetch_factor`, wrapped by a
   `CudaPrefetcher` that keeps one batch ahead on a dedicated CUDA stream with
   `non_blocking=True` copies. Adopted primitives, thin glue; no custom dataloader
   machinery beyond the wrapper.
2. **RAM-disk staging** `training/data/staging.py`: optionally copy the hot shard
   set to a tmpfs mount before the run and point the dataset there. Refuses to stage
   beyond `staging.max_gb` (default 32) and falls back to NVMe mmap, which under the
   inference-down policy is usually already RAM-speed via page cache; staging exists
   for controlled experiments and multi-epoch small corpora. Compose adds a tmpfs
   mount `/staging` to the training service sized by the same setting.
3. **Core-pinning wrapper** (`training/affinity.py` plus a pin-run shell script):
   the 9950X is one NUMA node but two CCDs (cores 0-7 and 8-15, separate L3). The
   wrapper pins the main training process to one CCD and dataloader workers plus the
   checkpoint writer to the other, via `taskset`/`os.sched_setaffinity`, driven by a
   recipe block (`affinity: {main: "0-7", workers: "8-15"}`), default off. SMT
   siblings (16-31) follow their physical cores. This is CCD/L3 pinning, not NUMA
   pinning.
4. **AOCL-LibMem toggle**: the container entrypoint honors
   `TRAIN_LD_PRELOAD=/opt/aocl/libmem.so`; the library is absent by default,
   single-digit expected gain, off until profiled.
5. **Run lock** `training/lock.py`: a lock file with pid and run name written while
   a job runs; `serve_eval` and any second job check it. Enforces the one-GPU-consumer
   concurrency policy mechanically.
6. **Proof of overlap**: the loop logs `data_wait_ms` per step (time blocked on the
   next batch). Acceptance uses this in-band metric; nsys is the diagnostic tool.

Acceptance in the document: on the 150M reference workload, mean `data_wait_ms` under
5 percent of step time with the prefetcher on, versus a measurably higher figure with
`pin_memory=False, num_workers=0`; staging respects `max_gb` and falls back cleanly;
with affinity on, worker and main-process CPU masks are verifiably disjoint
(`taskset -pc`) and throughput is not worse than affinity-off (park the feature
off-by-default if it does not help); the lock file blocks a second concurrent run and
`serve_eval` during training.

## Current repository state

D9's hooks are pre-wired; none of its modules exist yet.

- `docker/training/entrypoint.sh` (lines 21-27) already implements the
  `TRAIN_LD_PRELOAD` -> `LD_PRELOAD` interposer slot, and
  `docker-compose.training.yaml` (lines 34-35) passes it through, empty by default.
  Decision 4's plumbing is therefore complete; only the optional library install
  remains, and it stays out of scope for this pass (see resolved decisions).
- `scripts/eval/serve_eval.py` (lines 47-48, 140-146) already checks
  `/mnt/ai-data/training/.train.lock` and refuses to swap the served model while it
  exists. The lock path is settled by the repository.
- `training/loop.py` reads data by calling `ShardedTokenDataset.next_window()`
  directly (line 118 and the step loop); its header comment (line 16) reserves the
  D9 DataLoader wrap point. The loop already measures `step_time` per step and logs
  `tokens_per_s` and `grad_norm` through `Tracker`, so `data_wait_ms` is an
  incremental metric, not new machinery.
- `training/data/shards.py`: `ShardedTokenDataset` is already index-addressable
  (`__len__`, `__getitem__(index)`) in addition to the cursor API
  (`next_window`, `state_dict`/`load_state_dict`). The map-style interface the
  DataLoader needs exists today.
- `training/recipes/schema.py` merges recipes in struct mode: unknown keys fail at
  load. New `loader`, `staging`, and `affinity` blocks must be added to the schema
  (and invariants to `training/recipes/validate.py`) before any recipe can carry
  them. The schema already ships not-yet-consumed blocks (`watchdog`, fp8 fields)
  as the established pattern for this.
- Training-related scripts live in `scripts/training/`; the tests mirror source in
  `tests/training/` and run under `make test`; Makefile targets carry
  `(Phase 0 Dn)` tags in their `##` help strings.
- D8 (benchmark harness) is not implemented, but D9 does not depend on it.

## Deviations from the document

| Document | Repository (authoritative) |
|----------|---------------------------|
| Lock path `/mnt/ai-data/training/.lock` | `/mnt/ai-data/training/.train.lock`, already hardcoded in `scripts/eval/serve_eval.py:48` |
| Pin script at `scripts/train/pin_run.sh` | Training scripts live in `scripts/training/`; the script goes there |
| Style guides at `.claude/*.md` | Style guides live in `.styles/` |
| Silent on recipe schema changes | Struct-mode schema requires explicit `loader`/`staging`/`affinity` dataclasses in `training/recipes/schema.py` plus `validate.py` rules |
| DataLoader "over ShardedTokenDataset" with the cursor left implicit | The cursor is the authoritative resume state (D6); multi-worker prefetch decouples cursor position from consumed batches, so resume moves to a consumed-window index (see resolved decisions) |

## Work breakdown

1. **Recipe schema extension** (`training/recipes/schema.py`, `validate.py`).
   `LoaderConfig` (`num_workers: int = 4`, `prefetch_factor: int = 2`,
   `pin_memory: bool = True`), `StagingConfig` (`enabled: bool = False`,
   `max_gb: float = 32.0`, `dir: str = "/staging"`), `AffinityConfig`
   (`enabled: bool = False`, `main: str = "0-7"`, `workers: str = "8-15"`).
   Validation: affinity masks parse and are disjoint; `prefetch_factor` requires
   `num_workers > 0`; `staging.max_gb > 0`.
2. **Run lock** (`training/lock.py`). Atomic create (`O_CREAT | O_EXCL`) of
   `/mnt/ai-data/training/.train.lock` containing pid, run name, and timestamp as
   JSON; context-manager acquire/release in `run_training`; stale-lock detection by
   pid liveness with an explicit `--force`-style override left to the operator
   (delete the file manually; the error message says so). `serve_eval.py` keeps its
   existing check unchanged.
3. **Index-based resume rework** (`training/loop.py`, `training/checkpoint.py`
   consumers). The checkpoint's dataloader state becomes the index of the last
   consumed window (plus epoch/wrap count); the loop iterates a sequential sampler
   starting at that index instead of relying on the dataset mutating its own cursor.
   `next_window()` and the cursor API remain for existing tests and callers; the
   loop stops using them. Resume acceptance (identical loss trajectory in
   deterministic mode) must still pass with `num_workers: 0` and `num_workers: 4`.
4. **Prefetch dataloader** (`training/data/loader.py`). `build_dataloader(recipe,
   dataset, start_index)` returning a `DataLoader` with the schema settings, plus
   `CudaPrefetcher` (dedicated stream, one batch ahead, `non_blocking=True`, exposes
   per-batch wait time). CPU-only fallback (no CUDA available) passes batches
   through unprefetched so unit tests run in the sandbox.
5. **data_wait_ms metric** (`training/loop.py`, `training/tracking.py` stream). Time
   blocked on the next batch per step, logged alongside `tokens_per_s`; the run
   summary records the mean for the acceptance comparison.
6. **RAM-disk staging** (`training/data/staging.py`). Manifest-driven size check
   against `max_gb`, copy with hash verification into `staging.dir`, return the
   effective shard directory (staged dir or original on fallback), `[SKIP]`-style
   message on fallback per the output style guides. Compose: tmpfs mount `/staging`
   on the training service, size from `TRAIN_STAGING_GB` (default 32g).
7. **Affinity** (`training/affinity.py`, `scripts/training/pin_run.sh`).
   `os.sched_setaffinity` applied to the main process and via DataLoader
   `worker_init_fn` to workers, from the recipe block; the shell script is the
   host-side `taskset` equivalent for wrapping arbitrary commands. Comments state
   CCD/L3 pinning, not NUMA pinning.
8. **Tests** (`tests/training/`). Lock: acquire/contend/release, stale pid, and the
   `serve_eval` path against a temp lock file. Loader: exact batch sequence versus
   `next_window()` ground truth, resume-from-index equivalence with 0 and >0
   workers. Staging: cap respected, fallback path, hash verification. Affinity:
   mask parsing, disjointness validation, `sched_getaffinity` round trip. All
   CPU-only, sandbox-runnable.
9. **Makefile and docs**. `##` help text tagged `(Phase 0 D9)` where targets are
   added or changed; a companion reference doc after implementation, following the
   D3-D7 pattern.

Suggested order: 1, 2 (independent), then 3, 4, 5 (the coupled core), then 6, 7,
then 8 throughout and 9 last.

## Acceptance checks

Host-side, run by the human; the sandbox cannot reach the GPU or `/mnt/ai-data`.

1. Unit tests (sandbox-runnable): `make test`.
2. Overlap: run the 150M reference recipe twice, once with the prefetcher on and
   once with `loader: {num_workers: 0, pin_memory: false}`; compare mean
   `data_wait_ms` from the run summaries. Pass: prefetcher-on mean under 5 percent
   of mean step time and lower than the baseline.
3. Staging: run with `staging.enabled: true` on a shard set under the cap and one
   over it; verify staged run reads from `/staging` and the oversized run logs the
   fallback and completes from NVMe.
4. Affinity: with `affinity.enabled: true`, `taskset -pc <main-pid>` and
   `taskset -pc <worker-pid>` show disjoint masks; tokens/s not worse than
   affinity-off on the reference workload.
5. Lock: start a training run, then attempt a second run and
   `scripts/eval/serve_eval.py`; both must refuse with the lock message; after the
   run exits the lock file is gone.

## Risks

- **Resume semantics under multi-worker prefetch** is the only genuinely subtle
  piece: off-by-one between "fetched" and "consumed" windows silently breaks the D6
  identical-loss-trajectory guarantee. Mitigated by the index-based design and the
  0-worker/4-worker equivalence test.
- **Worker memory duplication**: each worker mmaps the shard files; memory-mapped
  pages are shared, so this is cheap, but worker startup cost with `num_workers: 4`
  on tiny e2e runs may exceed the benefit. Default stays 4 per the document; tiny
  reference recipes may set 0.
- **Affinity may not help** on a workload this small; the document pre-authorizes
  parking it off-by-default if throughput is not improved. The deliverable is the
  measured wrapper, not a win.
- **tmpfs mount interaction with `read_only: true`** rootfs: `/staging` must be an
  explicit tmpfs entry in the compose service, following the existing `/tmp` and
  `/app/cache` pattern.

## Resolved decisions

1. **Lock path is `.train.lock`** (repository over document); `training/lock.py`
   writes the path `serve_eval.py` already checks.
2. **Index-based resume**: rework the loop to a consumed-window index rather than
   restricting prefetch to `num_workers: 0` or accepting approximate resume.
3. **All five pieces land in this pass** (loader, staging, affinity, LD_PRELOAD
   toggle already present, lock, metric), with the AOCL-LibMem library itself not
   installed in the image: the hook exists, the library remains absent by default
   per the document, and profiling it is future work.
4. **Investigation doc first**: implementation follows review of this document.

## Open questions

1. Should the lock live in `training/lock.py` as a context manager acquired inside
   `run_training` (proposed), or in the CLI wrapper so non-loop jobs (export, future
   TRL runs) reuse it without importing the loop? Proposal: module in
   `training/lock.py`, acquired by `run_training`, importable by any future job.
2. Stale-lock policy: pid-liveness check with manual delete on staleness (proposed)
   versus automatic removal of locks whose pid is dead. Automatic removal is
   convenient but races two simultaneous starters; proposal keeps manual delete.
3. The 150M reference recipe (`pretrain-150m.yaml`) predates the new schema blocks;
   it should gain explicit `loader` defaults as part of this work so the acceptance
   run is reproducible from the recipe alone.
