# Tracking and resumable checkpointing

Phase 0 D6: the reference training loop, atomic resumable checkpoints, and metric
tracking. One recipe YAML fully drives a run; every run is observable and resumable.
Design record:
[the D6 investigation](investigations/d6_tracking_checkpointing_investigation.md).

## Table of contents

- [Running a training job](#running-a-training-job)
- [Directory layout](#directory-layout)
- [Checkpoint contents](#checkpoint-contents)
- [Atomicity](#atomicity)
- [Retention](#retention)
- [Resume](#resume)
- [Tracking](#tracking)
- [Secrets](#secrets)
- [Module map](#module-map)

## Running a training job

Host-side, in the training container:

```bash
$ make train CONFIG=training/recipes/reference/tiny-e2e.yaml
$ make train CONFIG=training/recipes/reference/tiny-e2e.yaml ARGS="checkpoint.resume=latest"
```

The `train` subcommand loads and validates the recipe (D5), then runs
`training/runtime/loop.py`: model from the recipe (HF hub id or scratch spec), AdamW or
AdamW8bit, cosine or WSD schedule with warmup, gradient accumulation, and a
checkpoint every `checkpoint.interval_steps` plus one at the final step. The loop is
the Phase 0 reference skeleton, not the Phase 1 SFT trainer.

Data is read sequentially from D2 shard directories through
`ShardedTokenDataset.next_window()`; the cursor wraps to the start when the corpus
is exhausted. The dataset cursor is the authoritative dataloader resume state.

## Directory layout

```
${run.output_root}/${run.name}/          default root /mnt/ai-data/training/runs
  <timestamp>/                           one per invocation: config.yaml snapshot,
                                         tensorboard/, wandb/, mlruns/
  checkpoints/                           shared across invocations
    step_000000005/
    step_000000010/
    latest -> step_000000010
```

Checkpoints live under the run name, not the timestamped invocation directory, so
`checkpoint.resume=latest` finds them across restarts.

## Checkpoint contents

Each `step_NNNNNNNNN/` directory holds five components:

| File | Contents |
|------|----------|
| `model/` | HF `save_pretrained` sharded safetensors (D7 export consumes this directly) |
| `optimizer.pt` | `torch.save` optimizer state dict |
| `scheduler.pt` | `torch.save` scheduler state dict |
| `dataloader.json` | `ShardedTokenDataset` cursor: shard index, window offset, window size |
| `rng.pt` | python, numpy, torch CPU, and torch CUDA RNG states |
| `meta.json` | step, tokens seen, loss, recipe snapshot hash, git SHA, library versions, wall-clock, checkpoint format |

## Atomicity

`training/runtime/checkpoint.py` writes every file into `step_X.tmp/`, fsyncs each file and
directory, then publishes with a single `os.rename`. The `latest` symlink is updated
the same way (temp symlink, rename). A crash at any point leaves either the previous
complete state or ignorable `*.tmp` debris — never a half checkpoint that resume
could pick up. Retention sweeps `*.tmp` debris on its next pass.

## Retention

Applied immediately after each successful save: keep the last
`checkpoint.retain_last` checkpoints plus every step divisible by
`checkpoint.retain_every`; delete the rest, oldest first. The checkpoint `latest`
points to is never deleted. The D10 disk-free watchdog is the backstop once it
lands.

Checkpoint-at-stable convention (P2-D3): for a `schedule.type=wsd` run, every
checkpoint written during the stable phase (steps in
`[warmup_steps, decay_start_frac * total_steps)`) is decay-branchable — a decay
leg can later be re-run from it with a different decay configuration. The
retention settings for a wsd run must keep at least one stable-phase checkpoint
(set `checkpoint.retain_every` so a multiple lands inside the plateau).
Multi-branch tooling is deferred to later phases; Phase 2 records the convention
only.

## Resume

`checkpoint.resume=latest` (or an explicit checkpoint path) restores all five
components, then continues from the saved step to `schedule.total_steps`. The loop
warns when the resuming recipe's hash differs from the checkpoint's recorded hash
(`checkpoint.resume` itself is excluded from the hash), since a changed
`total_steps` shifts the LR schedule.

The acceptance test (`tests/training/test_loop.py`) runs N+M steps straight versus
N steps, checkpoint, restart, M steps — with
`torch.use_deterministic_algorithms(True)` and fixed seeds, test only — and asserts
identical per-step loss values. The production default remains nondeterministic for
speed.

## Tracking

`training/runtime/tracking.py` fans one `log(dict, step)` to the sinks enabled in the
recipe's `tracking` block: tensorboard (default on), wandb, mlflow. Metrics logged
per step: `loss`, `lr`, `grad_norm`, `tokens_per_s`, `step_time_s`, and
`gpu_mem_allocated_gb` on CUDA.

Offline-first: wandb runs with `WANDB_MODE=offline` unless the environment sets
otherwise; sync later with `wandb sync <run.dir>`. mlflow logs to a local
`mlruns/` file store under the run directory unless `MLFLOW_TRACKING_URI` is set.
A sink that fails to initialize or log is disabled with a stderr warning instead of
killing the run.

## Secrets

Tokens come from an untracked `.env.training` passed via compose `env_file`
(optional); the template is `env.training.example`. Nothing is baked into the
image.

```bash
$ cp env.training.example .env.training
```

## Module map

| Path | Purpose |
|------|---------|
| `training/runtime/loop.py` | Reference training loop |
| `training/runtime/checkpoint.py` | Atomic save/load, retention, resume resolution |
| `training/runtime/tracking.py` | Metric sink fan-out |
| `training/models/model.py` | Shared model/optimizer/scheduler builders (also used by the D4 calibrator) |
| `tests/training/test_checkpoint.py` | Atomicity, retention, resume-resolution unit tests |
| `tests/training/test_loop.py` | Deterministic resume and tracking acceptance tests |
