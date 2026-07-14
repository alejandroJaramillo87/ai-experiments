# D6 tracking and resumable checkpointing: implementation investigation

Comparison of the Phase 0 D6 design (roadmap document, "D6: Tracking and resumable
checkpointing") against the current repository state, with a work breakdown, acceptance
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

## Design summary

D6 makes every run observable and every run resumable, and ships the minimal reference
training loop those facilities are tested against. Tag: orchestrate. Depends on D1
(container) and D5 (recipe schema), both complete. The document specifies:

1. **Reference loop, deliberately small** (`training/loop.py`): dense model, standard
   forward/backward/step with gradient accumulation, driven entirely by a recipe. A
   known-good skeleton for later phases, not the Phase 1 SFT trainer (TRL) nor the
   Phase 3 production loop.
2. **Checkpoint contents**, one directory per step
   (`/mnt/ai-data/training/checkpoints/<run>/step_000123400/`):
   - `model/` — sharded safetensors, HF `save_pretrained` layout so D7 export consumes
     checkpoints directly.
   - `optimizer.pt`, `scheduler.pt` — `torch.save` (DeepSpeed runs use DeepSpeed's own
     checkpoint API into the same directory, recorded in metadata).
   - `dataloader.json` — the D2 shard cursor plus sampler state.
   - `rng.pt` — python, numpy, torch CPU, torch CUDA states.
   - `meta.json` — step, tokens seen, loss, recipe snapshot hash, git SHA, library
     versions, wall-clock.
3. **Atomicity**: write to `step_X.tmp/`, fsync files, `os.rename` to the final name,
   then update a `latest` symlink. A crash never leaves a half checkpoint that resume
   could pick up.
4. **Retention** after each successful save: keep the last `retain_last` checkpoints
   plus every `retain_every`-th step; delete the rest oldest first; never delete the
   checkpoint `latest` points to. Retention runs first; the D10 disk-free watchdog is
   the backstop.
5. **Resume**: `checkpoint.resume=latest|<path>` restores all five components and
   continues. Acceptance compares N+M straight steps against N steps, checkpoint,
   restart, M steps with `torch.use_deterministic_algorithms(True)` and fixed seeds
   (test only) and asserts identical loss values.
6. **Tracking** (`training/tracking.py`): one `log(dict, step)` fanned to enabled sinks
   (wandb, mlflow, tensorboard). Offline-first: wandb offline by default; syncing is
   manual. Reference-loop metrics: loss, lr, grad norm, tokens/s, step time, GPU
   memory, plus D10 telemetry once D10 exists.
7. **Secrets**: `WANDB_API_KEY`, `HF_TOKEN` come from an untracked `.env.training`
   passed via compose `env_file`; nothing baked into the image.

## Current repository state

D6 is greenfield for its three core modules, but nearly all of its contract surface
already exists and constrains the implementation.

| Area | State |
|------|-------|
| `training/loop.py`, `training/checkpoint.py`, `training/tracking.py` | Absent — the actual D6 deliverables |
| D5 recipe contract | Done: `CheckpointConfig` (`interval_steps`, `retain_last`, `retain_every`, `resume`) and `TrackingConfig` (`wandb`, `wandb_project`, `mlflow`, `tensorboard`) in `training/recipes/schema.py`; `_check_checkpoint` invariants in `training/recipes/validate.py` |
| D2 resume hook | Done: `ShardedTokenDataset.state_dict()`/`load_state_dict()` in `training/data/shards.py` serialize the cursor (`shard`, `offset`, `window`), documented as "for D6 resume" |
| Model/optimizer builders | `training/calculator/calibrate.py` has `build_model` (scratch spec as HF `LlamaConfig` via `AutoModelForCausalLM.from_config`, so `save_pretrained` works) and `build_optimizer` (adamw, adamw8bit); its docstring says it is not the trainer, but the builders are directly reusable or extractable |
| Reference recipe for tests | Done: `training/recipes/reference/tiny-e2e.yaml` — a scratch 2-layer model, `interval_steps: 5`, `retain_last: 2`, `retain_every: 10`, `total_steps: 20`, CPU-runnable, written explicitly for the D6 resume test |
| Tracking dependencies | Pinned in `pyproject.toml`/`uv.lock`: `wandb==0.28.0`, `mlflow==3.14.0`, `tensorboard==2.21.0`; hydra-core 1.3.2 for config |
| Entry-point convention | `python -m training <subcommand>` (`training/cli.py`); its docstring reserves the option for the D6 trainer to live outside the dispatcher if it adopts `hydra.main` |
| Container | Done (D1): compose mounts `/mnt/ai-data/training` read-write, `./training` and `./scripts` read-only; `read_only: true` root; `init: true`; no `env_file` entry yet |
| Makefile | `train-*` targets exist through D5; no `make train` target |
| Secrets plumbing | `.gitignore` covers per-tier secrets with `env.<tier>.example` templates; `env.example` and `env.interp.example` exist; no `env.training.example` and compose does not reference an env file |
| Tests | `tests/training/` is CPU-only and sandbox-runnable; `conftest.py` fixtures exist for tiny shard/tokenizer fixtures |

## Deviations from the document

1. **The reference loop must support scratch models now, not "once Phase 3 lands".**
   The document scopes the loop to HF `AutoModelForCausalLM` with scratch support
   deferred to Phase 3, but the repository's acceptance recipe (`tiny-e2e.yaml`) is a
   scratch model, and `calibrate.build_model` already instantiates scratch specs as HF
   Llama models. The loop should build models the same way (HF hub id via
   `from_pretrained`, scratch via the existing spec-to-`LlamaConfig` path); both yield
   HF models, so the `save_pretrained` checkpoint layout holds for both.
2. **`dataloader.json` cursor has no epoch field.** The document lists "shard index,
   offset, epoch"; the repository's `ShardedTokenDataset.state_dict()` serializes
   `shard`, `offset`, `window` and has no epoch counter or sampler. The checkpoint
   stores exactly what `state_dict()` returns; if multi-epoch runs need an epoch
   counter later, it is added to the dataset first, not invented in the checkpoint
   writer. Phase 0 windows are read sequentially (no sampler state exists).
3. **Checkpoint directory location.** The document says
   `/mnt/ai-data/training/checkpoints/<run>/`; the D5 run-directory convention places
   run outputs under `${run.output_root}/${run.name}` (default
   `/mnt/ai-data/training/runs`). The schema has no checkpoint-root field. Options:
   keep checkpoints inside the run directory (one tree per run, simplest retention),
   or add `checkpoint.root` defaulting to `/mnt/ai-data/training/checkpoints`.
   Recorded as an open question; the document's path is advisory.
4. **DeepSpeed checkpointing is out of scope for the reference loop.** The schema's
   `memory.offload: zero_offload` exists for D4 sizing, but the Phase 0 loop is a
   plain single-GPU loop; the DeepSpeed checkpoint-API branch mentioned in the
   document lands when a DeepSpeed-driven trainer lands (Phase 1). `meta.json`
   records the checkpoint format so the branch can be added compatibly.
5. **D10 telemetry in the metric set is deferred.** The watchdog does not exist;
   the loop logs the metrics it can produce (loss, lr, grad norm, tokens/s, step
   time, GPU memory via `torch.cuda` when available). The tracking fan-out is the
   stable surface D10 later plugs into.

## Work breakdown

1. `training/checkpoint.py` — `save_checkpoint(...)` and `load_checkpoint(...)`:
   step-numbered directories, HF `save_pretrained` for `model/`, `torch.save` for
   optimizer/scheduler, `dataloader.json` from the dataset `state_dict()`, `rng.pt`
   (python/numpy/torch CPU/torch CUDA), `meta.json`. Atomic protocol: write into
   `step_X.tmp/`, fsync every file and the directory, `os.rename`, update `latest`
   symlink (also atomically, via temp symlink + rename). `apply_retention(...)`
   implementing last-N plus every-Mth, never deleting the `latest` target,
   oldest-first deletion. Resolution of `resume: latest|<path>` including ignoring
   stray `.tmp` directories.
2. `training/tracking.py` — a `Tracker` with `log(metrics: dict, step: int)` and
   `close()`, fanning to sinks enabled by `TrackingConfig`: tensorboard
   (`SummaryWriter`), wandb (offline mode by default via `WANDB_MODE=offline` unless
   the environment overrides), mlflow. Sinks are lazily imported and individually
   optional; a sink failure warns and disables that sink rather than killing a run.
3. `training/loop.py` — `train(recipe)`: build tokenizer-free token dataset from D2
   shards, model (hf or scratch, reusing/extracting the builders in
   `calibrate.py`), optimizer (adamw, adamw8bit), scheduler (cosine, wsd, with
   warmup), gradient accumulation, bf16 autocast on CUDA (fp32 fallback on CPU so
   tiny-e2e runs in the sandbox), checkpoint save on `interval_steps` and at
   `total_steps`, retention after each save, resume restoring all five components,
   metrics through the tracker every step. fp8 branches are stubs until D8.
4. Entry point: `python -m training train --config <recipe> [overrides]` following
   the existing CLI subcommand pattern (see open question 1), plus a Makefile target
   `make train CONFIG=... ARGS=...` running it in the container.
5. Secrets: `env.training.example` (WANDB_API_KEY, HF_TOKEN, WANDB_MODE) and an
   optional `env_file` entry in `docker-compose.training.yaml`; `.gitignore` already
   covers the untracked real file.
6. `tests/training/test_checkpoint.py` — atomic-save unit tests (simulated kill
   between tmp-write and rename leaves previous `latest` intact and resumable),
   retention set exactness (last-N plus every-Mth, `latest` never deleted), resume
   path resolution.
7. `tests/training/test_loop.py` — the deterministic resume test on tiny fixtures:
   N+M straight versus N, checkpoint, restart, M with
   `torch.use_deterministic_algorithms(True)` and fixed seeds, asserting identical
   per-step losses; a tracking smoke test asserting tensorboard event files and an
   offline wandb run directory appear.
8. `docs/training/` usage doc for the loop, checkpoint layout, and resume workflow
   after implementation.

Items 1-3 and 6-7 are sandbox-executable (tiny-e2e is CPU-sized; torch is CPU-usable
in the sandbox). Item 4's containerized `make train` and any GPU-path verification
are host-side. The wandb/mlflow smoke may need network isolation flags verified in
the sandbox; tensorboard is fully local.

## Acceptance checks

Per the document, mapped to the repository:

- Sandbox: `pytest tests/training/` passes, including the deterministic resume test
  (identical loss trajectory), the SIGKILL-between-tmp-and-rename atomicity test, and
  the retention-set test, all on the tiny-e2e recipe against generated fixtures.
- Sandbox or host: a tiny recipe run with `tracking.tensorboard: true` and
  `tracking.wandb: true` produces tensorboard event files and an offline wandb run
  directory under the run directory.
- Host: `make train CONFIG=training/recipes/reference/tiny-e2e.yaml` completes inside
  the container writing checkpoints under the training root, then re-running with
  `checkpoint.resume=latest` continues from the last step. Human runs and reports
  output.

## Risks

- **Deterministic-resume strictness.** Exact loss equality requires
  `torch.use_deterministic_algorithms(True)`, fixed seeds, restored RNG state, and a
  dataloader with `workers: 0` (tiny-e2e already sets this). Any nondeterministic op
  in the model path breaks exactness; the fallback is documented tolerance, but exact
  match is the target and is realistic for the tiny CPU model.
- **`os.rename` and symlinks on the training root.** Atomic rename requires tmp and
  final on the same filesystem (guaranteed by writing the tmp dir alongside the final
  dir). `latest` as a symlink assumes the checkpoint root is not consumed across a
  filesystem boundary that strips symlinks; a `latest` file containing the step name
  is the fallback if symlinks prove awkward.
- **wandb/mlflow import cost and network behavior.** Both are heavyweight imports and
  wandb attempts network access unless offline mode is set before `wandb.init`. Lazy
  imports plus offline-by-default contain this; tests force offline mode.
- **Scheduler restore across resume.** `LambdaLR`/cosine schedules restore cleanly
  via `state_dict`, but `total_steps` changes between save and resume would silently
  shift the schedule; `meta.json` records the recipe snapshot hash and resume warns
  on mismatch.

## Resolved decisions

Confirmed with the repository owner during this investigation:

1. **Entry point**: `python -m training train` subcommand (consistent with `memfit`/
   `calibrate`, argparse-based), reusing `_load_validated`. Hydra's own run-dir/argv
   management is not needed since D5's loader handles overrides.
2. **Checkpoint location**: inside the D5 run directory,
   `/mnt/ai-data/training/runs/<run>/checkpoints/`, keeping one tree per run. The
   document's separate `/mnt/ai-data/training/checkpoints/<run>/` path is superseded;
   D7 consumes a path either way.
3. **Builder reuse**: extract `build_model`/`build_optimizer` from
   `training/calculator/calibrate.py` into a shared module (`training/model.py`)
   consumed by both the calibrator and the loop; the calibrator keeps its public
   behavior. The shared builder gains an attention-implementation fallback
   (`sdpa` when CUDA or flash-attn is unavailable) so the sandbox resume test runs
   on CPU; the hardcoded `flash_attention_2` remains the CUDA default.
