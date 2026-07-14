# F3 supervised fine-tuning entry point: implementation investigation

Comparison of the Phase 1 F3 design (roadmap document, "F3: Supervised
fine-tuning byb entry point") against the current repository state, with a work
breakdown, acceptance checks, deviations, and open questions. Per operating
rules, the repository is the source of truth; where the roadmap document
disagrees with the repository, this investigation records the repository state
and treats the document as advisory.

## Table of contents

- [Design summary](#design-summary)
- [Current repository state](#current-repository-state)
- [Deviations from the document](#deviations-from-the-document)
- [Blocking dependency: F1 is not implemented](#blocking-dependency-f1-is-not-implemented)
- [Work breakdown](#work-breakdown)
- [Acceptance checks](#acceptance-checks)
- [Risks](#risks)
- [Open questions](#open-questions)

## Design summary

F3 is one command that takes a recipe and runs a supervised fine-tune on TRL,
supporting full fine-tuning and QLoRA, wired into the Phase 0 tracking,
checkpointing, and watchdog machinery. Tag: orchestrate. Depends on: D1
(training container), D5/F2 (recipe schema with PEFT), D6 (tracking and
checkpointing), D10 (watchdog), F1 (chat and instruction data preparation).
The document specifies:

1. **TRL `SFTTrainer` is the trainer.** The D6 reference loop is not used for
   SFT. The entry point builds the tokenizer and model from the recipe (with
   F2 PEFT wrapping), the dataset via F1's loader, an `SFTConfig` mapped from
   the recipe (sequence length, packing, batch and accumulation, schedule,
   precision), and runs `trainer.train()` with resume support.
2. **Unsloth is an optional backend, off by default.** `sft.backend:
   trl|unsloth` in the recipe. The unsloth path swaps model loading to
   `FastLanguageModel` and keeps everything else identical. Unsloth failures
   are not release blockers; every tracking run records the backend used.
3. **Checkpointing and resume** go through TRL/transformers' native checkpoint
   mechanism into the D6 directory layout, with D6 retention applied by a
   `TrainerCallback` after each save. Adapter-only checkpoints (PEFT) are the
   default for QLoRA runs; `meta.json` records whether a checkpoint is
   adapter-only or full, because F6 (export) needs to know.
4. **Watchdog and tracking attach as callbacks** (the D10 TRL integration
   path). Metrics: loss, learning rate, gradient norm, tokens per second,
   trainable-parameter report, GPU telemetry.
5. **Held-out loss is computed during training** on F1's held-out split at
   `eval_steps` intervals, so F7's scorecard has a training-time curve.

Document interface: `make train-sft CONFIG=recipes/sft-qlora-gemma-e4b.yaml`
with override pass-through, script `scripts/train/train_sft.py`, logic in
`training/sft.py`. Preflight, in order: recipe `validate()`, D4 memory-fit
check (warn or refuse per a `--force` flag), D9 lock acquisition,
inference-services-down check.

Document acceptance: a QLoRA fine-tune of a small instruction dataset
completes end to end and logs to tracking; a tiny SFT run checkpoints, is
killed, and resumes with a loss trajectory meeting D6's deterministic-resume
bar; the same tiny run executes on both backends (or the unsloth failure is
documented); preflight refuses to start while llama-gpu is up and releases the
lock on every exit path.

## Current repository state

Infrastructure F3 consumes is largely in place:

| Dependency | State | Repository location |
|------------|-------|---------------------|
| D1 training container | Done | `docker/training/Dockerfile.training`, `docker-compose.training.yaml`; TRL 0.24.0 pinned in the pyproject training group |
| D5 recipe system | Done | `training/recipes/` (schema, loader with override handling, `validate()`); `hydra.main` deliberately not used |
| F2 PEFT schema | Done | `peft` group in `training/recipes/schema.py` (lora/dora, nf4 QLoRA, `target_modules: auto`), `training/peft_utils.py` (`build_peft_model`, `trainable_parameter_report`), `training/families.py` registry |
| D4 memory-fit | Done, QLoRA-aware | `training/calculator/`, `python -m training memfit` |
| D6 checkpoint/tracking | Done | `training/checkpoint.py` (atomic `step_N/` layout, `latest` symlink, retention, `meta.json`), `training/tracking.py` (`Tracker.log` fan-out: tensorboard, wandb, mlflow) |
| D9 lock | Done | `training/lock.py` (`/mnt/ai-data/training/.train.lock`, context manager, staleness reported not auto-removed) |
| D10 watchdog | Done for the D6 loop | `training/watchdog.py` (`on_step` plus NVML background thread); the TRL callback adapter is explicitly anticipated but not written |
| D7 serving/export | Done | `training/serving.py` (service specs for llama-gpu 8002, vllm-gpu 8003, health checks), `training/export/` |
| F4 teacher harness | Done | `training/data/teacher.py` writes messages-format JSONL that "F1's prepare_sft.py later consumes" |
| Unified CLI | Done | `python -m training <subcommand>` (`training/cli.py`: memfit, calibrate, train, export); resolved D4 decision 1 says script entry points migrate to CLI subcommands |
| F2 reference recipe | Done | `training/recipes/reference/sft-qlora-1b.yaml` (QLoRA stand-in for the F8 SFT recipe; `data.source` points at a placeholder) |

Not present:

- `training/sft.py` and any SFT entry point (script or CLI subcommand).
- An `sft` group in the recipe schema (backend selection, packing, chat-data
  fields, eval interval for held-out loss).
- F1 in its entirety: no `training/data/sft.py`, no chat-template registry
  entries in `training/families.py` (F2 populated only `lora_target_modules`;
  the module docstring says F1 extends each entry with its chat template), no
  `prepare_sft.py`, no prompt-loss masking, no held-out split machinery.
- Unsloth: not pinned anywhere in `pyproject.toml` or the training Dockerfile.
- A TRL `TrainerCallback` adapter for the watchdog, tracker, or checkpoint
  retention.

## Deviations from the document

Repository conventions that override the document's wording:

1. **Entry-point path and shape.** The document names
   `scripts/train/train_sft.py`. The repository uses `scripts/training/` and,
   per resolved D4 decision 1, new tooling lands as `python -m training`
   subcommands (D6's trainer is `python -m training train`). F3 should be a
   `train-sft` subcommand in `training/cli.py` wrapped by a `make train-sft`
   target, matching `make train`.
2. **Override handling.** The document says "Hydra overrides pass through".
   The repository's D5 loader owns override handling and `hydra.main` is
   explicitly not used (`training/cli.py` docstring). Overrides flow as
   `ARGS="optimizer.lr=1e-4"` through the existing loader path.
3. **Model naming.** The document says "gemma-4-E4B". The repository treats
   these checkpoints as Gemma 3n (`model_type` `gemma3n`/`gemma3n_text`,
   MatFormer/PLE) in `training/families.py`. All F3 work keys off the HF
   `model_type`, not marketing names.
4. **Checkpoint layout.** The document says TRL's native mechanism writes
   "into the D6 directory layout". These are not the same format: D6's
   `step_N/` layout (atomic rename, `latest` symlink, `meta.json`,
   `dataloader.json`, RNG state) is custom, while transformers writes
   `checkpoint-<step>/` with its own trainer state. Reconciliation is a real
   design task, not a wording nit (see open questions).
5. **Preflight "inference-services-down check".** No such check exists as a
   named utility; `training/serving.py` has per-service health probes that can
   be inverted (refuse when any service answers). D9's lock alone does not
   cover this — the lock stops serving swaps, not an already-running llama-gpu.

## Blocking dependency: F1 is not implemented

The document orders F1 before F3 ("the trainer consumes prepared data") and
the repository confirms F1 has not landed. F1 is itself a substantial,
correctness-critical task: canonical messages format, per-family chat-template
registry with assistant-span locators, prompt-loss masking to -100 with golden
fixture tests, sequence packing with correct attention separation,
materialized artifacts with manifests, and a seeded held-out split. F3's
held-out-loss decision and its acceptance criteria all assume F1's loader
(`load_sft_dataset`) and artifacts exist.

Practical consequence: F3 cannot meet its acceptance without F1, so either F1
is scheduled first as its own task (document order, recommended) or F3's scope
expands to include a minimal F1 subset — which would still require the masking
and template registry, the highest-risk part, so little is saved.

## Work breakdown

Assuming F1 lands first (see open questions):

1. **Schema: `sft` group** in `training/recipes/schema.py` — `backend:
   trl|unsloth` (default trl), packing flag, `eval_steps`, held-out
   evaluation toggle, plus mapping fields SFTConfig needs that the existing
   groups do not carry. `validate()` invariants (for example unsloth combined
   with unsupported precision) with unit tests. Extend
   `training/recipes/reference/sft-qlora-1b.yaml` or add the real SFT recipe.
2. **`training/sft.py`** — builds tokenizer and model from the recipe
   (delegating PEFT wrapping to `training/peft_utils.py`), loads the F1
   dataset (train plus held-out), maps recipe fields to `SFTConfig`,
   constructs `SFTTrainer`, runs `trainer.train(resume_from_checkpoint=...)`.
3. **Callback adapters** — one module bridging TRL's `TrainerCallback` API to
   the existing D6/D10 objects: `Tracker.log` on `on_log`, watchdog `on_step`
   evaluation with `WatchdogAbort` translated to an emergency checkpoint and
   abort, D6 retention applied after each `on_save`, and `meta.json`
   augmentation recording adapter-only versus full (for F6).
4. **Preflight sequence** in the entry point — recipe `validate()`, D4
   memory-fit (warn or refuse per `--force`), D9 lock acquisition via the
   existing context manager (released on every exit path), inference-services
   health probe inverted into a refuse-if-up check.
5. **CLI and Makefile** — `train-sft` subcommand in `training/cli.py`, `make
   train-sft CONFIG=... ARGS=...` target in the Makefile, following the
   existing `train` target exactly.
6. **Unsloth backend (optional, gated)** — `sft.backend: unsloth` path
   swapping model construction to `FastLanguageModel`; requires a pinned
   unsloth version compatible with `torch`/`transformers==5.13.1` and a
   container smoke-test step verifying Gemma 3n (PLE) support. Failure mode is
   documented, not release-blocking.
7. **Tests** — unit tests for schema validation and SFTConfig mapping;
   callback-adapter tests with a stub trainer; preflight tests (lock held,
   service up, memory-fit refusal) following `tests/training/` conventions.
   GPU acceptance runs are host-side.

## Acceptance checks

From the document, restated against repository machinery (all host-side, run
by the human):

1. A QLoRA fine-tune of a Gemma 3n E4B-class base on a small instruction
   dataset completes end to end and logs to tracking.
2. A tiny SFT run (minutes) checkpoints, is killed, and resumes; the loss
   trajectory meets D6's deterministic-resume bar (`tests/training/test_loop.py`
   documents the existing bar for the reference loop; the TRL path needs its
   equivalent defined).
3. The same tiny run executes on both backends (trl, unsloth), or the unsloth
   failure is documented; tracking records the backend.
4. Preflight refuses to start while llama-gpu is up, and the D9 lock is
   released on every exit path (including `WatchdogAbort` and SIGKILL
   staleness reporting).

## Risks

- **TRL/PEFT/unsloth support for Gemma 3n (PLE)** must be verified in the
  container smoke test before any long run (document risk, owned by F1/F3).
  TRL 0.24.0 and transformers 5.13.1 are already pinned; the smoke test in
  `scripts/training/smoke_test.py` is the natural place to extend.
- **Checkpoint-format reconciliation** (deviation 4) touches F6: export
  consumes D6's `meta.json` and `model/` layout, so whatever F3 writes must
  keep `training/export/` working or extend it.
- **Deterministic resume through TRL** is weaker than D6's loop, which
  restores dataloader cursor and full RNG state explicitly. Transformers'
  resume replays dataloader state differently; the D6 bar may need an
  explicitly stated tolerance for the TRL path.
- **Unsloth pinning**: unsloth pins tightly to specific torch/transformers
  versions; a compatible pin against transformers 5.13.1 may not exist, which
  the document anticipates (may-lag, not blocking).

## Open questions

1. **F1 sequencing**: implement F1 as its own task first (document order,
   recommended), or fold a minimal subset into F3?
2. **Checkpoint layout**: wrap TRL's native `checkpoint-<step>/` output and
   add D6's `meta.json` plus retention on top (least invasive), or teach
   `training/checkpoint.py` to write its full atomic layout from a
   `TrainerCallback` (stronger guarantees, more work, and duplicates
   transformers' own resume state)? F6's export path is the constraint.
3. **Unsloth in scope now?** Pinning and smoke-testing unsloth is meaningful
   extra surface for a non-blocking optimization; it could land as a
   follow-up once the trl path is accepted.
4. **Deterministic-resume bar for TRL**: is bit-exact loss-curve match (D6's
   bar) required, or is a stated tolerance acceptable for the TRL path?
