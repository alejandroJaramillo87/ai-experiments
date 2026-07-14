# F3 supervised fine-tuning entry point: implementation investigation

Comparison of the Phase 1 F3 design (roadmap document, "F3: Supervised
fine-tuning byb entry point") against the current repository state, with a work
breakdown, acceptance checks, deviations, and open questions. Per operating
rules, the repository is the source of truth; where the roadmap document
disagrees with the repository, this investigation records the repository state
and treats the document as advisory. Revised after F1 (chat and instruction
data preparation) merged and the Gemma 4 primary / Qwen3 fallback model policy
was adopted.

## Table of contents

- [Design summary](#design-summary)
- [Current repository state](#current-repository-state)
- [Deviations from the document](#deviations-from-the-document)
- [Inherited F1 handoffs](#inherited-f1-handoffs)
- [Work breakdown](#work-breakdown)
- [Acceptance checks](#acceptance-checks)
- [Risks](#risks)
- [Resolved decisions](#resolved-decisions)
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

Document interface: `make train-sft CONFIG=...` with override pass-through,
script `scripts/train/train_sft.py`, logic in `training/sft.py`. Preflight, in
order: recipe `validate()`, D4 memory-fit check (warn or refuse per a
`--force` flag), D9 lock acquisition, inference-services-down check.

Model policy (repository, post-F1): Gemma 4 E-models (HF `model_type`
`gemma4`/`gemma4_text`; `google/gemma-4-E2B` and `-E4B`) are the primary
family for every fine-tuning deliverable; Qwen3 is the single fallback family
so a Gemma 4 library-support failure never blocks the phase. F3's acceptance
runs target Gemma 4 E2B first (the reference recipe), with a Qwen3 recipe as
the fallback proof.

## Current repository state

Every F3 dependency has landed:

| Dependency | State | Repository location |
|------------|-------|---------------------|
| D1 training container | Done | `docker/training/Dockerfile.training`, `docker-compose.training.yaml`; TRL 0.24.0 and transformers 5.13.1 pinned in the pyproject training group |
| D5 recipe system | Done | `training/recipes/` (schema, loader with override handling, `validate()`); `hydra.main` deliberately not used |
| F2 PEFT schema | Done | `peft` group in `training/recipes/schema.py` (lora/dora, nf4 QLoRA, `target_modules: auto`), `training/peft_utils.py` (`build_peft_model`, `trainable_parameter_report`) |
| F1 data preparation | Done | `training/data/sft.py`: segment-by-segment rendering from each family's `ChatTemplateSpec`, prompt-loss masking to -100 by construction, `mask_debug`, materialized `train.jsonl`/`holdout.jsonl` artifacts with hash-verified manifests, `load_sft_dataset(dir, tokenizer)` returning `(train, holdout)` lists of `{"input_ids", "labels"}`; `scripts/training/prepare_sft.py` + `make prepare-sft`; golden-fixture mask tests in `tests/training/test_sft.py` |
| Family registry | Done, Gemma 4 + Qwen3 only | `training/families.py`: `FamilySpec(model_type, lora_target_modules, chat_template)`; `CHAT_TEMPLATE_VERSION = 2` encodes the official gemma4 template (`<|turn>role ... <turn|>`, native system turn, assistant role "model") and ChatML for qwen3 |
| Recipe `sft` group (data side) | Done | `SftConfig(enabled, dataset)` in `training/recipes/schema.py`; `_check_sft` validation (requires `model.hf`, a prepared dataset with `manifest.json`). Trainer-side fields explicitly deferred to F3 per the schema docstring |
| D4 memory-fit | Done for Qwen3, **missing gemma4** | `training/calculator/components.py` `HF_SUPPORTED_MODEL_TYPES` has no `gemma4`/`gemma4_text`; per-layer embeddings (262k vocab x 30 layers x 256 dims packed into one table) add parameters the dense formulas do not model |
| D6 checkpoint/tracking | Done | `training/checkpoint.py` (atomic `step_N/` layout, `latest` symlink, retention, `meta.json`), `training/tracking.py` (`Tracker.log` fan-out) |
| D9 lock | Done | `training/lock.py` (context manager; staleness reported, never auto-removed) |
| D10 watchdog | Done for the D6 loop | `training/watchdog.py`; the TRL callback adapter is explicitly anticipated in the D10 investigation but not written |
| D7 serving/export | Done | `training/serving.py` (health probes for llama-gpu 8002, vllm-gpu 8003), `training/export/` (scoped to the Gemma 4 / Qwen3 policy) |
| Unified CLI | Done | `python -m training <subcommand>` (`training/cli.py`: memfit, calibrate, train, export) |
| Reference recipe | Done | `training/recipes/reference/sft-qlora-e2b.yaml` (Gemma 4 E2B QLoRA; F8 stand-in; `data.source` still a placeholder, `sft` group not yet referenced) |

Not present (F3's actual surface):

- `training/sft.py` and any SFT entry point (script or CLI subcommand).
- Trainer-side `sft` schema fields: `backend`, `packing`, `eval_steps`,
  epochs, and whatever else `SFTConfig` needs that existing groups do not
  carry.
- Sequence packing anywhere: F1 deliberately ships unpacked, mask-correct
  examples; the F1 investigation records packing as owned by F3 (TRL's
  packing with flash-attention varlen, plus packed-mode mask tests).
- Unsloth: not pinned in `pyproject.toml` or the training Dockerfile; no
  smoke-test step covers it.
- TRL `TrainerCallback` adapters for the watchdog, tracker, or D6 retention.
- gemma4 support in the D4 calculator (see inherited handoffs).

## Deviations from the document

Repository conventions that override the document's wording:

1. **Entry-point path and shape.** The document names
   `scripts/train/train_sft.py`. The repository uses `scripts/training/` and,
   per resolved D4 decision 1, new tooling lands as `python -m training`
   subcommands (D6's trainer is `python -m training train`). F3 should be a
   `train-sft` subcommand in `training/cli.py` wrapped by a `make train-sft`
   target, matching `make train` and F1's `make prepare-sft`.
2. **Override handling.** The document says "Hydra overrides pass through".
   The D5 loader owns override handling and `hydra.main` is explicitly not
   used; overrides flow as `ARGS="optimizer.lr=1e-4"` through the loader.
3. **Model naming and policy.** The document's examples span gemma and llama
   families. The repository policy is Gemma 4 primary (`model_type gemma4`,
   verified against transformers' gemma4 configuration), Qwen3 the single
   fallback; no Llama family is registered. F3 keys everything off the HF
   `model_type` through `training/families.py`.
4. **Checkpoint layout.** The document says TRL's native mechanism writes
   "into the D6 directory layout". These are not the same format: D6's
   `step_N/` layout (atomic rename, `latest` symlink, `meta.json`,
   `dataloader.json`, RNG state) is custom, while transformers writes
   `checkpoint-<step>/` with its own trainer state. Reconciliation is a real
   design task, not a wording nit; F6 export reads D6's `meta.json` and
   `model/` layout, so whatever F3 writes must keep `training/export/`
   working or extend it (see open questions).
5. **Preflight "inference-services-down check".** No such named utility
   exists; `training/serving.py`'s per-service health probes can be inverted
   (refuse when any service answers). The D9 lock alone does not cover an
   already-running llama-gpu.
6. **Dataset loading.** The document says the trainer "can also tokenize on
   the fly for small datasets". The repository's F1 decision is stricter:
   recipes reference materialized `prepare_sft.py` artifacts only, enforced
   by `_check_sft` (a dataset directory without `manifest.json` fails
   validation). F3 consumes `load_sft_dataset` and does not add an
   on-the-fly path.

## Inherited F1 handoffs

The F1 investigation explicitly assigns three items to F3:

1. **Sequence packing.** F1 artifacts are unpacked. F3 owns the recipe flag,
   TRL packing integration (position ids reset per example via
   flash-attention varlen), and masking-correctness tests in packed mode.
   Note: F1 artifacts are pre-tokenized with labels; TRL's packing paths are
   built around text/messages columns, so packing pre-masked `input_ids`/
   `labels` pairs needs verification against TRL 0.24.0's API (it may require
   a custom collator that concatenates examples and labels in step).
2. **D4 calculator gemma4 support.** `HF_SUPPORTED_MODEL_TYPES` excludes
   gemma4; the memfit preflight for the primary family fails today. This must
   land before F3's acceptance run sizes the E2B QLoRA reference recipe
   (per-layer embeddings are frozen under QLoRA, so the QLoRA memory model
   mostly needs correct parameter counting, but that counting is the work).
   Qwen3 recipes are fully supported today, so the fallback path is not
   blocked.
3. **Thinking-mode training decision.** The official gemma4 template strips
   `<|channel>thought ... <channel|>` sections from conversation history, so
   F1's rendered training conversations contain no thinking tokens. Whether
   SFT examples should carry a masked empty-thought prefix to mirror the
   inference-time prompt (where `<|think|>` appears in the system turn) is an
   F3 decision, checked during the host-side smoke fine-tune; a spec change
   bumps `CHAT_TEMPLATE_VERSION` and invalidates prepared artifacts by
   design.

## Work breakdown

1. **Schema: trainer-side `sft` fields** extending F1's `SftConfig` —
   `backend: trl|unsloth` (default trl), `packing: bool`, `eval_steps`,
   `num_epochs` (or max-steps precedence rule against `schedule.total_steps`),
   plus any `SFTConfig` mapping fields the existing groups lack. `validate()`
   invariants (backend value, unsloth-incompatible combinations) with unit
   tests in `tests/training/test_recipes.py`. Point
   `sft-qlora-e2b.yaml` at the `sft` group and drop its placeholder
   `data.source`.
2. **D4 calculator: gemma4 architecture support** — parameter counting for
   per-layer embeddings and the E-model attention layout in
   `training/calculator/components.py`, unblocking the memfit preflight for
   the primary family. Separable, and a prerequisite for the acceptance run.
3. **`training/sft.py`** — builds tokenizer and model from the recipe
   (delegating PEFT wrapping to `training/peft_utils.py`; `target_modules:
   auto` resolves through `training/families.py`), loads train and held-out
   splits via `load_sft_dataset` (tokenizer-hash check included), maps recipe
   fields to `SFTConfig`, constructs `SFTTrainer` with the held-out split as
   `eval_dataset`, and runs `trainer.train(resume_from_checkpoint=...)`.
4. **Packing** — recipe-gated, implemented against TRL 0.24.0's API for
   pre-tokenized, pre-masked examples; packed-mode golden mask tests extend
   `tests/training/test_sft.py` (F1 handoff 1).
5. **Callback adapters** — one module bridging TRL's `TrainerCallback` API to
   the existing D6/D10 objects: `Tracker.log` on `on_log` (loss, lr, grad
   norm, tokens/s, backend tag), watchdog signal evaluation with
   `WatchdogAbort` translated to an emergency checkpoint and abort, D6
   retention applied after each `on_save`, and `meta.json` written per
   checkpoint recording adapter-only versus full (for F6).
6. **Preflight sequence** in the entry point — recipe `validate()`, D4
   memory-fit (warn or refuse per `--force`), D9 lock via the existing
   context manager (released on every exit path), inference-services
   refuse-if-up probe built on `training/serving.py`.
7. **CLI and Makefile** — `train-sft` subcommand in `training/cli.py`, `make
   train-sft CONFIG=... ARGS=...` target, following `make train` exactly.
8. **Unsloth backend (optional, gated)** — `sft.backend: unsloth` swapping
   model construction to `FastLanguageModel`; requires a pinned unsloth
   version compatible with torch/transformers 5.13.1 and a
   `scripts/training/smoke_test.py` step verifying Gemma 4 (PLE) support.
   Documented as may-lag; failures are not release blockers; tracking records
   the backend on every run.
9. **Tests** — schema validation and SFTConfig mapping; callback adapters
   against a stub trainer; preflight refusals (lock held, service up,
   memory-fit); packed-mode masks. GPU acceptance runs are host-side.

Suggested order: 1 and 2 in parallel (schema and calculator are independent),
then 3–7 as the core, 8 last.

## Acceptance checks

From the document, restated against repository machinery (all host-side, run
by the human, per the sandbox rules):

1. A QLoRA fine-tune of Gemma 4 E2B (`sft-qlora-e2b.yaml`) on a small
   prepared instruction dataset completes end to end and logs to tracking. A
   Qwen3 QLoRA run is the documented fallback if a Gemma 4 library failure
   appears.
2. A tiny SFT run (minutes) checkpoints, is killed, and resumes; the loss
   trajectory meets the resume bar defined for the TRL path (see open
   question 2).
3. The same tiny run executes on both backends (trl, unsloth), or the unsloth
   failure is documented; tracking records the backend.
4. Preflight refuses to start while llama-gpu is up, and the D9 lock is
   released on every exit path (including `WatchdogAbort`).
5. Packed-mode golden mask tests pass (loss applies only to assistant content
   and end-of-turn tokens, packed and unpacked) — the scope document's
   required unit test, extended to the packed case F1 deferred.

## Risks

- **TRL/PEFT support for Gemma 4 (per-layer embeddings)** must be verified in
  the container smoke test before any long run. transformers 5.13.1 ships the
  gemma4 architecture; the risk is TRL/PEFT interaction (for example,
  gradient checkpointing or `prepare_model_for_kbit_training` against the PLE
  embedding table). The registered Qwen3 fallback bounds the damage.
- **Packing pre-masked examples through TRL** may not be supported directly
  by TRL 0.24.0's packing utilities (built around raw-text datasets); a
  custom packing collator is plausible extra scope. If it grows, packing can
  ship behind a default-off flag without blocking acceptance checks 1–4.
- **Checkpoint-format reconciliation** touches F6: export consumes D6's
  `meta.json` and layout; the chosen wrapping (open question 1) must keep
  `training/export/` working for both adapter-only and full checkpoints.
- **Deterministic resume through TRL** is weaker than D6's loop, which
  restores dataloader cursor and full RNG state explicitly; transformers'
  resume replays state differently, so the D6 bar may need a stated
  tolerance.
- **Unsloth pinning**: unsloth pins tightly to specific torch/transformers
  versions; a pin compatible with transformers 5.13.1 and gemma4 may not
  exist yet. Anticipated by the document: may-lag, not blocking.
- **Thinking-mode fidelity** (F1 handoff 3): if the smoke fine-tune shows the
  tuned model regressing on thinking-mode prompts because training examples
  lack the `<|think|>` system-turn framing, the fix is a `ChatTemplateSpec`
  change plus a `CHAT_TEMPLATE_VERSION` bump and re-preparation — cheap in
  code, expensive in re-tokenization time for large datasets.

## Resolved decisions

1. **Checkpoint layout: wrap TRL's native checkpoints (owner decision).**
   `trainer.train(resume_from_checkpoint=...)` only understands the
   `checkpoint-<step>/` directories the `Trainer` itself writes; making it
   resume from D6's `step_N/` layout would mean re-implementing transformers'
   resume loading against private internals, version-coupled to every
   transformers bump, to preserve a format whose extra state
   (`dataloader.json`, explicit RNG capture) transformers already tracks its
   own way. Instead, a `TrainerCallback` runs after each save and (a) writes a
   D6-style `meta.json` into the checkpoint directory — step, adapter-only
   versus full flag, base model id and revision, config snapshot hash, library
   versions — which is the part F6 export actually reads; (b) applies D6's
   retention policy; (c) updates a `latest` symlink via atomic rename only
   after verifying the checkpoint directory is complete. The verification
   step compensates for transformers' non-atomic writes: a crash mid-save
   leaves a directory without `meta.json`, which resume and export ignore.
   `training/export/` learns to read a TRL checkpoint through its `meta.json`;
   the D6 `step_N/` reader stays for the reference loop. Two formats coexist,
   distinguished by `meta.json`, each owned by its writer.
2. **Deterministic-resume bar: stated tolerance (owner decision).** Bit-exact
   resume remains D6's bar for the reference loop (and Phase 2/3 pretraining,
   which stays on that loop). The TRL path's bar: the resumed run's held-out
   loss at matching eval steps agrees within a documented tolerance, and the
   training-loss trajectory shows no step discontinuity. Forward check
   against later deliverables: F5 (distillation) reuses this trainer and
   inherits the same bar; F6 (export) and F7 (evaluation) consume final
   checkpoints and scorecards, which do not depend on resume determinism;
   nothing downstream compares two resumed halves of one run bit-for-bit.
   The one real cost is debugging — a divergence after resume cannot be ruled
   out as nondeterminism — accepted and documented.
3. **Unsloth: pinned and smoke-tested in F3 scope (owner decision).** The
   pin lands in the pyproject training group, the container smoke test gains
   an unsloth Gemma 4 load step, and `sft.backend: unsloth` is exercised in
   the tiny-run acceptance. Its failures remain documented-not-blocking per
   the roadmap.
4. **Packing: ships in F3 (owner decision).** Including the custom collator
   for pre-masked examples if TRL 0.24.0's packing utilities cannot consume
   them directly, and the packed-mode golden mask tests.
5. **F1 sequencing: moot** — F1 merged before F3 work begins.

## Open questions

1. **Reference SFT dataset** (carried over from F1): F3's end-to-end
   acceptance needs a real instruction dataset to fine-tune on — which
   public dataset to use (the roadmap's example is
   `HuggingFaceH4/ultrachat_200k`) and its license string for the prepared
   artifact's manifest.
2. **Thinking-mode prefix** (F1 handoff 3): Gemma 4 inference prompts enable
   thinking via `<|think|>` in the system turn, but F1's training data
   contains no thinking tokens (the official template strips thought
   channels from history). The open question is whether fine-tuning on data
   without that framing degrades the tuned model's thinking mode, and if so
   whether training examples should carry a masked empty-thought prefix so
   they match what the model sees at inference. Decided empirically during
   the host-side smoke fine-tune, not up front.
