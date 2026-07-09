# Phase 1 product designs: fine-tuning

This document is the implementation-ready product design for every Phase 1
deliverable in the training scope ("Phase 1: Fine-tuning"). It follows the same
contract as the Phase 0 product designs: each section can be picked up in a
fresh session and implemented without re-deriving context. It records the
decisions already made, the interfaces to build, the repository and ecosystem
facts the design depends on, and the tests that define done.

The scope document defines *what* Phase 1 delivers; this document defines
*how*. Where the two disagree, this document wins and the scope document
should be corrected.

Phase 1 delivers supervised fine-tuning (SFT) and distillation of a strong
open base: the smallest increment over the Phase 0 foundation, the fastest
path to a useful model, and the proof of the train-and-serve loop.
Preference alignment (DPO, RLHF) is out of scope by policy.

## Table of contents

- [Phase 0 contract](#phase-0-contract)
- [Cross-cutting conventions](#cross-cutting-conventions)
- [Model policy: Gemma 4](#model-policy-gemma-4)
- [Build order](#build-order)
- [F1: Chat and instruction data preparation](#f1-chat-and-instruction-data-preparation)
- [F2: Parameter-efficient options in the recipe schema](#f2-parameter-efficient-options-in-the-recipe-schema)
- [F3: Supervised fine-tuning entry point](#f3-supervised-fine-tuning-entry-point)
- [F4: Teacher and synthetic data harness](#f4-teacher-and-synthetic-data-harness)
- [F5: Distillation harness](#f5-distillation-harness)
- [F6: Adapter merge and serve](#f6-adapter-merge-and-serve)
- [F7: Fine-tuning evaluation wrapper](#f7-fine-tuning-evaluation-wrapper)
- [F8: Recipe cards](#f8-recipe-cards)
- [Phase 1 exit checklist](#phase-1-exit-checklist)

## Phase 0 contract

Phase 1 is implemented only after Phase 0 lands. It consumes these Phase 0
interfaces as given and does not reimplement them. If an interface named here
differs from what Phase 0 actually shipped, update this list and adapt; do not
fork the foundation.

| Phase 0 deliverable | What Phase 1 consumes |
| --- | --- |
| D1 training container | The single training image and `make train-shell`. Phase 1 adds packages to the same image (see conventions), never a second image. |
| D2 lite data pipeline | `training/data/shards.py` conventions for manifests and artifact hashing. SFT data does not use token shards (see F1) but reuses the manifest discipline. |
| D5 recipe system | The Hydra dataclass schema in `training/recipes/schema.py`. Phase 1 extends it with `sft`, `peft`, and `distill` groups; it does not add a parallel config system. `validate(recipe)` gains the new cross-field invariants. |
| D6 tracking and checkpointing | `training/tracking.py` sinks and the checkpoint directory layout. TRL runs hook the watchdog and tracking through TRL's callback API as D10 specified. |
| D7 train-and-serve round trip | `training/export/` (PEFT `merge_and_unload`, pinned llama.cpp conversion, export manifest) and `scripts/eval/serve_eval.py` (serving swap, restore-on-failure, quantization-confound rule). F6 and F7 build on these, they do not duplicate them. |
| D9 heterogeneous compute | The run lock (`training/lock.py`). Every serving-dependent Phase 1 step takes the same lock discipline: teacher generation, judge scoring, and serve evaluation refuse to run while a training job holds the lock, and vice versa. |
| D10 watchdogs | The in-process callback, attached to TRL trainers via `transformers.TrainerCallback`. |

## Cross-cutting conventions

Inherited unchanged from Phase 0: inference ports from docker-compose.yaml
(llama-cpu 8001, llama-gpu 8002, vllm-gpu 8003); code and recipes in git,
artifacts under `/mnt/ai-data/training/`; Python 3.12, black + ruff, plain
technical prose, no emojis; tests in `tests/training/` run by `make test`;
every CLI gets a Makefile target in the `##@ Training` section; secrets via
the untracked `.env.training`.

Phase 1 additions:

- **Concurrency phasing is structural.** Teacher generation (F4), judge
  scoring (F7), and serve evaluation run with the inference services up and
  training idle; training runs with inference down. Each tool preflights the
  D9 lock and the compose state and fails with a clear message rather than
  contending for the GPU.
- **Dataset locations.** Chat/instruction and distillation datasets live
  under `/mnt/ai-data/training/datasets/sft/<name>/` and
  `/mnt/ai-data/training/datasets/distill/<name>/`, each with a
  `manifest.json` (source, license string, counts, tokenizer hash where
  applicable, git SHA, creation CLI, content hash) following the D2 manifest
  discipline. Downstream loaders refuse a dataset whose manifest hash does
  not match.
- **Never resize embeddings.** Phase 1 uses each base model's existing chat
  template and special tokens. No new tokens are added and
  `resize_token_embeddings` is never called. A template registry entry that
  would require new tokens is a design error.
- **Quantization-confound rule (extends D7).** Every base-versus-fine-tuned
  comparison is run at matched quantization and, for judge scoring, with the
  same judge model and prompt. The scorecard records quant type, judge
  identity, and prompt hash so mismatched comparisons are detectable.
- **New packages** (pin exact versions at implementation, in the D1 image and
  mirrored as ranges in the Poetry training group where pure-Python):
  `unsloth` (optional backend, see F3 risk), `mergekit`, `distilabel`,
  `lm-eval` (lm-evaluation-harness), `openai` (client for the local
  OpenAI-compatible endpoints). TRL, PEFT, and bitsandbytes are already in
  the D1 image.

## Model policy: Gemma 4

The Phase 1 reference models are the Gemma 4 effective-parameter models:

- **google/gemma-4-E2B** — effective 2B, the distillation student.
- **google/gemma-4-E4B** — effective 4B, the SFT target and the in-process
  distillation teacher.

Facts the design depends on (verified July 2026; re-verify at implementation):

- Gemma 4 shipped April 2026 in four sizes (E2B, E4B, 26B A4B, 31B). The
  E-models use per-layer embeddings (PLE) for on-device efficiency.
- llama.cpp supports Gemma 4 from launch. **Constraint on D7:** the pinned
  llama.cpp revision must be at or after Gemma 4 support; bump the pin with a
  note if it predates it.
- Gemma 4 uses standard system/user/assistant roles plus explicit thinking
  control, and its BOS/EOS tokens differ from Llama defaults. The template
  registry (F1) owns these details; serving through llama.cpp uses the
  model's built-in template.
- Weights are gated behind license acceptance on Hugging Face; `HF_TOKEN`
  comes from `.env.training` per the D6 secrets convention.

Risks, owned by F1/F3: TRL, PEFT, and Unsloth support for the PLE
architecture must be verified in the container smoke test before any long
run — target-module selection for LoRA on a PLE model may differ from
vanilla decoder-only models. If Gemma 4 E-models prove unsupported by a
required library at implementation time, the documented fallback pair is
gemma-4-E4B full-precision teacher with a supported dense student of similar
size; record the deviation in this document.

Memory envelope (to be confirmed by the D4 calculator, which gains these as
reference configs): E2B full fine-tuning and E4B QLoRA fit comfortably in
32GB; E4B full fine-tuning with bf16 optimizer state requires ZeRO-Offload
and is supported but not the reference path. In-process logit distillation
holds the E4B teacher (frozen, no optimizer state, optionally 4-bit) plus
the E2B student with optimizer state; the calculator must confirm the fit
before F5 acceptance runs.

## Build order

Dependencies among deliverables, not calendar:

```
D5 schema extension (F2) ── F1 SFT data prep ── F3 train-sft ──┬── F6 export-gguf
                                                               │
F4 teacher harness ── F5 distillation harness ─────────────────┤
                                                               │
                                                               └── F7 eval_ft ── F8 recipe cards
```

F2 (schema) comes first for the same reason D5 did: it is the contract.
F1 before F3 because the trainer consumes prepared data. F4 before F5
because response-level distillation trains on F4's artifacts. F8 is last:
recipe cards are validated by actually running them through F3/F5/F6/F7.

## F1: Chat and instruction data preparation

**Tag:** build. **Depends on:** D1, D5 (F2 schema). Module:
`training/data/sft.py`.

### Purpose

Turn chat and instruction datasets into tokenized, loss-masked training
examples correctly for each supported model family. This is the highest-risk
correctness surface in Phase 1: a wrong template or mask silently destroys a
fine-tune while the loss curve looks healthy.

### Decisions

1. **Canonical record format.** All SFT data normalizes to the OpenAI-style
   messages list: `{"messages": [{"role": ..., "content": ...}, ...]}`, one
   JSON object per line. Loaders accept Hugging Face datasets or local JSONL
   in this shape, plus a converter for prompt/completion pairs. Multi-turn is
   the native case, single-turn a special case of it.
2. **Per-model chat-template registry.** `training/data/templates.py` maps a
   model family to (a) the template application function and (b) the
   assistant-span locator used for masking. Default implementation delegates
   to the tokenizer's own `apply_chat_template`; the registry exists because
   masking needs to know where assistant spans begin and end, which the
   generic template API does not expose uniformly. Registered families in
   Phase 1: `gemma4` (including thinking-control tokens) and `llama3`
   (second family to prove the registry is not a single-purpose shim).
   Unknown families fail loudly with the registration instructions; there is
   no silent generic fallback.
3. **Prompt-loss masking.** Labels are the input ids with every non-assistant
   token set to -100, computed from the template registry's span locator, not
   from string matching on the rendered text. Masking covers system and user
   turns, the assistant role header, and template control tokens; loss
   applies to assistant content tokens and the turn's end-of-turn token only.
   A `mask_debug(sample)` helper renders a token-by-token table
   (token, id, label) for eyeball verification; the unit tests assert exact
   mask boundaries per family against golden fixtures.
4. **Sequence packing.** Multiple short examples pack into one
   `max_seq_len` sequence with correct attention separation
   (position ids reset per example; use TRL's packing with
   flash-attention varlen support rather than a custom implementation).
   Packing on/off is a recipe flag; masking correctness must hold in both
   modes and is tested in both.
5. **Artifacts, not on-the-fly surprises.** `scripts/train/prepare_sft.py`
   materializes the tokenized, masked dataset to
   `/mnt/ai-data/training/datasets/sft/<name>/` with a manifest (source,
   license, tokenizer hash, template registry version, example counts,
   token counts, held-out split seed). The trainer can also tokenize
   on the fly for small datasets, but recipe cards reference materialized
   artifacts so runs are reproducible. A held-out split (default 2 percent,
   seeded) is always reserved for F7's held-out loss.

### Interfaces

```
scripts/train/prepare_sft.py \
    --source HuggingFaceH4/ultrachat_200k --split train_sft \
    --model google/gemma-4-E4B \
    --out /mnt/ai-data/training/datasets/sft/ultrachat-gemma4 \
    --max-seq-len 4096 --holdout 0.02
```

`make prepare-sft ARGS=...` wraps it. Library surface:
`load_sft_dataset(recipe.sft.data)` returning train and held-out splits ready
for the trainer.

### Acceptance

- Golden-fixture tests per registered family assert exact label masks for
  single-turn, multi-turn, and system-prompt cases, packed and unpacked.
  This is the scope document's required unit test: loss applies only to
  completion tokens.
- A round-trip test detokenizes a prepared sample and reproduces the
  template-rendered conversation exactly (no lost or duplicated control
  tokens).
- Re-running `prepare_sft.py` with the same inputs is deterministic
  (identical manifest hashes), including the held-out split membership.
- An unknown model family fails with the registration message.

## F2: Parameter-efficient options in the recipe schema

**Tag:** adopt. **Depends on:** D5. Module: extends
`training/recipes/schema.py`.

### Purpose

Expose LoRA, QLoRA, and DoRA through the recipe schema so every Phase 1 entry
point consumes one config surface. Adoption of Hugging Face PEFT and
bitsandbytes; the only code here is schema, validation, and construction glue.

### Decisions

1. **New `peft` group** in the schema: `enabled`, `method: lora|dora`,
   `rank`, `alpha`, `dropout`, `target_modules` (list or `"auto"`),
   `quantize_base: none|nf4` (nf4 via bitsandbytes = QLoRA),
   `double_quant`, `compute_dtype`. DoRA is `method: dora`
   (PEFT `use_dora=True`), valid with and without base quantization.
2. **`target_modules: "auto"`** resolves per model family from the same
   registry file as F1's templates (one place that knows each family's
   architecture), so a PLE-model quirk is fixed once. The resolved list is
   written into the run's config snapshot; "auto" never appears in a
   snapshot.
3. **Validation invariants** added to D5's `validate()`: `quantize_base: nf4`
   requires `peft.enabled` (no full fine-tuning of a 4-bit base);
   fp8 precision and nf4 base quantization are mutually exclusive;
   `optimizer: cpu_adam`/offload combinations follow D5's existing rules.
   The D4 calculator gains a QLoRA-aware memory model (frozen 4-bit base +
   trainable adapter + optimizer state on adapter only).
4. **Construction glue** in `training/peft_utils.py`:
   `build_peft_model(model, recipe.peft)` returning the wrapped model, and
   `trainable_parameter_report(model)` logged at run start (count and
   percentage; a QLoRA run that reports 100 percent trainable is a
   misconfiguration caught immediately).

### Acceptance

- Loading a recipe with each of LoRA, QLoRA, and DoRA produces a typed
  config; invalid combinations fail in `validate()` with clear messages,
  each covered by a unit test.
- `build_peft_model` on gemma-4-E4B with `target_modules: "auto"` yields a
  model whose trainable-parameter percentage is under 5 percent and whose
  resolved module list lands in the config snapshot.
- The D4 calculator's QLoRA prediction for the F8 SFT recipe is within
  10 percent of measured peak VRAM.

## F3: Supervised fine-tuning entry point

**Tag:** orchestrate. **Depends on:** D1, D5/F2, D6, D10, F1. Entry:
`make train-sft CONFIG=...`, script `scripts/train/train_sft.py`, logic in
`training/sft.py`.

### Purpose

One command that takes a recipe and runs a supervised fine-tune on TRL,
supporting full fine-tuning and QLoRA, wired into the Phase 0 tracking,
checkpointing, and watchdog machinery.

### Decisions

1. **TRL `SFTTrainer` is the trainer.** The D6 reference loop is not used for
   SFT. The script builds: tokenizer and model from the recipe (with F2 PEFT
   wrapping), dataset via F1's loader, `SFTConfig` mapped from the recipe
   (sequence length, packing, batch and accumulation, schedule, precision),
   and runs `trainer.train()` with resume support.
2. **Unsloth is an optional backend, off by default.**
   `sft.backend: trl|unsloth` in the recipe. The unsloth path swaps model
   loading to `FastLanguageModel` and keeps everything else identical.
   Unsloth pins tightly to specific torch/transformers versions and its
   Gemma 4 PLE support must be verified in the smoke test; it is a
   throughput optimization, documented as may-lag, and its failures are not
   release blockers. Every tracking run records the backend used.
3. **Checkpointing and resume** go through TRL/transformers' native
   checkpoint mechanism into the D6 directory layout
   (`/mnt/ai-data/training/checkpoints/<run>/`), with D6's retention applied
   by a `TrainerCallback` after each save. Adapter-only checkpoints (PEFT)
   are the default for QLoRA runs; `meta.json` records whether a checkpoint
   is adapter-only or full, because F6 needs to know.
4. **Watchdog and tracking attach as callbacks** (D10's TRL integration
   path). Metrics: loss, learning rate, grad norm, tokens per second,
   trainable-parameter report, GPU telemetry.
5. **Held-out loss is computed during training** on F1's held-out split at
   `eval_steps` intervals, so F7's scorecard has a training-time curve to
   reference, not just a final number.

### Interfaces

- `make train-sft CONFIG=recipes/sft-qlora-gemma-e4b.yaml` (Hydra overrides
  pass through: `make train-sft CONFIG=... EXTRA="optimizer.lr=1e-4"`).
- Preflight, in order: recipe `validate()`, D4 memory-fit check (warn or
  refuse per a `--force` flag), D9 lock acquisition, inference-services-down
  check.

### Acceptance

- A QLoRA fine-tune of gemma-4-E4B on a small instruction dataset completes
  end to end and logs to tracking (scope acceptance criterion, run via the
  F8 recipe card).
- A tiny SFT run (minutes) checkpoints, is killed, and resumes; loss
  trajectory matches D6's deterministic-resume bar.
- The same tiny run executes on both backends (`trl`, `unsloth`) or the
  unsloth failure is documented; tracking records the backend.
- Preflight refuses to start while llama-gpu is up, and releases the lock on
  every exit path.

## F4: Teacher and synthetic data harness

**Tag:** orchestrate. **Depends on:** D7 (serving swap), D9 (lock), the
inference services. Module: `training/data/teacher.py`, script
`scripts/train/generate_teacher.py`.

### Purpose

Batch-generate teacher responses and synthetic data through the existing
inference services, producing dataset artifacts that later training consumes.
Runs as its own phase: inference up, training idle.

### Decisions

1. **distilabel is the pipeline engine**, pointed at the local
   OpenAI-compatible endpoints (vllm-gpu 8003 preferred for batch throughput;
   llama-gpu 8002 as fallback). The harness contributes: endpoint
   configuration and health preflight, the serving-swap step (reuse D7's
   machinery to put the intended teacher model up, restore afterward),
   concurrency tuned to the service (vLLM takes high request concurrency;
   llama.cpp does not), retry/backoff, and resumability — generation state
   checkpoints so a multi-hour run over a large prompt set survives
   interruption without redoing completed prompts.
2. **Two built-in pipeline shapes**, both thin distilabel graphs:
   `respond` (prompts in, teacher responses out — the distillation dataset)
   and `synthesize` (seed instructions in, evolved instruction/response
   pairs out, self-instruct style). Custom distilabel pipelines can be
   dropped in; the harness owns I/O and serving, not pipeline authorship.
3. **Output is a normalized F1-format dataset** under
   `/mnt/ai-data/training/datasets/distill/<name>/` (messages-format JSONL
   plus manifest recording teacher model and quantization, service,
   generation parameters, prompt-set hash, distilabel version, timestamps).
   The manifest's teacher-identity block is what F7 cites when claiming a
   student improved against a given teacher.
4. **Sampling parameters are explicit** in the harness config (temperature,
   top-p, max tokens, optional thinking-mode control for Gemma 4 teachers);
   defaults favor distillation (low temperature, long max tokens). The
   config is snapshotted into the manifest.

### Interfaces

```
scripts/train/generate_teacher.py \
    --pipeline respond \
    --prompts /mnt/ai-data/training/datasets/sft/ultrachat-gemma4 \
    --teacher gemma-4-E4B-it-Q8_0.gguf --service vllm-gpu \
    --out /mnt/ai-data/training/datasets/distill/e4b-responses-v1
```

`make generate-teacher ARGS=...` wraps it.

### Acceptance

- A 1,000-prompt `respond` run against vllm-gpu completes, writes a valid
  F1-format dataset with a complete manifest, and restores prior serving
  state.
- Killing the run at 50 percent and re-invoking completes the remaining
  prompts only (resume proof).
- The harness refuses to start while the D9 training lock is held.
- A `synthesize` run from 50 seed instructions produces a dataset that
  `prepare_sft.py` ingests without modification.

## F5: Distillation harness

**Tag:** build. **Depends on:** F1, F2, F3, F4. Modules:
`training/data/distill.py`, `training/distill.py`, script
`scripts/train/distill.py`, entry `make train-distill CONFIG=...`.

### Purpose

Train a small student from a larger teacher. Two tiers, reflecting what the
serving stack can and cannot provide:

1. **Response-level distillation** — SFT on teacher outputs captured by F4
   through the inference services. Any teacher the services can serve,
   including quantized 26B-class models.
2. **Logit distillation** — token-level KL against full teacher logits,
   which HTTP serving cannot provide, so the teacher runs in-process in
   PyTorch. Limited to teachers that fit alongside the student
   (reference: frozen gemma-4-E4B teacher, gemma-4-E2B student).

### Decisions

1. **Response-level is a thin specialization of F3**: a recipe whose dataset
   is an F4 artifact. No new trainer. The recipe's `distill` block records
   the teacher-manifest reference so the run's provenance chain is complete.
   This tier ships first and de-risks the deliverable.
2. **Logit tier requires a shared tokenizer.** Teacher and student must have
   identical tokenizers (hash-checked at startup, per the D3 warning that
   cross-tokenizer distillation breaks logit KL). Gemma-4 E4B and E2B share
   one; cross-family logit distillation is out of scope for Phase 1 and the
   error message says so.
3. **Loss**: `L = alpha * CE(student, labels) + (1 - alpha) * T^2 * KL(student/T || teacher/T)`,
   with `alpha` and temperature `T` in the recipe's `distill` block, KL
   computed only where labels are unmasked (assistant tokens, reusing F1's
   masking). Forward KL is the default; reverse KL is a flag. Implemented as
   a subclass of TRL's trainer overriding `compute_loss`; if the installed
   TRL version ships a maintained distillation trainer (GKD), prefer
   adopting it and record the choice — the recipe surface stays the same
   either way.
4. **Teacher execution modes**, per recipe: `online` (teacher forward pass
   per batch, frozen, `torch.no_grad`, optionally nf4-quantized to save
   VRAM) and `cached` (a pre-pass materializes teacher top-K logprobs per
   token — default K=64, K in the recipe — to disk; training reads the
   cache and computes KL over the top-K support with a renormalization
   correction). Online is exact and simple; cached frees the VRAM and turns
   teacher compute into a phase, matching the concurrency philosophy. Both
   modes hash-stamp the teacher revision into the run.
5. **Memory-fit before run**: preflight calls D4 with the combined
   teacher+student model (online mode); the F8 distill recipe is the
   calibration case.

### Acceptance

- Masked-KL unit test: KL contributes zero on prompt tokens (golden fixture
  reusing F1's masks).
- Online and cached modes on a tiny model pair produce loss values that
  agree within a documented tolerance attributable to top-K truncation.
- Tokenizer-mismatch preflight fails with the documented message.
- The scope criterion, executed via the F8 recipe: a distilled gemma-4-E2B
  measurably improves on a target evaluation (F7 scorecard) over its
  pre-distillation baseline.

## F6: Adapter merge and serve

**Tag:** orchestrate. **Depends on:** D7, F3/F5 checkpoints. Entry:
`make export-gguf CHECKPOINT=... [QUANT=Q8_0]`.

### Purpose

Take a Phase 1 checkpoint (adapter-only or full), produce a served GGUF, and
feed the D7 round trip. Mostly wiring over D7's export pipeline.

### Decisions

1. **Adapter-into-base merging stays PEFT's `merge_and_unload`** inside D7's
   `export_gguf.py`, which already handles it. F6 adds: resolving an
   adapter-only checkpoint to its base model (recorded in the checkpoint's
   `meta.json` by F3), and a preflight that the base revision at export time
   hash-matches the one trained against.
2. **mergekit enters for multi-adapter and multi-model merges only**
   (`scripts/export/merge_models.py`, `make merge CONFIG=...`), driven by a
   standard mergekit YAML kept in `training/export/merges/`. Output is an HF
   directory that feeds the same GGUF pipeline. Phase 1 ships the tool and
   one validated example (two LoRA adapters merged via TIES onto one base);
   broad merge recipes are future work.
3. **Export manifest** (D7's) gains fields: adapter path and hash, base
   model and revision, merge method if any. Nothing else about D7's flow
   changes: pinned llama.cpp, quantize, write under
   `/mnt/ai-data/models/trained/<run>/`.
4. **llama.cpp pin check**: the export preflight verifies the pinned
   revision supports the model architecture (Gemma 4 requires a post-April-
   2026 pin) using D7's existing architecture-support check.

### Acceptance

- An adapter-only QLoRA checkpoint from F3 exports to Q8_0 GGUF, serves on
  llama-gpu via D7's swap, and answers the smoke prompts; serving state is
  restored.
- A full-fine-tune checkpoint exports through the same command with no
  adapter step (the tool detects checkpoint type from `meta.json`).
- The two-adapter mergekit example produces a servable GGUF and a manifest
  recording the merge method.
- Re-export of the same checkpoint is hash-identical (D7's reproducibility
  bar holds through the new path).

## F7: Fine-tuning evaluation wrapper

**Tag:** build. **Depends on:** D7 (serve_eval), F1 (held-out split), F6.
Script: `scripts/eval/eval_ft.py`, logic in `training/eval/`.

### Purpose

Answer "did the fine-tune help" with one command producing a base-versus-
fine-tuned scorecard from three signal types: held-out loss, instruction
benchmarks, and model-as-judge scoring.

### Decisions

1. **Three evaluators behind one runner**, each independently skippable:
   - **Held-out loss** — perplexity/loss on F1's held-out split, computed
     in-process in the training container (GPU, inference down) for both
     base and fine-tuned checkpoints at matched precision.
   - **lm-evaluation-harness** — a pinned task set in
     `training/eval/tasks.yaml` (default: ifeval, gsm8k, mmlu subset,
     hellaswag; pinned lm-eval version and task revisions in the manifest).
     Runs in-process against the HF checkpoint (before quantization), so
     benchmark numbers measure the fine-tune, not the quant; the scorecard
     labels them accordingly.
   - **Model-as-judge** — pairwise comparison on a fixed prompt set
     (`training/eval/judge_prompts.jsonl`, seeded from public sets, held in
     git): base and fine-tuned GGUFs each generate through the serving
     stack (D7 swap, matched quantization per the confound rule), then a
     judge model served on vllm-gpu scores pairs with position-swapped
     duplicates to cancel order bias. Judge model is recipe-configured;
     default is the strongest locally servable model (a quantized Gemma 4
     26B A4B or similar — pick at implementation and pin in the config).
     Verdicts: win/loss/tie plus rationale, all persisted.
2. **Phasing**: held-out loss and lm-eval run inference-down; generation and
   judging run inference-up. The runner sequences these itself, using D7's
   swap and D9's lock, so one command is still honest about the concurrency
   policy.
3. **Scorecard artifact**: `scorecard.json` in the run directory, extending
   D7's evaluation JSON shape — metadata block (models, revisions, quant
   types, judge identity and prompt hash, lm-eval pins, git SHA), then one
   section per evaluator with base, fine-tuned, and delta; plus a rendered
   `scorecard.md` table. The JSON is the machine-readable record the scope's
   acceptance criteria are checked against.
4. **Distillation mode**: `--baseline` accepts any checkpoint or HF model,
   so a student is compared against its pre-distillation self (the F5
   criterion) with the same machinery as base-versus-fine-tuned.

### Interfaces

```
scripts/eval/eval_ft.py \
    --base google/gemma-4-E4B \
    --tuned /mnt/ai-data/training/checkpoints/<run>/latest \
    --gguf-quant Q8_0 --judge-service vllm-gpu \
    --evaluators holdout,lmeval,judge
```

`make eval-ft ARGS=...` wraps it.

### Acceptance

- On a deliberately-good case (the F8 SFT run), the scorecard shows the
  fine-tuned model beating its base on held-out loss and judge win rate;
  this is the scope's serve-and-scorecard criterion.
- A null test — evaluating a model against itself — produces deltas
  within noise and a judge win rate near 50 percent (validates the harness
  before trusting any real result).
- Judge scoring with position-swapped duplicates is verified by a unit test
  on the pairing logic; a judge-refusal or malformed verdict is counted and
  reported, never silently dropped.
- Interrupting the runner mid-judging leaves serving restored (inherits
  D7's kill-mid-swap bar).

## F8: Recipe cards

**Tag:** orchestrate. **Depends on:** all of the above. Files:
`training/recipes/sft-qlora-gemma-e4b.yaml`,
`training/recipes/distill-gemma-e2b.yaml`.

The scope document names these `sft-qlora-8b.yaml` and `distill-3b.yaml`;
the names change to match the chosen Gemma 4 models and the scope document
should be updated.

### Decisions

1. **`sft-qlora-gemma-e4b.yaml`** — google/gemma-4-E4B, QLoRA (nf4, rank and
   alpha tuned during validation), a public instruction dataset prepared via
   F1 (default: ultrachat-derived; pin the exact dataset and revision at
   implementation), sequence length 4096, packing on, cosine schedule,
   tracking on, watchdog defaults. Comments in the YAML state measured peak
   VRAM, wall-clock on this machine, and the scorecard result of the
   validation run — a recipe card is a validated artifact, not a template.
2. **`distill-gemma-e2b.yaml`** — student google/gemma-4-E2B, teacher
   google/gemma-4-E4B (instruction-tuned variant), two-stage as documented
   in the card: response-level pass on an F4 dataset, then a logit-KL pass
   (online mode, alpha and T recorded from validation). The card records
   the teacher dataset manifest it was validated against.
3. Both cards are exercised end to end (train, export, evaluate) as the
   final Phase 1 validation, and their scorecards are committed under
   `docs/training/scorecards/` as the reference results.

### Acceptance

- Each recipe loads through D5 validation and runs end to end via its entry
  point with no overrides.
- Committed scorecards demonstrate the two scope criteria: the SFT model
  beats its base, and the distilled student beats its pre-distillation
  baseline.

## Phase 1 exit checklist

Restating the scope document's acceptance criteria against the deliverables:

1. A QLoRA fine-tune of a small-model base (gemma-4-E4B) completes and logs
   to tracking (F3, F8).
2. A unit test confirms loss applies only to completion tokens under
   prompt-loss masking (F1).
3. A distilled student (gemma-4-E2B) measurably improves on a target
   evaluation over its pre-distillation baseline (F5, F7, F8).
4. The fine-tuned model is served through the inference services and
   returns a scorecard beating its base (F6, F7, F8).

Additional bars this document adds: the null-test scorecard (F7), resumable
teacher generation (F4), export reproducibility through the adapter path
(F6), and both recipe cards validated with committed scorecards (F8).

---

*Last updated: 2026-07-09. Written against the Phase 0 product designs;
implement only after Phase 0's exit checklist passes.*
