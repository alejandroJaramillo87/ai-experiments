# P2-D5: Optimizer offload — implementation investigation

Investigation of what implementing P2-D5 (optimizer offload) entails, reconciling the
Phase 2 product design against the current state of the repository. Per the operating
rules, the repository is the source of truth; every point where the design document
disagrees with shipped code is called out and the repository convention wins. Findings
verified as of 2026-07.

## Table of contents

- [Scope summary](#scope-summary)
- [Current repository state](#current-repository-state)
- [Design-document drift](#design-document-drift)
- [Proposed design decisions](#proposed-design-decisions)
- [Resolved design decisions](#resolved-design-decisions)
- [Model family considerations](#model-family-considerations)
- [Module layout](#module-layout)
- [Acceptance criteria](#acceptance-criteria)
- [Open questions](#open-questions)

## Scope summary

P2-D5 adds the DeepSpeed ZeRO-Offload configuration that makes full-parameter
continual pre-training of the Gemma-4-E2B transformer stack fit on this workstation:
bf16 weights and gradients on the 32GB RTX 5090, fp32 master weights and Adam moments
in system RAM, the optimizer step running on the 9950X via DeepSpeed's CPUAdam op.
Tag: orchestrate. Depends on: Phase 0 D1 (CPUAdam op in the training image), D6
(reference loop and checkpoint layout), D9 (affinity, pinned memory), and P2-D2/D4
(real mixture/replay dataloader shape). The design document's core decisions:

1. ZeRO stage 2 with CPU optimizer offload (`offload_optimizer: {device: cpu,
   pin_memory: true}`), single GPU. Stage 3 and NVMe/ZeRO-Infinity offload are
   explicitly out of scope: the ~bf16 weights fit in VRAM.
2. The DeepSpeed config is generated from the recipe (`training/offload.py`), never a
   loose `ds_config.json`. Surfaced knobs: gradient accumulation, allgather/reduce
   bucket sizes (default 2e8), `pin_memory`.
3. Frozen parameters (embeddings, LM head, per-layer embedding tables) stay out of
   DeepSpeed's optimizer partitioning: a `trainable_filter` is applied before engine
   initialization.
4. Host-RAM budget is computed by the D4 calculator and enforced by `validate()`
   against 128GB minus the watchdog RAM floor, with a calibration recipe
   `reference/continual-gemma4-e2b.yaml`.
5. CPU placement: main process plus CPUAdam threads on CCD0 (cores 0-7), dataloader
   workers plus checkpoint writer on CCD1 (8-15), via the D9 wrapper; measured, and
   parked off by default if it does not help.
6. Checkpointing uses DeepSpeed's checkpoint API inside the D6 directory layout; the
   deterministic-resume test must pass with offload enabled, first on a tiny model.

## Current repository state

The recipe- and calculator-side hooks for offload already exist; the execution path
does not. Verified state:

- `training/recipes/schema.py`: `OPTIMIZER_NAMES` includes `cpu_adam` and
  `OFFLOAD_MODES` includes `zero_offload`; `MemoryConfig.offload` defaults to
  `none`. There are no bucket-size or `pin_memory` knobs under `memory`, and no
  `trainable_filter` field anywhere in the schema.
- `training/recipes/validate.py:85` already cross-checks `cpu_adam` against
  `memory.offload=zero_offload`. The host-RAM budget check at line 442 covers only
  `replay.max_gb`; it does not add offloaded optimizer state to the budget.
- `training/model.py:106`: `build_optimizer()` raises on `cpu_adam` with a comment
  deferring the branch to a DeepSpeed engine. `build_model()` has no freeze logic;
  all parameters are trainable unless PEFT wraps the model.
- `training/loop.py`: plain single-GPU loop — `loss.backward()`, manual
  `clip_grad_norm_`, `optimizer.step()`. Checkpoint meta hardcodes
  `"checkpoint_format": "torch"` with the comment "DeepSpeed runs record their own",
  anticipating exactly this deliverable. No engine branch exists.
- `training/checkpoint.py`: atomic `step_XXXXXXXXX/` directories with
  `model/` (safetensors), `optimizer.pt`, `scheduler.pt`, `dataloader.json`,
  `rng.pt`, `meta.json`. No DeepSpeed-format path.
- `training/calculator/components.py`: `OFFLOAD_HOST_BYTES_PER_PARAM = 16`
  (fp32 optimizer states 8 B/param + fp32 master params 4 B/param + pinned fp32
  gradient transfer buffers 4 B/param). `optimizer_bytes()` returns 0 VRAM and
  `host_ram_bytes()` returns `trainable * 16 + replay budget` when
  `offload == "zero_offload"` or the optimizer is `cpu_adam`. Gemma 4 shapes
  (per-layer embedding tables, `gemma4`/`gemma4_text` model types) are already
  modeled. `trainable_param_count()` knows only two cases: PEFT adapter or all
  parameters — there is no freeze-policy case.
- `training/affinity.py` (D9): `resolve_masks`/`apply_mask`/`worker_init_fn` with
  defaults main `0-7`, workers `8-15`, `include_smt: true`, off by default. This
  matches the design document's decision 5 layout as-is.
- `training/watchdog.py`: RAM-floor alert at `ram_floor_gb` (default 8.0) and abort
  below half of it, already wired into the loop with emergency checkpointing.
- `docker/training/Dockerfile.training`: DeepSpeed ops are pre-built at image build
  time with `DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1` and Zen 5 flags (Phase 0 D1 is
  done). `pyproject.toml` pins `deepspeed==0.19.2` (released 2026-06-16; current
  stable as of this investigation).
- `training/recipes/reference/`: no `continual-gemma4-e2b.yaml`; the closest shapes
  are `pretrain-1b.yaml` (scratch, adamw8bit) and `tiny-e2e.yaml`.
- `training/families.py`: `gemma4`/`gemma4_text` and `qwen3` are registered
  families with target-module lists; per-layer embeddings are acknowledged for the
  Gemma 4 E-models.

## Design-document drift

Points where the design document does not match shipped code; the repository wins.

1. **Host-RAM formula.** The document says "12 x trainable params + pinned gradient
   buckets + replay.max_gb + a page-cache allowance". The shipped calculator uses
   `OFFLOAD_HOST_BYTES_PER_PARAM = 16`, which already folds the pinned gradient
   buffers into the per-parameter constant, and `host_ram_bytes()` already adds the
   replay budget. The remaining work is not "extend the calculator" but "make
   `validate()` consume `host_ram_bytes()` for the offload warning" — the
   document's framing predates the merged P2-D4 calculator work. A page-cache
   allowance term is genuinely absent and is a design decision below.
2. **`trainable_filter` is not a shipped convention.** The document references it as
   an existing cross-cutting convention ("the trainable_filter (cross-cutting
   conventions) is applied before engine initialization"). No such field, function,
   or pattern exists in the repository. It must be designed and built inside P2-D5
   (schema field, freeze application in `build_model` or a new helper, and a
   `trainable_param_count()` case in the calculator), or split out as a
   prerequisite. This is the largest unstated piece of work in the deliverable.
3. **Bucket-size knobs.** The document lists allgather/reduce bucket sizes as recipe
   knobs; `MemoryConfig` has neither. New fields are required (with the 2e8
   default), plus `pin_memory` for the offload block — note the existing
   `data.pin_memory` is the dataloader's flag, a different thing.
4. **Checkpoint format.** The document says the D6 layout "already anticipated"
   DeepSpeed via a metadata note. What is actually shipped is one meta field
   (`checkpoint_format: "torch"`) and a comment; `checkpoint.py` has no DeepSpeed
   branch, and the atomicity contract (tmp-dir + fsync + rename) must be
   re-established around `engine.save_checkpoint()`, which writes its own tag
   directories.
5. **Parameter counts are unverified.** The document states ~5.1B total / ~2.3B
   trainable for Gemma-4-E2B with the freeze policy. Public sources (transformers
   `configuration_gemma4.py`) confirm the architecture (30 layers, hidden 2304,
   262144 vocab, per-layer input tables packed as
   `[vocab_size, num_layers * hidden_size_per_layer_input]`, tied word embeddings)
   but not the E2B envelope numbers. Treat the counts as calculator outputs to be
   asserted within a tolerance envelope at calibration, not as fixed constants.

## Proposed design decisions

1. **Engine integration as a branch inside the existing loop, not a second loop.**
   `_train_locked()` gains an offload path: when `memory.offload == "zero_offload"`,
   build the model, apply the trainable filter, call `deepspeed.initialize()` with
   the generated config and only the trainable parameter groups, and route
   forward/backward/step through `engine.backward(loss)` / `engine.step()`.
   Gradient accumulation moves into the DeepSpeed config
   (`gradient_accumulation_steps`) so the engine owns boundary semantics; the
   loop's manual `clip_grad_norm_` is replaced by the config's `gradient_clipping`
   for the engine path (the current loop clips at infinity, i.e. measures only —
   the engine's `get_global_grad_norm()` supplies the metric instead).
2. **`training/offload.py` generates the DeepSpeed config dict** from the recipe:
   ZeRO stage 2, `offload_optimizer: {device: cpu, pin_memory: <recipe>}`,
   bucket sizes, bf16 enabled from `precision.dtype`, `gradient_clipping`, and the
   CPUAdam optimizer block from `optimizer.*`. Unit-testable as a pure function
   (recipe in, dict out), no GPU needed.
3. **`trainable_filter` as a schema field** (`model.trainable_filter`, values
   `all` (default) and `transformer_stack`), applied by a helper that sets
   `requires_grad = False` on parameters matching frozen patterns
   (`embed_tokens`, tied LM head, per-layer embedding/projection tables) and
   returns the trainable parameter list for the engine. The calculator's
   `trainable_param_count()` gains the matching case so VRAM/host-RAM predictions
   and validate() warnings track the same policy. A unit test asserts no
   embed/per-layer-embedding pattern appears in the optimizer groups and the
   trainable count falls within the expected envelope.
4. **Frozen parameters never reach `deepspeed.initialize()`'s optimizer groups.**
   ZeRO-2 with frozen parameters and CPU offload has documented sharp edges in
   DeepSpeed (gradient-bucket population bugs when the autograd graph touches
   parameters absent from the partitioner, for example deepspeedai/DeepSpeed issue
   #7595); passing an explicit trainable-only parameter list is the mitigation,
   and the tiny-model resume test is the guard.
5. **Checkpointing:** on the engine path, `engine.save_checkpoint(tmp_dir, tag)`
   writes into the same atomic tmp-dir protocol as `save_checkpoint()`, alongside
   the existing `dataloader.json`, `rng.pt`, and `meta.json`
   (`checkpoint_format: "deepspeed"`); the HF `model/` export remains so D7
   export keeps working from any retained checkpoint. Resume calls
   `engine.load_checkpoint()` and then restores dataloader/RNG state exactly as
   today. Retention and emergency-checkpoint semantics are unchanged.
6. **validate() host-RAM enforcement:** when offload is active, warn when
   `host_ram_bytes()` (which already includes the replay budget) plus a fixed
   page-cache allowance exceeds `128GB - watchdog.ram_floor_gb`. Proposed
   allowance: 16GB, recorded as a named constant next to
   `OFFLOAD_HOST_BYTES_PER_PARAM` so calibration can tune it.
7. **Affinity defaults already match the design** (main 0-7 = CCD0 for the loop and
   CPUAdam threads, workers 8-15 = CCD1); P2-D5 adds no affinity code, only a
   calibration measurement with `affinity.enabled: true` vs `false`, keeping the
   D9 rule (off by default unless it measurably helps).
8. **Calibration recipe `reference/continual-gemma4-e2b.yaml`:** Gemma-4-E2B at a
   pinned revision, `trainable_filter: transformer_stack`, seq 4096,
   `optimizer.name: cpu_adam`, `memory.offload: zero_offload`, mixture + replay
   0.2 per P2-D2/D4, wsd schedule per P2-D3, watchdog on. Run through the D4
   calibration harness; predictions within 10 percent of measured.
9. **Step-time instrumentation:** the engine path logs an optimizer-wait metric
   (wall time of `engine.step()` minus GPU compute) so the CPUAdam-bound case is
   visible; the documented remedy order is raise `grad_accum` first.

## Resolved design decisions

Reviewed and resolved 2026-07:

1. **`model.trainable_filter` (string enum, values `all` and `transformer_stack`,
   default `all`).** The freeze policy is a property of how the model is built, so
   it lives in the `model:` block. Family-specific frozen-parameter patterns are
   hardcoded in `training/offload.py`, not recipe fields, so the calculator can
   always price the policy; promoting the string to a block is a mechanical
   migration if a finer policy is ever needed.
2. **The LR scheduler stays loop-owned (external to DeepSpeed).** The engine is
   initialized without a scheduler; the loop steps the P2-D3 scheduler once per
   accumulation boundary (once per outer step, matching the config's
   `gradient_accumulation_steps`). `scheduler.pt` save/restore stays uniform
   across the plain and engine paths. Contingency: if the tiny-model
   deterministic-resume test exposes an ordering problem, switch to
   engine-owned.
3. **Page-cache allowance starts at 16GB** (`PAGE_CACHE_ALLOWANCE_BYTES` in
   `training/calculator/components.py`), corrected from calibration data when
   acceptance criterion 3 produces the measurement.
4. **The host-RAM budget check is a hard validation error, not a warning.** The
   design document says "warn", but the shipped convention for the equivalent
   check (`replay.max_gb` versus the host budget in `validate.py`) is a hard
   problem, and `validate()` has no warning channel. A recipe whose predicted
   host RAM cannot fit is a run that will abort at the watchdog floor;
   repository convention wins.

## Model family considerations

Gemma 4 is the primary target; Qwen3 is the fallback (standing project decision).

- **Gemma 4 E2B** (`model_type` `gemma4`/`gemma4_text`, shipped 2026-04): tied word
  embeddings (`tie_word_embeddings: true`) mean there is no separate `lm_head`
  weight to freeze — freezing `embed_tokens` covers both ends. The per-layer input
  tables are the dominant frozen mass: `vocab_size_per_layer_input` (262144) x
  `num_layers * hidden_size_per_layer_input`, which is why the freeze policy is
  what makes offload fit. The freeze filter must match the per-layer embedding
  parameter names as they appear in the transformers implementation, verified
  against the pinned revision at implementation time, not hardcoded from the
  design document.
- **Qwen3 fallback**: standard dense Llama-lineage shape, no per-layer embeddings,
  untied embeddings on larger variants. `trainable_filter: transformer_stack`
  must degrade cleanly: freeze `embed_tokens` and `lm_head` (when present), train
  the decoder stack. The calculator already models `qwen3`; the freeze-count unit
  test should run against both families.
- The chat template, thinking-mode, and prompt-formatting differences between the
  families are irrelevant to P2-D5 (raw token shards, no chat formatting); they
  matter again at P2-D6/D7 evaluation and serving.

## Module layout

- `training/offload.py` — recipe-to-DeepSpeed-config generation, trainable-filter
  application, engine construction helper. New.
- `training/loop.py` — engine branch in `_train_locked()`; engine checkpoint
  save/load wiring; optimizer-wait metric. Modified.
- `training/checkpoint.py` — DeepSpeed-format save/load inside the atomic
  directory protocol. Modified.
- `training/recipes/schema.py` — `model.trainable_filter`, `memory` bucket/pin
  knobs. Modified.
- `training/recipes/validate.py` — offload host-RAM warning via
  `host_ram_bytes()`; trainable-filter value check. Modified.
- `training/calculator/components.py` — `trainable_param_count()` freeze case;
  page-cache allowance constant. Modified.
- `training/recipes/reference/continual-gemma4-e2b.yaml` — calibration recipe. New.
- `tests/training/` — config-generation unit tests, freeze-filter parameter-group
  tests, tiny-model deterministic-resume test with offload enabled (CPU-sandbox
  parts unit-tested; the resume and calibration runs are host-side).

## Acceptance criteria

From the design document, restated against the repository:

1. The tiny-model offload run passes the existing deterministic-resume acceptance
   test with `memory.offload: zero_offload` (host-side; run first, before any
   Gemma-scale attempt).
2. Gemma-4-E2B with `trainable_filter: transformer_stack` trains 500+ steps at seq
   4096 without VRAM OOM; peak VRAM, host RAM, and tokens/s recorded as a
   benchmark JSON (Phase 0 D8 schema) under `docs/training/optimizations/`.
3. Calculator host-RAM and VRAM predictions for `continual-gemma4-e2b.yaml` land
   within 10 percent of measured (D4 calibration harness).
4. The watchdog RAM-floor alert fires when `replay.max_gb` is set adversarially
   high against a lowered floor (integration test; the watchdog and replay
   plumbing for this already exist from P2-D4/D10).
5. Unit tests: trainable-parameter count within the expected envelope; no frozen
   pattern in optimizer groups; generated DeepSpeed config matches the recipe
   knobs.

## Open questions

1. **Where does `trainable_filter` live in the schema?** Proposed:
   `model.trainable_filter` beside the hf/scratch blocks, since it is a property
   of how the model is built, not of memory management. Alternative: a
   `freeze:` block for future finer-grained policies.
2. **Scheduler ownership under DeepSpeed.** The engine can own the LR scheduler or
   the loop can keep stepping the existing `training/schedules.py` object
   externally. Keeping it external preserves the P2-D3 wsd implementation and its
   tests unchanged; proposed default, to be confirmed against
   `engine.load_checkpoint()` resume behavior in the tiny-model test.
3. **Page-cache allowance size** (proposed 16GB) — a calibration output more than a
   design input; acceptable to start at 16GB and adjust from measured data.
4. **DeepSpeed version policy.** `deepspeed==0.19.2` is pinned and current; any
   bump triggered by frozen-parameter or resume bugs follows the pin-then-fix-
   then-update rule, one bump at a time with the tiny-model resume test as the
   acceptance check.
