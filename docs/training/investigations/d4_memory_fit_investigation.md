# D4 memory-fit calculator: implementation investigation

**Frozen record — 2026-07-14. Drift against the current repository is expected and
correct; this document is never updated.**

Comparison of the Phase 0 D4 design (roadmap document, "D4: Memory-fit calculator")
against the current repository state, with a work breakdown, acceptance checks,
deviations, and open questions. Per operating rules, the repository is the source of
truth; where the roadmap document disagrees with the repository, this investigation
records the repository state and treats the document as advisory.

## Table of contents

- [Design summary](#design-summary)
- [Current repository state](#current-repository-state)
- [Deviations from the document](#deviations-from-the-document)
- [Memory model details](#memory-model-details)
- [Work breakdown](#work-breakdown)
- [Acceptance checks](#acceptance-checks)
- [Risks](#risks)
- [Resolved decisions](#resolved-decisions)

## Design summary

D4 answers, before a run starts, whether a configuration fits in 32GB VRAM and what
the maximum sequence length and batch size are. It must be accurate on this machine,
not generally. The document specifies:

1. Peak VRAM estimated as the sum of five components, all in bytes: weights
   (including NF4 packing plus quantization constants for QLoRA-style 4-bit bases),
   gradients (`P_trainable * bytes(grad_dtype)`), optimizer state (AdamW fp32 = 8
   bytes/param plus fp32 master weights under mixed precision; 8-bit Adam = 2
   bytes/param plus per-block constants; zero when ZeRO-offloaded, in which case the
   host RAM cost is reported instead), activations (analytic per-layer formula for
   the dense attention + SwiGLU family, GQA-aware, flash-attn assumed so attention
   memory is linear in sequence length, with a gradient-checkpointing mode), and
   framework overhead (calibrated constant `overhead_gb`, initial value 1.5).
2. Two entry results: `estimate(recipe)` returns the numbers for the given
   batch/seq; `solve(recipe, axis="batch"|"seq")` binary-searches the largest value
   whose estimate fits. The fits flag tests estimated-reserved against 32GB minus a
   1GB default safety margin.
3. Input is a recipe file (D5 schema), never a parallel argument set. Architecture
   comes from the recipe `model.scratch` fields or, for HF checkpoints, AutoConfig
   with a mapping table; unknown architectures fail loudly.
4. Calibration harness in-tree: `training/calculator/calibrate.py` runs a short real
   training loop, records `torch.cuda.max_memory_allocated/reserved` and NVML peak,
   and writes a calibration record JSON. The two reference configurations live in
   `training/recipes/reference/`.
5. Output: human table to stdout plus `--json`; fields are per-component bytes,
   totals (allocated, reserved-estimate), max seq at given batch, max batch at given
   seq, fits flag, host RAM estimate when offload is on.
6. Interfaces: `make memfit CONFIG=...`, `python -m training.calculator --config ...
   [--solve batch|seq] [--json]`, package
   `training/calculator/{__init__.py,components.py,solve.py,calibrate.py}`.

Acceptance per the document: within 10 percent of measured peak on the two reference
configurations (component sum versus `max_memory_allocated`; the reserved estimate
must not under-predict the NVML peak); the 1B reference config reported as fitting
in 32GB without offload, confirmed by calibration; unit tests pin the closed-form
component numbers for a hand-computed tiny config, no GPU needed.

## Current repository state

| Area | State |
|------|-------|
| D5 recipe system (the input contract) | Done and merged: `training/recipes/{schema.py,loader.py,validate.py}`, struct-mode load with dotlist overrides, `validate(recipe)` collecting all violations |
| Reference recipes | Done: `reference/pretrain-150m.yaml` (SmolLM2-135M shape, bf16, adamw, no checkpointing), `reference/pretrain-1b.yaml` (Llama-3.2-1B shape, adamw8bit, gradient checkpointing on), `reference/tiny-e2e.yaml` — exactly the two calibration configs the document asks for, already shipped |
| Schema fields D4 needs | Present: `model.scratch` carries layers, hidden, heads, kv_heads, ffn_mult, vocab, tie_embeddings; `data` carries seq_len, micro_batch; `precision` carries dtype, grad_dtype; `memory` carries gradient_checkpointing, offload, compile; `optimizer.name` covers adamw, adamw8bit, cpu_adam |
| D1 container | Done: `docker-compose.training.yaml`, `docker/training/Dockerfile.training`, Makefile targets `train-build`, `train-shell`, `train-smoke`, `train-prepare`, `train-tokenizer`, `train-validate` |
| Dependencies | Already pinned in `pyproject.toml`: `torch==2.13.0`, `flash-attn==2.8.3.post1`, `pynvml==12.0.0`, `bitsandbytes==0.49.2`, `transformers==5.13.1` (AutoConfig). No new dependencies expected |
| Calculator package | Absent: no `training/calculator/`, no `make memfit` target |
| D6 trainer loop | Absent: `training/loop.py` does not exist; the calibration harness cannot reuse a shared loop yet |
| Entry-point pattern | Established: `scripts/training/validate_recipe.py` shows the load → override → validate → styled-output CLI shape; `.styles/PYTHON_OUTPUT_STYLE.md` governs table and `--json` behavior |
| Tests | `tests/training/` is CPU-only and sandbox-runnable; `test_recipes.py` shows the fixture pattern for loading shipped recipes |

Unlike D5, no drift or blockers were found: the D5 branch already fixed the
`scripts/train` renames and container visibility of `training/`, and the recipe
contract D4 consumes exists exactly as the document assumes.

## Deviations from the document

1. **Package interface vs script interface.** The document specifies
   `python -m training.calculator`, which requires a `__main__.py` (not listed in its
   own package layout). The repository's established pattern is thin CLIs under
   `scripts/training/`. Suggested resolution: implement the logic in
   `training/calculator/` with a `__main__.py` so both `python -m
   training.calculator` and a `make memfit` target work; skip a separate
   `scripts/training/` wrapper unless consistency is preferred.
2. **Calibration harness precedes D6.** The document has `calibrate.py` run "a short
   real training loop", but `training/loop.py` (D6) does not exist yet. The harness
   must therefore embed its own minimal forward/backward/step loop over synthetic
   batches. That loop should stay private to `calibrate.py` and not become a de facto
   D6 — D6 later replaces it or the harness adopts D6's loop.
3. **QLoRA / NF4 weights term is forward-looking.** Nothing in the current schema
   expresses 4-bit base loading or adapters (that arrives with the Phase 1 PEFT
   work). Phase 0 D4 should implement the weights term for the dtypes the schema can
   express today (bf16, fp8) and structure `components.py` so the NF4 + adapter term
   is an additive extension, not a rewrite.
4. **HF AutoConfig mapping table is minimally exercised in Phase 0.** Both reference
   recipes are `model.scratch`. The AutoConfig path should ship (the document is
   explicit, and Phase 1 fine-tunes HF models), but its mapping table can start with
   the Llama/Gemma-family fields and fail loudly on anything unmapped, per decision 2
   of the document. Its tests can run offline against a locally constructed config
   object rather than a hub download.
5. **`overhead_gb` and the safety margin are calculator constants, not recipe
   fields.** The schema has no `memory.overhead_gb`; adding one would let recipes
   drift from calibration. Keep both as constants in `training/calculator/` updated
   by calibration records, matching decision 1 (no parallel argument set).

## Memory model details

Formulas the implementation and the hand-computed unit test must agree on, resolved
against the schema fields:

- **Parameter count (scratch).** Embeddings `vocab * hidden` (once when
  `tie_embeddings`, twice otherwise); per layer: attention
  `hidden * (hidden + 2 * kv_dim) + hidden * hidden` with
  `kv_dim = hidden * kv_heads / heads`, SwiGLU MLP `3 * hidden * ffn` with
  `ffn = ffn_mult * hidden` (the reference recipes' ffn_mult values, 2.67 and 4.0,
  already encode the SwiGLU convention of the source architectures — verify against
  SmolLM2-135M's 1536 and Llama-3.2-1B's 8192 intermediate sizes), plus RMSNorm
  vectors. Rounding of `ffn_mult * hidden` must match the reference architectures
  (open question 3).
- **Gradients.** `P_trainable * bytes(grad_dtype)`; both reference recipes set
  `grad_dtype: fp32`, so 4 bytes/param, not the bf16 default the document mentions.
- **Optimizer.** adamw: 8 bytes/param (fp32 exp_avg + exp_avg_sq) plus 4 bytes/param
  fp32 master weights when `precision.dtype` is bf16 and grads are fp32; adamw8bit:
  2 bytes/param plus per-block constants (bitsandbytes block size 256); cpu_adam:
  zero VRAM, host-RAM figure reported instead.
- **Activations.** Per layer with flash-attn assumed:
  attention and MLP intermediates linear in `micro_batch * seq_len`; with
  `gradient_checkpointing`, only layer inputs (`micro_batch * seq_len * hidden *
  bytes(dtype)` per layer) plus one layer's recompute workspace; logits
  (`micro_batch * seq_len * vocab`) counted separately since the vocab term dominates
  small models. The exact intermediate inventory is the main accuracy risk and the
  reason the calibration harness exists.
- **Fits test.** `estimated_reserved <= 32GB - safety_margin (default 1GB)`, with
  `estimated_reserved = allocated_estimate + overhead_gb` as the starting model,
  refined by calibration.

## Work breakdown

1. `training/calculator/components.py` — closed-form component functions
   (`param_count`, `weights_bytes`, `grad_bytes`, `optimizer_bytes`,
   `activation_bytes`, `host_ram_bytes`) taking the typed recipe; an
   `ArchitectureSpec` derived from `model.scratch` or from AutoConfig via the mapping
   table, failing loudly on unknown architectures.
2. `training/calculator/solve.py` — `estimate(recipe)` returning a structured result
   (per-component bytes, totals, fits flag, host RAM when offload is on) and
   `solve(recipe, axis)` binary-searching micro-batch or seq_len.
3. `training/cli.py` plus `training/__main__.py` — unified CLI (`python -m
   training memfit --config ... [--solve batch|seq] [--json]`, see resolved
   decision 1), calling `load_recipe` and `validate` first, human table per
   `.styles/PYTHON_OUTPUT_STYLE.md`, machine mode emitting only JSON.
4. `training/calculator/calibrate.py` — GPU-only harness: builds the model from the
   recipe, runs a few real steps on synthetic batches, records
   `torch.cuda.max_memory_allocated/reserved` and NVML peak (pynvml), writes a
   calibration record JSON, and prints estimate-versus-measured deltas.
5. Makefile: `memfit` target (`docker compose -f docker-compose.training.yaml run
   --rm training python -m training.calculator --config $(CONFIG) $(ARGS)`) and a
   `train-calibrate` target for the harness.
6. `tests/training/test_calculator.py` — hand-computed tiny config pinning every
   component number; estimates for both shipped reference recipes load and produce
   consistent totals; solve() monotonicity and boundary behavior; offload switches
   the report to host RAM. All CPU-only, sandbox-runnable.
7. Documentation: this file plus a `docs/training/memory_fit.md` usage doc, and
   filling in the memory-fit section of the `training-runs` skill once the commands
   are real.

Items 1-3 and 6-7 are sandbox-executable. Items 4-5's verification (calibration on
the two reference configs, the 10 percent acceptance) is host-side.

## Acceptance checks

- Sandbox: `pytest tests/training/` passes, including the pinned closed-form numbers
  for the hand-computed tiny config (document acceptance bullet 3).
- Host: `make memfit CONFIG=training/recipes/reference/pretrain-1b.yaml` reports the
  config as fitting in 32GB without offload; `make train-calibrate CONFIG=...` for
  both reference configs produces measured peaks within 10 percent of the estimate,
  with the reserved estimate not under-predicting the NVML peak. Human runs and
  reports output; the reported commands and outputs are recorded verbatim.

## Risks

- **Activation formula accuracy** is the hard part: torch.compile (both reference
  recipes set `memory.compile: true`) changes activation lifetimes and allocator
  behavior, and flash-attn workspace sizes are version-dependent. Mitigation is the
  calibration loop the document already mandates; expect an iterate-measure-adjust
  cycle on the host before the 10 percent bar is met.
- **The 1B-fits-in-32GB acceptance is a real constraint, not a formality.** 1B
  params with adamw8bit, bf16 weights, fp32 grads, checkpointing on: roughly 2 + 4 +
  2.5 GB for weights/grads/optimizer, leaving the activation and overhead terms to
  determine the margin at micro_batch 4, seq 2048. If calibration shows it does not
  fit, the recipe (micro-batch) is adjusted, not the calculator.
- **`ffn_mult` as float** invites off-by-rounding drift between the calculator's
  param count and the real model built by the calibration harness. The harness
  should print the built model's true parameter count so the discrepancy is caught
  immediately.
- **Calibration harness scope creep** into a proto-D6 trainer; keep it to a private
  few-step loop with no checkpointing, tracking, or resume.

## Resolved decisions

Recorded from review of this investigation (2026-07-14).

1. **Unified CLI.** D4 lands as the first subcommand of a unified training CLI:
   `training/cli.py` exposed as `python -m training <subcommand>`, starting with
   `memfit` and `calibrate` and migrating the existing `scripts/training/` entry
   points (`validate`, `prepare-data`, `train-tokenizer`) as thin subcommands that
   call the same underlying modules. Makefile targets keep their current names and
   become one-line invocations of the CLI. Caveat: the D6 trainer stays outside the
   subcommand dispatcher if it adopts `hydra.main` (Hydra owns argv there); it can
   join later via Hydra's compose API. Migration of existing scripts may land as a
   separate commit from the calculator itself.
2. **Calibration model construction:** HF `LlamaConfig` +
   `AutoModelForCausalLM.from_config` builds the scratch architectures, so the
   measured model matches what D6 training runs will instantiate.
3. **SwiGLU rounding convention:** pinned from the source architectures the
   reference recipes cite — SmolLM2-135M: hidden 576, intermediate 1536
   (`ffn_mult` 2.67); Llama-3.2-1B: hidden 2048, intermediate 8192 (`ffn_mult`
   4.0). The unit test asserts these intermediate sizes; the calculator and the
   calibration harness must derive them identically from `ffn_mult * hidden`
   (rounded to a hardware-friendly multiple where the product is inexact).
4. **Phase 0 weights term scope:** NF4/QLoRA support is deferred to Phase 1, where
   all fine-tuning happens. Phase 0 ships bf16/fp8 dtypes only, with
   `components.py` structured so the 4-bit base + adapter term is an additive
   extension.
