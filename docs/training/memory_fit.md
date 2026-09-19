# Memory-fit calculator

Pre-run VRAM estimation for training recipes (Phase 0 D4). The calculator answers,
before a run starts, whether a recipe's peak VRAM fits the RTX 5090's 32GB, from
two kinds of term: static terms priced from the recipe's parameter count, and an
activation cost measured on this workstation for the recipe's profile. Input is
always a D5 recipe file, never a parallel argument set, so the calculator and the
trainer read identical configuration by construction. Design record:
[the D4 investigation](investigations/d4_memory_fit_investigation.md); the
2026-09-19 reduction that replaced its analytic activation model is recorded
below.

## Usage

Estimate a recipe (host, in the training container):

```bash
make memfit CONFIG=training/recipes/reference/sft-qlora-e2b.yaml ARGS="data.micro_batch=8"
```

Direct invocation, with dotlist overrides applied before estimation:

```bash
python -m training memfit --config training/recipes/reference/sft-qlora-e2b.yaml \
    data.micro_batch=8 --json
```

Exit codes: 0 when the configuration fits or when its profile is unmeasured
(a warning, not a verdict); 1 on a load or validation failure, and when a
measured profile does not fit; 2 on invalid usage. With `--json`, stdout carries
only JSON: one `estimate` object with the fields of `Estimate`
(`training/calculator/estimate.py`), `rate`, `activations` and `fits` being
`null` for an unmeasured profile.

`make train-sft` prints the same line as a `[OK]` or a `[WARN]` and proceeds
either way: an out-of-memory run on this machine fails at its first step, after
a minute of model load, so the estimate is advice the operator reads before
launching, not a gate the run waits on.

## Memory model

Peak VRAM is static terms plus measured activations plus a calibrated overhead:

| Term | Model |
|------|-------|
| Weights | `P * bytes(precision.dtype)`; under QLoRA the linear projections at NF4 rates, the rest and the adapter in `peft.compute_dtype` (`weights_bytes` in `training/calculator/components.py`) |
| Gradients | `P_trainable * bytes(precision.dtype)` (`peft.compute_dtype` under PEFT) — `backward()` produces gradients in the parameter dtype, and no trainer implements separate-dtype accumulation (KB-55); all parameters are trainable in the pretrain path, only the adapter under PEFT (`adapter_param_count`) |
| Optimizer state | `adamw`: 8 bytes/param fp32 states plus 4 bytes/param master copy under mixed precision; `adamw8bit`: 2 bytes/param plus per-block fp32 constants (block size 256); zero when offloaded, with the host RAM cost reported instead (16 bytes/param against the 128GB budget) |
| Activations | `rate * micro_batch * seq_len`, where `rate` is the bytes per token `training/calculator/measured.py` records for the recipe's profile; unmeasured when the profile has no row |
| Framework overhead | Calibrated constant (`OVERHEAD_BYTES` in `training/calculator/estimate.py`), calibrated to 2.5GB from the 2026-08-01 `pretrain-1b` record: 1.38GB allocator slack plus 1.08GB CUDA context, 2.46GB above the measured allocated |

The fits flag tests the reserved estimate (the terms above) against 32GB minus a
1GB safety margin. For an unmeasured profile the estimate is the static terms
plus overhead, reported as a lower bound with a `[WARN]` naming the profile and
the calibrate command that measures it; `fits` is `None`, and nothing is
refused on it.

**Profiles.** A rate is filed under `family/layersxhidden/mode/ckpt`
(`profile()` in `components.py`): the HF text `model_type` or `scratch`, the
geometry, `full` | `lora` | `qlora` (PEFT with `quantize_base: nf4`), and
whether gradient checkpointing is on. Two recipes with the same profile share a
per-token activation cost; a different base, width, depth or training mode is a
different measurement. The rate is one point on a line through the origin:
activations at another micro-batch or sequence length are assumed linear in
tokens, which flash attention makes true to within the per-head softmax
statistics.

**Why measured, not modeled (2026-09-19).** The calculator once carried an
analytic activation inventory (per-layer saved tensors, a checkpointed
temporal-max peak model, KB-56) and a binary search for the largest fitting
micro-batch or sequence length. One measured point, a scratch Llama in August,
calibrated it; on the workload this repository runs, QLoRA on Gemma 4 through
TRL, its accuracy was unproven, and its formula did not model Gemma 4's
geometry (four norms a layer, two head widths, per-layer embedding inputs,
double-wide MLPs). An analytic activation term is a second implementation of
the model that drifts with every family and every transformers release, so it
was replaced by the measurement calibrate already made. `solve` went with it:
its answer rested on the term that left.

The host RAM report (`host_ram_bytes`) sums the offloaded optimizer state with
an explicit replay-buffer budget (`replay.max_gb`, P2-D4). A replay block with
`max_gb` unset contributes nothing: its budget is derived from the remaining
host headroom (`training/continual/replay.resolve_max_bytes`), so counting it
would be circular.

## Parameter accounting

Architecture comes from the recipe's `model.scratch` fields, with the SwiGLU
intermediate size derived as `round(ffn_mult * hidden / 64) * 64` (pinned against
SmolLM2-135M and Llama-3.2-1B), or from HF `AutoConfig` for `model.hf` recipes via
a supported-model-type mapping (`HF_SUPPORTED_MODEL_TYPES` in
`training/calculator/components.py`). Unknown architectures fail loudly rather
than guess. The Gemma 4 E-models (`model_type` `gemma4`/`gemma4_text`) are
supported: `spec_from_hf_config` descends into `text_config`, reads the
`layer_types` array so the full-attention layers price at `global_head_dim`,
counts the q and k norms and the four block norms, and models the per-layer
embedding path in full (table, model-level projection, per-layer gate,
projection and norm). `tests/training/test_calculator.py` holds the E2B and E4B
counts equal to the exact figures in
[gemma-4-training.md](gemma-4-training.md#parameter-accounting). Only a config
with the MoE block enabled (`enable_moe_block`) fails loudly — the dense
formulas do not model expert routing.

A composite checkpoint (Gemma 4's `text_config` beside vision and audio configs)
loads its towers through `from_pretrained` whether or not text-only work uses
them, so their size must be in the weights term. It is read, not modeled:
`spec_from_recipe` reads the `model.safetensors` header of a local base
directory and sums every tensor outside `model.language_model.` into
`ArchitectureSpec.extra_params`, priced in `precision.dtype` and never
quantized (the safe direction). A hub id resolved through a cache has no such
directory and prices the towers at zero.

## Calibration

`python -m training calibrate` (or `make train-calibrate CONFIG=...`) runs a short
real training loop on the workstation GPU, records
`torch.cuda.max_memory_allocated/reserved` and the NVML peak, and derives the
profile's activation rate:

```bash
make train-calibrate CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="--out /mnt/ai-data/training/verify/calibrate-sft-e2b.json"
```

The model is built through the shared builders in `training/models/model.py`, so
the measured model is exactly what training runs instantiate: scratch recipes via
HF `LlamaConfig` and `AutoModelForCausalLM.from_config`, `model.hf` recipes via
`from_pretrained` with the recipe's nf4 `BitsAndBytesConfig`, wrapped with the
recipe's PEFT adapter when `peft.enabled`. The record includes the built model's
true parameter count (and trainable count, under PEFT) so drift from the analytic
counts is caught immediately.

The rate is what the peak held above the model: right after the model is built
and on the device, `torch.cuda.memory_allocated` is recorded as the resident
static bytes (NF4 packing, towers and all, so none of that leaks into the rate);
after the steps, `(max_memory_allocated - resident - grads - optimizer) /
(micro_batch * seq_len)`, floored at zero, is printed as

```
Activation rate:     123,456 bytes/token
  Record: "gemma4_text/35x1536/qlora/ckpt": 123456,
```

and the operator pastes the `Record:` line into `RATES` in
`training/calculator/measured.py` with a comment naming the date, the runbook
entry and the measured figures. The training container bind-mounts `training/`,
so the next `make memfit` reads the row without a rebuild.

Acceptance, once a rate is on file: the allocated estimate within 10 percent of
`max_memory_allocated` (or within a 128MiB absolute floor,
`ALLOCATED_FLOOR_BYTES`, for tiny models where fixed CUDA library workspaces
dominate), and the reserved estimate not under-predicting the NVML peak. The
first calibrate of a profile has nothing to hold to that bar; it reports
`Allocated error: not on file`, prints the rate, and exits 0. A later calibrate
of the same profile, after a transformers upgrade or a trainer change, is what
detects a moved peak: when it reports drift, re-record the rate (never the
recipe), or raise `OVERHEAD_BYTES` in `training/calculator/estimate.py` when
the reserved check fails.

One recipe class is refused as uncalibratable rather than measured wrongly:
`memory.offload: zero_offload` (the calibration loop is plain torch, not the
DeepSpeed ZeRO engine).

## Tests

`tests/training/test_calculator.py` pins the static component numbers for a
hand-computed tiny configuration, the reference-recipe parameter counts, the
Gemma 4 E2B and E4B counts against the checkpoint's exact figures, the
checkpoint-header reader, the measured-rate estimate and its unmeasured
advisory, the HF mapping table, and the CLI JSON contract.
`tests/training/test_cli_handlers_memfit.py` covers the `memfit` and `calibrate`
handlers with the rate table under test control. All are CPU-only:

```bash
pytest tests/training/test_calculator.py tests/training/test_cli_handlers_memfit.py
```

On the workstation, `tests/hardware/test_memfit_sft.py` (`hw_smoke`) holds the
E2B QLoRA recipe's verdicts once its rate is recorded, and
`tests/hardware/test_gpu.py` runs calibrate on the tiny recipe.
