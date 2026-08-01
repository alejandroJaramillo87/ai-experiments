# Memory-fit calculator

Pre-run VRAM estimation for training recipes (Phase 0 D4). The calculator answers,
before a run starts, whether a recipe fits in the RTX 5090's 32GB VRAM and what the
maximum micro-batch or sequence length is. Input is always a D5 recipe file, never a
parallel argument set, so the calculator and the trainer read identical
configuration by construction.

## Usage

Estimate a recipe (host, in the training container):

```bash
$ make memfit CONFIG=training/recipes/reference/pretrain-1b.yaml
```

Solve for the largest fitting micro-batch or sequence length, with machine output:

```bash
$ make memfit CONFIG=training/recipes/reference/pretrain-1b.yaml ARGS="--solve batch --json"
```

Direct invocation, with dotlist overrides applied before estimation:

```bash
$ python -m training memfit --config training/recipes/reference/pretrain-1b.yaml \
    data.micro_batch=8 --solve seq
```

Exit codes: 0 when the configuration fits (or `--solve` was requested), 1 on a
load, validation, or fit failure, 2 on invalid usage. With `--json`, stdout carries
only JSON.

## Memory model

Peak VRAM is the sum of five components, computed in
`training/calculator/components.py`:

| Component | Model |
|-----------|-------|
| Weights | `P * bytes(precision.dtype)` |
| Gradients | `P_trainable * bytes(precision.dtype)` (`peft.compute_dtype` under PEFT) — `backward()` produces gradients in the parameter dtype, and no trainer implements separate-dtype accumulation (KB-55); all parameters are trainable in the pretrain path, only the adapter under PEFT (`adapter_param_count`) |
| Optimizer state | `adamw`: 8 bytes/param fp32 states plus 4 bytes/param master copy under mixed precision; `adamw8bit`: 2 bytes/param plus per-block fp32 constants (block size 256); zero when offloaded, with the host RAM cost reported instead (16 bytes/param against the 128GB budget) |
| Activations | Analytic per-layer inventory for the dense attention + SwiGLU family, GQA-aware, flash-attn assumed (linear in sequence length); gradient checkpointing keeps only layer inputs plus one layer of recompute workspace; logits include the fp32 upcast for cross-entropy |
| Framework overhead | Calibrated constant (`OVERHEAD_BYTES` in `training/calculator/solve.py`), calibrated to 2.5GB from the 2026-08-01 `pretrain-1b` record: 1.38GB allocator slack plus 1.08GB CUDA context above the allocated estimate |

The host RAM report (`host_ram_bytes`) sums the offloaded optimizer state with
an explicit replay-buffer budget (`replay.max_gb`, P2-D4). A replay block with
`max_gb` unset contributes nothing: its budget is derived from the remaining
host headroom (`training/continual/replay.resolve_max_bytes`), so counting it
would be circular.

Under gradient checkpointing the allocated total is not the component sum but a
temporal maximum over two backward phases (KB-56): during head backward the
logits and their fp32 cross-entropy upcast are live but parameter gradients are
not yet materialized (`layer_inputs + head`), and by late backward the gradients
are full but the head tensors are freed (`grads + layer_inputs + per_layer`).
The 1B calibration that motivated this measured 7.00GB against an 11.42GB
component sum; the temporal-max model predicts 7.2GB. The `Estimate.peak_phase`
field records which phase bounds the total (`sum` when checkpointing is off,
where the conservative component sum is kept).

The fits flag tests the reserved estimate (allocated total plus overhead) against
32GB minus a 1GB safety margin. `solve(recipe, axis)` binary-searches micro-batch
(integers) or sequence length (128-token steps) for the largest fitting value.

Architecture comes from the recipe's `model.scratch` fields, with the SwiGLU
intermediate size derived as `round(ffn_mult * hidden / 64) * 64` (pinned against
SmolLM2-135M and Llama-3.2-1B), or from HF `AutoConfig` for `model.hf` recipes via
a supported-model-type mapping (`HF_SUPPORTED_MODEL_TYPES` in
`training/calculator/components.py`). Unknown architectures fail loudly rather
than guess. The Gemma 4 E-models (`model_type` `gemma4`/`gemma4_text`, the
Phase 1 primary family) are supported: `spec_from_hf_config` descends into
`text_config` and models the per-layer embedding tables. Only a config with
the MoE block enabled (`enable_moe_block`) fails loudly — the dense formulas
do not model expert routing. `qwen3` (the fallback family) is fully supported.

## Calibration

`python -m training calibrate` (or `make train-calibrate CONFIG=...`) runs a short
real training loop on the workstation GPU, records
`torch.cuda.max_memory_allocated/reserved` and the NVML peak, and compares them to
the estimate:

```bash
$ make train-calibrate CONFIG=training/recipes/reference/pretrain-150m.yaml \
    ARGS="--out /mnt/ai-data/training/calibration/pretrain-150m.json"
```

The model is built through HF `LlamaConfig` and `AutoModelForCausalLM.from_config`
so the measured model is exactly what training runs instantiate; the record includes
the built model's true parameter count so drift from the analytic count is caught
immediately.

Acceptance (roadmap D4): the allocated estimate is within 10 percent of
`max_memory_allocated` (or within a 128MiB absolute floor,
`ALLOCATED_FLOOR_BYTES`, for tiny models where fixed CUDA library workspaces
dominate) on `pretrain-150m.yaml` and `pretrain-1b.yaml`, the reserved
estimate does not under-predict the NVML peak, and the 1B config fits in 32GB
without offload. When calibration reports drift, adjust the calibrated constants in
`training/calculator/solve.py` (never the recipe) and re-run.

## Tests

`tests/training/test_calculator.py` pins the closed-form component numbers for a
hand-computed tiny configuration, the reference-recipe parameter counts
(134,515,008 and 1,073,809,408), solve boundary behavior, the HF mapping table,
and the CLI JSON contract. All tests are CPU-only:

```bash
$ pytest tests/training/test_calculator.py
```
