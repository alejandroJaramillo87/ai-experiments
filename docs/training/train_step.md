# Stepping a recipe on the GPU

`make train-step` builds a recipe's real model, runs a few optimizer steps through the
trainer's loss, and prints what the card held. It is the answer to "does this recipe fit"
and to "is the model the tree builds the model the checkpoint describes", read before a long
run is launched. It succeeds the memory-fit calculator, retired on 2026-09-20: the
calculator's per-profile activation rate needed a workstation run per family, width,
training mode and checkpointing flag before it could say anything, and on a one-GPU
workstation that run answers the question by itself. The design record of the calculator
and of its 2026-09-19 reduction is
[the D4 investigation](investigations/d4_memory_fit_investigation.md).

## Usage

Host, in the training container, one GPU consumer at a time:

```bash
make train-step CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="--out /mnt/ai-data/training/verify/step-sft-e2b.json"
```

Direct invocation, with dotlist overrides applied before the build:

```bash
python -m training step --config training/recipes/reference/sft-qlora-e2b.yaml \
    data.micro_batch=1 --steps 3 --json
```

Exit codes: 0 when the steps ran; 1 on a load or validation failure, when the recipe cannot
be stepped here (`memory.offload: zero_offload`, whose runtime is the DeepSpeed engine, not
this loop), when a step runs out of GPU memory, and without a GPU; 2 on invalid usage. With
`--json`, stdout carries only the record (the fields of `StepRecord` in
`training/commands/train_step.py`); `--out` writes the same record to a file.

## What it prints

```
[RUN] Stepping sft-qlora-e2b: 5 steps on cuda:0
[OK] Parameter count: 5,140,325,376 (analytic matches model)

Resident after build:   6.37 GB
Peak allocated:         13.10 GB
Peak reserved:          13.80 GB
NVML peak:              15.20 GB
Tokens per micro-batch: 4,096
[OK] Stepped 5 steps
```

- **Parameter count** compares the built model's torch-reported count (bitsandbytes 4-bit
  parameters counted at their logical size) with `ArchitectureSpec.param_count()` from
  `training/models/architecture.py`, plus the adapter under PEFT; the trainable count is
  compared the same way. A drift is a `[WARN]` with the module to align, not a failure: on
  Gemma 4 the modeling code does not construct the shared-KV residue the checkpoint carries,
  a known drift on E2B of 18,881,440 as measured, 18,880,512 of it that residue (KB-82,
  [gemma-4-training.md](gemma-4-training.md#parameter-accounting)). The adapter count is
  priced per layer on the modules the model builds (KB-90), so under PEFT it matches exactly.
- **Resident after build** is `torch.cuda.memory_allocated` once the model is on the device:
  the weights as loaded (NF4 packing, the towers of a composite checkpoint, the embedding
  tables at the load dtype since the PEFT wrap upcasts the norms only, KB-85).
- **Peak allocated** and **Peak reserved** are torch's high-water marks over the steps;
  **NVML peak** is the driver's view, which includes the CUDA context and allocator slack
  and is the figure to hold against the card's 32 GB.
- **Tokens per micro-batch** is `data.micro_batch x data.seq_len`, the geometry the figures
  belong to.

A step that runs the GPU out of memory is a geometry finding, not a failure to diagnose:

```
[ERROR] Step ran out of GPU memory at step 1: resident after build 6.37 GB, peak allocated 30.57 GB of 31.36 GB, at 4,096 tokens per micro-batch
  Solution: lower data.micro_batch or data.seq_len
```

The resident figure tells a model that loaded larger than it should (KB-85 was 5.5 GB of
fp32 embedding tables) from a micro-batch that does not fit beside it (KB-86).

## What it measures

The model is built through the shared builders in `training/models/model.py`, so the stepped
model is exactly what training instantiates: scratch recipes via HF `LlamaConfig`, `model.hf`
recipes via `from_pretrained` with the recipe's nf4 `BitsAndBytesConfig`, wrapped with the
recipe's PEFT adapter when `peft.enabled` (KB-03), and stepped in training mode as the
Trainer runs it: `from_pretrained` returns a model in eval mode, and transformers'
checkpointing layers checkpoint only while `self.training`, so a loop that skips `train()`
keeps every layer's activations and measures a model training never runs (KB-89). The loss
is the trainer's:
`training/models/head_loss.py` runs the decoder once and then the head, Gemma 4's logit
softcap and the cross-entropy over `memory.head_chunk_tokens` positions at a time under torch
checkpointing (KB-87), so the 262144-wide logits are resident one chunk at a time. Unchunked,
the `micro_batch x seq_len x vocab` logits were held in about six copies through the loss and
the backward, about 24 GB at 4096 tokens, a term outside every checkpointed block. A peak
printed by an out-of-memory step is the card less the request that failed, not the size of
any term: FT.calibrate-qlora's fourth, fifth and sixth attempts all read 30.57 GB of 31.36,
before and after the chunked loss.

## Parameter accounting

Architecture comes from the recipe's `model.scratch` fields, with the SwiGLU intermediate
size derived as `round(ffn_mult * hidden / 64) * 64` (pinned against SmolLM2-135M and
Llama-3.2-1B), or from HF `AutoConfig` for `model.hf` recipes via a supported-model-type
mapping (`HF_SUPPORTED_MODEL_TYPES` in `training/models/architecture.py`). Unknown
architectures fail loudly rather than guess. The Gemma 4 E-models (`model_type`
`gemma4`/`gemma4_text`) are supported: `spec_from_hf_config` descends into `text_config`,
reads the `layer_types` array so the full-attention layers count at `global_head_dim`, counts
the q and k norms and the four block norms, and models the per-layer embedding path in full
(table, model-level projection, per-layer gate, projection and norm).
`tests/training/test_architecture.py` holds the E2B and E4B counts equal to the exact figures
in [gemma-4-training.md](gemma-4-training.md#parameter-accounting). Only a config with the MoE
block enabled (`enable_moe_block`) fails loudly, since the dense formulas do not model expert
routing.

A composite checkpoint (Gemma 4's `text_config` beside vision and audio configs) loads its
towers through `from_pretrained` whether or not text-only work uses them, so their size is in
the count. It is read, not modeled: `spec_from_recipe` reads the `model.safetensors` header of
a local base directory and sums every tensor outside `model.language_model.` into
`ArchitectureSpec.extra_params`. A hub id resolved through a cache has no such directory and
counts the towers at zero.

The same module carries the one byte term two invariants still need: `host_ram_bytes`, the
offloaded optimizer state (16 bytes per trainable parameter under `zero_offload`) plus an
explicit replay budget, which validate's offload budget check holds under the 128 GB host
and the replay buffer's derived headroom subtracts from.

## Tests

`tests/training/test_architecture.py` pins the parameter counts for a hand-computed tiny
configuration, the reference recipes, the Gemma 4 E2B and E4B checkpoints, the header reader,
the HF mapping table and the adapter counts. `tests/training/test_cli_handlers_step.py`
covers the `step` handler with the GPU loop under test control. Both are CPU-only:

```bash
pytest tests/training/test_architecture.py tests/training/test_cli_handlers_step.py
```

On the workstation, `tests/hardware/test_gpu.py` steps the tiny recipe in the `hw_smoke`
tier, and `tests/hardware/test_step_qlora.py` steps the reference E2B QLoRA recipe as committed
in the `hw_full` tier, holding the adapter count, the parameter drift, the resident figure and
the peak to what FT.calibrate-qlora recorded, so a library bump that moves any of them fails
there before a long run.
