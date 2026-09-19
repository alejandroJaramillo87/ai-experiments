# Numeric precision policy

The dtype decisions of the three tiers — the Rust inference harness, the interpretability
tier, and the training tier — collected in one place. Each tier picks its own numeric
formats for its own reasons; this document names the concern, states what each tier does
about it, and cites the file that decides it. Where a precision choice in one tier
invalidates an artifact produced by another, that hazard is stated here rather than left
implicit.

The code is the authority. Nothing below restates a value that a source file owns; every
row names the file and the symbol so the fact can be re-read at the source.

## Table of contents

- [Why bf16 is the house compute dtype](#why-bf16-is-the-house-compute-dtype)
- [Concerns at a glance](#concerns-at-a-glance)
- [Model compute and weights](#model-compute-and-weights)
- [Activation storage on disk](#activation-storage-on-disk)
- [Gradients and optimizer state](#gradients-and-optimizer-state)
- [KV cache](#kv-cache)
- [Quantization](#quantization)
- [Cross-tier hazard: probes are fitted at a precision](#cross-tier-hazard-probes-are-fitted-at-a-precision)
- [Widening and narrowing: which direction is lossy](#widening-and-narrowing-which-direction-is-lossy)

## Why bf16 is the house compute dtype

bfloat16 and fp16 are both 16 bits, and that is all they share. bfloat16 spends 8 bits on
the exponent and 7 on the mantissa; IEEE half spends 5 on the exponent and 10 on the
mantissa. The 8-bit exponent is the same width as fp32's, so bfloat16 covers the fp32
dynamic range and trades precision, not range, for its two bytes.

That trade matches transformer numerics. Residual-stream activations carry outlier
features whose magnitude grows with depth, and fp16's exponent saturates at 65504: an
activation past that becomes `inf`, and the `inf` propagates through the rest of the
forward pass. bfloat16 has no such ceiling at any magnitude fp32 can represent, so a
bf16 forward pass either matches fp32 in range or overflows where fp32 would too. The
cost is mantissa: bfloat16 resolves roughly three decimal digits against fp16's four.

Two consequences follow, and both are load-bearing elsewhere in this document. bfloat16
is bit-exactly the high half of an f32, so widening is a shift and never rounds — the
`bf16_widening_is_a_shift_for_every_input` test in
[`rust-harness/runtime/src/tensor.rs`](../../rust-harness/runtime/src/tensor.rs) asserts
this over a strided sweep of every 16-bit pattern. And fp32 is a lossless *storage*
substitute for bf16 at twice the bytes, which is why an fp32 capture is a valid stand-in
for a bf16 one.

None of this makes fp16 a compute dtype here. The argument above is about a forward pass,
where an `inf` at any layer poisons every layer after it and the magnitudes are unbounded
in advance. The argument about *storage* is narrower and comes out differently, because a
stored activation is a quantity that can be measured before it is written — see
[widening and narrowing](#widening-and-narrowing-which-direction-is-lossy). fp16 is a
storage format in this repository and nowhere a compute one.

## Concerns at a glance

| Concern | rust-harness | interp | training |
|---------|--------------|--------|----------|
| Model compute | f32 throughout the CPU backend | Subject model loaded at `harvest.dtype` | `precision.dtype`, default bf16 |
| Weights in memory | Held at the checkpoint's published width (`tensor::Weight`), widened per use | Subject-model dtype | `precision.dtype`; frozen base optionally nf4 |
| Weights on disk | Hugging Face safetensors, F32, F16, or BF16 | n/a (consumes a served or loaded model) | Checkpoint in parameter dtype |
| Activations on disk | Probe capture shards, f32 | Arrow capture at `harvest.dtype`; Arrow stores fp16 or fp32 only | n/a |
| Gradients | n/a (inference only) | n/a | Parameter dtype (KB-55) |
| Optimizer state | n/a | n/a | AdamW fp32 states plus fp32 master copy |
| KV cache | f32; global layers hold the context, sliding layers hold the window | Inherited from the serving engine | n/a |
| Quantization accepted | None. A quantized checkpoint is refused by name | None; consumes what it is served | nf4 base under QLoRA; 8-bit optimizer state under `adamw8bit` |

## Model compute and weights

**rust-harness.** The CPU backend computes in plain f32 and performs no arithmetic in any
other numeric type. Weights are *stored* at the width their checkpoint declares — bf16 stays
bf16 — in the `Weight` enum in
[`rust-harness/runtime/src/tensor.rs`](../../rust-harness/runtime/src/tensor.rs), and are
widened to f32 at the point of use. Storage width and compute precision are separate
decisions here, and conflating them is the most common misreading of this tier: there is no
path that multiplies two bf16 values in bf16.

The accepted set is the three float widths safetensors can carry: F32, F16, and BF16. It is
enforced by `widen` in
[`rust-harness/runtime/src/checkpoint.rs`](../../rust-harness/runtime/src/checkpoint.rs),
which refuses any other dtype by name rather than approximating it — the error text states
that the backend "computes at 16-bit precision or wider by design". A quantized checkpoint
therefore cannot be loaded at all, deliberately, because the numerics have to line up with
what the interpretability pipelines record. Because BF16 is the published storage type for
Gemma 4 and the format the training tier emits, the widening to f32 is exact and the
checkpoint's own values reach the forward pass unrounded.

No GGUF reader remains in the runtime. `gguf.rs` and the `GGML_*` storage-type
constants beside it were deleted at R10, along with the last artifact that needed them: the
control vector, which is now a safetensors image
([`schemas/control_vector_artifact.json`](../../schemas/control_vector_artifact.json)) as the
probe already was. Every tensor the runtime reads — weights, probes, control vectors — arrives
as safetensors, widened through one decoder.

**training.** `precision.dtype` in
[`training/recipes/schema.py`](../../training/recipes/schema.py) (`PrecisionConfig`) selects
the run's compute dtype, defaults to `bf16`, and drives both the torch dtype
(`training/models/model.py`) and the trainer flags (`training/tune/sft.py`,
`training/runtime/offload.py`).

**interp.** The interp tier does not choose a compute dtype independently: the subject
model is loaded at `harvest.dtype` for a capture, or is whatever the serving container
loaded for the write-back path.

### The two allowed dtype sets in the training schema

[`training/recipes/schema.py`](../../training/recipes/schema.py) declares two tuples of
allowed dtype names, and they differ on purpose:

| Constant | Governs | Members |
|----------|---------|---------|
| `PRECISION_DTYPES` | `precision.dtype` — the run-level compute dtype | `bf16` |
| `COMPUTE_DTYPES` | `peft.compute_dtype` (the PEFT compute dtype over an nf4 base) and `harvest.dtype` (the interp capture dtype) | `bf16`, `fp16`, `fp32` |

Both are enforced in [`training/recipes/validate.py`](../../training/recipes/validate.py):
`precision.dtype` is checked against `PRECISION_DTYPES`, `peft.compute_dtype` against
`COMPUTE_DTYPES`. The policy, stated on 2026-09-19 when the unbuilt `fp8` field left the
schema: a run computes in bf16, full stop, for the reason the first section gives. fp16
and fp32 remain reachable where they are storage or adapter choices rather than a run
precision — the compute dtype of a LoRA adapter over an nf4 base, and the Arrow storage
dtype of a harvest — and never for the run itself. The two dead branches that the old
ambiguity left behind (an `fp16` trainer flag in `training/tune/sft.py`, an `!= "fp32"`
master-copy test in `training/calculator/components.py`) were removed with the field.

**What is narrower than 16 bits in a training recipe is storage, not arithmetic.**
`peft.quantize_base: nf4` holds the frozen base in 4-bit and dequantizes each block to
bf16 at the point of use; the adapter, its gradients and the matmul are bf16.
`optimizer: adamw8bit` holds the Adam moments in 8-bit with fp32 per-block constants; the
update is computed in fp32. Neither changes the precision of a forward or backward pass.

**A QLoRA adapter is fitted against a base the export does not serve.** The adapter is
trained against the nf4-dequantized base, and `training/export/merge.py` merges it into
the bf16 base, so the served `model.safetensors` is not bit-identical to the model the
adapter was optimized against. The difference is the nf4 rounding of the frozen weights,
small and standard practice, but it is a precision fact of the same kind as the probe
hazard below: an artifact carries a precision it was not fitted at, and nothing errors.
The export manifest records `mode: merge` and the adapter's own hash, which is enough to
know the fact applies; no consumer today measures its size.

## Activation storage on disk

**interp.** A harvest writes residual-stream activations to Arrow through the SAELens
activation-caching runner. Two constants near the top of
[`interp/harvest.py`](../../interp/harvest.py) govern the format:

| Constant | Meaning |
|----------|---------|
| `DTYPE_BYTES` | Bytes per element for `bf16`, `fp16`, `fp32` — the disk and budget arithmetic |
| `ARROW_STORAGE_DTYPES` | The subset Arrow can store, and therefore the subset a capture can actually be taken at: `fp16` and `fp32` |
| `DTYPE_NAMES` | Recipe spelling to the SAELens/torch dtype name; shared with `interp/sae/train.py` |

`bf16` is in `DTYPE_BYTES` and absent from `ARROW_STORAGE_DTYPES` because Arrow has no
bfloat16 type in any version. This is not a tunable: a `bf16` capture cannot run at all.
`tests/interp/test_harvest.py` enforces the split against the committed recipes.
[`docs/interp/harvest-precision-and-resume.md`](../interp/harvest-precision-and-resume.md)
owns the defect (KB-57), the SAELens reading behind it, and what the options cost; the
ledger row is in [`docs/verification/bugs/known-bugs.md`](../verification/bugs/known-bugs.md).

**One field, two type systems.** `harvest.dtype` is a single recipe field that sets both
the Arrow on-disk storage type *and* the precision the SAE is later trained at. The SAE
side reads it back out of the capture manifest: `capture_dtype()` in
[`interp/sae/train.py`](../../interp/sae/train.py) refuses a manifest dtype it cannot train
at, and the trainer job dict takes its `dtype` from that manifest rather than from the SAE
recipe — the comment there records the reason, that training an SAE at a precision the
activations were not harvested at is a silent numerics mismatch. The consistency is
deliberate; the conflation of *storage format* with *training precision* into one field is
what makes an Arrow limitation into a compute-precision constraint.

**rust-harness.** Probe-capture activations are a separate artifact and a separate format:
`rust-harness/runtime/src/bin/activation-capture.rs` writes shards of f32 rows in the
layout `interp/probes/capture.py` defines, so the Python probe toolkit consumes
rust-served activations unchanged.

float32 is not a free choice on either side: it is required by
[`schemas/activation_capture.manifest.json`](../../schemas/activation_capture.manifest.json),
which is the contract both writers answer to. The reasoning recorded there is the reverse of
the Arrow case below — a shard holds activations of a bf16 model, widening bf16 to float32
is exact, and float16 is refused outright because its five exponent bits saturate on
residual-stream outliers that bf16 represents. bfloat16 on disk is unavailable for a third
reason again: numpy has no bfloat16 dtype and `.npy` has no descriptor for it. Both writers
emitted float16 before the contract constrained the field; those captures still load, but
they are not numerically comparable to float32 ones, and a capture is a regenerable cache,
so the remedy is to re-run rather than migrate.

The two activation formats in this repository therefore reach opposite conclusions about
fp16 from the same bf16 source data, and both are correct. The Arrow harvest admits fp16
because magnitudes are measured per subject before a capture runs; the capture-shard
contract refuses it because it constrains two writers across two languages with no such
measurement in the loop. See
[widening and narrowing](#widening-and-narrowing-which-direction-is-lossy) for the
underlying asymmetry.

## Gradients and optimizer state

[`docs/training/memory_fit.md`](../training/memory_fit.md) owns this concern in full: its
memory-model table prices gradients, optimizer states, and the fp32 logit upcast, and its
figures are calibrated against measured workstation runs. The three facts to carry away,
with the detail left where it lives:

- Gradients are produced in the parameter dtype, not a wider accumulation dtype. No
  trainer in this repository implements separate-dtype gradient accumulation; the
  calculator was corrected to match (KB-55).
- AdamW keeps fp32 optimizer states, plus an fp32 master copy of the parameters under
  mixed precision. `adamw8bit` is the quantized-state alternative, with per-block fp32
  constants.
- Logits are upcast to fp32 for the cross-entropy computation, and the upcast is a costed
  line in the activation inventory rather than a rounding detail.

Read [`training/memory_fit.md`](../training/memory_fit.md) for the byte-level model and the
calibration record before sizing anything.

## KV cache

The rust-harness CPU backend stores K and V as f32 — four bytes per element, and no KV
quantization option exists. This is the one place in the tier where f32 is a *storage*
choice rather than a compute one, and it is the conservative direction: attention consumes K
and V in f32, so caching them in f32 stores exactly the values the forward pass produced.
Narrowing the cache to bf16 would introduce a rounding step the arithmetic does not
otherwise have.

**The cache is not narrowed, but it is bounded.** A sliding-window layer retains
`window - 1 + group` rows rather than the whole sequence, where `group` is the prefill batch
width (`DEFAULT_BATCH`, 64), plus a quarter of that as compaction slack. The arithmetic is
`window_capacity` and `budget_bytes` in
[`rust-harness/runtime/src/kv.rs`](../../rust-harness/runtime/src/kv.rs); the storage is the
compacting `LayerKv` buffer in
[`rust-harness/runtime/src/model/cpu.rs`](../../rust-harness/runtime/src/model/cpu.rs).
Global layers still hold the full context, and shared-KV layers hold nothing of their own.
The figure printed at startup is the design rather than an upper bound over it.

Measured on the workstation against real E4B at its trained context of 131072 tokens:

```
[INFO] Weights: 13.90 GiB resident at bf16
[INFO] KV cache: 131072 tokens, 4.05 GiB f32 resident (sliding layers capped at the window plus a batch of 64)
[INFO] Rewind horizon: 208 tokens behind the tip once the window fills (compaction past it re-prefills)
```

Of that 4.05 GiB, the four global layers at head width 512 account for 4.000 GiB and the
20 sliding layers at head width 256 for 0.055 GiB; the 18 shared-KV layers cost nothing.
Uncapped, the sliding layers would have held 10.0 GiB. The cache is now dominated by the
global layers, which no window can shrink — so the remaining lever on KV memory is dtype or
`--context`, not residency.

`--context` caps the budget below the model's trained window and is refused above it. The
2 GiB warning it exists to trigger now fires almost entirely on account of the global
layers.

**The cost of bounding it is rewindability**, which is a precision-adjacent fact worth
stating here: a windowed layer cannot return to a position it has evicted, so `truncate`
below `rewindable_from()` is an error rather than a silently short prefix. Conversation
compaction routinely rewinds further than the horizon, and `Session::sync` clears and
re-prefills when it does. The result is bit-identical, not approximate.

The `HKV1` session snapshot records a dtype byte (0 = f32, 1 reserved for f16 payloads) in
[`rust-harness/runtime/src/session.rs`](../../rust-harness/runtime/src/session.rs), so
narrowing the cache later does not need a format break. Nothing writes the reserved value.

For the containerized engines, the KV cache dtype is the engine's own setting; see
[`docs/inference/parameters/llama-cpp-parameters.md`](../inference/parameters/llama-cpp-parameters.md)
and [`docs/inference/parameters/vllm-parameters.md`](../inference/parameters/vllm-parameters.md).

## Quantization

| Tier | What is quantized | Where decided |
|------|-------------------|---------------|
| rust-harness | Nothing. Weights are held at the checkpoint's published width (`tensor::Weight`) and widened to f32 per use; no quantized arithmetic | `rust-harness/runtime/src/tensor.rs` |
| training | `peft.quantize_base = nf4` loads the frozen base in 4-bit via bitsandbytes (QLoRA), with `peft.compute_dtype` as the compute dtype over it | `training/recipes/schema.py`, `training/recipes/validate.py` |
| training | `optimizer.name = adamw8bit` holds the Adam moments in 8-bit (bitsandbytes), with fp32 per-block constants and an fp32 update | `training/models/model.py` (`build_optimizer`) |
| interp | None of its own; it consumes whatever the serving container loaded | `docker-compose.inference.yaml` |

`validate.py` enforces the one exclusion rather than leaving it to fail at run time: nf4
requires a PEFT adapter, since there is no full fine-tuning of a 4-bit base.

## Cross-tier hazard: probes are fitted at a precision

This is the sharpest precision coupling in the repository, and it crosses tier boundaries
silently.

Linear probes are fitted on bf16 activations. Scoring one against a quantized model means
the model that answered the request and the model whose activations the probe was fitted on
are not the same model. Nothing errors; the numbers simply stop meaning what they appear to
mean.

**R10.5 removed the hazard from this path rather than mitigating it.** Probes are now scored
in the process that answers the request, by the tap seam in the rust harness, which loads
bf16 safetensors and refuses a quantized checkpoint. The score comes from the weights that
produced the text, so there is no second model to compare against and no fidelity claim to
qualify.

Two mitigations existed before that and are gone with it. `activation-capture.rs` once
captured from a quantized serving path, so probes fitted on those rows described the model
that would answer -- retired when the harness stopped reading quantized checkpoints. And an
opt-in bf16 sibling of `llama-gpu` answered monitored traffic at the probes' precision so
that an offline replay would at least match it -- deleted with the replay, since a
precision-matched endpoint has nothing left to match. See
[`docs/interp/writeback.md`](../interp/writeback.md).

The hazard itself is undiminished — it is a property of the quantized llama.cpp and vLLM
services, which are still what most traffic is answered by. What changed is that interp no
longer observes those services at all: probe scoring happens only where the weights are
bf16. That is a narrowing of what is monitored, not a resolution of the quantization
question for the services themselves.

The banner comment at the top of `activation-capture.rs` still describes the rows as
"captured from the quantized serving path itself". That sentence is stale and this section
is the correct account.

## Widening and narrowing: which direction is lossy

The asymmetry is easy to invert, so it is stated once, explicitly:

| Conversion | Lossless | Why |
|------------|----------|-----|
| bf16 to fp32 | Yes, exactly | bf16 is the high 16 bits of the f32 encoding; widening is a left shift by 16 with zero fill |
| fp32 to bf16 | No | Discards 16 mantissa bits, with rounding |
| bf16 to fp16 | Only inside fp16's range | Mantissa *widens* (7 stored bits to 10), so no significand is lost; the exponent narrows from 8 bits to 5, so magnitudes above 65504 saturate to `inf` and those below 6.1e-5 denormalize |
| fp16 to fp32 | Yes | Range and precision both strictly widen |

The bf16-to-fp16 row is the one that surprises people, and it is the row the interp capture
format rests on. Because fp16 carries *more* significand bits than bf16, a bf16 value whose
magnitude lands inside fp16's exponent range round-trips bit for bit — this is not an
approximation that happens to be good enough, it is exact. The conversion is lossy only at
the ends of the range, and only there.

That turns "is fp16 safe for this capture?" into a question about magnitudes, which is
measurable rather than arguable. It is measured per subject by
[`scripts/interp/activation_magnitude_probe.py`](../../scripts/interp/activation_magnitude_probe.py),
and the results are recorded in
[`docs/interp/harvest-precision-and-resume.md`](../interp/harvest-precision-and-resume.md).

The practical consequence: storing a bf16 activation as fp32 preserves every bit at twice
the disk cost, and storing it as fp16 preserves every bit at the same cost as bf16 *provided
the subject's magnitudes have been measured*. Never assume the measurement transfers between
models — the two measured so far differ by 23x.
