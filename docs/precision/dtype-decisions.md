# Numeric dtype decisions

Why each tier picked the numeric format it did, and what it would take to change. The companion
document, [README.md](README.md), records *what* each tier does; this one records *why*, and
answers the question that keeps recurring: whether the three tiers can converge on one dtype.

The short answer is that **storage can converge and compute cannot**, for a reason that is not
a matter of taste: a stored value has already been measured, and a value about to flow through
a forward pass has not.

## Table of contents

- [The one-sentence answer per tier](#the-one-sentence-answer-per-tier)
- [Why rust-harness computes in f32](#why-rust-harness-computes-in-f32)
- [Why the harness accepts three float widths and no quantization](#why-the-harness-accepts-three-float-widths-and-no-quantization)
- [Why training computes in bf16 and not fp16](#why-training-computes-in-bf16-and-not-fp16)
- [Where the dtype constants live, and whether to pin one](#where-the-dtype-constants-live-and-whether-to-pin-one)
- [Can all three tiers share one dtype](#can-all-three-tiers-share-one-dtype)

## The one-sentence answer per tier

| Tier | Compute | Storage | The governing reason |
|------|---------|---------|----------------------|
| rust-harness | f32 | Holds weights at their published width; writes f32 probe shards | It is the reference implementation the CUDA backend is diffed against; a second *compute* type is a second source of disagreement, but storage width is free because widening is exact |
| interp | The subject model's dtype, so bf16 in practice | fp16 or fp32 in Arrow, per measured magnitudes | Arrow has no bfloat16 in any version, so the storage format cannot follow the compute one |
| training | bf16 | Checkpoints in the parameter dtype, so bf16 | Activation magnitudes in a forward pass are unbounded in advance and fp16 saturates at 65504 |

## Why rust-harness computes in f32

The reason is **reference-implementation simplicity, not numerical safety**. This is worth
stating explicitly because the obvious-sounding argument is wrong.

The wrong argument: "activations are unbounded and a single `inf` at layer 3 poisons every
layer after it, so the harness needs the wider format." That argument is real, but it is an
argument about **fp16**, whose 5-bit exponent saturates at 65504. It says nothing about f32
versus bf16, because f32 has bf16's exponent width exactly — 8 bits — plus 16 more mantissa
bits. Anything bf16 can represent, f32 represents at least as well, and neither overflows where
the other does not. Overflow does not choose between them.

What does choose between them is the harness's purpose.
[`docs/rust-harness/observability.md`](../rust-harness/observability.md) records it: the CPU
backend is hand-rolled plain f32 loops, Candle was evaluated and rejected, and "the bar is
correctness and readability, not throughput" — the backend exists to be the reference the CUDA
path is diffed against. Under that bar:

- **One numeric type in the runtime is one fewer thing that can disagree.** Every arithmetic
  operation in the forward pass is on f32, so there is a single representation to reason about
  and no conversion site inside the numerics to get wrong.
- **Rust has no scalar bf16 arithmetic.** `f32` and `f64` are language primitives with hardware
  operations; bf16 is a library type (`half::bf16`) whose scalar ops are emulated as
  widen-compute-narrow. Computing in bf16 on the CPU would mean converting to f32 per operation
  anyway, paying the conversion without gaining the reference clarity.
- **Widening from the published format is exact.** BF16 is Gemma 4's published storage type and
  what the training tier emits, and bf16-to-f32 is a left shift by 16 with zero fill. The
  checkpoint's own values reach the forward pass unrounded, so f32 costs no fidelity against
  the weights as published.

### Computing in f32 is not the same as storing in f32

The harness holds each weight tensor in **the width its checkpoint declares** and widens per
use: `tensor::Weight` in the runtime, loaded by `Checkpoint::weight`. A bf16 checkpoint stays
bf16 in memory. The arithmetic is still f32 — every multiply and every accumulation — so
nothing above changes.

The two are separable precisely because widening is exact. Widening at load and widening at use
produce the same f32 bits in the same order, so the choice buys memory and bandwidth and costs
no fidelity and no determinism: `cpu.rs`'s
`a_bf16_checkpoint_and_its_widened_twin_produce_identical_logits` compares the two arrangements
bit for bit. E4B was measured at 42 GB resident before the change — a 15 GB file expanded to
28 GiB of f32 with the map still faulted in behind it — and holding it at bf16 removes the
expansion: 28.15 GiB peak on the workstation afterwards, with the ladder table identical in
every digit. The residual is the map itself, still counted while its pages are resident.

Halving the bytes read per token did *not* halve the time — 5%, at the same thread width —
and the reason is recorded in
[`rust-harness-optimization-candidates.md`](../optimizations/ideation/rust-harness-optimization-candidates.md):
the engine sat at both its bandwidth ceiling and a scalar-compute floor at once, and this
change moved only the first. That is a performance finding and not a precision one. Nothing
above depends on it.

Two constraints on that, both load-bearing:

- **Storage-preserving means never narrowing.** An f32 checkpoint stays f32. Narrowing one to
  bf16 to save memory would be a real precision change, and it would be a change to the model
  rather than to the loader. `an_f32_checkpoint_is_not_narrowed` is the guard.
- **This applies to weights, not to the KV cache.** A weight is read from a file that already
  chose its width; a KV entry is *computed* in f32 and would be narrowed on write, which is
  lossy. Narrowing the cache is a separate decision with a separate measurement, and the `HKV1`
  snapshot format reserves a dtype byte for the day it is taken.

  The cache was instead made smaller without touching its dtype: a sliding-window layer now
  retains `window - 1 + group` rows rather than the whole sequence, which took the E4B cache
  from 14.05 GiB to 4.05 GiB at the trained context of 131072 tokens. That is the residency
  lever spent. What remains is 4.000 GiB of global layers, which hold the context by
  definition, so any further reduction is a dtype decision and would be the lossy kind.
  [README.md](README.md#kv-cache) records the measurement.

This is also a fidelity statement and not only a memory one. Interpretability runs at bf16, so a
substrate that stores what it studies is describing the same object its experiments are about.

What remains resident beyond the weights themselves — the mapping they were copied out of —
is a residency question rather than a dtype one, and is tracked in
[`docs/optimizations/ideation/rust-harness-optimization-candidates.md`](../optimizations/ideation/rust-harness-optimization-candidates.md).
Neither that nor anything else on that list changes the compute dtype.

**What would change this decision:** the harness acquiring a throughput obligation, or a CUDA
backend whose numerics need diffing against a bf16 CPU path rather than an f32 one. Neither is
true today.

## Why the harness accepts three float widths and no quantization

Model weights arrive as a Hugging Face checkpoint — `config.json` plus `model.safetensors` —
and `widen` in
[`rust-harness/runtime/src/checkpoint.rs`](../../rust-harness/runtime/src/checkpoint.rs) admits
exactly F32, F16, and BF16. Any other dtype is a hard error naming the tensor and the reason.

**The refusal of quantized weights is the decision, and it is deliberate rather than
unimplemented.** For a model format the reference implementation and the specification are the
same artifact, so a converter's metadata can only imply what a checkpoint states; that argument
is recorded in the repository's `AGENTS.md` and is why GGUF stopped being a model format here.
On top of it sits the precision argument: this workspace computes at bf16 or wider on purpose,
because the numerics have to line up with what the interpretability pipelines record. A probe
fitted on bf16 activations and evaluated against int8-served activations is measuring a
different model, and no error announces it.

**The three float widths are kept, and the set is not a preference.** It is the set of widths a
checkpoint we might be handed can declare. F32 and F16 cost one conversion function each and
nothing in the hot path, and refusing them would remove the ability to *read* a file rather than
the ability to *prefer* BF16. BF16 is the reference format rather than a tolerated one: it is
what Gemma 4 is published in and what the training tier produces, so widening it to f32 is
lossless and the checkpoint's own values reach the forward pass unrounded.

**GGUF is no longer read here.** `gguf.rs` and the `GGML_*` storage-type constants
in `tensor.rs` were deleted at R10. They had survived P8b to read one artifact, the
control-vector image `interp/steering/export.py` writes, kept in that container only because
llama.cpp read it too; retiring the llama.cpp steering path removed that reason, and the
artifact followed the probe to safetensors
([`schemas/control_vector_artifact.json`](../../schemas/control_vector_artifact.json),
[`schemas/probe_artifact.json`](../../schemas/probe_artifact.json)). `GGML_Q8_0` had already
gone with the model path that needed it. What remains in `tensor.rs` is the F16 and BF16
widening the safetensors loader uses, which is a dtype question and not a container one.

**What would change this decision:** a checkpoint we need to read that is published only in a
quantized format, which would be a decision about which model to study rather than about the
loader. Nothing about performance would change it; quantization here buys memory at the cost of
the thing the tier exists to measure.

## Why training computes in bf16 and not fp16

Here the exponent argument *is* the argument, and it is the one recorded in
[README.md](README.md#why-bf16-is-the-house-compute-dtype). A forward pass generates values
nobody measured in advance; residual-stream outlier features grow with depth; fp16 saturates at
65504; and an `inf` at any layer propagates to every layer after it. bf16 has fp32's exponent
range, so a bf16 forward pass either matches fp32 or overflows where fp32 would too.

fp16 training is viable in the wider world, but only with **loss scaling** — a dynamic
multiplier applied before backward and divided out after, plus overflow detection that discards
poisoned steps. That is machinery this repository does not have and does not need: the hardware
is a single RTX 5090, which supports bf16 natively at the same throughput as fp16, so the entire
apparatus would buy nothing.

This is visible in the schema as a narrowing rather than a stated policy.
`PRECISION_DTYPES` admits only `bf16` and `fp8`, while `COMPUTE_DTYPES` (governing
`peft.compute_dtype` and `harvest.dtype`) admits `bf16`, `fp16`, and `fp32`. Two artifacts of
that split are unreachable as written — `training/tune/sft.py` computes an `fp16` trainer flag
and `training/calculator/components.py` branches on `!= "fp32"`, neither of which
`PRECISION_DTYPES` permits. Whether that is deliberate or an untrimmed narrowing cannot be
determined from the source; [README.md](README.md#the-two-allowed-dtype-sets-in-the-training-schema)
records the ambiguity rather than resolving it.

**What would change this decision:** hardware without native bf16, or a numerical result showing
bf16's 7 mantissa bits limiting convergence on a run we care about. The second would be a
measurement, not an argument.

## Where the dtype constants live, and whether to pin one

The constants exist and are already the single source per tier:

| Constant | Governs | File |
|----------|---------|------|
| `widen` | Safetensors dtypes the model loader accepts: F32, F16, BF16 | `rust-harness/runtime/src/checkpoint.rs` |
| `GGML_F32`, `GGML_F16`, `GGML_BF16` | GGUF storage types accepted in the interp control-vector artifact, not in models or probes | `rust-harness/runtime/src/tensor.rs` |
| `DTYPE_BYTES` | Bytes per element for the disk and budget arithmetic | `interp/harvest.py` |
| `ARROW_STORAGE_DTYPES` | The subset Arrow can store, so the subset a capture can be taken at | `interp/harvest.py` |
| `DTYPE_NAMES` | Recipe spelling to the SAELens/torch name; shared with `interp/sae/train.py` | `interp/harvest.py` |
| `PRECISION_DTYPES`, `COMPUTE_DTYPES` | The two allowed sets for training recipe fields | `training/recipes/schema.py` |

**Do not add a single house-wide dtype constant.** It would be the wrong shape for the fact it
claims to hold. A capture dtype is correct or incorrect *per subject*, and the measurement does
not transfer: Gemma 4 E2B peaks at 127.5 and Gemma-2-2B at 2960, a 23x gap, measured on the
workstation 2026-08-07 with
[`scripts/interp/activation_magnitude_probe.py`](../../scripts/interp/activation_magnitude_probe.py).
A constant named something like `STORAGE_DTYPE = "fp16"` would encode one subject's measurement
as a global truth, and the next subject to overflow would do so silently.

What the recipe field plus the per-subject probe already gives is better: the dtype is stated
per capture, the probe exits non-zero when the margin is under 8x so it can gate a capture, and
`tests/interp/test_harvest.py` requires every shipped recipe to name its dtype explicitly rather
than inherit a default.

## Can all three tiers share one dtype

Not one. Two, split by what the number is for.

**Compute cannot converge**, and the reason is not that the tiers disagree. It is that they run
on different machines under different obligations. Training and interp both compute in bf16 on
the GPU and already agree. rust-harness computes in f32 on the CPU because it is a reference
implementation in a language with no scalar bf16 arithmetic — and since bf16-to-f32 is exact,
its f32 is not a *different* answer from bf16, it is the same answer carried at higher
precision. There is nothing to reconcile.

**Storage has converged, and the convergence is real.** bf16 is the published Gemma 4 format,
what training checkpoints emit, and what rust-harness reads as its reference type. The one
apparent exception is the interp activation cache, which stores fp16 — and that is not a
precision compromise. bf16 carries 8 significand bits and fp16 carries 11, so a bf16 value
inside fp16's exponent range is stored **exactly**; the mantissa widens and only the exponent
narrows. Measured per subject, both current subjects sit far inside that range. The interp
capture is bf16 data, losslessly held in a container Arrow can express.

So the honest summary is:

- **bf16 wherever a real forward pass happens**, and wherever weights are stored.
- **fp16 for measured activation storage**, exact for bf16 data whose magnitudes have been
  probed, and half the disk of fp32 — which is what made the 1.38 TB reference capture fit.
  This is the Arrow harvest only. Probe capture shards are float32 by contract, because that
  format has two writers in two languages and no per-subject measurement in the loop; see
  [README.md](README.md#activation-storage-on-disk).
- **f32 in the Rust reference backend**, exactly widened from bf16 and therefore not a divergent
  answer.

The one thing that must not be assumed is that a magnitude measurement transfers between models.
It does not, and the two measured so far differ by 23x.

## Related

- [README.md](README.md) — what each tier does, by concern
- [`docs/interp/harvest-precision-and-resume.md`](../interp/harvest-precision-and-resume.md) —
  the Arrow bfloat16 defect (KB-57), the per-subject magnitude measurements, and chunked capture
- [`docs/rust-harness/observability.md`](../rust-harness/observability.md) — why the CPU backend
  is hand-rolled f32 loops
