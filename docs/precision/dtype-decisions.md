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
- [Why the harness accepts four GGUF storage types](#why-the-harness-accepts-four-gguf-storage-types)
- [Why training computes in bf16 and not fp16](#why-training-computes-in-bf16-and-not-fp16)
- [Where the dtype constants live, and whether to pin one](#where-the-dtype-constants-live-and-whether-to-pin-one)
- [Can all three tiers share one dtype](#can-all-three-tiers-share-one-dtype)

## The one-sentence answer per tier

| Tier | Compute | Storage | The governing reason |
|------|---------|---------|----------------------|
| rust-harness | f32 | Reads F32, F16, BF16, Q8_0; writes f32 probe shards | It is the reference implementation the CUDA backend is diffed against; a second numeric type is a second source of disagreement |
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

- **One numeric type in the runtime is one fewer thing that can disagree.** Every supported GGUF
  storage type dequantizes to a dense `Vec<f32>` at load, so the forward pass has a single
  representation and no conversion sites to get wrong.
- **Rust has no scalar bf16 arithmetic.** `f32` and `f64` are language primitives with hardware
  operations; bf16 is a library type (`half::bf16`) whose scalar ops are emulated as
  widen-compute-narrow. Computing in bf16 on the CPU would mean converting to f32 per operation
  anyway, paying the conversion without gaining the reference clarity.
- **Widening from the published format is exact.** BF16 is Gemma 4's published storage type and
  what the training tier emits, and bf16-to-f32 is a left shift by 16 with zero fill. The
  checkpoint's own values reach the forward pass unrounded, so f32 costs no fidelity against
  the weights as published.

The cost is memory: f32 weights are twice the bytes of the file they came from. That is a real
cost and it is priced as a candidate optimization in
[`docs/optimizations/ideation/rust-harness-optimization-candidates.md`](../optimizations/ideation/rust-harness-optimization-candidates.md),
which proposes keeping weights resident in their storage format. That proposal is about
residency; it does not change the compute dtype.

**What would change this decision:** the harness acquiring a throughput obligation, or a CUDA
backend whose numerics need diffing against a bf16 CPU path rather than an f32 one. Neither is
true today.

## Why the harness accepts four GGUF storage types

`GGML_F32`, `GGML_F16`, `GGML_BF16`, and `GGML_Q8_0` are declared in
[`rust-harness/runtime/src/tensor.rs`](../../rust-harness/runtime/src/tensor.rs), and any other
GGML type is a hard error naming the tensor. The question is whether to narrow that set to BF16
alone, since BF16 is the reference format.

**Keep all four.** The set is not a policy about what we should use; it is the set of things
that exist in files we might be handed:

- Narrowing removes the ability to *read* a file, not the ability to *prefer* a format. The
  preference is already expressed where it matters —
  [`docs/rust-harness/gemma-4-ingestion.md`](../rust-harness/gemma-4-ingestion.md) records BF16
  as "the reference format rather than a tolerated one", and refuses requantization to Q8_0
  because interp fits probes on those exact weights.
- Q8_0 support is what makes the quantized serving path measurable at all. The rust probe
  capture (`rust-harness/runtime/src/bin/activation-capture.rs`) exists specifically to capture
  from the quantized path, so probes describe the model that will answer requests. Dropping Q8_0
  would delete that capability.
- The dequantize-at-load design means each type costs one small conversion function and nothing
  in the hot path. There is no ongoing complexity to recover by removing them.

**What would change this decision:** a supported type acquiring a nontrivial maintenance burden,
or evidence that one of them is silently wrong. Neither is a reason to preemptively narrow.

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
| `GGML_F32`, `GGML_F16`, `GGML_BF16`, `GGML_Q8_0` | GGUF storage types the runtime accepts | `rust-harness/runtime/src/tensor.rs` |
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
