# x86 CPU inference optimization candidates

A recorded menu of optimizations for the CPU backend
(`runtime/src/model/cpu.rs`), for implementation after host verification.
Nothing here is a commitment: every item is blocked on runbook entry RH.1
(the real E4B checkpoint loading at all) and a recorded baseline benchmark,
and each item lands as its own commit with a measured before/after
tokens/second comparison, mirroring the repository's pin-then-fix-then-update
discipline. Target hardware is the AMD Ryzen 9950X (16 Zen 5 cores, AVX-512,
DDR5-6000).

## Table of contents

- [The determinism constraint](#the-determinism-constraint)
- [Tier A: order-of-magnitude candidates](#tier-a-order-of-magnitude-candidates)
- [Tier B: measured moderate wins](#tier-b-measured-moderate-wins)
- [Tier C: situational](#tier-c-situational)
- [Sequencing](#sequencing)

## The determinism constraint

The CPU backend exists to be a boring, rr-debuggable, bit-reproducible
reference: golden tests pin exact logits, the flight recorder replays runs,
and the KV snapshot restore is bit-identical. Floating-point addition is not
associative, so any change to reduction order — threading a dot product,
widening it with SIMD — can change logits bit-for-bit even when both answers
are correct. Most inference engines optimize without this budget; this one
cannot.

The standing rule for every item below: parallel and vectorized paths ship
behind a flag with a reference fallback (in the style of the planned
`--single-thread`), the golden tests continue to pin the reference path, and
the optimized path gets its own tolerance-based comparison against the
reference rather than inheriting the bit-exact goldens.

## Tier A: order-of-magnitude candidates

**Multithreaded matvec.** Every weight matrix-vector product is
embarrassingly parallel across output rows. `std::thread::scope` splits the
row range across cores with no unsafe code and compiler-verified absence of
data races, and avoids a new dependency (rayon would also work but costs a
crate in an offline-first repository). Expect near-linear scaling on
prefill; decode matvec is memory-bandwidth-bound, so DDR5-6000 will likely
saturate at roughly six to eight effective cores. Requires the
`--single-thread` fallback for rr and for the bit-exact goldens.

**Quantized-weight compute.** The backend currently dequantizes all weights
to f32 at load time, an approximately 4x memory blow-up for Q8_0 and worse
for the planned k-quants. The mature alternative — the core of llama.cpp's
CPU performance — keeps weights quantized in memory and computes block-wise
integer dot products, dequantizing only per-block scales on the fly. Matvec
is bandwidth-bound, so reading 4-bit weights moves roughly 8x less data than
f32; a Q4 E4B stays near 4 GB resident instead of tens of GB. This item
subsumes the deferred Q4_K/Q6_K dequantization work: the kernels replace it
rather than sit beside it. Largest effort on the list.

**SIMD.** Zen 5 executes full-width AVX-512. Three rungs, cheapest first:

1. **Autovectorization.** Build with `-C target-cpu=znver5` and shape hot
   loops so the compiler vectorizes them; `chunks_exact` also eliminates
   bounds checks. A build-configuration change plus loop hygiene, no unsafe.
2. **Portable SIMD.** `std::simd` remains nightly-only; on stable, the
   idiomatic path is rung 1 plus inspecting generated assembly to confirm
   vectorization.
3. **Intrinsics.** `core::arch::x86_64` AVX-512 intrinsics behind
   `is_x86_feature_detected!`, in `unsafe` blocks with safety comments —
   only for loops where profiles show rung 1 left real throughput on the
   table.

## Tier B: measured moderate wins

**Prefill batching.** Prefill processes one position at a time; processing
all prompt positions per layer as one matrix-matrix multiply reuses each
weight block across positions, turning a bandwidth-bound loop into a
compute-bound one. This is the "prefill-heavy optimization" goal named in
the runtime design.

**Scratch-buffer reuse.** If profiling (flight recorder plus `make flame`)
shows per-token temporary allocations in the forward pass, a preallocated
scratch workspace removes allocator traffic and stabilizes latency. Profile
first; do not assume.

**f16 KV cache and windowed ring buffers.** Halving KV element size cuts the
128k-token worst case from roughly 17.5 GB to 8.7 GB, and bounding windowed
layers to their attention window (a ring buffer instead of full-sequence
storage) shrinks it further. The `HKV1` snapshot format already reserves a
dtype byte for f16, so the on-disk format absorbs this without a version
bump. Numerics effect on golden tests must be measured, not assumed.

**Huge pages.** The TLB-pressure argument behind the parent repository's
26% llama.cpp improvement transfers to any pass that walks gigabytes of
weights, but the mechanism does not: the existing
`hugepage_mmap_wrapper.cpp` intercepts `mmap()` of large files, and this
runtime reads the GGUF with `std::fs::read` and walks dequantized heap
buffers, so there is no mapping to intercept. Owning the allocation sites
allows a direct approach: `madvise(MADV_HUGEPAGE)` on the multi-MB weight
and KV allocations first (one libc call per buffer, fail-safe, no host
reservation needed), and explicit `MAP_HUGETLB` allocation with a fallback
chain only if measurement shows transparent huge pages under-delivering.
Acceptance evidence is host-side: tokens/second with and without, plus
`AnonHugePages` in `/proc/meminfo`.

## Tier C: situational

**Speculative decoding.** A draft model proposing tokens the target model
verifies in one batched pass. Real decode speedups elsewhere, but it
requires a second resident model and complicates the determinism, session,
and KV-snapshot stories; parked without a design.

**Sampler and tokenizer micro-optimization.** Negligible next to the forward
pass; only worth attention if profiles disagree.

**The cuda backend.** Roadmap phase A remains the largest speedup available
and deliberately obsoletes CPU-side serving speed. The CPU path is optimized
to make workstation experiments pleasant while remaining the rr-debuggable
reference — not optimized maximally.

## Sequencing

1. RH.1 decides whether the real checkpoint loads; no optimization work
   before that verdict.
2. A recorded baseline benchmark on the workstation (per the repository
   benchmarking protocol) anchors all comparisons.
3. Items land one at a time, ordered by measured value, each with its own
   before/after numbers and with the reference path still pinned by the
   golden tests.
