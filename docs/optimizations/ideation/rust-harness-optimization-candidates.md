# x86 CPU inference optimization candidates

**Parked proposals — recorded 2026-07-30. Blocked on runbook entry RH.1 and a recorded
baseline; this document is edited only as items graduate out.**

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
associative, so a change to *reduction order* can change logits bit-for-bit
even when both answers are correct. Most inference engines optimize without
this budget; this one cannot.

Two kinds of change get confused here, and the distinction decides how much
apparatus an item needs:

- **Splitting a reduction** — threading a dot product, widening one with SIMD,
  folding a multiply and an add into an FMA. The summation order changes, so
  the result changes. These ship behind a flag with a reference fallback, the
  golden tests continue to pin the reference path, and the optimized path gets
  its own tolerance-based comparison rather than inheriting the bit-exact
  goldens.
- **Splitting across independent outputs** — assigning whole output rows of a
  matvec to different cores. Nothing is reordered: each output element is
  still one sequential fold in the original order, and the partition is a
  function of the output length and the width alone. The result is identical
  bit-for-bit, so the goldens pin the optimized path directly and there is no
  second answer to compare against.

This document originally applied the first rule to both. The multithreaded
matvec, which landed as the second kind, is the correction: `cpu.rs`'s
`par_rows` carries the argument, and
`a_threaded_forward_pass_produces_bit_identical_logits` measures it. Before
budgeting a fallback and a tolerance for an item below, check which kind it
is.

## Tier A: order-of-magnitude candidates

**Multithreaded matvec.** Landed. Every weight matrix-vector product is
embarrassingly parallel across output rows, and the tied LM head — a
hand-rolled row fold rather than a `matvec_into` call, reading 2.1 GiB of
E4B's roughly 28 per token — is the same shape and would otherwise have been
the Amdahl floor under the rest. `cpu.rs`'s `par_rows` splits the row range
with `std::thread::scope`: no unsafe code, compiler-verified absence of data
races, and no new dependency (rayon would also work but costs a crate in an
offline-first repository, and its work stealing is wasted on equal chunks
whose partition has to stay fixed anyway).

Bit-identical at every width, per the second case above, so the goldens pin
the parallel path and `--single-thread` exists for rr ergonomics rather than
as a reference fallback. `PARALLEL_MIN_ROWS` keeps small projections serial,
which is also why the Tier T fixture's goldens still exercise the serial path.

Measured on E4B, `SIZE=quick`, 32-core workstation, width 32:

| | serial | width 32 |
|---|---|---|
| wall | 18:07.69 | 4:25.43 |
| per token | 1.63 s | 0.40 s |
| user | 1093.99s | 3463.40s |
| system | 3.95s | 424.64s |
| cpu | 100% | 1464% |

**4.10x, and the ladder table unchanged in every digit.** Two things the
speedup alone hides. Only 14.6 of 32 cores were ever busy, and system time
rose 107x — that is `clone` and futex, roughly 12,000 thread creations per
token at width 32 over ~380 `par_rows` calls. And 30.1 GB of weights read in
0.398s is about **76 GB/s**, which on dual-channel DDR5-6000 is at or near
what the memory controller will deliver.

If that reading holds, the row split has taken essentially all there is to
take, and the wasted CPU is cores idling on memory rather than lost wall
clock. `DEFAULT_MAX_THREADS` caps the default at 8 on that basis, with
`--threads` for the sweep that will set the real number — llama.cpp
optimizes near 12 on this class of machine, but its prefill is a batched GEMM
and this engine's is `forward_one` in a loop, so the number does not carry
over. The next real speedup is fewer bytes per token, not more cores: see
native-width weight storage below.

**Native-width weight storage.** The backend currently widens all weights to
f32 at load time: a 2x blow-up for BF16, roughly 4x for Q8_0, worse for the
planned k-quants. BF16 is the case that matters first, because it is the
published Gemma 4 storage type and the only one the interp and training
tiers can share without losing fidelity. Holding weights as bf16 and
widening per use is a shift in the inner loop, needs no new kernel, and
halves E4B from about 28 GiB resident to 14 — which is the difference
between not fitting and fitting in the 5090's 32 GB for the CUDA tier.
`tensor::tensor_bytes` exists as the seam: it does the bounds checking a
storage-preserving loader needs, so such a loader reuses it rather than
restating it.

**Quantized-weight compute.** Beyond bf16, the mature step is block-wise
integer arithmetic. The core of llama.cpp's CPU performance keeps
weights quantized in memory and computes block-wise integer dot products,
dequantizing only per-block scales on the fly. Matvec
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
128k-token worst case from 14.00 GiB to 7.00 GiB. Shared KV removes 18 of
E4B's 42 layers from the cache, leaving 24 owning layers: four global at head
dim 512 and 20 sliding at 256, each holding K and V for two KV heads across
131072 positions. Bounding windowed layers to their attention window (a ring
buffer instead of full-sequence storage) shrinks it further, but not across
all 20 sliding owners — a sliding layer that donates its states to a shared
layer must retain them full-length (`cpu.rs:24-27`). The `HKV1` snapshot
format already reserves a dtype byte for f16, so the on-disk format absorbs
this without a version bump. Numerics effect on golden tests must be
measured, not assumed.

**Huge pages.** The TLB-pressure argument behind the parent repository's
26% llama.cpp improvement transfers to any pass that walks gigabytes of
weights, but the mechanism does not: the existing
`hugepage_mmap_wrapper.cpp` intercepts `mmap()` of large files, and this
runtime reads the GGUF with `std::fs::read` and walks dequantized heap
buffers, so there is no mapping to intercept. Owning the allocation sites
allows direct allocation instead. The workstation disables transparent huge
pages entirely and reserves a 90 GB explicit pool (`vm.nr_hugepages`, see
`docs/optimizations/README.md` in the parent repository), so
`madvise(MADV_HUGEPAGE)` is a no-op there; the working approach is explicit
`MAP_HUGETLB` allocation for the weight and KV buffers with a fallback
chain to regular pages, drawing from the same reserved pool llama-cpu
uses. Acceptance evidence is host-side: tokens/second with and without,
plus `HugePages_Free` movement in `/proc/meminfo`.

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
