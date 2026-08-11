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
clock. `DEFAULT_MAX_THREADS` caps the default at 8 on that basis, and the
subsequent sweep confirmed the number rather than merely permitting it:

| threads | wall | user | system | cpu |
|---|---|---|---|---|
| 4 | 319.44s | 1166.85s | 20.81s | 371% |
| 8 | 236.43s | 1548.63s | 40.35s | 672% |
| 12 | 237.21s | 2244.08s | 69.46s | 975% |
| 16 | 221.72s | 2636.56s | 94.97s | 1231% |

Scaling stops at 8. Twelve is *slower* than eight for 45% more CPU, sixteen
buys 6% of wall clock for 70% more, and every one of the four beats the
width-32 run above. llama.cpp optimizes near 12 on this class of machine, but
its prefill is a batched GEMM and this engine's is `forward_one` in a loop, so
the number did not carry over and had to be measured.

The shape of that curve says something the speedup does not. Fitting Amdahl to
the 4-and-8 points leaves roughly 150 of the 236 seconds not scaling with
width at all. Some of that is the checkpoint load, which is fixed and serial;
some is attention, which the row split deliberately left sequential. The split
between them is unmeasured, and `make flame` is the cheapest way to settle it.

**Native-width weight storage.** Landed. The backend used to widen every
weight to f32 at load, a 2x blow-up on the published BF16 — and, because the
widening reads every page of the map, it left the 15 GB file resident behind
the 28 GiB expansion. That is where E4B's 42 GB came from.

`tensor::Weight` now holds each tensor at the width its checkpoint declares and
widens per use. This is the second kind of change from the determinism section
above, not the first: widening is exact for bf16 (a 16-bit shift) and for f16
(10 mantissa bits into 23), so widening at use produces the same f32 operands
in the same order as widening at load. The goldens pin the native path
directly, and `a_bf16_checkpoint_and_its_widened_twin_produce_identical_logits`
compares the two arrangements bit for bit.

Storage-preserving rather than bf16-converting: an f32 checkpoint stays f32,
because narrowing one would be a precision change to the model rather than a
loader optimization. `an_f32_checkpoint_is_not_narrowed` is the guard.

Measured on E4B, `SIZE=quick`, width 8, against the same ladder at the same
width before the change:

| | f32 storage | native bf16 |
|---|---|---|
| wall | 236.43s | 224.94s |
| user | 1548.63s | 1538.83s |
| max RSS | — | 28.15 GiB |

**The five-rung table is identical in every digit**, which is the claim the
package existed to make.

Memory behaved as designed. 42 GB against 28.15 is the 14 GiB of f32 expansion
removed, and the arithmetic closes: 14 GiB of owned bf16 plus roughly 15 GiB of
still-resident mapped pages, the latter visible as 3.9M minor faults. Peak RSS
counts mapped file pages while they are resident, which is why the figure is
not the ~15 GiB predicted — the prediction was about the owned copy and the
measurement is about both.

**Speed did not follow, and that corrects the diagnosis above.** 5%, not the
near-halving the bandwidth reading implied, and user time barely moved — if
the engine had been purely bandwidth-bound, halving the bytes would have cut
wall clock and CPU-seconds together. The two rates settle it: the f32 run
sustained 30.1 GB per token in 0.354s, about **85 GB/s**, near the ~96 GB/s
dual-channel DDR5-6000 ceiling; the bf16 run needs only ~45 GB/s and is no
faster. The same machine had just demonstrated it can deliver twice that.

So the engine was within a few percent of *both* ceilings at once. Halving the
bytes moved the memory ceiling out of the way and landed immediately on a
compute floor underneath it — the scalar multiply-accumulate loop, plus one
widening shift per element, which is why user time is flat rather than lower.

That makes this change an enabler rather than a win. It bought the headroom
SIMD needs: widening the arithmetic without it would have hit 85 GB/s within a
factor of two.

The paired change is `checkpoint::fingerprint`, which identified a checkpoint
by hashing all 15 GB of it, one byte at a time, at every startup, to decide
whether a KV snapshot was still valid. It now hashes the safetensors header,
`config.json`, and the file length. Both were mmap-laziness questions, which is
why they were decided together.

**Dropping the map after load.** Small, and worth roughly 14 GiB. Once
`CpuModel::from_checkpoint` has copied every tensor it wants, the mapping is
dead weight — but `harness/src/main.rs` keeps `CheckpointFiles` alive for the
lazy `sha256_hex(data)` in the control-vector and probe identity checks, so
15 GiB of clean page cache stays resident and counted. Those two callers can
re-read the file on demand; they run once, at startup, and only when the
corresponding flag is set. Unmapping puts peak RSS at roughly the owned copy,
~14 GiB, which is the figure the entry above predicted. The evidence is the
`Maximum resident set size` line, and the risk is that a caller of `files` is
missed — `cargo test` plus the ladder cover both identity paths.

Still open beyond that: **zero-copy weights**, borrowing the bf16 bytes out of
the map instead of copying them into an owned `Vec<u16>`. It would remove the
owned copy rather than the mapping, at the cost of a self-referential
`CpuModel`. Taken together with the map drop it is the same ~14 GiB, reached by
the harder route, so the drop comes first and this is revisited only if load
time becomes the complaint.

**Quantized-weight compute.** Beyond bf16, the mature step is block-wise
integer arithmetic. The core of llama.cpp's CPU performance keeps
weights quantized in memory and computes block-wise integer dot products,
dequantizing only per-block scales on the fly. Matvec
is bandwidth-bound, so reading 4-bit weights moves roughly 8x less data than
f32; a Q4 E4B stays near 4 GB resident instead of tens of GB. This item
subsumes the deferred Q4_K/Q6_K dequantization work: the kernels replace it
rather than sit beside it. Largest effort on the list.

**SIMD.** Zen 5 executes full-width AVX-512. This was parked on the grounds
that widening the arithmetic buys nothing while the engine is bandwidth-bound;
the bf16 measurement above falsified that premise, and it is now the item with
the clearest case. The scalar loop is the floor, the bf16 widening vectorizes
into a shift and interleave across a whole vector rather than costing per
element, and there is roughly 2x of bandwidth headroom to spend before the
memory wall binds again.

It stays bit-identical on the same terms as everything else here: vectorize
across *output rows*, where each element is still one sequential fold, and not
inside the reduction. Three rungs, cheapest first:

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
