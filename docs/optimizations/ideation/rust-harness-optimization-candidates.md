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

- **Splitting a reduction** — partitioning one dot product's summation across
  lanes or threads, or folding a multiply and an add into an FMA. The
  summation order changes, so the result changes. These ship behind a flag
  with a reference fallback, the golden tests continue to pin the reference
  path, and the optimized path gets its own tolerance-based comparison rather
  than inheriting the bit-exact goldens.
- **Splitting across independent outputs** — assigning whole output rows of a
  matvec to different cores, or evaluating several output rows in the lanes of
  one vector register. Nothing is reordered: each output element is still one
  sequential fold in the original order, and the partition is a function of
  the output length and the width alone. The result is identical bit-for-bit,
  so the goldens pin the optimized path directly and there is no second answer
  to compare against.

This document originally applied the first rule to both, and has now been
corrected twice in the same direction. The multithreaded matvec was the first
correction: `cpu.rs`'s `par_rows` carries the argument and
`a_threaded_forward_pass_produces_bit_identical_logits` measures it. **SIMD was
the second.** Vectorizing was listed above as inherently order-changing, which
is true only of vectorizing *within* a reduction; putting R output rows in R
lanes of one register leaves every fold in its original order and is the second
kind. That distinction was worth 1.87x at no cost, and the measured
order-changing alternative added 1%.

Before budgeting a fallback and a tolerance for an item below, check which kind
it is — and check whether the item can be *rewritten* as the second kind, which
is what both corrections turned out to be.

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

**Dropping the map after load.** Landed. Once `CpuModel::from_checkpoint` has
copied every tensor it wants, the mapping is dead weight, but
`harness/src/main.rs` kept `CheckpointFiles` alive for the lazy
`sha256_hex(data)` in the control-vector and probe identity checks — so 15 GiB
of clean page cache stayed resident and counted for the whole run.

The load now happens in a block that yields the model and the snapshot
fingerprint and drops everything else. Because `Checkpoint` borrows from
`CheckpointFiles`, the borrow checker enforces it: a later use of either fails
to compile rather than quietly keeping the map alive.

The two identity checks re-read the weights file instead, through a streaming
`sha256_file`. That also removes a second copy nobody had noticed — the
one-shot `sha256_hex` began with `bytes.to_vec()`, so hashing the weights had
been allocating a *further* 15 GB whenever a probe or control vector was
passed. Peak memory of the digest is now the 1 MiB read buffer.

Expected peak RSS is roughly the owned copy, ~14 GiB, against the 28.15
measured. That is a prediction and not yet a measurement; the previous one on
this axis was wrong, though for a reason now understood and accounted for.

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

**SIMD.** Landed, as the second kind of change: **1.87x, bit-identical, and
without AVX-512.** `SIZE=quick` on E4B at width 8 went 220.38s to 118.08s, and
user time 1501.93s to 713.15s — half the CPU-seconds, so this is work deleted
rather than overlap improved.

`Weight::row_dot_block` folds R output rows against one input in a single pass.
Each row keeps its own accumulator over `in_dim` in index order, so nothing is
reordered and the goldens pin the optimized path directly. What makes it vector
code is that LLVM puts the R row accumulators in the R lanes of one register
and broadcasts `x[k]` across them: `vpmovzxwd` widens R bf16 values, one per
row, `vmulps` against the broadcast, `vaddps` into the lane accumulators. The
bf16 widening that cost 40.43% of `row_dot`'s children disappears into the load
path.

The measurements, five dot kinds against two builds, `SIZE=quick`, threads 8,
wall clock in seconds:

| dot | baseline | znver5 | vs sequential | logits |
|-----|---------:|-------:|--------------:|--------|
| sequential | 220.38 | 216.56 | 1.00x | reference |
| **rows4** | **118.08** | **116.18** | **1.87x** | identical |
| rows8 | 118.90 | 117.64 | 1.85x | identical |
| rows16 | 124.26 | 122.68 | 1.77x | identical |
| lanes16 | 116.82 | 108.34 | 1.89x | differs |

Four findings, three of which contradict what this section said before.

**Loop shape is nearly everything; the build target is not.** `-C
target-cpu=znver5` is worth 1.6% on rows4 and 7% on lanes16, and nothing at all
on the sequential fold. Rung 1's premise — that the win comes from building for
the machine — was backwards. Codegen inspection under the pinned toolchain
shows why: the multi-row shape emits packed arithmetic on the *baseline* build,
because four f32 lanes is one xmm register and needs no ISA extension.

**Wider is worse.** rows16 loses to rows4 by 5%. Sixteen accumulators need four
registers and sixteen concurrent read streams; four need one register and four
streams. The sweep is the reason this is known rather than assumed, and it is
why R is a named constant with a measurement beside it.

**AVX-512 goes unused, and cannot help — for decode.** At 118s the decode moves
roughly 15 GB per token in about 0.2s — back within sight of the ~96 GB/s
ceiling the f32 engine hit before P10.5. The headroom that measurement bought
has now been spent. Rungs 2 and 3 below are therefore retired rather than
pending: portable SIMD and intrinsics would widen arithmetic that is no longer
the constraint.

That retirement is scoped to decode, and the scope is the point. Decode reads a
weight row to use it once, about 2 FLOP per byte, which is why every arithmetic
widening measured against it came back flat. Prefill over N positions reads the
same row and uses it N times, about 2N FLOP per byte, which walks off the
bandwidth wall onto the compute wall — and the compute wall is where wide
vectors pay. Prefill batching below is what moves the engine there, so these
rungs are retired for decode and live again for prefill. The prefill batching
measurement below now supports that in evidence rather than in argument: with
reuse saturated at batch 64, prefill is compute-bound, which is the condition
under which wide vectors pay at all.

One constraint travels with them. `mul_add` is prohibited workspace-wide
because fused multiply-add keeps a wider intermediate and breaks bit-identical
replay, and a conventional GEMM gets roughly half its peak FLOPS from exactly
that fusion. Any future prefill kernel is therefore capped near half of the
machine's nominal peak. That is an accepted cost of the determinism guarantee,
recorded here so it is not rediscovered as a disappointing benchmark.

**The reordered fold was measured and rejected.** `row_dot_lanes` splits the
reduction into 16 fixed lanes — deterministic, but a different summation order,
so it needs the flag-and-fallback apparatus this document prescribes. It is the
fastest number in the table and it is not adopted: 1.89x against rows4's 1.87x
does not buy a change to every logit the project has published. It stays in the
tree behind `--dot lanes16`, off, because a CUDA kernel will face the identical
question and the precedent is worth keeping cheap.

Worth recording for that future: **the reordered fold is measurably more
accurate.** Against the transformers reference it reports L1 rel_l2 0.000003
where sequential reports 0.000005, and perplexity 55.5427 against sequential's
55.5420 with the reference at 55.5438. That is the textbook result — naive
sequential summation accumulates error as O(n * eps) against O((n/k + k) * eps)
for k accumulators — and it means today's reference path is the least accurate
of the three, not the most. Bit-reproducibility is what it is chosen for;
accuracy is not the argument for it.

Diagnostic that gated all of this: `perf stat` on the ladder reported IPC 2.82
with frontend stalls at 3.2% of cycles, against a stated prediction of
memory-bound at IPC below 1.5. 265 billion instructions for a load and a
six-token prefill is a scalar loop paying one instruction per element per
operation. `stalled-cycles-backend` and `LLC-load-misses` are `<not supported>`
on this kernel and CPU, so the backend-stall fraction is inferred from IPC
rather than counted.

## Tier B: measured moderate wins

**Prefill batching.** Landed, bit-identical, measured at **1.65x on L3 and
1.47x on the ladder**. `CpuModel::forward_batch` puts N consecutive positions through the
stack in one pass, so each layer's weights are read once for the whole batch
instead of once per position: a 256-token window read roughly 14 GiB of weights
256 times and now reads them once. This is the "prefill-heavy optimization"
goal named in the runtime design.

Bit-identity is by construction rather than by tolerance. Each output element
is still one sequential fold over `in_dim` in index order, because position
`p`'s dot against weight row `o` sums the same products in the same order
however many positions are in flight; batching changes only when the fold runs
and which rows are resident while it runs.
`batched_prefill_matches_stepped_decode` asserts exact equality against the
one-position-at-a-time path at every batch width.

Two consequences worth stating rather than discovering. The traversal is now
layer-major, so a `Tap` observes `(0, 0), (0, 1), ..., (1, 0)` instead of
`(0, 0), (1, 0), (0, 1)`; every (layer, position) pair still fires exactly once
and positions still ascend within a layer. And the tied LM head — 1.07 GB, about
7 percent of E4B's weights — is now computed only for the positions a caller
asked for, where `prefill` previously folded the embedding table at every
prefilled position and discarded all but the last.

`--batch stepped|whole|<n>` selects the width and the ladder header reports it.

E4B, `--size quick --dot rows4 --threads 8`, per-rung wall clock:

| batch | L1 | L2 | L3 | L4 | rung sum | process wall | user |
|---|---|---|---|---|---|---|---|
| stepped | 0.91s | 1.04s | 90.36s | 23.64s | 116.45s | 125.08s | 727.53s |
| 64 | 0.52s | 0.65s | 54.88s | 22.66s | 79.18s | 84.78s | 508.73s |
| 256 | 0.51s | 0.65s | 55.02s | 22.65s | 79.31s | 84.97s | 510.04s |

All three tables agree in every digit, including `ppl 55.5420 against 55.5438`
and `worst token nll 0.001248`. Peak RSS 14.27 to 14.28 GiB — ten megabytes,
against a predicted "tens of MB".

**The recorded prediction was 120s to 50-70s with L3 collapsing several-fold,
and it was wrong.** The shape held: L4 is 96 percent decode-bound work and did
not move. The magnitude did not: L3 fell 1.65x, not several-fold. The 1.65x is
itself a floor on the prefill speedup, because L3 prefills a window and then
scores it, and the scoring does not batch.

Two readings of this table matter more than the speedup, and both are about
what is no longer true:

**Batch 64 and batch 256 are the same run** — 54.88s against 55.02s, inside
noise. Quadrupling the reuse bought nothing. Reuse only pays while memory
traffic is the limiter, so this is the direct statement that after batching,
**it is not**. The engine is off the bandwidth wall it sat against for every
measurement in this document above, which is what the package was for.

**User time fell nearly in proportion to wall clock**, 727.5s to 508.7s at a
roughly constant 6x CPU. Had the stepped arm been stalled on memory, batching
would have cut wall clock while leaving user time closer to flat. The CPU is
doing measurably less work, not less waiting, so the remainder is arithmetic.

The default is therefore **64, not the 256 it shipped at**. The original
reasoning — amortize as far as scratch allows — assumed a bandwidth limit that
this table refutes above 64. Reserving four times the scratch for a width
measured to buy nothing is indefensible once measured.

**Next experiment, stated before running it.** Kernel (a) calls
`row_dot_block::<R>` once per position, so it widens the same R bf16 weight
rows to f32 again for every position: at N=64 each weight element is converted
64 times to be used 64 times. Batching removed the redundant reads and left the
redundant conversions untouched. Widening the R x `in_dim` block into an f32
scratch once and looping the N positions against it is 32 KB at R=4 — L1
resident — needs no change to the activation layout, and is bit-identical for
the same reason batching is. If L3 falls again, the R x N micro-kernel and its
`[k][N]` packing earn their complexity. If L3 does not move, the remaining time
is the scalar multiply-add chain itself and the next lever is vectorizing it,
under the FMA cap recorded above. Deliberately not bundled with the packing:
two bets in one commit make a disappointing result unattributable, which is the
mistake the reordered-fold experiment already caught once.

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

**Huge pages.** Now the most likely next lever, and the argument against it has
expired. The TLB-pressure reasoning behind the parent repository's 26%
llama.cpp improvement transfers to any pass that walks gigabytes of weights,
and the mechanism now transfers too: `hugepage_mmap_wrapper.cpp` intercepts
`mmap()` of large files, which this runtime no longer relies on — since P10.5
the weights are owned `Vec<u16>` buffers this crate allocates itself, and since
the SIMD measurement the engine is bandwidth-bound again with the arithmetic no
longer in the way. 14 GiB of weights on 4 KiB pages is roughly 3.7 million
pages against an L2 TLB holding a few thousand entries; the last measured run
took 3,887,891 minor faults, which is that number. 2 MiB pages would cut it by
512x.

This converges with the loader change under "Zero-copy weights" in Tier A: reading
each tensor with `pread` into a buffer allocated `MAP_HUGETLB` from the
reserved pool gets huge pages, drops the load-time RSS peak from 28 GiB to
~14, and removes `memmap2` in one change. The cost is that it keeps
`deny(unsafe_code)` where a `pread`-only version would restore `forbid`.

**Not yet measured**, and it should be before it is written: the `perf stat`
run that produced the IPC figure omitted the TLB counters, so the page-walk
cost is currently inferred from the fault count rather than counted.
`dTLB-load-misses` plus a table-walk event on the multi-row path is the
prerequisite.

The workstation disables transparent huge
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
