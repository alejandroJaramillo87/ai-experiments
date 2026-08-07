# Python pipeline optimization candidates

**Parked proposals — recorded 2026-07-30. Items graduate into the training and interp
tier documentation once measured; this document is edited only as they do.**

A recorded menu of optimizations for the training tier (`training/runtime/`)
and the interpretability tier (`interp/probes/`, `interp/sae/`,
`interp/steering/`), for implementation after the training and interp
verification passes produce real-model baselines. Nothing here is a
commitment: each item lands as its own change with a measured before/after
comparison, one at a time, mirroring the pin-then-fix-then-update
discipline. The sibling menu for the rust runtime is
`docs/optimizations/ideation/rust-harness-optimization-candidates.md`.

These pipelines are batch pipelines, so the measurement currency is
wall-clock per campaign and artifacts per hour, not tokens/second. Target
hardware is the RTX 5090 (32 GB VRAM, Blackwell, native FP8 tensor cores),
AMD Ryzen 9950X, and 128 GB DDR5-6000.

## Table of contents

- [The provenance constraint](#the-provenance-constraint)
- [Tier A: order-of-magnitude candidates](#tier-a-order-of-magnitude-candidates)
- [Tier B: measured moderate wins](#tier-b-measured-moderate-wins)
- [Tier C: situational](#tier-c-situational)
- [Sequencing](#sequencing)

## The provenance constraint

Probes and steering directions are valid only against activations from the
same numerical world, which is why capture manifests carry model hashes and
tokenizer fingerprints and why the rust capture path exists at all. Most
attractive optimizations — `torch.compile`, FP8, attention-kernel swaps —
change forward-pass numerics. Not by much, but "not by much" is exactly the
bf16-versus-quantized gap the rust substrate was built to close.

The standing rule: any optimization that alters capture-path numerics
forces a re-capture and re-train of downstream artifacts, bumps the
fingerprints the manifests carry, and lands only between experiment
generations, never mid-campaign. The analysis half (probe fitting,
evaluation, export) has no such coupling; its speed is unconstrained.

## Tier A: order-of-magnitude candidates

**`torch.compile` on the training and SAE loops.** `training/runtime/loop.py`
and `interp/sae/train.py` hold the sustained GPU work; compilation typically
buys 1.3-2x on training steps, more where kernel-launch overhead dominates,
and Blackwell benefits from CUDA-graph capture. Compiling the training loop
is provenance-safe (the artifact is a checkpoint, not intermediate
numerics); compiling the capture forward pass is a numerics change and
follows the constraint above.

**FP8 training on Blackwell.** The RTX 5090 executes FP8 matmuls natively;
Transformer Engine-style FP8 in the fine-tuning path can approach 2x over
bf16 with loss-scaling care. Highest effort on the list, cleanly confined
to the training tier. Caveat: for LoRA-scale fine-tunes the win shrinks,
because matmul time dominates less; the full per-variant feasibility
arithmetic lives in
[training-scale-reference.md](../../training/training-scale-reference.md).

**Batched LLM-judge and generation stages through vllm-gpu.** The steering
scorecard judge and generation-heavy evaluations are throughput workloads,
and a batch-serving endpoint sits on port 8003. Sequential requests to a
latency-tuned endpoint are the classic order-of-magnitude loss; submitting
a whole scorecard as one batched run reorders requests only, so this is the
rare Tier A item with zero numerics implications.

## Tier B: measured moderate wins

**Capture throughput.** `interp/probes/capture.py` already batches at bf16
under nnsight. Remaining knobs: batch size tuned to measured VRAM headroom,
pinned-memory host transfers for harvested residuals, capturing only the
layers a campaign needs, and measuring nnsight tracing overhead against a
plain hooked forward before assuming it is free.

**Shard format and loading.** The npz shards are compressed and fully
materialized on load. For probe and SAE training that re-reads shards
across epochs, memory-mapped uncompressed storage lets the 128 GB of host
RAM act as the cache it is. Profile first: for linear probes, loading may
dominate the entire job; for SAE training it may be invisible.

**Dataloader and CPU topology.** The training tier already has
`training/runtime/affinity.py` and `offload.py`; the remaining standard
items are pinned-memory dataloaders with prefetch overlapped into the
step and worker counts matched to the core-allocation conventions used by
the inference services.

**Gradient checkpointing and paged optimizers, per model size.** Both trade
compute for memory; the right setting on a 32 GB card depends on which
fine-tune sizes actually fit. The optimization is recording the choice per
model size in the memory-fit calculator rather than fixing a global
default.

## Tier C: situational

**Pipelining capture into training.** Streaming shards into probe or SAE
training as they are produced would hide capture latency entirely, but it
creates two GPU consumers, and the repository rule is one GPU consumer at a
time for VRAM and result-integrity reasons. Parked; if a campaign's
wall-clock ever justifies it, it runs as a single orchestrated job, not
two concurrent ones.

**Probe-training micro-optimization.** Linear probes on cached activations
complete in seconds; work here is noise.

**Host-side huge pages.** The 26% llama.cpp result does not transfer:
PyTorch pinned allocations and GPU-resident compute have a different
bottleneck profile, and the host disables transparent huge pages in favor
of the explicit pool reserved for llama-cpu. Only worth revisiting if
profiling shows host TLB pressure.

## Sequencing

1. The training and interp verification passes produce real-model baselines
   with recorded wall-clocks; no optimization work before those numbers
   exist.
2. Profiles decide the order; several items above are only decidable with
   baseline measurements in hand.
3. Items land one at a time with before/after wall-clock, and anything
   touching capture numerics carries an explicit provenance bump and a
   scheduled re-capture.
