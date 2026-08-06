# Interpretability scale reference

A reference for how the training-scale conclusions in
[training-scale-reference.md](../training/training-scale-reference.md)
shape the mechanistic-interpretability program, and for how
interpretability works at inference time on the rust runtime, including
the planned CUDA backend. Nothing in this document is scheduled work: it
records feasibility conclusions and design constraints so later roadmap
tasks start from settled reasoning instead of re-deriving it. Related
documents: [rust-harness-interp.md](rust-harness-interp.md) (the tap-seam
design this document builds on) and
[python-pipeline-optimization-candidates.md](../optimizations/ideation/python-pipeline-optimization-candidates.md)
(the provenance constraint and capture-throughput items), and
[observation-frames.md](../rust-harness/observation-frames.md)
(the agent and chat frames the live-interpretability sections observe
through).

## Table of contents

- [The ladder, inverted](#the-ladder-inverted)
- [Sub-1B from-scratch: the interpretability payoff](#sub-1b-from-scratch-the-interpretability-payoff)
- [Inference-time interpretability on the CPU backend](#inference-time-interpretability-on-the-cpu-backend)
- [The CUDA backend](#the-cuda-backend)
- [Net effect on the interpretability roadmap](#net-effect-on-the-interpretability-roadmap)
- [Sequencing](#sequencing)

## The ladder, inverted

The Gemma 4 training ladder is also an interpretability ladder, inverted
in one important way. Training feasibility decays with model size, while
capture feasibility decays much more slowly: capture is a forward pass,
and forward passes fit where training does not. But interpretability of
models trained on this machine is where the interesting science is, so
the training ladder gates the interpretability program anyway.

| Variant | Capture | Interp-of-training loop | Notes |
|---------|---------|------------------------|-------|
| 2B | full, bf16 | closes fully | before/after studies at experiment cadence |
| 4B | full, bf16 | closes fully | model of record; training ceiling coincides |
| 12B | full, bf16 | generation cadence | PEFT deltas as primary interp objects |
| 31B | quantized only | none | transfer spot-checks; ~62 GB bf16 exceeds VRAM |
| 26B MoE | excluded | excluded | no runtime support, per the training doc |

Reading the table:

- **2B and 4B close the full loop**: continued-pretrain or preference-tune
  a model, re-capture activations through the rust tap seam, retrain
  probes and SAEs, and compare against the base model's artifacts. That
  before/after-training comparison — how did domain adaptation move the
  features, did preference tuning shift the steering directions — is the
  highest-value interpretability experiment this substrate enables, and it
  is only cheap where training is cheap. The 4B coinciding with both the
  harness's model of record and the pure-GPU training ceiling means the
  model of record is also the model where interpretability of training is
  fully practical.
- **12B** supports capture and probe work without difficulty, but training
  turns over in days, so before/after studies happen at generation
  cadence, not experiment cadence. Interpretability here is mostly of the
  base model, plus PEFT deltas — which are a good interpretability object
  in their own right: a LoRA adapter is a low-rank, directly inspectable
  weight change.
- **31B** is capture-only, and only against a quantized base: bf16 weights
  are ~62 GB and exceed the 32 GB of VRAM. Quantized capture is exactly
  the numerical-world divergence the provenance machinery exists to flag,
  so 31B serves as a "does the finding transfer to scale?" spot-check with
  the quantization caveat stamped on every manifest.

Re-capture economics reinforce the same shape. The provenance rule says
every trained checkpoint is a new numerical world requiring full
re-capture and probe/SAE retraining, and that cost scales with model size.
At 2B and 4B a re-capture campaign is hours, so the discipline that looks
expensive in the abstract is nearly free exactly where training happens
most often.

## Sub-1B from-scratch: the interpretability payoff

Phase 4's sub-1B scope reads as a limitation from the training side. From
the interpretability side it is the main event, for four compounding
reasons.

1. **Full-visibility scale.** Sub-1B is where complete-mechanism work
   lives: every layer and head can be captured, SAEs can be trained over
   the whole residual stream, and all artifacts fit in host RAM at once.
   At 4B a campaign samples layers; at a few hundred million parameters it
   takes the whole model. The capture-throughput and shard-format items in
   the optimization-candidates document mostly stop mattering at this
   scale.
2. **Known training history.** Every interpretability study on downloaded
   weights carries a confound: the training data is unknown. A
   from-scratch sub-1B model comes with its corpus, data order, and
   checkpoint series, unlocking experiments that are impossible otherwise:
   feature formation over training time (capture at successive
   checkpoints), memorization-versus-generalization attribution against a
   known corpus, and clean ablation-retrain loops.
3. **Same block, same seams.** Because from-scratch models use the Gemma 4
   block at reduced scale and `Config::from_gguf`
   (`rust-harness/runtime/src/model/cpu.rs:56`) reads every dimension from
   GGUF metadata, every interpretability asset transfers unchanged: tap
   seam, steering hook, probe path, npz capture, goldens methodology. The
   interpretability tier needs no "small model" mode — a sub-1B checkpoint
   exported to GGUF is just another model to the whole stack. The phase 4
   checkpoint-to-GGUF exporter is therefore an interpretability enabler as
   much as a training one, and checkpoint-series capture is its second
   customer.
4. **Bit-reproducibility becomes end-to-end.** On downloaded models the
   runtime's determinism buys clean provenance. On a model whose full
   training history is also controlled, it buys end-to-end reproducible
   science: same corpus, same seed, same checkpoint, same capture, same
   probe. Sub-1B is where that stops being infrastructure virtue and
   becomes a differentiated capability.

The Titans-class research option recorded in the training document
composes with this: the entire scientific interest of a test-time-learning
experiment is interpretability of the memory module, which only means
anything on a substrate that can capture and reproduce it.

## Inference-time interpretability on the CPU backend

The rust runtime makes interpretability a property of the serving path,
not a separate offline mode:

- **The tap seam lives inside the decode loop.** Capture (npz writer),
  steering (control-vector addition), and probing (`ProbeMonitor`) are
  implementations of one seam, with zero-cost passthrough when no tap is
  armed. There is no instrumented build versus production build.
- **Probes run live, per token.** The `--probe`/`--probe-out` wiring
  scores every generated token as it is produced: "is the model in state X
  right now," not "was it in state X when the logs were analyzed later."
- **Steering runs live** during generation, so interventions are online.
- **Determinism makes live observations scientific.** The flight recorder
  plus bit-reproducible forward passes mean any live observation can be
  replayed exactly, including under rr. A probe firing mid-session is not
  an anecdote; it is a reproducible experiment that can be re-run with a
  different tap armed.

Because the harness is the inference engine — tools, subagent forks,
memory files, compaction, KV serialization — the observation frame is an
agent, not a completion. That opens questions no capture-a-dataset
workflow reaches: probe trajectories across a whole tool-use episode,
feature shifts at compaction boundaries, and activation-space divergence
between forked subagents and their parent starting from an identical,
bit-exact HKV1 state. A saved KV snapshot doubles as a resumable
experimental condition.

Phase 5 gives live probes an operational role: each redeployed generation
is a new numerical world, its probes are retrained against the new
checkpoint (mandatory per provenance anyway), and the serving harness
ships every generation with its own monitoring. Human feedback and the
probe stream are complementary signals feeding the same loop — one records
what humans noticed, the other what the model's internals were doing at
the time.

Constraints on the CPU backend are latency, not memory:

- **Live ladder**: sub-1B runs fully instrumented at interactive speed on
  the 9950X; 2B is good; 4B is usable for experiments but too slow for
  daily serving. Larger variants are offline batch capture through the
  Python tier.
- **Tap-cost tiering**: a probe dot product per token is negligible and
  can stay always-on; full residual capture is I/O-bound and is armed
  per episode or campaign, not always-on.
- **Concurrency**: instrumented CPU inference touches no GPU, so a live
  interpretability session runs concurrently with a GPU training job
  without violating the one-GPU-consumer rule.

## The CUDA backend

A CUDA backend is the headline Tier A item in
`docs/optimizations/ideation/rust-harness-optimization-candidates.md`. The design constraints
below are stated now, before it exists, so they shape the implementation.

**Tap-seam parity.** The GPU backend implements the same `Tap` interface
with the same semantics, per the
[constraint on future backends](rust-harness-interp.md#constraint-on-future-backends)
already recorded in the bridge document. Mechanically, tap points become
explicit device-to-host sync boundaries: after the tapped layer's residual
is computed, copy it back — or run the probe dot product on-device and
copy back one scalar. Taps stay optional so the untapped path pays
nothing.

**Run-to-run determinism.** The CUDA backend must be run-to-run
deterministic on fixed hardware: fixed reduction orders, no atomics in
accumulation, deterministic kernels. It will never be bit-identical to the
CPU path, and rr does not reach GPU execution — the CPU backend keeps the
reproducibility crown.

**A third numerical world.** GPU-rust capture is distinct from both
CPU-rust and Python/torch capture. Probes served on the GPU backend
validate against GPU-captured activations; this is one more fingerprint
row handled by exactly the existing provenance machinery, not a new
mechanism.

**Revised live ladder.** With tap parity, the live-instrumentation reach
extends: 4B fully instrumented at interactive speed (the model of record
becomes the primary live subject), 12B enters live scope (~24 GB of bf16
weights fit VRAM for inference), 31B remains quantized-only. Tap cost
becomes transfer-bound instead of compute-bound, but the tiering is
unchanged: always-on scalar probes, on-demand heavy capture. Phase 5
probes-as-generation-monitoring becomes practical for the deployed 4B,
which is where the pattern matters.

**Two tiers, both kept.** The conclusion is a deliberate two-tier
architecture, not a replacement:

- **CPU backend: the reference and the microscope.** Bit-exact numerical
  world, rr replay, goldens, maximum-rigor capture, and the tier where an
  anomaly found live gets taken for full-fidelity dissection. It also
  keeps the concurrency niche: it runs alongside a GPU training job.
- **CUDA backend: the deployed, instrumented eye.** Fast live serving with
  probes armed. As a GPU consumer it serializes with training under the
  one-GPU rule.

## Net effect on the interpretability roadmap

The program's center of gravity becomes a two-pole one:

- **4B as the model of record**: live probes, steering, and capture on the
  deployed model — CPU-instrumented for experiments now, GPU-instrumented
  at speed once the CUDA backend lands.
- **Sub-1B from-scratch as the science pole**: full visibility, known
  training history, end-to-end reproducibility.
- **12B and 31B as transfer checks**, not primary subjects.

One new requirement surfaces: **checkpoint-series capture**. The capture
and manifest machinery currently assumes one model hash per campaign; a
feature-formation study wants an ordered family of hashes — one per
training checkpoint — treated as a unit. This is a small manifest-schema
consideration, recorded here so it is settled before the phase 4 exporter
is designed.

## Sequencing

1. Everything above sits behind the existing verification gates: the
   interpretability verification pass produces real-model baselines first;
   no new work before those numbers exist.
2. CUDA-backend interpretability constraints apply at backend design time,
   not as separate tasks.
3. Items graduate from this reference into roadmap tasks individually,
   each with its own acceptance check, per the repository's
   pin-then-fix-then-update discipline.
