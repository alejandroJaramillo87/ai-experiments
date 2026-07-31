# Training scale reference

A reference for which training modes fit the workstation (RTX 5090 with
32 GB VRAM, AMD Ryzen 9950X, 128 GB DDR5-6000), mapped to the Gemma 4 dense
model family and to the from-scratch (phase 4) and continual-learning
(phase 5) roadmap phases. Nothing in this document is scheduled work: it
records feasibility conclusions and design constraints so later roadmap
tasks start from settled arithmetic instead of re-deriving it. Items marked
aspirational are exactly that. Related documents:
[python-pipeline-optimization-candidates.md](../optimizations/python-pipeline-optimization-candidates.md)
(the FP8 Tier A item this document expands),
[memory_fit.md](memory_fit.md) (the memory-fit calculator these budgets
feed), and
[interp-scale-reference.md](../interpretability/interp-scale-reference.md)
(how these conclusions shape the interpretability program).

## Table of contents

- [FP8 on Blackwell](#fp8-on-blackwell)
- [Memory arithmetic](#memory-arithmetic)
- [The Gemma 4 ladder](#the-gemma-4-ladder)
- [Phase 4: from-scratch training](#phase-4-from-scratch-training)
- [Phase 5: continual learning](#phase-5-continual-learning)
- [Titans-class architectures](#titans-class-architectures)
- [Sequencing](#sequencing)

## FP8 on Blackwell

The RTX 5090 executes FP8 matmuls natively. The payoff of FP8 training
scales with how matmul-dominated the workload is: continued pretraining and
full-parameter training make every forward and backward matmul eligible and
approach the 1.5-2x step-time win over bf16, while LoRA-scale fine-tunes
shrink the win because the trainable matmuls are a small fraction of the
step. FP8 therefore belongs to the continued-pretraining and full-training
modes, not the fine-tuning path.

The standard recipe keeps master weights in bf16, layer norms and
embeddings in higher precision, and applies delayed or per-block scaling to
contain outlier activations (the Transformer Engine recipe, or torchao
float8, which composes with `torch.compile`). Stability divides by mode:
from-scratch FP8 pretraining is where the known instabilities live;
continued pretraining from a healthy bf16 checkpoint starts in a tame loss
landscape and is the FP8-friendly case. That combination — continued
pretraining on domain data, in FP8 — is both the highest-value training
mode planned for this machine and the numerically easiest.

Provenance: the training artifact is a checkpoint, not intermediate
numerics, so FP8 inside the training loop forces nothing downstream. Any
activation capture against the resulting checkpoint is a new numerical
world regardless of training precision, so FP8 adds no provenance cost
beyond what training itself already incurs.

## Memory arithmetic

Full-parameter training cost per parameter, with FP8 compute and the
standard state layout (bf16 master weights 2 bytes, gradients 2 bytes, FP8
compute copy 1 byte, Adam moments):

| Configuration | Bytes per parameter | States location |
|---------------|--------------------:|-----------------|
| fp32 Adam, pure GPU | ~13 | VRAM |
| 8-bit Adam, pure GPU | ~7 | VRAM |
| fp32 Adam, offloaded | ~12 host + ~5 GPU | host RAM |
| 8-bit Adam, offloaded | ~6 host + ~5 GPU | host RAM |

Ceilings that follow, assuming gradient checkpointing and a few GB reserved
for activations and CUDA workspace:

- **Pure GPU, fp32 Adam**: ~2B parameters (26 GB of states in 32 GB VRAM).
- **Pure GPU, 8-bit Adam**: 3B comfortable, 4B the realistic ceiling
  (~28 GB of states). The optimizer, not FP8, is the lever that moves this
  ceiling — FP8 shrinks compute cost, not the states that dominate.
- **Offloaded (ZeRO-Offload pattern)**: master weights and Adam states live
  in host RAM and the optimizer step runs on the 9950X. Against ~100 GB of
  usable RAM: ~8B parameters comfortable, ~13-14B the practical maximum
  with 8-bit states and chunked gradient streaming. The GPU-side residency
  (FP8 weights, current gradients, activations) is the second ceiling and
  is what forces the chunking above ~8B.

Offload buys capacity, not speed. Every optimizer step round-trips
gradients out and updated weights back over PCIe 5.0 x16 (~60 GB/s each
way nominal, less in practice), and the CPU-side Adam step over billions
of parameters is itself nontrivial. Expect step times 2-3x the resident
case even with transfer-compute overlap; the FP8 compute win largely
disappears behind transfer time. Offloaded campaigns are planned in days.

These budgets are inputs to the memory-fit calculator
([memory_fit.md](memory_fit.md)); the per-model-size gradient-checkpointing
and paged-optimizer choices recorded there apply on top.

## The Gemma 4 ladder

The Gemma 4 dense variants are 2B, 4B, 12B, and 31B. The 26B
mixture-of-experts variant is excluded from the program: expert routing
touches the rust runtime's forward pass, GGUF tensor layout, tap seam, and
KV-cache determinism story simultaneously, and none of that cost serves
the research questions the substrate exists for.

| Variant | Continued pretraining | Full fine-tune | Preference tuning | Notes |
|---------|----------------------|----------------|-------------------|-------|
| 2B | pure GPU | pure GPU | pure GPU, full | fits with fp32 Adam |
| 4B | pure GPU | pure GPU | pure GPU, tight | requires 8-bit Adam; at the resident ceiling |
| 12B | offload, slow | offload, slow | offload or PEFT | days-scale campaigns; PEFT for fast iteration |
| 31B | no | no | PEFT only | full-parameter states ~370 GB exceed host RAM |
| 26B MoE | excluded | excluded | excluded | runtime cost, see above |

Reading the table:

- **2B and 4B** support every training mode fully resident on the GPU.
  The 4B sits exactly at the 8-bit-Adam ceiling, which is convenient: the
  4B class is the model family the rust harness targets, so the pure-GPU
  sweet spot and the model of record coincide.
- **12B** is capacity-feasible for full-parameter work only via offload,
  with the 2-3x step-time tax. Parameter-efficient fine-tuning (PEFT)
  remains fast and resident.
- **31B** is PEFT-only, and the distinction matters: full-parameter
  training is not merely slow but impossible on this machine — the state
  footprint exceeds the 128 GB of host RAM, not just VRAM. QLoRA-style
  tuning (4-bit base weights ~16 GB resident plus adapter states) fits in
  VRAM; bf16-base LoRA fits with offload.

One boundary condition applies to every row: a single 5090 sustains on the
order of a few billion training tokens per week at small-model scale.
Continued pretraining on this machine is domain adaptation at the
few-billion-token scale, not a base-model rebuild.

## Phase 4: from-scratch training

For from-scratch pretraining the binding constraint is token throughput,
not memory. Compute-optimal training of even a 2B model wants tens of
billions of tokens; at single-GPU throughput the realistic phase 4 scope is
**sub-1B research models**. That scope is not a consolation prize — sub-1B
is the scale where full-visibility interpretability studies live and where
the substrate's capture seams pay off most.

**Design constraint: from-scratch models use the Gemma 4 block at reduced
scale.** No Gemma 4 variant is sub-1B, but from-scratch training chooses
its own architecture, and choosing the Gemma block (attention layout, norm
placement, activation, local/global attention pattern) at smaller
dimensions makes phase 4 models first-class citizens of the entire
substrate — runtime, tap seam, probe path, KV snapshots, goldens — on day
one. This is verified, not assumed: `Config::from_gguf`
(`rust-harness/runtime/src/model/cpu.rs:56`) gates only on the architecture
string starting with `gemma` and reads every dimension (layers, heads,
embedding width, FFN width, sliding-window pattern, RoPE base) from GGUF
metadata. A scaled-down Gemma exported with the expected metadata keys
loads with zero runtime changes.

Two consequences for the training roadmap:

- **Checkpoint-to-GGUF exporter.** Phase 4 requires a Python-side exporter
  from training checkpoints to well-formed GGUF with the runtime's
  expected metadata keys. This is a training-tier artifact, not runtime
  work.
- **Tokenizer.** The cheapest and safest choice is reusing the Gemma
  tokenizer verbatim, which keeps the runtime's SentencePiece path and the
  provenance fingerprints untouched. Note the tension with
  [tokenizer_training.md](tokenizer_training.md), which plans byte-level
  BPE tokenizers for the from-scratch path: a custom tokenizer is a
  runtime change (new decode path) and a provenance change, and the choice
  between the two is deliberately left open here.

## Phase 5: continual learning

Phase 5 is defined as **orchestrated generations**, not literal
always-on learning: deploy a model, generate trajectories and capture
activations through the harness, collect human feedback, train on it
(offline preference tuning, periodic continued pretraining), export to
GGUF, redeploy. Every arrow in that loop is built or planned — the harness
is the trajectory generator and capture substrate, the training tier does
the weight updates, the phase 4 exporter closes the loop — and the
provenance machinery (model hashes, tokenizer fingerprints, re-capture
discipline) is precisely what makes repeated model turnover safe. Discrete,
versioned generations are a feature for a research substrate: every
generation is a reproducible artifact.

Every Gemma 4 variant can enter phase 5, with two recorded constraints:

- **Reference-model overhead.** Preference methods in the DPO family need
  a frozen reference model alongside the policy, roughly doubling resident
  weight memory. This pushes each variant one rung down the ladder above.
  Standard mitigations recover the headroom: precomputing reference
  log-probabilities offline — a natural fit here, since the rust harness
  can serve as the log-probability generator — or adapter-based training
  where the frozen base model is the reference. With those, 2B and 4B
  remain pure-GPU, 12B is offload-plus-mitigations, and 31B uses
  adapter-based preference tuning, which is standard practice at that
  scale.
- **Offline over online.** Online reinforcement-learning methods (PPO,
  GRPO) interleave generation and training on the same GPU, colliding with
  the repository's one-GPU-consumer rule; running them as a single
  orchestrated alternating job is possible but halves effective throughput.
  The human-in-the-loop pattern — collect feedback, then train on it — is
  the offline family's natural shape, so **offline preference methods are
  the deliberate phase 5 default**, recorded here as a choice rather than
  a discovery.

## Titans-class architectures

Test-time-learning architectures (Titans and the family descending from
it) place a neural memory module inside the model that updates its own
parameters during inference. Honest verdict: **not feasible with what is
built; plausibly feasible with what the substrate is for.**

What breaks: the rust runtime's load-bearing assumption is frozen weights —
load GGUF once, walk read-only buffers, KV cache as the only mutable
state. A test-time-learning memory makes parameters mutable in the forward
pass, which touches the model struct, the HKV1 snapshot format (memory
state must serialize alongside KV), fork/join semantics (whether forked
subagents share or copy learned memory), the flight recorder, and every
golden test. There is no GGUF convention and no reference implementation to
match; supporting Titans means implementing a research architecture from a
paper and defining its serialization. The effort is comparable to the
original CPU backend, on top of a from-scratch training pipeline for an
unproven architecture.

The counterweight: a bit-reproducible test-time-learning implementation is
exactly what mainstream frameworks cannot provide and this runtime could,
and sub-1B — where Titans-style experiments live in the literature — is
phase 4's scope. Titans is therefore recorded as a **research option inside
phase 4's sub-1B from-scratch scope**: aspirational, sequenced strictly
after a vanilla scaled-Gemma from-scratch run proves the exporter and
training loop, and justified only when a research question demands
test-time memory.

## Sequencing

1. All of the above sits behind the existing verification gates: real
   bf16 training baselines with recorded wall-clocks come first; no
   FP8 or scale work before those numbers exist.
2. FP8 lands as one measured change: step time, a loss-curve overlay
   against the bf16 baseline, and a downstream evaluation confirming
   quality held.
3. Items graduate from this reference into roadmap tasks individually,
   each with its own acceptance check, per the repository's
   pin-then-fix-then-update discipline.
