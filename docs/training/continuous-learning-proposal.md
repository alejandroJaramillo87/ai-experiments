# Continuous learning: the parked goal of the training tier

**Parked proposal — recorded 2026-09-18. Nothing here is scheduled; this document is
edited only as its gating conditions are met or an item graduates out.**

The long-term goal of this repository's training tier is a model that learns over time:
weights that move as the model reads, at more than one rate, rather than static weights
updated by a separate training run. This document records that goal, the architecture
that is the current candidate, the order of work that precedes it, and the design
problems any implementation has to clear first, so that none of it is derived a second
time. The rest of the tier is documented in [README.md](README.md).

## Table of Contents

<!-- toc:start -->

1. [The goal, stated against what this lab can do](#the-goal-stated-against-what-this-lab-can-do)
2. [The progression that precedes it](#the-progression-that-precedes-it)
3. [The current candidate](#the-current-candidate)
4. [Design problems to clear first](#design-problems-to-clear-first)
5. [What is ruled out, and why](#what-is-ruled-out-and-why)
6. [Graduation condition](#graduation-condition)

<!-- toc:end -->

## The goal, stated against what this lab can do

Every training rung in this repository produces an artifact with a known change, which
the rust harness serves and the interpretability tier recovers: a fine-tune with an
implanted behaviour, a continued pretraining with a planted fact, a scratch model whose
whole history is on disk. The tier is an instrument for testing claims about where and
how knowledge enters a network, not a factory for better models.

A model that learns continuously is a claim of exactly that kind, about where knowledge
lives and how fast it moves. The contribution a one-GPU lab can make is not the
architecture that wins but the instrument that can tell whether an architecture's memory
claim is true. That is the frame this goal is held in: build the model small enough to
train from nothing, plant what it should learn and when, and check with the probes,
captures and organisms the earlier rungs verified.

## The progression that precedes it

Three rungs, in this order, each leaving behind a verified organism the later work
reuses:

1. **Supervised fine-tuning.** Exists; verified by
   [../verification/training/fine-tuning-loop.md](../verification/training/fine-tuning-loop.md).
   Proves the probe, steering and identity chain on a real implant.
2. **Continual pretraining.** Code exists (the Phase 2 data pipeline, mixtures, the
   warmup-stable-decay schedule, the replay buffer, optimizer offload; design records
   under [investigations/](investigations/)). Its runbook is written when its first entry
   can be. Proves the lab can measure what a corpus did to a model it did not train.
3. **Training from scratch.** Code exists (the D6 reference loop on `pretrain-150m` and
   `pretrain-1b`, calibrated by the archived P0.3). Its runbook follows the second. The
   only model whose entire history the lab holds, and the substrate the candidate below
   would run on, since it is small and trained from nothing.

The goal below waits on all three. It is not a fourth rung on the same ladder: it is a
different architecture, and the earlier rungs are what make its claims testable.

## The current candidate

Nested Learning (Behrouz, Razaviyayn, Zhong and Mirrokni, Google Research, NeurIPS 2025)
reframes a model as nested optimization problems, each with its own memory and its own
update frequency. Three parts matter here:

- A "deep optimizer" reading in which optimizer state such as momentum is itself an
  associative memory.
- A continuum memory system: a chain of blocks each updated at a different rate, so
  short-term and long-term knowledge live at different levels.
- HOPE, a self-modifying variant of Titans that combines the continuum memory with a
  sequence model that updates its own fast weights at test time.

Continual learning in that paper means the fast levels update during inference on the
stream the model reads, while slow levels change rarely. The reported models are small
(roughly 340M to 1.3B parameters); the results are language modelling and long-context
recall. It was a research release: there is no pinned implementation to build on, which
under the repository's pin-then-fix-then-update rule means the implementation would be
this lab's own, written from the paper. Check every claim above against the source before
relying on it; the sandbox that recorded this document could not fetch the paper.

What makes it a fit rather than a distraction: a model whose levels update at known
frequencies is an organism by construction. The lab knows which block should hold a fact
planted at time t, and the probes and captures can check.

## Design problems to clear first

Each is a real design question, not a task, and none is scheduled.

1. **Artifact identity.** Every consumer in this repository assumes a served artifact is a
   fixed file with a sha256 of `model.safetensors`: the harness preflight, the probe
   sidecar's `model_hash`, the control-vector manifest, the capture manifest, the
   served-request log. A model that rewrites fast weights while answering has no stable
   hash across requests. The likely answer is that the artifact is the slow weights and the
   identity check covers those alone, but that has to be decided and threaded through
   `rust-harness/harness/src/preflight.rs` and the schemas under `schemas/`.
2. **What a capture means.** `interp/probes/capture.py` and the harness's
   `activation-capture` assume the activations for a prompt depend on the prompt and the
   weights alone. With fast weights that depend on request history, a capture also depends
   on what the model read before, and the manifest would have to record it.
3. **A new architecture, not a new training method.** The harness's config reader, the
   family registry in `interp/families.py`, the export path and the calculator's parameter
   accounting all assume a Hugging Face Gemma 4 checkpoint. The candidate has none of
   that; it needs its own model code, a training loop for multi-frequency updates, a
   serving path that performs test-time updates, and an interp family. That is a fourth
   tier's worth of work, and the reason the three rungs come first.
4. **Nothing to pin.** The repository's operating rule is pin, then fix, then update, and
   there is no upstream to pin. The implementation would be written from the paper and
   validated against its reported numbers on the small models, which is a research
   sprint with a postmortem, not a roadmap task.

The three rungs should not deepen the assumptions in problems 1 and 2 in ways that are
hard to undo. Concretely: keep the identity check a function of "the weights being
served" rather than "the file", and keep the capture format indifferent to whether the
model changed between requests. Neither costs anything today.

## What is ruled out, and why

- **Distillation.** Removed from the tree on 2026-09-18. Its only distinct capability was
  a logit-level KL trainer whose output is a model that is better by scorecard, not a
  model with a known change; no interp consumer treated a distilled student differently
  from any checkpoint. The design record,
  [investigations/f5_distillation_investigation.md](investigations/f5_distillation_investigation.md),
  is frozen and stays. `make generate-teacher` stays with it, because the organism
  generator draws ground truth through it.
- **Human-feedback training** (RLHF, DPO and relatives). Answers "how do I make a model
  people prefer", which is not the question this repository asks, and brings a reward
  model, a preference pipeline and a second training loop, each needing its own runbook,
  observability and ledger rows. The one honest counterargument, that refusal and
  sycophancy arise from preference training and are the behaviours people most want to
  reverse engineer, is answered by the organism pipeline: a refusal-shaped behaviour can
  be implanted with supervised fine-tuning on a taboo-style corpus, which is what it
  already does.

The test both fail is the one every rung must pass: state what the interpretability tier
is supposed to find afterwards. Neither can.

## Graduation condition

This document is revisited when the third rung, training from scratch, is verified with
an organism of its own. At that point the first item to schedule is problem 1 above, as a
design investigation under [investigations/](investigations/), before any model code.
