# Local LLM training scope

This document defines the scope for adding local model-training infrastructure to
this repository, alongside the existing inference services. Training and inference
share one repository, one workstation, and one set of conventions: a model trained
here is served by the existing llama.cpp and vLLM services in this repository,
closing a single train-and-serve loop. The document records the training stages
the repository implements, what it adopts versus builds, and a phased roadmap.

This is a planning document. No training code is implied to exist yet.

## Table of Contents

- [Purpose](#purpose)
- [Goals](#goals)
- [Hardware envelope](#hardware-envelope)
- [Training stages and methods](#training-stages-and-methods)
  - [Pretraining from scratch](#pretraining-from-scratch)
  - [Continual pre-training](#continual-pre-training)
  - [Fine-tuning](#fine-tuning)
  - [Continual learning](#continual-learning)
- [Design principle: adopt versus build](#design-principle-adopt-versus-build)
  - [Adopt](#adopt)
  - [Build](#build)
- [Learning track](#learning-track)
- [Continual and lifelong learning](#continual-and-lifelong-learning)
  - [Why it fits a local sub-6B lab](#why-it-fits-a-local-sub-6b-lab)
  - [Taxonomy the repository organizes around](#taxonomy-the-repository-organizes-around)
  - [State of the field as of mid-2026](#state-of-the-field-as-of-mid-2026)
  - [What is shared, built, and adopted](#what-is-shared-built-and-adopted)
- [Roadmap](#roadmap)
  - [Phase 0: Foundation](#phase-0-foundation)
  - [Phase 1: Fine-tuning](#phase-1-fine-tuning)
  - [Phase 2: Continual pre-training](#phase-2-continual-pre-training)
  - [Phase 3: Pretraining from scratch](#phase-3-pretraining-from-scratch)
  - [Phase 4: Continual learning](#phase-4-continual-learning)
  - [Optional and ongoing tracks](#optional-and-ongoing-tracks)
- [Proposed repository layout](#proposed-repository-layout)
- [References](#references)

## Purpose

Provide a reproducible local environment to train and adapt language models of 6
billion parameters and smaller on this single-node workstation, and to serve them
through the inference services already in this repository. The training additions
target three outcomes at once: producing models that are useful, teaching the
mathematics and mechanics of deep learning, and saturating the full machine rather
than the GPU alone.

A single RTX 5090 cannot pretrain a state-of-the-art model from scratch. The
reference small model of this era, SmolLM3-3B, consumed approximately 9,200
GPU-days on 11.2 trillion tokens; reproducing that on one card is on the order of
50 GPU-years. The repository therefore separates two distinct paths and serves
both:

1. **Pretraining from scratch.** Primarily serves the learning pillar and the goal
   of owning the full stack. The realistic target is a coherent and servable
   model, not a state-of-the-art one.
2. **Specializing a strong open base.** Continual pre-training and fine-tuning of
   an existing model such as SmolLM3, Qwen, or Llama-3.2. This is the path that
   yields genuinely useful local models within days rather than weeks, and it is
   where continual learning lives.

## Goals

The project has three co-equal pillars. Every design decision serves at least one
of them.

1. **Produce useful models.** Specialize strong open bases and, where practical,
   train small models from scratch, up to 6 billion parameters through
   parameter-efficient methods. The output is a coherent model, not a
   demonstration.
2. **Teach deep learning internals.** Serve as a working environment to understand
   backpropagation, the matrix algebra of the forward and backward pass, gradient
   flow, optimization dynamics, tokenization, and embeddings. Annotated
   from-scratch implementations and visualizations are first-class deliverables.
3. **Use the whole machine.** Place each workload on the correct hardware tier and
   overlap tiers so none sits idle: GPU for the training step, CPU threads for the
   data pipeline and evaluation, system RAM and NVMe for optimizer and activation
   offload.

## Hardware envelope

| Component | Specification | Primary training role |
|-----------|---------------|-----------------------|
| GPU | RTX 5090, 32GB VRAM, Blackwell | Forward and backward pass, matrix multiplication, attention and state-space kernels |
| CPU | AMD Ryzen 9950X, 16 cores, 32 threads | Data pipeline, tokenization, deduplication, concurrent evaluation, checkpoint serialization |
| RAM | 128GB DDR5-6000 | Optimizer-state offload, dataset cache, memory-mapped token shards |
| Storage | NVMe at `/mnt/ai-data` | Activation and parameter spill, checkpoint store, dataset shards |

The binding constraint is 32GB of VRAM. The design is memory-first: fit the
largest useful model and longest useful context into 32GB, then use system RAM and
NVMe to extend that ceiling through offload.

## Training stages and methods

The repository implements four training stages. Each is a code path built on an
adopted trainer, and each applies to both a strong open base and a from-scratch
model. Preference alignment, for example direct preference optimization and
reinforcement learning from human feedback, is out of scope.

| Stage | Objective | Primary adopted tool | Scale on 32GB |
|-------|-----------|----------------------|---------------|
| Pretraining from scratch | Next-token prediction | nanotron, litGPT, custom loop | 100M to 1B full |
| Continual pre-training | Next-token prediction on a corpus stream | TRL, Axolotl | 1B to 3B full with offload, larger with LoRA |
| Fine-tuning | Instruction, chat, and distillation | TRL, torchtune, Unsloth | 1.5B to 2B full, 6B to 8B with QLoRA |
| Continual learning | Retain prior stages while learning new ones | The stages above plus EasyEdit | Inherits the wrapped stage |

### Pretraining from scratch

Train a model from random initialization with a causal language-modeling
objective. This stage owns the tokenizer, the architecture, and the full data
pipeline. Kernels come from FlashAttention and Liger-Kernel. The realistic local
target is 100 million to 1 billion parameters. The primary value is the learning
pillar and control of the entire stack rather than model quality.

### Continual pre-training

Resume training on a strong open base over new corpora, most often for domain
adaptation. The objective is unchanged, but two techniques govern quality:
learning-rate rewarming and re-decay, typically through a warmup-stable-decay
schedule, and replay of a slice of the original data distribution to limit
forgetting. Data-mixing weights control the balance of new and original domains.
Full training reaches 1 billion to 3 billion parameters with optimizer offload to
system RAM, or larger through low-rank adaptation. Run as a sequence of such
adaptations rather than once, this becomes the continual-pre-training stage of
continual learning.

### Fine-tuning

Fine-tuning adapts a base model to instruction and chat data, called supervised
fine-tuning (SFT), or to a larger teacher's behavior, called distillation. The
stage handles chat templates, multi-turn conversations, prompt-loss masking so
that loss applies only to completions, and sequence packing. Full fine-tuning
reaches roughly 1.5 billion to 2 billion parameters. Parameter-efficient
fine-tuning (PEFT), namely low-rank adaptation (LoRA), its quantized form (QLoRA),
and weight-decomposed adaptation (DoRA), adopted from Hugging Face PEFT and
bitsandbytes, reaches 6 billion to 8 billion. Adapter merging through mergekit
combines separately trained adapters, and the same adapter-isolation mechanism
underpins the architecture family of continual-learning methods. Distillation, in
which a small model learns from a larger teacher's responses or output
distribution, is the most direct route to a capable sub-3-billion-parameter model
on this hardware.

### Continual learning

Continual learning runs the stages above as a sequence over a stream of new
knowledge and tasks, rather than once, while retaining prior capability. It adds no
new trainer, only an outer loop and anti-forgetting machinery around the existing
paths. Because it is both a research-grade pillar and the subject of active 2026
work, it has its own detailed treatment in
[Continual and lifelong learning](#continual-and-lifelong-learning).

## Design principle: adopt versus build

Roughly 80 percent of the training additions are integration, configuration, and
orchestration. Roughly 20 percent is code specific to this hardware envelope and
the combined train-and-serve lifecycle. The test for every component is a single
question: does it depend on the specific envelope of 32GB Blackwell, 128GB RAM,
local single node, and 6 billion parameters and under, or on the combined
train-and-serve lifecycle in this repository, or is it the same problem faced by
anyone with a GPU. Shared problems are adopted. Specific problems are built.

Writing a custom trainer, dataloader, deduplication primitive, or optimizer is
treated as an anti-pattern. Each is a maintenance treadmill against well-funded
upstream projects.

### Adopt

| Area | Tools |
|------|-------|
| Kernels and precision | FlashAttention, Liger-Kernel, Unsloth, mamba-ssm, causal-conv1d, flash-linear-attention, Transformer Engine for FP8 |
| Trainers | TRL, Axolotl, torchtune, Unsloth, Llama-Factory |
| Memory and offload engine | DeepSpeed ZeRO-Offload and ZeRO-Infinity, FSDP2, bitsandbytes |
| Data pipeline primitives | Hugging Face datatrove, AI2 dolma, NVIDIA NeMo-Curator, text-dedup |
| Synthetic data generation | distilabel, driven by the in-repo inference services |
| Tokenizer training | Hugging Face tokenizers, SentencePiece |
| Model architectures | Hugging Face transformers, litGPT, official Mamba and RWKV repositories, titans-pytorch and open HOPE reproductions |
| Pretraining at scale | nanotron, torchtitan, OLMo-core |
| Continual learning | EasyEdit for knowledge editing, Hugging Face PEFT for adapter isolation |
| Evaluation | lm-evaluation-harness, lighteval, MTEB |
| Experiment tracking, configuration, search | Weights and Biases, MLflow, TensorBoard, Hydra, Optuna, Ray Tune |
| Quantization, export, merge | llama.cpp for GGUF, AutoAWQ, GPTQModel, mergekit |

### Build

The differentiated core, in priority order. Each item depends on the specific
hardware envelope or the train-and-serve lifecycle and cannot be downloaded.

1. **Memory-fit solver and recipe library.** A calculator that maps a model
   configuration, optimizer, checkpointing strategy, and offload setting to peak
   VRAM, maximum sequence length, and maximum batch size on 32GB, paired with a
   curated set of validated recipes tuned to this exact machine. No existing tool
   answers this for a single prosumer node.
2. **Heterogeneous-compute orchestration.** The whole-machine choreography that
   runs the CPU data pipeline, the GPU training step, concurrent CPU-side
   evaluation, and inference-driven synthetic-data generation at the same time,
   with NUMA and core pinning for the 9950X and RAM-disk staging of hot shards.
3. **Blackwell FP8 training enablement and benchmarks.** Making the FP8 training
   stack run on the sm_120 architecture and measuring the result in the
   reproducible JSON format already used for inference benchmarks. This is
   enablement and measurement, not kernel authorship.
4. **Train-and-serve integration.** Export a checkpoint, merge any adapter,
   quantize to a servable format, serve it on the existing inference services in
   this repository, and evaluate by serving. One loop within one repository.
5. **From-scratch educational implementations.** Annotated transformer and Mamba
   implementations built for understanding, described in the learning track. These
   serve the learning pillar rather than production.
6. **Continual-learning regime and test-time-memory architectures.** The outer
   loop, replay buffer, anti-forgetting hooks, retention evaluation, and the
   architectural memory models that let a model keep learning after pretraining,
   detailed in [Continual and lifelong learning](#continual-and-lifelong-learning).
   The custom footprint is small and reuses the shared trainer, data pipeline, and
   offload.

## Learning track

The learning pillar is a build item because no external project is organized
around this goal on this machine. The track builds understanding from first
principles upward and connects each concept to a runnable artifact.

- **Autograd from scratch.** A minimal reverse-mode automatic differentiation
  engine over scalars, then over tensors, to make backpropagation concrete before
  any framework abstraction is introduced.
- **Matrix algebra of the forward and backward pass.** Written derivations of the
  shapes and gradients for each layer of a transformer block: embedding lookup,
  attention projections, softmax, the feed-forward network, and the language-model
  head. Each derivation pairs with code that checks the manual gradient against the
  framework gradient.
- **Single-file annotated transformer.** A readable decoder-only model with every
  tensor operation commented, tied to the derivations above.
- **Single-file annotated state-space model.** A readable Mamba-style model that
  contrasts the recurrence and scan of a state-space model with the attention of a
  transformer.
- **Training-dynamics visualizations.** Notebooks that plot gradient norms,
  activation statistics, dead-neuron counts, learning-rate schedules, loss curves,
  and the evolution of the embedding space during a real run.
- **Ablation harness.** A configuration surface to toggle one architectural or
  optimization choice at a time and observe the effect on loss and stability.

The learning track uses the same data pipeline, tracking, and hardware as the
production track, so lessons transfer directly to real runs.

## Continual and lifelong learning

Continual learning is the paradigm in which a model adapts to a stream of new
knowledge and tasks over time while retaining prior capability, rather than being
trained once and frozen. It is distinct from a single continual pre-training run:
it is a training regime that spans pretraining and fine-tuning, and its defining
problem is catastrophic forgetting, the loss of earlier capability when new
gradients overwrite the weights that held it.

The repository treats continual learning as a first-class regime rather than a
separate model type. It is not a parallel training stack. It is the existing
trainer, data pipeline, checkpointing, and offload, wrapped in an outer sequential
loop with the anti-forgetting and retention-measurement machinery that every stage
shares.

### Why it fits a local sub-6B lab

Continual learning research is bounded by method design and iteration speed, not by
scale. A single node that runs a full task sequence on a 1 billion to 3 billion
parameter model in hours is better suited to this work than a cluster optimized for
one large run. Continual fine-tuning at this scale completes in days on the RTX
5090. The hardware shape is an asset: replay buffers and external memory stores
reside in the 128GB of system RAM while the GPU trains. This is a research
direction where a local lab can produce original results that larger organizations
are not optimizing for.

### Taxonomy the repository organizes around

The repository follows the taxonomy of the mid-2026 survey by Chen and colleagues,
which structures continual learning across training stages, each addressed by three
method families. The stages are exactly those defined in
[Training stages and methods](#training-stages-and-methods), run as a sequence
rather than once. Continual learning adds no new trainer; it adds the outer loop
and the anti-forgetting machinery around those existing paths. The survey also
defines a continual-alignment stage, which is out of scope here because alignment
itself is out of scope.

Stages implemented:

- **Continual pre-training.** A stream of corpora, adapted with replay and
  learning-rate rewarming on the pretraining code path.
- **Continual fine-tuning.** A sequence of tasks or instruction sets on the
  fine-tuning code path.

Three method families:

- **Rehearsal and replay.** Mix samples of prior data into new batches. Implemented
  as a data-pipeline component with the buffer resident in system RAM.
- **Regularization.** Penalize changes to weights important to prior tasks, for
  example elastic weight consolidation and learning-without-forgetting. Implemented
  as hooks into the loss and optimizer.
- **Architecture.** Isolate new knowledge in new parameters, for example adapter
  and low-rank isolation, orthogonal low-rank adaptation, and mixture-of-experts
  routing. Implemented mostly through adopted parameter-efficient libraries with
  thin routing glue.

A fourth mechanism, targeted knowledge editing such as MEMIT and WISE, does not use
the training loop at all. It performs closed-form or optimization-based weight
surgery on specific layers and is adopted as a bolt-on library rather than built.

### State of the field as of mid-2026

Continual learning is an active frontier and is not solved. Three threads are
relevant to this repository.

- **Architectural test-time memory.** Google's Nested Learning paradigm and its
  HOPE architecture, published in late 2025, reframe a model as a set of nested
  optimization problems that update at different timescales. HOPE couples a
  self-modifying core, derived from the earlier Titans test-time memory work, with
  a continuum memory system of components that learn at different rates. It moves
  continual learning from a training-time addition to a property of the
  architecture itself. Several open reproductions already exist at small scale,
  including nano-scale educational implementations of a few tens of millions of
  parameters, which makes this directly reproducible on the RTX 5090 and a natural
  addition to the model registry alongside Mamba.
- **Improved replay and time-continual benchmarks.** Surprise-driven prioritized
  replay, multi-timescale memory dynamics, and web-scale time-continual pretraining
  benchmarks such as TiC-LM sharpen the parametric approach.
- **Agent and on-device memory.** Just-in-time reinforcement learning that adjusts
  logits in closed form without gradient updates, and on-device continual learning
  through budget-curated memory, target exactly the local and private setting this
  repository occupies.

The honest caveat: self-modifying and self-editing models still degrade on prior
tasks as updates accumulate, official weights for HOPE are not released, and the
strongest reproductions are unofficial. This is a research pillar, not a solved
integration, which is why it sits at the end of the roadmap.

### What is shared, built, and adopted

| Component | Relationship to the repository | Verdict |
|-----------|-------------------------------|---------|
| Model, trainer, optimizer, checkpointing | Identical to the rest of the repository | Reuse |
| Data pipeline, tokenizer, shards | Identical | Reuse |
| Replay buffer, resident in system RAM | Data-pipeline component | Build, thin |
| Outer loop over a task or corpus sequence | Continual-learning specific | Build, thin |
| Regularization hooks, for example Fisher penalty | Hooks into loss and optimizer | Build, thin |
| Retention evaluation, forgetting and transfer metrics | Wraps the adopted evaluation harness | Build |
| Knowledge editing, MEMIT and WISE | Separate weight-surgery code path | Adopt EasyEdit |
| Test-time-memory architectures, Titans and HOPE | New entries in the model registry | Build from open reproductions |

## Roadmap

The roadmap is ordered by training type, not by calendar. Phase 0 builds the shared
foundation that every later phase depends on. Phase 1 delivers production value
fastest and proves the train-and-serve loop. Phase 2 adds domain adaptation. Phase
3 is the educational from-scratch run, deferred because it needs the heaviest data
pipeline, which Phases 1 and 2 build up. Phase 4 is the experimental
continual-learning regime, which requires the earlier stages to exist because it
runs them as a sequence.

The learning pillar is decoupled from Phase 3: the educational learning track is
small and is seeded in Phase 0, so the deep-learning-internals payoff arrives early
rather than waiting on the from-scratch run.

Each phase below lists concrete **deliverables**, each tagged **[adopt]** (external
tool pulled in unchanged), **[orchestrate]** (configuration and pipelines wiring
adopted pieces together), or **[build]** (custom code specific to this hardware
envelope and the train-and-serve lifecycle), followed by **acceptance criteria**
that define when the phase is complete. Module paths and commands describe the
intended surface, not a fixed contract.

### Phase 0: Foundation

The shared substrate every later phase depends on, plus the learning-track seed and
Blackwell FP8 enablement. FP8 is enabled here, not as a training type, so every
subsequent phase inherits the speedup. **Depends on:** nothing.

**Deliverables**

1. **[orchestrate] Training container and dependency group.** A `docker/training/`
   image with PyTorch on CUDA 13, FlashAttention, Liger-Kernel, bitsandbytes,
   DeepSpeed, FSDP2, TRL, torchtune, PEFT, and Transformer Engine, a `training`
   Poetry group in `pyproject.toml`, and a `make train-shell` target.
2. **[orchestrate] Lite data pipeline.** `scripts/train/prepare_data.py` and
   `training/data/` that ingest a Hugging Face dataset or local JSONL, tokenize
   across 32 threads, and write memory-mapped shards with a manifest of token
   counts and length statistics.
3. **[build] Tokenizer training.** `training/data/tokenizer.py` that trains and
   evaluates a byte-pair-encoding tokenizer, reporting fertility and compression
   ratio.
4. **[build] Memory-fit calculator.** `training/calculator/` exposing
   `estimate(model, optimizer, precision, checkpointing, offload)` returning peak
   VRAM, maximum sequence length, maximum batch size, and a fits flag, with a
   `make memfit CONFIG=...` entry point.
5. **[orchestrate] Recipe system.** A Hydra-based schema in `training/recipes/`
   where one YAML file fully specifies a run: model, data, optimizer, schedule,
   precision, offload, and tracking.
6. **[orchestrate] Tracking and resumable checkpointing.** Weights and Biases and
   MLflow wiring, and sharded safetensors checkpoints that save and restore model,
   optimizer, scheduler, dataloader position, and random-number state.
7. **[build] Train-and-serve round trip.** `scripts/export/` that merges adapters,
   converts to Hugging Face format, and quantizes to GGUF, plus
   `scripts/eval/serve_eval.py` that queries the model through the existing
   inference services.
8. **[build] FP8 enablement and benchmark harness.** Transformer Engine behind a
   `precision: fp8` recipe flag on the sm_120 architecture, and
   `scripts/benchmark/train_bench.py` that records a bfloat16 baseline against FP8
   for tokens per second, model FLOPs utilization, and peak VRAM as JSON under
   `docs/training/optimizations/`.
9. **[orchestrate] Heterogeneous-compute skeleton.** A pinned-memory prefetch
   dataloader, a RAM-disk shard-staging helper, and a NUMA and core-pinning wrapper
   for the 9950X.
10. **[orchestrate] Operational watchdogs.** A monitor that aborts or alerts on
    not-a-number loss, GPU temperature, and power-draw anomalies, wired to DCGM.
11. **[build] Learning-track seed.** `training/learning/` with a reverse-mode
    automatic-differentiation engine, a single-file annotated transformer carrying
    manual-backpropagation notes, and one gradient-and-activation visualization
    notebook.

**Acceptance criteria**

- The container builds and imports Torch, transformers, TRL, and PEFT with CUDA
  available.
- A recipe drives a tiny end-to-end training step and writes a checkpoint that
  resumes to the same loss trajectory.
- Memory-fit predictions land within 10 percent of measured peak VRAM on two
  reference configurations.
- A base model round-trips through export and answers a prompt through the
  llama-gpu service.
- The FP8 benchmark completes and the JSON records the bfloat16-versus-FP8
  comparison.
- The autograd engine reproduces framework gradients on a test network within
  numerical tolerance.

### Phase 1: Fine-tuning

Supervised and distillation fine-tuning of a strong open base. The smallest
increment over the foundation, the fastest path to a useful model, and the proof of
the train-and-serve loop. Production. **Depends on:** Phase 0.

**Deliverables**

1. **[orchestrate] Supervised fine-tuning entry point.** `make train-sft
   CONFIG=...` wrapping the TRL supervised fine-tuning trainer with an optional
   Unsloth backend, supporting full fine-tuning and QLoRA.
2. **[build] Chat and instruction data preparation.** `training/data/sft.py`
   providing a per-model chat-template registry, prompt-loss masking so loss
   applies only to completion tokens, multi-turn handling, and sequence packing.
3. **[adopt] Parameter-efficient options.** LoRA, QLoRA, and DoRA exposed in the
   recipe schema through Hugging Face PEFT, with 4-bit base loading through
   bitsandbytes.
4. **[build] Distillation harness.** `training/data/distill.py` and
   `scripts/train/distill.py` that capture a teacher's responses and, where
   available, logits, and apply a knowledge-distillation loss covering
   response-level and logit-KL variants.
5. **[orchestrate] Teacher and synthetic data harness.** A distilabel-based client
   that batches generation to the existing inference services to build distillation
   and synthetic datasets.
6. **[orchestrate] Adapter merge and serve.** `make export-gguf` merges the adapter
   through mergekit and quantizes, feeding the Phase 0 round trip.
7. **[build] Fine-tuning evaluation wrapper.** `scripts/eval/eval_ft.py` reporting
   held-out loss, instruction benchmarks through lm-evaluation-harness, and
   model-as-judge scoring served by the inference services, as a
   base-versus-fine-tuned scorecard.
8. **[orchestrate] Recipe cards.** Validated `recipes/sft-qlora-8b.yaml` and
   `recipes/distill-3b.yaml`.

**Acceptance criteria**

- A QLoRA fine-tune of a 3-to-8-billion-parameter base completes and logs to
  tracking.
- A unit test confirms loss applies only to completion tokens under prompt-loss
  masking.
- A distilled student measurably improves on a target evaluation over its
  pre-distillation baseline.
- The fine-tuned model is served through the inference services and returns a
  scorecard beating its base.

### Phase 2: Continual pre-training

Domain adaptation of a strong base over new corpora, with the forgetting controls
that distinguish it from a naive resume. Production. **Depends on:** Phase 0, and
reuses the Phase 1 evaluation wrapper.

**Deliverables**

1. **[orchestrate] Full-scale data pipeline.** `training/data/pipeline.py` adding
   MinHash-LSH deduplication, heuristic and quality filtering, language
   identification, and benchmark decontamination ahead of shard writing, parallel
   across 32 threads.
2. **[orchestrate] Data-mixing configuration.** A `mixture` recipe block giving
   weighted sampling proportions across sources.
3. **[orchestrate] Warmup-stable-decay schedule.** A learning-rate schedule with
   rewarming and re-decay for resuming from a base.
4. **[build] Replay buffer.** `training/continual/replay.py`, a
   system-RAM-resident buffer that mixes a configurable fraction of the original
   distribution into new batches.
5. **[orchestrate] Optimizer offload.** A DeepSpeed ZeRO-Offload configuration
   tuned to the 32GB and 128GB ratio, enabling full training at 1 billion to 3
   billion parameters.
6. **[build] Forgetting evaluation.** `scripts/eval/forgetting.py` comparing base
   and adapted models on both the new domain and original-domain benchmarks,
   reporting retention.
7. **[orchestrate] Recipe card.** `recipes/continual-pretrain-1b.yaml` specializing
   a base on a domain.

**Acceptance criteria**

- A 3-billion-parameter base runs full continual pre-training with optimizer state
  offloaded to system RAM.
- Batches contain the configured replay fraction and mixture proportions, verified
  by source counts.
- An adapted model improves its on-domain benchmark while keeping original-domain
  regression under a set threshold.

### Phase 3: Pretraining from scratch

A dense transformer trained from random initialization, the centerpiece of the
learning pillar and full ownership of the stack. Educational, because a single RTX
5090 cannot reach a state-of-the-art result from scratch. **Depends on:** Phase 0
and the Phase 2 data pipeline.

**Deliverables**

1. **[build] From-scratch model and loop.** `training/models/transformer.py` and
   `scripts/train/pretrain.py`, a configurable dense transformer with RMSNorm,
   SwiGLU, rotary position embeddings, and grouped-query attention, trained with a
   causal language-modeling objective on FlashAttention and Liger-Kernel, or a
   hardened nanotron or litGPT wrapper.
2. **[build] Architecture registry.** `training/models/` with dense-transformer and
   Mamba entries behind a common interface.
3. **[build] Scaling-law and token-budget calculator.**
   `training/calculator/scaling.py` that fits a compute-optimal token budget and
   predicts final loss for a target parameter count and wall-clock.
4. **[orchestrate] Validation run.** A 360-to-500-million-parameter model on 10
   billion to 20 billion FineWeb-Edu tokens, completing in two to four days, with
   resumable multi-week checkpointing.
5. **[orchestrate] First useful run.** A 1-to-1.5-billion-parameter model on 50
   billion to 100 billion tokens using a SmolLM2-style mixture, reusing a proven
   tokenizer, served through the round trip.
6. **[build] Learning-track tie-in.** Notebooks that visualize gradient and
   activation dynamics from the real run, using the same architecture as the
   annotated transformer.

**Acceptance criteria**

- The model overfits a single batch as a correctness sanity check before any full
  run.
- The 360-to-500-million-parameter run completes and produces coherent text at the
  target validation perplexity.
- The trained model is served through the inference services.
- The scaling calculator's predicted loss is within a set tolerance of the run's
  measured loss.

### Phase 4: Continual learning

The experimental regime that runs the earlier stages as a sequence without
forgetting, in two sub-stages: the parametric methods, then architectural test-time
memory. Experimental. **Depends on:** Phases 1, 2, and 3.

**Deliverables, sub-stage A, parametric regime**

1. **[build] Outer sequential loop.** `training/continual/loop.py` orchestrating a
   sequence of tasks or corpora, invoking the Phase 1, 2, and 3 code paths per step
   with checkpointing between steps.
2. **[build] Anti-forgetting hooks.** `training/continual/regularizers.py`
   implementing elastic weight consolidation with a Fisher penalty,
   learning-without-forgetting with a distillation penalty, and gradient
   projection, each a recipe toggle.
3. **[build] Retention metrics.** `training/continual/metrics.py` computing average
   accuracy, forgetting, backward transfer, and forward transfer across a task
   sequence.
4. **[adopt] Knowledge editing.** `scripts/train/edit.py` integrating EasyEdit for
   MEMIT and WISE, reporting edit success and locality.
5. **[orchestrate] Continual-learning benchmark suite.** Integration of a
   task-incremental benchmark such as TiC-LM.

**Deliverables, sub-stage B, architectural test-time memory**

6. **[build] Titans and HOPE in the registry.** `training/models/hope.py`
   implementing the self-modifying core and continuum memory system from open
   reproductions, with any custom kernels required for continuum-memory chunking.
7. **[orchestrate] Scale-up and benchmarking.** Training at nano scale, then toward
   1 billion to 3 billion parameters, with long-context memory and retention
   measured against a transformer baseline.
8. **[build] Self-editing harness.** `training/continual/self_edit.py` implementing
   a Self-Adapting-Language-Models-style loop in which the model generates its own
   updates, reporting the retention curve across edits.

**Acceptance criteria**

- Each anti-forgetting method runs and measurably reduces forgetting on a task
  sequence versus a no-regularizer baseline.
- A knowledge edit inserts a fact the model then recalls, without breaking
  unrelated behavior.
- A nano-scale HOPE reproduction trains and tracks a reference loss curve.
- Retention metrics are reported for both sub-stages.

### Optional and ongoing tracks

These reuse the foundation and slot in where they add value rather than occupying a
numbered phase. Several are experimental directions that became active through 2025
and 2026 and suit a memory-constrained sub-6B local node well.

Specialized model types:

- **Embedding and reranker models.** Contrastive training with the InfoNCE
  objective, hard-negative mining, and Matryoshka representations, plus
  cross-encoder rerankers, evaluated on the Massive Text Embedding Benchmark. A
  different objective and model type, and a strong local fit.
- **Reward and process reward models.** Sequence and step-level scorers that enable
  reinforcement-learning tracks.
- **Speculative-decoding drafts.** Medusa and EAGLE heads and small draft models
  trained specifically to accelerate the inference services in this repository.

Reasoning and reinforcement learning:

- **Online reinforcement learning and RLVR.** Group relative policy optimization
  and related methods with verifiable rewards on math and code, feasible on small
  models. Preference alignment remains out of scope; verifiable-reward reasoning is
  a distinct objective.
- **On-policy distillation.** Transfer of a frontier teacher's reasoning into a
  small student using a dense on-policy signal. As of 2026 this is the standard and
  most sample-efficient route to a strong small reasoning model, more so than
  reinforcement learning alone at this scale.
- **Agentic and tool-use training.** Function-calling and multi-step tool-use
  fine-tuning.

Efficiency and new precision:

- **Native low-bit training.** Training models with 1-bit or ternary weights from
  scratch, following BitNet b1.58 and its 4-bit-activation successors. A 2-billion
  parameter native 1-bit model is now competitive with full-precision peers, and
  native low-bit training is uniquely suited to a memory-constrained local box.
- **Quantization-aware training.** Training with quantization in the loop to
  produce servable low-precision checkpoints.
- **Native sparse attention.** Trainable sparse-attention mechanisms for
  long-context efficiency.
- **New optimizers.** Matrix-orthogonalized updates such as Muon and its
  successors, with attention to the infrastructure quirks they require.

New architectures and paradigms:

- **Diffusion language models.** Masked and continuous text-diffusion models, a
  fast-moving 2026 area that added roughly 90 papers in a single quarter, covering
  scaling, decoding, and alignment. A non-autoregressive alternative worth
  prototyping at small scale.
- **Tokenizer-free and byte-level models.** Byte-latent and byte-level state-space
  models that remove the tokenizer from the stack.
- **Additional state-space and linear-attention backbones.** RWKV-7, xLSTM, and
  gated linear attention beyond Mamba, through the flash-linear-attention library.
- **Mixture-of-experts with expert offload.** Sparse models whose inactive experts
  reside in system RAM while active experts stream to the GPU.

Data, context, and interpretability:

- **Long-context extension.** Positional scaling such as YaRN with continued
  long-context training, evaluated with RULER and needle-in-a-haystack tests.
- **Synthetic data and self-improvement.** Self-instruct, rejection sampling, and
  self-rewarding loops driven by the inference services.
- **Multimodal projector training.** Vision-language and audio adapters over a
  frozen language backbone.
- **Interpretability training.** Sparse-autoencoder training and probing as a
  research and blog track.
- **Model merging.** Combining checkpoints through TIES, DARE, spherical
  interpolation, and model soups with mergekit. A gradient-free technique rather
  than a training type, listed for completeness.

FP8 hardening continues across all phases.

## Proposed repository layout

Training is added to the existing repository rather than a separate one. Training
work is job-based, launched from a recipe and exiting on completion, whereas the
inference services are persistent on ports. The only new persistent surfaces are
observability: JupyterLab, TensorBoard, and MLflow. Many training dependencies,
including transformers, datasets, accelerate, peft, deepspeed, bitsandbytes,
Weights and Biases, and MLflow, are already present in `pyproject.toml`.

```
ai-experiments/
  docker/
    llama-cpu/  llama-gpu/  vllm-gpu/  comfyui-flux/   # existing inference services
    training/                                          # new: trainer image, PyTorch, CUDA 13, FlashAttention, TRL, torchtune, Unsloth
  training/                                            # new: training source
    data/                                              # ingest, dedup, filter, tokenize, shard pipelines
    models/                                            # architecture registry: transformer, Mamba, Titans, HOPE
    recipes/                                           # Hydra and YAML run configs: sft-qlora, continual-pretrain, pretrain-500m
    continual/                                         # outer loop, replay buffer, anti-forgetting hooks, retention eval
    learning/                                          # autograd, annotated models, visualization notebooks
    calculator/                                        # memory-fit solver
  scripts/
    benchmark/  setup/  update/  utils/                # existing
    train/  eval/  export/                             # new training scripts
  docs/
    inference/  optimizations/  os/                    # existing
    training/                                          # this scope doc, internals, training optimizations, benchmark JSON
  Makefile                                             # extended: train-sft, continual-pretrain, pretrain, train-cl, eval, export-gguf, board
  docker-compose.yaml                                  # existing inference services
  pyproject.toml                                       # training dependencies, largely already present
```

## References

- Chen et al. Continual Learning in Large Language Models: Methods, Challenges, and
  Opportunities. arXiv:2603.12658, 2026.
- Behrouz et al. Nested Learning: The Illusion of Deep Learning Architectures.
  arXiv:2512.24695, 2025. Google Research.
- Behrouz et al. Titans: Learning to Memorize at Test Time. 2025.
- Allal et al. SmolLM2: When Smol Goes Big, Data-Centric Training of a Small
  Language Model. arXiv:2502.02737, 2025.
- SmolLM3: smol, multilingual, long-context reasoner. Hugging Face, 2025.
- The Smol Training Playbook. Hugging Face, 2025.
- BitNet b1.58 2B4T Technical Report. arXiv:2504.12285, 2025.
- On-Policy Distillation. Thinking Machines Lab, 2026.
- Awesome-Diffusion-LLM. Community survey of diffusion language models, 2026.

---
*Last Updated: 2026-07-07*
