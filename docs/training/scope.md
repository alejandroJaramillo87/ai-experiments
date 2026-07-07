# Local LLM training repository scope

This document defines the scope for a proposed sibling repository, working name
`ai-experiments-training`, that provides infrastructure and tooling to train
language models locally on the RTX 5090 workstation. It is the training-side
counterpart to this inference repository. The document records what the project
adopts from existing tools, what it builds itself, and the minimum version 0.1
milestone that proves the repository is useful before broader investment.

This is a planning document. No code is implied to exist yet.

## Table of Contents

- [Purpose](#purpose)
- [Goals](#goals)
- [Hardware envelope](#hardware-envelope)
- [Trainable model scale](#trainable-model-scale)
- [Training stages and methods](#training-stages-and-methods)
  - [Pretraining from scratch](#pretraining-from-scratch)
  - [Continued pretraining](#continued-pretraining)
  - [Supervised fine-tuning](#supervised-fine-tuning)
  - [Preference alignment](#preference-alignment)
  - [Parameter-efficient fine-tuning toolkit](#parameter-efficient-fine-tuning-toolkit)
- [Design principle: adopt versus build](#design-principle-adopt-versus-build)
  - [Adopt](#adopt)
  - [Build](#build)
- [Learning track](#learning-track)
- [Continual and lifelong learning](#continual-and-lifelong-learning)
  - [Why it fits a local sub-6B lab](#why-it-fits-a-local-sub-6b-lab)
  - [Taxonomy the repository organizes around](#taxonomy-the-repository-organizes-around)
  - [State of the field as of mid-2026](#state-of-the-field-as-of-mid-2026)
  - [What is shared, built, and adopted](#what-is-shared-built-and-adopted)
- [Version 0.1 scope](#version-01-scope)
  - [Deliverables](#deliverables)
  - [Non-goals for version 0.1](#non-goals-for-version-01)
  - [Success criteria](#success-criteria)
- [Roadmap beyond version 0.1](#roadmap-beyond-version-01)
- [Proposed repository structure](#proposed-repository-structure)
- [Relationship to the inference repository](#relationship-to-the-inference-repository)
- [Resolved decisions](#resolved-decisions)
- [Open questions](#open-questions)
- [References](#references)

## Purpose

Provide a reproducible local environment to train, adapt, and align language
models of 6 billion parameters and smaller on a single-node workstation. The
repository targets three outcomes at once: producing models that are useful,
teaching the mathematics and mechanics of deep learning, and saturating the
full machine rather than the GPU alone.

A single RTX 5090 cannot pretrain a state-of-the-art model from scratch. The
reference small model of this era, SmolLM3-3B, consumed approximately 9,200
GPU-days on 11.2 trillion tokens; reproducing that on one card is on the order
of 50 GPU-years. The repository therefore separates two distinct paths and
serves both:

1. **Pretraining from scratch.** Primarily serves the learning pillar and the
   goal of owning the full stack. The realistic target is a coherent and
   servable model, not a state-of-the-art one.
2. **Specializing a strong open base.** Continued pretraining, fine-tuning, and
   alignment of an existing model such as SmolLM3, Qwen, or Llama-3.2. This is
   the path that yields genuinely useful local models within days rather than
   weeks, and it is where continual learning lives.

## Goals

The project has three co-equal pillars. Every design decision serves at least
one of them.

1. **Produce useful models.** Train dense transformer models in the 1 billion
   to 3 billion parameter range from scratch, continue pretraining on domain
   data, and fine-tune or align models up to 6 billion parameters. The output
   is a coherent model, not a demonstration.

2. **Teach deep learning internals.** Serve as a working environment to
   understand backpropagation, the matrix algebra of the forward and backward
   pass, gradient flow, optimization dynamics, tokenization, and embeddings.
   Annotated from-scratch implementations and visualizations are first-class
   deliverables, not documentation afterthoughts.

3. **Use the whole machine.** Place each workload on the correct hardware tier
   and overlap tiers so none sits idle: GPU for the training step, CPU threads
   for the data pipeline and evaluation, system RAM and NVMe for optimizer and
   activation offload.

## Hardware envelope

| Component | Specification | Primary training role |
|-----------|---------------|-----------------------|
| GPU | RTX 5090, 32GB VRAM, Blackwell | Forward and backward pass, matrix multiplication, attention and state-space kernels |
| CPU | AMD Ryzen 9950X, 16 cores, 32 threads | Data pipeline, tokenization, deduplication, concurrent evaluation, checkpoint serialization |
| RAM | 128GB DDR5-6000 | Optimizer-state offload, dataset cache, memory-mapped token shards |
| Storage | NVMe at `/mnt/ai-data` | Activation and parameter spill, checkpoint store, dataset shards |

The binding constraint is 32GB of VRAM. The repository design is memory-first:
fit the largest useful model and longest useful context into 32GB, then use
system RAM and NVMe to extend that ceiling through offload.

## Trainable model scale

Training memory is the sum of parameters, gradients, optimizer states, and
activations. Optimizer states dominate. The table below estimates the practical
ceiling on approximately 28GB of usable VRAM. Values exclude activations, which
gradient checkpointing reduces substantially.

| Method | Bytes per parameter | Practical ceiling | Notes |
|--------|---------------------|-------------------|-------|
| Full training, AdamW, bf16 with fp32 master | ~16 | ~1B, tight at 1.5B | The from-scratch learning zone |
| Full training, 8-bit AdamW, gradient checkpointing | ~10 | ~2.5B to 3B | bitsandbytes 8-bit optimizer |
| Full training with optimizer offload to RAM | ~2 on GPU | ~3B to 6B | Slower step time, RAM absorbs optimizer state |
| LoRA, base frozen in bf16 | ~2 | 6B to 8B | Adapters are small relative to base |
| QLoRA, base in 4-bit | ~0.5 | 6B to 8B, 13B feasible | Context length becomes the limiter |
| Pretraining from scratch | Author-defined | 100M to 1B for full runs | Full control of tokenizer and architecture |

The capability statement for the repository: full pretraining is the 100
million to 1 billion parameter zone, continued pretraining reaches 1 billion to
3 billion, and fine-tuning or alignment reaches 6 billion through QLoRA.

## Training stages and methods

The repository provides four training stages. Each is a code path built on an
adopted trainer, and each applies to both a strong open base and a from-scratch
model. These stages are the foundation the continual-learning regime later runs
as a sequence rather than once. The table summarizes them, and the subsections
give detail.

| Stage | Objective | Primary adopted tool | Scale on 32GB | Continual-mode extension |
|-------|-----------|----------------------|---------------|--------------------------|
| Pretraining from scratch | Next-token prediction | nanotron, litGPT, custom loop | 100M to 1B full | Not applicable |
| Continued pretraining | Next-token prediction on new corpora | TRL, Axolotl | 1B to 3B full with offload, larger with LoRA | Continual pre-training over a corpus stream |
| Supervised fine-tuning | Instruction and chat completion | TRL, torchtune, Unsloth | 1.5B to 2B full, 6B to 8B with QLoRA | Continual fine-tuning over a task sequence |
| Preference alignment | Match human or model preferences | TRL, Axolotl | 6B to 8B with QLoRA | Continual alignment over evolving preferences |

### Pretraining from scratch

Train a model from random initialization with a causal language-modeling
objective. This stage owns the tokenizer, the architecture, and the full data
pipeline. Kernels come from FlashAttention and Liger-Kernel. The realistic local
target is 100 million to 1 billion parameters. The primary value is the learning
pillar and control of the entire stack rather than model quality.

### Continued pretraining

Resume training on a strong open base over new corpora, most often for domain
adaptation. The objective is unchanged, but two techniques govern quality:
learning-rate rewarming and re-decay, typically through a warmup-stable-decay
schedule, and replay of a slice of the original data distribution to limit
forgetting. Data mixing weights control the balance of new and original domains.
Full training reaches 1 billion to 3 billion parameters with optimizer offload
to system RAM, or larger through low-rank adaptation. This single run is the
precursor to continual pre-training, which streams many such adaptations in
sequence.

### Supervised fine-tuning

Supervised fine-tuning (SFT) trains a base model on instruction and chat data.
The stage handles chat templates, multi-turn conversations, prompt-loss masking
so that loss applies only to completions, and sequence packing for efficiency.
Full fine-tuning reaches roughly 1.5 billion to 2 billion parameters; parameter-
efficient methods reach 6 billion to 8 billion. A distinct and high-value variant
is distillation fine-tuning, in which a smaller model learns from the responses
or output distribution of a larger teacher. Distillation is the most direct route
to a capable sub-3 billion parameter model on this hardware.

### Preference alignment

Align a fine-tuned model to human or model-generated preferences. The default
methods are those that need no separate reward model and suit a single GPU:
direct preference optimization (DPO), odds-ratio preference optimization (ORPO),
Kahneman-Tversky optimization (KTO), and simple preference optimization (SimPO).
Online reinforcement learning, through reward modeling with proximal policy
optimization or group relative policy optimization (GRPO), supports reasoning
alignment with verifiable rewards and is deferred to a later phase. Preference
alignment reaches 6 billion to 8 billion parameters through QLoRA.

### Parameter-efficient fine-tuning toolkit

Parameter-efficient fine-tuning (PEFT) is the substrate shared by the fine-tuning
and alignment stages and by architecture-based continual learning. The toolkit
adopts Hugging Face PEFT and bitsandbytes and covers low-rank adaptation (LoRA),
its quantized form (QLoRA), weight-decomposed adaptation (DoRA), rank-stabilized
LoRA, IA3, and prompt and prefix tuning. Adapter merging through TIES, DARE, and
spherical interpolation, using mergekit, combines separately trained adapters.
The same adapter-isolation mechanism underpins the architecture family of
continual-learning methods, for example orthogonal LoRA.

## Design principle: adopt versus build

Roughly 80 percent of the repository is integration, configuration, and
orchestration. Roughly 20 percent is code specific to this hardware envelope
and the two-repository lifecycle. The test for every component is a single
question: does the component depend on the specific envelope of 32GB Blackwell,
128GB RAM, local single node, and 6 billion parameters and under, or is it the
same problem faced by anyone with a GPU. Shared problems are adopted. Envelope-
specific problems are built.

Writing a custom trainer, dataloader, deduplication primitive, or optimizer is
treated as an anti-pattern. Each is a maintenance treadmill against well-funded
upstream projects.

### Adopt

| Area | Tools |
|------|-------|
| Kernels | FlashAttention, Liger-Kernel, Unsloth, mamba-ssm, causal-conv1d, flash-linear-attention, Transformer Engine |
| Trainers | TRL, Axolotl, torchtune, Unsloth, Llama-Factory |
| Memory and offload engine | DeepSpeed ZeRO-Offload and ZeRO-Infinity, FSDP2, bitsandbytes |
| Data pipeline primitives | Hugging Face datatrove, AI2 dolma, NVIDIA NeMo-Curator, text-dedup |
| Synthetic data generation | distilabel wired to the inference repository |
| Tokenizer training | Hugging Face tokenizers, SentencePiece |
| Model architectures for production | Hugging Face transformers, litGPT, official Mamba and RWKV repositories |
| Pretraining at scale | torchtitan, nanotron, OLMo-core |
| Evaluation | lm-evaluation-harness, lighteval, MTEB |
| Experiment tracking, configuration, search | Weights and Biases, MLflow, TensorBoard, Hydra, Optuna, Ray Tune |
| Quantization, export, merge | llama.cpp for GGUF, AutoAWQ, GPTQModel, mergekit |

### Build

The differentiated core, in priority order. Each item depends on the specific
hardware envelope or the two-repository lifecycle and cannot be downloaded.

1. **Memory-fit solver and recipe library.** A calculator that maps a model
   configuration, optimizer, checkpointing strategy, and offload setting to
   peak VRAM, maximum sequence length, and maximum batch size on 32GB. Paired
   with a curated set of validated recipes tuned to this exact machine. No
   existing tool answers this question for a single prosumer node.

2. **Heterogeneous-compute orchestration.** The whole-machine choreography that
   runs the CPU data pipeline, the GPU training step, concurrent CPU-side
   evaluation, and inference-repository synthetic-data generation at the same
   time, with NUMA and core pinning for the 9950X and RAM-disk staging of hot
   shards.

3. **Blackwell FP8 training enablement and benchmarks.** Making the existing
   FP8 training stack run on the sm_120 architecture and measuring the result
   in the reproducible JSON format this project already uses for inference
   benchmarks. This is enablement and measurement, not kernel authorship.

4. **Round-trip integration to the inference repository.** Export, quantize,
   serve, and evaluate-by-serving as one loop specific to this two-repository
   setup.

5. **From-scratch educational implementations.** Annotated transformer and
   Mamba implementations built for understanding, described in the next
   section. These serve the learning pillar rather than production.

6. **Continual-learning regime and test-time-memory architectures.** The outer
   loop, replay buffer, anti-forgetting hooks, retention evaluation, and the
   architectural memory models that let a model keep learning after pretraining.
   Detailed in the [Continual and lifelong learning](#continual-and-lifelong-learning)
   section. The custom footprint is small and reuses the shared trainer, data
   pipeline, and offload.

## Learning track

The learning pillar is a build item because no external project is organized
around this specific goal on this specific machine. The track builds
understanding from first principles upward and connects each concept to a
runnable artifact.

- **Autograd from scratch.** A minimal reverse-mode automatic differentiation
  engine over scalars, then over tensors, to make backpropagation concrete
  before any framework abstraction is introduced.
- **Matrix algebra of the forward and backward pass.** Written derivations of
  the shapes and gradients for each layer of a transformer block: embedding
  lookup, attention projections, softmax, the feed-forward network, and the
  language-model head. Each derivation pairs with code that checks the manual
  gradient against the framework gradient.
- **Single-file annotated transformer.** A readable decoder-only model with
  every tensor operation commented, tied to the derivations above.
- **Single-file annotated state-space model.** A readable Mamba-style model
  that contrasts the recurrence and scan of a state-space model with the
  attention of a transformer.
- **Training-dynamics visualizations.** Notebooks that plot gradient norms,
  activation statistics, dead-neuron counts, learning-rate schedules, loss
  curves, and the evolution of the embedding space during a real run.
- **Ablation harness.** A configuration surface to toggle one architectural or
  optimization choice at a time and observe the effect on loss and stability.

The learning track uses the same data pipeline, tracking, and hardware as the
production track, so lessons transfer directly to real runs.

## Continual and lifelong learning

Continual learning is the paradigm in which a model adapts to a stream of new
knowledge and tasks over time while retaining prior capability, rather than
being trained once and frozen. It is distinct from a single continued
pretraining run: it is a training regime that spans pretraining, fine-tuning,
and alignment, and its defining problem is catastrophic forgetting, the loss of
earlier capability when new gradients overwrite the weights that held it.

The repository treats continual learning as a first-class regime rather than a
separate model type. It is not a parallel training stack. It is the existing
trainer, data pipeline, checkpointing, and offload, wrapped in an outer
sequential loop with the anti-forgetting and retention-measurement machinery
that every stage shares.

### Why it fits a local sub-6B lab

Continual learning research is bounded by method design and iteration speed, not
by scale. A single node that runs a full task sequence on a 1 billion to 3
billion parameter model in hours is better suited to this work than a cluster
optimized for one large run. Continual fine-tuning and continual alignment at
this scale complete in days on the RTX 5090. The hardware shape is an asset:
replay buffers and external memory stores reside in the 128GB of system RAM
while the GPU trains. This is a research direction where a local lab can produce
original results that larger organizations are not optimizing for.

### Taxonomy the repository organizes around

The repository follows the taxonomy of the mid-2026 survey by Chen and
colleagues, which structures continual learning across three training stages,
each addressed by three method families. The three stages are exactly the stages
defined in [Training stages and methods](#training-stages-and-methods), run as a
sequence rather than once. Continual learning adds no new trainer; it adds the
outer loop and the anti-forgetting machinery around those existing paths.

Three stages:

- **Continual pre-training.** A stream of corpora, adapted with replay and
  learning-rate rewarming on the pretraining code path.
- **Continual fine-tuning.** A sequence of tasks or instruction sets on the
  supervised fine-tuning code path.
- **Continual alignment.** Preferences that evolve over time, re-run on the
  preference-optimization code path.

Three method families:

- **Rehearsal and replay.** Mix samples of prior data into new batches. Implemented
  as a data-pipeline component with the buffer resident in system RAM.
- **Regularization.** Penalize changes to weights important to prior tasks, for
  example elastic weight consolidation and learning-without-forgetting. Implemented
  as hooks into the loss and optimizer.
- **Architecture.** Isolate new knowledge in new parameters, for example
  adapter and low-rank isolation, orthogonal low-rank adaptation, and
  mixture-of-experts routing. Implemented mostly through adopted
  parameter-efficient libraries with thin routing glue.

A fourth mechanism, targeted knowledge editing such as MEMIT and WISE, does not
use the training loop at all. It performs closed-form or optimization-based
weight surgery on specific layers and is adopted as a bolt-on library rather
than built.

### State of the field as of mid-2026

Continual learning is an active frontier and is not solved. Three threads are
relevant to this repository.

- **Architectural test-time memory.** Google's Nested Learning paradigm and its
  HOPE architecture, published in late 2025, reframe a model as a set of nested
  optimization problems that update at different timescales. HOPE couples a
  self-modifying core, derived from the earlier Titans test-time memory work,
  with a continuum memory system of components that learn at different rates. It
  reports strong language-modeling and long-context memory results. It moves
  continual learning from a training-time addition to a property of the
  architecture itself. Several open reproductions already exist at small scale,
  including nano-scale educational implementations of a few tens of millions of
  parameters, which makes this directly reproducible on the RTX 5090 and a
  natural addition to the model registry alongside Mamba.
- **Improved replay and time-continual benchmarks.** Surprise-driven prioritized
  replay, multi-timescale memory dynamics, and web-scale time-continual
  pretraining benchmarks such as TiC-LM sharpen the parametric approach.
- **Agent and on-device memory.** Just-in-time reinforcement learning that
  adjusts logits in closed form without gradient updates, and on-device
  continual learning through budget-curated memory, target exactly the local
  and private setting this repository occupies.

The honest caveat: self-modifying and self-editing models still degrade on prior
tasks as updates accumulate, official weights for HOPE are not released, and the
strongest reproductions are unofficial. This is a research pillar, not a solved
integration. That is the reason it belongs in the research-grade part of the
roadmap rather than in version 0.1.

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

## Version 0.1 scope

Version 0.1 proves the repository is useful with the minimum surface area. It
combines mostly adopted infrastructure with the two highest-value build items:
the memory-fit recipe system and one real pretraining run. It also seeds the
learning track.

### Deliverables

1. **Training container and environment.** One Docker image with PyTorch on
   CUDA 13, FlashAttention, bitsandbytes, and a trainer stack of TRL and
   torchtune. A Poetry dependency group parallel to the inference project.
2. **Data path for one corpus.** Ingest a public corpus sample, train a
   byte-pair-encoding tokenizer, and produce memory-mapped token shards using
   an adopted pipeline. Record token counts and length distributions.
3. **Memory-fit calculator and three seed recipes.** The VRAM solver plus three
   validated recipes: a 360 million to 500 million parameter dense pretraining
   run, a QLoRA fine-tune of a 3 billion parameter base, and a small
   from-scratch configuration for the learning track.
4. **One real pretraining run.** Train a dense transformer of 360 million to 500
   million parameters from scratch on 10 billion to 20 billion tokens of
   FineWeb-Edu to a target validation perplexity, with Weights and Biases
   tracking, resumable and sharded checkpointing, and evaluation through
   lm-evaluation-harness. This run completes in two to four days on the RTX 5090
   and validates the whole stack. The first 1 billion parameter run is deferred
   to Phase 1.5, once the pipeline is proven.
5. **Round-trip proof.** Export the trained checkpoint, quantize to GGUF, serve
   it on the inference repository, and evaluate by serving.
6. **Learning-track seed.** The from-scratch autograd engine, the single-file
   annotated transformer with manual-backpropagation notes, and one
   gradient-and-activation visualization notebook.
7. **Benchmark record.** Model FLOPs utilization, tokens per second, and peak
   VRAM captured in the reproducible JSON format used elsewhere in the project.

### Non-goals for version 0.1

The following are deferred to keep version 0.1 tractable. Each is valuable and
appears on the roadmap.

- Mixture-of-experts models and expert offload
- Reinforcement-learning alignment: PPO, GRPO, and reward modeling
- Continual learning, which is a research-grade pillar deferred to Phase 2 and later
- FP8 training on Blackwell
- Multimodal training
- Multi-GPU or distributed parallelism
- Diffusion language models and custom kernels

### Success criteria

- The 360 million to 500 million parameter model trains to completion locally
  and produces coherent text at the target perplexity.
- Measured peak VRAM for each seed recipe matches the calculator prediction
  within 10 percent.
- The round-trip path serves the trained model through the inference
  repository and returns evaluation numbers.
- The learning-track autograd engine reproduces framework gradients on a test
  network within numerical tolerance.

## Roadmap beyond version 0.1

- **Phase 1.5.** The first 1 billion to 1.5 billion parameter pretraining run on
  50 billion to 100 billion tokens using a SmolLM2-style data mixture, reusing a
  proven tokenizer. This is a multi-week run that produces a useful, though not
  state-of-the-art, small base.
- **Phase 2.** FP8 training enablement and benchmarks on Blackwell. Continued
  pretraining with replay and learning-rate rewarming. Preference optimization
  through DPO and ORPO. A full Mamba run at approximately 1.3 billion
  parameters. A scaling-law and token-budget calculator. Continual learning,
  first stage: the parametric regime, comprising the outer sequential loop,
  replay buffer, regularization hooks, retention evaluation, and knowledge
  editing through EasyEdit.
- **Phase 3.** Mixture-of-experts training with inactive experts resident in
  system RAM. Reasoning alignment through GRPO with verifiable rewards.
  Distillation from a larger teacher model. Embedding-model training with
  contrastive objectives. Speculative-decoding draft models built for the
  inference repository. Continual learning, second stage: architectural
  test-time memory, adding Titans and a HOPE reproduction to the model registry.
- **Phase 4.** Interpretability tooling, including the logit lens and sparse
  autoencoders. Long-context extension. Broader evaluation suites and paper
  reproductions. Continual learning, third stage: the forgetting-mitigation
  ablation harness and self-editing experiments. Integration with the technical
  blog.

## Proposed repository structure

```
ai-experiments-training/
  docker/
    trainer/          # PyTorch, CUDA 13, FlashAttention, TRL, torchtune
    data-prep/        # CPU tokenization, deduplication, filtering
  configs/            # Hydra and YAML recipes, one file per run
  data/
    pipelines/        # ingest, dedup, filter, tokenize, shard
  models/             # architecture registry: transformer, Mamba, Titans, HOPE
  recipes/            # validated run configs: pretrain-500m, sft-qlora-3b
  continual/          # outer loop, replay buffer, anti-forgetting hooks, retention eval
  learning/           # autograd, annotated models, visualization notebooks
  scripts/
    train/  eval/  export/
  docs/
    internals/        # tokenization, embeddings, backpropagation, attention, state-space models
    optimizations/    # memory, offload, FP8, with benchmark JSON
  Makefile
```

Service surfaces that remain persistent are observability only: JupyterLab,
TensorBoard, and MLflow. The training workload itself is job-based and launched
from a recipe, not a long-running service on a port.

## Relationship to the inference repository

The two repositories form one lifecycle. The training repository produces a
checkpoint, merges any adapter, and quantizes to a servable format. The
inference repository serves that artifact on the existing GPU services and
returns latency and quality measurements. The training repository can consume
the inference repository as a submodule so that evaluate-by-serving uses the
real inference stack. The result is a closed loop: train, quantize, serve,
benchmark, and refine on the same hardware.

## Resolved decisions

- **Corpus and token budget.** The version 0.1 validation run uses FineWeb-Edu
  at 10 billion to 20 billion tokens on a 360 million to 500 million parameter
  model, with a purpose-trained 32k byte-pair-encoding tokenizer. The first
  useful model, in Phase 1.5, uses a SmolLM2-style mixture of FineWeb-Edu, DCLM,
  synthetic Cosmopedia data, and a code and math slice at 50 billion to 100
  billion tokens, reusing a proven tokenizer.
- **Learning-track location.** The learning track remains in this repository as
  a self-contained `learning/` module. A split into a companion repository is
  reconsidered at Phase 3, triggered by the track growing a distinct
  blog-driven audience.

## Open questions

- Whether optimizer offload to RAM at 3 billion parameters and above yields an
  acceptable step time on this machine.
- Whether FP8 training on the sm_120 architecture is stable enough to move from
  a Phase 2 experiment to a default in a later phase.
- Whether a HOPE reproduction scales usefully from the tens-of-millions
  parameter educational range into the 1 billion to 3 billion parameter range on
  a single node, given that no official weights exist.
- Which continual-learning benchmark suite to standardize on for retention
  metrics, given several competing 2025 and 2026 benchmarks.

## References

- Chen et al. Continual Learning in Large Language Models: Methods, Challenges,
  and Opportunities. arXiv:2603.12658, 2026.
- Behrouz et al. Nested Learning: The Illusion of Deep Learning Architectures.
  arXiv:2512.24695, 2025. Google Research.
- Behrouz et al. Titans: Learning to Memorize at Test Time. 2025.
- Allal et al. SmolLM2: When Smol Goes Big, Data-Centric Training of a Small
  Language Model. arXiv:2502.02737, 2025.
- SmolLM3: smol, multilingual, long-context reasoner. Hugging Face, 2025.
- The Smol Training Playbook. Hugging Face, 2025.

---
*Last Updated: 2026-07-07*
