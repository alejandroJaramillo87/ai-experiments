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
- [Design principle: adopt versus build](#design-principle-adopt-versus-build)
  - [Adopt](#adopt)
  - [Build](#build)
- [Learning track](#learning-track)
- [Version 0.1 scope](#version-01-scope)
  - [Deliverables](#deliverables)
  - [Non-goals for version 0.1](#non-goals-for-version-01)
  - [Success criteria](#success-criteria)
- [Roadmap beyond version 0.1](#roadmap-beyond-version-01)
- [Proposed repository structure](#proposed-repository-structure)
- [Relationship to the inference repository](#relationship-to-the-inference-repository)
- [Open questions](#open-questions)

## Purpose

Provide a reproducible local environment to train, adapt, and align language
models of 6 billion parameters and smaller on a single-node workstation. The
repository targets three outcomes at once: producing models that are useful,
teaching the mathematics and mechanics of deep learning, and saturating the
full machine rather than the GPU alone.

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
   validated recipes: a 1 billion parameter dense pretraining run, a QLoRA
   fine-tune of a 3 billion parameter base, and a small from-scratch
   configuration for the learning track.
4. **One real pretraining run.** Train a dense transformer of approximately 1
   billion parameters from scratch to a target validation perplexity on a fixed
   token budget, with Weights and Biases tracking, resumable and sharded
   checkpointing, and evaluation through lm-evaluation-harness.
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
- FP8 training on Blackwell
- Multimodal training
- Multi-GPU or distributed parallelism
- Diffusion language models and custom kernels

### Success criteria

- The 1 billion parameter model trains to completion locally and produces
  coherent text at the target perplexity.
- Measured peak VRAM for each seed recipe matches the calculator prediction
  within 10 percent.
- The round-trip path serves the trained model through the inference
  repository and returns evaluation numbers.
- The learning-track autograd engine reproduces framework gradients on a test
  network within numerical tolerance.

## Roadmap beyond version 0.1

- **Phase 2.** FP8 training enablement and benchmarks on Blackwell. Continued
  pretraining with replay and learning-rate rewarming. Preference optimization
  through DPO and ORPO. A full Mamba run at approximately 1.3 billion
  parameters. A scaling-law and token-budget calculator.
- **Phase 3.** Mixture-of-experts training with inactive experts resident in
  system RAM. Reasoning alignment through GRPO with verifiable rewards.
  Distillation from a larger teacher model. Embedding-model training with
  contrastive objectives. Speculative-decoding draft models built for the
  inference repository.
- **Phase 4.** Interpretability tooling, including the logit lens and sparse
  autoencoders. Long-context extension. Broader evaluation suites and paper
  reproductions. Integration with the technical blog.

## Proposed repository structure

```
ai-experiments-training/
  docker/
    trainer/          # PyTorch, CUDA 13, FlashAttention, TRL, torchtune
    data-prep/        # CPU tokenization, deduplication, filtering
  configs/            # Hydra and YAML recipes, one file per run
  data/
    pipelines/        # ingest, dedup, filter, tokenize, shard
  recipes/            # validated run configs: pretrain-1b, sft-qlora-3b
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

## Open questions

- Which public corpus and token budget give the best 1 billion parameter
  baseline within a practical local wall-clock time.
- Whether optimizer offload to RAM at 3 billion parameters and above yields an
  acceptable step time on this machine.
- Whether FP8 training on the sm_120 architecture is stable enough to move from
  Phase 2 experiment to default in a later phase.
- Whether the from-scratch learning track lives in this repository or in a
  dedicated companion, given its different audience.

---
*Last Updated: 2026-07-07*
