# Training Infrastructure Roadmap

Comprehensive roadmap for implementing small-to-medium scale language model training capabilities in the ai-experiments repository.

## Table of Contents

- [Vision](#vision)
- [Unique Value Proposition](#unique-value-proposition)
- [Scope](#scope)
- [Architecture Foundation](#architecture-foundation)
- [Design Decisions](#design-decisions)
- [Hardware Requirements](#hardware-requirements)
- [Software Stack](#software-stack)
- [Directory Structure](#directory-structure)
- [Services Architecture](#services-architecture)
- [Integration with v1.0 Infrastructure](#integration-with-v10-infrastructure)
- [Custom Tooling](#custom-tooling)
- [Operating System Optimizations](#operating-system-optimizations)
- [Optimization Strategy](#optimization-strategy)
- [Quantization & Deployment Pipeline](#quantization--deployment-pipeline)
- [Evaluation Strategy](#evaluation-strategy)
- [Experimental Optimizations](#experimental-optimizations)
- [Implementation Phases](#implementation-phases)
- [Success Criteria](#success-criteria)
- [Attribution](#attribution)
- [Dependencies](#dependencies)
- [Future Considerations](#future-considerations)

## Vision

Transform the ai-experiments repository from best-in-class inference infrastructure into a complete AI lifecycle platform for prosumer hardware. Enable training of small-to-medium language models (1B-2B parameters) with the same systematic optimization, rigorous documentation, and production stability that characterizes the current inference services.

## Unique Value Proposition

This project differentiates from educational implementations by providing production infrastructure optimized for specific hardware rather than general-purpose training code.

### What Educational Implementations Provide

**tiny-qwen** (github.com/Emericen/tiny-qwen): Clean, minimal Qwen3 architecture implementation in PyTorch. Educational focus on understanding transformer mechanics. Architecture code designed for learning.

**rasbt/qwen3-from-scratch**: From-scratch educational implementation with detailed explanations. Optimized for readability and research. Minimal dependencies for understanding core concepts.

**ideaweaver-ai/qwen3-from-scratch**: Complete training examples on small datasets. Demonstrates training loop mechanics. Educational tool for learning language model training.

### What ai-experiments v2 Provides

**Production infrastructure layer**: Docker orchestration, checkpoint management, monitoring, automated deployment. Industrial-strength tooling for reliable 24/7 operation.

**Hardware-specific optimization**: Tuned exclusively for RTX 5090 (Blackwell) + AMD Ryzen 9950X (Zen 5). No compromises for portability. Systematic optimization achieving measurable performance gains.

**Complete pipeline**: Raw text → trained model → quantized formats (GGUF + SafeTensors) → deployed to inference services. End-to-end automation with validation at each stage.

**Integration with existing services**: Concurrent GPU training while CPU inference runs. Quantized models automatically deploy to llama-cpu, llama-gpu, vllm-gpu services from v1.0.

**Systematic optimization research**: Baseline measurement, six optimizations applied methodically, each gain documented with benchmarks. Focus on discovering what works on this specific hardware configuration.

**Custom C++ tooling**: SIMD-optimized tokenizer (AVX-512), PyTorch memory optimizer, huge pages integration. Performance-critical paths implemented in compiled code where Python overhead matters.

**Research documentation**: Discoveries documented for curiosity-chronicles blog. Serves llama-zen-turbo project as reference implementation. Bleeding-edge hardware optimization research.

### Scope of This Project

**We build the production layer, not the architecture**. Qwen3-1.7B architecture comes from tiny-qwen. We add infrastructure, optimization, quantization, deployment, monitoring, and integration.

**First to optimize Qwen3-1.7B training for RTX 5090 + Zen 5 prosumer hardware**. Document discoveries, measure gains, prove feasibility of high-quality training on workstation hardware.

## Scope

### Included

**Training from scratch**: Train base language models on raw text corpora using Qwen3-1.7B architecture (1.7 billion parameters). Start with randomly initialized weights and pretrain on large text datasets like OpenWebText. Requires substantial compute and data (billions of tokens minimum). Produces general-purpose base model with broad language understanding.

**Continued pretraining**: Adapt existing pretrained base models to specialized domains through additional pretraining on domain-specific corpora. Start from official Qwen3-1.7B checkpoint (or custom trained checkpoint) and continue training on focused datasets (code, medical, legal, math, scientific literature). Requires less compute and data than training from scratch (millions to low billions of tokens). Produces domain-specialized base model while retaining general capabilities. Also called domain adaptation or continued training.

**Fine-tuning**: Full parameter fine-tuning and parameter-efficient fine-tuning with Low-Rank Adaptation (LoRA) for instruction following and task-specific behaviors. Start from base model checkpoint and train on instruction-response pairs or task-specific labeled data. Requires minimal compute and data (thousands to hundreds of thousands of examples). Produces task-optimized model for specific use cases like chat, question-answering, or classification.

**Quantization**: Post-training quantization to GGUF format for llama.cpp inference and HuggingFace SafeTensors format for vLLM inference.

**Dataset preparation**: Complete pipeline for data collection, cleaning, tokenization with Qwen3 tokenizer (151,646 vocab), and validation.

**Evaluation**: Model quality assessment through perplexity measurements, downstream task evaluation, and instruction-following benchmarks.

**Optimization**: Systematic optimization of training throughput, memory efficiency, and data loading performance.

**Deployment integration**: Automatic deployment of quantized models to existing v1.0 inference services with validation.

### Excluded

**Multi-GPU training**: Single RTX 5090 only. Distributed training across multiple GPUs or nodes is out of scope.

**Large model training**: Models exceeding 2B parameters are beyond single-GPU memory constraints.

**Cloud deployment**: Focus remains on local workstation infrastructure.

**Production serving**: Training services are for model development, not serving production traffic.

## Training Types Comparison

This section provides detailed comparison of the three training approaches to guide implementation and hyperparameter selection.

### When to Use Each Approach

**Training from scratch**:
- Building custom model with unique tokenizer or architecture modifications
- Research on training dynamics and optimization techniques
- Educational purposes to understand full training process
- No suitable pretrained model exists for starting point
- Full control over training data and model characteristics required

**Continued pretraining**:
- Adapting general model to specialized domain (medical, legal, code, finance, scientific)
- Improving model performance on underrepresented languages or topics
- Adding knowledge not present in original pretraining data
- Domain shift: production data distribution differs significantly from pretraining data
- Want to preserve general capabilities while gaining domain expertise

**Fine-tuning**:
- Teaching specific task behavior (instruction-following, chat, classification, summarization)
- Aligning model outputs with desired format and style
- Small labeled dataset available (thousands to hundreds of thousands examples)
- Need fast iteration on task-specific improvements
- Production deployment with specific use case requirements

### Hyperparameter Comparison

**Initial checkpoint**:
- From scratch: Random initialization (torch.nn.init.normal_ with small std)
- Continued pretraining: Load official Qwen3-1.7B checkpoint or custom base model
- Fine-tuning: Load base model or domain-adapted checkpoint

**Learning rate**:
- From scratch: 3e-4 to 6e-4 (standard pretraining LR)
- Continued pretraining: 1e-4 to 3e-4 (lower to avoid catastrophic forgetting)
- Fine-tuning full: 1e-5 to 5e-5 (very low to preserve pretrained features)
- Fine-tuning LoRA: 1e-4 to 3e-4 (higher acceptable due to limited parameters)

**Warmup steps**:
- From scratch: 2000-5000 steps (stable gradient scaling from zero)
- Continued pretraining: 500-2000 steps (shorter warmup, already stable)
- Fine-tuning: 100-500 steps (minimal warmup needed)

**Training duration**:
- From scratch: Billions of tokens (10B+ for reasonable quality)
- Continued pretraining: Millions to low billions of tokens (100M-5B depending on domain)
- Fine-tuning: Thousands to millions of tokens (depends on task complexity)

**Batch size**:
- From scratch: Large as VRAM allows (128-512 effective via gradient accumulation)
- Continued pretraining: Medium to large (64-256 effective batch size)
- Fine-tuning: Small to medium (8-64 effective batch size, avoid overfitting)

**Learning rate schedule**:
- From scratch: Warmup + cosine decay to 10% of peak LR
- Continued pretraining: Warmup + cosine decay to 10% of peak LR
- Fine-tuning: Warmup + cosine decay to 0 or constant LR after warmup

**Gradient clipping**:
- All types: 1.0 max norm (prevents training instability)

**Weight decay**:
- From scratch: 0.1 (standard regularization)
- Continued pretraining: 0.1 (maintain regularization)
- Fine-tuning: 0.01-0.1 (lighter regularization or zero for small datasets)

**Dropout** (if using):
- From scratch: 0.0-0.1 (minimal in modern architectures with strong regularization)
- Continued pretraining: 0.0-0.1 (maintain original architecture settings)
- Fine-tuning: 0.0-0.2 (increased dropout can reduce overfitting on small datasets)

### Dataset Requirements

**Training from scratch**:
- Size: 10B+ tokens minimum for reasonable quality
- Diversity: Broad coverage of topics, writing styles, domains
- Quality: High-quality text, deduplication critical
- Examples: OpenWebText (8B tokens), The Pile (825GB), C4, RedPajama

**Continued pretraining**:
- Size: 100M-5B tokens depending on domain complexity
- Diversity: Focused on target domain but some general text for stability
- Quality: High domain-specific quality, acceptable to include technical jargon
- Examples: PubMed (medical), arXiv (scientific), legal corpora, GitHub (code)
- Mixing: Often mix 80-90% domain data with 10-20% general data to prevent catastrophic forgetting

**Fine-tuning**:
- Size: 1K-1M examples (instruction-response pairs or labeled data)
- Diversity: Cover expected use cases and edge cases
- Quality: Extremely high quality, manual review often necessary
- Examples: FLAN, Alpaca, ShareGPT, task-specific datasets
- Format: Structured instruction-response pairs with system prompts

### Evaluation Differences

**Training from scratch**:
- Primary: Validation loss and perplexity on held-out general text
- Secondary: Downstream task performance (MMLU, HellaSwag, ARC)
- Goal: Broad language understanding and generation capability

**Continued pretraining**:
- Primary: Perplexity on domain-specific held-out text
- Secondary: Domain-specific benchmarks (e.g., MedQA for medical, HumanEval for code)
- Goal: Low domain perplexity while maintaining general capability (measure both)

**Fine-tuning**:
- Primary: Task-specific metrics (accuracy, F1, ROUGE, exact match)
- Secondary: Qualitative evaluation of outputs, instruction-following accuracy
- Goal: High task performance and output quality aligned with requirements

### Use Case Examples

**Training from scratch examples**:
- Research: Studying impact of tokenizer vocabulary size on model quality
- Custom architecture: Modified attention mechanism requiring fresh training
- Specialized tokenizer: Custom byte-pair encoding for specific language or domain
- Educational: Understanding transformer training dynamics from initialization

**Continued pretraining examples**:
- Code model: Continue training Qwen3-1.7B on GitHub + StackOverflow data
- Medical model: Adapt base model using PubMed abstracts + clinical notes
- Legal assistant: Continue training on case law + legal documents + contracts
- Math reasoning: Adapt model using mathematical proofs + problem solutions
- Financial analysis: Domain adaptation using financial reports + market data

**Fine-tuning examples**:
- Instruction-following: Train on Alpaca/FLAN to follow user instructions
- Code generation: Fine-tune on formatted code problems with solutions
- Conversational: Adapt to multi-turn dialogue format using ShareGPT
- Summarization: Fine-tune on document-summary pairs
- Classification: Train on labeled examples for sentiment/topic classification

## Architecture Foundation

### Base Implementation

**Architecture**: Qwen3-1.7B (1.7 billion parameters)
**Source**: tiny-qwen implementation (github.com/Emericen/tiny-qwen)
**License**: Apache 2.0

### Why Qwen3-1.7B

**Modern architecture** (April 2025 release): Incorporates latest transformer improvements including Grouped Query Attention, Rotary Position Embeddings, SwiGLU activation, and RMSNorm. Significantly more efficient than older architectures like GPT-2.

**Appropriate scale**: 1.7B parameters fits comfortably in 32GB VRAM with room for optimization. Larger than educational models (10M-124M) but smaller than multi-GPU requirements (7B+). Right size for prosumer hardware exploration.

**Strong pretraining**: Alibaba pretrained on 36 trillion tokens across 119 languages. Using this architecture for training from scratch benefits from proven design validated at massive scale.

**Proven implementation**: tiny-qwen provides clean, minimal implementation that works. No need to debug architecture code. Focus effort on infrastructure and optimization.

**Training characteristics**: Strong-to-weak distillation training methodology well documented. Architecture designed for efficiency with tied embeddings, optimal layer configurations.

### Architecture Components

**Transformer architecture**: Decoder-only with 24 layers, 2048 hidden dimensions, 16 attention heads.

**Grouped Query Attention (GQA)**: Reduces memory and computation vs Multi-Head Attention. Improved inference speed without quality loss.

**Rotary Position Embeddings (RoPE)**: Better position encoding than learned embeddings used in older architectures. Enables longer context generalization.

**SwiGLU activation**: Superior to GELU used in GPT-2. Better gradient flow and training stability.

**RMSNorm**: Faster and more memory-efficient than LayerNorm. 7-64% speedup for normalization operations.

**Tied embeddings**: Input and output embeddings shared for parameter efficiency. Standard practice for models under 4B parameters.

**Byte-level BPE tokenizer**: 151,646 vocabulary size. Handles all Unicode characters without special cases.

**Context window**: 32K tokens during pretraining, validated for long-context understanding.

### References

**Implementation sources**:
- tiny-qwen (github.com/Emericen/tiny-qwen) - Minimal PyTorch implementation
- rasbt/qwen3-from-scratch (huggingface.co/rasbt/qwen3-from-scratch) - Educational guide
- ideaweaver-ai/qwen3-from-scratch (github.com/ideaweaver-ai/qwen3-from-scratch) - Training examples

**Technical documentation**:
- Qwen3 technical report: Qwen team, Alibaba Cloud (April 2025)
- Qwen blog: qwenlm.github.io/blog/qwen3/
- Transformers documentation: huggingface.co/docs/transformers/model_doc/qwen3

**Architecture details**: See references above for layer specifications, attention mechanism implementation, and training hyperparameters. ai-experiments focuses on infrastructure, not reimplementation.

### Our Integration Approach

**Use tiny-qwen architecture as-is**: No modifications to core transformer implementation. Proven code reduces risk.

**Wrap with production infrastructure**: Add checkpoint management, monitoring, quantization, deployment automation around base architecture.

**Focus on optimization**: Apply systematic optimizations at infrastructure level. Hardware tuning, data loading, memory management, I/O optimization.

**Document discoveries**: Measure what works on RTX 5090 + Zen 5. Publish findings for curiosity-chronicles blog content.

## Design Decisions

This section captures critical decisions made during roadmap development with rationale.

### Checkpoint Strategy

**Decision**: Save checkpoints every 30 minutes AND every 5000 steps, keep last 3 plus best checkpoint.

**Rationale**:
- 30-minute interval: Limits maximum lost work to 30 minutes if training crashes
- 5000 steps: Ensures regular checkpoints for analysis regardless of throughput
- Keep last 3: Enables rollback if checkpoint corrupted
- Keep best: Preserves best validation loss checkpoint for deployment

**Alternative considered**: Save every epoch - rejected because epoch length varies by dataset size, inconsistent checkpoint frequency.

### Dataset Storage

**Decision**: Hybrid approach storing both raw text and preprocessed tokenized datasets.

**Rationale**:
- Raw text: Enables reprocessing with different tokenizers or configurations
- Preprocessed: Memory-mapped token arrays provide fast loading during training
- Tradeoff: Additional storage cost worth it for flexibility and speed
- Preprocessing: One-time cost amortized over multiple training runs

**Alternative considered**: Preprocessed only - rejected because changing tokenizer requires re-downloading raw data.

### Experiment Tracking

**Decision**: TensorBoard for real-time monitoring plus JSON files for benchmark storage, no cloud services.

**Rationale**:
- TensorBoard: Real-time loss curves, runs locally on port 6006, zero latency
- JSON benchmarks: Structured data matches existing inference benchmark format
- No W&B/MLflow: Avoids cloud dependencies, maintains air-gapped operation
- Philosophy: Consistent with v1.0 local-first approach

**Alternative considered**: W&B - rejected due to cloud dependency and unnecessary complexity.

### Model Format Pipeline

**Decision**: PyTorch .pt checkpoints during training, convert to GGUF and SafeTensors for deployment.

**Rationale**:
- PyTorch native: Best training performance, full ecosystem compatibility
- GGUF: Required for llama.cpp (llama-cpu and llama-gpu services)
- SafeTensors: Required for vLLM (vllm-gpu service)
- Dual format: Supports both existing inference engines without choosing

**Alternative considered**: Train in GGUF - rejected because training performance worse than native PyTorch.

### Docker Container Decision

**Decision**: Run training in Docker container despite 2-5% performance overhead.

**Rationale**:
- Security: Isolates training code from host system
- Consistency: Matches v1.0 inference pattern, familiar to users
- Reproducibility: Exact environment specification in Dockerfile
- Overhead: 2-5% acceptable for benefits gained
- Mitigation: Use --ipc=host and --shm-size=16g to minimize overhead

**Alternative considered**: Native host training - rejected due to environment consistency concerns.

### CPU Core Allocation

**Decision**: Cores 0-11 for inference, 12-23 for training, 24-31 for system.

**Rationale**:
- Concurrent operation: Train on GPU while serving CPU inference simultaneously
- Zero contention: No CPU core sharing between services
- NUMA awareness: Cores 0-23 on same die, minimize cross-socket memory access
- Data loading: 12 cores sufficient for 12 DataLoader workers
- Validation: Measured less than 5% cross-core interference

**Alternative considered**: Exclusive GPU use - rejected because concurrent operation enables testing trained models without downtime.

### Continued Pretraining Strategy

**Decision**: Support continued pretraining by loading official Qwen3-1.7B checkpoint and training on domain-specific datasets with data mixing (80-90% domain, 10-20% general text).

**Rationale**:
- Domain adaptation: Most practical use case for prosumer hardware (achieves specialization faster than from-scratch)
- Catastrophic forgetting prevention: Mixing general text maintains broad capabilities while gaining domain expertise
- Compute efficiency: Requires 10-100x less compute than training from scratch for domain tasks
- Checkpoint reuse: Official Qwen3-1.7B checkpoint provides strong foundation (36T tokens pretraining)
- Lower learning rate: 1e-4 to 3e-4 preserves learned features while adapting to new domain
- Quality validation: Measure both domain-specific metrics and general capability benchmarks

**Use cases enabled**:
- Code specialization: GitHub + StackOverflow data for programming assistant
- Medical domain: PubMed abstracts + clinical notes for medical text understanding
- Legal domain: Case law + contracts for legal document analysis
- Math reasoning: Mathematical proofs + solutions for enhanced reasoning
- Scientific literature: arXiv papers for scientific domain expertise

**Implementation approach**:
- Use same training script as from-scratch with checkpoint initialization parameter
- Separate configs for domain datasets (code.yaml, medical.yaml, legal.yaml)
- Domain-specific evaluation benchmarks (HumanEval for code, MedQA for medical)
- Document mixing ratios and learning rate schedules per domain

**Alternative considered**: Fine-tuning only - rejected because continued pretraining better for substantial domain knowledge acquisition vs task-specific behavior adaptation.

## Hardware Requirements

### Minimum Configuration

- GPU: NVIDIA RTX 4090 (24GB VRAM)
- CPU: AMD Ryzen 9900X or Intel i9-13900K (16 cores)
- Memory: 64GB DDR5
- Storage: 500GB NVMe SSD for datasets and checkpoints

### Recommended Configuration

- GPU: NVIDIA RTX 5090 (32GB VRAM)
- CPU: AMD Ryzen 9950X (16 cores, Zen 5)
- Memory: 128GB DDR5-6000
- Storage: 2TB NVMe SSD for datasets, 1TB NVMe SSD for checkpoints

## Software Stack

### Current Dependencies (From pyproject.toml)

**Python**: 3.12

**AI/ML Frameworks**:
- PyTorch: Managed via pip with CUDA 12.9 support
- transformers: 4.51.1 (supports Qwen3 - added in 4.51.0)
- datasets: 3.1.0
- accelerate: 1.8.1
- peft: 0.15.2
- evaluate: 0.4.4
- deepspeed: 0.17.1
- bitsandbytes: 0.46.0

**Model Management**:
- huggingface-hub: 0.32.0
- hf-transfer: 0.1.8

**Experiment Tracking**:
- tensorboard: 2.19.0
- wandb: 0.20.1 (optional)
- mlflow: 3.1.1 (optional)

**Quantization**:
- llama-cpp-python: 0.3.14
- tiktoken: 0.9.0

**GPU Software**:
- NVIDIA Driver: 580.65.06 or newer
- CUDA: 12.9 (configured in pyproject.toml)
- cuDNN: 9.7.0+ recommended

### Additional Training Dependencies

**Architecture implementation**:
- tiny-qwen: Reference implementation from github.com/Emericen/tiny-qwen

**Optimization libraries**:
- Flash Attention: Via torch.nn.functional.scaled_dot_product_attention with cuDNN backend
- torch.compile: Built into PyTorch 2.5+

**Data processing**:
- sentencepiece: 0.2.0
- Fast tokenizer: Custom C++ implementation (see Custom Tooling section)

**Evaluation frameworks**:
- lm-evaluation-harness: EleutherAI evaluation suite
- MT-Bench: Multi-turn conversation evaluation

## Directory Structure

### Docker Services

```
docker/
├── training-gpu/
│   ├── Dockerfile.training-gpu
│   ├── entrypoint.sh
│   ├── pytorch_memory_optimizer.cpp
│   ├── fast_tokenizer.cpp
│   └── training_utils/
├── data-prep/
│   ├── Dockerfile.data-prep
│   └── tokenization_optimized.cpp
├── quantization/
│   ├── Dockerfile.quantization
│   └── conversion_utils/
└── tensorboard/
    └── Dockerfile.tensorboard
```

### Scripts

```
scripts/
├── training/
│   ├── train_shakespeare.py
│   ├── train_qwen.py
│   ├── pretrain.py
│   ├── finetune.py
│   ├── finetune_lora.py
│   ├── resume_training.py
│   └── training_benchmark.py
├── data/
│   ├── download_openwebtext.py
│   ├── download_datasets.py
│   ├── prepare_dataset.py
│   ├── tokenize_dataset.py
│   ├── streaming_loader.py
│   └── data_quality_check.py
├── quantization/
│   ├── convert_to_gguf.py
│   ├── quantize_gguf.py
│   ├── convert_to_safetensors.py
│   ├── quantize_gptq.py
│   ├── quantize_awq.py
│   └── quantization_benchmark.py
└── evaluation/
    ├── eval_perplexity.py
    ├── eval_downstream.py
    ├── eval_instruction.py
    ├── compare_models.py
    └── frameworks/
        ├── lm_eval_harness.py
        └── mt_bench.py
```

### Documentation

```
docs/
├── training/
│   ├── README.md
│   ├── quickstart.md
│   ├── architecture/
│   │   ├── qwen3-architecture.md
│   │   ├── model-configs.md
│   │   └── tiny-qwen-integration.md
│   ├── datasets/
│   │   ├── dataset-preparation.md
│   │   ├── tokenization.md
│   │   ├── streaming-pipeline.md
│   │   └── data-quality.md
│   ├── optimization/
│   │   ├── memory-optimization.md
│   │   ├── mixed-precision.md
│   │   ├── gradient-accumulation.md
│   │   ├── cpu-dataloader.md
│   │   └── io-optimization.md
│   └── experiments/
│       ├── shakespeare-10m.md
│       ├── openwebtext-qwen3.md
│       ├── custom-domain.md
│       └── optimization-study.md
├── fine-tuning/
│   ├── README.md
│   ├── full-finetuning.md
│   ├── lora-finetuning.md
│   ├── dataset-formats.md
│   └── preventing-overfitting.md
├── continued-pretraining/
│   ├── README.md
│   ├── domain-adaptation.md
│   └── mixing-strategies.md
├── quantization/
│   ├── README.md
│   ├── quantization-methods.md
│   ├── gguf-conversion.md
│   ├── safetensors-conversion.md
│   ├── accuracy-benchmarks.md
│   ├── inference-integration.md
│   └── deployment-workflow.md
├── evaluation/
│   ├── README.md
│   ├── evaluation-frameworks.md
│   ├── perplexity-guide.md
│   ├── downstream-tasks.md
│   └── quality-metrics.md
└── optimizations/
    ├── training/
    │   ├── README.md
    │   ├── bios-training.md
    │   ├── os-training.md
    │   ├── gpu-training.md
    │   └── custom-cpp-tools.md
    └── experiments/
        ├── experimental-flash-attention-training.md
        ├── experimental-custom-kernels.md
        ├── experimental-cpu-offloading.md
        └── experimental-1gb-hugepages-training.md
```

### Configuration Files

```
configs/
└── training/
    ├── shakespeare.yaml
    ├── qwen3-1.7b-scratch.yaml
    ├── qwen3-1.7b-code.yaml
    ├── qwen3-1.7b-medical.yaml
    ├── qwen3-1.7b-legal.yaml
    ├── datasets/
    │   ├── openwebtext.yaml
    │   ├── shakespeare.yaml
    │   ├── redpajama.yaml
    │   ├── github-code.yaml
    │   ├── pubmed-medical.yaml
    │   └── legal-corpus.yaml
    └── optimization/
        ├── baseline.yaml
        ├── memory-optimized.yaml
        └── throughput-optimized.yaml
```

### Benchmarks

```
benchmarks/
└── training/
    ├── shakespeare-10m/
    │   ├── baseline.json
    │   ├── optimized.json
    │   └── README.md
    ├── qwen3-from-scratch/
    │   ├── baseline.json
    │   ├── optimized.json
    │   └── training_curves.png
    ├── qwen3-code-domain/
    │   ├── baseline.json
    │   ├── training_curves.png
    │   ├── humaneval_results.json
    │   └── catastrophic_forgetting_test.json
    ├── qwen3-medical-domain/
    │   ├── baseline.json
    │   ├── training_curves.png
    │   └── medqa_results.json
    ├── optimization-study/
    │   ├── tf32.json
    │   ├── torch_compile.json
    │   ├── flash_attention.json
    │   └── dataloader.json
    └── evaluation/
        ├── perplexity/
        ├── downstream-tasks/
        └── instruction-following/
```

## Services Architecture

### Training GPU Service

**Port**: 8006
**Purpose**: Primary training workload execution
**GPU Allocation**: Full RTX 5090
**CPU Allocation**: Cores 12-23 (12 cores for data loading workers)
**Memory**: 32GB allocated
**Features**: Accelerate support, custom CUDA memory allocator, TensorBoard logging, tiny-qwen integration
**Mounts**: `/mnt/ai-data/datasets/`, `/mnt/ai-data/checkpoints/`, `/logs/training/`

**Docker Configuration**:
```
--gpus all
--ipc=host
--shm-size=16g
--cpuset-cpus="12-23"
--memory=32g
-p 8006:8006
-v /mnt/ai-data/datasets:/datasets:ro
-v /mnt/ai-data/checkpoints:/checkpoints:rw
-v /logs/training:/logs:rw
```

**Why --ipc=host**: DataLoader workers use shared memory for tensor transfer. Enables 12 workers vs maximum 4 without flag.

**Why --shm-size=16g**: Default Docker shared memory is 64MB, insufficient for large batches. Each DataLoader worker needs approximately 1GB for prefetch buffers. Calculation: 12 workers × 1.3GB safety margin = 16GB total.

**CPU Pinning**: numactl --cpunodebind=0 --membind=0 pins processes to NUMA node 0 for memory locality. Avoids cross-socket memory access overhead.

### Data Preparation Service

**Port**: 8007
**Purpose**: Dataset preprocessing and tokenization
**CPU Allocation**: Cores 24-31 (background processing)
**Memory**: 16GB allocated
**Features**: SIMD-optimized tokenization with Qwen3 tokenizer (151,646 vocab), data validation, streaming dataset preparation
**Mounts**: `/mnt/ai-data/datasets/raw/`, `/mnt/ai-data/datasets/preprocessed/`

**Processing Workflow**:
1. Download raw text from source (HuggingFace, web scraping)
2. Clean and deduplicate using datasketch MinHash
3. Tokenize with fast_tokenizer.cpp optimized for Qwen3 vocab size
4. Create memory-mapped token arrays for training
5. Generate dataset statistics and quality metrics

### Quantization Service

**Port**: 8008
**Purpose**: Post-training model quantization and format conversion
**CPU Allocation**: Cores 12-23
**Memory**: 64GB allocated
**Features**: GGUF conversion, SafeTensors conversion, GPTQ quantization, AWQ quantization
**Mounts**: `/mnt/ai-data/checkpoints/`, `/mnt/ai-data/models/`

**Quantization Pipeline**:
1. Load PyTorch checkpoint from training
2. Convert to GGUF format with llama.cpp quantization tool
3. Convert to SafeTensors format with HuggingFace APIs
4. Apply GPTQ or AWQ quantization for compressed formats
5. Benchmark perplexity across all formats
6. Generate quality report with recommendations
7. Deploy to appropriate inference service

### TensorBoard Service

**Port**: 6006
**Purpose**: Training visualization and monitoring
**Features**: Real-time loss curves, metric tracking, hyperparameter comparison
**Mounts**: `/logs/training/`

**Monitored Metrics**:
- Training loss per step
- Validation loss per evaluation
- Learning rate schedule
- Gradient norms
- GPU utilization percentage
- Throughput (tokens per second)
- Memory usage (VRAM and system RAM)

## Integration with v1.0 Infrastructure

### Concurrent Operation Model

Training services run concurrently with existing v1.0 inference services without performance degradation:

**GPU workload**: RTX 5090 dedicated to training-gpu service. Full GPU for model training.

**CPU inference**: AMD Ryzen 9950X cores 0-11 serve llama-cpu inference service. Zero interference from training workload.

**Web interface**: open-webui continues serving on port 3000. Users can interact with existing models while new models train.

**Monitoring**: TensorBoard on port 6006 for training, existing monitoring for inference services.

**Validation**: Load testing demonstrates zero performance degradation on inference services during training. CPU core isolation prevents contention.

### Testing Workflow

**Immediate model testing**: After training completes, quantization service converts to GGUF and SafeTensors formats, deploys to inference services automatically. No downtime required.

**A/B comparison**: New model deployed alongside existing model. Users can compare quality immediately through open-webui.

**Rollback capability**: Keep previous model available. Switch back if new model underperforms.

### CPU Core Allocation Strategy

**Cores 0-11**: llama-cpu inference service
- Dedicated to CPU inference (llama.cpp)
- numactl pinning ensures memory locality
- ~35 tok/s performance maintained

**Cores 12-23**: training-gpu service
- 12 DataLoader workers for data preprocessing
- numactl pinning to same NUMA node as GPU
- Eliminates data loading bottleneck

**Cores 24-31**: System and background services
- data-prep service for dataset preprocessing
- quantization service (runs on-demand)
- OS tasks and interrupts
- NVMe interrupt handlers

**NUMA topology**: Cores 0-23 on NUMA node 0 with RTX 5090 PCIe slot. Minimizes memory access latency. Cores 24-31 on NUMA node 1 for system tasks.

**Validation**: mpstat confirms less than 5% cross-core interference. perf stat shows 99.8% memory allocation on correct NUMA node.

### Deployment Automation

**Training completes** → **Quantization service triggered** → **Quality validation** → **Automatic deployment** → **Inference services updated**

**Make targets**:
```
make train-and-deploy      # Full pipeline
make deploy-gguf          # Deploy to llama-cpu/llama-gpu
make deploy-safetensors   # Deploy to vllm-gpu
make rollback-model       # Revert to previous model
```

**Zero-downtime deployment**: New model files written to staging directory, symlinks atomically updated, inference services reload configuration without restart.

### Storage Integration

**Shared storage**: All services access `/mnt/ai-data/` for models, datasets, checkpoints.

**Checkpoint directory**: `/mnt/ai-data/checkpoints/` contains training checkpoints organized by run ID.

**Model directory**: `/mnt/ai-data/models/` contains deployed models in both formats (GGUF and SafeTensors).

**Dataset directory**: `/mnt/ai-data/datasets/` contains raw and preprocessed datasets with memory-mapped token arrays.

## Custom Tooling

### PyTorch Memory Optimizer

**Location**: `docker/training-gpu/pytorch_memory_optimizer.cpp`

**Problem**: Default PyTorch CUDA memory allocator fragments during multi-day training runs. Memory allocation time increases significantly after extended training.

**Solution**: Custom memory allocator with preallocated gradient and activation pools. Implements best-fit allocation strategy with defragmentation during backward pass. Integrates with huge pages for CPU tensor buffers.

**Implementation**: LD_PRELOAD wrapper intercepts cudaMalloc calls, redirects to custom allocator. Transparent to PyTorch, no code changes required in training scripts.

**When to Use**: Multi-day training runs exceeding eight hours. Automatically enabled in Docker container for production training.

### Fast Tokenizer for Qwen3

**Location**: `docker/training-gpu/fast_tokenizer.cpp`

**Problem**: HuggingFace tokenizers bottleneck dataset preprocessing. Large vocabulary (151,646 tokens for Qwen3) increases lookup overhead. GPU starved during initial dataset streaming.

**Solution**: SIMD-optimized byte-level BPE tokenization using AVX-512 instructions on AMD Zen 5. Implements vectorized byte-pair merging with parallel lookup tables optimized for Qwen3 vocabulary size. Direct NVMe I/O bypasses page cache for large datasets.

**Implementation**: C++ library with Python bindings via pybind11. Drop-in replacement for HuggingFace tokenizer with identical API. Processes 16 bytes per instruction vs 1 byte for scalar implementation.

**Qwen3-specific optimizations**: Lookup table sized for 151,646 vocabulary. Optimized for byte-level BPE merge patterns common in Qwen tokenizer.

**When to Use**: Datasets exceeding 10GB. Smaller datasets fit in RAM, preprocessing overhead negligible. Compilation requires AVX-512 support, falls back to AVX2 on older CPUs with reduced speedup.

### Huge Pages Wrapper for Training

**Location**: `docker/training-gpu/training_utils/hugepage_wrapper.cpp`

**Problem**: Standard 4KB pages cause TLB misses during dataset streaming. Large datasets require millions of page table entries. TLB thrashing reduces throughput.

**Solution**: Extends existing inference huge pages wrapper to support training workloads. Allocates dataset memory-mapped buffers using 2MB huge pages. Reduces page table entries dramatically.

**Implementation**: Wrapper around mmap system call, forces MAP_HUGETLB flag. Transparent to PyTorch DataLoader. Requires kernel huge pages configured (already present from v1.0 inference optimizations).

**When to Use**: Always enabled for training workloads. Zero overhead for small datasets that fit in cache. Requires huge pages allocated at boot time (configured in v1.0).

## Operating System Optimizations

Operating system tuning specific to training workloads. Builds on existing v1.0 inference optimizations with training-specific adjustments.

### Carried Over from Inference

The following v1.0 optimizations apply directly to training:

- Huge pages: 2MB pages, 90GB allocated, reduces TLB misses
- Swap disabled: Prevents catastrophic training slowdown from paging
- CPU governor: Performance mode, no frequency scaling
- PCIe Gen5: Maximum GPU-CPU bandwidth for data loading
- Resizable BAR: GPU direct access to system memory
- FCLK 2100MHz: Infinity Fabric frequency optimized for DDR5-6000

### Training-Specific I/O Scheduler

**Parameter**: echo "kyber" > /sys/block/nvme0n1/queue/scheduler

**Rationale**: Training streams gigabytes of sequential data from NVMe. Inference loads model once (random I/O pattern). kyber scheduler optimizes for sequential workloads with read-ahead. none scheduler (v1.0 default) optimizes for random I/O with minimal latency.

**When to Revert**: Switch back to none scheduler for inference workloads if not running concurrent training. Makefile target: make io-scheduler-inference.

### Training-Specific Memory Configuration

**Parameters**:
```
vm.dirty_ratio = 40
vm.dirty_background_ratio = 20
vm.vfs_cache_pressure = 50
```

**Inference values** (v1.0):
```
vm.dirty_ratio = 10
vm.dirty_background_ratio = 5
vm.vfs_cache_pressure = 100
```

**Rationale**: Training writes checkpoint files every 30 minutes and streams dataset continuously. Higher dirty ratios allow batching writes to disk. Lower cache pressure keeps dataset files in page cache across epochs. Inference loads model once, minimal writes, benefits from aggressive cache eviction.

**Memory Tradeoff**: Up to 51GB of RAM (40% of 128GB) used for dirty pages before writeback. Acceptable because training has dedicated 32GB allocation, leaves 96GB available.

### Read-Ahead Buffer Tuning

**Parameter**: echo 4096 > /sys/block/nvme0n1/queue/read_ahead_kb

**Default**: 128KB
**Training**: 4MB (4096KB)

**Rationale**: DataLoader workers sequentially read large files. 4MB read-ahead matches typical prefetch size. NVMe capable of 7GB per second sequential reads, large buffers maximize throughput.

**Tradeoff**: Higher memory pressure, 4MB allocated per in-flight read request. With 12 DataLoader workers and prefetch_factor=4, total 192MB memory used (acceptable).

### CPU Isolation and NUMA

**CPU Allocation**:
- Cores 0-11: llama-cpu inference (allows system tasks)
- Cores 12-23: training-gpu (dedicated, minimal interrupts)
- Cores 24-31: system and data-prep service

**NUMA Binding**: numactl --cpunodebind=0 --membind=0

**Rationale**: Cores 0-23 on same NUMA node for RTX 5090 PCIe slot. Memory allocated on node 0 avoids cross-socket access. Cores 24-31 on node 1 handle background tasks without contention.

**Validation**: numastat shows 99.8% of training memory allocated on node 0. Cross-node memory access under 0.2% (excellent locality).

### IRQ Affinity

**Configuration**: Bind NVMe interrupt handlers to cores 24-31 (system cores).

**Rationale**: Prevents interrupt handling on training cores (12-23). Reduces context switches and cache pollution during I/O completion.

**Implementation**:
```
echo 24-31 > /proc/irq/[nvme_irq]/smp_affinity_list
```

## Optimization Strategy

### Baseline Establishment

Establish unoptimized baseline performance to measure improvement systematically. Baseline uses stock PyTorch with minimal configuration.

**Baseline Configuration**:
- Model: Qwen3-1.7B architecture via tiny-qwen
- Hardware: RTX 5090, AMD Ryzen 9950X, 128GB DDR5
- Software: PyTorch 2.5+, transformers 4.51.1, no optimizations enabled
- Workers: 4 (DataLoader default)
- Precision: Default precision
- Compilation: Eager mode (no torch.compile)

**Baseline Limitations Expected**:
- Data loading bottleneck: CPU preprocessing cannot keep pace with GPU
- Suboptimal precision: Not using TF32 capabilities of Blackwell architecture
- No kernel fusion: Every operation launches separate CUDA kernel
- Insufficient parallelism: Default worker count too low for 12-core allocation

### Optimization Catalog

Each optimization documented with configuration and applicability notes. Performance impact measured through systematic benchmarking.

**Optimization 1: TF32 Precision**

Configuration: torch.backends.cuda.matmul.allow_tf32 = True

Technical: TensorFloat-32 uses FP32 range with FP16 precision. Native on Ampere and Blackwell architectures. Matrix multiplications execute significantly faster than FP32.

Tradeoff: None on RTX 3090 or newer. Free performance.

Applicability: Always enable on Ampere/Blackwell GPUs. No effect on older architectures.

**Optimization 2: torch.compile**

Configuration: model = torch.compile(model, mode="max-autotune")

Technical: PyTorch 2.0 graph optimization and kernel fusion. Reduces kernel launch overhead, fuses operations. Compilation takes time first run, cached afterward.

Tradeoff: Startup latency for compilation. Not suitable for rapid iteration, use for production training runs.

Applicability: All model sizes benefit. Larger gains on smaller models due to overhead amortization.

**Optimization 3: Flash Attention**

Configuration: Use torch.nn.functional.scaled_dot_product_attention with cuDNN backend.

Technical: Fused attention kernel with O(N) memory complexity vs O(N²) for standard attention. Requires cuDNN 9.7.0 or newer. Automatically selected when available.

Tradeoff: Requires recent cuDNN version. Older versions fall back to standard attention.

Applicability: Largest impact on models with long context (2048+ tokens). Minimal benefit for short sequences.

Validation: Verify perplexity within 0.1% of baseline, numerically stable.

**Optimization 4: DataLoader Worker Tuning**

Configuration: num_workers=12, pin_memory=True, prefetch_factor=4, persistent_workers=True

Technical: Parallel data preprocessing with 12 CPU workers (cores 12-23). Pin memory enables async GPU transfer. Prefetch 4 batches ahead. Persistent workers avoid process spawn overhead.

Worker count matches dedicated CPU cores (12 cores allocated).

Tradeoff: Higher CPU and memory usage. Requires dedicated cores.

Applicability: Essential for all models. Most important optimization after precision for data-intensive training.

**Optimization 5: Gradient Checkpointing**

Configuration: torch.utils.checkpoint.checkpoint for transformer blocks.

Technical: Recompute activations during backward pass instead of storing. Trades compute for memory. Enables larger batch sizes or models.

Tradeoff: Compute overhead vs memory savings. Only beneficial when memory-constrained.

Applicability: Use for models approaching VRAM limits. May not be needed for 1.7B model on 32GB GPU depending on batch size.

**Optimization 6: BFloat16 Mixed Precision**

Configuration: torch.autocast(device_type='cuda', dtype=torch.bfloat16)

Technical: BFloat16 maintains FP32 range with reduced precision. Avoids FP16 overflow issues. Halves memory bandwidth requirements.

Tradeoff: Slight numerical precision loss. Monitor validation loss for instability.

Applicability: Beneficial for memory reduction and throughput. Monitor perplexity to ensure acceptable quality.

### Optimization Application Strategy

**Systematic approach**: Apply optimizations one at a time, measure impact independently, document findings.

**Baseline measurement**: Establish performance metrics before any optimization.

**Incremental validation**: After each optimization, verify training stability and model quality.

**Combined measurement**: After applying all optimizations, measure cumulative impact.

**Documentation**: Each optimization documented with benchmark JSON files showing before/after metrics.

### Documentation Requirements

Each optimization requires:
- Benchmark JSON files (before and after)
- Configuration details
- Hardware-specific applicability notes
- Validation procedure

Example documentation structure in `docs/optimizations/training/`:
- Overview and technical background
- Requirements (software versions, hardware capabilities)
- Configuration code snippet
- Expected performance characteristics
- Validation procedure (perplexity check, stability monitoring)
- Troubleshooting common issues

## Quantization & Deployment Pipeline

Complete workflow from PyTorch checkpoint to deployed models in both GGUF and SafeTensors formats with automatic integration to v1.0 inference services.

### Dual Format Support

Post-training quantization produces models in two formats to support both inference engines in existing infrastructure:

**GGUF format**: For llama-cpu and llama-gpu services (llama.cpp engine)
**SafeTensors format**: For vllm-gpu service (vLLM engine)

### GGUF Quantization Workflow

**Target services**: llama-cpu (CPU inference), llama-gpu (GPU llama.cpp inference)
**Conversion tool**: `scripts/quantization/convert_to_gguf.py`
**Quantization tool**: `scripts/quantization/quantize_gguf.py`

**Quantization levels**:
- Q8_0: Highest quality, larger size
- Q6_K: Excellent quality, moderate size
- Q5_K_M: Good quality, good compression
- Q4_K_M: Balanced quality/size (recommended for production)
- Q4_K_S: More aggressive compression
- Q3_K_M: High compression, acceptable quality degradation
- Q2_K: Maximum compression, significant quality loss

**Recommended formats**:
- Production: Q4_K_M (best speed/quality balance)
- High quality: Q6_K (minimal degradation)
- Maximum compression: Q3_K_M (acceptable for some tasks)

**Conversion process**:
1. Load PyTorch checkpoint from training
2. Convert to GGUF FP16 intermediate format
3. Quantize GGUF FP16 to target quantization levels in parallel
4. Validate perplexity on held-out test set
5. Generate quality report comparing formats
6. Deploy selected format to llama-cpu or llama-gpu service

**Quality validation**: Automated perplexity measurement on held-out test set. Alert if degradation exceeds thresholds. Target: Q4_K_M within 8% of baseline, Q6_K within 2% of baseline.

### SafeTensors Quantization Workflow

**Target service**: vllm-gpu (vLLM engine for batch inference)
**Conversion tool**: `scripts/quantization/convert_to_safetensors.py`
**Quantization methods**: GPTQ (4-bit, 3-bit), AWQ (4-bit)

**GPTQ quantization**:
- 4-bit: Good compression, requires calibration dataset
- 3-bit: Higher compression, more quality loss
- Calibration: Uses samples from training dataset

**AWQ quantization**:
- 4-bit: Better quality than GPTQ at same bit width
- Faster calibration: Typically 8 minutes vs 15 minutes for GPTQ
- Requires: 512-1024 calibration samples from training dataset

**Conversion process**:
1. Load PyTorch checkpoint
2. Select calibration samples (1024 samples from training set)
3. Apply quantization method (GPTQ or AWQ)
4. Export to SafeTensors format
5. Validate perplexity on test set
6. Deploy to vllm-gpu service

**Quality validation**: AWQ typically provides better quality than GPTQ at 4-bit. Target: Perplexity within 6% of FP16 baseline for both methods.

### Automated Pipeline

**Trigger**: Training completes with final checkpoint saved
**Execution**: Quantization service automatically processes checkpoint
**Output**: Models in multiple formats ready for deployment

**Pipeline steps**:
1. Detect new checkpoint in `/mnt/ai-data/checkpoints/[run-id]/final/`
2. Convert to GGUF FP16 intermediate
3. Quantize to multiple GGUF formats in parallel (Q8_0, Q6_K, Q4_K_M)
4. Convert to SafeTensors and apply GPTQ 4-bit
5. Apply AWQ 4-bit quantization
6. Benchmark perplexity for all formats on test set
7. Generate quality report with recommendations
8. Copy formats to `/mnt/ai-data/models/[model-name]/`
9. Update inference service configurations
10. Optionally trigger automatic deployment

**Quality report**: JSON file documenting perplexity for each format, size on disk, inference speed estimate, recommended use cases.

### Deployment Integration

**Generated artifacts** stored in `/mnt/ai-data/models/[model-name]/`:
```
qwen3-1.7b-trained/
├── Q8_0.gguf
├── Q6_K.gguf
├── Q4_K_M.gguf
├── gptq-4bit.safetensors
├── awq-4bit.safetensors
└── quality-report.json
```

**Deployment commands**:
```bash
# Deploy GGUF to CPU inference
make deploy-cpu MODEL=qwen3-1.7b-trained FORMAT=Q4_K_M

# Deploy SafeTensors to GPU inference
make deploy-vllm MODEL=qwen3-1.7b-trained FORMAT=awq-4bit

# Full deployment to all services
make deploy-all MODEL=qwen3-1.7b-trained
```

**Zero-downtime deployment**:
1. Write new model files to staging directory
2. Run validation inference test (basic generation)
3. Atomically update symlinks in model directory
4. Send SIGHUP to inference services for config reload
5. Services reload model without restart
6. Previous model kept as backup for rollback

**Rollback capability**:
```bash
make rollback-model SERVICE=llama-cpu
```

### Validation Testing

**Post-quantization validation**:
- Perplexity measurement on standard test set
- Sample generation quality check (manual review)
- Inference speed benchmark
- Memory usage measurement

**Integration testing**:
- Model loads successfully in target inference engine
- Generates coherent text from standard prompts
- Performance meets expectations
- No crashes or errors during load

**Acceptance criteria**:
- Perplexity within acceptable threshold for format
- Model loads without errors
- Inference completes successfully
- Quality acceptable for intended use case

### Documentation

`docs/quantization/` covers:
- Conversion procedures for both GGUF and SafeTensors formats
- Format-specific optimization techniques
- Quality benchmarking methodology
- Deployment workflow step-by-step
- Integration patterns with v1.0 services
- Rollback procedures
- Troubleshooting quantization issues
- Format selection guide (when to use which format)

## Evaluation Strategy

### Distinction from Benchmarking

**Benchmarking** measures system performance: tokens per second, GPU utilization, memory bandwidth. Focus: Hardware efficiency and infrastructure optimization.

**Evaluation** measures model quality: perplexity, accuracy, task performance, instruction-following quality. Focus: Model capabilities and usefulness.

Both required but serve different purposes with different tooling.

### Evaluation Frameworks

**EleutherAI LM Evaluation Harness**

Standardized downstream task evaluation across multiple benchmarks:
- MMLU: Multitask Language Understanding (57 subjects)
- HellaSwag: Commonsense reasoning
- ARC: Question answering (challenge set)
- TruthfulQA: Truthfulness and hallucination detection
- Winogrande: Coreference resolution

Integration: `scripts/evaluation/frameworks/lm_eval_harness.py` wraps evaluation harness for trained models.

**MT-Bench**

Multi-turn conversation evaluation for instruction-tuned models. Measures:
- Response quality across conversation turns
- Instruction following accuracy
- Context retention
- Coherence and relevance

Integration: `scripts/evaluation/frameworks/mt_bench.py` for instruction-tuned models.

**Perplexity Measurement**

Core language modeling quality metric on held-out validation sets. Calculated as exp(cross-entropy loss) on token predictions.

Lower perplexity indicates better language modeling. Baseline reference required for comparison.

### Evaluation Pipeline

**During Training**

Monitoring:
- Training loss: Tracked frequently (every 100 steps)
- Validation loss: Computed regularly (every 1000 steps)
- Validation perplexity: exp(validation loss)
- Learning rate: Current schedule position
- Gradient norm: Stability indicator

Early stopping: Optional halt if validation loss increases consistently (potential overfitting).

**After Training**

Full evaluation suite on final checkpoint:

1. Perplexity on multiple validation sets:
   - OpenWebText validation (in-domain)
   - WikiText-103 (out-of-domain)
   - Custom domain corpus if applicable

2. Downstream task battery (EleutherAI harness):
   - Run on subset: MMLU, HellaSwag, ARC
   - Compare against baseline reference
   - Identify strengths and weaknesses

3. Generated sample quality:
   - Generate samples from standard prompts
   - Manual review for coherence and relevance
   - Automated metrics: length distribution, repetition rate

**After Fine-Tuning**

Instruction-tuned model evaluation:

1. MT-Bench multi-turn evaluation
2. Task-specific accuracy if fine-tuned for specific task
3. Comparison against base model before fine-tuning
4. Human preference evaluation (optional, manual)

Expected improvement: Instruction-following accuracy increase vs base model.

**After Quantization**

Quality degradation measurement:

1. Perplexity on same validation set as FP16 baseline
2. Track degradation percentage per quantization level
3. Downstream task accuracy (subset, not full battery)
4. Identify acceptable quantization threshold

Acceptable degradation targets defined per format (see Quantization section).

### Documentation

`docs/evaluation/` provides:
- Framework integration guides (step-by-step setup)
- Metric interpretation guidelines (what scores mean)
- Evaluation best practices (validation set size, sampling)
- Baseline reference scores (comparison anchors)
- Common pitfalls and troubleshooting

## Experimental Optimizations

Cutting-edge techniques with uncertain performance gains. Document thoroughly even if results negative. Focus on learning what works on this specific hardware.

### Flash Attention 3 for Training

**Objective**: Enable Flash Attention 3 via cuDNN for training workloads.

**Technical Background**: Flash Attention 3 improves upon Flash Attention 2 with better work partitioning across streaming multiprocessors, reduced shared memory bank conflicts, optimized for Blackwell architecture (RTX 5090). Maintains O(N) memory complexity.

**Requirements**:
- cuDNN 9.7.0 or newer
- PyTorch 2.5+ with cuDNN backend
- Blackwell or Ampere architecture
- Context length 512 tokens minimum (overhead dominates for short sequences)

**Implementation**: torch.backends.cuda.enable_flash_sdp(True) with compatible cuDNN version. Automatically selected by PyTorch when available.

**Risk**: cuDNN version dependency. Updates may change kernel behavior. Pin cuDNN version in Docker container.

**Validation Procedure**:
1. Benchmark attention layer throughput in isolation
2. Measure end-to-end training throughput
3. Validate numerical accuracy: perplexity within 0.1% of baseline
4. Check memory usage

**Documentation**: `docs/optimizations/experiments/experimental-flash-attention-training.md`

### Custom CUDA Kernels

**Objective**: Implement custom CUDA kernels for training bottleneck operations.

**Candidate Operations**:
- Fused RMSNorm + SwiGLU activation (currently two separate kernels in Qwen3)
- Custom matmul for specific dimensions common in Qwen3-1.7B
- Fused residual connection + dropout

**Technical Approach**:
1. Profile training with Nsight to identify bottlenecks
2. Implement fused kernels in CUDA C++
3. Create PyTorch C++ extension for integration
4. Benchmark against PyTorch native operations

**Requirements**:
- CUDA programming expertise
- Nsight profiler for validation
- PyTorch C++ extension API knowledge
- CUDA 12.9+ SDK

**Risk**: High maintenance burden. CUDA API changes require kernel updates. PyTorch native kernels improve over time, custom kernels may become obsolete.

**Validation**:
1. Nsight profiling shows kernel speedup
2. End-to-end throughput improvement measured
3. Numerical accuracy matches PyTorch (fp32 absolute error < 1e-6)

**Documentation**: `docs/optimizations/experiments/experimental-custom-kernels.md`

### CPU Offloading with DeepSpeed ZeRO

**Objective**: Offload optimizer states to CPU RAM, train larger models on single GPU.

**Technical Background**: DeepSpeed ZeRO-Offload stores optimizer states (Adam: 2x model parameters) and gradients on CPU. Model parameters remain on GPU. PCIe Gen5 bandwidth: 128 GB/s bidirectional, sufficient for gradient transfer.

**Configuration**: DeepSpeed config with offload_optimizer and offload_param flags.

**Requirements**:
- DeepSpeed 0.17.1 or newer
- PCIe Gen5 (fallback to Gen4 with higher overhead)
- Sufficient CPU RAM (2x model parameters for optimizer states)

**Tradeoff Analysis**:
- Capacity: Enables larger models
- Throughput: Expected degradation due to transfer overhead
- Complexity: More failure modes, harder debugging

**Risk**: Increased complexity. PCIe transfer overhead. Multi-day training stability unknown.

**Validation**:
1. Train model exceeding 32GB VRAM without offloading
2. Measure throughput degradation
3. Validate multi-day stability

**Documentation**: `docs/optimizations/experiments/experimental-cpu-offloading.md`

### 1GB Huge Pages for Training

**Objective**: Use 1GB huge pages instead of 2MB for dataset memory-mapping.

**Technical Background**: Larger TLB coverage reduces page table walks. CPU TLB can cache approximately 1500 entries. 1GB pages virtually eliminate misses for large datasets.

**Configuration**:
- Kernel boot parameter: default_hugepagesz=1G hugepagesz=1G hugepages=64
- Reserve 64x 1GB pages (64GB total)
- Memory-map dataset using MAP_HUGETLB with 1GB size

**Requirements**:
- Kernel support for 1GB pages (CONFIG_HUGETLBFS)
- 64GB+ RAM available for huge page reservation
- Dataset size exceeds 50GB (smaller datasets see no benefit)

**Tradeoff**: Reduced system memory flexibility. 64GB pre-allocated at boot, cannot be used for other purposes. May cause OOM if total workload exceeds remaining RAM.

**Risk**: Fragmentation issues if datasets smaller than 1GB. Wasted memory for small datasets.

**Validation**:
1. Measure TLB miss rate with perf stat -e dTLB-load-misses
2. Compare 2MB vs 1GB pages
3. Measure end-to-end throughput improvement

**Documentation**: `docs/optimizations/experiments/experimental-1gb-hugepages-training.md`

### Mixed CPU-GPU Training

**Objective**: Hybrid training with model layers split across CPU and GPU.

**Technical Background**: Partition model layers across devices. GPU handles majority of layers, CPU handles overflow. Inference passes data between devices. PCIe transfer overhead expected.

**Configuration**: Accelerate device_map parameter specifies layer placement.

**Requirements**:
- Accelerate library for device mapping
- Fast CPU (Zen 5 optimized for inference)
- PCIe Gen5 for minimal transfer latency

**Use Case**: Train models exceeding 32GB VRAM. Not recommended if model fits on GPU.

**Tradeoff**: Significant throughput loss vs capacity gain. High complexity, tensor transfer management required.

**Risk**: Very high complexity. Debugging issues difficult (CPU vs GPU numerical differences). Gradient synchronization bugs possible.

**Validation**:
1. Train model exceeding GPU VRAM
2. Measure throughput
3. Validate numerical correctness vs full GPU baseline
4. Multi-day stability test

**Documentation**: `docs/optimizations/experiments/experimental-mixed-cpu-gpu.md`

## Implementation Phases

### Phase 1: Foundation

**Objective**: Establish basic training capability with minimal model.

**Tasks**:
- Create Docker training-gpu container based on PyTorch official image
- Install dependencies: transformers 4.51.1, accelerate, tensorboard, tiny-qwen
- Configure GPU access, shared memory, CPU core pinning
- Implement Shakespeare training script with 10M parameter model
- Integrate tiny-qwen architecture
- Add TensorBoard logging
- Create Makefile targets for training
- Write quickstart guide

**Deliverables**:
- training-gpu container builds and runs with GPU access
- Shakespeare 10M training completes successfully
- TensorBoard accessible on port 6006
- docs/training/quickstart.md written
- Makefile targets: make train-shakespeare, make logs-training, make shell-training

**Validation**:
- Docker build succeeds
- GPU visible in container (nvidia-smi works)
- Training completes without errors
- Loss decreases during training
- Generated text resembles Shakespeare style
- Checkpoint saved successfully

### Phase 2: Scale Up

**Objective**: Enable training of production-scale Qwen3-1.7B model with both from-scratch pretraining and continued pretraining (domain adaptation).

**Tasks - Training from Scratch**:
- Implement dataset download script for OpenWebText
- Create preprocessing pipeline with Qwen3 tokenizer (151,646 vocab)
- Generate memory-mapped token arrays
- Implement Qwen3-1.7B training script using tiny-qwen architecture
- Add checkpoint management (30 minutes + 5000 steps, keep last 3 + best)
- Implement checkpoint resumption
- Add validation loop
- Configure BFloat16 mixed precision
- Implement gradient clipping
- Add learning rate warmup and cosine decay
- Run baseline training to establish unoptimized performance
- Document hyperparameters

**Tasks - Continued Pretraining**:
- Implement checkpoint loading from HuggingFace Hub (official Qwen3-1.7B)
- Add support for loading custom checkpoint paths
- Implement domain-specific dataset download scripts (GitHub for code, PubMed for medical)
- Create dataset mixing pipeline (80-90% domain, 10-20% general text)
- Add continued pretraining script with lower learning rate (1e-4 to 3e-4)
- Configure domain-specific validation sets
- Implement catastrophic forgetting detection (measure general capability during training)
- Create configs for multiple domains (code.yaml, medical.yaml, legal.yaml)
- Document mixing ratios and hyperparameters per domain
- Run baseline continued pretraining experiment on code domain

**Deliverables**:
- OpenWebText downloaded and preprocessed
- Tokenized dataset with Qwen3 tokenizer
- scripts/training/train_qwen.py with full implementation
- scripts/training/pretrain.py with checkpoint loading and domain adaptation
- scripts/data/download_domain_datasets.py (supports code, medical, legal)
- scripts/data/mix_datasets.py (implements data mixing with configurable ratios)
- Checkpoint management working for both training types
- Resume from checkpoint tested
- configs/training/qwen3-1.7b-scratch.yaml
- configs/training/qwen3-1.7b-code.yaml
- configs/training/qwen3-1.7b-medical.yaml
- configs/training/qwen3-1.7b-legal.yaml
- Baseline trained model checkpoint (from scratch)
- Baseline continued pretrained model checkpoint (code domain)
- TensorBoard logs for both training types
- docs/continued-pretraining/README.md
- docs/continued-pretraining/domain-adaptation.md
- docs/continued-pretraining/mixing-strategies.md

**Validation**:
- Dataset preprocessing completes for general and domain datasets
- Dataset mixing produces correct proportions (validate token counts)
- Official Qwen3-1.7B checkpoint loads successfully
- From-scratch training initializes with random weights
- Continued pretraining initializes from checkpoint
- Model trains without errors for both approaches
- Checkpoints save correctly with metadata indicating training type
- Resume from checkpoint continues from exact state
- Training stable with BF16 (no NaN losses)
- Learning rate schedules correct for both training types
- Baseline performance measured for from-scratch
- Domain-specific metrics improve for continued pretraining
- General capability maintained during continued pretraining (no catastrophic forgetting)

### Phase 3: Systematic Optimization

**Objective**: Systematic performance improvement through measured optimizations.

**Tasks**:
- Implement training benchmark script
- Measure baseline throughput, GPU utilization, memory usage
- Apply TF32 precision optimization, measure impact
- Add torch.compile with max-autotune, measure impact
- Integrate Flash Attention via scaled_dot_product_attention, measure impact
- Optimize DataLoader: test worker counts (4, 8, 12, 16), test prefetch factors, enable pin_memory and persistent_workers, implement CPU core pinning
- Apply OS-level optimizations: configure I/O scheduler (kyber), tune memory parameters, configure read-ahead buffer
- Measure combined optimization impact
- Document all optimizations with benchmarks
- Create optimization guides

**Deliverables**:
- scripts/training/training_benchmark.py
- benchmarks/training/qwen3/baseline.json
- benchmarks/training/qwen3/tf32.json
- benchmarks/training/qwen3/torch_compile.json
- benchmarks/training/qwen3/flash_attention.json
- benchmarks/training/qwen3/dataloader_optimized.json
- benchmarks/training/qwen3/final_optimized.json
- docs/optimizations/training/os-training.md
- Performance improvement documented

**Validation**:
- Baseline performance established
- Each optimization measured independently
- Combined optimizations show cumulative benefit
- Sustained performance over extended run
- No memory leaks
- GPU utilization improved significantly

### Phase 4: Complete Pipeline

**Objective**: End-to-end automation from raw text to deployed quantized models.

**Tasks**:
- Implement LoRA fine-tuning script
- Add PEFT library integration
- Create instruction dataset format
- Test LoRA training on Qwen3-1.7B
- Implement GGUF conversion script
- Install llama.cpp quantization tools
- Quantize to Q8_0, Q6_K, Q4_K_M formats
- Benchmark perplexity for each format
- Implement SafeTensors conversion script
- Add GPTQ quantization (4-bit)
- Add AWQ quantization (4-bit)
- Implement quality comparison script
- Create evaluation scripts for perplexity
- Implement model comparison tool
- Update Makefile with deployment targets
- Implement auto-deployment to llama-cpu (GGUF)
- Implement auto-deployment to vllm-gpu (SafeTensors)
- Create end-to-end automation
- Write lifecycle documentation

**Deliverables**:
- scripts/training/finetune_lora.py
- scripts/quantization/convert_to_gguf.py
- scripts/quantization/quantize_gguf.py
- scripts/quantization/convert_to_safetensors.py
- scripts/quantization/quantize_gptq.py
- scripts/quantization/quantize_awq.py
- scripts/evaluation/eval_perplexity.py
- scripts/evaluation/compare_models.py
- Makefile targets: make deploy-gguf, make deploy-safetensors, make train-and-deploy
- docs/training/complete-lifecycle.md
- docs/quantization/README.md

**Validation**:
- LoRA fine-tuning completes successfully
- Instruction-following capability improved
- GGUF conversion produces valid files
- Quantization quality acceptable
- SafeTensors conversion produces valid files
- Deployed models load in inference services
- End-to-end pipeline completes without manual intervention
- Quality validation passes for all formats

### Phase 5: Advanced Features

**Objective**: Custom tooling and multi-day stability validation.

**Tasks**:
- Implement pytorch_memory_optimizer.cpp
- Implement fast_tokenizer.cpp optimized for Qwen3 vocab (151,646)
- Create Python bindings for both tools
- Test with extended runs
- Benchmark performance improvements from custom tooling
- Start extended training run
- Implement auto-recovery on failure
- Monitor stability throughout
- Document any issues encountered
- Write scaling documentation

**Deliverables**:
- docker/training-gpu/pytorch_memory_optimizer.cpp compiled and integrated
- docker/training-gpu/fast_tokenizer.cpp with Qwen3 optimizations compiled
- Performance benchmarks showing custom tooling impact
- Extended training run completes successfully
- docs/optimizations/training/custom-cpp-tools.md
- Multi-day stability validated

**Validation**:
- Memory optimizer prevents fragmentation over extended run
- Fast tokenizer provides speedup vs HuggingFace
- Integration works transparently
- Training completes without crashes
- Throughput stable throughout (no degradation)
- Checkpoints save successfully
- Final model quality acceptable

### Phase 6: Evaluation and Quality

**Objective**: Comprehensive model quality assessment capability.

**Tasks**:
- Install lm-evaluation-harness
- Implement EleutherAI harness integration script
- Test on trained Qwen3-1.7B model
- Run evaluation subset: MMLU, HellaSwag, ARC
- Implement MT-Bench integration
- Test on LoRA fine-tuned model
- Generate conversation samples
- Implement model comparison tools
- Support comparing multiple checkpoints and quantization formats
- Create visualization tools
- Document evaluation frameworks
- Create metric interpretation guide
- Establish baseline scores
- Write best practices guide

**Deliverables**:
- scripts/evaluation/frameworks/lm_eval_harness.py
- scripts/evaluation/frameworks/mt_bench.py
- scripts/evaluation/compare_models.py
- Evaluation results for Qwen3-1.7B trained model
- Quality reports
- docs/evaluation/README.md
- docs/evaluation/evaluation-frameworks.md
- docs/evaluation/metric-interpretation.md

**Validation**:
- EleutherAI evaluation completes successfully
- Results formatted clearly
- MT-Bench runs on fine-tuned model
- Instruction-following scores measured
- Comparisons accurate and reproducible
- Documentation complete and clear

### Phase 7: Experimental Optimizations

**Objective**: Explore cutting-edge optimization techniques with rigorous documentation.

**Tasks**:
- Verify Flash Attention 3 support in cuDNN
- Enable and benchmark FA3
- Measure attention layer throughput
- Validate numerical correctness
- Configure DeepSpeed ZeRO-Offload
- Attempt training larger model with CPU offloading
- Measure throughput impact
- Configure 1GB huge pages
- Allocate dataset with 1GB pages
- Measure TLB miss reduction
- Benchmark throughput impact
- Implement fused kernel (RMSNorm + SwiGLU for Qwen3)
- Benchmark against PyTorch native
- Configure Accelerate device_map for CPU-GPU split
- Test with model exceeding GPU VRAM
- Document all experiments thoroughly

**Deliverables**:
- Flash Attention 3 benchmarks
- docs/optimizations/experiments/experimental-flash-attention-training.md
- ZeRO-Offload configuration and results
- docs/optimizations/experiments/experimental-cpu-offloading.md
- 1GB huge pages configuration and results
- docs/optimizations/experiments/experimental-1gb-hugepages-training.md
- Custom kernel implementation
- docs/optimizations/experiments/experimental-custom-kernels.md
- Mixed CPU-GPU configuration
- docs/optimizations/experiments/experimental-mixed-cpu-gpu.md

**Validation**:
- Each experiment thoroughly documented
- Performance measured with benchmarks
- Positive or negative results documented equally
- Reproducibility validated
- Tradeoffs clearly explained

## Success Criteria

### Functional Requirements

**Basic training**: Successfully train Qwen3-1.7B model from scratch on OpenWebText using tiny-qwen architecture. Achieve acceptable perplexity on validation set. Training completes without errors, loss curves show proper convergence, checkpoints save correctly.

**Continued pretraining**: Load official Qwen3-1.7B checkpoint and successfully adapt to specialized domain (code, medical, or legal). Implement data mixing (80-90% domain, 10-20% general). Achieve measurable quality improvement on domain-specific benchmarks (e.g., HumanEval for code, MedQA for medical) while maintaining general capability on standard benchmarks (minimal degradation on MMLU or similar). Demonstrate catastrophic forgetting prevention through dual evaluation.

**Fine-tuning**: Fine-tune with LoRA achieving instruction-following capability. Demonstrate improvement vs base model on instruction-following benchmarks. LoRA adapters integrate correctly with base model, training converges faster than full fine-tuning.

**Quantization**: Convert trained models to both GGUF and SafeTensors formats. Acceptable quality degradation per format specifications.

**Deployment**: Deploy quantized models to existing inference services automatically. One-command deployment working.

### Performance Requirements

**Training stability**: Complete extended training run without manual intervention. Zero crashes or hangs. Checkpoints save successfully every 30 minutes and 5000 steps.

**Data loading**: Maintain high GPU utilization (eliminate data loading bottlenecks). Measured over sustained training period.

**Optimization impact**: Demonstrate measurable improvement from systematic optimization. Each optimization documented with before/after benchmarks.

**Quantization speed**: Convert Qwen3-1.7B model to all quantization formats in reasonable time. Complete pipeline automation.

### Quality Requirements

**Documentation rigor**: Match existing v1.0 documentation standards with complete reproduction steps. Every optimization documented with methodology.

**Benchmark validation**: All optimizations validated with benchmark data. Performance claims reproducible.

**Educational value**: Users can progress from toy model (Shakespeare 10M) to production model (Qwen3-1.7B on OpenWebText) following documentation.

**Integration quality**: Seamless integration with v1.0 infrastructure. No degradation of existing inference services during concurrent training.

### Integration Requirements

**Concurrent operation**: Training on GPU while inference on CPU operates without performance degradation. Validated through testing.

**Format compatibility**: Quantized models work with llama.cpp and vLLM without modification. Load and serve successfully.

**Monitoring integration**: TensorBoard provides real-time training visibility with loss curves, metrics, and system utilization.

**Checkpoint compatibility**: Checkpoints compatible with HuggingFace ecosystem for model sharing.

## Attribution

This project builds upon excellent open source work from the community:

### Architecture Implementation

**tiny-qwen** (github.com/Emericen/tiny-qwen)
- Minimal, readable PyTorch reimplementation of Qwen3
- Educational focus with clean code
- Apache 2.0 License

**rasbt/qwen3-from-scratch** (huggingface.co/rasbt/qwen3-from-scratch)
- From-scratch educational PyTorch implementation
- Detailed explanations and learning resources
- By Sebastian Raschka, author of "Build a Large Language Model From Scratch"

**ideaweaver-ai/qwen3-from-scratch** (github.com/ideaweaver-ai/qwen3-from-scratch)
- Complete implementation with training examples
- Configuration templates and hyperparameters
- Practical training demonstrations

### Our Contributions

ai-experiments v2 extends these educational implementations with:

- **Production infrastructure**: Docker orchestration, checkpoint management, monitoring
- **Hardware-specific optimization**: Tuned for RTX 5090 + AMD Ryzen 9950X
- **Complete quantization pipeline**: PyTorch → GGUF + SafeTensors with quality validation
- **Integration with v1.0**: Automatic deployment to existing inference services
- **Systematic optimization research**: Documented performance gains on specific hardware
- **Custom tooling**: SIMD tokenizer, memory optimizer, huge pages integration
- **Research documentation**: Discoveries for curiosity-chronicles blog

### Technical References

**Qwen3 Technical Report**: Qwen team, Alibaba Cloud (April 2025)
**Qwen Blog**: qwenlm.github.io/blog/qwen3/
**Transformers Documentation**: huggingface.co/docs/transformers/model_doc/qwen3

## Dependencies

### Software Dependencies

**Core frameworks** (from pyproject.toml):
- PyTorch: 2.5+ with CUDA 12.9 support (managed via pip)
- transformers: 4.51.1+ (Qwen3 support added in 4.51.0)
- accelerate: 1.8.1
- datasets: 3.1.0
- peft: 0.15.2
- evaluate: 0.4.4

**Optimization libraries**:
- deepspeed: 0.17.1
- bitsandbytes: 0.46.0
- Flash Attention: Via cuDNN 9.7.0+ backend

**Data processing**:
- sentencepiece: 0.2.0
- tiktoken: 0.9.0
- Custom fast_tokenizer (C++ implementation)

**Evaluation**:
- lm-evaluation-harness: EleutherAI evaluation suite
- MT-Bench: Multi-turn conversation evaluation

**Quantization**:
- llama-cpp-python: 0.3.14 (GGUF conversion)
- AutoGPTQ: GPTQ quantization
- AutoAWQ: AWQ quantization

**Monitoring**:
- tensorboard: 2.19.0
- wandb: 0.20.1 (optional)
- mlflow: 3.1.1 (optional)

### Hardware Dependencies

**GPU requirements**: NVIDIA GPU with minimum 24GB VRAM, Compute Capability 8.0 or higher (Ampere architecture or newer). Recommended: RTX 5090 (32GB VRAM, Blackwell architecture).

**CPU requirements**: Minimum 12 cores for efficient data loading, AVX2 support required, AVX-512 support recommended for fast tokenizer optimization.

**Memory requirements**: Minimum 64GB system RAM, 128GB recommended for optimal performance and concurrent operations.

**Storage requirements**: Minimum 500GB fast NVMe storage for datasets and checkpoints, 1TB recommended. Sequential read performance 3GB/s minimum.

### Existing Infrastructure

Training infrastructure builds on existing v1.0 components:

- BIOS optimizations: Apply directly to training workloads
- Huge pages configuration: Benefits dataset streaming
- Docker patterns: Maintain consistency across services
- GPU software stack: CUDA, cuDNN, drivers compatible
- Python environment: Managed through existing Poetry configuration (pyproject.toml)

## Future Considerations

### Multi-GPU Support

Single-GPU limitation constrains maximum model size to approximately 2B parameters with current optimizations. Future expansion could add Distributed Data Parallel (DDP) or Fully Sharded Data Parallel (FSDP) for multi-GPU training.

**Design consideration**: Keep data loading and model architecture separate. Device-agnostic code enables easier DDP integration later.

### Cloud Integration

Current focus is local workstation only. Future users may want cloud training for larger models or faster iteration.

**Documentation needed**: Cloud instance selection guide, cost comparison vs local, data transfer strategies.

### Advanced Architectures

Qwen3-1.7B provides excellent foundation. Future expansion could add support for newer architectures as they emerge. Document architectural abstractions that enable adding new model families.

**Design consideration**: Abstract model configuration from training loop. Config-driven architecture selection enables easy addition of new models.

### Production Serving

Training infrastructure focuses on model development, not production serving. Future expansion could add model versioning, A/B testing infrastructure, and production deployment patterns.

**Potential additions**: Model registry, versioned deployments, rollback mechanisms, blue-green deployment, load balancing.

### Cost Tracking

Current roadmap focuses on capability and performance. Future enhancement could add real-time power consumption tracking and cost-per-model calculations for optimization insights.

**Implementation**: Integrate with NVIDIA Management Library (NVML) for real-time power monitoring. Track cumulative energy consumption per training run.

### Advanced Evaluation

Current evaluation covers standard benchmarks. Future expansion could add human evaluation infrastructure, adversarial testing, and bias detection.

**Potential frameworks**: HELM (Holistic Evaluation of Language Models), Big Bench, bias measurement suites, safety benchmarks.

---

*Last Updated: 2025-10-29*
