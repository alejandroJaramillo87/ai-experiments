# Training Infrastructure Roadmap

Comprehensive roadmap for implementing small-to-medium scale language model training capabilities in the ai-experiments repository.

## Table of Contents

- [Vision](#vision)
- [Scope](#scope)
- [Hardware Requirements](#hardware-requirements)
- [Directory Structure](#directory-structure)
- [Services Architecture](#services-architecture)
- [Custom Tooling](#custom-tooling)
- [Optimization Strategy](#optimization-strategy)
- [Quantization Strategy](#quantization-strategy)
- [Evaluation Strategy](#evaluation-strategy)
- [Implementation Phases](#implementation-phases)
- [Success Criteria](#success-criteria)
- [Dependencies](#dependencies)
- [Future Considerations](#future-considerations)

## Vision

Transform the ai-experiments repository from best-in-class inference infrastructure into a complete AI lifecycle platform for prosumer hardware. Enable training of small-to-medium language models (10M to 1.5B parameters) with the same systematic optimization, rigorous documentation, and production stability that characterizes the current inference services.

## Scope

### Included

**Training from scratch**: Train base language models on raw text corpora using GPT-2 architecture variants (124M, 350M, 774M, 1.5B parameters).

**Continued pretraining**: Adapt existing base models to specialized domains through additional pretraining on domain-specific corpora.

**Fine-tuning**: Full parameter fine-tuning and parameter-efficient fine-tuning with Low-Rank Adaptation (LoRA) for instruction following and task-specific behaviors.

**Quantization**: Post-training quantization to GGUF format for llama.cpp inference and HuggingFace SafeTensors format for vLLM inference.

**Dataset preparation**: Complete pipeline for data collection, cleaning, tokenization, and validation.

**Evaluation**: Model quality assessment through perplexity measurements, downstream task evaluation, and instruction-following benchmarks.

**Optimization**: Systematic optimization of training throughput, memory efficiency, and data loading performance.

### Excluded

**Multi-GPU training**: Single RTX 5090 only. Distributed training across multiple GPUs or nodes is out of scope.

**Large model training**: Models exceeding 2B parameters are beyond single-GPU memory constraints.

**Cloud deployment**: Focus remains on local workstation infrastructure.

**Production serving**: Training services are for model development, not serving production traffic.

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

### Software Stack

- NVIDIA Driver: 580.65.06 or newer
- CUDA: 13.0 or newer
- cuDNN: 9.13.0 or newer
- PyTorch: 2.3.0 or newer
- Python: 3.12

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
│   ├── train_gpt.py
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
│   ├── create_vocab.py
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
│   │   ├── gpt-architecture.md
│   │   ├── model-configs.md
│   │   └── scaling-laws.md
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
│   ├── experiments/
│   │   ├── shakespeare-10m.md
│   │   ├── openwebtext-124m.md
│   │   ├── custom-domain-350m.md
│   │   └── scaling-study.md
│   └── benchmarks/
│       ├── training-throughput.md
│       ├── convergence-curves.md
│       └── hardware-utilization.md
├── fine-tuning/
│   ├── README.md
│   ├── full-finetuning.md
│   ├── lora-finetuning.md
│   ├── dataset-formats.md
│   ├── evaluation-metrics.md
│   └── preventing-overfitting.md
├── continued-pretraining/
│   ├── README.md
│   ├── domain-adaptation.md
│   ├── catastrophic-forgetting.md
│   └── mixing-strategies.md
├── quantization/
│   ├── README.md
│   ├── quantization-methods.md
│   ├── gguf-conversion.md
│   ├── safetensors-conversion.md
│   ├── accuracy-benchmarks.md
│   ├── inference-integration.md
│   └── optimal-formats.md
├── evaluation/
│   ├── README.md
│   ├── evaluation-frameworks.md
│   ├── perplexity-guide.md
│   ├── downstream-tasks.md
│   ├── instruction-following.md
│   └── quality-metrics.md
└── optimizations/
    ├── training/
    │   ├── README.md
    │   ├── bios-training.md
    │   ├── os-training.md
    │   ├── gpu-training.md
    │   ├── cpu-dataloader.md
    │   └── custom-cpp-tools.md
    └── experiments/
        ├── experimental-flash-attention-training.md
        ├── experimental-custom-kernels.md
        ├── experimental-cpu-offloading.md
        ├── experimental-1gb-hugepages-training.md
        └── experimental-mixed-cpu-gpu.md
```

### Configuration Files

```
configs/
└── training/
    ├── shakespeare.yaml
    ├── gpt-124m.yaml
    ├── gpt-350m.yaml
    ├── gpt-774m.yaml
    ├── gpt-1.5b.yaml
    ├── datasets/
    │   ├── openwebtext.yaml
    │   ├── shakespeare.yaml
    │   ├── redpajama.yaml
    │   └── custom-code.yaml
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
    ├── openwebtext-124m/
    │   ├── baseline.json
    │   ├── optimized.json
    │   └── training_curves.png
    ├── scaling-study/
    │   ├── 124m.json
    │   ├── 350m.json
    │   ├── 774m.json
    │   └── analysis.md
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
**Features**: DeepSpeed integration, Accelerate support, custom CUDA memory allocator, TensorBoard logging
**Mounts**: `/mnt/ai-data/datasets/`, `/mnt/ai-data/checkpoints/`, `/logs/training/`

### Data Preparation Service

**Port**: 8007
**Purpose**: Dataset preprocessing and tokenization
**CPU Allocation**: Cores 24-31 (background processing)
**Memory**: 16GB allocated
**Features**: SIMD-optimized tokenization, data validation, streaming dataset preparation
**Mounts**: `/mnt/ai-data/datasets/raw/`, `/mnt/ai-data/datasets/preprocessed/`

### Quantization Service

**Port**: 8008
**Purpose**: Post-training model quantization
**CPU Allocation**: Cores 12-23
**Memory**: 64GB allocated
**Features**: GGUF conversion, SafeTensors conversion, GPTQ quantization, AWQ quantization
**Mounts**: `/mnt/ai-data/checkpoints/`, `/mnt/ai-data/models/`

### TensorBoard Service

**Port**: 6006
**Purpose**: Training visualization and monitoring
**Features**: Real-time loss curves, metric tracking, hyperparameter comparison
**Mounts**: `/logs/training/`

### Concurrent Operation

Training services run concurrently with existing inference services:

- GPU training: RTX 5090 (training-gpu service)
- CPU inference: AMD Ryzen 9950X cores 0-11 (llama-cpu service)
- Web interface: open-webui on port 3000
- Monitoring: TensorBoard on port 6006

This configuration enables testing newly trained models immediately after quantization without interrupting inference availability.

## Custom Tooling

### PyTorch Memory Optimizer

**Location**: `docker/training-gpu/pytorch_memory_optimizer.cpp`
**Purpose**: Prevents CUDA memory fragmentation during multi-day training runs
**Integration**: Custom PyTorch memory allocator with huge pages support for CPU tensors
**Performance Impact**: Prevents 15-20% throughput degradation on day three of multi-day training runs
**Implementation**: Preallocates gradient and activation memory, reduces fragmentation through custom allocation strategies

### Fast Tokenizer

**Location**: `docker/training-gpu/fast_tokenizer.cpp`
**Purpose**: SIMD-optimized BPE tokenization leveraging AVX-512 on AMD Zen 5
**Performance Impact**: 3-5x faster than HuggingFace fast tokenizers
**Implementation**: Vectorized byte-pair encoding with NVMe streaming integration
**Use Case**: Dataset preprocessing bottleneck elimination

### Huge Pages Wrapper for Training

**Location**: `docker/training-gpu/training_utils/hugepage_wrapper.cpp`
**Purpose**: Extends inference huge pages wrapper for training workloads
**Integration**: Dataset memory-mapping with transparent huge pages
**Performance Impact**: 8-12% improvement in dataset streaming throughput

## Optimization Strategy

### Baseline Establishment

Establish unoptimized baseline performance for each model size to measure improvement systematically.

**Target Baseline**: 220,000 tokens per second, 38% Model FLOPS Utilization (MFU) for 124M parameter model on RTX 5090.

### Systematic Optimization Areas

**GPU memory**: TensorFloat-32 (TF32) precision, gradient checkpointing, Flash Attention integration, CUDA graph optimization.

**Mixed precision**: BFloat16 training, dynamic loss scaling, selective FP32 operations for numerical stability.

**Data loading**: DataLoader worker tuning (4, 8, 12 workers), CPU core pinning, prefetch factor optimization, memory-mapped datasets.

**I/O pipeline**: NVMe scheduler tuning (kyber vs none), read-ahead buffer sizing, dataset caching strategies, streaming optimizations.

**Compilation**: torch.compile integration, CUDA kernel fusion, custom kernel implementations.

**Memory bandwidth**: Huge pages (2MB, 1GB experimental), NUMA awareness, memory allocation strategies.

### Target Performance

Achieve 310,000 tokens per second with 53% MFU for 124M parameter model through systematic optimization, representing 41% improvement over baseline.

### Documentation Requirements

Each optimization requires:

- Benchmark JSON files (before and after)
- Step-by-step reproduction guide
- Hardware-specific applicability notes
- Cost-benefit analysis for different model sizes

## Quantization Strategy

### Dual Format Support

Post-training quantization produces models in two formats to support both inference engines in the existing infrastructure.

### GGUF Format (llama.cpp)

**Target Inference Engine**: llama-cpu and llama-gpu services
**Conversion Tool**: `scripts/quantization/convert_to_gguf.py`
**Quantization Levels**: Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_K_S, Q3_K_M, Q2_K
**Quality Benchmarking**: Perplexity measurements across quantization levels
**Integration**: Direct deployment to existing llama.cpp inference services

### SafeTensors Format (vLLM)

**Target Inference Engine**: vllm-gpu service
**Conversion Tool**: `scripts/quantization/convert_to_safetensors.py`
**Quantization Methods**: GPTQ (4-bit, 3-bit), AWQ (4-bit)
**Quality Benchmarking**: Downstream task accuracy across quantization methods
**Integration**: Compatible with vLLM batch inference service

### Quantization Workflow

Training produces PyTorch checkpoint files. Quantization service converts checkpoints to both GGUF and SafeTensors formats simultaneously. Automated benchmarking validates quality degradation remains within acceptable thresholds. Models deploy directly to appropriate inference services based on format.

### Documentation

`docs/quantization/` covers conversion procedures, format-specific optimization techniques, quality benchmarking methodology, and inference integration patterns for both formats.

## Evaluation Strategy

### Distinction from Benchmarking

**Benchmarking** measures system performance: tokens per second, MFU percentage, GPU utilization, memory bandwidth.

**Evaluation** measures model quality: perplexity, accuracy, task performance, instruction-following quality.

Both are required but serve different purposes and use different tooling.

### Evaluation Frameworks

**EleutherAI LM Evaluation Harness**: Standardized downstream task evaluation including MMLU, HellaSwag, ARC, TruthfulQA, and Winogrande.

**MT-Bench**: Multi-turn conversation evaluation for instruction-tuned models.

**Perplexity Measurement**: Core language modeling quality metric on held-out validation sets.

### Evaluation Pipeline

**During training**: Monitor training loss, validation loss, and validation perplexity every N steps.

**After training**: Run full evaluation suite including perplexity on multiple corpora and downstream task battery.

**After fine-tuning**: Instruction-following evaluation, task-specific accuracy, comparison against base model.

**After quantization**: Quality degradation measurement across quantization levels to establish acceptable thresholds.

### Documentation

`docs/evaluation/` provides framework integration guides, metric interpretation guidelines, evaluation best practices, and quality benchmark baselines for reference models.

## Implementation Phases

### Phase 1: Foundation (Week 1)

**Objective**: Establish basic training capability with minimal model.

**Deliverables**:

- Docker service: training-gpu with basic configuration
- Script: train_shakespeare.py implementing simple training loop
- Config: shakespeare.yaml model configuration
- Documentation: quickstart guide for first training run
- Makefile targets: train-shakespeare, logs-training
- TensorBoard integration for real-time monitoring

**Success Criteria**: Users can train 10M parameter model on Shakespeare corpus in five minutes and generate text samples.

### Phase 2: Scale Up (Week 2)

**Objective**: Enable training of production-scale models.

**Deliverables**:

- Script: train_gpt.py with full GPT-2 training loop
- Script: prepare_dataset.py for OpenWebText preprocessing
- Config: gpt-124m.yaml model configuration
- Checkpoint management: save, resume, best-model tracking
- Gradient accumulation for memory-constrained training
- Mixed precision training with BFloat16
- Documentation: complete training guide

**Success Criteria**: Train 124M parameter model on OpenWebText to GPT-2 quality (perplexity approximately 22) in 10-hour run with checkpoint recovery capability.

### Phase 3: Optimization (Week 3)

**Objective**: Systematic performance improvement through measured optimizations.

**Deliverables**:

- Benchmark suite: training_benchmark.py measuring MFU and throughput
- OS optimizations: I/O scheduler tuning, memory configuration
- GPU optimizations: TF32, torch.compile, Flash Attention
- DataLoader optimizations: worker tuning, CPU pinning, prefetch optimization
- Documentation: optimization guide with before/after benchmarks
- Config: optimized.yaml with best-practice settings
- Benchmark JSONs: baseline and optimized results

**Success Criteria**: Achieve 310,000 tokens per second with 53% MFU, reducing 124M model training time from 10 hours to six hours.

### Phase 4: Complete Pipeline (Week 4)

**Objective**: End-to-end automation from raw text to deployed model.

**Deliverables**:

- Script: finetune_lora.py for parameter-efficient fine-tuning
- Script: convert_to_gguf.py for llama.cpp format conversion
- Script: convert_to_safetensors.py for vLLM format conversion
- Script: eval_perplexity.py for quality validation
- Integration: automated deployment to inference services
- Documentation: complete lifecycle guide
- Makefile target: train-and-deploy

**Success Criteria**: Single command execution that trains base model, fine-tunes with LoRA, quantizes to both formats, and deploys to appropriate inference services.

### Phase 5: Scale and Advanced Features (Weeks 5-6)

**Objective**: Support larger models and advanced training techniques.

**Deliverables**:

- Configs: gpt-350m.yaml, gpt-774m.yaml, gpt-1.5b.yaml
- Memory optimization: gradient checkpointing for larger models
- Script: pretrain.py for continued pretraining workflows
- Multi-day stability: checkpoint frequency tuning, auto-recovery
- Scaling study: comparative analysis across model sizes
- Documentation: scaling study results, optimal configurations per model size
- Custom tooling: pytorch_memory_optimizer.cpp, fast_tokenizer.cpp

**Success Criteria**: Successfully train 774M parameter model through four-day run with zero manual intervention and deploy quantized versions to inference services.

### Phase 6: Evaluation and Quality (Week 7)

**Objective**: Comprehensive model quality assessment capability.

**Deliverables**:

- Script: eval_downstream.py integrating LM Evaluation Harness
- Script: eval_instruction.py for instruction-following assessment
- Script: compare_models.py for A/B testing
- Documentation: evaluation frameworks guide
- Documentation: metric interpretation guide
- Benchmark baselines: reference model quality metrics

**Success Criteria**: Automated evaluation pipeline produces comprehensive quality reports for trained models with comparisons to reference models.

### Phase 7: Experimental Optimizations (Week 8)

**Objective**: Explore cutting-edge optimization techniques.

**Deliverables**:

- Documentation: experimental-flash-attention-training.md
- Documentation: experimental-custom-kernels.md
- Documentation: experimental-cpu-offloading.md
- Documentation: experimental-1gb-hugepages-training.md
- Documentation: experimental-mixed-cpu-gpu.md
- Benchmark results: experimental optimization performance data

**Success Criteria**: Document experimental optimizations with reproducible benchmarks, even if performance gains are marginal or negative.

## Success Criteria

### Functional Requirements

**Basic training**: Successfully train 124M parameter model from scratch to GPT-2 quality.

**Continued pretraining**: Adapt base model to specialized domain with measurable quality improvement.

**Fine-tuning**: Fine-tune with LoRA and achieve instruction-following capability.

**Quantization**: Convert trained models to both GGUF and SafeTensors formats with minimal quality loss.

**Deployment**: Deploy quantized models to existing inference services without manual configuration.

### Performance Requirements

**Training throughput**: Achieve minimum 310,000 tokens per second for 124M model (53% MFU).

**Data loading**: Maintain minimum 95% GPU utilization (eliminate data loading bottlenecks).

**Multi-day stability**: Complete four-day 774M training run without manual intervention.

**Quantization speed**: Convert 124M model to all quantization formats in under 30 minutes.

### Quality Requirements

**Documentation rigor**: Match existing v1.0 documentation standards with complete reproduction steps.

**Benchmark validation**: All optimizations validated with before/after benchmark JSONs.

**Educational value**: Users can progress from toy model to production model in under one week.

**Cost efficiency**: Train 350M parameter model for under 20 dollars in electricity costs.

### Integration Requirements

**Concurrent operation**: Training on GPU while inference on CPU operates without degradation.

**Format compatibility**: Quantized models work with llama.cpp and vLLM without modification.

**Monitoring integration**: TensorBoard provides real-time training visibility.

**Checkpoint compatibility**: Checkpoints compatible with HuggingFace ecosystem for model sharing.

## Dependencies

### Software Dependencies

**Core frameworks**: PyTorch 2.3.0 or newer with CUDA 13.0 support, Transformers 4.51.1, Accelerate 1.8.1.

**Optimization libraries**: DeepSpeed 0.17.1, bitsandbytes 0.46.0, Flash Attention via cuDNN 9.13.0.

**Data processing**: Datasets 3.1.0, tokenizers (HuggingFace), sentencepiece 0.2.0.

**Evaluation**: lm-evaluation-harness, MT-Bench, evaluate library.

**Quantization**: llama.cpp (GGUF conversion), GPTQ-for-LLaMa, AutoGPTQ, AutoAWQ.

**Monitoring**: TensorBoard 2.19.0, Weights and Biases (optional), MLflow (optional).

### Hardware Dependencies

**GPU requirements**: NVIDIA GPU with minimum 24GB VRAM, Compute Capability 8.0 or higher (Ampere architecture or newer).

**CPU requirements**: Minimum 12 cores for efficient data loading, AVX2 support required, AVX-512 support optional but beneficial.

**Memory requirements**: Minimum 64GB system RAM for 124M models, 128GB recommended for 774M models.

**Storage requirements**: Minimum 500GB fast NVMe storage for datasets and checkpoints, 1TB recommended.

### Existing Infrastructure

Training infrastructure builds on existing v1.0 components:

- BIOS optimizations apply directly to training workloads
- Huge pages configuration benefits dataset streaming
- Docker patterns and Makefile conventions maintain consistency
- GPU software stack (CUDA, cuDNN, drivers) requires no changes
- Python environment managed through existing Poetry configuration

## Future Considerations

### Multi-GPU Support

Single-GPU limitation constrains maximum model size to approximately 1.5B parameters. Future expansion could add Distributed Data Parallel (DDP) or Fully Sharded Data Parallel (FSDP) for multi-GPU training. Documentation should note architectural decisions that facilitate or hinder future multi-GPU support.

### Cloud Integration

Current focus is local workstation only. Future users may want cloud training for larger models. Document cloud-specific considerations without implementing cloud deployment.

### Advanced Architectures

GPT-2 architecture is well-understood and stable. Future expansion could add support for Mistral, Llama 3, or other architectures. Document architectural abstractions that enable adding new model families.

### Production Serving

Training infrastructure focuses on model development, not production serving. Future expansion could add model versioning, A/B testing infrastructure, and production deployment patterns.

### Cost Tracking

Current roadmap mentions cost efficiency but does not implement power monitoring. Future enhancement could add real-time power consumption tracking and cost-per-model calculations.

### Advanced Evaluation

Current evaluation covers standard benchmarks. Future expansion could add human evaluation infrastructure, adversarial testing, and bias detection.

---

*Last Updated: 2025-10-28*
