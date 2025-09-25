# Poetry Dependencies for AI Workstation

Understanding the Python dependency ecosystem for AI experiments and inference infrastructure.

## Table of Contents

- [Introduction to Poetry](#introduction-to-poetry)
  - [Why Poetry?](#why-poetry)
  - [Poetry vs Traditional Tools](#poetry-vs-traditional-tools)
- [Understanding pyproject.toml](#understanding-pyprojecttoml)
  - [File Structure Overview](#file-structure-overview)
  - [Project Metadata](#project-metadata)
  - [Dependency Management](#dependency-management)
- [Core AI/ML Framework Dependencies](#core-aiml-framework-dependencies)
  - [Transformers Ecosystem](#transformers-ecosystem)
  - [Training Optimization](#training-optimization)
  - [Quantization Tools](#quantization-tools)
- [Data Science & Visualization Stack](#data-science--visualization-stack)
  - [Interactive Development](#interactive-development)
  - [Data Processing](#data-processing)
  - [Visualization Tools](#visualization-tools)
- [MLOps & Experiment Tracking](#mlops--experiment-tracking)
  - [Experiment Management](#experiment-management)
  - [Model Versioning](#model-versioning)
- [Utility & Performance Tools](#utility--performance-tools)
  - [Progress Monitoring](#progress-monitoring)
  - [API Development](#api-development)
- [Quantization & Inference Dependencies](#quantization--inference-dependencies)
- [PyTorch Special Handling](#pytorch-special-handling)
  - [Why PyTorch is Different](#why-pytorch-is-different)
  - [CUDA Compatibility](#cuda-compatibility)
- [Development Dependencies](#development-dependencies)
  - [Code Quality Tools](#code-quality-tools)
  - [Testing Framework](#testing-framework)
- [Integration with GPU Stack](#integration-with-gpu-stack)
- [Common Tasks](#common-tasks)
  - [Daily Operations](#daily-operations)
  - [Dependency Updates](#dependency-updates)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)
- [Next Steps](#next-steps)

## Introduction to Poetry

Poetry is a Python dependency management tool that simplifies package management, virtual environments, and project building. For our AI workstation, it provides consistent environments for reliable model performance.

### Why Poetry?

Poetry solves several problems for AI development:

1. **Dependency Resolution** - Automatically resolves complex dependency trees
2. **Lock Files** - Ensures exact same versions across all environments
3. **Virtual Environments** - Isolated Python environments per project
4. **Single Configuration** - One `pyproject.toml` replaces multiple files
5. **Build System** - Modern PEP 517/518 compliant packaging

### Poetry vs Traditional Tools

| Feature | pip + venv | conda | Poetry |
|---------|------------|-------|--------|
| **Dependency Resolution** | Manual | Automatic | Automatic |
| **Lock Files** | requirements.txt | environment.yml | poetry.lock |
| **Virtual Env Management** | Separate | Integrated | Integrated |
| **PyPI Support** | Yes | Limited | Full |
| **Custom Repositories** | Complex setup | Yes | Yes |
| **Build System** | setuptools | conda-build | PEP 517 |

## Understanding pyproject.toml

The `pyproject.toml` file is the heart of our Python environment, defining everything from dependencies to code formatting rules.

### File Structure Overview

```toml
[tool.poetry]              # Project metadata
[tool.poetry.dependencies]  # Runtime dependencies
[build-system]             # Build configuration
[[tool.poetry.source]]     # Package repositories
[tool.poetry.group.dev.dependencies]  # Dev dependencies
[tool.black]               # Code formatter config
[tool.ruff]                # Linter configuration
```

### Project Metadata

```toml
[tool.poetry]
name = "ai-toolkit"
version = "0.1.0"
description = "Base Python project for AI engineering on RTX 5090"
authors = ["Alejandro <alejandroj1234@proton.me>"]
license = "MIT"
package-mode = false  # Not building a distributable package
```

This metadata identifies our project and its configuration for AI experiments.

### Dependency Management

Dependencies are grouped by purpose, with strict version constraints for consistency:

```toml
python = ">=3.12,<3.13"  # Python 3.12 for latest performance
transformers = ">=4.51.1,<4.52.0"  # Exact minor version
```

## Core AI/ML Framework Dependencies

These packages form the backbone of our AI capabilities:

### Transformers Ecosystem

```
┌──────────────────────────────────────────────┐
│            Application Layer                  │
│         (Your Models & Scripts)              │
└──────────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────┐
│          Transformers (4.51.1)               │
│    • Pre-trained models (GPT, BERT, T5)     │
│    • Tokenizers & processors                 │
│    • Training loops & pipelines              │
└──────────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────┐
│           Accelerate (1.8.1)                 │
│    • Multi-GPU training                      │
│    • Mixed precision (FP16/BF16)            │
│    • DeepSpeed integration                   │
└──────────────────────────────────────────────┘
```

#### Key Packages Explained:

**transformers (4.51.1)**
- **Purpose**: Access to thousands of pre-trained models
- **Our Use**: Loading and fine-tuning LLMs for inference
- **Impact**: Enables model experimentation without training from scratch
- **GPU Integration**: Automatic CUDA acceleration when available

**datasets (3.1.0)**
- **Purpose**: Efficient data loading and preprocessing
- **Our Use**: Managing training datasets for fine-tuning
- **Features**: Memory-mapped files, streaming, built-in datasets
- **Performance**: Can handle datasets larger than RAM

**accelerate (1.8.1)**
- **Purpose**: Simplifies mixed-precision training
- **Our Use**: Optimizing training on RTX 5090
- **Key Features**:
  - Automatic mixed precision (AMP)
  - Gradient accumulation
  - DeepSpeed integration
  - Multi-GPU support (future expansion)

**peft (0.15.2)**
- **Purpose**: Parameter-Efficient Fine-Tuning techniques
- **Our Use**: LoRA, QLoRA for fine-tuning large models
- **Benefits**:
  - Fine-tune 70B models on single GPU
  - 10-100x fewer trainable parameters
  - Maintains base model performance

### Training Optimization

**deepspeed (0.17.1)**
- **Purpose**: Memory and speed optimization for large models
- **Our Use**: ZeRO optimization stages for training
- **Benefits**:
  - ZeRO-1: Optimizer state sharding
  - ZeRO-2: + Gradient sharding
  - ZeRO-3: + Parameter sharding
  - Enables training models 10x larger

**einops (0.8.1)**
- **Purpose**: Tensor operations with Einstein notation
- **Our Use**: Clean, readable tensor manipulations
- **Example**: `rearrange(x, 'b h w c -> b c h w')`

### Quantization Tools

**bitsandbytes (0.46.0)**
- **Purpose**: 8-bit and 4-bit quantization for LLMs
- **Our Use**: Running larger models in limited VRAM
- **Impact**:
  - 4-bit: 75% memory reduction
  - 8-bit: 50% memory reduction
  - Minimal performance loss (~2-3%)
- **GPU Requirement**: CUDA compute capability ≥ 7.5

**sentence-transformers (5.0.0)**
- **Purpose**: Semantic embeddings and similarity
- **Our Use**: RAG systems, semantic search
- **Features**: Pre-trained embedding models
- **Performance**: Optimized for batch processing on GPU

## Data Science & Visualization Stack

### Interactive Development

**jupyterlab (4.4.3)**
- **Purpose**: Interactive notebook environment
- **Our Use**: Experimentation, visualization, prototyping
- **GPU Integration**: Direct access to CUDA kernels
- **Extensions**: Variable inspector, git integration

### Data Processing

**pandas (2.3.0)**
- **Purpose**: Dataframe operations and analysis
- **Our Use**: Dataset preparation, results analysis
- **Performance**: Leverages NumPy's vectorization
- **Memory**: Efficient columnar storage

**scikit-learn (1.7.0)**
- **Purpose**: Classical ML algorithms
- **Our Use**: Data preprocessing, baseline models
- **Integration**: Works seamlessly with deep learning pipelines

### Visualization Tools

**matplotlib (3.10.3) & seaborn (0.13.2)**
- **Purpose**: Publication-quality plots
- **Our Use**: Performance metrics, loss curves, attention maps
- **GPU Monitoring**: Visualize GPU utilization over time

## MLOps & Experiment Tracking

These tools ensure consistency and tracking of our experiments:

```
Experiment Lifecycle:
┌────────────┐    ┌────────────┐    ┌────────────┐
│   wandb    │───▶│   mlflow   │───▶│tensorboard │
│  (Remote)  │    │  (Local)   │    │(Real-time) │
└────────────┘    └────────────┘    └────────────┘
     ↓                 ↓                   ↓
  Metrics         Model Registry      Visualization
  Artifacts       Versioning          Loss Curves
  Hyperparams     Versioning          Embeddings
```

### Experiment Management

**wandb (0.20.1)**
- **Purpose**: Weights & Biases experiment tracking
- **Our Use**: Track training runs, hyperparameters, metrics
- **Features**:
  - Automatic GPU metrics logging
  - Hyperparameter sweeps
  - Model artifact storage
  - Team collaboration

**mlflow (3.1.1)**
- **Purpose**: End-to-end ML lifecycle management
- **Our Use**: Model versioning and management
- **Components**:
  - Tracking: Log parameters, metrics, artifacts
  - Projects: Consistent runs
  - Models: Standardized model packaging
  - Registry: Model versioning and staging

### Model Versioning

**tensorboard (2.19.0)**
- **Purpose**: TensorFlow's visualization toolkit
- **Our Use**: Real-time training monitoring
- **Visualizations**:
  - Scalar metrics (loss, accuracy)
  - Histograms (weights, gradients)
  - Images (attention maps)
  - Embeddings (t-SNE, PCA)

## Utility & Performance Tools

### Parallel Computing

**ray (2.47.1)**
- **Purpose**: Parallel computing framework
- **Our Use**: Parallel hyperparameter tuning on local workstation
- **Features**:
  - Ray Tune: Hyperparameter optimization
  - Ray Serve: Model serving
  - Ray Data: Parallel data processing
- **Local Usage**: Utilizes all 32 CPU cores efficiently

**optuna (4.4.0)**
- **Purpose**: Hyperparameter optimization framework
- **Our Use**: Automatic hyperparameter search
- **Algorithms**:
  - Tree-structured Parzen Estimator (TPE)
  - CMA-ES
  - Random search
  - Grid search
- **Integration**: Works with any ML framework

### Progress Monitoring

**tqdm (4.67.1)**
- **Purpose**: Progress bars for loops
- **Our Use**: Training progress, data loading
- **Features**: Nested bars, notebook support, ETA

**rich (14.0.0)**
- **Purpose**: Beautiful terminal formatting
- **Our Use**: Colored logs, formatted tables, trees
- **Benefits**: Better debugging, clearer output

### API Development

**fastapi (0.115.0) & uvicorn (0.34.3)**
- **Purpose**: High-performance API framework
- **Our Use**: Model serving endpoints
- **Features**:
  - Automatic API documentation
  - Async support for high throughput
  - Type hints validation
- **Performance**: Handles thousands of requests/sec

## Quantization & Inference Dependencies

Specialized tools for inference optimization:

**nemo-toolkit (2.0.0)**
- **Purpose**: NVIDIA's neural modules toolkit
- **Our Use**: Speech and NLP model optimization
- **Features**: Pre-trained models, optimization recipes
- **GPU Optimization**: TensorRT integration

**llama-cpp-python (0.3.14)**
- **Purpose**: Python bindings for llama.cpp
- **Our Use**: CPU/GPU inference with quantized models
- **Performance**: Enables our 286.85 tok/s inference
- **Integration**: Direct GGUF model loading

**Supporting Libraries:**
- `tiktoken (0.9.0)` - OpenAI's tokenizer
- `rouge-score (0.1.2)` - Text generation metrics
- `transformers-stream-generator (0.0.5)` - Streaming generation
- `sseclient (0.0.27)` - Server-sent events for streaming

## PyTorch Special Handling

### Why PyTorch is Different

PyTorch requires special handling due to CUDA version dependencies:

```python
# NOT in pyproject.toml dependencies
# Installed manually via pip with CUDA version
torch = "2.3.0+cu129"  # CUDA 12.9 version
```

### CUDA Compatibility

Our setup uses CUDA 13.0, but PyTorch currently supports up to CUDA 12.9:

| Component | Version | Compatibility |
|-----------|---------|--------------|
| **System CUDA** | 13.0.88 | Latest |
| **PyTorch CUDA** | 12.9 | Backward compatible |
| **cuDNN** | 9.13.0 | Cross-compatible |

**Installation Process:**
```bash
# Manual installation with correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

See [pytorch_cuda_update_guide.md](pytorch_cuda_update_guide.md) for detailed instructions.

## Development Dependencies

### Code Quality Tools

**black (23.0.0)**
- **Purpose**: Opinionated code formatter
- **Configuration**: 88 character line length
- **Benefit**: Consistent code style, no debates

**ruff (0.1.0)**
- **Purpose**: Fast Python linter (Rust-based)
- **Checks**: pycodestyle, pyflakes, isort, more
- **Speed**: 10-100x faster than traditional linters

**mypy (1.0.0)**
- **Purpose**: Static type checking
- **Benefit**: Catch errors before runtime
- **Integration**: Works with type hints

### Testing Framework

**pytest Suite:**
- `pytest (7.0.0)` - Testing framework
- `pytest-asyncio (0.21.0)` - Async test support
- `pytest-mock (3.11.0)` - Mocking utilities
- `pytest-xdist (3.7.0)` - Parallel test execution
- `pytest-cov (6.2.0)` - Coverage reporting

## Integration with GPU Stack

Our Python dependencies leverage the GPU stack for acceleration:

```
Python Package          GPU Stack Component    Performance Impact
─────────────          ──────────────────    ─────────────────
PyTorch                → CUDA 13.0           → 8x speedup
transformers           → cuDNN 9.13          → Flash Attention
bitsandbytes          → CUDA Kernels        → 4-bit quantization
accelerate            → NCCL                → Multi-GPU support
deepspeed             → CUDA + cuDNN        → ZeRO optimization
```

### Memory Management

Python packages must coordinate with GPU memory:

```python
# Example: Loading model with memory awareness
import torch
from transformers import AutoModel

# Check available VRAM
print(f"Available VRAM: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")

# Load model with appropriate precision
model = AutoModel.from_pretrained(
    "model_name",
    torch_dtype=torch.float16,  # Use FP16 to save memory
    device_map="auto"  # Automatic device placement
)
```

## Common Tasks

### Daily Operations

```bash
# Install all dependencies
poetry install

# Activate virtual environment
poetry shell

# Run Python script
poetry run python script.py

# Add new dependency
poetry add pandas

# Add dev dependency
poetry add --group dev pytest

# Update specific package
poetry update transformers

# Show installed packages
poetry show

# Show dependency tree
poetry show --tree
```

### Dependency Updates

```bash
# Update all dependencies (respecting version constraints)
poetry update

# Update lock file without installing
poetry lock

# Check outdated packages
poetry show --outdated

# Export requirements.txt (for compatibility)
poetry export -f requirements.txt --output requirements.txt
```

## Troubleshooting

### Common Issues

#### 1. PyTorch CUDA Mismatch
```bash
# Error: "CUDA error: no kernel image available"
# Solution: Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

#### 2. Poetry Lock Conflicts
```bash
# Error: "SolverProblemError"
# Solution: Clear cache and re-lock
poetry cache clear pypi --all
poetry lock --no-update
poetry install
```

#### 3. Virtual Environment Issues
```bash
# Poetry not using correct Python version
poetry env remove python
poetry env use python3.12
poetry install
```

#### 4. Memory Errors
```python
# Error: "CUDA out of memory"
# Solution: Use gradient accumulation or smaller batch size
import torch
torch.cuda.empty_cache()  # Clear cache

# Or use automatic mixed precision
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="fp16")
```

### Performance Optimization

#### Dependency Loading Speed
```bash
# Use parallel installation
poetry config installer.parallel true

# Pre-download packages
poetry export -f requirements.txt | pip download -r /dev/stdin -d ./pip-cache
```

#### Import Optimization
```python
# Lazy imports for faster startup
def get_transformers():
    global transformers
    import transformers
    return transformers
```

## Best Practices

1. **Version Pinning**
   - Pin minor versions for stability
   - Use `~=` for compatible releases
   - Document why specific versions are required

2. **Dependency Groups**
   ```toml
   [tool.poetry.group.dev]  # Development only
   [tool.poetry.group.test]  # Testing only
   [tool.poetry.group.docs]  # Documentation
   ```

3. **Security Updates**
   ```bash
   # Regular security audits
   poetry audit
   ```

4. **Performance Considerations**
   - Keep dependencies minimal
   - Use optional dependencies for rare features
   - Regular cleanup of unused packages

5. **Consistency**
   - Always commit `poetry.lock`
   - Document Python version requirements
   - Use Docker for full environment capture

## Next Steps

- Review [setup_python.md](setup_python.md) for Python environment setup
- See [pytorch_cuda_update_guide.md](pytorch_cuda_update_guide.md) for PyTorch management
- Explore [GPU Stack Documentation](../gpu-stack/README.md) for acceleration details
- Check [Inference Optimization](../../inference/README.md) for performance tuning

---

