# GPU Stack for AI Inference

Understanding the NVIDIA GPU software stack that powers our AI experiments workstation.

## Table of Contents

- [Overview](#overview)
- [The Three-Layer Stack](#the-three-layer-stack)
- [Our Production Environment](#our-production-environment)
- [Quick Start](#quick-start)
  - [Verify Installation](#verify-installation)
  - [Installation Scripts](#installation-scripts)
- [Why Each Layer Matters](#why-each-layer-matters)
  - [For AI Inference Performance](#for-ai-inference-performance)
  - [Real Impact on Our Workload](#real-impact-on-our-workload)
- [Component Documentation](#component-documentation)
- [Common Issues and Solutions](#common-issues-and-solutions)
  - [Driver Issues](#driver-issues)
  - [CUDA Issues](#cuda-issues)
  - [cuDNN Issues](#cudnn-issues)
- [Performance Verification](#performance-verification)
- [Architecture-Specific Features](#architecture-specific-features)
- [Next Steps](#next-steps)

## Overview

Our AI inference workstation achieves **286.85 tokens/second** on the RTX 5090 through a carefully configured GPU software stack. This documentation explains each layer of the stack and how they work together to enable high-performance AI computing.

## The Three-Layer Stack

```
┌─────────────────────────────────────────────┐
│     Application Layer (Your AI Models)      │
│   llama.cpp, PyTorch, TensorFlow, vLLM     │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│         cuDNN 9.13.0 (Deep Learning)        │
│  Optimized neural network primitives:       │
│  • Flash Attention • Convolutions           │
│  • Batch Normalization • Activations        │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│       CUDA 13.0.88 (GPU Programming)        │
│  Parallel compute platform:                 │
│  • Tensor Cores • CUDA Cores               │
│  • Memory Management • Kernel Execution     │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│    NVIDIA Driver 580.65.06 (Hardware)       │
│  Hardware interface and resource control:   │
│  • PCIe Gen5 • Power Management (600W)     │
│  • VRAM Access (32GB) • GPU Scheduling     │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│         RTX 5090 (Physical GPU)             │
│  • 21,760 CUDA Cores                       │
│  • 32GB GDDR7 VRAM                         │
│  • 5th Gen Tensor Cores                    │
└─────────────────────────────────────────────┘
```

## Our Production Environment

| Component | Version | Purpose |
|-----------|---------|---------|
| **NVIDIA Driver** | 580.65.06 | Hardware interface, GPU resource management |
| **CUDA Toolkit** | 13.0.88 | GPU programming platform, parallel computing |
| **cuDNN** | 9.13.0.50-1 | Deep learning acceleration library |

## Quick Start

### Verify Installation

Check your current GPU stack status:

```bash
# Check all components at once
nvidia-smi  # Shows driver version, GPU status, VRAM usage

# Check CUDA version
nvcc --version

# Check cuDNN version
apt list --installed | grep cudnn
```

### Installation Scripts

Our automated setup scripts handle the complete stack installation:

```bash
# Install NVIDIA Driver
scripts/setup/setup_nvidia.sh

# Install CUDA Toolkit
scripts/setup/setup_cuda.sh

# Install cuDNN
scripts/setup/setup_cudnn.sh
```

## Why Each Layer Matters

### For AI Inference Performance

1. **Driver Layer**: Manages the physical GPU, enabling access to 32GB of high-bandwidth memory
2. **CUDA Layer**: Provides parallel processing for matrix operations (core of neural networks)
3. **cuDNN Layer**: Optimizes specific deep learning operations, improving performance by 30-50%

### Real Impact on Our Workload

- **Without GPU stack**: ~35 tokens/sec (CPU only)
- **With complete stack**: ~287 tokens/sec (8x faster)
- **Memory efficiency**: Can load 20B parameter models entirely in VRAM
- **Batch processing**: Optimal batch size of 2048 tokens

## Component Documentation

Detailed documentation for each layer:

1. **[NVIDIA Drivers](nvidia-drivers.md)** - Hardware interface and GPU management
2. **[CUDA Platform](cuda.md)** - Parallel computing and GPU programming
3. **[cuDNN Library](cudnn.md)** - Deep learning optimization layer

## Common Issues and Solutions

### Driver Issues
```bash
# Driver not loaded
nvidia-smi  # Should show GPU, not "No devices found"

# Fix: Reinstall driver
sudo apt-get purge nvidia-*
scripts/setup/setup_nvidia.sh
```

### CUDA Issues
```bash
# CUDA not found
nvcc --version  # Should show CUDA 13.0.88

# Fix: Update PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### cuDNN Issues
```bash
# cuDNN not detected by PyTorch
python -c "import torch; print(torch.backends.cudnn.version())"

# Fix: Reinstall cuDNN
scripts/setup/setup_cudnn.sh
```

## Performance Verification

After installation, verify performance:

```bash
# Run GPU benchmark
scripts/benchmark.py --service llama-gpu --label "gpu_test"

# Expected performance
# ~287 tokens/second with optimal configuration
```

## Architecture-Specific Features

Our RTX 5090 (Blackwell architecture) provides:

- **SM 12.0 Compute Capability**: Latest CUDA features
- **5th Gen Tensor Cores**: FP8 precision for faster inference
- **PCIe Gen5**: 128 GB/s bidirectional bandwidth
- **32GB GDDR7**: High-bandwidth memory for large models

## Next Steps

1. Read component-specific documentation for deeper understanding
2. Review optimization guides in `docs/optimizations/gpu/`
3. Run benchmarks to verify your setup
4. Explore inference configurations in `docs/inference/`

---

*This GPU stack powers our AI experiments workstation, enabling production-ready inference at 286.85 tokens/second on the RTX 5090.*