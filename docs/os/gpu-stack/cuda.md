# CUDA - The Parallel Computing Platform

Understanding CUDA and how it transforms GPUs into AI acceleration engines.

## Table of Contents

- [What is CUDA?](#what-is-cuda)
  - [The Parallel Computing Paradigm](#the-parallel-computing-paradigm)
  - [Key CUDA Concepts](#key-cuda-concepts)
- [Why Essential for AI Workloads?](#why-essential-for-ai-workloads)
  - [Matrix Operations - The Heart of AI](#matrix-operations---the-heart-of-ai)
  - [Our llama.cpp Inference](#our-llamacpp-inference)
- [Our Specific Setup: CUDA 13.0.88](#our-specific-setup-cuda-13088)
  - [Why CUDA 13.0?](#why-cuda-130)
  - [Installation and Configuration](#installation-and-configuration)
  - [Environment Configuration](#environment-configuration)
- [Impact on Our AI Workloads](#impact-on-our-ai-workloads)
  - [Performance Comparison](#performance-comparison)
  - [Real Example: Model Loading](#real-example-model-loading)
  - [Batch Processing Optimization](#batch-processing-optimization)
- [CUDA Programming Model](#cuda-programming-model)
  - [Grid, Blocks, and Threads](#grid-blocks-and-threads)
  - [Memory Access Patterns](#memory-access-patterns)
- [CUDA Libraries We Use](#cuda-libraries-we-use)
  - [cuBLAS - Linear Algebra](#cublas---linear-algebra)
  - [cuDNN - Deep Learning](#cudnn---deep-learning)
  - [NCCL - Multi-GPU Communication](#nccl---multi-gpu-communication)
- [Troubleshooting CUDA](#troubleshooting-cuda)
  - [Common Issues](#common-issues)
- [Monitoring CUDA Performance](#monitoring-cuda-performance)
  - [Real-time Monitoring](#real-time-monitoring)
  - [Key Metrics](#key-metrics)
- [CUDA and Container Optimization](#cuda-and-container-optimization)
- [Advanced CUDA Features We Use](#advanced-cuda-features-we-use)
  - [1. CUDA Graphs](#1-cuda-graphs)
  - [2. Unified Memory](#2-unified-memory)
  - [3. Tensor Cores](#3-tensor-cores)
- [Best Practices](#best-practices)
- [Next Steps](#next-steps)

## What is CUDA?

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform that allows developers to use GPUs for general-purpose processing. Instead of just rendering graphics, CUDA enables your RTX 5090's 21,760 cores to work on AI computations simultaneously.

### The Parallel Computing Paradigm

Traditional CPU vs GPU with CUDA:

```
CPU (Sequential)                 GPU with CUDA (Parallel)
─────────────────                ─────────────────────────
Core 1: Task A (10ms)            21,760 Cores simultaneously:
Core 1: Task B (10ms)            ├── Tasks A1-A21760 (0.1ms)
Core 1: Task C (10ms)            ├── Tasks B1-B21760 (0.1ms)
Core 1: Task D (10ms)            └── Tasks C1-C21760 (0.1ms)
Total: 40ms                      Total: 0.3ms

Result: 133x speedup for parallelizable workloads
```

### Key CUDA Concepts

1. **CUDA Cores** - Individual processing units
   - RTX 5090 has 21,760 CUDA cores
   - Each can execute one floating-point operation per clock

2. **Streaming Multiprocessors (SMs)** - Groups of cores
   - RTX 5090 has 170 SMs (SM 12.0 architecture)
   - Each SM manages 128 CUDA cores

3. **Tensor Cores** - Specialized AI processors
   - 5th generation on RTX 5090
   - Accelerate matrix operations by 5x over CUDA cores

4. **Memory Hierarchy**
   ```
   Registers (fastest, per-thread)
       ↓
   Shared Memory (per-SM, ~100 KB)
       ↓
   L2 Cache (96 MB on RTX 5090)
       ↓
   Global Memory/VRAM (32 GB)
   ```

## Why Essential for AI Workloads?

### Matrix Operations - The Heart of AI

Neural networks are fundamentally matrix multiplications:

```python
# What happens in a neural network layer:
output = activation(weights @ input + bias)

# For a 4096x4096 matrix multiplication:
# CPU: ~500ms (sequential)
# GPU with CUDA: ~2ms (parallel)
# GPU with Tensor Cores: ~0.4ms (specialized hardware)
```

### Our llama.cpp Inference

When you run inference with `--n-gpu-layers 999`:

```
Model Layer Processing:
├── Attention Mechanism
│   ├── Q, K, V matrix multiplications → CUDA Tensor Cores
│   ├── Attention scores computation → CUDA Cores
│   └── Softmax normalization → Optimized CUDA kernels
├── Feed-Forward Network
│   ├── Linear transformations → Tensor Cores
│   └── Activation functions → CUDA Cores
└── Output Generation
    ├── Vocabulary projection → CUDA Cores
    └── Sampling → CPU (minimal work)

Result: 286.85 tokens/second
```

## Our Specific Setup: CUDA 13.0.88

### Why CUDA 13.0?

CUDA 13.0 is the minimum version supporting:
- **RTX 5090's SM 12.0** compute capability
- **5th Gen Tensor Cores** with FP8 precision
- **Hardware acceleration** for Flash Attention
- **Larger kernel grids** for bigger models

### Installation and Configuration

Our setup script (`scripts/setup/setup_cuda.sh`) installs:

```bash
# CUDA Toolkit components
/usr/local/cuda-13.0/
├── bin/
│   ├── nvcc         # CUDA compiler
│   ├── cuda-gdb     # CUDA debugger
│   └── nvprof       # Profiler
├── lib64/
│   ├── libcudart.so    # CUDA runtime
│   ├── libcublas.so    # Linear algebra library
│   └── libcudnn.so     # Deep learning primitives
└── include/
    └── cuda.h          # CUDA headers
```

### Environment Configuration

```bash
# Essential environment variables
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

# Performance tuning (in our Docker containers)
export CUDA_LAUNCH_BLOCKING=0    # Asynchronous kernel execution
export CUDA_DEVICE_ORDER=FASTEST_FIRST
export CUDA_VISIBLE_DEVICES=0    # Use RTX 5090
```

## Impact on Our AI Workloads

### Performance Comparison

| Configuration | Inference Speed | Relative Performance |
|--------------|----------------|---------------------|
| **CPU-only** (12 cores) | 35.44 tok/s | 1.0x (baseline) |
| **GPU with CUDA** | 286.85 tok/s | **8.1x faster** |
| **Without CUDA** (driver only) | 0 tok/s | No acceleration |

### Real Example: Model Loading

```python
# With CUDA - Direct GPU loading
import torch
model = torch.load("model.pt", map_location="cuda:0")
# Load time: 8.2 seconds for 20B parameters
# Memory: Stored in 32GB VRAM

# Without CUDA - CPU fallback
model = torch.load("model.pt", map_location="cpu")
# Load time: 45+ seconds
# Memory: Uses system RAM, constant swapping
```

### Batch Processing Optimization

Our optimal batch configuration leverages CUDA:

```python
# Batch size 2048 - fits in L2 cache
# Each token processed in parallel across CUDA cores
batch_tokens = 2048
cuda_cores = 21760

# Theoretical parallelism
operations_per_token = batch_tokens / cuda_cores  # 0.094
# Near-perfect parallelization for our workload
```

## CUDA Programming Model

Understanding how CUDA works helps optimize AI workloads:

### Grid, Blocks, and Threads

```
Kernel Launch Hierarchy:
Grid (entire computation)
├── Block 0 (runs on SM 0)
│   ├── Thread (0,0) - processes token 0
│   ├── Thread (0,1) - processes token 1
│   └── ... (up to 1024 threads)
├── Block 1 (runs on SM 1)
└── ... (up to 65535 blocks)

For our 2048 token batch:
- 2 blocks of 1024 threads each
- Executes on 2 SMs simultaneously
```

### Memory Access Patterns

Efficient CUDA programming requires coalesced memory access:

```cuda
// Good - Coalesced access (all threads access sequential memory)
output[threadIdx.x] = input[threadIdx.x] * weight;

// Bad - Random access (cache misses, 10x slower)
output[random_index[threadIdx.x]] = input[threadIdx.x] * weight;
```

## CUDA Libraries We Use

### cuBLAS - Linear Algebra
```bash
# Used for matrix multiplications in neural networks
# 95% of inference time is matrix ops
libcublas.so → Optimized BLAS operations on GPU
```

### cuDNN - Deep Learning
```bash
# Provides optimized implementations of:
# - Convolutions
# - Attention mechanisms
# - Normalization layers
# Covered in detail in cudnn.md
```

### NCCL - Multi-GPU Communication
```bash
# For future multi-GPU setups
# Enables efficient GPU-to-GPU communication
```

## Troubleshooting CUDA

### Common Issues

#### 1. CUDA Not Found
```bash
# Check installation
nvcc --version
# Should show: V13.0.88

# Fix PATH if missing
export PATH=/usr/local/cuda/bin:$PATH
```

#### 2. Version Mismatch
```bash
# Check compatibility
nvidia-smi  # Driver: 580.65.06
nvcc --version  # CUDA: 13.0.88

# These must be compatible
```

#### 3. Out of Memory
```cuda
# Error: CUDA out of memory
# Solution: Reduce batch size or context length
export CUDA_VISIBLE_DEVICES=0  # Ensure using correct GPU
nvidia-smi  # Check VRAM usage
```

#### 4. Slow Performance
```bash
# Verify CUDA is being used
nvidia-smi  # Should show llama-server process
# GPU-Util should be >90% during inference

# Check for CPU fallback
ldd $(which llama-server) | grep cuda
# Should show CUDA libraries linked
```

## Monitoring CUDA Performance

### Real-time Monitoring
```bash
# GPU utilization and memory
watch -n 1 nvidia-smi

# Detailed CUDA metrics
nvidia-smi dmon -s pucvmet

# Profile CUDA kernels
nsys profile --stats=true ./llama-server
```

### Key Metrics
- **GPU Utilization**: Should be >90% during inference
- **Memory Bandwidth**: ~1 TB/s on RTX 5090
- **SM Efficiency**: >80% for well-optimized code
- **Tensor Core Usage**: High for matrix operations

## CUDA and Container Optimization

Our Docker containers are CUDA-optimized:

```dockerfile
# From Dockerfile.llama-gpu
ENV CUDA_MODULE_LOADING=LAZY          # Faster startup
ENV CUDA_DEVICE_ORDER=FASTEST_FIRST   # Select RTX 5090
ENV GGML_CUDA_GRAPHS=1               # Enable CUDA graphs
ENV CUDA_LAUNCH_BLOCKING=0           # Asynchronous execution
```

These settings reduce kernel launch overhead by 20%.

## Advanced CUDA Features We Use

### 1. CUDA Graphs
Reduce kernel launch overhead by recording and replaying GPU operations:
```
Normal: Launch → Execute → Launch → Execute (overhead each time)
Graphs: Record → Replay → Replay → Replay (overhead once)
Result: 15-20% performance improvement
```

### 2. Unified Memory
Simplifies memory management between CPU and GPU:
```cuda
// Automatically migrates data between CPU/GPU as needed
cudaMallocManaged(&data, size);
```

### 3. Tensor Cores
Specialized hardware for AI:
- FP16 operations: 2x throughput
- FP8 operations: 4x throughput (new in RTX 5090)
- Automatic mixed precision in cuDNN

## Best Practices

1. **Always verify CUDA is active** before benchmarking
2. **Monitor GPU memory** to prevent OOM errors
3. **Use appropriate batch sizes** for your VRAM (2048 for 32GB)
4. **Keep CUDA and driver versions compatible**
5. **Profile your workloads** to identify bottlenecks

## Next Steps

- Continue to **[cuDNN Documentation](cudnn.md)** for deep learning optimizations
- Review **[CUDA Optimization Guide](../../optimizations/gpu/gpu-optimizations.md)**
- Explore **[Benchmarking Results](../../inference/README.md)**

---

*CUDA transforms the RTX 5090 from a graphics card into a parallel supercomputer, enabling our 286.85 tokens/second inference speed. It's the bridge between raw GPU hardware and AI applications.*