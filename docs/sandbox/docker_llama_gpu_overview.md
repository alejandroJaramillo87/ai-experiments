# AI Engineering GPU Docker Implementation

Explanation of the RTX 5090 Blackwell optimized Dockerfile for GPU-accelerated AI model inference on the AI workstation, detailing the multi-stage build process, CUDA 13.0.2 integration, and Blackwell architecture compilation.

> Note: This document explains the actual implementation in `docker/llama-gpu/Dockerfile.llama-gpu`. The Dockerfile uses CUDA 13.0.2 with cuDNN and Blackwell sm_120a architecture targeting for RTX 5090 GPU acceleration.

## Table of Contents

<!-- toc:start -->

1. [Implementation Overview](#implementation-overview)
2. [Stage 1: Builder Environment](#stage-1-builder-environment)
   1. [CUDA Development Base](#cuda-development-base)
   2. [RTX 5090 Blackwell Configuration](#rtx-5090-blackwell-configuration)
   3. [CUDA Components Installation](#cuda-components-installation)
   4. [CUDA Environment Setup](#cuda-environment-setup)
   5. [llama.cpp CUDA Compilation](#llamacpp-cuda-compilation)
3. [Stage 2: Runtime Environment](#stage-2-runtime-environment)
   1. [CUDA Runtime Base](#cuda-runtime-base)
   2. [Runtime Dependencies](#runtime-dependencies)
   3. [GPU Performance Tuning](#gpu-performance-tuning)
   4. [Security Configuration](#security-configuration)
   5. [Application Deployment](#application-deployment)
4. [Entrypoint Configuration](#entrypoint-configuration)
   1. [Basic Environment Variables](#basic-environment-variables)
   2. [GPU/CPU Split Configuration](#gpucpu-split-configuration)
   3. [Advanced Server Features](#advanced-server-features)
   4. [Nsight Systems Profiling](#nsight-systems-profiling)
   5. [Server Flags](#server-flags)
5. [GPU Performance Optimization Strategy](#gpu-performance-optimization-strategy)
   1. [Blackwell Architecture Targeting](#blackwell-architecture-targeting)
   2. [CUDA Memory Management](#cuda-memory-management)
   3. [Compute Optimization](#compute-optimization)
   4. [Inference Configuration](#inference-configuration)
6. [Container Security Implementation](#container-security-implementation)
7. [Tested But Ineffective Optimizations](#tested-but-ineffective-optimizations)
8. [Reference Implementation](#reference-implementation)

<!-- toc:end -->

## Implementation Overview

The GPU Docker implementation leverages a multi-stage build strategy specifically optimized for the RTX 5090 Blackwell architecture. The implementation utilizes CUDA 13.0.2 with cuDNN for maximum compatibility with the latest GPU features and provides AI inference capabilities.

**Key Implementation Features:**
- **Multi-stage build**: Separates CUDA development tools from runtime environment
- **Blackwell optimization**: sm_120a architecture targeting for RTX 5090 maximum performance
- **CUDA 13.0.2 with cuDNN**: Base images include cuDNN for deep learning acceleration
- **CMake build system**: Modern build configuration for llama.cpp CUDA compilation
- **Entrypoint script**: Parameterized server configuration via environment variables
- **Nsight Systems integration**: Optional deep profiling support
- **Security hardening**: Non-root user execution with minimal runtime dependencies

## Stage 1: Builder Environment

### CUDA Development Base

**NVIDIA CUDA Development Foundation**
```dockerfile
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS builder
```
- **Purpose**: Provides complete CUDA development environment with compiler tools and cuDNN
- **Version**: CUDA 13.0.2 ensures RTX 5090 Blackwell architecture compatibility
- **cuDNN included**: Deep learning primitives included in base image
- **Base OS**: Ubuntu 24.04 LTS provides stable foundation with modern libraries

### RTX 5090 Blackwell Configuration

**Blackwell Architecture Environment Variables**
```dockerfile
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_DOCKER_ARCH=sm_120a
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV CUDA_ARCHITECTURES=120a
ENV CUDAFLAGS="-O3 --use_fast_math -arch=sm_120a"
ENV LDFLAGS="-Wl,-O3 -Wl,--as-needed -L/usr/local/cuda/lib64/stubs"
```

**Architecture-Specific Optimization**
- **CUDA_DOCKER_ARCH=sm_120a**: Targets RTX 5090 Blackwell compute capability 12.0. The `a`
  suffix selects the architecture-accelerated variant, not plain `sm_120`
- **TORCH_CUDA_ARCH_LIST="12.0"**: PyTorch/tensor operation optimization for Blackwell
- **CUDA_ARCHITECTURES=120a**: carried for tools that read it; the llama.cpp build below does
  **not** select the target this way
- **CUDAFLAGS**: NVCC compiler flags with fast math optimizations for inference
- **Aggressive optimization**: -O3 and --use_fast_math for maximum performance

### CUDA Components Installation

**CUDA Development Stack**
```dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        gcc g++ build-essential cmake git curl ca-certificates \
        libssl-dev pkg-config python3 python3-pip \
        ninja-build ccache \
        cuda-driver-dev-13-0 \
        cuda-cudart-dev-13-0 \
        cuda-cupti-dev-13-0 \
        cuda-nvml-dev-13-0 \
        cuda-nvtx-13-0 \
        libcublas-dev-13-0
```

**CUDA Component Selection**
- **cuda-driver-dev-13-0**: CUDA driver development headers
- **cuda-cudart-dev-13-0**: CUDA runtime API development libraries
- **cuda-cupti-dev-13-0**: CUDA Profiling Tools Interface for performance analysis
- **cuda-nvml-dev-13-0**: NVIDIA Management Library for GPU monitoring
- **cuda-nvtx-13-0**: NVIDIA Tools Extension for profiling and debugging
- **libcublas-dev-13-0**: CUDA Basic Linear Algebra Subroutines for matrix operations

Note: cuDNN is included in the base image (`nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04`) and does not need separate installation.

### CUDA Environment Setup

**CUDA Development Environment**
```dockerfile
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV PKG_CONFIG_PATH=/usr/local/cuda/lib64/pkgconfig:${PKG_CONFIG_PATH}

ENV CUDA_MODULE_LOADING=LAZY
ENV CUDA_DEVICE_ORDER=FASTEST_FIRST

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
```

**Environment Configuration**
- **CUDA_MODULE_LOADING=LAZY**: Defers CUDA module loading for faster container startup
- **CUDA_DEVICE_ORDER=FASTEST_FIRST**: Ensures RTX 5090 is selected when multiple GPUs present
- **Library stub linking**: Enables compilation in containerized environment without GPU driver

### llama.cpp CUDA Compilation

**CUDA-Optimized Build Process**
```dockerfile
# Pin to an exact commit for reproducible builds. Bump deliberately, not on rebuild.
ARG LLAMA_CPP_COMMIT=e3546c7948e3af463d0b401e6421d5a4c2faf565   # tag b9976 (2026-07-11)
RUN rm -rf /tmp/llama.cpp && \
    git clone https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp && \
    cd /tmp/llama.cpp && \
    git checkout "${LLAMA_CPP_COMMIT}" && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=OFF \
        -DCMAKE_CUDA_FLAGS="-gencode arch=compute_120a,code=sm_120a" \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FORCE_CUBLAS=ON \
        -DGGML_CUDA_GRAPHS=ON \
        -DGGML_CUDA_PEER_MAX_BATCH_SIZE=256 \
        -DGGML_CUDA_NO_VMM=OFF \
        -DGGML_NATIVE=ON \
        -DGGML_LTO=ON \
        -DGGML_BUILD_TESTS=OFF \
        -DGGML_BUILD_EXAMPLES=OFF \
        -DLLAMA_CURL=OFF \
        -DGGML_CCACHE=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON && \
     cmake --build . --target llama-server --config Release -j$(nproc)
```

**CUDA Build Configuration**
- **CMAKE_CUDA_ARCHITECTURES=OFF** with an explicit
  `-DCMAKE_CUDA_FLAGS="-gencode arch=compute_120a,code=sm_120a"`: the Blackwell target is
  forced through the gencode flag rather than selected by CMake. `Dockerfile.llama-gpu:35-37`
  records why — CMake/ggml arch selection "proved unreliable in this image (kept emitting
  compute_120)". Setting `CMAKE_CUDA_ARCHITECTURES` to a value here would reintroduce that bug
- **LLAMA_CPP_COMMIT**: the build is pinned to an exact llama.cpp commit, not to `master`.
  Bump it deliberately; a rebuild must not move it
- **CMAKE_VERSION=3.31.6**: CMake is installed from a pinned Kitware release rather than apt
  (`Dockerfile.llama-gpu:38`)
- **GGML_CUDA=ON**: Enable CUDA acceleration throughout llama.cpp
- **GGML_CUDA_FORCE_CUBLAS=ON**: Force cuBLAS usage for maximum performance
- **GGML_CUDA_GRAPHS=ON**: Enable CUDA graphs for reduced kernel launch overhead
- **GGML_CUDA_NO_VMM=OFF**: Enable Virtual Memory Management for large models
- **GGML_CUDA_FA_ALL_QUANTS=ON**: FlashAttention support for all quantization formats
- **GGML_CUDA_PEER_MAX_BATCH_SIZE=256**: Optimized batch size for peer-to-peer transfers
- **GGML_CCACHE=ON**: Enable compiler cache for faster rebuilds

## Stage 2: Runtime Environment

### CUDA Runtime Base

**Optimized Runtime Foundation**
```dockerfile
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04
```
- **Runtime focus**: Minimal CUDA runtime without development overhead
- **cuDNN included**: Deep learning primitives included in runtime base image
- **Version consistency**: Matching CUDA 13.0.2 for binary compatibility
- **Size optimization**: Excludes development tools and headers

### Runtime Dependencies

**Essential CUDA Runtime Libraries**
```dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        ca-certificates \
        libgomp1 \
        libcublas-13-0 \
        cuda-cupti-13-0 \
        cuda-nvtx-13-0 \
        nsight-systems-2025.3.2
```

**Runtime Components**
- **libgomp1**: OpenMP runtime library for parallel execution
- **libcublas-13-0**: CUDA Basic Linear Algebra Subroutines runtime
- **cuda-cupti-13-0**: Performance monitoring and profiling capabilities
- **cuda-nvtx-13-0**: NVIDIA Tools Extension for runtime analysis
- **nsight-systems-2025.3.2**: Deep profiling tool (optional, enabled via NSYS_PROFILE)

Note: cuDNN runtime is included in the base image and does not need separate installation.

### GPU Performance Tuning

**RTX 5090 Performance Configuration**
```dockerfile
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_MODULE_LOADING=LAZY
ENV CUDA_DEVICE_ORDER=FASTEST_FIRST

# Performance tuning for RTX 5090 (Blackwell, 32GB VRAM)
ENV CUDA_LAUNCH_BLOCKING=0
ENV CUDNN_LOGINFO_DBG=0
ENV CUDNN_LOGDEST_DBG=stderr
ENV CUDA_CACHE_DISABLE=0
ENV CUDA_CACHE_PATH=/tmp/cuda_cache
ENV CUDA_FORCE_PTX_JIT=0

# Enable CUDA graphs for improved performance
ENV GGML_CUDA_GRAPHS=1
# Optimize for single stream processing
ENV GGML_CUDA_MAX_STREAMS=1
# Force synchronous operations for lower latency
ENV GGML_CUDA_FORCE_SYNC=0
```

**Performance Optimization Variables**
- **CUDA_LAUNCH_BLOCKING=0**: Enable asynchronous kernel launches for performance
- **CUDA_CACHE_DISABLE=0**: Enable kernel caching for repeated operations
- **CUDA_FORCE_PTX_JIT=0**: Disable forced JIT compilation for faster startup
- **GGML_CUDA_GRAPHS=1**: Enable CUDA graphs for reduced kernel overhead
- **GGML_CUDA_MAX_STREAMS=1**: Optimize for single-stream inference workloads

**NVIDIA Container Configuration**
```dockerfile
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV NVIDIA_REQUIRE_CUDA="cuda>=13.0"
```

Note: `NVIDIA_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are set via docker-compose.inference.yaml, not in the Dockerfile.

### Security Configuration

**Non-Root User Implementation**
```dockerfile
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "Application User" appuser
USER appuser
WORKDIR /app
```

**Cache Directory Setup**
```dockerfile
RUN mkdir -p /tmp/cuda_cache
```

Note: Cache directory is created after switching to appuser for proper ownership.

### Application Deployment

**GPU-Optimized Binary Installation**
```dockerfile
COPY --from=builder --chown=appuser:appuser /tmp/llama.cpp/build/bin/llama-server /app/server
COPY --from=builder --chown=appuser:appuser /tmp/llama.cpp/build/bin/* /app/
COPY --chown=appuser:appuser docker/llama-gpu/entrypoint.sh /app/entrypoint.sh

EXPOSE 8002

ENTRYPOINT ["/app/entrypoint.sh"]
```

The server is started via the entrypoint script, which provides parameterized configuration through environment variables.

## Entrypoint Configuration

The entrypoint script (`entrypoint.sh`) provides parameterized server configuration with environment variable overrides.

### Basic Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_PORT` | 8002 | HTTP API server port |
| `MODEL_PATH` | (hardcoded path) | Path to GGUF model file |
| `SERVER_HOST` | 0.0.0.0 | Bind address for server |
| `N_GPU_LAYERS` | 999 | Layers to offload to GPU |
| `CTX_SIZE` | 65536 | Context window size |
| `BATCH_SIZE` | 2048 | Logical batch size |
| `UBATCH_SIZE` | 512 | Physical batch size |
| `THREADS` | 1 | CPU threads for inference |
| `THREADS_BATCH` | 1 | CPU threads for batch processing |
| `THREADS_HTTP` | 4 | HTTP handler threads |
| `PARALLEL` | 1 | Parallel request slots |
| `MAIN_GPU` | 0 | Primary GPU index |
| `FLASH_ATTN` | on | Flash attention mode |
| `CONT_BATCHING` | false | Enable continuous batching |

### GPU/CPU Split Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SPLIT_MODE` | none | Split mode (none/layer/row) |
| `TENSOR_SPLIT` | "" | Tensor split ratio for multi-GPU |
| `NO_KV_OFFLOAD` | false | Disable KV cache GPU offload |

### Advanced Server Features

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_REUSE` | 0 | KV cache reuse threshold |
| `JINJA_ENABLED` | false | Enable Jinja template processing |
| `REASONING_FORMAT` | "" | Reasoning output format |
| `KV_UNIFIED` | false | Enable unified KV buffer |

### Nsight Systems Profiling

| Variable | Default | Description |
|----------|---------|-------------|
| `NSYS_PROFILE` | false | Enable Nsight Systems profiling |
| `NSYS_OUTPUT` | /app/nsys_output/profile | Profile output path |

When `NSYS_PROFILE=true`, the server is wrapped with nsys for deep CUDA profiling:
```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --sample=cpu \
    --output="$NSYS_OUTPUT" \
    --force-overwrite=true \
    --stats=true \
    -- ./server ...
```

### Server Flags

The entrypoint builds command arguments dynamically. Always included:
```bash
./server \
    --model "$MODEL_PATH" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --ctx-size "$CTX_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --ubatch-size "$UBATCH_SIZE" \
    --threads "$THREADS" \
    --threads-batch "$THREADS_BATCH" \
    --metrics \
    --no-warmup \
    --threads-http "$THREADS_HTTP" \
    --flash-attn "$FLASH_ATTN" \
    --no-mmap \
    --main-gpu "$MAIN_GPU" \
    --parallel "$PARALLEL"
```

**Optional flags added based on environment:**
- `--cont-batching` (if CONT_BATCHING=true)
- `--split-mode $SPLIT_MODE` (if SPLIT_MODE != none)
- `--tensor-split $TENSOR_SPLIT` (if TENSOR_SPLIT is set)
- `--no-kv-offload` (if NO_KV_OFFLOAD=true)
- `--cache-reuse $CACHE_REUSE` (if CACHE_REUSE > 0)
- `--jinja` (if JINJA_ENABLED=true)
- `--reasoning-format $REASONING_FORMAT` (if REASONING_FORMAT is set)
- `--kv-unified` (if KV_UNIFIED=true)

**Entrypoint Features**
- Validates model file exists before starting
- Reports GPU memory status via nvidia-smi before model load
- Displays configuration summary on startup
- Supports optional Nsight Systems deep profiling

## GPU Performance Optimization Strategy

### Blackwell Architecture Targeting

**Compute Capability 12.0 Utilization**
The implementation targets specific RTX 5090 Blackwell capabilities:
- **sm_120a architecture**: Maximizes utilization of Blackwell compute units
- **Advanced tensor operations**: Leverages fourth-generation RT cores and third-generation Tensor cores
- **Memory bandwidth**: Optimized for 896 GB/s memory bandwidth with GDDR7
- **32GB VRAM**: Configured for large model inference with full GPU memory utilization

**Compiler Optimization Pipeline**
1. **Architecture targeting**: `-arch=sm_120a` generates Blackwell-specific CUDA code
2. **Fast math**: `--use_fast_math` enables hardware-accelerated mathematical operations
3. **Aggressive optimization**: -O3 level optimization for maximum performance
4. **FlashAttention**: Hardware-accelerated attention mechanisms for transformer models

### CUDA Memory Management

**VRAM Optimization Strategy**
- **32GB VRAM utilization**: Full model loading with `--n-gpu-layers=999`
- **No memory mapping**: `--no-mmap` forces direct GPU memory allocation
- **Batch optimization**: Asymmetric batching (2048/512) optimizes for Blackwell memory hierarchy

**CUDA Graphs and Kernel Optimization**
- **CUDA graphs**: Reduced kernel launch overhead through graph capture and replay
- **Kernel caching**: Persistent kernel cache at `/tmp/cuda_cache` for repeated operations
- **Stream optimization**: Single stream configuration optimized for inference latency
- **Virtual Memory Management**: Enabled for efficient large model handling

### Compute Optimization

**Inference-Specific Configuration**
- **Minimal CPU threading**: Single CPU thread (`--threads=1`) to maximize GPU utilization
- **GPU-first processing**: All compute layers offloaded to GPU (`--n-gpu-layers=999`)
- **FlashAttention**: Hardware-accelerated attention for transformer efficiency
- **Quantization support**: Full quantization format support with FlashAttention acceleration

**Memory Access Optimization**
- **Batch processing**: Large context size (65536) with optimized batch dimensions
- **Peer-to-peer**: Configured for multi-GPU environments (future expansion)

### Inference Configuration

**Server Performance Tuning**
- **HTTP threading**: 4 dedicated threads for API request handling
- **No warmup**: Immediate model availability without pre-computation overhead
- **Metrics enabled**: Performance monitoring and profiling capabilities
- **Main GPU specification**: Explicit GPU 0 targeting for multi-GPU systems

**Model Loading Strategy**
- **Direct GPU loading**: All model layers loaded into VRAM for maximum speed
- **Context window**: 65K context optimized for Blackwell memory subsystem
- **Parallel processing**: Single parallel stream optimized for latency-critical inference

## Container Security Implementation

**Multi-Layer Security Approach**
- **Minimal runtime**: Only essential CUDA libraries included in final image
- **Non-root execution**: Application runs as dedicated non-privileged user
- **No interactive access**: User account configured without shell access
- **Controlled GPU access**: Explicit GPU device specification and capability restrictions
- **Cache isolation**: CUDA cache confined to container temporary directory

**Docker Compose Integration**
When deployed via docker-compose, additional security measures include:
- **Capability dropping**: All Linux capabilities removed except GPU access
- **Privilege restrictions**: `no-new-privileges:true` prevents escalation
- **Memory constraints**: Unlimited memlock for CUDA operations with system oversight
- **Read-only containers**: Filesystem mounted read-only except for designated cache areas

## Tested But Ineffective Optimizations

**Runtime Environment Variables**
The following runtime environment variables are general CUDA and framework tuning knobs
that were evaluated as candidates but are deliberately not set in this service's
`docker/llama-gpu/Dockerfile.llama-gpu`, because they provided no performance improvement
for llama.cpp inference during external testing:

- **Memory Pool Optimizations**: CUDA_ALLOCATOR_BACKEND, CUDA_MALLOC_ASYNC_POOLS, PYTORCH_CUDA_ALLOC_CONF
- **Tensor Core Settings**: CUDA_TENSOR_CORES, CUBLAS_WORKSPACE_CONFIG, CUDNN_TENSOR_OPS
- **Blackwell Architecture Settings**: CUDA_L2_PERSISTING_SIZE, CUDA_DEVICE_MAX_CONNECTIONS, CUDA_COPY_SPLIT_THRESHOLD

These variables primarily benefit PyTorch and Python-based inference frameworks. The
llama.cpp implementation is already optimized at compile time through the CMake flags in
the Dockerfile, so the Dockerfile sets only the CUDA and GGML variables that affect it
(for example CUDA_MODULE_LOADING, CUDA_CACHE_PATH, and the GGML_CUDA_* settings).

**Important Performance Warning**
GPU clock locking (nvidia-smi -lgc/-lmc) causes a 38% performance degradation on RTX 5090 and should never be enabled.

## Reference Implementation

**File Structure**
- **docker/llama-gpu/Dockerfile.llama-gpu**: Complete multi-stage GPU-optimized Dockerfile
- **docker/llama-gpu/entrypoint.sh**: Parameterized server startup script
- **docker-compose.inference.yaml**: Container orchestration with GPU resource allocation and security hardening

**Integration with AI Workstation**
This GPU implementation works in conjunction with:
- **CPU containers**: Parallel execution for hybrid CPU/GPU workloads
- **Load balancing**: GPU container handles large models while CPU containers manage concurrent smaller models
- **Resource isolation**: Exclusive GPU access with controlled VRAM allocation
- **Performance monitoring**: CUDA profiling integration for optimization analysis

**Hardware Requirements**
- **GPU**: RTX 5090 32GB (Blackwell architecture)
- **CUDA**: Version 13.0 or higher driver support
- **System Memory**: Sufficient for container and CUDA context overhead
- **Storage**: Fast storage for model loading and CUDA cache operations

---

*This GPU Docker implementation provides AI model inference on the RTX 5090 Blackwell architecture. The multi-stage build process integrates CUDA while maintaining security and operational efficiency for high-throughput AI workloads.*
