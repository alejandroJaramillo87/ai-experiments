# AI Engineering GPU Docker Implementation

Comprehensive explanation of the RTX 5090 Blackwell optimized Dockerfile for GPU-accelerated AI model inference on the AI workstation, detailing the multi-stage build process, CUDA 12.9.1 integration, and Blackwell architecture optimizations.

> Note: This document explains the actual implementation in `docker/Dockerfile.llama-gpu`. The Dockerfile uses CUDA 12.9.1 with Blackwell sm_120 architecture targeting and comprehensive GPU performance optimizations.

## Table of Contents

- [AI Engineering GPU Docker Implementation](#ai-engineering-gpu-docker-implementation)
  - [Table of Contents](#table-of-contents)
  - [Implementation Overview](#implementation-overview)
  - [Stage 1: Builder Environment](#stage-1-builder-environment)
    - [CUDA Development Base](#cuda-development-base)
    - [RTX 5090 Blackwell Configuration](#rtx-5090-blackwell-configuration)
    - [CUDA Components Installation](#cuda-components-installation)
    - [CUDA Environment Setup](#cuda-environment-setup)
    - [llama.cpp CUDA Compilation](#llamacpp-cuda-compilation)
  - [Stage 2: Runtime Environment](#stage-2-runtime-environment)
    - [CUDA Runtime Base](#cuda-runtime-base)
    - [Runtime Dependencies](#runtime-dependencies)
    - [GPU Performance Tuning](#gpu-performance-tuning)
    - [Security Configuration](#security-configuration)
    - [Application Deployment](#application-deployment)
  - [GPU Performance Optimization Strategy](#gpu-performance-optimization-strategy)
    - [Blackwell Architecture Targeting](#blackwell-architecture-targeting)
    - [CUDA Memory Management](#cuda-memory-management)
    - [Compute Optimization](#compute-optimization)
    - [Inference Configuration](#inference-configuration)
  - [Container Security Implementation](#container-security-implementation)
  - [Reference Implementation](#reference-implementation)

## Implementation Overview

The GPU Docker implementation leverages a multi-stage build strategy specifically optimized for the RTX 5090 Blackwell architecture. The implementation utilizes CUDA 12.9.1 for maximum compatibility with the latest GPU features and provides comprehensive performance tuning for AI inference workloads.

**Key Implementation Features:**
- **Multi-stage build**: Separates CUDA development tools from runtime environment
- **Blackwell optimization**: sm_120 architecture targeting for RTX 5090 maximum performance
- **CUDA 12.9.1**: Latest CUDA toolkit with Blackwell support and optimizations
- **CMake build system**: Modern build configuration for llama.cpp CUDA compilation
- **Performance tuning**: Comprehensive GPU memory and compute optimizations
- **Security hardening**: Non-root user execution with minimal runtime dependencies

## Stage 1: Builder Environment

### CUDA Development Base

**NVIDIA CUDA Development Foundation**
```dockerfile
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS builder
```
- **Purpose**: Provides complete CUDA development environment with compiler tools
- **Version**: CUDA 12.9.1 ensures RTX 5090 Blackwell architecture compatibility
- **Base OS**: Ubuntu 24.04 LTS provides stable foundation with modern libraries

### RTX 5090 Blackwell Configuration

**Blackwell Architecture Environment Variables**
```dockerfile
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_DOCKER_ARCH=sm_120 
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV CUDA_ARCHITECTURES=120
ENV CUDAFLAGS="-O3 --use_fast_math -arch=sm_120"
ENV LDFLAGS="-Wl,-O3 -Wl,--as-needed -L/usr/local/cuda/lib64/stubs"
```

**Architecture-Specific Optimization**
- **CUDA_DOCKER_ARCH=sm_120**: Targets RTX 5090 Blackwell compute capability 12.0
- **TORCH_CUDA_ARCH_LIST="12.0"**: PyTorch/tensor operation optimization for Blackwell
- **CUDA_ARCHITECTURES=120**: CMake CUDA architecture specification
- **CUDAFLAGS**: NVCC compiler flags with fast math optimizations for inference
- **Aggressive optimization**: -O3 and --use_fast_math for maximum performance

### CUDA Components Installation

**Comprehensive CUDA Development Stack**
```dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        gcc g++ build-essential cmake git curl ca-certificates \
        libssl-dev pkg-config python3 python3-pip \
        ninja-build ccache \
        cuda-driver-dev-12-9 \
        cuda-cudart-dev-12-9 \
        cuda-cupti-dev-12-9 \
        cuda-nvml-dev-12-9 \
        cuda-nvtx-12-9 \
        cudnn9-cuda-12-9 \
        libcublas-dev-12-9
```

**CUDA Component Selection**
- **cuda-driver-dev-12-9**: CUDA driver development headers
- **cuda-cudart-dev-12-9**: CUDA runtime API development libraries
- **cuda-cupti-dev-12-9**: CUDA Profiling Tools Interface for performance analysis
- **cuda-nvml-dev-12-9**: NVIDIA Management Library for GPU monitoring
- **cuda-nvtx-12-9**: NVIDIA Tools Extension for profiling and debugging
- **cudnn9-cuda-12-9**: Deep Neural Network library optimized for Blackwell
- **libcublas-dev-12-9**: CUDA Basic Linear Algebra Subroutines for matrix operations

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
RUN rm -rf /tmp/llama.cpp && \
    git clone  https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp && \
    cd /tmp/llama.cpp && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=120 \
        -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs" \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FORCE_CUBLAS=ON \
        -DGGML_CUDA_F16=ON \
        -DGGML_CUDA_GRAPHS=ON \
        -DGGML_CUDA_PEER_MAX_BATCH_SIZE=256 \
        -DGGML_CUDA_NO_VMM=OFF \
        -DGGML_NATIVE=ON \
        -DGGML_LTO=ON \
        -DGGML_BUILD_TESTS=OFF \
        -DGGML_BUILD_EXAMPLES=OFF \
        -DLLAMA_CURL=OFF \
        -DGGML_CCACHE=ON \
        -DGGML_CUDA_DMMV_X=64 \
        -DGGML_CUDA_MMV_Y=2 \
        -DGGML_CUDA_GRAPHS=ON \
        -DGGML_CUDA_KQUANTS_ITER=2  \
         -DGGML_CUDA_FA_ALL_QUANTS=ON && \
     cmake --build . --target llama-server --config Release -j$(nproc)
```

**CUDA Build Configuration**
- **CMAKE_CUDA_ARCHITECTURES=120**: Explicit Blackwell architecture targeting
- **GGML_CUDA=ON**: Enable CUDA acceleration throughout llama.cpp
- **GGML_CUDA_FORCE_CUBLAS=ON**: Force cuBLAS usage for maximum performance
- **GGML_CUDA_F16=ON**: Enable FP16 precision for memory efficiency and speed
- **GGML_CUDA_GRAPHS=ON**: Enable CUDA graphs for reduced kernel launch overhead
- **GGML_CUDA_NO_VMM=OFF**: Enable Virtual Memory Management for large models
- **GGML_CUDA_FA_ALL_QUANTS=ON**: FlashAttention support for all quantization formats

**Advanced CUDA Optimizations**
- **GGML_CUDA_PEER_MAX_BATCH_SIZE=256**: Optimized batch size for peer-to-peer transfers
- **GGML_CUDA_DMMV_X=64**: Matrix-vector multiplication optimization parameters
- **GGML_CUDA_MMV_Y=2**: Memory access pattern optimization for Blackwell
- **GGML_CUDA_KQUANTS_ITER=2**: Quantization kernel iteration optimization

## Stage 2: Runtime Environment

### CUDA Runtime Base

**Optimized Runtime Foundation**
```dockerfile
FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04
```
- **Runtime focus**: Minimal CUDA runtime without development overhead
- **Version consistency**: Matching CUDA 12.9.1 for binary compatibility
- **Size optimization**: Excludes development tools and headers

### Runtime Dependencies

**Essential CUDA Runtime Libraries**
```dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        ca-certificates \
        libgomp1 \
        cudnn9-cuda-12-9 \
        libcublas-12-9 \
        cuda-driver-dev-12-9 \
        cuda-cupti-12-9 \
        cuda-nvtx-12-9
```

**Runtime Components**
- **cudnn9-cuda-12-9**: Deep learning primitive library runtime
- **libcublas-12-9**: CUDA Basic Linear Algebra Subroutines runtime
- **cuda-cupti-12-9**: Performance monitoring and profiling capabilities
- **cuda-nvtx-12-9**: NVIDIA Tools Extension for runtime analysis

### GPU Performance Tuning

**RTX 5090 Performance Configuration**
```dockerfile
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_MODULE_LOADING=LAZY
ENV CUDA_DEVICE_ORDER=FASTEST_FIRST
ENV CUDA_VISIBLE_DEVICES=0

# Performance tuning for RTX 5090 (Blackwell, 24GB VRAM)
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
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_REQUIRE_CUDA="cuda>=12.8"
```

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
> Note: Cache directory created before user switch to ensure proper permissions

### Application Deployment

**GPU-Optimized Binary Installation**
```dockerfile
COPY --from=builder --chown=appuser:appuser /tmp/llama.cpp/build/bin/llama-server /app/server
COPY --from=builder --chown=appuser:appuser /tmp/llama.cpp/build/bin/* /app/
```

**GPU Inference Server Configuration**
```dockerfile
EXPOSE 8004

CMD ["./server", \
    "--model", "/app/models/gguf/gpt-oss-20b-GGUF/gpt-oss-20b-UD-Q8_K_XL.gguf", \
    "--host", "0.0.0.0", \
    "--port", "8004", \
    "--n-gpu-layers", "999", \
    "--ctx-size", "65536", \
    "--batch-size", "2048", \
    "--ubatch-size", "512", \
    "--threads", "1", \
    "--threads-batch", "1", \
    "--metrics", \
    "--no-warmup", \
    "--threads-http", "4", \
    "--flash-attn", \
    "--mlock", \
    "--no-mmap", \
    "--main-gpu", "0", \
    "--parallel", "1" \
]
```

## GPU Performance Optimization Strategy

### Blackwell Architecture Targeting

**Compute Capability 12.0 Utilization**
The implementation targets specific RTX 5090 Blackwell capabilities:
- **sm_120 architecture**: Maximizes utilization of Blackwell compute units
- **Advanced tensor operations**: Leverages fourth-generation RT cores and third-generation Tensor cores
- **Memory bandwidth**: Optimized for 896 GB/s memory bandwidth with GDDR7
- **24GB VRAM**: Configured for large model inference with full GPU memory utilization

**Compiler Optimization Pipeline**
1. **Architecture targeting**: `-arch=sm_120` generates Blackwell-specific CUDA code
2. **Fast math**: `--use_fast_math` enables hardware-accelerated mathematical operations
3. **Aggressive optimization**: -O3 level optimization for maximum performance
4. **FlashAttention**: Hardware-accelerated attention mechanisms for transformer models

### CUDA Memory Management

**VRAM Optimization Strategy**
- **24GB VRAM utilization**: Full model loading with `--n-gpu-layers=999`
- **Memory locking**: `--mlock` prevents model swapping and ensures GPU residence
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
- **Matrix operations**: Optimized DMMV and MMV parameters for Blackwell memory controllers
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
- **Memory locking**: Prevents operating system from swapping model data
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

## Reference Implementation

**File Structure**
- **docker/Dockerfile.llama-gpu**: Complete multi-stage GPU-optimized Dockerfile
- **docker-compose.yaml**: Container orchestration with GPU resource allocation and security hardening

**Integration with AI Workstation**
This GPU implementation works in conjunction with:
- **CPU containers**: Parallel execution for hybrid CPU/GPU workloads
- **Load balancing**: GPU container handles large models while CPU containers manage concurrent smaller models
- **Resource isolation**: Exclusive GPU access with controlled VRAM allocation
- **Performance monitoring**: CUDA profiling integration for optimization analysis

**Hardware Requirements**
- **GPU**: RTX 5090 24GB (Blackwell architecture)
- **CUDA**: Version 12.8 or higher driver support
- **System Memory**: Sufficient for container and CUDA context overhead
- **Storage**: Fast storage for model loading and CUDA cache operations

---

*This GPU Docker implementation provides maximum performance AI model inference on the RTX 5090 Blackwell architecture as of mid-2025. The multi-stage build process ensures optimal CUDA integration while maintaining security and operational efficiency for high-throughput AI workloads.*