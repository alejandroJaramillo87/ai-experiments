# AI Engineering vLLM GPU Docker Implementation

Comprehensive explanation of the RTX 5090 Blackwell optimized vLLM Dockerfile for high-performance GPU-accelerated AI model inference on the AI workstation, detailing the multi-stage build process, vLLM v0.10.0 integration, and Blackwell architecture optimizations.

> Note: This document explains the actual implementation in `docker/Dockerfile.vllm-gpu`. The Dockerfile uses vLLM v0.10.0 with CUDA 12.9.1, Blackwell sm_120 architecture targeting, and comprehensive GPU performance optimizations for transformer model serving.

## Table of Contents

- [AI Engineering vLLM GPU Docker Implementation](#ai-engineering-vllm-gpu-docker-implementation)
  - [Table of Contents](#table-of-contents)
  - [Implementation Overview](#implementation-overview)
  - [Stage 1: Builder Environment](#stage-1-builder-environment)
    - [NVIDIA NGC CUDA Base](#nvidia-ngc-cuda-base)
    - [RTX 5090 Blackwell Configuration](#rtx-5090-blackwell-configuration)
    - [Build Dependencies Installation](#build-dependencies-installation)
    - [Compiler Cache Setup](#compiler-cache-setup)
    - [vLLM v0.10.0 Compilation](#vllm-v0100-compilation)
  - [Stage 2: Runtime Environment](#stage-2-runtime-environment)
    - [CUDA Runtime Base](#cuda-runtime-base)
    - [Runtime Dependencies](#runtime-dependencies)
    - [vLLM Performance Tuning](#vllm-performance-tuning)
    - [Security Configuration](#security-configuration)
    - [Application Deployment](#application-deployment)
  - [vLLM Performance Optimization Strategy](#vllm-performance-optimization-strategy)
    - [Blackwell Architecture Targeting](#blackwell-architecture-targeting)
    - [vLLM Engine Configuration](#vllm-engine-configuration)
    - [Memory Management](#memory-management)
    - [Inference Optimization](#inference-optimization)
  - [Container Security Implementation](#container-security-implementation)
  - [Reference Implementation](#reference-implementation)

## Implementation Overview

The vLLM GPU Docker implementation leverages a multi-stage build strategy specifically optimized for the RTX 5090 Blackwell architecture. The implementation utilizes vLLM v0.10.0 with CUDA 12.9.1 for maximum compatibility and performance, providing state-of-the-art transformer model serving capabilities.

**Key Implementation Features:**
- **Multi-stage build**: Separates vLLM compilation environment from runtime deployment
- **vLLM v0.10.0**: Latest vLLM release with Blackwell support and advanced optimizations
- **Blackwell optimization**: sm_120 architecture targeting for RTX 5090 maximum performance
- **CUDA 12.9.1**: Latest CUDA toolkit with comprehensive Blackwell support
- **Performance tuning**: Comprehensive vLLM engine and GPU memory optimizations
- **Security hardening**: Non-root user execution with controlled cache management

## Stage 1: Builder Environment

### NVIDIA NGC CUDA Base

**NVIDIA CUDA Development Foundation**
```dockerfile
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 AS builder
```
- **Purpose**: Provides complete CUDA development environment optimized for Blackwell
- **NGC container**: Recommended base per vLLM issue #14452 for optimal compatibility
- **CUDA 12.9.1**: Ensures RTX 5090 Blackwell architecture support and PyTorch 2.6+ compatibility
- **Ubuntu 24.04**: Latest LTS foundation with modern development libraries

### RTX 5090 Blackwell Configuration

**Blackwell Architecture Environment Variables**
```dockerfile
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV CUDA_DOCKER_ARCH=sm_120
ENV CUDA_ARCHITECTURES=120
ENV NVCC_THREADS=8
ENV CUDAFLAGS="-O3 --use_fast_math"
ENV LDFLAGS="-Wl,-O3 -Wl,--as-needed -L/usr/local/cuda/lib64/stubs"
```

**Architecture-Specific Optimization**
- **TORCH_CUDA_ARCH_LIST="12.0"**: PyTorch tensor operation optimization for Blackwell compute capability 12.0
- **CUDA_DOCKER_ARCH=sm_120**: Targets RTX 5090 Blackwell architecture specifically
- **CUDA_ARCHITECTURES=120**: CMake CUDA architecture specification for native compilation
- **NVCC_THREADS=8**: Parallel NVIDIA CUDA compiler threads for faster builds
- **CUDAFLAGS**: Aggressive optimization with fast math for inference workloads
- **Performance linker flags**: -O3 optimization with reduced binary size

### Build Dependencies Installation

**Development Environment Setup**
```dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ccache \
        ninja-build python3.12-dev\
        python3 python3-pip \
        cmake \
        git
```

**Essential Build Tools**
- **ccache**: Compiler cache for faster rebuilds (required per vLLM issue #14452)
- **ninja-build**: Fast parallel build system for vLLM compilation
- **python3.12-dev**: Python development headers for vLLM Python extension compilation
- **cmake**: Modern build system for complex vLLM dependency management
- **git**: Version control for vLLM source code retrieval

### Compiler Cache Setup

**Build Performance Optimization**
```dockerfile
ENV CCACHE_DIR=/tmp/ccache
ENV PATH=/usr/lib/ccache:${PATH}
```
- **CCACHE_DIR**: Dedicated cache directory for compiled objects
- **PATH modification**: Ensures ccache intercepts compiler calls for caching
- **Build acceleration**: Significantly reduces rebuild times during development

### vLLM v0.10.0 Compilation

**vLLM Source and Build Process**
```dockerfile
RUN git clone --depth 1 --branch v0.10.0 https://github.com/vllm-project/vllm.git /tmp/vllm

WORKDIR /tmp/vllm

RUN python3 use_existing_torch.py

RUN pip3 install --no-cache-dir --break-system-packages -r requirements/common.txt \
    setuptools_scm 

RUN mkdir -p /tmp/ccache && \
    CCACHE_DIR=/tmp/ccache \
    VLLM_TARGET_DEVICE=cuda \
    MAX_JOBS=$(nproc) \
    TORCH_CUDA_ARCH_LIST="12.0" \
    pip3 wheel --no-build-isolation . && \
    pip3 install --break-system-packages vllm-*.whl
```

**vLLM Build Configuration**
- **Specific version**: v0.10.0 tagged release for stability and Blackwell compatibility
- **Shallow clone**: `--depth 1` reduces download time and build context size
- **PyTorch integration**: `use_existing_torch.py` leverages optimized NGC container PyTorch
- **Build isolation disabled**: `--no-build-isolation` allows custom CUDA flags and optimization
- **Architecture targeting**: `TORCH_CUDA_ARCH_LIST="12.0"` ensures Blackwell-specific compilation
- **Parallel compilation**: `MAX_JOBS=$(nproc)` utilizes all available CPU cores

**Advanced Build Features**
- **VLLM_TARGET_DEVICE=cuda**: Explicitly enables CUDA device targeting
- **Wheel packaging**: Creates distributable package for runtime stage deployment
- **ccache integration**: Accelerated compilation through intelligent caching

## Stage 2: Runtime Environment

### CUDA Runtime Base

**Optimized Runtime Foundation**
```dockerfile
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04 
```
- **Runtime focus**: Includes development libraries for vLLM runtime requirements
- **Version consistency**: Matching CUDA 12.9.1 for binary compatibility
- **vLLM requirements**: Development environment needed for vLLM Python extensions

### Runtime Dependencies

**Essential Runtime Libraries**
```dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        python3.12-dev\
        python3 python3-pip \
        libgomp1
```

**Runtime Components**
- **python3.12-dev**: Python development libraries required for vLLM extensions
- **libgomp1**: OpenMP runtime library for parallel processing
- **curl**: HTTP client for health checks and API testing
- **ca-certificates**: SSL/TLS certificate authority bundle for secure communications

**vLLM Installation**
```dockerfile
COPY --from=builder /tmp/vllm/vllm-*.whl /tmp/
RUN pip3 install --break-system-packages /tmp/vllm-*.whl
```

### vLLM Performance Tuning

**CUDA Environment Configuration**
```dockerfile
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_MODULE_LOADING=LAZY
ENV CUDA_DEVICE_ORDER=FASTEST_FIRST
ENV CUDA_VISIBLE_DEVICES=0
```

**RTX 5090 Performance Optimization**
```dockerfile
ENV CUDA_LAUNCH_BLOCKING=0
ENV CUDNN_LOGINFO_DBG=0
ENV CUDNN_LOGDEST_DBG=stderr
ENV CUDA_CACHE_DISABLE=0
ENV CUDA_CACHE_PATH=/tmp/cuda_cache
ENV CUDA_FORCE_PTX_JIT=0
```

**vLLM Engine Configuration**
```dockerfile
ENV VLLM_USE_MODELSCOPE=False
ENV VLLM_FLASH_ATTN_VERSION=2
ENV VLLM_ATTENTION_BACKEND=FLASH_ATTN
ENV VLLM_USE_RAY=0
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV TOKENIZERS_PARALLELISM=false
ENV VLLM_USAGE_SOURCE=docker
ENV VLLM_LOGGING_LEVEL=INFO

ENV VLLM_USE_V1=1
ENV VLLM_ENABLE_PREFIX_CACHING=1
ENV VLLM_FP8_E4M3=1
```

**vLLM Optimization Variables**
- **VLLM_FLASH_ATTN_VERSION=2**: FlashAttention v2 for Blackwell compatibility (v3 not yet supported)
- **VLLM_ATTENTION_BACKEND=FLASH_ATTN**: Hardware-accelerated attention mechanisms
- **VLLM_USE_V1=1**: Enable vLLM v1 engine optimizations for improved performance
- **VLLM_ENABLE_PREFIX_CACHING=1**: Intelligent caching of common prompt prefixes
- **VLLM_FP8_E4M3=1**: FP8 quantization support for Blackwell architecture
- **VLLM_USE_RAY=0**: Disable Ray for single-GPU deployments to reduce overhead

### Security Configuration

**Non-Root User Implementation**
```dockerfile
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "Application User" appuser

RUN mkdir -p /tmp/cuda_cache /app/logs /app/cache

ENV HF_HOME=/app/cache
ENV HOME=/app

RUN chown -R appuser:appuser /app /tmp/cuda_cache

USER appuser
WORKDIR /app
```

**Security Features**
- **Dedicated user**: Non-root execution prevents privilege escalation
- **No shell access**: `/sbin/nologin` prevents interactive container access
- **Cache isolation**: Dedicated cache directories with proper ownership
- **Hugging Face integration**: `HF_HOME` directs model cache to writable location
- **Controlled permissions**: Explicit ownership changes before user switch

### Application Deployment

**Health Check Implementation**
```dockerfile
RUN echo '#!/bin/bash\n\
set -e\n\
curl -f -s -o /dev/null -w "%{http_code}" http://localhost:8005/health | grep -q "200" || exit 1' \
    > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=300s \
    CMD ["/app/healthcheck.sh"]
```

**vLLM Server Configuration**
```dockerfile
EXPOSE 8005

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

CMD ["--model", "/app/models/hf/DeepSeek-R1-0528-Qwen3-8b", \
     "--host", "0.0.0.0", \
     "--port", "8005", \
     "--tensor-parallel-size", "1", \
     "--gpu-memory-utilization", "0.85", \
     "--max-model-len", "32768", \
     "--max-num-seqs", "512", \
     "--max-num-batched-tokens", "65536", \
     "--enable-chunked-prefill", \
     "--enable-prefix-caching", \
     "--disable-log-stats", \
     "--trust-remote-code", \
     "--download-dir", "/app/cache", \
     "--dtype", "auto", \
     "--kv-cache-dtype", "auto", \
     "--quantization", "fp8", \
     "--tokenizer-mode", "auto"]
```

## vLLM Performance Optimization Strategy

### Blackwell Architecture Targeting

**Compute Capability 12.0 Utilization**
The implementation targets specific RTX 5090 Blackwell capabilities:
- **sm_120 architecture**: Maximizes utilization of Blackwell streaming multiprocessors
- **Advanced tensor operations**: Leverages fourth-generation RT cores and third-generation Tensor cores
- **Memory bandwidth**: Optimized for 896 GB/s memory bandwidth with GDDR7
- **24GB VRAM**: Configured for large transformer model inference with intelligent memory management

**vLLM Compilation Pipeline**
1. **Architecture targeting**: `TORCH_CUDA_ARCH_LIST="12.0"` generates Blackwell-specific PyTorch operations
2. **Fast math**: `--use_fast_math` enables hardware-accelerated mathematical operations
3. **Aggressive optimization**: -O3 level optimization for maximum inference performance
4. **FlashAttention integration**: Hardware-accelerated attention mechanisms for transformer efficiency

### vLLM Engine Configuration

**V1 Engine Optimizations**
- **VLLM_USE_V1=1**: Enables next-generation vLLM engine with improved memory management
- **Prefix caching**: `VLLM_ENABLE_PREFIX_CACHING=1` caches common prompt prefixes for faster response
- **Chunked prefill**: `--enable-chunked-prefill` optimizes large context processing
- **FP8 quantization**: `VLLM_FP8_E4M3=1` leverages Blackwell's native FP8 support for memory efficiency

**Attention Mechanism Optimization**
- **FlashAttention v2**: `VLLM_FLASH_ATTN_VERSION=2` provides memory-efficient attention computation
- **Backend specification**: `VLLM_ATTENTION_BACKEND=FLASH_ATTN` ensures optimal attention implementation
- **Hardware acceleration**: Leverages Blackwell's specialized tensor processing units

### Memory Management

**VRAM Optimization Strategy**
- **GPU memory utilization**: `--gpu-memory-utilization=0.85` reserves 85% of 24GB VRAM for models
- **Intelligent batching**: `--max-num-batched-tokens=65536` optimizes throughput for Blackwell memory hierarchy
- **Sequence management**: `--max-num-seqs=512` balances concurrent request handling with memory efficiency
- **Context optimization**: `--max-model-len=32768` provides large context support within memory constraints

**Cache Management Strategy**
- **Model cache**: `--download-dir=/app/cache` centralizes model storage in writable directory
- **CUDA cache**: `/tmp/cuda_cache` provides persistent kernel compilation cache
- **Hugging Face cache**: `HF_HOME=/app/cache` consolidates transformer model artifacts

### Inference Optimization

**Single-GPU Configuration**
- **Tensor parallelism**: `--tensor-parallel-size=1` optimized for single RTX 5090 deployment
- **Worker configuration**: `VLLM_WORKER_MULTIPROC_METHOD=spawn` ensures clean process isolation
- **Ray disabled**: `VLLM_USE_RAY=0` eliminates distributed computing overhead for single-node deployment

**Performance Tuning Parameters**
- **Batch optimization**: Asymmetric batching strategy optimized for Blackwell architecture
- **Quantization**: `--quantization=fp8` leverages native Blackwell FP8 tensor operations
- **Data type auto-detection**: `--dtype=auto` automatically selects optimal precision for each operation
- **Tokenizer optimization**: `--tokenizer-mode=auto` enables efficient text processing

## Container Security Implementation

**Multi-Layer Security Approach**
- **Minimal runtime**: Only essential libraries included in final image
- **Non-root execution**: Application runs as dedicated non-privileged user
- **No interactive access**: User account configured without shell access
- **Controlled GPU access**: Explicit GPU device specification and capability restrictions
- **Cache isolation**: vLLM and CUDA caches confined to container directories with proper permissions

**Health Monitoring**
- **HTTP health checks**: Automated container health monitoring via OpenAI-compatible API
- **Startup period**: 300-second grace period for large model loading
- **Failure detection**: Automatic container restart on health check failures
- **Performance monitoring**: Optional metrics collection for production deployments

**Docker Compose Integration**
When deployed via docker-compose, additional security measures include:
- **Capability dropping**: All Linux capabilities removed except GPU access
- **Privilege restrictions**: `no-new-privileges:true` prevents escalation
- **Memory constraints**: Controlled memory allocation with unlimited memlock for CUDA operations
- **Read-only containers**: Filesystem mounted read-only except for designated cache areas

## Reference Implementation

**File Structure**
- **docker/Dockerfile.vllm-gpu**: Complete multi-stage vLLM-optimized Dockerfile
- **docker-compose.yaml**: Container orchestration with GPU resource allocation and security hardening

**Integration with AI Workstation**
This vLLM implementation works in conjunction with:
- **CPU containers**: Parallel execution for hybrid vLLM/llama.cpp workloads
- **Load balancing**: vLLM container handles large transformer models while CPU containers manage concurrent smaller models
- **Resource isolation**: Exclusive GPU access with controlled VRAM allocation
- **Performance monitoring**: vLLM metrics integration for optimization analysis

**Hardware Requirements**
- **GPU**: RTX 5090 24GB (Blackwell architecture)
- **CUDA**: Version 12.8 or higher driver support
- **System Memory**: Sufficient for container overhead and model preprocessing
- **Storage**: Fast storage for model loading and cache operations

**Model Compatibility**
- **Transformer architectures**: GPT, LLaMA, Qwen, DeepSeek, and other modern transformer models
- **Quantization formats**: FP16, FP8, INT8, and other precision formats
- **Context lengths**: Up to 32K tokens with larger contexts possible depending on model size
- **Hugging Face integration**: Direct model loading from Hugging Face Hub with caching

---

*This vLLM GPU Docker implementation provides state-of-the-art transformer model serving on the RTX 5090 Blackwell architecture as of mid-2025. The multi-stage build process ensures optimal vLLM integration while maintaining security and operational efficiency for high-throughput AI inference workloads.*