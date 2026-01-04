# AI Engineering vLLM GPU Docker Implementation

Explanation of the RTX 5090 Blackwell optimized vLLM Dockerfile for high-performance GPU-accelerated AI model inference on the AI workstation, detailing the multi-stage build process, vLLM v0.12.0 integration, and Blackwell architecture optimizations.

> Note: This document explains the actual implementation in `docker/vllm-gpu/Dockerfile.vllm-gpu`. The Dockerfile uses vLLM v0.12.0 with CUDA 13.0.2, Blackwell sm_120 architecture targeting, FlashInfer attention kernels, and GPU performance optimizations for transformer model serving.

## Table of Contents

- [AI Engineering vLLM GPU Docker Implementation](#ai-engineering-vllm-gpu-docker-implementation)
  - [Table of Contents](#table-of-contents)
  - [Implementation Overview](#implementation-overview)
  - [Stage 1: Builder Environment](#stage-1-builder-environment)
    - [Build Arguments](#build-arguments)
    - [NVIDIA CUDA Base](#nvidia-cuda-base)
    - [RTX 5090 Blackwell Configuration](#rtx-5090-blackwell-configuration)
    - [Build Dependencies Installation](#build-dependencies-installation)
    - [uv Package Manager Setup](#uv-package-manager-setup)
    - [Compiler Cache Setup](#compiler-cache-setup)
    - [vLLM v0.12.0 Compilation](#vllm-v0120-compilation)
  - [Stage 2: Runtime Environment](#stage-2-runtime-environment)
    - [CUDA Runtime Base](#cuda-runtime-base)
    - [Runtime Dependencies](#runtime-dependencies)
    - [vLLM Installation](#vllm-installation)
    - [FlashInfer Attention Kernels](#flashinfer-attention-kernels)
    - [CUDA Runtime Environment](#cuda-runtime-environment)
    - [vLLM Settings](#vllm-settings)
    - [CUDA Performance Tuning](#cuda-performance-tuning)
    - [Experimental Blackwell Optimizations](#experimental-blackwell-optimizations)
    - [Security Configuration](#security-configuration)
    - [Application Deployment](#application-deployment)
  - [Entrypoint Configuration](#entrypoint-configuration)
    - [Required Environment Variables](#required-environment-variables)
    - [Server Configuration](#server-configuration)
    - [Feature Flags](#feature-flags)
    - [Nsight Systems Profiling](#nsight-systems-profiling)
    - [Server Command](#server-command)
  - [vLLM Performance Optimization Strategy](#vllm-performance-optimization-strategy)
    - [Blackwell Architecture Targeting](#blackwell-architecture-targeting)
    - [Memory Management](#memory-management)
    - [Inference Optimization](#inference-optimization)
  - [Container Security Implementation](#container-security-implementation)
  - [Reference Implementation](#reference-implementation)

## Implementation Overview

The vLLM GPU Docker implementation leverages a multi-stage build strategy specifically optimized for the RTX 5090 Blackwell architecture. The implementation utilizes vLLM v0.12.0 with CUDA 13.0.2 for maximum compatibility and performance, providing state-of-the-art transformer model serving capabilities.

**Key Implementation Features:**
- **Multi-stage build**: Separates vLLM compilation environment from runtime deployment
- **vLLM v0.12.0**: Latest vLLM release with full CUDA 13 and Blackwell support
- **Blackwell optimization**: sm_120 architecture targeting for RTX 5090 maximum performance
- **CUDA 13.0.2**: Latest CUDA toolkit with comprehensive Blackwell support
- **uv package manager**: Fast Python package installation (following upstream vLLM)
- **FlashInfer 0.5.3**: Optimized attention kernels for CUDA 13
- **Entrypoint script**: Parameterized server configuration via environment variables
- **Nsight Systems integration**: Optional deep profiling support
- **Security hardening**: Non-root user execution with controlled cache management

## Stage 1: Builder Environment

### Build Arguments

**Parameterized Build Configuration**
```dockerfile
ARG CUDA_VERSION=13.0.2
ARG VLLM_VERSION=v0.12.0
ARG PYTHON_VERSION=3.12
```

The Dockerfile uses build arguments for flexible version management:
- **CUDA_VERSION**: CUDA toolkit version (default: 13.0.2)
- **VLLM_VERSION**: vLLM git tag to build (default: v0.12.0)
- **PYTHON_VERSION**: Python version (default: 3.12)

### NVIDIA CUDA Base

**NVIDIA CUDA Development Foundation**
```dockerfile
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS builder
```
- **Purpose**: Provides complete CUDA development environment optimized for Blackwell
- **cuDNN included**: Deep learning primitives included in base image
- **CUDA 13.0.2**: Ensures RTX 5090 Blackwell architecture support with full vLLM compatibility
- **Ubuntu 24.04**: Latest LTS foundation with modern development libraries

### RTX 5090 Blackwell Configuration

**Blackwell Architecture Environment Variables**
```dockerfile
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV CUDA_DOCKER_ARCH=sm_120
ENV CUDA_ARCHITECTURES=120
ENV NVCC_THREADS=2
ENV CUDAFLAGS="-O3 --use_fast_math"
ENV LDFLAGS="-Wl,-O3 -Wl,--as-needed -L/usr/local/cuda/lib64/stubs"
```

**Architecture-Specific Optimization**
- **TORCH_CUDA_ARCH_LIST="12.0"**: PyTorch tensor operation optimization for Blackwell compute capability 12.0
- **CUDA_DOCKER_ARCH=sm_120**: Targets RTX 5090 Blackwell architecture specifically
- **CUDA_ARCHITECTURES=120**: CMake CUDA architecture specification for native compilation
- **NVCC_THREADS=2**: Parallel NVIDIA CUDA compiler threads (kept low to avoid oversubscription with high MAX_JOBS)
- **CUDAFLAGS**: Aggressive optimization with fast math for inference workloads
- **Performance linker flags**: -O3 optimization with reduced binary size

**CUDA Environment Setup**
```dockerfile
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV PKG_CONFIG_PATH=/usr/local/cuda/lib64/pkgconfig:${PKG_CONFIG_PATH}
ENV CUDA_MODULE_LOADING=LAZY
ENV CUDA_DEVICE_ORDER=FASTEST_FIRST

RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
```

### Build Dependencies Installation

**Development Environment Setup**
```dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        ccache \
        ninja-build \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python3 \
        python3-pip \
        cmake \
        git \
        curl \
        ca-certificates \
        pkg-config \
        cuda-driver-dev-13-0 \
        cuda-cudart-dev-13-0 \
        cuda-cupti-dev-13-0 \
        cuda-nvtx-13-0 \
        libcublas-dev-13-0
```

**Essential Build Tools**
- **ccache**: Compiler cache for faster rebuilds
- **ninja-build**: Fast parallel build system for vLLM compilation
- **python${PYTHON_VERSION}-dev/-venv**: Python development headers and virtual environment support
- **cmake**: Modern build system for complex vLLM dependency management
- **git/curl/ca-certificates**: Version control and package retrieval
- **pkg-config**: Build configuration helper
- **CUDA development packages**: Driver, runtime, profiling, and math library development headers

### uv Package Manager Setup

**Fast Python Package Management**
```dockerfile
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN uv venv /opt/venv --python ${PYTHON_VERSION}
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:${PATH}"
```

- **uv**: Modern, fast Python package manager (following upstream vLLM Dockerfile pattern)
- **Virtual environment**: Avoids PEP 668 externally-managed restrictions
- **PATH configuration**: Ensures uv and venv binaries are accessible

### Compiler Cache Setup

**Build Performance Optimization**
```dockerfile
ENV CCACHE_DIR=/tmp/ccache
ENV CCACHE_MAX_SIZE=20G
ENV CCACHE_COMPR_LEVEL=2
ENV PATH=/usr/lib/ccache:${PATH}
```
- **CCACHE_DIR**: Dedicated cache directory for compiled objects
- **CCACHE_MAX_SIZE=20G**: Large cache for vLLM's extensive CUDA compilation
- **CCACHE_COMPR_LEVEL=2**: Balanced compression for speed and space
- **PATH modification**: Ensures ccache intercepts compiler calls

### vLLM v0.12.0 Compilation

**vLLM Source and Build Process**
```dockerfile
RUN git clone --depth 1 --branch ${VLLM_VERSION} https://github.com/vllm-project/vllm.git /tmp/vllm

WORKDIR /tmp/vllm

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements/build.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements/cuda.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match

RUN mkdir -p /tmp/ccache && \
    CCACHE_DIR=/tmp/ccache \
    TORCH_CUDA_ARCH_LIST="12.0" \
    VLLM_TARGET_DEVICE=cuda \
    MAX_JOBS=20 \
    python3 setup.py bdist_wheel --dist-dir=dist
```

**vLLM Build Configuration**
- **Specific version**: v0.12.0 tagged release with full CUDA 13 support
- **Shallow clone**: `--depth 1` reduces download time and build context size
- **uv pip install**: Fast package installation with build caching
- **requirements/build.txt**: Build-time dependencies
- **requirements/cuda.txt**: CUDA-specific requirements with cu130 PyTorch wheels
- **VLLM_TARGET_DEVICE=cuda**: Explicitly enables CUDA device targeting
- **MAX_JOBS=20**: Parallel compilation jobs (tuned for build system resources)
- **setup.py bdist_wheel**: Creates distributable wheel for runtime stage

## Stage 2: Runtime Environment

### CUDA Runtime Base

**Optimized Runtime Foundation**
```dockerfile
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04
```
- **Runtime focus**: Minimal CUDA runtime without full development overhead
- **cuDNN included**: Deep learning primitives included in runtime base image
- **Version consistency**: Matching CUDA 13.0.2 for binary compatibility
- **Size optimization**: Excludes development tools and headers where possible

### Runtime Dependencies

**Essential Runtime Libraries**
```dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        curl \
        ca-certificates \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        python3 \
        python3-pip \
        libgomp1 \
        libcublas-13-0 \
        cuda-cupti-13-0 \
        cuda-nvtx-13-0 \
        cuda-nvrtc-13-0 \
        libnccl2 \
        nsight-systems-2025.3.2 \
        cuda-nvcc-13-0 \
        cuda-cudart-dev-13-0
```

**Runtime Components**
- **python${PYTHON_VERSION}/-dev/-venv**: Python runtime with development headers for extensions
- **libgomp1**: OpenMP runtime library for parallel processing
- **libcublas-13-0**: CUDA Basic Linear Algebra Subroutines runtime
- **cuda-cupti-13-0**: Performance monitoring and profiling capabilities
- **cuda-nvtx-13-0**: NVIDIA Tools Extension for runtime analysis
- **cuda-nvrtc-13-0**: NVIDIA Runtime Compilation library for JIT compilation
- **libnccl2**: NVIDIA Collective Communications Library for multi-GPU
- **nsight-systems-2025.3.2**: Deep profiling tool (optional, enabled via NSYS_PROFILE)
- **cuda-nvcc-13-0**: NVIDIA CUDA compiler for torch.compile support
- **cuda-cudart-dev-13-0**: CUDA runtime development headers

### vLLM Installation

**vLLM Wheel Installation**
```dockerfile
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN uv venv /opt/venv --python ${PYTHON_VERSION}
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:${PATH}"

COPY --from=builder /tmp/vllm/dist/*.whl /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install /tmp/*.whl \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match && \
    rm -f /tmp/*.whl
```

- **uv package manager**: Same fast package manager as builder stage
- **Virtual environment**: Isolated Python environment
- **Wheel installation**: Installs pre-built vLLM wheel from builder stage
- **cu130 wheel index**: Ensures compatible PyTorch CUDA 13 binaries

### FlashInfer Attention Kernels

**Optimized Attention Computation**
```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install flashinfer-cubin==0.5.3 && \
    uv pip install flashinfer-jit-cache==0.5.3 \
    --extra-index-url https://flashinfer.ai/whl/cu130
```

**FlashInfer Components**
- **flashinfer-cubin==0.5.3**: Pre-compiled CUDA binary attention kernels optimized for CUDA 13
- **flashinfer-jit-cache==0.5.3**: JIT compilation cache for custom attention kernel generation
- **CUDA 13 wheel index**: FlashInfer wheels specifically built for CUDA 13.x

**Optional Performance Package**
```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch-c-dlpack-ext
```
- **torch-c-dlpack-ext**: Enables EnvTensorAllocator for TVM/DLPack interoperability

### CUDA Runtime Environment

**CUDA Paths Configuration**
```dockerfile
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
```

**NVIDIA Docker Runtime**
```dockerfile
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV NVIDIA_REQUIRE_CUDA="cuda>=13.0"
```

- **NVIDIA_VISIBLE_DEVICES=all**: Exposes all available GPUs to the container
- **NVIDIA_DRIVER_CAPABILITIES**: Enables compute, utility, and graphics capabilities
- **NVIDIA_REQUIRE_CUDA**: Enforces minimum CUDA version requirement

### vLLM Settings

**Verified vLLM Environment Variables**
```dockerfile
ENV VLLM_USAGE_SOURCE=docker
ENV VLLM_LOGGING_LEVEL=INFO
ENV VLLM_USE_MODELSCOPE=False
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HUB_ENABLE_HF_TRANSFER=1
```

**vLLM Configuration**
- **VLLM_USAGE_SOURCE=docker**: Telemetry source identifier
- **VLLM_LOGGING_LEVEL=INFO**: Standard logging verbosity
- **VLLM_USE_MODELSCOPE=False**: Disable ModelScope downloads (use Hugging Face)
- **VLLM_WORKER_MULTIPROC_METHOD=spawn**: Use "spawn" for CUDA compatibility
- **TOKENIZERS_PARALLELISM=false**: Disable tokenizer parallelism to avoid fork issues
- **HF_HUB_ENABLE_HF_TRANSFER=1**: Enable fast Hugging Face model downloads

### CUDA Performance Tuning

**General CUDA Settings**
```dockerfile
ENV CUDA_MODULE_LOADING=LAZY
ENV CUDA_DEVICE_ORDER=FASTEST_FIRST
ENV CUDA_LAUNCH_BLOCKING=0
ENV CUDA_CACHE_DISABLE=0
ENV CUDA_CACHE_PATH=/tmp/cuda_cache
ENV CUDA_FORCE_PTX_JIT=0
ENV CUDNN_LOGINFO_DBG=0
ENV CUDNN_LOGDEST_DBG=stderr
```

**Performance Optimization Variables**
- **CUDA_MODULE_LOADING=LAZY**: Defers CUDA module loading for faster startup
- **CUDA_DEVICE_ORDER=FASTEST_FIRST**: Ensures RTX 5090 is selected when multiple GPUs present
- **CUDA_LAUNCH_BLOCKING=0**: Enable asynchronous kernel launches for performance
- **CUDA_CACHE_DISABLE=0**: Enable CUDA kernel caching
- **CUDA_FORCE_PTX_JIT=0**: Disable forced PTX JIT compilation for faster startup

**PyTorch Memory Configuration**
```dockerfile
ENV PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.95,max_split_size_mb:2048
```

- **expandable_segments:True**: Enables dynamic memory segment expansion
- **garbage_collection_threshold:0.95**: Aggressive garbage collection for 32GB VRAM
- **max_split_size_mb:2048**: Maximum memory block split size for efficient allocation

### Experimental Blackwell Optimizations

**RTX 5090 Specific Settings (Mostly Commented Out)**

The following settings are documented in the Dockerfile but require testing for vLLM compatibility. Only `CUDA_CACHE_MAXSIZE` is enabled by default:

```dockerfile
# 96MB L2 cache on RTX 5090
# ENV CUDA_L2_PERSISTING_SIZE=100663296
# Concurrent kernel connections
# ENV CUDA_DEVICE_MAX_CONNECTIONS=32
# MB threshold for split copies
# ENV CUDA_COPY_SPLIT_THRESHOLD=256
# Async memory allocation
# ENV CUDA_ALLOCATOR_BACKEND=cudaMallocAsync
# ENV CUDA_MALLOC_ASYNC_POOLS=1
# 2GB kernel cache (ENABLED)
ENV CUDA_CACHE_MAXSIZE=2147483648
# Tensor core optimization
# ENV CUDA_TENSOR_CORES=1
# ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
# ENV CUDNN_TENSOR_OPS=1
# Disable auto-boost for consistent performance
# ENV CUDA_AUTO_BOOST=0
```

Note: These experimental settings are commented out and should be enabled based on benchmarking results for your specific workload.

### Security Configuration

**Non-Root User Implementation**
```dockerfile
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "Application User" appuser

RUN mkdir -p /tmp/cuda_cache /app/logs /app/cache /app/nsys_output

ENV HF_HOME=/app/cache
ENV HOME=/app

RUN chown -R appuser:appuser /app /tmp/cuda_cache

COPY --chown=appuser:appuser docker/vllm-gpu/entrypoint.sh /app/entrypoint.sh

USER root
RUN chmod +x /app/entrypoint.sh

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
HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=300s \
    CMD curl -f http://localhost:8003/health || exit 1
```

**Health Check Parameters**
- **interval**: 30-second check frequency
- **timeout**: 10-second response deadline
- **retries**: 5 attempts before unhealthy status
- **start_period**: 300 seconds grace period for large model loading

**Server Configuration**
```dockerfile
EXPOSE 8003

ENTRYPOINT ["/app/entrypoint.sh"]
```

The server is started via the entrypoint script, which provides parameterized configuration through environment variables.

## Entrypoint Configuration

The entrypoint script (`entrypoint.sh`) provides parameterized server configuration with environment variable overrides.

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `MODEL_PATH` | **REQUIRED** - Path to the model (no default, will fail if not set) |

### Server Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_PORT` | 8003 | HTTP API server port |
| `SERVER_HOST` | 0.0.0.0 | Bind address for server |
| `GPU_MEMORY_UTILIZATION` | 0.85 | GPU memory fraction to use |
| `MAX_MODEL_LEN` | 16384 | Maximum context length |
| `MAX_NUM_SEQS` | 256 | Maximum concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | 32768 | Maximum tokens per batch |
| `TENSOR_PARALLEL_SIZE` | 1 | Number of GPUs for tensor parallelism |
| `DTYPE` | auto | Model data type (auto, float16, bfloat16) |
| `KV_CACHE_DTYPE` | auto | KV cache data type |
| `QUANTIZATION` | (none) | Quantization method (fp8, awq, etc.) |

### Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_PREFIX_CACHING` | true | Enable KV cache prefix caching |
| `ENABLE_CHUNKED_PREFILL` | true | Enable chunked prefill for large contexts |
| `TRUST_REMOTE_CODE` | true | Trust remote model code |
| `DISABLE_LOG_STATS` | true | Disable statistics logging |

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
    -- vllm serve ...
```

### Server Command

The entrypoint uses `vllm serve` (NOT `python -m vllm.entrypoints.openai.api_server`):

```bash
vllm serve $MODEL_PATH \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --dtype "$DTYPE" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --download-dir "/app/cache"
```

**Optional flags added based on environment:**
- `--quantization $QUANTIZATION` (if QUANTIZATION is set)
- `--enable-prefix-caching` (if ENABLE_PREFIX_CACHING=true)
- `--enable-chunked-prefill` (if ENABLE_CHUNKED_PREFILL=true)
- `--trust-remote-code` (if TRUST_REMOTE_CODE=true)
- `--disable-log-stats` (if DISABLE_LOG_STATS=true)

**Entrypoint Features**
- Validates MODEL_PATH is set (fails fast if missing)
- Reports GPU memory status via nvidia-smi before model load
- Displays configuration summary on startup
- Supports optional Nsight Systems deep profiling

## vLLM Performance Optimization Strategy

### Blackwell Architecture Targeting

**Compute Capability 12.0 Utilization**
The implementation targets specific RTX 5090 Blackwell capabilities:
- **sm_120 architecture**: Maximizes utilization of Blackwell streaming multiprocessors
- **Advanced tensor operations**: Leverages fourth-generation RT cores and third-generation Tensor cores
- **Memory bandwidth**: Optimized for 896 GB/s memory bandwidth with GDDR7
- **32GB VRAM**: Configured for large transformer model inference with intelligent memory management

**vLLM Compilation Pipeline**
1. **Architecture targeting**: `TORCH_CUDA_ARCH_LIST="12.0"` generates Blackwell-specific PyTorch operations
2. **Fast math**: `--use_fast_math` enables hardware-accelerated mathematical operations
3. **Aggressive optimization**: -O3 level optimization for maximum inference performance
4. **FlashInfer integration**: Hardware-accelerated attention mechanisms for transformer efficiency

### Memory Management

**VRAM Optimization Strategy**
- **GPU memory utilization**: `--gpu-memory-utilization=0.85` reserves 85% of 32GB VRAM for models
- **Intelligent batching**: `--max-num-batched-tokens=32768` optimizes throughput for Blackwell memory hierarchy
- **Sequence management**: `--max-num-seqs=256` balances concurrent request handling with memory efficiency
- **Context optimization**: `--max-model-len=16384` provides large context support within memory constraints

**Cache Management Strategy**
- **Model cache**: `--download-dir=/app/cache` centralizes model storage in writable directory
- **CUDA cache**: `/tmp/cuda_cache` provides persistent kernel compilation cache
- **Hugging Face cache**: `HF_HOME=/app/cache` consolidates transformer model artifacts
- **Nsight output**: `/app/nsys_output` for profiling data

### Inference Optimization

**Single-GPU Configuration**
- **Tensor parallelism**: `--tensor-parallel-size=1` optimized for single RTX 5090 deployment
- **Worker configuration**: `VLLM_WORKER_MULTIPROC_METHOD=spawn` ensures clean process isolation
- **Prefix caching**: Enabled by default for faster repeated prompt processing
- **Chunked prefill**: Enabled for efficient large context handling

**Performance Tuning Parameters**
- **Quantization**: Optional FP8, AWQ, or other quantization methods via QUANTIZATION env var
- **Data type auto-detection**: `--dtype=auto` automatically selects optimal precision for each operation
- **KV cache optimization**: `--kv-cache-dtype=auto` for memory-efficient key-value storage

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

**Docker Compose Integration**
When deployed via docker-compose, additional security measures include:
- **Capability dropping**: All Linux capabilities removed except GPU access
- **Privilege restrictions**: `no-new-privileges:true` prevents escalation
- **Memory constraints**: Controlled memory allocation with unlimited memlock for CUDA operations
- **Read-only containers**: Filesystem mounted read-only except for designated cache areas

## Reference Implementation

**File Structure**
- **docker/vllm-gpu/Dockerfile.vllm-gpu**: Complete multi-stage vLLM-optimized Dockerfile
- **docker/vllm-gpu/entrypoint.sh**: Parameterized server startup script
- **docker-compose.yaml**: Container orchestration with GPU resource allocation and security hardening

**Integration with AI Workstation**
This vLLM implementation works in conjunction with:
- **CPU containers**: Parallel execution for hybrid vLLM/llama.cpp workloads
- **Load balancing**: vLLM container handles large transformer models while CPU containers manage concurrent smaller models
- **Resource isolation**: Exclusive GPU access with controlled VRAM allocation
- **Performance monitoring**: vLLM metrics integration for optimization analysis

**Hardware Requirements**
- **GPU**: RTX 5090 32GB (Blackwell architecture)
- **CUDA**: Version 13.0 or higher driver support
- **System Memory**: Sufficient for container overhead and model preprocessing
- **Storage**: Fast storage for model loading and cache operations

**Model Compatibility**
- **Transformer architectures**: GPT, LLaMA, Qwen, DeepSeek, and other modern transformer models
- **Quantization formats**: FP16, FP8, AWQ, GPTQ and other precision formats
- **Context lengths**: Up to 16K tokens by default, configurable via MAX_MODEL_LEN
- **Hugging Face integration**: Direct model loading from Hugging Face Hub with fast transfers

---

*This vLLM GPU Docker implementation provides transformer model serving on the RTX 5090 Blackwell architecture. The multi-stage build process integrates vLLM v0.12.0 with FlashInfer attention kernels while maintaining security and operational efficiency for high-throughput AI inference workloads.*

*Last Updated: 2026-01-03*
