# AI Engineering vLLM GPU Docker Implementation

Explanation of the RTX 5090 Blackwell optimized vLLM Dockerfile for GPU-accelerated model inference on the AI workstation, detailing the single-stage image build, the vLLM 0.30.0 official cu130 wheel install, and Blackwell architecture settings.

> Note: This document explains the actual implementation in `docker/vllm-gpu/Dockerfile.vllm-gpu`. The Dockerfile installs vLLM 0.30.0 with CUDA 13.0.2, targets Blackwell sm_120 through the official cu130 wheel, uses FlashInfer 0.6.18.post1 attention kernels, and applies GPU runtime settings for transformer model serving.

## Table of Contents

<!-- toc:start -->

1. [Implementation Overview](#implementation-overview)
2. [Image Build](#image-build)
   1. [Build Arguments](#build-arguments)
   2. [Pinned uv Binary Source](#pinned-uv-binary-source)
   3. [CUDA Runtime Base](#cuda-runtime-base)
   4. [Runtime Dependencies](#runtime-dependencies)
   5. [Virtual Environment Setup](#virtual-environment-setup)
   6. [vLLM Wheel Installation](#vllm-wheel-installation)
   7. [FlashInfer Attention Kernels](#flashinfer-attention-kernels)
   8. [CUDA Runtime Environment](#cuda-runtime-environment)
   9. [vLLM Settings](#vllm-settings)
   10. [CUDA Performance Tuning](#cuda-performance-tuning)
   11. [Experimental Blackwell Optimizations](#experimental-blackwell-optimizations)
   12. [Security Configuration](#security-configuration)
   13. [Application Deployment](#application-deployment)
3. [Entrypoint Configuration](#entrypoint-configuration)
   1. [Required Environment Variables](#required-environment-variables)
   2. [Server Configuration](#server-configuration)
   3. [Feature Flags](#feature-flags)
   4. [Nsight Systems Profiling](#nsight-systems-profiling)
   5. [Server Command](#server-command)
4. [vLLM Performance Optimization Strategy](#vllm-performance-optimization-strategy)
   1. [Blackwell Architecture Targeting](#blackwell-architecture-targeting)
   2. [Memory Management](#memory-management)
   3. [Inference Optimization](#inference-optimization)
5. [Container Security Implementation](#container-security-implementation)
6. [Reference Implementation](#reference-implementation)

<!-- toc:end -->

## Implementation Overview

The vLLM GPU Docker implementation uses a single-stage image built on the CUDA 13.0.2 runtime. It installs the official vLLM 0.30.0 cu130 wheel, which ships sm_120 kernels for the RTX 5090 Blackwell architecture and pulls in PyTorch 2.13 cu130. The previous from-source build is no longer used: since vLLM v0.20.0 the official wheels default to CUDA 13.0 and include Blackwell (compute capability 12.0) kernel binaries, so no local compilation, git clone, ccache, or `setup.py` build step is required.

Key implementation features:

- **Single-stage image**: One `nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04` stage installs vLLM from the wheel; no builder stage.
- **vLLM 0.30.0**: Installed as the official cu130 wheel via `uv pip install vllm==0.30.0 --torch-backend=cu130`.
- **PyTorch 2.13 cu130**: Pulled in as a dependency of the vLLM cu130 wheel.
- **Blackwell support**: The vLLM wheel ships sm_120 kernel binaries for the RTX 5090.
- **CUDA 13.0.2**: CUDA runtime with cuDNN in the base image.
- **Pinned uv 0.11.28**: Copied from `ghcr.io/astral-sh/uv:0.11.28`, replacing the unpinned `curl | sh` installer.
- **FlashInfer 0.6.18.post1**: Pre-compiled attention kernels, the default attention backend on Blackwell.
- **Entrypoint script**: Parameterized server configuration through environment variables.
- **Nsight Systems integration**: Optional deep profiling support.
- **Security hardening**: Non-root user execution with controlled cache management.

## Image Build

### Build Arguments

```dockerfile
ARG CUDA_VERSION=13.0.2
ARG VLLM_VERSION=0.30.0
ARG TRANSFORMERS_VERSION=5.16.1
ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.11.28
ARG FLASHINFER_VERSION=0.6.18.post1
```

The Dockerfile uses build arguments for version management:

- **CUDA_VERSION**: CUDA runtime version (default: 13.0.2)
- **VLLM_VERSION**: vLLM wheel version to install (default: 0.30.0)
- **TRANSFORMERS_VERSION**: transformers version installed in the same resolve (default: 5.16.1); vLLM's own requirement is a floor, and the model-config API the two share is what a Gemma 4 load exercises (KB-100)
- **PYTHON_VERSION**: Python version (default: 3.12)
- **UV_VERSION**: Pinned uv binary version (default: 0.11.28)
- **FLASHINFER_VERSION**: FlashInfer package version (default: 0.6.18.post1)

### Pinned uv Binary Source

```dockerfile
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
```

The uv package manager is pinned to an exact version and copied from its official image, rather than fetched through an unpinned `curl | sh` installer. This stage exists only to provide the `uv` and `uvx` binaries to the runtime stage through a `COPY --from`.

### CUDA Runtime Base

```dockerfile
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04
```

- **Runtime focus**: The CUDA runtime image without the full development toolchain.
- **cuDNN included**: Deep learning primitives ship in the base image.
- **CUDA 13.0.2**: Provides the CUDA runtime that the vLLM cu130 wheel targets.
- **Ubuntu 24.04**: LTS foundation.

### Runtime Dependencies

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
        cuda-cudart-dev-13-0 && \
    rm -rf /var/lib/apt/lists/*
```

Runtime components:

- **python${PYTHON_VERSION}/-dev/-venv**: Python runtime with development headers for extensions.
- **libgomp1**: OpenMP runtime library for parallel processing.
- **libcublas-13-0**: CUDA Basic Linear Algebra Subroutines runtime.
- **cuda-cupti-13-0**: Performance monitoring and profiling support.
- **cuda-nvtx-13-0**: NVIDIA Tools Extension for runtime analysis.
- **cuda-nvrtc-13-0**: NVIDIA Runtime Compilation library for JIT compilation.
- **libnccl2**: NVIDIA Collective Communications Library for multi-GPU.
- **nsight-systems-2025.3.2**: Deep profiling tool (optional, enabled through `NSYS_PROFILE`).
- **cuda-nvcc-13-0**: NVIDIA CUDA compiler retained for `torch.compile` JIT compilation.
- **cuda-cudart-dev-13-0**: CUDA runtime development headers for `torch.compile`.

### Virtual Environment Setup

```dockerfile
COPY --from=uv /uv /uvx /usr/local/bin/

RUN uv venv /opt/venv --python ${PYTHON_VERSION}
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:${PATH}"
```

- **Pinned uv binary**: Copied from the `uv` stage defined above; no network installer runs during the build.
- **Virtual environment**: Avoids PEP 668 externally-managed restrictions.
- **PATH configuration**: Ensures the venv binaries are used.

### vLLM Wheel Installation

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install vllm==${VLLM_VERSION} transformers==${TRANSFORMERS_VERSION} \
    --torch-backend=cu130
```

- **Official wheel**: Installs vLLM 0.30.0 from the published cu130 wheel; no source checkout or compilation.
- **`--torch-backend=cu130`**: Selects the PyTorch cu130 index, pulling in PyTorch 2.13 cu130, the version the lock file pins for the other images.
- **Blackwell kernels**: The vLLM wheel itself ships sm_120 kernel binaries for the RTX 5090.
- **transformers pinned in the same resolve**: vLLM requires only a floor, so an unpinned build takes the release current that day; the two are bumped together (KB-100).
- **uv cache mount**: Reuses the uv download cache across builds.

### FlashInfer Attention Kernels

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install flashinfer-jit-cache==${FLASHINFER_VERSION} \
    --index-url https://flashinfer.ai/whl/cu130
```

FlashInfer components:

- **flashinfer-python**: Arrives with the vLLM wheel, which pins it exactly; nothing installs it by name.
- **flashinfer-jit-cache==0.6.18.post1**: Pre-warmed JIT cache for custom attention kernel generation.
- **cu130 wheel index**: FlashInfer wheels built for CUDA 13.x.

```dockerfile
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch-c-dlpack-ext==0.1.5
```

- **torch-c-dlpack-ext==0.1.5**: Enables `EnvTensorAllocator` for TVM/DLPack interoperability with `torch.compile`.

### CUDA Runtime Environment

```dockerfile
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
```

```dockerfile
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV NVIDIA_REQUIRE_CUDA="cuda>=13.0"
```

- **NVIDIA_VISIBLE_DEVICES=all**: Exposes all available GPUs to the container.
- **NVIDIA_DRIVER_CAPABILITIES**: Enables compute, utility, and graphics capabilities.
- **NVIDIA_REQUIRE_CUDA**: Enforces the minimum CUDA version requirement.

### vLLM Settings

```dockerfile
ENV VLLM_USAGE_SOURCE=docker
ENV VLLM_LOGGING_LEVEL=INFO
ENV VLLM_USE_MODELSCOPE=False
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HUB_ENABLE_HF_TRANSFER=1
```

- **VLLM_USAGE_SOURCE=docker**: Telemetry source identifier.
- **VLLM_LOGGING_LEVEL=INFO**: Standard logging verbosity.
- **VLLM_USE_MODELSCOPE=False**: Disables ModelScope downloads in favor of Hugging Face.
- **VLLM_WORKER_MULTIPROC_METHOD=spawn**: Uses "spawn" for CUDA compatibility.
- **TOKENIZERS_PARALLELISM=false**: Disables tokenizer parallelism to avoid fork issues.
- **HF_HUB_ENABLE_HF_TRANSFER=1**: Enables fast Hugging Face model downloads.

### CUDA Performance Tuning

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

- **CUDA_MODULE_LOADING=LAZY**: Defers CUDA module loading for faster startup.
- **CUDA_DEVICE_ORDER=FASTEST_FIRST**: Selects the RTX 5090 when multiple GPUs are present.
- **CUDA_LAUNCH_BLOCKING=0**: Keeps kernel launches asynchronous.
- **CUDA_CACHE_DISABLE=0**: Enables CUDA kernel caching.
- **CUDA_FORCE_PTX_JIT=0**: Disables forced PTX JIT compilation for faster startup.

```dockerfile
ENV PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.95,max_split_size_mb:2048
```

- **expandable_segments:True**: Enables dynamic memory segment expansion.
- **garbage_collection_threshold:0.95**: Aggressive garbage collection for 32GB VRAM.
- **max_split_size_mb:2048**: Maximum memory block split size for efficient allocation.

### Experimental Blackwell Optimizations

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

These experimental settings remain commented out and should be enabled based on benchmarking results for the target workload.

### Security Configuration

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

Security features:

- **Dedicated user**: Non-root execution prevents privilege escalation.
- **No shell access**: `/sbin/nologin` prevents interactive container access.
- **Cache isolation**: Dedicated cache directories with defined ownership.
- **Hugging Face integration**: `HF_HOME` directs the model cache to a writable location.
- **Controlled permissions**: Explicit ownership changes before the user switch.

### Application Deployment

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=300s \
    CMD curl -f http://localhost:8003/health || exit 1
```

Health check parameters:

- **interval**: 30-second check frequency.
- **timeout**: 10-second response deadline.
- **retries**: 5 attempts before unhealthy status.
- **start-period**: 300-second grace period for large model loading.

```dockerfile
EXPOSE 8003

ENTRYPOINT ["/app/entrypoint.sh"]
```

The server starts through the entrypoint script, which provides parameterized configuration through environment variables.

## Entrypoint Configuration

The entrypoint script (`entrypoint.sh`) provides parameterized server configuration with environment variable overrides.

### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `MODEL_PATH` | **REQUIRED** - Path to the model. No default; the entrypoint fails fast if unset. |

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
| `ATTENTION_BACKEND` | flashinfer | Attention backend passed to `--attention-backend` |

### Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `TRUST_REMOTE_CODE` | false | Add `--trust-remote-code`; opt in per model for trusted sources only |
| `DISABLE_LOG_STATS` | true | Add `--disable-log-stats` to suppress statistics logging |

Prefix caching and chunked prefill are default-on in the vLLM V1 engine. The legacy `--enable-prefix-caching` and `--enable-chunked-prefill` flags are no longer passed by the entrypoint.

### Nsight Systems Profiling

| Variable | Default | Description |
|----------|---------|-------------|
| `NSYS_PROFILE` | false | Enable Nsight Systems profiling |
| `NSYS_OUTPUT` | /app/nsys_output/profile | Profile output path |

When `NSYS_PROFILE=true`, the server runs under nsys for deep CUDA profiling:

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

The entrypoint uses `vllm serve`, not `python -m vllm.entrypoints.openai.api_server`:

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
    --attention-backend "$ATTENTION_BACKEND" \
    --download-dir "/app/cache"
```

Optional flags added based on environment:

- `--quantization $QUANTIZATION` (if `QUANTIZATION` is set)
- `--trust-remote-code` (if `TRUST_REMOTE_CODE=true`)
- `--disable-log-stats` (if `DISABLE_LOG_STATS=true`)

Entrypoint features:

- Validates that `MODEL_PATH` is set and fails fast if missing.
- Reports GPU memory status through `nvidia-smi` before the model load.
- Displays a configuration summary on startup, including the attention backend.
- Supports optional Nsight Systems deep profiling.

## vLLM Performance Optimization Strategy

### Blackwell Architecture Targeting

The vLLM 0.30.0 cu130 wheel ships pre-built sm_120 (compute capability 12.0) kernel binaries for the RTX 5090, so no local compilation is required to target Blackwell. The implementation relies on:

- **sm_120 kernels**: Included in the official wheel for Blackwell streaming multiprocessors.
- **32GB VRAM**: Configured for large transformer model inference with memory management tuning.
- **FlashInfer 0.6.18.post1**: The default attention backend on Blackwell, selectable through `ATTENTION_BACKEND`.

### Memory Management

VRAM strategy:

- **GPU memory utilization**: `--gpu-memory-utilization=0.85` reserves 85% of 32GB VRAM.
- **Batching**: `--max-num-batched-tokens=32768` sets the maximum tokens per batch.
- **Sequence management**: `--max-num-seqs=256` bounds concurrent request handling.
- **Context length**: `--max-model-len=16384` sets the default maximum context.

Cache management:

- **Model cache**: `--download-dir=/app/cache` centralizes model storage in a writable directory.
- **CUDA cache**: `/tmp/cuda_cache` holds the kernel compilation cache.
- **Hugging Face cache**: `HF_HOME=/app/cache` consolidates transformer model artifacts.
- **Nsight output**: `/app/nsys_output` holds profiling data.

### Inference Optimization

Single-GPU configuration:

- **Tensor parallelism**: `--tensor-parallel-size=1` for single RTX 5090 deployment.
- **Worker method**: `VLLM_WORKER_MULTIPROC_METHOD=spawn` for clean process isolation.
- **Prefix caching and chunked prefill**: Default-on in the vLLM V1 engine; no explicit flags.

Tuning parameters:

- **Quantization**: Optional FP8, AWQ, or other methods through the `QUANTIZATION` env var.
- **Data type**: `--dtype=auto` selects the precision based on the model.
- **KV cache**: `--kv-cache-dtype=auto` for the default key-value storage precision.

## Container Security Implementation

Security layers:

- **Minimal runtime**: Only the required libraries are installed in the image.
- **Non-root execution**: The application runs as a dedicated non-privileged user.
- **No interactive access**: The user account has no shell.
- **Controlled GPU access**: Explicit GPU device specification and capability restrictions.
- **Cache isolation**: vLLM and CUDA caches are confined to container directories with defined permissions.

Health monitoring:

- **HTTP health checks**: Container health monitored through the OpenAI-compatible API.
- **Startup period**: 300-second grace period for large model loading.
- **Failure detection**: Container restart on health check failure.

Docker Compose integration adds:

- **Capability dropping**: Linux capabilities removed except GPU access.
- **Privilege restrictions**: `no-new-privileges:true` prevents escalation.
- **Memory constraints**: Controlled memory allocation with unlimited memlock for CUDA operations.
- **Read-only containers**: Filesystem mounted read-only except for designated cache areas.

## Reference Implementation

File structure:

- **docker/vllm-gpu/Dockerfile.vllm-gpu**: Single-stage vLLM Dockerfile.
- **docker/vllm-gpu/entrypoint.sh**: Parameterized server startup script.
- **docker-compose.inference.yaml**: Container orchestration with GPU resource allocation and security hardening.
- **env.example**: Environment template documenting `VLLM_GPU_MODEL` and related variables.

Integration with the AI workstation:

- **CPU containers**: Parallel execution for hybrid vLLM/llama.cpp workloads.
- **Workload split**: The vLLM container handles large transformer models while CPU containers manage smaller models.
- **Resource isolation**: Exclusive GPU access with controlled VRAM allocation.

Hardware requirements:

- **GPU**: RTX 5090 32GB (Blackwell architecture).
- **CUDA**: Version 13.0 or higher driver support.
- **System Memory**: Sufficient for container overhead and model preprocessing.
- **Storage**: Fast storage for model loading and cache operations.

Model compatibility:

- **Transformer architectures**: GPT, LLaMA, Qwen, DeepSeek, and other modern transformer models.
- **Quantization formats**: FP16, FP8, AWQ, GPTQ, and other precision formats.
- **Context lengths**: Up to 16K tokens by default, configurable through `MAX_MODEL_LEN`.
- **Hugging Face integration**: Direct model loading from Hugging Face Hub with fast transfers.

---

*This vLLM GPU Docker implementation provides transformer model serving on the RTX 5090 Blackwell architecture. The single-stage image installs the vLLM 0.30.0 official cu130 wheel with FlashInfer 0.6.18.post1 attention kernels while maintaining non-root execution and cache isolation.*
