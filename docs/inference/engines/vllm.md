# vLLM - High-Throughput GPU Inference

Technical deep-dive into vLLM, the GPU-optimized inference engine designed for maximum throughput in serving large language models.

## Table of Contents

- [Overview](#overview)
- [Architecture Philosophy](#architecture-philosophy)
- [PagedAttention](#pagedattention)
  - [Memory Fragmentation Problem](#memory-fragmentation-problem)
  - [Virtual Memory for KV Cache](#virtual-memory-for-kv-cache)
  - [Block Management](#block-management)
- [Continuous Batching](#continuous-batching)
  - [Dynamic Request Scheduling](#dynamic-request-scheduling)
  - [Iteration-Level Scheduling](#iteration-level-scheduling)
- [Tensor Parallelism](#tensor-parallelism)
  - [Model Sharding](#model-sharding)
  - [Communication Patterns](#communication-patterns)
- [Quantization Support](#quantization-support)
  - [FP8 Quantization](#fp8-quantization)
  - [GPTQ and AWQ](#gptq-and-awq)
- [Memory Management](#memory-management)
  - [GPU Memory Pool](#gpu-memory-pool)
  - [Swap Space](#swap-space)
  - [Preemption](#preemption)
- [Request Processing](#request-processing)
  - [Prefill Phase](#prefill-phase)
  - [Decode Phase](#decode-phase)
  - [Chunked Prefill](#chunked-prefill)
- [Our Implementation](#our-implementation)
  - [Docker Configuration](#docker-configuration)
  - [CUDA 13 Compatibility](#cuda-13-compatibility)
  - [Pending Optimizations](#pending-optimizations)
- [Performance Characteristics](#performance-characteristics)
- [Comparison with llama.cpp](#comparison-with-llamacpp)
- [Integration Points](#integration-points)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

## Overview

vLLM is a high-throughput, memory-efficient inference engine for LLMs. Unlike llama.cpp's focus on edge deployment and quantization, vLLM optimizes for serving many concurrent requests on datacenter GPUs through innovative memory management and scheduling algorithms.

Key innovations:
- PagedAttention for 24x higher throughput
- Continuous batching for dynamic request handling
- Zero memory waste through virtual memory management
- Native tensor parallelism for multi-GPU scaling
- Production-ready OpenAI-compatible API server

## Architecture Philosophy

vLLM follows a systems-oriented design philosophy:

```
Design Principles:
├── Memory efficiency through paging
├── Dynamic scheduling over static batching
├── Zero-copy tensor operations
├── Kernel fusion for reduced overhead
└── Hardware-aware optimization
```

Architectural comparison:

| Aspect | vLLM | llama.cpp |
|--------|------|-----------|
| Target | Datacenter GPUs | Edge devices |
| Focus | Throughput | Latency |
| Memory | Paged allocation | Static allocation |
| Batching | Continuous | Static |
| Quantization | FP8/GPTQ/AWQ | GGUF formats |
| Multi-GPU | Native support | Limited |

## PagedAttention

### Memory Fragmentation Problem

Traditional attention implementations suffer from memory fragmentation:

```
Standard KV Cache Allocation:
Request 1: [████████████████████            ] 60% used
Request 2: [██████████                      ] 30% used
Request 3: [████████████████                ] 50% used

Problem: 40% average memory waste due to pre-allocation
```

### Virtual Memory for KV Cache

PagedAttention borrows virtual memory concepts from operating systems:

```
Physical blocks (GPU memory):
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│P0│P1│P2│P3│P4│P5│P6│P7│P8│P9│PA│PB│
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

Logical view (per request):
Request 1: L0→P2, L1→P5, L2→P8
Request 2: L0→P0, L1→P3
Request 3: L0→P1, L1→P4, L2→P6, L3→P7
```

### Block Management

Block allocation and deallocation:

```python
# Simplified block manager
class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.free_blocks = list(range(num_blocks))
        self.block_tables = {}  # request_id -> block_list

    def allocate(self, request_id, num_tokens):
        blocks_needed = (num_tokens + block_size - 1) // block_size
        allocated = []
        for _ in range(blocks_needed):
            if self.free_blocks:
                allocated.append(self.free_blocks.pop())
        self.block_tables[request_id] = allocated

    def free(self, request_id):
        self.free_blocks.extend(self.block_tables[request_id])
        del self.block_tables[request_id]
```

Benefits:
- Near 100% memory utilization
- Dynamic memory sharing between requests
- Copy-on-write for shared prefixes

## Continuous Batching

### Dynamic Request Scheduling

Unlike static batching, continuous batching allows requests to join/leave dynamically:

```
Time →
T0: [Req1 Req2 Req3] → Process batch
T1: [Req1 Req2 Req3 Req4] → Req4 joins
T2: [Req2 Req3 Req4] → Req1 completes
T3: [Req2 Req3 Req4 Req5 Req6] → New requests join
```

### Iteration-Level Scheduling

Scheduling happens at each iteration, not per batch:

```python
while True:
    # Get ready requests
    running_batch = scheduler.get_running_batch()

    # Add new requests if space available
    if has_capacity():
        waiting_requests = scheduler.get_waiting()
        running_batch.extend(waiting_requests)

    # Process one iteration
    outputs = model.forward(running_batch)

    # Handle completions
    for req in running_batch:
        if req.is_finished():
            scheduler.mark_finished(req)
            free_memory(req)
```

## Tensor Parallelism

### Model Sharding

Distributing model layers across GPUs:

```
Single GPU:              2-GPU Tensor Parallel:
┌─────────────┐         ┌───────┐ ┌───────┐
│   Layer 1   │   →     │ L1.0  │ │ L1.1  │
├─────────────┤         ├───────┤ ├───────┤
│   Layer 2   │   →     │ L2.0  │ │ L2.1  │
├─────────────┤         ├───────┤ ├───────┤
│   Layer 3   │   →     │ L3.0  │ │ L3.1  │
└─────────────┘         └───────┘ └───────┘
                         GPU 0     GPU 1
```

### Communication Patterns

All-reduce operations for tensor parallelism:

```
Matrix multiplication with TP=2:
GPU 0: X × W0 = Y0
GPU 1: X × W1 = Y1
All-reduce: Y = Y0 + Y1
```

## Quantization Support

### FP8 Quantization

Native FP8 support for RTX 5090 (Blackwell):

```
FP8 Formats:
E4M3: 1 sign, 4 exponent, 3 mantissa (range-focused)
E5M2: 1 sign, 5 exponent, 2 mantissa (precision-focused)

Performance on RTX 5090:
- FP32: 82 TFLOPS
- FP16: 660 TFLOPS
- FP8:  1320 TFLOPS (2x FP16)
```

### GPTQ and AWQ

Weight-only quantization methods:

| Method | Quantization | Activation | Speed | Quality |
|--------|--------------|------------|-------|---------|
| FP8 | 8-bit | 8-bit | Fastest | 99% |
| GPTQ | 4-bit | 16-bit | Fast | 98% |
| AWQ | 4-bit | 16-bit | Fast | 98.5% |

## Memory Management

### GPU Memory Pool

Pre-allocated memory pools to reduce allocation overhead:

```
Memory Layout (32GB RTX 5090):
┌──────────────────────────────┐
│ Model Weights (16GB)         │
├──────────────────────────────┤
│ KV Cache Blocks (12GB)       │
├──────────────────────────────┤
│ Activation Memory (2GB)      │
├──────────────────────────────┤
│ CUDA Workspace (2GB)         │
└──────────────────────────────┘
```

### Swap Space

CPU memory can serve as overflow for GPU KV cache. The current deployment does
not pass `--swap-space`; the entrypoint only sets `--gpu-memory-utilization`.
`--swap-space` is available as a tuning option if overflow handling is needed:

```python
# Deployed setting (docker/vllm-gpu/entrypoint.sh)
--gpu-memory-utilization 0.85  # Leave headroom

# Optional tuning (not currently deployed):
# --swap-space 4  # 4GB CPU swap space for LRU block overflow
```

### Preemption

Request preemption for priority handling:

```
Preemption modes:
1. Recompute: Drop KV cache, recompute when resumed
2. Swap: Move KV cache to CPU memory
3. Sliding window: Keep only recent KV entries
```

The entrypoint does not pass `--preemption-mode`; vLLM uses its default. Swap
mode can be selected as a tuning option for long-context workloads.

## Request Processing

### Prefill Phase

Initial prompt processing:

```
Input: "What is machine learning?"
Tokens: [1518, 338, 4933, 6509, 29973]

Prefill computation:
- Process all tokens in parallel
- Generate KV cache for attention
- Output: First generated token
```

### Decode Phase

Token-by-token generation:

```
Iteration 1: Generate token 6
Iteration 2: Generate token 7
...
Iteration N: Generate EOS token

Each iteration produces one token per sequence.
```

### Chunked Prefill

Breaking large prompts into chunks:

```
Standard prefill (blocks GPU):
[────────────────────────] 2000 tokens, 400ms

Chunked prefill (interleaved):
[────][decode][────][decode][────][decode]
Chunks of 256 tokens, better latency for other requests
```

## Our Implementation

### Docker Configuration

Current production setup (vLLM 0.25.0 with CUDA 13.0.2):

```dockerfile
# Single-stage image on the CUDA runtime base
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

# Install the official vLLM cu130 wheel (ships sm_120 kernels and PyTorch 2.11 cu130)
RUN uv pip install vllm==0.25.0 --torch-backend=cu130

# Server started via entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
```

The entrypoint runs `vllm serve` with the default flags below. Prefix caching and chunked prefill are default-on in the vLLM V1 engine, so the legacy `--enable-prefix-caching` and `--enable-chunked-prefill` flags are not passed. The attention backend defaults to FlashInfer through `--attention-backend flashinfer` (override with the `ATTENTION_BACKEND` env var):

```bash
vllm serve /app/models/example-model-hf \
    --port 8003 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --attention-backend flashinfer \
    --disable-log-stats
```

### CUDA 13 Compatibility

The vLLM 0.25.0 official cu130 wheel targets CUDA 13.0.2:

```
Status: vLLM 0.25.0 official cu130 wheel targets CUDA 13.0.2
Base:   nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04
PyTorch: 2.11 cu130
FlashInfer: 0.6.13
```

### Active Optimizations

The following CUDA environment variables are set in
`docker/vllm-gpu/Dockerfile.vllm-gpu`:

```bash
# 2GB kernel cache
ENV CUDA_CACHE_MAXSIZE=2147483648
# Lazy module loading for faster startup
ENV CUDA_MODULE_LOADING=LAZY
# Order GPU devices by performance
ENV CUDA_DEVICE_ORDER=FASTEST_FIRST
# Async kernel execution
ENV CUDA_LAUNCH_BLOCKING=0
```

### Experimental Optimizations (not currently enabled)

The following variables are present in the Dockerfile but commented out. They
are documented as tuning candidates and require benchmarking before enabling:

```bash
# Async memory allocation
# ENV CUDA_ALLOCATOR_BACKEND=cudaMallocAsync
# ENV CUDA_MALLOC_ASYNC_POOLS=1
# Tensor core / cuBLAS workspace tuning
# ENV CUDA_TENSOR_CORES=1
# ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
# ENV CUDNN_TENSOR_OPS=1
# Blackwell L2 cache (96MB) and connection tuning
# ENV CUDA_L2_PERSISTING_SIZE=100663296
# ENV CUDA_DEVICE_MAX_CONNECTIONS=32
# ENV CUDA_COPY_SPLIT_THRESHOLD=256
```

## Performance Characteristics

No benchmark has been run on this workstation. The values below are configured
settings, not measured results.

| Metric | Value | Notes |
|--------|-------|-------|
| Max sequences | 256 | `--max-num-seqs` setting |
| GPU memory utilization | 0.85 | `--gpu-memory-utilization` setting |
| FP8 support | Available | Native on RTX 5090 |

Qualitative comparison with llama.cpp on the same hardware:

| Aspect | vLLM | llama.cpp |
|--------|------|-----------|
| Concurrent throughput | Higher (continuous batching) | Lower (single stream) |
| Single-stream latency | Higher | Lower |
| Memory efficiency | Paged KV cache | Near model size |
| Quantization | FP8/GPTQ | GGUF |

## Comparison with llama.cpp

When to use each engine:

### Use vLLM when:
- Serving many concurrent users
- Throughput more important than latency
- Using standard HuggingFace models
- Multi-GPU deployment needed
- Dynamic batching required

### Use llama.cpp when:
- Single-user or few users
- Lowest latency critical
- Edge deployment (CPU/mobile)
- Custom quantization needed
- Memory constraints tight

## Integration Points

### API Compatibility

OpenAI-compatible endpoints:

```
POST /v1/completions
POST /v1/chat/completions
POST /v1/embeddings
GET /v1/models
GET /health
```

### Docker Integration

Service configuration:

```yaml
vllm-gpu:
  container_name: vllm-gpu
  ports:
    - "127.0.0.1:8003:8003"
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Monitoring

Metrics exposed by vLLM:

```json
{
  "num_running_requests": 5,
  "num_waiting_requests": 2,
  "gpu_memory_usage": 27648000000
}
```

## Troubleshooting

Common issues and solutions:

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM on startup | Model too large | Reduce --gpu-memory-utilization |
| Slow generation | Poor batching | Increase --max-num-seqs |
| High latency | Prefill blocking | Chunked prefill is default-on in the V1 engine; reduce --max-num-batched-tokens if it starves decode |
| CUDA error | Version mismatch | Ensure CUDA 13.0.2 and PyTorch 2.11 cu130 |
| Memory leak | Cache fragmentation | Restart service periodically |

## Next Steps

- Run benchmarks to validate projected performance numbers
- Review [parameters/vllm-parameters.md](../parameters/vllm-parameters.md) for tuning
- Compare with [llama-cpp.md](llama-cpp.md) for deployment decisions
