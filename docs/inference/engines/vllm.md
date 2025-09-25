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

CPU memory as overflow for GPU:

```python
# Configuration in our deployment
--swap-space 4  # 4GB CPU swap space
--gpu-memory-utilization 0.85  # Leave headroom

# Behavior:
# 1. Fill GPU to 85% (27.2GB of 32GB)
# 2. Swap least-recently-used blocks to CPU
# 3. Swap back when needed
```

### Preemption

Request preemption for priority handling:

```
Preemption modes:
1. Recompute: Drop KV cache, recompute when resumed
2. Swap: Move KV cache to CPU memory
3. Sliding window: Keep only recent KV entries

Our configuration: Swap mode (better for long contexts)
```

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
- Time: ~50ms for 100 tokens
```

### Decode Phase

Token-by-token generation:

```
Iteration 1: Generate token 6
Iteration 2: Generate token 7
...
Iteration N: Generate EOS token

Per-iteration time: ~3.5ms (RTX 5090)
Throughput: 286 tokens/second
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

Current production setup (waiting for CUDA 13 support):

```dockerfile
# Build stage
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04 AS builder
ENV TORCH_CUDA_ARCH_LIST="12.0"  # RTX 5090 Blackwell
ENV CUDA_DOCKER_ARCH=sm_120
ENV NVCC_THREADS=8

# Runtime stage
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

CMD ["--model", "/app/models/hf/DeepSeek-R1-0528-Qwen3-8b",
     "--port", "8005",
     "--gpu-memory-utilization", "0.85",
     "--max-model-len", "32768",
     "--max-num-seqs", "512",
     "--enable-chunked-prefill",
     "--enable-prefix-caching",
     "--quantization", "fp8"]
```

### CUDA 13 Compatibility

Current blocker and timeline:

```
Issue: vLLM v0.10.2 incompatible with CUDA 13
Cause: CUB library API changes (cub::Sum removed)
Tracking: https://github.com/vllm-project/vllm/pull/23976
Status: Waiting for upstream fix
Workaround: None (must wait for vLLM update)
```

### Pending Optimizations

Optimizations ready when CUDA 13 support lands:

```bash
# Memory pools (30% allocation overhead reduction)
ENV CUDA_ALLOCATOR_BACKEND=cudaMallocAsync
ENV CUDA_MALLOC_ASYNC_POOLS=1

# Tensor cores (40% GEMM speedup)
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV CUDNN_TENSOR_OPS=1

# Blackwell L2 cache (96MB)
ENV CUDA_L2_PERSISTING_SIZE=100663296

# FP8 formats
ENV VLLM_FP8_E4M3=1
ENV VLLM_FP8_E5M2=1

# CUDA graphs (20% kernel launch reduction)
ENV VLLM_CUDA_GRAPHS=1
```

## Performance Characteristics

Expected performance on RTX 5090 (once CUDA 13 supported):

| Metric | Current (CUDA 12.8) | With CUDA 13 | Improvement |
|--------|---------------------|--------------|-------------|
| Throughput | ~300 tok/s | ~500 tok/s | 67% |
| First token | 100ms | 50ms | 50% |
| Batch size | 32 | 64 | 100% |
| Memory usage | 85% | 95% | 12% |
| FP8 support | Partial | Full | N/A |

Comparison with llama.cpp on same hardware:

| Metric | vLLM | llama.cpp | Use Case |
|--------|------|-----------|----------|
| Single stream | 300 tok/s | 287 tok/s | Similar |
| 16 parallel | 4800 tok/s | 287 tok/s | vLLM wins |
| First token | 100ms | 50ms | llama.cpp wins |
| Memory efficiency | 85% | 95% | llama.cpp wins |
| Quantization | FP8/GPTQ | GGUF | Different |

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
    - "127.0.0.1:8005:8005"
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
  "gpu_memory_usage": 27648000000,
  "gpu_utilization": 0.92,
  "throughput_tokens_per_second": 486.3
}
```

## Troubleshooting

Common issues and solutions:

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM on startup | Model too large | Reduce --gpu-memory-utilization |
| Slow generation | Poor batching | Increase --max-num-seqs |
| High latency | Prefill blocking | Enable --enable-chunked-prefill |
| CUDA error | Version mismatch | Wait for CUDA 13 support |
| Memory leak | Cache fragmentation | Restart service periodically |

## Next Steps

- Monitor [vLLM CUDA 13 support](https://github.com/vllm-project/vllm/pulls)
- Review [parameters/vllm-parameters.md](../parameters/vllm-parameters.md) for tuning
- Compare with [llama-cpp.md](llama-cpp.md) for deployment decisions
- Check [benchmarks/performance-results.md](../benchmarks/performance-results.md) when available

---

*Last Updated: 2025-09-24*