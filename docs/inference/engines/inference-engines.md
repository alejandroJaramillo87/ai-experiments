# Inference Engines - The AI Model Runtime

Understanding inference engines and their role in AI model deployment on our workstation.

## Table of Contents

<!-- toc:start -->

1. [What Are Inference Engines?](#what-are-inference-engines)
   1. [Inference vs Training](#inference-vs-training)
   2. [Why Not Use Training Frameworks?](#why-not-use-training-frameworks)
2. [Core Functions of Inference Engines](#core-functions-of-inference-engines)
   1. [Memory Management](#memory-management)
   2. [Quantization Support](#quantization-support)
   3. [Batch Processing](#batch-processing)
   4. [Hardware Optimization](#hardware-optimization)
3. [Architecture Components](#architecture-components)
   1. [Model Loading](#model-loading)
   2. [Tokenization](#tokenization)
   3. [Execution Engine](#execution-engine)
   4. [API Server](#api-server)
4. [Integration with Hardware Stack](#integration-with-hardware-stack)
   1. [CPU Path](#cpu-path)
   2. [GPU Path](#gpu-path)
5. [Inference Strategies](#inference-strategies)
   1. [Latency Optimization](#latency-optimization)
   2. [Throughput Optimization](#throughput-optimization)
6. [Memory Optimization Techniques](#memory-optimization-techniques)
   1. [Weight Quantization](#weight-quantization)
   2. [KV Cache Management](#kv-cache-management)
   3. [Memory Mapping](#memory-mapping)
7. [Our Implementation](#our-implementation)
8. [Performance Characteristics](#performance-characteristics)
9. [Next Steps](#next-steps)

<!-- toc:end -->

## What Are Inference Engines?

Inference engines are specialized runtime systems designed to execute trained AI models efficiently. Unlike training frameworks that optimize for gradient computation and parameter updates, inference engines optimize for forward pass execution speed and memory efficiency.

### Inference vs Training

```
Training (PyTorch/TensorFlow)         Inference (llama.cpp/vLLM)
─────────────────────────────         ──────────────────────────
Forward pass     ───→                 Forward pass only
Gradient calc    ←───                 No gradients
Weight update    ←→                   Read-only weights
High memory      32GB+                Low memory 4-16GB
Batch oriented   Large batches        Single/small batch
Float32/16       Full precision       INT4/8 quantization
```

### Why Not Use Training Frameworks?

Training frameworks like PyTorch include overhead unnecessary for inference:

1. **Gradient Tracking** - Memory overhead for autograd graph
2. **Dynamic Graphs** - Flexibility costs performance
3. **Memory Inefficiency** - Stores activations for backpropagation
4. **Quantization Limitations** - Limited support for aggressive quantization
5. **Batch Requirements** - Optimized for large batch training

## Core Functions of Inference Engines

### Memory Management

Inference engines implement specialized memory management for model weights and activations:

```
Model Loading Pipeline:
1. Read model file (GGUF/SafeTensors)
2. Memory map weights (mmap)
3. Lock pages in RAM (mlock)
4. Allocate KV cache
5. Initialize compute buffers
```

### Quantization Support

Native support for reduced precision formats:

| Format | Bits | Memory | Speed | Quality |
|--------|------|--------|-------|---------|
| FP32   | 32   | 100%   | 1.0x  | 100%    |
| FP16   | 16   | 50%    | 2.0x  | 99.9%   |
| INT8   | 8    | 25%    | 3.5x  | 99.5%   |
| INT4   | 4    | 12.5%  | 5.0x  | 98%     |

### Batch Processing

Two primary strategies for handling multiple requests:

**Static Batching** (llama.cpp default):
- Process requests sequentially
- Consistent memory usage
- Predictable latency

**Continuous Batching** (vLLM):
- Process multiple requests simultaneously
- Dynamic memory allocation
- Higher throughput

### Hardware Optimization

Direct integration with hardware acceleration:

```
CPU Optimizations:
├── AVX-512 VNNI (Zen 5)
├── Memory prefetching
├── NUMA awareness
└── Thread affinity

GPU Optimizations:
├── CUDA kernels
├── Tensor cores
├── Flash Attention
└── CUDA graphs
```

## Architecture Components

### Model Loading

```
Model File → Parser → Weight Mapping → Memory Allocation
    ↓          ↓           ↓                ↓
  GGUF      Metadata    Quantized      RAM/VRAM
  Format    Extraction   Tensors       Placement
```

### Tokenization

Text to token conversion pipeline:

```
"Hello world" → Tokenizer → [15496, 1917] → Embedding
                    ↓              ↓            ↓
                BPE/WordPiece   Token IDs   Vector[4096]
```

### Execution Engine

Core computation loop:

```
for layer in model.layers:
    hidden = layer.attention(hidden, kv_cache)
    hidden = layer.mlp(hidden)
    hidden = layer.norm(hidden)
return model.lm_head(hidden)
```

### API Server

Request handling interface:

```
HTTP Request → Queue → Scheduler → Engine → Response
     ↓           ↓         ↓         ↓         ↓
   OpenAI     Priority  Batching  Compute   Stream
   Format     Queue     Logic     Execute   SSE/JSON
```

## Integration with Hardware Stack

### CPU Path

```
Application (llama.cpp)
    ↓
CPU Instructions (AVX-512)
    ↓
Memory Controller
    ↓
DDR5 RAM (96GB/s)
```

Optimizations:
- Huge pages (2MB) for reduced TLB misses
- Memory locking to prevent swapping
- Thread pinning to specific cores
- NUMA-aware memory allocation

### GPU Path

```
Application (llama.cpp/vLLM)
    ↓
CUDA Runtime
    ↓
cuBLAS/cuDNN Libraries
    ↓
CUDA Cores/Tensor Cores
    ↓
VRAM (32GB GDDR7)
```

Optimizations:
- Kernel fusion for reduced overhead
- Tensor core utilization for matrix ops
- Memory pooling for allocation efficiency
- CUDA graphs for kernel launch optimization

## Inference Strategies

### Latency Optimization

Minimize time to first token and total response time:

```
Request → Load Context → Generate Token 1 → ... → Token N
  0ms        50ms           55ms                   500ms

Optimizations:
- Prefill phase acceleration
- Speculative decoding
- KV cache reuse
- Memory pre-allocation
```

### Throughput Optimization

Maximize tokens generated per second across all requests:

```
Request 1 ─┐
Request 2 ─┼→ Batch → Process → Split → Response 1
Request 3 ─┘                            → Response 2
                                       → Response 3

Optimizations:
- Continuous batching
- Dynamic scheduling
- Memory paging (vLLM)
- Request prioritization
```

## Memory Optimization Techniques

### Weight Quantization

Reducing model weight precision:

```
Original (FP32):        Quantized (INT4):
[3.14159, 2.71828]  →  [7, 6] + scale=0.45
32 bits/weight      →  4 bits/weight + metadata
```

### KV Cache Management

Attention key-value storage optimization:

```
Standard KV Cache:          Paged KV Cache (vLLM):
┌─────────────────┐        ┌──┬──┬──┬──┐
│  Contiguous     │   →    │P1│P2│  │P3│
│  Memory Block   │        └──┴──┴──┴──┘
└─────────────────┘        Non-contiguous pages
```

### Memory Mapping

Direct file-to-memory mapping:

```
Model File on Disk
      ↓
   mmap() call
      ↓
Virtual Memory (lazy load)
      ↓
Physical RAM (on access)
```

Benefits:
- No explicit loading time
- OS manages paging
- Shared between processes
- Reduced memory duplication

## Our Implementation

Our workstation runs two inference engines optimized for different workloads:

| Engine | Target | Optimization | Use Case |
|--------|--------|--------------|----------|
| llama.cpp | CPU/GPU | Latency | Interactive chat |
| vLLM | GPU | Throughput | API serving |

Architecture flow:
```
User Request
     ↓
Docker Container
     ↓
Inference Engine (llama.cpp/vLLM)
     ├── CPU Path: 16 cores, 96GB RAM
     └── GPU Path: RTX 5090, 32GB VRAM
```

## Performance Characteristics

Key metrics for inference engines:

| Metric | Definition | llama.cpp | vLLM |
|--------|-----------|-----------|------|
| Time to First Token | Latency before first output | Lower | Higher |
| Generation speed | Tokens per second | Latency-optimized | Throughput-optimized |
| Memory Efficiency | Model size vs RAM usage | Near model size | Paged KV cache |
| Batch Efficiency | Performance scaling with batch | Limited | Scales with concurrency |

No benchmark has been run on this workstation; the entries above are
qualitative.

## Next Steps

- Review [llama.cpp](llama-cpp.md) for CPU/GPU unified inference
- Review [vLLM](vllm.md) for high-throughput GPU serving
- See [parameters/optimization-guide.md](../parameters/llama-cpp-parameters.md) for tuning
