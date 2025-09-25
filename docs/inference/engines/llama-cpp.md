# llama.cpp - High-Performance C++ Inference

Technical deep-dive into llama.cpp, the inference engine powering our CPU and GPU model serving.

## Table of Contents

- [Overview](#overview)
- [Architecture Philosophy](#architecture-philosophy)
- [GGUF Format](#gguf-format)
  - [File Structure](#file-structure)
  - [Quantization Schemes](#quantization-schemes)
- [Memory Management](#memory-management)
  - [Memory Mapping](#memory-mapping)
  - [Memory Locking](#memory-locking)
  - [Huge Pages Integration](#huge-pages-integration)
- [CPU Optimization](#cpu-optimization)
  - [SIMD Instructions](#simd-instructions)
  - [Thread Management](#thread-management)
  - [Cache Optimization](#cache-optimization)
- [GPU Acceleration](#gpu-acceleration)
  - [CUDA Integration](#cuda-integration)
  - [Layer Offloading](#layer-offloading)
  - [Unified Memory](#unified-memory)
- [Inference Pipeline](#inference-pipeline)
  - [Model Loading](#model-loading)
  - [Context Processing](#context-processing)
  - [Token Generation](#token-generation)
- [Batch Processing](#batch-processing)
  - [Static Batching](#static-batching)
  - [Continuous Batching](#continuous-batching)
- [Our Implementation](#our-implementation)
  - [CPU Configuration](#cpu-configuration)
  - [GPU Configuration](#gpu-configuration)
- [Performance Characteristics](#performance-characteristics)
- [Integration Points](#integration-points)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

## Overview

llama.cpp is a plain C/C++ implementation of LLaMA model inference, designed for maximum performance and minimal dependencies. Created by Georgi Gerganov, it enables running large language models on consumer hardware through aggressive optimization and quantization.

Key characteristics:
- Zero dependency on Python or frameworks
- Native support for 4-bit to 32-bit quantization
- Unified codebase for CPU and GPU execution
- Memory-efficient through mmap and streaming
- Production-ready server implementation

## Architecture Philosophy

llama.cpp follows a minimalist design philosophy:

```
Design Principles:
├── Single-file models (GGUF format)
├── Zero-copy memory mapping
├── Static memory allocation
├── Compiler-optimized hot paths
└── Platform-native SIMD usage
```

Comparison with framework-based inference:

| Aspect | llama.cpp | PyTorch/Transformers |
|--------|-----------|---------------------|
| Dependencies | None | Python, CUDA toolkit, 100+ packages |
| Memory overhead | ~10% | ~40% |
| Startup time | <1 second | 10-30 seconds |
| Binary size | ~10MB | ~10GB |
| Quantization | Native | External tools required |

## GGUF Format

GGUF (GGML Universal Format) is llama.cpp's model format, replacing the older GGML format.

### File Structure

```
GGUF File Layout:
┌──────────────────┐
│   Magic Number   │ "GGUF" (4 bytes)
├──────────────────┤
│     Version      │ uint32 (currently 3)
├──────────────────┤
│  Tensor Count    │ uint64
├──────────────────┤
│  Metadata KV     │ Model parameters
├──────────────────┤
│  Tensor Info     │ Names, shapes, types
├──────────────────┤
│ Alignment Padding│
├──────────────────┤
│  Tensor Data     │ Quantized weights
└──────────────────┘
```

### Quantization Schemes

Available quantization formats and their characteristics:

| Format | Bits/weight | Model Size (30B) | Speed | Quality |
|--------|------------|------------------|-------|---------|
| Q2_K   | 2.56       | 10.5 GB         | 5.0x  | 93%     |
| Q3_K_S | 3.16       | 12.3 GB         | 4.5x  | 95%     |
| Q4_0   | 4.34       | 16.9 GB         | 4.0x  | 97%     |
| Q4_K_S | 4.58       | 17.3 GB         | 3.8x  | 97.5%   |
| Q5_K_S | 5.21       | 19.6 GB         | 3.5x  | 98%     |
| Q6_K   | 6.33       | 23.5 GB         | 3.0x  | 98.5%   |
| Q8_0   | 8.50       | 31.0 GB         | 2.5x  | 99%     |
| F16    | 16.00      | 58.0 GB         | 2.0x  | 99.9%   |
| F32    | 32.00      | 116.0 GB        | 1.0x  | 100%    |

## Memory Management

### Memory Mapping

llama.cpp uses mmap for efficient model loading:

```c
// Simplified memory mapping flow
int fd = open("model.gguf", O_RDONLY);
void* addr = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

// Benefits:
// - No loading time (lazy page fault)
// - OS manages memory paging
// - Shared between processes
// - Automatic cache management
```

### Memory Locking

Prevents model swapping to disk:

```c
// Our configuration uses --mlock flag
mlock(model_data, model_size);

// Effect on our system:
// - 15.3GB locked in RAM for 30B model
// - Zero swap usage
// - Consistent inference latency
// - No disk I/O during generation
```

### Huge Pages Integration

Our custom wrapper enables huge pages for models >1GB:

```
Standard Pages (4KB):        Huge Pages (2MB):
15.3GB model                 15.3GB model
= 4,013,056 pages           = 7,834 pages
= High TLB pressure         = Low TLB pressure
```

Implementation via `hugepage_mmap_wrapper.so`:
1. Intercepts mmap() calls
2. Allocates huge pages with MAP_HUGETLB
3. Loads model into huge page memory
4. Transparent to llama.cpp

## CPU Optimization

### SIMD Instructions

CPU path utilizes architecture-specific SIMD:

```
AMD Zen 5 (our CPU):
├── AVX-512 (512-bit vectors)
├── AVX-512 VNNI (INT8 acceleration)
├── AVX-512 BF16 (bfloat16)
└── Native 512-bit datapath

Operations per cycle:
- FP32: 16 ops (AVX-512)
- INT8: 64 ops (VNNI)
- Matrix: 256 ops (tensor-like)
```

### Thread Management

Optimized threading strategy:

```
Our Configuration (12 cores):
├── Main thread: Coordination
├── Worker threads: 12 (one per core)
├── HTTP threads: 2 (API handling)
└── No SMT: Exclusive L1/L2 cache

Thread allocation:
- Cores 0-11: LLM computation
- Core 12-15: System/OS tasks
```

### Cache Optimization

Memory access patterns optimized for cache:

```
Cache Hierarchy (Zen 5):
L1: 32KB per core (private)
L2: 1MB per core (private)
L3: 32MB per CCX (shared)

Optimization strategies:
- Weight reuse within L2
- Activation streaming
- Prefetch next layer
- Cache-aligned allocation
```

## GPU Acceleration

### CUDA Integration

GPU acceleration through CUDA kernels:

```
Kernel types:
├── GEMM kernels (matrix multiply)
├── Attention kernels (scaled dot product)
├── Activation kernels (GELU, SiLU)
├── Normalization kernels (RMSNorm)
└── Sampling kernels (top-k, top-p)
```

### Layer Offloading

Flexible GPU memory management:

```
--n-gpu-layers parameter:
0    = Pure CPU inference
35   = Attention layers on GPU
70   = All layers except embed/output
999  = Full model on GPU (our default)

Memory distribution (30B model):
- GPU (999 layers): 15.3GB VRAM
- GPU (35 layers): 8GB VRAM + 7.3GB RAM
- CPU only: 15.3GB RAM
```

### Unified Memory

Seamless CPU-GPU data transfer:

```
Data flow:
Model weights (VRAM)
     ↓
CUDA kernels process
     ↓
Results to RAM (if needed)
     ↓
CPU post-processing

Our configuration:
- No --no-mmap: Direct GPU loading
- No CPU fallback needed
- Zero-copy where possible
```

## Inference Pipeline

### Model Loading

Loading sequence for our deployment:

```
1. Open GGUF file
2. Read metadata (model architecture, vocab)
3. mmap weight tensors
4. Allocate compute buffers
5. Initialize CUDA (if GPU)
6. Copy weights to VRAM (if GPU)
7. Create KV cache
```

Timing on our system:
- CPU model load: ~2 seconds
- GPU model load: ~5 seconds

### Context Processing

Prefill phase (initial prompt):

```
Input tokens: [1518, 9234, ...]
     ↓
Embedding lookup
     ↓
For each layer (1-80):
  - Self-attention
  - MLP
  - Normalization
     ↓
Final hidden states
```

Performance:
- CPU: ~1000 tokens/second prefill
- GPU: ~8000 tokens/second prefill

### Token Generation

Autoregressive generation loop:

```
While not EOS:
  1. Forward pass (single token)
  2. Logit computation
  3. Sampling (top-k, top-p, temp)
  4. Token selection
  5. Update KV cache
  6. Yield token

Our speeds:
- CPU: 35.44 tokens/second
- GPU: 286.85 tokens/second
```

## Batch Processing

### Static Batching

Default llama.cpp behavior:

```
Request 1 → Process → Complete
Request 2 → Process → Complete
Request 3 → Process → Complete

Characteristics:
- Sequential processing
- Predictable latency
- Fixed memory usage
- Simple scheduling
```

### Continuous Batching

Enabled with `--cont-batching`:

```
Request 1 ─┐
Request 2 ─┼→ Batch → Process → Stream
Request 3 ─┘

Benefits:
- Parallel processing
- Better GPU utilization
- Dynamic batching

Tradeoffs:
- Variable latency
- Complex memory management
- Higher memory usage
```

Our configuration:
- CPU: Continuous batching enabled
- GPU: Static batching (lower latency)

## Our Implementation

### CPU Configuration

Production parameters from `docker/llama-cpu/entrypoint.sh`:

```bash
./server \
    --model "$MODEL_PATH" \           # 30B Q4 model
    --ctx-size 32768 \                # 32K context window
    --batch-size 2048 \               # Optimal batch size
    --ubatch-size 2048 \              # Physical batch = logical
    --threads 12 \                    # 12 CPU cores
    --threads-batch 12 \              # Batch processing threads
    --cont-batching \                 # Continuous batching on
    --mlock \                         # Lock model in RAM
    --threads-http 2                  # API handler threads
```

Key optimizations:
- Huge pages via LD_PRELOAD wrapper
- Memory locked to prevent swapping
- Dedicated core allocation (0-11)
- No GPU layers (pure CPU)

### GPU Configuration

Production parameters from `docker/llama-gpu/entrypoint.sh`:

```bash
./server \
    --model "$MODEL_PATH" \           # 20B Q8 model
    --n-gpu-layers 999 \              # Full GPU offload
    --ctx-size 65536 \                # 64K context window
    --batch-size 2048 \               # Optimal batch size
    --ubatch-size 512 \               # Smaller physical batch
    --threads 1 \                     # Minimal CPU threads
    --flash-attn \                    # Flash Attention enabled
    --no-mmap \                       # Direct GPU loading
    --main-gpu 0 \                    # RTX 5090
    --parallel 1                      # Single sequence
```

Key optimizations:
- Full model in VRAM
- Flash Attention for memory efficiency
- Optimized batch/ubatch ratio
- No CPU computation

## Performance Characteristics

Measured performance on our hardware:

| Metric | CPU (Zen 5) | GPU (RTX 5090) |
|--------|-------------|----------------|
| Model | 30B Q4_XS | 20B Q8_K_XL |
| Prefill speed | 1000 tok/s | 8000 tok/s |
| Generation | 35.44 tok/s | 286.85 tok/s |
| First token | 200ms | 50ms |
| Memory usage | 15.3GB RAM | 10.5GB VRAM |
| Power draw | 120W | 437W |

## Integration Points

### Docker Integration

Container configuration:

```yaml
llama-cpu:
  cpuset: "0-11"        # CPU affinity
  memory: 96G           # RAM limit
  cap_add: [IPC_LOCK]   # For mlock
  volumes:
    - /mnt/ai-data/models:/app/models:ro
```

### API Compatibility

OpenAI-compatible endpoints:

```
POST /v1/completions
POST /v1/chat/completions
POST /v1/embeddings
GET  /v1/models
GET  /health
```

### Monitoring

Key metrics exposed:

```json
{
  "tokens_per_second": 35.44,
  "prompt_tokens": 256,
  "completion_tokens": 512,
  "kv_cache_used": 768,
  "queue_pending": 0
}
```

## Troubleshooting

Common issues and solutions:

| Issue | Cause | Solution |
|-------|-------|----------|
| Slow inference | Wrong batch size | Use 2048 for our hardware |
| High latency | Model swapping | Enable --mlock |
| GPU OOM | Context too large | Reduce --ctx-size |
| CPU 100% | Too many threads | Match physical cores (12) |
| Crash on load | Corrupted model | Re-download GGUF file |

## Next Steps

- Review [parameters/llama-cpp-parameters.md](../parameters/llama-cpp-parameters.md) for tuning
- See [vLLM](vllm.md) for throughput-optimized serving
- Check [benchmarks/performance-results.md](../benchmarks/performance-results.md) for measurements

---

*Last Updated: 2025-09-23*