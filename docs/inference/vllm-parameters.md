# vLLM Parameter Optimization

Detailed analysis of runtime parameters for vLLM server optimized for high-throughput concurrent inference on the RTX 5090 with focus on maximizing request handling capacity.

## Current Parameter Configuration

### vllm-gpu Service (Port 8005)

**Runtime Parameters (Dockerfile.vllm-gpu)**:
```bash
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

**Service Status**: Waiting for CUDA 13.0 support in vLLM v0.10.2

## Parameter Analysis

### GPU Resource Management

**`--tensor-parallel-size`**
- **Current**: 1 (single GPU)
- **Purpose**: Distribute model across multiple GPUs
- **Note**: Single RTX 5090 setup, no parallelism needed

**`--gpu-memory-utilization`**
- **Current**: 0.85 (85% of 32GB VRAM)
- **Recommended**: 0.99 (99% utilization for maximum performance)
- **Purpose**: Control VRAM allocation for KV cache
- **Improvement needed**: Increase to 0.99 for better throughput

### Sequence and Batching Configuration

**`--max-num-seqs`**
- **Current**: 512 concurrent sequences
- **Purpose**: Maximum number of simultaneous requests
- **Trade-off**: More sequences = higher throughput, more memory usage

**`--max-num-batched-tokens`**
- **Current**: 65536 tokens per batch
- **Purpose**: Maximum tokens processed in single batch
- **Optimization**: Large batches for GPU efficiency

**`--max-model-len`**
- **Current**: 32768 (32K context)
- **Purpose**: Maximum context length per sequence
- **Note**: Balanced for throughput vs memory

### Memory and Caching Optimizations

**`--enable-prefix-caching`**
- **Status**: Enabled
- **Purpose**: Cache common prompt prefixes
- **Benefit**: Faster processing for repeated patterns

**`--enable-chunked-prefill`**
- **Status**: Enabled
- **Purpose**: Split large prefill operations into chunks
- **Benefit**: Better batching efficiency

**`--kv-cache-dtype`**
- **Current**: "auto" (automatic selection)
- **Purpose**: KV cache quantization
- **Note**: Could optimize with explicit quantization

### Model and Data Configuration

**`--dtype`**
- **Current**: "auto"
- **Purpose**: Model computation precision
- **Note**: Automatic selection based on model

**`--quantization`**
- **Current**: "fp8"
- **Purpose**: FP8 quantization for memory efficiency
- **Benefit**: ~2x memory savings with minimal quality loss on RTX 5090

**`--trust-remote-code`**
- **Status**: Enabled
- **Purpose**: Allow custom model code execution
- **Required**: For many modern models

## Missing Parameters to Research

Based on 2025 vLLM documentation, we should consider:

### Performance Optimizations

**`--max-seq-len-to-capture`**
- **Purpose**: Maximum sequence length for Hip-graphs
- **Recommended**: Set to max-model-len (32768) for best performance
- **Priority**: High

**`--num-scheduler-steps`**
- **Purpose**: Number of scheduler steps per iteration
- **Benefit**: Fine-tune scheduling efficiency
- **Priority**: Medium

**`--speculative-model`**
- **Purpose**: Speculative decoding for throughput improvement
- **Requirement**: Needs smaller draft model
- **Priority**: Medium (requires additional setup)

### Advanced Features (2025)

**`--quantized-kv-cache`**
- **Purpose**: Quantize KV cache for memory savings
- **Benefit**: More sequences or longer contexts
- **Priority**: High

**`--disable-sliding-window`**
- **Purpose**: Control sliding window attention
- **Use case**: Model-specific optimization
- **Priority**: Low

### Memory Management

**`--swap-space`**
- **Purpose**: CPU memory swap space for overflow
- **Current**: Not configured
- **Benefit**: Handle temporary memory spikes
- **Priority**: Medium

**`--preemption-mode`**
- **Purpose**: How to handle memory pressure
- **Options**: "swap" or "recompute"
- **Priority**: Medium

## Optimization Opportunities

### Immediate Improvements
1. **Increase `--gpu-memory-utilization`** from 0.85 to 0.99
2. **Add `--max-seq-len-to-capture=32768`** for Hip-graphs
3. **Test `--quantized-kv-cache`** for memory efficiency
4. **Configure `--swap-space`** for overflow handling

### Throughput Optimization
1. **Benchmark different `--max-num-seqs`** values
2. **Test `--max-num-batched-tokens`** variations
3. **Evaluate `--speculative-model`** setup
4. **Optimize `--num-scheduler-steps`**

### Memory Efficiency
1. **Explicit `--kv-cache-dtype`** specification
2. **Test different quantization strategies**
3. **Evaluate `--preemption-mode`** options
4. **Monitor memory usage patterns**

## CUDA 13.0 Considerations

### Current Limitation
- **vLLM v0.10.2** does not support CUDA 13.0
- **Issue**: CUB library API changes breaking build
- **Status**: Waiting for official CUDA 13 support

### Pending Optimizations
These optimizations are ready to deploy once CUDA 13 is supported:

**Container Runtime Optimizations** (currently commented out):
```bash
# Memory Pool Optimizations
ENV CUDA_CACHE_MAXSIZE=2147483648
ENV CUDA_ALLOCATOR_BACKEND=cudaMallocAsync
ENV CUDA_MALLOC_ASYNC_POOLS=1

# Tensor Core Optimizations
ENV CUDA_TENSOR_CORES=1
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV CUDNN_TENSOR_OPS=1

# Blackwell Architecture Specific
ENV CUDA_L2_PERSISTING_SIZE=100663296
ENV CUDA_DEVICE_MAX_CONNECTIONS=32
ENV CUDA_COPY_SPLIT_THRESHOLD=256
```

### Expected Improvements with CUDA 13
- **Memory allocation**: 30% reduction in overhead
- **Tensor cores**: 40% improvement in GEMM operations
- **FP8 support**: Up to 2x throughput for compatible models
- **L2 cache**: 10% memory bandwidth improvement

## Benchmarking Plan

### Test Matrix (Once CUDA 13 Available)
- **GPU memory utilization**: 0.85 vs 0.99
- **Sequence limits**: Different max-num-seqs values
- **Batch sizes**: Optimal batched tokens configuration
- **Caching strategies**: Different KV cache configurations
- **Quantization**: FP8 vs FP16 vs INT8 comparisons

### Metrics to Track
- **Concurrent requests**: Maximum simultaneous handling
- **Total throughput**: Aggregate tokens/second
- **Request latency**: P50, P95, P99 response times
- **Memory usage**: VRAM utilization patterns
- **Queue efficiency**: Request scheduling performance

## Hardware-Specific Optimizations

### RTX 5090 Blackwell Features
- **32GB VRAM**: Support for large batch sizes and long contexts
- **FP8 tensor cores**: Native FP8 quantization support
- **High memory bandwidth**: Optimal for concurrent request handling
- **CUDA 13.0**: Full Blackwell architecture utilization

### vLLM Engine Configuration
- **Attention backend**: FlashAttention v2 (v3 not yet supported)
- **Memory pooling**: Advanced memory management for batching
- **Scheduling**: Priority-based request handling
- **Multimodal**: Ready for vision/audio model support

## Production Deployment Notes

### Monitoring Requirements
- **Memory pressure**: Watch for OOM conditions
- **Queue length**: Monitor request backlog
- **Response times**: Track latency distributions
- **GPU utilization**: Ensure high compute usage

### Scaling Considerations
- **Horizontal scaling**: Multiple vLLM instances
- **Load balancing**: Request distribution strategies
- **Model variants**: Different models for different workloads
- **Resource allocation**: Memory vs throughput trade-offs

---

*vLLM parameter optimization for the RTX 5090 workstation. Focus on high-throughput concurrent request handling for API server workloads. Configuration ready for CUDA 13.0 deployment.*