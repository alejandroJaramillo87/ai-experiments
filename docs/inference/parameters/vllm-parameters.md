# vLLM Parameter Optimization

Detailed analysis of runtime parameters for vLLM server optimized for high-throughput concurrent inference on the RTX 5090 with focus on maximizing request handling capacity.

## Current Parameter Configuration

### vllm-gpu Service (Port 8003)

**Default runtime command (entrypoint.sh)**:
```bash
vllm serve /app/models/example-model-hf \
    --host 0.0.0.0 \
    --port 8003 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --tensor-parallel-size 1 \
    --dtype auto \
    --kv-cache-dtype auto \
    --attention-backend flashinfer \
    --download-dir /app/cache \
    --disable-log-stats
```

Prefix caching and chunked prefill are default-on in the vLLM V1 engine, so the legacy `--enable-prefix-caching` and `--enable-chunked-prefill` flags are no longer passed. `--trust-remote-code` and `--quantization` are added only when `TRUST_REMOTE_CODE=true` or `QUANTIZATION` is set (both off by default). `--disable-log-stats` is likewise conditional — `entrypoint.sh:94` appends it only when `DISABLE_LOG_STATS=true`, which is the default, so it appears above but disappears if that variable is overridden.

**Service Status**: Active with vLLM 0.25.0 and CUDA 13.0.2

## Parameter Analysis

### GPU Resource Management

**`--tensor-parallel-size`**
- **Current**: 1 (single GPU)
- **Purpose**: Distribute model across multiple GPUs
- **Note**: Single RTX 5090 setup, no parallelism needed

**`--gpu-memory-utilization`**
- **Current**: 0.85 (85 percent of 32GB VRAM)
- **Purpose**: Control VRAM allocation for KV cache
- **Tuning**: Raising toward 0.99 allocates more KV cache; benchmark before changing

### Sequence and Batching Configuration

**`--max-num-seqs`**
- **Current**: 256 concurrent sequences
- **Purpose**: Maximum number of simultaneous requests
- **Trade-off**: More sequences = higher throughput, more memory usage

**`--max-num-batched-tokens`**
- **Current**: 32768 tokens per batch
- **Purpose**: Maximum tokens processed in single batch
- **Optimization**: Large batches for GPU efficiency

**`--max-model-len`**
- **Current**: 16384 (16K context)
- **Purpose**: Maximum context length per sequence
- **Note**: Balanced for throughput vs memory

### Memory and Caching Optimizations

**Prefix caching and chunked prefill**
- **Status**: Default-on in the vLLM V1 engine
- **Note**: The legacy `--enable-prefix-caching` and `--enable-chunked-prefill` flags are not passed by the entrypoint; the V1 engine enables both by default

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
- **Current**: Unset (no quantization applied by default)
- **Purpose**: FP8 quantization for memory efficiency
- **Note**: Opt-in per model via the `QUANTIZATION` env var (for example `QUANTIZATION=fp8`); FP8 is native on the RTX 5090

**`--trust-remote-code`**
- **Status**: Disabled by default (`TRUST_REMOTE_CODE=false`)
- **Purpose**: Allow custom model code execution
- **Note**: Opt-in per model via `TRUST_REMOTE_CODE=true`; enable only for trusted sources

## Missing Parameters to Research

Based on 2025 vLLM documentation, we should consider:

### Performance Optimizations

**`--max-seq-len-to-capture`**
- **Purpose**: Maximum sequence length for Hip-graphs
- **Recommended**: Set to max-model-len (16384) to match the configured context
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
2. **Add `--max-seq-len-to-capture=16384`** to match the configured context
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

## CUDA 13.0 Configuration

### Current Status
- **vLLM 0.25.0** installed from the official cu130 wheel, targeting CUDA 13.0.2
- **PyTorch**: 2.11 cu130
- **FlashInfer**: 0.6.13
- **Status**: Active and operational

### Active Optimizations
The following CUDA environment variables are set in
`docker/vllm-gpu/Dockerfile.vllm-gpu`:

```bash
ENV CUDA_CACHE_MAXSIZE=2147483648
ENV CUDA_MODULE_LOADING=LAZY
ENV CUDA_DEVICE_ORDER=FASTEST_FIRST
ENV CUDA_LAUNCH_BLOCKING=0
```

### Experimental Optimizations (not currently enabled)
The following variables are present in the Dockerfile but commented out. They
require benchmarking before enabling:

```bash
# ENV CUDA_ALLOCATOR_BACKEND=cudaMallocAsync
# ENV CUDA_MALLOC_ASYNC_POOLS=1
# ENV CUDA_TENSOR_CORES=1
# ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
# ENV CUDNN_TENSOR_OPS=1
# ENV CUDA_L2_PERSISTING_SIZE=100663296
# ENV CUDA_DEVICE_MAX_CONNECTIONS=32
# ENV CUDA_COPY_SPLIT_THRESHOLD=256
```

### Features of CUDA 13
- **Async memory allocation**: Available via cudaMallocAsync (experimental)
- **Tensor cores**: GEMM acceleration on Blackwell
- **FP8 support**: Native FP8 tensor cores on RTX 5090
- **L2 cache**: 96MB persisting cache on RTX 5090

No benchmark has been run to quantify these on this workstation.

## Benchmarking Plan

### Test Matrix
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
- **Attention backend**: FlashInfer 0.6.13 (default on Blackwell; set through `ATTENTION_BACKEND`)
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
