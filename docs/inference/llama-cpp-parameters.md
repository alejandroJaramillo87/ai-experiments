# llama.cpp Parameter Optimization

Detailed analysis of runtime parameters for llama.cpp server (CPU and GPU variants) optimized for low-latency single-request inference on the AMD Ryzen 9950X + RTX 5090 workstation.

## Current Parameter Configuration

### llama-cpu Service (Port 8001)

**Runtime Parameters (entrypoint.sh)**:
```bash
exec ./server \
    --model "$MODEL_PATH" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --ctx-size "$CTX_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --ubatch-size "$UBATCH_SIZE" \
    --threads "$THREADS" \
    --threads-batch "$THREADS_BATCH" \
    --cont-batching \
    --metrics \
    --no-warmup \
    --mlock \
    --threads-http "$THREADS_HTTP"
```

**Environment Defaults**:
- `THREADS=12` - Match available CPU cores (0-11)
- `THREADS_BATCH=12` - Same for batch processing
- `CTX_SIZE=32768` - 32K context window
- `BATCH_SIZE=2048` - Large batch for efficiency
- `UBATCH_SIZE=2048` - Unified batch size
- `THREADS_HTTP=2` - Minimal HTTP overhead
- `N_GPU_LAYERS=0` - CPU-only inference

### llama-gpu Service (Port 8004)

**Runtime Parameters (Dockerfile.llama-gpu)**:
```bash
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
    "--flash-attn", "on", \
    "--no-mmap", \
    "--main-gpu", "0", \
    "--parallel", "1"]
```

## Parameter Analysis

### Threading Parameters

**`--threads` / `--threads-batch`**
- **CPU**: 12 threads to match available cores (0-11)
- **GPU**: 1 thread (GPU handles parallelism internally)
- **Purpose**: Balance CPU utilization without oversubscription
- **Trade-off**: More threads = higher throughput but potential latency variance

**`--threads-http`**
- **CPU**: 2 (minimal overhead for API handling)
- **GPU**: 4 (can handle more HTTP load due to GPU offload)
- **Purpose**: Separate HTTP processing from inference threads

### Memory Parameters

**`--ctx-size`**
- **CPU**: 32768 (32K context)
- **GPU**: 65536 (64K context - GPU has more memory)
- **Purpose**: Maximum context length for long conversations
- **Trade-off**: Larger context = more memory usage, potentially slower

**`--batch-size` / `--ubatch-size`**
- **CPU**: 2048/2048 (large, unified batches)
- **GPU**: 2048/512 (different batch strategies)
- **Purpose**: Token processing efficiency
- **Trade-off**: Larger batches = better throughput, higher memory usage

**`--mlock`** (CPU only)
- **Purpose**: Lock model in RAM to prevent swapping
- **Benefit**: Consistent latency, no disk I/O delays
- **Note**: Not used on GPU (VRAM is already locked)

### GPU-Specific Parameters

**`--n-gpu-layers`**
- **CPU**: 0 (pure CPU inference)
- **GPU**: 999 (full GPU offload)
- **Purpose**: Offload computation to GPU for acceleration

**`--flash-attn`**
- **Setting**: "on" for GPU
- **Purpose**: Memory-efficient attention mechanism
- **Benefit**: Reduced VRAM usage, faster attention computation

**`--no-mmap`** (GPU only)
- **Purpose**: Disable memory mapping for direct GPU memory access
- **Benefit**: Better GPU memory management

**`--main-gpu`**
- **Setting**: "0" (first GPU)
- **Purpose**: Specify primary GPU for multi-GPU systems

### Optimization Parameters

**`--cont-batching`** (CPU only)
- **Purpose**: Continuous batching for better CPU utilization
- **Note**: Not used on GPU - different batching strategy

**`--parallel`** (GPU only)
- **Setting**: "1" (single parallel stream)
- **Purpose**: Optimize for single-request latency over throughput

**`--no-warmup`**
- **Purpose**: Skip model warmup for faster startup
- **Trade-off**: First request may be slightly slower

**`--metrics`**
- **Purpose**: Enable performance metrics collection
- **Benefit**: Monitor performance for optimization

## Tested Parameters - Not Available

The following parameters were tested but found to **not exist** in the current llama.cpp server:

### Invalid Parameters (Tested September 2025)
**`--kv-split`** - **NOT VALID**
- Causes error: `error: invalid argument: --kv-split`
- Not implemented in current llama.cpp server

**`--cache-reuse`** - **NOT VALID**
- Not recognized by llama.cpp server
- May have been planned but not implemented

**`--parallel`** - **NOT VALID for server**
- No errors but no effect observed
- May be valid for main llama.cpp CLI but not the server

### Advanced Features
**`--cpu-mask`**
- **Purpose**: CPU affinity mask for specific core binding
- **Current**: Using Docker cpuset, but this might be more precise
- **Test priority**: Medium

**`--no-context-shift`**
- **Purpose**: Disable context shifting when context fills up
- **Trade-off**: May need larger context sizes
- **Test priority**: Low

### Speculative Decoding (2025)
**`--draft-max` / `--draft-min`**
- **Purpose**: Speculative decoding for faster generation
- **Requirement**: Needs draft model
- **Test priority**: Medium (requires additional setup)

## Optimization Findings (September 2025 Testing)

### CPU Service - Benchmark Results
**Batch Size Testing** (on AMD Ryzen 9950X):
- **Batch 512**: 34.79 tokens/sec (worse)
- **Batch 2048**: 35.44 tokens/sec (OPTIMAL)
- **Batch 4096**: 34.95 tokens/sec (slightly worse)

**Key Finding**: The original batch size of 2048 is optimal for CPU inference. Smaller batches hurt performance due to inefficient vectorization, while larger batches cause cache thrashing.

### CPU Service Recommendations
1. **Keep batch size at 2048** - Testing confirmed this is optimal
2. **Keep `--cont-batching`** - Removing it didn't improve single-request performance
3. **Consider `--cpu-mask`** for more precise core binding (untested)
4. **Keep current thread configuration** (12 threads matching allocated cores)

### GPU Service Improvements
1. **Test different `--ubatch-size`** values (current: 512)
2. **Evaluate `--cont-batching`** for GPU (currently disabled)
3. **Research optimal `--threads` count** (current: 1)

### Cross-Service Questions
1. **Why different HTTP thread counts?** (2 vs 4)
2. **Should both services use `--cont-batching`?**
3. **Optimal context sizes** for each workload?
4. **Batch size optimization** for latency vs memory trade-off

## Benchmarking Completed

### Tested Configurations
- **Base configuration** (batch 2048) - 35.44 tokens/sec ✓
- **Small batch size** (batch 512) - 34.79 tokens/sec (worse)
- **Large batch size** (batch 4096) - 34.95 tokens/sec (worse)
- **Cache optimizations** (--kv-split, --cache-reuse) - Parameters don't exist
- **Single-request optimizations** (--parallel, --no-mmap, no --cont-batching) - No improvement

### Metrics to Track
- **First-token latency** (ms)
- **Tokens per second** (single request)
- **Memory usage** (RAM/VRAM)
- **CPU/GPU utilization**
- **Response consistency** (latency variance)

## Hardware-Specific Considerations

### AMD Zen 5 Optimizations
- **12 cores available** (0-11 via cpuset)
- **AVX-512 instruction set** (enabled in compilation)
- **L3 cache optimization** (32MB shared)
- **NUMA considerations** (single socket)

### RTX 5090 Optimizations
- **32GB VRAM** (can handle large contexts)
- **Blackwell architecture** (FlashAttention, FP8)
- **CUDA 13.0.1** (latest features)
- **High memory bandwidth** (optimal for large batches)

---

*llama.cpp parameter optimization for the AMD Ryzen 9950X + RTX 5090 workstation. Focus on low-latency single-request inference for interactive AI applications.*