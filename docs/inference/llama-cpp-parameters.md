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

## Missing Parameters to Research

Based on 2025 llama.cpp documentation, we should consider testing:

### Cache Optimization
**`--kv-split`**
- **Purpose**: Single unified KV buffer for all sequences
- **Potential benefit**: Better memory efficiency
- **Test priority**: High

**`--cache-reuse`** (2025 feature)
- **Purpose**: Cache reuse optimization
- **Potential benefit**: Faster subsequent requests
- **Test priority**: High

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

## Optimization Opportunities

### CPU Service Improvements
1. **Test `--kv-split`** for memory efficiency
2. **Evaluate `--cache-reuse`** for multi-turn conversations
3. **Consider `--cpu-mask`** for more precise core binding
4. **Benchmark different batch size combinations**

### GPU Service Improvements
1. **Add missing cache optimizations** (`--kv-split`, `--cache-reuse`)
2. **Test different `--ubatch-size`** values (current: 512)
3. **Evaluate `--cont-batching`** for GPU (currently disabled)
4. **Research optimal `--threads` count** (current: 1)

### Cross-Service Questions
1. **Why different HTTP thread counts?** (2 vs 4)
2. **Should both services use `--cont-batching`?**
3. **Optimal context sizes** for each workload?
4. **Batch size optimization** for latency vs memory trade-off

## Benchmarking Plan

### Test Matrix
- **Base configuration** (current parameters)
- **Cache optimizations** (+kv-split, +cache-reuse)
- **Threading variations** (different thread counts)
- **Batch size optimization** (different combinations)
- **Context size impact** (16K, 32K, 64K, 128K)

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