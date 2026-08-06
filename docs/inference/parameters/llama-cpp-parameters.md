# llama.cpp Parameter Optimization

Detailed analysis of runtime parameters for llama.cpp server (CPU and GPU variants) optimized for low-latency single-request inference on the AMD Ryzen 9950X + RTX 5090 workstation.

## Current Parameter Configuration

### llama-cpu Service (Port 8001)

**Runtime Parameters (entrypoint.cpu.sh)**:
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
    --threads-http "$THREADS_HTTP" \
    --flash-attn "on" \
    --cache-reuse 256 \
    --cont-batching \
    --metrics \
    --mlock \
    --no-repack
```

**Environment Defaults** (entrypoint.cpu.sh):
- `THREADS` - Required, set via docker-compose (typically 12-16)
- `THREADS_BATCH` - Required, set via docker-compose (typically 12-16)
- `CTX_SIZE=32768` - 32K context window
- `BATCH_SIZE=2048` - Large batch for efficiency
- `UBATCH_SIZE=512` - Micro-batch size for memory efficiency
- `THREADS_HTTP=2` - Minimal HTTP overhead
- `N_GPU_LAYERS=0` - CPU-only inference

**Note**: docker-compose.inference.yaml sets `THREADS=16` and `THREADS_BATCH=16` by default.

### llama-gpu Service (Port 8002)

**Runtime Parameters (entrypoint.sh)**:
```bash
# Environment variable configuration with defaults
SERVER_PORT=${SERVER_PORT:-8002}
MODEL_PATH=${MODEL_PATH:-"/app/models/gguf/gpt-oss-20b-GGUF/gpt-oss-20b-UD-Q8_K_XL.gguf"}
N_GPU_LAYERS=${N_GPU_LAYERS:-999}
CTX_SIZE=${CTX_SIZE:-65536}
BATCH_SIZE=${BATCH_SIZE:-2048}      # Optimal for GPU throughput
UBATCH_SIZE=${UBATCH_SIZE:-512}     # Different from CPU (2048)
THREADS=${THREADS:-1}
THREADS_BATCH=${THREADS_BATCH:-1}
THREADS_HTTP=${THREADS_HTTP:-4}
PARALLEL=${PARALLEL:-1}
FLASH_ATTN=${FLASH_ATTN:-"on"}
CONT_BATCHING=${CONT_BATCHING:-false}
```

**Override via docker-compose**:
```bash
# Test different configurations without rebuilding
BATCH_SIZE=1024 UBATCH_SIZE=1024 docker-compose up llama-gpu
```

## Parameter Analysis

### Threading Parameters

**`--threads` / `--threads-batch`**
- **CPU**: 16 threads to match available cores (0-15)
- **GPU**: 1 thread (GPU handles parallelism internally)
- **Purpose**: Balance CPU utilization without oversubscription
- **Trade-off**: More threads = higher throughput but potential latency variance

**`--threads-http`**
- **CPU**: two (minimal overhead for API handling)
- **GPU**: four (can handle more HTTP load due to GPU offload)
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

## Tested Parameters - Availability Notes

The following parameters were tested during optimization:

### Invalid Parameters (Tested September 2025)
**`--kv-split`** - **NOT VALID**
- Causes error: `error: invalid argument: --kv-split`
- Not implemented in current llama.cpp server

**`--parallel`** - **NOT VALID for server**
- No errors but no effect observed
- May be valid for main llama.cpp CLI but not the server

### Valid Parameters (Added since initial testing)
**`--cache-reuse <n>`** - **VALID** (added in later llama.cpp versions)
- Now used in production: `--cache-reuse 256`
- Enables KV cache reuse for repeated prompts
- Improves performance for similar requests

**`--flash-attn`** - **VALID**
- Used in production: `--flash-attn "on"`
- Enables Flash Attention for memory-efficient attention computation
- Works on both CPU and GPU builds

**`--no-repack`** - **VALID**
- Used in production to skip tensor repacking
- Can improve startup time for some models

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

## Optimization Notes

No benchmark has been run on this workstation. The batch size of 2048 is the
configured default (`BATCH_SIZE=2048` in `entrypoint.cpu.sh`); the sections
below list tuning candidates, not measured results.

### CPU Service Recommendations
1. **Batch size 2048** - Current configured default; benchmark before changing
2. **`--cont-batching`** - Enabled by default on the CPU service
3. **Consider `--cpu-mask`** for more precise core binding (untested)
4. **Keep current thread configuration** (16 threads matching allocated cores)

### GPU Service Configuration
1. **Batch size**: 2048/512 configured default (batch/ubatch)
2. **Configuration flexibility**: Uses environment variables for tuning without rebuilding
3. **Performance**: Not yet benchmarked on this workstation

### Cross-Service Questions
1. **Why different HTTP thread counts?** (2 vs 4)
2. **Should both services use `--cont-batching`?**
3. **Optimal context sizes** for each workload?
4. **Batch size optimization** for latency vs memory trade-off

## Benchmarking Plan

No benchmark has been run on this workstation. The items below describe the
intended test matrix, not completed results.

### CPU Service (llama-cpu)
- **Batch size sweep** (512, 2048, 4096) - Compare generation speed
- **Cache parameters** (`--kv-split`, `--cache-reuse`) - `--kv-split` is not a valid server flag
- **Single-request options** (`--parallel`, `--no-mmap`, no `--cont-batching`)

### GPU Service (llama-gpu)
- **Batch/ubatch sweep** - Compare 2048/512 against alternatives
- **Environment variables** - Configurable without rebuilding
- **GPU utilization and VRAM usage** - Measure under load

### Metrics to Track
- **First-token latency** (ms)
- **Tokens per second** (single request)
- **Memory usage** (RAM/VRAM)
- **CPU/GPU utilization**
- **Response consistency** (latency variance)

## Hardware-Specific Considerations

### AMD Zen 5 Optimizations
- **16 cores available** (0-15 via cpuset)
- **AVX-512 instruction set** (enabled in compilation)
- **L3 cache optimization** (32MB shared)
- **NUMA considerations** (single socket)

### RTX 5090 Optimizations
- **32GB VRAM** (can handle large contexts)
- **Blackwell architecture** (FlashAttention, FP8)
- **CUDA 13.0.2** (latest features)
- **High memory bandwidth** (optimal for large batches)

---

*Last Updated: 2026-01-03*