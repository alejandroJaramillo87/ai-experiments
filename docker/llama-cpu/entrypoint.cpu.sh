#!/bin/bash
set -e

# Hugepages entrypoint - AVX-512 with zen5_optimizer for hugepage allocation
# Uses LD_PRELOAD to intercept mmap() for large allocations

# Required environment variables (fail fast if missing)
MODEL_PATH=${LLAMA_CPU_MODEL:?LLAMA_CPU_MODEL environment variable is required}
SERVER_PORT=${SERVER_PORT:?SERVER_PORT environment variable is required}
THREADS=${THREADS:?THREADS environment variable is required}
THREADS_BATCH=${THREADS_BATCH:?THREADS_BATCH environment variable is required}

# Optional with sensible defaults
SERVER_HOST=${SERVER_HOST:-"0.0.0.0"}
N_GPU_LAYERS=${N_GPU_LAYERS:-0}
CTX_SIZE=${CTX_SIZE:-32768}
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH_SIZE=${UBATCH_SIZE:-512}
THREADS_HTTP=${THREADS_HTTP:-2}

# Feature flags (defaults preserve the historical hardcoded behavior)
FLASH_ATTN=${FLASH_ATTN:-"on"}
CACHE_REUSE=${CACHE_REUSE:-256}
CONT_BATCHING=${CONT_BATCHING:-true}
MLOCK=${MLOCK:-true}
# Weight repacking disabled by default (startup-time optimization).
# Benchmark NO_REPACK=false before changing: repacking is the main CPU
# throughput lever on AVX-512.
NO_REPACK=${NO_REPACK:-true}

# Advanced server features (appended only when set)
CACHE_TYPE_K=${CACHE_TYPE_K:-""}
CACHE_TYPE_V=${CACHE_TYPE_V:-""}
KV_UNIFIED=${KV_UNIFIED:-false}
CTX_CHECKPOINTS=${CTX_CHECKPOINTS:-""}
JINJA_ENABLED=${JINJA_ENABLED:-false}

echo "[HUGEPAGES] Starting llama-server (AVX-512 + hugepage optimization)"
echo "[HUGEPAGES] Configuration:"
echo "  - Architecture: znver5 (Zen 5 native)"
echo "  - BLAS: Disabled"
echo "  - AVX-512: Enabled (VNNI, VBMI, BF16)"
echo "  - Hugepages: Enabled (zen5_optimizer)"
echo "  - Port: ${SERVER_PORT}"
echo "  - Threads: ${THREADS}"
echo "  - Model: ${MODEL_PATH}"
echo "  - Flash attention: ${FLASH_ATTN}"
echo "  - Weight repacking: $([ "$NO_REPACK" == "true" ] && echo "disabled" || echo "enabled")"
if [[ "$CACHE_REUSE" -gt 0 ]]; then
    echo "  - Cache reuse: ${CACHE_REUSE}"
fi
if [[ -n "$CACHE_TYPE_K" || -n "$CACHE_TYPE_V" ]]; then
    echo "  - KV cache types: K=${CACHE_TYPE_K:-default} V=${CACHE_TYPE_V:-default}"
fi
if [[ "$KV_UNIFIED" == "true" ]]; then
    echo "  - KV unified buffer: enabled"
fi
if [[ -n "$CTX_CHECKPOINTS" ]]; then
    echo "  - Context checkpoints: ${CTX_CHECKPOINTS}"
fi
if [[ "$JINJA_ENABLED" == "true" ]]; then
    echo "  - Jinja templates: enabled"
fi

# Validate model file exists
if [ ! -f "${MODEL_PATH}" ]; then
    echo "[HUGEPAGES] ERROR: Model file not found at ${MODEL_PATH}"
    exit 1
fi

# Enable zen5_optimizer via LD_PRELOAD for hugepage mmap()
export LD_PRELOAD="/app/libzen5_optimizer.so"
echo "[HUGEPAGES] Memory allocator loaded"
echo "  LD_PRELOAD: $LD_PRELOAD"
echo "  zen5_optimizer: intercepts mmap() for allocations >= 1GB (MAP_HUGETLB)"

# Set OpenMP threads to match server threads
export OMP_NUM_THREADS=$THREADS

# Memory status before loading
echo "[HUGEPAGES] Memory status before model load:"
grep -E "MemTotal|MemFree|AnonHugePages|HugePages_Total|HugePages_Free|HugePages_Rsvd" /proc/meminfo | sed 's/^/  /'

# Build command arguments
CMD_ARGS=(
    "./server"
    "--model" "$MODEL_PATH"
    "--host" "$SERVER_HOST"
    "--port" "$SERVER_PORT"
    "--n-gpu-layers" "$N_GPU_LAYERS"
    "--ctx-size" "$CTX_SIZE"
    "--batch-size" "$BATCH_SIZE"
    "--ubatch-size" "$UBATCH_SIZE"
    "--threads" "$THREADS"
    "--threads-batch" "$THREADS_BATCH"
    "--threads-http" "$THREADS_HTTP"
    "--flash-attn" "$FLASH_ATTN"
    "--metrics"
)

# Add optional parameters based on environment variables
if [[ "$CACHE_REUSE" -gt 0 ]]; then
    CMD_ARGS+=("--cache-reuse" "$CACHE_REUSE")
fi

if [[ "$CONT_BATCHING" == "true" ]]; then
    CMD_ARGS+=("--cont-batching")
fi

if [[ "$MLOCK" == "true" ]]; then
    CMD_ARGS+=("--mlock")
fi

if [[ "$NO_REPACK" == "true" ]]; then
    CMD_ARGS+=("--no-repack")
fi

# Advanced server features
if [[ -n "$CACHE_TYPE_K" ]]; then
    CMD_ARGS+=("--cache-type-k" "$CACHE_TYPE_K")
fi

if [[ -n "$CACHE_TYPE_V" ]]; then
    CMD_ARGS+=("--cache-type-v" "$CACHE_TYPE_V")
fi

if [[ "$KV_UNIFIED" == "true" ]]; then
    CMD_ARGS+=("--kv-unified")
fi

if [[ -n "$CTX_CHECKPOINTS" ]]; then
    CMD_ARGS+=("--ctx-checkpoints" "$CTX_CHECKPOINTS")
fi

if [[ "$JINJA_ENABLED" == "true" ]]; then
    CMD_ARGS+=("--jinja")
fi

echo "[HUGEPAGES] Launch command:"
echo "  ${CMD_ARGS[*]}"

# Execute the server with all parameters
exec "${CMD_ARGS[@]}"
