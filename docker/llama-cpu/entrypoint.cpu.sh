#!/bin/bash
set -e

# Hugepages entrypoint - AVX-512 with zen5_optimizer for hugepage allocation
# Uses LD_PRELOAD to intercept mmap() for large allocations

# Default values (can be overridden by environment variables)
SERVER_PORT=${SERVER_PORT:-8001}
MODEL_PATH=${MODEL_PATH:-"/app/models/gguf/gpt-oss-120b-GGUF/Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"}
SERVER_HOST=${SERVER_HOST:-"0.0.0.0"}
N_GPU_LAYERS=${N_GPU_LAYERS:-0}
CTX_SIZE=${CTX_SIZE:-32768}
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH_SIZE=${UBATCH_SIZE:-512}
THREADS=${THREADS:-8}
THREADS_BATCH=${THREADS_BATCH:-8}
THREADS_HTTP=${THREADS_HTTP:-2}

echo "[HUGEPAGES] Starting llama-server (AVX-512 + hugepage optimization)"
echo "[HUGEPAGES] Configuration:"
echo "  - Architecture: znver5 (Zen 5 native)"
echo "  - BLAS: Disabled"
echo "  - AVX-512: Enabled (VNNI, VBMI, BF16)"
echo "  - Hugepages: Enabled (zen5_optimizer)"
echo "  - AOCL: Disabled"
echo "  - Port: ${SERVER_PORT}"
echo "  - Threads: ${THREADS}"
echo "  - Model: ${MODEL_PATH}"

# Validate required environment variables
if [ -z "${MODEL_PATH}" ]; then
    echo "[HUGEPAGES] ERROR: MODEL_PATH environment variable not set"
    exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
    echo "[HUGEPAGES] ERROR: Model file not found at ${MODEL_PATH}"
    exit 1
fi

# Enable zen5_optimizer (hugepage support via LD_PRELOAD)
# The optimizer intercepts mmap() and uses MAP_HUGETLB for allocations >= 1GB
export LD_PRELOAD=/app/libzen5_optimizer.so
echo "[HUGEPAGES] zen5_optimizer loaded"
echo "  LD_PRELOAD set to: $LD_PRELOAD"
echo "  MAP_HUGETLB enabled for allocations >= 1GB"

# Set OpenMP threads to match server threads
export OMP_NUM_THREADS=$THREADS

# Memory status before loading
echo "[HUGEPAGES] Memory status before model load:"
grep -E "MemTotal|MemFree|AnonHugePages|HugePages_Total|HugePages_Free|HugePages_Rsvd" /proc/meminfo | sed 's/^/  /'

# Launch llama-server with hugepage optimization and NUMA interleaving
exec numactl --interleave=all ./server \
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
