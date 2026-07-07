#!/bin/bash
set -e

# AOCL experiment entrypoint - ZenDNN backend (AOCL-DLP GEMM) with
# zen5_optimizer hugepage preload and optional AOCL-LibMem preload.
# See docs/optimizations/experiments/aocl-llama-cpp.md

# Required environment variables (fail fast if missing)
MODEL_PATH=${MODEL_PATH:?MODEL_PATH environment variable is required}
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

# Experiment toggles
ENABLE_LIBMEM=${ENABLE_LIBMEM:-false}
NO_REPACK=${NO_REPACK:-true}

echo "[AOCL] Starting llama-server (ZenDNN/AOCL-DLP backend + hugepage optimization)"
echo "[AOCL] Configuration:"
echo "  - Architecture: znver5 (Zen 5 native)"
echo "  - Backend: ZenDNN (AOCL-DLP LPGEMM, F32/BF16/Q8_0 mul_mat)"
echo "  - AVX-512: Enabled (VNNI, VBMI, BF16)"
echo "  - Hugepages: Enabled (zen5_optimizer)"
echo "  - AOCL-LibMem: ${ENABLE_LIBMEM}"
echo "  - Native repack disabled: ${NO_REPACK}"
echo "  - Port: ${SERVER_PORT}"
echo "  - Threads: ${THREADS}"
echo "  - Model: ${MODEL_PATH}"

# Validate model file exists
if [ ! -f "${MODEL_PATH}" ]; then
    echo "[AOCL] ERROR: Model file not found at ${MODEL_PATH}"
    exit 1
fi

# ZenDNN: select the blocked AOCL-DLP matmul algorithm (AMD-recommended)
export ZENDNNL_MATMUL_ALGO=1
echo "[AOCL] ZENDNNL_MATMUL_ALGO=1 (blocked AOCL-DLP)"

# Preload chain: zen5_optimizer intercepts mmap() >= 1GB (MAP_HUGETLB);
# AOCL-LibMem interposes memcpy/memset family (disjoint symbols)
PRELOAD="/app/libzen5_optimizer.so"
if [ "${ENABLE_LIBMEM}" = "true" ]; then
    PRELOAD="${PRELOAD}:/app/libaocl-libmem.so"
fi
export LD_PRELOAD="${PRELOAD}"
echo "[AOCL] Memory allocator loaded"
echo "  LD_PRELOAD: $LD_PRELOAD"

# Set OpenMP threads to match server threads (ggml and ZenDNN)
export OMP_NUM_THREADS=$THREADS

# Memory status before loading
echo "[AOCL] Memory status before model load:"
grep -E "MemTotal|MemFree|AnonHugePages|HugePages_Total|HugePages_Free|HugePages_Rsvd" /proc/meminfo | sed 's/^/  /'

# Native repack toggle: production config disables repack to keep weights
# in the mmap region (hugepage-backed); benchmark arm B enables it
REPACK_FLAG=""
if [ "${NO_REPACK}" = "true" ]; then
    REPACK_FLAG="--no-repack"
fi

# Launch llama-server with ZenDNN backend
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
    $REPACK_FLAG
