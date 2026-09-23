#!/bin/bash
set -e

# The llama-gpu service's entrypoint: starts llama-server from the environment, with
# optional nsys profiling.

# Default values (can be overridden by environment variables)
SERVER_PORT=${SERVER_PORT:-8002}
MODEL_PATH=${MODEL_PATH:-"/app/models/gguf/gpt-oss-20b-GGUF/gpt-oss-20b-UD-Q8_K_XL.gguf"}

# Nsight Systems profiling (set NSYS_PROFILE=true to enable)
NSYS_PROFILE=${NSYS_PROFILE:-false}
NSYS_OUTPUT=${NSYS_OUTPUT:-"/app/nsys_output/profile"}
SERVER_HOST=${SERVER_HOST:-"0.0.0.0"}
N_GPU_LAYERS=${N_GPU_LAYERS:-999}
CTX_SIZE=${CTX_SIZE:-65536}
BATCH_SIZE=${BATCH_SIZE:-2048}
UBATCH_SIZE=${UBATCH_SIZE:-512}
THREADS=${THREADS:-1}
THREADS_BATCH=${THREADS_BATCH:-1}
THREADS_HTTP=${THREADS_HTTP:-4}
PARALLEL=${PARALLEL:-1}
MAIN_GPU=${MAIN_GPU:-0}
FLASH_ATTN=${FLASH_ATTN:-"on"}
CONT_BATCHING=${CONT_BATCHING:-false}

# GPU/CPU split configuration
SPLIT_MODE=${SPLIT_MODE:-"none"}
TENSOR_SPLIT=${TENSOR_SPLIT:-""}
NO_KV_OFFLOAD=${NO_KV_OFFLOAD:-false}

# Advanced server features
CACHE_REUSE=${CACHE_REUSE:-0}
JINJA_ENABLED=${JINJA_ENABLED:-false}
REASONING_FORMAT=${REASONING_FORMAT:-""}
KV_UNIFIED=${KV_UNIFIED:-false}

echo "=== Starting llama.cpp GPU Server ==="
echo "  Port: $SERVER_PORT"
echo "  Model: $MODEL_PATH"
echo "  GPU layers: $N_GPU_LAYERS"
echo "  Context size: $CTX_SIZE"
echo "  Batch/UBatch: $BATCH_SIZE/$UBATCH_SIZE"
echo "  Threads: $THREADS"
echo "  Continuous batching: $CONT_BATCHING"
echo "  Split mode: $SPLIT_MODE"
if [[ -n "$TENSOR_SPLIT" ]]; then
    echo "  Tensor split: $TENSOR_SPLIT"
fi
echo "  KV offload: $([ "$NO_KV_OFFLOAD" == "true" ] && echo "disabled" || echo "enabled")"
if [[ "$CACHE_REUSE" -gt 0 ]]; then
    echo "  Cache reuse: $CACHE_REUSE"
fi
if [[ "$JINJA_ENABLED" == "true" ]]; then
    echo "  Jinja templates: enabled"
fi
if [[ -n "$REASONING_FORMAT" ]]; then
    echo "  Reasoning format: $REASONING_FORMAT"
fi
if [[ "$KV_UNIFIED" == "true" ]]; then
    echo "  KV unified buffer: enabled"
fi

# Verify model file exists
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    echo "Please ensure the model file exists at the specified path."
    exit 1
fi

# Memory status before loading
echo "GPU memory status before model load:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits | sed 's/^/  /'

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
    "--metrics"
    "--no-warmup"
    "--threads-http" "$THREADS_HTTP"
    "--flash-attn" "$FLASH_ATTN"
    "--no-mmap"
    "--main-gpu" "$MAIN_GPU"
    "--parallel" "$PARALLEL"
)

# Add optional parameters based on environment variables
# Upstream defaults continuous batching to on; pass the opt-out to preserve
# the historical default (disabled) when CONT_BATCHING=false
if [[ "$CONT_BATCHING" != "true" ]]; then
    CMD_ARGS+=("--no-cont-batching")
fi

# GPU/CPU split configuration
if [[ "$SPLIT_MODE" != "none" ]]; then
    CMD_ARGS+=("--split-mode" "$SPLIT_MODE")
fi

if [[ -n "$TENSOR_SPLIT" ]]; then
    CMD_ARGS+=("--tensor-split" "$TENSOR_SPLIT")
fi

if [[ "$NO_KV_OFFLOAD" == "true" ]]; then
    CMD_ARGS+=("--no-kv-offload")
fi

# Advanced server features
if [[ "$CACHE_REUSE" -gt 0 ]]; then
    CMD_ARGS+=("--cache-reuse" "$CACHE_REUSE")
fi

# Upstream defaults jinja templating to on; pass the opt-out to preserve
# the historical default (disabled) when JINJA_ENABLED=false
if [[ "$JINJA_ENABLED" != "true" ]]; then
    CMD_ARGS+=("--no-jinja")
fi

if [[ -n "$REASONING_FORMAT" ]]; then
    CMD_ARGS+=("--reasoning-format" "$REASONING_FORMAT")
fi

if [[ "$KV_UNIFIED" == "true" ]]; then
    CMD_ARGS+=("--kv-unified")
fi

# Execute the server with all parameters
# Optionally wrap with nsys for deep profiling
if [[ "$NSYS_PROFILE" == "true" ]]; then
    echo "=== NSYS Profiling Enabled ==="
    echo "  Output: ${NSYS_OUTPUT}"
    echo "  Stop container to finalize report"

    exec nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --sample=cpu \
        --output="$NSYS_OUTPUT" \
        --force-overwrite=true \
        --stats=true \
        -- "${CMD_ARGS[@]}"
else
    exec "${CMD_ARGS[@]}"
fi