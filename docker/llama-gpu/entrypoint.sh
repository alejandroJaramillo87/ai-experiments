#!/bin/bash
set -e

# Default values (can be overridden by environment variables)
SERVER_PORT=${SERVER_PORT:-8004}
MODEL_PATH=${MODEL_PATH:-"/app/models/gguf/gpt-oss-20b-GGUF/gpt-oss-20b-UD-Q8_K_XL.gguf"}
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

echo "=== Starting llama.cpp GPU Server ==="
echo "  Port: $SERVER_PORT"
echo "  Model: $MODEL_PATH"
echo "  GPU layers: $N_GPU_LAYERS"
echo "  Context size: $CTX_SIZE"
echo "  Batch/UBatch: $BATCH_SIZE/$UBATCH_SIZE"
echo "  Threads: $THREADS"
echo "  Continuous batching: $CONT_BATCHING"

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
if [[ "$CONT_BATCHING" == "true" ]]; then
    CMD_ARGS+=("--cont-batching")
fi

# Execute the server with all parameters
exec "${CMD_ARGS[@]}"