#!/bin/bash
set -e

# Default values (can be overridden by environment variables)
COMFYUI_PORT=${COMFYUI_PORT:-8188}
MODEL_BASE_PATH=${MODEL_BASE_PATH:-"/app/models"}
OUTPUT_PATH=${OUTPUT_PATH:-"/app/output"}
LISTEN_ADDRESS=${LISTEN_ADDRESS:-"0.0.0.0"}
VRAM_MODE=${VRAM_MODE:-"highvram"}
PREVIEW_METHOD=${PREVIEW_METHOD:-"auto"}

# Silence git warnings for non-critical operations
export GIT_PYTHON_REFRESH=quiet

# Create writable directories if they don't exist
mkdir -p /app/user 2>/dev/null || true
mkdir -p /app/.cache 2>/dev/null || true

echo "=== Starting ComfyUI v0.3.60 with FLUX.1-dev ==="
echo "  Port: $COMFYUI_PORT"
echo "  Models: $MODEL_BASE_PATH"
echo "  Output: $OUTPUT_PATH"
echo "  VRAM Mode: $VRAM_MODE"
echo "  Listen: $LISTEN_ADDRESS"

# Check for FLUX model
if [[ -d "$MODEL_BASE_PATH/diffusion_models/FLUX.1-dev" ]]; then
    echo "FLUX.1-dev model found at $MODEL_BASE_PATH/diffusion_models/FLUX.1-dev"
    # ComfyUI will auto-detect models in the mounted directory
else
    echo "WARNING: FLUX.1-dev model not found at $MODEL_BASE_PATH/diffusion_models/FLUX.1-dev"
    echo "To download the model, run:"
    echo "  make flux-model-download"
    echo "Then move it to the correct location:"
    echo "  sudo mv /mnt/ai-data/models/flux /mnt/ai-data/models/diffusion_models/FLUX.1-dev"
fi

# GPU memory status before loading
echo "GPU memory status before model load:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits | sed 's/^/  /'

# Activate virtual environment
source /app/venv/bin/activate

# Build command arguments
CMD_ARGS=(
    "python"
    "main.py"
    "--listen" "$LISTEN_ADDRESS"
    "--port" "$COMFYUI_PORT"
    "--preview-method" "$PREVIEW_METHOD"
    "--use-pytorch-cross-attention"
)

# Add VRAM mode flag
if [[ "$VRAM_MODE" == "highvram" ]]; then
    CMD_ARGS+=("--highvram")
elif [[ "$VRAM_MODE" == "normalvram" ]]; then
    CMD_ARGS+=("--normalvram")
elif [[ "$VRAM_MODE" == "lowvram" ]]; then
    CMD_ARGS+=("--lowvram")
elif [[ "$VRAM_MODE" == "novram" ]]; then
    CMD_ARGS+=("--novram")
fi

# Execute ComfyUI
exec "${CMD_ARGS[@]}"