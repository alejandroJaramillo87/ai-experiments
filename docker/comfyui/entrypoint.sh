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

echo "=== Starting ComfyUI v0.27.1 (image and video generation) ==="
echo "  Port: $COMFYUI_PORT"
echo "  Models: $MODEL_BASE_PATH"
echo "  Output: $OUTPUT_PATH"
echo "  VRAM Mode: $VRAM_MODE"
echo "  Listen: $LISTEN_ADDRESS"

# Report detected generation models; ComfyUI auto-detects the mounted layout
echo "Detected models:"
found_models=0
for dir in diffusion_models checkpoints text_encoders vae; do
    if [[ -d "$MODEL_BASE_PATH/$dir" ]]; then
        count=$(find "$MODEL_BASE_PATH/$dir" -type f \
            \( -name '*.safetensors' -o -name '*.gguf' \) 2>/dev/null | wc -l)
        if [[ "$count" -gt 0 ]]; then
            echo "  $dir: $count file(s)"
            found_models=1
        fi
    fi
done
if [[ "$found_models" -eq 0 ]]; then
    echo "WARNING: No generation models found under $MODEL_BASE_PATH"
    echo "To download the image and video model set, run on the host:"
    echo "  make comfy-models-image"
    echo "  make comfy-models-video"
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
