#!/bin/bash
set -e

# vLLM GPU Server Entrypoint
# Supports environment-based configuration and optional nsys profiling

# Color and helper setup (following BASH_OUTPUT_STYLE.md)
RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
BLUE='\033[0;34m' CYAN='\033[0;36m' NC='\033[0m'
[[ -n "${NO_COLOR:-}" ]] && RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''

print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
print_run()   { printf "${BLUE}[RUN]${NC} %s\n" "$*"; }

# Required environment variables (fail fast if missing)
MODEL_PATH=${MODEL_PATH:?MODEL_PATH environment variable is required}

# Server configuration with sensible defaults
SERVER_PORT=${SERVER_PORT:-8003}
SERVER_HOST=${SERVER_HOST:-"0.0.0.0"}

# Model loading configuration
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
DTYPE=${DTYPE:-auto}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}
QUANTIZATION=${QUANTIZATION:-}

# Nsight Systems profiling (set NSYS_PROFILE=true to enable)
NSYS_PROFILE=${NSYS_PROFILE:-false}
NSYS_OUTPUT=${NSYS_OUTPUT:-"/app/nsys_output/profile"}

# vLLM v0.12.0 feature flags (enabled by default)
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-true}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-true}

# Optional features
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-true}
DISABLE_LOG_STATS=${DISABLE_LOG_STATS:-true}

echo "=== Starting vLLM GPU Server ==="
echo
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Port: $SERVER_PORT"
echo "  GPU memory utilization: $GPU_MEMORY_UTILIZATION"
echo "  Max model length: $MAX_MODEL_LEN"
echo "  Max sequences: $MAX_NUM_SEQS"
echo "  Max batched tokens: $MAX_NUM_BATCHED_TOKENS"
echo "  Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "  Dtype: $DTYPE"
echo "  KV cache dtype: $KV_CACHE_DTYPE"
[[ -n "$QUANTIZATION" ]] && echo "  Quantization: $QUANTIZATION"
echo "  Prefix caching: $ENABLE_PREFIX_CACHING"
echo "  Chunked prefill: $ENABLE_CHUNKED_PREFILL"
echo

# GPU memory status before loading
echo "GPU memory status:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits | sed 's/^/  /'
echo

# Build command arguments
CMD_ARGS=(
    "vllm" "serve" "$MODEL_PATH"
    "--host" "$SERVER_HOST"
    "--port" "$SERVER_PORT"
    "--gpu-memory-utilization" "$GPU_MEMORY_UTILIZATION"
    "--max-model-len" "$MAX_MODEL_LEN"
    "--max-num-seqs" "$MAX_NUM_SEQS"
    "--max-num-batched-tokens" "$MAX_NUM_BATCHED_TOKENS"
    "--tensor-parallel-size" "$TENSOR_PARALLEL_SIZE"
    "--dtype" "$DTYPE"
    "--kv-cache-dtype" "$KV_CACHE_DTYPE"
    "--download-dir" "/app/cache"
)

# Disable torch.compile (requires nvcc which runtime image doesn't have)
# CMD_ARGS+=("--enforce-eager")

# Optional arguments based on environment variables
[[ -n "$QUANTIZATION" ]] && CMD_ARGS+=("--quantization" "$QUANTIZATION")
[[ "$ENABLE_PREFIX_CACHING" == "true" ]] && CMD_ARGS+=("--enable-prefix-caching")
[[ "$ENABLE_CHUNKED_PREFILL" == "true" ]] && CMD_ARGS+=("--enable-chunked-prefill")
[[ "$TRUST_REMOTE_CODE" == "true" ]] && CMD_ARGS+=("--trust-remote-code")
[[ "$DISABLE_LOG_STATS" == "true" ]] && CMD_ARGS+=("--disable-log-stats")

# Execute with optional nsys profiling
if [[ "$NSYS_PROFILE" == "true" ]]; then
    echo "=== NSYS Profiling Enabled ==="
    echo "  Output: ${NSYS_OUTPUT}"
    echo "  Stop container to finalize report"
    echo

    exec nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --sample=cpu \
        --output="$NSYS_OUTPUT" \
        --force-overwrite=true \
        --stats=true \
        -- "${CMD_ARGS[@]}"
else
    print_run "Starting vLLM server"
    exec "${CMD_ARGS[@]}"
fi
