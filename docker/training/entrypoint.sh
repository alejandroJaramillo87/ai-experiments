#!/bin/bash
set -e

# The training container's entrypoint: echo the environment, then exec the given command
# (Phase 0 D1).
# Thin: echo the environment, reserve an optional LD_PRELOAD hook for the D9
# heterogeneous-compute skeleton, then exec the given command. Unlike the inference
# entrypoints this starts no server: the container is a batch/interactive job runner,
# invoked via `docker compose -f docker-compose.training.yaml run --rm training <cmd>`.

# Color and helper setup (following BASH_OUTPUT_STYLE.md)
RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
BLUE='\033[0;34m' CYAN='\033[0;36m' NC='\033[0m'
[[ -n "${NO_COLOR:-}" ]] && RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''

print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
print_run()   { printf "${BLUE}[RUN]${NC} %s\n" "$*"; }

# Optional LD_PRELOAD hook slot for the D9 heterogeneous-compute skeleton. Empty by default;
# set TRAIN_LD_PRELOAD to inject an interposer (e.g. a NUMA/affinity or offload shim) without
# rebuilding the image.
if [[ -n "${TRAIN_LD_PRELOAD:-}" ]]; then
    export LD_PRELOAD="${TRAIN_LD_PRELOAD}${LD_PRELOAD:+:$LD_PRELOAD}"
    print_info "LD_PRELOAD: ${LD_PRELOAD}"
fi

echo "=== Training container ==="
echo
echo "Environment:"
echo "  Python: $(python --version 2>&1 | awk '{print $2}')"
echo "  CUDA arch list: ${TORCH_CUDA_ARCH_LIST:-unset}"
echo "  Working dir: $(pwd)"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
else
    print_warn "nvidia-smi not found: no GPU visible in this container"
fi
echo

print_run "exec: $*"
exec "$@"
