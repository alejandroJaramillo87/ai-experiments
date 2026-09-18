#!/bin/bash
set -e

# Interpretability workbench entrypoint (interp Phase 0 I1)
# Thin: bootstrap the /mnt/ai-data/interp storage tree, echo the environment, then exec the
# given command. The default command "jupyter-lab" expands to a localhost-bound, token-
# authenticated JupyterLab; exposure beyond 127.0.0.1 is prevented by the compose port
# mapping, never by arguments here.

# Color and helper setup (following BASH_OUTPUT_STYLE.md)
RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
BLUE='\033[0;34m' CYAN='\033[0;36m' NC='\033[0m'
[[ -n "${NO_COLOR:-}" ]] && RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''

print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
print_run()   { printf "${BLUE}[RUN]${NC} %s\n" "$*"; }

# Storage bootstrap (idempotent; mirrored host-side by `make interp-dirs`). hf-cache backs
# HF_HOME so model weights land on NVMe once and never in the image or the repository.
INTERP_ROOT=/mnt/ai-data/interp
if [[ -d "$(dirname "$INTERP_ROOT")" ]]; then
    mkdir -p "$INTERP_ROOT"/{activations,saes,probes,vectors,hf-cache,logs,monitor}
else
    print_warn "/mnt/ai-data not mounted: storage bootstrap skipped"
fi

# Jupyter writable dirs live on tmpfs (read_only rootfs; see Dockerfile ENV block)
mkdir -p "${JUPYTER_CONFIG_DIR:-/tmp/jupyter/config}" \
         "${JUPYTER_DATA_DIR:-/tmp/jupyter/data}" \
         "${JUPYTER_RUNTIME_DIR:-/tmp/jupyter/runtime}" \
         "${IPYTHONDIR:-/tmp/ipython}" "${MPLCONFIGDIR:-/tmp/mpl}" \
         "${XDG_CACHE_HOME:-/tmp/cache}"

echo "=== Interpretability workbench ==="
echo
echo "Environment:"
echo "  Python: $(python --version 2>&1 | awk '{print $2}')"
echo "  Torch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'import failed')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')"
echo "  HF_HOME: ${HF_HOME:-unset}"
echo "  Working dir: $(pwd)"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
else
    print_warn "nvidia-smi not found: no GPU visible in this container"
fi
echo

# Default command: JupyterLab, token-authenticated. JUPYTER_TOKEN is generated per launch by
# `make interp-up`; if it is absent, refuse rather than start an unauthenticated server.
if [[ "$1" == "jupyter-lab" ]]; then
    if [[ -z "${JUPYTER_TOKEN:-}" ]]; then
        print_error "JUPYTER_TOKEN is not set"
        echo "  Solution: launch through 'make interp-up', which generates a per-launch token"
        exit 1
    fi
    print_run "JupyterLab on 0.0.0.0:8888 (host exposure limited to 127.0.0.1 by compose)"
    exec jupyter lab \
        --ip 0.0.0.0 --port 8888 --no-browser \
        --ServerApp.root_dir=/workspace \
        --IdentityProvider.token="${JUPYTER_TOKEN}"
fi

print_run "exec: $*"
exec "$@"
