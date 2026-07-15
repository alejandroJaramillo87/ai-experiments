#!/bin/bash
set -euo pipefail

# SAEBench isolated environment (interp Phase 4 P4-D3)
#
# SAEBench 0.6.0 requires datasets<4.0 and numpy<2.0, which conflict with the
# repository's core datasets==5.0.0 pin, so it cannot join the interp
# dependency group (P4-D3 resolved decision). This script creates a dedicated
# uv venv on NVMe with sae-bench pinned; the full SAEBench battery runs only
# against TransformerLens-supported subjects (the Qwen3 calibration run),
# cross-checking the direct core-metric implementations in
# interp/sae/evaluate.py that carry the Gemma 4 rows.
#
# Host-side script: run on the workstation, not in the workbench container.
#
# Exit codes: 0 environment ready, 1 uv missing or install failure.

SAEBENCH_PIN="sae-bench==0.6.0"
ENV_DIR="${SAEBENCH_ENV:-/mnt/ai-data/interp/saebench-env}"

RED='\033[0;31m' GREEN='\033[0;32m' CYAN='\033[0;36m' NC='\033[0m'
if [[ -n "${NO_COLOR:-}" ]] || [[ "${1:-}" == "--no-color" ]]; then
    RED='' GREEN='' CYAN='' NC=''
fi
print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }

if ! command -v uv >/dev/null 2>&1; then
    print_error "uv not found on PATH"
    echo "  Solution: install uv (https://docs.astral.sh/uv/)"
    exit 1
fi

print_info "SAEBench environment: $ENV_DIR"
uv venv --allow-existing "$ENV_DIR"
uv pip install --python "$ENV_DIR/bin/python" "$SAEBENCH_PIN"

installed=$("$ENV_DIR/bin/python" -c "import importlib.metadata as m; print(m.version('sae-bench'))")
print_ok "sae-bench $installed installed (isolated from the interp group)"
print_info "Run evals with: $ENV_DIR/bin/python -m <sae_bench eval module>"
