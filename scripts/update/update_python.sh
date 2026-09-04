#!/bin/bash

# update_python.sh
# Updates the uv-managed Python environment and verifies the PyTorch CUDA stack.
# Direct dependencies are pinned exactly in pyproject.toml, so the update path
# only refreshes transitive dependencies; version bumps are edits to pyproject.toml
# followed by this script (pin, fix, then update -- see AGENTS.md).

RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
BLUE='\033[0;34m' CYAN='\033[0;36m' NC='\033[0m'

if [[ -n "${NO_COLOR:-}" ]] || [[ "${1:-}" == "--no-color" ]]; then
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''
    [[ "${1:-}" == "--no-color" ]] && shift
fi

print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
print_run()   { printf "${BLUE}[RUN]${NC} %s\n" "$*"; }

CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --check      Only check environment, don't update"
            echo "  --no-color   Disable colored output"
            echo "  --help       Show this help message"
            echo
            echo "Examples:"
            echo "  $0              # Refresh lock, sync, verify"
            echo "  $0 --check      # Check environment only"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "  Solution: Use --help for usage information"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."

if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
    print_error "No pyproject.toml found in $PROJECT_ROOT"
    exit 1
fi

cd "$PROJECT_ROOT" || exit 1

echo "=== Python Environment Update ==="
echo
echo "Environment check:"

if ! command -v uv &> /dev/null; then
    print_error "uv is not installed"
    echo "  Solution: Install uv from https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

print_ok "uv version: $(uv --version | awk '{print $2}')"
print_ok "Python pin: $(cat .python-version)"

if [ -d ".venv" ]; then
    print_ok "Project environment: .venv"

    # The base interpreter, by absolute path. A venv whose home has been removed
    # -- a retired pyenv, a snap revision directory -- fails every check below in
    # ways that name the symptom rather than the cause.
    VENV_HOME=$(sed -n 's/^home[[:space:]]*=[[:space:]]*//p' .venv/pyvenv.cfg 2>/dev/null)
    if [ -z "$VENV_HOME" ]; then
        print_warn "Base interpreter: no home entry in .venv/pyvenv.cfg"
    elif [ -d "$VENV_HOME" ]; then
        print_ok "Base interpreter: $VENV_HOME"
    else
        print_error "Base interpreter: $VENV_HOME does not exist"
        echo "  Solution: reinstall the interpreter and rebuild the environment"
        echo "  Command: uv python install \$(cat .python-version) && rm -rf .venv && uv sync --all-groups --all-extras"
    fi
else
    print_warn "No .venv found; will be created by uv sync"
fi

if command -v nvcc &> /dev/null; then
    SYSTEM_CUDA=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    print_ok "System CUDA (nvcc): $SYSTEM_CUDA"
else
    print_warn "System CUDA (nvcc): not found"
fi

check_torch() {
    uv run --no-sync python - <<'EOF'
import sys
try:
    import torch
except ImportError:
    print("PyTorch: not installed")
    sys.exit(1)
print(f"PyTorch: {torch.__version__}")
cuda = getattr(torch.version, "cuda", None)
print(f"PyTorch CUDA: {cuda or 'CPU-only build'}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF
}

echo
check_torch || print_warn "PyTorch not yet installed; run uv sync"

if $CHECK_ONLY; then
    echo
    print_ok "Check complete"
    exit 0
fi

echo
echo "Updating environment:"

print_run "Refreshing lock (transitive dependencies within pinned versions)"
uv lock --upgrade || { print_error "uv lock failed"; exit 1; }

print_run "Syncing environment (all groups and extras)"
uv sync --all-groups --all-extras || { print_error "uv sync failed"; exit 1; }

echo
echo "Verification:"

uv run --no-sync python - <<'EOF'
import sys
packages = ["torch", "transformers", "accelerate", "datasets"]
failed = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f"[OK] import {pkg}")
    except ImportError as e:
        print(f"[ERROR] import {pkg}: {e}")
        failed.append(pkg)
sys.exit(1 if failed else 0)
EOF
[ $? -eq 0 ] || { print_error "Package import verification failed"; exit 1; }

echo
check_torch || { print_error "PyTorch verification failed"; exit 1; }

# The torch trio is pinned to the cu130 index in pyproject.toml [tool.uv.sources];
# a cu12x build here means the pins or sources were edited incorrectly
TORCH_BUILD=$(uv run --no-sync python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
case "$TORCH_BUILD" in
    *cu121*|*cu128*|*cu129*)
        print_warn "PyTorch built for an older CUDA version: $TORCH_BUILD"
        echo "  Solution: Check [tool.uv.sources] in pyproject.toml points at the cu130 index"
        ;;
esac

echo
print_ok "Update complete"
echo "Rollback: git checkout uv.lock && uv sync --all-groups --all-extras"
