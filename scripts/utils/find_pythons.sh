#!/bin/bash

# find_pythons.sh
# Project-specific Python environment checker for AI experiments
# Verifies documented Python setup from docs/os/python/setup_python.md

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Disable colors if NO_COLOR is set
[[ -n "${NO_COLOR:-}" ]] && RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''

# Print helper functions
print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
print_skip()  { printf "${BLUE}[SKIP]${NC} %s\n" "$*"; }

echo "=== AI Experiments Python Environment Check ==="
echo "Checking Python setup as documented in docs/os/python/setup_python.md"
echo

EXPECTED_PIN="3.12"

# Derive project root from this script's location (scripts/utils/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)"

# uv installation check
echo "uv installation check:"
if command -v uv &> /dev/null; then
    print_ok "uv installation: $(uv --version)"
else
    print_error "uv installation: not installed"
    echo "  Install per: https://docs.astral.sh/uv/getting-started/installation/"
    echo
    echo "=== Environment Check Complete ==="
    exit 1
fi
echo

# Managed interpreter check
echo "Python interpreter check:"
if uv python find "$EXPECTED_PIN" &> /dev/null; then
    FOUND_PYTHON=$(uv python find "$EXPECTED_PIN")
    print_ok "Python $EXPECTED_PIN: $FOUND_PYTHON"
else
    print_warn "Python $EXPECTED_PIN: no interpreter found"
    echo "  Install with: uv python install $EXPECTED_PIN"
fi

# Reported rather than assumed: uv derives this from XDG_DATA_HOME unless
# UV_PYTHON_INSTALL_DIR overrides it, and a VS Code snap terminal sets that
# variable to a revision directory snapd later removes.
UV_PY_DIR=$(uv python dir 2>/dev/null)
if [ -n "$UV_PY_DIR" ]; then
    print_ok "Managed interpreter directory: $UV_PY_DIR"
else
    print_warn "Managed interpreter directory: uv python dir reported nothing"
fi

# pyenv reads .python-version too, so inside this repository it resolves the same
# pin to its own interpreter and puts it on PATH as python3.
if command -v pyenv &> /dev/null; then
    print_warn "pyenv is installed and reads .python-version, the file uv writes"
    echo "  Found: inside this repository pyenv resolves the $EXPECTED_PIN pin to its own"
    echo "         interpreter and shadows the project environment on PATH"
    echo "  Solution: remove pyenv per docs/os/python/setup_python.md"
fi
echo

# Project configuration check
echo "Project directory configuration:"
if [ -d "$PROJECT_DIR" ]; then
    print_ok "Project directory: $PROJECT_DIR"

    PYTHON_VERSION_FILE="$PROJECT_DIR/.python-version"
    if [ -f "$PYTHON_VERSION_FILE" ]; then
        CONFIGURED_PIN=$(cat "$PYTHON_VERSION_FILE")
        if [ "$CONFIGURED_PIN" = "$EXPECTED_PIN" ]; then
            print_ok "Python version file: .python-version -> $CONFIGURED_PIN"
        else
            print_warn "Python version file: .python-version -> $CONFIGURED_PIN, expected $EXPECTED_PIN"
            echo "  Fix with: uv python pin $EXPECTED_PIN"
        fi
    else
        print_warn "Python version file: .python-version not found"
        echo "  Create with: uv python pin $EXPECTED_PIN"
    fi

    if [ -f "$PROJECT_DIR/pyproject.toml" ]; then
        print_ok "Project configuration: pyproject.toml found"
    else
        print_error "Project configuration: pyproject.toml not found in $PROJECT_DIR"
    fi

    if [ -f "$PROJECT_DIR/uv.lock" ]; then
        print_ok "Lock file: uv.lock found"
    else
        print_warn "Lock file: uv.lock not found"
        echo "  Create with: uv lock"
    fi

    if [ -d "$PROJECT_DIR/.venv" ]; then
        print_ok "Project environment: .venv"

        # A venv records its base interpreter as an absolute path in pyvenv.cfg.
        # Remove whatever provided that path -- a retired pyenv, or the snap
        # revision directory XDG_DATA_HOME pointed at -- and the environment is
        # unusable, while every check that follows reports a symptom rather than
        # this cause. Checking the path is what separates the two.
        VENV_HOME=$(sed -n 's/^home[[:space:]]*=[[:space:]]*//p' \
            "$PROJECT_DIR/.venv/pyvenv.cfg" 2>/dev/null)
        REBUILD="uv python install $EXPECTED_PIN && rm -rf .venv && uv sync --all-groups --all-extras"
        if [ -z "$VENV_HOME" ]; then
            print_warn "Base interpreter: no home entry in .venv/pyvenv.cfg"
        elif [ ! -d "$VENV_HOME" ]; then
            print_error "Base interpreter: $VENV_HOME does not exist"
            echo "  Found: .venv is built from an interpreter that has been removed"
            echo "  Solution: reinstall the interpreter and rebuild the environment"
            echo "  Command: $REBUILD"
        elif [ -n "$UV_PY_DIR" ] && [[ "$VENV_HOME" != "$UV_PY_DIR"* ]]; then
            print_warn "Base interpreter: $VENV_HOME is outside $UV_PY_DIR"
            echo "  Found: .venv is not built from a uv-managed interpreter, so it breaks"
            echo "         whenever whatever provides that path moves or is removed"
            echo "  Solution: rebuild from the managed interpreter"
            echo "  Command: $REBUILD"
        else
            print_ok "Base interpreter: $VENV_HOME"
        fi
    else
        print_warn "Project environment: .venv not found"
        echo "  Create with: uv sync --all-groups --all-extras"
    fi
else
    print_warn "Project directory: not found"
fi
echo

# PyTorch CUDA verification
echo "PyTorch CUDA verification:"
if [ -d "$PROJECT_DIR/.venv" ]; then
    cd "$PROJECT_DIR" 2>/dev/null || true
    TORCH_CHECK=$(uv run --no-sync python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null)
    if [ -n "$TORCH_CHECK" ]; then
        print_ok "PyTorch installation"
        echo "$TORCH_CHECK" | sed 's/^/  /'
    else
        print_warn "PyTorch installation: not installed or not working"
        echo "  Install with: uv sync --all-groups --all-extras"
    fi
else
    print_skip "PyTorch installation: project environment not available"
fi

echo
echo "=== Environment Check Complete ==="
print_info "For setup instructions, see: docs/os/python/setup_python.md"
