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

# Project Python Environment Status
echo "Project Python environment status:"
echo

# Check current Python
CURRENT_PYTHON=$(which python3 2>/dev/null)
if [ -n "$CURRENT_PYTHON" ]; then
    CURRENT_VERSION=$(python3 --version 2>/dev/null)
    echo "Active Python: $CURRENT_PYTHON ($CURRENT_VERSION)"
else
    echo "Active Python: not found"
fi
echo

# Expected Python Version Check (3.12.11)
echo "Expected Python version check:"
EXPECTED_VERSION="3.12.11"
if command -v pyenv &> /dev/null; then
    if pyenv versions --bare | grep -q "^$EXPECTED_VERSION$"; then
        print_ok "Python $EXPECTED_VERSION: installed via pyenv"
    else
        print_warn "Python $EXPECTED_VERSION: not found in pyenv versions"
        echo "  Install with: pyenv install $EXPECTED_VERSION"
    fi
else
    print_error "Python $EXPECTED_VERSION: pyenv not found"
fi
echo

# AI Tools Virtual Environment Check
echo "AI tools virtual environment check:"
AI_TOOLS_ENV="ai-tools"
if command -v pyenv &> /dev/null; then
    if pyenv versions | grep -q "$AI_TOOLS_ENV"; then
        print_ok "Virtualenv $AI_TOOLS_ENV: found"
        AI_TOOLS_PYTHON="$HOME/.pyenv/versions/$EXPECTED_VERSION/envs/$AI_TOOLS_ENV/bin/python"
        if [ -f "$AI_TOOLS_PYTHON" ]; then
            AI_TOOLS_VERSION=$($AI_TOOLS_PYTHON --version 2>/dev/null)
            echo "  Python executable: $AI_TOOLS_PYTHON"
            echo "  Version: $AI_TOOLS_VERSION"
        fi
    else
        print_warn "Virtualenv $AI_TOOLS_ENV: not found"
        echo "  Create with: pyenv virtualenv $EXPECTED_VERSION $AI_TOOLS_ENV"
    fi
else
    print_error "Virtualenv $AI_TOOLS_ENV: pyenv not available"
fi
echo

# Project Directory and Python Version File
echo "Project directory configuration:"
PROJECT_DIR="$HOME/workspace/ai-experiments"
if [ -d "$PROJECT_DIR" ]; then
    print_ok "Project directory: $PROJECT_DIR"

    # Check .python-version file
    PYTHON_VERSION_FILE="$PROJECT_DIR/.python-version"
    if [ -f "$PYTHON_VERSION_FILE" ]; then
        CONFIGURED_ENV=$(cat "$PYTHON_VERSION_FILE")
        if [ "$CONFIGURED_ENV" = "$AI_TOOLS_ENV" ]; then
            print_ok "Python version file: .python-version -> $CONFIGURED_ENV"
        else
            print_warn "Python version file: .python-version -> $CONFIGURED_ENV, expected $AI_TOOLS_ENV"
        fi
    else
        print_warn "Python version file: .python-version not found"
        echo "  Create with: echo '$AI_TOOLS_ENV' > .python-version"
    fi
else
    print_warn "Project directory: $PROJECT_DIR not found"
fi
echo

# Poetry Configuration Check
echo "Poetry configuration check:"
if command -v poetry &> /dev/null; then
    POETRY_VERSION=$(poetry --version 2>/dev/null)
    print_ok "Poetry installation: $POETRY_VERSION"

    # Check if in project directory
    if [ -f "$PROJECT_DIR/pyproject.toml" ]; then
        print_ok "Project configuration: pyproject.toml found"

        # Check Poetry environment
        cd "$PROJECT_DIR" 2>/dev/null || true
        POETRY_ENV_PATH=$(poetry env info --path 2>/dev/null)
        if [ -n "$POETRY_ENV_PATH" ]; then
            print_ok "Poetry environment: $POETRY_ENV_PATH"

            # Check if it's using the correct Python
            if [[ "$POETRY_ENV_PATH" == *"$AI_TOOLS_ENV"* ]]; then
                print_ok "Environment link: using $AI_TOOLS_ENV"
            else
                print_warn "Environment link: not using $AI_TOOLS_ENV"
                echo "  Fix with: poetry env use ~/.pyenv/versions/$EXPECTED_VERSION/envs/$AI_TOOLS_ENV/bin/python"
            fi

            # Check PyTorch CUDA configuration
            if poetry config repositories.pytorch_cuda_cu129.url 2>/dev/null | grep -q "pytorch.org"; then
                print_ok "PyTorch CUDA repository: cu129 configured"
            else
                print_warn "PyTorch CUDA repository: cu129 not configured"
                echo "  Fix with: poetry config repositories.pytorch_cuda_cu129.url https://download.pytorch.org/whl/cu129"
            fi
        else
            print_warn "Poetry environment: not created"
            echo "  Create with: poetry install"
        fi
    else
        print_error "Project configuration: pyproject.toml not found in $PROJECT_DIR"
    fi
else
    print_error "Poetry installation: not installed"
    echo "  Install with: sudo apt install python3-poetry"
fi
echo

# PyTorch CUDA Verification
echo "PyTorch CUDA verification:"
if [ -n "$POETRY_ENV_PATH" ] && [ -f "$POETRY_ENV_PATH/bin/python" ]; then
    cd "$PROJECT_DIR" 2>/dev/null || true
    TORCH_CHECK=$(poetry run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null)
    if [ -n "$TORCH_CHECK" ]; then
        print_ok "PyTorch installation"
        echo "$TORCH_CHECK" | sed 's/^/  /'
    else
        print_warn "PyTorch installation: not installed or not working"
        echo "  Install with: poetry add torch torchvision torchaudio --source pytorch_cuda_cu129"
    fi
else
    print_skip "PyTorch installation: Poetry environment not available"
fi

echo
echo "=== Environment Check Complete ==="
print_info "For setup instructions, see: docs/os/python/setup_python.md"