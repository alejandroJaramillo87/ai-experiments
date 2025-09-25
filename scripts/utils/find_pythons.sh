#!/bin/bash

# find_pythons.sh
# Project-specific Python environment checker for AI experiments
# Verifies documented Python setup from docs/os/python/setup_python.md

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
        echo "Python $EXPECTED_VERSION: OK (installed via pyenv)"
    else
        echo "Python $EXPECTED_VERSION: WARN (not found in pyenv versions)"
        echo "  Install with: pyenv install $EXPECTED_VERSION"
    fi
else
    echo "Python $EXPECTED_VERSION: ERROR (pyenv not found)"
fi
echo

# AI Tools Virtual Environment Check
echo "AI tools virtual environment check:"
AI_TOOLS_ENV="ai-tools"
if command -v pyenv &> /dev/null; then
    if pyenv versions | grep -q "$AI_TOOLS_ENV"; then
        echo "Virtualenv $AI_TOOLS_ENV: OK (found)"
        AI_TOOLS_PYTHON="$HOME/.pyenv/versions/$EXPECTED_VERSION/envs/$AI_TOOLS_ENV/bin/python"
        if [ -f "$AI_TOOLS_PYTHON" ]; then
            AI_TOOLS_VERSION=$($AI_TOOLS_PYTHON --version 2>/dev/null)
            echo "  Python executable: $AI_TOOLS_PYTHON"
            echo "  Version: $AI_TOOLS_VERSION"
        fi
    else
        echo "Virtualenv $AI_TOOLS_ENV: WARN (not found)"
        echo "  Create with: pyenv virtualenv $EXPECTED_VERSION $AI_TOOLS_ENV"
    fi
else
    echo "Virtualenv $AI_TOOLS_ENV: ERROR (pyenv not available)"
fi
echo

# Project Directory and Python Version File
echo "Project directory configuration:"
PROJECT_DIR="$HOME/workspace/ai-experiments"
if [ -d "$PROJECT_DIR" ]; then
    echo "Project directory: OK ($PROJECT_DIR)"

    # Check .python-version file
    PYTHON_VERSION_FILE="$PROJECT_DIR/.python-version"
    if [ -f "$PYTHON_VERSION_FILE" ]; then
        CONFIGURED_ENV=$(cat "$PYTHON_VERSION_FILE")
        if [ "$CONFIGURED_ENV" = "$AI_TOOLS_ENV" ]; then
            echo "Python version file: OK (.python-version -> $CONFIGURED_ENV)"
        else
            echo "Python version file: WARN (.python-version -> $CONFIGURED_ENV, expected $AI_TOOLS_ENV)"
        fi
    else
        echo "Python version file: WARN (.python-version not found)"
        echo "  Create with: echo '$AI_TOOLS_ENV' > .python-version"
    fi
else
    echo "Project directory: WARN ($PROJECT_DIR not found)"
fi
echo

# Poetry Configuration Check
echo "Poetry configuration check:"
if command -v poetry &> /dev/null; then
    POETRY_VERSION=$(poetry --version 2>/dev/null)
    echo "Poetry installation: OK ($POETRY_VERSION)"

    # Check if in project directory
    if [ -f "$PROJECT_DIR/pyproject.toml" ]; then
        echo "Project configuration: OK (pyproject.toml found)"

        # Check Poetry environment
        cd "$PROJECT_DIR" 2>/dev/null || true
        POETRY_ENV_PATH=$(poetry env info --path 2>/dev/null)
        if [ -n "$POETRY_ENV_PATH" ]; then
            echo "Poetry environment: OK ($POETRY_ENV_PATH)"

            # Check if it's using the correct Python
            if [[ "$POETRY_ENV_PATH" == *"$AI_TOOLS_ENV"* ]]; then
                echo "Environment link: OK (using $AI_TOOLS_ENV)"
            else
                echo "Environment link: WARN (not using $AI_TOOLS_ENV)"
                echo "  Fix with: poetry env use ~/.pyenv/versions/$EXPECTED_VERSION/envs/$AI_TOOLS_ENV/bin/python"
            fi

            # Check PyTorch CUDA configuration
            if poetry config repositories.pytorch_cuda_cu129.url 2>/dev/null | grep -q "pytorch.org"; then
                echo "PyTorch CUDA repository: OK (cu129 configured)"
            else
                echo "PyTorch CUDA repository: WARN (cu129 not configured)"
                echo "  Fix with: poetry config repositories.pytorch_cuda_cu129.url https://download.pytorch.org/whl/cu129"
            fi
        else
            echo "Poetry environment: WARN (not created)"
            echo "  Create with: poetry install"
        fi
    else
        echo "Project configuration: ERROR (pyproject.toml not found in $PROJECT_DIR)"
    fi
else
    echo "Poetry installation: ERROR (not installed)"
    echo "  Install with: sudo apt install python3-poetry"
fi
echo

# PyTorch CUDA Verification
echo "PyTorch CUDA verification:"
if [ -n "$POETRY_ENV_PATH" ] && [ -f "$POETRY_ENV_PATH/bin/python" ]; then
    cd "$PROJECT_DIR" 2>/dev/null || true
    TORCH_CHECK=$(poetry run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null)
    if [ -n "$TORCH_CHECK" ]; then
        echo "PyTorch installation: OK"
        echo "$TORCH_CHECK" | sed 's/^/  /'
    else
        echo "PyTorch installation: WARN (not installed or not working)"
        echo "  Install with: poetry add torch torchvision torchaudio --source pytorch_cuda_cu129"
    fi
else
    echo "PyTorch installation: SKIP (Poetry environment not available)"
fi

echo
echo "=== Environment Check Complete ==="
echo "For setup instructions, see: docs/os/python/setup_python.md"