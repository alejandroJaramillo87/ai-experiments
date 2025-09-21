#!/bin/bash

# update_python.sh
# Updates Poetry, Python packages, and handles PyTorch CUDA dependencies
# Simplified version focusing on what actually works

echo "==================================================="
echo " Python/AI Package Updater"
echo "==================================================="
echo

# Parse command line arguments
CHECK_ONLY=false
UPDATE_PYTORCH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --pytorch)
            UPDATE_PYTORCH=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --check    Only check environment, don't update"
            echo "  --pytorch  Reinstall PyTorch with CUDA 12.9 support"
            echo "  --help     Show this help message"
            echo
            echo "Examples:"
            echo "  $0              # Standard update"
            echo "  $0 --check      # Check environment only"
            echo "  $0 --pytorch    # Fix PyTorch CUDA version"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get script directory for relative paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
UTILS_DIR="$SCRIPT_DIR/../utils"
PROJECT_ROOT="$SCRIPT_DIR/../.."

# Check if we're in a Poetry project
if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
    echo "ERROR: No pyproject.toml found in $PROJECT_ROOT"
    exit 1
fi

cd "$PROJECT_ROOT" || exit 1

echo "--- 1. Environment Check ---"

# Check Poetry installation
if ! command -v poetry &> /dev/null; then
    echo "ERROR: Poetry is not installed"
    echo "Install with: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

echo "Poetry version: $(poetry --version)"

# Check Python environment
echo "Python version: $(python3 --version)"
poetry_env=$(poetry env info --path 2>/dev/null)
if [ -n "$poetry_env" ]; then
    echo "Poetry environment: $poetry_env"
else
    echo "No Poetry environment found. Creating..."
    poetry install --no-root
fi

# Quick GPU check
echo
echo "Checking GPU configuration..."
poetry run python -c "
import sys
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('PyTorch not installed')
    sys.exit(1)
" 2>/dev/null || echo "PyTorch not yet installed or configured"

if $CHECK_ONLY; then
    echo
    echo "Check complete."
    exit 0
fi

echo
echo "--- 2. Updating Poetry and Tools ---"

# Update Poetry itself
echo "Checking for Poetry updates..."
poetry self update 2>/dev/null || echo "Poetry is at latest version"

# Update pip, setuptools, wheel
echo "Updating pip, setuptools, wheel..."
poetry run pip install --upgrade pip setuptools wheel -q

echo
echo "--- 3. Updating Packages ---"

# Backup poetry.lock
if [ -f "poetry.lock" ]; then
    cp poetry.lock poetry.lock.backup
    echo "Backed up poetry.lock to poetry.lock.backup"
fi

if $UPDATE_PYTORCH; then
    echo "Reinstalling PyTorch with CUDA 12.9 support..."

    # Get the CUDA index URL from pyproject.toml if defined
    CUDA_URL=$(grep -A1 'pytorch_cuda' "$PROJECT_ROOT/pyproject.toml" | grep 'url = ' | cut -d'"' -f2)
    if [ -z "$CUDA_URL" ]; then
        CUDA_URL="https://download.pytorch.org/whl/cu129"
    fi

    echo "Using PyTorch index: $CUDA_URL"

    # Uninstall existing PyTorch
    poetry run pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

    # Install PyTorch from the correct index
    poetry run pip install torch torchvision torchaudio --index-url "$CUDA_URL"

    echo "PyTorch reinstalled successfully"
else
    echo "Updating Poetry dependencies..."

    # Standard Poetry update
    poetry update

    # Check if PyTorch is using the wrong CUDA version
    PYTORCH_VERSION=$(poetry run python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
    if [[ "$PYTORCH_VERSION" == *"cu128"* ]] || [[ "$PYTORCH_VERSION" == *"cu121"* ]]; then
        echo
        echo "WARNING: PyTorch is using an older CUDA version: $PYTORCH_VERSION"
        echo "Run '$0 --pytorch' to update to CUDA 12.9"
    fi
fi

echo
echo "--- 4. Verification ---"

# Test key imports
echo "Testing key package imports..."
poetry run python -c "
import sys
packages = ['torch', 'transformers', 'accelerate', 'datasets']
failed = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError as e:
        print(f'✗ {pkg}: {e}')
        failed.append(pkg)
if failed:
    print(f'\\nFailed to import: {', '.join(failed)}')
    sys.exit(1)
"

# Final GPU check
echo
poetry run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Version: {torch.version.cuda}')
"

echo
echo "==================================================="
echo " Update Complete"
echo "==================================================="

# Get PyTorch source info
PYTORCH_SOURCE=$(grep -o 'pytorch_cuda_cu[0-9]*' "$PROJECT_ROOT/pyproject.toml" | head -1)
if [ -n "$PYTORCH_SOURCE" ]; then
    echo "PyTorch configured source: $PYTORCH_SOURCE"
fi

if [ -f "poetry.lock.backup" ]; then
    echo
    echo "Rollback available: cp poetry.lock.backup poetry.lock && poetry install"
fi