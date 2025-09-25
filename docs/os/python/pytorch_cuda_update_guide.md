# PyTorch CUDA Version Update Guide

## Overview

This guide documents the process for updating PyTorch to use different CUDA versions in Poetry-managed projects. Due to limitations in how Poetry handles PyTorch's custom package repositories, we use a hybrid approach where Poetry manages all packages except PyTorch, which is installed manually via pip.

## The Poetry-PyTorch Issue

Poetry has difficulty properly resolving PyTorch packages from custom CUDA-specific repositories. When you change the PyTorch source repository (e.g., from `cu128` to `cu129`), Poetry:
- Continues using cached package metadata from the old repository
- Fails to recognize the new repository during dependency resolution
- Shows "Repository does not exist" errors even when properly configured
- Will downgrade CUDA packages when running `poetry update` or `poetry install`

## When to Update PyTorch CUDA Version

Update PyTorch's CUDA version when:
- PyTorch releases support for a newer CUDA version
- You want to match your system's CUDA installation more closely
- You need features or optimizations from a newer CUDA version
- You see warnings about CUDA version mismatches

Check your current versions:
```bash
# System CUDA version
nvcc --version

# PyTorch CUDA version
poetry run python -c "import torch; print(torch.__version__)"
```

## Current Recommended Approach

Based on our experience, the most reliable approach is to **exclude PyTorch from Poetry management entirely** and install it manually. This prevents Poetry from downgrading CUDA packages during updates.

### pyproject.toml Configuration

Comment out PyTorch packages in your `pyproject.toml`:
```toml
# Core AI/ML Frameworks and Utilities
# PyTorch packages are managed manually via pip due to Poetry limitations with custom indexes
# Install with: ./scripts/update/update_python.sh --pytorch
# torch = {version = ">=2.3.0,<3.0.0", source = "pytorch_cuda_cu129"}
# torchvision = {version = ">=0.18.0,<1.0.0", source = "pytorch_cuda_cu129"}
# torchaudio = {version = ">=2.3.0,<3.0.0", source = "pytorch_cuda_cu129"}
```

Keep the repository source for documentation purposes:
```toml
[[tool.poetry.source]]
name = "pytorch_cuda_cu129"
url = "https://download.pytorch.org/whl/cu129"
priority = "explicit"
```

## Update Process

### Step 1: Update the CUDA URL in pyproject.toml comments

When a new CUDA version becomes available, update the URL in the commented sections:
```toml
# For CUDA 13.0 (when available):
# torch = {version = ">=2.3.0,<3.0.0", source = "pytorch_cuda_cu130"}
# torchvision = {version = ">=0.18.0,<1.0.0", source = "pytorch_cuda_cu130"}
# torchaudio = {version = ">=2.3.0,<3.0.0", source = "pytorch_cuda_cu130"}

[[tool.poetry.source]]
name = "pytorch_cuda_cu130"
url = "https://download.pytorch.org/whl/cu130"
priority = "explicit"
```

### Step 2: Update Poetry Lock File

Create a new lock file without PyTorch:
```bash
poetry lock
```

This will be fast since PyTorch packages are excluded.

### Step 3: Install Dependencies

Install all non-PyTorch dependencies:
```bash
poetry install --no-root
```

### Step 4: Manual PyTorch Installation

Install PyTorch using the update script:

```bash
# This will read the CUDA URL from pyproject.toml and install PyTorch
./scripts/update/update_python.sh --pytorch
```

Or manually:
```bash
# Remove any existing PyTorch packages
poetry run pip uninstall torch torchvision torchaudio -y

# Install PyTorch from the correct CUDA index
# For CUDA 12.9:
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# For CUDA 13.0 (when available):
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

## Verification

Verify the update was successful:

```bash
# Check PyTorch version and CUDA support
poetry run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

Expected output should show the new CUDA version, e.g., `2.8.0+cu129` or `2.x.x+cu130`.

## Important Discovery: Poetry Update Behavior

**WARNING**: Running `poetry update` or `poetry install` will downgrade PyTorch CUDA packages to an older version (usually 12.8) even after you've manually installed the correct version. This happens because:

1. Poetry's lock file contains references to CUDA 12.8 packages as dependencies of other packages
2. Even with PyTorch commented out in pyproject.toml, Poetry still tries to "fix" the CUDA packages
3. Poetry doesn't respect that these packages were manually installed

**Solution**:
- Always use `./scripts/update/update_python.sh --pytorch` after running any Poetry commands
- For updating non-PyTorch packages, prefer `poetry add package@latest` over `poetry update`
- Avoid running plain `poetry install` - use `poetry install --no-root` instead

## Troubleshooting

### Issue: Poetry downgraded CUDA packages

After running `poetry update` or `poetry install`, PyTorch shows an older CUDA version.

**Solution**: Re-run the PyTorch installation:
```bash
./scripts/update/update_python.sh --pytorch
```

### Issue: "Repository does not exist" error

Even after updating pyproject.toml, Poetry shows this error during installation.

**Solution**: This is expected since PyTorch is commented out. Use the manual installation method.

### Issue: CUDA version mismatch warnings

System has CUDA 13.0 but PyTorch uses CUDA 12.9.

**Solution**: This is usually fine due to CUDA's backward compatibility. CUDA 13.0 can run binaries built for CUDA 12.x.

### Issue: GPU not detected after update

PyTorch can't find the GPU after updating.

**Solution**:
1. Verify NVIDIA drivers: `nvidia-smi`
2. Reinstall PyTorch: `./scripts/update/update_python.sh --pytorch`
3. Check for conflicting CUDA installations

## Available PyTorch CUDA Versions

Check available versions at: https://pytorch.org/get-started/locally/

Common index URLs:
- CUDA 11.8: `https://download.pytorch.org/whl/cu118`
- CUDA 12.1: `https://download.pytorch.org/whl/cu121`
- CUDA 12.8: `https://download.pytorch.org/whl/cu128`
- CUDA 12.9: `https://download.pytorch.org/whl/cu129`
- CUDA 13.0: `https://download.pytorch.org/whl/cu130` (when available)

## Quick Reference

```bash
# Check current versions (system vs PyTorch)
nvcc --version  # System CUDA
poetry run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# Update PyTorch to CUDA 12.9 (current)
./scripts/update/update_python.sh --pytorch

# Manual update to specific CUDA version
poetry run pip uninstall torch torchvision torchaudio -y
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Fix after Poetry downgrades CUDA packages
./scripts/update/update_python.sh --pytorch

# Verify GPU detection
poetry run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Full environment check (shows both system and PyTorch CUDA)
poetry run python scripts/utils/check_py_deps_install.py

# Update non-PyTorch packages (safer than poetry update)
poetry add transformers@latest
poetry add accelerate@latest
```

## Key Learnings

1. **Poetry cannot properly manage PyTorch with custom CUDA repositories** - Even with correct configuration, Poetry fails to resolve the correct packages
2. **Commenting out PyTorch in pyproject.toml is the best solution** - This prevents Poetry from interfering with manual PyTorch installation
3. **Poetry will downgrade CUDA packages** - Running `poetry update` or `poetry install` will downgrade CUDA packages even when PyTorch is commented out
4. **Always reinstall PyTorch after Poetry commands** - Use `./scripts/update/update_python.sh --pytorch` after any Poetry operations
5. **System CUDA and PyTorch CUDA versions can differ** - CUDA 13.0 system works fine with PyTorch CUDA 12.9 due to backward compatibility

## Notes

- Always backup poetry.lock before updates: `cp poetry.lock poetry.lock.backup`
- The manual pip installation method is the ONLY reliable way to manage PyTorch CUDA versions
- CUDA backward compatibility allows newer system CUDA (13.0) with older PyTorch CUDA (12.9)
- Check PyTorch's official site for the latest supported CUDA versions
- PyTorch typically lags 1-2 versions behind the latest CUDA release