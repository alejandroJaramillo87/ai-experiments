# PyTorch CUDA Version Update Guide

## Overview

This guide documents the process for updating PyTorch to use different CUDA versions in Poetry-managed projects. Due to limitations in how Poetry handles PyTorch's custom package repositories, special steps are required when switching between CUDA versions.

## The Poetry-PyTorch Issue

Poetry has difficulty properly resolving PyTorch packages from custom CUDA-specific repositories. When you change the PyTorch source repository (e.g., from `cu128` to `cu129`), Poetry often:
- Continues using cached package metadata from the old repository
- Fails to recognize the new repository during dependency resolution
- Shows "Repository does not exist" errors even when properly configured

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

## Update Process

### Step 1: Update pyproject.toml

1. Update the repository definition:
```toml
# Example: Updating from CUDA 12.9 to 13.0
[[tool.poetry.source]]
name = "pytorch_cuda_cu130"  # Change name
url = "https://download.pytorch.org/whl/cu130"  # Change URL
priority = "explicit"
```

2. Update package source references:
```toml
torch = {version = ">=2.3.0,<3.0.0", source = "pytorch_cuda_cu130"}
torchvision = {version = ">=0.18.0,<1.0.0", source = "pytorch_cuda_cu130"}
torchaudio = {version = ">=2.3.0,<3.0.0", source = "pytorch_cuda_cu130"}
```

### Step 2: Clear Poetry Caches

Remove all cached package information:
```bash
# Clear Poetry and pip caches
rm -rf ~/.cache/pypoetry/
rm -rf ~/.cache/pip/

# Remove the lock file
rm poetry.lock
```

### Step 3: Attempt Poetry Resolution

Try to generate a new lock file:
```bash
poetry lock --no-cache
```

**Note**: This will likely still resolve to the wrong CUDA version due to Poetry's limitations.

### Step 4: Manual PyTorch Installation (Recommended)

Since Poetry often fails to properly resolve PyTorch from custom repositories, use this manual approach:

```bash
# Install all dependencies first
poetry install

# Remove incorrectly installed PyTorch packages
poetry run pip uninstall torch torchvision torchaudio -y

# Install PyTorch from the correct CUDA index
# For CUDA 12.9:
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# For CUDA 13.0 (when available):
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### Step 5: Using the Update Script

The simplified update script handles this automatically:
```bash
# Updates PyTorch to the CUDA version specified in pyproject.toml
./scripts/update/update_python.sh --pytorch
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

## Troubleshooting

### Issue: "Repository does not exist" error

Even after updating pyproject.toml, Poetry shows this error during installation.

**Solution**: This is expected. Use the manual installation method (Step 4).

### Issue: Poetry keeps installing old CUDA version

Poetry's lock file still references the old CUDA version even after regeneration.

**Solution**:
1. Completely remove caches: `rm -rf ~/.cache/pypoetry/ ~/.cache/pip/`
2. Use manual pip installation within Poetry environment

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
# Check current versions
poetry run python -c "import torch; print(torch.__version__)"
nvcc --version

# Update PyTorch (after modifying pyproject.toml)
./scripts/update/update_python.sh --pytorch

# Manual update to specific CUDA version
poetry run pip uninstall torch torchvision torchaudio -y
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Verify GPU detection
poetry run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Full environment check
poetry run python scripts/utils/check_py_deps_install.py
```

## Notes

- Always backup poetry.lock before updates: `cp poetry.lock poetry.lock.backup`
- The manual pip installation method is more reliable than Poetry's resolution
- CUDA backward compatibility usually allows newer system CUDA with older PyTorch CUDA
- Check PyTorch's official site for the latest supported CUDA versions