# PyTorch CUDA version update guide

This guide documents how to change the PyTorch CUDA build in the uv-managed environment.
uv resolves `torch` and `torchvision` directly from a
CUDA-specific PyTorch index, so PyTorch is a first-class locked dependency. No pip
sideload and no manual reinstall step exist.

## Table of contents

- [How PyTorch is locked](#how-pytorch-is-locked)
- [When to update the CUDA version](#when-to-update-the-cuda-version)
- [Check current versions](#check-current-versions)
- [Update process](#update-process)
- [Verification](#verification)
- [Available PyTorch CUDA indexes](#available-pytorch-cuda-indexes)
- [System CUDA and PyTorch CUDA compatibility](#system-cuda-and-pytorch-cuda-compatibility)
- [Troubleshooting](#troubleshooting)
- [Quick reference](#quick-reference)

## How PyTorch is locked

`pyproject.toml` pins `torch` and `torchvision` to exact versions and binds them to a named
CUDA index (`torchaudio` was dropped: unused and no release past 2.11.0):

```toml
dependencies = [
    "torch==2.13.0",
    "torchvision==0.28.0",
]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
torchvision = { index = "pytorch-cu130" }
```

`explicit = true` means only packages named in `[tool.uv.sources]` resolve from that
index; every other dependency resolves from PyPI. uv records the exact CUDA wheels in
`uv.lock` and installs them during `uv sync`.

deepspeed ships a source distribution whose `setup.py` imports torch at build time. uv
supplies torch to that build through `[tool.uv.extra-build-dependencies]`:

```toml
[tool.uv.extra-build-dependencies]
deepspeed = ["torch"]
```

## When to update the CUDA version

Update the PyTorch CUDA build when:

- PyTorch releases wheels for a newer CUDA version
- A newer CUDA build is required to match a system upgrade
- A newer CUDA build provides a required feature or fix

## Check current versions

Check the system CUDA version reported by the driver toolkit:

```bash
$ nvcc --version
```

Check the installed PyTorch build:

```bash
$ uv run python -c "import torch; print(torch.__version__)"
```

The version string includes the CUDA suffix, for example `2.9.1+cu130`.

## Update process

Changing the CUDA build follows the pin-then-fix-then-update rule (see AGENTS.md): edit
the pins, refresh the lock, sync, then run the acceptance checks.

### Step 1: Edit the index and the pins

To move to a different CUDA build, update the index URL and name, the `[tool.uv.sources]`
bindings, and the version pins in `pyproject.toml`. For example, to move to a
CUDA 13.1 build once it is published:

```toml
dependencies = [
    "torch==2.14.0",
    "torchvision==0.29.0",
]

[[tool.uv.index]]
name = "pytorch-cu131"
url = "https://download.pytorch.org/whl/cu131"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu131" }
torchvision = { index = "pytorch-cu131" }
```

Change one thing at a time. Do not combine a CUDA index change with an unrelated version
bump in the same step.

### Step 2: Refresh the lock file

```bash
$ uv lock
```

uv re-resolves `torch` and `torchvision` from the new index and rewrites `uv.lock`.

### Step 3: Sync the environment

```bash
$ uv sync --all-groups --all-extras
```

### Step 4: Run the acceptance checks

```bash
$ uv run python -c "import torch; assert torch.cuda.is_available()"
$ uv run python scripts/utils/check_py_deps_install.py
```

The update script runs the lock refresh, sync, and PyTorch verification in one step:

```bash
$ ./scripts/update/update_python.sh
```

The script runs `uv lock --upgrade` followed by `uv sync --all-groups --all-extras`, then
verifies imports and the CUDA build. It warns if the installed torch reports an older
CUDA build than expected, which indicates the pins or the `[tool.uv.sources]` bindings
were edited incorrectly.

## Verification

Confirm the installed build and GPU detection:

```bash
$ uv run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

The output shows the CUDA suffix, for example `2.9.1+cu130`, and reports the RTX 5090.

## Available PyTorch CUDA indexes

Check published builds at https://pytorch.org/get-started/locally/.

| CUDA version | Index URL |
|--------------|-----------|
| CUDA 11.8 | `https://download.pytorch.org/whl/cu118` |
| CUDA 12.1 | `https://download.pytorch.org/whl/cu121` |
| CUDA 12.8 | `https://download.pytorch.org/whl/cu128` |
| CUDA 12.9 | `https://download.pytorch.org/whl/cu129` |
| CUDA 13.0 | `https://download.pytorch.org/whl/cu130` |
| CUDA 13.1 | `https://download.pytorch.org/whl/cu131` (when published) |

## System CUDA and PyTorch CUDA compatibility

The system CUDA version and the PyTorch CUDA build do not need to match exactly. CUDA is
backward compatible: a system running CUDA 13.1 runs wheels built for CUDA 13.0. The
current environment pairs a system CUDA 13.1 toolkit with the cu130 PyTorch wheels, and
the cu130 wheels run without modification.

## Troubleshooting

### PyTorch installs a CPU-only build

The `torch` version string has no CUDA suffix.

Confirm `[tool.uv.sources]` in `pyproject.toml` binds `torch` and `torchvision`
to the CUDA index, then refresh:

```bash
$ uv lock
$ uv sync --all-groups --all-extras
```

### CUDA version mismatch warning

The system reports one CUDA version and PyTorch reports another.

This is expected and safe under CUDA backward compatibility. A cu130 build runs on a
system CUDA 13.1 toolkit.

### GPU not detected after update

`torch.cuda.is_available()` returns `False`.

1. Verify the driver: `nvidia-smi`
2. Re-sync the environment: `uv sync --all-groups --all-extras`
3. Confirm the installed build reports a CUDA version:
   `uv run python -c "import torch; print(torch.version.cuda)"`

## Quick reference

```bash
# Check current versions (system vs PyTorch)
$ nvcc --version
$ uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# Update the CUDA build: edit the pins and index in pyproject.toml, then
$ uv lock
$ uv sync --all-groups --all-extras

# Or run the update script (lock upgrade, sync, verify)
$ ./scripts/update/update_python.sh

# Verify GPU detection
$ uv run python -c "import torch; assert torch.cuda.is_available()"

# Full environment check (system and PyTorch CUDA)
$ uv run python scripts/utils/check_py_deps_install.py

# Roll back a bad update
$ git checkout uv.lock && uv sync --all-groups --all-extras
```

---

*Last updated: 2026-07-12*
