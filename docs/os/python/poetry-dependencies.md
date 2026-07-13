# Python dependency reference

Reference for the Python dependency set of the AI workstation. uv manages the environment
from a single `pyproject.toml` with standard `[project]` metadata, exact version pins, a
CUDA-specific PyTorch index, an optional `quant` extra, and a `dev` dependency group. The
lock file is `uv.lock`; it replaces the deleted `poetry.lock`.

## Table of contents

- [Layout of pyproject.toml](#layout-of-pyprojecttoml)
- [Exact-pin policy](#exact-pin-policy)
- [Core runtime dependencies](#core-runtime-dependencies)
  - [Transformers ecosystem](#transformers-ecosystem)
  - [Training optimization](#training-optimization)
  - [Data science and visualization](#data-science-and-visualization)
  - [MLOps and experiment tracking](#mlops-and-experiment-tracking)
  - [Utilities and serving](#utilities-and-serving)
- [PyTorch and the CUDA index](#pytorch-and-the-cuda-index)
- [The quant optional extra](#the-quant-optional-extra)
- [The dev dependency group](#the-dev-dependency-group)
- [Common tasks](#common-tasks)
- [Planned Phase B modernization](#planned-phase-b-modernization)

## Layout of pyproject.toml

The file uses standard PEP 621 `[project]` metadata and uv-specific tables:

| Table | Purpose |
|-------|---------|
| `[project]` | Name, version, `requires-python`, and the runtime `dependencies` list |
| `[project.optional-dependencies]` | The `quant` extra, isolated from the core set |
| `[dependency-groups]` | The `dev` group (development-only tools) |
| `[tool.uv]` | uv settings, including `package = false` |
| `[[tool.uv.index]]` | The `pytorch-cu130` CUDA wheel index |
| `[tool.uv.sources]` | Binds the torch trio to that index |
| `[tool.uv.extra-build-dependencies]` | Supplies torch to the deepspeed build |
| `[tool.ruff]` | Linter and formatter configuration |
| `[tool.pytest.ini_options]` | Test discovery configuration |

`requires-python` is `>=3.12,<3.13`. The interpreter is pinned to 3.12 in
`/.python-version`.

## Exact-pin policy

Every direct dependency uses an exact `==` pin. This is Phase A of a
pin-then-fix-then-update migration: the pins match the last Poetry lock resolution, so the
environment is reproducible before any version changes. Change a dependency by editing its
pin in `pyproject.toml`, refreshing the lock, and re-running the acceptance checks, one
change at a time.

The `dev` group and the `quant` extra keep heavy or narrow dependencies out of the core
resolution. The `quant` extra isolates `nemo-toolkit` and its companions so their
transitive constraints cannot block upgrades of the core environment.

## Core runtime dependencies

### Transformers ecosystem

| Package | Version | Role |
|---------|---------|------|
| `transformers` | 4.51.3 | Pre-trained model loading and fine-tuning |
| `datasets` | 3.6.0 | Memory-mapped and streaming dataset loading |
| `accelerate` | 1.10.1 | Mixed precision, gradient accumulation, device placement |
| `peft` | 0.15.2 | Parameter-efficient fine-tuning (LoRA, QLoRA) |
| `evaluate` | 0.4.6 | Metric computation for model evaluation |
| `sentencepiece` | 0.2.1 | Subword tokenization |
| `sentence-transformers` | 5.1.0 | Embedding models for retrieval and similarity |
| `huggingface-hub` | 0.32.6 | Model and dataset transfer, `hf` CLI |
| `hf-transfer` | 0.1.8 | Accelerated Hub downloads |

### Training optimization

| Package | Version | Role |
|---------|---------|------|
| `deepspeed` | 0.17.6 | ZeRO optimizer, gradient, and parameter sharding |
| `bitsandbytes` | 0.46.1 | 4-bit and 8-bit quantization kernels |
| `einops` | 0.8.1 | Tensor rearrangement with Einstein notation |

`deepspeed` ships a source distribution whose `setup.py` imports torch. uv supplies torch
to that build through `[tool.uv.extra-build-dependencies]`, so the sync does not fail on a
missing build-time import.

### Data science and visualization

| Package | Version | Role |
|---------|---------|------|
| `jupyterlab` | 4.4.7 | Interactive notebook environment |
| `pandas` | 2.3.3 | Dataframe operations (held at 2.x; see note below) |
| `scikit-learn` | 1.7.2 | Classical machine learning and preprocessing |
| `matplotlib` | 3.10.6 | Plotting |
| `seaborn` | 0.13.2 | Statistical visualization |

`pandas` is held at the latest 2.x (`2.3.3`). `mlflow` requires `pandas<3` in every release
through 3.14, so `pandas` 3.x cannot resolve while `mlflow` remains a dependency. Do not bump
`pandas` to 3.x without first removing or replacing `mlflow`.

### MLOps and experiment tracking

| Package | Version | Role |
|---------|---------|------|
| `wandb` | 0.20.1 | Experiment tracking and hyperparameter sweeps |
| `mlflow` | 3.4.0 | Experiment tracking and model registry |
| `tensorboard` | 2.20.0 | Metric and histogram visualization |

### Utilities and serving

| Package | Version | Role |
|---------|---------|------|
| `tqdm` | 4.67.1 | Progress bars |
| `rich` | 14.1.0 | Terminal formatting |
| `ray[default]` | 2.49.2 | Parallel hyperparameter tuning and serving |
| `optuna` | 4.5.0 | Hyperparameter optimization |
| `tiktoken` | 0.9.0 | Byte-pair tokenizer |
| `requests` | 2.32.5 | HTTP client |
| `llama-cpp-python` | 0.3.16 | Python bindings for llama.cpp |
| `fastapi` | 0.115.14 | Model-serving API framework |
| `uvicorn` | 0.34.3 | ASGI server for FastAPI |

## PyTorch and the CUDA index

The `torch`, `torchvision`, and `torchaudio` trio is a first-class part of the runtime
`dependencies` list, pinned exactly and resolved from a CUDA-specific index rather than
PyPI:

| Package | Version |
|---------|---------|
| `torch` | 2.9.1 |
| `torchvision` | 0.24.1 |
| `torchaudio` | 2.9.1 |

`[[tool.uv.index]]` declares the `pytorch-cu130` index at
`https://download.pytorch.org/whl/cu130` with `explicit = true`. `[tool.uv.sources]` binds
the three packages to that index. Because the index is explicit, only these three packages
resolve from it; all other dependencies resolve from PyPI. uv records the exact CUDA wheels
in `uv.lock` and installs them during `uv sync`. No pip sideload step exists.

The system CUDA 13.1 toolkit runs the cu130 wheels by CUDA backward compatibility. See
[pytorch_cuda_update_guide.md](pytorch_cuda_update_guide.md) for the update procedure.

## The quant optional extra

The `quant` extra isolates the quantization and evaluation tail whose transitive
constraints would otherwise slow or block core resolution:

| Package | Version | Role |
|---------|---------|------|
| `nemo-toolkit` | 2.1.0 | NVIDIA neural modules toolkit |
| `rouge-score` | 0.1.2 | Text-generation metrics |
| `transformers-stream-generator` | 0.0.5 | Streaming generation helper |
| `mpmath` | 1.3.0 | Arbitrary-precision arithmetic |
| `sseclient` | 0.0.27 | Server-sent-events client |

`make install` installs the extra because it passes `--all-extras`. Install it explicitly
with `uv sync --extra quant`.

## The dev dependency group

The `dev` group holds development-only tools, including the pytest family that previously
lived in the main dependency list:

| Package | Version | Role |
|---------|---------|------|
| `pytest` | 8.4.2 | Test framework |
| `pytest-asyncio` | 1.2.0 | Async test support |
| `pytest-mock` | 3.15.1 | Mocking fixtures |
| `pytest-xdist` | 3.8.0 | Parallel test execution |
| `pytest-cov` | 6.2.1 | Coverage reporting |
| `ruff` | 0.13.1 | Linter and formatter |
| `mypy` | 1.18.2 | Static type checking |

`black` is no longer a dependency. `ruff format` replaces it, and `ruff` targets Python
3.12 (`target-version = "py312"`). `make lint` runs `ruff check .` and `make format` runs
`ruff format .`.

## Common tasks

```bash
# Sync the environment (all groups and extras)
$ make install

# Update dependencies (lock upgrade, sync, verify)
$ make update

# Run tests, lint, and format
$ make test
$ make lint
$ make format

# Add or bump a dependency: edit the pin in pyproject.toml, then
$ uv lock
$ uv sync --all-groups --all-extras

# Install only the quant extra
$ uv sync --extra quant

# Show the resolved dependency tree
$ uv tree
```

Commit `uv.lock` with every dependency change. Do not edit `uv.lock` by hand.

## Planned Phase B modernization

Phase A pins the environment to the last known-good resolution. Phase B modernizes the
majors, one group at a time, re-running the acceptance checks after each bump:

- Move the torch trio to the 2.13.x line on the cu130 index.
- Move `transformers` to the 5.x line. TransformerLens 3.4 requires
  `transformers >= 5.4` for future mechanistic-interpretability work.
- Add a future `interp` dependency group (`transformer-lens`, `nnsight`) once
  `transformers` is on 5.x.
- Coordinate a future training tier on the same majors as the interp tier so the two do
  not diverge.

## Next steps

- Review [setup_python.md](setup_python.md) for environment setup and host cleanup.
- See [pytorch_cuda_update_guide.md](pytorch_cuda_update_guide.md) for the CUDA build
  update procedure.
- See [gpu-stack/README.md](../gpu-stack/README.md) for the GPU software stack.

---

*Last updated: 2026-07-12*
