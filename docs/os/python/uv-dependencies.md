# Python dependency reference

Reference for the Python dependency set of the AI workstation. uv manages the environment
from a single `pyproject.toml` with standard `[project]` metadata, exact version pins, a
CUDA-specific PyTorch index, an optional `quant` extra, and `dev` and `interp` dependency
groups. The lock file is `uv.lock`; it replaces the deleted `poetry.lock`.

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
- [The interp dependency group](#the-interp-dependency-group)
- [Common tasks](#common-tasks)
- [Version modernization](#version-modernization)

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
| `transformers` | 5.13.1 | Pre-trained model loading and fine-tuning |
| `datasets` | 5.0.0 | Memory-mapped and streaming dataset loading |
| `accelerate` | 1.14.0 | Mixed precision, gradient accumulation, device placement |
| `peft` | 0.19.1 | Parameter-efficient fine-tuning (LoRA, QLoRA) |
| `evaluate` | 0.4.6 | Metric computation for model evaluation |
| `sentencepiece` | 0.2.2 | Subword tokenization |
| `sentence-transformers` | 5.6.0 | Embedding models for retrieval and similarity |
| `huggingface-hub` | 1.23.0 | Model and dataset transfer, `hf` CLI |
| `hf-transfer` | 0.1.9 | Accelerated Hub downloads |

### Training optimization

| Package | Version | Role |
|---------|---------|------|
| `deepspeed` | 0.19.2 | ZeRO optimizer, gradient, and parameter sharding |
| `bitsandbytes` | 0.49.2 | 4-bit and 8-bit quantization kernels (CUDA 13 wheels) |
| `einops` | 0.8.2 | Tensor rearrangement with Einstein notation |

`deepspeed` ships a source distribution whose `setup.py` imports torch. uv supplies torch
to that build through `[tool.uv.extra-build-dependencies]`, so the sync does not fail on a
missing build-time import.

### Data science and visualization

| Package | Version | Role |
|---------|---------|------|
| `jupyterlab` | 4.6.1 | Interactive notebook environment |
| `pandas` | 2.3.3 | Dataframe operations (held at 2.x; see note below) |
| `scikit-learn` | 1.9.0 | Classical machine learning and preprocessing |
| `matplotlib` | 3.11.0 | Plotting |
| `seaborn` | 0.13.2 | Statistical visualization |

`pandas` is held at the latest 2.x (`2.3.3`). `mlflow` requires `pandas<3` in every release
through 3.14, so `pandas` 3.x cannot resolve while `mlflow` remains a dependency. Do not bump
`pandas` to 3.x without first removing or replacing `mlflow`.

### MLOps and experiment tracking

| Package | Version | Role |
|---------|---------|------|
| `wandb` | 0.28.0 | Experiment tracking and hyperparameter sweeps |
| `mlflow` | 3.14.0 | Experiment tracking and model registry |
| `tensorboard` | 2.21.0 | Metric and histogram visualization |

### Utilities and serving

| Package | Version | Role |
|---------|---------|------|
| `tqdm` | 4.68.4 | Progress bars |
| `rich` | 15.0.0 | Terminal formatting |
| `ray[default]` | 2.56.0 | Parallel hyperparameter tuning and serving |
| `optuna` | 4.9.0 | Hyperparameter optimization |
| `tiktoken` | 0.13.0 | Byte-pair tokenizer |
| `requests` | 2.34.2 | HTTP client |
| `fastapi` | 0.139.0 | Model-serving API framework |
| `uvicorn` | 0.51.0 | ASGI server for FastAPI |

## PyTorch and the CUDA index

The `torch` and `torchvision` pair is a first-class part of the runtime `dependencies` list,
pinned exactly and resolved from a CUDA-specific index rather than PyPI:

| Package | Version |
|---------|---------|
| `torch` | 2.13.0 |
| `torchvision` | 0.28.0 |

`torchaudio` was previously part of this set but is no longer a dependency: it is unused in
the codebase and has no release past 2.11.0, so it cannot pair with `torch` 2.13.0.

`[[tool.uv.index]]` declares the `pytorch-cu130` index at
`https://download.pytorch.org/whl/cu130` with `explicit = true`. `[tool.uv.sources]` binds
both packages to that index. Because the index is explicit, only these packages
resolve from it; all other dependencies resolve from PyPI. uv records the exact CUDA wheels
in `uv.lock` and installs them during `uv sync`. No pip sideload step exists.

The system CUDA 13.0.88 toolkit runs the cu130 wheels by CUDA backward compatibility. See
[pytorch_cuda_update_guide.md](pytorch_cuda_update_guide.md) for the update procedure.

## The quant optional extra

The `quant` extra isolates the quantization and evaluation tail whose transitive
constraints would otherwise slow or block core resolution:

| Package | Version | Role |
|---------|---------|------|
| `nemo-toolkit` | 2.1.0 | NVIDIA neural modules toolkit (held; see note) |
| `rouge-score` | 0.1.2 | Text-generation metrics |
| `transformers-stream-generator` | 0.0.5 | Streaming generation helper |
| `mpmath` | 1.3.0 | Arbitrary-precision arithmetic (held; see note) |
| `sseclient` | 0.0.27 | Server-sent-events client |

`make install` installs the extra because it passes `--all-extras`. Install it explicitly
with `uv sync --extra quant`.

`nemo-toolkit` is held at 2.1.0, the last release that leaves `protobuf` unpinned. Every
release from 2.4 onward hard-pins `protobuf~=5.29.5`, which conflicts with the core
`tensorboard` requirement of `protobuf>=6.31.1`. Bumping `nemo-toolkit` would force a
downgrade of core `tensorboard`, which the isolation of the quant extra exists to prevent.
Revisit once a `nemo-toolkit` release relaxes its `protobuf` pin to allow 6.x.

`mpmath` is held at 1.3.0 for the same class of reason: core `torch` depends on
`sympy>=1.13.3`, which requires `mpmath<1.4`. The next release (1.4.0) is therefore out of
reach until torch's `sympy` dependency allows `mpmath>=1.4`. Both quant packages that could
otherwise move are already at the ceiling the modernized core permits.

## The dev dependency group

The `dev` group holds development-only tools, including the pytest family that previously
lived in the main dependency list:

| Package | Version | Role |
|---------|---------|------|
| `pytest` | 9.1.1 | Test framework |
| `pytest-asyncio` | 1.4.0 | Async test support |
| `pytest-mock` | 3.15.1 | Mocking fixtures |
| `pytest-xdist` | 3.8.0 | Parallel test execution |
| `pytest-cov` | 7.1.0 | Coverage reporting |
| `ruff` | 0.15.21 | Linter and formatter |
| `mypy` | 2.3.0 | Static type checking |

`black` is no longer a dependency. `ruff format` replaces it, and `ruff` targets Python
3.12 (`target-version = "py312"`). `make lint` runs `ruff check .` and `make format` runs
`ruff format .`.

## The interp dependency group

The `interp` group holds mechanistic-interpretability tooling. It is gated behind its own
group so the core install stays lean; `make install` (which syncs all groups) includes it.

| Package | Version | Role |
|---------|---------|------|
| `transformer-lens` | 3.5.1 | Hooked transformer internals for activation analysis |
| `nnsight` | 0.7.0 | Tracing and intervention on model internals |

`transformer-lens` requires `transformers >= 5.4`, satisfied by the `transformers` 5.13.1
pin in the core dependencies.

## The training dependency group

The `training` group (Phase 0 D1) holds the training-tier libraries not already in the core
dependencies. Core already provides torch, torchvision, transformers, datasets, accelerate,
peft, deepspeed, bitsandbytes, sentencepiece, wandb, mlflow, and tensorboard, so the group adds
only the remainder. It tracks the same majors as the `interp` tier so the two do not diverge.
The container in `docker/training/Dockerfile.training` builds and CUDA-verifies this group; see
[docker_training_overview.md](../../sandbox/docker_training_overview.md).

| Package | Role |
|---------|------|
| `trl` | Fine-tuning trainers (SFT, DPO, GRPO) |
| `liger-kernel` | Fused Triton kernels for lower-memory training |
| `torchao` | float8 training (primary FP8 backend on sm_120) |
| `flash-attn` | FlashAttention-2 kernels (sm_120) |
| `hydra-core` | Recipe and configuration composition |
| `pynvml` | GPU telemetry through NVML |
| `xxhash` | Fast content hashing for shard and dataset manifests |
| `zstandard` | Shard and artifact compression |

`flash-attn` (like `deepspeed`) imports torch during its build, so it is listed in
`[tool.uv.extra-build-dependencies]`. The CUDA-compiled members (`flash-attn`, `liger-kernel`,
`torchao`) build against the cu130 torch pinned in `[tool.uv.sources]`.

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

## Version modernization

The environment has been modernized to current majors, one group at a time, re-running the
acceptance checks after each bump:

- `torch` moved to 2.13.0 on the cu130 index (`torchvision` 0.28.0; `torchaudio` dropped as
  unused with no release past 2.11).
- `transformers` moved to the 5.x line (5.13.1), which unblocks the `interp` group.
- The `interp` dependency group (`transformer-lens`, `nnsight`) is present and installed.
- `pandas` is held at 2.x under the `mlflow` `pandas<3` constraint (see above).

The `training` dependency group (Phase 0 D1) is present and tracks the same majors as the
`interp` tier so the two do not diverge.

## Next steps

- Review [setup_python.md](setup_python.md) for environment setup and host cleanup.
- See [pytorch_cuda_update_guide.md](pytorch_cuda_update_guide.md) for the CUDA build
  update procedure.
- See [gpu-stack/README.md](../gpu-stack/README.md) for the GPU software stack.

---

*Last updated: 2026-07-12*
