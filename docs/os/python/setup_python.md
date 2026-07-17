# Python environment setup

Configuration guide for the uv-managed Python environment on Ubuntu 24.04 for AI
experiments. uv manages the interpreter, the project virtual environment, the lock file,
and the PyTorch CUDA wheels from a single `pyproject.toml`.

## Table of contents

- [Prerequisites](#prerequisites)
- [Install uv](#install-uv)
- [Install the Python interpreter](#install-the-python-interpreter)
- [Sync the environment](#sync-the-environment)
- [Activate the environment](#activate-the-environment)
- [Acceptance checks](#acceptance-checks)
- [Common commands](#common-commands)
- [Retire the old pyenv and Poetry toolchain](#retire-the-old-pyenv-and-poetry-toolchain)
- [Troubleshooting](#troubleshooting)

## Prerequisites

System requirements:

- Ubuntu 24.04 LTS
- Internet connection for package downloads
- An RTX 5090 with the NVIDIA driver and CUDA stack installed (see
  [gpu-stack/README.md](../gpu-stack/README.md))

uv downloads a standalone CPython build, so the source-compilation packages that pyenv
required (`build-essential`, `libssl-dev`, and similar) are no longer needed.

## Install uv

Install uv with the official standalone installer:

```bash
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart the shell or source the environment file the installer prints, then verify:

```bash
$ uv --version
```

The installer places `uv` in `~/.local/bin`. Confirm that directory is on `PATH` if the
command is not found.

## Install the Python interpreter

The project pins Python 3.12 in `/.python-version`. Install a matching interpreter with
uv:

```bash
$ uv python install 3.12
```

uv manages this interpreter under its own data directory. It does not touch the system
Python or require pyenv.

## Sync the environment

From the repository root, create the virtual environment and install every dependency,
group, and extra:

```bash
$ make install
```

`make install` runs `uv sync --all-groups --all-extras`. uv reads `pyproject.toml` and
`uv.lock`, creates `.venv` in the repository root, and installs:

- the exact pinned runtime dependencies, including the `torch` and `torchvision` pair
  resolved from the CUDA 13.0 index
- the `quant` optional extra (nemo-toolkit and its quantization companions)
- the `dev` dependency group (pytest family, ruff, mypy)

The `torch` trio resolves from the `pytorch-cu130` index declared in
`[[tool.uv.index]]` and bound through `[tool.uv.sources]`. No separate pip install step
exists; uv installs the CUDA wheels directly.

## Activate the environment

uv creates the environment at `/.venv`. Prefix commands with `uv run` to use it without
activation:

```bash
$ uv run python --version
```

To activate the environment in the current shell:

```bash
$ source .venv/bin/activate
```

The `make shell` target prints this activation command rather than spawning a subshell.

## Acceptance checks

Run the following after `make install` to confirm the environment is correct.

Sync the environment:

```bash
$ uv sync --all-groups --all-extras
```

Confirm PyTorch detects the GPU:

```bash
$ uv run python -c "import torch; assert torch.cuda.is_available()"
```

Run the dependency reporter:

```bash
$ uv run python scripts/utils/check_py_deps_install.py
```

Run the Make quality targets:

```bash
$ make test
$ make lint
$ make format
```

Run the environment check without modifying the lock or environment:

```bash
$ ./scripts/update/update_python.sh --check
```

`--check` reports the uv version, the Python pin, the presence of `.venv`, the system
CUDA version, and the installed PyTorch build, then exits without running `uv lock` or
`uv sync`.

## Common commands

| Task | Command |
|------|---------|
| Sync environment (all groups and extras) | `make install` |
| Update dependencies (lock upgrade, sync, verify) | `make update` |
| Run tests | `make test` |
| Lint | `make lint` |
| Format | `make format` |
| Print activation command | `make shell` |
| Run a command in the environment | `uv run <command>` |
| Add a dependency | edit `pyproject.toml`, then `uv lock && uv sync --all-groups --all-extras` |
| Show the installed interpreter path | `uv python find 3.12` |

Direct dependencies are pinned exactly (Phase A of the pin-then-fix-then-update
migration). Add or bump a dependency by editing the pin in `pyproject.toml`, then running
`make update` or the sync command. Do not edit `uv.lock` by hand.

## Retire the old pyenv and Poetry toolchain

The host previously managed Python with pyenv and Poetry, with PyTorch installed by a pip
sideload. uv replaces all of it. Remove the retired components after `make install`
succeeds and the acceptance checks pass.

Before deleting the environment, clear any global or local pyenv pin that still selects it.
Deleting the `ai-tools` environment while the global pin `~/.pyenv/version` (or a stray
`.python-version`) still names it leaves pyenv pointing at a version that no longer exists, and
every shell that inherits the pin then fails with
`pyenv: version 'ai-tools' is not installed` on each `cd` outside a directory with its own
`.python-version`:

```bash
$ pyenv global               # if this prints ai-tools, reset it
$ pyenv global system        # or: pyenv global 3.12.11
$ pyenv global               # verify it no longer prints ai-tools
```

Then delete the pyenv-managed virtual environment used for AI tools:

```bash
$ pyenv virtualenv-delete ai-tools
```

If `pyenv virtualenv-delete` is unavailable, remove the environment directly:

```bash
$ pyenv uninstall 3.12.11/envs/ai-tools
```

Remove Poetry and its cache:

```bash
$ sudo apt remove python3-poetry
$ rm -rf ~/.cache/pypoetry
```

The old `poetry.lock` is deleted from the repository; `uv.lock` replaces it. No action is
required beyond pulling the current repository state.

Removing pyenv itself is optional. Keep pyenv if other projects on the host depend on it.
Removing the ai-tools environment and Poetry is mandatory; removing pyenv is not. If pyenv
is no longer used by any project, remove its shell initialization lines from `~/.bashrc`
or `~/.zshrc`:

```bash
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Then remove the pyenv data directory:

```bash
$ rm -rf ~/.pyenv
```

## Troubleshooting

### uv: command not found

Confirm the installer directory is on `PATH`:

```bash
$ ls ~/.local/bin/uv
```

Add `~/.local/bin` to `PATH` in the shell configuration if the binary exists but the
command is not found.

### No interpreter found for Python 3.12

Install a managed interpreter:

```bash
$ uv python install 3.12
```

### CUDA not detected

Verify the installed PyTorch build reports a CUDA version:

```bash
$ uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

The version string must include a CUDA suffix (for example `2.9.1+cu130`). A CPU-only
build indicates the `torch` sources in `pyproject.toml` no longer point at the
`pytorch-cu130` index. See
[pytorch_cuda_update_guide.md](pytorch_cuda_update_guide.md).

### Environment out of sync

Re-sync from the lock file:

```bash
$ uv sync --all-groups --all-extras
```

---

*Last updated: 2026-07-12*
