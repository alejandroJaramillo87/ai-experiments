# Python Environment Setup

Configuration guide for Python development environment using pyenv and Poetry on Ubuntu 24.04 for AI experiments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Install Build Dependencies](#install-build-dependencies)
- [Install pyenv](#install-pyenv)
- [Configure Shell](#configure-shell)
- [Install Python](#install-python)
- [Create Virtual Environment](#create-virtual-environment)
- [Install Poetry](#install-poetry)
- [Configure Project](#configure-project)
- [Install AI Dependencies](#install-ai-dependencies)
- [Verify Installation](#verify-installation)
- [Common Commands](#common-commands)
- [Troubleshooting](#troubleshooting)

## Prerequisites

System requirements:
- Ubuntu 24.04 LTS
- Internet connection for package downloads
- sudo privileges for system package installation

## Install Build Dependencies

Install packages required for compiling Python from source:

```bash
sudo apt update
sudo apt install -y build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev \
  libncurses5-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev
```

Additional development tools:
```bash
sudo apt install -y wget curl llvm xz-utils tk-dev
```

## Install pyenv

Clone pyenv repository and plugins:

```bash
# Clone pyenv
git clone https://github.com/pyenv/pyenv.git ~/.pyenv

# Clone pyenv-virtualenv plugin
git clone https://github.com/pyenv/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv

# Clone pyenv-update plugin
git clone https://github.com/pyenv/pyenv-update.git ~/.pyenv/plugins/pyenv-update

# Clone pyenv-doctor plugin
git clone https://github.com/pyenv/pyenv-doctor.git ~/.pyenv/plugins/pyenv-doctor
```

## Configure Shell

Add pyenv to shell configuration. For Zsh:

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
```

For Bash:

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
```

Reload shell configuration:

```bash
# For Zsh
source ~/.zshrc

# For Bash
source ~/.bashrc
```

## Install Python

Install Python 3.12.11:

```bash
pyenv install 3.12.11
```

Verify installation:

```bash
pyenv versions
```

## Create Virtual Environment

Create dedicated environment for AI tools:

```bash
pyenv virtualenv 3.12.11 ai-tools
```

## Install Poetry

Install Poetry via system package manager:

```bash
sudo apt install python3-poetry
```

Verify installation:

```bash
poetry --version
# Expected: Poetry (version 1.8.2)
```

## Configure Project

Navigate to project directory:

```bash
cd ~/workspace/ai-experiments
```

Set Python version for project:

```bash
echo "ai-tools" > .python-version
```

Configure Poetry to use existing virtualenv:

```bash
poetry env use ~/.pyenv/versions/3.12.11/envs/ai-tools/bin/python
```

Configure PyTorch repository:

```bash
poetry config repositories.pytorch_cuda_cu129.url https://download.pytorch.org/whl/cu129
```

Configure Poetry settings:

```bash
poetry config virtualenvs.in-project true
poetry config installer.parallel true
```

## Install AI Dependencies

Install project dependencies:

```bash
poetry install
```

Install PyTorch with CUDA support:

```bash
poetry add torch torchvision torchaudio --source pytorch_cuda_cu129
```

## Verify Installation

Check Python version:

```bash
python --version
# Expected: Python 3.12.11
```

Check active environment:

```bash
pyenv version
# Expected: ai-tools (set by /home/alejandro/workspace/ai-experiments/.python-version)
```

Check Poetry environment:

```bash
poetry env info
```

Verify CUDA availability:

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Common Commands

### pyenv Commands

```bash
pyenv versions                # List installed Python versions
pyenv install --list          # List available Python versions
pyenv virtualenv-delete ai-tools  # Delete virtual environment
pyenv update                  # Update pyenv itself
```

### Poetry Commands

```bash
poetry show                   # List installed packages
poetry add <package>          # Add dependency
poetry remove <package>       # Remove dependency
poetry update                 # Update all dependencies
poetry shell                  # Activate environment
poetry run python script.py   # Run script in environment
```

### Environment Activation

Automatic activation (via .python-version file):
```bash
cd ~/workspace/ai-experiments
# Environment activates automatically
```

Manual activation:
```bash
pyenv activate ai-tools
```

Deactivation:
```bash
pyenv deactivate
```

## Troubleshooting

### pyenv: command not found

Ensure shell configuration is correct:
```bash
echo $PYENV_ROOT
# Should show: /home/alejandro/.pyenv
```

### Poetry using wrong Python

Reset Poetry environment:
```bash
poetry env remove python
poetry env use ~/.pyenv/versions/3.12.11/envs/ai-tools/bin/python
```

### Python compilation fails

Install missing dependencies:
```bash
sudo apt install -y libncursesw5-dev
pyenv doctor  # Check for issues
```

### CUDA not detected

Verify PyTorch installation:
```bash
poetry show torch
# Check version includes +cu129
```

---

*Last Updated: 2025-09-24*