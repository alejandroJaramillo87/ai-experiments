# Python Environment Setup Guide

This guide sets up a complete Python development environment with pyenv and Poetry. Execute each section step by step, not all at once.

## Table of Contents

1. [Update System Packages](#step-1-update-system-packages)
2. [Install Python Build Dependencies](#step-2-install-python-build-dependencies)
3. [Install pyenv](#step-3-install-pyenv)
4. [Install Python 3.13](#step-4-install-python-313)
5. [Install pipx](#step-5-install-pipx)
6. [Configure pipx PATH](#step-6-configure-pipx-path)
7. [Install Poetry](#step-7-install-poetry)
8. [Configure Poetry](#step-8-configure-poetry)
9. [Usage Instructions](#usage-after-installation)

## Step 1: Update System Packages

Update system package lists to ensure you have the latest package information:

```bash
sudo apt update
```

## Step 2: Install Python Build Dependencies

Install packages required to compile Python from source via pyenv:

```bash
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
libffi-dev liblzma-dev
```

### Package Explanations:

- **make**: Build automation tool
- **build-essential**: Essential compilation tools (gcc, g++, etc.)
- **libssl-dev**: SSL/TLS library development files
- **zlib1g-dev**: Compression library
- **libbz2-dev**: Bzip2 compression library
- **libreadline-dev**: Command line editing library
- **libsqlite3-dev**: SQLite database library
- **wget, curl**: Download utilities
- **llvm**: Compiler infrastructure
- **libncursesw5-dev**: Terminal handling library
- **xz-utils**: XZ compression utilities
- **tk-dev**: Tkinter GUI toolkit
- **libxml2-dev, libxmlsec1-dev**: XML processing libraries
- **libffi-dev**: Foreign function interface library
- **liblzma-dev**: LZMA compression library

## Step 3: Install pyenv

Install pyenv (Python version manager) which allows you to install and switch between multiple Python versions:

```bash
curl https://pyenv.run | bash
```

### Important Post-Installation Steps:

After running the installation command, you need to:

1. **Add pyenv to your shell PATH** by adding these lines to `~/.bashrc` or `~/.zshrc`:

```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

2. **Restart your terminal** or run:

```bash
source ~/.bashrc
```

## Step 4: Install Python 3.13

Compile and install Python 3.13 from source using pyenv:

```bash
pyenv install 3.13
```

**Note**: This process may take several minutes as it compiles Python from source.

**Optional**: Set Python 3.13 as the global default:

```bash
pyenv global 3.13
```

## Step 5: Install pipx

Install pipx, which installs Python applications in isolated environments to prevent conflicts:

```bash
sudo apt install pipx
```

## Step 6: Configure pipx PATH

Ensure pipx-installed applications are available in your shell PATH:

```bash
pipx ensurepath
```

> **Note**: You may need to restart your terminal after this step.

## Step 7: Install Poetry

Install Poetry using pipx to keep it isolated from your projects:

```bash
pipx install poetry
```

Poetry is a dependency management and packaging tool for Python that provides better dependency resolution and project isolation.

## Step 8: Configure Poetry

Configure Poetry to create virtual environments inside project directories:

```bash
poetry config virtualenvs.in-project true
```

This creates `.venv` folders inside your projects instead of in a global location, making it easier to manage and find your project environments.

## Usage After Installation

### For Project-Specific Environments

#### If you have an existing `pyproject.toml` file:

```bash
# Navigate to your project directory
cd /path/to/your/project

# Install dependencies from pyproject.toml
poetry install

# Activate the virtual environment
poetry shell
```

#### If starting a new project:

```bash
# Create a new pyproject.toml file
poetry init

# Add dependencies
poetry add <package>

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

#### To use a specific Python version in your project:

```bash
# Set Python 3.13 for this project
pyenv local 3.13

# Tell Poetry to use Python 3.13
poetry env use 3.13
```

### For Global Environment with Poetry Dependencies

1. **Create a global poetry project:**

```bash
mkdir ~/global-python-env
cd ~/global-python-env
poetry init       # Create pyproject.toml with your global deps
poetry install    # Install the dependencies
```

2. **Activate the global environment:**

```bash
cd ~/global-python-env
poetry shell      # This activates the global environment
```

## Useful Commands

### pyenv Commands

```bash
pyenv versions              # List installed Python versions
pyenv global <version>      # Set global Python version
pyenv local <version>       # Set local Python version for current directory
```

### Poetry Commands

```bash
poetry --version            # Check Poetry version
poetry env info             # Show virtual environment info
poetry env list             # List virtual environments
poetry show                 # Show installed packages
poetry add <package>        # Add a new dependency
poetry remove <package>     # Remove a dependency
poetry update               # Update dependencies
poetry shell                # Activate virtual environment
```

### Environment Management

```bash
# Exit Poetry environment
exit

# Deactivate any virtual environment
deactivate
```

## Best Practices

1. **Use project-specific environments** for all development work
2. **Pin dependency versions** in `pyproject.toml` for reproducible builds
3. **Use `poetry.lock`** to ensure consistent dependency resolution across environments
4. **Keep global environment minimal** - only install essential tools globally
5. **Use `pyenv local`** to set Python versions per project
6. **Commit `pyproject.toml` and `poetry.lock`** to version control

## Troubleshooting

### If pyenv command is not found:
- Ensure you've added pyenv to your PATH and restarted your terminal
- Check that the export commands are in your shell configuration file

### If Poetry environment activation fails:
- Try using `poetry shell` instead of `poetry env activate`
- Ensure Poetry is properly installed with `poetry --version`

### If Python compilation fails:
- Ensure all build dependencies are installed
- Check for specific error messages and install any missing libraries

---

**Note**: This setup provides a robust Python development environment suitable for AI engineering workloads with proper dependency isolation and version management.

*Last Updated: 2025-09-23*