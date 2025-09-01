# Installation

Setup and prerequisites for the AI Model Evaluation Framework.

## Prerequisites

The framework requires Python 3.8 or higher and basic command-line familiarity.

### System Requirements

- Python 3.8+
- Git
- 4GB+ RAM recommended
- Internet connection for package installation

### Optional Requirements

For GPU monitoring features:
- NVIDIA GPU with drivers
- nvidia-ml-py3 package

## Installation Steps

Clone the repository:
```bash
git clone <repository-url>
cd benchmark_tests
```

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Verify installation:
```bash
make test MODE=quick
```

## Configuration

No configuration required for framework testing. For model evaluation, see [Basic Usage](./basic-usage.md).

## Common Issues

**Import Errors**: Ensure you're running from the benchmark_tests directory.

**Missing Dependencies**: Run `pip install -r requirements.txt` to install all required packages.

**Permission Errors**: Use `pip install --user` if you encounter permission issues.

## References

- [Basic Usage](./basic-usage.md)
- [Troubleshooting](./troubleshooting.md)