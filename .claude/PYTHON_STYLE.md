# Python Style Guide

This guide defines Python coding standards for the AI experiments repository. All Python code must follow these conventions to maintain professional quality and Unix compatibility.

## Table of Contents

- [Core Philosophy](#core-philosophy)
- [Code Standards](#code-standards)
- [Output Formatting](#output-formatting)
- [Error Handling](#error-handling)
- [Documentation Standards](#documentation-standards)
- [CLI Tool Standards](#cli-tool-standards)
- [Project Integration](#project-integration)
- [Import Management](#import-management)
- [Examples](#examples)
- [Prohibited Practices](#prohibited-practices)
- [Quality Checklist](#quality-checklist)

## Core Philosophy

### Unix Principles Applied to Python

All Python code follows traditional Unix development principles:

- **Do one thing well** - Functions and classes have single, clear responsibilities
- **Composable** - Code works well with other tools and scripts
- **Silent on success** - Minimal output unless explicitly requested
- **Clear on failure** - Actionable error messages with proper exit codes
- **Text-based** - Output compatible with standard Unix tools (grep, awk, sort)
- **Professional** - No decorative elements, marketing language, or subjective terminology

### Development Standards

```python
# Good: Clear, single responsibility
def parse_model_config(config_path: str) -> Dict[str, Any]:
    """Parse model configuration file and return settings."""

# Bad: Multiple responsibilities, unclear purpose
def do_model_stuff(file, other_things=None):
    """Does various model operations and other stuff."""
```

## Code Standards

### PEP 8 Compliance with Unix Emphasis

Follow PEP 8 with these additional requirements:

#### Naming Conventions
```python
# Functions and variables: lowercase with underscores
def check_cuda_availability() -> bool:
    model_path = "/path/to/model"

# Classes: PascalCase
class ModelManager:
    pass

# Constants: UPPERCASE with underscores
DEFAULT_BATCH_SIZE = 512
MAX_RETRY_ATTEMPTS = 3
```

#### Function Design
```python
# Good: Single responsibility, clear return type
def validate_model_path(path: str) -> bool:
    """Validate that model file exists and is accessible."""
    return Path(path).is_file() and os.access(path, os.R_OK)

# Bad: Multiple responsibilities, unclear behavior
def check_stuff(path):
    # Does validation, logging, and other operations
    pass
```

#### Variable Scope
```python
# Good: Minimal scope, clear names
def process_batch(data: List[str]) -> List[str]:
    processed_items = []
    for item in data:
        cleaned_item = item.strip().lower()
        processed_items.append(cleaned_item)
    return processed_items

# Bad: Unclear scope, generic names
def process_batch(data):
    result = []
    for x in data:
        tmp = x.strip().lower()
        result.append(tmp)
    return result
```

## Output Formatting

### Status Indicators

Use text-based status matching bash script conventions:

```python
# Status indicators (no emojis)
STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_ERROR = "ERROR"
STATUS_INFO = "INFO"

# Good: Consistent status formatting
print(f"Model loading: {STATUS_OK} (model.safetensors)")
print(f"CUDA availability: {STATUS_WARN} (driver version mismatch)")
print(f"Memory allocation: {STATUS_ERROR} (insufficient VRAM)")

# Bad: Inconsistent, decorative formatting
print("✅ Model loaded successfully!")
print("⚠️  CUDA might have issues")
print("❌ Failed to allocate memory!!!")
```

### Parseable Output

Structure output for Unix tool compatibility:

```python
# Good: Parseable format
def report_model_status(models: List[Model]) -> None:
    """Report model status in parseable format."""
    for model in models:
        status = STATUS_OK if model.is_loaded else STATUS_ERROR
        print(f"{model.name}: {status} ({model.size_gb:.1f}GB)")

# Output:
# gpt-4: OK (15.3GB)
# llama-70b: ERROR (model file not found)

# Bad: Decorative, unparseable format
def report_model_status_bad(models):
    print("🚀 Model Status Report 🚀")
    print("=" * 50)
    for model in models:
        if model.is_loaded:
            print(f"✅ {model.name} is working perfectly!")
        else:
            print(f"❌ Oh no! {model.name} failed to load!")
```

### Progress Reporting

When progress reporting is necessary:

```python
# Good: Minimal, informative progress
def download_model(url: str, destination: str) -> None:
    """Download model with minimal progress reporting."""
    total_size = get_remote_file_size(url)
    downloaded = 0

    with open(destination, 'wb') as f:
        for chunk in download_chunks(url):
            f.write(chunk)
            downloaded += len(chunk)

            # Only report significant progress
            if downloaded % (total_size // 10) == 0:
                percent = (downloaded / total_size) * 100
                print(f"Download progress: {percent:.0f}%")

# Bad: Excessive, decorative progress
def download_model_bad(url, dest):
    print("🚀 Starting amazing download! 🚀")
    # Prints every chunk with emojis and animations
```

## Error Handling

### Error Message Format

Professional, actionable error messages:

```python
# Good: Clear, actionable errors
class ModelLoadError(Exception):
    """Raised when model cannot be loaded."""
    pass

def load_model(model_path: str) -> Model:
    """Load model from specified path."""
    if not Path(model_path).exists():
        raise ModelLoadError(
            f"Model file not found: {model_path}\n"
            f"  Solution: Verify file path is correct\n"
            f"  Command: ls -la {Path(model_path).parent}"
        )

    try:
        return Model.from_file(model_path)
    except Exception as e:
        raise ModelLoadError(
            f"Failed to load model: {model_path}\n"
            f"  Error: {e}\n"
            f"  Solution: Check file format and permissions"
        ) from e

# Bad: Unclear, unhelpful errors
def load_model_bad(path):
    if not os.path.exists(path):
        raise Exception("File not found!")

    try:
        return Model.from_file(path)
    except:
        raise Exception("Something went wrong!")
```

### Exit Codes

Follow Unix conventions:

```python
import sys

# Standard exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_INVALID_USAGE = 2

def main() -> int:
    """Main function with proper exit code handling."""
    try:
        result = process_arguments()
        if not result:
            return EXIT_FAILURE
        return EXIT_SUCCESS
    except ValueError as e:
        print(f"ERROR: Invalid input: {e}", file=sys.stderr)
        return EXIT_INVALID_USAGE
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_FAILURE

if __name__ == "__main__":
    sys.exit(main())
```

## Documentation Standards

### Docstring Format

Follow documentation style guide principles:

```python
# Good: Clear, technical docstring
def optimize_model_parameters(
    model_path: str,
    target_size: int,
    quantization: str = "int8"
) -> Dict[str, float]:
    """Optimize model parameters for target deployment size.

    Args:
        model_path: Absolute path to model file
        target_size: Target model size in MB
        quantization: Quantization method (int8, int4, fp16)

    Returns:
        Dictionary containing optimization metrics:
        - compression_ratio: Achieved compression ratio
        - accuracy_loss: Measured accuracy degradation
        - size_reduction: Size reduction in MB

    Raises:
        ModelLoadError: If model file cannot be loaded
        ValueError: If target_size is invalid
    """

# Bad: Marketing language, subjective terms
def optimize_model_parameters_bad(model, size, quant="awesome"):
    """
    🚀 Amazing model optimization that makes your model super fast! 🚀

    This incredible function will optimize your model in the most
    efficient way possible, giving you outstanding performance!
    """
```

### Inline Comments

Minimal, essential comments only:

```python
# Good: Essential technical context
def calculate_memory_requirements(model_size: int, batch_size: int) -> int:
    """Calculate VRAM requirements for model inference."""
    # Base model memory (parameters + activations)
    base_memory = model_size * 1.2  # 20% overhead for activations

    # Batch processing overhead scales linearly
    batch_memory = batch_size * model_size * 0.1

    return int(base_memory + batch_memory)

# Bad: Obvious or decorative comments
def calculate_memory_requirements_bad(model_size, batch_size):
    # This function calculates memory! So cool!
    memory = model_size * 1.2  # Magic number that works perfectly
    # Add more memory for batches (this is important!)
    memory += batch_size * model_size * 0.1
    return memory  # Return the result
```

## CLI Tool Standards

### Argument Parsing

Consistent argument handling across all tools:

```python
import argparse
from pathlib import Path

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Model performance benchmark tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --model model.gguf --batch-size 32
  python benchmark.py --model model.gguf --output results.json
        """
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model file"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size (default: %(default)s)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results (default: stdout)"
    )

    return parser
```

### Help Text Format

Professional, informative help text:

```python
# Good: Clear, professional help
parser.add_argument(
    "--quantization",
    choices=["int8", "int4", "fp16"],
    default="int8",
    help="Model quantization method (default: %(default)s)"
)

# Bad: Marketing language in help
parser.add_argument(
    "--quantization",
    help="Choose your amazing quantization method! 🚀"
)
```

## Project Integration

### Consistency with Bash Scripts

Python scripts should behave consistently with existing bash scripts:

```python
# Output format matching bash scripts
def report_status(component: str, status: str, details: str = "") -> None:
    """Report status in format matching bash scripts."""
    detail_text = f" ({details})" if details else ""
    print(f"{component}: {status}{detail_text}")

# Usage
report_status("Python environment", STATUS_OK, "3.12.11")
report_status("CUDA availability", STATUS_ERROR, "driver not found")

# Output matches bash script format:
# Python environment: OK (3.12.11)
# CUDA availability: ERROR (driver not found)
```

### Shared Configuration

Use consistent configuration patterns:

```python
# Good: Consistent with project structure
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / ".config"

def load_project_config() -> Dict[str, Any]:
    """Load project configuration following established patterns."""
    config_file = CONFIG_DIR / "ai-experiments.json"
    if not config_file.exists():
        return get_default_config()

    with open(config_file) as f:
        return json.load(f)
```

## Import Management

### Import Organization

Standard import order and practices:

```python
# Good: Standard import order
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import transformers

from .utils import validate_model_path
from .config import load_config

# Bad: Disorganized imports
from typing import *
import torch, transformers, json
import sys
from pathlib import Path
from .utils import *
```

### Dependency Handling

Check for dependencies gracefully:

```python
# Good: Graceful dependency handling
def check_torch_availability() -> bool:
    """Check if PyTorch is available with CUDA support."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def main() -> int:
    if not check_torch_availability():
        print("ERROR: PyTorch with CUDA support required", file=sys.stderr)
        print("  Install with: poetry add torch --source pytorch_cuda_cu129")
        return EXIT_FAILURE

    # Continue with torch-dependent code

# Bad: Unclear dependency failures
import torch  # Will crash if not installed
```

## Examples

### Complete Script Example

```python
#!/usr/bin/env python3
"""
Model validation utility for AI experiments project.
Validates model files and reports compatibility status.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_INVALID_USAGE = 2

# Status indicators
STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_ERROR = "ERROR"

class ModelValidationError(Exception):
    """Raised when model validation fails."""
    pass

def validate_model_file(model_path: Path) -> Dict[str, str]:
    """Validate model file and return status information.

    Args:
        model_path: Path to model file

    Returns:
        Dictionary with validation results:
        - status: OK, WARN, or ERROR
        - message: Human-readable status message
        - size_gb: Model size in gigabytes

    Raises:
        ModelValidationError: If validation cannot be performed
    """
    if not model_path.exists():
        raise ModelValidationError(f"Model file not found: {model_path}")

    if not model_path.is_file():
        raise ModelValidationError(f"Path is not a file: {model_path}")

    # Get file size
    size_bytes = model_path.stat().st_size
    size_gb = size_bytes / (1024**3)

    # Validate file format
    if not model_path.suffix in ['.gguf', '.safetensors', '.bin']:
        return {
            'status': STATUS_WARN,
            'message': f'Unknown file format: {model_path.suffix}',
            'size_gb': f'{size_gb:.1f}'
        }

    return {
        'status': STATUS_OK,
        'message': f'Valid {model_path.suffix[1:]} format',
        'size_gb': f'{size_gb:.1f}'
    }

def main() -> int:
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: validate_model.py <model_path>", file=sys.stderr)
        return EXIT_INVALID_USAGE

    model_path = Path(sys.argv[1])

    try:
        result = validate_model_file(model_path)
        status = result['status']
        message = result['message']
        size = result['size_gb']

        print(f"Model validation: {status} ({message}, {size}GB)")

        return EXIT_SUCCESS if status == STATUS_OK else EXIT_FAILURE

    except ModelValidationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_FAILURE
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        return EXIT_FAILURE

if __name__ == "__main__":
    sys.exit(main())
```

## Prohibited Practices

### Never Use

- ❌ Emojis or Unicode symbols in output or code
- ❌ Marketing language ("amazing", "awesome", "incredible")
- ❌ Exclamation marks in normal output
- ❌ Colored terminal output (incompatible with logs)
- ❌ Star imports (`from module import *`)
- ❌ Bare except clauses (`except:`)
- ❌ Global variables for program state
- ❌ Print statements for debugging (use logging)

### Text Alternatives

| Prohibited | Use Instead |
|-----------|-------------|
| `print("✅ Success!")` | `print(f"Status: {STATUS_OK}")` |
| `raise Exception("Failed!")` | `raise SpecificError("Clear message")` |
| `from utils import *` | `from utils import specific_function` |
| `except:` | `except SpecificException:` |

## Quality Checklist

Before committing Python code:

### Code Review
- [ ] Follows PEP 8 formatting standards
- [ ] Functions have single, clear responsibilities
- [ ] Variable names are descriptive and professional
- [ ] No global variables or state

### Output Review
- [ ] Uses text-based status indicators (OK, WARN, ERROR)
- [ ] Output is parseable by Unix tools
- [ ] No emojis or decorative elements
- [ ] Silent on success unless explicitly requested

### Error Handling Review
- [ ] Specific exception types used
- [ ] Error messages are clear and actionable
- [ ] Proper exit codes for CLI tools
- [ ] No bare except clauses

### Documentation Review
- [ ] Docstrings follow professional standards
- [ ] Type hints provided for functions
- [ ] Comments are minimal and essential only
- [ ] No marketing language or subjective terms

### Integration Review
- [ ] Consistent with existing bash script behavior
- [ ] Uses project-standard configuration patterns
- [ ] Import statements are organized properly
- [ ] Dependencies checked gracefully

### Final Review
- [ ] Code follows Unix philosophy principles
- [ ] No prohibited elements present
- [ ] Professional appearance maintained
- [ ] Integrates well with existing toolchain

---

*Last Updated: 2025-09-24*