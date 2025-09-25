#!/usr/bin/env python3
"""
PyTorch GPU environment validation utility.
Verifies PyTorch GPU setup and AI package dependencies.
"""

import argparse
import json
import re
import subprocess
import sys
from typing import Dict, Any

# Status indicators
STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_ERROR = "ERROR"

# Exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_INVALID_USAGE = 2

class PyTorchValidationError(Exception):
    """Raised when PyTorch validation fails."""
    pass

def check_pytorch_gpu_setup() -> Dict[str, Any]:
    """Check PyTorch GPU configuration and CUDA compatibility.

    Returns:
        Dictionary containing validation results:
        - success: Boolean indicating overall success
        - pytorch_version: PyTorch version string
        - cuda_available: Boolean indicating CUDA availability
        - pytorch_cuda_version: PyTorch CUDA build version
        - system_cuda_version: System CUDA version from nvcc
        - gpu_detected: Boolean indicating GPU detection
        - gpu_name: GPU device name
        - cudnn_available: Boolean indicating cuDNN availability

    Raises:
        PyTorchValidationError: If critical validation steps fail
    """
    print("PyTorch GPU environment check:")
    print("Verifying PyTorch GPU, CUDA, and cuDNN configuration")
    print()

    result = {
        "success": False,
        "pytorch_version": None,
        "cuda_available": False,
        "pytorch_cuda_version": None,
        "system_cuda_version": None,
        "gpu_detected": False,
        "gpu_name": None,
        "cudnn_available": False
    }

    # Import PyTorch
    try:
        import torch
        result["pytorch_version"] = torch.__version__
        print(f"PyTorch version: {STATUS_OK} ({torch.__version__})")
    except ImportError:
        print(f"PyTorch installation: {STATUS_ERROR} (not installed)")
        return result

    print(f"Python version: {STATUS_OK} ({sys.version.split(' ')[0]})")
    print()

    # Check system CUDA version using nvcc
    try:
        nvcc_output = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if nvcc_output.returncode == 0:
            for line in nvcc_output.stdout.split('\n'):
                if 'release' in line:
                    match = re.search(r'release (\d+\.\d+)', line)
                    if match:
                        result["system_cuda_version"] = match.group(1)
                        print(f"System CUDA (nvcc): {STATUS_OK} ({result['system_cuda_version']})")
                        break
    except FileNotFoundError:
        print(f"System CUDA (nvcc): {STATUS_WARN} (not found in PATH)")

    # Check PyTorch CUDA version
    if hasattr(torch.version, 'cuda'):
        result["pytorch_cuda_version"] = torch.version.cuda
        print(f"PyTorch CUDA build: {STATUS_OK} ({torch.version.cuda})")

    # Check if CUDA is available to PyTorch
    result["cuda_available"] = torch.cuda.is_available()
    cuda_status = STATUS_OK if result["cuda_available"] else STATUS_ERROR
    print(f"CUDA availability: {cuda_status} ({result['cuda_available']})")

    if not result["cuda_available"]:
        print(f"GPU configuration: {STATUS_ERROR} (PyTorch cannot access CUDA)")
        print("  Solution: Verify NVIDIA drivers and PyTorch CUDA build")
        print("  Common causes: driver issues, CPU-only PyTorch, version mismatch")
        return result

    # Check for version mismatch
    if result["system_cuda_version"] and result["pytorch_cuda_version"]:
        sys_major = result["system_cuda_version"].split('.')[0]
        pytorch_cuda_major = result["pytorch_cuda_version"].split('.')[0] if result["pytorch_cuda_version"] else "unknown"

        if sys_major != pytorch_cuda_major:
            print(f"CUDA version compatibility: {STATUS_WARN} (version mismatch)")
            print(f"  System: {result['system_cuda_version']}, PyTorch: {result['pytorch_cuda_version']}")
            print("  Solution: Reinstall PyTorch for system CUDA version")

    # GPU detection
    device_count = torch.cuda.device_count()
    result["gpu_detected"] = device_count > 0
    gpu_status = STATUS_OK if device_count > 0 else STATUS_ERROR
    print(f"GPU devices: {gpu_status} ({device_count} detected)")

    if device_count > 0:
        result["gpu_name"] = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  Primary GPU: {result['gpu_name']}")
        print(f"  CUDA capability: {capability[0]}.{capability[1]}")
        print(f"  Memory: {total_memory:.1f}GB")

    # Check cuDNN
    result["cudnn_available"] = torch.backends.cudnn.is_available()
    cudnn_status = STATUS_OK if result["cudnn_available"] else STATUS_WARN
    if result["cudnn_available"]:
        cudnn_version = torch.backends.cudnn.version()
        print(f"cuDNN: {cudnn_status} (version {cudnn_version})")
    else:
        print(f"cuDNN: {cudnn_status} (not available)")

    # Test GPU operation
    print()
    try:
        tensor = torch.randn(10, 10).cuda()
        result_tensor = tensor * 2 + 1
        print(f"GPU tensor test: {STATUS_OK} (operations functional)")
        result["success"] = True
    except Exception as e:
        print(f"GPU tensor test: {STATUS_ERROR} ({str(e)})")

    return result

def check_key_packages() -> Dict[str, bool]:
    """Check key AI package availability.

    Returns:
        Dictionary mapping package names to installation status.
    """
    print()
    print("AI package availability:")
    packages = {
        "transformers": False,
        "accelerate": False,
        "datasets": False,
        "huggingface_hub": False,
        "bitsandbytes": False,
        "peft": False
    }

    for package in packages:
        try:
            __import__(package)
            packages[package] = True
            print(f"  {package}: {STATUS_OK}")
        except ImportError:
            print(f"  {package}: {STATUS_ERROR} (not installed)")

    return packages

def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="PyTorch GPU environment validation utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_py_deps_install.py
  python check_py_deps_install.py --json
        """
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )

    return parser

def main() -> int:
    """Main function to run all validation checks.

    Returns:
        Exit code: 0 for success, 1 for validation failures.
    """
    parser = create_parser()
    args = parser.parse_args()

    print("PyTorch GPU validation utility")
    print()
    try:
        # Run PyTorch GPU checks
        gpu_result = check_pytorch_gpu_setup()

        # Check key packages
        packages = check_key_packages()

        # Summary
        print()
        print("Validation summary:")

        gpu_summary_status = STATUS_OK if gpu_result["success"] else STATUS_ERROR
        print(f"PyTorch GPU setup: {gpu_summary_status}")

        if gpu_result["system_cuda_version"]:
            print(f"System CUDA: {gpu_result['system_cuda_version']}")

        if gpu_result["pytorch_cuda_version"]:
            print(f"PyTorch CUDA: {gpu_result['pytorch_cuda_version']}")

        # JSON output if requested
        if args.json:
            result = {
                "gpu": gpu_result,
                "packages": packages
            }
            print()
            print("JSON output:")
            print(json.dumps(result, indent=2))

        return EXIT_SUCCESS if gpu_result["success"] else EXIT_FAILURE

    except PyTorchValidationError as e:
        print(f"{STATUS_ERROR}: {e}", file=sys.stderr)
        return EXIT_FAILURE
    except Exception as e:
        print(f"{STATUS_ERROR}: Unexpected error: {e}", file=sys.stderr)
        return EXIT_FAILURE

if __name__ == "__main__":
    sys.exit(main())