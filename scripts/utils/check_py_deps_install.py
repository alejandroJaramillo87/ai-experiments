#!/usr/bin/env python3
import sys
import subprocess
import json

def check_pytorch_gpu_setup():
    """
    Checks if PyTorch is correctly configured to use the GPU
    and detects CUDA version information for compatibility checks.
    """
    print("--- PyTorch GPU Environment Check ---")
    print("This script verifies PyTorch's ability to detect and utilize your GPU, CUDA, and cuDNN.")
    print("------------------------------------------------------")

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
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("ERROR: PyTorch is not installed")
        return result

    print(f"Python version: {sys.version.split(' ')[0]}")
    print("-" * 50)

    # Check system CUDA version using nvcc
    try:
        nvcc_output = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if nvcc_output.returncode == 0:
            for line in nvcc_output.stdout.split('\n'):
                if 'release' in line:
                    import re
                    match = re.search(r'release (\d+\.\d+)', line)
                    if match:
                        result["system_cuda_version"] = match.group(1)
                        print(f"System CUDA version (nvcc): {result['system_cuda_version']}")
                        break
    except FileNotFoundError:
        print("WARNING: nvcc not found in PATH")

    # Check PyTorch CUDA version
    if hasattr(torch.version, 'cuda'):
        result["pytorch_cuda_version"] = torch.version.cuda
        print(f"PyTorch built with CUDA: {torch.version.cuda}")

    # Check if CUDA is available to PyTorch
    result["cuda_available"] = torch.cuda.is_available()
    print(f"CUDA available to PyTorch: {result['cuda_available']}")

    if not result["cuda_available"]:
        print("ERROR: PyTorch cannot find CUDA. This means PyTorch is not configured to use your GPU.")
        print("  Possible reasons:")
        print("  - NVIDIA drivers might not be correctly installed")
        print("  - PyTorch might have been installed without CUDA support")
        print("  - CUDA version mismatch between PyTorch and system")
        return result

    # Check for version mismatch
    if result["system_cuda_version"] and result["pytorch_cuda_version"]:
        sys_major = result["system_cuda_version"].split('.')[0]
        pytorch_cuda_major = result["pytorch_cuda_version"].split('.')[0] if result["pytorch_cuda_version"] else "unknown"

        if sys_major != pytorch_cuda_major:
            print(f"WARNING: CUDA version mismatch!")
            print(f"  System CUDA: {result['system_cuda_version']}")
            print(f"  PyTorch CUDA: {result['pytorch_cuda_version']}")
            print("  This may cause compatibility issues")

    # GPU detection
    device_count = torch.cuda.device_count()
    result["gpu_detected"] = device_count > 0
    print(f"CUDA device count: {device_count}")

    if device_count > 0:
        result["gpu_name"] = torch.cuda.get_device_name(0)
        print(f"  Device 0: {result['gpu_name']}")
        capability = torch.cuda.get_device_capability(0)
        print(f"    CUDA Capability: {capability}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"    Total Memory: {total_memory:.2f} GB")

    # Check cuDNN
    result["cudnn_available"] = torch.backends.cudnn.is_available()
    print(f"cuDNN available: {result['cudnn_available']}")
    if result["cudnn_available"]:
        print(f"cuDNN version: {torch.backends.cudnn.version()}")

    # Test GPU operation
    print("-" * 50)
    print("Testing GPU tensor operation...")
    try:
        tensor = torch.randn(10, 10).cuda()
        result_tensor = tensor * 2 + 1
        print("GPU tensor operation: SUCCESS")
        result["success"] = True
    except Exception as e:
        print(f"GPU tensor operation: FAILED - {e}")

    return result

def check_key_packages():
    """Check if key AI packages are installed and working"""
    print("\n--- Checking Key AI Packages ---")
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
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} not installed")

    return packages

def main():
    """Main function to run all checks."""
    # Run PyTorch GPU checks
    gpu_result = check_pytorch_gpu_setup()

    # Check key packages
    packages = check_key_packages()

    # Summary
    print("\n--- Summary ---")

    if gpu_result["success"]:
        print("✓ PyTorch GPU setup is working correctly")
    else:
        print("✗ PyTorch GPU setup has issues")

    if gpu_result["system_cuda_version"]:
        print(f"System CUDA: {gpu_result['system_cuda_version']}")

    if gpu_result["pytorch_cuda_version"]:
        print(f"PyTorch CUDA: {gpu_result['pytorch_cuda_version']}")

    # Return result as JSON for script consumption
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        result = {
            "gpu": gpu_result,
            "packages": packages
        }
        print("\n--- JSON Output ---")
        print(json.dumps(result, indent=2))

    sys.exit(0 if gpu_result["success"] else 1)

if __name__ == "__main__":
    main()