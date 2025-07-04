import torch
import sys

def check_pytorch_gpu_setup():
    """
    Checks if PyTorch is correctly configured to use the GPU (RTX 5090)
    and its underlying CUDA and cuDNN libraries, without making system calls.
    """
    print("--- PyTorch GPU Environment Check ---")
    print("This script verifies PyTorch's ability to detect and utilize your GPU, CUDA, and cuDNN.")
    print("------------------------------------------------------")

    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version.split(' ')[0]}")
    print("-" * 50)

    # 1. Check if CUDA is available to PyTorch
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available to PyTorch: {cuda_available}")

    if not cuda_available:
        print("ERROR: PyTorch cannot find CUDA. This means PyTorch is not configured to use your GPU.")
        print("  Possible reasons:")
        print("  - NVIDIA drivers, CUDA Toolkit, or cuDNN might not be correctly installed or configured on your OS.")
        print("  - PyTorch might have been installed without CUDA support (ensure you used the correct `cu128` index URL).")
        print("  - Your `pyenv` environment might not be correctly activated when running this script.")
        print("  Please ensure all these prerequisites are met.")
        return False

    # If CUDA is available, proceed with detailed checks
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.device_count() == 0:
        print("WARNING: CUDA is available, but no CUDA devices (GPUs) were found by PyTorch.")
        print("  This could indicate a problem with GPU detection or driver issues.")
        return False

    # Print details for each detected GPU
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"    CUDA Capability: {torch.cuda.get_device_capability(i)}")
        # Query memory usage (allocated and cached)
        # Note: These values are dynamic and depend on current GPU usage
        print(f"    Memory Allocated: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
        print(f"    Memory Cached: {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB")
        print(f"    Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")

    # 2. Check cuDNN availability
    cudnn_available = torch.backends.cudnn.is_available()
    print(f"cuDNN available to PyTorch: {cudnn_available}")
    if cudnn_available:
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    else:
        print("WARNING: cuDNN is not available. While PyTorch can still use CUDA, some operations (especially deep learning) might be slower.")
        print("  Ensure cuDNN is installed and correctly linked with your CUDA Toolkit installation.")

    # 3. Perform a simple tensor operation on the GPU to confirm functionality
    print("-" * 50)
    print("Attempting a simple tensor operation on GPU...")
    try:
        # Create a random tensor and move it to the GPU
        tensor_on_gpu = torch.randn(10, 10).to("cuda")
        # Perform a basic operation
        result_tensor = tensor_on_gpu * 2 + 1

        print(f"Successfully created and operated on a tensor on GPU: {tensor_on_gpu.device}")
        print(f"Example tensor (first 5 elements): {result_tensor.flatten()[:5].tolist()}")
        print("GPU tensor operation successful.")
        return True
    except Exception as e:
        print(f"ERROR: Failed to perform a tensor operation on GPU. This indicates a deeper issue.")
        print(f"  Details: {e}")
        return False

def main():
    """Main function to run the GPU setup checks."""
    all_checks_passed = check_pytorch_gpu_setup()

    print("\n--- Summary ---")
    if all_checks_passed:
        print("All PyTorch GPU checks passed successfully!")
        print("Your PyTorch installation appears to be correctly utilizing your RTX 5090, CUDA, and cuDNN.")
        print("You are ready to run AI models!")
        sys.exit(0)
    else:
        print("One or more PyTorch GPU checks failed or had warnings.")
        print("Please review the output above for error messages and warnings to troubleshoot.")
        sys.exit(1)

if __name__ == "__main__":
    main()