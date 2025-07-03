#!/bin/bash

# ai_deps_checker.sh
# This script checks and reports on various AI engineering related dependencies
# including Python environments, NVIDIA drivers/CUDA, and essential system tools.
# Designed for Ubuntu 24.04 with NVIDIA GPUs.

echo "==================================================="
echo " AI Engineering Dependency Report (Mid-2025)"
echo "==================================================="
echo

# --- 1. System Information ---
echo "--- System Information ---"
echo "Operating System: $(lsb_release -ds 2>/dev/null || cat /etc/*release 2>/dev/null | head -n 1)"
echo "Kernel Version: $(uname -r)"
echo "Architecture: $(uname -m)"
echo "Hostname: $(hostname)"
echo "Current User: $(whoami)"
echo "Current Date: $(date)"
echo
echo "--- CPU Information (AMD 9950) ---"
lscpu | grep -E 'Model name|Architecture|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|Vendor ID|Virtualization'
echo
echo "--- RAM Information (128GB DDR5 6000) ---"
free -h | grep -E 'Mem|Swap'
echo

# --- 2. NVIDIA GPU & CUDA/cuDNN Information ---
echo "--- NVIDIA GPU & CUDA/cuDNN Information (RTX 5090) ---"

if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA Driver Version:"
    nvidia-smi --query-gpu=driver_version --format=csv,noheader

    echo "GPU Details (Name, Total/Free/Used Memory):"
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv

    echo "Running GPU Processes (PID, Process Name, GPU, Used Memory):"
    # Check if any compute processes are running before listing them
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
        nvidia-smi --query-compute-apps=pid,process_name,gpu_name,used_gpu_memory --format=csv
    else
        echo "  No GPU processes currently running."
    fi
else
    echo "NVIDIA drivers or nvidia-smi not found. Please ensure NVIDIA drivers are installed and in your PATH."
fi

# Check for CUDA Toolkit installation (via nvcc)
# This is the authoritative source for the *installed* CUDA Toolkit version.
if command -v nvcc &> /dev/null; then
    echo "CUDA Toolkit (nvcc) Version: $(nvcc --version | grep 'release' | awk '{print $NF}')"
else
    echo "CUDA Toolkit (nvcc) not found in PATH. Ensure it's installed and configured."
fi

# Check for cuDNN (common path, might vary)
echo "cuDNN Check (common paths):"
if ls /usr/local/cuda/include/cudnn.h &> /dev/null; then
    echo "  Found cuDNN header: /usr/local/cuda/include/cudnn.h"
    echo "  (Version usually within the header file, e.g., CUDNN_MAJOR/MINOR/PATCH)"
else
    echo "  cuDNN header not found in /usr/local/cuda/include/. Please verify cuDNN installation."
fi
echo

# --- 3. Python Environment & Core AI Libraries ---
echo "--- Python Environment & Core AI Libraries ---"
if command -v python3 &> /dev/null; then
    echo "Default Python 3 Version: $(python3 --version)"
    echo "Active Python Environment (if any):"
    if [ -n "$VIRTUAL_ENV" ]; then
        echo "  Virtualenv: $(basename "$VIRTUAL_ENV")"
    else
        echo "  No specific virtual environment active (using system/default Python)."
    fi

    # echo "Installed Python Packages (in active environment):"
    # List common AI/ML packages and their versions
    # This will list ALL installed packages. For a more concise list,
    # you could filter this, but for a full dependency check, all are useful.
    # pip list | grep -E 'torch|tensorflow|numpy|scipy|pandas|scikit-learn|huggingface-hub|transformers|diffusers|accelerate|bitsandbytes|deepspeed|jax|flax|optax|onnxruntime|openvino|opencv-python|matplotlib|seaborn|jupyter|ipython|conda|mamba|gpustat|tqdm|tensorboard|wandb|mlflow' || echo "  No common AI/ML packages found or pip not available."

    # echo
    # echo "Pip Version: $(pip --version 2>/dev/null || echo 'pip not found')"
else
    echo "Python 3 not found. Please ensure Python 3 is installed."
fi
echo



# --- 5. Disk Usage for AI Assets ---
echo "--- Disk Usage for AI Assets ---"
echo "Root Partition (Ubuntu 24.04):"
df -h / | grep -E 'Filesystem|/'
echo "Samsung 990 Pro 2TB (assuming /mnt/data):"
if mountpoint -q /mnt/data; then
    df -h /mnt/data | grep -E 'Filesystem|/'
else
    echo "  /mnt/data not mounted. Please ensure your 2TB SSD is mounted there."
fi
echo "Home Directory Usage:"
du -sh "$HOME"
echo


# ---6. Advanced System Optimizations for AI ---
echo "--- Advanced System Optimizations for AI ---"

# Huge Pages
echo "Huge Pages (Transparent Huge Pages - THP):"
if [ -f /sys/kernel/mm/transparent_hugepage/enabled ]; then
    echo "  THP Enabled: $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
    echo "  THP Defrag: $(cat /sys/kernel/mm/transparent_hugepage/defrag)"
    echo "  Recommendation: For most AI workloads, consider disabling THP or setting to 'madvise'."
    echo "  To disable: echo 'never' | sudo tee /sys/kernel/mm/transparent_hugepage/enabled"
else
    echo "  Transparent Huge Pages status file not found."
fi
echo "Huge Pages (Explicit):"
grep HugePages_Total /proc/meminfo
grep Hugepagesize /proc/meminfo
echo "  Recommendation: For specific high-performance applications, explicit huge pages can be beneficial."
echo

# CPU Governor
echo "CPU Governor:"
if [ -d /sys/devices/system/cpu/cpu0/cpufreq ]; then
    echo "  Current Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
    echo "  Available Governors: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors 2>/dev/null)"
    echo "  Recommendation: For maximum performance, set to 'performance'."
    echo "  To set (requires sudo): echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
else
    echo "  CPU frequency scaling information not found."
fi
echo

# CPU Isolation (isolcpus)
echo "CPU Isolation (isolcpus):"
if grep -q "isolcpus" /proc/cmdline; then
    echo "  isolcpus detected in kernel boot parameters:"
    grep "isolcpus" /proc/cmdline
    echo "  Recommendation: Useful for dedicated CPU cores for specific AI processes, reducing noise."
else
    echo "  isolcpus not detected in kernel boot parameters."
fi
echo

# Memory Overcommit Policy
echo "Memory Overcommit Policy (vm.overcommit_memory):"
if [ -f /proc/sys/vm/overcommit_memory ]; then
    echo "  Current Policy: $(cat /proc/sys/vm/overcommit_memory)"
    echo "  0 = heuristic overcommit (default)"
    echo "  1 = always overcommit"
    echo "  2 = never overcommit (might cause OOM for large allocations)"
    echo "  Recommendation: For large AI models, '1' can prevent allocation failures but risks OOM."
    echo "  To set (requires sudo): echo '1' | sudo tee /proc/sys/vm/overcommit_memory"
else
    echo "  Memory overcommit policy file not found."
fi
echo

# NUMA-aware Placement
echo "NUMA-aware Placement:"
if command -v numactl &> /dev/null; then
    echo "  numactl installed. NUMA information:"
    numactl --hardware 2>/dev/null
    echo "  Recommendation: Use 'numactl' to bind processes to specific NUMA nodes for optimal memory access."
else
    echo "  numactl not found. Install 'numactl' for NUMA-aware process placement (sudo apt install numactl)."
fi
# Check if NUMA is detected by lscpu (already included in System Info, but good to re-emphasize)
if lscpu | grep -q "NUMA node(s):"; then
    echo "  NUMA nodes detected by lscpu."
else
    echo "  NUMA nodes not detected by lscpu (system might be UMA or NUMA disabled)."
fi
echo

# BLAS Libraries
echo "BLAS/LAPACK Libraries:"
echo "  Checking for common BLAS/LAPACK implementations:"
if ldconfig -p | grep -q "libopenblas"; then
    echo "  - OpenBLAS: Detected"
else
    echo "  - OpenBLAS: Not detected"
fi
if ldconfig -p | grep -q "libmkl_rt"; then
    echo "  - Intel MKL: Detected"
else
    echo "  - Intel MKL: Not detected"
fi
if ldconfig -p | grep -q "libblas"; then
    echo "  - Generic BLAS: Detected (might be reference or optimized)"
fi
echo "  Recommendation: Intel MKL or OpenBLAS are highly recommended for CPU-bound linear algebra."
echo "  Ensure your AI frameworks (NumPy, SciPy, PyTorch CPU) are linked against an optimized BLAS."
echo

# Low-Latency Kernel
echo "Low-Latency Kernel:"
KERNEL_VERSION=$(uname -r)
echo "  Current Kernel Version: $KERNEL_VERSION"
if [[ "$KERNEL_VERSION" == *"lowlatency"* || "$KERNEL_VERSION" == *"realtime"* ]]; then
    echo "  Detected a low-latency or real-time kernel variant."
    echo "  Recommendation: Good for applications requiring minimal latency, but might not be strictly necessary for all AI training."
else
    echo "  Standard kernel detected. Consider 'linux-lowlatency' package for reduced latency if needed."
fi
echo

# echo "==================================================="
# echo " Report Complete"
# echo "==================================================="
