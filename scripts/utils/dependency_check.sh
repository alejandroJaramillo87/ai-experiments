#!/bin/bash

# dependency_check.sh
# Checks and reports on AI engineering dependencies and repository-defined optimizations
# Designed for Ubuntu 24.04 with AMD Ryzen 9950X + RTX 5090

echo "==================================================="
echo " AI Engineering Dependency & Optimization Report"
echo "==================================================="
echo

# --- 1. System Information ---
echo "--- System Information ---"
echo "OS: $(lsb_release -ds 2>/dev/null || cat /etc/*release 2>/dev/null | head -n 1)"
echo "Kernel: $(uname -r)"
echo "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "RAM: $(free -h | grep '^Mem' | awk '{print $2}')"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo

# --- 2. Repository Optimizations Status ---
echo "--- Repository Optimizations Status ---"
echo "Checking optimizations from docs/optimizations/cpu-ram/os-optimizations.md"
echo

# Initialize counters
PASS_COUNT=0
FAIL_COUNT=0

# Function to check status
check_status() {
    local name="$1"
    local expected="$2"
    local actual="$3"

    if [ "$expected" = "$actual" ]; then
        echo "✓ $name: $actual"
        ((PASS_COUNT++))
    else
        echo "✗ $name: Expected '$expected', Got '$actual'"
        ((FAIL_COUNT++))
    fi
}

# Swap Status
SWAP_STATUS=$(swapon --show 2>/dev/null | wc -l)
if [ "$SWAP_STATUS" -eq 0 ]; then
    SWAP_VALUE="disabled"
else
    SWAP_VALUE="enabled"
fi
check_status "Swap" "disabled" "$SWAP_VALUE"

# CPU Governor
CPU_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
check_status "CPU Governor" "performance" "$CPU_GOV"

# Memory Locking
MEMLOCK=$(ulimit -l)
if [ "$MEMLOCK" = "unlimited" ]; then
    echo "✓ Memory Locking: unlimited"
    ((PASS_COUNT++))
else
    echo "✗ Memory Locking: Expected 'unlimited', Got '$MEMLOCK'"
    ((FAIL_COUNT++))
fi

# Huge Pages
HUGEPAGES_TOTAL=$(grep "^HugePages_Total:" /proc/meminfo | awk '{print $2}')
check_status "Huge Pages Total" "46080" "$HUGEPAGES_TOTAL"

# sysctl Settings
echo
echo "sysctl Configuration:"

# vm.nr_hugepages
SYSCTL_HUGEPAGES=$(sysctl -n vm.nr_hugepages 2>/dev/null)
check_status "  vm.nr_hugepages" "46080" "$SYSCTL_HUGEPAGES"

# vm.swappiness
SWAPPINESS=$(sysctl -n vm.swappiness 2>/dev/null)
check_status "  vm.swappiness" "0" "$SWAPPINESS"

# vm.zone_reclaim_mode
ZONE_RECLAIM=$(sysctl -n vm.zone_reclaim_mode 2>/dev/null)
check_status "  vm.zone_reclaim_mode" "0" "$ZONE_RECLAIM"

# vm.overcommit_memory
OVERCOMMIT=$(sysctl -n vm.overcommit_memory 2>/dev/null)
check_status "  vm.overcommit_memory" "1" "$OVERCOMMIT"

# vm.vfs_cache_pressure
VFS_CACHE=$(sysctl -n vm.vfs_cache_pressure 2>/dev/null)
check_status "  vm.vfs_cache_pressure" "50" "$VFS_CACHE"

# vm.dirty_ratio
DIRTY_RATIO=$(sysctl -n vm.dirty_ratio 2>/dev/null)
check_status "  vm.dirty_ratio" "5" "$DIRTY_RATIO"

# vm.dirty_background_ratio
DIRTY_BG_RATIO=$(sysctl -n vm.dirty_background_ratio 2>/dev/null)
check_status "  vm.dirty_background_ratio" "2" "$DIRTY_BG_RATIO"

# Docker Status
echo
if command -v docker &> /dev/null; then
    if systemctl is-active --quiet docker; then
        echo "✓ Docker: running"
        ((PASS_COUNT++))
    else
        echo "✗ Docker: not running"
        ((FAIL_COUNT++))
    fi
else
    echo "✗ Docker: not installed"
    ((FAIL_COUNT++))
fi

echo

# --- 3. NVIDIA GPU & CUDA Information ---
echo "--- NVIDIA GPU & CUDA Information ---"

if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)

    echo "Driver: $DRIVER_VERSION"
    echo "GPU: $GPU_NAME ($GPU_MEMORY)"

    # Check for running GPU processes
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    echo "GPU Processes: $GPU_PROCS running"
else
    echo "NVIDIA driver not found"
fi

# CUDA Toolkit version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep 'release' | awk '{print $NF}')
    echo "CUDA Toolkit: $CUDA_VERSION"
else
    echo "CUDA Toolkit: not found"
fi

# cuDNN Check
CUDNN_PACKAGE=$(apt list --installed 2>/dev/null | grep -E '^cudnn[0-9]*-cuda-[0-9]+' | head -1)
if [ -n "$CUDNN_PACKAGE" ]; then
    CUDNN_VERSION=$(echo "$CUDNN_PACKAGE" | awk '{print $2}' | cut -d'/' -f1)
    echo "cuDNN: $CUDNN_VERSION"
else
    echo "cuDNN: not detected via APT"
fi

echo

# --- 4. Python & PyTorch Environment ---
echo "--- Python & PyTorch Environment ---"

# Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "Python: $PYTHON_VERSION"

    # Check for Poetry
    if command -v poetry &> /dev/null; then
        echo "Poetry: $(poetry --version 2>&1 | awk '{print $3}')"

        # Check Poetry environment
        POETRY_ENV=$(poetry env info --path 2>/dev/null)
        if [ -n "$POETRY_ENV" ]; then
            echo "Poetry Environment: Active"

            # Check PyTorch if in Poetry environment
            PYTORCH_CHECK=$(poetry run python -c "
import sys
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    if torch.cuda.is_available():
        print(f'PyTorch CUDA: {torch.version.cuda}')
    else:
        print('PyTorch CUDA: Not available')
except ImportError:
    print('PyTorch: Not installed')
" 2>/dev/null)
            echo "$PYTORCH_CHECK"
        else
            echo "Poetry Environment: Not active"
        fi
    else
        echo "Poetry: Not installed"
    fi

    # Check key AI packages
    echo
    echo "Key AI Packages:"
    for pkg in transformers accelerate datasets huggingface-hub bitsandbytes deepspeed; do
        if poetry run python -c "import $pkg" 2>/dev/null; then
            echo "  ✓ $pkg"
        else
            echo "  ✗ $pkg"
        fi
    done
else
    echo "Python 3: Not found"
fi

echo

# --- 5. Disk Usage ---
echo "--- Disk Usage ---"

# Root partition
ROOT_USAGE=$(df -h / | awk 'NR==2 {print $3 " / " $2 " (" $5 " used)"}')
echo "Root (/): $ROOT_USAGE"

# AI data mount
if mountpoint -q /mnt/ai-data; then
    AI_USAGE=$(df -h /mnt/ai-data | awk 'NR==2 {print $3 " / " $2 " (" $5 " used)"}')
    echo "/mnt/ai-data: $AI_USAGE"
else
    echo "/mnt/ai-data: Not mounted"
fi

# Home directory
HOME_SIZE=$(du -sh "$HOME" 2>/dev/null | cut -f1)
echo "Home directory: $HOME_SIZE"

echo

# --- 6. Additional System Checks ---
echo "--- Additional System Checks ---"

# Transparent Huge Pages
THP_ENABLED=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null | grep -oP '\[\K[^\]]+')
echo "Transparent Huge Pages: $THP_ENABLED"

# NUMA nodes
NUMA_NODES=$(lscpu | grep "NUMA node(s):" | awk '{print $3}')
echo "NUMA Nodes: $NUMA_NODES"

# CPU isolation
if grep -q "isolcpus" /proc/cmdline; then
    ISOLCPUS=$(grep -oP 'isolcpus=\K[^ ]+' /proc/cmdline)
    echo "CPU Isolation: $ISOLCPUS"
else
    echo "CPU Isolation: Not configured"
fi

# BLAS Libraries
echo -n "BLAS Libraries: "
BLAS_FOUND=""
if ldconfig -p 2>/dev/null | grep -q "libopenblas"; then
    BLAS_FOUND="${BLAS_FOUND}OpenBLAS "
fi
if ldconfig -p 2>/dev/null | grep -q "libmkl_rt"; then
    BLAS_FOUND="${BLAS_FOUND}MKL "
fi
if [ -z "$BLAS_FOUND" ]; then
    echo "None detected"
else
    echo "$BLAS_FOUND"
fi

echo

# --- 7. Summary ---
echo "==================================================="
echo " Optimization Summary"
echo "==================================================="
echo "Repository Optimizations: $PASS_COUNT passed, $FAIL_COUNT failed"

if [ $FAIL_COUNT -gt 0 ]; then
    echo
    echo "To fix failed optimizations, refer to:"
    echo "  docs/optimizations/cpu-ram/os-optimizations.md"
fi

echo
echo "Report complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==================================================="