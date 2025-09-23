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

    echo
    echo "GPU Optimizations:"

    # Persistence Mode
    PERSISTENCE=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader 2>/dev/null)
    check_status "  Persistence Mode" "Enabled" "$PERSISTENCE"

    # Power Limit
    POWER_LIMIT=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | cut -d'.' -f1)
    check_status "  Power Limit" "600" "$POWER_LIMIT"

    # GPU Clock Settings (should NOT be locked for RTX 5090 performance)
    GPU_CLOCKS=$(nvidia-smi --query-gpu=clocks.applications.graphics --format=csv,noheader 2>/dev/null)
    if [ -n "$GPU_CLOCKS" ] && [ "$GPU_CLOCKS" != "[N/A]" ]; then
        echo "  GPU Clocks: $GPU_CLOCKS MHz (locked - WARNING: degrades performance)"
        ((FAIL_COUNT++))
    else
        echo "  ✓ GPU Clocks: Not locked (optimal)"
        ((PASS_COUNT++))
    fi

    # Memory Clock Settings (should NOT be locked for RTX 5090 performance)
    MEM_CLOCKS=$(nvidia-smi --query-gpu=clocks.applications.memory --format=csv,noheader 2>/dev/null)
    if [ -n "$MEM_CLOCKS" ] && [ "$MEM_CLOCKS" != "[N/A]" ]; then
        echo "  Memory Clocks: $MEM_CLOCKS MHz (locked - WARNING: degrades performance)"
        ((FAIL_COUNT++))
    else
        echo "  ✓ Memory Clocks: Not locked (optimal)"
        ((PASS_COUNT++))
    fi

    # Compute Mode (DEFAULT mode for optimal RTX 5090 performance)
    COMPUTE_MODE=$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader 2>/dev/null)
    check_status "  Compute Mode" "Default" "$COMPUTE_MODE"

    # PCIe Generation
    PCIE_GEN_MAX=$(nvidia-smi -q 2>/dev/null | grep -A 2 "GPU Link Info" | grep "Max" | head -1 | awk '{print $NF}')
    PCIE_GEN_CURRENT=$(nvidia-smi -q 2>/dev/null | grep -A 2 "GPU Link Info" | grep "Current" | head -1 | awk '{print $NF}')
    echo "  PCIe Generation: Current=$PCIE_GEN_CURRENT, Max=$PCIE_GEN_MAX"

    # GPU IRQ Affinity Check
    GPU_IRQS=$(cat /proc/interrupts | grep -E "nvidia|gpu" 2>/dev/null | awk '{print $1}' | tr -d ':')
    if [ -n "$GPU_IRQS" ]; then
        IRQ_CORRECT=0
        IRQ_TOTAL=0
        for irq in $GPU_IRQS; do
            ((IRQ_TOTAL++))
            AFFINITY=$(cat "/proc/irq/$irq/smp_affinity" 2>/dev/null || echo "N/A")
            if [ "$AFFINITY" = "f0000000" ]; then
                ((IRQ_CORRECT++))
            fi
        done
        if [ $IRQ_TOTAL -eq $IRQ_CORRECT ] && [ $IRQ_TOTAL -gt 0 ]; then
            echo "  ✓ GPU IRQ Affinity: Correct (cores 24-31)"
            ((PASS_COUNT++))
        else
            echo "  GPU IRQ Affinity: Not optimized"
            ((FAIL_COUNT++))
        fi
    else
        echo "  GPU IRQ Affinity: No IRQs found"
    fi
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

# CUDA Environment Variables
echo
echo "CUDA Environment:"
CUDA_ENV_COUNT=0
CUDA_ENV_CORRECT=0

# Check each expected CUDA environment variable
for var in CUDA_CACHE_DISABLE CUDA_CACHE_PATH CUDA_CACHE_MAXSIZE CUDA_DEVICE_ORDER CUDA_LAUNCH_BLOCKING CUDA_MODULE_LOADING; do
    ((CUDA_ENV_COUNT++))
    VAL=$(grep "^$var=" /etc/environment 2>/dev/null | cut -d'=' -f2)
    if [ -n "$VAL" ]; then
        ((CUDA_ENV_CORRECT++))
    fi
done

if [ $CUDA_ENV_CORRECT -eq $CUDA_ENV_COUNT ]; then
    echo "  ✓ System CUDA variables: Configured"
    ((PASS_COUNT++))
else
    echo "  System CUDA variables: $CUDA_ENV_CORRECT/$CUDA_ENV_COUNT configured"
    ((FAIL_COUNT++))
fi

# Kernel Module Parameters
if [ -f /etc/modprobe.d/nvidia-optimizations.conf ]; then
    echo "  ✓ Kernel modules: nvidia-optimizations.conf present"
    ((PASS_COUNT++))
else
    echo "  Kernel modules: nvidia-optimizations.conf missing"
    ((FAIL_COUNT++))
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

# GPU Kernel Parameters
echo "GPU Kernel Parameters:"
grep -q "pcie_aspm=off" /proc/cmdline && echo "  pcie_aspm=off: Present" || echo "  pcie_aspm=off: Missing"
grep -q "nvidia-drm.modeset=1" /proc/cmdline && echo "  nvidia-drm.modeset=1: Present" || echo "  nvidia-drm.modeset=1: Missing"
grep -q "rcutree.rcu_idle_gp_delay=1" /proc/cmdline && echo "  rcutree.rcu_idle_gp_delay=1: Present" || echo "  rcutree.rcu_idle_gp_delay=1: Missing"

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
echo "Total Checks: $PASS_COUNT passed, $FAIL_COUNT failed"

if [ $FAIL_COUNT -gt 0 ]; then
    echo
    echo "To fix failed optimizations, refer to:"
    echo "  docs/optimizations/os/os-optimizations.md (CPU/RAM and GPU)"
    echo "  /etc/modprobe.d/nvidia-optimizations.conf (Kernel modules)"
    echo
    echo "Note: Some GPU optimizations require a system reboot to take effect."
fi

echo
echo "Report complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==================================================="