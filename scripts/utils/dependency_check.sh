#!/bin/bash

# dependency_check.sh
# Checks and reports on AI engineering dependencies and repository-defined optimizations
# Designed for Ubuntu 24.04 with AMD Ryzen 9950X + RTX 5090

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Disable colors if NO_COLOR is set
[[ -n "${NO_COLOR:-}" ]] && RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''

# Project root for uv environment checks (script lives in scripts/utils/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Print helper functions
print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
print_skip()  { printf "${BLUE}[SKIP]${NC} %s\n" "$*"; }

echo "=== AI Engineering Dependency & Optimization Report ==="
echo

# --- 1. System Information ---
echo "System information:"
echo "OS: $(lsb_release -ds 2>/dev/null || cat /etc/*release 2>/dev/null | head -n 1)"
echo "Kernel: $(uname -r)"
echo "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "RAM: $(free -h | grep '^Mem' | awk '{print $2}')"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo

# --- 2. Repository Optimizations Status ---
echo "Repository optimizations status:"
echo "Checking optimizations from docs/optimizations/os/os-optimizations.md"
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
        print_ok "$name: $actual"
        ((PASS_COUNT++))
    else
        print_error "$name: expected $expected, got $actual"
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

# Huge Pages (1GB pages for large model loading)
# Note: 1GB hugepages are configured via hugepages-1048576kB, not vm.nr_hugepages (which is 2MB pages)
HUGEPAGES_1GB=$(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages 2>/dev/null || echo "0")
check_status "1GB Huge Pages" "80" "$HUGEPAGES_1GB"

# sysctl Settings
echo
echo "sysctl Configuration:"

# 1GB hugepages (configured via sysfs, not sysctl)
SYSCTL_HUGEPAGES_1GB=$(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages 2>/dev/null || echo "0")
check_status "  1GB hugepages (sysfs)" "80" "$SYSCTL_HUGEPAGES_1GB"

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
        print_ok "Docker: running"
        ((PASS_COUNT++))
    else
        print_error "Docker: not running"
        ((FAIL_COUNT++))
    fi

    # NVIDIA container toolkit - required for GPU containers (llama-gpu, vllm-gpu).
    # Detected either as a registered "nvidia" Docker runtime (legacy) or via the
    # nvidia-ctk binary / a CDI spec (modern nvidia-container-toolkit uses CDI and
    # does not register a runtime in `docker info`).
    if docker info --format '{{.Runtimes}}' 2>/dev/null | grep -q "nvidia" \
        || command -v nvidia-ctk &> /dev/null \
        || ls /etc/cdi/nvidia*.yaml &> /dev/null; then
        print_ok "NVIDIA container toolkit: available"
        ((PASS_COUNT++))
    else
        print_error "NVIDIA container toolkit: not found"
        echo "  Solution: install nvidia-container-toolkit (runtime or CDI mode)"
        ((FAIL_COUNT++))
    fi

    # Service health (informational) - the four documented services. Down is not
    # a failure here; services may be intentionally stopped.
    echo "Service status:"
    # Distinguish a compose failure from "nothing is running": both yield an empty result
    # once stderr is discarded, and reporting a broken invocation as "no services running"
    # hides the actual problem. This check is informational, so a failure warns rather than
    # incrementing FAIL_COUNT.
    if ! SERVICES=$(docker compose ps --format '{{.Name}}\t{{.State}}' 2>&1); then
        print_warn "  cannot list compose services: $SERVICES"
    elif [ -n "$SERVICES" ]; then
        echo "$SERVICES" | while IFS=$'\t' read -r name state; do
            [ -z "$name" ] && continue
            if [ "$state" = "running" ]; then
                print_ok "  $name: $state"
            else
                print_info "  $name: ${state:-stopped}"
            fi
        done
    else
        print_skip "  no compose services running"
    fi
else
    print_error "Docker: not installed"
    ((FAIL_COUNT++))
fi

echo

# --- 3. NVIDIA GPU & CUDA Information ---
echo "NVIDIA GPU and CUDA information:"

if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)

    echo "Driver: $DRIVER_VERSION"
    echo "GPU: $GPU_NAME ($GPU_MEMORY)"

    # Check for running GPU processes
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    echo "GPU Processes: $GPU_PROCS running"

    # Active clock throttle reasons - surfaces thermal/power/reliability slowdowns.
    # "Not Active" (or absence) means the GPU is running unthrottled.
    THROTTLE=$(nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader 2>/dev/null | head -1)
    if [ -z "$THROTTLE" ] || echo "$THROTTLE" | grep -qiE "0x0000000000000000|Not Active"; then
        print_ok "  GPU throttling: none active"
        ((PASS_COUNT++))
    else
        print_warn "  GPU throttling: active ($THROTTLE)"
        ((FAIL_COUNT++))
    fi

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
        print_warn "  GPU Clocks: $GPU_CLOCKS MHz (locked - degrades performance)"
        ((FAIL_COUNT++))
    else
        print_ok "  GPU clocks: not locked (optimal)"
        ((PASS_COUNT++))
    fi

    # Memory Clock Settings (should NOT be locked for RTX 5090 performance)
    MEM_CLOCKS=$(nvidia-smi --query-gpu=clocks.applications.memory --format=csv,noheader 2>/dev/null)
    if [ -n "$MEM_CLOCKS" ] && [ "$MEM_CLOCKS" != "[N/A]" ]; then
        print_warn "  Memory Clocks: $MEM_CLOCKS MHz (locked - degrades performance)"
        ((FAIL_COUNT++))
    else
        print_ok "  Memory clocks: not locked (optimal)"
        ((PASS_COUNT++))
    fi

    # PCIe Generation
    PCIE_GEN_MAX=$(nvidia-smi -q 2>/dev/null | grep -A 5 "GPU Link Info" | grep "Max" | head -1 | awk -F: '{print $2}' | xargs)
    PCIE_GEN_CURRENT=$(nvidia-smi -q 2>/dev/null | grep -A 5 "GPU Link Info" | grep "Current" | head -1 | awk -F: '{print $2}' | xargs)
    if [ -n "$PCIE_GEN_CURRENT" ] && [ -n "$PCIE_GEN_MAX" ]; then
        print_info "  PCIe Generation: Current=$PCIE_GEN_CURRENT, Max=$PCIE_GEN_MAX"
    else
        print_warn "  PCIe Generation: unable to detect"
    fi

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
            print_ok "  GPU IRQ affinity: cores 28-31"
            ((PASS_COUNT++))
        else
            print_error "  GPU IRQ affinity: not optimized"
            ((FAIL_COUNT++))
        fi
    else
        print_info "  GPU IRQ affinity: no IRQs found"
    fi
else
    print_error "NVIDIA driver: not found"
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

# Kernel Module Parameters - Config file verification
# Note: NVIDIA proprietary driver doesn't expose params via /sys/module/nvidia/parameters/
# Verify config file content instead (params applied at module load time)
echo
echo "NVIDIA Kernel Module Parameters:"

NVIDIA_CONF="/etc/modprobe.d/nvidia-optimizations.conf"

if [ -f "$NVIDIA_CONF" ]; then
    print_ok "  Config file: nvidia-optimizations.conf present"
    ((PASS_COUNT++))

    # Check for Resizable BAR (the only required parameter)
    if grep -q "NVreg_EnableResizableBar=1" "$NVIDIA_CONF" 2>/dev/null; then
        print_ok "  NVreg_EnableResizableBar=1"
        ((PASS_COUNT++))
    else
        print_error "  NVreg_EnableResizableBar=1: missing from config"
        ((FAIL_COUNT++))
    fi
else
    print_error "  Config file: nvidia-optimizations.conf missing"
    ((FAIL_COUNT++))
    print_info "  See: docs/optimizations/os/os-optimizations.md Section 3"
fi

echo

# --- 4. Python & PyTorch Environment ---
echo "Python and PyTorch environment:"

# Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "Python: $PYTHON_VERSION"

    # Check for uv
    if command -v uv &> /dev/null; then
        echo "uv: $(uv --version 2>&1 | awk '{print $2}')"

        # Check project environment
        if [ -d "$PROJECT_ROOT/.venv" ]; then
            echo "Project Environment: .venv present"

            # Check PyTorch in the project environment
            PYTORCH_CHECK=$(uv run --project "$PROJECT_ROOT" --no-sync python -c "
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
            echo "Project Environment: .venv not found"
        fi
    else
        echo "uv: Not installed"
    fi

    # Check key AI packages
    # Single interpreter start (imports all packages) instead of one `uv run`
    # per package, which spawns a fresh interpreter each time. Python emits a
    # "status<TAB>name" line per package; the shell colorizes via print helpers.
    echo
    echo "Key AI Packages:"
    PKG_STATUS=$(uv run --project "$PROJECT_ROOT" --no-sync python - <<'PYEOF' 2>/dev/null
import importlib
# Map display name -> import name (differ for hyphenated distributions)
packages = {
    "transformers": "transformers",
    "accelerate": "accelerate",
    "datasets": "datasets",
    "huggingface-hub": "huggingface_hub",
    "bitsandbytes": "bitsandbytes",
    "deepspeed": "deepspeed",
}
for name, module in packages.items():
    try:
        importlib.import_module(module)
        print(f"OK\t{name}")
    except Exception:
        print(f"MISSING\t{name}")
PYEOF
)
    if [ -n "$PKG_STATUS" ]; then
        while IFS=$'\t' read -r status pkg; do
            if [ "$status" = "OK" ]; then
                print_ok "  $pkg"
            else
                print_error "  $pkg: not found"
            fi
        done <<< "$PKG_STATUS"
    else
        print_error "  package check: project environment not available"
    fi
else
    print_error "Python 3: not found"
fi

echo

# --- 5. Disk Usage ---
echo "Disk usage:"

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
echo "Additional system checks:"

# Transparent Huge Pages
THP_ENABLED=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null | grep -oP '\[\K[^\]]+')
echo "Transparent Huge Pages: $THP_ENABLED"

# NUMA nodes
NUMA_NODES=$(lscpu | grep "NUMA node(s):" | awk '{print $3}')
echo "NUMA Nodes: $NUMA_NODES"

# HF Hub fast-transfer toggle (hf-transfer is a project dependency; the env var
# activates it for accelerated model downloads).
if [ "${HF_HUB_ENABLE_HF_TRANSFER:-}" = "1" ]; then
    print_ok "HF_HUB_ENABLE_HF_TRANSFER: enabled"
else
    print_info "HF_HUB_ENABLE_HF_TRANSFER: not set (downloads use default transfer)"
fi

# Memory Allocator - Container Check (zen5_optimizer only)
echo "Memory Allocator (CPU Container):"
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^llama-cpu$"; then
    ZEN5_LOADED=$(docker exec llama-cpu cat /proc/1/maps 2>/dev/null | grep -c "libzen5_optimizer" || echo "0")
    if [ "$ZEN5_LOADED" -gt 0 ]; then
        print_ok "  zen5_optimizer: loaded (hugepage mmap interception)"
        ((PASS_COUNT++))
    else
        print_error "  zen5_optimizer: not loaded"
        ((FAIL_COUNT++))
    fi
else
    print_skip "  llama-cpu container not running"
fi

echo

# --- 7. Summary ---
echo "=== Optimization Summary ==="
if [ $FAIL_COUNT -eq 0 ]; then
    print_ok "All checks passed ($PASS_COUNT total)"
else
    echo "Results: $PASS_COUNT passed, $FAIL_COUNT failed"
    echo
    print_info "To fix failed optimizations, refer to:"
    echo "  docs/optimizations/os/os-optimizations.md (CPU/RAM and GPU)"
    echo "  /etc/modprobe.d/nvidia-optimizations.conf (Kernel modules)"
    echo
    print_info "Note: Some GPU optimizations require a system reboot to take effect."
fi

echo
echo "Report complete: $(date '+%Y-%m-%d %H:%M:%S')"