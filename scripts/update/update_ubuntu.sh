#!/bin/bash

# update_ubuntu.sh
# Comprehensive Ubuntu update script for AI workstation
# Maintains system optimizations and cleans up unnecessary files

set -euo pipefail

echo "=== Ubuntu System Update ==="
echo

# Parse command line arguments
SECURITY_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --security)
            SECURITY_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --security  Only install security updates"
            echo "  --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Pre-update checks
echo "Pre-update checks:"

# Check available disk space
ROOT_USAGE=$(df / | awk 'NR==2 {print int($5)}')
ROOT_FREE=$(df -h / | awk 'NR==2 {print $4}')
if [ "$ROOT_USAGE" -gt 80 ]; then
    echo "WARN: Root partition ${ROOT_USAGE}% full (${ROOT_FREE} free)"
    echo "  Consider running: ./scripts/utils/storage_check.sh"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Update cancelled"
        exit 1
    fi
else
    echo "Disk space: OK (${ROOT_USAGE}% used, ${ROOT_FREE} free)"
fi

# Check kernel count for safety
KERNEL_COUNT=$(dpkg -l | grep -c "linux-image-[0-9]" || true)
if [ "$KERNEL_COUNT" -lt 2 ]; then
    echo "WARN: Only $KERNEL_COUNT kernel(s) installed (no fallback available)"
else
    echo "Kernels installed: OK ($KERNEL_COUNT available)"
fi

echo

# Main update process
if $SECURITY_ONLY; then
    echo "Installing security updates only:"
    sudo apt update
    sudo apt install -y unattended-upgrades
    sudo unattended-upgrade -d
else
    echo "Updating system packages:"

    # Standard updates
    echo "Updating package lists..."
    sudo apt update

    echo "Upgrading packages..."
    sudo apt -y upgrade

    echo "Performing distribution upgrade..."
    sudo apt -y dist-upgrade

    # AMD microcode updates (for Ryzen 9950X)
    echo "Updating AMD microcode..."
    sudo apt -y install amd64-microcode

    # Kernel headers and build tools
    echo "Installing kernel headers and build tools..."
    sudo apt -y install "linux-headers-$(uname -r)" dkms build-essential

    # Docker and NVIDIA runtime updates
    echo "Checking Docker and NVIDIA runtime..."
    if dpkg -l | grep nvidia-container-toolkit &> /dev/null; then
        sudo apt -y install nvidia-docker2 nvidia-container-toolkit
        sudo systemctl restart docker
        echo "NVIDIA container runtime: OK (updated)"
    fi
fi

echo

# Snap package updates
if command -v snap &> /dev/null; then
    echo "Updating snap packages:"
    sudo snap refresh

    # Remove old snap revisions (keeps only 2 versions per snap)
    echo "Removing old snap revisions..."
    LANG=C snap list --all | awk '/disabled/{print $1, $3}' | while read snapname revision; do
        sudo snap remove "$snapname" --revision="$revision" 2>/dev/null || true
    done
fi

# Flatpak updates (if installed)
if command -v flatpak &> /dev/null; then
    echo "Updating flatpak applications:"
    flatpak update -y
fi

echo

# Cleanup
echo "System cleanup:"

# Clean package cache
echo "Cleaning package cache..."
sudo apt -y autoclean

# Remove unnecessary packages and old kernels
echo "Removing unnecessary packages and old kernels..."
sudo apt autoremove --purge -y

# Clean journal logs (keep last 7 days)
JOURNAL_SIZE=$(journalctl --disk-usage 2>/dev/null | grep -oP '\d+\.\d+[KMGT]' || echo "0")
echo "Current journal size: $JOURNAL_SIZE"
if [ -n "$JOURNAL_SIZE" ]; then
    sudo journalctl --vacuum-time=7d
    NEW_SIZE=$(journalctl --disk-usage 2>/dev/null | grep -oP '\d+\.\d+[KMGT]' || echo "0")
    echo "Journal reduced to: $NEW_SIZE"
fi

echo

# Docker maintenance (if installed)
if command -v docker &> /dev/null && systemctl is-active --quiet docker; then
    echo "Docker maintenance:"

    # Update base images for AI containers
    echo "Updating Docker base images..."
    docker pull ubuntu:24.04 2>/dev/null || echo "  ubuntu:24.04 update failed"

    # Only try to pull CUDA image if NVIDIA runtime is available
    if docker info 2>/dev/null | grep -q nvidia; then
        docker pull nvidia/cuda:13.0-cudnn9-runtime-ubuntu24.04 2>/dev/null || echo "  CUDA image update failed"
    fi

    # Clean up dangling images
    DANGLING=$(docker images -f "dangling=true" -q | wc -l)
    if [ "$DANGLING" -gt 0 ]; then
        echo "Removing $DANGLING dangling Docker images..."
        docker image prune -f
    fi
fi

echo

# Verify system optimizations are still in place
echo "Verifying system optimizations:"

# Check CPU governor
CPU_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
if [ "$CPU_GOV" = "performance" ]; then
    echo "CPU governor: OK (performance mode)"
else
    echo "CPU governor: WARN ($CPU_GOV - expected performance)"
fi

# Check swap status
SWAP_STATUS=$(swapon --show 2>/dev/null | wc -l)
if [ "$SWAP_STATUS" -eq 0 ]; then
    echo "Swap: OK (disabled)"
else
    echo "Swap: WARN (enabled - should be disabled for AI workloads)"
fi

# Check huge pages
HUGEPAGES=$(grep "^HugePages_Total:" /proc/meminfo | awk '{print $2}')
if [ "$HUGEPAGES" = "46080" ]; then
    echo "Huge pages: OK (46080 pages, 90GB)"
elif [ "$HUGEPAGES" -gt 0 ]; then
    echo "Huge pages: WARN ($HUGEPAGES pages - expected 46080)"
else
    echo "Huge pages: WARN (not configured)"
fi

# Verify critical services
echo
echo "Checking critical services..."
for service in docker nvidia-persistenced; do
    if systemctl is-active --quiet "$service" 2>/dev/null; then
        echo "$service: OK (running)"
    else
        echo "$service: WARN (not running)"
    fi
done

echo

# Summary
echo "=== Update Summary ==="
echo "Kernel: $(uname -r)"
echo "NVIDIA Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'Not detected')"
echo "Docker: $(docker --version 2>/dev/null | awk '{print $3}' | sed 's/,$//' || echo 'Not installed')"
echo "Python: $(python3 --version 2>/dev/null | awk '{print $2}' || echo 'Not installed')"
echo "Free Space: $(df -h / | awk 'NR==2 {print $4}')"

# Check if reboot is required
if [ -f /var/run/reboot-required ]; then
    echo
    echo "Reboot required: YES"
    echo "  Packages requiring reboot:"
    if [ -f /var/run/reboot-required.pkgs ]; then
        cat /var/run/reboot-required.pkgs | sed 's/^/    - /'
    fi
else
    echo
    echo "Reboot required: NO"
fi

echo "Update completed: $(date '+%Y-%m-%d %H:%M:%S')"