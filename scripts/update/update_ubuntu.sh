#!/bin/bash

# update_ubuntu.sh
# Comprehensive Ubuntu update script for AI workstation
# Maintains system optimizations and cleans up unnecessary files

set -euo pipefail

echo "==================================================="
echo " Ubuntu System Update"
echo "==================================================="
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
echo "--- Pre-Update Checks ---"

# Check available disk space
ROOT_USAGE=$(df / | awk 'NR==2 {print int($5)}')
ROOT_FREE=$(df -h / | awk 'NR==2 {print $4}')
if [ "$ROOT_USAGE" -gt 80 ]; then
    echo "⚠ WARNING: Root partition is ${ROOT_USAGE}% full (${ROOT_FREE} free)"
    echo "  Consider running: ./scripts/utils/storage_check.sh"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Update cancelled"
        exit 1
    fi
else
    echo "✓ Disk space: ${ROOT_USAGE}% used (${ROOT_FREE} free)"
fi

# Check kernel count for safety
KERNEL_COUNT=$(dpkg -l | grep -c "linux-image-[0-9]" || true)
if [ "$KERNEL_COUNT" -lt 2 ]; then
    echo "⚠ WARNING: Only $KERNEL_COUNT kernel(s) installed. No fallback kernel available."
else
    echo "✓ Kernels installed: $KERNEL_COUNT"
fi

echo

# Main update process
if $SECURITY_ONLY; then
    echo "--- Installing Security Updates Only ---"
    sudo apt update
    sudo apt install -y unattended-upgrades
    sudo unattended-upgrade -d
else
    echo "--- Updating System Packages ---"

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
        echo "✓ NVIDIA container runtime updated"
    fi
fi

echo

# Snap package updates
if command -v snap &> /dev/null; then
    echo "--- Updating Snap Packages ---"
    sudo snap refresh

    # Remove old snap revisions (keeps only 2 versions per snap)
    echo "Removing old snap revisions..."
    LANG=C snap list --all | awk '/disabled/{print $1, $3}' | while read snapname revision; do
        sudo snap remove "$snapname" --revision="$revision" 2>/dev/null || true
    done
fi

# Flatpak updates (if installed)
if command -v flatpak &> /dev/null; then
    echo "--- Updating Flatpak Applications ---"
    flatpak update -y
fi

echo

# Cleanup
echo "--- System Cleanup ---"

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
    echo "--- Docker Maintenance ---"

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
echo "--- Verifying System Optimizations ---"

# Check CPU governor
CPU_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
if [ "$CPU_GOV" = "performance" ]; then
    echo "✓ CPU Governor: performance"
else
    echo "⚠ CPU Governor: $CPU_GOV (expected: performance)"
fi

# Check swap status
SWAP_STATUS=$(swapon --show 2>/dev/null | wc -l)
if [ "$SWAP_STATUS" -eq 0 ]; then
    echo "✓ Swap: disabled"
else
    echo "⚠ Swap: enabled (should be disabled for AI workloads)"
fi

# Check huge pages
HUGEPAGES=$(grep "^HugePages_Total:" /proc/meminfo | awk '{print $2}')
if [ "$HUGEPAGES" = "46080" ]; then
    echo "✓ Huge Pages: 46080 (90GB)"
elif [ "$HUGEPAGES" -gt 0 ]; then
    echo "⚠ Huge Pages: $HUGEPAGES (expected: 46080)"
else
    echo "⚠ Huge Pages: not configured"
fi

# Verify critical services
echo
echo "Checking critical services..."
for service in docker nvidia-persistenced; do
    if systemctl is-active --quiet "$service" 2>/dev/null; then
        echo "✓ $service: running"
    else
        echo "⚠ $service: not running"
    fi
done

echo

# Summary
echo "==================================================="
echo " Update Summary"
echo "==================================================="
echo "Kernel: $(uname -r)"
echo "NVIDIA Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'Not detected')"
echo "Docker: $(docker --version 2>/dev/null | awk '{print $3}' | sed 's/,$//' || echo 'Not installed')"
echo "Python: $(python3 --version 2>/dev/null | awk '{print $2}' || echo 'Not installed')"
echo "Free Space: $(df -h / | awk 'NR==2 {print $4}')"

# Check if reboot is required
if [ -f /var/run/reboot-required ]; then
    echo
    echo "⚠ REBOOT REQUIRED"
    echo "  Packages requiring reboot:"
    if [ -f /var/run/reboot-required.pkgs ]; then
        cat /var/run/reboot-required.pkgs | sed 's/^/    - /'
    fi
else
    echo
    echo "✓ No reboot required"
fi

echo "==================================================="
echo "Update completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==================================================="