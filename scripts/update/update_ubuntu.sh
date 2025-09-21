#!/bin/bash
# Script for updating Ubuntu and related packages

set -euo pipefail

echo "Updating Ubuntu system packages..."

# Check kernel count for safety
KERNEL_COUNT=$(dpkg -l | grep -c "linux-image-[0-9]" || true)
if [ "$KERNEL_COUNT" -lt 2 ]; then
    echo "WARNING: Only $KERNEL_COUNT kernel(s) installed. Update will proceed but no fallback kernel available."
fi

sudo apt update && sudo apt -y upgrade
sudo apt -y dist-upgrade
sudo apt -y autoclean

echo "Installing latest Linux kernel headers (for DKMS compatibility)..."
sudo apt -y install "linux-headers-$(uname -r)" dkms build-essential

echo "Checking for Docker NVIDIA runtime updates..."
if dpkg -l | grep nvidia-container-toolkit &> /dev/null; then
    sudo apt -y install nvidia-docker2 nvidia-container-toolkit
    sudo systemctl restart docker
else
    echo "NVIDIA container runtime not installed."
fi

echo "Cleaning old kernels and unnecessary packages..."
sudo apt autoremove --purge -y

echo "Ubuntu system update completed."


