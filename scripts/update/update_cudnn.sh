#!/bin/bash

# update_cudnn_simple.sh
# This script automatically checks for and applies updates for the
# cuDNN package matching your installed CUDA version.
#
# IMPORTANT:
# - This script does NOT reboot your system automatically.
# - It assumes the NVIDIA CUDA repository is correctly set up via the keyring.

# Auto-detect current CUDA version from nvcc if available
if command -v nvcc &> /dev/null; then
    CUDA_VERSION_FULL=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    if [ -n "$CUDA_VERSION_FULL" ]; then
        CUDA_MAJOR=$(echo "$CUDA_VERSION_FULL" | cut -d. -f1)
        DETECTED_PACKAGE="cudnn9-cuda-${CUDA_MAJOR}"
        echo "Detected CUDA version: ${CUDA_VERSION_FULL} (cuDNN package: ${DETECTED_PACKAGE})"
    fi
fi

# Make cuDNN version configurable via environment variable or use detected version
CUDNN_PACKAGE="${CUDNN_VERSION:-${DETECTED_PACKAGE:-cudnn9-cuda-13}}"

# Detect Ubuntu version for repository URLs
UBUNTU_VERSION=$(lsb_release -rs | tr -d '.')
if [ -z "$UBUNTU_VERSION" ]; then
    echo "ERROR: Could not detect Ubuntu version"
    exit 1
fi

echo "==================================================="
echo " cuDNN Updater for ${CUDNN_PACKAGE}"
echo "==================================================="
echo

echo "--- 1. Ensuring CUDA Keyring is in Place ---"

# Only download and install if not already present
if ! dpkg -s cuda-keyring &> /dev/null; then
    echo "Installing CUDA keyring..."
    CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64"
    wget -q "${CUDA_REPO_URL}/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb || { echo "Error: Failed to download cuda-keyring. Exiting."; exit 1; }
    sudo dpkg -i /tmp/cuda-keyring.deb || { echo "Error: Failed to install cuda-keyring. Exiting."; exit 1; }
    rm /tmp/cuda-keyring.deb
else
    echo "CUDA keyring already installed."
fi
echo

echo "--- 2. Updating APT Package Lists ---"
sudo apt-get update || { echo "Error: Failed to update APT package lists. Check internet connection."; exit 1; }
echo "APT package lists updated."
echo

echo "--- 3. Checking and Applying cuDNN Update ---"
sudo apt-get -y install "${CUDNN_PACKAGE}" || { echo "Error: Failed to install or update ${CUDNN_PACKAGE}. Exiting."; exit 1; }
echo "cuDNN update process completed."
echo

# Verify cuDNN installation
echo "Installed cuDNN version:"
dpkg -l | grep cudnn | head -1 || echo "WARNING: Could not verify cuDNN installation"

echo "==================================================="
echo " Script Finished"
echo "==================================================="
echo "No reboot is typically required after cuDNN updates."