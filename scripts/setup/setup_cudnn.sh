#!/bin/bash

# install_cudnn_simple.sh
# This script installs the cuDNN library matching your installed CUDA version,
# automatically detecting the CUDA version from nvcc.
#
# IMPORTANT:
# - This script does NOT reboot your system.
# - Ensure your NVIDIA GPU driver and CUDA Toolkit are already installed and working.
# - The 'cuda-keyring' DEB is assumed to correctly add the necessary repository for cuDNN packages.

# Auto-detect current CUDA version from nvcc if available
if command -v nvcc &> /dev/null; then
    CUDA_VERSION_FULL=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    if [ -n "$CUDA_VERSION_FULL" ]; then
        CUDA_MAJOR=$(echo "$CUDA_VERSION_FULL" | cut -d. -f1)
        DETECTED_PACKAGE="cudnn9-cuda-${CUDA_MAJOR}"
        echo "Detected CUDA version: ${CUDA_VERSION_FULL} (cuDNN package: ${DETECTED_PACKAGE})"
    fi
fi

# Allow override via environment variable or use detected version
CUDNN_PACKAGE="${CUDNN_VERSION:-${DETECTED_PACKAGE:-cudnn9-cuda-13}}"

echo "==================================================="
echo " Simple cuDNN Installation Script"
echo "==================================================="
echo

echo "--- 1. Downloading and Installing CUDA Keyring ---"
# This step adds the NVIDIA GPG key to your system, allowing APT to trust
# packages from NVIDIA's repositories.
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring_1.1-1_all.deb || { echo "Error: Failed to download cuda-keyring. Exiting."; exit 1; }
sudo dpkg -i /tmp/cuda-keyring_1.1-1_all.deb || { echo "Error: Failed to install cuda-keyring. Exiting."; exit 1; }
rm /tmp/cuda-keyring_1.1-1_all.deb # Clean up downloaded deb file
echo "CUDA keyring installed."
echo

echo "--- 2. Updating APT Package Lists ---"
# This refreshes your local list of available packages, including those
# from the NVIDIA repositories enabled by the keyring.
sudo apt-get update || { echo "Error: Failed to update APT package lists. Check internet connection."; exit 1; }
echo "APT package lists updated."
echo

echo "--- 3. Installing cuDNN Package (${CUDNN_PACKAGE}) ---"
# Installs the specified cuDNN package.
sudo apt-get -y install "${CUDNN_PACKAGE}" || { echo "Error: Failed to install ${CUDNN_PACKAGE}. Exiting."; exit 1; }
echo "cuDNN package '${CUDNN_PACKAGE}' installed successfully."
echo

echo "==================================================="
echo " cuDNN Installation Complete"
echo "==================================================="
echo "You can verify cuDNN installation by checking for the package:"
echo "  apt list --installed | grep cudnn"
echo "And by running a PyTorch/TensorFlow script that uses cuDNN."
echo "No reboot is typically required for cuDNN installations."
echo