#!/bin/bash

# cuda_toolkit_updater_simple.sh
# This script automatically checks for and applies updates for the
# cuda-toolkit-12-8 package.
#
# IMPORTANT:
# - This script does NOT reboot your system automatically, as CUDA Toolkit updates
#   do not always require a reboot (unlike NVIDIA drivers).
# - It is recommended to run this script from a TTY (text-only terminal, Ctrl+Alt+F2)
#   if you are also updating NVIDIA drivers alongside this.

CUDA_PACKAGE="cuda-toolkit-12-8"

echo "==================================================="
echo " Simple CUDA Toolkit Updater for ${CUDA_PACKAGE}"
echo "==================================================="
echo

echo "--- 1. Ensuring NVIDIA CUDA Network Repository is Configured ---"
# These steps are idempotent and ensure the official NVIDIA CUDA network repository
# is added and trusted by your system's APT. This is crucial for finding updates.

# Add the repository pin file (if not already there)
if [ ! -f "/etc/apt/preferences.d/cuda-repository-pin-600" ]; then
    echo "Adding CUDA repository pin file..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin -O /tmp/cuda-ubuntu2404.pin
    sudo mv /tmp/cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600 || { echo "Error moving pin file. Exiting."; exit 1; }
else
    echo "CUDA repository pin file already exists."
fi

# Install the CUDA GPG keyring (if not already installed)
if ! dpkg -s cuda-keyring &> /dev/null; then
    echo "Installing CUDA GPG keyring..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i /tmp/cuda-keyring_1.1-1_all.deb || { echo "Error installing cuda-keyring. Exiting."; exit 1; }
    rm /tmp/cuda-keyring_1.1-1_all.deb # Clean up downloaded deb
else
    echo "CUDA GPG keyring already installed."
fi

echo "--- 2. Updating APT Package Lists ---"
sudo apt update || { echo "Error: Failed to update APT package lists. Check internet connection."; exit 1; }
echo "APT package lists updated."
echo

echo "--- 3. Checking for Available CUDA Toolkit Updates ---"
# Check if the CUDA Toolkit package is installed
if ! dpkg -s "${CUDA_PACKAGE}" &> /dev/null; then
    echo "Error: Package '${CUDA_PACKAGE}' is not installed. Cannot check for updates."
    echo "Please ensure the CUDA Toolkit is installed correctly before running this script."
    exit 1
fi

# Use apt list --upgradable to check for a new version
UPGRADABLE_INFO=$(apt list --upgradable "${CUDA_PACKAGE}" 2>/dev/null | grep "${CUDA_PACKAGE}" | grep -E '\[upgradable from:')

if [ -n "$UPGRADABLE_INFO" ]; then
    CURRENT_VERSION=$(echo "$UPGRADABLE_INFO" | awk -F' ' '{print $2}' | sed 's/.*from: \(.*\)]/\1/')
    NEW_VERSION=$(echo "$UPGRADABLE_INFO" | awk -F' ' '{print $2}' | sed 's/\/.*//')
    
    echo "An update is available for ${CUDA_PACKAGE}!"
    echo "  Current Version: ${CURRENT_VERSION}"
    echo "  New Version:     ${NEW_VERSION}"
    echo

    echo "--- 4. Applying Update ---"
    echo "Attempting to install the new version of ${CUDA_PACKAGE}."
    
    sudo apt install "${CUDA_PACKAGE}" -y || { echo "Error: Failed to install update. Exiting."; exit 1; }
    echo "Update applied successfully."
    echo

    echo "==================================================="
    echo " CUDA Toolkit Update Completed"
    echo "==================================================="
    echo "The CUDA Toolkit has been updated."
    echo "Remember to re-source your ~/.zshrc (or open a new terminal) to ensure"
    echo "your environment variables point to the correct CUDA installation path."
    echo "You might also need to re-check PyTorch/TensorFlow compatibility."
else
    echo "No updates found for ${CUDA_PACKAGE}. You are already on the latest version."
    echo "Script finished."
fi

echo
echo "==================================================="
echo " Script Completed"
echo "==================================================="