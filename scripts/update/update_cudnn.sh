#!/bin/bash

# update_cudnn_simple.sh
# This script automatically checks for and applies updates for the
# cuDNN package (cudnn9-cuda-12).
#
# IMPORTANT:
# - This script does NOT reboot your system automatically.
# - It assumes the NVIDIA CUDA repository is correctly set up via the keyring.

CUDNN_PACKAGE="cudnn9-cuda-12" # This is the specific package name apt uses for cuDNN 9 for CUDA 12

echo "==================================================="
echo " Simple cuDNN Updater for ${CUDNN_PACKAGE}"
echo "==================================================="
echo

echo "--- 1. Ensuring CUDA Keyring is in Place ---"
# This step ensures the NVIDIA GPG key is installed, which is necessary
# for APT to trust packages from NVIDIA's repositories and find updates.
# It's idempotent, so safe to run repeatedly.
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring_1.1-1_all.deb || { echo "Error: Failed to download cuda-keyring. Exiting."; exit 1; }
sudo dpkg -i /tmp/cuda-keyring_1.1-1_all.deb || { echo "Error: Failed to install cuda-keyring. Exiting."; exit 1; }
rm /tmp/cuda-keyring_1.1-1_all.deb # Clean up downloaded deb file
echo "CUDA keyring checked/installed."
echo

echo "--- 2. Updating APT Package Lists ---"
# This refreshes your local list of available packages from all repositories,
# including the NVIDIA ones, to find if new cuDNN versions are available.
sudo apt-get update || { echo "Error: Failed to update APT package lists. Check internet connection."; exit 1; }
echo "APT package lists updated."
echo

echo "--- 3. Checking and Applying cuDNN Update ---"
# This command will check if a newer version of ${CUDNN_PACKAGE} is available
# and, if so, it will automatically download and install it.
# If it's already the newest version, it will simply state that.
sudo apt-get -y install "${CUDNN_PACKAGE}" || { echo "Error: Failed to install or update ${CUDNN_PACKAGE}. Exiting."; exit 1; }
echo "cuDNN update process completed."
echo

echo "==================================================="
echo " Script Finished"
echo "==================================================="
echo "You can verify the installed cuDNN version with: apt list --installed | grep cudnn"
echo "No reboot is typically required after cuDNN updates."
echo