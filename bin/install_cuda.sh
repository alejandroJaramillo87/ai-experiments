#!/bin/bash

# install_cuda_exact_steps.sh
# This script performs a CUDA Toolkit 12.8 installation exactly as per the provided steps,
# including setting up a local repository and then installing the toolkit.
# It also configures environment variables in ~/.zshrc and reboots the system.
#
# IMPORTANT:
# - This script will REBOOT your system automatically at the end.
# - This method primarily uses a *local* CUDA repository installer. While it adds
#   the general CUDA pin file, future updates for CUDA Toolkit might require
#   downloading new local installer DEBs if NVIDIA doesn't push updates
#   to the general network repository for this specific version.
# - Ensure your NVIDIA GPU driver is already installed and working
#   before running this script.

echo "==================================================="
echo " Installing NVIDIA CUDA Toolkit 12.8 (Exact Steps)"
echo "==================================================="
echo

echo "--- 1. Setting up CUDA Repository Pin ---"
# Downloads a pin file that tells APT to prefer NVIDIA's CUDA packages
# from their repositories over potentially older versions from Ubuntu's default.
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600 || { echo "Error moving cuda-ubuntu2404.pin. Exiting."; exit 1; }
echo "CUDA repository pin file moved to /etc/apt/preferences.d/"
echo

echo "--- 2. Installing Local CUDA Repository DEB ---"
# Downloads a specific local installer DEB for CUDA 12.8.1.
# This DEB sets up a local APT repository on your system (e.g., in /var/cuda-repo-...).
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.1-570.124.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.1-570.124.06-1_amd64.deb || { echo "Error installing local CUDA repo DEB. Exiting."; exit 1; }
echo "Local CUDA repository DEB installed."
echo

echo "--- 3. Copying CUDA Keyring ---"
# Copies the GPG keyring from the local repository to a system-wide location.
# This allows APT to verify the authenticity of packages from the local CUDA repository.
sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ || { echo "Error copying CUDA keyring. Exiting."; exit 1; }
echo "CUDA keyring copied to /usr/share/keyrings/"
echo

echo "--- 4. Updating APT Package Lists and Installing CUDA Toolkit 12.8 ---"
# Updates the list of available packages from all repositories (including the newly added local one).
# Then, installs the cuda-toolkit-12-8 package.
sudo apt-get update && sudo apt-get -y install cuda-toolkit-12-8 || { echo "Error during APT update or CUDA Toolkit installation. Exiting."; exit 1; }
echo "APT package lists updated and cuda-toolkit-12-8 installed."
echo

echo "--- 5. Configuring ~/.zshrc for CUDA Environment Variables ---"
ZSHRC_FILE="$HOME/.zshrc"
CUDA_PATH_LINE='export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}'
CUDA_LD_LIBRARY_PATH_LINE='export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}'

# Check if lines already exist to prevent duplicates
if ! grep -qxF "${CUDA_PATH_LINE}" "${ZSHRC_FILE}"; then
    echo "${CUDA_PATH_LINE}" >> "${ZSHRC_FILE}"
    echo "Added CUDA PATH to ${ZSHRC_FILE}"
else
    echo "CUDA PATH already present in ${ZSHRC_FILE}"
fi

if ! grep -qxF "${CUDA_LD_LIBRARY_PATH_LINE}" "${ZSHRC_FILE}"; then
    echo "${CUDA_LD_LIBRARY_PATH_LINE}" >> "${ZSHRC_FILE}"
    echo "Added CUDA LD_LIBRARY_PATH to ${ZSHRC_FILE}"
else
    echo "CUDA LD_LIBRARY_PATH already present in ${ZSHRC_FILE}"
fi

echo "Please remember to 'source ~/.zshrc' or open a new terminal after reboot to apply env vars."
echo

echo "==================================================="
echo " CUDA Toolkit Installation Complete"
echo "==================================================="
echo "The system will now reboot to finalize the installation."
echo "Press Ctrl+C to cancel reboot, otherwise rebooting in 5 seconds..."
sleep 5

# sudo shutdown -r now