#!/bin/bash

# nvidia_driver_updater_simple.sh
# This script automatically checks for and applies updates for the
# nvidia-driver-570-server-open package.
#
# IMPORTANT:
# - It is highly recommended to run this script from a TTY (text-only terminal, Ctrl+Alt+F2)
#   to minimize potential conflicts during driver updates.

# Auto-detect installed NVIDIA driver if not specified
if [ -z "$NVIDIA_DRIVER_VERSION" ]; then
    DRIVER_PACKAGE=$(dpkg -l | grep "^ii.*nvidia-driver-[0-9].*-open" | awk '{print $2}' | sort -V | tail -1)
    if [ -z "$DRIVER_PACKAGE" ]; then
        echo "ERROR: No NVIDIA driver found. Please install a driver first."
        exit 1
    fi
else
    DRIVER_PACKAGE="$NVIDIA_DRIVER_VERSION"
fi

echo "=== NVIDIA Driver Updater for ${DRIVER_PACKAGE} ==="
echo

# Check for running GUI sessions unless forced
# if pgrep -x "Xorg|Xwayland" > /dev/null && [ "${FORCE_UPDATE:-}" != "yes" ]; then
#     echo "ERROR: GUI session detected. Please switch to TTY (Ctrl+Alt+F2)"
#     echo "       or set FORCE_UPDATE=yes to proceed anyway (not recommended)"
#     exit 1
# fi

echo "Step 1 - Updating APT package lists:"
sudo apt update || { echo "Error: Failed to update APT package lists. Exiting."; exit 1; }
echo "APT package lists updated."
echo

echo "Step 2 - Checking for available driver updates:"
# Check if the driver package is installed
if ! dpkg -s "${DRIVER_PACKAGE}" &> /dev/null; then
    echo "Error: Package '${DRIVER_PACKAGE}' is not installed. Cannot check for updates."
    echo "Please ensure the driver is installed correctly before running this script."
    exit 1
fi

# Use apt list --upgradable to check for a new version
# We grep for the package name and then for the "upgradable from" pattern
UPGRADABLE_INFO=$(apt list --upgradable "${DRIVER_PACKAGE}" 2>/dev/null | grep "${DRIVER_PACKAGE}" | grep -E '\[upgradable from:')

if [ -n "$UPGRADABLE_INFO" ]; then
    CURRENT_VERSION=$(echo "$UPGRADABLE_INFO" | awk -F' ' '{print $2}' | sed 's/.*from: \(.*\)]/\1/')
    NEW_VERSION=$(echo "$UPGRADABLE_INFO" | awk -F' ' '{print $2}' | sed 's/\/.*//')
    
    echo "An update is available for ${DRIVER_PACKAGE}"
    echo "  Current Version: ${CURRENT_VERSION}"
    echo "  New Version:     ${NEW_VERSION}"
    echo

    echo "--- 3. Applying Update ---"
    echo "Attempting to install the new version of ${DRIVER_PACKAGE}."
    echo "This will proceed automatically. Please be aware a reboot will follow."
    
    sudo apt install "${DRIVER_PACKAGE}" -y || { echo "Error: Failed to install update. Exiting."; exit 1; }
    echo "Update applied successfully."
    echo


else
    echo "No updates found for ${DRIVER_PACKAGE}. You are already on the latest version."
    echo "Script finished without rebooting."
fi

echo
echo "=== Script Completed ==="