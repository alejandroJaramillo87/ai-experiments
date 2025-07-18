#!/bin/bash

# NVIDIA Driver Installation Script
# This script removes old NVIDIA drivers and installs nvidia-driver-570-server-open

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}


# Function to run command with error handling
run_command() {
    local cmd="$1"
    local description="$2"
    
    print_status "$description"
    echo "Running: $cmd"
    
    if eval "$cmd"; then
        print_status "✓ $description completed successfully"
        echo
    else
        print_error "✗ Failed: $description"
        print_error "Command that failed: $cmd"
        exit 1
    fi
}

# Main script execution
main() {
    print_status "Starting NVIDIA driver installation process..."
    echo
    

    
    # Confirm with user
    print_warning "This script will:"
    echo "  1. Remove all existing NVIDIA drivers"
    echo "  2. Clean up orphaned packages"
    echo "  3. Install nvidia-driver-570-server-open"
    echo
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Operation cancelled by user."
        exit 0
    fi
    
    echo
    print_status "Starting driver cleanup and installation..."
    echo
    
    # Step 1: Remove existing NVIDIA drivers
    run_command 'sudo apt-get --purge remove "nvidia-*"' "Removing existing NVIDIA drivers"
    
    # Step 2: Remove orphaned packages
    run_command "sudo apt autoremove --purge" "Removing orphaned packages"
    
    # Step 3: Clean package cache
    run_command "sudo apt clean" "Cleaning package cache"
    
    # Step 4: Update package list
    run_command "sudo apt update" "Updating package list"
    
    # Step 5: Install new NVIDIA driver
    run_command "sudo apt install nvidia-driver-570-server-open" "Installing nvidia-driver-570-server-open"
    
    # Success message
    print_status "✓ NVIDIA driver installation completed successfully!"
    echo
    print_warning "IMPORTANT: A system reboot is required for the new driver to take effect."
    echo
    read -p "Would you like to reboot now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Rebooting system..."
        sudo reboot
    else
        print_warning "Please remember to reboot your system before using the new NVIDIA driver."
    fi
}

# Trap to handle script interruption
trap 'print_error "Script interrupted by user"; exit 1' INT TERM

# Run main function
main "$@"