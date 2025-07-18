#!/bin/bash
set -e

# === Configuration ===
DRY_RUN=false

# === Handle dry-run flag ===
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE ENABLED ==="
fi

# === Helper Function ===
run() {
    if $DRY_RUN; then
        echo "[DRY-RUN] $*"
    else
        eval "$@"
    fi
}

# === Check if running as root ===
if [[ $EUID -eq 0 ]]; then
   echo "This script should not be run as root. Please run as a regular user with sudo privileges."
   exit 1
fi

# === Check for sudo privileges ===
if ! sudo -n true 2>/dev/null; then
    echo "This script requires sudo privileges. Please ensure you can run sudo commands."
    exit 1
fi

echo "=== Installing Docker CE and NVIDIA Container Toolkit ==="
echo "=== This will enable GPU support in Docker containers ==="

# === Update package lists ===
echo "=== Updating package lists ==="
run "sudo apt update"

# === Install prerequisites ===
echo "=== Installing prerequisites ==="
run "sudo apt install -y wget curl ca-certificates lsb-release"

# === Install Docker CE ===
echo "=== Installing Docker CE ==="

# Download Docker GPG key
echo "=== Downloading Docker GPG key ==="
run "sudo wget -qO /etc/apt/keyrings/docker.asc https://download.docker.com/linux/ubuntu/gpg"

# Add Docker repository
echo "=== Adding Docker repository ==="
run "echo \"deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \$(lsb_release -sc) stable\" | sudo tee /etc/apt/sources.list.d/docker.list"

# Update package lists and install Docker CE
echo "=== Installing Docker CE ==="
run "sudo apt update"
run "sudo apt install -y docker-ce"

# Add current user to docker group
echo "=== Adding current user to docker group ==="
run "sudo usermod -a -G docker \$USER"

# Start and enable Docker service
echo "=== Starting Docker service ==="
run "sudo systemctl start docker"
run "sudo systemctl enable docker"

# === Install NVIDIA Container Toolkit ===
echo "=== Installing NVIDIA Container Toolkit ==="

# Download NVIDIA Container Toolkit GPG key
echo "=== Downloading NVIDIA Container Toolkit GPG key ==="
run "sudo wget -qO /etc/apt/keyrings/nvidia-container-toolkit.asc https://nvidia.github.io/libnvidia-container/gpgkey"

# Add NVIDIA Container Toolkit repository
echo "=== Adding NVIDIA Container Toolkit repository ==="
run "echo \"deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.asc] https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /\" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"

# Update package lists and install NVIDIA Container Toolkit
echo "=== Installing NVIDIA Container Toolkit ==="
run "sudo apt update"
run "sudo apt install -y nvidia-container-toolkit"

# Restart Docker service to apply changes
echo "=== Restarting Docker service ==="
run "sudo systemctl restart docker"

# === Testing Installation ===
echo "=== Installation Complete ==="

if $DRY_RUN; then
    echo "=== DRY RUN COMPLETE ==="
    echo "No changes were made to the system."
    echo ""
    echo "To run for real, execute: $0"
else
    echo "✅ Docker CE and NVIDIA Container Toolkit have been installed successfully!"
    echo ""
    echo "⚠️  IMPORTANT: You need to log out and log back in for docker group changes to take effect."
    echo ""
    echo "After logging back in, you can test the installation with:"
    echo "  docker version"
    echo "  docker run hello-world"
    echo "  docker run --rm --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark"
    echo ""
    echo "The last command will test GPU access in containers (requires NVIDIA GPU and drivers)."
fi

# === Verification Commands (for reference) ===
if ! $DRY_RUN; then
    echo ""
    echo "=== Current Status ==="
    echo "Docker version:"
    docker version --format '{{.Client.Version}}' 2>/dev/null || echo "  Note: Log out and back in to use docker without sudo"
    echo "Docker service status:"
    sudo systemctl is-active docker
    echo "NVIDIA Container Toolkit installed:"
    dpkg -l | grep nvidia-container-toolkit | awk '{print "  " $2 " " $3}' || echo "  Package not found"
fi