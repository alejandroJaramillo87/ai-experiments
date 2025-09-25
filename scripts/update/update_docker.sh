#!/bin/bash

# update_docker.sh
# This script updates Docker CE, Docker Compose, and NVIDIA Container Toolkit
# It also updates Dockerfiles to match the host CUDA version

echo "=== Docker Infrastructure Updater ==="
echo

# Auto-detect current CUDA version from nvcc if available
if command -v nvcc &> /dev/null; then
    CUDA_VERSION_FULL=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    if [ -n "$CUDA_VERSION_FULL" ]; then
        CUDA_MAJOR=$(echo "$CUDA_VERSION_FULL" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VERSION_FULL" | cut -d. -f2)
        DETECTED_CUDA="${CUDA_MAJOR}.${CUDA_MINOR}"
        echo "Detected host CUDA version: ${DETECTED_CUDA}"
    fi
fi

# Allow override via environment variable or use detected version
CUDA_VERSION="${CUDA_VERSION:-${DETECTED_CUDA:-13.0}}"
echo "Using CUDA version: ${CUDA_VERSION}"
echo

echo "Step 1 - Updating Docker CE:"
# Check current Docker version
CURRENT_DOCKER_VERSION=$(docker --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
echo "Current Docker version: ${CURRENT_DOCKER_VERSION:-not installed}"

# Update Docker CE
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io || { echo "Error updating Docker CE"; exit 1; }

NEW_DOCKER_VERSION=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
echo "Docker CE updated to: ${NEW_DOCKER_VERSION}"
echo

echo "Step 2 - Updating Docker Compose:"
# Check current Docker Compose version
CURRENT_COMPOSE_VERSION=$(docker compose version 2>/dev/null | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' | head -1)
echo "Current Docker Compose version: ${CURRENT_COMPOSE_VERSION:-not installed}"

# Get latest compose version from GitHub
COMPOSE_VERSION="${DOCKER_COMPOSE_VERSION:-$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep tag_name | cut -d'"' -f4)}"
echo "Installing Docker Compose ${COMPOSE_VERSION}"

# Download and install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create symlink for 'docker compose' command if needed
sudo ln -sf /usr/local/bin/docker-compose /usr/libexec/docker/cli-plugins/docker-compose 2>/dev/null || true

NEW_COMPOSE_VERSION=$(docker compose version 2>/dev/null | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' | head -1)
echo "Docker Compose updated to: ${NEW_COMPOSE_VERSION}"
echo

echo "Step 3 - Updating NVIDIA Container Toolkit:"
# Check current NVIDIA Container Toolkit version
CURRENT_NCT_VERSION=$(dpkg -l | grep nvidia-container-toolkit | awk '{print $3}' | head -1)
echo "Current NVIDIA Container Toolkit version: ${CURRENT_NCT_VERSION:-not installed}"

# Update NVIDIA Container Toolkit
sudo apt update
sudo apt install -y nvidia-container-toolkit || { echo "Error updating NVIDIA Container Toolkit"; exit 1; }

NEW_NCT_VERSION=$(dpkg -l | grep nvidia-container-toolkit | awk '{print $3}' | head -1)
echo "NVIDIA Container Toolkit updated to: ${NEW_NCT_VERSION}"

# Restart Docker to apply NVIDIA runtime changes
sudo systemctl restart docker
echo "Docker service restarted"
echo

echo "Step 4 - Updating Dockerfiles to CUDA ${CUDA_VERSION}:"
# Find the latest CUDA image tag for the detected version
# Using the .1 patch version as that's the pattern (e.g., 12.9.1, 13.0.1)
CUDA_IMAGE_TAG="${CUDA_VERSION}.1"

# Update Dockerfile.llama-gpu
DOCKERFILE_LLAMA="/home/alejandro/workspace/ai-expirements/docker/Dockerfile.llama-gpu"
if [ -f "$DOCKERFILE_LLAMA" ]; then
    echo "Updating $DOCKERFILE_LLAMA to use CUDA ${CUDA_IMAGE_TAG}"
    sed -i "s/FROM nvidia\/cuda:[0-9]\+\.[0-9]\+\.[0-9]\+-devel-ubuntu[0-9]\+\.[0-9]\+/FROM nvidia\/cuda:${CUDA_IMAGE_TAG}-devel-ubuntu24.04/g" "$DOCKERFILE_LLAMA"
    sed -i "s/FROM nvidia\/cuda:[0-9]\+\.[0-9]\+\.[0-9]\+-runtime-ubuntu[0-9]\+\.[0-9]\+/FROM nvidia\/cuda:${CUDA_IMAGE_TAG}-runtime-ubuntu24.04/g" "$DOCKERFILE_LLAMA"
    echo "Updated Dockerfile.llama-gpu"
fi

# Update Dockerfile.vllm-gpu
DOCKERFILE_VLLM="/home/alejandro/workspace/ai-expirements/docker/Dockerfile.vllm-gpu"
if [ -f "$DOCKERFILE_VLLM" ]; then
    echo "Updating $DOCKERFILE_VLLM to use CUDA ${CUDA_IMAGE_TAG}"
    sed -i "s/FROM nvidia\/cuda:[0-9]\+\.[0-9]\+\.[0-9]\+-devel-ubuntu[0-9]\+\.[0-9]\+/FROM nvidia\/cuda:${CUDA_IMAGE_TAG}-devel-ubuntu24.04/g" "$DOCKERFILE_VLLM"
    echo "Updated Dockerfile.vllm-gpu"
fi

echo
echo "=== Docker Infrastructure Update Complete ==="
echo
echo "Docker CE: ${NEW_DOCKER_VERSION}"
echo "Docker Compose: ${NEW_COMPOSE_VERSION}"
echo "NVIDIA Container Toolkit: ${NEW_NCT_VERSION}"
echo "Dockerfiles updated to CUDA: ${CUDA_IMAGE_TAG}"
echo
echo "You may need to rebuild your Docker images to use the new CUDA version:"
echo "  make build-gpu"