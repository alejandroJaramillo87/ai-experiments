#!/bin/bash

# update_docker.sh
# This script updates Docker CE, Docker Compose, and NVIDIA Container Toolkit.
#
# It does not touch the Dockerfiles. Base image tags are pinned deliberately and
# bumped one at a time with an acceptance check after each, per the operating
# rules in AGENTS.md; a rewrite driven by whatever nvcc reports on the host is
# the opposite of that.

echo "=== Docker Infrastructure Updater ==="
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

echo "=== Docker Infrastructure Update Complete ==="
echo
echo "Docker CE: ${NEW_DOCKER_VERSION}"
echo "Docker Compose: ${NEW_COMPOSE_VERSION}"
echo "NVIDIA Container Toolkit: ${NEW_NCT_VERSION}"
echo
echo "Rebuild the GPU images to pick up the new runtime:"
echo "  make build-gpu"