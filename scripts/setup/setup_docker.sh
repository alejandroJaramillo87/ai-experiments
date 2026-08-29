#!/bin/bash
#
# setup_docker.sh -- install Docker CE and the NVIDIA Container Toolkit, the
# two halves of GPU-capable containers on this workstation.
#
# The compose plugin is installed alongside the engine and is not optional:
# every make target in this repository drives `docker compose`, so an engine
# without it provisions a machine that cannot start a single service.
#
# Runs as a regular user with sudo rights, not as root, because it adds the
# invoking user to the docker group. That membership only takes effect at the
# next login.

set -euo pipefail

# USER is not exported by every non-login shell, so it cannot be relied on
# under set -u; id(1) is always right.
TARGET_USER="${USER:-$(id -un)}"

DRY_RUN=false

RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
BLUE='\033[0;34m' CYAN='\033[0;36m' NC='\033[0m'

print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*" >&2; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
print_run()   { printf "${BLUE}[RUN]${NC} %s\n" "$*"; }

usage() {
    cat <<'EOF'
Usage: setup_docker.sh [OPTIONS]

Install Docker CE (engine, CLI, buildx, compose) and the NVIDIA Container
Toolkit, and add the current user to the docker group.

Options:
  --dry-run        Print every action without executing it
  --no-color       Disable ANSI color
  --help           Show this message

Run as a regular user with sudo rights, not as root. Log out and back in
afterwards for the docker group membership to take effect.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        --no-color) RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''; shift ;;
        --help|-h)  usage; exit 0 ;;
        *)          print_error "Unknown option: $1"; usage >&2; exit 2 ;;
    esac
done

if [[ -n "${NO_COLOR:-}" ]]; then
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''
fi

$DRY_RUN && print_info "Dry run: no changes will be made"

run() {
    if $DRY_RUN; then
        printf "[DRY-RUN] %s\n" "$*"
    else
        print_run "$*"
        "$@"
    fi
}

# Writing a root-owned file needs a pipe into tee, which run() cannot express
# because it takes argv. This keeps the dry-run contract for that one shape.
write_root_file() {
    local path="$1" content="$2"
    if $DRY_RUN; then
        printf "[DRY-RUN] %s\n" "write $path"
    else
        print_run "write $path"
        printf '%s\n' "$content" | sudo tee "$path" >/dev/null
    fi
}

if [[ $EUID -eq 0 ]]; then
    print_error "This script must not run as root"
    echo "  Context: it adds the invoking user to the docker group, and root is not that user"
    echo "  Command: ./scripts/setup/setup_docker.sh"
    exit 1
fi

if ! sudo -n true 2>/dev/null; then
    print_error "Passwordless sudo is not available"
    echo "  Solution: configure sudo for this user first"
    echo "  Command: sudo ./scripts/setup/setup_sudo_user.sh"
    exit 1
fi

echo "=== Installing Docker CE and the NVIDIA Container Toolkit ==="
echo

echo "=== Installing prerequisites ==="
run sudo apt-get update
run sudo apt-get install --yes wget curl ca-certificates lsb-release
# apt does not ship this directory on a fresh install, and writing a key into a
# path that does not exist fails with a bare redirection error.
run sudo install -m 0755 -d /etc/apt/keyrings
print_ok "Prerequisites installed"
echo

echo "=== Adding the Docker repository ==="
run sudo wget -qO /etc/apt/keyrings/docker.asc https://download.docker.com/linux/ubuntu/gpg
run sudo chmod a+r /etc/apt/keyrings/docker.asc
write_root_file /etc/apt/sources.list.d/docker.list \
    "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(lsb_release -sc) stable"
print_ok "Docker repository added"
echo

echo "=== Installing Docker CE ==="
run sudo apt-get update
# compose and buildx are plugins of the engine, not dependencies of it: an
# engine installed without them cannot run `docker compose` at all.
run sudo apt-get install --yes \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin
print_ok "Docker CE installed"

run sudo usermod -aG docker "$TARGET_USER"
run sudo systemctl enable docker
run sudo systemctl start docker
print_ok "Docker service enabled and started"
echo

echo "=== Adding the NVIDIA Container Toolkit repository ==="
run sudo wget -qO /etc/apt/keyrings/nvidia-container-toolkit.asc \
    https://nvidia.github.io/libnvidia-container/gpgkey
run sudo chmod a+r /etc/apt/keyrings/nvidia-container-toolkit.asc
write_root_file /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    "deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.asc] https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /"
print_ok "NVIDIA Container Toolkit repository added"
echo

echo "=== Installing the NVIDIA Container Toolkit ==="
run sudo apt-get update
run sudo apt-get install --yes nvidia-container-toolkit
run sudo systemctl restart docker
print_ok "NVIDIA Container Toolkit installed"
echo

echo "=== Complete ==="
if $DRY_RUN; then
    print_info "No changes were made"
    echo "  Run for real: $0"
    exit 0
fi

print_ok "Docker CE and the NVIDIA Container Toolkit are installed"
print_warn "Log out and back in before using docker without sudo"
echo "  Verify engine:  docker version"
echo "  Verify compose: docker compose version"
echo "  Verify GPU:     docker run --rm --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark"
echo

echo "=== Current status ==="
if CLIENT_VERSION=$(docker version --format '{{.Client.Version}}' 2>/dev/null); then
    print_ok "Docker client: $CLIENT_VERSION"
else
    print_warn "Docker client not reachable yet; log out and back in"
fi
print_ok "Docker service: $(sudo systemctl is-active docker)"
if dpkg -l nvidia-container-toolkit 2>/dev/null | grep -q '^ii'; then
    print_ok "nvidia-container-toolkit: $(dpkg-query -W -f='${Version}' nvidia-container-toolkit)"
else
    print_warn "nvidia-container-toolkit: not installed"
fi
