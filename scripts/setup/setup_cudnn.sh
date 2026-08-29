#!/bin/bash
#
# setup_cudnn.sh -- install the cuDNN runtime matching the installed CUDA
# toolkit.
#
# The package name carries the CUDA major version (cudnn9-cuda-13), so it is
# derived from nvcc rather than pinned here: a cuDNN built against a different
# major loads but fails at the first kernel launch. Override with CUDNN_PACKAGE
# when nvcc is absent or the derived name is wrong.
#
# No reboot is required. The NVIDIA driver and CUDA toolkit must already be
# installed -- see setup_nvidia.sh and setup_cuda.sh.

set -euo pipefail

KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
FALLBACK_PACKAGE="cudnn9-cuda-13"

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
Usage: setup_cudnn.sh [OPTIONS]

Install the cuDNN runtime matching the installed CUDA toolkit, adding NVIDIA's
apt repository first.

Options:
  --dry-run        Print every action without executing it
  --no-color       Disable ANSI color
  --help           Show this message

Environment:
  CUDNN_PACKAGE    Package to install, overriding the version derived from nvcc
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

echo "=== Resolving the cuDNN package ==="

DETECTED_PACKAGE=""
OVERRIDE="${CUDNN_PACKAGE:-}"

if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    if [[ -n "$CUDA_VERSION" ]]; then
        DETECTED_PACKAGE="cudnn9-cuda-${CUDA_VERSION%%.*}"
        print_ok "CUDA $CUDA_VERSION detected via nvcc"
    fi
fi

CUDNN_PACKAGE="${OVERRIDE:-${DETECTED_PACKAGE:-$FALLBACK_PACKAGE}}"

if [[ -z "$DETECTED_PACKAGE" && -z "$OVERRIDE" ]]; then
    print_warn "nvcc not on PATH; falling back to $FALLBACK_PACKAGE"
    echo "  Context: a cuDNN major mismatch fails at the first kernel launch, not at install"
    echo "  Solution: install the toolkit first, or name the package explicitly"
    echo "  Command: scripts/setup/setup_cuda.sh"
fi

print_info "Package to install: $CUDNN_PACKAGE"
echo

echo "=== Adding the NVIDIA apt repository ==="
if $DRY_RUN; then
    printf "[DRY-RUN] %s\n" "download and install $(basename "$KEYRING_URL")"
else
    WORKDIR=$(mktemp -d)
    trap 'rm -rf "$WORKDIR"' EXIT

    KEYRING_DEB="$WORKDIR/$(basename "$KEYRING_URL")"
    if ! run wget -q -O "$KEYRING_DEB" "$KEYRING_URL"; then
        print_error "Failed to download the CUDA keyring"
        echo "  Solution: check network access to developer.download.nvidia.com"
        exit 1
    fi
    run sudo dpkg -i "$KEYRING_DEB"
    print_ok "CUDA keyring installed"
fi

run sudo apt-get update
print_ok "Package lists updated"
echo

echo "=== Installing $CUDNN_PACKAGE ==="
if ! run sudo apt-get install --yes "$CUDNN_PACKAGE"; then
    print_error "Failed to install $CUDNN_PACKAGE"
    echo "  Solution: confirm the package exists for this CUDA major version"
    echo "  Command: apt-cache policy $CUDNN_PACKAGE"
    exit 1
fi
echo

echo "=== Complete ==="
if $DRY_RUN; then
    print_info "No changes were made"
else
    print_ok "cuDNN installed: $CUDNN_PACKAGE"
    echo "  Verify: python -c 'import torch; print(torch.backends.cudnn.version())'"
    print_info "No reboot required"
fi
