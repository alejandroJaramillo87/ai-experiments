#!/bin/bash
#
# setup_cuda.sh -- install the CUDA toolkit from NVIDIA's apt repository and
# put nvcc on the shell PATH.
#
# The network repository is used rather than a local installer DEB. An earlier
# version of this script downloaded the versioned local installer, which is a
# URL that 404s on every CUDA release; and it bought nothing, because the two
# repositories sit at equal apt priority and the newer network candidate wins
# regardless. On this workstation the local repo was pinned at 13.0.0 while the
# installed toolkit had already moved to 13.0.3 from the network repo.
#
# The same cuda-keyring package configures the repository for setup_cudnn.sh,
# so both scripts now take one path to NVIDIA's archive.
#
# The NVIDIA driver must already be installed; see setup_nvidia.sh. This script
# does not reboot.

set -euo pipefail

# Toolkit branch. Bump this one line to move branches; the patch version comes
# from whatever the repository currently serves. See AGENTS.md on pinning.
CUDA_BRANCH="13-0"
CUDA_PACKAGE="cuda-toolkit-${CUDA_BRANCH}"

KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"

ZSHRC_FILE="$HOME/.zshrc"
DRY_RUN=false

RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
BLUE='\033[0;34m' CYAN='\033[0;36m' NC='\033[0m'

print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*" >&2; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
print_run()   { printf "${BLUE}[RUN]${NC} %s\n" "$*"; }
print_skip()  { printf "${BLUE}[SKIP]${NC} %s\n" "$*"; }

usage() {
    cat <<'EOF'
Usage: setup_cuda.sh [OPTIONS]

Install the CUDA toolkit from NVIDIA's apt repository and append the CUDA
PATH and LD_LIBRARY_PATH exports to ~/.zshrc.

Options:
  --dry-run        Print every action without executing it
  --no-color       Disable ANSI color
  --help           Show this message

No reboot is performed. Open a new shell, or source ~/.zshrc, to pick up nvcc.
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

# The keyring DEB lands in a temp directory and is removed on exit. The earlier
# version left its downloads in whatever directory it was invoked from.
if ! $DRY_RUN; then
    WORKDIR=$(mktemp -d)
    trap 'rm -rf "$WORKDIR"' EXIT
fi

echo "=== Installing $CUDA_PACKAGE ==="
echo

echo "=== Adding the NVIDIA apt repository ==="
# cuda-keyring installs both the signing key and the repository definition, and
# carries the apt pin that makes NVIDIA's packages win over Ubuntu's older ones.
if $DRY_RUN; then
    printf "[DRY-RUN] %s\n" "download and install $(basename "$KEYRING_URL")"
else
    KEYRING_DEB="$WORKDIR/$(basename "$KEYRING_URL")"
    if ! run wget -q -O "$KEYRING_DEB" "$KEYRING_URL"; then
        print_error "Failed to download the CUDA keyring"
        echo "  Solution: check network access to developer.download.nvidia.com"
        exit 1
    fi
    run sudo dpkg -i "$KEYRING_DEB"
    print_ok "CUDA keyring installed"
fi
echo

echo "=== Installing the toolkit ==="
run sudo apt-get update
run sudo apt-get install --yes "$CUDA_PACKAGE"
print_ok "Toolkit installed: $CUDA_PACKAGE"
echo

echo "=== Configuring the shell environment ==="
CUDA_PATH_LINE='export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}'
CUDA_LIB_LINE='export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}'

# Appending to a file that does not exist yet is a first-boot case, not an
# error: grep on a missing file would otherwise abort the script under set -e.
if $DRY_RUN; then
    printf "[DRY-RUN] %s\n" "append CUDA PATH and LD_LIBRARY_PATH to $ZSHRC_FILE"
else
    [[ -f "$ZSHRC_FILE" ]] || run touch "$ZSHRC_FILE"

    for line in "$CUDA_PATH_LINE" "$CUDA_LIB_LINE"; do
        if grep -qxF "$line" "$ZSHRC_FILE"; then
            print_skip "Already present in $(basename "$ZSHRC_FILE"): ${line%%=*}"
        else
            printf '%s\n' "$line" >> "$ZSHRC_FILE"
            print_ok "Appended to $(basename "$ZSHRC_FILE"): ${line%%=*}"
        fi
    done
fi
echo

echo "=== Complete ==="
if $DRY_RUN; then
    print_info "No changes were made"
else
    print_ok "CUDA toolkit installed"
    echo "  Verify: source ~/.zshrc && nvcc --version"
    print_info "No reboot performed; a new shell is enough to pick up nvcc"
fi
