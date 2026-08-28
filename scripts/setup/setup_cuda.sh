#!/bin/bash
#
# setup_cuda.sh -- install the CUDA toolkit from NVIDIA's local repository DEB
# and put nvcc on the shell PATH.
#
# The local installer is used rather than the network repository because it
# pins the toolkit to one patch release: the network repo tracks the newest
# build in the branch, which is a silent version bump on every apt upgrade.
# The cost is that the artifact URL below carries a patch version and has to be
# updated by hand -- see the comment on CUDA_LOCAL_INSTALLER_URL.
#
# The NVIDIA driver must already be installed; see setup_nvidia.sh. This script
# does not reboot.

set -euo pipefail

# Toolkit branch. This is the apt package name and is patch-independent; the
# installer URL below must point at the same branch.
CUDA_BRANCH="13-0"
CUDA_PACKAGE="cuda-toolkit-${CUDA_BRANCH}"

# Local repository DEB. The filename encodes both the toolkit patch version and
# the driver bundled with it, so it changes on every CUDA release. Confirm the
# current artifact before bumping:
#   https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/
CUDA_LOCAL_INSTALLER_URL="https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda-repo-ubuntu2404-13-0-local_13.0.0-580.65.06-1_amd64.deb"
CUDA_PIN_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin"

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

Install the CUDA toolkit from NVIDIA's local repository DEB and append the
CUDA PATH and LD_LIBRARY_PATH exports to ~/.zshrc.

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

# Everything downloaded lands in a temp directory and is removed on exit. The
# earlier version left two DEBs in whatever directory it was invoked from.
if ! $DRY_RUN; then
    WORKDIR=$(mktemp -d)
    trap 'rm -rf "$WORKDIR"' EXIT
fi

echo "=== Installing $CUDA_PACKAGE ==="
print_info "Local installer: $(basename "$CUDA_LOCAL_INSTALLER_URL")"
echo

echo "=== Adding the CUDA repository pin ==="
# The pin makes apt prefer NVIDIA's packages over Ubuntu's older ones.
if $DRY_RUN; then
    printf "[DRY-RUN] %s\n" "download $CUDA_PIN_URL to /etc/apt/preferences.d/cuda-repository-pin-600"
else
    run wget -q -O "$WORKDIR/cuda-ubuntu2404.pin" "$CUDA_PIN_URL"
    run sudo install -m 644 "$WORKDIR/cuda-ubuntu2404.pin" \
        /etc/apt/preferences.d/cuda-repository-pin-600
    print_ok "Repository pin installed"
fi
echo

echo "=== Installing the local repository DEB ==="
if $DRY_RUN; then
    printf "[DRY-RUN] %s\n" "download and dpkg -i $(basename "$CUDA_LOCAL_INSTALLER_URL")"
else
    LOCAL_DEB="$WORKDIR/$(basename "$CUDA_LOCAL_INSTALLER_URL")"
    if ! run wget -q -O "$LOCAL_DEB" "$CUDA_LOCAL_INSTALLER_URL"; then
        print_error "Failed to download the local CUDA repository DEB"
        echo "  Context: the URL carries a patch version and goes stale on each release"
        echo "  Solution: find the current artifact and update CUDA_LOCAL_INSTALLER_URL"
        echo "  Command: xdg-open https://developer.download.nvidia.com/compute/cuda/"
        exit 1
    fi
    run sudo dpkg -i "$LOCAL_DEB"
    print_ok "Local repository registered"

    # The keyring lives inside the local repo; apt cannot verify it until the
    # key is copied somewhere apt trusts.
    run sudo cp /var/cuda-repo-ubuntu2404-"${CUDA_BRANCH}"-local/cuda-*-keyring.gpg \
        /usr/share/keyrings/
    print_ok "Repository keyring installed"
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
