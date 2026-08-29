#!/bin/bash
#
# setup_nvidia.sh -- purge any installed NVIDIA driver and install the branch
# this workstation is qualified on.
#
# The open kernel-module flavor is not optional: Blackwell (RTX 5090, SM 12.0)
# is only supported by the open packages. The proprietary flavor builds but
# will not bind to the card. Both -open and -server-open are open-kernel
# builds; this workstation is provisioned on -server-open, which is what the
# installed metapackage has always been.
#
# The purge step is destructive to a running display server. Run it from a TTY
# (Ctrl+Alt+F2), not from a desktop session, or X will die mid-transaction.

set -euo pipefail

# Driver branch. Bump this one line, then re-run; see AGENTS.md on pinning.
# Verify first: apt-cache policy nvidia-driver-<N>-server-open
DRIVER_BRANCH="580"
DRIVER_PACKAGE="nvidia-driver-${DRIVER_BRANCH}-server-open"

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
Usage: setup_nvidia.sh [OPTIONS]

Remove every installed NVIDIA driver and install the qualified branch, then
offer a reboot. The new module is not loaded until the machine reboots.

Options:
  --dry-run        Print every action without executing it
  --no-color       Disable ANSI color
  --help           Show this message

Run from a TTY (Ctrl+Alt+F2). Purging the driver under a live desktop session
kills the display server partway through the transaction.
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

# Takes argv, not a string: `run sudo apt-get install foo`, never
# `run "sudo apt-get install foo"`. The glob passed to the purge step below is
# quoted so apt expands it, not the shell against the current directory.
run() {
    if $DRY_RUN; then
        printf "[DRY-RUN] %s\n" "$*"
    else
        print_run "$*"
        "$@"
    fi
}

echo "=== NVIDIA driver installation ==="
echo
print_info "Target package: $DRIVER_PACKAGE"
echo "  1. Remove all existing NVIDIA drivers"
echo "  2. Remove orphaned packages and clean the apt cache"
echo "  3. Install $DRIVER_PACKAGE"
echo

if ! $DRY_RUN; then
    read -r -p "Continue? (y/N): " REPLY
    if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
        print_info "Cancelled; no changes made"
        exit 0
    fi
    echo
fi

echo "=== Removing existing drivers ==="
# A machine with no NVIDIA packages installed is a valid starting state, so a
# purge that matches nothing is not an error.
if $DRY_RUN; then
    printf "[DRY-RUN] %s\n" "sudo apt-get --purge --yes remove nvidia-*"
elif sudo apt-get --purge --yes remove 'nvidia-*'; then
    print_ok "Existing NVIDIA packages removed"
else
    print_warn "No NVIDIA packages matched the purge; continuing"
fi

run sudo apt-get --purge --yes autoremove
run sudo apt-get clean
print_ok "Orphaned packages removed and cache cleaned"
echo

echo "=== Installing $DRIVER_PACKAGE ==="
run sudo apt-get update
run sudo apt-get install --yes "$DRIVER_PACKAGE"
print_ok "Driver installed: $DRIVER_PACKAGE"
echo

if $DRY_RUN; then
    print_info "No changes were made"
    exit 0
fi

print_warn "A reboot is required before the new driver loads"
echo "  Verify after reboot: nvidia-smi"
read -r -p "Reboot now? (y/N): " REPLY
if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    print_info "Rebooting"
    sudo reboot
else
    print_warn "Reboot before using the GPU; nvidia-smi will fail until you do"
fi
