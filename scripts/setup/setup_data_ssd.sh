#!/bin/bash
#
# setup_data_ssd.sh — provision the AI data drive at /mnt/ai-data.
#
# Three independent phases, each idempotent and individually skippable:
#   1. format    — mkfs.ext4 + tune2fs (destructive; opt-in confirmation)
#   2. mount     — mount and persist an fstab entry by UUID
#   3. provision — directory tree, ownership, and cache symlinks
#
# The device is resolved by filesystem LABEL rather than a hardcoded /dev node,
# because node numbering is not stable across reboots or hardware changes. Pass
# --device to override when the label does not exist yet (first-ever format).

set -euo pipefail

# === Configuration ===
MOUNT_POINT="/mnt/ai-data"
LABEL="AIENGINEERING"
DEVICE=""
DRY_RUN=false
FORCE=false
SKIP_FORMAT=false

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
Usage: setup_data_ssd.sh [OPTIONS]

Options:
  --dry-run        Print every action without executing it
  --skip-format    Skip the destructive format/tune2fs phase
  --device DEV     Target device node (default: resolved from the LABEL)
  --force          Permit formatting a device that already holds a filesystem,
                   and replace real cache directories with symlinks
  --no-color       Disable ANSI color
  --help           Show this message

The format phase refuses to run against a mounted device under all
circumstances, and against a device that already carries a filesystem unless
--force is given. Both refusals are independent of --dry-run.
EOF
}

# === Argument parsing ===
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)     DRY_RUN=true; shift ;;
        --skip-format) SKIP_FORMAT=true; shift ;;
        --force)       FORCE=true; shift ;;
        --device)      DEVICE="${2:-}"; shift 2 ;;
        --no-color)    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''; shift ;;
        --help|-h)     usage; exit 0 ;;
        *)             print_error "Unknown option: $1"; usage >&2; exit 2 ;;
    esac
done

if [[ -n "${NO_COLOR:-}" ]]; then
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''
fi

$DRY_RUN && print_info "Dry run: no changes will be made"

# Execute a command, or print it under --dry-run. Takes argv, not a string:
# `run sudo mkfs.ext4 -F "$DEVICE"`, never `run "sudo mkfs.ext4 ..."`. The
# latter passes one word and bash searches for a command by that literal name.
run() {
    if $DRY_RUN; then
        printf "[DRY-RUN] %s\n" "$*"
    else
        print_run "$*"
        "$@"
    fi
}

# === Device resolution ===
echo "=== Resolving target device ==="

if [[ -z "$DEVICE" ]]; then
    if ! DEVICE=$(blkid -L "$LABEL" 2>/dev/null); then
        print_error "No device carries the filesystem label: $LABEL"
        echo "  Context: the label is how this script finds the drive across reboots"
        echo "  Solution: pass the node explicitly for a first-ever format"
        echo "  Command: lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT"
        exit 1
    fi
    print_ok "Device resolved from label $LABEL: $DEVICE"
else
    print_info "Device supplied on the command line: $DEVICE"
fi

if [[ ! -b "$DEVICE" ]]; then
    print_error "Not a block device: $DEVICE"
    exit 1
fi

lsblk -o NAME,SIZE,TYPE,FSTYPE,LABEL,MOUNTPOINT "$DEVICE"
echo

# === Format phase ===
if $SKIP_FORMAT; then
    print_skip "Format phase (--skip-format)"
else
    echo "=== Formatting $DEVICE ==="

    # Refuse a mounted device unconditionally. Formatting under a live mount
    # corrupts the running system, and --force must not be able to authorize it.
    if findmnt --source "$DEVICE" >/dev/null 2>&1; then
        MOUNTED_AT=$(findmnt -n -o TARGET --source "$DEVICE" | tr '\n' ' ')
        print_error "Device is mounted at: $MOUNTED_AT"
        echo "  Solution: unmount it, or pass --skip-format to provision in place"
        echo "  Command: sudo umount $DEVICE"
        exit 1
    fi

    EXISTING_FS=$(blkid -s TYPE -o value "$DEVICE" 2>/dev/null || true)
    if [[ -n "$EXISTING_FS" ]] && ! $FORCE; then
        print_error "Device already holds a $EXISTING_FS filesystem"
        echo "  Context: formatting destroys every byte on $DEVICE"
        echo "  Solution: pass --force to overwrite, or --skip-format to keep it"
        exit 1
    fi

    if ! $DRY_RUN; then
        print_warn "This destroys all data on $DEVICE"
        read -r -p "Type the device path to confirm: " CONFIRM
        if [[ "$CONFIRM" != "$DEVICE" ]]; then
            print_error "Confirmation did not match; no changes made"
            exit 1
        fi
    fi

    # Journal kept enabled and sized at 128MB; ordered data mode for large-file
    # reliability. tune2fs runs before the mount because -J requires the
    # filesystem to be offline.
    run sudo mkfs.ext4 -L "$LABEL" \
        -E lazy_itable_init=1,lazy_journal_init=1 \
        -O extent,huge_file,flex_bg,uninit_bg,dir_nlink,extra_isize \
        "$DEVICE"
    run sudo tune2fs -o journal_data_ordered "$DEVICE"
    run sudo tune2fs -J size=128 "$DEVICE"
    print_ok "Filesystem created and tuned"
fi
echo

# === Mount phase ===
echo "=== Mounting at $MOUNT_POINT ==="
run sudo mkdir -p "$MOUNT_POINT"

if findmnt --mountpoint "$MOUNT_POINT" >/dev/null 2>&1; then
    print_skip "Already mounted: $MOUNT_POINT"
else
    run sudo mount "$DEVICE" "$MOUNT_POINT"
    print_ok "Mounted $DEVICE at $MOUNT_POINT"
fi

# fstab is keyed by UUID, never by device node, for the same stability reason
# the device is resolved by label above.
if $DRY_RUN; then
    UUID="<resolved-at-run-time>"
    printf "[DRY-RUN] %s\n" "blkid -s UUID -o value $DEVICE"
else
    if ! UUID=$(sudo blkid -s UUID -o value "$DEVICE"); then
        print_error "Could not read the UUID of $DEVICE"
        exit 1
    fi
fi

FSTAB_LINE="UUID=${UUID} $MOUNT_POINT ext4 defaults,noatime,commit=30,data=ordered 0 2"
if $DRY_RUN; then
    printf "[DRY-RUN] %s\n" "append to /etc/fstab: $FSTAB_LINE"
elif grep -q "$UUID" /etc/fstab; then
    print_skip "fstab entry already present for UUID $UUID"
else
    echo "$FSTAB_LINE" | sudo tee -a /etc/fstab >/dev/null
    print_ok "fstab entry added"
fi
echo

# === Provision phase ===
echo "=== Creating directory structure ==="

# The tree the current stack actually uses. models/ is bound read-only into
# every inference and training service; models/trained is the narrow writable
# window for exported GGUFs. training/ and interp/ are identity-mounted rw by
# docker-compose.training.yaml and docker-compose.interp.yaml respectively.
DIRS=(
    "models/hf"
    "models/gguf"
    "models/trained"
    "models/checkpoints"
    "training/datasets"
    "training/shards"
    "training/tokenizers"
    "training/runs"
    "training/verify"
    "interp/activations"
    "interp/saes"
    "interp/probes"
    "interp/monitor"
    "interp/vectors"
    "interp/organisms"
    "interp/hf-cache"
    "interp/logs"
    "logs/training"
    "logs/interp"
    "logs/inference"
    "logs/crash_dumps"
    "benchmarks/gpu"
    "benchmarks/perplexity"
    "benchmarks/nsys"
    "cache/huggingface"
    "cache/torch"
    "cache/pip"
    "cache/uv"
)
for d in "${DIRS[@]}"; do
    run sudo mkdir -p "$MOUNT_POINT/$d"
done
print_ok "Directory tree created (${#DIRS[@]} paths)"

run sudo chown -R "$(whoami):$(whoami)" "$MOUNT_POINT"
print_ok "Ownership set to $(whoami)"
echo

# === Cache symlinks ===
echo "=== Linking AI caches to the data drive ==="

# Only AI caches are redirected. Desktop caches (browser, GNOME, thumbnails)
# stay on the OS drive: putting them here defeats the drive's purpose of
# isolating AI I/O from system I/O.
if [[ -L "$HOME/.cache" ]]; then
    print_error "$HOME/.cache is itself a symlink to $(readlink "$HOME/.cache")"
    echo "  Context: that redirects every desktop cache onto the AI drive"
    echo "  Solution: replace it with a real directory, then re-run this script"
    echo "  Command: rm $HOME/.cache && mkdir -p $HOME/.cache"
    exit 1
fi

run mkdir -p "$HOME/.cache"

link_cache() {
    local name="$1"
    local target="$MOUNT_POINT/cache/$name"
    local link="$HOME/.cache/$name"

    if [[ -L "$link" ]]; then
        if [[ "$(readlink -f "$link")" == "$target" ]]; then
            print_skip "$name (already linked)"
            return
        fi
        run rm "$link"
    elif [[ -d "$link" ]]; then
        # A real directory holds data. Never delete it silently.
        if ! $FORCE; then
            print_warn "$name: real directory, not a symlink; left untouched"
            echo "  Solution: move its contents onto the data drive, then re-run"
            echo "  Command: rsync -a --remove-source-files $link/ $target/ && rmdir $link"
            return
        fi
        run rsync -a --remove-source-files "$link/" "$target/"
        run rm -rf "$link"
    fi
    run ln -s "$target" "$link"
    print_ok "$name -> $target"
}

for c in huggingface torch pip uv; do
    link_cache "$c"
done
echo

echo "=== Complete ==="
if $DRY_RUN; then
    print_info "No changes were made"
else
    print_ok "Data drive provisioned at $MOUNT_POINT"
    echo "  Verify: ./scripts/utils/storage_check.sh"
fi
