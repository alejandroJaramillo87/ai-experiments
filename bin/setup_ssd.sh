#!/bin/bash
set -e

# === Configuration ===
DEVICE="/dev/nvme0n1p2"  # Use the 1.8T partition instead of the whole disk
MOUNT_POINT="/mnt/ai-data"
LABEL="AIENGINEERING"
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
        "$@"
    fi
}

# === Device Verification ===
echo "=== Verifying target device ==="
if [[ ! -b "$DEVICE" ]]; then
    echo "ERROR: Device $DEVICE does not exist"
    exit 1
fi

echo "Device info:"
lsblk "$DEVICE"

if [[ "$DRY_RUN" == false ]]; then
    read -p "Are you sure you want to format $DEVICE? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Operation cancelled."
        exit 1
    fi
fi

# === Start Script ===
echo "=== Target Device: ${DEVICE} ==="
echo "=== Mount Point: ${MOUNT_POINT} ==="

# === Format Device ===
echo "=== Formatting ${DEVICE} as ext4 with AI workload optimizations ==="
# run "sudo umount ${DEVICE} || true"
run "sudo mkfs.ext4 -F -L $LABEL -E lazy_itable_init=0,lazy_journal_init=0 -O ^has_journal,extent,huge_file,flex_bg,uninit_bg,dir_nlink,extra_isize ${DEVICE}"

# === Mount Drive ===
echo "=== Creating and mounting at ${MOUNT_POINT} ==="
run "sudo mkdir -p $MOUNT_POINT"
run "sudo mount ${DEVICE} $MOUNT_POINT"

# === Filesystem Optimizations ===
echo "=== Optimizing filesystem for AI workloads ==="
run "sudo tune2fs -o journal_data_writeback $DEVICE"

# === Directory Structure ===
echo "=== Creating directory structure ==="
run "sudo mkdir -p $MOUNT_POINT/{models/{hf,gguf,onnx,safetensors,quantized},datasets/{commoncrawl,audio,vision,parquet,mmap,preprocessed},embeddings/{chroma,faiss,doc_chunks,vector_cache},cache/{huggingface,torch,webui,pip},logs/{training,inference,crash_dumps},workspace/{experiments,checkpoints,temp}}"

# === Permissions ===
echo "=== Setting ownership to current user ==="
run "sudo chown -R $(whoami):$(whoami) $MOUNT_POINT"

# === fstab Entry ===
echo "=== Updating /etc/fstab ==="
if $DRY_RUN; then
    UUID="dummy-uuid-for-dry-run"
    echo "[DRY-RUN] Would get UUID for $DEVICE"
else
    if ! UUID=$(sudo blkid -s UUID -o value ${DEVICE}); then
        echo "ERROR: Could not get UUID for $DEVICE"
        exit 1
    fi
fi

FSTAB_LINE="UUID=${UUID} $MOUNT_POINT ext4 defaults,noatime,nodiratime,commit=60,barrier=0 0 2"

if $DRY_RUN; then
    echo "[DRY-RUN] Would add to /etc/fstab: $FSTAB_LINE"
else
    if ! grep -q "$UUID" /etc/fstab; then
        echo "$FSTAB_LINE" | sudo tee -a /etc/fstab
    fi
fi

# === Create target directories for symlinks ===
echo "=== Creating target directories for symlinks ==="
run "mkdir -p $MOUNT_POINT/cache/{huggingface,torch,pip}"
run "mkdir -p $MOUNT_POINT/models/gguf"
run "mkdir -p $MOUNT_POINT/models/ollama"  # Uncomment if you want separate Ollama dir

# === Symbolic Links ===
echo "=== Creating symbolic links for model caches ==="
run "mkdir -p ~/.cache"
run "rm -rf ~/.cache/huggingface"
run "ln -s $MOUNT_POINT/cache/huggingface ~/.cache/huggingface"
run "rm -rf ~/.cache/torch"
run "ln -s $MOUNT_POINT/cache/torch ~/.cache/torch"
run "rm -rf ~/.cache/pip"
run "ln -s $MOUNT_POINT/cache/pip ~/.cache/pip"

run "mkdir -p ~/.ollama"
run "rm -rf ~/.ollama/models"
run "ln -s $MOUNT_POINT/models/ollama ~/.ollama/models" 

run "mkdir -p ~/.local/share"
run "rm -rf ~/.local/share/models"
run "ln -s $MOUNT_POINT/models ~/.local/share/models"

echo "=== Script Complete ==="
if $DRY_RUN; then
    echo "No changes were made to the system."
else
    echo "✅ Your AI SSD is fully configured and optimized for AI workloads."
    echo "📁 Available directories:"
    echo "   - Models: $MOUNT_POINT/models/{hf,gguf,onnx,safetensors,quantized}"
    echo "   - Datasets: $MOUNT_POINT/datasets/{commoncrawl,audio,vision,parquet,mmap,preprocessed}"
    echo "   - Embeddings: $MOUNT_POINT/embeddings/{chroma,faiss,doc_chunks,vector_cache}"
    echo "   - Cache: $MOUNT_POINT/cache/{huggingface,torch,webui,pip}"
    echo "   - Workspace: $MOUNT_POINT/workspace/{experiments,checkpoints,temp}"
    echo "   - Logs: $MOUNT_POINT/logs/{training,inference,crash_dumps}"
fi