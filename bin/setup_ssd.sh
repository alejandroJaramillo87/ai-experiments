#!/bin/bash

set -e

# === Configuration ===
DEVICE="/dev/nvme1n1"  # Confirm with lsblk or nvme list
MOUNT_POINT="/mnt/ai-engineering"
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
        eval "$@"
    fi
}

# === Start Script ===
echo "=== Target Device: ${DEVICE} ==="
echo "=== Mount Point: ${MOUNT_POINT} ==="

# === Format Device ===
echo "=== Formatting ${DEVICE} as ext4 ==="
run "sudo umount ${DEVICE} || true"
run "sudo mkfs.ext4 -F -L $LABEL ${DEVICE}"

# === Mount Drive ===
echo "=== Creating and mounting at ${MOUNT_POINT} ==="
run "sudo mkdir -p $MOUNT_POINT"
run "sudo mount ${DEVICE} $MOUNT_POINT"

# === Directory Structure ===
echo "=== Creating directory structure ==="
run "sudo mkdir -p $MOUNT_POINT/{models/{hf,gguf,onnx,safetensors},datasets/{commoncrawl,audio,vision,parquet,mmap},embeddings/{chroma,faiss,doc_chunks,vector_cache},cache/{huggingface,torch,webui},logs/{training,inference,crash_dumps}}"

# === Permissions ===
echo "=== Setting ownership to current user ==="
run "sudo chown -R $(whoami):$(whoami) $MOUNT_POINT"

# === fstab Entry ===
echo "=== Updating /etc/fstab ==="
UUID=$(sudo blkid -s UUID -o value ${DEVICE})
FSTAB_LINE="UUID=${UUID} $MOUNT_POINT ext4 defaults,noatime,nodiratime 0 2"
if $DRY_RUN; then
    echo "[DRY-RUN] Would add to /etc/fstab: $FSTAB_LINE"
else
    if ! grep -q "$UUID" /etc/fstab; then
        echo "$FSTAB_LINE" | sudo tee -a /etc/fstab
    fi
fi

# === Symbolic Links ===
echo "=== Creating symbolic links for model caches ==="
run "mkdir -p ~/.cache"

run "rm -rf ~/.cache/huggingface"
run "ln -s $MOUNT_POINT/cache/huggingface ~/.cache/huggingface"

run "rm -rf ~/.cache/torch"
run "ln -s $MOUNT_POINT/cache/torch ~/.cache/torch"

run "mkdir -p ~/.ollama"
run "rm -rf ~/.ollama/models"
run "ln -s $MOUNT_POINT/models/gguf ~/.ollama/models"

echo "=== Script Complete ==="
if $DRY_RUN; then
    echo "No changes were made to the system."
else
    echo "✅ Your AI SSD is fully configured."
fi
