# Storage Configuration

Dual-SSD architecture configuration for AI workstation with dedicated OS and data drives optimized for model loading and inference workloads.

## Table of Contents

- [Storage Architecture](#storage-architecture)
- [Hardware Configuration](#hardware-configuration)
- [AI Data Drive Setup](#ai-data-drive-setup)
- [Filesystem Optimizations](#filesystem-optimizations)
- [Directory Structure](#directory-structure)
- [Symbolic Links](#symbolic-links)
- [Persistent Mounting](#persistent-mounting)
- [Automation Script](#automation-script)
- [Verification](#verification)
- [Maintenance](#maintenance)

## Storage Architecture

The workstation employs a dual-SSD configuration to isolate system operations from AI workloads:

| Drive | Model | Capacity | Mount Point | Purpose |
|-------|-------|----------|-------------|---------|
| nvme1n1 | Samsung 990 Pro | 2TB | / | Operating system, applications |
| nvme0n1p2 | Samsung 990 EVO | 1.8TB | /mnt/ai-data | AI models, datasets, caches |

### Performance Isolation

Separating OS and AI data prevents:
- I/O contention during model loading
- System slowdown during large file operations
- Cache conflicts between system and AI processes

### PCIe Lane Management

The configuration preserves all PCIe 5.0 x16 lanes for the RTX 5090. Additional M.2 drives would reduce GPU bandwidth from x16 to x8, impacting inference performance by up to 15%.

## Hardware Configuration

### Primary Drive: Samsung 990 Pro 2TB

System drive specifications:
- Interface: PCIe 5.0 x4
- Sequential read: 7,450 MB/s
- Mount point: / (root filesystem)
- Filesystem: ext4 with default options
- Usage: Ubuntu 24.04, Docker, development tools

### Secondary Drive: Samsung 990 EVO 1.8TB Partition

AI data drive specifications:
- Interface: PCIe 4.0 x4
- Sequential read: 5,000 MB/s
- Device: /dev/nvme0n1p2
- Mount point: /mnt/ai-data
- Filesystem: ext4 with AI-optimized settings
- Usage: Models, datasets, caches, workspaces

## AI Data Drive Setup

### Device Preparation

Format the data partition with ext4:

```bash
# Format with AI workload optimizations
sudo mkfs.ext4 -F -L AIENGINEERING \
  -E lazy_itable_init=1,lazy_journal_init=1 \
  -O extent,huge_file,flex_bg,uninit_bg,dir_nlink,extra_isize \
  /dev/nvme0n1p2
```

Options explained:
- `lazy_itable_init`: Faster formatting for large drives
- `huge_file`: Support for files >2TB
- `extent`: Efficient storage of large contiguous files
- `flex_bg`: Improved allocation for large files

### Mount Configuration

Create and mount the data drive:

```bash
# Create mount point
sudo mkdir -p /mnt/ai-data

# Mount the drive
sudo mount /dev/nvme0n1p2 /mnt/ai-data

# Set ownership
sudo chown -R $(whoami):$(whoami) /mnt/ai-data
```

## Filesystem Optimizations

### Journal Configuration

Optimize ext4 journal for large file operations:

```bash
# Set ordered journal mode
sudo tune2fs -o journal_data_ordered /dev/nvme0n1p2

# Configure 128MB journal size
sudo tune2fs -J size=128 /dev/nvme0n1p2
```

Journal settings balance reliability with performance:
- Ordered mode: Metadata committed after data write
- 128MB size: Reduced overhead for large files
- Commit interval: 30 seconds (via mount options)

### Mount Options

Optimized mount parameters in /etc/fstab:

```
defaults,noatime,commit=30,data=ordered
```

Options:
- `noatime`: Skip access time updates (15% performance gain)
- `commit=30`: Delay journal commits to 30 seconds
- `data=ordered`: Ensure data integrity with good performance

## Directory Structure

Standardized organization for AI assets:

```
/mnt/ai-data/
├── models/
│   ├── hf/           # HuggingFace models
│   ├── gguf/         # GGUF quantized models
│   ├── quantized/    # Other quantization formats
│   └── trtllm-engine/ # TensorRT-LLM engines
├── datasets/
│   ├── audio/        # Speech datasets
│   ├── commoncrawl/  # Web corpus data
│   ├── vision/       # Image datasets
│   ├── parquet/      # Columnar data
│   ├── mmap/         # Memory-mapped files
│   └── preprocessed/ # Processed datasets
├── embeddings/
│   ├── chroma/       # Chroma DB storage
│   ├── faiss/        # FAISS indices
│   ├── doc_chunks/   # Document chunks
│   └── vector_cache/ # Cached embeddings
├── cache/            # Mixed system and AI caches
│   ├── huggingface/  # HF transformers cache
│   ├── torch/        # PyTorch cache
│   ├── pip/          # pip download cache
│   ├── pypoetry/     # Poetry cache
│   ├── nvidia/       # NVIDIA GPU cache
│   ├── webui/        # Web UI caches
│   └── ...           # Other system caches
├── workspace/
│   ├── experiments/  # Experiment tracking
│   ├── checkpoints/  # Training checkpoints
│   └── temp/         # Temporary files
└── logs/
    ├── training/     # Training logs
    ├── inference/    # Inference logs
    └── crash_dumps/  # Debug information
```

## Symbolic Links

**Note**: The following symbolic links are optional. Currently, cache directories exist as regular directories in `~/.cache/`. Creating symbolic links redirects cache storage to the data drive for centralized management.

Map system cache directories to data drive (if not already configured):

### HuggingFace Cache

```bash
# Check current status
stat -c "%F" ~/.cache/huggingface 2>/dev/null || echo "Does not exist"

# Create symbolic link (optional)
mkdir -p ~/.cache
rm -rf ~/.cache/huggingface  # Warning: removes existing cache
ln -s /mnt/ai-data/cache/huggingface ~/.cache/huggingface
```

### PyTorch Cache

```bash
# Check current status
stat -c "%F" ~/.cache/torch 2>/dev/null || echo "Does not exist"

# Create symbolic link (optional)
rm -rf ~/.cache/torch  # Warning: removes existing cache
ln -s /mnt/ai-data/cache/torch ~/.cache/torch
```

### Pip Cache

```bash
# Check current status
stat -c "%F" ~/.cache/pip 2>/dev/null || echo "Does not exist"

# Create symbolic link (optional)
rm -rf ~/.cache/pip  # Warning: removes existing cache
ln -s /mnt/ai-data/cache/pip ~/.cache/pip
```

### Ollama Models

```bash
mkdir -p ~/.ollama
rm -rf ~/.ollama/models
ln -s /mnt/ai-data/models/ollama ~/.ollama/models
```

### Local Share Models

```bash
mkdir -p ~/.local/share
rm -rf ~/.local/share/models
ln -s /mnt/ai-data/models ~/.local/share/models
```

## Persistent Mounting

### Get UUID

```bash
# Find UUID for the data partition
sudo blkid -s UUID -o value /dev/nvme0n1p2
```

### Configure /etc/fstab

Add entry for automatic mounting at boot:

```bash
# Add to /etc/fstab
UUID=<your-uuid> /mnt/ai-data ext4 defaults,noatime,commit=30,data=ordered 0 2
```

Fields explained:
- UUID: Persistent identifier for the partition
- /mnt/ai-data: Mount point
- ext4: Filesystem type
- Mount options: Performance and reliability settings
- 0: Dump frequency (disabled)
- 2: fsck order (check after root filesystem)

## Automation Script

### Usage

Run the automated setup script:

```bash
# Execute setup
./scripts/setup/setup_data_ssd.sh

# Dry run mode (preview changes)
./scripts/setup/setup_data_ssd.sh --dry-run
```

### Script Features

Safety and automation features:
- Device verification before formatting
- User confirmation prompt
- Dry-run mode for testing
- Automatic directory creation
- Symbolic link configuration
- fstab entry management

### Script Location

```bash
/home/alejandro/workspace/ai-experiments/scripts/setup/setup_data_ssd.sh
```

## Verification

### Check Mount Status

```bash
# Verify mount
mount | grep ai-data

# Check disk usage
df -h /mnt/ai-data

# Verify filesystem
sudo tune2fs -l /dev/nvme0n1p2 | grep -E "Journal|Block size"
```

### Test Performance

```bash
# Write speed test
dd if=/dev/zero of=/mnt/ai-data/test.bin bs=1G count=10 oflag=direct

# Read speed test
dd if=/mnt/ai-data/test.bin of=/dev/null bs=1G iflag=direct

# Clean up
rm /mnt/ai-data/test.bin
```

### Verify Cache Configuration

```bash
# Check if using symbolic links or regular directories
for dir in huggingface torch pip; do
    if [ -L ~/.cache/$dir ]; then
        echo "$dir: symlink -> $(readlink ~/.cache/$dir)"
    elif [ -d ~/.cache/$dir ]; then
        echo "$dir: regular directory"
    else
        echo "$dir: does not exist"
    fi
done

# Verify HuggingFace cache location
python -c "from transformers import cache; print(cache.default_cache_path)" 2>/dev/null || echo "transformers not installed"
```

## Maintenance

### Disk Usage Monitoring

Monitor space utilization:

```bash
# Overall usage
df -h /mnt/ai-data

# Directory sizes
du -sh /mnt/ai-data/*

# Large files
find /mnt/ai-data -type f -size +1G -exec ls -lh {} \;
```

### Cache Cleanup

Regular cache maintenance (adjust paths based on actual cache location):

```bash
# Determine cache location
CACHE_BASE=$([ -L ~/.cache/huggingface ] && echo "/mnt/ai-data/cache" || echo "$HOME/.cache")

# Clear HuggingFace cache (keep models)
rm -rf $CACHE_BASE/huggingface/hub/temp*

# Clear pip cache
pip cache purge

# Clear PyTorch cache
rm -rf $CACHE_BASE/torch/hub/checkpoints/*
```

### Health Monitoring

Check drive health:

```bash
# SMART status
sudo smartctl -a /dev/nvme0n1

# Filesystem check (unmount first)
sudo umount /mnt/ai-data
sudo fsck.ext4 -f /dev/nvme0n1p2
sudo mount /mnt/ai-data
```

### Backup Considerations

Critical directories to backup:
- /mnt/ai-data/models/quantized (custom quantizations)
- /mnt/ai-data/workspace/experiments (experiment results)
- /mnt/ai-data/embeddings (processed vectors)

Exclude from backups:
- /mnt/ai-data/cache (regeneratable)
- /mnt/ai-data/workspace/temp (temporary files)
- /mnt/ai-data/models/hf (downloadable from HuggingFace)

## Cross-References

- Hardware specifications: [Hardware Build Documentation](../hardware/README.md#storage-layout)
- Backup strategies: [Backup and Recovery Guide](backup_and_recovery.md)
- Automation script: [setup_data_ssd.sh](/scripts/setup/setup_data_ssd.sh)
- Python environment: [Python Setup Guide](python/setup_python.md)
- GPU stack configuration: [GPU Stack Documentation](gpu-stack/README.md)

---

*Last Updated: 2025-09-24*