# Storage configuration

Dual-SSD architecture configuration for AI workstation with dedicated OS and data drives optimized for model loading and inference workloads.

## Table of Contents

- [Storage architecture](#storage-architecture)
- [Hardware configuration](#hardware-configuration)
- [AI data drive setup](#ai-data-drive-setup)
- [Filesystem optimizations](#filesystem-optimizations)
- [Directory structure](#directory-structure)
- [Symbolic links](#symbolic-links)
- [Persistent mounting](#persistent-mounting)
- [Automation script](#automation-script)
- [Verification](#verification)
- [Maintenance](#maintenance)
- [Cross-references](#cross-references)

## Storage architecture

The workstation employs a dual-SSD configuration to isolate system operations from AI workloads:

| Drive | Model | Capacity | Mount Point | Purpose |
|-------|-------|----------|-------------|---------|
| nvme0n1 | Samsung 990 EVO Plus 1TB | 931.5G | / | Operating system, applications |
| nvme1n1p2 | Samsung 990 PRO 2TB | 1.8T | /mnt/ai-data | AI models, training and interp tiers, caches |

**Warning!** The device node is not stable across reboots or hardware changes. Every
command below names `/dev/nvme1n1p2` for concreteness, but resolve the drive by its
filesystem label before running anything destructive:

```bash
blkid -L AIENGINEERING
```

### Performance isolation

Separating OS and AI data prevents:
- I/O contention during model loading
- System slowdown during large file operations
- Cache conflicts between system and AI processes

### PCIe lane management

The configuration preserves all PCIe 5.0 x16 lanes for the RTX 5090. Additional M.2 drives would reduce GPU bandwidth from x16 to x8, reducing host-to-device transfer bandwidth.

## Hardware configuration

### OS drive: Samsung 990 EVO Plus 1TB

System drive specifications:
- Device: /dev/nvme0n1 (931.5G; p1 is the 1G ESP, p2 the 930.5G root filesystem)
- Interface: PCIe 5.0 x4 (unverified, see note below)
- Sequential read: 7,450 MB/s (unverified, see note below)
- Mount point: / (root filesystem)
- Filesystem: ext4 with default options
- Usage: Ubuntu 24.04, Docker, development tools

### AI data drive: Samsung 990 PRO 2TB

AI data drive specifications:
- Interface: PCIe 4.0 x4 (unverified, see note below)
- Sequential read: 5,000 MB/s (unverified, see note below)
- Device: /dev/nvme1n1p2 (1.8T of the 2TB drive; p1 is a 1G unmounted vfat partition)
- Mount point: /mnt/ai-data
- Filesystem: ext4 with AI-optimized settings
- Usage: Model weights, the training and interp tiers, observability logs, host caches

**Note**: The interface generation and sequential-read figures above are vendor claims
carried over from earlier revisions of this document, not measurements taken on this
machine. They conflict with [Hardware Build Documentation](../hardware/README.md#storage-layout),
which states PCIe 5.0 x4 and 12,400 MB/s for the AI data drive. Only the model names,
device nodes, capacities, and mount points are confirmed, from `lsblk` and
`blkid -L AIENGINEERING`. Resolve the throughput figures with a measurement before
relying on them.

## AI data drive setup

### Device preparation

Format the data partition with ext4:

```bash
# Format with AI workload optimizations. The -F force flag is omitted so that
# mkfs.ext4 refuses a mounted device and prompts before overwriting an existing
# filesystem. The unconditional refusal for a device that already carries a
# filesystem is implemented by /scripts/setup/setup_data_ssd.sh (lines 116-130),
# not by mkfs.
sudo mkfs.ext4 -L AIENGINEERING \
  -E lazy_itable_init=1,lazy_journal_init=1 \
  -O extent,huge_file,flex_bg,uninit_bg,dir_nlink,extra_isize \
  /dev/nvme1n1p2
```

Prefer `./scripts/setup/setup_data_ssd.sh` over running these by hand. It resolves the
device by label, refuses a mounted target, and requires the device path typed back
before formatting.

Options explained:
- `lazy_itable_init`: Faster formatting for large drives
- `huge_file`: Support for files >2TB
- `extent`: Efficient storage of large contiguous files
- `flex_bg`: Improved allocation for large files

### Mount configuration

Create and mount the data drive:

```bash
# Create mount point
sudo mkdir -p /mnt/ai-data

# Mount the drive
sudo mount /dev/nvme1n1p2 /mnt/ai-data

# Set ownership
sudo chown -R $(whoami):$(whoami) /mnt/ai-data
```

## Filesystem optimizations

### Journal configuration

Optimize ext4 journal for large file operations:

```bash
# Set ordered journal mode
sudo tune2fs -o journal_data_ordered /dev/nvme1n1p2

# Configure 128MB journal size
sudo tune2fs -J size=128 /dev/nvme1n1p2
```

Journal settings balance reliability with performance:
- Ordered mode: Metadata committed after data write
- 128MB size: Reduced overhead for large files
- Commit interval: 30 seconds (via mount options)

### Mount options

Optimized mount parameters in /etc/fstab:

```
defaults,noatime,commit=30,data=ordered
```

Options:
- `noatime`: Skip access time updates, removing a metadata write per read
- `commit=30`: Delay journal commits to 30 seconds
- `data=ordered`: Ensure data integrity with good performance

## Directory structure

Five top-level directories. The compose files are the source of truth for what is
mounted where; this section follows them. Entries marked "on demand" are created by
the code that writes them and are absent on a fresh machine.

```
/mnt/ai-data/
├── models/           # All model weights; read-only in every service
│   ├── hf/           # HuggingFace format, one directory per model
│   ├── gguf/         # GGUF quantized models
│   ├── trained/      # Exported GGUFs; the one writable window
│   └── checkpoints/  # Imported third-party checkpoints
├── training/         # Training tier; identity-mounted read-write
│   ├── datasets/     # Prepared datasets
│   ├── shards/       # Tokenized shards with manifests
│   ├── tokenizers/   # Trained tokenizers
│   ├── runs/         # Checkpoints and run outputs
│   ├── verify/       # Calibration records from verification runs
│   ├── eval/         # Evaluation artifacts (on demand)
│   ├── staging/      # Large intermediates, e.g. export (on demand)
│   └── .train.lock   # Single-GPU-consumer lock (on demand)
├── interp/           # Interpretability tier; identity-mounted read-write
│   ├── activations/  # Captured activations
│   ├── saes/         # Sparse autoencoders
│   ├── probes/       # Trained probes
│   ├── monitor/      # Probe monitoring output
│   ├── vectors/      # Steering vectors; read-only in harness-serve
│   ├── organisms/    # Model organisms
│   ├── hf-cache/     # Interp HF_HOME (persistent, unlike the training tier)
│   └── logs/         # Per-request JSONL written by harness-serve
├── benchmarks/       # Benchmark output; no service reads it back
│   ├── gpu/          # scripts/benchmark/benchmark_gpu.sh (BENCH_RESULTS_DIR)
│   ├── perplexity/   # scripts/benchmark/test_perplexity.sh (BENCH_RESULTS_DIR)
│   └── nsys/         # Nsight reports, written in-container when NSYS_PROFILE=true
├── logs/             # Structured observability records (OBS_LOG_DIR)
│   ├── training/     # One directory per CLI: event files plus runs.jsonl
│   ├── interp/       # Interp CLI records
│   ├── harness/      # Rust harness records, host-side runs (on demand)
│   ├── inference/
│   └── crash_dumps/
└── cache/            # Host tooling, plus the prepare service's HuggingFace cache
    ├── huggingface/  # Also bound rw into the prepare service as HF_HOME
    ├── torch/
    ├── pip/
    └── uv/
```

### What mounts what

| Directory | Services | Mode |
|-----------|----------|------|
| `models/` | training, export, interp, harness, llama-cpu, llama-gpu, vllm-gpu | read-only |
| `models/trained/` | training, export | read-write |
| `training/` | training, export, interp | read-write |
| `interp/` | interp | read-write |
| `interp/vectors/` | harness-serve | read-only |
| `interp/logs/` | harness-serve | read-write |
| `logs/` | training, export, interp, prepare | read-write |
| `benchmarks/nsys/` | llama-gpu, vllm-gpu | read-write |
| `models/hf/` | prepare | read-write |
| `cache/huggingface/` | prepare | read-write |
| `cache/torch/`, `cache/pip/`, `cache/uv/` | none | n/a |

The `models/` tree is read-only everywhere, with `models/trained/` bound separately
as a narrow writable window for exported GGUFs. The `harness` service (compose
service key in `/docker-compose.rust-harness.yaml`; `rust-harness` is its
container and image name) mounts
`models/` but does not collect observability records inside the container; its
`logs/harness/` entries come from host-side runs.

### Caches

| Service | `HF_HOME` | Persists between runs |
|---------|-----------|-----------------------|
| prepare | `/mnt/ai-data/cache/huggingface` | Yes |
| interp | `/mnt/ai-data/interp/hf-cache` | Yes |
| training | `/app/cache` (tmpfs, 2GB) | No |
| export | `/app/cache` (tmpfs, 8GB) | No |
| vllm-gpu | `/app/cache` (tmpfs, 2GB) | No |

Only `prepare` and `interp` keep a HuggingFace cache across runs. The `training`
tmpfs is deliberately small and expected to stay empty: that service resolves no
hub identifiers, so anything appearing there indicates a path still reaching the
hub.

`cache/huggingface/` is shared between host tooling and the `prepare` service, so a
model already fetched by either is a cache hit for the other. The remaining
subdirectories (`torch`, `pip`, `uv`) are host-side only. All of it is
regeneratable — exclude `cache/` from backups.

Only AI caches belong here: redirecting all of `~/.cache` also pulls browser,
desktop, and thumbnail caches onto this drive, which defeats the I/O isolation
described above.

## Symbolic links

Four AI caches are redirected to the data drive, each linked individually:
`huggingface`, `torch`, `pip`, and `uv`.

**Warning!** Do not symlink `~/.cache` as a whole. Doing so redirects every desktop
cache — browser, GNOME, thumbnails, shader caches — onto the data drive, which
contradicts the I/O isolation this drive exists to provide. `~/.cache` must remain a
real directory containing per-cache symlinks. `./scripts/utils/storage_check.sh`
warns when it is a symlink.

**Warning!** The loop below deletes any existing cache directory rather than
migrating it. Move the contents onto the data drive first, or use the setup
script, which migrates instead of deleting.

```bash
# Verify ~/.cache is a real directory, not a symlink
[ -L ~/.cache ] && echo "WRONG: ~/.cache is a symlink" || echo "OK: real directory"

# Link each AI cache individually
mkdir -p ~/.cache
for c in huggingface torch pip uv; do
    mkdir -p "/mnt/ai-data/cache/$c"
    rm -rf ~/.cache/"$c"          # Warning: destroys the existing cache contents
    ln -s "/mnt/ai-data/cache/$c" ~/.cache/"$c"
done
```

`./scripts/setup/setup_data_ssd.sh --skip-format` links the same four caches, but
handles existing directories differently: it leaves a real cache directory
untouched and prints the migration command, and with `--force` it rsyncs the
contents onto the data drive before removing the directory. Neither path
discards data, unlike the manual loop above.

## Persistent mounting

### Get UUID

```bash
# Find UUID for the data partition
sudo blkid -s UUID -o value /dev/nvme1n1p2
```

### Configure /etc/fstab

Add entry for automatic mounting at boot:

Line to append to `/etc/fstab`:

```
UUID=<uuid> /mnt/ai-data ext4 defaults,noatime,commit=30,data=ordered 0 2
```

Fields explained:
- UUID: Persistent identifier for the partition
- /mnt/ai-data: Mount point
- ext4: Filesystem type
- Mount options: Performance and reliability settings
- 0: Dump frequency (disabled)
- 2: fsck order (check after root filesystem)

## Automation script

### Usage

Run the automated setup script:

```bash
# Execute setup
./scripts/setup/setup_data_ssd.sh

# Print every action without executing it
./scripts/setup/setup_data_ssd.sh --dry-run
```

Options:

| Option | Effect |
|--------|--------|
| `--dry-run` | Print every action without executing it |
| `--skip-format` | Skip the destructive format and tune2fs phase |
| `--device DEV` | Target device node; default is resolved from the `AIENGINEERING` label |
| `--force` | Permit formatting a device that already holds a filesystem, and replace real cache directories with symlinks (contents are rsynced to the data drive first) |
| `--no-color` | Disable ANSI color |
| `--help` | Print usage and exit 0 |

`--dry-run` is not a pure preview. Device resolution runs first and is
unconditional: the script exits 1 before previewing anything if the
`AIENGINEERING` label resolves to nothing, or if the `--device` argument is not
a block device. The refusal to format a mounted device, and the refusal to
format a device that already carries a filesystem without `--force`, are also
independent of `--dry-run`.

### Script features

Safety and automation features:
- Device verification before formatting
- User confirmation prompt
- Dry-run mode for testing
- Automatic directory creation
- Symbolic link configuration
- fstab entry management

### Script location

```
/scripts/setup/setup_data_ssd.sh
```

## Verification

### Check mount status

```bash
# Verify mount
mount | grep ai-data

# Check disk usage
df -h /mnt/ai-data

# Verify filesystem
sudo tune2fs -l /dev/nvme1n1p2 | grep -E "Journal|Block size"
```

### Test performance

```bash
# Write speed test
dd if=/dev/zero of=/mnt/ai-data/test.bin bs=1G count=10 oflag=direct

# Read speed test
dd if=/mnt/ai-data/test.bin of=/dev/null bs=1G iflag=direct

# Clean up
rm /mnt/ai-data/test.bin
```

### Verify cache configuration

```bash
# Check if using symbolic links or regular directories
for dir in huggingface torch pip uv; do
    if [ -L ~/.cache/$dir ]; then
        echo "$dir: symlink -> $(readlink ~/.cache/$dir)"
    elif [ -d ~/.cache/$dir ]; then
        echo "$dir: regular directory"
    else
        echo "$dir: does not exist"
    fi
done
```

## Maintenance

### Disk usage monitoring

Monitor space utilization:

```bash
# Overall usage
df -h /mnt/ai-data

# Directory sizes
du -sh /mnt/ai-data/*

# Large files
find /mnt/ai-data -type f -size +1G -exec ls -lh {} \;
```

### Cache cleanup

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

### Health monitoring

Check drive health:

```bash
# SMART status
sudo smartctl -a /dev/nvme1n1

# Filesystem check (unmount first)
sudo umount /mnt/ai-data
sudo fsck.ext4 -f /dev/nvme1n1p2
sudo mount /mnt/ai-data
```

### Backup considerations

Critical directories to backup:
- /mnt/ai-data/models/trained (exported GGUFs; not reproducible without a rerun)
- /mnt/ai-data/training/runs (checkpoints and run outputs)
- /mnt/ai-data/training/verify (calibration records)
- /mnt/ai-data/interp/saes, /interp/probes, /interp/vectors (trained artifacts)
- /mnt/ai-data/logs (observability records; the audit trail for completed runs)

Exclude from backups:
- /mnt/ai-data/cache (regeneratable)
- /mnt/ai-data/training/staging (large intermediates)
- /mnt/ai-data/interp/activations (recapturable, and large)
- /mnt/ai-data/models/hf (downloadable from HuggingFace at a pinned revision)

## Cross-references

- Hardware specifications: [Hardware Build Documentation](../hardware/README.md#storage-layout)
- Backup strategies: [Backup and Recovery Guide](backup_and_recovery.md)
- Automation script: [setup_data_ssd.sh](/scripts/setup/setup_data_ssd.sh)
- Python environment: [Python Setup Guide](python/setup_python.md)
- GPU stack configuration: [GPU Stack Documentation](gpu-stack/README.md)
