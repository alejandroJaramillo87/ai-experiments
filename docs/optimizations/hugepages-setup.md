# Huge Pages Setup and Management

Complete guide for configuring and managing huge pages for AI model inference.

## Overview

The huge pages system reduces memory management overhead by using 2MB pages instead of 4KB pages. This is especially beneficial for large AI models (>15GB).

## Initial Setup

### 1. Allocate Huge Pages

```bash
# Calculate pages needed: model_size_GB * 512
# For 30GB model: 30 * 512 = 15360 pages

# Allocate immediately
sudo sysctl vm.nr_hugepages=15360

# Make persistent across reboots
echo "vm.nr_hugepages=15360" | sudo tee -a /etc/sysctl.conf

# Verify allocation
grep "^HugePages" /proc/meminfo
```

### 2. Mount Hugetlbfs Filesystem

```bash
# Create mount point
sudo mkdir -p /mnt/models-hugepages

# Mount hugetlbfs
sudo mount -t hugetlbfs -o pagesize=2M,size=30G none /mnt/models-hugepages

# Add to /etc/fstab for persistence
echo "none /mnt/models-hugepages hugetlbfs pagesize=2M,size=30G 0 0" | sudo tee -a /etc/fstab

# Verify mount
mount | grep hugetlbfs
```

### 3. Install Systemd Service (Optional)

```bash
# Copy service file
sudo cp scripts/hugepages-model-manager.service /etc/systemd/system/

# Enable for automatic model loading at boot
sudo systemctl daemon-reload
sudo systemctl enable hugepages-model-manager.service
```

## Model Management

### Loading Models

```bash
# List available models
sudo /scripts/optimizations/manage-hugepages-models.sh list

# Load a model into huge pages
sudo /scripts/optimizations/manage-hugepages-models.sh load gguf/model.gguf

# Check loaded models
sudo /scripts/optimizations/manage-hugepages-models.sh loaded

# View complete status
sudo /scripts/optimizations/manage-hugepages-models.sh status
```

### Container Configuration

Update MODEL_PATH in docker-compose.yaml:

```yaml
environment:
  # For model loaded in huge pages
  - MODEL_PATH=/hugepages/Qwen3-Coder-30B-A3B-Instruct-IQ4_XS.gguf
```

### Dynamic Model Switching

```bash
# Stop containers
docker-compose stop llama-cpu-0 llama-cpu-1 llama-cpu-2

# Clear current model
sudo /scripts/optimizations/manage-hugepages-models.sh clear

# Load new model
sudo /scripts/optimizations/manage-hugepages-models.sh load path/to/new-model.gguf

# Start containers with new model
docker-compose start llama-cpu-0 llama-cpu-1 llama-cpu-2
```

## Verification Commands

### System Status

```bash
# Check huge pages allocation and usage
cat /proc/meminfo | grep "^HugePages"

# Output explanation:
# HugePages_Total: Total allocated pages
# HugePages_Free:  Available pages
# HugePages_Rsvd:  Reserved but not yet allocated
# HugePages_Surp:  Surplus pages over nr_hugepages
# Hugepagesize:    Page size (2048 kB = 2MB)
```

### Container Verification

```bash
# Check if container can see the model
docker exec llama-cpu-0 ls -la /hugepages/

# Monitor container memory usage
docker exec llama-cpu-0 cat /proc/1/status | grep "^Vm"

# Check if wrapper is loaded
docker logs llama-cpu-0 2>&1 | grep "hugepage_mmap_wrapper"
```

### Performance Monitoring

```bash
# Watch huge pages usage during model loading
watch -n 1 'grep "^HugePages" /proc/meminfo'

# Monitor memory pressure
cat /proc/pressure/memory

# Check for memory fragmentation
cat /proc/buddyinfo
```

## How It Works

1. **Model Loading**: The management script copies GGUF models to the hugetlbfs mount
2. **Container Access**: Containers mount `/mnt/models-hugepages` as `/hugepages`
3. **mmap Interception**: The `hugepage_mmap_wrapper.so` library:
   - Intercepts mmap() calls for hugetlbfs files
   - Allocates anonymous memory with MAP_HUGETLB flag
   - Reads file contents into huge page memory
   - Returns the memory pointer transparently to llama.cpp

## Troubleshooting

### Insufficient Huge Pages

```bash
# Check current allocation
grep HugePages_Free /proc/meminfo

# Increase if needed (may fail if memory fragmented)
sudo sysctl vm.nr_hugepages=20000

# If allocation fails, reboot and allocate early
```

### Model Not Found

```bash
# Verify model is loaded
sudo /scripts/optimizations/manage-hugepages-models.sh loaded

# Check container mount
docker exec llama-cpu-0 ls -la /hugepages/
```

### Container Startup Issues

```bash
# Check model path
docker-compose config | grep MODEL_PATH

# Verify wrapper exists
docker exec llama-cpu-0 ls -la /app/hugepage_mmap_wrapper.so

# Check container logs
docker logs llama-cpu-0
```

### Memory Fragmentation

If huge pages allocation fails:

```bash
# Drop caches to free memory
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

# Compact memory
echo 1 | sudo tee /proc/sys/vm/compact_memory

# Try allocation again
sudo sysctl vm.nr_hugepages=15360
```

## Benchmarking

Use the provided benchmark script to compare performance:

```bash
# Run benchmarks for different configurations
./scripts/benchmark_hugepages.sh

# Compares:
# - No huge pages (regular 4KB pages)
# - 2MB huge pages
# - 1GB huge pages (if supported)
```

## Files Reference

- **Management Script**: `/scripts/optimizations/manage-hugepages-models.sh`
- **Systemd Service**: `/scripts/hugepages-model-manager.service`
- **Wrapper Library**: `/docker/llama-cpu/hugepage_mmap_wrapper.cpp`
- **Container Entry**: `/docker/llama-cpu/entrypoint.sh`
- **Benchmark Script**: `/scripts/benchmark_hugepages.sh`