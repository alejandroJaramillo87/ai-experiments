# Operating System Optimizations

Host OS optimizations for containerized AI workloads on AMD Ryzen 9950X + RTX 5090 platform.

## Implemented Optimizations

### 1. Swap Disabled ✅

**Status:** IMPLEMENTED

Prevents model data from being swapped to disk.

```bash
# Disable swap
sudo swapoff -a

# Remove swap entries from fstab
sudo sed -i '/swap/d' /etc/fstab

# Verify
free -h  # Swap should show 0
```

### 2. CPU Governor - Performance Mode ✅

**Status:** IMPLEMENTED

Ensures consistent CPU frequency for inference workloads.

```bash
# Set performance governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Create systemd service for persistence
sudo tee /etc/systemd/system/cpufreq-performance.service << 'EOF'
[Unit]
Description=Set CPU Governor to Performance
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now cpufreq-performance.service

# Verify
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | uniq
```

### 3. Memory Locking ✅

**Status:** CONFIGURED IN CONTAINERS

Containers are configured with unlimited memory locking.

**Docker Compose Configuration:**
```yaml
ulimits:
  memlock:
    soft: -1
    hard: -1
```

**Host System Configuration:**
```bash
# Set system-wide limits
sudo tee -a /etc/security/limits.conf << 'EOF'
* soft memlock unlimited
* hard memlock unlimited
EOF
```

**Verification:**
```bash
# Check container limits
docker exec llama-cpu-0 bash -c "ulimit -l"  # Should show: unlimited

# Monitor locked memory on host
watch -n 2 "cat /proc/meminfo | grep MemLocked"

# Check container memory locking
docker exec llama-cpu-0 cat /proc/1/status | grep VmLck
```

### 4. Huge Pages ✅

**Status:** IMPLEMENTED WITH CUSTOM WRAPPER

Enables 2MB huge pages for model loading to reduce memory management overhead.

**Allocation:**
```bash
# Allocate huge pages (for 30GB model)
sudo sysctl vm.nr_hugepages=15360  # 15360 * 2MB = 30GB

# Make persistent
echo "vm.nr_hugepages=15360" | sudo tee -a /etc/sysctl.conf
```

**Mount hugetlbfs:**
```bash
# Create mount point
sudo mkdir -p /mnt/models-hugepages

# Mount hugetlbfs
sudo mount -t hugetlbfs -o pagesize=2M,size=30G none /mnt/models-hugepages

# Add to /etc/fstab for persistence
echo "none /mnt/models-hugepages hugetlbfs pagesize=2M,size=30G 0 0" | sudo tee -a /etc/fstab
```

**Model Loading:**
```bash
# Load model into huge pages
sudo /scripts/optimizations/manage-hugepages-models.sh load <model.gguf>

# Check status
sudo /scripts/optimizations/manage-hugepages-models.sh status
```

**Verification:**
```bash
# Check huge pages allocation
grep "^HugePages" /proc/meminfo

# Expected output:
# HugePages_Total:   15360
# HugePages_Free:    <available>
# HugePages_Rsvd:    0
# HugePages_Surp:    0
# Hugepagesize:      2048 kB
```

**Container Integration:**
- Uses custom `hugepage_mmap_wrapper.so` via LD_PRELOAD
- Automatically activated when MODEL_PATH points to `/hugepages/*`
- Transparently loads hugetlbfs files into anonymous huge page memory

## Container Resource Allocation

### CPU Pinning
Containers use Docker's cpuset to pin to specific cores:
- **llama-cpu-0**: Cores 0-7
- **llama-cpu-1**: Cores 8-15  
- **llama-cpu-2**: Cores 16-23
- **System/GPU**: Cores 24-31

### Memory Limits
Each CPU container is limited to 32GB RAM.

## Monitoring Commands

### System Performance
```bash
# CPU frequency monitoring
watch -n 1 'grep "cpu MHz" /proc/cpuinfo | head -8'

# Memory usage
free -h

# Per-container resources
docker stats
```

### Verification Script
```bash
#!/bin/bash
echo "=== OS Optimization Status ==="

# Swap
echo -n "Swap: "
if [ $(swapon --show | wc -l) -eq 0 ]; then
    echo "DISABLED"
else
    echo "ENABLED (should be disabled)"
fi

# CPU Governor
echo -n "CPU Governor: "
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Memory Locking
echo -n "System memlock limit: "
ulimit -l

# Huge Pages
echo "Huge Pages:"
grep "^HugePages" /proc/meminfo
```

## Security Considerations

All optimizations maintain container isolation:
- CPU pinning via cpuset (not isolcpus)
- Memory locking within container limits
- No privileged access required
- Read-only container filesystems