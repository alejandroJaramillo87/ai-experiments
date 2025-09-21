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
# Allocate huge pages for 3x 30GB models (90GB total)
sudo sysctl vm.nr_hugepages=46080  # 46080 * 2MB = 90GB

# Make persistent in /etc/sysctl.conf
echo "vm.nr_hugepages=46080" | sudo tee -a /etc/sysctl.conf
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
# HugePages_Total:   46080  # 90GB total
# HugePages_Free:    38267  # Available (varies based on usage)
# HugePages_Rsvd:    0
# HugePages_Surp:    0
# Hugepagesize:      2048 kB
```

**Container Integration:**
- Uses custom `hugepage_mmap_wrapper.so` via LD_PRELOAD
- Automatically activated when MODEL_PATH points to `/hugepages/*`
- Transparently loads hugetlbfs files into anonymous huge page memory

## Complete sysctl.conf Configuration

**Full /etc/sysctl.conf settings for AI workloads:**

```bash
# AI Workstation Optimizations
# AMD Ryzen 9950X + 128GB RAM

# Huge Pages: 90GB for 3x 30GB models
vm.nr_hugepages = 46080

# Memory Management
vm.swappiness = 0              # Disable swapping (swap is off)
vm.zone_reclaim_mode = 0       # Single NUMA node optimization
vm.overcommit_memory = 1       # Allow memory overcommit

# Cache and I/O Tuning
vm.vfs_cache_pressure = 50     # Favor application memory
vm.dirty_ratio = 5              # Aggressive writeback
vm.dirty_background_ratio = 2  # Background writeback
```

**Apply settings:**
```bash
sudo sysctl -p
```

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

## GPU Optimizations

### 5. NVIDIA Driver Persistence Mode ✅

**Status:** TO BE IMPLEMENTED

Keeps NVIDIA driver loaded in memory, reducing CUDA initialization time.

```bash
# Enable persistence mode
sudo nvidia-smi -pm 1

# Create systemd service for persistence
sudo tee /etc/systemd/system/nvidia-persistenced.service << 'EOF'
[Unit]
Description=NVIDIA Persistence Daemon
After=multi-user.target

[Service]
Type=forking
ExecStart=/usr/bin/nvidia-persistenced --user nvidia-persistenced
ExecStopPost=/bin/rm -rf /var/run/nvidia-persistenced
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now nvidia-persistenced.service

# Verify
nvidia-smi -q | grep "Persistence Mode"
```

### 6. GPU Power and Clock Management ✅

**Status:** TO BE IMPLEMENTED

Locks GPU in maximum performance state for consistent inference.

```bash
# Set power limit to maximum (600W for RTX 5090)
sudo nvidia-smi -pl 600

# Lock GPU clocks for consistency
sudo nvidia-smi -lgc 2400,2550  # Lock GPU clock range (min,max)
sudo nvidia-smi -lmc 3002       # Lock memory clock

# Create systemd service for GPU clocks
sudo tee /etc/systemd/system/gpu-clocks.service << 'EOF'
[Unit]
Description=Lock GPU Clocks for AI Inference
After=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/bin/bash -c '\
    nvidia-smi -pm 1 && \
    nvidia-smi -pl 600 && \
    nvidia-smi -lgc 2400,2550 && \
    nvidia-smi -lmc 3002'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now gpu-clocks.service
```

### 7. GPU Compute Mode ✅

**Status:** TO BE IMPLEMENTED

Configure GPU for optimal multi-container usage.

```bash
# Set to EXCLUSIVE_PROCESS mode (one process per GPU)
sudo nvidia-smi -c EXCLUSIVE_PROCESS

# Alternative for shared access:
# sudo nvidia-smi -c DEFAULT

# Verify
nvidia-smi --query-gpu=compute_mode --format=csv
```

### 8. NVIDIA Kernel Module Parameters

**Status:** TO BE IMPLEMENTED

Optimize NVIDIA driver behavior at kernel level.

```bash
# Create modprobe configuration
sudo tee /etc/modprobe.d/nvidia-optimizations.conf << 'EOF'
# RTX 5090 AI Inference Optimizations
options nvidia NVreg_PreserveVideoMemoryAllocations=1  # Preserve VRAM across suspend
options nvidia NVreg_TemporaryFilePath=/var/tmp        # Fast temp storage
options nvidia NVreg_EnableResizableBar=1              # Enable Resizable BAR
options nvidia NVreg_EnableGpuFirmware=1               # Enable GSP firmware
options nvidia NVreg_TCEBypass=1                       # Bypass IOMMU for performance
options nvidia NVreg_UsePageAttributeTable=1           # Better memory management
EOF

# Rebuild initramfs
sudo update-initramfs -u

# Apply without reboot (if possible)
sudo modprobe -r nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
```

### 9. GPU IRQ Affinity

**Status:** TO BE IMPLEMENTED

Pin GPU interrupts to dedicated CPU cores for reduced latency.

```bash
# Find GPU IRQ numbers
GPU_IRQS=$(cat /proc/interrupts | grep -E "nvidia|gpu" | awk '{print $1}' | tr -d ':')

# Pin to cores 24-31 (system/GPU cores)
for irq in $GPU_IRQS; do
    echo f0000000 | sudo tee /proc/irq/$irq/smp_affinity > /dev/null
done

# Create script for persistence
sudo tee /usr/local/bin/set-gpu-irq-affinity.sh << 'EOF'
#!/bin/bash
GPU_IRQS=$(cat /proc/interrupts | grep -E "nvidia|gpu" | awk '{print $1}' | tr -d ':')
for irq in $GPU_IRQS; do
    echo f0000000 > /proc/irq/$irq/smp_affinity 2>/dev/null
done
EOF

sudo chmod +x /usr/local/bin/set-gpu-irq-affinity.sh

# Add to rc.local or systemd service
```

### 10. CUDA System Environment

**Status:** TO BE IMPLEMENTED

System-wide CUDA optimization variables.

```bash
# Add to /etc/environment
sudo tee -a /etc/environment << 'EOF'
# CUDA Optimizations
CUDA_CACHE_DISABLE=0
CUDA_CACHE_PATH=/tmp/cuda_cache
CUDA_CACHE_MAXSIZE=2147483648
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_LAUNCH_BLOCKING=0
CUDA_MODULE_LOADING=LAZY
EOF

# Create CUDA cache directory
sudo mkdir -p /tmp/cuda_cache
sudo chmod 1777 /tmp/cuda_cache
```

### 11. NVIDIA Multi-Process Service (MPS)

**Status:** OPTIONAL

Enable for better GPU utilization with multiple containers.

```bash
# Create MPS directories
sudo mkdir -p /tmp/nvidia-mps /tmp/nvidia-log

# Create systemd service
sudo tee /etc/systemd/system/nvidia-mps.service << 'EOF'
[Unit]
Description=NVIDIA MPS Control Daemon
After=nvidia-persistenced.service

[Service]
Type=forking
Environment="CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps"
Environment="CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log"
ExecStart=/usr/bin/nvidia-cuda-mps-control -d
ExecStop=/usr/bin/echo quit | /usr/bin/nvidia-cuda-mps-control
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

# Enable for multi-container workloads
# sudo systemctl enable --now nvidia-mps.service
```

### 12. Kernel Boot Parameters

**Status:** TO BE IMPLEMENTED

Optimize kernel for GPU workloads.

```bash
# Edit /etc/default/grub
# Add to GRUB_CMDLINE_LINUX:
# nvidia-drm.modeset=1     - Enable DRM kernel mode setting
# pcie_aspm=off            - Disable PCIe power saving
# rcutree.rcu_idle_gp_delay=1  - Reduce RCU delays

sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 nvidia-drm.modeset=1 pcie_aspm=off rcutree.rcu_idle_gp_delay=1"/' /etc/default/grub

# Update GRUB
sudo update-grub
```

## GPU Monitoring and Verification

### GPU Status Script
```bash
#!/bin/bash
echo "=== GPU Optimization Status ==="

# Driver persistence
echo -n "Persistence Mode: "
nvidia-smi -q | grep "Persistence Mode" | head -1 | awk '{print $4}'

# Power limits
echo -n "Power Limit: "
nvidia-smi -q | grep "Power Limit" | head -1 | awk '{print $4 " " $5}'

# Clock locks
echo "GPU Clocks:"
nvidia-smi -q | grep -A 2 "Clocks"

# Compute mode
echo -n "Compute Mode: "
nvidia-smi --query-gpu=compute_mode --format=csv,noheader

# PCIe status
echo "PCIe Status:"
nvidia-smi -q | grep -A 4 "GPU Link Info"

# Memory usage
echo "GPU Memory:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
```

### Complete Setup Script

Create `/usr/local/bin/optimize-gpu.sh`:
```bash
#!/bin/bash
set -e

echo "Configuring RTX 5090 for optimal AI inference..."

# Enable persistence mode
sudo nvidia-smi -pm 1

# Set maximum power limit (600W for RTX 5090)
sudo nvidia-smi -pl 600

# Lock GPU clocks for consistency
sudo nvidia-smi -lgc 2400,2550
sudo nvidia-smi -lmc 3002

# Set compute mode
sudo nvidia-smi -c EXCLUSIVE_PROCESS

# Set GPU IRQ affinity
GPU_IRQS=$(cat /proc/interrupts | grep -E "nvidia|gpu" | awk '{print $1}' | tr -d ':')
for irq in $GPU_IRQS; do
    echo f0000000 | sudo tee /proc/irq/$irq/smp_affinity > /dev/null 2>&1
done

# Create CUDA cache directory
sudo mkdir -p /tmp/cuda_cache
sudo chmod 1777 /tmp/cuda_cache

echo "GPU optimization complete!"

# Verify settings
nvidia-smi -q | grep -E "Persistence Mode|Power Limit|Compute Mode"
```

## Security Considerations

All optimizations maintain container isolation:
- CPU pinning via cpuset (not isolcpus)
- Memory locking within container limits
- No privileged access required for containers
- Read-only container filesystems
- GPU access controlled via device cgroups