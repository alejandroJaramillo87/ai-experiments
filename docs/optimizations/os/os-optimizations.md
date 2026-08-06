# Operating System Optimizations

Host OS optimizations for containerized AI workloads on AMD Ryzen 9950X + RTX 5090 platform.
Latency optimizations target llama.cpp single-request inference; throughput optimizations target vLLM batch serving.

## Table of Contents

- [Operating System Optimizations](#operating-system-optimizations)
  - [Table of Contents](#table-of-contents)
  - [CPU/RAM Optimizations](#cpuram-optimizations)
    - [1. Swap Disabled \[IMPLEMENTED\]](#1-swap-disabled-implemented)
    - [2. CPU Governor - Performance Mode \[IMPLEMENTED\]](#2-cpu-governor---performance-mode-implemented)
    - [3. Huge Pages \[IMPLEMENTED\]](#3-huge-pages-implemented)
    - [4. Memory Allocator Tuning \[TESTED - NOT BENEFICIAL\]](#4-memory-allocator-tuning-tested---not-beneficial)
    - [5. System Memory Tuning \[IMPLEMENTED\]](#5-system-memory-tuning-implemented)
    - [6. CPU Pinning \[IMPLEMENTED\]](#6-cpu-pinning-implemented)
  - [GPU Optimizations](#gpu-optimizations)
    - [1. NVIDIA Driver Persistence Mode \[IMPLEMENTED\]](#1-nvidia-driver-persistence-mode-implemented)
    - [2. GPU Power and Clock Management \[PARTIALLY IMPLEMENTED\]](#2-gpu-power-and-clock-management-partially-implemented)
    - [3. NVIDIA Kernel Module Parameters \[IMPLEMENTED\]](#3-nvidia-kernel-module-parameters-implemented)
    - [4. GPU IRQ Affinity \[IMPLEMENTED\]](#4-gpu-irq-affinity-implemented)

## CPU/RAM Optimizations

### 1. Swap Disabled [IMPLEMENTED]

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

### 2. CPU Governor - Performance Mode [IMPLEMENTED]

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

### 3. Huge Pages [IMPLEMENTED]

**Status:** IMPLEMENTED WITH CUSTOM 1GB HUGEPAGE WRAPPER

Model weights are loaded into 1GB hugepages via LD_PRELOAD wrapper to reduce TLB pressure.

**Host Configuration:**
```bash
# Allocate 80 x 1GB hugepages (80GB total)
echo 80 | sudo tee /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages

# Make persistent in /etc/sysctl.conf
echo "vm.nr_hugepages_mempolicy=80" | sudo tee -a /etc/sysctl.conf
```

**Container Integration:**
- `libzen5_optimizer.so` intercepts mmap() calls for files >= 1GB
- Allocates anonymous 1GB hugepages via `MAP_HUGETLB | MAP_HUGE_1GB`
- Falls back to regular pages if allocation fails

**Verification:**
```bash
cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
# Should show: 80
```

**See:** `hugepages/src/memory/hugepage_wrapper.cpp` for implementation details.

### 4. Memory Allocator Tuning [TESTED - NOT BENEFICIAL]

**Status:** TESTED - CAUSED PERFORMANCE REGRESSION

Tested jemalloc as an alternative to glibc malloc. Results showed performance degradation for single-model latency workloads.

**Benchmark Results:**
- **Baseline (glibc malloc):** ~13.5 tok/s
- **With jemalloc:** 12.64 tok/s (-6.4% regression)

**Configuration Tested:**
```bash
export MALLOC_CONF="narenas:8,dirty_decay_ms:10000,muzzy_decay_ms:10000,background_thread:true"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libjemalloc.so.2:/app/libzen5_optimizer.so"
```

**Why jemalloc Did Not Help:**
- The `background_thread:true` setting competes with inference threads for CPU resources
- Decay timers add latency variance that hurts single-request performance
- llama.cpp's allocation pattern (large model load, then steady-state inference) doesn't benefit from jemalloc's server-oriented optimizations
- The zen5_optimizer already handles the critical large allocations via hugepages

**Current Configuration:**
```bash
# Only zen5_optimizer for hugepage mmap() interception
export LD_PRELOAD="/app/libzen5_optimizer.so"
```

**Recommendation:**
For single-model, latency-focused workloads, glibc malloc is sufficient. jemalloc may benefit high-concurrency scenarios with many simultaneous requests, but this use case is better served by vLLM.

### 5. System Memory Tuning [IMPLEMENTED]

**Status:** IMPLEMENTED

Kernel memory management optimizations for AI workloads.

**Configuration (/etc/sysctl.conf):**
```bash
# AI Workstation Optimizations
# AMD Ryzen 9950X + 128GB RAM

# Memory Management
vm.swappiness = 0              # Disable swapping (swap is off)
vm.zone_reclaim_mode = 0       # Single NUMA node optimization
vm.overcommit_memory = 1       # Allow overcommit
vm.vfs_cache_pressure = 50     # Favor application memory
vm.dirty_ratio = 5             # Aggressive writeback
vm.dirty_background_ratio = 2  # Background writeback
```

**Apply settings:**
```bash
sudo sysctl -p
```

### 6. CPU Pinning [IMPLEMENTED]

**Status:** IMPLEMENTED

Docker container resource limits and CPU pinning for AI workloads.

**CPU Container (llama-cpu):**

CPU pinning and OpenMP configuration for optimal inference performance:

```yaml
# docker-compose.inference.yaml
cpuset: "0-15"           # Pin to 16 cores (first CCX on Zen 5)
deploy:
  resources:
    limits:
      memory: 96G
environment:
  - THREADS=16
  - THREADS_BATCH=16
```

```dockerfile
# Dockerfile - OpenMP thread binding
ENV OMP_PROC_BIND=spread    # Spread threads across cores
ENV OMP_PLACES=cores        # Bind to physical cores
# OMP_NUM_THREADS set dynamically by entrypoint to match THREADS
```

**Why this works:**
- `cpuset: "0-15"` pins container to first 16 cores (single CCX on 9950X)
- `OMP_PROC_BIND=spread` distributes threads evenly across available cores
- `OMP_PLACES=cores` ensures threads bind to physical cores, not hyperthreads
- Keeps inference workload isolated from system processes on cores 16-31
- 

**Verification:**
```bash
# Check CPU pinning
docker exec llama-cpu taskset -p 1

# Check memory limits
docker stats --no-stream

# Verify OpenMP settings inside container
docker exec llama-cpu env | grep OMP
```

## GPU Optimizations

### 1. NVIDIA Driver Persistence Mode [IMPLEMENTED]

**Status:** IMPLEMENTED

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

### 2. GPU Power and Clock Management [PARTIALLY IMPLEMENTED]

**Status:** PARTIALLY IMPLEMENTED

**WARNING: Clock locking on RTX 5090 causes performance degradation**
- Testing showed 38% performance loss with locked clocks (287 to 178 tokens/sec)
- GPU requires dynamic frequency scaling for optimal performance
- Only power limit setting is recommended

Sets GPU power limit for maximum performance while allowing dynamic frequency scaling.

```bash
# Set power limit to maximum (600W for RTX 5090)
sudo nvidia-smi -pl 600

# DO NOT lock GPU/memory clocks - causes performance degradation
# sudo nvidia-smi -lgc 2400,2550  # DISABLED - reduces performance by 38%
# sudo nvidia-smi -lmc 3002       # DISABLED - reduces performance

# Actual implementation in optimize-gpu.sh:
echo "Setting power limit to 600W..."
nvidia-smi -pl 600

# Clock locking disabled - causes performance degradation on RTX 5090
# nvidia-smi -lgc 2400,2550
# nvidia-smi -lmc 3002
```

**Verification:**
```bash
# Check power limit
nvidia-smi -q | grep "Power Limit"  # Should show 600W

# Monitor actual power draw during workload
watch -n 1 'nvidia-smi --query-gpu=power.draw --format=csv,noheader'
# Note: May not reach 600W if workload doesn't require it (437W typical for llama.cpp)
```

### 3. NVIDIA Kernel Module Parameters [IMPLEMENTED]

**Status:** IMPLEMENTED

Enable Resizable BAR for full GPU VRAM access from CPU.

```bash
# Create modprobe configuration
sudo tee /etc/modprobe.d/nvidia-optimizations.conf << 'EOF'
# RTX 5090 - Enable Resizable BAR for full VRAM access
options nvidia NVreg_EnableResizableBar=1
EOF

# Rebuild initramfs
sudo update-initramfs -u

# Reboot required for changes to take effect
```

**Note:** Other parameters (EnableGpuFirmware, UsePageAttributeTable) are enabled by default on driver 580.x with RTX 5090.


### 4. GPU IRQ Affinity [IMPLEMENTED]

**Status:** IMPLEMENTED

Pin GPU interrupts to dedicated CPU cores for reduced latency.

```bash
# Find GPU IRQ numbers
GPU_IRQS=$(cat /proc/interrupts | grep -E "nvidia|gpu" | awk '{print $1}' | tr -d ':')

# Pin to cores 28-31 (GPU IRQ handling)
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



---

*Last Updated: 2025-09-23*