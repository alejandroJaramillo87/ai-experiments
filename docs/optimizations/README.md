# System Optimizations

Performance optimizations for AI workstation (AMD Ryzen 9950X + RTX 5090).

## Synopsis

System optimizations are divided into three categories:

- BIOS firmware settings
- Operating system configuration  
- Memory management with huge pages

## Files

### bios-optimizations.md
BIOS/UEFI settings for Gigabyte X870E Aorus Elite WiFi motherboard.
- CPU performance (C-states, PBO)
- Memory performance (DDR5-6000)
- System configuration

### os-optimizations.md
Host operating system optimizations for containerized workloads.
- Swap disabled
- CPU governor (performance mode)
- Memory locking configuration
- CPU pinning via Docker cpuset

### hugepages-setup.md
Huge pages configuration and model management.
- 2MB huge pages allocation
- Model loading into hugetlbfs
- Container integration via mmap wrapper

## Implementation Status

| Optimization | Status | Verification |
|--------------|--------|--------------|
| BIOS CPU settings | DONE | Check boost frequencies in CPU-Z |
| BIOS memory settings | DONE | Verify DDR5-6000 in BIOS |
| Swap disabled | DONE | `free -h` shows Swap: 0 |
| CPU governor | DONE | `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` |
| Memory locking | DONE | `docker exec <container> ulimit -l` |
| CPU pinning | DONE | `docker inspect <container> | grep CpusetCpus` |
| Huge pages 2MB | DONE | `grep ^HugePages /proc/meminfo` |
| THP disabled | DONE | `cat /sys/kernel/mm/transparent_hugepage/enabled` |

## Quick Verification

Check all optimizations:

```bash
#!/bin/bash
# Verify system optimizations

echo "Swap status:"
swapon --show || echo "  Disabled"

echo -e "\nCPU Governor:"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

echo -e "\nHuge Pages:"
grep "^HugePages_Total\|^HugePages_Free" /proc/meminfo

echo -e "\nMemory Locking:"
ulimit -l

echo -e "\nTHP Status:"
cat /sys/kernel/mm/transparent_hugepage/enabled
```

## Container Integration

Optimizations are applied to containers via:

1. Docker Compose configuration (`docker-compose.yaml`)
   - CPU pinning via cpuset (cores 0-7, 8-15, 16-23)
   - Memory limits (32GB per container) and ulimits
   - Volume mounts for huge pages (/mnt/models-hugepages)

2. Custom components (`docker/llama-cpu/`)
   - hugepage_mmap_wrapper.so for hugetlbfs support
   - entrypoint.sh for model verification

3. Management scripts (`scripts/optimizations/`)
   - manage-hugepages-models.sh for model loading
   - benchmark_hugepages.sh for performance testing

## Performance Impact

Measured improvements with optimizations enabled:

- CPU inference: Consistent frequency, no throttling
- Memory access: Reduced TLB misses with huge pages
- Container startup: Faster model loading from hugetlbfs

## See Also

- docker-compose.yaml - Container orchestration
- docker/Dockerfile.llama-cpu - CPU container build
- scripts/optimizations/ - Management scripts