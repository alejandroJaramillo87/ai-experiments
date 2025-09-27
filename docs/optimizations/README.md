# System Optimizations

System optimizations for LLM inference on AMD Ryzen 9950X + RTX 5090: latency-focused (llama.cpp on CPU/GPU) and throughput-focused (vLLM on GPU).

## Table of Contents

- [System Optimizations](#system-optimizations)
  - [Table of Contents](#table-of-contents)
  - [Optimization Philosophy: Latency vs Throughput](#optimization-philosophy-latency-vs-throughput)
  - [Synopsis](#synopsis)
  - [Files](#files)
    - [bios/](#bios)
    - [os/](#os)
    - [gpu/](#gpu)
    - [benchmark-guide.md](#benchmark-guidemd)
  - [Implementation Status](#implementation-status)
  - [Quick Verification](#quick-verification)
  - [Container Integration](#container-integration)
  - [System Configuration](#system-configuration)
  - [Validated Performance Results](#validated-performance-results)
    - [Benchmark Summary](#benchmark-summary)
    - [Key Validated Settings Impact](#key-validated-settings-impact)
  - [Performance Impact](#performance-impact)
  - [See Also](#see-also)

## Optimization Philosophy: Latency vs Throughput

This repository implements two distinct optimization strategies based on inference engine:

### Latency Optimization (llama.cpp)
For interactive chatbot applications prioritizing single-request response time:

**llama-cpu** (port 8001):
- **12 dedicated cores** (0-11) for single model instance
- **CPU optimizations**: Huge pages, C-states disabled, memory locking
- **Result**: 35 tokens/sec with 26% latency improvement

**llama-gpu** (port 8004):
- **RTX 5090** for single-request GPU inference
- **GPU optimizations**: Persistence mode, DEFAULT compute mode, no clock locking
- **Result**: 287 tokens/sec optimal performance

Both llama.cpp services optimize for minimum first-token latency and consistent response times.

### Throughput Optimization (vLLM)
**vllm-gpu** (port 8005):
- **RTX 5090** for batch serving and concurrent requests
- **Optimizations**: CUDA graphs, tensor cores, FP8 support, memory pools
- **Goal**: Maximum requests/second for API endpoints

## Synopsis

System optimizations are organized by component:

- **bios/** - BIOS firmware settings and benchmark results
- **os/** - Operating system and memory optimizations
- **gpu/** - GPU-specific performance tuning
- **experiments/** - Experimental and testing configurations

## Files

### bios/
- **bios-optimizations.md** - BIOS/UEFI settings for Gigabyte X870E Aorus Elite WiFi
  - CPU performance (C-states disabled, PBO configured)
  - Memory performance (DDR5-6000 with power-down disabled)
  - FCLK 2100 MHz for interconnect optimization
  - Single-model optimization focus
- **benchmark_results_*.json** - Benchmark data showing 26% improvement

### os/
- **os-optimizations.md** - Host operating system optimizations for containerized workloads
  - Swap disabled
  - CPU governor (performance mode)
  - Memory locking configuration
  - CPU pinning via Docker cpuset
- **hugepages-explicit.md** - Explicit huge pages implementation using MAP_HUGETLB
  - Automatic huge page allocation for models larger than 1GB via wrapper
  - No special filesystem required
  - Approximately 35 tokens/sec performance with optimizations

### gpu/
- **gpu-optimizations.md** - RTX 5090 specific optimizations
  - CUDA configuration and tuning
  - GPU memory management
  - Multi-GPU scaling considerations
  - **Important**: Clock locking causes 38% performance degradation on RTX 5090
  - **Note**: vLLM sections focus on throughput optimization, different from llama.cpp latency goals

### benchmark-guide.md
Performance benchmarking tool for optimization validation.
- Measures tokens per second
- Multiple test prompts for different workloads
- JSON and human-readable output

## Implementation Status

| Optimization | Status | Verification |
|--------------|--------|--------------|
| BIOS CPU settings | DONE | Check boost frequencies in CPU-Z |
| BIOS memory settings | DONE | Verify DDR5-6000 in BIOS |
| Swap disabled | DONE | `free -h` shows Swap: 0 |
| CPU governor | DONE | `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` |
| Memory locking | DONE | `docker exec <container> ulimit -l` |
| CPU pinning (12 cores) | DONE | `docker inspect llama-cpu | grep CpusetCpus` should show "0-11" |
| Huge pages (90GB) | DONE | `grep ^HugePages_Total /proc/meminfo` shows 46080 |
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

echo -e "\nCPU Count:"
nproc

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
   - Single CPU service with 12 dedicated cores (cpuset: "0-11")
   - Optimized memory allocation (96GB for single high-performance instance)
   - Environment variables from .env file

2. Explicit huge pages (`docker/llama-cpu/`)
   - hugepage_mmap_wrapper.so uses MAP_HUGETLB for models larger than 1GB
   - No special filesystem or model copying required
   - Direct huge page allocation, not THP

3. Performance testing (`scripts/`)
   - benchmark.py for measuring latency and tokens/second
   - Focus on first-token latency and response time consistency
   - Validated 26% performance improvement with optimizations

## System Configuration

Complete `/etc/sysctl.conf` settings:
```bash
vm.nr_hugepages = 46080         # 90GB huge pages
vm.swappiness = 0               # Disable swapping
vm.zone_reclaim_mode = 0        # Single NUMA optimization
vm.overcommit_memory = 1        # Allow overcommit
vm.vfs_cache_pressure = 50      # Favor app memory
vm.dirty_ratio = 5              # Aggressive writeback
vm.dirty_background_ratio = 2   # Background writeback
```

## Validated Performance Results

Production system benchmarking with BIOS optimizations enabled:

### Benchmark Summary
- **Before optimizations**: 28.09 tokens/second average
- **After optimizations**: 35.44 tokens/second average
- **Performance gain**: 26.2% increase
- **Consistency improvement**: 59% reduction in standard deviation

### Key Validated Settings Impact
- **FCLK 2100 MHz**: Successfully stable, major contributor to performance
- **Conservative Curve Optimizer (-15)**: Achieved excellent results with stability
- **CPU Boost 200 MHz increase**: Stable at maximum recommended setting
- **Power optimizations**: C-states disabled, power down disabled for consistency

## Performance Impact

Validated improvements with optimizations enabled:

- **Latency**: 26% improvement in tokens per second (validated via benchmarking)
- **Cache Efficiency**: Optimized cache access patterns
- **Memory Access**: Reduced TLB misses with explicit huge pages
- **Consistency**: Predictable token generation timing (0.08-0.28 stdev vs 0.08-0.84)

## See Also

- bios/bios-optimizations.md - Detailed BIOS configuration guide
- os/os-optimizations.md - Operating system tuning
- os/hugepages-explicit.md - Memory optimization details
- gpu/gpu-optimizations.md - GPU performance tuning
- docker-compose.yaml - Container orchestration
- docker/llama-cpu/Dockerfile.llama-cpu - Latency-optimized CPU container
- scripts/benchmark.py - Latency measurement tool

---

*Last Updated: 2025-09-23*