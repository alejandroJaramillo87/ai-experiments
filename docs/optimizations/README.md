# System Optimizations

System optimizations for LLM inference on AMD Ryzen 9950X + RTX 5090: latency-focused (llama.cpp on CPU/GPU) and throughput-focused (vLLM on GPU).

## Table of Contents

- [System Optimizations](#system-optimizations)
  - [Table of Contents](#table-of-contents)
  - [Optimization Philosophy: Latency vs Throughput](#optimization-philosophy-latency-vs-throughput)
  - [Synopsis](#synopsis)
  - [Files](#files)
    - [roadmap.md](#roadmapmd)
    - [bios/](#bios)
    - [os/](#os)
    - [Parked proposals](#parked-proposals)
  - [Implementation Status](#implementation-status)
  - [Quick Verification](#quick-verification)
  - [Container Integration](#container-integration)
  - [System Configuration](#system-configuration)
  - [Validated Performance Results](#validated-performance-results)
    - [Benchmark Summary](#benchmark-summary)
    - [Key Validated Settings Impact](#key-validated-settings-impact)
  - [See Also](#see-also)

## Optimization Philosophy: Latency vs Throughput

This repository implements two distinct optimization strategies based on inference engine:

### Latency Optimization (llama.cpp)
For interactive chatbot applications prioritizing single-request response time:

**llama-cpu** (port 8001):
- **16 dedicated cores** (0-15) for single model instance
- **CPU optimizations**: Huge pages, C-states disabled, memory locking
- **Result**: unmeasured on this configuration. The one recorded figure was taken at 12
  cores and is superseded ([../bench/results.md](../bench/results.md))

**llama-gpu** (port 8002):
- **RTX 5090** for single-request GPU inference
- **GPU optimizations**: Persistence mode, DEFAULT compute mode, no clock locking
- **Result**: 287 tokens/sec optimal performance

Both llama.cpp services optimize for minimum first-token latency and consistent response times.

### Throughput Optimization (vLLM)
**vllm-gpu** (port 8003):
- **RTX 5090** for batch serving and concurrent requests
- **Optimizations**: CUDA graphs, tensor cores, FP8 support, memory pools
- **Goal**: Maximum requests/second for API endpoints

## Synopsis

System optimizations are organized by component:

- **bios/** - BIOS firmware settings and benchmark results
- **os/** - Operating system and memory optimizations
- **ideation/** - Parked proposals: recorded menus of optimization work that is
  not scheduled. Items graduate into the documents above once measured. See
  [Parked proposals](#parked-proposals) below.

## Files

### roadmap.md

[roadmap.md](roadmap.md) owns the ordering of the measurement campaign: which layer is
measured when, why that order and not the stack order, and what each stage accepts on. It is
the entry point for re-measuring anything in this directory, including the superseded figures
below.

### bios/
- **bios-optimizations.md** - BIOS/UEFI settings for Gigabyte X870E Aorus Elite WiFi
  - CPU performance (C-states disabled, PBO configured)
  - Memory performance (DDR5-6000 with power-down disabled)
  - FCLK 2100 MHz for interconnect optimization
  - Single-model optimization focus
- **cpu_benchmark_results_*.json** - The raw before and after captures behind the
  superseded figure in [../bench/results.md](../bench/results.md); not regenerable by
  anything in the tree

### os/
- **os-optimizations.md** - Host operating system optimizations for containerized workloads
  - Swap disabled
  - CPU governor (performance mode)
  - Memory locking configuration
  - CPU pinning via Docker cpuset

### Parked proposals

Recorded menus under `ideation/`. Nothing in them is scheduled or implemented; each item
requires host-side verification with before/after numbers before it graduates into the
documents above.

- **[ideation/os-optimization-candidates.md](ideation/os-optimization-candidates.md)** -
  Host-level OS and GPU candidates (2026-07-30); graduate into `os/os-optimizations.md`
- **[ideation/python-pipeline-optimization-candidates.md](ideation/python-pipeline-optimization-candidates.md)** -
  Training and interp Python pipeline candidates (2026-07-30)
- **[ideation/rust-harness-optimization-candidates.md](ideation/rust-harness-optimization-candidates.md)** -
  x86 CPU backend candidates (2026-07-30); blocked on runbook entry RH.1
- **[ideation/future-inference-optimizations.md](ideation/future-inference-optimizations.md)** -
  Parked custom CUDA and C++ findings (2026-07-27); gating conditions not met

## Implementation Status

| Optimization | Status | Verification |
|--------------|--------|--------------|
| BIOS CPU settings | DONE | Check boost frequencies in CPU-Z |
| BIOS memory settings | DONE | Verify DDR5-6000 in BIOS |
| Swap disabled | DONE | `free -h` shows Swap: 0 |
| CPU governor | DONE | `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` |
| Memory locking | DONE | `docker exec <container> ulimit -l` |
| CPU pinning (16 cores) | DONE | `docker inspect llama-cpu | grep CpusetCpus` should show "0-15" |
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

1. Docker Compose configuration (`docker-compose.inference.yaml`)
   - Single CPU service with 16 dedicated cores (cpuset: "0-15")
   - Optimized memory allocation (96GB for single high-performance instance)
   - Environment variables from .env file

2. Explicit huge pages (`docker/llama-cpu/`)
   - libzen5_optimizer.so uses MAP_HUGETLB for models larger than 1GB
   - No special filesystem or model copying required
   - Direct huge page allocation, not THP

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

### Benchmark Summary

The repository's one recorded before-and-after measurement, the 26.2 percent CPU
figure, is owned by [../bench/results.md](../bench/results.md), which states the four
grounds on which it is superseded and the commands that re-anchor it. It is not restated
here: it was measured across a configuration this document no longer describes, and
keeping a copy beside the settings is what let it be quoted as a validated property of
them.

### Key Validated Settings Impact
- **FCLK 2100 MHz**: Successfully stable, major contributor to performance
- **Conservative Curve Optimizer (-15)**: Achieved excellent results with stability
- **CPU Boost 200 MHz increase**: Stable at maximum recommended setting
- **Power optimizations**: C-states disabled, power down disabled for consistency

## See Also

- [roadmap.md](roadmap.md) — the measurement campaign order and its reasoning
- [bios/bios-optimizations.md](bios/bios-optimizations.md) — BIOS configuration guide
- [os/os-optimizations.md](os/os-optimizations.md) — operating system tuning
- [../bench/results.md](../bench/results.md) — what has actually been measured, and its status
- `docker-compose.inference.yaml` — container orchestration, and the source of truth for
  every service argument and cpuset named here
- `docker/llama-cpu/Dockerfile.cpu` — the latency-optimized CPU container
