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
    - [Parked proposals](#parked-proposals)
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
- **16 dedicated cores** (0-15) for single model instance
- **CPU optimizations**: Huge pages, C-states disabled, memory locking
- **Result**: 35 tokens/sec with 26% latency improvement

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

### bios/
- **bios-optimizations.md** - BIOS/UEFI settings for Gigabyte X870E Aorus Elite WiFi
  - CPU performance (C-states disabled, PBO configured)
  - Memory performance (DDR5-6000 with power-down disabled)
  - FCLK 2100 MHz for interconnect optimization
  - Single-model optimization focus
- **cpu_benchmark_results_*.json** - Benchmark data showing 26% improvement

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

**SUPERSEDED — awaiting re-measurement with `python -m bench`.** The figures below are
retained because they are the only before/after pair the repository has, and because a
new measurement has to be reconciled against them. Four things disqualify them from
being quoted as current:

- **Measured at 12 cores.** The run used `cpuset 0-11` and `THREADS=12`; the llama-cpu
  service in `docker-compose.inference.yaml` now runs `cpuset 0-15` and `THREADS=16`.
- **The model is a mixture of experts.** Both arms served
  `Qwen3-Coder-30B-A3B-Instruct-IQ4_XS` — roughly 3B active parameters of 30B total.
  An MoE's decode rate moves with which experts a prompt routes to, and this document
  presented the number as the CPU track's headline without recording the architecture.
  It is now recorded in `schemas/bench_models.json`.
- **Prefill and decode are conflated.** `tokens_per_second` in the archived files is
  `completion_tokens / total_time`, so prefill is folded into a decode-shaped figure.
  A change to either phase moves it, and it identifies neither.
- **The generating script is not in the repository.** The two JSON files under `bios/`
  cannot be regenerated by anything in the tree.

Recorded figures, on those terms:

- **Before optimizations**: 28.09 tokens/second average (end-to-end, prefill included)
- **After optimizations**: 35.44 tokens/second average (same basis)
- **Performance gain**: 26.2% increase
- **Consistency improvement**: 59% reduction in standard deviation. Note this is not a
  stored field: the archived files hold per-prompt standard deviations, and the quoted
  figure is a derivation whose method is not recorded with it.

To re-anchor the claim, run the arms the current configuration actually uses:

```bash
python3 -m bench noise --service llama-cpu --model qwen3-coder-30b-a3b-IQ4_XS
python3 -m bench run   --service llama-cpu --model qwen3-coder-30b-a3b-IQ4_XS \
    --profile publish --arm t16:THREADS=16 --arm t12:THREADS=12
```

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
- docker-compose.inference.yaml - Container orchestration
- docker/llama-cpu/Dockerfile.cpu - Latency-optimized CPU container
