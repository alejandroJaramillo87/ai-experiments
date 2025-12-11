# AMD μProf Setup Guide

AMD μProf (microprocessor profiler) provides deep CPU performance analysis for AMD processors. This guide covers installation and usage for benchmarking llama-zen-turbo optimizations.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Kernel Module Setup](#kernel-module-setup)
- [Verification](#verification)
- [Usage with Docker Containers](#usage-with-docker-containers)
- [Profile Types](#profile-types)
- [Interpreting Results](#interpreting-results)

## Prerequisites

- AMD Zen 4 (Ryzen 7000+) or Zen 5 (Ryzen 9000+) processor
- Ubuntu 22.04+ or Debian 12+
- Linux kernel 5.4+
- Root access for kernel module installation

## Installation

### Step 1: Download AMD μProf

Download the latest version from the AMD Developer site:
https://www.amd.com/en/developer/uprof.html

Or use wget (update version as needed):
```bash
# Check AMD developer site for latest version URL
wget https://download.amd.com/developer/eula/uprof/AMDuProf_Linux_x64_5.0.1449.tar.bz2
```

### Step 2: Extract and Install

```bash
# Extract
tar -xjf AMDuProf_Linux_x64_*.tar.bz2
cd AMDuProf_Linux_x64_*

# Install (copies to /opt/AMDuProf/)
sudo ./AMDuProf-Install.run

# Add to PATH
echo 'export PATH=$PATH:/opt/AMDuProf/bin' >> ~/.bashrc
source ~/.bashrc
```

### Alternative: DEB Package (if available)

```bash
sudo dpkg -i amduprof_*.deb
sudo apt-get install -f  # Fix any dependency issues
```

## Kernel Module Setup

AMD μProf requires the `amdpmc` kernel module for full PMU access.

### Load Module (Temporary)

```bash
sudo modprobe amdpmc
```

### Load Module (Persistent)

```bash
echo "amdpmc" | sudo tee /etc/modules-load.d/amdpmc.conf
```

### Set Permissions

Option 1: Add user to uprof group (if created by installer):
```bash
sudo usermod -aG uprof $USER
# Log out and back in
```

Option 2: Set perf_event_paranoid (same as for perf):
```bash
sudo sysctl kernel.perf_event_paranoid=-1

# Make persistent
echo "kernel.perf_event_paranoid=-1" | sudo tee /etc/sysctl.d/99-perf.conf
```

## Verification

### Check Installation

```bash
AMDuProfCLI --version
```

### Check System Support

```bash
AMDuProfCLI info --system
```

Expected output shows:
- CPU model (should show Zen 4/5)
- Available PMU events
- Supported profile configurations

### Test Basic Collection

```bash
# Profile a simple command
AMDuProfCLI collect --config assess -o /tmp/test-profile -- sleep 2

# Generate report
AMDuProfCLI report -i /tmp/test-profile --summary
```

## Usage with Docker Containers

μProf runs on the host and attaches to container processes by PID.

### Manual Profiling

```bash
# Get container's main process PID
SERVER_PID=$(docker inspect --format '{{.State.Pid}}' llama-hugepages)

# Start profiling
sudo AMDuProfCLI collect --config tbp -o ./uprof_results/test -p $SERVER_PID &
UPROF_PID=$!

# Run your benchmark
python3 llama.cpp/scripts/server-bench.py --path_server http://127.0.0.1:8003 ...

# Stop profiling
sudo kill -INT $UPROF_PID
wait $UPROF_PID

# Generate report
AMDuProfCLI report -i ./uprof_results/test --summary
```

### Using the Wrapper Script

```bash
# Basic usage (defaults to llama-hugepages, assess profile)
./scripts/benchmark_with_uprof.sh

# Specify container and profile type
./scripts/benchmark_with_uprof.sh llama-base tbp baseline-tbp

# Memory bandwidth analysis
./scripts/benchmark_with_uprof.sh llama-hugepages memory hugepages-memory
```

## Profile Types

### assess (General Performance)

Best for: Overall performance assessment
```bash
AMDuProfCLI collect --config assess -p <PID> -o results/
```

Collects:
- CPU utilization
- IPC (Instructions Per Cycle)
- Branch prediction accuracy
- Cache hit rates

### tbp (Time-Based Profiling)

Best for: Memory bandwidth and hotspot analysis
```bash
AMDuProfCLI collect --config tbp -p <PID> -o results/
```

Collects:
- Memory bandwidth utilization
- Time-based sampling
- Function-level hotspots
- Call stacks

### Custom PMU Events

Best for: Specific metrics (TLB, AVX-512, etc.)
```bash
AMDuProfCLI collect -e "cycles,instructions,L3_MISSES,RETIRED_SSE_AVX_OPS" -p <PID> -o results/
```

#### Useful Events for llama-zen-turbo

| Event | Description |
|-------|-------------|
| `cycles` | CPU cycles |
| `instructions` | Instructions retired |
| `L3_MISSES` | L3 cache misses |
| `DRAM_CHANNEL_0` | Memory bandwidth channel 0 |
| `DRAM_CHANNEL_1` | Memory bandwidth channel 1 |
| `RETIRED_SSE_AVX_OPS` | SIMD operations |
| `L2_CACHE_MISS_FROM_DC_MISS` | L2 misses |

List all available events:
```bash
AMDuProfCLI info --list-events
```

## Interpreting Results

### Report Generation

```bash
# Summary report
AMDuProfCLI report -i results/ --summary

# CSV export for analysis
AMDuProfCLI report -i results/ -o report.csv

# Detailed function-level report
AMDuProfCLI report -i results/ --hotspots
```

### Key Metrics for Memory Optimization

| Metric | Good Value | Indicates |
|--------|------------|-----------|
| IPC | > 2.0 | Efficient execution |
| L3 Hit Rate | > 90% | Good cache utilization |
| Memory Bandwidth | < 80% of max | Not bandwidth-limited |
| TLB Miss Rate | < 1% | Effective huge pages |

### Comparing Results

When comparing base vs hugepages:
1. Run same workload on both containers
2. Compare memory bandwidth utilization
3. Check for TLB miss rate differences
4. Look at IPC changes

## Troubleshooting

### "Permission denied" errors

```bash
# Ensure perf_event_paranoid is set
cat /proc/sys/kernel/perf_event_paranoid
# Should be -1 or 0

# Check if amdpmc module is loaded
lsmod | grep amdpmc
```

### "Cannot find process" errors

```bash
# Verify container is running
docker ps | grep llama

# Check PID is valid
ps -p $(docker inspect --format '{{.State.Pid}}' llama-hugepages)
```

### Empty or incomplete results

- Ensure profiling runs long enough (minimum 10 seconds)
- Check that the workload is actually running during profiling
- Try increasing sampling frequency

## References

- [AMD μProf User Guide](https://www.amd.com/en/developer/uprof.html)
- [AMD PMU Events Reference](https://developer.amd.com/resources/developer-guides-manuals/)
- [Zen 5 Optimization Guide](https://developer.amd.com/resources/developer-guides-manuals/)
