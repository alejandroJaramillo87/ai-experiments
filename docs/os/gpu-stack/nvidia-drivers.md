# NVIDIA Drivers - The Foundation Layer

Understanding NVIDIA GPU drivers and their role in AI computing.

## Table of Contents

<!-- toc:start -->

1. [What Are NVIDIA Drivers?](#what-are-nvidia-drivers)
   1. [Key Components](#key-components)
2. [Why Critical for Our AI Workstation?](#why-critical-for-our-ai-workstation)
   1. [1. Hardware Resource Management](#1-hardware-resource-management)
   2. [2. PCIe Communication](#2-pcie-communication)
   3. [3. CUDA Runtime Foundation](#3-cuda-runtime-foundation)
3. [Our Specific Setup](#our-specific-setup)
   1. [Driver Version: 580.65.06](#driver-version-5806506)
   2. [Key Configuration](#key-configuration)
   3. [Installation Process](#installation-process)
4. [Impact on AI Performance](#impact-on-ai-performance)
   1. [Memory Management](#memory-management)
   2. [Real-World Example](#real-world-example)
5. [Troubleshooting](#troubleshooting)
   1. [Common Issues](#common-issues)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
   1. [Key Metrics to Monitor](#key-metrics-to-monitor)
   2. [Regular Maintenance](#regular-maintenance)
7. [Server vs Desktop Drivers](#server-vs-desktop-drivers)
8. [Integration with CUDA Stack](#integration-with-cuda-stack)
9. [Best Practices](#best-practices)
10. [Next Steps](#next-steps)

<!-- toc:end -->

## What Are NVIDIA Drivers?

NVIDIA drivers are the software layer that enables your operating system to communicate with your GPU hardware. They handle communication between Linux and your RTX 5090, from basic display output to CUDA operations.

### Key Components

1. **Kernel Modules** - Low-level code that runs in kernel space
   - `nvidia` - Core driver module
   - `nvidia-modeset` - Display mode setting
   - `nvidia-uvm` - Unified Virtual Memory for CUDA
   - `nvidia-drm` - Direct Rendering Manager

2. **User-Space Libraries** - Application interfaces
   - `libnvidia-gl` - OpenGL support
   - `libnvidia-compute` - CUDA compute libraries
   - `libnvidia-ml` - Management Library (nvidia-ml)

3. **Management Tools**
   - `nvidia-smi` - System Management Interface
   - `nvidia-settings` - GUI configuration tool
   - `nvidia-persistenced` - Persistence mode daemon

## Why Critical for Our AI Workstation?

### 1. Hardware Resource Management

The driver manages all GPU resources:

```
Driver Responsibilities:
├── VRAM Allocation (32GB total)
│   ├── Model storage (15-20GB for large models)
│   ├── Inference buffers (5-10GB)
│   └── CUDA workspace (remaining)
├── Compute Resources
│   ├── 21,760 CUDA cores scheduling
│   ├── 680 Tensor cores allocation
│   └── SM (Streaming Multiprocessor) dispatch
└── Power & Thermal
    ├── 600W power limit enforcement
    ├── Clock speed management (up to 2.55 GHz)
    └── Temperature monitoring
```

### 2. PCIe Communication

Our RTX 5090 communicates over PCIe Gen5:
- **Bandwidth**: 128 GB/s bidirectional
- **Latency**: Sub-microsecond for small transfers
- **Model Loading**: 20GB model loads in ~0.3 seconds

The driver manages this high-speed data highway between CPU and GPU.

### 3. CUDA Runtime Foundation

Without drivers, CUDA cannot function:
```
Application (llama.cpp)
    ↓ CUDA API calls
CUDA Runtime
    ↓ Driver API
NVIDIA Driver
    ↓ Hardware commands
RTX 5090 GPU
```

## Our Specific Setup

### Driver Version: 580.65.06

We use the latest stable driver for several reasons:

1. **Blackwell Support**: Full support for RTX 5090's SM 12.0 architecture
2. **CUDA 13 Compatibility**: Required for CUDA 13.0.88
3. **Performance Features**:
   - Hardware scheduling improvements
   - Optimized memory management
   - Better PCIe Gen5 utilization

### Key Configuration

```bash
# Current driver configuration
nvidia-smi -q

# Key settings we use:
Driver Version                  : 580.65.06
Persistence Mode                : Enabled    # Faster CUDA init (200ms saved)
Power Limit                    : 600W       # Maximum performance
Compute Mode                   : Default    # Multi-process support
```

### Installation Process

Our automated script (`scripts/setup/setup_nvidia.sh`) performs:

```bash
# 1. Remove old drivers
sudo apt-get purge nvidia-*

# 2. Add NVIDIA repository
# (Handled by script)

# 3. Install the qualified driver branch
# An open-kernel-module build is required: Blackwell (SM 12.0) is only
# supported there. This workstation is provisioned on the -server-open flavor.
sudo apt install nvidia-driver-580-server-open

# 4. Configure persistence mode
sudo nvidia-smi -pm 1

# 5. Set power limit
sudo nvidia-smi -pl 600
```

## Impact on AI Performance

### Memory Management

The driver's memory management directly impacts inference:

| Configuration | Impact on Performance |
|--------------|----------------------|
| **Without Persistence Mode** | +200ms CUDA initialization per request |
| **Default Power Limit (450W)** | -15% inference speed |
| **Wrong Driver Version** | CUDA incompatibility, no GPU acceleration |
| **Optimal Settings** | Full GPU acceleration |

### Real-World Example

Loading a 20B parameter model:

```python
# With proper driver configuration
model_load_time = 8.2 seconds   # Direct VRAM loading
# Full GPU acceleration

# With misconfigured driver
model_load_time = 45+ seconds   # System RAM fallback
# CPU-only fallback performance
```

## Troubleshooting

### Common Issues

#### 1. Driver Not Loaded
```bash
nvidia-smi
# Output: "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver"

# Fix:
sudo modprobe nvidia
sudo systemctl restart nvidia-persistenced
```

#### 2. Version Mismatch
```bash
# Check for mismatches
nvidia-smi  # Shows driver version
nvcc --version  # Shows CUDA version

# These must be compatible (580.x supports CUDA 13.0)
```

#### 3. Persistence Mode Disabled
```bash
# Check status
nvidia-smi -q | grep "Persistence Mode"

# Enable
sudo nvidia-smi -pm 1
```

#### 4. PCIe Generation Issues
```bash
# Check PCIe status
nvidia-smi -q | grep "PCIe Generation"
# Should show: Current 5, Max 5

# If showing Gen1-4, check BIOS settings
```

## Monitoring and Maintenance

### Key Metrics to Monitor

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Detailed query
nvidia-smi -q -d PERFORMANCE,MEMORY,UTILIZATION

# Driver logs
sudo journalctl -u nvidia-persistenced
sudo dmesg | grep nvidia
```

### Regular Maintenance

1. **Check for Updates** (monthly)
   ```bash
   # Check available versions
   apt list -a nvidia-driver-*
   ```

2. **Verify Configuration** (after reboots)
   ```bash
   # Driver and CUDA runtime visible to the host
   nvidia-smi
   # The repository's GPU-dependent test tier
   uv run pytest -m hw_smoke tests/hardware/
   ```

3. **Monitor Performance**
   ```bash
   # Benchmark to ensure consistent performance
   scripts/benchmark/benchmark_gpu.sh
   ```

## Server vs Desktop Drivers

We specifically chose server-oriented configurations:

| Feature | Desktop | Server (Our Choice) |
|---------|---------|---------------------|
| **Focus** | Graphics, Gaming | Compute, AI |
| **Persistence Mode** | Off | **On** |
| **Power Management** | Dynamic | **Fixed High** |
| **Display Support** | Priority | Minimal |
| **Stability** | Frequent updates | **LTS Focus** |

## Integration with CUDA Stack

The driver provides the foundation for CUDA:

```
nvidia-driver-580 provides:
├── libnvidia-compute-580  → Required by CUDA Runtime
├── libnvidia-ml-580       → Used by nvidia-ml library
└── nvidia-uvm kernel mod  → Enables CUDA Unified Memory
```

Without these components, CUDA 13.0 cannot initialize, and our inference engines fall back to CPU-only mode.

## Best Practices

1. **Always Use Persistence Mode** for AI workloads
2. **Set Power Limits** to maximum for inference
3. **Monitor PCIe Generation** to ensure Gen5 speeds
4. **Keep Driver and CUDA Versions Compatible**
5. **Use Server/Compute Focused Settings** over desktop defaults

## Next Steps

- Continue to **[CUDA Documentation](cuda.md)** to understand the compute layer
- Review **[GPU Optimizations](../../optimizations/README.md)** for performance tuning
- Check **[System Monitoring](../../optimizations/README.md)** for ongoing maintenance

---

