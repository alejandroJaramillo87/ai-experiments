# BIOS Optimizations Guide

Comprehensive BIOS optimization guide for the AMD Ryzen 9950X + RTX 5090 AI engineering workstation, focusing on firmware-level performance tuning for **minimum inference latency** in interactive chatbot-style LLM applications (llama.cpp).

This documentation provides detailed BIOS/UEFI firmware configuration settings specifically tuned for single-model LLM inference with an emphasis on first-token latency and consistent token generation timing on the Gigabyte X870E Aorus Elite WiFi motherboard platform.

**Note**: These optimizations build upon the foundational BIOS configuration in `docs/bios/README.md` and provide advanced performance tuning for AI workloads. Review base configuration requirements before implementing these performance optimizations.

## Table of Contents

- [BIOS Optimizations Guide](#bios-optimizations-guide)
  - [Table of Contents](#table-of-contents)
  - [Implementation Overview](#implementation-overview)
    - [Latency vs Throughput: A Critical Trade-off](#latency-vs-throughput-a-critical-trade-off)
  - [Production-Validated Configuration](#production-validated-configuration)
    - [Validated CPU Settings](#validated-cpu-settings)
    - [Validated Memory Settings](#validated-memory-settings)
    - [Benchmark Validation Results](#benchmark-validation-results)
  - [BIOS Performance Optimizations](#bios-performance-optimizations)
    - [CPU Performance Configuration](#cpu-performance-configuration)
    - [Memory Performance Optimization](#memory-performance-optimization)
    - [AVX-512 Optimization (Zen 5 Advantage)](#avx-512-optimization-zen-5-advantage)
    - [GPU Performance Configuration](#gpu-performance-configuration)
  - [Reference Implementation](#reference-implementation)

## Implementation Overview

The AI engineering workstation optimization strategy focuses on **minimizing inference latency** for interactive chatbot-style LLM workloads using llama.cpp. These optimizations prioritize single-request response time over multi-request throughput, aligning with how llama.cpp and similar CPU inference engines are designed to operate.

**Note**: These optimizations are specific to llama.cpp latency goals. vLLM users should focus on different settings that optimize for throughput rather than single-request latency.

### Latency vs Throughput: A Critical Trade-off

CPU-based LLM inference presents a fundamental choice between optimizing for:
- **Throughput**: Processing many requests simultaneously (batch processing, API servers)
- **Latency**: Minimizing response time for individual requests (chatbots, interactive assistants)

This guide optimizes for **latency**, with validated benchmarks showing 26 percent performance improvement for interactive workloads.

**Key Optimization Features:**
- **12-Core Dedication**: Allocates 12 cores to single LLM instance (cores 0-11)
- **Memory Performance**: Advanced DDR5-6000 optimization for model parameter access
- **Power Management**: Disable power-saving features that introduce latency variance
- **Single-Model Focus**: All optimizations target one high-performance model instance
- **FCLK Optimization**: 2100 MHz interconnect for maximum bandwidth

**Hardware Configuration:**
- **Motherboard**: Gigabyte X870E Aorus Elite WiFi (AMD X870E chipset)
- **CPU**: AMD Ryzen 9950X (16 cores, 32 threads, Zen 5 architecture)
- **Memory**: G.SKILL Flare X5 128GB (2 x 64GB) DDR5-6000 CL36 with EXPO support
- **GPU**: RTX 5090 32GB VRAM (Blackwell architecture, PCIe 5.0 x16)

**Performance Optimization Priorities:**
- Minimum first-token latency for interactive chat responses
- Consistent token generation timing without variance
- Dedicated core resources for single high-performance LLM instance
- Predictable performance for real-time conversational AI

## Production-Validated Configuration

The following settings have been validated in production with a **26.2 percent performance improvement**:

### Validated CPU Settings
- **PBO Limits**: Motherboard
- **PBO Scalar**: 1X
- **CPU Boost Clock Override**: 200 MHz increase (maximum recommended)
- **Curve Optimizer**: -15 (conservative, stable)
- **Curve Shaper**:
  - Med Frequency - Med Temperature: -20
  - Med Frequency - High Temperature: -25
  - Max Frequency - High Temperature: -10
- **Global C-States**: Disabled
- **ECO Mode**: Disabled
- **X3D Turbo Mode**: Disabled
- **AVX-512**: Enabled

### Validated Memory Settings
- **XMP/EXPO High Bandwidth Support**: Enabled
- **Core Tuning Config**: Auto
- **Infinity Fabric Frequency**: 2100 MHz (critical for performance)
- **UCLK DIV1 MODE**: UCLK=MEMCLK
- **Power Down Enable**: Disabled
- **Memory Context Restore**: Disabled (more aggressive than Auto)
- **CSM Support**: Disabled

### Benchmark Validation Results

Measured at the previous 12-core CPU configuration; the current llama-cpu service runs 16 cores (cpuset 0-15).

- **Before optimizations**: 28.09 tokens/second average
- **After optimizations**: 35.44 tokens/second average
- **Performance gain**: 26.2 percent increase
- **Consistency improvement**: 59 percent reduction in standard deviation

## BIOS Performance Optimizations

### CPU Performance Configuration

**AMD Zen 5 Maximum Performance Settings**
Advanced CPU configuration for optimal AI inference performance, maximizing processing capability while maintaining system stability under sustained computational workloads.

**CPU Performance Settings:**
- **Global C-States**: `Disabled`
  - **Purpose**: Eliminates CPU sleep states for minimum first-token latency
  - **Benefit**: Instant response without wake-up delays
  - **Impact**: Cores remain at full readiness for immediate token generation

- **Precision Boost Overdrive**: `Advanced Configuration`
  - **Purpose**: Maximizes sustained performance for 12-core AI workload
  - **Configuration**: 
    - **PBO Mode**: `Advanced`
    - **PBO Limits**: `Motherboard` 
    - **PBO Scalar**: `1X` (Auto) - critical for longevity
  - **Benefit**: Provides maximum power headroom for sustained AI inference

- **ECO mode**: `Disabled`
  - **Purpose**: Ensures maximum power delivery for consistent token timing
  - **Benefit**: Eliminates power-related latency variations
  - **Impact**: Maintains peak performance for single-model inference

- **CPU Boost Clock Override**: `100 to 200 MHz increase`
  - **Purpose**: Extends maximum boost frequency ceiling for token generation
  - **Benefit**: Higher peak performance for single-threaded operations
  - **Configuration**: Start with 100 MHz increase, test stability before 200 MHz increase
  - **Safety**: CPU's internal FIT monitoring prevents dangerous voltages

- **Curve Optimizer**: `Negative -15 to -25` ⚠️ **PERFORMANCE CRITICAL**
  - **Purpose**: Undervolting to create thermal/power headroom for higher clocks
  - **Configuration**:
    - **Mode**: `Per CCD` or `All Cores`
    - **Sign**: `Negative`
    - **Validated**: `-15` (stable, achieved 26% performance improvement)
    - **Maximum**: `-20 to -25` (potential for additional gains with extensive testing)
    - **Method**: Start conservative, increase incrementally, validate with CoreCycler
  - **Benefit**: Reduces power consumption (P∝V²), creates headroom for higher sustained frequencies
  - **Impact**: Validated 26% improvement with conservative -15 setting
  - **Revalidation**: Validated against bursty chat inference only; sustained combined CPU+GPU workloads (training runs) require the stability soak in `docs/verification/hardware-stress.md` before first use

- **Curve Shaper**: `Frequency-Specific Optimization` (Zen 5 Feature)
  - **Purpose**: Granular voltage control for different frequency bands
  - **Validated Configuration**:
    - **Med Frequency - Med Temperature**: `-20`
    - **Med Frequency - High Temperature**: `-25`
    - **Max Frequency - High Temperature**: `-10`
  - **Theoretical Maximum**:
    - **Medium Frequency**: `-30` (more aggressive, requires validation)
    - **Max Frequency**: `-15` (maintains boost stability)
  - **Benefit**: Fine-tuned optimization for AI workload frequency range
  - **Advantage**: Eliminates single-offset limitation of traditional Curve Optimizer

- **X3D Turbo Mode**: `Disabled`
  - **Purpose**: Maintains standard Zen 5 operation (9950X does not have 3D V-Cache)
  - **Note**: Setting appropriate for non-X3D CPU configuration

**Latency Optimization Benefits:**
- **Optimized Cache Access**: Enhanced cache utilization patterns
- **Reduced First-Token Latency**: 26 percent validated improvement in tokens/second
- **Consistent Token Timing**: Reduced variance provides predictable generation
- **Single-Model Performance**: 12 cores dedicated to one LLM for maximum speed

**Core Allocation Strategy:**
- **Cores 0-11**: Dedicated to LLM inference
- **Remaining cores**: Reserved for OS, system tasks, and housekeeping
- **NUMA Optimization**: Primarily uses first CCD for cache locality
- **Docker Configuration (benchmark run)**: Single container with `cpuset: "0-11"` and `THREADS=12`. The current llama-cpu service in `docker-compose.inference.yaml` uses `cpuset: "0-15"` and `THREADS=16`.

### Memory Performance Optimization

**DDR5-6000 Advanced Performance Configuration**
Enhanced memory configuration building upon EXPO profile activation to maximize memory bandwidth and minimize latency for AI model parameter access and large dataset processing.

**Memory Performance Settings:**
- **XMP/EXPO Profile**: `DDR5-6000` (Primary)
  - **Purpose**: Foundational high-speed memory configuration
  - **Benefit**: Optimal balance of bandwidth and latency for Zen 5
  - **Configuration**: Enable highest stable EXPO profile for DDR5-6000 kit

- **Infinity Fabric Clock (FCLK)**: `2100 MHz` **BANDWIDTH CRITICAL**
  - **Purpose**: Eliminates interconnect bottleneck for 12-core AI workload
  - **Configuration**:
    - **FCLK**: `2100 MHz` (up from 1800 MHz default)
    - **Ratio**: Maintain 1:1:1 with UCLK/MCLK where possible
    - **AI Cache Boost**: `Enabled` (ASUS boards - automates FCLK optimization)
  - **Benefit**: Research shows 15 percent performance uplift in LLM workloads
  - **Validation**: Monitor stability, may require SoC voltage adjustment

- **Memory Timing Optimization**: `Enhanced Beyond EXPO`
  - **Purpose**: Further latency reduction through tighter sub-timings
  - **Configuration**: Use motherboard "Tighter" or "High-Efficiency" presets
  - **Method**: MSI "High-Efficiency Tightest", ASUS memory optimization presets
  - **Validation**: Always follow with memory stress testing

- **Power Down Enable**: `Disabled`
  - **Purpose**: Prevents memory modules from entering power-saving states
  - **Benefit**: Eliminates memory wake-up delays during intensive AI operations
  - **Impact**: Consistent memory performance for sustained AI workloads

- **Memory Context Restore**: `Disabled` (Validated) or `Auto`
  - **Purpose**: Controls memory context preservation across power states
  - **Validated**: `Disabled` - More aggressive, worked well in production
  - **Conservative**: `Auto` - Maintains memory training stability
  - **Benefit**: Disabled setting provides slightly better performance when stable

**Memory Architecture Optimization:**
- **Bandwidth Maximization**: FCLK 2100 MHz feeds 12 cores without bottleneck
- **Latency Minimization**: Tighter sub-timings reduce parameter access delays
- **Synchronous Operation**: 1:1:1 ratio prevents async penalty
- **Cache Efficiency**: Optimized FCLK improves cache hit rates
- **AVX-512 Native Support**: Full 512-bit datapath utilizes memory bandwidth

### AVX-512 Optimization (Zen 5 Advantage)

**Native 512-bit Datapath Configuration**
Zen 5 implements a native 512-bit AVX-512 datapath, which increases vector throughput for AI workloads compared to the double-pumped 256-bit implementation in prior generations.

**AVX-512 Settings:**
- **AVX-512 Support**: `Enabled` (default, but verify)
  - **Purpose**: Unlocks native 512-bit vector processing for AI frameworks
  - **Benefit**: 2x theoretical throughput vs Zen 4's "double-pumped" 256-bit
  - **Impact**: Massive performance gains in PyTorch, TensorFlow, llama.cpp
  - **Efficiency**: No significant power/thermal penalty on Zen 5

**Latency-Focused Memory Benefits:**
- **Model Loading**: Fast initial model load for quick first response
- **Parameter Streaming**: Optimized bandwidth for sequential token generation
- **Cache Utilization**: Better L3 cache efficiency without SMT contention
- **Memory Allocation**: Single container gets full memory bandwidth

### GPU Performance Configuration

**RTX 5090 PCIe 5.0 Optimization**
Firmware-level GPU configuration for maximum AI inference throughput with RTX 5090 Blackwell architecture.

**GPU BIOS Settings:**
- **PCIe x16 Slot Link**: `Gen5`
  - **Purpose**: Enable full PCIe 5.0 bandwidth for RTX 5090
  - **Benefit**: Maximum 128 GB/s bidirectional throughput
  - **Impact**: Critical for large model parameter transfers

- **PCIe Slot Link Speed**: `Gen5`
  - **Purpose**: Ensures PCIe 5.0 operation at full speed
  - **Benefit**: No bandwidth bottlenecks for GPU-CPU communication
  - **Verification**: Confirm Gen5 x16 in nvidia-smi after boot

- **Above 4G Decoding**: `Enabled`
  - **Purpose**: Allows addressing of RTX 5090's 32GB VRAM
  - **Requirement**: Essential for large BAR support
  - **Impact**: Full GPU memory accessible to CPU

- **Resizable BAR**: `Enabled`
  - **Purpose**: CPU can access entire GPU VRAM as single block
  - **Benefit**: Improved data transfer efficiency
  - **Impact**: Up to 15% performance gain in AI workloads

- **Initial Display Output**: `IGD Video`
  - **Purpose**: Routes display through integrated graphics
  - **Benefit**: RTX 5090 dedicated exclusively to compute
  - **Configuration**: Preserves GPU resources for AI inference

**GPU Performance Benefits:**
- **Maximum Bandwidth**: Full PCIe 5.0 x16 bandwidth utilized
- **VRAM Access**: Complete 32GB VRAM addressable by CPU
- **Compute Dedication**: GPU resources preserved for AI workloads
- **Transfer Efficiency**: Resizable BAR improves model loading speed


## Reference Implementation

**Hardware Optimization Integration:**
This optimization guide works in conjunction with the complete AI workstation configuration:

**System Integration:**
- **BIOS Foundation**: Base configuration requirements in `docs/bios/README.md`
- **Hardware Platform**: Component specifications in `docs/hardware/README.md`
- **OS Optimizations**: System-level performance tuning in `docs/optimizations/os/os-optimizations.md`
- **Huge Pages Setup**: Memory optimization in `hugepages/docs/hugepages-explicit.md`
- **Container Deployment**: Single latency-optimized service in `docker-compose.inference.yaml`

**Performance Validation:**
Verify optimization success through:
- CPU-Z confirmation of CPU boost frequencies and memory speeds
- GPU-Z validation of RTX 5090 performance states and memory bandwidth
- AI inference benchmarking with actual model workloads
- System stability testing under sustained AI computational loads

**Optimization Philosophy:**
All optimizations prioritize **inference latency** over throughput. These settings are specifically designed for interactive chatbot applications where response time matters more than handling multiple simultaneous requests. The configuration trades multi-request capacity for single-request speed.


---

*This hardware optimization guide minimizes inference latency on the AMD Ryzen 9950X + RTX 5090 AI engineering workstation. Validated Zen 5 optimizations including Curve Optimizer (-15), Curve Shaper, and FCLK 2100 MHz achieved 26.2% performance improvement in testing. Settings prioritize stability and consistency for interactive chatbot applications.*
