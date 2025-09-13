# BIOS Optimizations Guide

Comprehensive BIOS optimization guide for the AMD Ryzen 9950X + RTX 5090 AI engineering workstation, focusing on firmware-level performance tuning for **minimum inference latency** in interactive chatbot-style LLM applications.

This documentation provides detailed BIOS/UEFI firmware configuration settings specifically tuned for single-model LLM inference with an emphasis on first-token latency and consistent token generation timing on the Gigabyte X870E Aorus Elite WiFi motherboard platform.

> Note: These optimizations build upon the foundational BIOS configuration in `docs/bios/README.md` and provide advanced performance tuning for AI workloads. Review base configuration requirements before implementing these performance optimizations.

## Table of Contents

- [BIOS Optimizations Guide](#bios-optimizations-guide)
  - [Table of Contents](#table-of-contents)
  - [Implementation Overview](#implementation-overview)
  - [BIOS Performance Optimizations](#bios-performance-optimizations)
    - [CPU Performance Configuration](#cpu-performance-configuration)
    - [Memory Performance Optimization](#memory-performance-optimization)
    - [System Configuration Settings](#system-configuration-settings)
  - [OS Performance Optimizations](#os-performance-optimizations)
  - [Reference Implementation](#reference-implementation)

## Implementation Overview

The AI engineering workstation optimization strategy focuses on **minimizing inference latency** for interactive chatbot-style LLM workloads. These optimizations prioritize single-request response time over multi-request throughput, aligning with how llama.cpp and similar CPU inference engines are designed to operate.

### Latency vs Throughput: A Critical Trade-off

CPU-based LLM inference presents a fundamental choice between optimizing for:
- **Throughput**: Processing many requests simultaneously (batch processing, API servers)
- **Latency**: Minimizing response time for individual requests (chatbots, interactive assistants)

This guide optimizes for **latency**, based on research showing that disabling SMT and dedicating cores to a single model provides 20-30% better response times for interactive workloads.

**Key Optimization Features:**
- **SMT Disabled**: Eliminates thread contention for exclusive L1/L2 cache access per core
- **12-Core Dedication**: Allocates 12 physical cores to single LLM instance (cores 0-11)
- **Memory Performance**: Advanced DDR5-6000 optimization for model parameter access  
- **Power Management**: Disable power-saving features that introduce latency variance
- **Single-Model Focus**: All optimizations target one high-performance model instance

**Hardware Configuration:**
- **Motherboard**: Gigabyte X870E Aorus Elite WiFi (AMD X870E chipset)
- **CPU**: AMD Ryzen 9950X (16 cores, 32 threads, Zen 5 architecture)
- **Memory**: G.SKILL Flare X5 128GB (2 x 64GB) DDR5-6000 CL36 with EXPO support
- **GPU**: RTX 5090 32GB VRAM (Blackwell architecture, PCIe 5.0 x16)

**Performance Optimization Priorities:**
- Minimum first-token latency for interactive chat responses
- Consistent token generation timing without variance
- Exclusive core resources for single high-performance LLM instance
- Predictable performance for real-time conversational AI

## BIOS Performance Optimizations

### CPU Performance Configuration

**AMD Zen 5 Maximum Performance Settings**
Advanced CPU configuration for optimal AI inference performance, maximizing processing capability while maintaining system stability under sustained computational workloads.

**CPU Performance Settings:**
- **SMT (Simultaneous Multithreading)**: `Disabled` ⚠️ **CRITICAL SETTING**
  - **Purpose**: Provides exclusive L1/L2 cache access per physical core
  - **Benefit**: 20-30% reduction in inference latency for chatbot workloads
  - **Impact**: 16 physical cores operate as 16 threads (not 32)
  - **Rationale**: Research shows SMT hurts latency-sensitive workloads due to:
    - Shared L1/L2 cache contention between logical threads
    - Thread scheduling overhead and context switching
    - Memory bandwidth competition between sibling threads
    - Reduced per-thread cache capacity

- **Global C-States**: `Disabled`
  - **Purpose**: Eliminates CPU sleep states for minimum first-token latency
  - **Benefit**: Instant response without wake-up delays
  - **Impact**: Cores remain at full readiness for immediate token generation

- **Precision Boost Overdrive**: `Advanced Configuration`
  - **Purpose**: Maximizes sustained performance for 12-core AI workload
  - **Configuration**: 
    - **PBO Mode**: `Advanced`
    - **PBO Limits**: `Motherboard` (or manual values below)
    - **PPT (Package Power)**: `200W` (optimized for 12-core focus)
    - **TDC (Thermal Design Current)**: `160A`
    - **EDC (Electrical Design Current)**: `225A`
    - **PBO Scalar**: `1X` (Auto) - critical for longevity
  - **Benefit**: Provides maximum power headroom for sustained AI inference

- **ECO mode**: `Disabled`
  - **Purpose**: Ensures maximum power delivery for consistent token timing
  - **Benefit**: Eliminates power-related latency variations
  - **Impact**: Maintains peak performance for single-model inference

- **CPU Boost Clock Override**: `+100MHz to +200MHz`
  - **Purpose**: Extends maximum boost frequency ceiling for token generation
  - **Benefit**: Higher peak performance for single-threaded operations
  - **Configuration**: Start with +100MHz, test stability before +200MHz
  - **Safety**: CPU's internal FIT monitoring prevents dangerous voltages

- **Curve Optimizer**: `Negative -20 to -25` ⚠️ **PERFORMANCE CRITICAL**
  - **Purpose**: Undervolting to create thermal/power headroom for higher clocks
  - **Configuration**:
    - **Mode**: `Per CCD` or `All Cores`
    - **Sign**: `Negative`
    - **Magnitude**: `-20` (conservative start), target `-25` after testing
    - **Method**: Start conservative, increase incrementally, validate with CoreCycler
  - **Benefit**: Reduces power consumption (P∝V²), creates headroom for higher sustained frequencies
  - **Impact**: Can provide 10-15% performance improvement through higher sustained clocks

- **Curve Shaper**: `Frequency-Specific Optimization` (Zen 5 Feature)
  - **Purpose**: Granular voltage control for different frequency bands
  - **Configuration**:
    - **Medium Frequency**: `Negative -30` (where AI inference primarily operates)
    - **Max Frequency**: `Negative -15` (maintains boost stability)
    - **Low Frequency**: `Negative -10` (prevents idle crashes)
  - **Benefit**: Aggressive optimization for AI workload frequency range
  - **Advantage**: Eliminates single-offset limitation of traditional Curve Optimizer

- **X3D Turbo Mode**: `Disabled`
  - **Purpose**: Maintains standard Zen 5 operation (9950X does not have 3D V-Cache)
  - **Note**: Setting appropriate for non-X3D CPU configuration

**Latency Optimization Benefits:**
- **Exclusive Cache Access**: Each core has full L1/L2 cache without SMT sharing
- **Reduced First-Token Latency**: 20-30% improvement over SMT-enabled configuration
- **Consistent Token Timing**: Eliminated thread contention provides predictable generation
- **Single-Model Performance**: All 12 cores dedicated to one LLM for maximum speed

**Core Allocation Strategy:**
- **Cores 0-11**: Dedicated to LLM inference (12 physical cores)
- **Cores 12-15**: Reserved for OS, system tasks, and housekeeping
- **NUMA Optimization**: Primarily uses first CCD for cache locality
- **Docker Configuration**: Single container with `cpuset: "0-11"` and `THREADS=12`

### Memory Performance Optimization

**DDR5-6000 Advanced Performance Configuration**
Enhanced memory configuration building upon EXPO profile activation to maximize memory bandwidth and minimize latency for AI model parameter access and large dataset processing.

**Memory Performance Settings:**
- **XMP/EXPO Profile**: `DDR5-6000` (Primary)
  - **Purpose**: Foundational high-speed memory configuration
  - **Benefit**: Optimal balance of bandwidth and latency for Zen 5
  - **Configuration**: Enable highest stable EXPO profile for DDR5-6000 kit

- **Infinity Fabric Clock (FCLK)**: `2100MHz` ⚠️ **BANDWIDTH CRITICAL**
  - **Purpose**: Eliminates interconnect bottleneck for 12-core AI workload
  - **Configuration**:
    - **FCLK**: `2100MHz` (up from 1800MHz default)
    - **Ratio**: Maintain 1:1:1 with UCLK/MCLK where possible
    - **AI Cache Boost**: `Enabled` (ASUS boards - automates FCLK optimization)
  - **Benefit**: Research shows 15% performance uplift in LLM workloads
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

- **Memory Context Restore**: `Auto`
  - **Purpose**: Allows BIOS to automatically manage memory context preservation
  - **Benefit**: Maintains memory training stability while optimizing for performance
  - **Setting**: Auto provides optimal balance for high-capacity configurations

**Memory Architecture Optimization:**
- **Bandwidth Maximization**: FCLK 2100MHz feeds 12 cores without bottleneck
- **Latency Minimization**: Tighter sub-timings reduce parameter access delays
- **Synchronous Operation**: 1:1:1 ratio prevents async penalty
- **Cache Efficiency**: SMT-disabled + optimized FCLK improves cache hit rates
- **AVX-512 Native Support**: Full 512-bit datapath utilizes memory bandwidth

### AVX-512 Optimization (Zen 5 Advantage)

**Native 512-bit Datapath Configuration**
Zen 5's revolutionary native 512-bit AVX-512 implementation provides massive AI performance gains.

**AVX-512 Settings:**
- **AVX-512 Support**: `Enabled` (default, but verify)
  - **Purpose**: Unlocks native 512-bit vector processing for AI frameworks
  - **Benefit**: 2x theoretical throughput vs Zen 4's "double-pumped" 256-bit
  - **Impact**: Massive performance gains in PyTorch, TensorFlow, llama.cpp
  - **Efficiency**: No significant power/thermal penalty on Zen 5

- **AVX-512 VNNI**: `Enabled`
  - **Purpose**: Accelerates convolution and inner-product operations
  - **Benefit**: Optimized for deep learning mathematical operations
  - **Target**: Neural network inference and training workloads

- **bfloat16 Optimization**: `Auto`
  - **Purpose**: Enhanced support for AI-optimized data types
  - **Benefit**: Balance between FP32 range and FP16 efficiency
  - **Configuration**: Automatic optimization for AI frameworks

**Latency-Focused Memory Benefits:**
- **Model Loading**: Fast initial model load for quick first response
- **Parameter Streaming**: Optimized bandwidth for sequential token generation  
- **Cache Utilization**: Better L3 cache efficiency without SMT contention
- **Memory Allocation**: Single container gets full memory bandwidth

### Advanced System Configuration

**Platform Optimization for Single-Model Latency**
System-level configuration settings optimizing the AMD X870E platform for minimum-latency AI inference with 12-core dedication strategy.

**System Configuration Settings:**
- **CPU Core Count**: `16 cores (SMT Disabled)`
  - **Purpose**: Verify SMT disabled shows 16 threads, not 32
  - **Validation**: Check with `nproc` command in Linux
  - **Benefit**: Confirms exclusive L1/L2 cache access per core
  - **Impact**: Critical for latency optimization validation

- **ASUS AI Cache Boost**: `Enabled` (ASUS X870E boards)
  - **Purpose**: One-click FCLK optimization to 2100MHz for AI workloads
  - **Benefit**: Automatic interconnect optimization with 15% LLM performance gain
  - **Alternative**: Manual FCLK setting to 2100MHz on non-ASUS boards
  - **Compensation**: Increase PPT by 15-20W to account for higher I/O die power

- **Samsung/AMD Eco Mode**: `Disabled`
  - **Purpose**: Disables Samsung SSD and AMD platform power-saving features
  - **Benefit**: Maintains consistent storage and system performance under AI workloads
  - **Impact**: Optimal storage throughput for model loading and data processing

- **Initial Display Output**: `IGD Video`
  - **Purpose**: Routes initial display output through integrated graphics (AMD RDNA)
  - **Benefit**: Preserves RTX 5090 resources exclusively for AI computational tasks
  - **Configuration**: Enables headless GPU operation for AI inference optimization

**Platform Integration Benefits:**
- **GPU Resource Preservation**: RTX 5090 resources dedicated entirely to AI computation
- **Storage Performance**: Consistent NVMe SSD performance for model and dataset access
- **System Efficiency**: Reduced platform overhead for maximum AI workload resource allocation
- **Power Management**: Optimal power delivery prioritizing computational performance

**AI Development Optimization:**
- **Headless Operation**: RTX 5090 available exclusively for CUDA-based AI inference
- **Resource Allocation**: Maximum system resources dedicated to AI computational tasks
- **Storage Consistency**: Reliable high-performance storage access for AI development workflows
- **Platform Stability**: Optimized system configuration for sustained AI development operations


## Reference Implementation

**Hardware Optimization Integration:**
This optimization guide works in conjunction with the complete AI workstation configuration:

**System Integration:**
- **BIOS Foundation**: Base configuration requirements in `docs/bios/README.md`
- **Hardware Platform**: Component specifications in `docs/hardware/README.md`
- **OS Optimizations**: System-level performance tuning in `docs/optimizations/os-optimizations.md`
- **Huge Pages Setup**: Memory optimization in `docs/optimizations/hugepages-setup.md`
- **Container Deployment**: Single latency-optimized service in `docker-compose.yaml`

**Performance Validation:**
Verify optimization success through:
- CPU-Z confirmation of CPU boost frequencies and memory speeds
- GPU-Z validation of RTX 5090 performance states and memory bandwidth
- AI inference benchmarking with actual model workloads
- System stability testing under sustained AI computational loads

**Optimization Philosophy:**
All optimizations prioritize **inference latency** over throughput. These settings are specifically designed for interactive chatbot applications where response time matters more than handling multiple simultaneous requests. The configuration trades multi-request capacity for single-request speed.

**Validation Methodology:**
Rigorous testing protocol for stability validation:

**Stage 1: Basic Stability**
- Boot stability and 30-minute idle test
- Verify SMT disabled: `nproc` shows 16 (not 32)
- Check FCLK: Ensure 2100MHz in monitoring tools

**Stage 2: All-Core Thermal**
- Cinebench R23 30-minute loop
- Monitor temperatures <85°C target
- Validate sustained boost clocks

**Stage 3: Memory/Fabric Stability**
- TestMem5 or Karhu RAM Test overnight
- Validates FCLK 2100MHz stability
- Zero errors required for production use

**Stage 4: Per-Core Validation**
- CoreCycler with y-cruncher (19-ZN2 Kagari mode)
- Tests individual core stability at max boost
- Validates Curve Optimizer settings
- 30-60 minutes per core for confidence

**Performance Monitoring:**
- `htop` and `nvtop` for CPU utilization patterns
- AI framework performance metrics during model inference
- Token generation latency and consistency measurements
- First-token latency optimization validation

---

*This hardware optimization guide minimizes inference latency on the AMD Ryzen 9950X + RTX 5090 AI engineering workstation as of mid-2025. Advanced Zen 5 optimizations including Curve Optimizer (-20 to -25), Curve Shaper, and FCLK 2100MHz targeting single-model inference. Expected 15-25% total performance improvement through combined CPU, memory, and interconnect optimization for interactive chatbot applications.*