# BIOS Optimizations Guide

Comprehensive BIOS optimization guide for the AMD Ryzen 9950X + RTX 5090 AI engineering workstation, focusing on firmware-level performance tuning for **minimum inference latency** in interactive chatbot-style LLM applications (llama.cpp).

This documentation provides detailed BIOS/UEFI firmware configuration settings specifically tuned for single-model LLM inference with an emphasis on first-token latency and consistent token generation timing on the Gigabyte X870E Aorus Elite WiFi motherboard platform.

**Note**: These optimizations build upon the foundational BIOS configuration in `docs/bios/README.md` and provide advanced performance tuning for AI workloads. Review base configuration requirements before implementing these performance optimizations.

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
- **Docker Configuration**: Single container with `cpuset: "0-11"` and `THREADS=12`

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

### Advanced System Configuration

**Platform Optimization for Single-Model Latency**
System-level configuration settings optimizing the AMD X870E platform for minimum-latency AI inference with 12-core dedication strategy.

**System Configuration Settings:**
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

## Benchmark Results

### Test Configuration
- **Model**: Qwen3-Coder-30B-A3B-Instruct (IQ4_XS quantization)
- **Test Suite**: Six workload patterns for comprehensive validation
- **Methodology**: 5 iterations per test for consistency measurement
- **Platform**: llama.cpp with 12 dedicated cores

### Performance Improvements by Workload Type

| Workload | Before (tok/s) | After (tok/s) | Improvement |
|----------|----------------|---------------|-------------|
| Memory Sequential | 28.19 | 35.14 | +24.7% |
| Cache Loops | 28.32 | 35.44 | +25.1% |
| Compute Arithmetic | 27.29 | 35.50 | +30.1% |
| Memory Structured | 28.22 | 35.50 | +25.8% |
| Throughput Sustained | 28.29 | 35.52 | +25.6% |
| Memory Bandwidth | 28.20 | 35.52 | +26.0% |

### Consistency Improvements
- **Before**: Standard deviation 0.08-0.84 across workloads
- **After**: Standard deviation 0.08-0.28 across workloads
- **Result**: 59% reduction in performance variance

**Validation Methodology:**
Rigorous testing protocol for stability validation:

**Stage 1: Basic Stability**
- Boot stability and 30-minute idle test
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

### Production Validation Notes

**Key Findings from Production Testing:**
1. **Conservative settings work**: Curve Optimizer at -15 achieved excellent 26% gains
2. **FCLK 2100 MHz stable**: No SoC voltage adjustment required on test system
3. **Memory Context Restore disabled**: More aggressive than Auto, but stable
4. **Consistency matters**: Reduced variance is as valuable as raw performance

**Potential for Further Optimization:**
- After extended stability testing, Curve Optimizer could be pushed to -20
- Curve Shaper Medium Frequency could potentially reach -30
- These adjustments may yield additional 3-5% improvement

---

*This hardware optimization guide minimizes inference latency on the AMD Ryzen 9950X + RTX 5090 AI engineering workstation. Validated Zen 5 optimizations including Curve Optimizer (-15), Curve Shaper, and FCLK 2100 MHz achieved 26.2% performance improvement in testing. Settings prioritize stability and consistency for interactive chatbot applications.*

*Last Updated: 2025-09-23*