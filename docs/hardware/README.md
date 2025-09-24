# AI Engineering Hardware Build

Comprehensive hardware specification and configuration guide for the AMD Ryzen 9950X + RTX 5090 AI engineering workstation, optimized for running multiple models simultaneously across both GPU (VRAM) and CPU/RAM configurations with emphasis on thermal management and power efficiency.

This documentation provides detailed component analysis, installation guidance, and optimization strategies for sustained AI development workloads, including large language model inference, computer vision processing, and multi-container AI deployment scenarios.

**Note**: This build serves as a reference implementation for high-performance AI engineering workstations. Component selections prioritize AI workload performance, system stability, and thermal efficiency. Adapt specifications based on your specific requirements, budget, and regional availability.

## Table of Contents

- [AI Engineering Hardware Build](#ai-engineering-hardware-build)
  - [Table of Contents](#table-of-contents)
  - [Build Overview](#build-overview)
  - [Performance Specifications](#performance-specifications)
  - [Component Details](#component-details)
    - [Motherboard](#motherboard)
    - [CPU and Cooling](#cpu-and-cooling)
    - [Graphics Card](#graphics-card)
    - [Memory Configuration](#memory-configuration)
    - [Storage Layout](#storage-layout)
    - [Power Supply](#power-supply)
    - [Case and Cooling](#case-and-cooling)
    - [Networking](#networking)
  - [AI Workload Optimization](#ai-workload-optimization)
    - [GPU Configuration](#gpu-configuration)
    - [CPU Configuration](#cpu-configuration)
    - [Thermal Management](#thermal-management)
  - [Installation Notes](#installation-notes)
    - [Memory Installation](#memory-installation)
    - [Storage Configuration](#storage-configuration)
    - [PCIe Configuration](#pcie-configuration)

## Build Overview

The AI engineering workstation leverages a dual-processing architecture specifically designed to maximize AI development productivity through parallel GPU and CPU model execution strategies.

**Dual Processing Architecture:**
- **GPU-accelerated inference**: Large models (30B or larger parameters) loaded into 32GB GDDR7 VRAM for maximum throughput
- **CPU-based inference**: Multiple smaller models (7B to 13B parameters) running concurrently across 128GB DDR5 system memory
- **Hybrid workloads**: Simultaneous GPU and CPU model execution enabling complex AI pipelines and multi-modal workflows
- **Container optimization**: Hardware foundation supporting Docker-based AI model isolation and deployment

**AI Development Workflow Support:**
- **Model experimentation**: Rapid switching between different model architectures and sizes
- **Multi-model deployment**: Concurrent hosting of specialized models for different AI tasks
- **Development flexibility**: Support for both local development and production-ready containerized deployments
- **Performance scaling**: Architecture enables both single-model optimization and multi-model concurrency

## Performance Specifications

| Component | Model | Power Consumption | Notes |
|-----------|-------|------------------|-------|
| **Total System** | - | Approximately 800W under load | Estimated peak consumption |
| **GPU** | RTX 5090 32GB | Approximately 575W max load | Primary AI inference engine |
| **CPU** | AMD Ryzen 9 9950X | 170W TDP | 16-core Zen 5 for parallel processing |
| **Cooling** | Dark Rock Pro 5 | - | 270W TDP capacity with dual fans |

## Component Details

### Motherboard

**Gigabyte X870E Aorus Elite WiFi Platform Foundation**
The X870E chipset provides comprehensive support for AMD Zen 5 architecture with advanced I/O capabilities and expansion flexibility essential for AI development workstations.

**Platform Specifications:**
- **Socket**: AM5 (AMD Ryzen 9000 series with native DDR5 and PCIe 5.0 support)
- **Chipset**: AMD X870E with enhanced I/O and connectivity features
- **PCIe Configuration**: PCIe 5.0 x16 slot ensuring maximum RTX 5090 bandwidth utilization
- **Memory Support**: Native DDR5-6000+ with EXPO profile support for high-capacity configurations
- **Expansion**: Multiple M.2 slots with dedicated CPU and chipset connections

**AI Workstation Features:**
- **GPU Bandwidth**: Full PCIe 5.0 x16 lane allocation for maximum AI inference performance
- **Memory Capacity**: Support for 128GB+ DDR5 configurations essential for multi-model deployment
- **Storage Flexibility**: Multiple NVMe M.2 slots for model storage and system separation
- **Diagnostic Capabilities**: Onboard POST code display for efficient hardware troubleshooting

### CPU and Cooling

**AMD Ryzen 9 9950X Processing Foundation**
The Zen 5 architecture delivers exceptional parallel processing capabilities essential for CPU-based AI model inference and multi-container AI deployment scenarios.

**Zen 5 Architecture Specifications:**
- **Architecture**: Zen 5 (Granite Ridge) with advanced IPC improvements and AI instruction optimizations
- **Core Configuration**: 16 cores / 32 threads enabling massive parallel AI model inference
- **Clock Speeds**: 4.3 GHz base, up to 5.7 GHz boost for single-threaded AI preprocessing tasks
- **TDP**: 170W with dynamic power management for sustained AI workloads
- **Memory Controller**: Native DDR5-6000 support with optimized memory bandwidth for large models

**AI Processing Optimizations:**
- **Parallel Inference**: 32 threads support concurrent execution of multiple 7B to 13B parameter models
- **Memory Bandwidth**: Zen 5 memory controller optimized for high-capacity DDR5 configurations
- **Instruction Set**: Enhanced AI/ML instruction support for accelerated mathematical operations
- **Container Performance**: Optimal CPU resource allocation for Docker-based AI model deployment

**be quiet! Dark Rock Pro 5 Thermal Solution**
Professional-grade air cooling designed for sustained high-performance computing workloads with whisper-quiet operation.

**Cooling Specifications:**
- **TDP Capacity**: 270W thermal dissipation capability with significant headroom for sustained AI workloads
- **Configuration**: Dual-fan asymmetrical design with optimized airflow patterns
- **Compatibility**: Native AM5 socket support with secure mounting system
- **Acoustic Performance**: Ultra-quiet operation essential for 24/7 AI development environments

### Graphics Card

**Gigabyte Gaming GeForce RTX 5090 32GB AI Acceleration Engine**
The RTX 5090 represents the pinnacle of AI inference acceleration, featuring Blackwell architecture optimizations specifically designed for transformer models and large-scale AI workloads.

**Blackwell Architecture Specifications:**
- **VRAM Configuration**: 32GB GDDR7 with 896 GB/s memory bandwidth for large model residence
- **Compute Capability**: sm_120 architecture with advanced tensor processing units
- **Interface**: PCIe 5.0 x16 ensuring maximum data transfer rates for model loading
- **Power Consumption**: Approximately 575W maximum load with intelligent power management
- **Cooling**: Triple-fan open-air design optimized for sustained AI inference workloads

**AI Workload Optimization:**
- **Large Language Models**: Native support for 30B or larger parameter models in VRAM
- **Computer Vision**: Hardware-accelerated image processing and neural network inference
- **Generative AI**: Optimized for stable diffusion, image generation, and multimodal AI
- **Mixed Precision**: FP32, FP16, and experimental FP8 support for memory efficiency

**Professional AI Features:**
- **CUDA 12.9.1**: Latest CUDA toolkit support with Blackwell-specific optimizations
- **Tensor Cores**: Fourth-generation tensor cores for accelerated AI mathematical operations
- **NVLink Readiness**: Prepared for future multi-GPU scaling in advanced configurations
- **Container Integration**: Full Docker and NVIDIA Container Toolkit compatibility

### Memory Configuration

**G.SKILL Flare X5 128GB High-Capacity AI Memory System**
High-density DDR5 memory configuration specifically selected for multi-model AI deployment scenarios requiring substantial system memory allocation.

**Memory Specifications:**
- **Capacity**: 128GB (2 x 64GB) providing extensive memory for concurrent model hosting
- **Type**: DDR5-6000 (PC5-48000) with AMD EXPO profile optimization
- **Configuration**: Dual-channel installation in slots A1 and B1 for optimal memory controller utilization
- **Timings**: CL36-36-36-96 with optimized sub-timings for AI workload patterns
- **Voltage**: 1.35V EXPO specification with stable operation under sustained loads

**AI Workload Benefits:**
- **Multi-Model Hosting**: Concurrent execution of multiple 7B to 13B parameter models
- **Container Memory**: Extensive memory allocation for Docker-based AI model isolation
- **Data Processing**: Large dataset loading and preprocessing capabilities
- **Development Flexibility**: Memory headroom for experimental model configurations

**EXPO Profile Optimization:**
- **Performance**: DDR5-6000 speeds essential for CPU-based AI inference efficiency
- **Compatibility**: Native AMD Zen 5 memory controller optimization
- **Stability**: Validated memory configuration for sustained AI development workloads
- **Bandwidth**: Maximized memory throughput for large model parameter access

**Critical Configuration**: Install memory modules exclusively in slots A1 and B1 to maintain DDR5-6000 EXPO speeds. Adding additional modules significantly reduces memory frequency on current AM5 platforms, impacting AI workload performance.

### Storage Layout

**Dual-SSD Architecture for AI Workload Optimization**
Strategic storage configuration separating operating system from AI models and datasets for optimal performance and organization.

**Primary SSD: Samsung 990 Pro 2TB (AI Data Storage)**
High-performance NVMe storage dedicated to AI models, datasets, and active development projects.

**Storage Specifications:**
- **Interface**: PCIe 5.0 x4 with CPU-direct connection for maximum throughput
- **Performance**: Up to 12,400 MB/s sequential read optimized for large model loading
- **Capacity**: 2TB providing extensive storage for multiple large AI models
- **Mount Point**: `/mnt/ai-data` for organized AI asset management
- **Use Cases**: GGUF models, Safetensors, HuggingFace cache, Docker volumes

**Secondary SSD: Samsung 990 EVO 1TB (System Storage)**
Reliable system storage providing fast boot times and application performance.

**System Configuration:**
- **Interface**: PCIe 4.0 x4 chipset connection with dedicated bandwidth allocation
- **Operating System**: Ubuntu 24.04 LTS optimized for AI development workflows
- **Applications**: Development tools, Docker, CUDA toolkit, Python environments
- **Performance**: Optimized for system responsiveness and development tool access

**Storage Architecture Benefits:**
- **Performance Isolation**: Separate storage domains prevent I/O conflicts between system and AI operations
- **Organization**: Clear separation between system files and AI development assets
- **Backup Strategy**: Independent backup schedules for system and data partitions
- **Container Integration**: Dedicated AI storage mount points for Docker container access

**PCIe Lane Management**: This dual-SSD configuration preserves all PCIe 5.0 x16 lanes for RTX 5090 maximum bandwidth. Additional M.2 installations would reduce GPU performance by sharing PCIe lanes, impacting AI inference throughput.

### Power Supply

**Super Flower Leadex VII XP PRO 1200W Professional Power Platform**
High-efficiency, fully modular power supply designed for sustained high-performance computing workloads with comprehensive protections and modern connectivity standards.

**Power Supply Specifications:**
- **Capacity**: 1200W continuous power delivery with significant headroom for peak AI workloads
- **Efficiency**: 80+ Platinum and Cybenetics Platinum certification ensuring optimal power efficiency
- **Modularity**: Fully modular cable configuration for optimized airflow and cable management
- **Standards Compliance**: ATX 3.1 and PCIe 5.1 support for latest hardware compatibility

**AI Workstation Power Management:**
- **GPU Power Delivery**: Native 12VHPWR cables (2x 8-pin to 16-pin) for RTX 5090 power requirements
- **Load Distribution**: Balanced power delivery across multiple +12V rails for system stability
- **Transient Response**: Advanced power regulation for sudden AI workload power spikes
- **Protection Systems**: Comprehensive OVP, UVP, OCP, and SCP protection for hardware safety

**Sustained AI Workload Features:**
- **Thermal Management**: 140mm fluid dynamic bearing fan with intelligent speed control
- **24/7 Operation**: Designed for continuous operation supporting round-the-clock AI development
- **Power Efficiency**: High efficiency reduces heat generation and operating costs
- **Modular Connectivity**: Custom cable configuration optimized for AI workstation requirements

### Case and Cooling

**Corsair iCUE 5000D RGB Airflow Thermal Management System**
Professional-grade case design optimized for high-airflow thermal management supporting sustained AI workloads with comprehensive cooling infrastructure.

**Case Specifications:**
- **Form Factor**: Mid-tower ATX with optimized internal layout for airflow efficiency
- **Airflow Design**: High-airflow front panel and strategic ventilation for maximum thermal performance
- **Expansion Support**: Full support for RTX 5090 dimensions and professional cooling solutions
- **Build Quality**: Steel construction with tempered glass side panel for professional aesthetics

**Comprehensive Cooling Configuration:**
- **Total Fan Count**: 10 case fans providing positive pressure airflow optimization
- **Intake Configuration**: 3x front intake fans for cool air delivery to components
- **Exhaust Configuration**: 3x top exhaust + 1x rear exhaust for efficient heat removal
- **Additional Cooling**: 3x supplementary fans for optimal airflow distribution

**Thermal Management Strategy:**
- **Positive Pressure**: Intake fans exceed exhaust capacity reducing dust accumulation
- **Component Cooling**: Dedicated airflow paths for GPU, CPU, and memory thermal management
- **Sustained Operation**: Cooling capacity designed for 24/7 AI development workloads
- **Temperature Control**: Comprehensive thermal solution supporting peak performance operation

**Control System:**
- **Controllers**: iCUE Commander Core XT (x2) for comprehensive fan and RGB management
- **Monitoring**: Real-time temperature and fan speed monitoring capabilities
- **Customization**: Programmable fan curves optimized for AI workload thermal profiles

**Linux Compatibility Consideration**: Corsair iCUE software lacks native Linux support. Fan control and RGB customization require alternative solutions such as OpenRGB, liquidctl, or manual BIOS-based configuration for Ubuntu 24.04 environments.

### Networking

**Panda Wireless PAU0F AXE3000 High-Performance Wireless Connectivity**
Advanced WiFi 6E adapter providing reliable high-bandwidth connectivity for AI development workflows requiring cloud model access and remote development capabilities.

**Wireless Specifications:**
- **Standard**: WiFi 6E tri-band support (2.4 GHz, 5 GHz, 6 GHz) with latest 802.11ax features
- **Interface**: USB 3.0 connection ensuring adequate bandwidth for wireless throughput
- **Performance**: AXE3000 speeds supporting high-bandwidth AI development tasks
- **Linux Compatibility**: Native driver support for Ubuntu 24.04 without additional configuration

**AI Development Connectivity:**
- **Model Downloads**: High-speed connectivity for downloading large AI models from HuggingFace Hub
- **Remote Development**: Stable connection for SSH, remote Jupyter notebooks, and cloud GPU access
- **Container Registries**: Reliable connectivity for Docker image downloads and updates
- **Collaboration**: Bandwidth adequate for Git operations, video conferencing, and collaborative AI development

**Network Performance Features:**
- **6 GHz Band**: Access to less congested 6 GHz spectrum for maximum performance
- **MIMO Technology**: Multiple antenna configuration for improved signal strength and stability
- **Beamforming**: Directional signal optimization for improved connection quality
- **Enterprise Security**: WPA3 support and advanced encryption for secure development environments

## AI Workload Optimization

### GPU Configuration

**RTX 5090 AI Acceleration Optimization**
- **VRAM Utilization**: 32GB GDDR7 capacity supports large models (30B or larger parameters) with full model residence
- **Compute Capability**: Blackwell sm_120 architecture optimized for transformer and attention mechanisms
- **Model Deployment**: Single large model deployment or multiple smaller models with dynamic memory allocation
- **Memory Bandwidth**: 896 GB/s throughput ensuring rapid model parameter access and processing
- **Precision Support**: FP32, FP16, and experimental FP8 precision modes for memory efficiency optimization
- **Container Integration**: Full NVIDIA Container Toolkit support for isolated AI model deployment

### CPU Configuration

**AMD Zen 5 Multi-Model Processing Architecture**
- **Memory Bandwidth**: 128GB DDR5-6000 configuration enables concurrent hosting of multiple AI models
- **Thread Distribution**: 32 threads across 16 cores providing massive parallel inference capability
- **Model Capacity**: Optimal performance for 7B to 13B parameter models with efficient CPU-based inference
- **Container Allocation**: CPU pinning and memory allocation for isolated Docker-based model deployment
- **NUMA Optimization**: Memory controller configuration optimized for large model parameter access
- **Instruction Set**: Zen 5 AI/ML instruction enhancements for accelerated mathematical operations

### Thermal Management

**Comprehensive Cooling Architecture for Sustained AI Workloads**
- **GPU Cooling**: Triple-fan open-air design with optimized case airflow delivering cool air directly to RTX 5090
- **CPU Cooling**: be quiet! Dark Rock Pro 5 with 270W TDP capacity and dual-fan configuration for silent operation
- **Case Airflow**: 10-fan positive pressure configuration ensuring optimal component temperatures under load
- **Total Cooling**: 15 total fans (10 case + 2 CPU + 3 GPU) providing comprehensive thermal management
- **Thermal Monitoring**: Real-time temperature monitoring for all critical components during AI workloads
- **Sustained Performance**: Cooling capacity designed for 24/7 AI development and inference operations

## Installation Notes

### Memory Installation

**DDR5 High-Capacity Configuration Process**
1. **Slot Configuration**: Install G.SKILL Flare X5 modules exclusively in slots A1 and B1 for optimal dual-channel performance
2. **EXPO Activation**: Enable EXPO profile in Gigabyte UEFI for DDR5-6000 speeds (see `docs/bios/README.md`)
3. **Stability Validation**: Verify memory stability under sustained AI workloads using memtest and stress testing
4. **Performance Verification**: Confirm DDR5-6000 operation using `sudo dmidecode --type 17` in Ubuntu
5. **AI Workload Testing**: Test memory performance with actual AI model loading and inference scenarios

### Storage Configuration

**Dual-SSD Optimization Installation Process**
1. **Primary M.2 Installation**: Install Samsung 990 Pro 2TB in CPU-connected M.2_1 slot for maximum AI data throughput
2. **Secondary M.2 Installation**: Install Samsung 990 EVO 1TB in chipset M.2 slot (M.2_2 or M.2_3) avoiding GPU lane sharing
3. **PCIe Lane Preservation**: Leave remaining M.2 slots empty to maintain full PCIe 5.0 x16 bandwidth for RTX 5090
4. **Mount Point Configuration**: Configure `/mnt/ai-data` mount point for AI model and dataset storage (see `docs/os/README.md`)
5. **Permission Setup**: Configure appropriate user permissions for AI development workflow access

### PCIe Configuration

**GPU Bandwidth Optimization Setup**
1. **GPU Installation**: Install RTX 5090 in primary PCIe 5.0 x16 slot (closest to CPU) for maximum bandwidth
2. **Configuration Verification**: Verify full x16 lane allocation using `nvidia-smi` and PCIe monitoring tools
3. **Lane Preservation**: Avoid expansion cards that would reduce GPU to x8 configuration impacting AI performance
4. **BIOS Configuration**: Enable Resizable BAR and Above 4G Decoding for optimal GPU memory access
5. **Performance Validation**: Test GPU performance under AI workloads to confirm optimal PCIe utilization
---

---

*Last Updated: 2025-09-23*