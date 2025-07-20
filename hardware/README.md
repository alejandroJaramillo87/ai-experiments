# AI Engineering Hardware Build

A high-performance desktop PC specifically designed for AI engineering workloads, optimized for running multiple models simultaneously on both GPU (VRAM) and CPU/RAM configurations.
> Note: This build serves as a reference guide for the type of hardware specifications needed for AI engineering. Component choices should be adapted based on your specific requirements, budget, and availability rather than followed exactly.

## Build Overview

This system is engineered to support dual AI inference patterns:
- **GPU-accelerated inference**: One large model loaded into 32GB VRAM for maximum performance
- **CPU-based inference**: Multiple smaller models running concurrently on 128GB system RAM
- **Hybrid workloads**: Simultaneous GPU and CPU model execution for complex AI pipelines

## Performance Specifications

| Component | Model | Power Consumption | Notes |
|-----------|-------|------------------|-------|
| **Total System** | - | ~800W under load | Estimated peak consumption |
| **GPU** | RTX 5090 32GB | ~575W max load | Primary AI inference engine |
| **CPU** | AMD Ryzen 9 9950X | 170W TDP | 16-core Zen 5 for parallel processing |
| **Cooling** | Dark Rock Pro 5 | - | 270W TDP capacity with dual fans |

## Component Details

### Motherboard
**Gigabyte X870E Aorus Elite WiFi**
- Socket: AM5 (AMD Ryzen 9000 series)
- Chipset: AMD X870E
- PCIe 5.0 support for maximum GPU bandwidth

### CPU & Cooling
**AMD Ryzen 9 9950X**
- Architecture: Zen 5 (Granite Ridge)
- Cores/Threads: 16C/32T
- Base Clock: 4.3 GHz
- TDP: 170W
- Optimized for parallel AI model inference

**be quiet! Dark Rock Pro 5 (BK036)**
- TDP Capacity: 270W
- Configuration: Dual-fan air cooling
- Socket Support: AM5, AM4, Intel LGA series

### Graphics Card
**Gigabyte Gaming GeForce RTX 5090 32GB**
- VRAM: 32GB GDDR7
- Interface: PCIe 5.0 x16
- Power: ~575W max load
- Optimized for: Large language models, computer vision, generative AI

### Memory Configuration
**G.SKILL Flare X5 128GB (2 x 64GB)**
- Type: DDR5-6000 (PC5-48000)
- Configuration: Dual-channel in slots A1 and B1
- EXPO Profile: Enabled for optimal performance
- Use Case: Multiple concurrent CPU-based AI models

> **Important**: Only use 2 memory modules in slots A1 and B1 to maintain EXPO speeds. Adding 4 modules significantly reduces overclocking capability with current AM5 hardware.

### Storage Layout
**Primary SSD: Samsung 990 Pro 2TB**
- Interface: PCIe 5.0 x4 (CPU-direct connection)
- Use Case: AI model storage, datasets, active projects

**Secondary SSD: Samsung 990 EVO 1TB**
- Interface: PCIe 4.0 x4 (chipset connection)
- Current OS: Ubuntu 24.04 LTS
- Use Case: Operating system and applications

> **PCIe Lane Management**: Only these two M.2 slots are populated to preserve all PCIe 5.0 x16 lanes for the GPU. Additional M.2 slots would reduce GPU bandwidth.

### Power Supply
**Super Flower Leadex VII XP PRO 1200W**
- Efficiency: 80+ Platinum, Cybenetics Platinum
- Modularity: Fully modular
- Standards: ATX 3.1 & PCIe 5.1 compliant
- GPU Power: Native 12VHPWR cables (2x8pin-16pin)

### Case & Cooling
**Corsair iCUE 5000D RGB Airflow**
- Form Factor: Mid-tower ATX
- Airflow: High-airflow design optimized for thermal performance
- Fan Configuration: 10 total case fans + 3x AF120 RGB Elite included
- Controller: iCUE Commander Core XT (x2)

> **Linux Compatibility Note**: Corsair iCUE software is not compatible with Linux. RGB and fan control may require alternative solutions or manual configuration.

### Networking
**Panda Wireless PAU0F AXE3000**
- Standard: WiFi 6E tri-band (2.4/5/6 GHz)
- Interface: USB 3.0
- Use Case: Main connectivity for stronger signal

## AI Workload Optimization

### GPU Configuration
- **VRAM Utilization**: 32GB allows for large models (70B+ parameters)
- **Compute Capability**: Optimized for transformer architectures
- **Concurrent Models**: Single large model or multiple smaller models

### CPU Configuration
- **Memory Bandwidth**: 128GB enables multiple model hosting
- **Thread Distribution**: 32 threads for parallel inference
- **Model Sizes**: Ideal for 7B-13B parameter models on CPU

### Thermal Management
- **GPU**: Triple-fan open-air design with case airflow optimization
- **CPU**: 270W air cooling capacity with two fans
- **Case**: 10-fan configuration ensures optimal component temperatures
- **Total Fans**: A total of 15 fans provide optimal thermal management

## Installation Notes

### Memory Installation
1. Install memory modules in slots A1 and B1 only
2. Enable EXPO profile in BIOS for DDR5-6000 speeds
3. Verify stability under AI workloads before production use

### Storage Configuration
1. Install Samsung 990 Pro 2TB in CPU-connected M.2 slot (usually M.2_1)
2. Install Samsung 990 EVO 1TB in chipset M.2 slot (avoiding PCIe lane sharing)
3. Leave remaining M.2 slots empty to preserve GPU bandwidth

### PCIe Configuration
1. Install GPU in primary PCIe 5.0 x16 slot
2. Verify full x16 configuration in system monitoring
3. Avoid using expansion cards that would reduce GPU lanes
---

*This hardware configuration is optimized for AI engineering workflows as of mid-2025. Component selection prioritizes AI inference performance, memory capacity, and thermal management for sustained workloads.*