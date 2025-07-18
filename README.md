# AI Engineering Workstation Build Guide

A comprehensive guide for building a high-performance local AI engineering workstation optimized for running large language models and machine learning workloads.

## 🚀 Build Overview

This guide documents the complete process of building a professional AI workstation capable of:
- **Simultaneous GPU + CPU inference** - Run one model on GPU while serving multiple models from CPU/RAM
- **Large model support** - Handle 70B+ parameter models locally with 128GB DDR5 memory
- **Multiple agents support** - Run multiple agentic worflows on CPU/RAM
- **High-throughput inference** - Optimized for sustained AI workloads with proper cooling and power delivery
- **Development flexibility** - Full containerization support for reproducible AI environments
- **Security** - Hardened at every level for the upmost security
- **Backup and Recovery** - Backup and recovery strategy outlined

## 🔧 Target Hardware Configuration

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **CPU** | AMD Ryzen 9950X | High core count for CPU inference + compilation |
| **GPU** | NVIDIA RTX 5090 | Maximum VRAM for large model inference |
| **Memory** | 128GB DDR5-6000 | Large model loading + multi-model serving |
| **Storage** | Samsung 990 PRO 2TB + 990 EVO 1TB | High-speed model storage + OS |
| **Motherboard** | Gigabyte X870E Aorus Elite WiFi | PCIe 5.0 support + robust power delivery |


## 📚 Documentation Structure

### 🔩 [Hardware](docs/hardware/)
- **[Component Selection](docs/hardware/component-selection.md)** - Why these specific parts for AI workloads
- **[Compatibility Matrix](docs/hardware/compatibility-matrix.md)** - PCIe lanes, memory channels, thermal considerations
- **[Assembly Guide](docs/hardware/assembly-guide.md)** - Step-by-step build process with photos
- **[Troubleshooting](docs/hardware/troubleshooting.md)** - Common hardware issues and solutions

### ⚙️ [BIOS Configuration](docs/bios/)
- **[Initial Setup](docs/bios/initial-setup.md)** - Essential BIOS settings for AI workstations
- **[Memory Overclocking](docs/bios/memory-overclocking.md)** - DDR5-6000 tuning and stability testing
- **[CPU Optimization](docs/bios/cpu-optimization.md)** - PBO, curve optimizer, thermal management
- **[Settings Reference](docs/bios/settings-reference.md)** - Complete BIOS configuration table

### 🖥️ [Operating System](docs/operating-system/)
- **[Ubuntu Installation](docs/operating-system/ubuntu-installation.md)** - AI-optimized Ubuntu 24.04 setup
- **[System Optimization](docs/operating-system/system-optimization.md)** - Huge pages, CPU governor, NUMA tuning
- **[Driver Installation](docs/operating-system/driver-installation.md)** - NVIDIA drivers, CUDA, cuDNN setup
- **[Environment Setup](docs/operating-system/environment-setup.md)** - Python, Docker, development tools

### 🤖 [AI Inference Setup](docs/inference/)
- **[GPU Inference](docs/inference/gpu-setup.md)** - TensorRT-LLM, vLLM configuration
- **[CPU Inference](docs/inference/cpu-setup.md)** - llama.cpp optimization for high-memory systems
- **[Dual Inference](docs/inference/dual-setup.md)** - Running GPU + CPU models simultaneously
- **[Benchmarking](docs/inference/benchmarking.md)** - Performance testing and optimization

## 🛠️ Automation & Scripts

Configuration scripts and docs are in ```/utils```:

```

## 🚦 Quick Start

1. **Hardware Assembly**: Follow the [assembly guide](docs/hardware/assembly-guide.md)
2. **BIOS Configuration**: Apply settings from [BIOS setup](docs/bios/initial-setup.md)
3. **OS Installation**: Install Ubuntu using [installation guide](docs/operating-system/ubuntu-installation.md)
4. **AI Inference Setup**: Configure inference engines using [AI setup guides](docs/inference/)

## 🎯 Who This Guide Is For

- **AI Engineers** building local development environments
- **ML Researchers** needing high-performance local inference
- **Software Engineers** interested in self-hosted AI capabilities
- **Enthusiasts** building professional-grade AI workstations

## 💡 Key Differentiators

This guide focuses on:
- **Dual inference architecture** - Maximize hardware utilization
- **Real-world performance data** - Actual benchmarks, not theoretical specs
- **Complete automation** - Reproducible setup with minimal manual intervention
- **Professional stability** - Configurations tested under sustained AI workloads
- **Cost optimization** - Maximum performance per dollar invested

## 📋 Prerequisites

- **Budget**: $3,000-6,000 for complete build
- **Technical Level**: Intermediate Linux/hardware knowledge
- **Time Investment**: several days to potentially several weeks
- **Space Requirements**: Full ATX case with robust cooling

## 🤝 Contributing

This is a living document based on real-world AI workstation experience. Contributions welcome:
- Performance benchmarks on similar hardware
- Alternative configuration optimizations
- Additional model compatibility testing
- Documentation improvements

## 📄 License

This guide is open source and available under the MIT License. Hardware costs and performance results are documented for transparency and reproducibility.

---

**Built by the AI engineering community, for the AI engineering community.**

*Last updated: July 2025*