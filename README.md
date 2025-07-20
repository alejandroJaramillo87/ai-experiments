# AI Engineering Workstation Guide
A comprehensive guide for building a high-performance local AI engineering workstation optimized for running large language models and agentic workloads on modern hardware platforms.

## Table of Contents
- [AI Engineering Workstation Guide](#ai-engineering-workstation-guide)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Reference Hardware Configuration](#reference-hardware-configuration)
  - [Project Structure](#project-structure)
  - [Documentation Sections](#documentation-sections)
    - [Hardware](#hardware)
    - [BIOS Configuration](#bios-configuration)
    - [Operating System](#operating-system)
    - [Sandbox Environment](#sandbox-environment)
    - [AI Inference Configuration](#ai-inference-configuration)
    - [System Optimizations](#system-optimizations)
  - [Target Audience](#target-audience)
  - [Key Features](#key-features)
  - [Performance Benchmarks](#performance-benchmarks)
  - [Contributing](#contributing)
  - [Community \& Support](#community--support)
  - [License](#license)

## Overview
This guide documents the complete process of building a professional AI workstation capable of:

- **Simultaneous GPU + CPU inference** - Run one model on GPU while serving multiple models from CPU/RAM
- **Large model support** - Handle 30-34B quantized parameter models locally with KV cache + context on GPU
- **Multiple agent support** - Run multiple agentic workflows on CPU/RAM simultaneously
- **High-throughput inference** - Optimized for sustained AI workloads with proper cooling and power delivery
- **Development flexibility** - Full containerization support for reproducible AI environments
- **Security and sandboxing** - Hardened configuration at every level and Docker sandboxed environment for utmost security standards
- **Backup and recovery** - Comprehensive data protection strategy
- **RAG operations** - Run entire RAG operations in memory for faster inference
- **Optimized storage** - Store data/models on Gen 5 SSD with direct access to CPU while running OS on separate SSD
- **Full Ubuntu setup** - A full suite of docs and scripts outlining setting up Ubuntu for AI Engineering
- **Benchmarking** - Outlined strategy and tooling for benchmarking models on GPU and CPU/RAM

## Reference Hardware Configuration
This guide is optimized for and tested on the following hardware configuration:

| Component | Specification | Purpose |
|-----------|---------------|---------|
| CPU | AMD Ryzen 9950X | 16-core/32-thread for CPU inference + compilation |
| GPU | NVIDIA RTX 5090 | 32GB VRAM for large model inference |
| Memory | 128GB DDR5-6000 (2x64GB EXPO) | Large model loading + multi-model serving |
| Storage | Samsung 990 PRO 2TB (models)<br>Samsung 990 EVO 1TB (Ubuntu 24.04) | High-speed model storage + OS separation |
| Motherboard | Gigabyte X870E Aorus Elite WiFi | PCIe 5.0 support + robust power delivery |

**Note:** This configuration represents a mid-2025 high-performance setup. The guide supports scaling up/down based on requirements and budget.

## Project Structure
```
ai-workstation/
├── hardware/                  # Hardware selection and assembly
├── bios/                      # BIOS configuration and optimization
├── operating-system/          # OS installation and optimization
├── sandbox/                   # Docker sandbox environment 
├── inference/                 # Inference Configuration for GPU and CPU/RAM
├── optimization/              # Optimizations for increasing inference speeds
└── README.md                  # This overview document
```

## Documentation Sections

### Hardware
**Reference:** hardware/README.md

Complete hardware selection, compatibility, and assembly guidance documented in the hardware README.

### BIOS Configuration
**Reference:** bios/README.md

Minimal BIOS configuration and troubleshooting are documented in the BIOS README.

### Operating System
**Reference:** OS/README.md, OS/scripts/ and OS/docs/

Ubuntu 24.04 LTS installation and AI engineering tooling documented in the operating system README, scripts, and detailed documentation.

### Sandbox Environment
**Reference:** sandbox/README.md and sandbox/scripts/

Guide for setting up a Docker sandboxed environment that provides optimal security and development flexibility.

### AI Inference Configuration
**Reference:** inference/README.md and inference/scripts/

Dual GPU + CPU inference setup for maximum hardware utilization documented in the inference README, scripts, and detailed documentation.

### System Optimizations
**Reference:** optimizations/README.md and optimizations/scripts/

Optimizations at each level of the workstation aimed at improving overall model inference performance.

## Target Audience
This guide is designed for:

- AI Engineers building local development environments
- ML Researchers needing high-performance local inference
- Software Engineers interested in self-hosted AI capabilities
- Tech Enthusiasts building professional-grade AI workstations

## Key Features
This guide focuses on:

- **Dual inference architecture** - Maximize hardware utilization across GPU and CPU
- **Real-world performance data** - Actual benchmarks, not theoretical specifications
- **Code Examples** - Examples of code used are provided
- **Professional stability** - Configurations tested under sustained AI workloads
- **Cost optimization** - Maximum performance per dollar invested
- **Modern hardware support** - Optimized for latest AMD Zen 5 + NVIDIA RTX 50 series

## Performance Benchmarks
Based on the reference hardware configuration:

- **GPU Inference:** TBD
- **CPU Inference:** TBD
- **Memory Utilization:** TBD
- **Thermal Performance:** TBD

Detailed benchmarks available in docs/inference/benchmarking.md

## Contributing
This is a living document based on real-world AI workstation experience. Contributions welcome:

- Performance benchmarks on similar hardware configurations
- Alternative optimization strategies
- Additional model compatibility testing
- Documentation improvements and corrections
- Cost-effective hardware alternatives

Please see our Contributing Guidelines for details.

## Community & Support
- **Issues:** Report bugs or request features via GitHub Issues
- **Discussions:** Share configurations and ask questions in GitHub Discussions
- **Updates:** Follow repository for hardware and software updates

## License
This guide is open source and available under the MIT License. Hardware costs and performance results are documented for transparency and reproducibility.

Built by the AI engineering community, for the AI engineering community.

---
**Last updated:** July 2025  
**Template version:** 2.0  
**Hardware revision:** AMD Zen 5 + RTX 50 series