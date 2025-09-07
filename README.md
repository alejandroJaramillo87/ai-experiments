# AI Engineering Expirements Base Infrastructure
A base infrastructure repository for AI engineering expirements, providing a consistent hardware-optimized environment via git submodules. This repo contains the foundation for building high-performance local AI engineering setups optimized for running large language models and agentic workloads on modern hardware platforms.

## Table of Contents
- [AI Engineering Expirements Base Infrastructure](#ai-engineering-expirements-base-infrastructure)
  - [Table of Contents](#table-of-contents)
  - [Repository Architecture](#repository-architecture)
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
  - [Usage as Base Repository](#usage-as-base-repository)
  - [Community \& Support](#community--support)
  - [License](#license)

## Repository Architecture

This repository follows the **Unix philosophy**: do one thing well and compose with other programs. It serves as a **base infrastructure repository** that other AI engineering projects consume via **git submodules**.

### Design Pattern
- **This repo**: Contains hardware-optimized infrastructure (Docker configs, Python environment, RTX 5090 setup)
- **Project repos**: Include this as a submodule to inherit infrastructure, focus on their specific AI tasks
- **Benefits**: Centralized infrastructure management, consistent environments, easy hardware updates

### Git Submodule Integration
```bash
# In your AI project repo
git submodule add https://github.com/user/ai-expirements base-infrastructure
git submodule update --init --recursive

# Your project now has access to:
# - Docker services on ports 8001-8005  
# - RTX 5090 optimized configurations
# - Poetry environment with AI packages
# - Makefile commands for infrastructure management
```

## Overview
This base infrastructure repository provides the foundation for AI expirements capable of:

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
- **Base infrastructure pattern** - Designed for consumption via git submodules by AI project repositories

## Reference Hardware Configuration
This base infrastructure is optimized for and tested on the following hardware configuration:

| Component | Specification | Purpose |
|-----------|---------------|---------|
| CPU | AMD Ryzen 9950X | 16-core/32-thread for CPU inference + compilation |
| GPU | NVIDIA RTX 5090 | 32GB VRAM for large model inference |
| Memory | 128GB DDR5-6000 (2x64GB EXPO) | Large model loading + multi-model serving |
| Storage | Samsung 990 PRO 2TB (models)<br>Samsung 990 EVO 1TB (Ubuntu 24.04) | High-speed model storage + OS separation |
| Motherboard | Gigabyte X870E Aorus Elite WiFi | PCIe 5.0 support + robust power delivery |

**Note:** This configuration represents a mid-2025 high-performance setup. The base infrastructure supports scaling up/down based on requirements and budget.

## Project Structure
```
ai-expirements/                # Base Infrastructure Repository
├── .claude/                   # Claude Code configuration
│   └── CLAUDE.md             # Project context and service documentation
├── docker/                    # Container definitions and Dockerfiles
├── docs/                      # Infrastructure documentation
├── scripts/                   # Automation and maintenance scripts
├── docker-compose.yaml        # Multi-service AI inference orchestration
├── pyproject.toml            # Poetry Python environment (CUDA optimized)
├── Makefile                  # Infrastructure management commands
└── README.md                 # This overview document

# Consumed by project repositories via git submodules:
project-repo/
├── base-infrastructure/      # This repository as submodule
├── src/                      # Project-specific code
├── tests/                    # Project-specific tests
└── README.md                # Project-specific documentation
```

## Documentation Sections

### Hardware
**Reference:** [hardware/README.md](./hardware/README.md)

Complete hardware selection, compatibility, and assembly guidance documented in the hardware README.

### BIOS Configuration
**Reference:** [bios/README.md](./bios/README.md)

Minimal BIOS configuration and troubleshooting are documented in the BIOS README.

### Operating System
**Reference:** [OS/README.md](./OS/README.md), [OS/scripts/](./OS/scripts/) and [OS/docs/](./OS/docs/)

Ubuntu 24.04 LTS installation and AI engineering tooling documented in the operating system README, scripts, and detailed documentation.

### Sandbox Environment
**Reference:** [sandbox/README.md](./sandbox/README.md) and [sandbox/scripts/](./sandbox/scripts/)

Guide for setting up a Docker sandboxed environment that provides optimal security and development flexibility.

### AI Inference Configuration
**Reference:** [inference/README.md](./inference/README.md) and [inference/scripts/](./inference/scripts/)

Dual GPU + CPU inference setup for maximum hardware utilization documented in the inference README, scripts, and detailed documentation.

### System Optimizations
**Reference:** [optimizations/README.md](./optimization/README.md) and [optimizations/scripts/](./optimizations/scripts/)

Optimizations at each level of the workstation aimed at improving overall model inference performance.

## Target Audience
This base infrastructure is designed for:

- AI Engineers building local development environments
- ML Researchers needing high-performance local inference
- Software Engineers interested in self-hosted AI capabilities
- Tech Enthusiasts building professional-grade AI expirements setups

## Key Features
This base infrastructure focuses on:

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

## Usage as Base Repository

### For AI Project Repositories
When consuming this base infrastructure via git submodule:

1. **Add as submodule**:
   ```bash
   git submodule add https://github.com/user/ai-expirements base-infrastructure
   cd base-infrastructure && make up  # Start all AI services
   ```

2. **Access infrastructure services**:
   - **GPU inference**: `http://localhost:8004/v1` (llama-gpu)
   - **High-performance GPU**: `http://localhost:8005/v1` (vllm-gpu)  
   - **Load-balanced CPU**: `http://localhost:8001-8003/v1` (llama-cpu services)
   - **Web interface**: `http://localhost:3000` (open-webui)

3. **Use infrastructure commands**:
   ```bash
   cd base-infrastructure
   make status      # Check service health
   make logs-gpu    # Monitor GPU services
   make demo        # Quick start GPU + UI
   ```

### Infrastructure Updates
When hardware or service configurations change:
```bash
# In your project repo
cd base-infrastructure
git pull origin main
cd .. && git add base-infrastructure && git commit -m "Update infrastructure"
```

## Contributing
This is a living base infrastructure repository based on real-world AI expirements experience. Contributions welcome:

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