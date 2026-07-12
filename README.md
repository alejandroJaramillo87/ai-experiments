# AI Experiments Base Infrastructure v1.0

AI infrastructure for local LLM inference, optimized for RTX 5090 + AMD Ryzen 9950X. Designed as a git submodule base for AI engineering projects.

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Key Achievements](#key-achievements)
- [Infrastructure Services](#infrastructure-services)
- [Performance Benchmarks](#performance-benchmarks)
- [Repository Architecture](#repository-architecture)
- [Project Structure](#project-structure)
- [Documentation Map](#documentation-map)
- [Hardware Configuration](#hardware-configuration)
- [Tooling and Scripts](#tooling-and-scripts)
- [Python Environment](#python-environment)
- [GPU Software Stack](#gpu-software-stack)
- [System Optimizations](#system-optimizations)
- [Usage as Base Repository](#usage-as-base-repository)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

```bash
# 1. Clone repository
$ git clone https://github.com/user/ai-experiments
$ cd ai-experiments

# 2. Install GPU stack (if needed)
$ scripts/setup/setup_nvidia.sh
$ scripts/setup/setup_cuda.sh
$ scripts/setup/setup_cudnn.sh

# 3. Configure models
$ cp .env.example .env
# Edit .env with model paths

# 4. Start services
$ make up          # All services
$ make gpu-up      # GPU only
$ make cpu-up      # CPU only

# 5. Verify
$ make status      # Check health
$ make logs-gpu    # Monitor GPU

# 6. Access services
# GPU API: http://localhost:8002/v1
# CPU API: http://localhost:8001/v1
# Web UI: http://localhost:3000
```

## Overview

This base infrastructure repository provides AI inference capabilities with validated performance:

- **286.85 tokens/second** GPU inference (8x faster than CPU)
- **35.44 tokens/second** CPU inference with optimizations
- **Complete GPU software stack** documentation (NVIDIA Driver, CUDA 13.0, cuDNN 9.13)
- **Parameterized Docker services** for easy tuning without rebuilds
- **26% CPU performance improvement** through BIOS and OS optimizations
- **Git submodule architecture** for AI project repositories
- **Comprehensive benchmarking suite** with JSON output for tracking
- **28 documentation files** covering every aspect of the infrastructure
- **24 automation scripts** for setup, updates, and utilities

## Key Achievements

### Performance Milestones
- Achieved **286.85 tok/s** on RTX 5090 through systematic optimization
- Validated **35.44 tok/s** on CPU with 26% improvement from optimizations
- Discovered optimal batch configurations: GPU (2048/512), CPU (2048/2048)
- Implemented Flash Attention v3 support via cuDNN 9.13

### Technical Accomplishments
- Complete documentation of NVIDIA GPU software stack
- Environment variable parameterization for all services
- uv-based dependency management with CUDA compatibility
- Automated setup and update scripts for all components
- Git submodule architecture
- Comprehensive benchmarking with multiple workload types

## Infrastructure Services

### llama-gpu (Port 8002)
- **Purpose**: Primary GPU inference service
- **Performance**: 286.85 tokens/second
- **Configuration**: Batch 2048, Ubatch 512, Flash Attention enabled
- **GPU Utilization**: 95% with 15.3GB VRAM usage
- **Features**: CUDA-optimized, Blackwell architecture support

### llama-cpu (Port 8001)
- **Purpose**: Latency-optimized CPU inference
- **Performance**: 35.44 tokens/second
- **Configuration**: 16 dedicated cores (0-15), 96GB RAM
- **Optimizations**: Huge pages enabled
- **Features**: Single-model focus for minimum latency

### vllm-gpu (Port 8003)
- **Purpose**: High-throughput batch inference
- **Status**: Awaiting CUDA 13.0 support
- **Features**: Optimized for concurrent requests
- **Note**: Will enable multi-user serving when operational

### open-webui (Port 3000)
- **Purpose**: Web interface for model interaction
- **Access**: http://localhost:3000
- **Features**: Pre-configured for all inference services

## Performance Benchmarks

TBD

## Repository Architecture

This repository follows the **Unix philosophy**: do one thing well and compose with other tools.

### Design Pattern
- **This repo**: Hardware-optimized infrastructure (Docker configs, GPU setup, Python environment)
- **Project repos**: Include this as a git submodule to inherit infrastructure
- **Benefits**: Centralized management, consistent environments, easy updates

### Git Submodule Integration
```bash
# In AI project repository
$ git submodule add https://github.com/user/ai-experiments base-infrastructure
$ git submodule update --init --recursive

# Project now has access to:
# - Inference services on ports 8001-8005
# - RTX 5090 optimized configurations
# - uv environment with AI packages
# - Makefile commands for infrastructure
```

## Project Structure

```
ai-experiments/
├── .claude/                   # Claude Code configuration
├── .styles/                   # Style guides
├── CLAUDE.md                  # Project instructions
├── docker/                    # Container definitions
│   ├── llama-cpu/            # CPU service
│   │   ├── Dockerfile.cpu
│   │   └── entrypoint.cpu.sh
│   ├── llama-gpu/            # GPU service
│   │   ├── Dockerfile.llama-gpu
│   │   └── entrypoint.sh
│   └── vllm-gpu/             # vLLM service
│       ├── Dockerfile.vllm-gpu
│       └── entrypoint.sh
├── docs/                      # Comprehensive documentation (28 files)
│   ├── bios/                 # BIOS configuration
│   ├── hardware/             # Hardware selection guide
│   ├── inference/            # Parameter optimization
│   │   ├── engines/         # Inference engine docs
│   │   └── parameters/      # Parameter tuning guides
│   ├── optimizations/        # System optimizations
│   │   ├── bios/            # BIOS settings & benchmarks
│   │   └── os/              # OS-level optimizations
│   ├── os/                   # Operating system setup
│   │   ├── gpu-stack/       # NVIDIA GPU software stack
│   │   └── python/          # Python environment
│   └── sandbox/              # Container documentation
├── hugepages/                 # zen5_optimizer hugepage wrapper
│   ├── src/                  # Source code
│   │   ├── zen5_optimizer.cpp
│   │   └── memory/hugepage_wrapper.cpp
│   └── tests/                # Test suite
├── scripts/                   # Automation tooling (24 scripts)
│   ├── setup/                # Installation scripts (8)
│   ├── update/               # Update scripts (7)
│   ├── utils/                # Utilities (5)
│   └── benchmark/            # Performance testing (4)
│       ├── server-bench.py
│       ├── benchmark_cpu.sh
│       ├── benchmark_gpu.sh
│       └── datasets/
├── datasets/                  # Training/test datasets
├── logs/                      # Service logs
├── docker-compose.yaml        # Service orchestration
├── pyproject.toml            # uv project and exact dependency pins
├── uv.lock                   # Locked dependencies
├── Makefile                  # Infrastructure commands
├── .env                      # Environment configuration
├── .gitignore                # Git exclusions
└── README.md                 # This file
```

## Documentation map

### Getting started
- **Overview**: [docs/README.md](docs/README.md)
- **Hardware Guide**: [docs/hardware/README.md](docs/hardware/README.md)
- **BIOS Setup**: [docs/bios/README.md](docs/bios/README.md)
- **OS Installation**: [docs/os/README.md](docs/os/README.md)

### GPU software stack
- **Overview**: [docs/os/gpu-stack/README.md](docs/os/gpu-stack/README.md)
- **NVIDIA Drivers**: [docs/os/gpu-stack/nvidia-drivers.md](docs/os/gpu-stack/nvidia-drivers.md)
- **CUDA Platform**: [docs/os/gpu-stack/cuda.md](docs/os/gpu-stack/cuda.md)
- **cuDNN Library**: [docs/os/gpu-stack/cudnn.md](docs/os/gpu-stack/cudnn.md)

### Python environment
- **Poetry Dependencies**: [docs/os/python/poetry-dependencies.md](docs/os/python/poetry-dependencies.md)
- **Python Setup**: [docs/os/python/setup_python.md](docs/os/python/setup_python.md)
- **PyTorch CUDA**: [docs/os/python/pytorch_cuda_update_guide.md](docs/os/python/pytorch_cuda_update_guide.md)

### Inference optimization
- **Overview**: [docs/inference/README.md](docs/inference/README.md)
- **llama.cpp Parameters**: [docs/inference/parameters/llama-cpp-parameters.md](docs/inference/parameters/llama-cpp-parameters.md)
- **vLLM Parameters**: [docs/inference/parameters/vllm-parameters.md](docs/inference/parameters/vllm-parameters.md)

### System optimizations
- **Overview**: [docs/optimizations/README.md](docs/optimizations/README.md)
- **BIOS Optimizations**: [docs/optimizations/bios/bios-optimizations.md](docs/optimizations/bios/bios-optimizations.md)
- **OS Optimizations**: [docs/optimizations/os/os-optimizations.md](docs/optimizations/os/os-optimizations.md)

### Container documentation
- **Docker Compose**: [docs/sandbox/docker_compose_overview.md](docs/sandbox/docker_compose_overview.md)
- **CPU Service**: [docs/sandbox/docker_llama_cpu_overview.md](docs/sandbox/docker_llama_cpu_overview.md)
- **GPU Service**: [docs/sandbox/docker_llama_gpu_overview.md](docs/sandbox/docker_llama_gpu_overview.md)
- **vLLM Service**: [docs/sandbox/docker_vllm_gpu_overview.md](docs/sandbox/docker_vllm_gpu_overview.md)

## Hardware configuration

### Current environment (v1.0)

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **CPU** | AMD Ryzen 9950X | 16-core/32-thread, 16 cores for inference |
| **GPU** | NVIDIA RTX 5090 | 32GB VRAM, SM 12.0 Blackwell architecture |
| **Memory** | 128GB DDR5-6000 | Large model loading, 90GB huge pages |
| **Storage** | Samsung 990 PRO 2TB (models)<br>Samsung 990 EVO 1TB (OS) | High-speed model storage |
| **Motherboard** | Gigabyte X870E Aorus Elite WiFi | PCIe 5.0, robust power delivery |

### GPU software stack

| Component | Version | Purpose |
|-----------|---------|---------|
| **Driver** | 580.65.06 | Hardware interface, resource management |
| **CUDA** | 13.0.88 | Parallel computing platform |
| **cuDNN** | 9.13.0.50-1 | Deep learning acceleration |

## Tooling and scripts

### Setup scripts
- `setup_nvidia.sh` - NVIDIA driver installation
- `setup_cuda.sh` - CUDA toolkit setup
- `setup_cudnn.sh` - cuDNN library installation
- `setup_docker.sh` - Docker and NVIDIA runtime
- `setup_data_ssd.sh` - Storage configuration
- `setup_security.sh` - Security hardening
- `setup_sudo_user.sh` - User configuration
- `setup_terminal.sh` - Terminal environment

### Update scripts
- `update_nvidia.sh` - Update NVIDIA drivers
- `update_cuda.sh` - Update CUDA toolkit
- `update_cudnn.sh` - Update cuDNN library
- `update_docker.sh` - Update Docker
- `update_python.sh` - Update Python/PyTorch
- `update_security.sh` - Security updates
- `update_ubuntu.sh` - System updates

### Utilities
- `check_py_deps_install.py` - Dependency verification

Model downloads use the Hugging Face CLI directly (ships with `huggingface-hub`):

```bash
# Full repository
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download <repo-id> --local-dir <dir>
# Single file (e.g. a GGUF quant)
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download <repo-id> <filename> --local-dir <dir>
```

- `dependency_check.sh` - System dependency check
- `find_pythons.sh` - Python installation finder
- `storage_check.sh` - Storage verification

### Benchmarking
```bash
# Run GPU benchmark
$ scripts/benchmark/benchmark_gpu.sh

# Run CPU benchmark
$ scripts/benchmark/benchmark_cpu.sh

# Run server benchmark
$ scripts/benchmark/server-bench.py --service llama-gpu
```

## Python Environment

### uv-based management
- **Python 3.12** via uv for a reproducible managed interpreter
- **uv** for reproducible dependency management with exact version pins
- **Key Packages**:
  - torch 2.9.1 (cu130) - GPU tensor computation
  - transformers 4.51.3 - Pre-trained models
  - accelerate 1.10.1 - Distributed training
  - deepspeed 0.17.6 - Memory optimization
  - bitsandbytes 0.46.1 - Quantization
  - peft 0.15.2 - Parameter-efficient fine-tuning

### Installation
```bash
# Sync the environment (all groups and extras, including the cu130 PyTorch trio)
$ make install

# Activate the environment
$ source .venv/bin/activate
```

uv resolves the PyTorch trio from the CUDA 13.0 index declared in `pyproject.toml`, so no
separate PyTorch install step is required. See
[docs/os/python/setup_python.md](docs/os/python/setup_python.md) for details.

## GPU Software Stack

### Three-layer architecture
```
Application Layer (llama.cpp, PyTorch)
        ↓
cuDNN 9.13.0 (Deep Learning Primitives)
        ↓
CUDA 13.0.88 (Parallel Computing)
        ↓
NVIDIA Driver 580.65.06 (Hardware Interface)
        ↓
RTX 5090 (Physical GPU)
```

### Key features
- **5th Gen Tensor Cores** with FP8 support
- **Flash Attention v3** via cuDNN 9.13
- **96MB L2 Cache** on Blackwell architecture
- **PCIe Gen5** with 128 GB/s bandwidth
- **32GB GDDR7 VRAM** at 1.8 TB/s

## System Optimizations

### BIOS level
- CPU C-states disabled for consistent latency
- Precision Boost Overdrive enabled
- DDR5-6000 EXPO profiles optimized
- PCIe Gen5, Resizable BAR enabled

### OS level
- Swap disabled to prevent paging
- CPU governor in performance mode
- 90GB huge pages configured
- CPU pinning for dedicated cores
- THP disabled for consistency

### Container level
- Memory locking enabled
- CPU affinity set (cores 0-15)
- GPU persistence mode enabled
- CUDA memory pools optimized
- Environment variable configuration

## Usage as base repository

### For AI project repositories

1. **Add as submodule**:
```bash
$ git submodule add https://github.com/user/ai-experiments base-infrastructure
$ cd base-infrastructure && make up
```

2. **Access services**:
```python
# In project code
GPU_API = "http://localhost:8002/v1"
CPU_API = "http://localhost:8001/v1"
```

3. **Update infrastructure**:
```bash
$ cd base-infrastructure
$ git pull origin main
$ cd .. && git add base-infrastructure
$ git commit -m "Update infrastructure to v1.0"
```

## Contributing

Contributions welcome! Areas of interest:

- Performance benchmarks on similar hardware
- Alternative optimization strategies
- Additional model compatibility testing
- Documentation improvements
- Cost-effective hardware alternatives

Please see our Contributing Guidelines for details.

## License

This infrastructure is open source and available under the MIT License.

---

**Version:** 1.0.0
**Last Updated:** January 2026
**Status:** Stable
**Performance:** 286.85 tok/s (GPU) | 35.44 tok/s (CPU)
**Hardware:** RTX 5090 + AMD Ryzen 9950X

---

*Last Updated: 2026-01-03*