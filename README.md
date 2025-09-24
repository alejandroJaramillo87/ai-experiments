# AI Experiments Base Infrastructure v1.0

Production-ready AI infrastructure for local LLM inference, optimized for RTX 5090 + AMD Ryzen 9950X. Designed as a git submodule base for AI engineering projects.

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
git clone https://github.com/user/ai-experiments
cd ai-experiments

# 2. Install GPU stack (if needed)
scripts/setup/setup_nvidia.sh
scripts/setup/setup_cuda.sh
scripts/setup/setup_cudnn.sh

# 3. Configure models
cp .env.example .env
# Edit .env with your model paths

# 4. Start services
make up          # All services
make gpu-up      # GPU only
make cpu-up      # CPU only

# 5. Verify
make status      # Check health
make logs-gpu    # Monitor GPU

# 6. Access services
# GPU API: http://localhost:8004/v1
# CPU API: http://localhost:8001/v1
# Web UI: http://localhost:3000
```

## Overview

This base infrastructure repository provides production-ready AI inference capabilities with validated performance:

- **286.85 tokens/second** GPU inference (8x faster than CPU)
- **35.44 tokens/second** CPU inference with optimizations
- **Complete GPU software stack** documentation (NVIDIA Driver, CUDA 13.0, cuDNN 9.13)
- **Parameterized Docker services** for easy tuning without rebuilds
- **26% CPU performance improvement** through BIOS and OS optimizations
- **Production-ready git submodule architecture** for AI project repositories
- **Comprehensive benchmarking suite** with JSON output for tracking
- **31 documentation files** covering every aspect of the infrastructure
- **21 automation scripts** for setup, updates, and utilities

## Key Achievements

### Performance Milestones
- Achieved **286.85 tok/s** on RTX 5090 through systematic optimization
- Validated **35.44 tok/s** on CPU with 26% improvement from optimizations
- Discovered optimal batch configurations: GPU (2048/512), CPU (2048/2048)
- Implemented Flash Attention v3 support via cuDNN 9.13

### Technical Accomplishments
- Complete documentation of NVIDIA GPU software stack
- Environment variable parameterization for all services
- Poetry-based dependency management with CUDA compatibility
- Automated setup and update scripts for all components
- Production-ready git submodule architecture
- Comprehensive benchmarking with multiple workload types

## Infrastructure Services

### llama-gpu (Port 8004)
- **Purpose**: Primary GPU inference service
- **Performance**: 286.85 tokens/second
- **Configuration**: Batch 2048, Ubatch 512, Flash Attention enabled
- **GPU Utilization**: 95% with 15.3GB VRAM usage
- **Features**: CUDA-optimized, Blackwell architecture support

### llama-cpu (Port 8001)
- **Purpose**: Latency-optimized CPU inference
- **Performance**: 35.44 tokens/second
- **Configuration**: 12 dedicated cores (0-11), 96GB RAM
- **Optimizations**: SMT disabled, huge pages enabled
- **Features**: Single-model focus for minimum latency

### vllm-gpu (Port 8005)
- **Purpose**: High-throughput batch inference
- **Status**: Awaiting CUDA 13.0 support
- **Features**: Optimized for concurrent requests
- **Note**: Will enable multi-user serving when operational

### open-webui (Port 3000)
- **Purpose**: Web interface for model interaction
- **Access**: http://localhost:3000
- **Features**: Pre-configured for all inference services

## Performance Benchmarks

Based on production testing (September 2025):

### GPU Inference (RTX 5090)
- **Model**: gpt-oss-20b (Q8_K_XL quantization)
- **Performance**: 286.85 tokens/second
- **Configuration**: Batch 2048, Ubatch 512
- **GPU Utilization**: 95%
- **VRAM Usage**: 15.3GB of 32GB
- **Power Draw**: 438W of 600W limit
- **Benchmark**: `docs/optimizations/os/gpu_baseline.json`

### CPU Inference (AMD Ryzen 9950X)
- **Model**: Qwen3-Coder-30B (IQ4_XS quantization)
- **Performance**: 35.44 tokens/second
- **Configuration**: Batch 2048, 12 threads
- **Optimization Gain**: +26% with BIOS/OS optimizations
- **Memory**: Huge pages (90GB allocated)
- **Benchmark**: `docs/optimizations/bios/cpu_benchmark_results_final_bios.json`

## Repository Architecture

This repository follows the **Unix philosophy**: do one thing well and compose with other tools.

### Design Pattern
- **This repo**: Hardware-optimized infrastructure (Docker configs, GPU setup, Python environment)
- **Project repos**: Include this as a git submodule to inherit infrastructure
- **Benefits**: Centralized management, consistent environments, easy updates

### Git Submodule Integration
```bash
# In your AI project repository
git submodule add https://github.com/user/ai-experiments base-infrastructure
git submodule update --init --recursive

# Your project now has access to:
# - Inference services on ports 8001-8005
# - RTX 5090 optimized configurations
# - Poetry environment with AI packages
# - Makefile commands for infrastructure
```

## Project Structure

```
ai-experiments/
├── .claude/                   # Claude Code configuration
│   └── CLAUDE.md             # Project context and guidelines
├── docker/                    # Container definitions
│   ├── llama-cpu/            # CPU service
│   │   ├── Dockerfile.llama-cpu
│   │   ├── entrypoint.sh     # Parameterized startup
│   │   └── hugepage_mmap_wrapper.cpp
│   ├── llama-gpu/            # GPU service
│   │   └── entrypoint.sh     # Parameterized startup
│   ├── Dockerfile.llama-gpu  # GPU container definition
│   └── Dockerfile.vllm-gpu   # vLLM container definition
├── docs/                      # Comprehensive documentation (31 files)
│   ├── bios/                 # BIOS optimization guides
│   ├── hardware/             # Hardware selection guide
│   ├── inference/            # Parameter optimization
│   │   ├── README.md
│   │   ├── llama-cpp-parameters.md
│   │   ├── vllm-parameters.md
│   │   └── parameter-optimization-guide.md
│   ├── optimizations/        # System optimizations
│   │   ├── bios/            # BIOS settings & benchmarks
│   │   ├── os/              # OS-level optimizations
│   │   ├── gpu/             # GPU optimizations
│   │   ├── experiments/     # Experimental features
│   │   └── benchmark-guide.md
│   ├── os/                   # Operating system setup
│   │   ├── gpu-stack/       # NVIDIA GPU software stack
│   │   │   ├── README.md
│   │   │   ├── nvidia-drivers.md
│   │   │   ├── cuda.md
│   │   │   └── cudnn.md
│   │   ├── python/          # Python environment
│   │   │   ├── poetry-dependencies.md
│   │   │   ├── setup_python.md
│   │   │   └── pytorch_cuda_update_guide.md
│   │   └── backup_and_recovery.md
│   └── sandbox/              # Container documentation
│       ├── docker_compose_overview.md
│       ├── docker_llama_cpu_overview.md
│       ├── docker_llama_gpu_overview.md
│       └── docker_vllm_gpu_overview.md
├── scripts/                   # Automation tooling (21 scripts)
│   ├── setup/                # Installation scripts (8)
│   │   ├── setup_nvidia.sh
│   │   ├── setup_cuda.sh
│   │   ├── setup_cudnn.sh
│   │   ├── setup_docker.sh
│   │   ├── setup_data_ssd.sh
│   │   ├── setup_security.sh
│   │   ├── setup_sudo_user.sh
│   │   └── setup_terminal.sh
│   ├── update/               # Update scripts (8)
│   │   ├── update_nvidia.sh
│   │   ├── update_cuda.sh
│   │   ├── update_cudnn.sh
│   │   ├── update_docker.sh
│   │   ├── update_python.sh
│   │   ├── update_security.sh
│   │   └── update_ubuntu.sh
│   ├── utils/                # Utilities (5)
│   │   ├── check_py_deps_install.py
│   │   ├── dependency_check.sh
│   │   ├── download_model_hf.py
│   │   ├── find_pythons.sh
│   │   └── storage_check.sh
│   ├── optimizations/        # Performance tuning
│   │   └── optimize-gpu.sh
│   └── benchmark.py          # Performance testing tool
├── logs/                      # Service logs
│   ├── cpu/                  # CPU service logs
│   └── gpu/                  # GPU service logs
├── docker-compose.yaml        # Service orchestration
├── pyproject.toml            # Poetry dependencies
├── poetry.lock               # Locked dependencies
├── Makefile                  # Infrastructure commands
├── .env.example              # Environment template
├── .gitignore                # Git exclusions
└── README.md                 # This file
```

## Documentation Map

### Getting Started
- **Overview**: [docs/README.md](docs/README.md)
- **Hardware Guide**: [docs/hardware/README.md](docs/hardware/README.md)
- **BIOS Setup**: [docs/bios/README.md](docs/bios/README.md)
- **OS Installation**: [docs/os/README.md](docs/os/README.md)

### GPU Software Stack
- **Overview**: [docs/os/gpu-stack/README.md](docs/os/gpu-stack/README.md)
- **NVIDIA Drivers**: [docs/os/gpu-stack/nvidia-drivers.md](docs/os/gpu-stack/nvidia-drivers.md)
- **CUDA Platform**: [docs/os/gpu-stack/cuda.md](docs/os/gpu-stack/cuda.md)
- **cuDNN Library**: [docs/os/gpu-stack/cudnn.md](docs/os/gpu-stack/cudnn.md)

### Python Environment
- **Poetry Dependencies**: [docs/os/python/poetry-dependencies.md](docs/os/python/poetry-dependencies.md)
- **Python Setup**: [docs/os/python/setup_python.md](docs/os/python/setup_python.md)
- **PyTorch CUDA**: [docs/os/python/pytorch_cuda_update_guide.md](docs/os/python/pytorch_cuda_update_guide.md)

### Inference Optimization
- **Overview**: [docs/inference/README.md](docs/inference/README.md)
- **llama.cpp Parameters**: [docs/inference/llama-cpp-parameters.md](docs/inference/llama-cpp-parameters.md)
- **vLLM Parameters**: [docs/inference/vllm-parameters.md](docs/inference/vllm-parameters.md)
- **Optimization Guide**: [docs/inference/parameter-optimization-guide.md](docs/inference/parameter-optimization-guide.md)

### System Optimizations
- **Overview**: [docs/optimizations/README.md](docs/optimizations/README.md)
- **BIOS Optimizations**: [docs/optimizations/bios/bios-optimizations.md](docs/optimizations/bios/bios-optimizations.md)
- **OS Optimizations**: [docs/optimizations/os/os-optimizations.md](docs/optimizations/os/os-optimizations.md)
- **GPU Optimizations**: [docs/optimizations/gpu/gpu-optimizations.md](docs/optimizations/gpu/gpu-optimizations.md)
- **Benchmarking**: [docs/optimizations/benchmark-guide.md](docs/optimizations/benchmark-guide.md)

### Container Documentation
- **Docker Compose**: [docs/sandbox/docker_compose_overview.md](docs/sandbox/docker_compose_overview.md)
- **CPU Service**: [docs/sandbox/docker_llama_cpu_overview.md](docs/sandbox/docker_llama_cpu_overview.md)
- **GPU Service**: [docs/sandbox/docker_llama_gpu_overview.md](docs/sandbox/docker_llama_gpu_overview.md)
- **vLLM Service**: [docs/sandbox/docker_vllm_gpu_overview.md](docs/sandbox/docker_vllm_gpu_overview.md)

## Hardware Configuration

### Current Environment (v1.0)

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **CPU** | AMD Ryzen 9950X | 16-core/32-thread, cores 0-11 for inference |
| **GPU** | NVIDIA RTX 5090 | 32GB VRAM, SM 12.0 Blackwell architecture |
| **Memory** | 128GB DDR5-6000 | Large model loading, 90GB huge pages |
| **Storage** | Samsung 990 PRO 2TB (models)<br>Samsung 990 EVO 1TB (OS) | High-speed model storage |
| **Motherboard** | Gigabyte X870E Aorus Elite WiFi | PCIe 5.0, robust power delivery |

### GPU Software Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| **Driver** | 580.65.06 | Hardware interface, resource management |
| **CUDA** | 13.0.88 | Parallel computing platform |
| **cuDNN** | 9.13.0.50-1 | Deep learning acceleration |

## Tooling and Scripts

### Setup Scripts
- `setup_nvidia.sh` - NVIDIA driver installation
- `setup_cuda.sh` - CUDA toolkit setup
- `setup_cudnn.sh` - cuDNN library installation
- `setup_docker.sh` - Docker and NVIDIA runtime
- `setup_data_ssd.sh` - Storage configuration
- `setup_security.sh` - Security hardening
- `setup_sudo_user.sh` - User configuration
- `setup_terminal.sh` - Terminal environment

### Update Scripts
- `update_nvidia.sh` - Update NVIDIA drivers
- `update_cuda.sh` - Update CUDA toolkit
- `update_cudnn.sh` - Update cuDNN library
- `update_docker.sh` - Update Docker
- `update_python.sh` - Update Python/PyTorch
- `update_security.sh` - Security updates
- `update_ubuntu.sh` - System updates

### Utilities
- `benchmark.py` - Comprehensive performance testing
- `download_model_hf.py` - HuggingFace model downloader
- `check_py_deps_install.py` - Dependency verification
- `dependency_check.sh` - System dependency check
- `storage_check.sh` - Storage verification

### Benchmarking
```bash
# Run GPU benchmark
scripts/benchmark.py --service llama-gpu --label "gpu_test"

# Run CPU benchmark
scripts/benchmark.py --service llama-cpu --label "cpu_test"

# Compare results
scripts/benchmark.py --compare gpu_baseline.json gpu_test.json
```

## Python Environment

### Poetry-Based Management
- **Python 3.12** via pyenv for latest performance
- **Poetry** for reproducible dependency management
- **Key Packages**:
  - transformers 4.51.1 - Pre-trained models
  - accelerate 1.8.1 - Distributed training
  - deepspeed 0.17.1 - Memory optimization
  - bitsandbytes 0.46.0 - Quantization
  - peft 0.15.2 - Parameter-efficient fine-tuning

### Installation
```bash
# Install dependencies
poetry install

# Activate environment
poetry shell

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

## GPU Software Stack

### Three-Layer Architecture
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

### Key Features
- **5th Gen Tensor Cores** with FP8 support
- **Flash Attention v3** via cuDNN 9.13
- **96MB L2 Cache** on Blackwell architecture
- **PCIe Gen5** with 128 GB/s bandwidth
- **32GB GDDR7 VRAM** at 1.8 TB/s

## System Optimizations

### BIOS Level
- CPU C-states disabled for consistent latency
- Precision Boost Overdrive enabled
- DDR5-6000 EXPO profiles optimized
- PCIe Gen5, Resizable BAR enabled

### OS Level
- Swap disabled to prevent paging
- CPU governor in performance mode
- 90GB huge pages configured
- CPU pinning for dedicated cores
- THP disabled for consistency

### Container Level
- Memory locking enabled
- CPU affinity set (cores 0-11)
- GPU persistence mode enabled
- CUDA memory pools optimized
- Environment variable configuration

## Usage as Base Repository

### For AI Project Repositories

1. **Add as submodule**:
```bash
git submodule add https://github.com/user/ai-experiments base-infrastructure
cd base-infrastructure && make up
```

2. **Access services**:
```python
# In your project code
GPU_API = "http://localhost:8004/v1"
CPU_API = "http://localhost:8001/v1"
```

3. **Update infrastructure**:
```bash
cd base-infrastructure
git pull origin main
cd .. && git add base-infrastructure
git commit -m "Update infrastructure to v1.0"
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
**Last Updated:** September 2025
**Status:** Production Ready
**Performance:** 286.85 tok/s (GPU) | 35.44 tok/s (CPU)
**Hardware:** RTX 5090 + AMD Ryzen 9950X

Built by the AI engineering community, for the AI engineering community.