# AI inference workstation infrastructure

Local AI inference infrastructure optimized for an RTX 5090 and AMD Ryzen 9950X workstation.
It provides containerized LLM inference tuned for both low latency (llama.cpp) and high
throughput (vLLM), a reproducible uv-managed Python environment, and documented setup for the
full GPU software stack.

## Table of contents

- [Quick start](#quick-start)
- [Overview](#overview)
- [Infrastructure services](#infrastructure-services)
- [Project structure](#project-structure)
- [Documentation map](#documentation-map)
- [Hardware configuration](#hardware-configuration)
- [Tooling and scripts](#tooling-and-scripts)
- [Python environment](#python-environment)
- [GPU software stack](#gpu-software-stack)
- [System optimizations](#system-optimizations)
- [Contributing](#contributing)
- [License](#license)

## Quick start

```bash
# 1. Clone the repository
$ git clone https://github.com/user/ai-experiments
$ cd ai-experiments

# 2. Install the GPU stack (if needed)
$ scripts/setup/setup_nvidia.sh
$ scripts/setup/setup_cuda.sh
$ scripts/setup/setup_cudnn.sh

# 3. Configure models
$ cp env.example .env
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
# vLLM API: http://localhost:8003/v1
# Web UI:  http://localhost:3000
```

## Overview

This repository is a self-contained, hardware-optimized base for local LLM inference. It
follows the Unix philosophy: simple, composable, text-based components. The infrastructure
provides:

- Containerized inference on both CPU (llama.cpp) and GPU (llama.cpp and vLLM)
- Complete GPU software stack documentation (NVIDIA driver, CUDA 13.0, cuDNN 9.13)
- Parameterized Docker services for tuning without image rebuilds
- BIOS and OS optimizations for consistent inference latency
- uv-based Python dependency management with CUDA 13.0 wheel pinning
- A benchmarking suite with JSON output for tracking performance across changes
- Documentation covering hardware, BIOS, OS, GPU stack, inference tuning, and containers
- Automation scripts for setup, updates, and utilities

## Infrastructure services

### llama-gpu (port 8002)
- **Purpose**: Primary GPU inference service
- **Configuration**: Batch 2048, ubatch 512, Flash Attention enabled
- **Features**: CUDA-optimized, Blackwell architecture support

### llama-cpu (port 8001)
- **Purpose**: Latency-optimized CPU inference
- **Configuration**: 16 dedicated cores (0-15), huge pages enabled
- **Features**: Single-model focus for minimum latency

### vllm-gpu (port 8003)
- **Purpose**: High-throughput batch inference
- **Configuration**: vLLM on CUDA 13.0, PagedAttention in VRAM
- **Features**: Optimized for concurrent multi-user serving

### open-webui (port 3000)
- **Purpose**: Web interface for model interaction
- **Access**: http://localhost:3000
- **Features**: Configured for the inference services

### training (Phase 0 D1, no port)
- **Purpose**: Reproducible GPU environment for fine-tuning and pretraining
- **Configuration**: PyTorch cu130 on sm_120 with TRL, PEFT, DeepSpeed, torchao, flash-attn
- **Features**: Separate compose file (`docker-compose.training.yaml`); runs jobs via
  `make train-shell` / `make train-smoke`, never alongside inference (one GPU consumer at a time)

## Project structure

```
ai-experiments/
├── .claude/                   # Claude Code configuration
├── .styles/                   # Style guides
├── AGENTS.md                  # Agent operating rules
├── CLAUDE.md                  # Project instructions
├── docker/                    # Container definitions
│   ├── agents/               # Agent sandbox service
│   ├── comfyui-flux/         # ComfyUI + FLUX image service
│   ├── llama-cpu/            # CPU service
│   │   ├── Dockerfile.cpu
│   │   └── entrypoint.cpu.sh
│   ├── llama-gpu/            # GPU service
│   │   ├── Dockerfile.llama-gpu
│   │   └── entrypoint.sh
│   ├── vllm-gpu/             # vLLM service
│   │   ├── Dockerfile.vllm-gpu
│   │   └── entrypoint.sh
│   └── training/             # Training tier (Phase 0 D1)
│       ├── Dockerfile.training
│       └── entrypoint.sh
├── docs/                      # Documentation (30 files)
│   ├── agents/               # Agent tooling and parity
│   ├── bios/                 # BIOS configuration
│   ├── hardware/             # Hardware selection guide
│   ├── inference/            # Inference tuning
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
├── scripts/                   # Automation tooling
│   ├── setup/                # Installation scripts (8)
│   ├── update/               # Update scripts (7)
│   ├── utils/                # Utilities (4)
│   ├── agents/               # Agent guard and preflight (4)
│   └── benchmark/            # Performance testing (4)
│       ├── server-bench.py
│       ├── benchmark_cpu.sh
│       ├── benchmark_gpu.sh
│       ├── test_perplexity.sh
│       └── datasets/
├── tests/                     # pytest suite
├── docker-compose.yaml        # Service orchestration
├── pyproject.toml            # uv project and exact dependency pins
├── uv.lock                   # Locked dependencies
├── Makefile                  # Infrastructure commands
├── env.example               # Environment configuration template
├── .gitignore                # Git exclusions
└── README.md                 # This file
```

## Documentation map

### Getting started
- **Overview**: [docs/README.md](docs/README.md)
- **Hardware guide**: [docs/hardware/README.md](docs/hardware/README.md)
- **BIOS setup**: [docs/bios/README.md](docs/bios/README.md)
- **OS installation**: [docs/os/README.md](docs/os/README.md)

### GPU software stack
- **Overview**: [docs/os/gpu-stack/README.md](docs/os/gpu-stack/README.md)
- **NVIDIA drivers**: [docs/os/gpu-stack/nvidia-drivers.md](docs/os/gpu-stack/nvidia-drivers.md)
- **CUDA platform**: [docs/os/gpu-stack/cuda.md](docs/os/gpu-stack/cuda.md)
- **cuDNN library**: [docs/os/gpu-stack/cudnn.md](docs/os/gpu-stack/cudnn.md)

### Python environment
- **uv dependencies**: [docs/os/python/uv-dependencies.md](docs/os/python/uv-dependencies.md)
- **Python setup**: [docs/os/python/setup_python.md](docs/os/python/setup_python.md)
- **PyTorch CUDA**: [docs/os/python/pytorch_cuda_update_guide.md](docs/os/python/pytorch_cuda_update_guide.md)

### Inference optimization
- **Overview**: [docs/inference/README.md](docs/inference/README.md)
- **llama.cpp parameters**: [docs/inference/parameters/llama-cpp-parameters.md](docs/inference/parameters/llama-cpp-parameters.md)
- **vLLM parameters**: [docs/inference/parameters/vllm-parameters.md](docs/inference/parameters/vllm-parameters.md)

### System optimizations
- **Overview**: [docs/optimizations/README.md](docs/optimizations/README.md)
- **BIOS optimizations**: [docs/optimizations/bios/bios-optimizations.md](docs/optimizations/bios/bios-optimizations.md)
- **OS optimizations**: [docs/optimizations/os/os-optimizations.md](docs/optimizations/os/os-optimizations.md)

### Container documentation
- **Docker Compose**: [docs/sandbox/docker_compose_overview.md](docs/sandbox/docker_compose_overview.md)
- **CPU service**: [docs/sandbox/docker_llama_cpu_overview.md](docs/sandbox/docker_llama_cpu_overview.md)
- **GPU service**: [docs/sandbox/docker_llama_gpu_overview.md](docs/sandbox/docker_llama_gpu_overview.md)
- **vLLM service**: [docs/sandbox/docker_vllm_gpu_overview.md](docs/sandbox/docker_vllm_gpu_overview.md)

## Hardware configuration

### Current environment

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
- `update_python.sh` - Update the uv-managed Python environment
- `update_security.sh` - Security updates
- `update_ubuntu.sh` - System updates

### Utilities
- `check_py_deps_install.py` - Dependency verification
- `dependency_check.sh` - System dependency check
- `find_pythons.sh` - Python installation finder
- `storage_check.sh` - Storage verification

Model downloads use the Hugging Face CLI directly (ships with `huggingface-hub`):

```bash
# Full repository
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download <repo-id> --local-dir <dir>
# Single file (e.g. a GGUF quant)
HF_HUB_ENABLE_HF_TRANSFER=1 uv run hf download <repo-id> <filename> --local-dir <dir>
```

### Benchmarking

The benchmark suite measures inference performance and writes JSON output for tracking
across changes:

```bash
# Run GPU benchmark
$ scripts/benchmark/benchmark_gpu.sh

# Run CPU benchmark
$ scripts/benchmark/benchmark_cpu.sh

# Run server benchmark
$ scripts/benchmark/server-bench.py --service llama-gpu
```

## Python environment

### uv-based management
- **Python 3.12** via uv for a reproducible managed interpreter
- **uv** for reproducible dependency management with exact version pins
- **Key packages**:
  - torch (cu130) - GPU tensor computation
  - transformers - Pre-trained models
  - accelerate - Distributed training
  - deepspeed - Memory optimization
  - bitsandbytes - Quantization
  - peft - Parameter-efficient fine-tuning

Exact pinned versions are recorded in `pyproject.toml` and `uv.lock`.

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

## GPU software stack

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
- **96MB L2 cache** on Blackwell architecture
- **PCIe Gen5** with 128 GB/s bandwidth
- **32GB GDDR7 VRAM** at 1.8 TB/s

## System optimizations

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

## Contributing

Contributions are welcome. Areas of interest:

- Performance benchmarks on similar hardware
- Alternative optimization strategies
- Additional model compatibility testing
- Documentation improvements
- Cost-effective hardware alternatives

## License

This infrastructure is open source and available under the MIT License.

---

**Hardware:** RTX 5090 + AMD Ryzen 9950X
