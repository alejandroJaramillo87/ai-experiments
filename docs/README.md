# AI Experiments Documentation Hub

Central index for all project documentation. This repository contains 36 comprehensive guides covering hardware setup, software configuration, optimization strategies, and performance benchmarking for a local AI inference workstation.

## Quick Navigation

- **[Back to Main README](../README.md)** - Project overview and quick start
- **[Hardware Setup](hardware/README.md)** - Component selection and assembly
- **[Getting Started](os/README.md)** - Ubuntu installation and initial setup
- **[Performance Results](optimizations/README.md)** - 286.85 tok/s GPU | 35.44 tok/s CPU

## Documentation by Category

### Infrastructure Setup

Complete guides for building your AI workstation from hardware to software:

| Document | Description |
|----------|-------------|
| [hardware/README.md](hardware/README.md) | Hardware selection, compatibility, and assembly guide |
| [bios/README.md](bios/README.md) | BIOS configuration for Gigabyte X870E Aorus Elite WiFi |
| [os/README.md](os/README.md) | Ubuntu 24.04 LTS installation and configuration |
| [os/storage-configuration.md](os/storage-configuration.md) | Dual-SSD architecture and AI data storage setup |
| [os/backup_and_recovery.md](os/backup_and_recovery.md) | Data protection and recovery strategies |
| [sandbox/README.md](sandbox/README.md) | Docker sandboxed environment setup |
| [sandbox/docker_compose_overview.md](sandbox/docker_compose_overview.md) | Multi-service orchestration configuration |

### GPU Software Stack

NVIDIA GPU software stack documentation for RTX 5090:

| Document | Description | Version |
|----------|-------------|---------|
| [os/gpu-stack/README.md](os/gpu-stack/README.md) | Three-layer GPU stack overview | - |
| [os/gpu-stack/nvidia-drivers.md](os/gpu-stack/nvidia-drivers.md) | NVIDIA driver installation and configuration | 580.65.06 |
| [os/gpu-stack/cuda.md](os/gpu-stack/cuda.md) | CUDA parallel computing platform | 13.0.88 |
| [os/gpu-stack/cudnn.md](os/gpu-stack/cudnn.md) | cuDNN deep learning acceleration library | 9.13.0.50-1 |

### Python Environment

Python and dependency management with Poetry:

| Document | Description |
|----------|-------------|
| [os/python/poetry-dependencies.md](os/python/poetry-dependencies.md) | Comprehensive guide to all AI/ML dependencies |
| [os/python/setup_python.md](os/python/setup_python.md) | Python 3.12 installation via pyenv |
| [os/python/pytorch_cuda_update_guide.md](os/python/pytorch_cuda_update_guide.md) | PyTorch installation with CUDA 12.9 compatibility |

### Inference Optimization

Parameter tuning and optimization for maximum performance:

**Inference Engines:**
| Document | Description | Performance |
|----------|-------------|-------------|
| [inference/engines/inference-engines.md](inference/engines/inference-engines.md) | Understanding inference engines vs training frameworks | - |
| [inference/engines/llama-cpp.md](inference/engines/llama-cpp.md) | llama.cpp architecture and memory management | - |
| [inference/engines/vllm.md](inference/engines/vllm.md) | vLLM PagedAttention and continuous batching | - |

**Parameter Configuration:**
| Document | Description | Performance |
|----------|-------------|-------------|
| [inference/README.md](inference/README.md) | Optimization philosophy and overview | - |
| [inference/parameters/llama-cpp-parameters.md](inference/parameters/llama-cpp-parameters.md) | llama.cpp parameter tuning guide | 286.85 tok/s (GPU) |
| [inference/parameters/vllm-parameters.md](inference/parameters/vllm-parameters.md) | vLLM configuration for throughput | Awaiting CUDA 13 |
| [inference/parameters/optimization-guide.md](inference/parameters/optimization-guide.md) | Systematic benchmarking methodology | - |

**Performance Benchmarks:**
| Document | Description | Performance |
|----------|-------------|-------------|
| [inference/benchmarks/performance-results.md](inference/benchmarks/performance-results.md) | Validated performance measurements and analysis | 286.85 tok/s GPU, 35.44 tok/s CPU |

### System Optimizations

Performance optimizations at every level:

| Document | Description | Impact |
|----------|-------------|--------|
| [optimizations/README.md](optimizations/README.md) | Optimization overview and verification | +26% CPU performance |
| [optimizations/benchmark-guide.md](optimizations/benchmark-guide.md) | Performance testing tool documentation | - |
| **BIOS Optimizations** | | |
| [optimizations/bios/bios-optimizations.md](optimizations/bios/bios-optimizations.md) | BIOS settings for AMD Ryzen 9950X | FCLK 2100MHz |
| **OS Optimizations** | | |
| [optimizations/os/os-optimizations.md](optimizations/os/os-optimizations.md) | Operating system tuning | Performance governor |
| [optimizations/os/hugepages-explicit.md](optimizations/os/hugepages-explicit.md) | Huge pages memory optimization | 90GB allocated |
| **GPU Optimizations** | | |
| [optimizations/gpu/gpu-optimizations.md](optimizations/gpu/gpu-optimizations.md) | RTX 5090 specific optimizations | 95% utilization |

### Experimental Features

Advanced and experimental optimizations:

| Document | Description | Status |
|----------|-------------|--------|
| [optimizations/experiments/experimental-1gb-hugepages.md](optimizations/experiments/experimental-1gb-hugepages.md) | 1GB huge pages testing | Experimental |
| [optimizations/experiments/experimental-gpu-rtx5090.md](optimizations/experiments/experimental-gpu-rtx5090.md) | RTX 5090 advanced features | Testing |
| [optimizations/experiments/experimental-memory-bandwidth-optimizations.md](optimizations/experiments/experimental-memory-bandwidth-optimizations.md) | Memory bandwidth tuning | Research |
| [optimizations/experiments/experimental-vnni-wrapper.md](optimizations/experiments/experimental-vnni-wrapper.md) | VNNI instruction optimization | Experimental |

### Container Services

Docker container documentation for inference services:

| Document | Service | Performance |
|----------|---------|-------------|
| [sandbox/docker_llama_cpu_overview.md](sandbox/docker_llama_cpu_overview.md) | llama-cpu service (Port 8001) | 35.44 tok/s |
| [sandbox/docker_llama_gpu_overview.md](sandbox/docker_llama_gpu_overview.md) | llama-gpu service (Port 8004) | 286.85 tok/s |
| [sandbox/docker_vllm_gpu_overview.md](sandbox/docker_vllm_gpu_overview.md) | vllm-gpu service (Port 8005) | Awaiting CUDA 13 |
| [sandbox/docker_compose_overview.md](sandbox/docker_compose_overview.md) | Service orchestration | - |

## Performance Summary

### Validated Benchmarks

| Service | Hardware | Model | Performance | Benchmark File |
|---------|----------|-------|-------------|----------------|
| **llama-gpu** | RTX 5090 | gpt-oss-20b (Q8_K_XL) | **286.85 tok/s** | `optimizations/os/gpu_baseline.json` |
| **llama-cpu** | AMD Ryzen 9950X | Qwen3-Coder-30B (IQ4_XS) | **35.44 tok/s** | `optimizations/bios/cpu_benchmark_results_final_bios.json` |

### Key Optimizations Applied

- **GPU**: Batch 2048, Ubatch 512, Flash Attention v3
- **CPU**: Batch 2048, 12 threads, huge pages enabled
- **BIOS**: FCLK 2100MHz, C-states disabled, PBO enabled
- **OS**: Performance governor, swap disabled, CPU pinning

## Documentation Standards

This documentation follows Linux/Unix conventions:

- **Plain text focus** - Clear, technical writing without decorative elements
- **No emojis in code** - Professional documentation uses standard ASCII
- **Command-line oriented** - Examples use shell commands and Unix tools
- **Minimalist approach** - Essential information only, no redundancy

## Quick Reference

### Most Frequently Accessed

1. **[GPU Stack Overview](os/gpu-stack/README.md)** - Understanding the NVIDIA software stack
2. **[Inference Parameters](inference/llama-cpp-parameters.md)** - Tuning for performance
3. **[Benchmarking Guide](optimizations/benchmark-guide.md)** - Testing your setup
4. **[Poetry Dependencies](os/python/poetry-dependencies.md)** - Python package management

### Troubleshooting Guides

- **GPU Issues**: See [nvidia-drivers.md](os/gpu-stack/nvidia-drivers.md#troubleshooting)
- **Performance Problems**: See [benchmark-guide.md](optimizations/benchmark-guide.md)
- **Python/CUDA Conflicts**: See [pytorch_cuda_update_guide.md](os/python/pytorch_cuda_update_guide.md)
- **Container Issues**: See [docker_compose_overview.md](sandbox/docker_compose_overview.md)

## Contributing to Documentation

When adding new documentation:

1. **Placement**:
   - Hardware guides: `docs/hardware/`
   - Software setup: `docs/os/`
   - Optimization guides: `docs/optimizations/`
   - Container docs: `docs/sandbox/`
   - Inference tuning: `docs/inference/`

2. **Naming Convention**:
   - Use lowercase with hyphens: `feature-name.md`
   - Be descriptive but concise
   - Include category prefix when helpful

3. **Content Standards**:
   - Include table of contents for long documents
   - Provide concrete examples and commands
   - Document actual performance measurements
   - Reference related documentation

## Documentation Coverage

| Category | File Count | Status |
|----------|------------|--------|
| Infrastructure Setup | 7 | Complete |
| GPU Software Stack | 4 | Complete |
| Python Environment | 3 | Complete |
| Inference Optimization | 8 | Complete |
| System Optimizations | 8 | Complete |
| Container Services | 4 | Complete |
| Experimental | 4 | Ongoing |
| **Total** | **36** | **Complete** |

---

*Last Updated: September 2025 | Version 1.0.0 | [Main README](../README.md)*