# AI Engineering Workstation OS Setup Guide

Comprehensive guide for configuring Ubuntu 24.04 LTS as a high-performance local AI engineering workstation optimized for running large language models and AI workloads on the AMD Ryzen 9950X + RTX 5090 hardware configuration.

This documentation covers the complete system setup process, from initial OS installation through advanced GPU acceleration, containerization, and Python environment configuration. The setup supports both GPU-accelerated inference (keeping models loaded in VRAM) and CPU-based model execution for running multiple models simultaneously.

**Note**: This guide references a comprehensive suite of automation scripts in `scripts/` and configuration files. Review all scripts before execution and adapt them to your specific requirements and security policies.

## Table of Contents

- [AI Engineering Workstation OS Setup Guide](#ai-engineering-workstation-os-setup-guide)
  - [Table of Contents](#table-of-contents)
  - [Implementation Overview](#implementation-overview)
  - [Prerequisites](#prerequisites)
  - [1. Operating System Setup](#1-operating-system-setup)
    - [Ubuntu Installation and Configuration](#ubuntu-installation-and-configuration)
    - [Firefox Browser Configuration](#firefox-browser-configuration)
    - [Git and GitHub Integration](#git-and-github-integration)
    - [VS Code Development Environment](#vs-code-development-environment)
  - [2. Linux Environment Configuration](#2-linux-environment-configuration)
    - [User Setup](#user-setup)
    - [Terminal Environment](#terminal-environment)
    - [System Security](#system-security)
    - [Storage Configuration](#storage-configuration)
    - [System Information and Dependencies](#system-information-and-dependencies)
  - [3. NVIDIA GPU and CUDA Setup](#3-nvidia-gpu-and-cuda-setup)
    - [NVIDIA Driver Installation](#nvidia-driver-installation)
    - [CUDA Toolkit Configuration](#cuda-toolkit-configuration)
    - [cuDNN Installation](#cudnn-installation)
  - [4. Docker and Containerization](#4-docker-and-containerization)
    - [Docker Engine and NVIDIA Container Toolkit Setup](#docker-engine-and-nvidia-container-toolkit-setup)
  - [5. Python Development Environment](#5-python-development-environment)
    - [Python Version Management and Dependency Management](#python-version-management-and-dependency-management)
  - [6. Backup and Recovery Strategy](#6-backup-and-recovery-strategy)
    - [System Backup Configuration](#system-backup-configuration)
  - [Reference Implementation](#reference-implementation)

## Implementation Overview

The AI engineering workstation implementation leverages a comprehensive automation strategy designed specifically for the AMD Ryzen 9950X + RTX 5090 hardware configuration. The setup maximizes performance for AI workloads while maintaining system security and development flexibility.

**Key Implementation Features:**
- **Hardware optimization**: AMD Zen 5 and NVIDIA Blackwell architecture targeting
- **Automated setup**: Comprehensive script collection for consistent environment deployment
- **Security hardening**: Multi-layer security with AI development workflow preservation
- **Containerization**: Docker-based model isolation with GPU acceleration support
- **Python ecosystem**: Modern dependency management with AI framework optimization

**Hardware Configuration:**
- **CPU**: AMD Ryzen 9950X (16 cores, 32 threads, Zen 5 architecture)
- **GPU**: RTX 5090 32GB VRAM (Blackwell architecture, sm_120 compute capability)
- **RAM**: 128GB DDR5 EXPO for large model loading and multi-model deployment
- **Storage**: Samsung 990 Pro 2TB (OS) + Samsung 990 EVO 1TB (data) NVMe SSDs

## Prerequisites

**System Requirements:**
- Fresh Ubuntu 24.04 LTS installation on dedicated SSD
- Basic familiarity with Linux command line and system administration
- Internet connection for package downloads and updates
- Backup storage for system snapshots and configuration files

**Account Setup:**
- GitHub account for version control and configuration backup
- Cloud storage account for automated backup and recovery
- NVIDIA Developer account for proprietary driver and CUDA toolkit access

## 1. Operating System Setup

### Ubuntu Installation and Configuration

**Ubuntu 24.04 LTS Foundation**
Ubuntu 24.04 LTS provides excellent hardware support for NVIDIA RTX 5090 Blackwell architecture and AMD Ryzen 9950X Zen 5 processors, making it the optimal foundation for AI development workstations.

**System Installation Process:**
- Install Ubuntu 24.04 LTS on dedicated Samsung 990 Pro 2TB SSD
- Configure UEFI/BIOS settings for optimal performance (see `docs/bios/README.md`)
- Enable hardware acceleration features and memory configurations
- Set up initial user account with administrative privileges

**System Updates and Maintenance**
```bash
# Automated system updates
./scripts/update/update_ubuntu.sh
```

**Reference Files:**
- `scripts/update/update_ubuntu.sh`: Automated Ubuntu system updates and security patches
- `docs/bios/README.md`: UEFI/BIOS optimization guide for AI workstation hardware

### Firefox Browser Configuration

**Development-Focused Web Browser**
Firefox provides superior privacy controls, extension ecosystem, and developer tools essential for AI development workflows, including model repository access, documentation research, and API testing.

**Browser Setup Process:**
- Replace default browser with latest Firefox release
- Install development extensions: uBlock Origin, Privacy Badger, Developer tools
- Configure privacy settings for AI research and model downloading
- Set up bookmark synchronization for development resources
- Configure security settings for GitHub, Hugging Face, and AI platform access

### Git and GitHub Integration

**Version Control for AI Development**
Git integration provides essential version control for AI projects, model versioning, experiment tracking, and collaborative development across the AI engineering workflow.

**Git Configuration Process:**
- Configure Git with proper user credentials and commit signing
- Generate and configure SSH keys for secure GitHub authentication
- Set up GPG signing for commit verification and security
- Configure automated backup of configuration files to cloud repositories
- Establish repository structure for AI projects and model management

**GitHub Integration Features:**
- Model weight versioning with Git LFS for large files
- Experiment tracking through branch management and tagging
- Automated configuration backup to private repositories
- Collaborative development workflow with pull request management

### VS Code Development Environment

**AI-Optimized Development Environment**
Visual Studio Code provides comprehensive support for AI/ML development with Python, Jupyter notebooks, Docker integration, and specialized extensions for model development and deployment.

**VS Code Setup Process:**
- Install VS Code with AI/ML extension pack
- Configure Python interpreter integration with pyenv environments
- Set up Jupyter notebook support for interactive model development
- Install Docker extensions for container development and debugging
- Configure Git integration with SSH key authentication

**Essential AI Development Extensions:**
- Python: IntelliSense, linting, debugging, and code formatting
- Jupyter: Interactive notebook support with kernel management
- Docker: Container development and debugging capabilities
- Remote-SSH: Development on remote GPU servers
- GitLens: Advanced Git integration and history visualization
- Thunder Client: API testing for model inference endpoints

## 2. Linux Environment Configuration

### User Setup

**Secure AI Development User Management**
Proper user configuration ensures secure AI development environment with appropriate privileges for GPU access, container management, and system resource utilization.

**User Configuration Process:**
```bash
# Automated user setup with AI development privileges
./scripts/setup/setup_sudo_user.sh
```
- Create dedicated AI development user with controlled sudo privileges
- Configure user groups for NVIDIA GPU access (video, docker groups)
- Set up appropriate file permissions for model storage and cache directories
- Implement security controls while maintaining development flexibility
- Configure shell environment and development tool access

**Reference Files:**
- `scripts/setup/setup_sudo_user.sh`: Automated user configuration with AI development privileges

### Terminal Environment

**High-Performance Terminal Configuration**
Advanced terminal environment optimized for AI development workflows, model management, and system administration tasks across multiple concurrent sessions.

**Terminal Setup Process:**
```bash
# Comprehensive terminal environment setup
./scripts/setup/setup_terminal.sh
```
- Install and configure Zsh shell with AI-focused customizations and themes
- Set up Tilix terminal emulator with multi-pane and session management
- Install essential development packages: git, curl, wget, htop, nvtop, build tools
- Create custom aliases for Docker commands, model management, and GPU monitoring
- Configure environment variables for CUDA, Python, and development tools

**AI Development Aliases and Functions:**
- GPU monitoring: `nvidia-smi`, `nvtop`, and custom GPU memory tracking
- Docker shortcuts: Container management, image building, and log monitoring
- Model management: Download, conversion, and deployment automation
- System monitoring: CPU, memory, and storage utilization tracking

**Reference Files:**
- `scripts/setup/setup_terminal.sh`: Terminal configuration for Tilix and Zsh with AI development optimizations

### System Security

**AI Workstation Security Hardening**
Comprehensive security implementation protects AI models, data, and development environment from threats while preserving optimal performance for AI workloads and development flexibility.

**Security Implementation Process:**
```bash
# Comprehensive security hardening for AI development
./scripts/setup/setup_security.sh
```

**Automated Security Components:**
- **Automated Security Updates**: Configures unattended-upgrades with scheduled 3 AM kernel update reboots
- **AI Development Firewall**: UFW configuration allowing Jupyter (8888), TensorBoard (6006), Gradio (7860), Streamlit (8501), and Docker ports while blocking unnecessary traffic
- **Intrusion Prevention**: Fail2Ban deployment with AI development server protection rules
- **System Auditing**: Comprehensive logging for file access, authentication events, and system changes
- **Kernel Security Hardening**: 20+ security parameters including ASLR, SYN flood protection, IP spoofing prevention, and memory protection
- **AI Resource Limits**: System limits optimized for large model loading (unlimited memory lock, 65,000 file descriptors, 32,000 processes)
- **Daily Security Scanning**: Automated rootkit detection, security audits, and system integrity verification
- **Container Security**: Docker networking security with AI application development flexibility
- **AppArmor Integration**: Ubuntu mandatory access controls for additional process isolation

**AI Development Security Features:**
- Jupyter notebook security with token authentication and HTTPS
- Model repository access controls and secure credential management
- GPU resource access controls preventing unauthorized compute usage
- Container isolation for untrusted model execution environments

**Reference Files:**
- `scripts/setup/setup_security.sh`: Comprehensive security hardening with AI development workflow preservation


### Storage Configuration

**High-Performance Storage Optimization**
Storage configuration maximizes performance for large model loading, data processing, and AI development workloads across the dual-SSD storage architecture.

**Storage Setup Process:**
```bash
# Automated storage configuration for AI workloads
./scripts/setup/setup_data_ssd.sh
```
- Mount and configure Samsung 990 EVO 1TB SSD for model and data storage at `/mnt/ai-data`
- Optimize file system settings for large model files (>10GB) with appropriate block sizes
- Configure automated storage monitoring with alerts for disk usage thresholds
- Set up storage cleanup automation for temporary files and model cache management
- Implement SSD health monitoring and wear leveling optimization

**Storage Architecture:**
- **OS Drive**: Samsung 990 Pro 2TB (Ubuntu 24.04, applications, development tools)
- **Data Drive**: Samsung 990 EVO 1TB (models, datasets, Docker volumes, project data)
- **Model Storage**: Organized directory structure for GGUF, Safetensors, and HuggingFace models
- **Cache Management**: Automated cleanup for PyTorch, Transformers, and CUDA caches

**Performance Optimizations:**
- NVMe-specific mount options for maximum throughput
- File system tuning for large sequential reads (model loading)
- Swap configuration optimized for 128GB RAM AI workloads
- Automated defragmentation and maintenance scheduling

**Reference Files:**
- `scripts/setup/setup_data_ssd.sh`: Storage configuration automation for AI workstation dual-SSD setup

### System Information and Dependencies

**System State Analysis and Monitoring**
Comprehensive system analysis tools provide crucial insights for optimal AI model deployment, resource management, and troubleshooting across the hardware and software stack.

**System Analysis Process:**
```bash
# Comprehensive system analysis and dependency verification
./scripts/utils/dependency_check.sh
```
- Hardware detection and capability reporting (CPU, GPU, RAM, storage)
- NVIDIA driver and CUDA toolkit version verification
- Python environment and AI framework dependency analysis
- Docker and container runtime status verification
- System resource utilization and performance metrics
- Security configuration validation and compliance checking

**Dependency Verification Features:**
- **Hardware Analysis**: CPU features, GPU compute capability, memory configuration
- **Software Stack**: NVIDIA drivers, CUDA toolkit, cuDNN versions
- **Python Ecosystem**: PyTorch, TensorFlow, Transformers library versions
- **Container Environment**: Docker, NVIDIA Container Toolkit, runtime status
- **Performance Metrics**: Memory bandwidth, GPU utilization, storage throughput

**System Monitoring Integration:**
- Real-time hardware monitoring with `nvtop`, `htop`, and custom monitoring tools
- Automated health checks for critical AI development components
- Performance baseline establishment for optimization tracking
- Dependency conflict detection and resolution recommendations

**Reference Files:**
- `scripts/utils/dependency_check.sh`: Comprehensive system information and dependency analysis
- `scripts/utils/find_pythons.sh`: Python installation discovery and analysis
- `scripts/utils/check_py_deps_install.py`: Python dependency verification and analysis

## 3. NVIDIA GPU and CUDA Setup

### NVIDIA Driver Installation

**RTX 5090 Blackwell Architecture Support**
The RTX 5090 requires the latest NVIDIA drivers for optimal AI workload performance, Blackwell architecture support, and compatibility with CUDA 12.9.1 and advanced AI frameworks.

**Driver Installation Process:**
```bash
# Automated NVIDIA driver installation for RTX 5090
./scripts/setup/setup_nvidia.sh
```
- Install NVIDIA drivers (version 570+) with RTX 5090 Blackwell architecture support
- Configure driver persistence and power management for AI workloads
- Verify GPU detection, compute capability (sm_120), and VRAM accessibility (32GB)
- Set up automated driver update monitoring and installation
- Configure X11/Wayland integration for development environment compatibility

**Driver Verification:**
- GPU detection: `nvidia-smi` output validation
- Compute capability: CUDA device query verification
- Memory access: Full 32GB VRAM availability confirmation
- Performance: GPU boost clock and thermal management validation

**Reference Files:**
- `scripts/setup/setup_nvidia.sh`: Automated NVIDIA driver installation for RTX 5090
- `scripts/update/update_nvidia.sh`: Driver update automation and version management


### CUDA Toolkit Configuration

**CUDA 12.9.1 for Blackwell Architecture**
CUDA toolkit provides the foundation for GPU-accelerated AI model inference, training, and deployment with comprehensive Blackwell architecture support and optimization.

**CUDA Installation Process:**
```bash
# Automated CUDA 12.9.1 installation with Blackwell support
./scripts/setup/setup_cuda.sh
```
- Install CUDA Toolkit 12.9.1 with Blackwell sm_120 compute capability support
- Configure CUDA environment variables in `~/.zshrc` for development workflow
- Set up CUDA library paths and binary accessibility
- Verify CUDA installation with `nvcc --version` and device query
- Configure CUDA memory management for large model inference

**CUDA Configuration Features:**
- **Blackwell Support**: sm_120 compute capability targeting for RTX 5090
- **Environment Setup**: PATH and LD_LIBRARY_PATH configuration for development
- **Compiler Integration**: NVCC compiler accessibility for custom CUDA code
- **Library Management**: cuBLAS, cuDNN, and specialized AI library integration
- **Memory Optimization**: CUDA memory pool configuration for large model loading

**CUDA Development Tools:**
- NVCC compiler for custom kernel development
- CUDA profiler and debugger integration
- cuBLAS and cuSPARSE for optimized mathematical operations
- CUDA graphs support for inference optimization

**Reference Files:**
- `scripts/setup/setup_cuda.sh`: CUDA 12.9.1 installation with Blackwell architecture support
- `scripts/update/update_cuda.sh`: CUDA toolkit update automation

### cuDNN Installation

**Deep Learning Acceleration Library**
cuDNN provides hardware-accelerated deep neural network primitives essential for optimal AI model performance, with specialized RTX 5090 optimizations and Blackwell architecture support.

**cuDNN Installation Process:**
```bash
# Automated cuDNN installation matching CUDA 12.9.1
./scripts/setup/setup_cudnn.sh
```
- Install cuDNN 9.x libraries compatible with CUDA 12.9.1
- Configure cuDNN optimization settings for RTX 5090 Blackwell architecture
- Verify library installation and framework compatibility
- Set up cuDNN environment variables and library paths
- Configure cuDNN algorithm selection for inference optimization

**cuDNN Optimization Features:**
- **Blackwell Optimization**: Native RTX 5090 architecture support
- **Memory Management**: Optimized for 32GB VRAM utilization
- **Algorithm Selection**: Automatic optimization for transformer architectures
- **Precision Support**: FP32, FP16, and experimental FP8 support
- **Batch Processing**: Optimized batch sizes for inference workloads

**Framework Integration:**
- PyTorch: Native cuDNN backend integration
- TensorFlow: cuDNN acceleration for all operations
- JAX: cuDNN primitives for XLA compilation
- ONNX Runtime: cuDNN execution provider support

**Reference Files:**
- `scripts/setup/setup_cudnn.sh`: cuDNN installation automation with RTX 5090 optimization
- `scripts/update/update_cudnn.sh`: cuDNN update scripts and version management

## 4. Docker and Containerization

### Docker Engine and NVIDIA Container Toolkit Setup

**Containerized AI Model Deployment**
Docker enables consistent AI model deployment and simplified dependency management across development and production environments. The NVIDIA Container Toolkit provides seamless GPU access for containerized AI inference and training workloads.

**Docker Installation Process:**
```bash
# Automated Docker and NVIDIA Container Toolkit installation
./scripts/setup/setup_docker.sh
```
- Install Docker CE with proper user permissions and GPU access
- Configure Docker daemon for optimal AI container performance
- Install NVIDIA Container Toolkit for GPU acceleration in containers
- Set up Docker Compose for multi-container AI application orchestration
- Configure container resource limits and security policies

**NVIDIA Container Integration:**
- **GPU Access**: Seamless RTX 5090 access from Docker containers
- **CUDA Runtime**: Containerized CUDA 12.9.1 support
- **Memory Management**: Container VRAM allocation and limits
- **Multi-Container**: GPU sharing across multiple AI inference containers

**AI Container Architecture:**
The implementation supports the containerized AI model deployment documented in `docs/sandbox/`:
- **CPU Containers**: Three llama-cpu containers with AMD Zen 5 optimization
- **GPU Containers**: llama-gpu and vLLM containers with RTX 5090 acceleration
- **Security**: Container isolation with read-only filesystems and capability dropping
- **Networking**: Custom bridge networking with localhost-only access

**Container Orchestration:**
```yaml
# Multi-container AI deployment
docker-compose up -d
```

**Reference Files:**
- `scripts/setup/setup_docker.sh`: Docker and NVIDIA Container Toolkit installation
- `docker-compose.yaml`: Multi-container AI inference orchestration
- `docker/`: Specialized Dockerfiles for CPU and GPU AI inference
- `docs/sandbox/README.md`: Comprehensive containerized AI security architecture

## 5. Python Development Environment

### Python Version Management and Dependency Management

**Modern Python Ecosystem for AI Development**
Pyenv provides flexible Python version management essential for AI development with different framework requirements. Poetry offers robust dependency management and virtual environment handling optimized for AI projects and large framework installations.

**Python Environment Setup:**
- Install Pyenv for multiple Python version management
- Install Python 3.12.x optimized for AI workloads and framework compatibility
- Configure global and project-specific Python version selection
- Set up Poetry for advanced dependency management and virtual environment isolation
- Install PyTorch 2.6+ with CUDA 12.8 support for RTX 5090
- Configure TensorFlow, Transformers, and specialized AI frameworks

**AI Framework Configuration:**
The project uses Poetry for dependency management with optimized AI framework versions:

```toml
# Core AI/ML Frameworks from pyproject.toml
torch = {version = ">=2.3.0,<3.0.0", source = "pytorch_cuda_cu128"}
torchvision = {version = ">=0.18.0,<1.0.0", source = "pytorch_cuda_cu128"}
torchaudio = {version = ">=2.3.0,<3.0.0", source = "pytorch_cuda_cu128"}
transformers = ">=4.51.1,<4.52.0"
datasets = ">=3.1.0,<4.0.0"
accelerate = ">=1.8.1,<2.0.0"
```

**Specialized AI Libraries:**
- **Model Management**: huggingface-hub, hf-transfer for efficient model downloads
- **Quantization**: bitsandbytes for 4-bit and 8-bit model quantization
- **Optimization**: deepspeed for distributed training and inference
- **Experiment Tracking**: wandb, mlflow, tensorboard for development workflows
- **High-Performance**: tensorrt-llm, llama-cpp-python for optimized inference

**Development Environment Features:**
- **JupyterLab**: Interactive development with GPU acceleration
- **API Development**: FastAPI and uvicorn for model serving endpoints
- **Visualization**: matplotlib, seaborn for model performance analysis
- **Testing Framework**: pytest with async support for AI application testing

**GPU Verification and Testing:**
```bash
# Python dependency verification
python scripts/utils/check_py_deps_install.py
```

**Reference Files:**
- `pyproject.toml`: Complete AI framework dependency specification with CUDA support
- `docs/os/setup_python.md`: Detailed Python environment configuration guide
- `scripts/utils/check_py_deps_install.py`: Python dependency verification and GPU testing
- `scripts/utils/find_pythons.sh`: Python installation discovery and version management

## 6. Backup and Recovery Strategy

### System Backup Configuration

**Comprehensive AI Development Environment Protection**
Robust backup strategy protects AI development environment, valuable model data, experimental results, and system configurations against hardware failure, data corruption, and accidental deletion.

**Backup Implementation Strategy:**
- Configure Timeshift for automated system snapshots with AI development environment preservation
- Set up external backup storage (USB/network) for model data and experimental results
- Create specialized backup scripts for AI models, datasets, and training checkpoints
- Implement automated backup verification, integrity checking, and recovery testing
- Configure cloud backup integration for critical configuration files and code repositories

**AI-Specific Backup Components:**
- **System Snapshots**: Complete OS and development environment preservation
- **Model Storage**: Large model file backup with deduplication and compression
- **Configuration Files**: Docker configs, Python environments, and development settings
- **Experimental Data**: Training logs, checkpoints, and model evaluation results
- **Code Repositories**: Automated Git repository backup to multiple locations

**Backup Architecture:**
- **Local Snapshots**: Timeshift on dedicated SSD partitions
- **External Storage**: USB/NAS backup for model data and long-term archival
- **Cloud Integration**: GitHub for code, cloud storage for critical configurations
- **Automated Scheduling**: Daily incremental, weekly full system backups

**Recovery Testing:**
- Regular backup integrity verification
- Automated recovery simulation and testing
- Documentation of recovery procedures and time estimates
- Emergency recovery from different backup sources

**Reference Files:**
- `docs/os/backup_and_recovery.md`: Comprehensive backup system configuration and recovery procedures

## Reference Implementation

**File Structure and Automation:**
- `scripts/setup/`: Complete automation scripts for system configuration
- `scripts/update/`: Automated update and maintenance scripts
- `scripts/utils/`: System analysis and dependency verification tools
- `docs/os/`: Comprehensive setup and configuration documentation
- `pyproject.toml`: AI framework dependency specification
- `docker-compose.yaml`: Containerized AI inference orchestration

**Integration with AI Workstation:**
This OS setup works in conjunction with:
- **Hardware Optimization**: BIOS/UEFI configuration in `docs/bios/`
- **Docker Containers**: Containerized AI deployment in `docker/`
- **Security Architecture**: AI model sandboxing in `docs/sandbox/`
- **Performance Optimization**: System tuning in `docs/optimizations/`

---

*Last Updated: 2025-09-23*

*This AI engineering workstation OS setup provides a complete foundation for high-performance AI development on the AMD Ryzen 9950X + RTX 5090 platform as of mid-2025. All automation scripts and configurations are optimized for AI workloads while maintaining system security and development flexibility.*