# AI Engineering Workstation OS Setup Guide

## Table of Contents

- [AI Engineering Workstation OS Setup Guide](#ai-engineering-workstation-os-setup-guide)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [1. Operating System Setup](#1-operating-system-setup)
    - [1.1 Ubuntu Installation and Configuration](#11-ubuntu-installation-and-configuration)
    - [1.2 Firefox Browser Configuration](#12-firefox-browser-configuration)
    - [1.3 Git and GitHub Integration](#13-git-and-github-integration)
    - [1.4 VS Code Development Environment](#14-vs-code-development-environment)
  - [2. Linux Environment Configuration](#2-linux-environment-configuration)
    - [2.1 User Setup](#21-user-setup)
    - [2.2 Terminal Environment](#22-terminal-environment)
    - [2.3 System Security](#23-system-security)
    - [2.4 Storage Configuration](#24-storage-configuration)
    - [2.5 System Information and Dependencies](#25-system-information-and-dependencies)
  - [3. NVIDIA GPU and CUDA Setup](#3-nvidia-gpu-and-cuda-setup)
    - [3.1 NVIDIA Driver Installation](#31-nvidia-driver-installation)
    - [3.2 CUDA Toolkit Configuration](#32-cuda-toolkit-configuration)
    - [3.3 cuDNN Installation](#33-cudnn-installation)
  - [4. Docker and Containerization](#4-docker-and-containerization)
    - [4.1 Docker Engine and NVIDIA Container Toolkit Setup](#41-docker-engine-and-nvidia-container-toolkit-setup)
  - [5. Python Development Environment](#5-python-development-environment)
    - [5.1 Python Version Management and Dependency Management](#51-python-version-management-and-dependency-management)
  - [6. Backup and Recovery Strategy](#6-backup-and-recovery-strategy)
    - [6.1 System Backup Configuration](#61-system-backup-configuration)

---

## Overview

This guide provides a comprehensive walkthrough for configuring Ubuntu LTS 24.04 a high-performance local AI engineering workstation optimized for running large language models and AI workloads. The setup supports both GPU-accelerated inference (keeping models loaded in VRAM) and CPU-based model execution for running multiple models simultaneously. This guide has a suit of scripts and markup docs that can be referenced for seting up and maintaining AI engineering dependencies. Do **NOT** execute scripts or the commands in the docs; but instead use them as guides for what you may want to do on your OS.


## Prerequisites

- Basic familiarity with Linux command line
- GitHub account for version control
- Cloud storage account for configuration backups

---

## 1. Operating System Setup

### 1.1 Ubuntu Installation and Configuration

Ubuntu 24.04 LTS provides excellent hardware support for the Nvidia GPUs and modern AMD processors, making it the ideal foundation for AI development.

**Key Steps:**
- Install Ubuntu 24.04 on the dedicated OS SSD.
- Perform initial system updates 
- Configure system settings 
  
**Reference Files:**
- `scripts/update_ubuntu.sh` - updates ubuntu system

### 1.2 Firefox Browser Configuration

Firefox offers better privacy controls and extension ecosystem for AI development workflows compared to default browsers.

**Key Steps:**
- Replace default browser with Firefox
- Install essential extensions for development and productivity
- Configure privacy and security settings
- Sync settings across development environments

### 1.3 Git and GitHub Integration

Version control is essential for AI project management, model versioning, and collaboration.

**Key Steps:**
- Configure Git with proper credentials and signing
- Set up GitHub integration with SSH keys
- Implement automated backup of configuration files to cloud storage
- Create and configure GitHub account

### 1.4 VS Code Development Environment

VS Code provides excellent support for AI/ML development with Python, Jupyter notebooks, and model management extensions.

**Key Steps:**
- Install VS Code with AI-focused extensions
- Remove or disable Copilot if preferred
- Configure workspace settings for Python AI development
- Set up extension synchronization and backup

---

## 2. Linux Environment Configuration

### 2.1 User Setup

Proper user management ensures a stable and secure AI development environment.

**Key Steps:**
- Create dedicated AI development user with appropriate sudo privileges
- Configure user groups for GPU and hardware access
- Implement security best practices and access controls

**Reference Files:**
- `scripts/setup_sudo_user.sh` - User configuration automation

### 2.2 Terminal Environment

A powerful terminal environment accelerates AI development workflows and model management tasks.

**Key Steps:**
- Install and configure Zsh with AI-focused customizations
- Set up Tilix terminal emulator for multi-session management
- Install essential Linux packages for AI development
- Create custom aliases and functions for common AI tasks

**Reference Files:**
- `scripts/setup_terminal.sh` - Terminal configuration for Tillix and Zsh

### 2.3 System Security

Security hardening protects your AI models, data, and development environment from threats while maintaining optimal performance for AI workloads.

**Key Steps:**
- **Automated Security Updates**: Configures unattended-upgrades to automatically install security patches with scheduled 3 AM reboots for kernel updates
- **AI Development Firewall**: Sets up UFW with rules specifically for AI workflows - allows Jupyter (8888), TensorBoard (6006), Gradio (7860), Streamlit (8501), and common development server ports while blocking unnecessary traffic
- **Intrusion Prevention**: Deploys Fail2Ban to automatically block malicious login attempts and common attack patterns
- **System Auditing**: Enables comprehensive logging of file access, authentication events, and system changes for security monitoring
- **Kernel Security Hardening**: Applies 20+ security parameters including ASLR, SYN flood protection, IP spoofing prevention, and memory protection specifically tuned for AI workstation requirements
- **Resource Limits for AI**: Configures system limits to handle large AI model loading (unlimited memory lock, 65K file descriptors, 32K processes)
- **Daily Security Scanning**: Automated rootkit detection, security audits, and system integrity checks
- **Container Security**: Secures Docker networking for containerized AI applications while maintaining development flexibility
- **AppArmor Integration**: Leverages Ubuntu's built-in mandatory access controls for additional process isolation

**Reference Files:**
- `scripts/setup_security.sh` - Firewall configuration


### 2.4 Storage Configuration

Proper storage setup maximizes performance for model loading and data processing workloads.

**Key Steps:**
- Mount and configure the SDD for model/data storage
- Set up optimal file system settings for large model files
- Configure automated storage monitoring and cleanup
- Implement storage performance optimization

**Reference Files:**
- `scripts/setup_data_ssd.sh` - Storage configuration automation

### 2.5 System Information and Dependencies

Understanding your system's state and dependencies is crucial for optimal AI model deployment and resource management.

**Key Steps:**
- Create comprehensive system information and dependency gathering script

**Reference Files:**
- `scripts/dependency_check.sh` - System information script

---

## 3. NVIDIA GPU and CUDA Setup

### 3.1 NVIDIA Driver Installation

The RTX 5090 requires the latest NVIDIA drivers for optimal AI workload performance and compatibility.

**Key Steps:**
- Install NVIDIA drivers compatible with RTX 5090
- Verify GPU detection and functionality
- Set up driver update automation

**Reference Files:**
- `scripts/setup_nvidia.sh` - Driver installation automation
- `scripts/update_nvidia.sh` - Update drivers


### 3.2 CUDA Toolkit Configuration

CUDA provides the foundation for GPU-accelerated AI model inference and training.

**Key Steps:**
- Install CUDA toolkit compatible with your AI frameworks
- Configure CUDA environment variables and paths

**Reference Files:**
- `scripts/setup_cuda.sh` - CUDA installation script
- `scripts//update_cuda.sh` - CUDA update script

### 3.3 cuDNN Installation

cuDNN accelerates deep neural network computations and is essential for optimal model performance.

**Key Steps:**
- Install cuDNN libraries matching your CUDA version
- Configure cuDNN for optimal performance on RTX 5090

**Reference Files:**
- `scripts/setup_cudnn.sh` - cuDNN installation automation
- `scripts/update_cudnn.sh` - cuDNN update scripts

---

## 4. Docker and Containerization

### 4.1 Docker Engine and NVIDIA Container Toolkit Setup

Docker enables consistent AI model deployment and simplified dependency management across different environments. The NVIDIA Container Toolkit allows Docker containers to access GPU resources for AI model inference.

**Key Steps:**
- Install Docker CE with proper user permissions
- Install NVIDIA Container Toolkit

**Reference Files:**
- `scripts/setup_docker.sh` - Docker installation script
---

## 5. Python Development Environment

### 5.1 Python Version Management and Dependency Management

Pyenv provides flexible Python version management essential for AI development with different framework requirements. Poetry provides robust dependency management and virtual environment handling for AI projects.

**Key Steps:**
- Install Pyenv for Python version management
- Install Python 3.13 optimized for AI workloads
- Set up global and project-specific Python versions
- Install Poetry for dependency management
- Configure Poetry for AI-specific package management
- Install PyTorch, TensorFlow, and other AI frameworks
- Create comprehensive GPU and CPU functionality tests

**Reference Files:**
- `docs/setup_python` - Framework installation

---

## 6. Backup and Recovery Strategy

### 6.1 System Backup Configuration

A robust backup strategy protects your AI development environment and valuable model data.

**Key Steps:**
- Configure Timeshift for automated system snapshots
- Set up external backup storage (USB/network)
- Create backup scripts for AI models and data
- Implement automated backup verification and testing

**Reference Files:**
- `docs/backup_and_recovery.md` - Backup system configuration

---