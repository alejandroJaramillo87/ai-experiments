# AI Model Sandboxing and Security Architecture

Comprehensive guide for secure AI model deployment using containerized sandboxing environments, addressing security risks inherent in local AI model execution and providing isolation strategies for the AMD Ryzen 9950X + RTX 5090 AI workstation.

This documentation covers the specific Docker implementation deployed in this project, including three specialized Dockerfiles (`Dockerfile.llama-cpu`, `Dockerfile.llama-gpu`, `Dockerfile.vllm-gpu`) and the docker-compose orchestration that enables concurrent model hosting across CPU and GPU resources.

**Note**: This guide focuses on defensive security measures for AI model deployment. All recommendations prioritize system security and stability over performance optimization.

## Table of Contents

- [AI Model Sandboxing and Security Architecture](#ai-model-sandboxing-and-security-architecture)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Why Sandboxing is Required](#why-sandboxing-is-required)
  - [Security Risks in Local AI Model Execution](#security-risks-in-local-ai-model-execution)
    - [Model-Specific Vulnerabilities](#model-specific-vulnerabilities)
    - [System Resource Exploitation](#system-resource-exploitation)
    - [Data Exposure Risks](#data-exposure-risks)
    - [Network Attack Vectors](#network-attack-vectors)
  - [Docker Container Isolation Implementation](#docker-container-isolation-implementation)
    - [Process Isolation](#process-isolation)
    - [File System Security](#file-system-security)
    - [Network Segmentation](#network-segmentation)
    - [Resource Constraints](#resource-constraints)
  - [HTTP Server Architecture](#http-server-architecture)
    - [API Gateway Pattern](#api-gateway-pattern)
    - [Authentication and Authorization](#authentication-and-authorization)
    - [Request Validation](#request-validation)
    - [Health Monitoring and DoS Protection](#health-monitoring-and-dos-protection)
  - [Container Security Configuration](#container-security-configuration)
    - [Runtime Security](#runtime-security)
    - [Image Security](#image-security)
    - [Network Security](#network-security)
  - [Security Best Practices](#security-best-practices)
    - [Principle of Least Privilege](#principle-of-least-privilege)
  - [Implementation Guidelines](#implementation-guidelines)
    - [Current Deployment Architecture](#current-deployment-architecture)
    - [Production Deployment](#production-deployment)
  - [Reference Files](#reference-files)
    - [Docker Configuration Files](#docker-configuration-files)
    - [Security Features Implemented](#security-features-implemented)

## Overview

AI model sandboxing provides essential security isolation for local AI workloads, protecting both the host system and sensitive data from potential threats inherent in AI model execution. This architecture uses Docker containers as the primary sandboxing mechanism, combined with HTTP server security patterns to create a robust defense framework.

The implementation consists of:
- **Three CPU inference containers** (`llama-cpu-0`, `llama-cpu-1`, `llama-cpu-2`) utilizing AMD Zen 5 optimization with AOCL libraries
- **GPU inference container** (`llama-gpu`) optimized for RTX 5090 with CUDA 12.9.1 and Blackwell architecture support
- **vLLM GPU container** (`vllm-gpu`) for high-performance transformer model serving
- **Open WebUI container** providing a secure web interface for model interaction
- **Isolated Docker network** (`ai-network`) with controlled inter-service communication

The sandboxing approach addresses three critical security domains:
- **Computational Security**: CPU pinning, memory limits, and resource constraints prevent resource exhaustion
- **Data Protection**: Read-only model mounts and isolated file systems protect sensitive data
- **Network Security**: Localhost-only port binding and custom bridge networking control access

## Why Sandboxing is Required

**Untrusted Code Execution**
AI models, particularly those from third-party sources, execute complex computational graphs that may contain malicious or unintended behaviors. Without proper isolation, these models can access system resources, file systems, and network interfaces beyond their intended scope.

**Resource Management**
Large AI models can consume substantial system resources (CPU, memory, GPU), potentially causing system instability or denial of service. Sandboxing provides resource limits and prevents resource exhaustion attacks.

**Model Integrity**
Containerized environments ensure consistent model execution environments, preventing environmental inconsistencies that could affect model behavior or introduce security vulnerabilities.

**Regulatory Compliance**
Many industries require strict data isolation and processing controls. Sandboxing provides auditable boundaries for data processing and model execution activities.

## Security Risks in Local AI Model Execution

### Model-Specific Vulnerabilities

**Adversarial Model Attacks**
- Poisoned models designed to execute malicious code during inference
- Models trained with backdoors that activate under specific input conditions
- Supply chain attacks through compromised model repositories

**Prompt Injection Vulnerabilities**
- Malicious prompts designed to bypass safety guardrails
- Information extraction attacks through carefully crafted inputs
- Model jailbreaking attempts to access restricted functionality

**Model Inversion Attacks**
- Attempts to extract training data from model parameters
- Membership inference attacks to determine if specific data was used in training
- Model stealing through query-based extraction

### System Resource Exploitation

**Memory Exhaustion**
- Large model parameters consuming excessive system memory
- Memory leak vulnerabilities in model inference engines
- Out-of-memory conditions causing system instability

**CPU and GPU Abuse**
- Infinite loops or computationally expensive operations
- Cryptocurrency mining through hijacked compute resources
- Distributed computing exploitation using local hardware

**Storage System Attacks**
- Excessive disk I/O operations causing system slowdown
- Temporary file proliferation leading to disk exhaustion
- Log file manipulation to hide malicious activities

### Data Exposure Risks

**Training Data Leakage**
- Unintended exposure of sensitive training data through model outputs
- Model memorization of personally identifiable information (PII)
- Cross-contamination between different model training datasets

**Local File System Access**
- Unauthorized access to sensitive files outside model directories
- Configuration file manipulation affecting system security
- Credential harvesting from unprotected configuration files

**Network Data Exfiltration**
- Unauthorized transmission of processed data to external servers
- Covert channels through model output patterns
- DNS tunneling for data exfiltration

### Network Attack Vectors

**Lateral Movement**
- Using model servers as pivot points for network reconnaissance
- Exploitation of shared network segments for additional system access
- Inter-container communication exploitation

**Command and Control**
- Models establishing persistent connections to external command servers
- Use of model inference endpoints as proxy servers
- Covert communication through model API responses

## Docker Container Isolation Implementation

### Process Isolation

**Namespace Separation**
Docker containers utilize Linux namespaces to provide process-level isolation, ensuring that containerized AI models cannot interact with host processes or other containers without explicit configuration.

**User Space Isolation**
All containers implement non-root user execution:
- **CPU containers**: Use `aiuser` with UID 1001 and dedicated group
- **GPU containers**: Use `appuser` for privilege separation
- This prevents privilege escalation and limits container breakout impact

**Control Group (cgroup) Constraints**
Implemented through docker-compose resource limits:
- **CPU containers**: Limited to 8 cores each with 32GB RAM per container
- **GPU containers**: Controlled memory allocation with unlimited memlock for CUDA operations
- **CPU pinning**: Each CPU container bound to specific core sets (0-7, 8-15, 16-23)

### File System Security

**Read-Only Root File Systems**
Implemented across all inference containers:
```yaml
read_only: true
```
Immutable container file systems prevent runtime modifications and reduce attack surface for persistent threats.

**Volume Mount Restrictions**
Selective file system mounting implemented:
- **Model storage**: `/mnt/ai-data/models/:/app/models:ro` (read-only access)
- **Log directories**: `./logs/cpu:/app/logs` and `./logs/gpu:/app/logs` (write access only to designated areas)
- **CUDA cache**: vLLM container uses tmpfs mount `/tmp:size=2G,mode=1777` for temporary GPU operations

**Temporary File System Controls**
vLLM container implements isolated tmpfs for CUDA operations while other containers use read-only root filesystems with designated log directories.

### Network Segmentation

**Bridge Network Isolation**
Custom Docker network `ai-network` with dedicated subnet:
```yaml
networks:
  ai-network:
    driver: bridge
    internal: false
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

**Port Mapping Controls**
Localhost-only port binding prevents external access:
- **llama-cpu-0**: `127.0.0.1:8001:8001`
- **llama-cpu-1**: `127.0.0.1:8002:8002`  
- **llama-cpu-2**: `127.0.0.1:8003:8003`
- **llama-gpu**: `127.0.0.1:8004:8004`
- **vllm-gpu**: `127.0.0.1:8005:8005`
- **open-webui**: `3000:8080`

**Network Policy Enforcement**
Internal Docker networking allows controlled inter-service communication while preventing external access to inference engines.

### Resource Constraints

**Memory Limits**
Hard memory limits implemented per container:
- **CPU containers**: 32GB RAM limit each, distributed across 16-core CPU
- **GPU containers**: Unlimited memlock for CUDA operations with controlled system memory
- **tmpfs limits**: 2GB temporary storage for vLLM CUDA cache

**CPU Throttling**
CPU usage limits implemented through cpuset pinning:
- **llama-cpu-0**: Cores 0-7 (8 cores, 8.0 CPU limit)
- **llama-cpu-1**: Cores 8-15 (8 cores, 8.0 CPU limit)  
- **llama-cpu-2**: Cores 16-23 (8 cores, 8.0 CPU limit)
- Prevents compute resource monopolization while maximizing utilization

**GPU Access Controls**
Selective GPU access implemented:
```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=0
  - CUDA_VISIBLE_DEVICES=0
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```
Ensures only authorized containers access RTX 5090 hardware acceleration.

## HTTP Server Architecture

### API Gateway Pattern

**Centralized Entry Point**
Open WebUI container provides secure web interface:
- **Port binding**: `3000:8080` for web access
- **Internal API routing**: Communicates with model containers via Docker network
- **Authentication**: No external API keys required for internal communication

**Request Routing**
Model-specific endpoint distribution:
- **CPU models**: Ports 8001, 8002, 8003 for parallel inference
- **GPU models**: Port 8004 (llama-gpu), Port 8005 (vLLM)
- **Load balancing**: Multiple CPU containers enable concurrent request handling

**Protocol Translation**
OpenAI-compatible API endpoints:
```yaml
environment:
  - 'OPENAI_API_BASE_URL=http://llama-gpu:8006/v1'
  - 'OPENAI_API_KEY=NONE'
```
Internal Docker networking eliminates need for external authentication keys.

### Authentication and Authorization

**Token-Based Authentication**
- Stateless authentication mechanisms prevent session hijacking and enable scalable security across multiple model instances
- JWT token validation and refresh mechanisms
- Session timeout and automatic token expiration

**Role-Based Access Control (RBAC)**
- Granular permissions systems ensure users only access authorized model functionality and data
- User role definitions and access matrices
- Dynamic permission evaluation

**API Key Management**
- Secure API key generation, distribution, and rotation prevent unauthorized model access
- Key lifecycle management and revocation
- Rate limiting per API key

### Request Validation

**Input Sanitization**
- Comprehensive input validation prevents prompt injection attacks and malformed request exploitation
- Content filtering and payload size limits
- Character encoding validation

**Schema Enforcement**
- Strict API schema validation ensures requests conform to expected patterns and prevent bypass attempts
- JSON schema validation
- Request structure verification

**Content Security**
- Request content filtering prevents malicious payload delivery through model inputs
- Malware scanning for uploaded content
- Content type validation

### Health Monitoring and DoS Protection

**Health Check Implementation**
All containers implement comprehensive health monitoring:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8001/v1/health"]
  interval: 30s
  timeout: 10s
  retries: 5
  start_period: 180s
```
- **CPU containers**: 180s start period for model loading into RAM
- **GPU containers**: 180s-300s start period for VRAM allocation
- **Automatic restart**: `unless-stopped` policy ensures service availability

**Resource-Based Protection**
Hard resource limits prevent resource exhaustion:
- **Memory limits**: 32GB per CPU container prevents memory exhaustion
- **CPU pinning**: Isolates workloads to specific core sets
- **tmpfs limits**: 2GB limit on temporary storage prevents disk exhaustion

**Container Restart Policies**
Automatic recovery from failures:
```yaml
restart: unless-stopped
```
Ensures service continuity while maintaining security boundaries.

## Container Security Configuration

### Runtime Security

**Security Profiles**
All containers implement comprehensive security hardening:
```yaml
security_opt:
  - no-new-privileges:true
cap_drop:
  - ALL
```
Prevents privilege escalation and removes all Linux capabilities by default.

**Capability Dropping**
All containers drop ALL capabilities, implementing principle of least privilege:
- Reduces attack surface for container breakout attempts
- Prevents access to privileged kernel functions
- Applied consistently across CPU, GPU, and vLLM containers

**Memory Lock Controls**
Unlimited memory lock for AI workloads:
```yaml
ulimits:
  memlock:
    soft: -1
    hard: -1
```
Required for large model loading while maintaining security boundaries.

### Image Security

**Base Image Hardening**
- Minimal base images reduce attack surface and eliminate unnecessary packages
- Multi-stage builds separate build dependencies from runtime (as implemented in project Dockerfiles)
- Regular base image updates and security patches

### Network Security

**Firewall Integration**
- Container-aware firewall rules provide additional network security layers beyond Docker networking
- Host-level firewall configuration
- Network policy enforcement

## Security Best Practices

### Principle of Least Privilege

**Minimal Container Permissions**
- Containers run with the minimum required privileges and capabilities necessary for model execution
- All capabilities dropped by default (`cap_drop: ALL`)
- No new privileges allowed (`no-new-privileges:true`)

**Network Access Restrictions**
- Outbound network access limited to essential services and destinations
- Localhost-only port binding prevents external access
- Custom bridge networking for controlled communication

**File System Access Controls**
- Read-only access to model files with write access only to designated temporary directories
- Read-only root filesystems (`read_only: true`)
- Selective volume mounting for necessary directories

## Implementation Guidelines

### Current Deployment Architecture

**Multi-Container CPU Strategy**
Three dedicated CPU containers leverage the 16-core AMD Ryzen 9950X:
- **Parallel inference**: Each container handles different models or requests simultaneously
- **Resource isolation**: 8 cores and 32GB RAM per container with cpuset pinning
- **AMD optimization**: AOCL libraries and Zen 5 compiler flags for maximum performance

**Dual GPU Strategy**
Two GPU containers optimize RTX 5090 utilization:
- **llama-gpu**: llama.cpp with CUDA optimization for GGUF models
- **vllm-gpu**: vLLM engine for high-throughput transformer serving
- **Blackwell architecture**: CUDA 12.9.1 with sm_120 targeting

**Development to Production**
Docker Compose configuration supports:
```yaml
networks:
  ai-network:
    internal: false  # Set to true in production for complete isolation
```
Allows image pulling during development while supporting production hardening.

### Production Deployment

**Container Build Process**
Multi-stage Dockerfiles optimize security and performance:
- **Builder stage**: Compiles optimized binaries with security hardening
- **Runtime stage**: Minimal runtime environment with non-root users
- **AMD AOCL integration**: Specialized CPU container uses pre-installed AOCL libraries

**Security Validation**
Docker Compose implements defense-in-depth:
```yaml
security_opt:
  - no-new-privileges:true
cap_drop:
  - ALL
read_only: true
```
Multiple security layers prevent privilege escalation and system compromise.

**Volume Security**
Read-only model mounts prevent tampering:
```yaml
volumes:
  - /mnt/ai-data/models/:/app/models:ro
  - ./logs/cpu:/app/logs
```
Separates read-only model access from writable log directories.


## Reference Files

### Docker Configuration Files

- **docker/Dockerfile.llama-cpu**: AMD Zen 5 optimized CPU inference with AOCL libraries
- **docker/Dockerfile.llama-gpu**: RTX 5090 optimized GPU inference with CUDA 12.9.1
- **docker/Dockerfile.vllm-gpu**: vLLM high-performance transformer serving
- **docker-compose.yaml**: Complete orchestration with security hardening
- **docker/requirements-cpu.txt**: Python dependencies for CPU containers
- **docker/aocl-linux-gcc-5.1.0_1_amd64.deb**: AMD Optimized CPU Libraries package

### Security Features Implemented

- **Process isolation**: Non-root users, capability dropping, privilege restrictions
- **Resource constraints**: CPU pinning, memory limits, tmpfs controls
- **Network security**: Localhost binding, custom bridge networking, internal communication
- **File system security**: Read-only containers, selective volume mounts, log isolation
- **Health monitoring**: Comprehensive health checks, automatic restarts, failure recovery

---

*This sandboxing architecture provides comprehensive security for AI model deployment on the AMD Ryzen 9950X + RTX 5090 workstation. Security configurations prioritize protection over performance and reflect the actual implementation in the Docker containers and compose orchestration.*

*Last Updated: 2025-09-23*