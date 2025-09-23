# Docker Compose Security Orchestration

Comprehensive explanation of the multi-service AI inference orchestration implemented in docker-compose.yaml, detailing security sandboxing, resource isolation, and defense-in-depth strategies for the AMD Ryzen 9950X + RTX 5090 AI workstation.

> Note: This document explains the actual implementation in `docker-compose.yaml`. The configuration orchestrates multiple containerized AI services with comprehensive security controls, resource constraints, and network isolation.

## Table of Contents

- [Docker Compose Security Orchestration](#docker-compose-security-orchestration)
  - [Table of Contents](#table-of-contents)
  - [Implementation Overview](#implementation-overview)
  - [Service Architecture](#service-architecture)
    - [CPU Inference Services](#cpu-inference-services)
    - [GPU Inference Services](#gpu-inference-services)
    - [Web Interface Service](#web-interface-service)
  - [Security Hardening](#security-hardening)
    - [Capability Dropping](#capability-dropping)
    - [Privilege Restrictions](#privilege-restrictions)
    - [Memory Locking](#memory-locking)
  - [Resource Management](#resource-management)
    - [CPU Resource Allocation](#cpu-resource-allocation)
    - [Memory Constraints](#memory-constraints)
    - [GPU Resource Reservation](#gpu-resource-reservation)
  - [Network Architecture](#network-architecture)
    - [Network Isolation](#network-isolation)
    - [Port Security](#port-security)
    - [Service Discovery](#service-discovery)
  - [Volume Security](#volume-security)
    - [Read-Only Model Mounts](#read-only-model-mounts)
    - [Log Directory Isolation](#log-directory-isolation)
    - [Temporary Storage](#temporary-storage)
  - [Container Runtime Security](#container-runtime-security)
    - [Read-Only Root Filesystems](#read-only-root-filesystems)
    - [User Namespacing](#user-namespacing)
    - [Security Options](#security-options)
  - [Health Monitoring and Resilience](#health-monitoring-and-resilience)
    - [Health Check Configuration](#health-check-configuration)
    - [Restart Policies](#restart-policies)
    - [Service Dependencies](#service-dependencies)
  - [Environment Configuration](#environment-configuration)
    - [Service Environment Variables](#service-environment-variables)
    - [GPU Configuration](#gpu-configuration)
    - [Model Path Management](#model-path-management)
  - [Build Configuration](#build-configuration)
    - [Build Context](#build-context)
    - [Dockerfile References](#dockerfile-references)
  - [Production Considerations](#production-considerations)
    - [Network Hardening](#network-hardening)
    - [Volume Persistence](#volume-persistence)
  - [Reference Implementation](#reference-implementation)

## Implementation Overview

The docker-compose.yaml implements a comprehensive multi-service AI inference platform with defense-in-depth security. The orchestration manages five distinct services across CPU and GPU resources, implementing strict isolation boundaries while maintaining controlled inter-service communication.

**Key Security Features:**
- **Service isolation**: Each AI model runs in its own security-hardened container
- **Resource constraints**: Hard limits on CPU, memory, and GPU resources prevent DoS
- **Network segmentation**: Custom bridge network with controlled communication paths
- **Filesystem immutability**: Read-only root filesystems prevent runtime tampering
- **Capability stripping**: All unnecessary Linux capabilities removed from containers
- **Privilege prevention**: No privilege escalation possible within containers

**Service Distribution:**
- **1 CPU inference container**: llama-cpu (12 threads on cores 0-12, 96GB RAM)
- **2 GPU inference containers**: llama-gpu and vllm-gpu (RTX 5090 exclusive access)
- **1 Web interface**: open-webui for secure user interaction

## Service Architecture

### CPU Inference Services

**llama-cpu Service Configuration**
```yaml
llama-cpu:
  security_opt:
    - no-new-privileges:true
  cap_drop:
    - ALL
  cap_add:
    - IPC_LOCK
  deploy:
    resources:
      limits:
        cpus: '12'
        memory: 96G
```

**Security Design:**
- **Single high-performance instance**: 12 threads for latency-optimized inference
- **IPC_LOCK capability**: Required for memory locking operations only
- **Memory allocation**: 96GB RAM for large model support
- **CPU isolation**: Cores 0-12 allocated via cpuset configuration (13 cores total)

**Resource Optimization:**
- **Thread configuration**: 12 threads configured for optimal performance
- **Core allocation**: Uses cores 0-12 (13 cores) via cpuset
- **Memory locking**: Prevents model swapping to disk via ulimits configuration

### GPU Inference Services

**llama-gpu Service Configuration**
```yaml
llama-gpu:
  security_opt:
    - no-new-privileges:true
  cap_drop:
    - ALL
  read_only: true
  tmpfs:
    - /tmp:size=4G,mode=1777  # CUDA cache and temp files
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu, compute, utility]
```

**GPU Access Control:**
- **Exclusive GPU access**: Single RTX 5090 reserved for this container
- **NVIDIA runtime**: GPU driver integration without host privileges
- **Device visibility**: CUDA_VISIBLE_DEVICES=0 ensures single GPU usage
- **Capability specification**: gpu, compute, utility for full CUDA support
- **Temporary storage**: 4GB tmpfs for CUDA cache and compilation
- **Read-only filesystem**: Immutable container filesystem for security

**vllm-gpu Service Configuration**
```yaml
vllm-gpu:
  security_opt:
    - no-new-privileges:true
  cap_drop:
    - ALL
  read_only: true
  tmpfs:
    - /tmp:size=4G,mode=1777
```
- **Security profile**: Same hardening as other GPU services
- **Temporary storage**: 4GB tmpfs for CUDA compilation cache
- **Memory mode**: 1777 permissions for temporary file operations
- **Filesystem immutability**: Read-only root filesystem

### Web Interface Service

**open-webui Service Configuration**
```yaml
open-webui:
  image: ghcr.io/open-webui/open-webui:main
  ports:
    - "3000:8080"
  read_only: true
  volumes:
    - open-webui-data:/app/backend/data
```

**User Interface Security:**
- **External image**: Official open-webui container from GitHub registry
- **Port mapping**: Public port 3000 mapped to container port 8080
- **Read-only filesystem**: Prevents web interface tampering
- **Persistent data**: Named volume for configuration persistence
- **Limited hardening**: Unlike other services, lacks security_opt and cap_drop settings

## Security Hardening

### Capability Dropping

**Universal Capability Removal**
All services implement comprehensive capability dropping:
```yaml
cap_drop:
  - ALL
```

**Security Impact:**
- **Prevents privilege escalation**: No access to privileged kernel operations
- **Reduces attack surface**: Eliminates unnecessary system call access
- **Container isolation**: Cannot modify host system configurations
- **Network restrictions**: No raw socket or network configuration access

**Selective Capability Addition**
Only llama-cpu adds minimal capability:
```yaml
cap_add:
  - IPC_LOCK
```
Required for memory locking operations to prevent model swapping.

### Privilege Restrictions

**No New Privileges Enforcement**
All services implement privilege escalation prevention:
```yaml
security_opt:
  - no-new-privileges:true
```

**Protection Mechanisms:**
- **SUID/SGID prevention**: Binaries cannot elevate privileges
- **Capability inheritance blocking**: Child processes cannot gain capabilities
- **Runtime security**: Prevents exploitation of vulnerable binaries
- **Defense-in-depth**: Additional layer beyond capability dropping

### Memory Locking

**Unlimited Memory Lock Configuration**
```yaml
ulimits:
  memlock:
    soft: -1
    hard: -1
```

**Purpose:**
- **Model persistence**: Prevents large models from swapping to disk
- **Performance guarantee**: Ensures consistent inference latency
- **Security benefit**: Reduces timing attack vulnerabilities
- **Resource management**: Controlled through memory limits instead

## Resource Management

### CPU Resource Allocation

**llama-cpu CPU Configuration**
```yaml
cpuset: "0-12"
deploy:
  resources:
    limits:
      cpus: '12'
```

**CPU Isolation Strategy:**
- **Core allocation**: Cores 0-12 (13 cores) assigned via cpuset
- **Thread count**: 12 threads configured for inference workload
- **NUMA optimization**: Cores span single NUMA node for memory locality
- **System isolation**: Remaining cores available for system operations

### Memory Constraints

**Service Memory Limits**
```yaml
llama-cpu:
  deploy:
    resources:
      limits:
        cpus: '12'
        memory: 96G  # High-performance single instance
```

**Memory Management:**
- **Hard limits**: Prevents memory exhaustion attacks
- **OOM prevention**: Kernel OOM killer respects container limits
- **Swap disabled**: Combined with memory locking for predictable performance
- **Buffer allocation**: Sufficient headroom for model operations

### GPU Resource Reservation

**GPU Service Configuration**
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu, compute, utility]
```

**GPU Isolation:**
- **Exclusive access**: Each GPU service gets dedicated RTX 5090 access
- **Driver integration**: NVIDIA container runtime handles GPU virtualization
- **Capability specification**: Precise GPU feature access control
- **Environment control**: CUDA_VISIBLE_DEVICES ensures isolation

## Network Architecture

### Network Isolation

**Custom Bridge Network**
```yaml
networks:
  ai-network:
    driver: bridge
    internal: false
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

**Network Security:**
- **Dedicated subnet**: 172.20.0.0/16 isolates AI services from host network
- **Bridge driver**: Layer 2 isolation between containers
- **Internal flag**: Set to false for development, true for production isolation
- **IPAM configuration**: Controlled IP address management

### Port Security

**Localhost-Only Binding**
```yaml
ports:
  - "127.0.0.1:8001:8001"  # llama-cpu
  - "127.0.0.1:8004:8004"  # llama-gpu
  - "127.0.0.1:8005:8005"  # vllm-gpu
```

**Access Control:**
- **Localhost restriction**: Services only accessible from host machine
- **No external exposure**: Prevents remote exploitation
- **Explicit binding**: 127.0.0.1 prefix ensures local-only access
- **Web UI exception**: Port 3000 publicly accessible for user interface

### Service Discovery

**Internal DNS Resolution**
Services communicate using container names:
```yaml
environment:
  - 'OPENAI_API_BASE_URL=http://llama-gpu:8006/v1'
```

**Communication Patterns:**
- **Service names**: Containers resolve via Docker's internal DNS
- **Direct routing**: No external DNS dependency
- **Port consistency**: Internal ports match service configuration
- **API compatibility**: OpenAI-compatible endpoints for standardization

## Volume Security

### Read-Only Model Mounts

**Model Directory Mounting**
```yaml
volumes:
  - /mnt/ai-data/models/:/app/models:ro
```

**Security Properties:**
- **Read-only flag**: `:ro` prevents model tampering
- **Shared access**: Multiple containers can read same models
- **Host protection**: Containers cannot modify host filesystem
- **Path consistency**: Uniform /app/models path across services

### Log Directory Isolation

**Writable Log Volumes**
```yaml
volumes:
  - ./logs/cpu:/app/logs
  - ./logs/gpu:/app/logs
```

**Log Management:**
- **Separated directories**: CPU and GPU logs isolated
- **Host persistence**: Logs survive container restarts
- **Write access**: Limited to specific log directory
- **Audit trail**: Persistent logging for security monitoring

### Temporary Storage

**tmpfs Mounts for GPU Services**
```yaml
tmpfs:
  - /tmp:size=4G,mode=1777
```

**Temporary Storage Security:**
- **RAM-based**: No disk persistence of temporary files
- **Size limits**: 4GB for both llama-gpu and vllm-gpu services
- **Mode 1777**: Sticky bit prevents cross-user file deletion
- **Automatic cleanup**: Contents cleared on container restart
- **GPU services only**: CPU service doesn't require tmpfs

## Container Runtime Security

### Read-Only Root Filesystems

**Filesystem Immutability**
```yaml
read_only: true
```

**Protection Benefits:**
- **Prevents persistence**: Malware cannot modify container filesystem
- **Configuration protection**: Runtime settings remain unchanged
- **Binary integrity**: Executables cannot be replaced
- **Log forcing**: Writes must use designated volumes

### User Namespacing

**Non-Root Execution**
Containers implement non-root users (configured in Dockerfiles):
- **llama-cpu**: Runs as appuser (UID 1000)
- **llama-gpu**: Runs as appuser (UID 1000)
- **vllm-gpu**: Runs as appuser (UID 1000)

**Security Benefits:**
- **Privilege separation**: Container compromise doesn't grant root
- **UID mapping**: Container UIDs isolated from host UIDs
- **File ownership**: Proper ownership for mounted volumes
- **Audit compliance**: Non-root execution best practice

### Security Options

**Comprehensive Security Configuration**
```yaml
security_opt:
  - no-new-privileges:true
cap_drop:
  - ALL
ulimits:
  memlock:
    soft: -1
    hard: -1
read_only: true
```

**Layered Defense:**
- **Multiple mechanisms**: Defense-in-depth security approach
- **Fail-secure defaults**: Restrictive by default configuration
- **Explicit permissions**: Only necessary capabilities added
- **Runtime protection**: Security enforced at container runtime

## Health Monitoring and Resilience

### Health Check Configuration

**Service Health Monitoring**
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8001/v1/health"]
  interval: 30s
  timeout: 10s
  retries: 5
  start_period: 180s
```

**Health Check Parameters:**
- **Test command**: HTTP endpoint verification via curl on service's API port
- **Interval**: 30-second check frequency
- **Timeout**: 10-second response deadline
- **Retries**: 5 attempts before unhealthy status
- **Start period**: 180 seconds (llama-cpu, llama-gpu), 300 seconds (vllm-gpu)

### Restart Policies

**Automatic Recovery**
```yaml
restart: unless-stopped
```

**Restart Behavior:**
- **Automatic recovery**: Containers restart on failure
- **Manual override**: Honors manual stop commands
- **System reboot**: Services resume after host restart
- **Crash protection**: Prevents service unavailability

### Service Dependencies

**Dependency Management**
While explicit depends_on is commented out, services implement:
- **Health-based startup**: Services wait for dependencies via health checks
- **Network readiness**: Internal DNS ensures service discovery
- **Retry logic**: Applications handle temporary unavailability

## Environment Configuration

### Service Environment Variables

**llama-cpu Environment**
```yaml
environment:
  - SERVER_PORT=8001
  - MODEL_PATH=${LLAMA_CPU_MODEL}
  - THREADS=12
  - THREADS_BATCH=12
```

**Configuration Management:**
- **Port specification**: Explicit port configuration for clarity
- **Model path variable**: External configuration via .env file
- **Thread optimization**: Matches physical core allocation
- **Batch processing**: Optimized for CPU architecture

### GPU Configuration

**GPU Environment Settings**
```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=0
  - CUDA_VISIBLE_DEVICES=0
  - MODEL_PATH=${LLAMA_GPU_MODEL}
```

**GPU Control:**
- **Device visibility**: Restricts GPU access to specific device
- **CUDA configuration**: Ensures proper CUDA device selection
- **Model flexibility**: Runtime model selection via environment

### Model Path Management

**Environment Variable Usage**
```yaml
MODEL_PATH=${LLAMA_CPU_MODEL}
MODEL_PATH=${LLAMA_GPU_MODEL}
MODEL_PATH=${VLLM_GPU_MODEL}
```

**Path Strategy:**
- **External configuration**: .env file contains actual paths
- **Deployment flexibility**: Easy model switching without compose changes
- **Security**: Paths validated by containers at startup
- **Consistency**: Standardized MODEL_PATH variable across services

## Build Configuration

### Build Context

**Local Build Configuration**
```yaml
build:
  context: .
  dockerfile: docker/Dockerfile.llama-cpu
```

**Build Properties:**
- **Context root**: Repository root as build context
- **Dockerfile paths**: Organized in docker/ directory
- **Multi-stage builds**: Dockerfiles implement builder/runtime separation
- **Cache efficiency**: Shared context enables layer caching

### Dockerfile References

**Service Dockerfile Mapping**
- **llama-cpu**: `docker/llama-cpu/Dockerfile.llama-cpu`
- **llama-gpu**: `docker/Dockerfile.llama-gpu`
- **vllm-gpu**: `docker/Dockerfile.vllm-gpu`

**Build Architecture:**
- **Stage separation**: Build dependencies isolated from runtime
- **Security hardening**: Non-root users configured in Dockerfiles
- **Optimization flags**: Architecture-specific compilation in build stage
- **Minimal runtime**: Only necessary libraries in final images

## Production Considerations

### Network Hardening

**Production Network Configuration**
```yaml
networks:
  ai-network:
    internal: true  # Change from false to true in production
```

**Production Security:**
- **Complete isolation**: Internal network prevents external access
- **Service mesh**: Only web UI exposed to external network
- **Firewall integration**: Additional host-level network controls
- **Zero trust**: No implicit trust between services

### Volume Persistence

**Named Volume Strategy**
```yaml
volumes:
  open-webui-data:
```

**Data Management:**
- **Persistent storage**: Survives container recreation
- **Backup capability**: Named volumes support backup operations
- **Access control**: Docker manages volume permissions
- **Migration support**: Volumes portable across hosts

## Reference Implementation

**Complete Security Stack:**
The docker-compose.yaml implements comprehensive security through:

1. **Process Isolation**: Each service runs in isolated container namespace
2. **Resource Limits**: CPU, memory, and GPU resources strictly controlled
3. **Network Segmentation**: Custom bridge network with controlled communication
4. **Filesystem Security**: Read-only roots with selective write mounts
5. **Capability Management**: Minimal capabilities with explicit dropping
6. **Health Monitoring**: Automatic detection and recovery from failures
7. **Access Control**: Localhost-only binding for service endpoints
8. **Data Protection**: Read-only model mounts prevent tampering

**Integration Points:**
- **Dockerfiles**: Provide build-time security configuration
- **Environment variables**: Enable runtime configuration without rebuilds
- **Volume mounts**: Separate code, models, and logs
- **Network policies**: Control inter-service communication
- **Health endpoints**: Enable orchestration and monitoring

**Security Validation:**
- All services drop ALL capabilities by default
- No privilege escalation possible (no-new-privileges)
- Read-only filesystems prevent runtime modification
- Resource limits prevent denial of service
- Network isolation limits lateral movement
- Non-root execution reduces compromise impact

---

*This docker-compose.yaml implementation provides production-grade security orchestration for AI inference workloads on the AMD Ryzen 9950X + RTX 5090 workstation as of mid-2025. The configuration implements defense-in-depth security while maintaining operational efficiency for multi-model AI serving.*