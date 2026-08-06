# Docker Compose Security Orchestration

Explanation of the multi-service AI inference orchestration implemented in docker-compose.inference.yaml, detailing security sandboxing, resource isolation, and defense-in-depth strategies for the AMD Ryzen 9950X + RTX 5090 AI experiments.

> Note: This document explains the actual implementation in `docker-compose.inference.yaml`. The configuration orchestrates multiple containerized AI services with comprehensive security controls, resource constraints, and network isolation.

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
  - [Reference Implementation](#reference-implementation)

## Implementation Overview

The docker-compose.inference.yaml implements a multi-service AI inference platform with defense-in-depth security. The orchestration manages five distinct services across CPU and GPU resources, implementing strict isolation boundaries while maintaining controlled inter-service communication.

**Key Security Features:**
- **Service isolation**: Each AI model runs in its own security-hardened container
- **Resource constraints**: Hard limits on CPU, memory, and GPU resources prevent DoS
- **Network segmentation**: Custom bridge network with controlled communication paths
- **Filesystem immutability**: Read-only root filesystems prevent runtime tampering
- **Capability stripping**: All unnecessary Linux capabilities removed from containers
- **Privilege prevention**: No privilege escalation possible within containers

**Service Distribution:**
- **1 CPU inference container**: llama-cpu (16 threads on cores 0-15, 96GB RAM)
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
  deploy:
    resources:
      limits:
        memory: 96G
```

**Security Design:**
- **Single high-performance instance**: 16 threads for latency-optimized inference
- **Memory allocation**: 96GB RAM for large model support
- **CPU isolation**: Cores 0-15 allocated via cpuset configuration (16 cores total)

**Resource Optimization:**
- **Thread configuration**: 16 threads configured for optimal performance
- **Core allocation**: Uses cores 0-15 (16 cores) via cpuset
- **Memory locking**: Handled via `--mlock` flag in server configuration

### GPU Inference Services

**llama-gpu Service Configuration**
```yaml
llama-gpu:
  init: true  # Use tini as PID 1 for proper signal forwarding
  user: "${UID:-1000}:${GID:-1000}"  # Match host user for volume write access
  security_opt:
    - no-new-privileges:true
  cap_drop:
    - ALL
  read_only: true
  tmpfs:
    - /tmp:size=4G,mode=1777  # CUDA cache and temp files
  environment:
    - NSYS_PROFILE=${NSYS_PROFILE:-false}
    - NSYS_OUTPUT=${NSYS_OUTPUT:-/app/nsys_output/profile}
  volumes:
    - ./gpu_benchmark_results:/app/nsys_output
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
- **Process manager**: tini (init: true) ensures proper signal handling for Nsight profiling
- **User mapping**: Host UID/GID for nsys output file permissions
- **Nsight profiling**: NSYS_PROFILE and NSYS_OUTPUT for GPU performance analysis

**vllm-gpu Service Configuration**
```yaml
vllm-gpu:
  init: true  # Use tini as PID 1 for proper signal forwarding
  security_opt:
    - no-new-privileges:true
  cap_drop:
    - ALL
  read_only: true
  tmpfs:
    - /tmp:size=4G,mode=1777           # CUDA cache and temp files
    - /app/.cache:size=1G,mode=1777    # vLLM model info cache
    - /app/cache:size=2G,mode=1777     # HuggingFace cache (HF_HOME)
    - /app/.config:size=100M,mode=1777 # vLLM usage stats
    - /app/.triton:size=2G,mode=1777,exec  # Triton compilation cache
  volumes:
    - ./logs/vllm:/app/logs
```
- **Security profile**: Same hardening as other GPU services
- **Process manager**: tini (init: true) for proper signal handling
- **Temporary storage**: Multiple tmpfs mounts for various caches:
  - `/tmp`: 4GB for CUDA temporary files
  - `/app/.cache`: 1GB for vLLM model info
  - `/app/cache`: 2GB for HuggingFace downloads
  - `/app/.config`: 100MB for vLLM configuration
  - `/app/.triton`: 2GB with exec for Triton JIT compilation
- **Log isolation**: Separate `./logs/vllm` directory for vLLM logs
- **Filesystem immutability**: Read-only root filesystem

### Web Interface Service

**open-webui Service Configuration**
```yaml
open-webui:
  image: ghcr.io/open-webui/open-webui:latest@sha256:a26effeb...
  security_opt:
    - no-new-privileges:true
  cap_drop:
    - ALL
  ports:
    - "127.0.0.1:3000:8080"
  read_only: true
  volumes:
    - open-webui-data:/app/backend/data
```

**User Interface Security:**
- **External image**: Official open-webui container, pinned to the stable release by digest
- **Port mapping**: Bound to 127.0.0.1 only; container port 8080 exposed as localhost:3000
- **Read-only filesystem**: Prevents web interface tampering
- **Persistent data**: Named volume for configuration persistence; open-webui auto-generates
  and persists its `WEBUI_SECRET_KEY` here (no literal key set in compose)
- **Hardening**: Runs with `no-new-privileges` and `cap_drop: ALL`, matching the other services

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

Memory locking is handled at the application level via the `--mlock` flag in the llama.cpp server entrypoint, rather than through container ulimits. This approach:
- **Model persistence**: Prevents large models from swapping to disk
- **Performance guarantee**: Ensures consistent inference latency
- **Simplified configuration**: No container-level ulimit configuration required

## Resource Management

### CPU Resource Allocation

**llama-cpu CPU Configuration**
```yaml
cpuset: "0-15"
deploy:
  resources:
    limits:
      memory: 96G
```

**CPU Isolation Strategy:**
- **Core allocation**: Cores 0-15 (16 cores) assigned via cpuset
- **Thread count**: 16 threads configured for inference workload
- **NUMA optimization**: Cores span single NUMA node for memory locality
- **System isolation**: Remaining cores available for system operations

### Memory Constraints

**Service Memory Limits**
```yaml
llama-cpu:
  deploy:
    resources:
      limits:
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
- **Internal flag**: Set to false to allow Docker image pulling
- **IPAM configuration**: Controlled IP address management

### Port Security

**Localhost-Only Binding**
```yaml
ports:
  - "127.0.0.1:8001:8001"  # llama-cpu
  - "127.0.0.1:8002:8002"  # llama-gpu
  - "127.0.0.1:8003:8003"  # vllm-gpu
  - "127.0.0.1:3000:8080"  # open-webui
```

**Access Control:**
- **Localhost restriction**: All services, including the web UI, bind to 127.0.0.1 only
- **No external exposure**: Prevents remote exploitation
- **Explicit binding**: 127.0.0.1 prefix ensures local-only access

### Service Discovery

**Internal DNS Resolution**
Services communicate using container names:
```yaml
environment:
  - 'OPENAI_API_BASE_URL=http://llama-gpu:8002/v1'
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
  - ./logs/cpu:/app/logs    # llama-cpu
  - ./logs/gpu:/app/logs    # llama-gpu
  - ./logs/vllm:/app/logs   # vllm-gpu
```

**Log Management:**
- **Separated directories**: CPU, GPU llama, and vLLM logs isolated
- **Host persistence**: Logs survive container restarts
- **Write access**: Limited to specific log directory
- **Audit trail**: Persistent logging for security monitoring

**Additional llama-cpu Volumes**
```yaml
volumes:
  - ./datasets:/app/datasets  # Dataset storage
  - /sys:/sys:ro              # Read-only sysfs access
```
- **Dataset access**: Writable directory for benchmark datasets
- **System info**: Read-only access to kernel sysfs for hardware information

### Temporary Storage

**tmpfs Mounts for GPU Services**

llama-gpu uses a single tmpfs:
```yaml
tmpfs:
  - /tmp:size=4G,mode=1777
```

vllm-gpu requires multiple tmpfs mounts for caches:
```yaml
tmpfs:
  - /tmp:size=4G,mode=1777
  - /app/.cache:size=1G,mode=1777
  - /app/cache:size=2G,mode=1777
  - /app/.config:size=100M,mode=1777
  - /app/.triton:size=2G,mode=1777,exec
```

**Temporary Storage Security:**
- **RAM-based**: No disk persistence of temporary files
- **Size limits**: Appropriate limits per cache type
- **Mode 1777**: Sticky bit prevents cross-user file deletion
- **Automatic cleanup**: Contents cleared on container restart
- **Triton exec**: Executable permission for JIT-compiled kernels

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
- **llama-cpu**: Runs as appuser
- **llama-gpu**: Runs as appuser
- **vllm-gpu**: Runs as appuser

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
  test: ["CMD", "curl", "-f", "http://localhost:PORT/v1/health"]  # llama-cpu: 8001, llama-gpu: 8002
  # vllm-gpu uses: ["CMD", "curl", "-f", "http://localhost:8003/health"]
  interval: 30s
  timeout: 10s
  retries: 5
  start_period: 180s  # 300s for vllm-gpu
```

**Health Check Parameters:**
- **Test command**: HTTP endpoint verification via curl on service's API port
- **Port mapping**: llama-cpu (8001), llama-gpu (8002), vllm-gpu (8003)
- **Endpoint paths**: llama services use /v1/health, vllm-gpu uses /health
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
  - MODEL_PATH=${LLAMA_MODEL:-${LLAMA_ZEN5_MODEL}}
  - THREADS=16
  - THREADS_BATCH=16
```

**Configuration Management:**
- **Port specification**: Explicit port configuration for clarity
- **Model path variable**: External configuration via .env file with LLAMA_ZEN5_MODEL fallback
- **Thread optimization**: Matches physical core allocation (16 cores)
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
# llama-cpu
MODEL_PATH=${LLAMA_MODEL:-${LLAMA_ZEN5_MODEL}}

# llama-gpu
MODEL_PATH=${LLAMA_GPU_MODEL}

# vllm-gpu
MODEL_PATH=${VLLM_GPU_MODEL}
```

**Path Strategy:**
- **External configuration**: .env file contains actual paths
- **Fallback support**: llama-cpu uses LLAMA_ZEN5_MODEL if LLAMA_MODEL unset
- **Deployment flexibility**: Easy model switching without compose changes
- **Security**: Paths validated by containers at startup
- **Consistency**: Standardized MODEL_PATH variable across services

## Build Configuration

### Build Context

**Local Build Configuration**
```yaml
build:
  context: .
  dockerfile: docker/llama-cpu/Dockerfile.cpu
```

**Build Properties:**
- **Context root**: Repository root as build context
- **Dockerfile paths**: Organized in docker/ directory
- **Multi-stage builds**: Dockerfiles implement builder/runtime separation
- **Cache efficiency**: Shared context enables layer caching

### Dockerfile References

**Service Dockerfile Mapping**
- **llama-cpu**: `docker/llama-cpu/Dockerfile.cpu`
- **llama-gpu**: `docker/llama-gpu/Dockerfile.llama-gpu`
- **vllm-gpu**: `docker/vllm-gpu/Dockerfile.vllm-gpu`

**Build Architecture:**
- **Stage separation**: Build dependencies isolated from runtime
- **Security hardening**: Non-root users configured in Dockerfiles
- **Optimization flags**: Architecture-specific compilation in build stage
- **Minimal runtime**: Only necessary libraries in final images


## Reference Implementation

**Complete Security Stack:**
The docker-compose.inference.yaml implements security through:

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

*Last Updated: 2026-01-03*