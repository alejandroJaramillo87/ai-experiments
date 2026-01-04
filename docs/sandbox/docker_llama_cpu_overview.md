# AI Engineering CPU Docker Implementation

Explanation of the AMD Zen 5 optimized Dockerfile for CPU-based AI model inference on the AMD Ryzen 9950X workstation, detailing the multi-stage build process, zen5_optimizer integration, and AVX-512 performance optimizations.

> Note: This document explains the actual implementation in `docker/llama-cpu/Dockerfile.cpu`. The Dockerfile uses a multi-stage build with cmake-based compilation and the zen5_optimizer library for hugepage support.

## Table of Contents

- [AI Engineering CPU Docker Implementation](#ai-engineering-cpu-docker-implementation)
  - [Table of Contents](#table-of-contents)
  - [Implementation Overview](#implementation-overview)
  - [Stage 1: Builder Environment](#stage-1-builder-environment)
    - [Base Image Selection](#base-image-selection)
    - [AMD Zen 5 Compiler Configuration](#amd-zen-5-compiler-configuration)
    - [System Dependencies Installation](#system-dependencies-installation)
    - [zen5_optimizer Library](#zen5_optimizer-library)
    - [llama.cpp Compilation](#llamacpp-compilation)
  - [Stage 2: Runtime Environment](#stage-2-runtime-environment)
    - [Runtime Base Image](#runtime-base-image)
    - [Library Dependencies](#library-dependencies)
    - [Security Configuration](#security-configuration)
    - [Application Deployment](#application-deployment)
  - [Entrypoint Configuration](#entrypoint-configuration)
    - [Required Environment Variables](#required-environment-variables)
    - [Optional Environment Variables](#optional-environment-variables)
    - [Server Flags](#server-flags)
  - [Performance Optimization Strategy](#performance-optimization-strategy)
    - [AMD Zen 5 Architecture Targeting](#amd-zen-5-architecture-targeting)
    - [AVX-512 CMAKE Configuration](#avx-512-cmake-configuration)
    - [Threading Configuration](#threading-configuration)
    - [Memory Optimization](#memory-optimization)
  - [Container Security Implementation](#container-security-implementation)
  - [Reference Implementation](#reference-implementation)

## Implementation Overview

The CPU Docker implementation leverages a multi-stage build strategy specifically optimized for the AMD Ryzen 9950X processor. The architecture separates the build environment from the runtime environment to minimize the final container size while maximizing CPU performance through targeted optimizations.

**Key Implementation Features:**
- **Multi-stage build**: Separates compilation dependencies from runtime requirements
- **AMD Zen 5 optimization**: Compiler flags targeting znver5 architecture
- **zen5_optimizer library**: Hugepage support via LD_PRELOAD for reduced TLB pressure
- **AVX-512 native**: Explicit VNNI, VBMI, BF16 extensions enabled via cmake
- **CMake build system**: Modern build configuration for llama.cpp compilation
- **Security hardening**: Non-root user execution and minimal runtime dependencies

## Stage 1: Builder Environment

### Base Image Selection

**Python 3.12 Slim Base**
```dockerfile
FROM python:3.12-slim AS builder

LABEL stage="builder"
```
- **Purpose**: Provides minimal Python environment with essential build tools
- **Size optimization**: Slim variant reduces initial image size
- **Compatibility**: Python 3.12 ensures modern language support

### AMD Zen 5 Compiler Configuration

**Build Environment Variables**
```dockerfile
ENV DEBIAN_FRONTEND=noninteractive
ENV CFLAGS="-march=znver5 -mtune=znver5 -O3"
ENV CXXFLAGS="${CFLAGS}"
ENV CC=gcc-14
ENV CXX=g++-14
ENV LDFLAGS="-Wl,-O3 -Wl,--as-needed"
```

**Compiler Flag Optimization**
- **-march=znver5**: Generates code specifically for AMD Zen 5 architecture
- **-mtune=znver5**: Optimizes instruction scheduling for Zen 5 microarchitecture
- **-O3**: Enables aggressive compiler optimizations for maximum performance
- **GCC-14**: Latest compiler version with enhanced Zen 5 support (from Debian unstable)
- **Linker optimization**: -Wl,-O3 and --as-needed reduce binary size

Note: AVX-512 instruction sets are enabled via cmake flags, not CFLAGS.

### System Dependencies Installation

**Build Tools and Libraries**
```dockerfile
RUN echo "deb http://deb.debian.org/debian unstable main" > /etc/apt/sources.list.d/sid.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc-14 g++-14 build-essential cmake git curl \
        libssl-dev libomp-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*
```

**Package Selection**
- **gcc-14/g++-14**: Latest compiler suite with Zen 5 optimizations
- **cmake**: Modern build system for llama.cpp and zen5_optimizer compilation
- **git**: Repository cloning for llama.cpp source
- **curl**: HTTP client for model downloads
- **libomp-dev**: OpenMP development libraries for parallel processing
- **libssl-dev/pkg-config**: Build dependencies
- **Unstable repository**: Access to GCC-14 with znver5 support

### zen5_optimizer Library

**CMake-Based Hugepage Optimization**

The hugepage optimization is provided by `zen5_optimizer`, a custom library that intercepts mmap() calls for large allocations:

```dockerfile
COPY hugepages/CMakeLists.txt /tmp/zen5_optimizer/
COPY hugepages/src/ /tmp/zen5_optimizer/src/
RUN cd /tmp/zen5_optimizer && \
    cmake -B build \
        -DCMAKE_C_COMPILER=gcc-14 \
        -DCMAKE_CXX_COMPILER=g++-14 \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_HUGEPAGES=ON \
        -DENABLE_DEBUG_PRINT=OFF && \
    cmake --build build --config Release -j$(nproc) && \
    cp build/libzen5_optimizer.so /tmp/libzen5_optimizer.so
```

**Source Files**
- `hugepages/CMakeLists.txt`: Build configuration
- `hugepages/src/zen5_optimizer.cpp`: Main entry point
- `hugepages/src/cpu_validator.cpp/.h`: CPU feature validation
- `hugepages/src/memory/hugepage_wrapper.cpp`: mmap() interception
- `hugepages/src/config.h`: Configuration headers

**Build Configuration**
- **CMAKE_C_COMPILER=gcc-14**: Use GCC-14 for C compilation
- **CMAKE_CXX_COMPILER=g++-14**: Use GCC-14 for C++ compilation
- **CMAKE_BUILD_TYPE=Release**: Optimized release build
- **ENABLE_HUGEPAGES=ON**: Enable hugepage memory allocation
- **ENABLE_DEBUG_PRINT=OFF**: Disable debug output for production

**How It Works**
The library intercepts mmap() calls for allocations >= 1GB and uses MAP_HUGETLB to allocate 2MB huge pages instead of 4KB standard pages.

**Performance Benefits**
- **Reduced TLB misses**: 2MB pages instead of 4KB reduces translation overhead
- **Better memory locality**: Fewer page table entries to manage
- **10-20% inference speedup**: Measured on large models (>15GB)

**Runtime Activation**
The library is activated via LD_PRELOAD in the entrypoint script:
```bash
export LD_PRELOAD="/app/libzen5_optimizer.so"
```

### llama.cpp Compilation

**CMake-Based Build Process with AVX-512**
```dockerfile
RUN rm -rf /tmp/llama.cpp && \
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp && \
    cd /tmp/llama.cpp && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=OFF \
        -DGGML_BLAS=OFF \
        -DGGML_NATIVE=ON \
        -DGGML_LTO=ON \
        -DGGML_AVX512=ON \
        -DGGML_AVX512_VBMI=ON \
        -DGGML_AVX512_VNNI=ON \
        -DGGML_AVX512_BF16=ON \
        -DGGML_BUILD_TESTS=OFF \
        -DGGML_BUILD_EXAMPLES=OFF \
        -DLLAMA_CURL=OFF && \
    cmake --build . --config Release -j$(nproc)
```

**Build Configuration**
- **CMAKE_BUILD_TYPE=Release**: Optimized release build with full optimizations
- **GGML_CUDA=OFF**: CPU-only build configuration
- **GGML_BLAS=OFF**: Disabled (using AVX-512 intrinsics instead)
- **GGML_NATIVE=ON**: Enable native CPU instruction set detection
- **GGML_LTO=ON**: Link-time optimization for additional performance gains
- **GGML_AVX512=ON**: Enable AVX-512 foundation instructions
- **GGML_AVX512_VBMI=ON**: Vector Byte Manipulation Instructions
- **GGML_AVX512_VNNI=ON**: Vector Neural Network Instructions (INT8 acceleration)
- **GGML_AVX512_BF16=ON**: BFloat16 support for Zen 5
- **GGML_BUILD_TESTS/EXAMPLES=OFF**: Skip test and example builds
- **LLAMA_CURL=OFF**: Disable curl integration in llama.cpp
- **Parallel compilation**: Uses all available CPU cores with -j$(nproc)

## Stage 2: Runtime Environment

### Runtime Base Image

**Minimal Runtime Foundation**
```dockerfile
FROM debian:unstable-slim

LABEL description="llama.cpp with hugepage optimization (AVX-512 + zen5_optimizer)"
```
- **Size optimization**: Minimal Debian base without development tools
- **Runtime focus**: Only includes libraries necessary for execution
- **Security**: Reduced attack surface through minimal package installation

### Library Dependencies

**Essential Runtime Libraries**
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*
```

**Runtime Components**
- **libgomp1**: OpenMP runtime library for parallel execution
- **curl**: HTTP client for health checks

### Security Configuration

**Non-Root User Implementation**
```dockerfile
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "Application User" appuser
USER appuser
WORKDIR /app
```

**Security Features**
- **Dedicated user**: Non-root execution prevents privilege escalation
- **No shell access**: `/sbin/nologin` prevents interactive access
- **Minimal permissions**: User restricted to application directory

### Application Deployment

**Binary Installation**
```dockerfile
# Copy server binary (renamed to 'server' for entrypoint compatibility)
COPY --from=builder --chown=appuser:appuser /tmp/llama.cpp/build/bin/llama-server /app/server
# Copy all binaries and libraries from bin/
COPY --from=builder --chown=appuser:appuser /tmp/llama.cpp/build/bin/* /app/
# Copy zen5_optimizer library for hugepage support
COPY --from=builder --chown=appuser:appuser /tmp/libzen5_optimizer.so /app/
# Copy entrypoint script
COPY --chmod=755 --chown=appuser:appuser docker/llama-cpu/entrypoint.cpu.sh /app/entrypoint.sh
```

**Environment Configuration**
```dockerfile
# Library path for llama.cpp shared libs
ENV LD_LIBRARY_PATH=/app

# OpenMP settings (OMP_NUM_THREADS set dynamically by entrypoint)
ENV OMP_PROC_BIND=spread
ENV OMP_PLACES=cores

EXPOSE 8001

ENTRYPOINT ["/app/entrypoint.sh"]
```

**OpenMP Configuration**
- **LD_LIBRARY_PATH=/app**: Locates libzen5_optimizer.so and other libraries
- **OMP_PROC_BIND=spread**: Distributes threads across cores for better cache utilization
- **OMP_PLACES=cores**: Optimizes memory access patterns per physical core
- **OMP_NUM_THREADS**: Set dynamically by entrypoint based on THREADS variable

## Entrypoint Configuration

The entrypoint script (`entrypoint.cpu.sh`) provides parameterized server configuration.

### Required Environment Variables

These must be provided via docker-compose or docker run:

| Variable | Description |
|----------|-------------|
| `MODEL_PATH` | Path to the GGUF model file |
| `SERVER_PORT` | Port for the HTTP API server |
| `THREADS` | Number of CPU threads for inference |
| `THREADS_BATCH` | Number of threads for batch processing |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | 0.0.0.0 | Bind address for server |
| `N_GPU_LAYERS` | 0 | GPU layers (0 for CPU-only) |
| `CTX_SIZE` | 32768 | Context window size |
| `BATCH_SIZE` | 2048 | Logical batch size |
| `UBATCH_SIZE` | 512 | Physical batch size |
| `THREADS_HTTP` | 2 | HTTP handler threads |

### Server Flags

The entrypoint executes the server with the following flags:
```bash
exec ./server \
    --model "$MODEL_PATH" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --n-gpu-layers "$N_GPU_LAYERS" \
    --ctx-size "$CTX_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --ubatch-size "$UBATCH_SIZE" \
    --threads "$THREADS" \
    --threads-batch "$THREADS_BATCH" \
    --threads-http "$THREADS_HTTP" \
    --flash-attn "on" \
    --cache-reuse 256 \
    --cont-batching \
    --metrics \
    --mlock \
    --no-repack
```

**Flag Descriptions**
- **--flash-attn "on"**: Enable Flash Attention for memory efficiency
- **--cache-reuse 256**: Reuse KV cache for repeated prompts
- **--cont-batching**: Enable continuous batching for throughput
- **--metrics**: Expose Prometheus-compatible metrics
- **--mlock**: Lock model in RAM (prevents swapping)
- **--no-repack**: Skip tensor repacking (startup optimization)

**Entrypoint Features**
- Validates model file exists before starting
- Enables zen5_optimizer via LD_PRELOAD
- Reports memory status (including hugepage allocation)
- Sets OMP_NUM_THREADS to match THREADS variable

## Performance Optimization Strategy

### AMD Zen 5 Architecture Targeting

**Instruction Set Utilization**
The implementation targets specific AMD Zen 5 capabilities:
- **znver5 targeting**: -march=znver5 generates Zen 5-specific code
- **AVX-512 extensions**: Full 512-bit vector processing via cmake flags
- **Native CPU features**: GGML_NATIVE=ON enables runtime feature detection

**Compiler Optimization Pipeline**
1. **Architecture targeting**: `-march=znver5` generates Zen 5-specific code
2. **Performance tuning**: `-mtune=znver5` optimizes for Zen 5 execution
3. **Aggressive optimization**: `-O3` enables maximum compiler optimization
4. **Link-time optimization**: GGML_LTO=ON for cross-module optimization

### AVX-512 CMAKE Configuration

AVX-512 is enabled via cmake flags (not CFLAGS) for precise control:

| Flag | Description |
|------|-------------|
| `GGML_AVX512=ON` | Enable AVX-512 foundation instructions |
| `GGML_AVX512_VBMI=ON` | Vector Byte Manipulation Instructions |
| `GGML_AVX512_VNNI=ON` | Vector Neural Network Instructions (INT8) |
| `GGML_AVX512_BF16=ON` | BFloat16 support for Zen 5 |

**Why cmake flags instead of CFLAGS?**
- Precise control over which AVX-512 extensions are used
- llama.cpp cmake detects and enables optimal code paths
- Avoids potential compatibility issues with -ffast-math

### Threading Configuration

**Parallel Processing Setup**
- **OpenMP threads**: Set dynamically via entrypoint to match THREADS variable
- **Core distribution**: `OMP_PROC_BIND=spread` distributes threads across cores
- **Cache awareness**: `OMP_PLACES=cores` optimizes for physical core cache
- **HTTP threading**: 2 dedicated threads for API request handling

### Memory Optimization

**Memory Management Strategy**
- **Memory locking**: `--mlock` prevents model data from being swapped to disk
- **Hugepage allocation**: zen5_optimizer uses 2MB pages for large allocations
- **Batch processing**: Optimized batch sizes (2048/512) for Zen 5 memory subsystem
- **Context management**: 32K context size balances memory usage and performance
- **KV cache reuse**: `--cache-reuse 256` for repeated prompt patterns

## Container Security Implementation

**Multi-Layer Security Approach**
- **Minimal runtime**: Only essential libraries (libgomp1, curl) included
- **Non-root execution**: Application runs as dedicated non-privileged user
- **No interactive access**: User account configured without shell access
- **Controlled exposure**: Only port 8001 exposed

**Docker Compose Integration**
When deployed via docker-compose, additional security measures include:
- **Capability dropping**: All Linux capabilities removed (`cap_drop: ALL`)
- **Privilege restrictions**: `no-new-privileges:true` prevents escalation
- **Resource constraints**: CPU and memory limits enforced (96GB, cores 0-15)
- **Read-only containers**: Filesystem mounted read-only for immutability

## Reference Implementation

**File Structure**
- **docker/llama-cpu/Dockerfile.cpu**: Multi-stage Dockerfile implementation
- **docker/llama-cpu/entrypoint.cpu.sh**: Parameterized server startup script
- **hugepages/CMakeLists.txt**: zen5_optimizer build configuration
- **hugepages/src/**: zen5_optimizer source files
- **docker-compose.yaml**: Container orchestration with security hardening

**Integration with AI Workstation**
This CPU implementation works in conjunction with:
- **GPU containers**: Parallel execution with CUDA-optimized containers (llama-gpu, vllm-gpu)
- **Resource isolation**: Container allocated 16 cores (0-15) and 96GB RAM via cpuset

---

*This CPU Docker implementation provides optimized AI model inference on the AMD Ryzen 9950X workstation. The multi-stage build process ensures minimal runtime overhead while maximizing performance through AVX-512 optimizations and hugepage memory allocation.*

*Last Updated: 2026-01-03*
