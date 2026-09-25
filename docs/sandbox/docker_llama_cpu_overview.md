# AI engineering CPU Docker implementation

Explanation of the AMD Zen 5 optimized Dockerfile for CPU-based AI model inference on the AMD Ryzen 9950X workstation, detailing the multi-stage build process, zen5_optimizer integration, and AVX-512 performance optimizations.

> Note: This document explains the actual implementation in `docker/llama-cpu/Dockerfile.cpu` and `docker/llama-cpu/entrypoint.cpu.sh`. The Dockerfile and entrypoint are the source of truth; this document follows them.

## Table of Contents

<!-- toc:start -->

1. [Implementation overview](#implementation-overview)
2. [Stage 1: builder environment](#stage-1-builder-environment)
   1. [Base image selection](#base-image-selection)
   2. [AMD Zen 5 compiler configuration](#amd-zen-5-compiler-configuration)
   3. [System dependencies installation](#system-dependencies-installation)
   4. [zen5_optimizer library](#zen5_optimizer-library)
   5. [llama.cpp compilation](#llamacpp-compilation)
3. [Stage 2: runtime environment](#stage-2-runtime-environment)
   1. [Runtime base image](#runtime-base-image)
   2. [Library dependencies](#library-dependencies)
   3. [Security configuration](#security-configuration)
   4. [Application deployment](#application-deployment)
4. [Entrypoint configuration](#entrypoint-configuration)
   1. [Required environment variables](#required-environment-variables)
   2. [Optional environment variables](#optional-environment-variables)
   3. [Feature flags](#feature-flags)
   4. [Advanced server knobs](#advanced-server-knobs)
   5. [Server command construction](#server-command-construction)
5. [Performance optimization strategy](#performance-optimization-strategy)
   1. [AMD Zen 5 architecture targeting](#amd-zen-5-architecture-targeting)
   2. [AVX-512 cmake configuration](#avx-512-cmake-configuration)
   3. [Threading configuration](#threading-configuration)
   4. [Memory optimization](#memory-optimization)
6. [Container security implementation](#container-security-implementation)
7. [Reference implementation](#reference-implementation)

<!-- toc:end -->

## Implementation overview

The CPU Docker implementation uses a multi-stage build strategy optimized for the AMD Ryzen 9950X processor. The architecture separates the build environment from the runtime environment to minimize the final container size while targeting CPU performance through Zen 5-specific optimizations.

**Key implementation features:**
- **Multi-stage build**: Separates compilation dependencies from runtime requirements.
- **Shared base image**: Both stages derive from `debian:trixie-slim` so glibc and libgomp stay in sync between build and runtime.
- **AMD Zen 5 optimization**: Compiler flags target the znver5 architecture.
- **zen5_optimizer library**: Hugepage support via LD_PRELOAD for reduced TLB pressure.
- **AVX-512 native**: Explicit VNNI, VBMI, and BF16 extensions enabled via cmake.
- **Pinned llama.cpp**: The source commit is pinned by SHA for reproducible builds.
- **Security hardening**: Non-root user execution and minimal runtime dependencies.

## Stage 1: builder environment

### Base image selection

**Shared Debian 13 trixie base**
```dockerfile
# Base image shared by builder and runtime so glibc/libgomp stay in sync.
# Pin the digest before building. Re-resolve with:
#   docker buildx imagetools inspect debian:trixie-slim
# then set BASE_IMAGE=debian:trixie-slim@sha256:<digest>
ARG BASE_IMAGE=debian:trixie-slim@sha256:28de0877c2189802884ccd20f15ee41c203573bd87bb6b883f5f46362d24c5c2

# --- Stage 1: The Builder ---
# Compiles llama.cpp with hugepage optimization (AVX-512 + zen5_optimizer)
FROM ${BASE_IMAGE} AS builder

LABEL stage="builder"
```
- **Purpose**: Provides a minimal Debian environment with the toolchain required to compile llama.cpp and the zen5_optimizer library.
- **Shared base**: Builder and runtime both use `${BASE_IMAGE}`, keeping the glibc and libgomp versions identical across stages and avoiding runtime symbol mismatches.
- **Digest pinning**: already done — the default is digest-pinned in the Dockerfile, so a
  build is reproducible without any extra step. To move it deliberately, re-resolve with
  `docker buildx imagetools inspect debian:trixie-slim` and edit the `ARG`. Bumping it is a
  version change and belongs in its own commit.

### AMD Zen 5 compiler configuration

**Build environment variables**
```dockerfile
# Build environment for AMD Zen 5 architecture
ENV DEBIAN_FRONTEND=noninteractive
ENV CFLAGS="-march=znver5 -mtune=znver5 -O3"
ENV CXXFLAGS="${CFLAGS}"
ENV CC=gcc-14
ENV CXX=g++-14
ENV LDFLAGS="-Wl,-O3 -Wl,--as-needed"
```

**Compiler flag optimization**
- **-march=znver5**: Generates code specifically for the AMD Zen 5 architecture.
- **-mtune=znver5**: Optimizes instruction scheduling for the Zen 5 microarchitecture.
- **-O3**: Enables aggressive compiler optimizations.
- **gcc-14**: Provides znver5 support, present since GCC 14.1 and shipped in Debian 13 trixie stable.
- **Linker optimization**: `-Wl,-O3` and `--as-needed` reduce binary size.

Note: AVX-512 instruction sets are enabled via cmake flags, not CFLAGS.

### System dependencies installation

**Build tools and libraries**
```dockerfile
# Install build dependencies (gcc-14 with znver5 support ships in trixie stable)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc-14 g++-14 build-essential cmake git ca-certificates \
        libssl-dev libomp-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*
```

**Package selection**
- **gcc-14/g++-14**: Compiler suite with znver5 support, available directly from the trixie stable repository. No third-party or unstable apt source is required.
- **cmake**: Build system for llama.cpp and zen5_optimizer.
- **git**: Clones the llama.cpp source.
- **ca-certificates**: Certificate bundle for HTTPS clones.
- **libomp-dev**: OpenMP development libraries for parallel processing.
- **libssl-dev/pkg-config**: Build dependencies for the OpenSSL-based HTTPS path in llama.cpp.

### zen5_optimizer library

**CMake-based hugepage optimization**

The hugepage optimization is provided by `zen5_optimizer`, a library that intercepts mmap() calls for large allocations:

```dockerfile
# Build zen5_optimizer library (hugepage support via LD_PRELOAD)
COPY docker/llama-cpu/hugepages/CMakeLists.txt /tmp/zen5_optimizer/
COPY docker/llama-cpu/hugepages/src/ /tmp/zen5_optimizer/src/
RUN cd /tmp/zen5_optimizer && \
    cmake -B build \
        -DCMAKE_C_COMPILER=gcc-14 \
        -DCMAKE_CXX_COMPILER=g++-14 \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_HUGEPAGES=ON \
        -DENABLE_DEBUG_PRINT=OFF && \
    cmake --build build --config Release -j$(nproc) && \
    cp build/libzen5_optimizer.so /tmp/libzen5_optimizer.so && \
    echo "Built libzen5_optimizer.so (hugepage optimization)"
```

**Source files**
- `docker/llama-cpu/hugepages/CMakeLists.txt`: Build configuration.
- `docker/llama-cpu/hugepages/src/zen5_optimizer.cpp`: Main entry point.
- `docker/llama-cpu/hugepages/src/cpu_validator.cpp` and `docker/llama-cpu/hugepages/src/cpu_validator.h`: CPU feature validation.
- `docker/llama-cpu/hugepages/src/memory/hugepage_wrapper.cpp`: mmap() interception.
- `docker/llama-cpu/hugepages/src/config.h`: Configuration headers.

**Build configuration**
- **CMAKE_C_COMPILER=gcc-14**: Use gcc-14 for C compilation.
- **CMAKE_CXX_COMPILER=g++-14**: Use g++-14 for C++ compilation.
- **CMAKE_BUILD_TYPE=Release**: Optimized release build.
- **ENABLE_HUGEPAGES=ON**: Enable hugepage memory allocation.
- **ENABLE_DEBUG_PRINT=OFF**: Disable debug output.

**How it works**
The library intercepts mmap() calls for allocations of 1GB or larger and uses MAP_HUGETLB to allocate 2MB huge pages instead of 4KB standard pages.

**Performance benefits**
- **Reduced TLB misses**: 2MB pages instead of 4KB reduces translation overhead.
- **Better memory locality**: Fewer page table entries to manage.
- **10-20% inference speedup**: Measured on large models greater than 15GB.

**Runtime activation**
The entrypoint script activates the library via LD_PRELOAD:
```bash
export LD_PRELOAD="/app/libzen5_optimizer.so"
```

### llama.cpp compilation

**Pinned source and AVX-512 build**

The llama.cpp source is pinned to an exact commit for reproducible builds and cloned in full from the canonical `ggml-org` repository. The commit is bumped deliberately, not on every rebuild.

```dockerfile
# Build llama.cpp with AVX-512 configuration
# Pin llama.cpp to an exact commit for reproducible builds. Bump deliberately, not on rebuild.
# LLAMA_CURL was removed upstream (deprecated no-op); HTTPS model fetch now uses OpenSSL.
ARG LLAMA_CPP_COMMIT=e3546c7948e3af463d0b401e6421d5a4c2faf565
# tag b9976, 2026-07-12
RUN rm -rf /tmp/llama.cpp && \
    git clone https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp && \
    cd /tmp/llama.cpp && \
    git checkout "${LLAMA_CPP_COMMIT}" && \
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
        -DGGML_BUILD_EXAMPLES=OFF && \
    cmake --build . --target llama-server --config Release -j$(nproc) && \
    echo "Built llama-server (AVX-512 VNNI, hugepages via LD_PRELOAD)"
```

**Source pinning**
- **LLAMA_CPP_COMMIT**: Pinned to `e3546c7948e3af463d0b401e6421d5a4c2faf565` (tag b9976, dated 2026-07-12).
- **Full clone**: The repository is cloned in full and checked out by SHA. A full clone is required because a shallow clone cannot check out an arbitrary historical commit.
- **Canonical repository**: The source is fetched from `https://github.com/ggml-org/llama.cpp.git`.

**Build target**
- **llama-server only**: The build targets `llama-server` (`cmake --build . --target llama-server`) rather than the full example set, reducing build time.

**Build configuration**
- **CMAKE_BUILD_TYPE=Release**: Optimized release build.
- **GGML_CUDA=OFF**: CPU-only build configuration.
- **GGML_BLAS=OFF**: Disabled in favor of AVX-512 intrinsics.
- **GGML_NATIVE=ON**: Enable native CPU instruction set detection.
- **GGML_LTO=ON**: Link-time optimization for cross-module optimization.
- **GGML_AVX512=ON**: Enable AVX-512 foundation instructions.
- **GGML_AVX512_VBMI=ON**: Vector Byte Manipulation Instructions.
- **GGML_AVX512_VNNI=ON**: Vector Neural Network Instructions (INT8 acceleration).
- **GGML_AVX512_BF16=ON**: BFloat16 support for Zen 5.
- **GGML_BUILD_TESTS/EXAMPLES=OFF**: Skip test and example builds.
- **Parallel compilation**: Uses all available CPU cores with `-j$(nproc)`.

The former libcurl build option is no longer set. Upstream replaced libcurl with OpenSSL for HTTPS model fetches, so the option became a no-op and was removed. See [Library dependencies](#library-dependencies) for the corresponding runtime change.

## Stage 2: runtime environment

### Runtime base image

**Shared base runtime foundation**
```dockerfile
# --- Stage 2: The Final Runtime Image ---
FROM ${BASE_IMAGE}

LABEL description="llama.cpp with hugepage optimization (AVX-512 + zen5_optimizer)"
```
- **Shared base**: The runtime reuses the same `${BASE_IMAGE}` (`debian:trixie-slim`) as the builder, guaranteeing matching glibc and libgomp versions.
- **Runtime focus**: The stage installs only the shared libraries required to run the server, keeping the attack surface small.

### Library dependencies

**Essential runtime libraries**
```dockerfile
# Runtime libraries only (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libssl3t64 \
    curl \
    && rm -rf /var/lib/apt/lists/*
```

**Runtime components**
- **libgomp1**: OpenMP runtime library for parallel execution.
- **libssl3t64**: OpenSSL runtime. llama.cpp now performs HTTPS model fetches through OpenSSL, having replaced its libcurl dependency in early 2026.
- **curl**: Used by the container healthcheck to poll the server. It is not used by llama.cpp itself.

### Security configuration

**Non-root user implementation**
```dockerfile
# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "Application User" appuser

USER appuser
WORKDIR /app
```

**Security features**
- **Dedicated user**: Non-root execution prevents privilege escalation.
- **No shell access**: `/sbin/nologin` prevents interactive access.
- **Minimal permissions**: The user is restricted to the application directory.

### Application deployment

**Binary and library installation**

The runtime copies only the server binary and the shared libraries it links against, rather than the entire build output directory:

```dockerfile
# Copy server binary (renamed to 'server' for entrypoint compatibility)
COPY --from=builder --chown=appuser:appuser /tmp/llama.cpp/build/bin/llama-server /app/server
# Copy the shared libraries llama-server links against (libllama, libggml*, libmtmd)
COPY --from=builder --chown=appuser:appuser /tmp/llama.cpp/build/bin/*.so* /app/
# Copy zen5_optimizer library for hugepage support
COPY --from=builder --chown=appuser:appuser /tmp/libzen5_optimizer.so /app/
COPY --chmod=755 --chown=appuser:appuser docker/llama-cpu/entrypoint.cpu.sh /app/entrypoint.sh
```

- **/app/server**: The `llama-server` binary, renamed to `server` for entrypoint compatibility.
- **/app/*.so***: The shared libraries the server links against (libllama, libggml*, libmtmd), copied explicitly rather than via a wholesale `bin/*` copy.
- **/app/libzen5_optimizer.so**: The hugepage optimization library.
- **/app/entrypoint.sh**: The parameterized startup script.

**Environment configuration**
```dockerfile
# Library path for llama.cpp shared libs
ENV LD_LIBRARY_PATH=/app

# OpenMP settings (OMP_NUM_THREADS set dynamically by entrypoint)
ENV OMP_PROC_BIND=spread
ENV OMP_PLACES=cores

EXPOSE 8001

ENTRYPOINT ["/app/entrypoint.sh"]
```

**OpenMP configuration**
- **LD_LIBRARY_PATH=/app**: Locates `libzen5_optimizer.so` and the copied shared libraries.
- **OMP_PROC_BIND=spread**: Distributes threads across cores for better cache utilization.
- **OMP_PLACES=cores**: Binds threads to physical cores.
- **OMP_NUM_THREADS**: Set dynamically by the entrypoint from the THREADS variable.

## Entrypoint configuration

The entrypoint script (`docker/llama-cpu/entrypoint.cpu.sh`) builds a `CMD_ARGS` bash array, gating optional flags on environment variables, then execs the server. It validates the model file, enables the zen5_optimizer via LD_PRELOAD, and reports memory status before loading.

### Required environment variables

The script fails fast if any of these are missing:

| Variable | Description |
|----------|-------------|
| `MODEL_PATH` | Path to the GGUF model file |
| `SERVER_PORT` | Port for the HTTP API server |
| `THREADS` | Number of CPU threads for inference |
| `THREADS_BATCH` | Number of threads for batch processing |

### Optional environment variables

These have defaults and always map to a server flag:

| Variable | Default | Server flag | Description |
|----------|---------|-------------|-------------|
| `SERVER_HOST` | `0.0.0.0` | `--host` | Bind address for the server |
| `N_GPU_LAYERS` | `0` | `--n-gpu-layers` | GPU layers (0 for CPU-only) |
| `CTX_SIZE` | `32768` | `--ctx-size` | Context window size |
| `BATCH_SIZE` | `2048` | `--batch-size` | Logical batch size |
| `UBATCH_SIZE` | `512` | `--ubatch-size` | Physical batch size |
| `THREADS_HTTP` | `2` | `--threads-http` | HTTP handler threads |

### Feature flags

These defaults preserve the historical hardcoded behavior. Each maps to a flag that is appended only when its condition holds, except `FLASH_ATTN`, which is always passed:

| Variable | Default | Server flag | Condition |
|----------|---------|-------------|-----------|
| `FLASH_ATTN` | `on` | `--flash-attn <value>` | Always appended |
| `CACHE_REUSE` | `256` | `--cache-reuse <value>` | Appended when greater than 0 |
| `CONT_BATCHING` | `true` | `--cont-batching` | Appended when `true` |
| `MLOCK` | `true` | `--mlock` | Appended when `true` |
| `NO_REPACK` | `true` | `--no-repack` | Appended when `true` |

- **FLASH_ATTN**: The upstream default is now `auto`. Passing `on` explicitly is deliberate, keeping Flash Attention enabled regardless of upstream default changes.
- **NO_REPACK**: Weight repacking rearranges weights into row-interleaved layouts for the AVX-512 GEMM paths. It is the main CPU throughput lever upstream, but it increases startup time. The default disables it (`NO_REPACK=true` passes `--no-repack`) as a documented tradeoff, pending a benchmark comparison against `NO_REPACK=false`. Set `NO_REPACK=false` to enable repacking and measure the throughput difference.

### Advanced server knobs

These are appended only when explicitly set, so the server falls back to its own defaults otherwise:

| Variable | Default | Server flag | Condition |
|----------|---------|-------------|-----------|
| `CACHE_TYPE_K` | unset | `--cache-type-k <value>` | Appended when non-empty |
| `CACHE_TYPE_V` | unset | `--cache-type-v <value>` | Appended when non-empty |
| `KV_UNIFIED` | `false` | `--kv-unified` | Appended when `true` |
| `CTX_CHECKPOINTS` | unset | `--ctx-checkpoints <value>` | Appended when non-empty |
| `JINJA_ENABLED` | `false` | `--jinja` | Appended when `true` |

### Server command construction

The script starts from a base `CMD_ARGS` array containing the required flags plus the always-on `--flash-attn` and `--metrics`:

```bash
# Build command arguments
CMD_ARGS=(
    "./server"
    "--model" "$MODEL_PATH"
    "--host" "$SERVER_HOST"
    "--port" "$SERVER_PORT"
    "--n-gpu-layers" "$N_GPU_LAYERS"
    "--ctx-size" "$CTX_SIZE"
    "--batch-size" "$BATCH_SIZE"
    "--ubatch-size" "$UBATCH_SIZE"
    "--threads" "$THREADS"
    "--threads-batch" "$THREADS_BATCH"
    "--threads-http" "$THREADS_HTTP"
    "--flash-attn" "$FLASH_ATTN"
    "--metrics"
)
```

It then appends the feature and advanced flags conditionally before executing the array:

```bash
# Add optional parameters based on environment variables
if [[ "$CACHE_REUSE" -gt 0 ]]; then
    CMD_ARGS+=("--cache-reuse" "$CACHE_REUSE")
fi

if [[ "$CONT_BATCHING" == "true" ]]; then
    CMD_ARGS+=("--cont-batching")
fi

if [[ "$MLOCK" == "true" ]]; then
    CMD_ARGS+=("--mlock")
fi

if [[ "$NO_REPACK" == "true" ]]; then
    CMD_ARGS+=("--no-repack")
fi
```

- **--metrics**: Exposes Prometheus-compatible metrics unconditionally.
- The advanced knobs (`--cache-type-k`, `--cache-type-v`, `--kv-unified`, `--ctx-checkpoints`, `--jinja`) follow the same conditional pattern.

The final array is executed with `exec "${CMD_ARGS[@]}"`, replacing the shell process with the server.

**Entrypoint responsibilities**
- Validates that the model file exists before starting, exiting with an error otherwise.
- Enables the zen5_optimizer via LD_PRELOAD.
- Reports memory status, including hugepage counters from `/proc/meminfo`, before the model loads.
- Sets `OMP_NUM_THREADS` to match the THREADS variable.

## Performance optimization strategy

### AMD Zen 5 architecture targeting

**Instruction set utilization**
The implementation targets specific AMD Zen 5 capabilities:
- **znver5 targeting**: `-march=znver5` generates Zen 5-specific code.
- **AVX-512 extensions**: Full 512-bit vector processing via cmake flags.
- **Native CPU features**: `GGML_NATIVE=ON` enables runtime feature detection.

**Compiler optimization pipeline**
1. **Architecture targeting**: `-march=znver5` generates Zen 5-specific code.
2. **Performance tuning**: `-mtune=znver5` optimizes for Zen 5 execution.
3. **Aggressive optimization**: `-O3` enables maximum compiler optimization.
4. **Link-time optimization**: `GGML_LTO=ON` for cross-module optimization.

### AVX-512 cmake configuration

AVX-512 is enabled via cmake flags rather than CFLAGS for precise control:

| Flag | Description |
|------|-------------|
| `GGML_AVX512=ON` | Enable AVX-512 foundation instructions |
| `GGML_AVX512_VBMI=ON` | Vector Byte Manipulation Instructions |
| `GGML_AVX512_VNNI=ON` | Vector Neural Network Instructions (INT8) |
| `GGML_AVX512_BF16=ON` | BFloat16 support for Zen 5 |

**Rationale for cmake flags over CFLAGS**
- Precise control over which AVX-512 extensions are used.
- The llama.cpp cmake logic detects and enables the optimal code paths.
- Avoids compatibility issues with unrelated CFLAGS changes.

### Threading configuration

**Parallel processing setup**
- **OpenMP threads**: Set dynamically by the entrypoint to match the THREADS variable.
- **Core distribution**: `OMP_PROC_BIND=spread` distributes threads across cores.
- **Cache awareness**: `OMP_PLACES=cores` binds threads to physical cores.
- **HTTP threading**: 2 dedicated threads for API request handling (`THREADS_HTTP`).

### Memory optimization

**Memory management strategy**
- **Memory locking**: `--mlock` prevents model data from being swapped to disk.
- **Hugepage allocation**: The zen5_optimizer uses 2MB pages for allocations of 1GB or larger.
- **Batch processing**: Batch sizes of 2048 (logical) and 512 (physical) for the Zen 5 memory subsystem.
- **Context management**: A 32768-token context balances memory usage and capacity.
- **KV cache reuse**: `--cache-reuse 256` for repeated prompt prefixes.
- **Weight repacking**: Disabled by default to shorten startup; see [Feature flags](#feature-flags) for the tradeoff.

## Container security implementation

**Multi-layer security approach**
- **Minimal runtime**: Only essential shared libraries (libgomp1, libssl3t64, curl) are installed.
- **Non-root execution**: The application runs as a dedicated non-privileged user.
- **No interactive access**: The user account is configured without shell access.
- **Controlled exposure**: Only port 8001 is exposed.

**Docker Compose integration**
The `llama-cpu` service in `docker-compose.inference.yaml` adds further constraints:
- **Capability dropping**: All Linux capabilities are removed (`cap_drop: ALL`).
- **Privilege restrictions**: `no-new-privileges:true` prevents escalation.
- **Resource constraints**: 96GB memory limit and CPU cores 0-15 via `cpuset`.
- **Read-only container**: The root filesystem is mounted read-only (`read_only: true`).
- **Thread allocation**: `THREADS=16` and `THREADS_BATCH=16`, matching the 16 pinned cores.
- **Loopback binding**: The port is published as `127.0.0.1:8001:8001`.
- **Health check**: `curl -f http://localhost:8001/v1/health` runs every 30 seconds with a 180-second start period. The server responds on both `/health` and `/v1/health`; the compose check uses `/v1/health`.

## Reference implementation

**File structure**
- `/docker/llama-cpu/Dockerfile.cpu`: Multi-stage Dockerfile implementation.
- `/docker/llama-cpu/entrypoint.cpu.sh`: Parameterized server startup script.
- `/docker/llama-cpu/hugepages/CMakeLists.txt`: zen5_optimizer build configuration.
- `/docker/llama-cpu/hugepages/src/`: zen5_optimizer source files.
- `/docker-compose.inference.yaml`: Container orchestration with security hardening.

**Integration with the AI workstation**
This CPU implementation runs alongside:
- **GPU containers**: The CUDA-optimized `llama-gpu` and `vllm-gpu` services.
- **Resource isolation**: The container is allocated 16 cores (0-15) and a 96GB memory limit via `cpuset` and `deploy.resources.limits`.

---

*This CPU Docker implementation targets AI model inference on the AMD Ryzen 9950X workstation. The multi-stage build keeps runtime dependencies minimal while enabling AVX-512 code paths and hugepage memory allocation.*
