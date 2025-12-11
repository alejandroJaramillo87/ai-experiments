# zen5_optimizer

Memory optimization library for llama.cpp on AMD Zen 5 / AVX-512 processors. Provides transparent huge page allocation via LD_PRELOAD.

## Overview

This library intercepts `mmap()` calls and uses huge pages (2MB) for large memory allocations, reducing TLB misses and improving inference performance.

**Phase 1A validated** with benchmark proof:
- **8.4% TTFT improvement** (Time to First Token)
- **43% TLB miss reduction** (verified via perf counters)
- **8.4% faster prompt processing** (77.80 → 84.37 tokens/s)

## Usage in This Repository

The library is built and used by the `llama-cpu` Docker container:

```
docker/llama-cpu/Dockerfile.cpu  # Builds zen5_optimizer from hugepages/
docker/llama-cpu/entrypoint.cpu.sh  # Loads via LD_PRELOAD
```

**Build process:**
1. `docker compose build llama-cpu` compiles the library
2. Container entrypoint sets `LD_PRELOAD=/app/libzen5_optimizer.so`
3. Library intercepts `mmap()` calls for allocations >= 1GB

See `docker/llama-cpu/Dockerfile.cpu` for build configuration.

## Standalone Usage

```bash
# Build the library
make

# Load optimizer before running llama.cpp
LD_PRELOAD=./build/libzen5_optimizer.so ./llama-server [args]

# Or export for session
export LD_PRELOAD=/path/to/libzen5_optimizer.so
./llama-server [args]
```

## Requirements

- x86-64 processor with AVX-512 support:
  - AMD: Zen 4 (Ryzen 7000+), Zen 5 (Ryzen 9000+)
  - Intel: Skylake-X, Ice Lake, Sapphire Rapids, and later
- Linux x86_64
- GCC 14+ with C++17 support (for Zen 5 march=znver5)
- Huge pages enabled in kernel

## Building

```bash
make                # Build library
make test          # Run all tests (verbose)
make test-quiet    # Run tests with minimal output
make install       # Install to /usr/local/lib
```

## Architecture

```
src/
├── zen5_optimizer.cpp      # Main LD_PRELOAD entry point
├── cpu_validator.cpp       # AVX-512 detection (AMD/Intel)
├── config.h                # Configuration parameters
└── memory/
    └── hugepage_wrapper.cpp # mmap() interception

tests/
├── unit/                   # Basic functionality tests
├── functional/             # Feature-level tests
└── integration/            # End-to-end validation
```

## Performance

Phase 1A validated results (80B MoE model, AMD Ryzen 9 9950X):

| Metric | Without Optimization | With Hugepages | Improvement |
|--------|---------------------|----------------|-------------|
| TTFT | 9844 ms | 9078 ms | **8.4% faster** |
| Prompt speed | 77.80 t/s | 84.37 t/s | **8.4% faster** |
| Generation | 9.54 t/s | 9.89 t/s | **3.7% faster** |
| dTLB misses | 9.95 billion | 5.71 billion | **43% reduction** |

## Development Roadmap

- **Phase 1A** - Transparent huge page allocation (complete)
- Phase 2A - NUMA-aware memory allocation (planned)
- Phase 2B - Layer-ahead prefetching (planned)
- Phase 2C - Mixed huge pages: 1GB/2MB/4KB (planned)

See `docs/zen_roadmap.md` for detailed benchmark results and optimization experiments.
