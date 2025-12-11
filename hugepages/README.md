# llama-zen-turbo

x86-64 AVX-512 optimizer for llama.cpp - transparent performance improvements via LD_PRELOAD.

## Overview

This project provides memory subsystem optimizations for running llama.cpp on modern x86-64 processors with AVX-512 support. It achieves measurable speedup through:

- **Transparent huge page allocation** for model files (Phase 1A - complete)
- NUMA-aware memory allocation (Phase 2A - planned)
- Layer-ahead prefetching (Phase 2B - planned)
- Mixed huge pages: 1GB/2MB/4KB (Phase 2C - planned)

## Current Status

**Phase 1A validated** with benchmark proof:
- **8.4% TTFT improvement** (Time to First Token)
- **43% TLB miss reduction** (verified via perf counters)
- **8.4% faster prompt processing** (77.80 → 84.37 tokens/s)

See `docs/zen_roadmap.md` for detailed benchmark results.

## Requirements

- x86-64 processor with AVX-512 support:
  - AMD: Zen 4 (Ryzen 7000+), Zen 5 (Ryzen 9000+)
  - Intel: Skylake-X, Ice Lake, Sapphire Rapids, and later
- Linux x86_64
- GCC with C++17 support
- Huge pages enabled in kernel

## Building

```bash
make                # Build library
make test          # Run all tests (verbose)
make test-quiet    # Run tests with minimal output
make install       # Install to /usr/local/lib
```

## Usage

```bash
# Load optimizer before running llama.cpp
LD_PRELOAD=/usr/local/lib/libzen5_optimizer.so ./llama-cli [args]

# Or export for session
export LD_PRELOAD=/usr/local/lib/libzen5_optimizer.so
./llama-cli [args]
```

## Architecture

```
src/
├── zen5_optimizer.cpp      # Main LD_PRELOAD entry point
├── cpu_validator.cpp       # AVX-512 detection (AMD/Intel)
├── memory/
│   └── hugepage_wrapper.cpp # mmap() interception
└── config.h                # Configuration parameters

tests/
├── unit/                   # Basic functionality tests
├── functional/             # Feature-level tests
└── integration/            # End-to-end validation

scripts/
└── benchmark_with_perf.sh  # Benchmark with perf counter collection
```

## Performance

Phase 1A validated results (80B MoE model, AMD Ryzen 9 9950X):

| Metric | Without Optimization | With Hugepages | Improvement |
|--------|---------------------|----------------|-------------|
| TTFT | 9844 ms | 9078 ms | **8.4% faster** |
| Prompt speed | 77.80 t/s | 84.37 t/s | **8.4% faster** |
| Generation | 9.54 t/s | 9.89 t/s | **3.7% faster** |
| dTLB misses | 9.95 billion | 5.71 billion | **43% reduction** |

## Development

The library requires AVX-512 support. It will detect CPU capabilities at load time and exit with an error message if AVX-512 is not available.

See `docs/zen_roadmap.md` for detailed development phases and optimization experiments.
