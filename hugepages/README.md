# zen5_optimizer

Memory optimization library for llama.cpp on AMD Zen 5 / AVX-512 processors. Provides transparent huge page allocation via LD_PRELOAD.

## Overview

This library intercepts `mmap()` calls and uses huge pages (2MB) for large memory allocations, reducing TLB misses and improving inference performance.

## Usage in This Repository

The library is built and used by the `llama-cpu` Docker container:

```
docker/llama-cpu/Dockerfile.cpu  # Builds zen5_optimizer from hugepages/
docker/llama-cpu/entrypoint.cpu.sh  # Loads via LD_PRELOAD
```

**Build process:**
1. `docker compose -f docker-compose.inference.yaml build llama-cpu` compiles the library
2. Container entrypoint sets `LD_PRELOAD=/app/libzen5_optimizer.so`, but only when
   `HUGEPAGES=on` (its default). The preload is conditional because the wrapper is a
   measured arm rather than a constant: `HUGEPAGES=off` runs the same image without it,
   which is the control arm `python -m bench` compares against.
3. Library intercepts `mmap()` calls for allocations >= 1GB

**Measure it, do not assume it:**

```bash
python3 -m bench noise --service llama-cpu --model <slug>
python3 -m bench run   --service llama-cpu --model <slug> --profile publish \
    --arm on:HUGEPAGES=on --arm off:HUGEPAGES=off
```

The report records `HugePages_Free` on both arms, which is the evidence the wrapper took
pages at all rather than falling back.

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


