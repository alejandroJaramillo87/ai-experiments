# Test suite documentation

Comprehensive test coverage for the AMD Zen 5 optimizer library.

## Table of Contents

- [Quick start](#quick-start)
- [Test organization](#test-organization)
- [Test categories](#test-categories)
- [Running tests](#running-tests)
- [Test output](#test-output)
- [Requirements](#requirements)
- [Test development](#test-development)
- [Troubleshooting](#troubleshooting)
- [Coverage status](#coverage-status)

## Quick start

Build and run all tests from project root:
```bash
$ make test          # Runs all tests with verbose output (default)
$ make test-quiet    # Runs all tests with minimal output
```

Run tests directly from test directory:
```bash
$ cd tests
$ ./run_tests.sh --verbose    # Verbose output
$ ./run_tests.sh              # Minimal output
```

## Test organization

Directory structure:
```
tests/
├── unit/                   # Basic functionality tests
├── functional/             # Feature-level tests
├── integration/            # End-to-end tests
└── include/                # Shared test utilities
```

## Test categories

### Unit tests (3 tests)

Basic component verification:

- **test_cpu** - AMD Zen 5 CPU detection and validation
- **test_load** - Library loading and initialization
- **test_hugepage** - mmap interception and concurrent operations (includes 3-thread concurrency test)

### Functional tests (6 tests)

Complete feature testing:

- **test_memory_boundaries** - 1GB threshold testing, MAP_FIXED handling, hugepage fallback
- **test_munmap** - Allocation tracking, double munmap protection, partial unmapping
- **test_fallback** - Graceful handling when hugepages unavailable
- **test_memory_tracking** - Track/untrack allocations, cleanup verification, fork handling
- **test_stress** - 50 rapid cycles, 8 concurrent threads, memory pressure, mixed sizes
- **test_performance** - Baseline measurements, throughput testing, TLB efficiency

### Integration tests (1 test)

Full system validation:

- **test_integration** - Complete LD_PRELOAD workflow with 1.5GB test files

## Running tests

### Using Makefile (from project root)

```bash
$ make test              # All tests with verbose output (default)
$ make test-quiet        # All tests with minimal output
$ make test-unit         # Unit tests only (verbose)
$ make test-functional   # Functional tests only (verbose)
$ make test-integration  # Integration test only (verbose)
$ make test-all          # Explicitly run all tests (verbose)
```

### Using test runner (from tests directory)

```bash
$ ./run_tests.sh              # All tests, minimal output
$ ./run_tests.sh --verbose    # All tests, detailed output
```

### Specific categories

```bash
$ ./run_tests.sh unit         # Unit tests only
$ ./run_tests.sh functional   # Functional tests only
$ ./run_tests.sh integration  # Integration tests only
```

### Individual tests

```bash
$ ./run_tests.sh test_cpu     # Run specific test
$ ./run_tests.sh memory       # Run all tests matching "memory"
$ ./run_tests.sh stress       # Run stress test
$ ./run_tests.sh performance  # Run performance baselines
```

### Advanced options

```bash
$ ./run_tests.sh --verbose    # Detailed output
$ ./run_tests.sh --list       # List available tests
$ ./run_tests.sh --no-color   # Disable colored output
$ ./run_tests.sh --help       # Show all options
```

## Test output

Tests use standardized output indicators:

| Indicator | Meaning |
|-----------|---------|
| **[OK]** | Test or assertion passed |
| **[FAIL]** | Test or assertion failed |
| **[WARN]** | Warning or non-AMD Zen 5 CPU |
| **[RUN]** | Test execution started |
| **[SKIP]** | Test was skipped |
| **[INFO]** | Informational message |

## Requirements

### Hardware requirements

- AMD Zen 5 CPU (tests skip with warning on other CPUs)

### Software requirements

- Linux kernel with mmap support
- GCC 14 or compatible C++ compiler
- ~3GB free disk space for test files
- pthread library (for concurrent tests)

### Optional dependencies

- Hugepage support (tests gracefully handle when unavailable)
- valgrind for memory leak detection (disabled by default due to hugepage interactions)

## Test development

### Adding new tests

Create test file in appropriate directory:

```cpp
// tests/unit/test_myfeature.cpp
#include "../include/test_colors.h"

int main() {
    PRINT_TEST("My feature test");
    // Test implementation
    PRINT_OK("Test passed");
    return 0;
}
```

Add test to Makefile TEST_SOURCES variable.

Build and run:
```bash
$ make clean && make test
```

### Test guidelines

- Follow TEST_OUTPUT_STYLE.md for consistent output formatting
- Keep unit tests under 1 second execution time
- Clean up all created files and resources
- Use descriptive test names
- Include positive and negative test cases
- Test edge cases and boundary conditions
- Use test_colors.h for consistent output markers
- Return proper exit codes (0=success, non-zero=failure)

## Troubleshooting

### Build failures

```bash
$ cd .. && make clean && make
```

### Library not found

Ensure library is built first:
```bash
$ cd .. && make
```

### Permission denied

Fix script permissions:
```bash
$ chmod +x integration/test_integration.sh
```

### Hugepage test failures

Check system hugepage configuration:
```bash
$ cat /proc/sys/vm/nr_hugepages
```

Value should be greater than 0 for hugepage tests.

## Coverage status

### Phase 1A: Complete

Current test coverage metrics:
- **10 executable tests** (3 unit, 6 functional, 1 integration)
- **2,800 lines** of test code
- **~2 seconds** full suite execution time
- **100% feature coverage** for Phase 1A

| Component | Status |
|-----------|--------|
| CPU detection and validation | Complete |
| Library loading mechanics | Complete |
| Hugepage allocation (>1GB files) | Complete |
| Memory threshold boundaries (1GB ±1 byte) | Complete |
| MAP_FIXED flag handling | Complete |
| Concurrent mmap operations | Complete |
| Hugepage fallback scenarios | Complete |
| Double munmap protection | Complete |
| Partial unmapping handling | Complete |
| Allocation tracking and cleanup | Complete |
| Memory leak detection | Complete |
| Stress testing (50+ cycles, 8 threads) | Complete |
| Performance baselines | Complete |
| Full LD_PRELOAD workflow | Complete |

### Future phases

| Component | Target Phase |
|-----------|-------------|
| NUMA/CCX optimization | Phase 1B |
| AOCL BLIS integration | Phase 2 |
| AVX-512 VNNI acceleration | Phase 3 |
| Memory prefetching | Phase 4 |
| Tensor core optimization | Phase 5 |

---

*Last Updated: 2025-09-27*