# Comment Style Guide

This guide establishes unified comment standards across all file types in the repository, following Unix/Linux documentation principles of clarity, technical precision, and professional presentation.

## Table of Contents

- [Comment Style Guide](#comment-style-guide)
  - [Table of Contents](#table-of-contents)
  - [Core Principles](#core-principles)
  - [Universal Standards](#universal-standards)
    - [Professional Tone](#professional-tone)
    - [Technical Accuracy](#technical-accuracy)
    - [Conciseness](#conciseness)
  - [Language-Specific Syntax](#language-specific-syntax)
    - [Shell Scripts and Dockerfiles](#shell-scripts-and-dockerfiles)
    - [Python](#python)
    - [C/C++](#cc)
    - [YAML and Configuration Files](#yaml-and-configuration-files)
  - [Comment Types and Usage](#comment-types-and-usage)
    - [Inline Comments](#inline-comments)
    - [Block Comments](#block-comments)
    - [Section Headers](#section-headers)
    - [File Headers](#file-headers)
  - [Status and TODO Comments](#status-and-todo-comments)
    - [Standard Formats](#standard-formats)
    - [Priority Levels](#priority-levels)
    - [Context Requirements](#context-requirements)
  - [Technical Explanations](#technical-explanations)
    - [Implementation Details](#implementation-details)
    - [Performance Notes](#performance-notes)
    - [Security Considerations](#security-considerations)
  - [What to Avoid](#what-to-avoid)
    - [Decorative Elements](#decorative-elements)
    - [Obvious Comments](#obvious-comments)
    - [Marketing Language](#marketing-language)
  - [Examples](#examples)
    - [Good Comment Examples](#good-comment-examples)
    - [Poor Comment Examples](#poor-comment-examples)
  - [Integration with Existing Style Guides](#integration-with-existing-style-guides)

## Core Principles

**Linux Philosophy Applied to Comments**
- **Clarity over decoration**: Comments explain "why" and "what", not "how" when obvious
- **Technical precision**: Use accurate technical terminology
- **Professional tone**: No marketing language, emojis, or casual expressions
- **Minimal yet sufficient**: Include necessary context without verbosity

**Comments Serve Three Purposes**
1. **Intent explanation**: Why this approach was chosen
2. **Context provision**: Information not obvious from code
3. **Maintenance guidance**: Critical information for future modifications

## Universal Standards

### Professional Tone

**Use technical language appropriate for the domain:**
```
# Configure TCP keepalive for persistent connections
# Allocate 2MB huge pages for reduced TLB pressure
# Validate input parameters before processing
```

**Avoid casual or marketing language:**
```
# Don't: Make our app super fast and awesome!
# Do: Optimize memory access patterns for performance
```

### Technical Accuracy

**Be precise about technical concepts:**
```
# Enable AVX-512 VNNI instructions for INT8 matrix operations
# Set file descriptor limit to handle concurrent connections
# Use copy-on-write semantics to reduce memory overhead
```

**Provide measurable context when relevant:**
```
# Reduces TLB misses by 95% for large model inference
# Increases throughput from 150 to 300 tokens/second
# Limits memory usage to 16GB maximum
```

### Conciseness

**Single line for simple explanations:**
```bash
# Disable swap to prevent model paging
export CUDA_VISIBLE_DEVICES=0  # Use first GPU only
```

**Multiple lines for complex concepts:**
```cpp
// Memory mapping wrapper for huge page support
// Maps file contents to anonymous huge page memory
// Returns pointer to huge page allocation instead of file mapping
void* intercept_mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset)
```

## Language-Specific Syntax

### Shell Scripts and Dockerfiles

**Use hash syntax with space after #:**
```bash
# Install build dependencies
apt-get update && apt-get install -y build-essential

# Set environment for AMD Zen 5 optimization
export CFLAGS="-march=znver5 -mtune=znver5 -O3"
```

**Multi-line explanations:**
```dockerfile
# Build llama.cpp with AOCL integration
# AOCL provides AMD-optimized BLAS routines
# Generic BLAS interface maintains compatibility
RUN cd /tmp/llama.cpp && mkdir build && cd build && cmake ..
```

### Python

**Follow existing Python style guide for docstrings. Use hash for inline comments:**
```python
# Configure GPU memory allocation strategy
torch.cuda.set_per_process_memory_fraction(0.95)

def optimize_model(model: torch.nn.Module) -> torch.nn.Module:
    """Optimize model for inference on AMD Zen 5 architecture.

    Applies quantization and AOCL acceleration.
    """
    # Enable AOCL BLAS backend for linear operations
    torch.backends.blas.set_backend('aocl')
    return model
```

### C/C++

**Use // for single line, /* */ for blocks:**
```cpp
// Function pointer to original mmap implementation
static mmap_fn real_mmap = nullptr;

/*
 * Intercept mmap calls for huge page optimization
 *
 * Checks file size and allocates huge pages for files larger than 1GB.
 * Falls back to standard mmap for smaller allocations.
 */
void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset)
```

### YAML and Configuration Files

**Use hash syntax, align with indentation:**
```yaml
services:
  llama-gpu:
    # GPU service for high-throughput inference
    image: llama-gpu:latest
    deploy:
      resources:
        reservations:
          devices:
            # Assign RTX 5090 with full capabilities
            - driver: nvidia
              count: 1
              capabilities: [gpu, compute]
```

## Comment Types and Usage

### Inline Comments

**Explain non-obvious decisions:**
```bash
ulimit -l unlimited  # Allow memory locking for model data
echo never > /sys/kernel/mm/transparent_hugepage/enabled  # Disable THP conflicts
```

**Provide context for magic numbers:**
```cpp
#define HUGE_PAGE_SIZE (2ULL << 20)  // 2MB standard huge page size
static const int MAX_CONNECTIONS = 1024;  // TCP connection limit
```

### Block Comments

**Use for complex algorithms or lengthy explanations:**
```cpp
/*
 * Huge page allocation strategy for large models
 *
 * 1. Check if file size exceeds 1GB threshold
 * 2. Allocate anonymous memory with MAP_HUGETLB flag
 * 3. Copy file contents to huge page memory via pread()
 * 4. Track allocation for proper cleanup in munmap
 *
 * Benefits: 95% reduction in TLB misses for 15GB+ models
 * Tradeoff: Additional memory copy during model loading
 */
```

### Section Headers

**Use simple text headers without decoration:**
```dockerfile
# Runtime environment configuration
FROM debian:unstable-slim

# Security hardening
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Application deployment
COPY --from=builder /app/server /app/
```

### File Headers

**Include purpose and key implementation details:**
```cpp
/*
 * hugepage_mmap_wrapper.cpp
 *
 * LD_PRELOAD library for transparent huge page allocation.
 * Intercepts mmap() calls and substitutes huge page memory
 * for large file mappings to reduce TLB pressure.
 */
```

## Status and TODO Comments

### Standard Formats

**Use consistent prefixes:**
```bash
# TODO: Implement CUDA 13.0 support when available
# FIXME: Memory leak in cleanup_allocations() function
# NOTE: Requires kernel 5.4+ for MAP_HUGE_1GB support
# WARNING: Clock locking reduces performance by 38%
```

### Priority Levels

**Include priority and context:**
```python
# TODO(high): Add FP8 quantization support for RTX 5090
# FIXME(critical): Race condition in concurrent model loading
# NOTE(documentation): Update performance benchmarks after optimization
```

### Context Requirements

**Provide enough information for action:**
```dockerfile
# TODO: Update base image to CUDA 13.0 when vLLM supports it
# Tracking: https://github.com/vllm-project/vllm/pull/23976
# Current blocker: CUB library API changes break compilation
```

## Technical Explanations

### Implementation Details

**Explain architectural decisions:**
```bash
# Use explicit huge pages instead of THP for predictable performance
# THP may fragment memory and cause allocation failures
echo 15360 > /proc/sys/vm/nr_hugepages
```

### Performance Notes

**Include measurable impact:**
```cpp
// Batch size optimization for AMD Zen 5 memory subsystem
// Tested values: 1024 (baseline), 2048 (+15%), 4096 (-5%)
const size_t OPTIMAL_BATCH_SIZE = 2048;
```

### Security Considerations

**Document security implications:**
```yaml
# Run as non-root user to prevent privilege escalation
# Disable shell access with /sbin/nologin
user: "appuser"
```

## What to Avoid

### Decorative Elements

**Never use decorative borders or emphasis:**
```bash
# Wrong:
# ============================================================================
# SUPER IMPORTANT CONFIGURATION SECTION
# ============================================================================

# Correct:
# Container runtime configuration
```

### Obvious Comments

**Avoid stating what code obviously does:**
```python
# Wrong:
x = x + 1  # Increment x by 1

# Correct:
x = x + 1  # Advance to next token position
```

### Marketing Language

**Use technical terminology:**
```dockerfile
# Wrong:
# This amazing optimization makes everything blazingly fast!

# Correct:
# AOCL BLAS integration reduces matrix operation latency by 40%
```

## Examples

### Good Comment Examples

**Dockerfile:**
```dockerfile
# Install AMD Optimized CPU Libraries for enhanced mathematical performance
COPY docker/llama-cpu/aocl-linux-gcc-5.1.0_1_amd64.deb /tmp/aocl.deb

# Create BLAS compatibility symlink for generic interface
RUN ln -s /opt/aocl_libs/libblis.so /opt/aocl_libs/libblas.so.3
```

**Shell script:**
```bash
# Pin GPU interrupts to cores 24-31 for better isolation
for irq in $(cat /proc/interrupts | grep nvidia | awk '{print $1}' | tr -d ':'); do
    echo f0000000 > /proc/irq/$irq/smp_affinity
done
```

**C++ code:**
```cpp
// Check file size threshold for huge page allocation
// Only files larger than 1GB benefit from huge pages due to allocation overhead
if (stat.st_size >= (1ULL << 30)) {
    return allocate_huge_pages(length);
}
```

### Poor Comment Examples

**Decorative and unprofessional:**
```dockerfile
# ============================================================================
# 🚀 SUPER AWESOME GPU OPTIMIZATION SECTION 🚀
# ============================================================================
# This section contains cutting-edge optimizations that will make your
# models run at lightning speed! Get ready to be amazed!
```

**Obvious and redundant:**
```bash
# Set variable to true
ENABLE_GPU=true

# Call the function
start_server()
```

**Vague and non-technical:**
```cpp
// Make it faster
optimize_performance();

// Fix the bug
if (error) handle_error();
```

## Integration with Existing Style Guides

**This comment style guide complements:**
- `.claude/DOCUMENTATION_STYLE.md`: For markdown documentation files
- `.claude/BASH_OUTPUT_STYLE.md`: For script output formatting
- `.claude/PYTHON_STYLE.md`: For Python code structure and docstrings

**Unified approach:** All guides follow Unix/Linux philosophy of technical precision, professional presentation, and functional clarity without decorative elements.

---

*Comments are documentation embedded in code. They should maintain the same professional standards as external documentation while serving the specific needs of developers working with the implementation.*