# Comment Style Guide

Unified comment standards across all file types. Follow Unix/Linux principles: clarity over
decoration, technical precision, professional tone (no marketing, emojis, or casual language),
minimal yet sufficient.

Comments explain **why** (intent, chosen approach), provide **context** not obvious from code,
and give **maintenance guidance** — not "how" when the code already shows it.

## Universal Standards

- Use accurate technical terminology: "Allocate 2MB huge pages for reduced TLB pressure", not
  "Make our app super fast".
- Provide measurable context when relevant: "Reduces TLB misses by 95% for large model inference".
- Single line for simple explanations; multiple lines for complex concepts.

## Language-Specific Syntax

```bash
# Shell/Dockerfile: hash with a space after #
export CFLAGS="-march=znver5 -mtune=znver5 -O3"   # AMD Zen 5 optimization
```

```python
# Python: hash for inline comments; docstrings per Python style guide
torch.cuda.set_per_process_memory_fraction(0.95)  # GPU memory allocation strategy
```

```cpp
// C/C++: // for single line, /* */ for blocks
static mmap_fn real_mmap = nullptr;   // pointer to original mmap
```

```yaml
# YAML: hash syntax, aligned with indentation
services:
  llama-gpu:            # GPU service for high-throughput inference
    image: llama-gpu:latest
```

## Comment Types

**Inline** — explain non-obvious decisions and magic numbers:

```bash
ulimit -l unlimited  # Allow memory locking for model data
```
```cpp
#define HUGE_PAGE_SIZE (2ULL << 20)  // 2MB standard huge page size
```

**Block** — complex algorithms or lengthy explanations:

```cpp
/*
 * Huge page allocation strategy for large models
 *
 * 1. Check if file size exceeds 1GB threshold
 * 2. Allocate anonymous memory with MAP_HUGETLB flag
 * 3. Copy file contents to huge page memory via pread()
 *
 * Benefits: 95% reduction in TLB misses for 15GB+ models
 * Tradeoff: additional memory copy during model loading
 */
```

**Section headers** — simple text, no decoration. **File headers** — state purpose and key
implementation details:

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

Use consistent prefixes, include priority and enough context to act:

```bash
# TODO: Implement CUDA 13.0 support when available
# FIXME(critical): Race condition in concurrent model loading
# NOTE: Requires kernel 5.4+ for MAP_HUGE_1GB support
# WARNING: Clock locking reduces performance by 38%
```

```dockerfile
# TODO: Update base image to CUDA 13.0 when vLLM supports it
# Tracking: https://github.com/vllm-project/vllm/pull/23976
# Current blocker: CUB library API changes break compilation
```

## Technical Explanations

Explain architectural decisions, include measurable impact, and document security implications:

```cpp
// Batch size optimization for AMD Zen 5 memory subsystem
// Tested values: 1024 (baseline), 2048 (+15%), 4096 (-5%)
const size_t OPTIMAL_BATCH_SIZE = 2048;
```
```yaml
# Run as non-root user to prevent privilege escalation
user: "appuser"
```

## What to Avoid

- Decorative borders or emphasis banners (`# ==== SUPER IMPORTANT ====`)
- Obvious comments (`x = x + 1  # Increment x`) — comment intent instead (`# Advance to next token`)
- Marketing/vague language ("blazingly fast", "make it faster") — state the technical effect

## Related Guides

- `.styles/DOCUMENTATION_STYLE.md` — markdown documentation
- `.styles/BASH_OUTPUT_STYLE.md` — script output formatting
