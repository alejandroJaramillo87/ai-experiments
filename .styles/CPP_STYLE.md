# C++ Style Guide

This guide defines C++ coding standards for system-level optimization libraries in the llama-zen-turbo project. All C++ code must follow these conventions to maintain professional quality and Unix compatibility.

## Table of Contents

- [Core Philosophy](#core-philosophy)
- [Code Standards](#code-standards)
  - [Naming Conventions](#naming-conventions)
  - [File Organization](#file-organization)
  - [Include Management](#include-management)
- [Output Formatting](#output-formatting)
  - [Console Output](#console-output)
  - [Error Messages](#error-messages)
  - [Debug Output](#debug-output)
- [Memory Management](#memory-management)
  - [RAII Principles](#raii-principles)
  - [Allocation Tracking](#allocation-tracking)
  - [Error Recovery](#error-recovery)
- [Library Interception](#library-interception)
  - [Function Hooking](#function-hooking)
  - [Symbol Resolution](#symbol-resolution)
  - [ABI Compatibility](#abi-compatibility)
- [Performance Guidelines](#performance-guidelines)
  - [Hot Path Optimization](#hot-path-optimization)
  - [Minimal Overhead](#minimal-overhead)
  - [Cache Considerations](#cache-considerations)
- [Error Handling](#error-handling)
  - [Error Reporting](#error-reporting)
  - [Fallback Strategies](#fallback-strategies)
  - [Resource Cleanup](#resource-cleanup)
- [Testing Conventions](#testing-conventions)
- [Examples](#examples)
  - [Complete Interceptor Example](#complete-interceptor-example)
  - [Error Handling Example](#error-handling-example)
- [Prohibited Practices](#prohibited-practices)
- [Quality Checklist](#quality-checklist)

## Core Philosophy

### Unix Principles Applied to System Programming

All C++ code follows traditional Unix system programming principles:

- **Zero overhead** - Optimizations must not slow down non-optimized paths
- **Transparent** - Intercepted functions behave identically to originals unless optimizing
- **Fail safe** - Always fall back to original behavior on error
- **Silent operation** - Minimal output unless debugging enabled
- **Professional** - No decorative elements, marketing language, or emojis
- **Technical precision** - Use accurate system programming terminology

### LD_PRELOAD Library Principles

```cpp
// Good: Clear interception with fallback
extern "C" void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    if (should_intercept(fd, length)) {
        return optimized_mmap(addr, length, prot, flags, fd, offset);
    }
    return real_mmap(addr, length, prot, flags, fd, offset);
}

// Bad: No fallback, unclear behavior
extern "C" void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    // Always forces our implementation
    return my_cool_mmap(addr, length, prot, flags, fd, offset);
}
```

## Code Standards

### Naming Conventions

Follow Linux kernel style naming conventions:

```cpp
// Functions: snake_case
void* allocate_huge_pages(size_t size);
static bool should_use_hugepages(int fd, size_t length);

// Structs and Classes: PascalCase
struct HugePageAllocation {
    void* addr;
    size_t size;
    HugePageAllocation* next;
};

// Type aliases: snake_case with _t suffix
typedef void* (*mmap_fn_t)(void*, size_t, int, int, int, off_t);

// Constants: UPPERCASE
const size_t MIN_HUGEPAGE_SIZE = 1ULL << 30;  // 1GB
#define MAX_ALLOCATIONS 1024

// Global variables (if needed): g_ prefix
static HugePageAllocation* g_allocations = nullptr;

// Member variables: trailing underscore
class MemoryTracker {
private:
    size_t total_allocated_;
    size_t peak_usage_;
};

// Namespaces: lowercase
namespace zen5_turbo {
    namespace memory {
        // Implementation
    }
}
```

### File Organization

Standard file structure for headers and implementation:

```cpp
/*
 * filename.cpp or filename.h
 *
 * Brief description of file purpose.
 * Additional implementation notes if necessary.
 */

// Header guards for .h files
#ifndef ZEN5_TURBO_MODULE_NAME_H
#define ZEN5_TURBO_MODULE_NAME_H

// System includes first
#include <sys/mman.h>
#include <unistd.h>

// Library includes second
#include <dlfcn.h>

// Project includes last
#include "config.h"
#include "utils.h"

// Content

#endif  // ZEN5_TURBO_MODULE_NAME_H
```

### Include Management

Minimal and ordered includes:

```cpp
// Good: Ordered, minimal includes
#include <sys/mman.h>      // For mmap, munmap
#include <errno.h>         // For errno
#include <string.h>        // For strerror

#include "config.h"        // Project configuration

// Bad: Unordered, unnecessary includes
#include <iostream>        // Avoid C++ streams in system libraries
#include <vector>          // Avoid STL in intercepted paths
#include "everything.h"    // Avoid umbrella headers
```

## Output Formatting

### Console Output

All output to stderr with consistent formatting:

```cpp
// Define consistent prefix
#define LOG_PREFIX "[zen5-optimizer]"

// Good: Technical, parseable output
fprintf(stderr, "%s Allocated %.2f GB with MAP_HUGETLB\n",
        LOG_PREFIX, size / (1024.0 * 1024.0 * 1024.0));

fprintf(stderr, "%s Memory allocation: OK (2MB pages)\n", LOG_PREFIX);

// Bad: Marketing language, emojis
printf("🚀 Awesome allocation successful! 🚀\n");
fprintf(stdout, "AMAZING: Your memory is now TURBO CHARGED!!!\n");
```

### Error Messages

Clear, actionable error reporting:

```cpp
// Good: Context, error detail, and solution
if (huge_mem == MAP_FAILED) {
    fprintf(stderr, "%s ERROR: mmap failed: %s\n",
            LOG_PREFIX, strerror(errno));
    fprintf(stderr, "  Context: Allocating %.2f GB with MAP_HUGETLB\n",
            length / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "  Solution: Check huge page availability\n");
    fprintf(stderr, "  Command: cat /proc/meminfo | grep HugePages\n");
    return nullptr;
}

// Bad: Vague, unhelpful errors
if (huge_mem == MAP_FAILED) {
    printf("ERROR: Something went wrong!\n");
    return nullptr;
}
```

### Debug Output

Conditional debug messages:

```cpp
// Good: Conditional, informative debug output
#ifdef DEBUG_OUTPUT
#define DEBUG_PRINT(fmt, ...) \
    fprintf(stderr, "%s DEBUG: " fmt "\n", LOG_PREFIX, ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...) do {} while(0)
#endif

// Usage
DEBUG_PRINT("Intercepting mmap for fd=%d, size=%zu", fd, length);

// Bad: Always-on debug output
fprintf(stderr, "DEBUG: Entering function mmap\n");
fprintf(stderr, "DEBUG: fd = %d\n", fd);
fprintf(stderr, "DEBUG: This is a debug message\n");
```

## Memory Management

### RAII Principles

Use RAII for automatic resource management:

```cpp
// Good: RAII wrapper for allocations
class ScopedHugePageAllocation {
public:
    explicit ScopedHugePageAllocation(size_t size)
        : addr_(allocate_huge_pages(size)), size_(size) {}

    ~ScopedHugePageAllocation() {
        if (addr_ && addr_ != MAP_FAILED) {
            munmap(addr_, size_);
        }
    }

    // Delete copy operations
    ScopedHugePageAllocation(const ScopedHugePageAllocation&) = delete;
    ScopedHugePageAllocation& operator=(const ScopedHugePageAllocation&) = delete;

    void* get() const { return addr_; }
    void release() { addr_ = nullptr; }

private:
    void* addr_;
    size_t size_;
};

// Bad: Manual resource management
void* addr = mmap(...);
// ... lots of code with multiple return paths ...
// Easy to forget cleanup
munmap(addr, size);
```

### Allocation Tracking

Track intercepted allocations for proper cleanup:

```cpp
// Good: Linked list tracking with clear ownership
struct AllocationNode {
    void* addr;
    size_t size;
    AllocationNode* next;
};

static AllocationNode* g_allocations = nullptr;
static pthread_mutex_t g_allocations_mutex = PTHREAD_MUTEX_INITIALIZER;

static void track_allocation(void* addr, size_t size) {
    AllocationNode* node = (AllocationNode*)malloc(sizeof(AllocationNode));
    if (!node) return;  // Fail silently, don't break application

    node->addr = addr;
    node->size = size;

    pthread_mutex_lock(&g_allocations_mutex);
    node->next = g_allocations;
    g_allocations = node;
    pthread_mutex_unlock(&g_allocations_mutex);
}
```

### Error Recovery

Clean error paths without leaks:

```cpp
// Good: Clean error handling with proper cleanup
void* optimized_mmap(size_t length) {
    void* huge_mem = mmap(nullptr, length, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);

    if (huge_mem == MAP_FAILED) {
        // Try fallback without MAP_HUGETLB
        huge_mem = mmap(nullptr, length, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        if (huge_mem == MAP_FAILED) {
            // Complete failure, return error
            return MAP_FAILED;
        }
    }

    // Success path
    track_allocation(huge_mem, length);
    return huge_mem;
}
```

## Library Interception

### Function Hooking

Standard pattern for function interception:

```cpp
// Good: Lazy initialization, proper function pointer type
typedef void* (*mmap_fn_t)(void*, size_t, int, int, int, off_t);
static mmap_fn_t real_mmap = nullptr;

static void init_real_functions() {
    if (!real_mmap) {
        real_mmap = (mmap_fn_t)dlsym(RTLD_NEXT, "mmap");
        if (!real_mmap) {
            fprintf(stderr, "%s ERROR: Failed to find real mmap: %s\n",
                   LOG_PREFIX, dlerror());
            abort();  // Fatal error, cannot continue
        }
    }
}

extern "C" void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    init_real_functions();

    // Optimization logic
    if (should_optimize(fd, length)) {
        return optimized_mmap(addr, length, prot, flags, fd, offset);
    }

    // Delegate to real function
    return real_mmap(addr, length, prot, flags, fd, offset);
}
```

### Symbol Resolution

Proper dlsym usage:

```cpp
// Good: Check for errors, use correct casting
void* symbol = dlsym(RTLD_NEXT, "function_name");
if (!symbol) {
    const char* error = dlerror();
    fprintf(stderr, "%s ERROR: dlsym failed: %s\n", LOG_PREFIX, error);
    return;
}
function_ptr = reinterpret_cast<function_type>(symbol);

// Bad: No error checking, C-style cast
function_ptr = (function_type)dlsym(RTLD_NEXT, "function_name");
```

### ABI Compatibility

Maintain C ABI compatibility:

```cpp
// Good: extern "C" for all intercepted functions
extern "C" {
    void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset);
    int munmap(void* addr, size_t length);
}

// Good: No exceptions across library boundaries
extern "C" void* mmap(...) {
    try {
        // Internal C++ code may use exceptions
        return internal_mmap(...);
    } catch (...) {
        // Never let exceptions escape
        errno = ENOMEM;
        return MAP_FAILED;
    }
}

// Bad: C++ features in C interface
extern "C" void* mmap(std::string path, size_t length);  // Wrong
extern "C" void* mmap(...) {
    throw std::runtime_error("error");  // Never do this
}
```

## Performance Guidelines

### Hot Path Optimization

Minimize overhead in frequently called paths:

```cpp
// Good: Inline for hot paths, minimal checks
inline bool should_use_hugepages(int fd, size_t length) {
    // Single comparison for common case
    return fd >= 0 && length >= MIN_HUGEPAGE_SIZE;
}

// Good: Early return for common case
extern "C" void* mmap(...) {
    // Fast path: small allocations
    if (length < MIN_HUGEPAGE_SIZE) {
        return real_mmap(addr, length, prot, flags, fd, offset);
    }

    // Slow path: optimization logic
    return optimized_mmap(...);
}
```

### Minimal Overhead

Avoid unnecessary work:

```cpp
// Good: Lazy initialization
static void init_once() {
    static bool initialized = false;
    if (!initialized) {
        perform_initialization();
        initialized = true;
    }
}

// Bad: Work on every call
void* mmap(...) {
    load_configuration();  // Don't reload every time
    check_cpu_features();   // Check once at startup
    // ...
}
```

### Cache Considerations

Optimize for cache locality:

```cpp
// Good: Pack related data together
struct AllocationInfo {
    void* addr;
    size_t size;
    uint32_t flags;
    uint32_t padding;  // Explicit padding for alignment
} __attribute__((packed));

// Good: Minimize cache line bouncing
alignas(64) static AllocationInfo g_allocation_cache[MAX_CACHED];

// Bad: Scattered data access
struct BadLayout {
    void* addr;
    char padding[4096];  // Wastes cache
    size_t size;
};
```

## Error Handling

### Error Reporting

Consistent error reporting pattern:

```cpp
// Good: Set errno appropriately, return standard error values
extern "C" void* mmap(...) {
    if (invalid_parameters) {
        errno = EINVAL;
        return MAP_FAILED;
    }

    void* result = perform_mmap(...);
    if (result == MAP_FAILED) {
        // errno already set by perform_mmap
        DEBUG_PRINT("mmap failed: %s", strerror(errno));
        return MAP_FAILED;
    }

    return result;
}

// Bad: Inconsistent error values
void* mmap(...) {
    if (error) return nullptr;  // Should return MAP_FAILED
    if (other_error) return (void*)-1;  // Inconsistent
}
```

### Fallback Strategies

Always provide fallback to original behavior:

```cpp
// Good: Multiple fallback levels
void* allocate_optimized_memory(size_t size) {
    // Try 1GB huge pages
    void* mem = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGE_1GB, -1, 0);

    if (mem == MAP_FAILED) {
        // Fall back to 2MB huge pages
        mem = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGE_2MB, -1, 0);
    }

    if (mem == MAP_FAILED) {
        // Fall back to regular pages
        mem = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    }

    return mem;
}
```

### Resource Cleanup

Ensure cleanup even on error paths:

```cpp
// Good: Cleanup on all paths
void* complex_allocation(size_t size) {
    void* temp_buffer = malloc(size);
    if (!temp_buffer) {
        return nullptr;
    }

    void* final_buffer = mmap(...);
    if (final_buffer == MAP_FAILED) {
        free(temp_buffer);  // Clean up temporary resources
        return nullptr;
    }

    // Process data
    memcpy(final_buffer, temp_buffer, size);
    free(temp_buffer);  // Clean up after success

    return final_buffer;
}
```

## Testing Conventions

### Test Output Standards

All test programs must follow the test output style guide:
- See `includes/TEST_OUTPUT_STYLE.md` for comprehensive output formatting rules
- Use consistent status indicators ([OK], [FAIL], [WARN], etc.)
- Follow Unix philosophy: simple, parseable, exit-code-driven

### Unit Test Structure

```cpp
// Good: Clear test structure
void test_hugepage_allocation() {
    // Setup
    size_t test_size = 2ULL << 30;  // 2GB

    // Execute
    void* result = allocate_huge_pages(test_size);

    // Verify
    assert(result != MAP_FAILED);
    assert(is_hugepage_backed(result));

    // Cleanup
    munmap(result, test_size);
}

// Good: Test error conditions
void test_allocation_failure() {
    // Test with invalid size
    void* result = allocate_huge_pages(0);
    assert(result == MAP_FAILED);
    assert(errno == EINVAL);
}
```

## Examples

### Complete Interceptor Example

```cpp
/*
 * mmap_interceptor.cpp
 *
 * Intercepts mmap calls to provide transparent huge page support.
 */

#include <sys/mman.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>

#define LOG_PREFIX "[zen5-optimizer]"

// Function pointer type
typedef void* (*mmap_fn_t)(void*, size_t, int, int, int, off_t);

// Real function pointer
static mmap_fn_t real_mmap = nullptr;

// Configuration
static const size_t MIN_HUGEPAGE_SIZE = 1ULL << 30;  // 1GB

// Initialize real function pointer
static void init_real_mmap() {
    if (!real_mmap) {
        real_mmap = (mmap_fn_t)dlsym(RTLD_NEXT, "mmap");
        if (!real_mmap) {
            fprintf(stderr, "%s ERROR: Cannot find real mmap\n", LOG_PREFIX);
            abort();
        }
    }
}

// Check if we should use huge pages
static inline bool should_use_hugepages(int fd, size_t length) {
    return fd >= 0 && length >= MIN_HUGEPAGE_SIZE;
}

// Optimized mmap with huge pages
static void* mmap_with_hugepages(void* addr, size_t length, int prot,
                                 int flags, int fd, off_t offset) {
    // Allocate huge pages
    void* huge_mem = real_mmap(nullptr, length, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);

    if (huge_mem == MAP_FAILED) {
        // Fallback to regular mmap
        return real_mmap(addr, length, prot, flags, fd, offset);
    }

    // Read file contents into huge page memory
    FILE* file = fdopen(fd, "rb");
    if (!file) {
        munmap(huge_mem, length);
        return real_mmap(addr, length, prot, flags, fd, offset);
    }

    size_t bytes_read = fread(huge_mem, 1, length, file);
    if (bytes_read != length) {
        munmap(huge_mem, length);
        return real_mmap(addr, length, prot, flags, fd, offset);
    }

    return huge_mem;
}

// Intercepted mmap function
extern "C" void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    init_real_mmap();

    if (should_use_hugepages(fd, length)) {
        void* result = mmap_with_hugepages(addr, length, prot, flags, fd, offset);
        if (result != MAP_FAILED) {
            fprintf(stderr, "%s Allocated %.2f GB with huge pages\n",
                   LOG_PREFIX, length / (1024.0 * 1024.0 * 1024.0));
            return result;
        }
    }

    return real_mmap(addr, length, prot, flags, fd, offset);
}
```

### Error Handling Example

```cpp
/*
 * Error handling patterns for system libraries
 */

#include <errno.h>
#include <string.h>

// Good: Comprehensive error handling
void* allocate_memory_with_fallback(size_t size, int numa_node) {
    // Validate input
    if (size == 0) {
        errno = EINVAL;
        fprintf(stderr, "%s ERROR: Invalid size: 0\n", LOG_PREFIX);
        return nullptr;
    }

    // Try NUMA-aware allocation
    void* mem = numa_alloc_onnode(size, numa_node);
    if (!mem) {
        int saved_errno = errno;
        fprintf(stderr, "%s WARN: NUMA allocation failed: %s\n",
                LOG_PREFIX, strerror(saved_errno));
        fprintf(stderr, "  Attempting standard allocation\n");

        // Fallback to standard allocation
        mem = malloc(size);
        if (!mem) {
            errno = saved_errno;  // Preserve original error
            fprintf(stderr, "%s ERROR: All allocation methods failed\n", LOG_PREFIX);
            return nullptr;
        }
    }

    return mem;
}
```

## Prohibited Practices

### Never Use in System Libraries

- ❌ **Exceptions in extern "C" functions** - Breaks ABI compatibility
- ❌ **C++ streams (iostream)** - Too much overhead, not async-signal-safe
- ❌ **STL containers in hot paths** - Hidden allocations, unpredictable performance
- ❌ **Global constructors with side effects** - Initialization order issues
- ❌ **Emojis or Unicode in output** - Not terminal-safe
- ❌ **Marketing language** - Unprofessional in system libraries
- ❌ **new/delete** - Use malloc/free for C compatibility
- ❌ **RTTI or dynamic_cast** - Overhead and ABI issues

### Alternatives

| Prohibited | Use Instead |
|-----------|-------------|
| `std::cout << "Error"` | `fprintf(stderr, "Error\n")` |
| `throw std::exception()` | `errno = EINVAL; return -1;` |
| `std::vector<int>` | `int* array = malloc(size * sizeof(int))` |
| `new char[size]` | `malloc(size)` |
| `🚀 Success!` | `Operation completed successfully` |

## Quality Checklist

Before committing C++ code:

### Code Structure
- [ ] Functions follow snake_case naming
- [ ] Types follow PascalCase naming
- [ ] Constants are UPPERCASE
- [ ] Headers have include guards

### Output and Logging
- [ ] All output goes to stderr
- [ ] Uses consistent LOG_PREFIX
- [ ] No emojis or decorative elements
- [ ] Error messages include context and solutions

### Memory Management
- [ ] All allocations have corresponding deallocations
- [ ] Error paths clean up resources
- [ ] No memory leaks on any code path
- [ ] RAII used where appropriate

### Library Interception
- [ ] Uses extern "C" for intercepted functions
- [ ] Proper fallback to real functions
- [ ] No exceptions cross library boundaries
- [ ] Thread-safe where necessary

### Performance
- [ ] Hot paths optimized for minimal overhead
- [ ] Early returns for common cases
- [ ] Cache-friendly data structures
- [ ] No unnecessary work in intercepted paths

### Error Handling
- [ ] Sets errno appropriately
- [ ] Returns standard error values
- [ ] Provides clear error messages
- [ ] Has fallback strategies

### Testing
- [ ] Unit tests for main functionality
- [ ] Tests error conditions
- [ ] Tests fallback behavior
- [ ] No test dependencies on system state

### Final Review
- [ ] Code follows Unix philosophy
- [ ] No prohibited practices present
- [ ] Professional appearance maintained
- [ ] Compatible with C code

---

*This guide ensures C++ code maintains the professional, efficient standards required for system-level optimization libraries.*

*Last Updated: 2025-09-27*