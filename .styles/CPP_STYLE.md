# C++ Style Guide

Standards for system-level optimization libraries in llama-zen-turbo. Unix system-programming
principles apply:

- **Zero overhead** — optimizations must not slow non-optimized paths.
- **Transparent** — intercepted functions behave identically to originals unless optimizing.
- **Fail safe** — always fall back to original behavior on error.
- **Silent** — minimal output unless debugging is enabled.
- Professional, technically precise; no decoration, marketing language, or emojis.

An LD_PRELOAD interceptor must always keep a working fallback:

```cpp
// Good: clear interception with fallback
extern "C" void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    if (should_intercept(fd, length))
        return optimized_mmap(addr, length, prot, flags, fd, offset);
    return real_mmap(addr, length, prot, flags, fd, offset);
}
```

## Naming (Linux kernel style)

```cpp
void* allocate_huge_pages(size_t size);           // functions: snake_case
struct HugePageAllocation { void* addr; };        // structs/classes: PascalCase
typedef void* (*mmap_fn_t)(void*, size_t, ...);   // type aliases: snake_case + _t
const size_t MIN_HUGEPAGE_SIZE = 1ULL << 30;      // constants: UPPERCASE
static HugePageAllocation* g_allocations;         // globals: g_ prefix
size_t total_allocated_;                          // members: trailing underscore
namespace zen5_turbo { }                          // namespaces: lowercase
```

## File Organization

```cpp
/*
 * filename.cpp — brief description of file purpose.
 */
#ifndef ZEN5_TURBO_MODULE_NAME_H          // header guards for .h files
#define ZEN5_TURBO_MODULE_NAME_H

#include <sys/mman.h>    // system includes first
#include <dlfcn.h>       // library includes second
#include "config.h"      // project includes last

#endif  // ZEN5_TURBO_MODULE_NAME_H
```

Keep includes minimal and ordered. Avoid C++ streams and STL in intercepted paths; avoid
umbrella headers.

## Output

All output goes to stderr with a consistent prefix. Error messages carry context and a solution:

```cpp
#define LOG_PREFIX "[zen5-optimizer]"

fprintf(stderr, "%s Allocated %.2f GB with MAP_HUGETLB\n",
        LOG_PREFIX, size / (1024.0 * 1024.0 * 1024.0));

if (huge_mem == MAP_FAILED) {
    fprintf(stderr, "%s ERROR: mmap failed: %s\n", LOG_PREFIX, strerror(errno));
    fprintf(stderr, "  Context: Allocating %.2f GB with MAP_HUGETLB\n", gb);
    fprintf(stderr, "  Solution: Check huge page availability\n");
    fprintf(stderr, "  Command: cat /proc/meminfo | grep HugePages\n");
    return nullptr;
}
```

Debug output is conditional, never always-on:

```cpp
#ifdef DEBUG_OUTPUT
#define DEBUG_PRINT(fmt, ...) fprintf(stderr, "%s DEBUG: " fmt "\n", LOG_PREFIX, ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...) do {} while(0)
#endif
```

## Memory Management

Use RAII for automatic cleanup; delete copy operations on owning wrappers:

```cpp
class ScopedHugePageAllocation {
public:
    explicit ScopedHugePageAllocation(size_t size)
        : addr_(allocate_huge_pages(size)), size_(size) {}
    ~ScopedHugePageAllocation() {
        if (addr_ && addr_ != MAP_FAILED) munmap(addr_, size_);
    }
    ScopedHugePageAllocation(const ScopedHugePageAllocation&) = delete;
    ScopedHugePageAllocation& operator=(const ScopedHugePageAllocation&) = delete;
    void* get() const { return addr_; }
private:
    void* addr_;
    size_t size_;
};
```

Track intercepted allocations for cleanup (guard shared state with a mutex; fail silently on
tracking-malloc failure rather than breaking the application). Error paths must free every
resource they acquired.

## Library Interception

Standard hooking pattern — lazy init, correct function-pointer type, checked `dlsym`:

```cpp
typedef void* (*mmap_fn_t)(void*, size_t, int, int, int, off_t);
static mmap_fn_t real_mmap = nullptr;

static void init_real_functions() {
    if (!real_mmap) {
        real_mmap = (mmap_fn_t)dlsym(RTLD_NEXT, "mmap");
        if (!real_mmap) {
            fprintf(stderr, "%s ERROR: Failed to find real mmap: %s\n", LOG_PREFIX, dlerror());
            abort();   // fatal, cannot continue
        }
    }
}
```

Use `reinterpret_cast` for symbols (not C-style casts) and check `dlerror()`.

**ABI compatibility**: use `extern "C"` for all intercepted functions and never let exceptions
cross the library boundary:

```cpp
extern "C" void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    try {
        return internal_mmap(addr, length, prot, flags, fd, offset);
    } catch (...) {
        errno = ENOMEM;
        return MAP_FAILED;   // never let exceptions escape
    }
}
```

## Performance

Minimize overhead on hot paths — inline cheap checks, early-return the common case:

```cpp
inline bool should_use_hugepages(int fd, size_t length) {
    return fd >= 0 && length >= MIN_HUGEPAGE_SIZE;
}

extern "C" void* mmap(...) {
    if (length < MIN_HUGEPAGE_SIZE)            // fast path
        return real_mmap(addr, length, prot, flags, fd, offset);
    return optimized_mmap(...);                // slow path
}
```

Initialize once (lazy), not on every call. Pack related data for cache locality; align hot
structures to 64 bytes; avoid wasting cache lines with oversized padding.

## Error Handling

Set `errno` appropriately and return standard error values (`MAP_FAILED`, not `nullptr` or
`(void*)-1` inconsistently). Always provide a fallback chain, degrading gracefully:

```cpp
void* allocate_optimized_memory(size_t size) {
    void* mem = mmap(nullptr, size, PROT_READ|PROT_WRITE,
                     MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGE_1GB, -1, 0);
    if (mem == MAP_FAILED)                          // fall back to 2MB pages
        mem = mmap(nullptr, size, PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGE_2MB, -1, 0);
    if (mem == MAP_FAILED)                          // fall back to regular pages
        mem = mmap(nullptr, size, PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    return mem;
}
```

Clean up temporary resources on every error path (e.g. `free()` a temp buffer before returning
on a later `mmap` failure).

## Testing

Follow `.styles/TEST_OUTPUT_STYLE.md` for output. Structure tests as setup / execute / verify /
cleanup, and cover error and fallback conditions:

```cpp
void test_hugepage_allocation() {
    size_t test_size = 2ULL << 30;                  // 2GB
    void* result = allocate_huge_pages(test_size);
    assert(result != MAP_FAILED);
    assert(is_hugepage_backed(result));
    munmap(result, test_size);
}
```

For a complete real-world interceptor, see `hugepages/src/memory/hugepage_wrapper.cpp`.

## Prohibited in System Libraries

- Exceptions in `extern "C"` functions — breaks ABI compatibility
- C++ streams (iostream) — overhead, not async-signal-safe → use `fprintf(stderr, ...)`
- STL containers in hot paths — hidden allocations → use `malloc`/`free`
- `new`/`delete` — use `malloc`/`free` for C compatibility
- RTTI or `dynamic_cast` — overhead and ABI issues
- Global constructors with side effects — initialization-order issues
- Emojis, Unicode, or marketing language in output
