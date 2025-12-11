/*
 * hugepage_wrapper.cpp
 *
 * Intercepts mmap() calls to provide transparent huge page support.
 * Allocates anonymous huge page memory for large file mappings
 * to reduce TLB pressure during model inference.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "../config.h"

// Define 1GB hugepage flag if not present in system headers
#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif

namespace zen5_turbo {

// 1GB hugepage size for alignment
const size_t HUGEPAGE_1GB = 1ULL * 1024 * 1024 * 1024;

// Round up to nearest 1GB boundary (required for 1GB hugepages)
static size_t round_up_1gb(size_t size) {
    return ((size + HUGEPAGE_1GB - 1) / HUGEPAGE_1GB) * HUGEPAGE_1GB;
}

// Function pointer to the real mmap
typedef void* (*mmap_fn)(void*, size_t, int, int, int, off_t);
static mmap_fn real_mmap = nullptr;

// Function pointer to the real munmap
typedef int (*munmap_fn)(void*, size_t);
static munmap_fn real_munmap = nullptr;

// Track our anonymous huge page allocations
struct HugePageAllocation {
    void* addr;
    size_t alloc_size;  // Rounded to 1GB boundary (for munmap)
    size_t file_size;   // Original file size (for logging)
    HugePageAllocation* next;
};
static HugePageAllocation* allocations = nullptr;

// Initialize function pointers to real functions
static void init_functions() {
    if (!real_mmap) {
        real_mmap = (mmap_fn)dlsym(RTLD_NEXT, "mmap");
        if (!real_mmap) {
            fprintf(stderr, "[%s] ERROR: Failed to find real mmap: %s\n",
                    ZEN5_OPTIMIZER_NAME, dlerror());
            exit(1);
        }
    }
    if (!real_munmap) {
        real_munmap = (munmap_fn)dlsym(RTLD_NEXT, "munmap");
        if (!real_munmap) {
            fprintf(stderr, "[%s] ERROR: Failed to find real munmap: %s\n",
                    ZEN5_OPTIMIZER_NAME, dlerror());
            exit(1);
        }
    }
}

// Check if we should use huge pages for this file
static bool should_use_hugepages(int /*fd*/, size_t length) {
#if ENABLE_HUGEPAGES
    return length >= MIN_SIZE_FOR_HUGEPAGES;
#else
    return false;
#endif
}

// Track an allocation so we can handle munmap properly
static void track_allocation(void* addr, size_t alloc_size, size_t file_size) {
    HugePageAllocation* alloc = (HugePageAllocation*)malloc(sizeof(HugePageAllocation));
    alloc->addr = addr;
    alloc->alloc_size = alloc_size;
    alloc->file_size = file_size;
    alloc->next = allocations;
    allocations = alloc;
}

// Find and remove a tracked allocation, returns alloc_size for munmap
static size_t untrack_allocation(void* addr) {
    HugePageAllocation** prev = &allocations;
    HugePageAllocation* curr = allocations;

    while (curr) {
        if (curr->addr == addr) {
            size_t alloc_size = curr->alloc_size;
            *prev = curr->next;
            free(curr);
            return alloc_size;
        }
        prev = &curr->next;
        curr = curr->next;
    }
    return 0;
}

// Cleanup function to be called on library unload
void cleanup_hugepage_allocations() {
    while (allocations) {
        HugePageAllocation* next = allocations->next;
        free(allocations);
        allocations = next;
    }
}

} // namespace zen5_turbo

// Our intercepted mmap function - must be extern "C" for LD_PRELOAD
extern "C" void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    using namespace zen5_turbo;

    init_functions();

    // Check if this is a file-backed mmap that could benefit from huge pages
    if (fd >= 0 && should_use_hugepages(fd, length)) {
        // Get file size to verify we're mapping the whole file
        struct stat st;
        if (fstat(fd, &st) != 0) {
            DEBUG_PRINT("WARNING: Failed to stat fd %d: %s", fd, strerror(errno));
            return real_mmap(addr, length, prot, flags, fd, offset);
        }

        // Only intercept if mapping the whole file from offset 0 (typical for model loading)
        if (offset == 0 && length == (size_t)st.st_size) {
            // Round up to 1GB boundary (required for 1GB hugepages)
            size_t alloc_size = round_up_1gb(length);
            DEBUG_PRINT("Intercepting mmap for %.2f GB file (allocating %.2f GB with 1GB huge pages)",
                    length / (1024.0 * 1024.0 * 1024.0),
                    alloc_size / (1024.0 * 1024.0 * 1024.0));

            // Allocate anonymous 1GB huge pages memory
            void* huge_mem = real_mmap(nullptr, alloc_size,
                                       PROT_READ | PROT_WRITE,
                                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB,
                                       -1, 0);

            bool using_hugepages = true;
            if (huge_mem == MAP_FAILED) {
                // Try without huge pages as fallback
                DEBUG_PRINT("1GB huge pages failed (%s), trying regular anonymous mmap", strerror(errno));
                using_hugepages = false;
                alloc_size = length;  // No rounding needed for regular pages
                huge_mem = real_mmap(nullptr, alloc_size,
                                    PROT_READ | PROT_WRITE,
                                    MAP_PRIVATE | MAP_ANONYMOUS,
                                    -1, 0);

                if (huge_mem == MAP_FAILED) {
                    fprintf(stderr, "[%s] ERROR: Anonymous mmap failed: %s\n",
                            ZEN5_OPTIMIZER_NAME, strerror(errno));
                    return MAP_FAILED;
                }
            } else {
                DEBUG_PRINT("Allocated %.2f GB with 1GB huge pages",
                        alloc_size / (1024.0 * 1024.0 * 1024.0));
            }

            // Read the file contents into huge pages memory
            DEBUG_PRINT("Loading file contents into huge pages memory...");

            size_t total_read = 0;
            const size_t chunk_size = 256 * 1024 * 1024; // 256MB chunks

            while (total_read < length) {
                size_t to_read = (length - total_read < chunk_size) ? (length - total_read) : chunk_size;
                ssize_t bytes_read = pread(fd, (char*)huge_mem + total_read, to_read, offset + total_read);

                if (bytes_read < 0) {
                    fprintf(stderr, "[%s] ERROR: Failed to read file: %s\n",
                            ZEN5_OPTIMIZER_NAME, strerror(errno));
                    real_munmap(huge_mem, alloc_size);
                    return MAP_FAILED;
                }

                if (bytes_read == 0) {
                    fprintf(stderr, "[%s] ERROR: Unexpected EOF at offset %zu\n",
                            ZEN5_OPTIMIZER_NAME, total_read);
                    real_munmap(huge_mem, alloc_size);
                    return MAP_FAILED;
                }

                total_read += bytes_read;

                // Progress indicator for large files
                if (total_read % (1024 * 1024 * 1024) == 0) {
                    DEBUG_PRINT("Loaded %.1f GB / %.1f GB",
                            total_read / (1024.0 * 1024.0 * 1024.0),
                            length / (1024.0 * 1024.0 * 1024.0));
                }
            }

            DEBUG_PRINT("Successfully loaded %.2f GB file into %s memory",
                    length / (1024.0 * 1024.0 * 1024.0),
                    using_hugepages ? "1GB huge pages" : "regular");

            // Set memory protection to match requested (usually PROT_READ for model files)
            // Note: mprotect on huge pages often fails with EINVAL, but this is non-fatal
            if (!(prot & PROT_WRITE)) {
                if (mprotect(huge_mem, alloc_size, prot) != 0) {
                    // Silently ignore - this is expected with huge pages
                }
            }

            // Track this allocation so we can handle munmap properly
            track_allocation(huge_mem, alloc_size, length);

            return huge_mem;
        }
    }

    // Not a candidate for huge pages, use regular mmap
    return real_mmap(addr, length, prot, flags, fd, offset);
}

// Check if address is one of our tracked allocations (without removing it)
static bool is_tracked_allocation(void* addr) {
    using namespace zen5_turbo;
    HugePageAllocation* curr = allocations;
    while (curr) {
        if (curr->addr == addr) {
            return true;
        }
        curr = curr->next;
    }
    return false;
}

// Our intercepted munmap function
extern "C" int munmap(void* addr, size_t length) {
    using namespace zen5_turbo;

    init_functions();

    // Check if this is one of our tracked hugepage allocations
    // We intentionally DO NOT unmap - the model weights must stay in memory
    // llama.cpp may call munmap but still expects to use the memory
    if (is_tracked_allocation(addr)) {
        DEBUG_PRINT("Keeping 1GB huge pages allocation at %p (munmap ignored - model weights must persist)",
                addr);
        return 0;  // Return success but don't actually unmap
    }

    // Regular munmap for non-hugepage allocations
    return real_munmap(addr, length);
}