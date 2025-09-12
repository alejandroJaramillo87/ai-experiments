/*
 * hugepage_mmap_wrapper.cpp
 * 
 * LD_PRELOAD library to intercept mmap() calls and handle hugetlbfs files
 * by loading them into anonymous huge page memory instead of file-backed mmap.
 * 
 * This solves the problem where llama.cpp cannot directly mmap files from hugetlbfs.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <linux/magic.h>

// Magic number for hugetlbfs
#ifndef HUGETLBFS_MAGIC
#define HUGETLBFS_MAGIC 0x958458f6
#endif

// Function pointer to the real mmap
typedef void* (*mmap_fn)(void*, size_t, int, int, int, off_t);
static mmap_fn real_mmap = nullptr;

// Function pointer to the real munmap
typedef int (*munmap_fn)(void*, size_t);
static munmap_fn real_munmap = nullptr;

// Track our anonymous huge page allocations
struct HugePageAllocation {
    void* addr;
    size_t size;
    HugePageAllocation* next;
};
static HugePageAllocation* allocations = nullptr;

// Initialize function pointers to real functions
static void init_functions() {
    if (!real_mmap) {
        real_mmap = (mmap_fn)dlsym(RTLD_NEXT, "mmap");
        if (!real_mmap) {
            fprintf(stderr, "ERROR: hugepage_wrapper: Failed to find real mmap: %s\n", dlerror());
            exit(1);
        }
    }
    if (!real_munmap) {
        real_munmap = (munmap_fn)dlsym(RTLD_NEXT, "munmap");
        if (!real_munmap) {
            fprintf(stderr, "ERROR: hugepage_wrapper: Failed to find real munmap: %s\n", dlerror());
            exit(1);
        }
    }
}

// Check if a file descriptor points to a hugetlbfs file
static bool is_hugetlbfs_fd(int fd) {
    struct statfs fs_info;
    if (fstatfs(fd, &fs_info) == 0) {
        return fs_info.f_type == HUGETLBFS_MAGIC;
    }
    return false;
}

// Track an allocation so we can handle munmap properly
static void track_allocation(void* addr, size_t size) {
    HugePageAllocation* alloc = (HugePageAllocation*)malloc(sizeof(HugePageAllocation));
    alloc->addr = addr;
    alloc->size = size;
    alloc->next = allocations;
    allocations = alloc;
}

// Find and remove a tracked allocation
static size_t untrack_allocation(void* addr) {
    HugePageAllocation** prev = &allocations;
    HugePageAllocation* curr = allocations;
    
    while (curr) {
        if (curr->addr == addr) {
            size_t size = curr->size;
            *prev = curr->next;
            free(curr);
            return size;
        }
        prev = &curr->next;
        curr = curr->next;
    }
    return 0;
}

// Our intercepted mmap function
extern "C" void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    init_functions();
    
    // Check if this is a file-backed mmap on hugetlbfs
    if (fd >= 0 && is_hugetlbfs_fd(fd)) {
        // Get file size
        struct stat st;
        if (fstat(fd, &st) != 0) {
            fprintf(stderr, "WARNING: hugepage_wrapper: Failed to stat fd %d: %s\n", fd, strerror(errno));
            return real_mmap(addr, length, prot, flags, fd, offset);
        }
        
        // Verify we're mapping the whole file from offset 0 (typical for model loading)
        if (offset == 0 && length == (size_t)st.st_size) {
            fprintf(stderr, "INFO: hugepage_wrapper: Intercepting hugetlbfs mmap for %.2f GB file\n", 
                    length / (1024.0 * 1024.0 * 1024.0));
            
            // Allocate anonymous huge pages memory
            void* huge_mem = real_mmap(nullptr, length, 
                                       PROT_READ | PROT_WRITE,
                                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                                       -1, 0);
            
            if (huge_mem == MAP_FAILED) {
                // Try without MAP_HUGETLB as fallback
                fprintf(stderr, "WARNING: hugepage_wrapper: MAP_HUGETLB failed, trying regular anonymous mmap\n");
                huge_mem = real_mmap(nullptr, length,
                                    PROT_READ | PROT_WRITE,
                                    MAP_PRIVATE | MAP_ANONYMOUS,
                                    -1, 0);
                
                if (huge_mem == MAP_FAILED) {
                    fprintf(stderr, "ERROR: hugepage_wrapper: Anonymous mmap failed: %s\n", strerror(errno));
                    return MAP_FAILED;
                }
            } else {
                fprintf(stderr, "hugepage_wrapper: Allocated %.2f GB with MAP_HUGETLB\n",
                        length / (1024.0 * 1024.0 * 1024.0));
            }
            
            // Read the file contents into our anonymous memory
            ssize_t total_read = 0;
            char* dest = (char*)huge_mem;
            
            // Reset file position to beginning
            if (lseek(fd, 0, SEEK_SET) != 0) {
                fprintf(stderr, "ERROR: hugepage_wrapper: Failed to seek: %s\n", strerror(errno));
                real_munmap(huge_mem, length);
                return MAP_FAILED;
            }
            
            // Read in chunks
            const size_t chunk_size = 64 * 1024 * 1024; // 64MB chunks
            while (total_read < (ssize_t)length) {
                size_t to_read = ((size_t)(length - total_read) < chunk_size) ? 
                                (length - total_read) : chunk_size;
                
                ssize_t bytes_read = read(fd, dest + total_read, to_read);
                if (bytes_read <= 0) {
                    if (bytes_read < 0) {
                        fprintf(stderr, "ERROR: hugepage_wrapper: Read failed at offset %zd: %s\n", 
                                total_read, strerror(errno));
                    } else {
                        fprintf(stderr, "ERROR: hugepage_wrapper: Unexpected EOF at offset %zd\n", total_read);
                    }
                    real_munmap(huge_mem, length);
                    return MAP_FAILED;
                }
                total_read += bytes_read;
                
                // Progress indicator every 1GB
                if (total_read % (1024 * 1024 * 1024) == 0) {
                    fprintf(stderr, "   ... loaded %.1f GB / %.1f GB\n",
                            total_read / (1024.0 * 1024.0 * 1024.0),
                            length / (1024.0 * 1024.0 * 1024.0));
                }
            }
            
            fprintf(stderr, "hugepage_wrapper: Successfully loaded %.2f GB model into huge pages memory\n",
                    total_read / (1024.0 * 1024.0 * 1024.0));
            
            // Set memory protection to match requested (usually PROT_READ for model files)
            if (!(prot & PROT_WRITE)) {
                if (mprotect(huge_mem, length, prot) != 0) {
                    fprintf(stderr, "WARNING: hugepage_wrapper: mprotect failed: %s\n", strerror(errno));
                    // Non-fatal, continue anyway
                }
            }
            
            // Track this allocation so we can handle munmap properly
            track_allocation(huge_mem, length);
            
            return huge_mem;
        }
    }
    
    // Not a hugetlbfs file or not a full file mapping, use regular mmap
    return real_mmap(addr, length, prot, flags, fd, offset);
}

// Our intercepted munmap function
extern "C" int munmap(void* addr, size_t length) {
    init_functions();
    
    // Check if this is one of our tracked allocations
    size_t tracked_size = untrack_allocation(addr);
    if (tracked_size > 0) {
        fprintf(stderr, "INFO: hugepage_wrapper: Unmapping %.2f GB huge pages allocation\n",
                tracked_size / (1024.0 * 1024.0 * 1024.0));
        // Use the tracked size, not the provided length (which might be wrong)
        return real_munmap(addr, tracked_size);
    }
    
    // Regular munmap
    return real_munmap(addr, length);
}

// Constructor - runs when library is loaded
__attribute__((constructor))
static void init() {
    fprintf(stderr, "hugepage_mmap_wrapper loaded (PID: %d)\n", getpid());
    init_functions();
}

// Destructor - cleanup when library is unloaded
__attribute__((destructor))
static void cleanup() {
    // Clean up any remaining tracked allocations
    while (allocations) {
        HugePageAllocation* next = allocations->next;
        free(allocations);
        allocations = next;
    }
}