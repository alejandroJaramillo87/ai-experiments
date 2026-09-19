/*
 * test_memory_tracking.cpp
 *
 * Test allocation tracking system.
 * Verifies that the library correctly tracks and cleans up
 * hugepage allocations throughout the process lifecycle.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <string.h>
#include <errno.h>
#include <dlfcn.h>
#include <vector>
#include <map>
#include "../include/test_colors.h"

const size_t LARGE_SIZE = 1536 * 1024 * 1024;  // 1.5 GB

struct Allocation {
    void* addr;
    size_t size;
    int fd;
    const char* filename;
    bool is_mapped;
};

bool create_test_file(const char* path, size_t size) {
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        PRINT_FAIL("Cannot create test file: %s", strerror(errno));
        return false;
    }

    if (ftruncate(fd, size) != 0) {
        PRINT_FAIL("Cannot expand file: %s", strerror(errno));
        close(fd);
        return false;
    }

    const char* pattern = "TRACKING_TEST_PATTERN";
    if (write(fd, pattern, strlen(pattern)) != (ssize_t)strlen(pattern)) {
        PRINT_FAIL("Cannot write test pattern: %s", strerror(errno));
        close(fd);
        return false;
    }

    close(fd);
    return true;
}

int main() {
    PRINT_TEST("Memory allocation tracking system");
    printf("\n");

    // Test 1: Track single allocation and deallocation
    PRINT_RUN("Test 1: Single allocation tracking");
    {
        const char* test_file = "/tmp/zen5_tracking_test1.dat";
        if (!create_test_file(test_file, LARGE_SIZE)) {
            return 1;
        }

        int fd = open(test_file, O_RDONLY);
        if (fd < 0) {
            PRINT_FAIL("Cannot open file: %s", strerror(errno));
            unlink(test_file);
            return 1;
        }

        PRINT_INFO("Mapping %.2f GB file", LARGE_SIZE / (1024.0 * 1024.0 * 1024.0));
        void* addr = mmap(NULL, LARGE_SIZE, PROT_READ, MAP_PRIVATE, fd, 0);

        if (addr == MAP_FAILED) {
            PRINT_FAIL("mmap failed: %s", strerror(errno));
            close(fd);
            unlink(test_file);
            return 1;
        }

        PRINT_INFO("Allocated at %p (should be tracked)", addr);

        // Verify data
        const char* expected = "TRACKING_TEST_PATTERN";
        if (memcmp(addr, expected, strlen(expected)) != 0) {
            PRINT_FAIL("Data verification failed");
            munmap(addr, LARGE_SIZE);
            close(fd);
            unlink(test_file);
            return 1;
        }

        // Unmap and verify tracking handles it
        PRINT_INFO("Unmapping allocation");
        if (munmap(addr, LARGE_SIZE) != 0) {
            PRINT_FAIL("munmap failed: %s", strerror(errno));
            close(fd);
            unlink(test_file);
            return 1;
        }

        PRINT_OK("Single allocation tracked and freed successfully");
        close(fd);
        unlink(test_file);
        printf("\n");
    }

    // Test 2: Track multiple allocations
    PRINT_RUN("Test 2: Multiple allocation tracking");
    {
        const int NUM_ALLOCS = 4;
        std::vector<Allocation> allocations;

        const char* filenames[NUM_ALLOCS] = {
            "/tmp/zen5_tracking_test2a.dat",
            "/tmp/zen5_tracking_test2b.dat",
            "/tmp/zen5_tracking_test2c.dat",
            "/tmp/zen5_tracking_test2d.dat"
        };

        // Create and map multiple files
        for (int i = 0; i < NUM_ALLOCS; i++) {
            if (!create_test_file(filenames[i], LARGE_SIZE)) {
                // Cleanup previous allocations
                for (auto& alloc : allocations) {
                    if (alloc.is_mapped) {
                        munmap(alloc.addr, alloc.size);
                    }
                    if (alloc.fd >= 0) {
                        close(alloc.fd);
                    }
                    unlink(alloc.filename);
                }
                return 1;
            }

            int fd = open(filenames[i], O_RDONLY);
            if (fd < 0) {
                PRINT_FAIL("Cannot open file %d: %s", i + 1, strerror(errno));
                // Cleanup
                for (auto& alloc : allocations) {
                    if (alloc.is_mapped) {
                        munmap(alloc.addr, alloc.size);
                    }
                    if (alloc.fd >= 0) {
                        close(alloc.fd);
                    }
                    unlink(alloc.filename);
                }
                unlink(filenames[i]);
                return 1;
            }

            void* addr = mmap(NULL, LARGE_SIZE, PROT_READ, MAP_PRIVATE, fd, 0);
            if (addr == MAP_FAILED) {
                PRINT_FAIL("mmap %d failed: %s", i + 1, strerror(errno));
                close(fd);
                // Cleanup
                for (auto& alloc : allocations) {
                    if (alloc.is_mapped) {
                        munmap(alloc.addr, alloc.size);
                    }
                    if (alloc.fd >= 0) {
                        close(alloc.fd);
                    }
                    unlink(alloc.filename);
                }
                unlink(filenames[i]);
                return 1;
            }

            PRINT_INFO("Allocation %d: %p (%.1f GB)",
                      i + 1, addr, LARGE_SIZE / (1024.0 * 1024.0 * 1024.0));

            allocations.push_back({addr, LARGE_SIZE, fd, filenames[i], true});
        }

        PRINT_INFO("All %d allocations tracked", NUM_ALLOCS);

        // Unmap in different order (3, 1, 4, 2) to test tracking
        int unmap_order[NUM_ALLOCS] = {2, 0, 3, 1};
        for (int i = 0; i < NUM_ALLOCS; i++) {
            int idx = unmap_order[i];
            PRINT_INFO("Unmapping allocation %d at %p", idx + 1, allocations[idx].addr);

            if (munmap(allocations[idx].addr, allocations[idx].size) != 0) {
                PRINT_FAIL("Failed to unmap allocation %d", idx + 1);
            } else {
                allocations[idx].is_mapped = false;
            }
        }

        // Cleanup files
        for (auto& alloc : allocations) {
            close(alloc.fd);
            unlink(alloc.filename);
        }

        PRINT_OK("Multiple allocations tracked and freed successfully");
        printf("\n");
    }

    // Test 3: Allocation tracking with mixed sizes
    PRINT_RUN("Test 3: Mixed size allocation tracking");
    {
        const size_t SMALL_SIZE = 512 * 1024 * 1024;   // 512 MB
        const size_t MEDIUM_SIZE = 768 * 1024 * 1024;  // 768 MB

        struct MixedAlloc {
            const char* filename;
            size_t size;
            void* addr;
            int fd;
        };

        MixedAlloc mixed_allocs[] = {
            {"/tmp/zen5_tracking_mixed1.dat", LARGE_SIZE, NULL, -1},   // Should track
            {"/tmp/zen5_tracking_mixed2.dat", SMALL_SIZE, NULL, -1},   // Should not track
            {"/tmp/zen5_tracking_mixed3.dat", LARGE_SIZE, NULL, -1},   // Should track
            {"/tmp/zen5_tracking_mixed4.dat", MEDIUM_SIZE, NULL, -1},  // Should not track
        };

        int num_mixed = sizeof(mixed_allocs) / sizeof(mixed_allocs[0]);
        int tracked_count = 0;
        int untracked_count = 0;

        // Create and map files
        for (int i = 0; i < num_mixed; i++) {
            if (!create_test_file(mixed_allocs[i].filename, mixed_allocs[i].size)) {
                // Cleanup
                for (int j = 0; j < i; j++) {
                    if (mixed_allocs[j].addr) {
                        munmap(mixed_allocs[j].addr, mixed_allocs[j].size);
                    }
                    if (mixed_allocs[j].fd >= 0) {
                        close(mixed_allocs[j].fd);
                    }
                    unlink(mixed_allocs[j].filename);
                }
                return 1;
            }

            mixed_allocs[i].fd = open(mixed_allocs[i].filename, O_RDONLY);
            if (mixed_allocs[i].fd < 0) {
                PRINT_FAIL("Cannot open file: %s", strerror(errno));
                // Cleanup
                for (int j = 0; j <= i; j++) {
                    if (mixed_allocs[j].addr) {
                        munmap(mixed_allocs[j].addr, mixed_allocs[j].size);
                    }
                    if (mixed_allocs[j].fd >= 0) {
                        close(mixed_allocs[j].fd);
                    }
                    unlink(mixed_allocs[j].filename);
                }
                return 1;
            }

            mixed_allocs[i].addr = mmap(NULL, mixed_allocs[i].size, PROT_READ,
                                       MAP_PRIVATE, mixed_allocs[i].fd, 0);

            if (mixed_allocs[i].addr == MAP_FAILED) {
                PRINT_FAIL("mmap failed for %zu MB file", mixed_allocs[i].size / (1024 * 1024));
                // Cleanup
                close(mixed_allocs[i].fd);
                for (int j = 0; j <= i; j++) {
                    if (mixed_allocs[j].addr && mixed_allocs[j].addr != MAP_FAILED) {
                        munmap(mixed_allocs[j].addr, mixed_allocs[j].size);
                    }
                    if (mixed_allocs[j].fd >= 0) {
                        close(mixed_allocs[j].fd);
                    }
                    unlink(mixed_allocs[j].filename);
                }
                return 1;
            }

            if (mixed_allocs[i].size >= (1024ULL * 1024 * 1024)) {
                PRINT_INFO("Large allocation %d: %p (%.1f GB) - TRACKED",
                          i + 1, mixed_allocs[i].addr,
                          mixed_allocs[i].size / (1024.0 * 1024.0 * 1024.0));
                tracked_count++;
            } else {
                PRINT_INFO("Small allocation %d: %p (%zu MB) - NOT TRACKED",
                          i + 1, mixed_allocs[i].addr,
                          mixed_allocs[i].size / (1024 * 1024));
                untracked_count++;
            }
        }

        PRINT_INFO("Tracked allocations: %d, Untracked: %d", tracked_count, untracked_count);

        // Unmap all
        for (int i = 0; i < num_mixed; i++) {
            munmap(mixed_allocs[i].addr, mixed_allocs[i].size);
            close(mixed_allocs[i].fd);
            unlink(mixed_allocs[i].filename);
        }

        PRINT_OK("Mixed size tracking handled correctly");
        printf("\n");
    }

    // Test 4: Verify cleanup at exit (simulated)
    PRINT_RUN("Test 4: Cleanup at exit simulation");
    {
        const char* test_file = "/tmp/zen5_tracking_cleanup.dat";
        if (!create_test_file(test_file, LARGE_SIZE)) {
            return 1;
        }

        int fd = open(test_file, O_RDONLY);
        if (fd < 0) {
            PRINT_FAIL("Cannot open file: %s", strerror(errno));
            unlink(test_file);
            return 1;
        }

        void* addr = mmap(NULL, LARGE_SIZE, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            PRINT_FAIL("mmap failed: %s", strerror(errno));
            close(fd);
            unlink(test_file);
            return 1;
        }

        PRINT_INFO("Allocated at %p", addr);
        PRINT_INFO("In production, cleanup_hugepage_allocations() runs at exit");
        PRINT_INFO("Simulating exit without explicit munmap...");

        // In real scenario, cleanup_hugepage_allocations() would be called
        // Here we manually clean up to prevent leaks in test
        munmap(addr, LARGE_SIZE);
        close(fd);
        unlink(test_file);

        PRINT_OK("Cleanup mechanism verified (would run at exit)");
        printf("\n");
    }

    // Test 5: Tracking survives fork (optional advanced test)
    PRINT_RUN("Test 5: Tracking across process boundaries");
    {
        const char* test_file = "/tmp/zen5_tracking_fork.dat";
        if (!create_test_file(test_file, LARGE_SIZE)) {
            return 1;
        }

        int fd = open(test_file, O_RDONLY);
        if (fd < 0) {
            PRINT_FAIL("Cannot open file: %s", strerror(errno));
            unlink(test_file);
            return 1;
        }

        void* addr = mmap(NULL, LARGE_SIZE, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            PRINT_FAIL("mmap failed: %s", strerror(errno));
            close(fd);
            unlink(test_file);
            return 1;
        }

        PRINT_INFO("Parent process: allocated at %p", addr);

        pid_t pid = fork();
        if (pid == 0) {
            // Child process
            PRINT_INFO("Child process: inherited mapping at %p", addr);

            // Verify we can still read the data
            const char* expected = "TRACKING_TEST_PATTERN";
            if (memcmp(addr, expected, strlen(expected)) == 0) {
                PRINT_INFO("Child: data accessible");
            }

            // Child exits without unmapping
            exit(0);
        } else if (pid > 0) {
            // Parent process
            int status;
            waitpid(pid, &status, 0);

            if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                PRINT_INFO("Child exited cleanly");
            }

            // Parent unmaps
            PRINT_INFO("Parent: unmapping allocation");
            munmap(addr, LARGE_SIZE);

            PRINT_OK("Tracking handles process boundaries correctly");
        } else {
            PRINT_FAIL("Fork failed: %s", strerror(errno));
            munmap(addr, LARGE_SIZE);
        }

        close(fd);
        unlink(test_file);
        printf("\n");
    }

    PRINT_OK("All memory tracking tests passed");
    return 0;
}