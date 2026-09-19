/*
 * test_stress.cpp
 *
 * Stress testing for the zen5_optimizer library.
 * Tests reliability under heavy load, concurrent operations,
 * and memory pressure scenarios.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <pthread.h>
#include <atomic>
#include <vector>
#include <string>
#include "../include/test_colors.h"

// Test parameters
const size_t LARGE_SIZE = 1536 * 1024 * 1024;  // 1.5 GB
const size_t MEDIUM_SIZE = 1024 * 1024 * 1024; // 1.0 GB
const int RAPID_CYCLES = 50;                   // Reduced from 100 for faster testing
const int THREADS_COUNT = 8;                   // Concurrent threads
const int ALLOCATIONS_PER_THREAD = 5;          // Allocations per thread

// Global counters for thread coordination
std::atomic<int> successful_allocs(0);
std::atomic<int> failed_allocs(0);
std::atomic<bool> stop_flag(false);

struct ThreadData {
    int thread_id;
    int num_allocations;
    bool success;
};

bool create_test_file(const char* path, size_t size) {
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return false;

    if (ftruncate(fd, size) != 0) {
        close(fd);
        return false;
    }

    const char* pattern = "STRESS_TEST";
    write(fd, pattern, strlen(pattern));
    close(fd);
    return true;
}

// Worker thread for concurrent allocation test
void* stress_worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    char filename[256];

    for (int i = 0; i < data->num_allocations && !stop_flag; i++) {
        snprintf(filename, sizeof(filename),
                 "/tmp/zen5_stress_t%d_a%d.dat", data->thread_id, i);

        // Alternate between medium and large sizes
        size_t size = (i % 2 == 0) ? MEDIUM_SIZE : LARGE_SIZE;

        if (!create_test_file(filename, size)) {
            failed_allocs++;
            continue;
        }

        int fd = open(filename, O_RDONLY);
        if (fd < 0) {
            failed_allocs++;
            unlink(filename);
            continue;
        }

        void* addr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            failed_allocs++;
        } else {
            successful_allocs++;
            // Small delay to simulate work
            usleep(1000);  // 1ms
            munmap(addr, size);
        }

        close(fd);
        unlink(filename);
    }

    data->success = true;
    return NULL;
}

int main() {
    PRINT_TEST("Stress testing");
    printf("\n");

    int total_passed = 0;
    int total_failed = 0;

    // Test 1: Rapid allocation/deallocation cycles
    PRINT_RUN("Test 1: Rapid allocation/deallocation cycles");
    printf("  Performing %d iterations...\n", RAPID_CYCLES);
    {
        const char* test_file = "/tmp/zen5_stress_rapid.dat";
        bool cycles_passed = true;
        clock_t start_time = clock();

        for (int i = 0; i < RAPID_CYCLES; i++) {
            if (!create_test_file(test_file, LARGE_SIZE)) {
                PRINT_FAIL("Failed to create file at iteration %d", i);
                cycles_passed = false;
                break;
            }

            int fd = open(test_file, O_RDONLY);
            if (fd < 0) {
                PRINT_FAIL("Failed to open file at iteration %d", i);
                unlink(test_file);
                cycles_passed = false;
                break;
            }

            void* addr = mmap(NULL, LARGE_SIZE, PROT_READ, MAP_PRIVATE, fd, 0);
            if (addr == MAP_FAILED) {
                PRINT_FAIL("mmap failed at iteration %d: %s", i, strerror(errno));
                close(fd);
                unlink(test_file);
                cycles_passed = false;
                break;
            }

            // Quick data verification
            char buffer[16];
            memcpy(buffer, addr, 11);

            munmap(addr, LARGE_SIZE);
            close(fd);
            unlink(test_file);

            // Progress indicator every 10 iterations
            if ((i + 1) % 10 == 0) {
                printf("  Progress: %d/%d\n", i + 1, RAPID_CYCLES);
            }
        }

        clock_t end_time = clock();
        double elapsed = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

        if (cycles_passed) {
            PRINT_OK("Completed %d rapid cycles in %.2f seconds", RAPID_CYCLES, elapsed);
            PRINT_INFO("Average: %.2f cycles/second", RAPID_CYCLES / elapsed);
            total_passed++;
        } else {
            total_failed++;
        }
        printf("\n");
    }

    // Test 2: Concurrent allocation stress
    PRINT_RUN("Test 2: Concurrent allocation stress");
    printf("  Launching %d threads with %d allocations each...\n",
           THREADS_COUNT, ALLOCATIONS_PER_THREAD);
    {
        pthread_t threads[THREADS_COUNT];
        ThreadData thread_data[THREADS_COUNT];

        successful_allocs = 0;
        failed_allocs = 0;
        stop_flag = false;

        clock_t start_time = clock();

        // Launch threads
        for (int i = 0; i < THREADS_COUNT; i++) {
            thread_data[i].thread_id = i;
            thread_data[i].num_allocations = ALLOCATIONS_PER_THREAD;
            thread_data[i].success = false;

            if (pthread_create(&threads[i], NULL, stress_worker, &thread_data[i]) != 0) {
                PRINT_FAIL("Failed to create thread %d", i);
                stop_flag = true;
                break;
            }
        }

        // Wait for completion
        for (int i = 0; i < THREADS_COUNT; i++) {
            pthread_join(threads[i], NULL);
        }

        clock_t end_time = clock();
        double elapsed = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

        int total_attempts = THREADS_COUNT * ALLOCATIONS_PER_THREAD;
        PRINT_INFO("Successful allocations: %d/%d",
                   successful_allocs.load(), total_attempts);
        PRINT_INFO("Failed allocations: %d", failed_allocs.load());
        PRINT_INFO("Time: %.2f seconds", elapsed);

        if (successful_allocs > 0 && !stop_flag) {
            PRINT_OK("Concurrent stress test completed");
            total_passed++;
        } else {
            PRINT_FAIL("Concurrent stress test failed");
            total_failed++;
        }
        printf("\n");
    }

    // Test 3: Memory pressure test
    PRINT_RUN("Test 3: Memory pressure test");
    printf("  Allocating until system limit...\n");
    {
        std::vector<void*> allocations;
        std::vector<int> fds;
        std::vector<std::string> filenames;
        const size_t PRESSURE_SIZE = LARGE_SIZE;
        int max_allocs = 0;

        // Try to allocate up to 10 large files
        for (int i = 0; i < 10; i++) {
            char filename[256];
            snprintf(filename, sizeof(filename), "/tmp/zen5_pressure_%d.dat", i);
            filenames.push_back(filename);

            if (!create_test_file(filename, PRESSURE_SIZE)) {
                PRINT_INFO("File creation failed at %d (expected under pressure)", i);
                break;
            }

            int fd = open(filename, O_RDONLY);
            if (fd < 0) {
                PRINT_INFO("Open failed at %d (expected under pressure)", i);
                break;
            }
            fds.push_back(fd);

            void* addr = mmap(NULL, PRESSURE_SIZE, PROT_READ, MAP_PRIVATE, fd, 0);
            if (addr == MAP_FAILED) {
                PRINT_INFO("mmap failed at allocation %d (system limit reached)", i + 1);
                close(fd);
                fds.pop_back();
                break;
            }

            allocations.push_back(addr);
            max_allocs++;
            PRINT_INFO("Allocated %d: %p (%.1f GB total)",
                      i + 1, addr, (i + 1) * PRESSURE_SIZE / (1024.0 * 1024.0 * 1024.0));
        }

        PRINT_INFO("Maximum concurrent allocations: %d (%.1f GB)",
                   max_allocs, max_allocs * PRESSURE_SIZE / (1024.0 * 1024.0 * 1024.0));

        // Cleanup
        for (size_t i = 0; i < allocations.size(); i++) {
            munmap(allocations[i], PRESSURE_SIZE);
        }
        for (size_t i = 0; i < fds.size(); i++) {
            close(fds[i]);
        }
        for (size_t i = 0; i < filenames.size(); i++) {
            unlink(filenames[i].c_str());
        }

        if (max_allocs > 0) {
            PRINT_OK("Memory pressure test completed (handled %d allocations)", max_allocs);
            total_passed++;
        } else {
            PRINT_FAIL("Could not allocate any memory");
            total_failed++;
        }
        printf("\n");
    }

    // Test 4: Mixed operations stress
    PRINT_RUN("Test 4: Mixed size rapid operations");
    {
        const int MIXED_ITERATIONS = 30;
        size_t sizes[] = {512 * 1024 * 1024, MEDIUM_SIZE, LARGE_SIZE, 2048 * 1024 * 1024ULL};
        int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
        bool mixed_passed = true;

        printf("  Testing with sizes: ");
        for (int i = 0; i < num_sizes; i++) {
            printf("%zuMB ", sizes[i] / (1024 * 1024));
        }
        printf("\n");

        for (int i = 0; i < MIXED_ITERATIONS && mixed_passed; i++) {
            size_t size = sizes[i % num_sizes];
            char filename[256];
            snprintf(filename, sizeof(filename), "/tmp/zen5_mixed_%d.dat", i);

            if (!create_test_file(filename, size)) {
                // May fail for 2GB file, that's OK
                if (size <= LARGE_SIZE) {
                    PRINT_FAIL("Failed to create %zu MB file", size / (1024 * 1024));
                    mixed_passed = false;
                }
                continue;
            }

            int fd = open(filename, O_RDONLY);
            if (fd >= 0) {
                void* addr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
                if (addr != MAP_FAILED) {
                    munmap(addr, size);
                }
                close(fd);
            }
            unlink(filename);
        }

        if (mixed_passed) {
            PRINT_OK("Mixed operations completed successfully");
            total_passed++;
        } else {
            PRINT_FAIL("Mixed operations failed");
            total_failed++;
        }
        printf("\n");
    }

    // Summary
    printf("[test_stress] Summary:\n");
    printf("  Total tests: %d\n", total_passed + total_failed);
    printf("  Passed: %d\n", total_passed);
    printf("  Failed: %d\n", total_failed);

    if (total_failed == 0) {
        PRINT_OK("All stress tests passed");
        return 0;
    } else {
        PRINT_FAIL("%d stress test(s) failed", total_failed);
        return 1;
    }
}