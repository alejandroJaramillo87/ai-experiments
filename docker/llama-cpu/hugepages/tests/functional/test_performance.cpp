/*
 * test_performance.cpp
 *
 * Performance measurement and baseline establishment.
 * Compares hugepage vs regular page performance,
 * measures overhead, and provides metrics for optimization tracking.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <vector>
#include "../include/test_colors.h"

// Test parameters
const size_t HUGE_SIZE = 1536 * 1024 * 1024;    // 1.5 GB (triggers hugepage)
const size_t REGULAR_SIZE = 768 * 1024 * 1024;  // 768 MB (no hugepage)
const int ITERATIONS = 10;                       // Number of test iterations
const size_t ACCESS_STRIDE = 4096;              // Page-sized stride for access tests

// Timing helpers
double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

bool create_test_file(const char* path, size_t size) {
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        PRINT_FAIL("Cannot create test file: %s", strerror(errno));
        return false;
    }

    // Write random-ish data to prevent compression optimizations
    char* buffer = (char*)malloc(4096);
    if (!buffer) {
        close(fd);
        return false;
    }

    // Fill buffer with pattern
    for (int i = 0; i < 4096; i++) {
        buffer[i] = (char)(i % 256);
    }

    // Write throughout the file
    size_t written = 0;
    while (written < size) {
        size_t to_write = (size - written > 4096) ? 4096 : (size - written);
        if (write(fd, buffer, to_write) != (ssize_t)to_write) {
            free(buffer);
            close(fd);
            return false;
        }
        written += to_write;
    }

    free(buffer);
    close(fd);
    return true;
}

int main() {
    PRINT_TEST("Performance measurements");
    printf("\n");

    // Test 1: Allocation speed comparison
    PRINT_RUN("Test 1: mmap allocation speed");
    printf("  Comparing %d allocations...\n", ITERATIONS);
    {
        const char* huge_file = "/tmp/zen5_perf_huge.dat";
        const char* regular_file = "/tmp/zen5_perf_regular.dat";

        // Create test files
        PRINT_INFO("Creating test files...");
        if (!create_test_file(huge_file, HUGE_SIZE) ||
            !create_test_file(regular_file, REGULAR_SIZE)) {
            return 1;
        }

        // Measure hugepage allocations
        printf("\n  Hugepage allocations (%.1f GB):\n", HUGE_SIZE / (1024.0 * 1024.0 * 1024.0));
        std::vector<double> huge_times;

        for (int i = 0; i < ITERATIONS; i++) {
            int fd = open(huge_file, O_RDONLY);
            if (fd < 0) continue;

            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);

            void* addr = mmap(NULL, HUGE_SIZE, PROT_READ, MAP_PRIVATE, fd, 0);

            clock_gettime(CLOCK_MONOTONIC, &end);

            if (addr != MAP_FAILED) {
                double elapsed = get_time_diff(start, end);
                huge_times.push_back(elapsed * 1000);  // Convert to ms
                printf("    Iteration %d: %.3f ms\n", i + 1, elapsed * 1000);
                munmap(addr, HUGE_SIZE);
            }

            close(fd);
        }

        // Measure regular allocations
        printf("\n  Regular allocations (%zu MB):\n", REGULAR_SIZE / (1024 * 1024));
        std::vector<double> regular_times;

        for (int i = 0; i < ITERATIONS; i++) {
            int fd = open(regular_file, O_RDONLY);
            if (fd < 0) continue;

            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);

            void* addr = mmap(NULL, REGULAR_SIZE, PROT_READ, MAP_PRIVATE, fd, 0);

            clock_gettime(CLOCK_MONOTONIC, &end);

            if (addr != MAP_FAILED) {
                double elapsed = get_time_diff(start, end);
                regular_times.push_back(elapsed * 1000);  // Convert to ms
                printf("    Iteration %d: %.3f ms\n", i + 1, elapsed * 1000);
                munmap(addr, REGULAR_SIZE);
            }

            close(fd);
        }

        // Calculate averages
        if (huge_times.size() > 0 && regular_times.size() > 0) {
            double huge_avg = 0, regular_avg = 0;
            for (double t : huge_times) huge_avg += t;
            for (double t : regular_times) regular_avg += t;
            huge_avg /= huge_times.size();
            regular_avg /= regular_times.size();

            printf("\n  Results:\n");
            PRINT_INFO("Hugepage avg: %.3f ms for %.1f GB",
                      huge_avg, HUGE_SIZE / (1024.0 * 1024.0 * 1024.0));
            PRINT_INFO("Regular avg: %.3f ms for %zu MB",
                      regular_avg, REGULAR_SIZE / (1024 * 1024));

            // Normalize to per-GB for comparison
            double huge_per_gb = huge_avg / (HUGE_SIZE / (1024.0 * 1024.0 * 1024.0));
            double regular_per_gb = regular_avg / (REGULAR_SIZE / (1024.0 * 1024.0 * 1024.0));

            PRINT_INFO("Normalized: %.3f ms/GB (huge) vs %.3f ms/GB (regular)",
                      huge_per_gb, regular_per_gb);

            if (huge_per_gb < regular_per_gb * 1.5) {
                PRINT_OK("Hugepage allocation is efficient");
            } else {
                PRINT_WARN("Hugepage allocation may have overhead");
            }
        }

        unlink(huge_file);
        unlink(regular_file);
        printf("\n");
    }

    // Test 2: Sequential memory access speed
    PRINT_RUN("Test 2: Sequential memory access speed");
    {
        const char* test_file = "/tmp/zen5_perf_access.dat";
        const size_t TEST_SIZE = 1024 * 1024 * 1024;  // 1GB for both tests

        PRINT_INFO("Creating 1GB test file...");
        if (!create_test_file(test_file, TEST_SIZE)) {
            return 1;
        }

        int fd = open(test_file, O_RDONLY);
        if (fd < 0) {
            unlink(test_file);
            return 1;
        }

        // Map the file
        void* addr = mmap(NULL, TEST_SIZE, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) {
            PRINT_FAIL("Failed to map test file");
            close(fd);
            unlink(test_file);
            return 1;
        }

        // Sequential read test
        PRINT_INFO("Performing sequential read of 1GB...");
        volatile long sum = 0;  // Prevent optimization
        struct timespec start, end;

        clock_gettime(CLOCK_MONOTONIC, &start);

        char* ptr = (char*)addr;
        for (size_t i = 0; i < TEST_SIZE; i += ACCESS_STRIDE) {
            sum += ptr[i];
        }

        clock_gettime(CLOCK_MONOTONIC, &end);

        double elapsed = get_time_diff(start, end);
        double throughput = (TEST_SIZE / (1024.0 * 1024.0 * 1024.0)) / elapsed;

        PRINT_INFO("Sequential read completed in %.3f seconds", elapsed);
        PRINT_OK("Throughput: %.2f GB/s", throughput);

        munmap(addr, TEST_SIZE);
        close(fd);
        unlink(test_file);
        printf("\n");
    }

    // Test 3: Random access pattern (TLB stress)
    PRINT_RUN("Test 3: Random access pattern (TLB efficiency)");
    {
        const char* huge_file = "/tmp/zen5_perf_tlb_huge.dat";
        const char* regular_file = "/tmp/zen5_perf_tlb_regular.dat";
        const int NUM_ACCESSES = 100000;

        PRINT_INFO("Creating test files...");
        if (!create_test_file(huge_file, HUGE_SIZE)) {
            return 1;
        }
        if (!create_test_file(regular_file, REGULAR_SIZE)) {
            unlink(huge_file);
            return 1;
        }

        // Test with hugepage file
        int fd_huge = open(huge_file, O_RDONLY);
        if (fd_huge >= 0) {
            void* addr = mmap(NULL, HUGE_SIZE, PROT_READ, MAP_PRIVATE, fd_huge, 0);
            if (addr != MAP_FAILED) {
                printf("  Testing random access with hugepages...\n");
                volatile long sum = 0;
                struct timespec start, end;

                // Generate random offsets
                srand(42);  // Fixed seed for reproducibility
                std::vector<size_t> offsets;
                for (int i = 0; i < NUM_ACCESSES; i++) {
                    offsets.push_back((rand() % (HUGE_SIZE / ACCESS_STRIDE)) * ACCESS_STRIDE);
                }

                clock_gettime(CLOCK_MONOTONIC, &start);

                char* ptr = (char*)addr;
                for (size_t offset : offsets) {
                    sum += ptr[offset];
                }

                clock_gettime(CLOCK_MONOTONIC, &end);

                double huge_time = get_time_diff(start, end);
                PRINT_INFO("Hugepage random access: %.3f ms for %d accesses",
                          huge_time * 1000, NUM_ACCESSES);
                PRINT_INFO("Average: %.3f ns per access",
                          (huge_time * 1e9) / NUM_ACCESSES);

                munmap(addr, HUGE_SIZE);
            }
            close(fd_huge);
        }

        // Test with regular file
        int fd_regular = open(regular_file, O_RDONLY);
        if (fd_regular >= 0) {
            void* addr = mmap(NULL, REGULAR_SIZE, PROT_READ, MAP_PRIVATE, fd_regular, 0);
            if (addr != MAP_FAILED) {
                printf("  Testing random access with regular pages...\n");
                volatile long sum = 0;
                struct timespec start, end;

                // Generate random offsets
                srand(42);  // Same seed for fair comparison
                std::vector<size_t> offsets;
                for (int i = 0; i < NUM_ACCESSES; i++) {
                    offsets.push_back((rand() % (REGULAR_SIZE / ACCESS_STRIDE)) * ACCESS_STRIDE);
                }

                clock_gettime(CLOCK_MONOTONIC, &start);

                char* ptr = (char*)addr;
                for (size_t offset : offsets) {
                    sum += ptr[offset];
                }

                clock_gettime(CLOCK_MONOTONIC, &end);

                double regular_time = get_time_diff(start, end);
                PRINT_INFO("Regular page random access: %.3f ms for %d accesses",
                          regular_time * 1000, NUM_ACCESSES);
                PRINT_INFO("Average: %.3f ns per access",
                          (regular_time * 1e9) / NUM_ACCESSES);

                munmap(addr, REGULAR_SIZE);
            }
            close(fd_regular);
        }

        unlink(huge_file);
        unlink(regular_file);
        printf("\n");
    }

    // Test 4: Library overhead measurement
    PRINT_RUN("Test 4: Library interception overhead");
    {
        const char* test_file = "/tmp/zen5_perf_overhead.dat";
        const int OVERHEAD_ITERATIONS = 100;

        PRINT_INFO("Creating small test file...");
        const size_t SMALL_SIZE = 1024 * 1024;  // 1MB
        if (!create_test_file(test_file, SMALL_SIZE)) {
            return 1;
        }

        int fd = open(test_file, O_RDONLY);
        if (fd < 0) {
            unlink(test_file);
            return 1;
        }

        printf("  Measuring %d mmap/munmap cycles...\n", OVERHEAD_ITERATIONS);
        std::vector<double> cycle_times;

        for (int i = 0; i < OVERHEAD_ITERATIONS; i++) {
            struct timespec start, end;

            clock_gettime(CLOCK_MONOTONIC, &start);

            void* addr = mmap(NULL, SMALL_SIZE, PROT_READ, MAP_PRIVATE, fd, 0);
            if (addr != MAP_FAILED) {
                munmap(addr, SMALL_SIZE);
            }

            clock_gettime(CLOCK_MONOTONIC, &end);

            double elapsed = get_time_diff(start, end);
            cycle_times.push_back(elapsed * 1e6);  // Convert to microseconds
        }

        if (cycle_times.size() > 0) {
            double avg = 0;
            for (double t : cycle_times) avg += t;
            avg /= cycle_times.size();

            PRINT_INFO("Average mmap/munmap cycle: %.2f microseconds", avg);

            // Find min/max
            double min_time = cycle_times[0], max_time = cycle_times[0];
            for (double t : cycle_times) {
                if (t < min_time) min_time = t;
                if (t > max_time) max_time = t;
            }

            PRINT_INFO("Range: %.2f - %.2f microseconds", min_time, max_time);

            if (avg < 100) {  // Less than 100 microseconds
                PRINT_OK("Library overhead is acceptable");
            } else {
                PRINT_WARN("Library overhead may be high");
            }
        }

        close(fd);
        unlink(test_file);
        printf("\n");
    }

    // Summary with baseline metrics
    PRINT_INFO("Performance baseline established");
    printf("\nKey metrics for future comparison:\n");
    printf("  - Hugepage allocation: ~1-3 ms per GB\n");
    printf("  - Sequential throughput: >1 GB/s expected\n");
    printf("  - Random access: Hugepages should be faster\n");
    printf("  - Library overhead: <100 microseconds per call\n");
    printf("\n");

    PRINT_OK("Performance measurements completed");
    printf("  Use these baselines to measure optimization impact\n");

    return 0;
}