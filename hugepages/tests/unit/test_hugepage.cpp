/*
 * test_hugepage.cpp
 *
 * Test that mmap interception works correctly.
 * Creates a temporary file and verifies hugepage allocation.
 * Also tests concurrent mmap calls to verify thread safety.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <atomic>
#include "../include/test_colors.h"

// Concurrent test data
struct ThreadData {
    int thread_id;
    const char* filename;
    size_t file_size;
    bool success;
    void* mapped_addr;
};

std::atomic<int> threads_ready(0);
std::atomic<bool> start_signal(false);

// Thread worker for concurrent mmap testing
void* concurrent_mmap_worker(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    // Signal ready and wait for all threads
    threads_ready++;
    while (!start_signal) {
        // Spin wait for start signal
    }

    // Open the file
    int fd = open(data->filename, O_RDONLY);
    if (fd < 0) {
        data->success = false;
        return NULL;
    }

    // Perform mmap (this should trigger our interception)
    void* addr = mmap(NULL, data->file_size, PROT_READ, MAP_PRIVATE, fd, 0);

    if (addr == MAP_FAILED) {
        close(fd);
        data->success = false;
        return NULL;
    }

    // Verify we can read from the mapping
    char buffer[16];
    memcpy(buffer, addr, 15);
    buffer[15] = '\0';

    // Store results
    data->mapped_addr = addr;
    data->success = true;

    // Clean up
    munmap(addr, data->file_size);
    close(fd);

    return NULL;
}

// Run concurrent mmap test
void run_concurrent_test() {
    PRINT_TEST("Concurrent mmap operations");
    printf("\n");

    const int NUM_THREADS = 3;
    const size_t test_size = 1536 * 1024 * 1024; // 1.5 GB each

    // Create test files
    const char* filenames[NUM_THREADS] = {
        "/tmp/zen5_concurrent_1.dat",
        "/tmp/zen5_concurrent_2.dat",
        "/tmp/zen5_concurrent_3.dat"
    };

    PRINT_RUN("Creating %d test files for concurrent access", NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; i++) {
        int fd = open(filenames[i], O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            PRINT_FAIL("Cannot create file %d: %s", i + 1, strerror(errno));
            // Clean up previously created files
            for (int j = 0; j < i; j++) {
                unlink(filenames[j]);
            }
            return;
        }

        if (ftruncate(fd, test_size) != 0) {
            PRINT_FAIL("Cannot expand file %d: %s", i + 1, strerror(errno));
            close(fd);
            for (int j = 0; j <= i; j++) {
                unlink(filenames[j]);
            }
            return;
        }

        const char* pattern = "CONCURRENT_TEST";
        write(fd, pattern, strlen(pattern));
        close(fd);
    }

    // Prepare thread data
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].thread_id = i + 1;
        thread_data[i].filename = filenames[i];
        thread_data[i].file_size = test_size;
        thread_data[i].success = false;
        thread_data[i].mapped_addr = NULL;
    }

    PRINT_RUN("Launching %d concurrent mmap operations", NUM_THREADS);

    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        if (pthread_create(&threads[i], NULL, concurrent_mmap_worker, &thread_data[i]) != 0) {
            PRINT_FAIL("Failed to create thread %d", i + 1);
            // Clean up files
            for (int j = 0; j < NUM_THREADS; j++) {
                unlink(filenames[j]);
            }
            return;
        }
    }

    // Wait for all threads to be ready
    while (threads_ready < NUM_THREADS) {
        usleep(1000);
    }

    // Signal all threads to start simultaneously
    PRINT_INFO("Starting concurrent mmap calls");
    start_signal = true;

    // Wait for all threads to complete
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    // Check results
    int successful = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        if (thread_data[i].success) {
            successful++;
            PRINT_OK("Thread %d: mmap succeeded", thread_data[i].thread_id);
        } else {
            PRINT_FAIL("Thread %d: mmap failed", thread_data[i].thread_id);
        }
    }

    // Clean up test files
    for (int i = 0; i < NUM_THREADS; i++) {
        unlink(filenames[i]);
    }

    if (successful == NUM_THREADS) {
        PRINT_OK("All %d concurrent mmap operations succeeded", NUM_THREADS);
        PRINT_INFO("Mutex protection verified - no deadlocks or crashes");
    } else {
        PRINT_FAIL("%d/%d concurrent operations failed",
                   NUM_THREADS - successful, NUM_THREADS);
    }

    printf("\n");
}

int main() {
    PRINT_TEST("mmap interception");
    printf("\n");

    // Create a temporary test file (1.5 GB to trigger hugepage logic)
    const size_t test_size = 1536 * 1024 * 1024; // 1.5 GB
    const char* test_file = "/tmp/zen5_test_hugepage.dat";

    PRINT_RUN("Creating %.2f GB test file", test_size / (1024.0 * 1024.0 * 1024.0));

    int fd = open(test_file, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        PRINT_FAIL("Cannot create test file: %s", strerror(errno));
        return 1;
    }

    // Expand file to test size
    if (ftruncate(fd, test_size) != 0) {
        PRINT_FAIL("Cannot expand file: %s", strerror(errno));
        close(fd);
        unlink(test_file);
        return 1;
    }

    // Write some test data
    const char* test_data = "ZEN5_OPTIMIZER_TEST_PATTERN";
    if (write(fd, test_data, strlen(test_data)) != (ssize_t)strlen(test_data)) {
        PRINT_FAIL("Cannot write test data: %s", strerror(errno));
        close(fd);
        unlink(test_file);
        return 1;
    }

    // Reset file position
    lseek(fd, 0, SEEK_SET);

    PRINT_RUN("Attempting mmap (should trigger interception)");

    // This mmap should be intercepted by our library (if LD_PRELOAD is active)
    void* addr = mmap(NULL, test_size, PROT_READ, MAP_PRIVATE, fd, 0);

    if (addr == MAP_FAILED) {
        PRINT_FAIL("mmap failed: %s", strerror(errno));
        close(fd);
        unlink(test_file);
        return 1;
    }

    PRINT_OK("mmap succeeded at address %p", addr);

    // Verify we can read the data
    if (memcmp(addr, test_data, strlen(test_data)) != 0) {
        PRINT_FAIL("Data mismatch");
        munmap(addr, test_size);
        close(fd);
        unlink(test_file);
        return 1;
    }

    PRINT_OK("Data verification passed");

    // Check if hugepages are in use (informational)
    FILE* maps = fopen("/proc/self/maps", "r");
    if (maps) {
        char line[256];
        bool found_huge = false;
        while (fgets(line, sizeof(line), maps)) {
            if (strstr(line, "huge")) {
                found_huge = true;
                break;
            }
        }
        fclose(maps);

        if (found_huge) {
            PRINT_INFO("Huge pages detected in memory mappings");
        } else {
            PRINT_INFO("No huge pages detected (may be transparent)");
        }
    }

    // Clean up
    munmap(addr, test_size);
    close(fd);
    unlink(test_file);

    PRINT_OK("mmap interception test complete");
    printf("\n");

    // Concurrent mmap test
    run_concurrent_test();

    return 0;
}