/*
 * test_load.cpp
 *
 * Test that the zen5_optimizer library loads correctly.
 */

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include "../include/test_colors.h"

int main() {
    PRINT_TEST("Library loading");
    PRINT_RUN("Attempting to load libzen5_optimizer.so");
    printf("\n");

    // Try to load the library
    // When test binary is in build dir, library is in same dir
    void* handle = dlopen("./libzen5_optimizer.so", RTLD_NOW);
    if (!handle) {
        // Try alternate path if running from tests directory
        handle = dlopen("../build/libzen5_optimizer.so", RTLD_NOW);
    }

    if (!handle) {
        PRINT_FAIL("%s", dlerror());
        return 1;
    }

    PRINT_OK("Library loaded successfully");
    printf("\n");

    // Clean up
    dlclose(handle);

    return 0;
}