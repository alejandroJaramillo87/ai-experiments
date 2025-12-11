/*
 * zen5_optimizer.cpp
 *
 * Main LD_PRELOAD library for AMD Zen 5 optimizations.
 * Phase 1A implements hugepage support.
 */

#include <stdio.h>
#include <unistd.h>
#include "config.h"
#include "cpu_validator.h"

// Forward declare cleanup function
namespace zen5_turbo {
    void cleanup_hugepage_allocations();
}

// Library initialization
__attribute__((constructor))
static void zen5_optimizer_init() {
    fprintf(stderr, "[%s] Version %s (PID %d)\n",
            ZEN5_OPTIMIZER_NAME, ZEN5_OPTIMIZER_VERSION, getpid());

    // Validate CPU before doing anything else
    zen5_turbo::validate_zen5_or_exit();

#if ENABLE_HUGEPAGES
    fprintf(stderr, "[%s] Hugepage support: ON (threshold %.1f GB)\n",
            ZEN5_OPTIMIZER_NAME, MIN_SIZE_FOR_HUGEPAGES / (1024.0 * 1024.0 * 1024.0));
#else
    fprintf(stderr, "[%s] Hugepage support: OFF\n", ZEN5_OPTIMIZER_NAME);
#endif

#ifdef DEBUG_OUTPUT
    fprintf(stderr, "[%s] Debug mode: ON\n", ZEN5_OPTIMIZER_NAME);
#endif
}

// Library cleanup
__attribute__((destructor))
static void zen5_optimizer_fini() {
    DEBUG_PRINT("Cleaning up");

    // Release tracked hugepage allocations
    zen5_turbo::cleanup_hugepage_allocations();

    fprintf(stderr, "[%s] Unloaded\n", ZEN5_OPTIMIZER_NAME);
}