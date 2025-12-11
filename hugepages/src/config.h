/*
 * config.h
 *
 * Configuration parameters for zen5_optimizer.
 * Phase 1A includes hugepage support only.
 */

#pragma once

#include <cstddef>

// Feature flags
#define ENABLE_HUGEPAGES 1
#define DEBUG_OUTPUT 1

// Memory thresholds
const size_t MIN_SIZE_FOR_HUGEPAGES = 1ULL * 1024 * 1024 * 1024; // 1GB

// Version information
#define ZEN5_OPTIMIZER_VERSION "0.1.0"
#define ZEN5_OPTIMIZER_NAME "zen5-optimizer"

// Debug output control
#ifdef DEBUG_OUTPUT
#define DEBUG_PRINT(fmt, ...) \
    fprintf(stderr, "[%s] " fmt "\n", ZEN5_OPTIMIZER_NAME, ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...) do {} while(0)
#endif