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
#define DEBUG_OUTPUT 0

// Memory thresholds
const size_t MIN_SIZE_FOR_HUGEPAGES = 1ULL * 1024 * 1024 * 1024; // 1GB

// Version information
#define ZEN5_OPTIMIZER_VERSION "0.1.0"
#define ZEN5_OPTIMIZER_NAME "zen5-optimizer"

// Debug output control. Gate on the value (#if), not mere definition (#ifdef):
// DEBUG_OUTPUT is always defined (here and via the CMake compile definition), so
// #ifdef would leave debug prints on even in a release build where it is 0.
#if DEBUG_OUTPUT
#define DEBUG_PRINT(fmt, ...) \
    fprintf(stderr, "[%s] " fmt "\n", ZEN5_OPTIMIZER_NAME, ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...) do {} while(0)
#endif