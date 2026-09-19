/*
 * test_colors.h
 *
 * Common color definitions for test output.
 * Makes test results easier to read at a glance.
 */

#pragma once

// ANSI color codes
#define COLOR_RESET   "\033[0m"
#define COLOR_RED     "\033[0;31m"
#define COLOR_GREEN   "\033[0;32m"
#define COLOR_YELLOW  "\033[1;33m"
#define COLOR_BLUE    "\033[0;34m"
#define COLOR_CYAN    "\033[0;36m"
#define COLOR_WHITE   "\033[1;37m"
#define COLOR_BOLD    "\033[1m"

// Test output helpers
#define PRINT_OK(fmt, ...) \
    printf(COLOR_GREEN "[OK] " COLOR_RESET fmt "\n", ##__VA_ARGS__)

#define PRINT_FAIL(fmt, ...) \
    fprintf(stderr, COLOR_RED "[FAIL] " COLOR_RESET fmt "\n", ##__VA_ARGS__)

#define PRINT_WARN(fmt, ...) \
    printf(COLOR_YELLOW "[WARN] " COLOR_RESET fmt "\n", ##__VA_ARGS__)

#define PRINT_INFO(fmt, ...) \
    printf(COLOR_CYAN "[INFO] " COLOR_RESET fmt "\n", ##__VA_ARGS__)

#define PRINT_TEST(fmt, ...) \
    printf(COLOR_BOLD "[TEST] " COLOR_RESET fmt "\n", ##__VA_ARGS__)

#define PRINT_RUN(fmt, ...) \
    printf(COLOR_BLUE "[RUN] " COLOR_RESET fmt "\n", ##__VA_ARGS__)
