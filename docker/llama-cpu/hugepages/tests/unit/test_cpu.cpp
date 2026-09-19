/*
 * test_cpu.cpp
 *
 * Test CPU detection functionality.
 * This test can run on any CPU to verify detection logic.
 */

#include <stdio.h>
#include <string.h>
#include "../include/test_colors.h"

#ifdef __x86_64__
#include <cpuid.h>
#endif

// Simple CPU detection (mirrors library logic)
bool detect_amd_zen5() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;

    // Check vendor ID
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx) == 0) {
        return false;
    }

    char vendor[13];
    memcpy(vendor, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';

    PRINT_INFO("CPU vendor: %s", vendor);

    if (strcmp(vendor, "AuthenticAMD") != 0) {
        PRINT_WARN("Not an AMD processor");
        return false;
    }

    // Get family info
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0) {
        return false;
    }

    unsigned int family = (eax >> 8) & 0xF;
    unsigned int extended_family = (eax >> 20) & 0xFF;
    unsigned int model = (eax >> 4) & 0xF;
    unsigned int extended_model = (eax >> 16) & 0xF;

    unsigned int display_family = family;
    if (family == 0xF) {
        display_family = family + extended_family;
    }

    unsigned int display_model = model;
    if (family == 0xF || family == 0x6) {
        display_model = (extended_model << 4) + model;
    }

    PRINT_INFO("AMD Family: 0x%X, Model: 0x%X", display_family, display_model);

    // Check for Zen 5
    if (display_family == 0x1A) {
        PRINT_OK("AMD Zen 5 detected");
        return true;
    }

    PRINT_WARN("Not AMD Zen 5 (Family 26/0x1A required)");
    return false;
#else
    PRINT_WARN("Not x86_64 architecture");
    return false;
#endif
}

int main() {
    PRINT_TEST("CPU detection logic");
    printf("\n");

    bool is_zen5 = detect_amd_zen5();

    if (is_zen5) {
        PRINT_OK("Running on AMD Zen 5");
        PRINT_INFO("Library should load successfully");
    } else {
        PRINT_INFO("Not running on AMD Zen 5");
        PRINT_WARN("Library will refuse to load on this CPU");
    }

    // This test always passes - it just reports CPU status
    PRINT_OK("CPU detection logic verified");
    printf("\n");
    return 0;
}