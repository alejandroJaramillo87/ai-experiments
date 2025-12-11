/*
 * cpu_validator.cpp
 *
 * x86-64 AVX-512 CPU detection using CPUID instruction.
 * Supports AMD (Zen 4+) and Intel (Skylake-X, Ice Lake, Sapphire Rapids, etc.)
 */

#include "cpu_validator.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __x86_64__
#include <cpuid.h>
#endif

namespace zen5_turbo {

// Check if current CPU supports AVX-512 Foundation
bool has_avx512_support() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;

    // Check max supported CPUID level (function 0)
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx) == 0) {
        return false;
    }

    // Need at least CPUID level 7 for AVX-512 detection
    if (eax < 7) {
        return false;
    }

    // Check for AVX-512 Foundation (CPUID.7.0:EBX.AVX512F[bit 16])
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) {
        return false;
    }

    // AVX512F is bit 16 of EBX
    bool avx512f = (ebx >> 16) & 1;

    return avx512f;
#else
    // Not x86_64 architecture
    return false;
#endif
}

// Check if current CPU supports AVX-512 VNNI
bool has_avx512_vnni() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;

    // Check max supported CPUID level
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx) == 0 || eax < 7) {
        return false;
    }

    // Check for AVX-512 VNNI (CPUID.7.0:ECX.AVX512_VNNI[bit 11])
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) {
        return false;
    }

    // AVX512_VNNI is bit 11 of ECX
    bool vnni = (ecx >> 11) & 1;

    return vnni;
#else
    return false;
#endif
}

// Get CPU vendor string for informational purposes
static void get_cpu_vendor(char* vendor) {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx)) {
        memcpy(vendor, &ebx, 4);
        memcpy(vendor + 4, &edx, 4);
        memcpy(vendor + 8, &ecx, 4);
        vendor[12] = '\0';
    } else {
        strcpy(vendor, "Unknown");
    }
#else
    strcpy(vendor, "Non-x86");
#endif
}

// Validate CPU has AVX-512 and exit if not supported
void validate_cpu_or_exit() {
    if (!has_avx512_support()) {
        char vendor[13];
        get_cpu_vendor(vendor);
        fprintf(stderr, "[%s] ERROR: CPU does not support AVX-512\n", ZEN5_OPTIMIZER_NAME);
        fprintf(stderr, "[%s] Detected CPU vendor: %s\n", ZEN5_OPTIMIZER_NAME, vendor);
        fprintf(stderr, "[%s] This optimizer requires AVX-512 support:\n", ZEN5_OPTIMIZER_NAME);
        fprintf(stderr, "[%s]   - AMD: Zen 4 (Ryzen 7000+), Zen 5 (Ryzen 9000+)\n", ZEN5_OPTIMIZER_NAME);
        fprintf(stderr, "[%s]   - Intel: Skylake-X, Ice Lake, Sapphire Rapids, etc.\n", ZEN5_OPTIMIZER_NAME);
        exit(1);
    }

    char vendor[13];
    get_cpu_vendor(vendor);
    bool vnni = has_avx512_vnni();

    DEBUG_PRINT("CPU validation: OK (AVX-512 detected)");
    DEBUG_PRINT("  Vendor: %s", vendor);
    DEBUG_PRINT("  AVX-512 VNNI: %s", vnni ? "yes" : "no");
}

} // namespace zen5_turbo