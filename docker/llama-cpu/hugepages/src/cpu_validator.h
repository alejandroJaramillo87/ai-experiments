/*
 * cpu_validator.h
 *
 * x86-64 AVX-512 CPU detection and validation.
 * Ensures the library only runs on CPUs with AVX-512 support.
 * Supports AMD Zen 4+, Intel Skylake-X, Ice Lake, Sapphire Rapids, etc.
 */

#pragma once

namespace zen5_turbo {

// Check if current CPU supports AVX-512 (Foundation instruction set)
bool has_avx512_support();

// Check if current CPU supports AVX-512 VNNI (Vector Neural Network Instructions)
bool has_avx512_vnni();

// Validate CPU has AVX-512 and exit if not supported
void validate_cpu_or_exit();

// Legacy alias for backward compatibility
inline bool is_zen5_cpu() { return has_avx512_support(); }
inline void validate_zen5_or_exit() { validate_cpu_or_exit(); }

} // namespace zen5_turbo