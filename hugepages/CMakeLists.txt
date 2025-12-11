cmake_minimum_required(VERSION 3.16)
project(zen5_optimizer VERSION 1.0.0 LANGUAGES C CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Configuration options
option(ENABLE_HUGEPAGES "Enable huge page support" ON)
option(ENABLE_DEBUG_PRINT "Enable debug output" OFF)

# Zen 5 specific compiler flags
set(ZEN5_FLAGS
    -march=znver5
    -mtune=znver5
    -mavx512f
    -mavx512vl
    -mavx512bw
    -mavx512dq
    -mavx512cd
    -mavx512vnni
    -mavx512vbmi
    -mavx512vbmi2
    -mavx512ifma
    -mavx512vpopcntdq
)

# Warning flags
set(WARNING_FLAGS
    -Wall
    -Wextra
    -Wpedantic
)

# Optimization flags
set(RELEASE_FLAGS
    -O3
    -ffast-math
    -fno-finite-math-only
    -flto
)

# Add flags based on build type
if(CMAKE_BUILD_TYPE MATCHES Release)
    add_compile_options(${ZEN5_FLAGS} ${WARNING_FLAGS} ${RELEASE_FLAGS})
else()
    add_compile_options(${ZEN5_FLAGS} ${WARNING_FLAGS} -O0 -g)
endif()

# Library source files
set(LIB_SOURCES
    src/zen5_optimizer.cpp
    src/cpu_validator.cpp
    src/memory/hugepage_wrapper.cpp
)

# Create shared library
add_library(zen5_optimizer SHARED ${LIB_SOURCES})

# Set library properties
set_target_properties(zen5_optimizer PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(zen5_optimizer
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Compile definitions
# Note: ENABLE_HUGEPAGES, DEBUG_OUTPUT, MIN_SIZE_FOR_HUGEPAGES, and
# ZEN5_OPTIMIZER_NAME are already defined in src/config.h
# CMake options here override config.h defaults if needed

if(NOT ENABLE_HUGEPAGES)
    target_compile_definitions(zen5_optimizer PRIVATE ENABLE_HUGEPAGES=0)
endif()

if(ENABLE_DEBUG_PRINT)
    target_compile_definitions(zen5_optimizer PRIVATE DEBUG_OUTPUT=1)
else()
    target_compile_definitions(zen5_optimizer PRIVATE DEBUG_OUTPUT=0)
endif()

# Link libraries
target_link_libraries(zen5_optimizer
    PRIVATE
        dl  # For dlsym
)

# Install targets
install(TARGETS zen5_optimizer
    LIBRARY DESTINATION lib
)

# Add tests subdirectory only if it exists (optional for production builds)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print configuration summary
message(STATUS "===========================================")
message(STATUS "zen5_optimizer Configuration")
message(STATUS "===========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Enable hugepages: ${ENABLE_HUGEPAGES}")
message(STATUS "Enable debug print: ${ENABLE_DEBUG_PRINT}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "===========================================")
