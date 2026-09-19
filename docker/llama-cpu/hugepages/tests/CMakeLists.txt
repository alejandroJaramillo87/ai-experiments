# tests/CMakeLists.txt
# Build configuration for zen5_optimizer test suite

# Test include directories
set(TEST_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Compiler flags for tests
set(TEST_CXX_FLAGS
    -std=c++11
    -Wall
    -Wextra
)

# Function to create a test executable
function(add_zen5_test test_name test_source)
    add_executable(${test_name} ${test_source})

    target_include_directories(${test_name}
        PRIVATE
            ${TEST_INCLUDE_DIRS}
    )

    target_compile_options(${test_name}
        PRIVATE
            ${TEST_CXX_FLAGS}
    )

    # Link against pthread for concurrent tests
    target_link_libraries(${test_name}
        PRIVATE
            pthread
    )

    # Set output directory
    set_target_properties(${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    # Add as CTest
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Unit tests
file(GLOB UNIT_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_*.cpp")
foreach(test_source ${UNIT_TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_zen5_test(${test_name} ${test_source})
endforeach()

# Functional tests
file(GLOB FUNCTIONAL_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/functional/test_*.cpp")
foreach(test_source ${FUNCTIONAL_TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_zen5_test(${test_name} ${test_source})
endforeach()

# Integration tests (shell scripts - don't need to be compiled)
# These are run by tests/run_tests.sh

# Print test configuration
message(STATUS "Test configuration:")
message(STATUS "  Unit tests: ${UNIT_TEST_SOURCES}")
message(STATUS "  Functional tests: ${FUNCTIONAL_TEST_SOURCES}")
