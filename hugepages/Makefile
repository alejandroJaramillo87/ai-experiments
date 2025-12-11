# Makefile for zen5_optimizer
#
# Simple Unix-style build system for AMD Zen 5 optimizer library.

# Compiler settings (matching ai-experiments)
CXX = g++-14
CXXFLAGS = -std=c++17 -march=znver5 -mtune=znver5 -O3 -fPIC -Wall -Wextra \
          -ffast-math -fno-finite-math-only
LDFLAGS = -shared -ldl -lpthread

# Directories
SRC_DIR = src
BUILD_DIR = build
TEST_DIR = tests
PREFIX = /usr/local

# Target library
LIB_NAME = libzen5_optimizer.so
LIB_PATH = $(BUILD_DIR)/$(LIB_NAME)

# Source files
SOURCES = $(SRC_DIR)/zen5_optimizer.cpp \
          $(SRC_DIR)/memory/hugepage_wrapper.cpp \
          $(SRC_DIR)/cpu_validator.cpp

OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))

# Test programs
UNIT_TESTS = $(TEST_DIR)/unit/test_load.cpp \
             $(TEST_DIR)/unit/test_cpu.cpp \
             $(TEST_DIR)/unit/test_hugepage.cpp

FUNCTIONAL_TESTS = $(TEST_DIR)/functional/test_memory_boundaries.cpp \
                   $(TEST_DIR)/functional/test_munmap.cpp \
                   $(TEST_DIR)/functional/test_fallback.cpp \
                   $(TEST_DIR)/functional/test_memory_tracking.cpp \
                   $(TEST_DIR)/functional/test_stress.cpp \
                   $(TEST_DIR)/functional/test_performance.cpp

TEST_SOURCES = $(UNIT_TESTS) $(FUNCTIONAL_TESTS)

# Extract test names and build paths
TEST_PROGRAMS = $(patsubst $(TEST_DIR)/unit/%.cpp,$(BUILD_DIR)/%, \
                  $(patsubst $(TEST_DIR)/functional/%.cpp,$(BUILD_DIR)/%, \
                    $(TEST_SOURCES)))

# Default target
all: $(LIB_PATH)

# Build library
$(LIB_PATH): $(OBJECTS)
	@printf "\033[0;36m[BUILD]\033[0m Linking $@\n"
	@$(CXX) $(LDFLAGS) -o $@ $^
	@printf "\033[0;32m[OK]\033[0m Library built: $@\n"

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	@printf "\033[0;36m[BUILD]\033[0m Compiling $<\n"
	@$(CXX) $(CXXFLAGS) -c -o $@ $<

# Build test_hugepage with pthread support
$(BUILD_DIR)/test_hugepage: $(TEST_DIR)/unit/test_hugepage.cpp
	@mkdir -p $(BUILD_DIR)
	@printf "\033[0;36m[BUILD]\033[0m Compiling test: $@ (with pthread)\n"
	@$(CXX) -std=c++17 -o $@ $< -ldl -lpthread

# Build test_stress with pthread support
$(BUILD_DIR)/test_stress: $(TEST_DIR)/functional/test_stress.cpp
	@mkdir -p $(BUILD_DIR)
	@printf "\033[0;36m[BUILD]\033[0m Compiling test: $@ (with pthread)\n"
	@$(CXX) -std=c++17 -o $@ $< -ldl -lpthread

# Build unit test programs
$(BUILD_DIR)/test_%: $(TEST_DIR)/unit/test_%.cpp
	@mkdir -p $(BUILD_DIR)
	@printf "\033[0;36m[BUILD]\033[0m Compiling test: $@\n"
	@$(CXX) -std=c++17 -o $@ $< -ldl

# Build functional test programs
$(BUILD_DIR)/test_%: $(TEST_DIR)/functional/test_%.cpp
	@mkdir -p $(BUILD_DIR)
	@printf "\033[0;36m[BUILD]\033[0m Compiling test: $@\n"
	@$(CXX) -std=c++17 -o $@ $< -ldl

# Run tests with verbose output by default
test: $(LIB_PATH) $(TEST_PROGRAMS)
	@cd $(TEST_DIR) && ./run_tests.sh --verbose

# Run unit tests only (verbose by default)
test-unit: $(LIB_PATH) $(patsubst $(TEST_DIR)/unit/%.cpp,$(BUILD_DIR)/%,$(UNIT_TESTS))
	@cd $(TEST_DIR) && ./run_tests.sh unit --verbose

# Run functional tests only (verbose by default)
test-functional: $(LIB_PATH) $(patsubst $(TEST_DIR)/functional/%.cpp,$(BUILD_DIR)/%,$(FUNCTIONAL_TESTS))
	@cd $(TEST_DIR) && ./run_tests.sh functional --verbose

# Run integration test (verbose by default)
test-integration: $(LIB_PATH)
	@cd $(TEST_DIR) && ./run_tests.sh integration --verbose

# Run all tests with test runner (verbose by default)
test-all: $(LIB_PATH) $(TEST_PROGRAMS)
	@cd $(TEST_DIR) && ./run_tests.sh all --verbose

# Run tests without verbose output (quiet mode)
test-quiet: $(LIB_PATH) $(TEST_PROGRAMS)
	@cd $(TEST_DIR) && ./run_tests.sh

# Install library
install: $(LIB_PATH)
	@printf "\033[0;33m[INSTALL]\033[0m Installing to $(PREFIX)/lib\n"
	@install -D -m 755 $(LIB_PATH) $(PREFIX)/lib/$(LIB_NAME)
	@printf "\033[0;32m[OK]\033[0m Installed to $(PREFIX)/lib/$(LIB_NAME)\n"

# Uninstall library
uninstall:
	@printf "\033[0;33m[UNINSTALL]\033[0m Removing $(PREFIX)/lib/$(LIB_NAME)\n"
	@rm -f $(PREFIX)/lib/$(LIB_NAME)

# Clean build artifacts
clean:
	@printf "\033[0;33m[CLEAN]\033[0m Removing build artifacts\n"
	@rm -rf $(BUILD_DIR)
	@printf "\033[0;32m[OK]\033[0m Clean complete\n"

# Show usage information
help:
	@echo "zen5_optimizer Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build the library (default)"
	@echo "  test             - Build and run all tests (verbose)"
	@echo "  test-unit        - Run unit tests only (verbose)"
	@echo "  test-functional  - Run functional tests only (verbose)"
	@echo "  test-integration - Run integration test only (verbose)"
	@echo "  test-all         - Run all tests with test runner (verbose)"
	@echo "  test-quiet       - Run all tests without verbose output"
	@echo "  install          - Install library to $(PREFIX)/lib"
	@echo "  uninstall        - Remove installed library"
	@echo "  clean            - Remove build artifacts"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  PREFIX    - Installation prefix (default: /usr/local)"
	@echo "  CXX       - C++ compiler (default: g++)"

.PHONY: all test test-unit test-functional test-integration test-all test-quiet install uninstall clean help