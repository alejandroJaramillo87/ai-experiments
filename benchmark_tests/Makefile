# AI Workstation Benchmark Tests
# Professional Grade Test Automation Makefile
#
# This Makefile provides comprehensive test automation for the AI workstation
# benchmark testing suite with automatic cleanup and professional reporting.
#
# Author: Claude Code
# Version: 2.0.0

# Configuration
PYTHON := python3
PYTEST := python -m pytest
TEST_TIMEOUT := 300
COVERAGE_MIN := 80

# Test directories
UNIT_TEST_DIR := tests/unit
INTEGRATION_TEST_DIR := tests/integration
ANALYSIS_TEST_DIR := tests/analysis
MODULAR_TEST_DIR := tests/unit/modular
PHASE2_TEST_DIR := tests/unit/modular/phase2
DEBUG_DIR := tests/debug

# Common pytest arguments
PYTEST_BASE_ARGS := --tb=short --strict-markers --disable-warnings
PYTEST_VERBOSE_ARGS := $(PYTEST_BASE_ARGS) -v -s
PYTEST_QUIET_ARGS := $(PYTEST_BASE_ARGS) -q
PYTEST_COVERAGE_ARGS := $(PYTEST_BASE_ARGS) --cov=evaluator --cov-report=term-missing --cov-fail-under=$(COVERAGE_MIN)

# Color output
BOLD := \033[1m
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
MAGENTA := \033[35m
CYAN := \033[36m
RESET := \033[0m

# Phony targets
.PHONY: help test test-unit test-integration test-modular test-phase2 test-analysis test-quick test-verbose test-coverage test-specific test-watch clean clean-all check-env setup debug-registry debug-help

# Default target
help:
	@echo "$(BOLD)$(BLUE)🧪 AI Workstation Benchmark Tests$(RESET)"
	@echo "$(BOLD)================================$(RESET)"
	@echo ""
	@echo "$(BOLD)Core Test Commands:$(RESET)"
	@echo "  $(GREEN)make test$(RESET)              - Run all tests (unit + integration + modular)"
	@echo "  $(GREEN)make test-unit$(RESET)         - Run unit tests only"
	@echo "  $(GREEN)make test-integration$(RESET)  - Run integration tests only"
	@echo "  $(GREEN)make test-modular$(RESET)      - Run modular evaluator tests only"
	@echo "  $(GREEN)make test-phase2$(RESET)       - Run Phase 2 core functionality tests only"
	@echo "  $(GREEN)make test-analysis$(RESET)     - Run analysis and validation scripts"
	@echo ""
	@echo "$(BOLD)Debug Tools:$(RESET)"
	@echo "  $(GREEN)make debug-registry$(RESET)    - Debug evaluator registry issues"
	@echo "  $(GREEN)make debug-help$(RESET)        - Show available debug utilities"
	@echo ""
	@echo "$(BOLD)Analysis Tools:$(RESET)"
	@echo "  $(GREEN)make domain-audit$(RESET)      - Show comprehensive domain coverage analysis"
	@echo ""
	@echo "$(BOLD)Enhancement Testing:$(RESET)"
	@echo "  $(GREEN)make test-enhanced-evaluator$(RESET)     - Test Phase 1 enhanced universal evaluator"
	@echo "  $(GREEN)make test-phase1-quality$(RESET)         - Test Phase 1 quality fixes validation"
	@echo "  $(GREEN)make convert-creativity-tests$(RESET)    - Convert creativity base tests to instruct format"
	@echo "  $(GREEN)make test-creativity-conversion$(RESET)  - Test enhanced evaluator with converted creativity tests"
	@echo "  $(GREEN)make convert-core-domains$(RESET)        - Convert core domains to instruct format (680 tests)"
	@echo "  $(GREEN)make test-core-domains-conversion$(RESET) - Test enhanced evaluator with converted core domain tests"
	@echo ""
	@echo "$(BOLD)Development Commands:$(RESET)"
	@echo "  $(GREEN)make test-quick$(RESET)        - Run essential test subset for quick validation"
	@echo "  $(GREEN)make test-verbose$(RESET)      - Run all tests with verbose output"
	@echo "  $(GREEN)make test-coverage$(RESET)     - Run tests with coverage reporting"
	@echo "  $(GREEN)make test-watch$(RESET)        - Continuously run tests on file changes"
	@echo ""
	@echo "$(BOLD)Specific Testing:$(RESET)"
	@echo "  $(GREEN)make test-specific FILE=path/to/test.py$(RESET)  - Run specific test file"
	@echo "  $(GREEN)make test-specific FILE=path::TestClass::test_method$(RESET)  - Run specific test method"
	@echo ""
	@echo "$(BOLD)Maintenance:$(RESET)"
	@echo "  $(GREEN)make clean$(RESET)             - Clean test artifacts and cache"
	@echo "  $(GREEN)make clean-all$(RESET)         - Deep clean including coverage reports"
	@echo "  $(GREEN)make check-env$(RESET)         - Verify test environment setup"
	@echo "  $(GREEN)make setup$(RESET)             - Install test dependencies"
	@echo ""
	@echo "$(BOLD)Examples:$(RESET)"
	@echo "  $(CYAN)make test ARGS='-k test_evaluator'$(RESET)        # Run tests matching pattern"
	@echo "  $(CYAN)make test-unit ARGS='--maxfail=1'$(RESET)         # Stop after first failure"
	@echo "  $(CYAN)make test-specific FILE=tests/unit/test_universal_evaluator.py$(RESET)"
	@echo "  $(CYAN)make test-coverage ARGS='--cov-report=html'$(RESET)  # Generate HTML coverage"
	@echo ""
	@echo "$(BOLD)Environment Variables:$(RESET)"
	@echo "  $(YELLOW)ARGS$(RESET)     - Additional pytest arguments"
	@echo "  $(YELLOW)FILE$(RESET)     - Specific test file/method for test-specific"
	@echo "  $(YELLOW)TIMEOUT$(RESET)  - Test timeout in seconds (default: $(TEST_TIMEOUT))"

# Environment check
check-env:
	@echo "$(BOLD)$(BLUE)🔍 Checking Test Environment$(RESET)"
	@echo "=============================="
	@$(PYTHON) --version || (echo "$(RED)❌ Python 3 not found$(RESET)" && exit 1)
	@$(PYTHON) -c "import pytest; print('✅ pytest available:', pytest.__version__)" || (echo "$(RED)❌ pytest not installed$(RESET)" && exit 1)
	@$(PYTHON) -c "import sys; print('✅ Python path:', sys.executable)"
	@echo "✅ Working directory: $(PWD)"
	@echo "✅ Test timeout: $(TEST_TIMEOUT) seconds"
	@echo "$(GREEN)Environment check passed!$(RESET)"

# Setup test dependencies
setup:
	@echo "$(BOLD)$(BLUE)📦 Installing Test Dependencies$(RESET)"
	@echo "================================"
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install pytest pytest-cov pytest-mock pytest-asyncio
	@echo "$(GREEN)✅ Dependencies installed!$(RESET)"

# Pre-test cleanup and setup
pre-test:
	@echo "$(BOLD)$(YELLOW)🧹 Pre-test Cleanup$(RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# Post-test cleanup
post-test:
	@echo "$(BOLD)$(YELLOW)🧹 Post-test Cleanup$(RESET)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup complete!$(RESET)"

# Run all tests
test: pre-test
	@echo "$(BOLD)$(BLUE)🧪 Running All Tests$(RESET)"
	@echo "====================="
	@$(PYTEST) $(PYTEST_BASE_ARGS) $(UNIT_TEST_DIR) $(INTEGRATION_TEST_DIR) $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Run unit tests only
test-unit: pre-test
	@echo "$(BOLD)$(BLUE)🔬 Running Unit Tests$(RESET)"
	@echo "======================"
	@$(PYTEST) $(PYTEST_QUIET_ARGS) $(UNIT_TEST_DIR) $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Run integration tests only
test-integration: pre-test
	@echo "$(BOLD)$(BLUE)🔗 Running Integration Tests$(RESET)"
	@echo "============================="
	@$(PYTEST) $(PYTEST_QUIET_ARGS) $(INTEGRATION_TEST_DIR) $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Run modular evaluator tests
test-modular: pre-test
	@echo "$(BOLD)$(BLUE)🧩 Running Modular Evaluator Tests$(RESET)"
	@echo "==================================="
	@$(PYTEST) $(PYTEST_VERBOSE_ARGS) $(MODULAR_TEST_DIR) $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Run Phase 2 core functionality tests
test-phase2: pre-test
	@echo "$(BOLD)$(BLUE)⚡ Running Phase 2 Core Functionality Tests$(RESET)"
	@echo "============================================="
	@$(PYTEST) $(PYTEST_VERBOSE_ARGS) $(PHASE2_TEST_DIR) $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Run analysis scripts
test-analysis: pre-test
	@echo "$(BOLD)$(BLUE)🔍 Running Analysis Scripts$(RESET)"
	@echo "============================"
	@if [ -f "$(ANALYSIS_TEST_DIR)/analyze_scoring_patterns.py" ]; then \
		$(PYTHON) $(ANALYSIS_TEST_DIR)/analyze_scoring_patterns.py; \
	fi
	@if [ -f "$(INTEGRATION_TEST_DIR)/comprehensive_validation_suite.py" ]; then \
		$(PYTHON) $(INTEGRATION_TEST_DIR)/comprehensive_validation_suite.py; \
	fi
	@if [ -f "$(INTEGRATION_TEST_DIR)/run_edge_case_tests.py" ]; then \
		$(PYTHON) $(INTEGRATION_TEST_DIR)/run_edge_case_tests.py; \
	fi
	@$(MAKE) post-test

# Quick test subset for development
test-quick: pre-test
	@echo "$(BOLD)$(BLUE)⚡ Running Quick Test Subset$(RESET)"
	@echo "============================="
	@$(PYTEST) $(PYTEST_QUIET_ARGS) \
		$(UNIT_TEST_DIR)/test_universal_evaluator.py \
		$(UNIT_TEST_DIR)/test_evaluation_config.py \
		$(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Verbose test run
test-verbose: pre-test
	@echo "$(BOLD)$(BLUE)📢 Running Tests (Verbose)$(RESET)"
	@echo "==========================="
	@$(PYTEST) $(PYTEST_VERBOSE_ARGS) $(UNIT_TEST_DIR) $(INTEGRATION_TEST_DIR) $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Test with coverage
test-coverage: pre-test
	@echo "$(BOLD)$(BLUE)📊 Running Tests with Coverage$(RESET)"
	@echo "==============================="
	@$(PYTEST) $(PYTEST_COVERAGE_ARGS) $(UNIT_TEST_DIR) $(INTEGRATION_TEST_DIR) $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Test specific file or method
test-specific: pre-test
	@if [ -z "$(FILE)" ]; then \
		echo "$(RED)❌ Error: FILE parameter required$(RESET)"; \
		echo "Usage: make test-specific FILE=path/to/test.py"; \
		echo "   or: make test-specific FILE=path::TestClass::test_method"; \
		exit 1; \
	fi
	@echo "$(BOLD)$(BLUE)🎯 Running Specific Test: $(FILE)$(RESET)"
	@echo "=========================================="
	@$(PYTEST) $(PYTEST_VERBOSE_ARGS) $(FILE) $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Watch for file changes and run tests
test-watch:
	@echo "$(BOLD)$(BLUE)👀 Watching for File Changes$(RESET)"
	@echo "============================="
	@echo "$(YELLOW)Press Ctrl+C to stop$(RESET)"
	@while true; do \
		$(MAKE) test-quick; \
		echo ""; \
		echo "$(CYAN)⏱️  Waiting for changes... (modify any .py file to re-run)$(RESET)"; \
		if command -v inotifywait >/dev/null 2>&1; then \
			inotifywait -q -e modify,create,delete -r . --include='.*\.py$$' 2>/dev/null; \
		else \
			echo "$(YELLOW)⚠️  inotifywait not available, using polling$(RESET)"; \
			sleep 5; \
		fi; \
		clear; \
	done

# Clean test artifacts
clean:
	@echo "$(BOLD)$(YELLOW)🧹 Cleaning Test Artifacts$(RESET)"
	@echo "==========================="
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Basic cleanup complete!$(RESET)"

# Deep clean including coverage reports
clean-all: clean
	@echo "$(BOLD)$(YELLOW)🧹 Deep Cleaning$(RESET)"
	@echo "================="
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".coverage.*" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "coverage.xml" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✅ Deep cleanup complete!$(RESET)"

# Debug target for troubleshooting
debug:
	@echo "$(BOLD)$(BLUE)🐛 Debug Information$(RESET)"
	@echo "===================="
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Pytest: $(shell $(PYTHON) -c 'import pytest; print(pytest.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "Working Directory: $(PWD)"
	@echo "Test Directories:"
	@echo "  Unit: $(UNIT_TEST_DIR) $(shell [ -d $(UNIT_TEST_DIR) ] && echo '✅' || echo '❌')"
	@echo "  Integration: $(INTEGRATION_TEST_DIR) $(shell [ -d $(INTEGRATION_TEST_DIR) ] && echo '✅' || echo '❌')"
	@echo "  Modular: $(MODULAR_TEST_DIR) $(shell [ -d $(MODULAR_TEST_DIR) ] && echo '✅' || echo '❌')"
	@echo "  Debug: $(DEBUG_DIR) $(shell [ -d $(DEBUG_DIR) ] && echo '✅' || echo '❌')"
	@echo "Environment Variables:"
	@echo "  ARGS: '$(ARGS)'"
	@echo "  FILE: '$(FILE)'"
	@echo "  TIMEOUT: $(TEST_TIMEOUT)"

# Debug utilities
debug-help:
	@echo "$(BOLD)$(BLUE)🐛 Debug Utilities$(RESET)"
	@echo "=================="
	@echo ""
	@echo "$(BOLD)Available Debug Scripts:$(RESET)"
	@echo "  $(GREEN)make debug-enhanced-system$(RESET)      - Debug enhanced universal evaluator system (comprehensive)"
	@echo "  $(GREEN)make debug-enhanced-evaluator$(RESET)   - Debug Phase 1 enhanced universal evaluator (detailed)"
	@echo "  $(GREEN)make debug-scoring-calibration$(RESET)  - Debug Phase 1 scoring calibration fixes"
	@echo ""
	@echo "$(BOLD)Usage:$(RESET)"
	@echo "  $(CYAN)make debug-enhanced-system$(RESET)       # Run comprehensive enhanced system debug"
	@echo "  $(CYAN)make debug-enhanced-evaluator$(RESET)     # Run enhanced evaluator debug with multi-tier scoring"
	@echo "  $(CYAN)make debug-scoring-calibration$(RESET)    # Test and validate Phase 1 scoring calibration fixes"

# Debug enhanced universal evaluator system (comprehensive)
debug-enhanced-system:
	@echo "$(BOLD)$(BLUE)🔧 Debugging Enhanced Universal Evaluator System$(RESET)"
	@echo "====================================================="
	@$(PYTHON) $(DEBUG_DIR)/debug_enhanced_system.py

# Debug enhanced universal evaluator
debug-enhanced-evaluator:
	@echo "$(BOLD)$(BLUE)🔧 Debugging Enhanced Universal Evaluator$(RESET)"
	@echo "============================================"
	@$(PYTHON) $(DEBUG_DIR)/debug_enhanced_evaluator.py

# Debug scoring calibration fixes
debug-scoring-calibration:
	@echo "$(BOLD)$(BLUE)🎯 Debugging Scoring Calibration Fixes$(RESET)"
	@echo "========================================="
	@$(PYTHON) $(DEBUG_DIR)/debug_scoring_calibration.py

# Domain coverage analysis
domain-audit:
	@echo "$(BOLD)$(BLUE)📊 Domain Coverage Analysis$(RESET)"
	@echo "============================"
	@echo "$(CYAN)📋 Comprehensive audit report available:$(RESET)"
	@echo "  docs/domain_coverage_audit_report.md"
	@echo ""
	@echo "$(CYAN)🔧 Evaluator requirements specification:$(RESET)" 
	@echo "  docs/evaluator_requirements_specification.md"
	@echo ""
	@echo "$(BOLD)Key Findings:$(RESET)"
	@echo "  • $(GREEN)30 domains analyzed$(RESET) across 3 sophistication tiers"
	@echo "  • $(YELLOW)6 production-ready domains$(RESET) with 200+ tests each"
	@echo "  • $(MAGENTA)10 specialized research domains$(RESET) requiring advanced evaluators"
	@echo "  • $(RED)Base/Instruct imbalance$(RESET) across most domains"
	@echo ""
	@echo "$(BOLD)Next Steps:$(RESET)"
	@echo "  1. $(CYAN)Expand instruct model coverage$(RESET) for core domains"
	@echo "  2. $(CYAN)Develop specialized evaluators$(RESET) for research domains"
	@echo "  3. $(CYAN)Implement advanced scoring$(RESET) for quantum philosophy content"

# Test enhanced universal evaluator
test-enhanced-evaluator: pre-test
	@echo "$(BOLD)$(BLUE)🔬 Testing Enhanced Universal Evaluator$(RESET)"
	@echo "=========================================="
	@$(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/validation/test_enhanced_universal_evaluator.py $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Test Phase 1 quality fixes
test-phase1-quality: pre-test
	@echo "$(BOLD)$(BLUE)🔧 Testing Phase 1 Quality Fixes$(RESET)"
	@echo "=================================="
	@$(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/validation/test_phase1_quality_fixes.py $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Convert creativity base models to instruct models  
convert-creativity-tests:
	@echo "$(BOLD)$(BLUE)🎭 Converting Creativity Base Models to Instruct Models$(RESET)"
	@echo "========================================================="
	@$(PYTHON) scripts/convert_base_to_instruct_creativity.py

# Test enhanced evaluator with converted creativity tests
test-creativity-conversion:
	@echo "$(BOLD)$(BLUE)🎭 Testing Enhanced Evaluator with Converted Creativity Tests$(RESET)"
	@echo "================================================================="
	@$(PYTHON) $(DEBUG_DIR)/test_creativity_conversion.py

# Convert core domains (language, integration, knowledge, social) to instruct format
convert-core-domains:
	@echo "$(BOLD)$(BLUE)🌍 Converting Core Domains to Instruct Model Format$(RESET)"
	@echo "=========================================================="
	@$(PYTHON) scripts/convert_core_domains_to_instruct.py

# Test enhanced evaluator with converted core domain tests
test-core-domains-conversion:
	@echo "$(BOLD)$(BLUE)🌍 Testing Enhanced Evaluator with Converted Core Domain Tests$(RESET)"
	@echo "======================================================================"
	@$(PYTHON) $(DEBUG_DIR)/test_core_domains_conversion.py