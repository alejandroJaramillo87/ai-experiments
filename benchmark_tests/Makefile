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
CALIBRATION_TEST_DIR := tests/calibration
ANALYSIS_TEST_DIR := tests/analysis
MODULAR_TEST_DIR := tests/unit/modular
PHASE2_TEST_DIR := tests/unit/modular/phase2
DEBUG_DIR := tests/debug
EXAMPLES_DIR := examples

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
.PHONY: help test test-unit test-integration test-calibration test-modular test-phase2 test-analysis test-quick test-verbose test-coverage test-specific test-watch clean clean-all check-env setup debug-registry debug-help calibration-demo examples check-prerequisites check-docker check-server check-model check-calibration check-system calibration-status calibration-validate

# Default target
help:
	@echo "$(BOLD)$(BLUE)🧪 AI Workstation Benchmark Tests$(RESET)"
	@echo "$(BOLD)================================$(RESET)"
	@echo ""
	@echo "$(BOLD)Core Test Commands:$(RESET)"
	@echo "  $(GREEN)make test$(RESET)              - Run all tests (unit + integration + modular)"
	@echo "  $(GREEN)make test-unit$(RESET)         - Run unit tests only"
	@echo "  $(GREEN)make test-integration$(RESET)  - Run integration tests only"
	@echo "  $(GREEN)make test-calibration$(RESET)  - Run calibration validation framework tests"
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
	@echo "$(BOLD)Calibration & Quality Assurance:$(RESET)"
	@echo "  $(GREEN)make calibration-validate$(RESET)        - Run live calibration validation (requires LLM server)"
	@echo "  $(GREEN)make calibration-demo$(RESET)            - Run calibration framework demo"
	@echo "  $(GREEN)make examples$(RESET)                    - Run example scripts and demos"
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
	@echo "$(BOLD)Prerequisite Checking:$(RESET)"
	@echo "  $(GREEN)make check-prerequisites$(RESET)  - Comprehensive calibration readiness check"
	@echo "  $(GREEN)make check-docker$(RESET)         - Check Docker container status"
	@echo "  $(GREEN)make check-server$(RESET)         - Test LLM server connectivity"
	@echo "  $(GREEN)make check-model$(RESET)          - Verify model loading and response"
	@echo "  $(GREEN)make check-calibration$(RESET)    - Check calibration framework readiness"
	@echo "  $(GREEN)make check-system$(RESET)         - System resources and GPU status"
	@echo "  $(GREEN)make calibration-status$(RESET)   - Complete status dashboard"
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

# Run calibration validation framework tests  
test-calibration: pre-test
	@echo "$(BOLD)$(BLUE)🎯 Running Calibration Validation Framework Tests$(RESET)"
	@echo "==================================================="
	@echo "$(CYAN)🧪 Running calibration unit tests directly...$(RESET)"
	@cd $(CALIBRATION_TEST_DIR) && $(PYTHON) test_calibration_suite.py
	@$(MAKE) post-test

# Run live calibration validation (requires LLM server)
calibration-validate:
	@echo "$(BOLD)$(BLUE)🎯 Running Live Calibration Validation$(RESET)"
	@echo "========================================"
	@echo "$(YELLOW)⚠️  This requires a running LLM server at http://localhost:8004$(RESET)"
	@echo ""
	@cd $(CALIBRATION_TEST_DIR) && $(PYTHON) calibration_validator.py

# Run calibration framework demo
calibration-demo:
	@echo "$(BOLD)$(BLUE)🎯 Calibration Framework Demo$(RESET)"
	@echo "=============================="
	@echo "$(CYAN)📋 Validating reference test cases...$(RESET)"
	@cd $(CALIBRATION_TEST_DIR) && $(PYTHON) reference_test_cases.py
	@echo ""
	@echo "$(CYAN)🧪 Running calibration unit tests...$(RESET)"
	@cd $(CALIBRATION_TEST_DIR) && $(PYTHON) test_calibration_suite.py

# Run example scripts and demos  
examples:
	@echo "$(BOLD)$(BLUE)🎭 Running Example Scripts and Demos$(RESET)"
	@echo "======================================"
	@if [ -f "$(EXAMPLES_DIR)/enhanced_evaluation_demo.py" ]; then \
		echo "$(CYAN)🔬 Enhanced Evaluation Demo:$(RESET)"; \
		cd $(EXAMPLES_DIR) && $(PYTHON) enhanced_evaluation_demo.py; \
		echo ""; \
	fi
	@echo "$(GREEN)✅ Examples complete!$(RESET)"

# Enhanced integration test commands
test-enhanced-integration: pre-test
	@echo "$(BOLD)$(BLUE)🚀 Running Enhanced Integration Tests$(RESET)"
	@echo "======================================"
	@if [ -f "$(INTEGRATION_TEST_DIR)/test_enhanced_evaluation.py" ]; then \
		echo "$(CYAN)🔬 Testing Enhanced Evaluation Integration:$(RESET)"; \
		$(PYTHON) $(INTEGRATION_TEST_DIR)/test_enhanced_evaluation.py; \
		echo ""; \
	fi
	@if [ -f "$(INTEGRATION_TEST_DIR)/test_integration_full.py" ]; then \
		echo "$(CYAN)🔗 Testing Full Integration:$(RESET)"; \
		$(PYTHON) $(INTEGRATION_TEST_DIR)/test_integration_full.py; \
		echo ""; \
	fi
	@$(MAKE) post-test

# ========================================================================
# PREREQUISITE CHECKING COMMANDS
# ========================================================================

# Docker container status check
check-docker:
	@echo "$(BOLD)$(BLUE)🐳 Docker Container Status Check$(RESET)"
	@echo "=================================="
	@echo ""
	@echo "$(CYAN)🔍 Checking Docker service status:$(RESET)"
	@if ! command -v docker >/dev/null 2>&1; then \
		echo "$(RED)❌ Docker command not found$(RESET)"; \
		echo "$(YELLOW)💡 Install Docker first$(RESET)"; \
		exit 1; \
	fi
	@if ! docker info >/dev/null 2>&1; then \
		echo "$(RED)❌ Docker daemon not running$(RESET)"; \
		echo "$(YELLOW)💡 Start Docker service: sudo systemctl start docker$(RESET)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✅ Docker service running$(RESET)"
	@echo ""
	@echo "$(CYAN)🔍 Checking for llama-gpu containers:$(RESET)"
	@if docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(llama|gpu)" >/dev/null 2>&1; then \
		echo "$(GREEN)✅ LLM containers found:$(RESET)"; \
		docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | head -1; \
		docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(llama|gpu)" | sed 's/^/  /'; \
	else \
		echo "$(YELLOW)🟡 No llama/gpu containers running$(RESET)"; \
		echo "$(CYAN)📋 All running containers:$(RESET)"; \
		docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | sed 's/^/  /'; \
	fi
	@echo ""
	@echo "$(CYAN)🔍 Checking port 8004 mapping:$(RESET)"
	@if docker ps --format "{{.Ports}}" | grep ":8004" >/dev/null 2>&1; then \
		echo "$(GREEN)✅ Port 8004 is mapped$(RESET)"; \
		docker ps --format "{{.Names}}: {{.Ports}}" | grep ":8004" | sed 's/^/  /'; \
	else \
		echo "$(YELLOW)🟡 Port 8004 not found in container mappings$(RESET)"; \
		echo "$(YELLOW)💡 Check if your LLM container exposes port 8004$(RESET)"; \
	fi

# LLM server connectivity check
check-server:
	@echo "$(BOLD)$(BLUE)🌐 LLM Server Connectivity Check$(RESET)"
	@echo "================================="
	@echo ""
	@echo "$(CYAN)🔍 Testing port 8004 connectivity:$(RESET)"
	@if command -v nc >/dev/null 2>&1; then \
		if nc -z localhost 8004 2>/dev/null; then \
			echo "$(GREEN)✅ Port 8004 is reachable$(RESET)"; \
		else \
			echo "$(RED)❌ Port 8004 not reachable$(RESET)"; \
			echo "$(YELLOW)💡 Ensure LLM container is running and port is mapped$(RESET)"; \
			exit 1; \
		fi; \
	else \
		echo "$(YELLOW)🟡 netcat not available, skipping port test$(RESET)"; \
	fi
	@echo ""
	@echo "$(CYAN)🔍 Testing HTTP endpoint connectivity:$(RESET)"
	@if command -v curl >/dev/null 2>&1; then \
		echo "$(CYAN)  Testing http://localhost:8004/health...$(RESET)"; \
		if curl -s --connect-timeout 5 "http://localhost:8004/health" >/dev/null 2>&1; then \
			echo "$(GREEN)✅ Health endpoint responding$(RESET)"; \
		else \
			echo "$(YELLOW)🟡 Health endpoint not available$(RESET)"; \
			echo "$(CYAN)  Testing http://localhost:8004/...$(RESET)"; \
			if curl -s --connect-timeout 5 "http://localhost:8004/" >/dev/null 2>&1; then \
				echo "$(GREEN)✅ Base endpoint responding$(RESET)"; \
			else \
				echo "$(RED)❌ Server not responding to HTTP requests$(RESET)"; \
				echo "$(YELLOW)💡 Check if LLM server is fully started$(RESET)"; \
				exit 1; \
			fi; \
		fi; \
	else \
		echo "$(YELLOW)🟡 curl not available, skipping HTTP test$(RESET)"; \
	fi
	@echo "$(GREEN)✅ Server connectivity check complete$(RESET)"

# Model loading and response test
check-model:
	@echo "$(BOLD)$(BLUE)🤖 Model Loading and Response Test$(RESET)"
	@echo "===================================="
	@echo ""
	@echo "$(CYAN)🔍 Testing model inference capability:$(RESET)"
	@if command -v curl >/dev/null 2>&1; then \
		echo "$(CYAN)  Sending test completion request...$(RESET)"; \
		response=$$(curl -s --connect-timeout 10 -X POST \
			-H "Content-Type: application/json" \
			-d '{"prompt":"Test","max_tokens":5,"temperature":0.1}' \
			"http://localhost:8004/v1/completions" 2>/dev/null || \
			curl -s --connect-timeout 10 -X POST \
			-H "Content-Type: application/json" \
			-d '{"prompt":"Test","n_predict":5,"temperature":0.1}' \
			"http://localhost:8004/completion" 2>/dev/null); \
		if [ -n "$$response" ] && echo "$$response" | grep -E "(choices|content)" >/dev/null 2>&1; then \
			echo "$(GREEN)✅ Model responding with valid completions$(RESET)"; \
			echo "$(CYAN)  Sample response length: $$(echo "$$response" | wc -c) chars$(RESET)"; \
		elif [ -n "$$response" ]; then \
			echo "$(YELLOW)🟡 Server responding but format unclear$(RESET)"; \
			echo "$(CYAN)  Response preview: $$(echo "$$response" | head -c 100)...$(RESET)"; \
		else \
			echo "$(RED)❌ No response from model endpoint$(RESET)"; \
			echo "$(YELLOW)💡 Check if model is loaded and server is ready$(RESET)"; \
			exit 1; \
		fi; \
	else \
		echo "$(YELLOW)🟡 curl not available, skipping model test$(RESET)"; \
	fi
	@echo "$(GREEN)✅ Model test complete$(RESET)"

# Calibration framework readiness check
check-calibration:
	@echo "$(BOLD)$(BLUE)🎯 Calibration Framework Readiness Check$(RESET)"
	@echo "============================================="
	@echo ""
	@echo "$(CYAN)🔍 Checking calibration framework files:$(RESET)"
	@calibration_files="$(CALIBRATION_TEST_DIR)/calibration_validator.py $(CALIBRATION_TEST_DIR)/reference_test_cases.py $(CALIBRATION_TEST_DIR)/calibration_reporter.py $(CALIBRATION_TEST_DIR)/test_calibration_suite.py"; \
	all_present=true; \
	for file in $$calibration_files; do \
		if [ -f "$$file" ]; then \
			echo "$(GREEN)✅ $$file$(RESET)"; \
		else \
			echo "$(RED)❌ Missing: $$file$(RESET)"; \
			all_present=false; \
		fi; \
	done; \
	if [ "$$all_present" = "false" ]; then \
		echo "$(RED)❌ Some calibration files missing$(RESET)"; \
		exit 1; \
	fi
	@echo ""
	@echo "$(CYAN)🔍 Testing Python imports:$(RESET)"
	@cd $(CALIBRATION_TEST_DIR) && \
	if $(PYTHON) -c "import calibration_validator, reference_test_cases, calibration_reporter" 2>/dev/null; then \
		echo "$(GREEN)✅ All calibration modules importable$(RESET)"; \
	else \
		echo "$(RED)❌ Import errors in calibration modules$(RESET)"; \
		echo "$(YELLOW)💡 Check Python path and dependencies$(RESET)"; \
		exit 1; \
	fi
	@echo ""
	@echo "$(CYAN)🔍 Checking benchmark runner integration:$(RESET)"
	@if [ -f "benchmark_runner.py" ]; then \
		echo "$(GREEN)✅ benchmark_runner.py found$(RESET)"; \
		if $(PYTHON) -c "from benchmark_runner import BenchmarkTestRunner" 2>/dev/null; then \
			echo "$(GREEN)✅ BenchmarkTestRunner importable$(RESET)"; \
		else \
			echo "$(YELLOW)🟡 BenchmarkTestRunner import issues$(RESET)"; \
		fi; \
	else \
		echo "$(RED)❌ benchmark_runner.py not found$(RESET)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✅ Calibration framework ready$(RESET)"

# System resources and GPU status check
check-system:
	@echo "$(BOLD)$(BLUE)💻 System Resources and GPU Status$(RESET)"
	@echo "===================================="
	@echo ""
	@echo "$(CYAN)🔍 GPU Status:$(RESET)"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "$(GREEN)✅ NVIDIA GPU detected$(RESET)"; \
		nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits | \
		awk -F',' '{printf "  GPU: %s | VRAM: %s/%s GB | Util: %s%%\n", $$1, int($$3/1024), int($$2/1024), $$4}'; \
	else \
		echo "$(YELLOW)🟡 No NVIDIA GPU or nvidia-smi not available$(RESET)"; \
	fi
	@echo ""
	@echo "$(CYAN)🔍 System Memory:$(RESET)"
	@if command -v free >/dev/null 2>&1; then \
		free -h | grep -E "(Mem|Swap)" | sed 's/^/  /' || true; \
	fi
	@echo ""
	@echo "$(CYAN)🔍 Disk Space:$(RESET)"
	@df -h . | sed 's/^/  /' 2>/dev/null || echo "  $(YELLOW)Disk info unavailable$(RESET)"
	@echo ""
	@echo "$(CYAN)🔍 Docker Resources:$(RESET)"
	@if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then \
		echo "  $(GREEN)Docker System Info:$(RESET)"; \
		docker system df 2>/dev/null | sed 's/^/    /' || echo "    $(YELLOW)Docker stats unavailable$(RESET)"; \
	fi
	@echo "$(GREEN)✅ System check complete$(RESET)"

# Complete status dashboard
calibration-status:
	@echo "$(BOLD)$(BLUE)📊 Calibration Framework Status Dashboard$(RESET)"
	@echo "=============================================="
	@echo ""
	@overall_status=0; \
	echo "$(BOLD)Running comprehensive status check...$(RESET)"; \
	echo ""; \
	if $(MAKE) check-docker >/dev/null 2>&1; then \
		echo "$(GREEN)✅ Docker Status:      READY$(RESET)"; \
	else \
		echo "$(RED)❌ Docker Status:      NOT READY$(RESET)"; \
		overall_status=1; \
	fi; \
	if $(MAKE) check-server >/dev/null 2>&1; then \
		echo "$(GREEN)✅ Server Status:      RESPONDING$(RESET)"; \
	else \
		echo "$(RED)❌ Server Status:      NOT RESPONDING$(RESET)"; \
		overall_status=1; \
	fi; \
	if $(MAKE) check-model >/dev/null 2>&1; then \
		echo "$(GREEN)✅ Model Status:       LOADED$(RESET)"; \
	else \
		echo "$(RED)❌ Model Status:       NOT LOADED$(RESET)"; \
		overall_status=1; \
	fi; \
	if $(MAKE) check-calibration >/dev/null 2>&1; then \
		echo "$(GREEN)✅ Calibration:        READY$(RESET)"; \
	else \
		echo "$(RED)❌ Calibration:        NOT READY$(RESET)"; \
		overall_status=1; \
	fi; \
	if $(MAKE) check-system >/dev/null 2>&1; then \
		echo "$(GREEN)✅ System Resources:   AVAILABLE$(RESET)"; \
	else \
		echo "$(YELLOW)🟡 System Resources:   LIMITED$(RESET)"; \
	fi; \
	echo ""; \
	if [ $$overall_status -eq 0 ]; then \
		echo "$(BOLD)$(GREEN)🎯 STATUS: READY FOR CALIBRATION$(RESET)"; \
		echo "$(CYAN)▶️  Next step: make calibration-validate$(RESET)"; \
	else \
		echo "$(BOLD)$(RED)🚫 STATUS: NOT READY$(RESET)"; \
		echo "$(YELLOW)💡 Run individual checks for details:$(RESET)"; \
		echo "   $(CYAN)make check-docker$(RESET)"; \
		echo "   $(CYAN)make check-server$(RESET)"; \
		echo "   $(CYAN)make check-model$(RESET)"; \
		echo "   $(CYAN)make check-calibration$(RESET)"; \
	fi

# Comprehensive prerequisite check (master command)
check-prerequisites: check-docker check-server check-model check-calibration check-system
	@echo ""
	@echo "$(BOLD)$(GREEN)🎯 CALIBRATION PREREQUISITES CHECK COMPLETE$(RESET)"
	@echo "$(CYAN)▶️  All systems ready - you can now run:$(RESET)"
	@echo "   $(GREEN)make calibration-validate$(RESET)  - Run live calibration validation"
	@echo "   $(GREEN)make calibration-demo$(RESET)       - Run calibration framework demo"

# ========================================================================
# CALIBRATION COMMANDS (Re-implemented)
# ========================================================================

# Run calibration validation framework tests  
test-calibration: pre-test
	@echo "$(BOLD)$(BLUE)🎯 Running Calibration Validation Framework Tests$(RESET)"
	@echo "=================================================="
	@cd $(CALIBRATION_TEST_DIR) && $(PYTHON) test_calibration_suite.py
	@$(MAKE) post-test

# Run live calibration validation (requires LLM server)
calibration-validate:
	@echo "$(BOLD)$(BLUE)🎯 Running Live Calibration Validation$(RESET)"
	@echo "======================================="
	@echo "$(YELLOW)⚠️  This requires a running LLM server at http://localhost:8004$(RESET)"
	@echo "$(CYAN)💡 Run 'make check-prerequisites' first to verify readiness$(RESET)"
	@echo ""
	@cd $(CALIBRATION_TEST_DIR) && $(PYTHON) calibration_validator.py

# Run calibration framework demo
calibration-demo:
	@echo "$(BOLD)$(BLUE)🎯 Calibration Framework Demo$(RESET)"
	@echo "=============================="
	@echo "$(CYAN)📋 Validating reference test cases...$(RESET)"
	@cd $(CALIBRATION_TEST_DIR) && $(PYTHON) reference_test_cases.py
	@echo ""
	@echo "$(CYAN)🧪 Running calibration unit tests...$(RESET)"
	@cd $(CALIBRATION_TEST_DIR) && $(PYTHON) test_calibration_suite.py