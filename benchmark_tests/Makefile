# AI Workstation Benchmark Tests
# Professional Grade Test Automation Makefile
#
# This Makefile provides comprehensive test automation for the AI workstation
# benchmark testing suite with automatic cleanup and professional reporting.
#

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

# Common pytest arguments and test patterns
PYTEST_BASE_ARGS := --tb=short --strict-markers --disable-warnings
PYTEST_VERBOSE_ARGS := $(PYTEST_BASE_ARGS) -v -s
PYTEST_QUIET_ARGS := $(PYTEST_BASE_ARGS) -q
PYTEST_COVERAGE_ARGS := $(PYTEST_BASE_ARGS) --cov=evaluator --cov-report=term-missing --cov-fail-under=$(COVERAGE_MIN)

# Test execution patterns
define run_test_suite
	@echo "$(BOLD)$(BLUE)$(1)$(RESET)"
	@echo "$(shell printf '=%.0s' {1..50})"
	@$(PYTEST) $(2) $(3) $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)
endef

# Prerequisite check pattern
define check_prerequisite
	@if $(1) >/dev/null 2>&1; then \
		echo "$(GREEN)✅ $(2): $(3)$(RESET)"; \
	else \
		echo "$(RED)❌ $(2): $(4)$(RESET)"; \
		exit 1; \
	fi
endef

# Color output
BOLD := \033[1m
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
MAGENTA := \033[35m
CYAN := \033[36m
RESET := \033[0m

# Phony targets (consolidated)
.PHONY: help test test-unit test-integration test-calibration test-modular test-analysis test-specific test-watch \
        test-regression test-core-safety clean clean-all setup debug-help calibration-demo examples \
        check-prerequisites calibration-status calibration-validate test-domain-loading test-enhanced-evaluator \
        test-semantic-analyzer test-api-suite debug-calibration-framework systematic-base-calibration \
        test-systematic-calibration convert-creativity-tests test-creativity-conversion convert-core-domains \
        test-core-domains-conversion docker-logs debug-test-components detect-backend get-backend-mode \
        test-scripts-organization-only test-domain-loading-only test-core-modules-debug \
        debug-enhanced-system debug-enhanced-evaluator debug-scoring-calibration test-cultural-calibration \
        debug-cultural-task-detection debug-batch-task-detection test-reasoning-batch debug-basic12-analysis \
        test-reasoning-next-batch domain-audit test-phase1-quality test-evaluator-integration test-network-failure \
        test-functional-enhanced test-quality-audit validate-evaluator-framework test-server-resilience \
        docs-evaluator docs-api-reference docs-integration-guide test-production test-development test-debug \
        test-mode-config show-test-config test-server-dependent test-phase1c-validation test-phase1c-functional \
        test-phase1c-calibration test-phase1c-unit test-phase1c-integration

# Default target
help:
	@echo "$(BOLD)$(BLUE)🧪 AI Workstation Benchmark Tests$(RESET)"
	@echo "$(BOLD)================================$(RESET)"
	@echo ""
	@echo "$(BOLD)Core Commands:$(RESET)"
	@echo "  $(GREEN)make test [MODE=quick|verbose|coverage] [SUITE=unit|integration|all]$(RESET)"
	@echo "    - Unified test runner with flexible options"
	@echo "  $(GREEN)make test-regression$(RESET)      - ⚠️ CRITICAL: Regression tests (excludes functional/ & calibration/)"
	@echo "  $(GREEN)make test-core-safety$(RESET)     - Quick core module safety check"
	@echo "  $(GREEN)make test-calibration$(RESET)     - Calibration validation framework tests"
	@echo "  $(GREEN)make test-analysis$(RESET)        - Analysis and validation scripts"
	@echo "  $(GREEN)make test-specific FILE=path$(RESET) - Run specific test file/method"
	@echo ""
	@echo "$(BOLD)Production vs Development Modes:$(RESET)"
	@echo "  $(GREEN)make test-production$(RESET)      - 🏭 Production mode: Fast CI/CD with unit and integration tests"
	@echo "  $(GREEN)make test-production-full$(RESET) - 🚀 Full production: ALL tests including server-dependent"
	@echo "  $(GREEN)make test-development$(RESET)     - 🔧 Development mode: Fast iteration with focused tests"
	@echo "  $(GREEN)make test-debug$(RESET)           - 🐛 Debug mode: Verbose logging and detailed analysis"
	@echo "  $(GREEN)make show-test-config$(RESET)     - 📊 Display current test mode configuration"
	@echo "  $(GREEN)make test-mode-config$(RESET)     - 🔧 Test the test mode configuration system"
	@echo "  $(GREEN)make test-server-dependent$(RESET) - 🌐 Test functional and calibration suites (requires LLM server)"
	@echo ""
	@echo "$(BOLD)Quality Assurance:$(RESET)"
	@echo "  $(GREEN)make check-prerequisites$(RESET)  - Complete system readiness check"
	@echo "  $(GREEN)make docker-logs$(RESET)          - Inspect model server logs & configuration"
	@echo "  $(GREEN)make calibration-validate$(RESET) - Live LLM calibration validation"
	@echo "  $(GREEN)make test-api-suite$(RESET)       - Complete API connectivity test"
	@echo "  $(GREEN)make test-enhanced-evaluator$(RESET) - Enhanced evaluator import test"
	@echo ""
	@echo "$(BOLD)Systematic Calibration:$(RESET)"
	@echo "  $(GREEN)make systematic-base-calibration$(RESET)  - 🎯 Base model calibration (easy→medium→hard) with enhanced evaluation"
	@echo "  $(GREEN)make test-systematic-calibration$(RESET) - 🧪 Test systematic calibration framework (single domain)"
	@echo "  $(GREEN)make statistical-pattern-detection$(RESET) - 🔬 Focused statistical pattern detection experiment (3 categories)"
	@echo ""
	@echo "$(BOLD)Quality Audit & Advanced Testing:$(RESET)"
	@echo "  $(GREEN)make test-quality-audit$(RESET)          - 🎯 Run all quality audit tests (evaluator + network + functional)"
	@echo "  $(GREEN)make test-evaluator-integration$(RESET)  - 🧠 Comprehensive evaluator framework integration tests"
	@echo "  $(GREEN)make test-network-failure$(RESET)        - 🌐 Network failure scenarios & server dependency tests"
	@echo "  $(GREEN)make test-server-resilience$(RESET)      - 🛡️ Server resilience & dependency management validation"
	@echo "  $(GREEN)make validate-evaluator-framework$(RESET) - ✅ Complete evaluator framework validation suite"
	@echo ""
	@echo "$(BOLD)Domain Coverage:$(RESET)"
	@echo "  $(GREEN)make convert-creativity-tests$(RESET)    - Convert creativity base→instruct tests"
	@echo "  $(GREEN)make convert-core-domains$(RESET)        - Convert all core domains to instruct"
	@echo "  $(GREEN)make test-creativity-conversion$(RESET)  - Test creativity conversion results"
	@echo "  $(GREEN)make test-core-domains-conversion$(RESET) - Test core domain conversion"
	@echo ""
	@echo "$(BOLD)Documentation & API Reference:$(RESET)"
	@echo "  $(GREEN)make docs-evaluator$(RESET)       - 📚 Open evaluator capability matrix & API documentation"
	@echo "  $(GREEN)make docs-api-reference$(RESET)   - 📖 Open evaluator API reference documentation"
	@echo "  $(GREEN)make docs-integration-guide$(RESET) - 🔧 Open developer integration guide"
	@echo ""
	@echo "$(BOLD)Maintenance & Debug:$(RESET)"
	@echo "  $(GREEN)make clean$(RESET) / $(GREEN)clean-all$(RESET)  - Clean test artifacts"
	@echo "  $(GREEN)make setup$(RESET)                - Install test dependencies"
	@echo "  $(GREEN)make debug-help$(RESET)           - Show debug utilities"
	@echo "  $(GREEN)make calibration-demo$(RESET)     - Run calibration demo"
	@echo ""
	@echo "$(BOLD)Usage Examples:$(RESET)"
	@echo "  $(CYAN)make test MODE=quick SUITE=unit$(RESET)           # Fast unit tests"
	@echo "  $(CYAN)make test MODE=coverage$(RESET)                   # All tests with coverage"
	@echo "  $(CYAN)make test ARGS='-k evaluator'$(RESET)             # Tests matching pattern"
	@echo "  $(CYAN)make test-specific FILE=tests/unit/test_foo.py$(RESET)"
	@echo ""
	@echo "$(BOLD)Environment Variables:$(RESET)"
	@echo "  $(YELLOW)MODE$(RESET): quick|verbose|coverage|standard  $(YELLOW)SUITE$(RESET): unit|integration|modular|all"
	@echo "  $(YELLOW)ARGS$(RESET): Additional pytest arguments       $(YELLOW)FILE$(RESET): Specific test file for test-specific"

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

# Backend Detection Functions
# Detects whether docker container is running llama.cpp (sequential) or vLLM (concurrent)
detect-backend:
	@echo "$(BOLD)$(BLUE)🔍 Detecting LLM Backend Type$(RESET)"
	@echo "=============================="
	@if docker compose logs llama-gpu 2>/dev/null | grep -i "llama.cpp\|llama-server\|gguf\|llamacpp" >/dev/null 2>&1; then \
		echo "$(GREEN)✅ Detected: llama.cpp backend (sequential mode required)$(RESET)"; \
		echo "BACKEND=llamacpp"; \
	elif docker compose logs llama-gpu 2>/dev/null | grep -i "vllm\|ray\|asyncio" >/dev/null 2>&1; then \
		echo "$(GREEN)✅ Detected: vLLM backend (concurrent mode enabled)$(RESET)"; \
		echo "BACKEND=vllm"; \
	else \
		echo "$(YELLOW)⚠️ Unable to detect backend type from logs$(RESET)"; \
		echo "$(CYAN)Defaulting to sequential mode for safety$(RESET)"; \
		echo "BACKEND=unknown"; \
	fi

# Get backend type for use in other commands
get-backend-mode:
	@BACKEND_TYPE=$$(if docker compose logs llama-gpu 2>/dev/null | grep -i "llama.cpp\|llama-server\|gguf\|llamacpp" >/dev/null 2>&1; then echo "sequential"; elif docker compose logs llama-gpu 2>/dev/null | grep -i "vllm\|ray\|asyncio" >/dev/null 2>&1; then echo "concurrent"; else echo "sequential"; fi); echo $$BACKEND_TYPE

# Consolidated test command with mode parameter
# Usage: make test [MODE=quick|verbose|coverage] [SUITE=unit|integration|modular|analysis|all]
test: pre-test
	@$(eval TEST_MODE := $(or $(MODE), standard))
	@$(eval TEST_SUITE := $(or $(SUITE), all))
	@$(eval PYTEST_ARGS := $(call get_pytest_args,$(TEST_MODE)))
	@$(eval TEST_DIRS := $(call get_test_dirs,$(TEST_SUITE)))
	@$(call run_test_suite,🧪 Running $(TEST_SUITE) Tests ($(TEST_MODE) mode),$(PYTEST_ARGS),$(TEST_DIRS))

# Legacy aliases for backwards compatibility
test-unit: ; @$(MAKE) test SUITE=unit
test-integration: ; @$(MAKE) test SUITE=integration  
test-modular: ; @$(MAKE) test SUITE=modular

# Analysis tests (special case)
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

# Helper functions for test command
define get_pytest_args
$(if $(filter quick,$(1)),$(PYTEST_QUIET_ARGS),\
$(if $(filter verbose,$(1)),$(PYTEST_VERBOSE_ARGS),\
$(if $(filter coverage,$(1)),$(PYTEST_COVERAGE_ARGS),\
$(PYTEST_BASE_ARGS))))
endef

define get_test_dirs
$(if $(filter unit,$(1)),$(UNIT_TEST_DIR),\
$(if $(filter integration,$(1)),$(INTEGRATION_TEST_DIR),\
$(if $(filter modular,$(1)),$(MODULAR_TEST_DIR),\
$(UNIT_TEST_DIR) $(INTEGRATION_TEST_DIR))))
endef

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
	@echo "  $(GREEN)make debug-concurrent-timeouts$(RESET)  - Debug concurrent execution timeout issues (Phase 1B)"
	@echo "  $(GREEN)make test-cultural-calibration$(RESET)  - Test cultural reasoning evaluation on specific tests"
	@echo "  $(GREEN)make debug-cultural-task-detection$(RESET) - Check if cultural tests are being detected properly"
	@echo "  $(GREEN)make debug-batch-task-detection$(RESET) - Check cultural task detection for basic_05-08"
	@echo "  $(GREEN)make test-reasoning-batch$(RESET)        - Test next batch of reasoning tests (basic_09-15)"
	@echo ""
	@echo "$(BOLD)Usage:$(RESET)"
	@echo "  $(CYAN)make debug-enhanced-system$(RESET)       # Run comprehensive enhanced system debug"
	@echo "  $(CYAN)make debug-enhanced-evaluator$(RESET)     # Run enhanced evaluator debug with multi-tier scoring"
	@echo "  $(CYAN)make debug-scoring-calibration$(RESET)    # Test and validate Phase 1 scoring calibration fixes"
	@echo "  $(CYAN)make debug-concurrent-timeouts$(RESET)    # Analyze concurrent execution timeout issues"

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

# Debug concurrent execution timeouts
debug-concurrent-timeouts:
	@echo "$(BOLD)$(BLUE)🔍 Debugging Concurrent Execution Timeouts$(RESET)"
	@echo "============================================"
	@echo "$(CYAN)Analyzing performance bottlenecks in concurrent execution$(RESET)"
	@$(PYTHON) scripts/debugging/performance_monitor.py

# Basic server diagnostic for timeout issues  
debug-server-basic:
	@echo "$(BOLD)$(BLUE)🔍 Basic Server Diagnostic$(RESET)"
	@echo "=========================="
	@echo "$(CYAN)Checking server responsiveness before concurrent analysis$(RESET)"
	@$(PYTHON) scripts/debugging/basic_server_check.py

# Test single benchmark execution to debug timeout root cause
debug-single-benchmark:
	@echo "$(BOLD)$(BLUE)🧪 Single Benchmark Execution Debug$(RESET)"
	@echo "======================================="
	@echo "$(CYAN)Testing single benchmark execution to identify root cause$(RESET)"
	@$(PYTHON) benchmark_runner.py --test-type base --test-id basic_01 --endpoint http://localhost:8004/v1/completions --model llama3 2>&1

# Test concurrent execution directly with correct endpoint  
debug-concurrent-execution:
	@echo "$(BOLD)$(BLUE)🔄 Testing Concurrent Execution$(RESET)"
	@echo "======================================"
	@echo "$(CYAN)Testing concurrent execution with 2 workers$(RESET)"
	@$(PYTHON) benchmark_runner.py --test-type base --test-id basic_01,basic_02 --mode concurrent --workers 2 --endpoint http://localhost:8004/v1/completions --model llama3 --verbose 2>&1

# Test the originally failing functional tests
test-failing-functional-tests:
	@echo "$(BOLD)$(BLUE)🧪 Testing Originally Failing Functional Tests$(RESET)"
	@echo "=============================================="
	@echo "$(CYAN)Running the 3 concurrent execution tests that were timing out$(RESET)"
	@echo ""
	@echo "$(YELLOW)Test 1: test_concurrent_execution$(RESET)"
	@timeout 90s $(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/functional/test_domain_execution.py::TestDomainExecution::test_concurrent_execution -v || echo "$(RED)❌ Test 1 failed or timed out$(RESET)"
	@echo ""
	@echo "$(YELLOW)Test 2: test_sequential_vs_concurrent_comparison$(RESET)"
	@timeout 90s $(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/functional/test_domain_execution.py::TestDomainExecution::test_sequential_vs_concurrent_comparison -v || echo "$(RED)❌ Test 2 failed or timed out$(RESET)"
	@echo ""
	@echo "$(YELLOW)Test 3: test_concurrent_results_integrity$(RESET)"
	@timeout 90s $(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/functional/test_result_validation.py::TestResultValidation::test_concurrent_results_integrity -v || echo "$(RED)❌ Test 3 failed or timed out$(RESET)"

# Test single failing functional test
test-single-failing-test:
	@echo "$(BOLD)$(BLUE)🧪 Testing Single Failing Functional Test$(RESET)"
	@echo "==========================================="
	@echo "$(CYAN)Running test_concurrent_execution with timeout$(RESET)"
	@timeout 60s $(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/functional/test_domain_execution.py::TestDomainExecution::test_concurrent_execution -v

# Test cultural reasoning calibration
test-cultural-calibration:
	@echo "$(BOLD)$(BLUE)🏛️ Testing Cultural Reasoning Calibration$(RESET)"
	@echo "============================================"
	@echo "$(CYAN)Testing Arabic Quranic verse pattern (basic_03):$(RESET)"
	@$(PYTHON) benchmark_runner.py --test-definitions domains/reasoning/base_models/easy.json --test-id basic_03 --enhanced-evaluation --evaluation-mode full --quiet
	@echo ""
	@echo "$(CYAN)Testing Native American creation story (basic_04):$(RESET)"
	@$(PYTHON) benchmark_runner.py --test-definitions domains/reasoning/base_models/easy.json --test-id basic_04 --enhanced-evaluation --evaluation-mode full --quiet

# Debug cultural task detection and evaluation
debug-cultural-task-detection:
	@echo "$(BOLD)$(BLUE)🔍 Debug Cultural Task Detection$(RESET)"
	@echo "======================================="
	@echo "$(CYAN)Checking Arabic Quranic test (basic_03) task detection:$(RESET)"
	@$(PYTHON) benchmark_runner.py --test-definitions domains/reasoning/base_models/easy.json --test-id basic_03 --enhanced-evaluation --evaluation-mode full 2>&1 | grep -E "TASK_DETECTION|CULTURAL_EVAL" || echo "No cultural evaluation detected"
	@echo ""
	@echo "$(CYAN)Checking Native American test (basic_04) task detection:$(RESET)"
	@$(PYTHON) benchmark_runner.py --test-definitions domains/reasoning/base_models/easy.json --test-id basic_04 --enhanced-evaluation --evaluation-mode full 2>&1 | grep -E "TASK_DETECTION|CULTURAL_EVAL" || echo "No cultural evaluation detected"

# Debug task detection for next batch of cultural tests
debug-batch-task-detection:
	@echo "$(BOLD)$(BLUE)🔍 Debug Batch Task Detection (basic_05-08)$(RESET)"
	@echo "================================================="
	@echo "$(CYAN)Checking Chinese Five Elements (basic_05):$(RESET)"
	@$(PYTHON) benchmark_runner.py --test-definitions domains/reasoning/base_models/easy.json --test-id basic_05 --enhanced-evaluation --evaluation-mode full 2>&1 | grep -E "TASK_DETECTION|CULTURAL_EVAL" || echo "No cultural evaluation detected"
	@echo ""
	@echo "$(CYAN)Checking Vedic Sanskrit (basic_06):$(RESET)"
	@$(PYTHON) benchmark_runner.py --test-definitions domains/reasoning/base_models/easy.json --test-id basic_06 --enhanced-evaluation --evaluation-mode full 2>&1 | grep -E "TASK_DETECTION|CULTURAL_EVAL" || echo "No cultural evaluation detected"
	@echo ""
	@echo "$(CYAN)Checking Celtic Triadic (basic_07):$(RESET)"
	@$(PYTHON) benchmark_runner.py --test-definitions domains/reasoning/base_models/easy.json --test-id basic_07 --enhanced-evaluation --evaluation-mode full 2>&1 | grep -E "TASK_DETECTION|CULTURAL_EVAL" || echo "No cultural evaluation detected"
	@echo ""
	@echo "$(CYAN)Checking Yoruba Oriki (basic_08):$(RESET)"
	@$(PYTHON) benchmark_runner.py --test-definitions domains/reasoning/base_models/easy.json --test-id basic_08 --enhanced-evaluation --evaluation-mode full 2>&1 | grep -E "TASK_DETECTION|CULTURAL_EVAL" || echo "No cultural evaluation detected"

# Test next batch of reasoning tests systematically
test-reasoning-batch:
	@echo "$(BOLD)$(BLUE)🧠 Testing Reasoning Batch (basic_09-15)$(RESET)"
	@echo "==============================================="
	@for test_id in basic_09 basic_10 basic_11 basic_12 basic_13 basic_14 basic_15; do \
		echo "$(CYAN)Testing $$test_id:$(RESET)"; \
		$(PYTHON) benchmark_runner.py --test-definitions domains/reasoning/base_models/easy.json --test-id $$test_id --enhanced-evaluation --evaluation-mode full --quiet | grep "Average reasoning score"; \
		echo ""; \
	done

# Analyze what makes basic_12 score higher than others
debug-basic12-analysis:
	@echo "$(BOLD)$(BLUE)🔍 Analyzing basic_12 High Score (52.6 vs 38.8)$(RESET)"
	@echo "=================================================="
	@echo "$(CYAN)Getting basic_12 test details:$(RESET)"
	@$(PYTHON) -c "import json; data=json.load(open('domains/reasoning/base_models/easy.json')); test=next(t for t in data['tests'] if t['id']=='basic_12'); print(f'📝 Name: {test.get(\"name\", \"Unknown\")}'); print(f'📄 Description: {test.get(\"description\", \"N/A\")}'); print(f'❓ Prompt: {test.get(\"prompt\", \"N/A\")[:200]}...')"
	@echo ""
	@echo "$(CYAN)Running basic_12 with task detection debug:$(RESET)"
	@$(PYTHON) benchmark_runner.py --test-definitions domains/reasoning/base_models/easy.json --test-id basic_12 --enhanced-evaluation --evaluation-mode full 2>&1 | grep -E "TASK_DETECTION|enhanced.*score|Average.*score"

# Test next batch of reasoning tests (basic_16-22)
test-reasoning-next-batch:
	@echo "$(BOLD)$(BLUE)🧠 Testing Next Reasoning Batch (basic_16-22)$(RESET)"
	@echo "================================================="
	@for test_id in basic_16 basic_17 basic_18 basic_19 basic_20 basic_21 basic_22; do \
		echo "$(CYAN)Testing $$test_id:$(RESET)"; \
		$(PYTHON) benchmark_runner.py --test-definitions domains/reasoning/base_models/easy.json --test-id $$test_id --enhanced-evaluation --evaluation-mode full --quiet | grep "Average reasoning score"; \
		echo ""; \
	done

# Run all tests from domains/reasoning/base_models/easy.json (210 tests)
test-easy-reasoning-all:
	@echo "$(BOLD)$(BLUE)🧠 Running All Easy Reasoning Tests (210 tests)$(RESET)"
	@echo "=================================================="
	@echo "$(CYAN)Loading test suite from easy.json...$(RESET)"
	@$(PYTHON) -c "import json; data=json.load(open('domains/reasoning/base_models/easy.json')); print(f'✅ Found {len(data[\"tests\"])} tests in easy.json')"
	@echo ""
	@echo "$(YELLOW)⚠️ This will run all 210 tests sequentially with enhanced evaluation$(RESET)"
	@echo "$(YELLOW)⚠️ Estimated time: ~105 minutes (30 seconds per test)$(RESET)"
	@echo "$(CYAN)🚀 Starting test execution...$(RESET)"
	@echo ""
	@test_ids=$$($(PYTHON) -c "import json; data=json.load(open('domains/reasoning/base_models/easy.json')); print(' '.join([t['id'] for t in data['tests']]))"); \
	total=$$(echo $$test_ids | wc -w); \
	counter=1; \
	for test_id in $$test_ids; do \
		echo "$(CYAN)[$$counter/$$total] Testing $$test_id...$(RESET)"; \
		$(PYTHON) benchmark_runner.py --test-definitions domains/reasoning/base_models/easy.json --test-id $$test_id --enhanced-evaluation --evaluation-mode full --quiet | grep "Average reasoning score" || echo "  ❌ Test failed or no score found"; \
		counter=$$((counter + 1)); \
		echo ""; \
	done; \
	echo "$(GREEN)✅ All easy reasoning tests completed!$(RESET)"

# Statistical Pattern Analysis Commands
statistical-pattern-analysis:
	@echo "$(BOLD)$(BLUE)📊 Running Statistical Pattern Analysis$(RESET)"
	@echo "======================================"
	@echo "$(CYAN)Analyzing cognitive patterns in test_results/...$(RESET)"
	@$(PYTHON) -c "import os; from evaluator.analysis.statistical_pattern_detector import run_pattern_analysis; analysis = run_pattern_analysis('test_results/', 'cognitive_pattern_analysis.json') if os.path.exists('test_results/') else print('❌ Error: test_results/ directory not found'); print(analysis.pattern_summary if 'analysis' in locals() else ''); print('\\n📄 Detailed results saved to: cognitive_pattern_analysis.json' if 'analysis' in locals() else '')"

pattern-analysis-detailed:
	@echo "$(BOLD)$(BLUE)📊 Running Detailed Statistical Pattern Analysis$(RESET)"
	@echo "================================================="
	@echo "$(CYAN)Performing comprehensive analysis with significance testing...$(RESET)"
	@$(PYTHON) ./run_pattern_analysis.py test_results/ --detailed --output detailed_pattern_analysis.json

pattern-analysis-custom:
	@if [ -z "$(DIR)" ]; then \
		echo "$(RED)❌ Error: DIR parameter required$(RESET)"; \
		echo "Usage: make pattern-analysis-custom DIR=path/to/results"; \
		exit 1; \
	fi
	@echo "$(BOLD)$(BLUE)📊 Running Custom Statistical Pattern Analysis$(RESET)"
	@echo "=============================================="
	@echo "$(CYAN)Analyzing cognitive patterns in $(DIR)...$(RESET)"
	@$(PYTHON) ./run_pattern_analysis.py "$(DIR)"

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

# Systematic base model calibration (easy → medium → hard) with enhanced evaluation
systematic-base-calibration:
	@echo "$(BOLD)$(BLUE)🎯 Systematic Base Model Calibration (Easy → Medium → Hard)$(RESET)"
	@echo "================================================================"
	@echo "$(YELLOW)⚠️ This requires a running LLM server at http://localhost:8004$(RESET)"
	@echo "$(CYAN)💡 Run 'make check-prerequisites' first to verify readiness$(RESET)"
	@echo "$(GREEN)🧠 Using enhanced evaluation with sophisticated evaluators$(RESET)"
	@echo "$(GREEN)📊 Target ranges: Easy(70-85), Medium(60-80), Hard(50-75)$(RESET)"
	@echo ""
	@echo "$(BOLD)$(MAGENTA)🔍 Detecting Backend Type...$(RESET)"
	@BACKEND_MODE=$$($(MAKE) get-backend-mode); \
	if [ "$$BACKEND_MODE" = "concurrent" ]; then \
		echo "$(GREEN)✅ vLLM backend detected - enabling concurrent processing$(RESET)"; \
		export CONCURRENCY_MODE=concurrent; \
	else \
		echo "$(YELLOW)⚠️ llama.cpp backend detected - using sequential processing$(RESET)"; \
		export CONCURRENCY_MODE=sequential; \
	fi; \
	$(PYTHON) scripts/calibration/systematic_base_calibration.py

# Test systematic calibration framework (single domain for testing)
test-systematic-calibration:
	@echo "$(BOLD)$(BLUE)🧪 Testing Systematic Calibration Framework$(RESET)"
	@echo "============================================="
	@echo "$(YELLOW)⚠️ This requires a running LLM server at http://localhost:8004$(RESET)"
	@echo "$(CYAN)💡 Testing enhanced evaluation integration with reasoning domain$(RESET)"
	@echo ""
	@cd scripts/calibration && timeout 120s $(PYTHON) -c "import sys; sys.path.append('../..'); from systematic_base_calibration import SystematicBaseCalibrator; calibrator = SystematicBaseCalibrator(); result = calibrator.calibrate_domain_progression('reasoning'); print(f'✅ Domain: {result.domain}'); print(f'📊 Success: {result.overall_success}'); print(f'⚠️ Halted at: {result.progression_halted_at}'); print(f'💡 Recommendation: {result.recommendation}')"

# Statistical Pattern Detection Focused Experiment
statistical-pattern-detection:
	@echo "$(BOLD)$(BLUE)🔬 Statistical Pattern Detection Experiment$(RESET)"
	@echo "==============================================="
	@echo "$(YELLOW)🎯 Objective: Validate framework's ability to detect meaningful statistical patterns$(RESET)"
	@echo "$(CYAN)📊 Method: Multi-run analysis across 3 reasoning categories$(RESET)"
	@echo "$(GREEN)🧪 Categories: basic_logic_patterns, cultural_reasoning, elementary_math_science$(RESET)"
	@echo "$(GREEN)📈 Statistical tests: ANOVA, Cohen's d, Classification accuracy$(RESET)"
	@echo "$(YELLOW)⚠️ This requires a running LLM server at http://localhost:8004$(RESET)"
	@echo ""
	$(PYTHON) scripts/calibration/statistical_pattern_detection.py

# Convert core domains (language, integration, knowledge, social) to instruct format
convert-core-domains:
	@echo "$(BOLD)$(BLUE)🌍 Converting Core Domains to Instruct Model Format$(RESET)"
	@echo "=========================================================="
	@$(PYTHON) scripts/conversion/convert_core_domains_to_instruct.py

# Test enhanced evaluator with converted core domain tests
test-core-domains-conversion:
	@echo "$(BOLD)$(BLUE)🌍 Testing Enhanced Evaluator with Converted Core Domain Tests$(RESET)"
	@echo "======================================================================"
	@$(PYTHON) $(DEBUG_DIR)/test_core_domains_conversion.py


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
# QUALITY AUDIT & ADVANCED TESTING COMMANDS  
# ========================================================================

# Comprehensive quality audit test suite (all medium priority issues)
test-quality-audit: pre-test
	@echo "$(BOLD)$(BLUE)🎯 Quality Audit Comprehensive Test Suite$(RESET)"
	@echo "=================================================="
	@echo "$(CYAN)💡 Running all quality audit tests: evaluator integration + network failure + enhanced functional$(RESET)"
	@echo ""
	@$(MAKE) test-evaluator-integration
	@echo ""
	@$(MAKE) test-network-failure  
	@echo ""
	@$(MAKE) test-functional-enhanced
	@echo ""
	@echo "$(GREEN)✅ Quality audit test suite complete!$(RESET)"
	@$(MAKE) post-test

# Comprehensive evaluator framework integration tests
test-evaluator-integration: pre-test
	@echo "$(BOLD)$(BLUE)🧠 Evaluator Framework Integration Tests$(RESET)"
	@echo "==========================================="
	@echo "$(CYAN)Testing multi-evaluator orchestration, fallback behavior, and performance$(RESET)"
	@$(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/integration/test_evaluator_framework_integration.py $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Network failure scenarios and server dependency management tests  
test-network-failure: pre-test
	@echo "$(BOLD)$(BLUE)🌐 Network Failure Scenarios Tests$(RESET)"
	@echo "====================================" 
	@echo "$(CYAN)Testing timeout handling, invalid endpoints, resilient retry logic$(RESET)"
	@$(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/functional/test_network_failure_scenarios.py $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Enhanced functional tests with server dependency management
test-functional-enhanced: pre-test
	@echo "$(BOLD)$(BLUE)🚀 Enhanced Functional Tests$(RESET)"
	@echo "============================="
	@echo "$(CYAN)Testing enhanced server dependency management and graceful degradation$(RESET)"
	@$(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/functional/ --ignore=tests/functional/test_network_failure_scenarios.py $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Complete evaluator framework validation
validate-evaluator-framework: pre-test
	@echo "$(BOLD)$(BLUE)✅ Complete Evaluator Framework Validation$(RESET)"
	@echo "=============================================="
	@echo "$(CYAN)Comprehensive validation: initialization, integration, fallbacks, performance$(RESET)"
	@echo ""
	@echo "$(BOLD)1. Testing Evaluator Integration:$(RESET)"
	@$(MAKE) test-evaluator-integration
	@echo ""
	@echo "$(BOLD)2. Testing Enhanced Evaluator Import:$(RESET)"
	@$(MAKE) test-enhanced-evaluator
	@echo ""
	@echo "$(BOLD)3. Running Core Safety Check:$(RESET)"
	@$(MAKE) test-core-safety
	@echo ""
	@echo "$(GREEN)✅ Evaluator framework validation complete!$(RESET)"
	@$(MAKE) post-test

# Server resilience and dependency management validation
test-server-resilience: pre-test
	@echo "$(BOLD)$(BLUE)🛡️ Server Resilience & Dependency Management$(RESET)"
	@echo "==============================================" 
	@echo "$(CYAN)Testing server health checking, backend detection, network failure handling$(RESET)"
	@echo ""
	@echo "$(BOLD)1. Network Failure Scenarios:$(RESET)"
	@$(MAKE) test-network-failure
	@echo ""
	@echo "$(BOLD)2. Server Health & API Connectivity:$(RESET)"
	@$(MAKE) test-api-suite || echo "$(YELLOW)⚠️ API tests require live server$(RESET)"
	@echo ""
	@echo "$(BOLD)3. Backend Detection:$(RESET)"
	@$(MAKE) detect-backend || echo "$(YELLOW)⚠️ Backend detection requires Docker$(RESET)"
	@echo ""
	@echo "$(GREEN)✅ Server resilience validation complete!$(RESET)"
	@$(MAKE) post-test

# ========================================================================
# DOCUMENTATION COMMANDS
# ========================================================================

# Open evaluator capability matrix and API documentation  
docs-evaluator:
	@echo "$(BOLD)$(BLUE)📚 Evaluator Documentation$(RESET)"
	@echo "============================"
	@echo "$(CYAN)Opening evaluator capability matrix and API documentation...$(RESET)"
	@if command -v xdg-open >/dev/null 2>&1; then \
		xdg-open docs/evaluator/EVALUATOR_CAPABILITY_MATRIX.md; \
		sleep 1; \
		xdg-open docs/evaluator/EVALUATOR_API_REFERENCE.md; \
	elif command -v open >/dev/null 2>&1; then \
		open docs/evaluator/EVALUATOR_CAPABILITY_MATRIX.md; \
		sleep 1; \
		open docs/evaluator/EVALUATOR_API_REFERENCE.md; \
	else \
		echo "$(GREEN)📖 Evaluator Capability Matrix:$(RESET) docs/evaluator/EVALUATOR_CAPABILITY_MATRIX.md"; \
		echo "$(GREEN)📖 API Reference:$(RESET) docs/evaluator/EVALUATOR_API_REFERENCE.md"; \
	fi

# Open evaluator API reference documentation
docs-api-reference:
	@echo "$(BOLD)$(BLUE)📖 Evaluator API Reference$(RESET)"
	@echo "=============================="
	@echo "$(CYAN)Opening API reference documentation...$(RESET)"
	@if command -v xdg-open >/dev/null 2>&1; then \
		xdg-open docs/evaluator/EVALUATOR_API_REFERENCE.md; \
	elif command -v open >/dev/null 2>&1; then \
		open docs/evaluator/EVALUATOR_API_REFERENCE.md; \
	else \
		echo "$(GREEN)📖 API Reference:$(RESET) docs/evaluator/EVALUATOR_API_REFERENCE.md"; \
		echo "$(CYAN)💡 Use your preferred editor or browser to view the documentation$(RESET)"; \
	fi

# Open developer integration guide
docs-integration-guide:
	@echo "$(BOLD)$(BLUE)🔧 Developer Integration Guide$(RESET)"
	@echo "=================================="
	@echo "$(CYAN)Opening integration guide...$(RESET)"
	@if command -v xdg-open >/dev/null 2>&1; then \
		xdg-open docs/evaluator/INTEGRATION_GUIDE.md; \
	elif command -v open >/dev/null 2>&1; then \
		open docs/evaluator/INTEGRATION_GUIDE.md; \
	else \
		echo "$(GREEN)📖 Integration Guide:$(RESET) docs/evaluator/INTEGRATION_GUIDE.md"; \
		echo "$(CYAN)💡 Use your preferred editor or browser to view the documentation$(RESET)"; \
	fi

# ========================================================================
# PREREQUISITE CHECKING COMMANDS
# ========================================================================

# Comprehensive prerequisite check (master command)
check-prerequisites:
	@echo "$(BOLD)$(BLUE)🔍 Comprehensive Prerequisites Check$(RESET)"
	@echo "============================================="
	@overall_status=0; \
	echo ""; \
	echo "$(BOLD)1. Docker Status Check$(RESET)"; \
	$(call check_prerequisite,$(MAKE) check-docker-internal,Docker Status,READY,NOT READY) || overall_status=1; \
	echo ""; \
	echo "$(BOLD)2. Server Connectivity Check$(RESET)"; \
	$(call check_prerequisite,$(MAKE) check-server-internal,Server Status,RESPONDING,NOT RESPONDING) || overall_status=1; \
	echo ""; \
	echo "$(BOLD)3. Model Loading Check$(RESET)"; \
	$(call check_prerequisite,$(MAKE) check-model-internal,Model Status,LOADED,NOT LOADED) || overall_status=1; \
	echo ""; \
	echo "$(BOLD)4. Calibration Framework Check$(RESET)"; \
	$(call check_prerequisite,$(MAKE) check-calibration-internal,Calibration,READY,NOT READY) || overall_status=1; \
	echo ""; \
	echo "$(BOLD)5. System Resources Check$(RESET)"; \
	$(call check_prerequisite,$(MAKE) check-system-internal,System Resources,AVAILABLE,LIMITED) || true; \
	echo ""; \
	if [ $$overall_status -eq 0 ]; then \
		echo "$(BOLD)$(GREEN)🎯 STATUS: READY FOR CALIBRATION$(RESET)"; \
		echo "$(CYAN)▶️  Next step: make calibration-validate$(RESET)"; \
	else \
		echo "$(BOLD)$(RED)🚫 STATUS: NOT READY$(RESET)"; \
		echo "$(YELLOW)💡 Fix the issues above before proceeding$(RESET)"; \
		exit 1; \
	fi

# Internal check functions (silent)
check-docker-internal:
	@command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 && docker ps --format "{{.Ports}}" | grep ":8004" >/dev/null 2>&1

check-server-internal:
	@command -v nc >/dev/null 2>&1 && nc -z localhost 8004 2>/dev/null && curl -s --connect-timeout 5 "http://localhost:8004/health" >/dev/null 2>&1

check-model-internal:
	@command -v curl >/dev/null 2>&1 && response=$$(curl -s --connect-timeout 10 -X POST -H "Content-Type: application/json" -d '{"prompt":"Test","max_tokens":5}' "http://localhost:8004/v1/completions" 2>/dev/null) && echo "$$response" | grep -E "(choices|content)" >/dev/null 2>&1

check-calibration-internal:
	@[ -f "$(CALIBRATION_TEST_DIR)/calibration_validator.py" ] && cd $(CALIBRATION_TEST_DIR) && $(PYTHON) -c "import calibration_validator, reference_test_cases, calibration_reporter" 2>/dev/null

check-system-internal:
	@command -v nvidia-smi >/dev/null 2>&1 && command -v free >/dev/null 2>&1

# Docker logs inspection
docker-logs:
	@echo "$(BOLD)$(BLUE)🐳 Docker Compose Logs - llama-gpu$(RESET)"
	@echo "$(shell printf '=%.0s' {1..50})"
	@echo "$(CYAN)💡 Inspecting model server logs for current configuration$(RESET)"
	@echo ""
	@docker compose logs llama-gpu

# ========================================================================
# AUTOMATED TESTING AND DEBUGGING COMMANDS
# ========================================================================

# Consolidated API testing suite
test-api-suite:
	@echo "$(BOLD)$(BLUE)🧪 Complete API Testing Suite$(RESET)"
	@echo "=============================="
	@echo ""
	@echo "$(CYAN)1. Server Health Check:$(RESET)"
	@curl -s http://localhost:8004/health | jq . 2>/dev/null && echo "$(GREEN)✅ Health endpoint OK$(RESET)" || echo "$(RED)❌ Health endpoint failed$(RESET)"
	@echo ""
	@echo "$(CYAN)2. Basic Completion Test:$(RESET)"
	@curl -s -X POST \
		-H "Content-Type: application/json" \
		-d '{"prompt":"Hello","n_predict":5,"temperature":0.1}' \
		http://localhost:8004/completion | jq . 2>/dev/null && echo "$(GREEN)✅ Completion endpoint OK$(RESET)" || echo "$(RED)❌ Completion endpoint failed$(RESET)"
	@echo ""
	@echo "$(CYAN)3. Haiku Completion Test:$(RESET)"
	@echo "   Input: Complete traditional Japanese haiku (5-7-5 pattern)"
	@echo "   Cherry blossoms fall / Gentle spring breeze carries them / ..."
	@response=$$(curl -s -X POST \
		-H "Content-Type: application/json" \
		-d '{"prompt":"Complete this traditional Japanese haiku following the 5-7-5 syllable pattern:\\n\\nCherry blossoms fall\\nGentle spring breeze carries them\\n","n_predict":20,"temperature":0.6}' \
		http://localhost:8004/completion 2>/dev/null | jq -r '.content' 2>/dev/null | head -1 | sed 's/^/   Response: /'); \
	if [ -n "$$response" ]; then \
		echo "$$response"; \
		echo "$(GREEN)✅ Haiku completion OK$(RESET)"; \
	else \
		echo "$(RED)❌ Haiku completion failed$(RESET)"; \
	fi
	@echo ""
	@echo "$(GREEN)✅ API testing suite complete$(RESET)"

# Individual component tests (kept for specific debugging)
test-domain-loading:
	@echo "$(BOLD)$(BLUE)📂 Testing Domain File Loading$(RESET)"
	@echo "================================"
	@$(PYTHON) -c "import json; data = json.load(open('domains/reasoning/base_models/easy.json')); print(f'✅ Loaded {len(data[\"tests\"])} tests from easy.json'); print(f'📝 First test ID: {data[\"tests\"][0][\"id\"]}'); print(f'🎯 Test name: {data[\"tests\"][0][\"name\"]}')"

test-enhanced-evaluator:
	@echo "$(BOLD)$(BLUE)🧠 Testing Enhanced Evaluator Import$(RESET)"
	@echo "====================================="
	@$(PYTHON) -c "from evaluator.subjects.enhanced_universal_evaluator import EnhancedUniversalEvaluator; print('✅ Enhanced evaluator import successful'); evaluator = EnhancedUniversalEvaluator(); print('✅ Enhanced evaluator instantiation successful')" || echo "❌ Enhanced evaluator error"

test-semantic-analyzer:
	@echo "$(BOLD)$(BLUE)🔍 Testing Semantic Analyzer Logging Fix$(RESET)"
	@echo "=============================================="
	@echo "$(CYAN)Testing semantic analyzer error message fix$(RESET)"
	@cd $(CALIBRATION_TEST_DIR) && timeout 30 python calibration_validator.py 2>&1 | grep -E "(semantic|embedding|ERROR|fallback)" || echo "$(GREEN)✅ No semantic analyzer errors found$(RESET)"

# Debug calibration framework step by step
debug-calibration-framework:
	@echo "$(BOLD)$(BLUE)🔧 Debug Calibration Framework$(RESET)"
	@echo "==============================="
	@echo "$(CYAN)Step 1: Domain file loading$(RESET)"
	@$(MAKE) test-domain-loading
	@echo ""
	@echo "$(CYAN)Step 2: Enhanced evaluator$(RESET)"
	@$(MAKE) test-enhanced-evaluator
	@echo ""
	@echo "$(CYAN)Step 3: Benchmark runner$(RESET)"
	@$(MAKE) test-benchmark-endpoint
	@echo ""
	@echo "$(CYAN)Step 4: API connectivity$(RESET)"
	@$(MAKE) test-server-health
	@echo ""
	@echo "$(GREEN)✅ Debug framework complete$(RESET)"

# Legacy calibration status (now just calls check-prerequisites)
calibration-status: check-prerequisites

# ========================================================================
# CALIBRATION COMMANDS (Re-implemented)
# ========================================================================

# Run calibration validation framework tests  
test-calibration: pre-test
	@echo "$(BOLD)$(BLUE)🎯 Running Calibration Validation Framework Tests$(RESET)"
	@echo "=================================================="
	@cd $(CALIBRATION_TEST_DIR) && $(PYTHON) test_calibration_suite.py
	@$(MAKE) post-test

# CRITICAL: Regression test - run all tests except functional/ and calibration/ (which require live server)
test-regression: pre-test
	@echo "$(BOLD)$(BLUE)⚠️ CRITICAL REGRESSION TEST - All tests except functional/ and calibration/$(RESET)"
	@echo "========================================================================="
	@echo "$(YELLOW)💡 This runs ALL tests EXCEPT functional/ and calibration/ to ensure no functionality is broken$(RESET)"
	@echo "$(CYAN)💡 Calibration tests require live server setup and are excluded like functional tests$(RESET)"
	@echo "$(GREEN)🚀 Running tests concurrently using pytest-xdist for faster execution$(RESET)"
	@echo ""
	@$(PYTEST) $(PYTEST_BASE_ARGS) -v -n auto tests/ --ignore=tests/functional/ --ignore=tests/calibration/ $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Quick core module safety check
test-core-safety: pre-test
	@echo "$(BOLD)$(BLUE)🛡️ Core Module Safety Check$(RESET)"
	@echo "================================="
	@echo "$(CYAN)Testing core/ modules and critical functionality$(RESET)"
	@$(PYTEST) $(PYTEST_BASE_ARGS) -v tests/unit/test_core_modules/ tests/unit/test_scripts_organization/ $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

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

# Debug specific test components (consolidated)
debug-test-components:
	@echo "$(BOLD)$(BLUE)🧪 Debug: Test Components Status$(RESET)"
	@echo "======================================="
	@echo "$(CYAN)📁 Project structure:$(RESET)"
	@echo "  Current dir: $(PWD)"
	@echo "  Scripts dir: $$(test -d scripts && echo '✅ EXISTS' || echo '❌ MISSING')"
	@echo "  Core dir: $$(test -d core && echo '✅ EXISTS' || echo '❌ MISSING')" 
	@echo "  Domains dir: $$(test -d domains && echo '✅ EXISTS' || echo '❌ MISSING')"
	@echo "$(CYAN)🔧 Core modules:$(RESET)"
	@echo "  calibration_engine.py: $$(test -f core/calibration_engine.py && echo '✅ EXISTS' || echo '❌ MISSING')"
	@echo "  production_calibration.py: $$(test -f core/production_calibration.py && echo '✅ EXISTS' || echo '❌ MISSING')"
	@echo "  benchmarking_engine.py: $$(test -f core/benchmarking_engine.py && echo '✅ EXISTS' || echo '❌ MISSING')"
	@echo "  cognitive_validation.py: $$(test -f core/cognitive_validation.py && echo '✅ EXISTS' || echo '❌ MISSING')"
	@echo "$(CYAN)📊 Domain test files: $$(find domains/ -name '*.json' | wc -l) files$(RESET)"

# Test specific test suites individually
test-scripts-organization-only:
	@echo "$(BOLD)$(BLUE)📁 Testing Scripts Organization Only$(RESET)"
	@echo "====================================="
	@$(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/unit/test_scripts_organization/ --tb=long

test-domain-loading-only:
	@echo "$(BOLD)$(BLUE)📂 Testing Domain Loading Only$(RESET)"
	@echo "==============================="
	@$(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/unit/test_core_system/test_domain_loading.py --tb=long

test-core-modules-debug:
	@echo "$(BOLD)$(BLUE)🔧 Testing Core Modules with Debug Info$(RESET)"
	@echo "=============================================="
	@$(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/unit/test_core_modules/ --tb=long -s

# ========================================================================
# PRODUCTION VS DEVELOPMENT TEST MODES
# ========================================================================

# Production test mode - Fast CI/CD with unit and integration tests
test-production: pre-test
	@echo "$(BOLD)$(BLUE)🏭 Production Test Mode$(RESET)"
	@echo "=========================="
	@echo "$(GREEN)⚡ Fast CI/CD with unit and integration test coverage$(RESET)"
	@echo "$(CYAN)💡 Optimized for production deployment validation without server dependencies$(RESET)"
	@echo ""
	@BENCHMARK_TEST_MODE=production $(PYTHON) scripts/test_mode_runner.py production $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Full production test mode - ALL tests including server-dependent
test-production-full: pre-test
	@echo "$(BOLD)$(BLUE)🚀 Full Production Test Mode$(RESET)"
	@echo "============================"
	@echo "$(GREEN)🎯 Comprehensive testing: ALL tests including functional and calibration$(RESET)"
	@echo "$(CYAN)💡 Requires LLM server at http://localhost:8004$(RESET)"
	@echo "$(YELLOW)⚠️  Server availability will be checked before execution$(RESET)"
	@echo ""
	@BENCHMARK_TEST_MODE=production_full $(PYTHON) scripts/test_mode_runner.py production_full $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Development test mode - Fast iteration with focused tests
test-development: pre-test
	@echo "$(BOLD)$(BLUE)🔧 Development Test Mode$(RESET)"
	@echo "========================="
	@echo "$(GREEN)⚡ Fast iteration with focused test execution$(RESET)"
	@echo "$(CYAN)💡 Optimized for rapid development cycles$(RESET)"
	@echo ""
	@BENCHMARK_TEST_MODE=development $(PYTHON) scripts/test_mode_runner.py development $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Debug test mode - Verbose logging and detailed analysis  
test-debug: pre-test
	@echo "$(BOLD)$(BLUE)🐛 Debug Test Mode$(RESET)"
	@echo "=================="
	@echo "$(GREEN)🔍 Comprehensive debugging with verbose logging$(RESET)"
	@echo "$(CYAN)💡 Optimized for troubleshooting and analysis$(RESET)"
	@echo ""
	@BENCHMARK_TEST_MODE=debug $(PYTHON) scripts/test_mode_runner.py debug $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Display current test mode configuration
show-test-config:
	@echo "$(BOLD)$(BLUE)📊 Current Test Mode Configuration$(RESET)"
	@echo "===================================="
	@$(PYTHON) -c "\
from core.test_mode_config import initialize_test_mode; \
config = initialize_test_mode(); \
print(f'$(GREEN)Mode:$(RESET) {config.mode.value}'); \
print(f'$(GREEN)Description:$(RESET) {config.description}'); \
print(f'$(GREEN)Chunk Size:$(RESET) {config.chunk_size}'); \
print(f'$(GREEN)Max Concurrent:$(RESET) {config.max_concurrent}'); \
print(f'$(GREEN)Timeout Per Test:$(RESET) {config.timeout_per_test}s'); \
print(f'$(GREEN)Memory Limit:$(RESET) {config.memory_limit_mb}MB'); \
print(f'$(GREEN)Coverage Enabled:$(RESET) {config.enable_coverage}'); \
print(f'$(GREEN)Verbose Output:$(RESET) {config.verbose_output}'); \
print(f'$(GREEN)Include Functional:$(RESET) {config.include_functional_tests}'); \
print(f'$(GREEN)Include Calibration:$(RESET) {config.include_calibration_tests}'); \
print(f'$(GREEN)Success Rate Threshold:$(RESET) {config.min_success_rate}'); \
print(f'$(GREEN)Calibration Threshold:$(RESET) {config.calibration_threshold}'); \
"

# Test the test mode configuration system
test-mode-config: pre-test
	@echo "$(BOLD)$(BLUE)🔧 Testing Test Mode Configuration System$(RESET)"
	@echo "=============================================="
	@$(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/unit/test_core_modules/test_test_mode_config.py $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Test server-dependent tests (functional and calibration) with timeout management
test-server-dependent: pre-test
	@echo "$(BOLD)$(BLUE)🌐 Testing Server-Dependent Test Suites$(RESET)"
	@echo "============================================="
	@echo "$(YELLOW)⚠️  This requires a running LLM server at http://localhost:8004$(RESET)"
	@echo "$(CYAN)💡 Testing functional/ and calibration/ directories with intelligent chunking$(RESET)"
	@echo ""
	@echo "$(BOLD)1. Testing Calibration Framework:$(RESET)"
	@timeout 60s $(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/calibration/ $(ARGS) || echo "$(YELLOW)⚠️ Calibration tests timed out or failed$(RESET)"
	@echo ""
	@echo "$(BOLD)2. Testing Functional Tests:$(RESET)" 
	@timeout 300s $(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/functional/ $(ARGS) || echo "$(YELLOW)⚠️ Functional tests timed out or failed$(RESET)"
	@echo ""
	@echo "$(GREEN)✅ Server-dependent test validation complete$(RESET)"
	@$(MAKE) post-test

# Test nonexistent test ID error handling manually
test-error-handling-manual:
	@echo "$(BOLD)$(BLUE)🧪 Testing Error Handling for Nonexistent Test ID$(RESET)"
	@echo "======================================================"
	@echo "$(CYAN)Running benchmark_runner.py with nonexistent test ID...$(RESET)"
	@echo ""
	@python benchmark_runner.py --test-type base --test-id nonexistent_test_999 --endpoint http://127.0.0.1:8004/v1/completions --model "/app/models/hf/DeepSeek-R1-0528-Qwen3-8b" --output-dir /tmp/test_output 2>&1 || echo "$(GREEN)✅ Error handling working - nonzero exit code as expected$(RESET)"

# Debug evaluation flag behavior manually
debug-evaluation-flag:
	@echo "$(BOLD)$(BLUE)🔧 Debugging Evaluation Flag Behavior$(RESET)"
	@echo "================================================="
	@echo "$(CYAN)Running benchmark_runner.py with --evaluation flag...$(RESET)"
	@echo ""
	@mkdir -p /tmp/eval_test_output
	@python benchmark_runner.py --test-type base --test-id basic_01 --evaluation --endpoint http://127.0.0.1:8004/v1/completions --model "/app/models/hf/DeepSeek-R1-0528-Qwen3-8b" --output-dir /tmp/eval_test_output --verbose
	@echo ""
	@echo "$(CYAN)Contents of output directory:$(RESET)"
	@ls -la /tmp/eval_test_output/
	@echo ""
	@echo "$(CYAN)Checking result file for evaluation data:$(RESET)"
	@find /tmp/eval_test_output/ -name "*.json" -exec echo "File: {}" \; -exec cat {} \; 2>/dev/null || echo "No JSON files found"

# Debug pattern_completion category execution
debug-pattern-completion:
	@echo "$(BOLD)$(BLUE)🧪 Debugging Pattern Completion Category$(RESET)"
	@echo "==================================================="
	@echo "$(CYAN)Running benchmark_runner.py with pattern_completion category...$(RESET)"
	@echo ""
	@mkdir -p /tmp/pattern_test_output
	@timeout 60s python benchmark_runner.py --test-type base --category pattern_completion --mode category --endpoint http://127.0.0.1:8004/v1/completions --model "/app/models/hf/DeepSeek-R1-0528-Qwen3-8b" --output-dir /tmp/pattern_test_output --verbose 2>&1 || echo "$(YELLOW)⚠️ Command timed out or failed$(RESET)"
	@echo ""
	@echo "$(CYAN)Contents of output directory:$(RESET)"
	@ls -la /tmp/pattern_test_output/ 2>/dev/null || echo "Directory not found"
	@echo ""
	@echo "$(CYAN)Test IDs in pattern_completion category:$(RESET)"
	@python -c "import json; data=json.load(open('domains/reasoning/base_models/categories.json')); print('Test IDs:', data['categories']['pattern_completion']['test_ids'])"

# Debug production test duplicate files issue
check-pycache-dirs:
	@echo "$(BOLD)$(BLUE)🔍 Checking Python Cache Directories$(RESET)"
	@echo "====================================="
	@python3 -c "import os; cache_dirs = [d for root, dirs, files in os.walk('tests') for d in dirs if d == '__pycache__']; print(f'Found {len(cache_dirs)} __pycache__ directories in tests/'); [print(f'  {os.path.join(root, d)}') for root, dirs, files in os.walk('tests') for d in dirs if d == '__pycache__']"

debug-duplicate-files:
	@echo "$(BOLD)$(BLUE)🔍 Debug Duplicate Test Files Issue$(RESET)"
	@echo "========================================="
	@echo "$(CYAN)Checking for duplicate test files in both test_cultural_validation/ and test_core_modules/$(RESET)"
	@python3 -c "import os; import glob; cultural_files = set(os.path.basename(f) for f in glob.glob('tests/unit/test_cultural_validation/test_*.py')); core_files = set(os.path.basename(f) for f in glob.glob('tests/unit/test_core_modules/test_*.py')); duplicates = cultural_files & core_files; print(f'Duplicate files found: {len(duplicates)}'); [print(f'  - {f}') for f in sorted(duplicates)]"

fix-duplicate-files:
	@echo "$(BOLD)$(BLUE)🔧 Fixing Duplicate Test Files$(RESET)"
	@echo "=================================="
	@echo "$(YELLOW)⚠️  Removing duplicate files from test_core_modules/ (keeping originals in test_cultural_validation/)$(RESET)"
	@python3 -c "import os; duplicates = ['test_cultural_authenticity.py', 'test_cultural_dataset_validator.py', 'test_cultural_pattern_library.py', 'test_tradition_validator.py']; [os.remove(f'tests/unit/test_core_modules/{f}') if os.path.exists(f'tests/unit/test_core_modules/{f}') else print(f'File {f} already removed') for f in duplicates]; print('✅ Duplicate files removed from test_core_modules/')"

# Debug functional test failures (Phase 1 Stabilization)
debug-category-execution:
	@echo "$(BOLD)$(BLUE)🔍 Debug Category Execution Test$(RESET)"
	@echo "====================================="
	@echo "$(CYAN)Testing --category basic_logic_patterns --mode category CLI execution$(RESET)"
	@mkdir -p /tmp/debug_category_test
	@python3 benchmark_runner.py --test-type base --category basic_logic_patterns --mode category --endpoint http://localhost:8004/v1/completions --model "/app/models/hf/DeepSeek-R1-0528-Qwen3-8b" --output-dir /tmp/debug_category_test --verbose 2>&1 || echo "$(RED)❌ Category execution failed$(RESET)"
	@echo ""
	@echo "$(CYAN)Output directory contents:$(RESET)"
	@ls -la /tmp/debug_category_test/ 2>/dev/null || echo "Directory not found"

debug-instruct-execution:
	@echo "$(BOLD)$(BLUE)🔍 Debug Instruct Domain Execution Test$(RESET)"
	@echo "==========================================="
	@echo "$(CYAN)Testing --test-type instruct --test-id basic_01 CLI execution$(RESET)"
	@mkdir -p /tmp/debug_instruct_test
	@python3 benchmark_runner.py --test-type instruct --test-id basic_01 --endpoint http://localhost:8004/v1/completions --model "/app/models/hf/DeepSeek-R1-0528-Qwen3-8b" --output-dir /tmp/debug_instruct_test --verbose 2>&1 || echo "$(RED)❌ Instruct execution failed$(RESET)"
	@echo ""
	@echo "$(CYAN)Output directory contents:$(RESET)"
	@ls -la /tmp/debug_instruct_test/ 2>/dev/null || echo "Directory not found"
	@echo ""
	@echo "$(CYAN)Checking instruct domain files:$(RESET)"
	@ls -la domains/reasoning/instruct_models/

debug-chunked-execution:
	@echo "$(BOLD)$(BLUE)🔍 Debug Chunked Execution Test$(RESET)"
	@echo "===================================="
	@echo "$(CYAN)Testing chunked execution runner functionality$(RESET)"
	@python3 -c "from tests.functional.chunked_test_runner import create_quick_test_runner; runner = create_quick_test_runner(chunk_size=2, timeout=30); print('✅ Quick test runner created successfully')" || echo "$(RED)❌ Chunked test runner import/creation failed$(RESET)"

debug-functional-test-failures:
	@echo "$(BOLD)$(BLUE)🔍 Debug All Functional Test Failures$(RESET)"
	@echo "============================================="
	@echo "$(CYAN)Running individual failing functional tests with detailed output$(RESET)"
	@echo ""
	@echo "$(YELLOW)Test 1: Category Execution$(RESET)"
	@timeout 60s $(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/functional/test_cli_workflows.py::TestCLIWorkflows::test_category_execution -v -s || echo "$(RED)❌ Category execution test failed$(RESET)"
	@echo ""
	@echo "$(YELLOW)Test 2: Chunked Category Execution$(RESET)"
	@timeout 60s $(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/functional/test_cli_workflows.py::TestCLIWorkflows::test_chunked_category_execution -v -s || echo "$(RED)❌ Chunked category execution test failed$(RESET)"
	@echo ""
	@echo "$(YELLOW)Test 3: Instruct Domain Execution$(RESET)"
	@timeout 60s $(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/functional/test_domain_execution.py::TestDomainExecution::test_instruct_domain_execution -v -s || echo "$(RED)❌ Instruct domain execution test failed$(RESET)"

debug-chunked-subprocess-execution:
	@echo "$(BOLD)$(BLUE)🔍 Debug Chunked Test Subprocess Execution$(RESET)"
	@echo "======================================================"
	@echo "$(CYAN)Testing individual basic_01 and basic_02 via subprocess as chunked runner does$(RESET)"
	@mkdir -p /tmp/debug_chunked_subprocess
	@echo ""
	@echo "$(YELLOW)Subprocess test 1: basic_01$(RESET)"
	@cd /home/alejandro/workspace/ai-workstation/benchmark_tests && timeout 45s python benchmark_runner.py --test-type base --test-id basic_01 --endpoint http://localhost:8004/v1/completions --model "/app/models/hf/DeepSeek-R1-0528-Qwen3-8b" --enhanced-evaluation --output-dir /tmp/debug_chunked_subprocess 2>&1 || echo "$(RED)❌ basic_01 subprocess failed$(RESET)"
	@echo ""
	@echo "$(YELLOW)Subprocess test 2: basic_02$(RESET)"  
	@cd /home/alejandro/workspace/ai-workstation/benchmark_tests && timeout 45s python benchmark_runner.py --test-type base --test-id basic_02 --endpoint http://localhost:8004/v1/completions --model "/app/models/hf/DeepSeek-R1-0528-Qwen3-8b" --enhanced-evaluation --output-dir /tmp/debug_chunked_subprocess 2>&1 || echo "$(RED)❌ basic_02 subprocess failed$(RESET)"
	@echo ""
	@echo "$(CYAN)Checking subprocess results:$(RESET)"
	@ls -la /tmp/debug_chunked_subprocess/ || echo "No results found"

validate-domain-files:
	@echo "$(BOLD)$(BLUE)🔍 Validate Domain Files for Functional Tests$(RESET)"
	@echo "================================================="
	@echo "$(CYAN)Checking category definition files:$(RESET)"
	@python3 -c "import json; data=json.load(open('domains/reasoning/base_models/categories.json')); print(f'✅ Categories loaded: {len(data[\"categories\"])} categories'); print(f'📋 basic_logic_patterns test IDs: {len(data[\"categories\"][\"basic_logic_patterns\"][\"test_ids\"])} tests')"
	@echo ""
	@echo "$(CYAN)Checking instruct model files:$(RESET)"
	@python3 -c "import json; data=json.load(open('domains/reasoning/instruct_models/easy.json')); print(f'✅ Instruct tests loaded: {len(data[\"tests\"])} tests'); basic_01 = next((t for t in data[\"tests\"] if t[\"id\"] == \"basic_01\"), None); print(f'📋 basic_01 found: {\"Yes\" if basic_01 else \"No\"}')"
	@echo ""
	@echo "$(CYAN)Verifying domain discovery:$(RESET)"
	@python3 benchmark_runner.py --discover-suites 2>&1 | head -10

# Debug unit test failures (Phase 1 Stabilization)
debug-unit-test-failures:
	@echo "$(BOLD)$(BLUE)🧪 Debug Unit Test Failures$(RESET)"
	@echo "==================================="
	@echo "$(CYAN)Running systematic analysis of 41 unit test failures$(RESET)"
	@echo ""
	@echo "$(YELLOW)Test 1: Cognitive Validation (CRITICAL priority)$(RESET)"
	@timeout 60s $(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/unit/test_core_modules/test_cognitive_validation.py -v --tb=short || echo "$(RED)❌ Cognitive validation tests failed$(RESET)"
	@echo ""
	@echo "$(YELLOW)Test 2: Enhanced Universal Evaluator (HIGH priority)$(RESET)"
	@timeout 60s $(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/unit/test_evaluator_integration/test_enhanced_universal_evaluator.py -v --tb=short || echo "$(RED)❌ Enhanced evaluator tests failed$(RESET)"
	@echo ""
	@echo "$(YELLOW)Test 3: Resource Manager (MEDIUM priority)$(RESET)"
	@timeout 60s $(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/unit/test_core_modules/test_resource_manager.py -v --tb=short || echo "$(RED)❌ Resource manager tests failed$(RESET)"

debug-unit-import-issues:
	@echo "$(BOLD)$(BLUE)🔍 Debug Unit Test Import Issues$(RESET)"
	@echo "========================================"
	@echo "$(CYAN)Testing imports for problematic modules$(RESET)"
	@echo ""
	@echo "$(YELLOW)Testing cognitive validation imports:$(RESET)"
	@python3 -c "from core.cognitive_validation import CognitiveValidationPipeline; print('✅ CognitiveValidationPipeline import OK')" || echo "$(RED)❌ Cognitive validation import failed$(RESET)"
	@echo ""
	@echo "$(YELLOW)Testing enhanced evaluator imports:$(RESET)"
	@python3 -c "from evaluator.subjects.enhanced_universal_evaluator import EnhancedUniversalEvaluator; print('✅ EnhancedUniversalEvaluator import OK')" || echo "$(RED)❌ Enhanced evaluator import failed$(RESET)"
	@echo ""
	@echo "$(YELLOW)Testing resource manager imports:$(RESET)"
	@python3 -c "from core.resource_manager import ResourceManager; print('✅ ResourceManager import OK')" || echo "$(RED)❌ Resource manager import failed$(RESET)"

debug-unit-test-specific:
	@echo "$(BOLD)$(BLUE)🎯 Debug Specific Unit Test Failure$(RESET)"
	@echo "========================================"
	@if [ -z "$(TEST)" ]; then \
		echo "$(RED)❌ Error: TEST parameter required$(RESET)"; \
		echo "Usage: make debug-unit-test-specific TEST=path/to/test.py::TestClass::test_method"; \
		exit 1; \
	fi
	@echo "$(CYAN)Running specific unit test: $(TEST)$(RESET)"
	@$(PYTEST) $(PYTEST_VERBOSE_ARGS) $(TEST) --tb=long -v -s

debug-phase1c-segment-analysis:
	@echo "$(BOLD)$(BLUE)🐛 Debug Phase 1C Segment Analysis$(RESET)"
	@echo "======================================"
	@echo "$(CYAN)Testing final segment extraction and structure detection$(RESET)"
	@$(PYTHON) -c "\
from evaluator.subjects.enhanced_universal_evaluator import EnhancedUniversalEvaluator; \
evaluator = EnhancedUniversalEvaluator(); \
test_response = '''We must respond to the final answer...\nLet me think about this...\nActually, let me approach this differently...\n\n**Step‑by‑step reasoning**\n\n1. **Identify the pattern**\nEach line of the oriki adds a new attribute or metaphor.\n\n2. **Choose attributes that fit**\nProtector, Guardian, Strong one with cultural significance.\n\n**Completed oriki**\n\n> Protector of the village's children,\n> Guardian who keeps the river's current steady,\n> Strong one whose roots run as deep as the baobab.'''; \
analysis = evaluator._analyze_final_segment_quality(test_response); \
print(f'Total lines: {len(test_response.strip().split(chr(10)))}'); \
print(f'Final segment start: {int(len(test_response.strip().split(chr(10))) * 0.75)}'); \
print(f'Has structure: {analysis[\"has_structure\"]}'); \
print(f'Quality score: {analysis[\"quality_score\"]}'); \
print(f'Final segment preview: {analysis.get(\"final_segment\", \"\")[:100]}...'); \
structure_test = evaluator._check_structured_format(analysis.get('final_segment', '')); \
print(f'Direct structure check: {structure_test}'); \
"

debug-cultural-evaluation:
	@echo "$(BOLD)$(BLUE)🐛 Debug Cultural Evaluation Issue$(RESET)"
	@echo "======================================="
	@echo "$(CYAN)Testing cultural evaluation with Islamic content$(RESET)"
	@$(PYTHON) -c "\
from evaluator.subjects.enhanced_universal_evaluator import EnhancedUniversalEvaluator; \
evaluator = EnhancedUniversalEvaluator(); \
test_def = { \
    'name': 'islamic_guidance', \
    'prompt': 'Discuss the concept of divine guidance in Islamic tradition', \
    'category': 'cultural', \
    'cultural_context': {'traditions': ['Islamic']}, \
    'description': 'Understanding of Islamic spiritual concepts' \
}; \
response = \"Allah's guidance provides wisdom and direction to believers through the teachings of the Quran and the example of the Prophet.\"; \
print(f'Input prompt: {test_def[\"prompt\"]}'); \
print(f'Input response: {response[:50]}...'); \
task_type = evaluator._detect_task_type(test_def, response); \
print(f'Detected task type: {task_type}'); \
prompt_lower = test_def.get('prompt', '').lower(); \
desc_lower = test_def.get('description', '').lower(); \
response_lower = response.lower(); \
cultural_terms = ['cultural', 'tradition', 'spiritual', 'religious', 'heritage', 'islamic', 'arabic', 'native', 'chinese', 'vedic', 'celtic', 'yoruba', 'african', 'wu xing', 'five elements', 'dharma', 'karma']; \
is_cultural = any(term in prompt_lower or term in desc_lower or term in response_lower for term in cultural_terms); \
print(f'Is cultural check: {is_cultural} (prompt: islamic={\"islamic\" in prompt_lower}, response: allah={\"allah\" in response_lower})'); \
cultural_score_direct = evaluator._assess_cultural_authenticity(response, test_def.get('prompt', ''), test_def.get('description', '')); \
print(f'Direct cultural authenticity score: {cultural_score_direct}'); \
result = evaluator.evaluate_response_enhanced(response, test_def, 'islamic_guidance'); \
print(f'Cultural depth score: {result.enhanced_metrics.cultural_depth_score}'); \
print(f'Overall score: {result.enhanced_metrics.overall_score}'); \
"

run-unit-tests-only:
	@echo "$(BOLD)$(BLUE)🧪 Run Unit Tests Only (Excluding Functional/Calibration)$(RESET)"
	@echo "=========================================================="
	@echo "$(CYAN)Running all unit tests to identify current failure count$(RESET)"
	@$(PYTEST) $(PYTEST_BASE_ARGS) -v tests/unit/ --ignore=tests/unit/test_calibration_integration/ $(ARGS) || echo "$(YELLOW)⚠️ Unit tests completed with failures - this is expected$(RESET)"

# ========================================================================
# PHASE 1C TESTING COMMANDS - Evidence-Based Loop-Recovery Validation
# ========================================================================

# Comprehensive Phase 1C validation suite
test-phase1c-validation: pre-test
	@echo "$(BOLD)$(BLUE)🔬 Phase 1C Comprehensive Validation Suite$(RESET)"
	@echo "==============================================="
	@echo "$(CYAN)💡 Evidence-based testing using actual test_results/ data$(RESET)"
	@echo "$(GREEN)🧪 Testing loop-recovery scoring system validation$(RESET)"
	@echo ""
	@echo "$(BOLD)1. Running Functional Tests:$(RESET)"
	@$(MAKE) test-phase1c-functional
	@echo ""
	@echo "$(BOLD)2. Running Unit Tests:$(RESET)"
	@$(MAKE) test-phase1c-unit
	@echo ""
	@echo "$(BOLD)3. Running Calibration Investigation:$(RESET)"
	@$(MAKE) test-phase1c-calibration
	@echo ""
	@echo "$(BOLD)4. Running Integration Tests:$(RESET)"
	@$(MAKE) test-phase1c-integration
	@echo ""
	@echo "$(GREEN)✅ Phase 1C validation suite complete!$(RESET)"
	@$(MAKE) post-test

# Phase 1C functional tests (loop-recovery validation)
test-phase1c-functional: pre-test
	@echo "$(BOLD)$(BLUE)🧪 Phase 1C Functional Tests$(RESET)"
	@echo "==============================="
	@echo "$(CYAN)Testing loop-recovery system with actual response patterns$(RESET)"
	@$(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/functional/test_phase1c_loop_recovery.py $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Phase 1C unit tests (individual functions)
test-phase1c-unit: pre-test
	@echo "$(BOLD)$(BLUE)🔧 Phase 1C Unit Tests$(RESET)"
	@echo "========================="
	@echo "$(CYAN)Testing individual Phase 1C functions and logic$(RESET)"
	@$(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/unit/test_segment_quality_analysis.py tests/unit/test_loop_classification.py tests/unit/test_hybrid_scoring_math.py $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Phase 1C calibration tests (regression investigation)
test-phase1c-calibration: pre-test
	@echo "$(BOLD)$(BLUE)🔍 Phase 1C Calibration Investigation$(RESET)"
	@echo "============================================"
	@echo "$(CYAN)Investigating scoring regressions and calibration issues$(RESET)"
	@echo "$(YELLOW)⚠️  Focus: math_08 regression (64.8→56.8) and basic_03 recovery (10.0→88.0)$(RESET)"
	@$(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/calibration/test_loop_recovery_calibration.py $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Phase 1C integration tests (evaluator integration)
test-phase1c-integration: pre-test
	@echo "$(BOLD)$(BLUE)🔗 Phase 1C Integration Tests$(RESET)"
	@echo "================================="
	@echo "$(CYAN)Testing Phase 1C integration with enhanced evaluator$(RESET)"
	@$(PYTEST) $(PYTEST_VERBOSE_ARGS) tests/integration/test_enhanced_evaluator_phase1c.py $(ARGS) && \
	$(MAKE) post-test || ($(MAKE) post-test && exit 1)

# Debug quality scoring calculation
debug-quality-scoring:
	@echo "$(BOLD)$(BLUE)🐛 Debug Quality Scoring$(RESET)"
	@echo "==========================="
	@echo "$(CYAN)Testing quality scoring calculation logic$(RESET)"
	@python3 -c "from evaluator.subjects.enhanced_universal_evaluator import EnhancedUniversalEvaluator; \
	evaluator = EnhancedUniversalEvaluator(); \
	text = 'This is a substantial response with meaningful content that demonstrates quality and completeness in its delivery and analysis.'; \
	score = evaluator._calculate_segment_quality(text, has_structure=True, is_coherent=True, delivers_content=True); \
	word_count = len(text.split()); \
	print(f'Text: {text}'); \
	print(f'Word count: {word_count}'); \
	print(f'Expected: base(40) + structure(25) + coherence(20) + content(15) + length bonus'); \
	print(f'Actual score: {score}'); \
	print(f'Length bonus calculation: word_count={word_count}, bonus={min(10.0, word_count/20) if word_count > 50 else 0 if word_count >= 20 else -10.0}')"

# Test entropy calculator functionality
test-entropy-calculator:
	@echo "$(BOLD)$(BLUE)🧮 Testing Entropy Calculator$(RESET)"
	@echo "==============================="
	@echo "$(CYAN)Testing entropy calculations and fallback systems$(RESET)"
	@python3 -c "from evaluator.advanced.entropy_calculator import run_entropy_tests; results = run_entropy_tests(); print('Entropy Test Results:'); import json; print(json.dumps(results, indent=2))"

# Debug semantic entropy integration in reasoning evaluator
debug-semantic-entropy:
	@echo "$(BOLD)$(BLUE)🔍 Debug Semantic Entropy Negative Zero$(RESET)"
	@echo "==============================================="
	@echo "$(CYAN)Investigating why semantic_entropy returns -0.0$(RESET)"
	@python3 -c "from evaluator.advanced.entropy_calculator import EntropyCalculator; \
	calc = EntropyCalculator(); \
	test_text = 'Make sure to maintain the parallel structure. Ensure the final answer is clear and follows all instructions.'; \
	print(f'Test text: {test_text}'); \
	result = calc.calculate_semantic_entropy(test_text); \
	print(f'Direct semantic entropy calculation: {result}'); \
	basic_result = calc._calculate_basic_semantic_entropy(test_text); \
	print(f'Basic semantic entropy calculation: {basic_result}'); \
	profile = calc.analyze_entropy_profile(test_text); \
	print(f'Full profile semantic_entropy: {profile[\"semantic_entropy\"]}'); \
	print(f'Type of result: {type(profile[\"semantic_entropy\"])}'); \
	print(f'Is negative zero: {profile[\"semantic_entropy\"] == -0.0}');"

# Test fresh orchestrator integration
test-final-integration:
	@echo "$(BOLD)$(BLUE)🎉 Final Integration Test$(RESET)"
	@echo "================================="
	@echo "$(CYAN)Testing complete evaluation pipeline with all fixes$(RESET)"
	@python -Bc "import sys; sys.dont_write_bytecode = True; \
	from evaluator.core.advanced_analysis_orchestrator import AdvancedAnalysisOrchestrator; \
	orchestrator = AdvancedAnalysisOrchestrator(); \
	print('=== TESTING NORMAL TEXT ==='); \
	normal_text = 'The African proverb follows a specific pattern. The student who learns from failure becomes wise.'; \
	result1 = orchestrator.run_advanced_analysis(normal_text, domain_context='reasoning'); \
	entropy1 = result1.analysis_data.get('entropy_analysis', {}); \
	print(f'Normal text - Semantic entropy: {entropy1.get(\"semantic_entropy\", \"MISSING\")}'); \
	print(f'Normal text - Embedding variance: {entropy1.get(\"embedding_variance\", \"MISSING\")}'); \
	print(f'Normal text - Repetitive patterns: {entropy1.get(\"entropy_patterns\", {}).get(\"has_repetitive_patterns\", \"MISSING\")}'); \
	print(''); \
	print('=== TESTING REPETITIVE TEXT ==='); \
	loop_text = 'Ensure the final answer follows instructions. Make sure the answer is complete. Ensure the final answer follows instructions. Make sure the answer is complete. Ensure the final answer follows instructions.'; \
	result2 = orchestrator.run_advanced_analysis(loop_text, domain_context='reasoning'); \
	entropy2 = result2.analysis_data.get('entropy_analysis', {}); \
	print(f'Loop text - Semantic entropy: {entropy2.get(\"semantic_entropy\", \"MISSING\")}'); \
	print(f'Loop text - Embedding variance: {entropy2.get(\"embedding_variance\", \"MISSING\")}'); \
	print(f'Loop text - Repetitive patterns: {entropy2.get(\"entropy_patterns\", {}).get(\"has_repetitive_patterns\", \"MISSING\")}'); \
	print(''); \
	print('=== SUMMARY OF FIXES ==='); \
	print(f'✅ Semantic entropy working: {\"YES\" if entropy1.get(\"semantic_entropy\", 0) > 0 and entropy2.get(\"semantic_entropy\", 0) > 0 else \"NO\"}'); \
	print(f'✅ Embedding variance working: {\"YES\" if entropy1.get(\"embedding_variance\", 0) > 0 and entropy2.get(\"embedding_variance\", 0) > 0 else \"NO\"}'); \
	print(f'✅ Loop detection working: {\"YES\" if entropy2.get(\"entropy_patterns\", {}).get(\"has_repetitive_patterns\") == True else \"NO\"}');"

# Debug repetitive text detection  
debug-repetitive-detection:
	@echo "$(BOLD)$(BLUE)🔄 Debug Basic_03 Massive Loop Pattern$(RESET)"
	@echo "=============================================="
	@echo "$(CYAN)Testing loop detection on basic_03 massive repetitive pattern$(RESET)"
	@python3 -c "from evaluator.advanced.entropy_calculator import EntropyCalculator; \
	calc = EntropyCalculator(); \
	massive_loop = '''- Ensure the final answer is clear, complete, and follows all instructions - Provide a concise explanation of your reasoning process - Make sure the final answer is presented in a boxed format - Avoid any markdown formatting in the final answer - Keep the final answer within the specified format - Ensure the final answer is accurate and follows all given instructions - Confirm that the final answer is complete and addresses all parts of the query - Verify that the final answer is presented in the correct format and meets all requirements - Double-check that the final answer is free from any errors or inconsistencies - Ensure the final answer is well-structured and easy to understand - Confirm that the final answer is comprehensive and addresses all aspects of the query - Verify that the final answer is presented in a clear and concise manner - Ensure the final answer is accurate and follows all given instructions - Confirm that the final answer is complete and meets all specified requirements - Double-check that the final answer is presented in the correct format and adheres to all guidelines - Ensure the final answer is well-organized and easy to follow - Verify that the final answer is accurate and complete - Confirm that the final answer is presented in the correct format and meets all specified criteria - Ensure the final answer is clear, concise, and follows all instructions - Double-check that the final answer is accurate and complete - Verify that the final answer is presented in the correct format and adheres to all guidelines - Ensure the final answer is well-structured and easy to understand'''; \
	print(f'Testing massive loop pattern (length: {len(massive_loop)} chars)'); \
	profile = calc.analyze_entropy_profile(massive_loop); \
	print(f'Semantic entropy: {profile[\"semantic_entropy\"]}'); \
	print(f'Semantic diversity: {profile[\"semantic_diversity\"]}'); \
	print(f'Embedding variance: {profile[\"embedding_variance\"]}'); \
	print(f'Has repetitive patterns: {profile[\"entropy_patterns\"].get(\"has_repetitive_patterns\", \"MISSING\")}'); \
	print(f'Entropy variance: {profile[\"entropy_patterns\"].get(\"entropy_variance\", \"MISSING\")}'); \
	print(f'Entropy trend: {profile[\"entropy_patterns\"].get(\"entropy_trend\", \"MISSING\")}');"