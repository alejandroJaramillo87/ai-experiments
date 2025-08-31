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
.PHONY: help test test-unit test-integration test-calibration test-modular test-analysis test-specific test-watch clean clean-all setup debug-help calibration-demo examples check-prerequisites calibration-status calibration-validate test-domain-loading test-enhanced-evaluator test-semantic-analyzer test-api-suite debug-calibration-framework convert-creativity-tests test-creativity-conversion convert-core-domains test-core-domains-conversion

# Default target
help:
	@echo "$(BOLD)$(BLUE)🧪 AI Workstation Benchmark Tests$(RESET)"
	@echo "$(BOLD)================================$(RESET)"
	@echo ""
	@echo "$(BOLD)Core Commands:$(RESET)"
	@echo "  $(GREEN)make test [MODE=quick|verbose|coverage] [SUITE=unit|integration|all]$(RESET)"
	@echo "    - Unified test runner with flexible options"
	@echo "  $(GREEN)make test-calibration$(RESET)     - Calibration validation framework tests"
	@echo "  $(GREEN)make test-analysis$(RESET)        - Analysis and validation scripts"
	@echo "  $(GREEN)make test-specific FILE=path$(RESET) - Run specific test file/method"
	@echo ""
	@echo "$(BOLD)Quality Assurance:$(RESET)"
	@echo "  $(GREEN)make check-prerequisites$(RESET)  - Complete system readiness check"
	@echo "  $(GREEN)make calibration-validate$(RESET) - Live LLM calibration validation"
	@echo "  $(GREEN)make test-api-suite$(RESET)       - Complete API connectivity test"
	@echo "  $(GREEN)make test-enhanced-evaluator$(RESET) - Enhanced evaluator import test"
	@echo ""
	@echo "$(BOLD)Domain Coverage:$(RESET)"
	@echo "  $(GREEN)make convert-creativity-tests$(RESET)    - Convert creativity base→instruct tests"
	@echo "  $(GREEN)make convert-core-domains$(RESET)        - Convert all core domains to instruct"
	@echo "  $(GREEN)make test-creativity-conversion$(RESET)  - Test creativity conversion results"
	@echo "  $(GREEN)make test-core-domains-conversion$(RESET) - Test core domain conversion"
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
	@echo "$(GREEN)✅ Cleanup complete!$(RESET)"

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
	@python3 -c "import json; data = json.load(open('domains/reasoning/base_models/easy.json')); print(f'✅ Loaded {len(data[\"tests\"])} tests from easy.json'); print(f'📝 First test ID: {data[\"tests\"][0][\"id\"]}'); print(f'🎯 Test name: {data[\"tests\"][0][\"name\"]}')"

test-enhanced-evaluator:
	@echo "$(BOLD)$(BLUE)🧠 Testing Enhanced Evaluator Import$(RESET)"
	@echo "====================================="
	@python3 -c "from evaluator.subjects.enhanced_universal_evaluator import EnhancedUniversalEvaluator; print('✅ Enhanced evaluator import successful'); evaluator = EnhancedUniversalEvaluator(); print('✅ Enhanced evaluator instantiation successful')" || echo "❌ Enhanced evaluator error"

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