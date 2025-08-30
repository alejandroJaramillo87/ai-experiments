# Benchmark Tests Makefile
# Provides convenient commands for running tests with automatic cleanup

.PHONY: test test-unit test-integration test-analysis clean help

# Default target
help:
	@echo "🧪 Benchmark Tests Automation"
	@echo "=============================="
	@echo ""
	@echo "Available commands:"
	@echo "  make test           - Run all tests (unit + integration) with cleanup"
	@echo "  make test-unit      - Run unit tests only with cleanup"
	@echo "  make test-integration - Run integration tests only with cleanup"
	@echo "  make test-analysis  - Run analysis scripts with cleanup"
	@echo "  make test-quick     - Run quick test subset with cleanup"
	@echo "  make clean          - Clean test artifacts without running tests"
	@echo "  make help           - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make test-unit"
	@echo "  make test ARGS='-v -k test_evaluator'"
	@echo "  make clean"

# Run all tests with cleanup
test:
	@python run_tests_with_cleanup.py tests/unit/ tests/integration/ --tb=short $(ARGS)

# Run unit tests only
test-unit:
	@python run_tests_with_cleanup.py tests/unit/ -q $(ARGS)

# Run integration tests only  
test-integration:
	@python run_tests_with_cleanup.py tests/integration/ -q $(ARGS)

# Run analysis scripts
test-analysis:
	@echo "🔍 Running analysis scripts with cleanup..."
	@python tests/analysis/analyze_scoring_patterns.py
	@python tests/integration/comprehensive_validation_suite.py
	@python tests/integration/run_edge_case_tests.py
	@python cleanup_test_artifacts.py

# Quick test subset
test-quick:
	@python run_tests_with_cleanup.py tests/unit/test_universal_evaluator.py tests/unit/test_evaluation_config.py -q $(ARGS)

# Clean artifacts without running tests
clean:
	@python cleanup_test_artifacts.py

# Development targets
test-watch:
	@echo "👀 Watching for file changes... (Ctrl+C to stop)"
	@while true; do \
		make test-quick; \
		echo ""; \
		echo "⏱️  Waiting for changes... (modify any .py file to re-run)"; \
		inotifywait -q -e modify,create,delete *.py **/*.py 2>/dev/null || sleep 5; \
		clear; \
	done

# Verbose test run
test-verbose:
	@python run_tests_with_cleanup.py tests/unit/ tests/integration/ -v $(ARGS)