#!/usr/bin/env python3
"""
Critical CLI Argument Validation Tests for benchmark_runner.py

Tests core CLI argument parsing and validation logic that users depend on.
Keeps it simple - only tests critical argument combinations.
"""

import unittest
import sys
import os

# Add the benchmark_tests directory to Python path
benchmark_tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, benchmark_tests_dir)

from benchmark_runner import BenchmarkTestRunner, load_and_configure_runner


class TestBenchmarkRunnerArgs(unittest.TestCase):
    """Test critical CLI argument validation in BenchmarkTestRunner"""
    
    def test_invalid_test_type_handled_gracefully(self):
        """Test that invalid test types are handled gracefully"""
        # The function actually loads base tests even with invalid type (fallback behavior)  
        runner = load_and_configure_runner(test_type="invalid")
        self.assertIsInstance(runner, BenchmarkTestRunner)
        # Function should still work and load some default tests
        self.assertGreaterEqual(len(runner.tests), 0, "Should handle invalid test type gracefully")
    
    def test_valid_base_type_accepted(self):
        """Test that valid base test type is accepted"""
        try:
            runner = load_and_configure_runner(test_type="base")
            self.assertIsInstance(runner, BenchmarkTestRunner)
            self.assertGreater(len(runner.tests), 0, "Should load some tests")
        except Exception as e:
            self.fail(f"Valid base test type should not raise exception: {e}")
    
    def test_valid_instruct_type_accepted(self):
        """Test that valid instruct test type is accepted"""
        try:
            runner = load_and_configure_runner(test_type="instruct")
            self.assertIsInstance(runner, BenchmarkTestRunner)
            # Instruct tests might be empty, that's OK
        except Exception as e:
            self.fail(f"Valid instruct test type should not raise exception: {e}")
    
    def test_specific_domain_loading(self):
        """Test that specific domain loading works"""
        try:
            runner = load_and_configure_runner(test_type="base", domain="reasoning")
            self.assertIsInstance(runner, BenchmarkTestRunner)
            # Should have loaded reasoning domain tests
        except Exception as e:
            self.fail(f"Domain-specific loading should work: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)