#!/usr/bin/env python3
"""
Functional Test Suite Runner

Runs the complete functional test suite for the benchmark system.
Tests the end-to-end workflow with real API calls and actual domain test files.

Usage:
    python run_functional_tests.py [--verbose] [--specific-test TestClass.test_method]
"""

import sys
import os
import unittest
import argparse
import time
from typing import Optional

# Add benchmark_tests to path
benchmark_tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, benchmark_tests_dir)

# Import test modules
from tests.functional.test_cli_workflows import TestCLIWorkflows
from tests.functional.test_domain_execution import TestDomainExecution  
from tests.functional.test_result_validation import TestResultValidation


def create_test_suite(specific_test: Optional[str] = None) -> unittest.TestSuite:
    """
    Create test suite with all functional tests or specific test.
    
    Args:
        specific_test: Optional specific test in format "TestClass.test_method"
        
    Returns:
        Configured test suite
    """
    suite = unittest.TestSuite()
    
    if specific_test:
        # Run specific test
        if '.' in specific_test:
            class_name, method_name = specific_test.split('.', 1)
            test_classes = {
                'TestCLIWorkflows': TestCLIWorkflows,
                'TestDomainExecution': TestDomainExecution,
                'TestResultValidation': TestResultValidation
            }
            
            if class_name in test_classes:
                suite.addTest(test_classes[class_name](method_name))
            else:
                print(f"Unknown test class: {class_name}")
                return suite
        else:
            print(f"Invalid test format: {specific_test}. Use 'TestClass.test_method'")
            return suite
    else:
        # Add all test classes
        suite.addTest(unittest.makeSuite(TestCLIWorkflows))
        suite.addTest(unittest.makeSuite(TestDomainExecution))
        suite.addTest(unittest.makeSuite(TestResultValidation))
    
    return suite


def run_tests(verbose: bool = False, specific_test: Optional[str] = None) -> bool:
    """
    Run functional test suite.
    
    Args:
        verbose: Enable verbose output
        specific_test: Run specific test only
        
    Returns:
        True if all tests passed, False otherwise
    """
    print("🚀 Benchmark System Functional Test Suite")
    print("=" * 50)
    print("Testing end-to-end workflows with real API calls...")
    print()
    
    # Create test suite
    suite = create_test_suite(specific_test)
    
    if suite.countTestCases() == 0:
        print("❌ No tests found to run")
        return False
    
    print(f"Running {suite.countTestCases()} functional tests...")
    print()
    
    # Configure test runner
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        stream=sys.stdout,
        buffer=True  # Capture stdout/stderr during tests
    )
    
    # Run tests with timing
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print results summary
    print()
    print("=" * 50)
    print("🎯 Test Results Summary")
    print(f"Total Tests: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Execution Time: {end_time - start_time:.1f}s")
    
    # Print failure details
    if result.failures:
        print()
        print("❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0]}")
    
    if result.errors:
        print()
        print("💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.split('\\n')[-2]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print()
        print("✅ All functional tests passed! The benchmark system is working correctly.")
    else:
        print()
        print("❌ Some functional tests failed. Check the output above for details.")
    
    return success


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run functional tests for benchmark system")
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose test output"
    )
    parser.add_argument(
        "--specific-test", "-t",
        help="Run specific test (format: TestClass.test_method)"
    )
    parser.add_argument(
        "--list-tests", "-l",
        action="store_true", 
        help="List all available tests"
    )
    
    args = parser.parse_args()
    
    if args.list_tests:
        print("Available functional tests:")
        print()
        
        test_classes = [TestCLIWorkflows, TestDomainExecution, TestResultValidation]
        for test_class in test_classes:
            print(f"{test_class.__name__}:")
            methods = [method for method in dir(test_class) if method.startswith('test_')]
            for method in sorted(methods):
                print(f"  {test_class.__name__}.{method}")
            print()
        
        return
    
    # Check if we're in the right directory
    if not os.path.exists("benchmark_runner.py"):
        print("❌ Error: Please run this script from the benchmark_tests directory")
        print("   Current directory:", os.getcwd())
        print("   Expected files: benchmark_runner.py, domains/, evaluator/")
        sys.exit(1)
    
    # Simple server check without blocking
    print("🔍 Pre-flight checks...")
    print("✅ Assuming LLM server is running on localhost:8004")
    print("   (If tests fail, ensure your LLM server is running)")
    print()
    
    # Run tests
    success = run_tests(args.verbose, args.specific_test)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()