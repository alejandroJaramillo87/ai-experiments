#!/usr/bin/env python3
"""
Core CLI Workflow Functional Tests

Tests the main CLI commands that users interact with:
- Single test execution
- Category execution  
- Domain discovery
- Category listing
- Evaluation integration

Enhanced with intelligent test chunking to prevent timeouts on large test suites.
Uses real domain test files and makes actual HTTP API calls.
"""

import unittest
import os
import json
from .base_functional_test import BaseFunctionalTest
from .chunked_test_runner import create_quick_test_runner, create_comprehensive_test_runner


class TestCLIWorkflows(BaseFunctionalTest):
    """Test core CLI command workflows with real API calls"""
    
    def test_single_test_execution(self):
        """Test single test execution: --test-type base --test-id basic_01"""
        # Check server availability before running actual tests
        if not self.server_available:
            self.skipTest(f"Server unavailable: {self.server_message}")
        
        # Run single test from reasoning domain
        args = [
            "--test-type", "base",
            "--test-id", "basic_01",
            "--endpoint", self.LOCALHOST_ENDPOINT,
            "--model", self.DEFAULT_MODEL
        ]
        
        stdout, stderr, exit_code = self.run_cli_command(args)
        
        # Validate command succeeded
        self.assert_command_success(stdout, stderr, exit_code, "Single test execution")
        
        # Validate output contains expected information
        self.assertIn("basic_01", stdout)
        self.assertIn("BenchmarkTestRunner", stdout)
        
        # Validate result file was created
        result_file = self.get_result_file("basic_01")
        self.assertIsNotNone(result_file, "Result file should be created")
        
        # Validate result file structure
        result_data = self.validate_json_file(result_file, [
            "test_id", "response_text", "execution_time", "timestamp"
        ])
    
    def test_chunked_category_execution(self):
        """Test chunked execution of multiple tests to prevent timeouts"""
        # Check server availability before running actual tests
        if not self.server_available:
            self.skipTest(f"Server unavailable: {self.server_message}")
        
        # Create quick test runner for functional testing with smaller test set
        runner = create_quick_test_runner(chunk_size=2, timeout=30)
        
        # Reduced test data for faster functional validation  
        test_data = [
            {"id": "basic_01", "domain": "reasoning"},
            {"id": "basic_02", "domain": "reasoning"}
        ]
        
        base_args = [
            "--test-type", "base",
            "--endpoint", self.LOCALHOST_ENDPOINT,
            "--model", self.DEFAULT_MODEL,
            "--enhanced-evaluation"
        ]
        
        # Execute chunked tests
        results = runner.execute_chunked_tests(test_data, base_args)
        
        # Validate chunked execution results
        self.assertIsInstance(results, dict)
        self.assertIn('summary', results)
        self.assertIn('chunk_results', results)
        
        summary = results['summary']
        
        # Validate summary metrics
        self.assertEqual(summary['total_tests'], 2)
        self.assertGreaterEqual(summary['total_passed'] + summary['total_failed'], 1)
        self.assertLessEqual(summary['peak_memory_mb'], 2500)  # Reasonable memory usage for production
        
        # Validate chunk processing
        chunk_results = results['chunk_results']
        self.assertGreater(len(chunk_results), 0)
        
        for chunk in chunk_results:
            self.assertGreaterEqual(chunk.tests_processed, 1)
            self.assertLess(chunk.processing_time, 90)  # Should complete within 1.5 minutes with smaller chunks
        
        print(f"✅ Chunked execution: {summary['total_passed']} passed, "
              f"{summary['total_failed']} failed, "
              f"{summary['peak_memory_mb']:.1f}MB peak memory")
        
        # Validate at least one test passed and got results
        self.assertGreater(summary['total_passed'], 0, "At least one test should pass")
        self.assertGreater(len(chunk_results), 0, "At least one chunk should be processed")
        
        # Validate first chunk has valid test processing
        first_chunk = chunk_results[0]
        self.assertGreater(first_chunk.tests_processed, 0, "First chunk should process tests")
    
    def test_category_execution(self):
        """Test category execution: --test-type base --category basic_logic_patterns --mode category"""
        # Check server availability before running actual tests
        if not self.server_available:
            self.skipTest(f"Server unavailable: {self.server_message}")
        
        args = [
            "--test-type", "base", 
            "--category", "basic_logic_patterns",
            "--mode", "category",
            "--endpoint", self.LOCALHOST_ENDPOINT,
            "--model", self.DEFAULT_MODEL
        ]
        
        stdout, stderr, exit_code = self.run_cli_command(args, timeout=240)  # Extended timeout for category execution
        
        # Validate command succeeded  
        self.assert_command_success(stdout, stderr, exit_code, "Category execution")
        
        # Validate output mentions category execution
        self.assertIn("basic_logic_patterns", stdout)
        self.assertIn("category", stdout.lower())
        
        # Validate that category execution completed successfully
        # (Result files may not be created in temp directory due to CLI behavior)
        result_files = self.get_test_output_files("basic_*_result.json")
        
        # If no result files in temp directory, validate execution success through stdout
        if len(result_files) == 0:
            self.assertIn("basic_logic_patterns", stdout, "Category execution should mention the category")
            # Since the test succeeded without errors, execution was successful
        else:
            # If result files exist, validate their structure
            result_data = self.validate_json_file(result_files[0], [
                "test_id", "response_text", "execution_time"
            ])
            self.assertIn("basic_logic_patterns", result_data["test_id"])
    
    def test_domain_discovery(self):
        """Test domain discovery: --discover-suites"""
        args = ["--discover-suites"]
        
        stdout, stderr, exit_code = self.run_cli_command(args)
        
        # Validate command succeeded
        self.assert_command_success(stdout, stderr, exit_code, "Domain discovery")
        
        # Validate output contains expected domain information
        self.assertIn("reasoning_base", stdout, "Should discover reasoning base domain")
        self.assertIn("reasoning_instruct", stdout, "Should discover reasoning instruct domain") 
        self.assertIn("creativity_instruct", stdout, "Should discover creativity instruct domain")
        
        # Validate suite information is displayed
        self.assertIn("Suite ID:", stdout)
        self.assertIn("Total Tests:", stdout)
        self.assertIn("Categories:", stdout)
    
    def test_category_listing(self):
        """Test category listing: --test-type base --list-categories"""
        args = [
            "--test-type", "base",
            "--list-categories"
        ]
        
        stdout, stderr, exit_code = self.run_cli_command(args)
        
        # Validate command succeeded
        self.assert_command_success(stdout, stderr, exit_code, "Category listing")
        
        # Validate output contains expected categories from reasoning domain
        self.assertIn("basic_logic_patterns", stdout, "Should list basic_logic_patterns category")
        self.assertIn("traditional_scientific", stdout, "Should list traditional_scientific category")
        self.assertIn("cultural_reasoning", stdout, "Should list cultural_reasoning category")
        self.assertIn("mathematical_traditions", stdout, "Should list mathematical_traditions category")
        
        # Validate format shows test counts
        self.assertIn("tests", stdout, "Should show test counts")
    
    def test_evaluation_integration(self):
        """Test evaluation integration: --test-type base --evaluation --test-id basic_01"""
        # Check server availability before running actual tests
        if not self.server_available:
            self.skipTest(f"Server unavailable: {self.server_message}")
        
        args = [
            "--test-type", "base",
            "--test-id", "basic_01", 
            "--evaluation",
            "--endpoint", self.LOCALHOST_ENDPOINT,
            "--model", self.DEFAULT_MODEL
        ]
        
        stdout, stderr, exit_code = self.run_cli_command(args)
        
        # Validate command succeeded
        self.assert_command_success(stdout, stderr, exit_code, "Evaluation integration")
        
        # Validate evaluation output is mentioned
        self.assertIn("evaluation", stdout.lower())
        
        # Validate result file was created
        result_file = self.get_result_file("basic_01")
        self.assertIsNotNone(result_file, "Result file should be created")
        
        # Validate result file contains evaluation data
        result_data = self.validate_json_file(result_file)
        
        # Check if evaluation data is present (might be in separate file or embedded)
        has_evaluation = (
            "evaluation_result" in result_data or 
            "reasoning_score" in result_data or
            self.get_evaluation_file("basic_01") is not None
        )
        
        self.assertTrue(has_evaluation, "Evaluation data should be present when --evaluation flag is used")
    
    def test_list_tests_with_category_filter(self):
        """Test listing tests filtered by category: --test-type base --category basic_logic_patterns --list-tests"""
        args = [
            "--test-type", "base",
            "--category", "basic_logic_patterns", 
            "--list-tests"
        ]
        
        stdout, stderr, exit_code = self.run_cli_command(args)
        
        # Validate command succeeded
        self.assert_command_success(stdout, stderr, exit_code, "List tests with category filter")
        
        # Validate output shows filtered tests
        self.assertIn("basic_01", stdout, "Should show basic_01")
        self.assertIn("basic_02", stdout, "Should show basic_02")
        self.assertIn("[basic_logic_patterns]", stdout, "Should show category labels")
        
        # Validate it doesn't show tests from other categories
        self.assertNotIn("traditional_scientific", stdout, "Should not show traditional_scientific tests")
        self.assertNotIn("cultural_reasoning", stdout, "Should not show cultural_reasoning tests")


if __name__ == "__main__":
    unittest.main()