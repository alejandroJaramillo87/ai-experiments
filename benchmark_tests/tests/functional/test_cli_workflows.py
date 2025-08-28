#!/usr/bin/env python3
"""
Core CLI Workflow Functional Tests

Tests the main CLI commands that users interact with:
- Single test execution
- Category execution
- Domain discovery
- Category listing
- Evaluation integration

Uses real domain test files and makes actual HTTP API calls.
"""

import unittest
import os
import json
from .base_functional_test import BaseFunctionalTest


class TestCLIWorkflows(BaseFunctionalTest):
    """Test core CLI command workflows with real API calls"""
    
    def test_single_test_execution(self):
        """Test single test execution: --test-type base --test-id text_continuation_01"""
        # Run single test from reasoning domain
        args = [
            "--test-type", "base",
            "--test-id", "text_continuation_01",
            "--endpoint", self.LOCALHOST_ENDPOINT,
            "--model", self.DEFAULT_MODEL
        ]
        
        stdout, stderr, exit_code = self.run_cli_command(args)
        
        # Validate command succeeded
        self.assert_command_success(stdout, stderr, exit_code, "Single test execution")
        
        # Validate output contains expected information
        self.assertIn("text_continuation_01", stdout)
        self.assertIn("BenchmarkTestRunner", stdout)
        
        # Validate result file was created
        result_file = self.get_result_file("text_continuation_01")
        self.assertIsNotNone(result_file, "Result file should be created")
        
        # Validate result file structure
        result_data = self.validate_json_file(result_file, [
            "test_id", "response_text", "execution_time", "timestamp"
        ])
        
        self.assertEqual(result_data["test_id"], "text_continuation_01")
        self.assertGreater(len(result_data["response_text"]), 0, "Response should not be empty")
        self.assertGreater(result_data["execution_time"], 0, "Execution time should be positive")
    
    def test_category_execution(self):
        """Test category execution: --test-type base --category text_continuation --mode category"""
        args = [
            "--test-type", "base", 
            "--category", "text_continuation",
            "--mode", "category",
            "--endpoint", self.LOCALHOST_ENDPOINT,
            "--model", self.DEFAULT_MODEL
        ]
        
        stdout, stderr, exit_code = self.run_cli_command(args)
        
        # Validate command succeeded  
        self.assert_command_success(stdout, stderr, exit_code, "Category execution")
        
        # Validate output mentions category execution
        self.assertIn("text_continuation", stdout)
        self.assertIn("category", stdout.lower())
        
        # Validate multiple result files were created (text_continuation has 10 tests)
        result_files = self.get_test_output_files("text_continuation_*_result.json")
        self.assertGreater(len(result_files), 1, "Multiple result files should be created for category")
        
        # Validate at least one result file has correct structure
        result_data = self.validate_json_file(result_files[0], [
            "test_id", "response_text", "execution_time"
        ])
        self.assertIn("text_continuation", result_data["test_id"])
    
    def test_domain_discovery(self):
        """Test domain discovery: --discover-suites"""
        args = ["--discover-suites"]
        
        stdout, stderr, exit_code = self.run_cli_command(args)
        
        # Validate command succeeded
        self.assert_command_success(stdout, stderr, exit_code, "Domain discovery")
        
        # Validate output contains expected domain information
        self.assertIn("reasoning_base", stdout, "Should discover reasoning base domain")
        self.assertIn("reasoning_instruct", stdout, "Should discover reasoning instruct domain") 
        self.assertIn("linux_instruct", stdout, "Should discover linux instruct domain")
        
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
        self.assertIn("text_continuation", stdout, "Should list text_continuation category")
        self.assertIn("pattern_completion", stdout, "Should list pattern_completion category")
        self.assertIn("style_mimicry", stdout, "Should list style_mimicry category")
        self.assertIn("context_coherence", stdout, "Should list context_coherence category")
        
        # Validate format shows test counts
        self.assertIn("tests", stdout, "Should show test counts")
    
    def test_evaluation_integration(self):
        """Test evaluation integration: --test-type base --evaluation --test-id text_continuation_01"""
        args = [
            "--test-type", "base",
            "--test-id", "text_continuation_01", 
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
        result_file = self.get_result_file("text_continuation_01")
        self.assertIsNotNone(result_file, "Result file should be created")
        
        # Validate result file contains evaluation data
        result_data = self.validate_json_file(result_file)
        
        # Check if evaluation data is present (might be in separate file or embedded)
        has_evaluation = (
            "evaluation_result" in result_data or 
            "reasoning_score" in result_data or
            self.get_evaluation_file("text_continuation_01") is not None
        )
        
        self.assertTrue(has_evaluation, "Evaluation data should be present when --evaluation flag is used")
    
    def test_list_tests_with_category_filter(self):
        """Test listing tests filtered by category: --test-type base --category text_continuation --list-tests"""
        args = [
            "--test-type", "base",
            "--category", "text_continuation", 
            "--list-tests"
        ]
        
        stdout, stderr, exit_code = self.run_cli_command(args)
        
        # Validate command succeeded
        self.assert_command_success(stdout, stderr, exit_code, "List tests with category filter")
        
        # Validate output shows filtered tests
        self.assertIn("text_continuation_01", stdout, "Should show text_continuation_01")
        self.assertIn("text_continuation_02", stdout, "Should show text_continuation_02")
        self.assertIn("[text_continuation]", stdout, "Should show category labels")
        
        # Validate it doesn't show tests from other categories
        self.assertNotIn("pattern_completion", stdout, "Should not show pattern_completion tests")
        self.assertNotIn("style_mimicry", stdout, "Should not show style_mimicry tests")


if __name__ == "__main__":
    unittest.main()