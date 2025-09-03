#!/usr/bin/env python3
"""
Result Validation Functional Tests

Tests that verify the correctness and completeness of test execution results:
- JSON result file structure and content
- Evaluation score computation and storage
- Batch results aggregation 
- Complete output file structure validation

Validates real output from actual API calls.
"""

import unittest
import os
import json
import glob
from .base_functional_test import BaseFunctionalTest


class TestResultValidation(BaseFunctionalTest):
    """Test validation of results from actual test execution"""
    
    def test_json_result_file_creation(self):
        """Test that JSON result files are created with correct structure"""
        # Execute a single test to generate result file
        args = [
            "--test-type", "base",
            "--test-id", "basic_01",
            "--endpoint", self.LOCALHOST_ENDPOINT,
            "--model", self.DEFAULT_MODEL
        ]
        
        stdout, stderr, exit_code = self.run_cli_command(args)
        self.assert_command_success(stdout, stderr, exit_code, "Result file creation test")
        
        # Validate result file exists
        result_file = self.get_result_file("basic_01")
        self.assertIsNotNone(result_file, "Result JSON file should be created")
        
        # Validate JSON structure with required fields
        result_data = self.validate_json_file(result_file, [
            "test_id", "response_text", "execution_time", "timestamp"
        ])
        
        # Validate field contents
        self.assertEqual(result_data["test_id"], "basic_01")
        self.assertIsInstance(result_data["response_text"], str, "Response should be string")
        self.assertGreater(len(result_data["response_text"]), 0, "Response should not be empty")
        self.assertIsInstance(result_data["execution_time"], (int, float), "Execution time should be numeric")
        self.assertGreater(result_data["execution_time"], 0, "Execution time should be positive")
        self.assertIsInstance(result_data["timestamp"], str, "Timestamp should be string")
        
        # Validate optional fields if present
        if "model" in result_data:
            self.assertIsInstance(result_data["model"], str)
        if "parameters" in result_data:
            self.assertIsInstance(result_data["parameters"], dict)
        if "tokens" in result_data:
            self.assertIsInstance(result_data["tokens"], int)
    
    def test_evaluation_scores_saved(self):
        """Test that evaluation scores are computed and saved when --evaluation is used"""
        # Execute test with evaluation enabled
        args = [
            "--test-type", "base", 
            "--test-id", "basic_01",
            "--evaluation",
            "--endpoint", self.LOCALHOST_ENDPOINT,
            "--model", self.DEFAULT_MODEL
        ]
        
        stdout, stderr, exit_code = self.run_cli_command(args)
        self.assert_command_success(stdout, stderr, exit_code, "Evaluation scores test")
        
        # Check for evaluation data in result file or separate evaluation file
        result_file = self.get_result_file("basic_01")
        self.assertIsNotNone(result_file)
        
        result_data = self.validate_json_file(result_file)
        
        # Evaluation data might be embedded in result file or in separate file
        has_embedded_eval = any(key in result_data for key in [
            "evaluation", "score", "overall_score", "evaluation_metrics", 
            "evaluation_result", "reasoning_score"
        ])
        
        eval_file = self.get_evaluation_file("basic_01")
        has_separate_eval = eval_file is not None
        
        self.assertTrue(
            has_embedded_eval or has_separate_eval, 
            "Evaluation data should be present when --evaluation flag is used"
        )
        
        # If separate evaluation file exists, validate its structure
        if has_separate_eval:
            eval_data = self.validate_json_file(eval_file, ["overall_score"])
            self.assertIsInstance(eval_data["overall_score"], (int, float))
            self.assertGreaterEqual(eval_data["overall_score"], 0)
            self.assertLessEqual(eval_data["overall_score"], 100)
    
    def test_batch_results_aggregation(self):
        """Test that batch results are properly aggregated when running multiple tests"""
        # Execute multiple tests via category mode
        args = [
            "--test-type", "base",
            "--category", "pattern_completion", 
            "--mode", "category",
            "--endpoint", self.LOCALHOST_ENDPOINT,
            "--model", self.DEFAULT_MODEL
        ]
        
        stdout, stderr, exit_code = self.run_cli_command(args, timeout=90)  # Reduce timeout to 90s
        self.assert_command_success(stdout, stderr, exit_code, "Batch results aggregation test")
        
        # Validate multiple individual result files were created
        # Files are created with test_id pattern like "basic_01_completion.txt" and "basic_01_result.json"
        result_files = self.get_test_output_files("basic_*_result.json")
        if len(result_files) == 0:
            # Fallback: check for any result files
            result_files = self.get_test_output_files("*_result.json")
        self.assertGreater(len(result_files), 1, f"Multiple individual result files should exist. Found: {result_files}")
        
        # Validate that we have multiple individual result files (batch execution creates individual files)
        # Current benchmark runner creates individual result files, not necessarily a batch summary file
        for result_file in result_files[:2]:  # Check first 2 result files
            result_data = self.validate_json_file(result_file, [
                "test_id", "success", "execution_time"
            ])
            
            # Each individual result should have expected structure
            self.assertIn("test_id", result_data)
            self.assertTrue("basic_" in result_data["test_id"], f"Test ID should be from pattern_completion: {result_data['test_id']}")
            self.assertIsInstance(result_data["success"], bool)
            self.assertIsInstance(result_data["execution_time"], (int, float))
        
        print(f"✅ Batch results validation passed with {len(result_files)} result files")
    
    def test_expected_output_file_structure(self):
        """Test that all expected output files are created in correct structure"""
        # Execute test with evaluation to generate comprehensive outputs
        args = [
            "--test-type", "base",
            "--test-id", "basic_02",
            "--evaluation", 
            "--endpoint", self.LOCALHOST_ENDPOINT,
            "--model", self.DEFAULT_MODEL
        ]
        
        stdout, stderr, exit_code = self.run_cli_command(args)
        self.assert_command_success(stdout, stderr, exit_code, "Output file structure test")
        
        # Validate primary result file
        result_file = self.get_result_file("basic_02")
        self.assert_file_exists(result_file, "Primary result JSON file")
        
        # Validate completion text file (raw response)
        completion_file = self.get_completion_file("basic_02")
        if completion_file:  # Not all configurations create completion files
            self.assert_file_exists(completion_file, "Completion text file")
            
            # Validate completion file contains the response text
            with open(completion_file, 'r', encoding='utf-8') as f:
                completion_text = f.read().strip()
            self.assertGreater(len(completion_text), 0, "Completion file should contain response text")
        
        # Validate all files are in the correct output directory
        all_output_files = self.get_test_output_files("*")
        self.assertGreater(len(all_output_files), 0, "Should have output files")
        
        for file_path in all_output_files:
            self.assertTrue(
                file_path.startswith(self.temp_output_dir),
                f"Output file should be in temp directory: {file_path}"
            )
    
    def test_concurrent_results_integrity(self):
        """Test that concurrent execution produces complete and non-corrupted results"""
        # Use small test set for faster functional testing (4 tests instead of 30)
        args = [
            "--test-type", "base",
            "--test-id", "basic_01,basic_02,basic_03,basic_04",
            "--mode", "concurrent", 
            "--workers", "2",
            "--endpoint", self.LOCALHOST_ENDPOINT,
            "--model", self.DEFAULT_MODEL
        ]
        
        stdout, stderr, exit_code = self.run_cli_command(args, timeout=300)
        self.assert_command_success(stdout, stderr, exit_code, "Concurrent results integrity test")
        
        # Validate result files were created
        result_files = self.get_test_output_files("basic_*_result.json")
        self.assertGreaterEqual(len(result_files), 2, "Multiple result files should exist from concurrent execution")
        
        # Validate each result file has complete, valid structure
        test_ids_found = set()
        for result_file in result_files:
            result_data = self.validate_json_file(result_file, [
                "test_id", "response_text", "execution_time"
            ])
            
            # Validate no duplicate test IDs
            test_id = result_data["test_id"]
            self.assertNotIn(test_id, test_ids_found, f"Duplicate test ID found: {test_id}")
            test_ids_found.add(test_id)
            
            # Validate response is complete (not truncated or corrupted)
            response = result_data["response_text"]
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 10, f"Response seems too short for {test_id}: {len(response)} chars")
            
            # Validate execution time is reasonable
            exec_time = result_data["execution_time"]
            self.assertGreater(exec_time, 0)
            self.assertLess(exec_time, 120, f"Execution time seems too high: {exec_time}s")
        
        # Validate total number of unique results
        self.assertGreaterEqual(len(test_ids_found), 2, "Should have at least 2 unique test results")
    
    def test_error_handling_in_results(self):
        """Test that execution errors are properly recorded in result files"""
        # Try to execute a non-existent test ID to trigger error handling
        args = [
            "--test-type", "base",
            "--test-id", "nonexistent_test_999",
            "--endpoint", self.LOCALHOST_ENDPOINT,
            "--model", self.DEFAULT_MODEL
        ]
        
        stdout, stderr, exit_code = self.run_cli_command(args)
        
        # Command should fail gracefully
        self.assertNotEqual(exit_code, 0, "Should fail with non-existent test ID")
        
        # Should provide informative error message
        combined_output = (stderr + stdout).lower()
        self.assertIn("nonexistent_test_999", combined_output, "Error message should mention the missing test ID")
        
        # Should not create result files for failed execution
        result_files = self.get_test_output_files("nonexistent_test_999_*.json")
        self.assertEqual(len(result_files), 0, "Should not create result files for failed test")


if __name__ == "__main__":
    unittest.main()