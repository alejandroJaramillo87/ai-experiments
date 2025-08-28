#!/usr/bin/env python3
"""
Multi-Domain Execution Functional Tests

Tests execution across different domains and model types:
- Reasoning domain (base models)
- Linux domain (instruct models) 
- Cross-domain execution
- Concurrent execution modes

Uses real domain test files from the restructured domains/ directory.
"""

import unittest
import os
import time
from .base_functional_test import BaseFunctionalTest


class TestDomainExecution(BaseFunctionalTest):
    """Test multi-domain execution with real domain test files"""
    
    def test_reasoning_domain_execution(self):
        """Test reasoning domain execution using real domains/reasoning/base_models/ tests"""
        # Execute a reasoning test to verify domain loading
        args = [
            "--test-type", "base",
            "--test-id", "text_continuation_01",
            "--endpoint", self.LOCALHOST_ENDPOINT,
            "--model", self.DEFAULT_MODEL
        ]
        
        stdout, stderr, exit_code = self.run_cli_command(args)
        
        # Validate command succeeded
        self.assert_command_success(stdout, stderr, exit_code, "Reasoning domain execution")
        
        # Validate it loaded from reasoning domain
        self.assertIn("text_continuation_01", stdout)
        
        # Validate result file created
        result_file = self.get_result_file("text_continuation_01") 
        self.assertIsNotNone(result_file)
        
        # Validate result contains reasoning-appropriate content
        result_data = self.validate_json_file(result_file, ["test_id", "response_text"])
        self.assertEqual(result_data["test_id"], "text_continuation_01")
        
        # Reasoning tests should generate text continuations
        self.assertGreater(len(result_data["response_text"]), 10, "Reasoning response should have substantial content")
    
    def test_linux_domain_execution(self):
        """Test linux domain execution using real domains/linux/instruct_models/ tests"""
        # Execute a linux instruct test
        args = [
            "--test-type", "instruct",
            "--test-id", "linux_test_01", 
            "--endpoint", self.LOCALHOST_ENDPOINT,
            "--model", self.DEFAULT_MODEL
        ]
        
        stdout, stderr, exit_code = self.run_cli_command(args)
        
        # Validate command succeeded
        self.assert_command_success(stdout, stderr, exit_code, "Linux domain execution")
        
        # Validate it executed linux test
        self.assertIn("linux_test_01", stdout)
        
        # Validate result file created
        result_file = self.get_result_file("linux_test_01")
        self.assertIsNotNone(result_file)
        
        # Validate result structure
        result_data = self.validate_json_file(result_file, ["test_id", "response_text"])
        self.assertEqual(result_data["test_id"], "linux_test_01")
        
        # Linux tests should generate technical responses
        response = result_data["response_text"].lower()
        # Should contain technical/linux-related content
        has_technical_content = any(term in response for term in [
            "system", "monitor", "script", "command", "process", "cpu", "memory", "disk"
        ])
        self.assertTrue(has_technical_content, "Linux response should contain technical content")
    
    def test_cross_domain_execution(self):
        """Test execution across both reasoning and linux domains sequentially"""
        # First execute reasoning test
        reasoning_args = [
            "--test-type", "base",
            "--test-id", "text_continuation_01",
            "--endpoint", self.LOCALHOST_ENDPOINT,
            "--model", self.DEFAULT_MODEL
        ]
        
        stdout1, stderr1, exit_code1 = self.run_cli_command(reasoning_args)
        self.assert_command_success(stdout1, stderr1, exit_code1, "Cross-domain reasoning execution")
        
        # Then execute linux test (different output dir to avoid conflicts)
        linux_args = [
            "--test-type", "instruct", 
            "--test-id", "linux_test_01",
            "--endpoint", self.LOCALHOST_ENDPOINT,
            "--model", self.DEFAULT_MODEL
        ]
        
        stdout2, stderr2, exit_code2 = self.run_cli_command(linux_args)
        self.assert_command_success(stdout2, stderr2, exit_code2, "Cross-domain linux execution")
        
        # Validate both result files exist
        reasoning_result = self.get_result_file("text_continuation_01")
        linux_result = self.get_result_file("linux_test_01") 
        
        self.assertIsNotNone(reasoning_result, "Reasoning result should exist")
        self.assertIsNotNone(linux_result, "Linux result should exist")
        
        # Validate both contain different types of content
        reasoning_data = self.validate_json_file(reasoning_result)
        linux_data = self.validate_json_file(linux_result)
        
        self.assertNotEqual(
            reasoning_data["response_text"], 
            linux_data["response_text"], 
            "Cross-domain tests should generate different responses"
        )
    
    def test_concurrent_execution(self):
        """Test concurrent execution: --mode concurrent --workers 2"""
        # Use multiple text continuation tests for concurrent execution
        args = [
            "--test-type", "base",
            "--category", "text_continuation", 
            "--mode", "concurrent",
            "--workers", "2",
            "--endpoint", self.LOCALHOST_ENDPOINT,
            "--model", self.DEFAULT_MODEL
        ]
        
        # Measure execution time
        start_time = time.time()
        stdout, stderr, exit_code = self.run_cli_command(args, timeout=300)  # Longer timeout for concurrent
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Validate command succeeded
        self.assert_command_success(stdout, stderr, exit_code, "Concurrent execution")
        
        # Validate concurrent execution mentions
        self.assertIn("concurrent", stdout.lower())
        self.assertIn("workers", stdout.lower())
        
        # Validate multiple result files were created
        result_files = self.get_test_output_files("text_continuation_*_result.json")
        self.assertGreater(len(result_files), 1, "Multiple result files should be created in concurrent mode")
        
        # Validate all result files have valid structure
        for result_file in result_files[:3]:  # Check first 3 files
            result_data = self.validate_json_file(result_file, ["test_id", "response_text", "execution_time"])
            self.assertIn("text_continuation", result_data["test_id"])
            self.assertGreater(len(result_data["response_text"]), 0)
        
        # Concurrent execution should complete within reasonable time
        # (This is a basic sanity check, not a rigorous performance test)
        self.assertLess(execution_time, 180, "Concurrent execution should complete within 3 minutes")
    
    def test_sequential_vs_concurrent_comparison(self):
        """Compare sequential vs concurrent execution to verify concurrent is working"""
        # Test with a small set of pattern completion tests (faster than text continuation)
        test_args_base = [
            "--test-type", "base",
            "--category", "pattern_completion",
            "--endpoint", self.LOCALHOST_ENDPOINT, 
            "--model", self.DEFAULT_MODEL
        ]
        
        # First run sequential
        sequential_args = test_args_base + ["--mode", "sequential"]
        start_sequential = time.time()
        stdout1, stderr1, exit_code1 = self.run_cli_command(sequential_args, timeout=180)
        sequential_time = time.time() - start_sequential
        
        self.assert_command_success(stdout1, stderr1, exit_code1, "Sequential execution for comparison")
        
        # Clear output directory for second run
        import shutil
        shutil.rmtree(self.temp_output_dir, ignore_errors=True)
        os.makedirs(self.temp_output_dir, exist_ok=True)
        
        # Then run concurrent with 2 workers  
        concurrent_args = test_args_base + ["--mode", "concurrent", "--workers", "2"]
        start_concurrent = time.time()
        stdout2, stderr2, exit_code2 = self.run_cli_command(concurrent_args, timeout=180)
        concurrent_time = time.time() - start_concurrent
        
        self.assert_command_success(stdout2, stderr2, exit_code2, "Concurrent execution for comparison")
        
        # Both should produce similar number of results
        result_files = self.get_test_output_files("pattern_completion_*_result.json")
        self.assertGreater(len(result_files), 1, "Both modes should produce multiple results")
        
        # Log timing for manual verification (concurrent should be faster for multiple tests)
        print(f"\\nTiming comparison: Sequential={sequential_time:.1f}s, Concurrent={concurrent_time:.1f}s")
        
        # Basic sanity check - neither should take excessively long
        self.assertLess(sequential_time, 120, "Sequential execution should complete within 2 minutes")
        self.assertLess(concurrent_time, 120, "Concurrent execution should complete within 2 minutes")


if __name__ == "__main__":
    unittest.main()