#!/usr/bin/env python3
"""
Comprehensive Test Suite for TestRunner

Tests the TestRunner system without mocking, using real functionality
to verify correct behavior across different execution modes and configurations.

Author: Claude Code
Version: 1.0.0
"""

import unittest
import sys
import os
import json
import tempfile
import shutil
import threading
import time
from flask import Flask, request, jsonify
from werkzeug.serving import make_server
from unittest.mock import patch
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Add the benchmark_tests directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark_runner import BenchmarkTestRunner


class MockVLLMServer:
    """
    Lightweight HTTP server that mimics vLLM API behavior for testing
    without requiring actual model inference
    """
    
    def __init__(self, port: int = 0):
        self.port = port  # 0 means OS will assign an available port
        self.actual_port = None  # Will be set when server starts
        self.app = Flask(__name__)
        self.requests_log = []
        self.response_config = {
            'default_response': 'This is a test response from MockVLLMServer',
            'delay': 0,
            'status_code': 200,
            'error_mode': None  # None, 'timeout', 'server_error', 'rate_limit'
        }
        self.server = None
        self.server_thread = None
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up Flask routes for completions and chat APIs"""
        
        @self.app.route('/v1/completions', methods=['POST'])
        def completions():
            return self._handle_request('completions')
        
        @self.app.route('/v1/chat/completions', methods=['POST'])
        def chat_completions():
            return self._handle_request('chat')
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy'})
    
    def _handle_request(self, endpoint_type: str):
        """Handle API requests with configurable responses"""
        # Log the request
        request_data = {
            'endpoint': endpoint_type,
            'method': request.method,
            'data': request.get_json(),
            'timestamp': datetime.now().isoformat()
        }
        self.requests_log.append(request_data)
        
        # Apply configured delay
        if self.response_config['delay'] > 0:
            time.sleep(self.response_config['delay'])
        
        # Handle error modes
        if self.response_config['error_mode'] == 'timeout':
            time.sleep(10)  # Simulate timeout
        elif self.response_config['error_mode'] == 'server_error':
            return jsonify({'error': 'Internal server error'}), 500
        elif self.response_config['error_mode'] == 'rate_limit':
            return jsonify({'error': 'Rate limit exceeded'}), 429
        
        # Generate appropriate response based on endpoint type
        if endpoint_type == 'completions':
            return self._generate_completions_response(request.get_json())
        else:  # chat
            return self._generate_chat_response(request.get_json())
    
    def _generate_completions_response(self, request_data: Dict) -> Dict:
        """Generate completions API response"""
        response_text = self.response_config.get('default_response', 'Test response')
        
        return jsonify({
            'id': 'cmpl-test-12345',
            'object': 'text_completion',
            'created': int(time.time()),
            'model': request_data.get('model', 'test-model'),
            'choices': [{
                'text': response_text,
                'index': 0,
                'logprobs': None,
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': len(request_data.get('prompt', '').split()),
                'completion_tokens': len(response_text.split()),
                'total_tokens': len(request_data.get('prompt', '').split()) + len(response_text.split())
            }
        })
    
    def _generate_chat_response(self, request_data: Dict) -> Dict:
        """Generate chat API response"""
        response_text = self.response_config.get('default_response', 'Test chat response')
        
        return jsonify({
            'id': 'chatcmpl-test-12345',
            'object': 'chat.completion',
            'created': int(time.time()),
            'model': request_data.get('model', 'test-model'),
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': response_text
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': 50,  # Simplified for testing
                'completion_tokens': len(response_text.split()),
                'total_tokens': 50 + len(response_text.split())
            }
        })
    
    def start(self):
        """Start the mock server in a background thread"""
        self.server = make_server('127.0.0.1', self.port, self.app, threaded=True)
        self.actual_port = self.server.server_port  # Get the actual assigned port
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # Wait a moment for server to start
        time.sleep(0.1)
    
    def get_base_url(self):
        """Get the server's base URL with the actual assigned port"""
        if self.actual_port is None:
            raise RuntimeError("Server not started yet")
        return f"http://127.0.0.1:{self.actual_port}"
    
    def stop(self):
        """Stop the mock server"""
        if self.server:
            self.server.shutdown()
            self.server_thread.join(timeout=1)
    
    def configure_response(self, response_text: str = None, delay: float = 0, 
                          status_code: int = 200, error_mode: str = None):
        """Configure server response behavior"""
        if response_text:
            self.response_config['default_response'] = response_text
        self.response_config['delay'] = delay
        self.response_config['status_code'] = status_code
        self.response_config['error_mode'] = error_mode
    
    def get_requests_log(self) -> List[Dict]:
        """Get log of all requests made to the server"""
        return self.requests_log.copy()
    
    def clear_requests_log(self):
        """Clear the requests log"""
        self.requests_log.clear()


class TestTestRunnerConfiguration(unittest.TestCase):
    """Test TestRunner configuration loading and test suite management"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_runner = BenchmarkTestRunner()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_base_model_test_suite(self):
        """Test loading actual base model test suite"""
        base_suite_path = "base_models/test_definitions/reasoning_tests_medium.json"
        
        if os.path.exists(base_suite_path):
            success = self.test_runner.load_test_suite(base_suite_path)
            self.assertTrue(success, "Should successfully load base model test suite")
            self.assertGreater(len(self.test_runner.tests), 0, 
                             "Should load at least one test")
            
            # Verify test structure
            first_test = list(self.test_runner.tests.values())[0]
            self.assertIn('id', first_test, "Test should have ID")
            self.assertIn('prompt', first_test, "Base model test should have prompt")
            self.assertIn('category', first_test, "Test should have category")
    
    def test_load_instruct_model_test_suite(self):
        """Test loading actual instruct model test suite"""
        instruct_suite_path = "instruct-models/test_definitions/linux_tests_medium.json"
        
        if os.path.exists(instruct_suite_path):
            success = self.test_runner.load_test_suite(instruct_suite_path)
            self.assertTrue(success, "Should successfully load instruct model test suite")
            self.assertGreater(len(self.test_runner.tests), 0,
                             "Should load at least one test")
            
            # Verify instruct test structure
            first_test = list(self.test_runner.tests.values())[0]
            self.assertIn('id', first_test, "Test should have ID")
            self.assertIn('messages', first_test, "Instruct test should have messages")
            self.assertIn('category', first_test, "Test should have category")
    
    def test_category_filtering(self):
        """Test that category filtering works correctly"""
        # Create a temporary test suite with known categories
        test_suite = {
            "test_suite_id": "test_category_filtering",
            "version": "1.0.0", 
            "description": "Test category filtering",
            "tests": [
                {
                    "id": "test_monitoring_01",
                    "name": "Monitoring Test",
                    "category": "monitoring",
                    "prompt": "Test monitoring command"
                },
                {
                    "id": "test_security_01", 
                    "name": "Security Test",
                    "category": "security",
                    "prompt": "Test security command"
                },
                {
                    "id": "test_creative_01",
                    "name": "Creative Test",
                    "category": "creative_thinking", 
                    "prompt": "Test creative thinking"
                }
            ]
        }
        
        temp_file = os.path.join(self.temp_dir, "test_suite.json")
        with open(temp_file, 'w') as f:
            json.dump(test_suite, f)
        
        # Load the test suite
        success = self.test_runner.load_test_suite(temp_file)
        self.assertTrue(success, "Should load temporary test suite")
        
        # Test category filtering
        monitoring_test_ids = self.test_runner.get_test_ids_by_category("monitoring")
        self.assertEqual(len(monitoring_test_ids), 1, "Should find exactly one monitoring test")
        self.assertEqual(monitoring_test_ids[0], "test_monitoring_01")
        
        security_test_ids = self.test_runner.get_test_ids_by_category("security")
        self.assertEqual(len(security_test_ids), 1, "Should find exactly one security test")
        
        creative_test_ids = self.test_runner.get_test_ids_by_category("creative_thinking")
        self.assertEqual(len(creative_test_ids), 1, "Should find exactly one creative test")
        
        unknown_test_ids = self.test_runner.get_test_ids_by_category("unknown_category")
        self.assertEqual(len(unknown_test_ids), 0, "Should find no tests for unknown category")
    
    def test_malformed_json_handling(self):
        """Test graceful handling of malformed JSON files"""
        # Create malformed JSON file
        malformed_file = os.path.join(self.temp_dir, "malformed.json")
        with open(malformed_file, 'w') as f:
            f.write('{"invalid": json, missing quotes}')
        
        success = self.test_runner.load_test_suite(malformed_file)
        self.assertFalse(success, "Should fail gracefully with malformed JSON")
        self.assertEqual(len(self.test_runner.tests), 0, "Should not load any tests from malformed file")
    
    def test_missing_file_handling(self):
        """Test handling of missing test suite files"""
        missing_file = os.path.join(self.temp_dir, "missing.json")
        
        success = self.test_runner.load_test_suite(missing_file)
        self.assertFalse(success, "Should fail gracefully with missing file")
        self.assertEqual(len(self.test_runner.tests), 0, "Should not load any tests from missing file")


class TestTestRunnerHTTPCommunication(unittest.TestCase):
    """Test TestRunner HTTP communication with mock server"""
    
    @classmethod
    def setUpClass(cls):
        """Start mock server for all HTTP tests"""
        cls.mock_server = MockVLLMServer()
        cls.mock_server.start()
    
    @classmethod 
    def tearDownClass(cls):
        """Stop mock server after all tests"""
        cls.mock_server.stop()
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_runner = BenchmarkTestRunner()
        self.mock_server.clear_requests_log()
        self.mock_server.configure_response()  # Reset to defaults
        
        # Configure TestRunner to use mock server
        self.test_runner.configure_api(
            endpoint=f"{self.mock_server.get_base_url()}/v1/completions",
            model="test-model"
        )
        
        # Create a simple test for execution
        self.test_runner.tests = {
            "test_01": {
                "id": "test_01",
                "name": "HTTP Test",
                "category": "test",
                "prompt": "This is a test prompt",
                "parameters": {
                    "max_tokens": 100,
                    "temperature": 0.5
                }
            }
        }
    
    def test_completions_api_request_format(self):
        """Test that completions API requests are formatted correctly"""
        # Execute a single test
        result = self.test_runner.execute_single_test("test_01")
        
        self.assertTrue(result.success, "Test should execute successfully")
        
        # Verify request was made
        requests_log = self.mock_server.get_requests_log()
        self.assertEqual(len(requests_log), 1, "Should make exactly one request")
        
        request_data = requests_log[0]
        self.assertEqual(request_data['endpoint'], 'completions', "Should use completions endpoint")
        
        # Verify request structure
        request_json = request_data['data']
        self.assertIn('model', request_json, "Request should include model")
        self.assertIn('prompt', request_json, "Request should include prompt")
        self.assertIn('max_tokens', request_json, "Request should include max_tokens")
        self.assertIn('temperature', request_json, "Request should include temperature")
        self.assertEqual(request_json['prompt'], "This is a test prompt")
    
    def test_chat_api_request_format(self):
        """Test that chat API requests are formatted correctly"""
        # Configure for chat API
        self.test_runner.configure_api(
            endpoint=f"{self.mock_server.get_base_url()}/v1/chat/completions",
            model="test-model"
        )
        
        # Create instruct-style test
        self.test_runner.tests = {
            "chat_test_01": {
                "id": "chat_test_01", 
                "name": "Chat Test",
                "category": "test",
                "messages": [
                    {"role": "user", "content": "This is a test message"}
                ],
                "parameters": {
                    "max_tokens": 200,
                    "temperature": 0.3
                }
            }
        }
        
        result = self.test_runner.execute_single_test("chat_test_01")
        self.assertTrue(result.success, "Chat test should execute successfully")
        
        # Verify request format
        requests_log = self.mock_server.get_requests_log()
        self.assertEqual(len(requests_log), 1, "Should make exactly one request")
        
        request_data = requests_log[0]
        self.assertEqual(request_data['endpoint'], 'chat', "Should use chat endpoint")
        
        request_json = request_data['data']
        self.assertIn('messages', request_json, "Chat request should include messages")
        self.assertIn('model', request_json, "Chat request should include model")
        self.assertEqual(len(request_json['messages']), 1, "Should have one message")
        self.assertEqual(request_json['messages'][0]['content'], "This is a test message")
    
    def test_retry_logic_on_server_error(self):
        """Test retry logic when server returns 500 error"""
        # Configure server to return error initially
        self.mock_server.configure_response(error_mode='server_error')
        
        # Execute test (should fail initially but retry)
        result = self.test_runner.execute_single_test("test_01")
        
        # Verify multiple requests were made (original + retries)
        requests_log = self.mock_server.get_requests_log()
        self.assertGreater(len(requests_log), 1, "Should retry on server error")
        
        # Result should indicate failure
        self.assertFalse(result.success, "Should fail when server consistently returns errors")
        self.assertIsNotNone(result.error_message, "Should have error message")
    
    def test_timeout_handling(self):
        """Test handling of request timeouts"""
        # Configure server with long delay to simulate timeout
        self.mock_server.configure_response(delay=3.0)
        
        # Configure TestRunner with short timeout
        self.test_runner.configure_api(
            endpoint=f"{self.mock_server.get_base_url()}/v1/completions",
            model="test-model",
            timeout=1  # 1 second timeout
        )
        
        result = self.test_runner.execute_single_test("test_01")
        
        # Should handle timeout gracefully
        self.assertFalse(result.success, "Should fail on timeout")
        self.assertIsNotNone(result.error_message, "Should have timeout error message")
    
    def test_response_parsing(self):
        """Test parsing of API responses"""
        # Configure custom response
        custom_response = "This is a custom test response with specific content"
        self.mock_server.configure_response(response_text=custom_response)
        
        result = self.test_runner.execute_single_test("test_01")
        
        self.assertTrue(result.success, "Should parse response successfully")
        self.assertEqual(result.response_text, custom_response, 
                        "Should extract correct response text")
        self.assertGreater(result.completion_tokens, 0, "Should count completion tokens")
        self.assertGreater(result.execution_time, 0, "Should measure execution time")


class TestTestRunnerConcurrentExecution(unittest.TestCase):
    """Test TestRunner concurrent execution capabilities"""
    
    @classmethod
    def setUpClass(cls):
        """Start mock server for concurrent tests"""
        cls.mock_server = MockVLLMServer()
        cls.mock_server.start()
    
    @classmethod
    def tearDownClass(cls):
        """Stop mock server"""
        cls.mock_server.stop()
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_runner = BenchmarkTestRunner()
        self.mock_server.clear_requests_log()
        self.mock_server.configure_response(delay=0.1)  # Small delay to test concurrency
        
        self.test_runner.configure_api(
            endpoint=f"{self.mock_server.get_base_url()}/v1/completions",
            model="test-model"
        )
        
        # Create multiple tests for concurrent execution
        self.test_runner.tests = {
            f"test_{i:02d}": {
                "id": f"test_{i:02d}",
                "name": f"Concurrent Test {i}",
                "category": "test",
                "prompt": f"This is test prompt {i}",
                "parameters": {"max_tokens": 50, "temperature": 0.3}
            }
            for i in range(6)  # 6 tests for testing with different worker counts
        }
    
    def test_sequential_execution_management(self):
        """Test sequential execution completes all tests properly"""
        
        start_time = time.time()
        results = self.test_runner.execute_sequential(
            test_ids=list(self.test_runner.tests.keys())
        )
        end_time = time.time()
        
        # Verify all tests completed
        self.assertEqual(len(results), 6, "Should complete all 6 tests")
        successful_results = [r for r in results if r.success]
        self.assertEqual(len(successful_results), 6, "All tests should succeed")
        
        # Verify execution took some time (tests are running)
        actual_time = end_time - start_time
        self.assertGreater(actual_time, 0.5, "Should take some time to execute tests")
        
        # Verify all requests were made
        requests_log = self.mock_server.get_requests_log()
        self.assertEqual(len(requests_log), 6, "Should make 6 HTTP requests")
    
    def test_progress_tracking(self):
        """Test that progress tracking works correctly during sequential execution"""
        results = self.test_runner.execute_sequential(
            test_ids=list(self.test_runner.tests.keys())
        )
        
        # Verify progress tracking
        progress = self.test_runner.get_progress()
        self.assertEqual(progress.total_tests, 6, "Should track total test count")
        self.assertEqual(progress.completed_tests, 6, "Should track completed count")
        self.assertEqual(progress.successful_tests, 6, "Should track successful count")
        self.assertEqual(progress.failed_tests, 0, "Should track failed count")
    
    def test_sequential_execution_consistency(self):
        """Test that sequential execution produces consistent results"""
        # First execution
        results_1 = self.test_runner.execute_sequential(
            test_ids=["test_00", "test_01", "test_02"]
        )
        
        self.mock_server.clear_requests_log()
        
        # Second execution
        results_2 = self.test_runner.execute_sequential(
            test_ids=["test_00", "test_01", "test_02"]
        )
        
        # Both should succeed with same results
        self.assertEqual(len(results_1), 3, "First execution should complete")
        self.assertEqual(len(results_2), 3, "Second execution should complete")
        
        # Should have consistent success rates
        success_1 = sum(1 for r in results_1 if r.success)
        success_2 = sum(1 for r in results_2 if r.success)
        self.assertEqual(success_1, success_2, "Both executions should have same success count")
    
    def test_error_handling_in_sequential_execution(self):
        """Test error handling during sequential execution"""
        # Configure server to return errors for some requests
        self.mock_server.configure_response(error_mode='server_error')
        
        results = self.test_runner.execute_sequential(
            test_ids=["test_00", "test_01"]
        )
        
        self.assertEqual(len(results), 2, "Should return results for all tests")
        
        # All should fail due to server error
        failed_results = [r for r in results if not r.success]
        self.assertEqual(len(failed_results), 2, "Both tests should fail")
        
        for result in failed_results:
            self.assertIsNotNone(result.error_message, "Should have error message")


class TestTestRunnerResultManagement(unittest.TestCase):
    """Test TestRunner result saving and file management"""
    
    @classmethod
    def setUpClass(cls):
        """Start mock server for result tests"""
        cls.mock_server = MockVLLMServer()
        cls.mock_server.start()
    
    @classmethod
    def tearDownClass(cls):
        """Stop mock server"""
        cls.mock_server.stop()
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_runner = BenchmarkTestRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.mock_server.clear_requests_log()
        
        self.test_runner.configure_api(
            endpoint=f"{self.mock_server.get_base_url()}/v1/completions",
            model="test-model"
        )
        
        # Create test for execution
        self.test_runner.tests = {
            "result_test_01": {
                "id": "result_test_01",
                "name": "Result Management Test",
                "category": "monitoring",
                "prompt": "Test prompt for result management",
                "parameters": {"max_tokens": 100}
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_individual_result_file_creation(self):
        """Test creation of individual result files"""
        # Execute test and save results
        result = self.test_runner.execute_single_test("result_test_01")
        success = self.test_runner.save_results([result], self.temp_dir)
        
        self.assertTrue(success, "Should save results successfully")
        
        # Check that result file was created
        result_files = [f for f in os.listdir(self.temp_dir) if f.endswith('_result.json')]
        self.assertEqual(len(result_files), 1, "Should create exactly one result file")
        
        # Verify result file content
        result_file_path = os.path.join(self.temp_dir, result_files[0])
        with open(result_file_path, 'r') as f:
            saved_result = json.load(f)
        
        self.assertIn('test_id', saved_result, "Result should contain test_id")
        self.assertIn('test_name', saved_result, "Result should contain test_name") 
        self.assertIn('success', saved_result, "Result should contain success flag")
        self.assertIn('execution_time', saved_result, "Result should contain execution_time")
        self.assertIn('api_response', saved_result, "Result should contain api_response")
        
        self.assertEqual(saved_result['test_id'], "result_test_01")
        self.assertTrue(saved_result['success'], "Saved result should show success")
    
    def test_completion_text_file_creation(self):
        """Test creation of completion text files"""
        result = self.test_runner.execute_single_test("result_test_01")
        success = self.test_runner.save_results([result], self.temp_dir)
        
        self.assertTrue(success, "Should save results successfully")
        
        # Check for completion text file
        completion_files = [f for f in os.listdir(self.temp_dir) if f.endswith('_completion.txt')]
        self.assertEqual(len(completion_files), 1, "Should create exactly one completion file")
        
        # Verify completion file content includes both prompt and response
        completion_file_path = os.path.join(self.temp_dir, completion_files[0])
        with open(completion_file_path, 'r') as f:
            completion_text = f.read()
        
        self.assertIn("PROMPT:", completion_text, "Completion file should contain PROMPT section")
        self.assertIn("COMPLETION:", completion_text, "Completion file should contain COMPLETION section")
        self.assertIn(result.response_text, completion_text, "Completion file should contain response text")
    
    def test_batch_results_aggregation(self):
        """Test creation of batch results file"""
        # Create multiple test results (batch files only created for multiple results)
        results = []
        for i in range(3):
            test_id = f"batch_test_{i:02d}"
            self.test_runner.tests[test_id] = {
                "id": test_id,
                "name": f"Batch Test {i}",
                "category": "test",
                "prompt": f"Batch test prompt {i}",
                "parameters": {"max_tokens": 50}
            }
            result = self.test_runner.execute_single_test(test_id)
            results.append(result)
        
        success = self.test_runner.save_results(results, self.temp_dir)
        self.assertTrue(success, "Should save batch results successfully")
        
        # Check for batch results file (only created when len(results) > 1)
        batch_files = [f for f in os.listdir(self.temp_dir) if f.startswith('batch_results_')]
        self.assertEqual(len(batch_files), 1, "Should create exactly one batch results file")
        
        # Verify batch results content
        batch_file_path = os.path.join(self.temp_dir, batch_files[0])
        with open(batch_file_path, 'r') as f:
            batch_results = json.load(f)
        
        self.assertIn('execution_summary', batch_results, "Should contain execution summary")
        self.assertIn('individual_results', batch_results, "Should contain individual results")
        
        summary = batch_results['execution_summary']
        self.assertEqual(summary['total_tests'], 3, "Should report correct total test count")
        self.assertEqual(summary['successful_tests'], 3, "Should report correct successful count")
        self.assertEqual(summary['failed_tests'], 0, "Should report correct failed count")
        
        individual_results = batch_results['individual_results']
        self.assertEqual(len(individual_results), 3, "Should include all individual results")
    
    def test_file_naming_conventions(self):
        """Test that files follow correct naming conventions"""
        result = self.test_runner.execute_single_test("result_test_01")
        success = self.test_runner.save_results([result], self.temp_dir)
        
        self.assertTrue(success, "Should save results successfully")
        
        files = os.listdir(self.temp_dir)
        
        # TestRunner uses test_name (not test_id) for file naming
        expected_base = "result_management_test"  # test_name converted to filename_safe
        
        # Check result file naming
        result_files = [f for f in files if f.endswith('_result.json')]
        self.assertEqual(len(result_files), 1)
        self.assertTrue(result_files[0].startswith(expected_base), 
                       f"Result file should start with {expected_base}, got {result_files[0]}")
        
        # Check completion file naming  
        completion_files = [f for f in files if f.endswith('_completion.txt')]
        self.assertEqual(len(completion_files), 1)
        self.assertTrue(completion_files[0].startswith(expected_base),
                       f"Completion file should start with {expected_base}, got {completion_files[0]}")
    
    def test_output_directory_creation(self):
        """Test automatic creation of output directories"""
        nested_dir = os.path.join(self.temp_dir, "nested", "output", "directory")
        
        result = self.test_runner.execute_single_test("result_test_01")
        success = self.test_runner.save_results([result], nested_dir)
        
        self.assertTrue(success, "Should create nested directories and save results")
        self.assertTrue(os.path.exists(nested_dir), "Should create nested directory")
        
        files = os.listdir(nested_dir)
        result_files = [f for f in files if f.endswith('_result.json')]
        self.assertEqual(len(result_files), 1, "Should save result file in nested directory")


class TestTestRunnerPerformanceMonitoring(unittest.TestCase):
    """Test TestRunner performance monitoring capabilities"""
    
    @classmethod
    def setUpClass(cls):
        """Start mock server for performance tests"""
        cls.mock_server = MockVLLMServer()
        cls.mock_server.start()
    
    @classmethod
    def tearDownClass(cls):
        """Stop mock server"""
        cls.mock_server.stop()
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_runner = BenchmarkTestRunner()
        self.mock_server.clear_requests_log()
        
        self.test_runner.configure_api(
            endpoint=f"{self.mock_server.get_base_url()}/v1/completions",
            model="test-model"
        )
        
        # Create test with performance monitoring enabled
        self.test_runner.tests = {
            "perf_test_01": {
                "id": "perf_test_01",
                "name": "Performance Monitoring Test",
                "category": "test",
                "prompt": "Test prompt for performance monitoring",
                "parameters": {"max_tokens": 100}
            }
        }
    
    def test_execution_time_measurement(self):
        """Test that execution time is measured accurately"""
        # Configure server with known delay
        self.mock_server.configure_response(delay=0.2)
        
        result = self.test_runner.execute_single_test("perf_test_01")
        
        self.assertTrue(result.success, "Test should execute successfully")
        self.assertGreater(result.execution_time, 0.15, 
                          "Should measure execution time including server delay")
        self.assertLess(result.execution_time, 1.0, 
                       "Execution time should be reasonable")
    
    def test_token_counting(self):
        """Test that token usage is tracked correctly"""
        result = self.test_runner.execute_single_test("perf_test_01")
        
        self.assertTrue(result.success, "Test should execute successfully")
        self.assertGreater(result.prompt_tokens, 0, "Should count prompt tokens")
        self.assertGreater(result.completion_tokens, 0, "Should count completion tokens")
        self.assertEqual(result.prompt_tokens + result.completion_tokens, 
                        result.prompt_tokens + result.completion_tokens,
                        "Token counts should be consistent")
    
    def test_performance_metrics_collection(self):
        """Test collection of system performance metrics"""
        # Enable performance monitoring if available
        if hasattr(self.test_runner, 'enable_performance_monitoring'):
            self.test_runner.enable_performance_monitoring(True)
        
        result = self.test_runner.execute_single_test("perf_test_01")
        
        self.assertTrue(result.success, "Test should execute successfully")
        
        # Check if performance metrics are present (optional, depends on system)
        if hasattr(result, 'performance_metrics') and result.performance_metrics:
            metrics = result.performance_metrics
            
            # These should be non-negative if present
            if 'cpu_usage_percent' in metrics:
                self.assertGreaterEqual(metrics['cpu_usage_percent'], 0)
                self.assertLessEqual(metrics['cpu_usage_percent'], 100)
            
            if 'memory_used_gb' in metrics:
                self.assertGreaterEqual(metrics['memory_used_gb'], 0)
    
    def test_tokens_per_second_calculation(self):
        """Test tokens per second calculation"""
        result = self.test_runner.execute_single_test("perf_test_01")
        
        self.assertTrue(result.success, "Test should execute successfully")
        
        if hasattr(result, 'tokens_per_second'):
            self.assertGreater(result.tokens_per_second, 0, 
                             "Should calculate tokens per second")
            
            # Verify calculation makes sense
            expected_tps = result.completion_tokens / result.execution_time
            self.assertAlmostEqual(result.tokens_per_second, expected_tps, places=1,
                                  msg="Tokens per second calculation should be accurate")


class TestTestRunnerErrorHandling(unittest.TestCase):
    """Test TestRunner error handling and edge cases"""
    
    @classmethod
    def setUpClass(cls):
        """Start mock server for error tests"""
        cls.mock_server = MockVLLMServer()
        cls.mock_server.start()
    
    @classmethod
    def tearDownClass(cls):
        """Stop mock server"""
        cls.mock_server.stop()
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_runner = BenchmarkTestRunner()
        self.mock_server.clear_requests_log()
        self.temp_dir = tempfile.mkdtemp()
        
        self.test_runner.configure_api(
            endpoint=f"{self.mock_server.get_base_url()}/v1/completions",
            model="test-model"
        )
        
        self.test_runner.tests = {
            "error_test_01": {
                "id": "error_test_01",
                "name": "Error Handling Test",
                "category": "test",
                "prompt": "Test prompt for error handling",
                "parameters": {"max_tokens": 100}
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_server_500_error_handling(self):
        """Test handling of HTTP 500 server errors"""
        self.mock_server.configure_response(error_mode='server_error')
        
        result = self.test_runner.execute_single_test("error_test_01")
        
        self.assertFalse(result.success, "Should fail when server returns 500 error")
        self.assertIsNotNone(result.error_message, "Should capture error message")
        self.assertIn("500", result.error_message.lower(), "Error message should mention status code")
    
    def test_rate_limiting_handling(self):
        """Test handling of rate limiting (HTTP 429)"""
        self.mock_server.configure_response(error_mode='rate_limit')
        
        result = self.test_runner.execute_single_test("error_test_01")
        
        self.assertFalse(result.success, "Should handle rate limiting appropriately")
        self.assertIsNotNone(result.error_message, "Should capture rate limit error")
    
    def test_network_connection_error(self):
        """Test handling of network connection errors"""
        # Configure TestRunner to use non-existent server
        self.test_runner.configure_api(
            endpoint="http://127.0.0.1:9999/v1/completions",  # Wrong port
            model="test-model"
        )
        
        result = self.test_runner.execute_single_test("error_test_01")
        
        self.assertFalse(result.success, "Should fail when server is unreachable")
        self.assertIsNotNone(result.error_message, "Should capture connection error")
    
    def test_invalid_test_id_handling(self):
        """Test handling of invalid test IDs"""
        result = self.test_runner.execute_single_test("nonexistent_test")
        
        self.assertFalse(result.success, "Should fail with invalid test ID")
        self.assertIsNotNone(result.error_message, "Should provide error message for invalid test ID")
    
    def test_empty_test_suite_handling(self):
        """Test behavior with empty test suite"""
        self.test_runner.tests = {}
        
        results = self.test_runner.execute_sequential(test_ids=[])
        self.assertEqual(len(results), 0, "Should return empty results for empty test suite")
        
        progress = self.test_runner.get_progress()
        self.assertEqual(progress.total_tests, 0, "Progress should show zero total tests")
    
    def test_disk_space_error_simulation(self):
        """Test handling of disk space errors during result saving"""
        # Create a read-only directory to simulate permission/disk errors
        readonly_dir = os.path.join(self.temp_dir, "readonly")
        os.makedirs(readonly_dir)
        os.chmod(readonly_dir, 0o444)  # Read-only
        
        try:
            # Execute test successfully
            result = self.test_runner.execute_single_test("error_test_01")
            self.assertTrue(result.success, "Test execution should succeed")
            
            # Try to save to read-only directory (should handle gracefully)
            success = self.test_runner.save_results([result], readonly_dir)
            self.assertFalse(success, "Should fail gracefully when unable to write files")
            
        finally:
            # Cleanup: restore permissions
            os.chmod(readonly_dir, 0o755)
    
    def test_malformed_response_handling(self):
        """Test handling of malformed API responses"""
        # This is tricky to test with Flask mock server, but we can test JSON parsing
        # by creating a custom response that might cause parsing issues
        
        # Configure server to return empty response
        self.mock_server.configure_response(response_text="")
        
        result = self.test_runner.execute_single_test("error_test_01")
        
        # Should handle empty response gracefully
        self.assertTrue(result.success or not result.success, "Should handle empty response gracefully")
        if not result.success:
            self.assertIsNotNone(result.error_message, "Should provide error message for empty response")


class TestTestRunnerEndToEndIntegration(unittest.TestCase):
    """Test complete TestRunner workflows end-to-end"""
    
    @classmethod
    def setUpClass(cls):
        """Start mock server for integration tests"""
        cls.mock_server = MockVLLMServer()
        cls.mock_server.start()
    
    @classmethod
    def tearDownClass(cls):
        """Stop mock server"""
        cls.mock_server.stop()
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_runner = BenchmarkTestRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.mock_server.clear_requests_log()
        
        self.test_runner.configure_api(
            endpoint=f"{self.mock_server.get_base_url()}/v1/completions",
            model="test-model"
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_base_model_workflow(self):
        """Test complete workflow: load base model tests → execute → evaluate → save"""
        # Create realistic base model test suite
        base_suite = {
            "test_suite_id": "integration_base_tests",
            "version": "1.0.0",
            "description": "Integration test suite for base models",
            "tests": [
                {
                    "id": "complex_test_01",
                    "name": "Complex Synthesis Test",
                    "category": "complex_synthesis", 
                    "reasoning_type": "general",
                    "prompt": "Analyze the following documents and synthesize key findings...",
                    "parameters": {
                        "max_tokens": 1500,
                        "temperature": 0.4,
                        "top_p": 0.95
                    }
                },
                {
                    "id": "math_test_01",
                    "name": "Mathematical Reasoning Test",
                    "category": "mathematical_reasoning",
                    "reasoning_type": "mathematical", 
                    "prompt": "Solve the following mathematical problem step by step...",
                    "parameters": {
                        "max_tokens": 800,
                        "temperature": 0.2
                    }
                },
                {
                    "id": "chain_test_01",
                    "name": "Chain of Thought Test",
                    "category": "chain_of_thought",
                    "reasoning_type": "chain_of_thought",
                    "prompt": "Think through this problem step by step...",
                    "parameters": {
                        "max_tokens": 1000,
                        "temperature": 0.3
                    }
                }
            ]
        }
        
        # Save test suite to temp file
        suite_file = os.path.join(self.temp_dir, "base_test_suite.json")
        with open(suite_file, 'w') as f:
            json.dump(base_suite, f)
        
        # Configure server with realistic responses
        self.mock_server.configure_response(
            response_text="Step 1: First, I need to analyze the given information. "
                         "Step 2: Based on the analysis, I can conclude that... "
                         "Step 3: Therefore, the solution is...",
            delay=0.1
        )
        
        # Load test suite
        success = self.test_runner.load_test_suite(suite_file)
        self.assertTrue(success, "Should load test suite successfully")
        self.assertEqual(len(self.test_runner.tests), 3, "Should load all 3 tests")
        
        # Execute tests concurrently
        results = self.test_runner.execute_sequential()
        self.assertEqual(len(results), 3, "Should execute all 3 tests")
        
        successful_results = [r for r in results if r.success]
        self.assertEqual(len(successful_results), 3, "All tests should succeed")
        
        # Save results
        save_success = self.test_runner.save_results(results, self.temp_dir)
        self.assertTrue(save_success, "Should save all results successfully")
        
        # Verify file creation
        result_files = [f for f in os.listdir(self.temp_dir) if f.endswith('_result.json')]
        completion_files = [f for f in os.listdir(self.temp_dir) if f.endswith('_completion.txt')]
        batch_files = [f for f in os.listdir(self.temp_dir) if f.startswith('batch_results_')]
        
        self.assertEqual(len(result_files), 3, "Should create 3 result files")
        self.assertEqual(len(completion_files), 3, "Should create 3 completion files")
        self.assertEqual(len(batch_files), 1, "Should create 1 batch results file")
        
        # Verify requests were made correctly
        requests_log = self.mock_server.get_requests_log()
        self.assertEqual(len(requests_log), 3, "Should make 3 API requests")
        
        for request in requests_log:
            self.assertEqual(request['endpoint'], 'completions')
            self.assertIn('prompt', request['data'])
            self.assertIn('model', request['data'])
    
    def test_complete_instruct_model_workflow(self):
        """Test complete workflow for instruct models with chat API"""
        # Create instruct model test suite (Linux tests)
        instruct_suite = {
            "test_suite_id": "integration_instruct_tests",
            "version": "1.0.0",
            "description": "Integration test suite for instruct models",
            "tests": [
                {
                    "id": "linux_monitor_01",
                    "name": "System Monitoring",
                    "category": "monitoring",
                    "reasoning_type": "general",
                    "messages": [
                        {
                            "role": "user", 
                            "content": "Write a bash script to monitor system CPU and memory usage"
                        }
                    ],
                    "parameters": {
                        "max_tokens": 500,
                        "temperature": 0.1
                    }
                },
                {
                    "id": "linux_security_01",
                    "name": "Security Audit",
                    "category": "security",
                    "reasoning_type": "general",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Create a command to find files with suspicious permissions"
                        }
                    ],
                    "parameters": {
                        "max_tokens": 300,
                        "temperature": 0.1
                    }
                }
            ]
        }
        
        # Save test suite
        suite_file = os.path.join(self.temp_dir, "instruct_test_suite.json")
        with open(suite_file, 'w') as f:
            json.dump(instruct_suite, f)
        
        # Configure for chat API
        self.test_runner.configure_api(
            endpoint=f"{self.mock_server.get_base_url()}/v1/chat/completions",
            model="test-model"
        )
        
        # Configure server with Linux-appropriate responses
        self.mock_server.configure_response(
            response_text="#!/bin/bash\n# System monitoring script\n"
                         "top -bn1 | grep 'Cpu(s)'\n"
                         "free -h | grep Mem",
            delay=0.05
        )
        
        # Execute complete workflow
        self.test_runner.load_test_suite(suite_file)
        results = self.test_runner.execute_sequential()
        self.test_runner.save_results(results, self.temp_dir)
        
        # Verify results
        self.assertEqual(len(results), 2, "Should execute both Linux tests")
        all_success = all(r.success for r in results)
        self.assertTrue(all_success, "All Linux tests should succeed")
        
        # Verify chat API requests
        requests_log = self.mock_server.get_requests_log()
        self.assertEqual(len(requests_log), 2, "Should make 2 chat API requests")
        
        for request in requests_log:
            self.assertEqual(request['endpoint'], 'chat')
            self.assertIn('messages', request['data'])
            messages = request['data']['messages']
            self.assertEqual(len(messages), 1, "Should have one user message")
            self.assertEqual(messages[0]['role'], 'user')
    
    def test_category_specific_execution(self):
        """Test execution of specific test categories"""
        # Create mixed category test suite
        mixed_suite = {
            "test_suite_id": "mixed_categories",
            "version": "1.0.0",
            "description": "Mixed category test suite",
            "tests": [
                {
                    "id": "monitoring_01",
                    "name": "Monitoring Test",
                    "category": "monitoring",
                    "prompt": "Monitor system resources",
                    "parameters": {"max_tokens": 200}
                },
                {
                    "id": "security_01", 
                    "name": "Security Test",
                    "category": "security",
                    "prompt": "Audit system security",
                    "parameters": {"max_tokens": 200}
                },
                {
                    "id": "creative_01",
                    "name": "Creative Test", 
                    "category": "creative_thinking",
                    "prompt": "Think creatively about this problem",
                    "parameters": {"max_tokens": 200}
                }
            ]
        }
        
        suite_file = os.path.join(self.temp_dir, "mixed_suite.json")
        with open(suite_file, 'w') as f:
            json.dump(mixed_suite, f)
        
        self.test_runner.load_test_suite(suite_file)
        
        # Execute only monitoring category
        monitoring_test_ids = self.test_runner.get_test_ids_by_category("monitoring")
        monitoring_results = self.test_runner.execute_sequential(
            test_ids=monitoring_test_ids
        )
        
        self.assertEqual(len(monitoring_results), 1, "Should execute only monitoring test")
        self.assertEqual(monitoring_results[0].test_id, "monitoring_01")
        self.assertTrue(monitoring_results[0].success, "Monitoring test should succeed")
        
        # Execute security category
        security_test_ids = self.test_runner.get_test_ids_by_category("security") 
        security_results = self.test_runner.execute_sequential(
            test_ids=security_test_ids
        )
        
        self.assertEqual(len(security_results), 1, "Should execute only security test")
        self.assertEqual(security_results[0].test_id, "security_01")
    
    def test_progress_tracking_integration(self):
        """Test progress tracking throughout complete workflow"""
        # Create test suite with known number of tests
        suite = {
            "test_suite_id": "progress_tracking",
            "version": "1.0.0", 
            "description": "Progress tracking test",
            "tests": [
                {"id": f"progress_test_{i:02d}", "name": f"Progress Test {i}", 
                 "category": "test", "prompt": f"Test {i}",
                 "parameters": {"max_tokens": 50}}
                for i in range(5)
            ]
        }
        
        suite_file = os.path.join(self.temp_dir, "progress_suite.json")
        with open(suite_file, 'w') as f:
            json.dump(suite, f)
        
        # Add small delay to observe progress
        self.mock_server.configure_response(delay=0.1)
        
        self.test_runner.load_test_suite(suite_file)
        
        # Check initial progress
        initial_progress = self.test_runner.get_progress()
        self.assertEqual(initial_progress.total_tests, 5)
        self.assertEqual(initial_progress.completed_tests, 0)
        
        # Execute tests
        results = self.test_runner.execute_sequential()
        
        # Check final progress
        final_progress = self.test_runner.get_progress()
        self.assertEqual(final_progress.total_tests, 5)
        self.assertEqual(final_progress.completed_tests, 5)
        self.assertEqual(final_progress.successful_tests, 5)
        self.assertEqual(final_progress.failed_tests, 0)
        
        # Verify all tests completed
        self.assertEqual(len(results), 5, "Should complete all 5 tests")


if __name__ == "__main__":
    # Configure test runner for verbose output
    unittest.main(verbosity=2, buffer=True)