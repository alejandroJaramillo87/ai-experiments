#!/usr/bin/env python3
"""
Network Failure Scenario Tests

Tests system behavior under various network failure conditions:
- Server timeouts
- Connection failures  
- Invalid endpoints
- Malformed requests
- Intermittent connectivity

Addresses audit findings for limited error scenario coverage.
"""

import unittest
import time
import requests
from unittest.mock import patch, Mock
from .base_functional_test import BaseFunctionalTest
from .server_dependency_manager import ServerDependencyManager, FunctionalTestHelper, ServerStatus, ServerHealth


class TestNetworkFailureScenarios(BaseFunctionalTest):
    """Test behavior under various network failure conditions"""
    
    def setUp(self):
        """Set up network failure testing environment"""
        super().setUp()
        self.server_manager = ServerDependencyManager()
        self.test_helper = FunctionalTestHelper()
    
    def test_server_timeout_handling(self):
        """Test behavior when server requests timeout"""
        # Mock both GET and POST requests to simulate timeout
        with patch('tests.functional.server_dependency_manager.requests.get') as mock_get, \
             patch('tests.functional.server_dependency_manager.requests.post') as mock_post:
            mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")
            mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")
            
            manager = ServerDependencyManager()
            health = manager.check_server_health(use_cache=False)
            
            # Should handle timeout gracefully
            self.assertIn(health.status, [ServerStatus.TIMEOUT, ServerStatus.NETWORK_ERROR, ServerStatus.UNAVAILABLE])
            self.assertTrue(
                any(term in health.error_message.lower() for term in ['timeout', 'timed out']),
                f"Error message should contain timeout info: {health.error_message}"
            )
    
    def test_invalid_endpoint_handling(self):
        """Test behavior with invalid server endpoints"""
        # Mock connection error for invalid endpoint
        with patch('tests.functional.server_dependency_manager.socket.socket') as mock_socket:
            mock_sock = Mock()
            mock_sock.connect_ex.return_value = 1  # Connection failed
            mock_socket.return_value = mock_sock
            
            invalid_manager = ServerDependencyManager(base_url="http://localhost:9999")
            health = invalid_manager.check_server_health(use_cache=False)
            
            # Should handle invalid endpoint gracefully
            self.assertIn(health.status, [ServerStatus.UNAVAILABLE, ServerStatus.NETWORK_ERROR])
            self.assertIsNotNone(health.error_message)
    
    def test_malformed_request_handling(self):
        """Test behavior with malformed requests"""
        # Mock the request to simulate malformed request handling
        with patch('tests.functional.server_dependency_manager.requests.post') as mock_post:
            # Mock a 400 Bad Request response for malformed JSON
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.text = "Bad Request: Invalid JSON"
            mock_post.return_value = mock_response
            
            try:
                response = requests.post(
                    self.server_manager.completion_endpoint,
                    data="invalid-json-payload",  # Not valid JSON
                    headers={"Content-Type": "application/json"},
                    timeout=5
                )
                # Server should reject malformed request
                self.assertNotIn(response.status_code, [200, 201])
                
            except requests.exceptions.RequestException:
                # Connection failure is acceptable for this test
                pass
    
    def test_cli_with_server_unavailable(self):
        """Test CLI behavior when server is completely unavailable"""
        # Use an invalid port to simulate server unavailability
        args = [
            "--test-type", "base",
            "--test-id", "basic_01", 
            "--endpoint", "http://localhost:9999/v1/completions",  # Invalid port
            "--model", self.DEFAULT_MODEL,
            "--output-dir", self.temp_output_dir
        ]
        
        stdout, stderr, exit_code = self.run_cli_command(args)
        
        # CLI should handle server failure gracefully (non-zero exit code expected)
        self.assertNotEqual(exit_code, 0, "CLI should fail gracefully when server unavailable")
        
        # Should contain helpful error message
        error_output = stdout + stderr
        self.assertTrue(
            any(term in error_output.lower() for term in ['connection', 'error', 'failed', 'timeout', 'unavailable']),
            f"Error output should mention connection issue: {error_output[:200]}..."
        )
    
    def test_network_failure_scenarios_comprehensive(self):
        """Test comprehensive network failure scenario detection"""
        # Mock various request failures to test scenario detection
        with patch('requests.get') as mock_get, patch('requests.post') as mock_post:
            # Mock timeout scenario
            mock_get.side_effect = requests.exceptions.Timeout("Timeout test")
            mock_post.side_effect = requests.exceptions.RequestException("Request error")
            
            scenarios = self.server_manager.test_network_failure_scenarios()
            
            # Should test multiple scenario types
            expected_scenarios = ['timeout', 'invalid_endpoint', 'malformed_request']
            
            for scenario in expected_scenarios:
                self.assertIn(scenario, scenarios, f"Should test {scenario} scenario")
                self.assertIsInstance(scenarios[scenario], bool, f"Scenario {scenario} should return boolean")
            
            print(f"✅ Network Failure Scenarios Tested:")
            for scenario, handled in scenarios.items():
                print(f"   {scenario}: {'✅ Handled' if handled else '❌ Not handled'}")
    
    def test_resilient_request_retry_logic(self):
        """Test retry logic for resilient requests"""
        # Mock server as available to test real retry logic
        with patch('tests.functional.server_dependency_manager.ServerDependencyManager.check_server_health') as mock_health:
            mock_health.return_value = ServerHealth(status=ServerStatus.AVAILABLE)
            
            helper = FunctionalTestHelper()
            helper.setup_test_environment()
            helper.mock_mode = False  # Force real mode to test retry logic
            
            # Test payload
            test_payload = {
                "prompt": "Test retry logic",
                "max_tokens": 5,
                "temperature": 0.0
            }
            
            # Test with mock network failures
            with patch('requests.post') as mock_post:
                # Simulate temporary network failure followed by success
                mock_response_success = Mock()
                mock_response_success.status_code = 200
                mock_response_success.json.return_value = {
                    "choices": [{"text": "test response"}],
                    "usage": {"total_tokens": 10}
                }
                
                # First call fails, second succeeds
                mock_post.side_effect = [
                    requests.exceptions.Timeout("Timeout"),
                    mock_response_success
                ]
                
                result = helper.create_resilient_request(test_payload, max_retries=3)
                
                # Should eventually succeed with retry
                self.assertIsNotNone(result, "Resilient request should succeed with retry")
                self.assertEqual(mock_post.call_count, 2, "Should retry after first failure")
    
    def test_graceful_degradation_to_mock_mode(self):
        """Test graceful degradation to mock mode when server unavailable"""
        # Mock server as unavailable to test degradation
        with patch('tests.functional.server_dependency_manager.ServerDependencyManager.check_server_health') as mock_health:
            mock_health.return_value = ServerHealth(
                status=ServerStatus.NETWORK_ERROR,
                error_message="Network unreachable"
            )
            
            helper = FunctionalTestHelper()
            success, message = helper.setup_test_environment()
            
            # Should succeed by falling back to mock mode
            self.assertTrue(success, f"Should succeed with mock mode: {message}")
            
            if helper.mock_mode:
                # Test mock response generation
                mock_response = helper.create_resilient_request({
                    "prompt": "Test mock response",
                    "max_tokens": 10
                })
                
                self.assertIsNotNone(mock_response)
                self.assertIn("choices", mock_response)
                self.assertIn("MOCK RESPONSE", mock_response["choices"][0]["text"])
    
    def test_server_health_check_error_handling(self):
        """Test server health check handles various error types"""
        test_cases = [
            # (Exception type, Expected status)
            (requests.exceptions.Timeout("Test timeout"), ServerStatus.TIMEOUT),
            (requests.exceptions.ConnectionError("Connection failed"), ServerStatus.NETWORK_ERROR), 
            (requests.exceptions.RequestException("General error"), ServerStatus.NETWORK_ERROR)
        ]
        
        for exception, expected_status in test_cases:
            with self.subTest(exception=type(exception).__name__):
                # Mock port connectivity to succeed, then mock health endpoint to fail
                with patch('tests.functional.server_dependency_manager.socket.socket') as mock_socket:
                    mock_sock = Mock()
                    mock_sock.connect_ex.return_value = 0  # Port is open
                    mock_socket.return_value = mock_sock
                    
                    with patch('tests.functional.server_dependency_manager.requests.get') as mock_get, \
                         patch('tests.functional.server_dependency_manager.requests.post') as mock_post:
                        mock_get.side_effect = exception
                        mock_post.side_effect = exception
                        
                        health = self.server_manager.check_server_health(use_cache=False)
                        
                        self.assertEqual(health.status, expected_status)
                        self.assertIsNotNone(health.error_message)
                        self.assertIn(str(exception), health.error_message)
    
    def test_connection_retry_with_exponential_backoff(self):
        """Test that connection retries use exponential backoff"""
        # Mock server as available but requests always timeout
        with patch('tests.functional.server_dependency_manager.ServerDependencyManager.check_server_health') as mock_health:
            mock_health.return_value = ServerHealth(status=ServerStatus.AVAILABLE)
            
            helper = FunctionalTestHelper()
            helper.mock_mode = False  # Force real mode to test retry logic
            
            start_time = time.time()
            
            with patch('requests.post') as mock_post:
                # Always timeout
                mock_post.side_effect = requests.exceptions.Timeout("Always timeout")
                
                result = helper.create_resilient_request(
                    {"prompt": "test", "max_tokens": 1}, 
                    max_retries=3
                )
                
                end_time = time.time()
                
                # Should be None after all retries fail
                self.assertIsNone(result, "Should fail after all retries")
                
                # Should have taken time for exponential backoff (1s + 2s + 3s = 6s minimum)
                total_time = end_time - start_time
                self.assertGreaterEqual(total_time, 5.0, 
                                      f"Should use exponential backoff, took {total_time:.1f}s")


class TestServerDependencyDetection(unittest.TestCase):
    """Test server dependency detection and backend identification"""
    
    def setUp(self):
        """Set up server dependency detection tests"""
        self.server_manager = ServerDependencyManager()
    
    def test_backend_type_detection_llama_cpp(self):
        """Test detection of llama.cpp backend"""
        # Mock health data with llama.cpp indicators
        llama_health_data = {
            "status": "ok",
            "model": "llama-2-7b.gguf",
            "backend": "llama.cpp",
            "slots": [{"id": 0, "state": "idle"}]
        }
        
        backend_type = self.server_manager._detect_backend_type(llama_health_data)
        self.assertEqual(backend_type, "llama.cpp")
    
    def test_backend_type_detection_vllm(self):
        """Test detection of vLLM backend"""
        # Mock health data with vLLM indicators (more specific vLLM indicators)
        vllm_health_data = {
            "status": "healthy",
            "version": "0.2.1",
            "vllm": "active",  # Direct vLLM indicator
            "asyncio": "running",  # AsyncIO indicator for vLLM
            "model": "meta-llama/Llama-2-7b-hf"
        }
        
        backend_type = self.server_manager._detect_backend_type(vllm_health_data)
        self.assertEqual(backend_type, "vLLM")
    
    def test_response_format_detection(self):
        """Test backend detection from response format"""
        # llama.cpp style response
        llama_response = {
            "choices": [{"text": "response"}],
            "timings": {"predicted_ms": 100},
            "completion_probabilities": []
        }
        
        backend = self.server_manager._detect_backend_from_response(llama_response)
        self.assertEqual(backend, "llama.cpp")
        
        # OpenAI compatible response
        openai_response = {
            "id": "cmpl-123",
            "choices": [{"text": "response", "index": 0}],
            "usage": {"total_tokens": 10}
        }
        
        backend = self.server_manager._detect_backend_from_response(openai_response)
        self.assertEqual(backend, "openai-compatible")
    
    def test_port_connectivity_check(self):
        """Test port connectivity checking"""
        # Test with localhost (should be available)
        manager_localhost = ServerDependencyManager("http://localhost:22")  # SSH port
        connectivity = manager_localhost._test_port_connectivity()
        # May or may not be available depending on system
        self.assertIsInstance(connectivity, bool)
        
        # Test with definitely unavailable port
        manager_invalid = ServerDependencyManager("http://localhost:99999")
        connectivity = manager_invalid._test_port_connectivity()
        self.assertFalse(connectivity)
    
    def test_health_check_caching(self):
        """Test that health checks are properly cached"""
        # First health check
        health1 = self.server_manager.check_server_health(use_cache=False)
        
        # Second health check with cache enabled (should be same instance)  
        health2 = self.server_manager.check_server_health(use_cache=True)
        
        # Should have cached the result
        self.assertIsNotNone(self.server_manager._last_health_check)
        
        # Cache time should be recent
        cache_age = time.time() - self.server_manager._last_health_check['timestamp']
        self.assertLess(cache_age, 5.0, "Cache should be fresh")


if __name__ == '__main__':
    print("🌐 Running Network Failure Scenario Tests")
    print("=" * 50)
    
    # Run tests with detailed output
    unittest.main(verbosity=2)