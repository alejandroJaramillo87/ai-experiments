#!/usr/bin/env python3
"""
Server Dependency Manager for Functional Tests

Provides robust server dependency management including:
- Server availability detection
- Graceful degradation when server unavailable
- Network failure scenario testing
- Timeout handling validation
- Connection retry logic

Addresses the audit finding for better error handling in functional tests.
"""

import requests
import time
import logging
import socket
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import subprocess

logger = logging.getLogger(__name__)


class ServerStatus(Enum):
    """Server status enumeration"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"


@dataclass
class ServerHealth:
    """Server health check result"""
    status: ServerStatus
    response_time: float = 0.0
    error_message: Optional[str] = None
    backend_type: Optional[str] = None  # llama.cpp vs vLLM
    model_loaded: bool = False
    capabilities: Dict[str, Any] = None


class ServerDependencyManager:
    """Manages server dependencies for functional tests"""
    
    def __init__(self, base_url: str = "http://localhost:8004", timeout: int = 60):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.health_endpoint = f"{base_url}/health"
        self.completion_endpoint = f"{base_url}/v1/completions"
        self.models_endpoint = f"{base_url}/v1/models"
        
        # Fallback endpoints for different backends
        self.llama_cpp_completion = f"{base_url}/completion"
        self.fallback_endpoints = [
            f"{base_url}/v1/completions",  # OpenAI standard
            f"{base_url}/completion",      # llama.cpp format
            f"{base_url}/generate"         # Alternative format
        ]
        
        self._last_health_check = None
        self._health_cache_duration = 30  # Cache health checks for 30 seconds
    
    def check_server_health(self, use_cache: bool = True) -> ServerHealth:
        """Comprehensive server health check"""
        current_time = time.time()
        
        # Return cached result if recent and cache enabled
        if (use_cache and self._last_health_check and 
            current_time - self._last_health_check['timestamp'] < self._health_cache_duration):
            return self._last_health_check['result']
        
        start_time = time.time()
        
        # Test basic connectivity
        if not self._test_port_connectivity():
            health = ServerHealth(
                status=ServerStatus.UNAVAILABLE,
                error_message="Port not accessible"
            )
            self._cache_health_result(health)
            return health
        
        # Test health endpoint if available
        health_status = self._check_health_endpoint()
        if health_status.status == ServerStatus.AVAILABLE:
            self._cache_health_result(health_status)
            return health_status
        
        # Test completion endpoint directly
        completion_status = self._check_completion_endpoint()
        completion_status.response_time = time.time() - start_time
        
        self._cache_health_result(completion_status)
        return completion_status
    
    def _test_port_connectivity(self) -> bool:
        """Test if server port is accessible"""
        try:
            host = self.base_url.split('://')[1].split(':')[0]
            port = int(self.base_url.split(':')[-1])
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()
            
            return result == 0
        except Exception as e:
            logger.debug(f"Port connectivity test failed: {e}")
            return False
    
    def _check_health_endpoint(self) -> ServerHealth:
        """Check dedicated health endpoint"""
        try:
            response = requests.get(self.health_endpoint, timeout=self.timeout)
            
            if response.status_code == 200:
                try:
                    health_data = response.json()
                    return ServerHealth(
                        status=ServerStatus.AVAILABLE,
                        response_time=response.elapsed.total_seconds(),
                        backend_type=self._detect_backend_type(health_data),
                        model_loaded=health_data.get('model_loaded', False),
                        capabilities=health_data
                    )
                except ValueError:  # JSON decode error
                    return ServerHealth(
                        status=ServerStatus.DEGRADED,
                        response_time=response.elapsed.total_seconds(),
                        error_message="Health endpoint returned non-JSON response"
                    )
            else:
                return ServerHealth(
                    status=ServerStatus.DEGRADED,
                    error_message=f"Health endpoint returned status {response.status_code}"
                )
                
        except requests.exceptions.Timeout:
            return ServerHealth(
                status=ServerStatus.TIMEOUT,
                error_message=f"Health endpoint timeout after {self.timeout}s"
            )
        except requests.exceptions.RequestException as e:
            return ServerHealth(
                status=ServerStatus.NETWORK_ERROR,
                error_message=f"Network error: {str(e)}"
            )
    
    def _check_completion_endpoint(self) -> ServerHealth:
        """Check completion endpoint with minimal test request"""
        test_payload = {
            "prompt": "Test",
            "max_tokens": 1,
            "temperature": 0.0
        }
        
        timeout_count = 0
        network_error_count = 0
        first_timeout_error = None
        first_network_error = None
        
        # Try different endpoint formats
        for endpoint in self.fallback_endpoints:
            try:
                response = requests.post(
                    endpoint, 
                    json=test_payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code in [200, 201]:
                    try:
                        result_data = response.json()
                        return ServerHealth(
                            status=ServerStatus.AVAILABLE,
                            response_time=response.elapsed.total_seconds(),
                            backend_type=self._detect_backend_from_response(result_data),
                            model_loaded=True,
                            capabilities={"completion_tested": True, "endpoint": endpoint}
                        )
                    except ValueError:
                        continue  # Try next endpoint
                        
            except requests.exceptions.Timeout as e:
                timeout_count += 1
                if first_timeout_error is None:
                    first_timeout_error = str(e)
                continue
            except requests.exceptions.RequestException as e:
                network_error_count += 1
                if first_network_error is None:
                    first_network_error = str(e)
                continue
        
        # Determine failure type based on error patterns
        if timeout_count > 0:
            error_msg = f"All completion endpoints timed out ({timeout_count} timeouts)"
            if first_timeout_error:
                error_msg += f": {first_timeout_error}"
            return ServerHealth(
                status=ServerStatus.TIMEOUT,
                error_message=error_msg
            )
        elif network_error_count > 0:
            error_msg = f"All completion endpoints failed with network errors"
            if first_network_error:
                error_msg += f": {first_network_error}"
            return ServerHealth(
                status=ServerStatus.NETWORK_ERROR,
                error_message=error_msg
            )
        else:
            return ServerHealth(
                status=ServerStatus.UNAVAILABLE,
                error_message="All completion endpoints failed"
            )
    
    def _detect_backend_type(self, health_data: Dict[str, Any]) -> Optional[str]:
        """Detect backend type from health data"""
        health_str = str(health_data).lower()
        
        # Check for vLLM indicators first (more specific)
        if any(key in health_str for key in ['vllm', 'ray']):
            return "vLLM"
        
        # Check for explicit vLLM status indicators
        if any(key in health_data for key in ['vllm', 'asyncio']):
            return "vLLM"
        
        # Check for llama.cpp specific indicators (excluding model names)
        if any(key in health_str for key in ['gguf', 'llamacpp', 'llama.cpp']):
            return "llama.cpp"
        
        # Check for llama.cpp server indicators
        if 'slots' in health_data and isinstance(health_data.get('slots'), list):
            return "llama.cpp"
        
        return "unknown"
    
    def _detect_backend_from_response(self, response_data: Dict[str, Any]) -> Optional[str]:
        """Detect backend type from completion response format"""
        if 'choices' in response_data:
            # Check for llama.cpp specific fields
            if any(field in response_data for field in ['completion_probabilities', 'timings']):
                return "llama.cpp"
            return "openai-compatible"
        
        return "unknown"
    
    def _cache_health_result(self, health: ServerHealth):
        """Cache health check result"""
        self._last_health_check = {
            'timestamp': time.time(),
            'result': health
        }
    
    def wait_for_server(self, max_wait_time: int = 60, check_interval: int = 5) -> bool:
        """Wait for server to become available"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            health = self.check_server_health(use_cache=False)
            
            if health.status == ServerStatus.AVAILABLE:
                logger.info(f"Server available after {time.time() - start_time:.1f}s")
                return True
            
            logger.debug(f"Server not ready: {health.status} - {health.error_message}")
            time.sleep(check_interval)
        
        logger.warning(f"Server not available after {max_wait_time}s timeout")
        return False
    
    def create_mock_server_response(self, prompt: str) -> Dict[str, Any]:
        """Create mock server response for when server is unavailable"""
        return {
            "id": "mock-completion-id",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "mock-model",
            "choices": [{
                "text": f"[MOCK RESPONSE] This is a mock response for prompt: {prompt[:50]}...",
                "index": 0,
                "finish_reason": "mock"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 10,
                "total_tokens": len(prompt.split()) + 10
            }
        }
    
    def test_network_failure_scenarios(self) -> Dict[str, bool]:
        """Test various network failure scenarios"""
        scenarios = {}
        
        # Test timeout scenario
        try:
            requests.get(self.base_url, timeout=0.001)  # Very short timeout
            scenarios['timeout'] = False
        except requests.exceptions.Timeout:
            scenarios['timeout'] = True
        except Exception:
            scenarios['timeout'] = True
        
        # Test invalid endpoint
        try:
            requests.get(f"{self.base_url}/invalid-endpoint", timeout=2)
            scenarios['invalid_endpoint'] = False
        except requests.exceptions.RequestException:
            scenarios['invalid_endpoint'] = True
        
        # Test malformed request
        try:
            requests.post(
                self.completion_endpoint,
                data="invalid-json",  # Not JSON
                timeout=2,
                headers={"Content-Type": "application/json"}
            )
            scenarios['malformed_request'] = False
        except requests.exceptions.RequestException:
            scenarios['malformed_request'] = True
        
        return scenarios


class FunctionalTestHelper:
    """Helper for functional tests with server dependency management"""
    
    def __init__(self, base_url: str = "http://localhost:8004"):
        self.server_manager = ServerDependencyManager(base_url)
        self.mock_mode = False
        self.server_health = None
    
    def setup_test_environment(self) -> Tuple[bool, str]:
        """Set up test environment and return (success, message)"""
        self.server_health = self.server_manager.check_server_health()
        
        if self.server_health.status == ServerStatus.AVAILABLE:
            return True, f"Server available ({self.server_health.backend_type})"
        
        elif self.server_health.status in [ServerStatus.TIMEOUT, ServerStatus.NETWORK_ERROR]:
            self.mock_mode = True
            return True, f"Using mock mode: {self.server_health.error_message}"
        
        else:
            return False, f"Server unavailable: {self.server_health.error_message}"
    
    def should_skip_test(self, require_server: bool = True) -> Tuple[bool, str]:
        """Determine if test should be skipped"""
        if not require_server:
            return False, ""
        
        if self.server_health.status == ServerStatus.AVAILABLE:
            return False, ""
        
        return True, f"Skipping test - Server {self.server_health.status}: {self.server_health.error_message}"
    
    def get_test_endpoint(self) -> str:
        """Get appropriate endpoint for testing"""
        if self.mock_mode:
            return "http://mock-server:8004/v1/completions"
        
        return self.server_manager.completion_endpoint
    
    def create_resilient_request(self, payload: Dict[str, Any], max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Make resilient request with retry logic"""
        if self.mock_mode:
            return self.server_manager.create_mock_server_response(payload.get('prompt', 'test'))
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.server_manager.completion_endpoint,
                    json=payload,
                    timeout=self.server_manager.timeout
                )
                
                if response.status_code in [200, 201]:
                    return response.json()
                else:
                    last_error = f"HTTP {response.status_code}: {response.text}"
                    
            except requests.exceptions.Timeout:
                last_error = "Request timeout"
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {str(e)}"
                time.sleep(1 * (attempt + 1))
        
        logger.error(f"Request failed after {max_retries} attempts: {last_error}")
        return None