#!/usr/bin/env python3
"""
Test Mode Configuration System

Manages production vs development test execution modes with optimized configurations
for different use cases:

- PRODUCTION: Full 26k+ test suite capability with extensive resource management
- DEVELOPMENT: Fast iteration with focused test execution and minimal overhead
- DEBUG: Verbose logging and detailed analysis for troubleshooting
"""

import os
import logging
from typing import Dict, Any, Optional, NamedTuple
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

class TestMode(Enum):
    """Test execution modes"""
    PRODUCTION = "production"
    PRODUCTION_FULL = "production_full"
    DEVELOPMENT = "development" 
    DEBUG = "debug"

class BackendType(Enum):
    """LLM Backend types detected from docker logs"""
    LLAMA_CPP = "llamacpp"
    VLLM = "vllm"
    UNKNOWN = "unknown"

@dataclass
class TestModeConfiguration:
    """Configuration for different test execution modes"""
    
    # Basic mode settings
    mode: TestMode
    description: str
    
    # Test execution settings
    chunk_size: int = 10
    max_concurrent: int = 1
    timeout_per_test: int = 30
    chunk_timeout: int = 600
    progress_interval: int = 5
    
    # Resource management
    memory_limit_mb: int = 8000
    enable_resource_monitoring: bool = True
    enable_aggressive_cleanup: bool = False
    
    # Coverage and reporting
    enable_coverage: bool = False
    enable_detailed_logging: bool = False
    enable_performance_metrics: bool = False
    
    # Test scope
    include_functional_tests: bool = True
    include_calibration_tests: bool = True
    include_integration_tests: bool = True
    include_network_failure_tests: bool = False
    
    # Backend optimization
    auto_detect_backend: bool = True
    force_sequential: bool = False
    
    # Quality thresholds
    min_success_rate: float = 0.7
    calibration_threshold: float = 60.0
    
    # Debugging
    verbose_output: bool = False
    save_intermediate_results: bool = False
    enable_debug_hooks: bool = False

class TestModeManager:
    """Manages test mode configurations and environment setup"""
    
    # Predefined configurations for different modes
    CONFIGURATIONS = {
        TestMode.PRODUCTION: TestModeConfiguration(
            mode=TestMode.PRODUCTION,
            description="Production deployment testing with comprehensive unit and integration coverage",
            chunk_size=25,  # Reduced for more manageable chunks
            max_concurrent=2,  # More conservative for stability
            timeout_per_test=45,  # Reduced timeout for faster feedback
            chunk_timeout=1800,  # 30 minutes - more reasonable
            progress_interval=5,  # More frequent progress updates
            memory_limit_mb=8000,  # 8GB limit - more conservative 
            enable_resource_monitoring=True,
            enable_aggressive_cleanup=True,
            enable_coverage=True,
            enable_detailed_logging=False,
            enable_performance_metrics=True,
            include_functional_tests=False,  # Exclude functional tests for faster execution
            include_calibration_tests=False,  # Exclude calibration tests for faster execution
            include_integration_tests=True,
            include_network_failure_tests=False,  # Exclude network failure tests
            auto_detect_backend=True,
            min_success_rate=0.85,  # Higher success rate threshold for production
            calibration_threshold=75.0,  # Slightly higher threshold
            verbose_output=False,
            save_intermediate_results=True,
            enable_debug_hooks=False
        ),
        
        TestMode.PRODUCTION_FULL: TestModeConfiguration(
            mode=TestMode.PRODUCTION_FULL,
            description="Comprehensive production testing including all server-dependent tests",
            chunk_size=15,  # Smaller chunks for server tests
            max_concurrent=1,  # Sequential execution for server stability
            timeout_per_test=120,  # Longer timeout for server operations
            chunk_timeout=7200,  # 2 hours for comprehensive testing
            progress_interval=10,  # Less frequent updates for long tests
            memory_limit_mb=10000,  # Higher memory limit for server tests
            enable_resource_monitoring=True,
            enable_aggressive_cleanup=True,
            enable_coverage=True,
            enable_detailed_logging=True,  # Detailed logging for server tests
            enable_performance_metrics=True,
            include_functional_tests=True,  # Include all functional tests
            include_calibration_tests=True,  # Include all calibration tests
            include_integration_tests=True,
            include_network_failure_tests=True,  # Include network failure tests
            auto_detect_backend=True,
            min_success_rate=0.95,  # Higher success rate for comprehensive testing
            calibration_threshold=80.0,  # Higher calibration threshold
            verbose_output=True,  # Verbose for comprehensive analysis
            save_intermediate_results=True,
            enable_debug_hooks=False
        ),
        
        TestMode.DEVELOPMENT: TestModeConfiguration(
            mode=TestMode.DEVELOPMENT,
            description="Fast iterative development with focused test execution",
            chunk_size=5,
            max_concurrent=1,  # Safe default for development
            timeout_per_test=20,
            chunk_timeout=300,  # 5 minutes for quick feedback
            progress_interval=2,
            memory_limit_mb=4000,  # Conservative for development
            enable_resource_monitoring=False,
            enable_aggressive_cleanup=False,
            enable_coverage=False,
            enable_detailed_logging=True,
            enable_performance_metrics=False,
            include_functional_tests=False,  # Skip slow tests in dev mode
            include_calibration_tests=False,
            include_integration_tests=True,
            include_network_failure_tests=False,
            auto_detect_backend=True,
            min_success_rate=0.6,
            calibration_threshold=50.0,
            verbose_output=True,
            save_intermediate_results=False,
            enable_debug_hooks=True
        ),
        
        TestMode.DEBUG: TestModeConfiguration(
            mode=TestMode.DEBUG,
            description="Comprehensive debugging with verbose logging and analysis",
            chunk_size=1,  # One test at a time for detailed analysis
            max_concurrent=1,
            timeout_per_test=120,  # Longer timeouts for debugging
            chunk_timeout=600,
            progress_interval=1,
            memory_limit_mb=8000,
            enable_resource_monitoring=True,
            enable_aggressive_cleanup=False,
            enable_coverage=True,
            enable_detailed_logging=True,
            enable_performance_metrics=True,
            include_functional_tests=True,
            include_calibration_tests=True,
            include_integration_tests=True,
            include_network_failure_tests=True,
            auto_detect_backend=True,
            force_sequential=True,  # Force sequential for debugging
            min_success_rate=0.0,  # No minimum for debugging
            calibration_threshold=0.0,
            verbose_output=True,
            save_intermediate_results=True,
            enable_debug_hooks=True
        )
    }
    
    def __init__(self):
        self.current_mode: Optional[TestMode] = None
        self.current_config: Optional[TestModeConfiguration] = None
        self.environment_overrides: Dict[str, Any] = {}
        
    def detect_backend_type(self) -> BackendType:
        """Detect LLM backend type from docker logs"""
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "compose", "logs", "llama-gpu"],
                capture_output=True, text=True, timeout=10
            )
            
            logs = result.stdout.lower()
            if any(keyword in logs for keyword in ['llama.cpp', 'llama-server', 'gguf', 'llamacpp']):
                return BackendType.LLAMA_CPP
            elif any(keyword in logs for keyword in ['vllm', 'ray', 'asyncio']):
                return BackendType.VLLM
            else:
                return BackendType.UNKNOWN
                
        except Exception as e:
            logger.warning(f"Backend detection failed: {e}")
            return BackendType.UNKNOWN
    
    def get_optimal_concurrency(self, backend_type: BackendType) -> int:
        """Get optimal concurrency based on backend type"""
        if backend_type == BackendType.LLAMA_CPP:
            return 1  # Sequential only for llama.cpp
        elif backend_type == BackendType.VLLM:
            import psutil
            return min(4, max(1, psutil.cpu_count() // 2))  # Conservative parallel
        else:
            return 1  # Safe default
    
    def load_mode_from_environment(self) -> TestMode:
        """Load test mode from environment variables"""
        mode_str = os.getenv('BENCHMARK_TEST_MODE', 'development').lower()
        
        # Map string values to enum
        mode_mapping = {
            'production': TestMode.PRODUCTION,
            'prod': TestMode.PRODUCTION,
            'development': TestMode.DEVELOPMENT,
            'dev': TestMode.DEVELOPMENT,
            'debug': TestMode.DEBUG,
            'dbg': TestMode.DEBUG
        }
        
        return mode_mapping.get(mode_str, TestMode.DEVELOPMENT)
    
    def collect_environment_overrides(self) -> Dict[str, Any]:
        """Collect configuration overrides from environment variables"""
        overrides = {}
        
        # Numeric overrides
        numeric_vars = {
            'BENCHMARK_CHUNK_SIZE': 'chunk_size',
            'BENCHMARK_MAX_CONCURRENT': 'max_concurrent', 
            'BENCHMARK_TIMEOUT': 'timeout_per_test',
            'BENCHMARK_MEMORY_LIMIT_MB': 'memory_limit_mb',
            'BENCHMARK_PROGRESS_INTERVAL': 'progress_interval'
        }
        
        for env_var, config_key in numeric_vars.items():
            value = os.getenv(env_var)
            if value:
                try:
                    overrides[config_key] = int(value)
                except ValueError:
                    logger.warning(f"Invalid numeric value for {env_var}: {value}")
        
        # Boolean overrides
        boolean_vars = {
            'BENCHMARK_ENABLE_COVERAGE': 'enable_coverage',
            'BENCHMARK_VERBOSE': 'verbose_output',
            'BENCHMARK_FORCE_SEQUENTIAL': 'force_sequential',
            'BENCHMARK_ENABLE_RESOURCE_MONITORING': 'enable_resource_monitoring',
            'BENCHMARK_ENABLE_DEBUG_HOOKS': 'enable_debug_hooks'
        }
        
        for env_var, config_key in boolean_vars.items():
            value = os.getenv(env_var)
            if value:
                overrides[config_key] = value.lower() in ('true', '1', 'yes', 'on')
        
        # Float overrides
        float_vars = {
            'BENCHMARK_MIN_SUCCESS_RATE': 'min_success_rate',
            'BENCHMARK_CALIBRATION_THRESHOLD': 'calibration_threshold'
        }
        
        for env_var, config_key in float_vars.items():
            value = os.getenv(env_var)
            if value:
                try:
                    overrides[config_key] = float(value)
                except ValueError:
                    logger.warning(f"Invalid float value for {env_var}: {value}")
        
        return overrides
    
    def initialize_mode(self, mode: Optional[TestMode] = None) -> TestModeConfiguration:
        """Initialize test mode configuration"""
        # Determine mode
        if mode is None:
            mode = self.load_mode_from_environment()
        
        self.current_mode = mode
        
        # Get base configuration
        base_config = self.CONFIGURATIONS[mode]
        
        # Collect environment overrides
        self.environment_overrides = self.collect_environment_overrides()
        
        # Apply overrides to create final configuration
        config_dict = base_config.__dict__.copy()
        config_dict.update(self.environment_overrides)
        
        # Create final configuration
        self.current_config = TestModeConfiguration(**config_dict)
        
        # Auto-detect backend and adjust concurrency if enabled
        if self.current_config.auto_detect_backend:
            backend_type = self.detect_backend_type()
            optimal_concurrency = self.get_optimal_concurrency(backend_type)
            
            # Override max_concurrent unless explicitly set in environment
            if 'max_concurrent' not in self.environment_overrides:
                self.current_config.max_concurrent = optimal_concurrency
                
            # Force sequential if backend is llama.cpp or forced
            if backend_type == BackendType.LLAMA_CPP or self.current_config.force_sequential:
                self.current_config.max_concurrent = 1
                
            logger.info(f"Backend detected: {backend_type.value}")
            logger.info(f"Concurrency adjusted to: {self.current_config.max_concurrent}")
        
        logger.info(f"Test mode initialized: {self.current_config.mode.value}")
        logger.info(f"Configuration: {self.current_config.description}")
        
        return self.current_config
    
    def get_current_config(self) -> Optional[TestModeConfiguration]:
        """Get current configuration"""
        return self.current_config
    
    def is_production_mode(self) -> bool:
        """Check if running in production mode"""
        return self.current_mode == TestMode.PRODUCTION
    
    def is_development_mode(self) -> bool:
        """Check if running in development mode"""
        return self.current_mode == TestMode.DEVELOPMENT
    
    def is_debug_mode(self) -> bool:
        """Check if running in debug mode"""
        return self.current_mode == TestMode.DEBUG
    
    def should_include_test_type(self, test_type: str) -> bool:
        """Check if specific test type should be included based on current mode"""
        if not self.current_config:
            return True
            
        test_type_mapping = {
            'functional': self.current_config.include_functional_tests,
            'calibration': self.current_config.include_calibration_tests,
            'integration': self.current_config.include_integration_tests,
            'network_failure': self.current_config.include_network_failure_tests
        }
        
        return test_type_mapping.get(test_type, True)
    
    def get_test_directories(self) -> list:
        """Get list of test directories to include based on current mode"""
        base_dirs = ['tests/unit']
        
        if self.should_include_test_type('integration'):
            base_dirs.append('tests/integration')
            
        if self.should_include_test_type('functional'):
            base_dirs.append('tests/functional')
            
        if self.should_include_test_type('calibration'):
            base_dirs.append('tests/calibration')
        
        return base_dirs
    
    def get_pytest_args(self) -> list:
        """Get pytest arguments based on current configuration"""
        args = ['--tb=short', '--strict-markers', '--disable-warnings']
        
        if not self.current_config:
            return args
            
        # Add verbosity
        if self.current_config.verbose_output:
            args.extend(['-v', '-s'])
        else:
            args.append('-q')
            
        # Add coverage if enabled
        if self.current_config.enable_coverage:
            args.extend([
                '--cov=evaluator', 
                '--cov=core',
                '--cov-report=term-missing',
                '--cov-fail-under=80',
                # Exclude validation modules with low coverage from production metrics
                '--cov-config=.coveragerc'
            ])
        
        # Add parallel execution for non-sequential modes
        if self.current_config.max_concurrent > 1:
            args.extend(['-n', str(self.current_config.max_concurrent)])
            
        return args
    
    def check_server_availability(self, server_url: str = "http://localhost:8004") -> bool:
        """Check if LLM server is available for server-dependent tests"""
        import requests
        import socket
        
        try:
            # First check port connectivity
            host = server_url.split('://')[1].split(':')[0]
            port = int(server_url.split(':')[-1])
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result != 0:
                logger.warning(f"Server port {port} not accessible")
                return False
            
            # Check health endpoint
            try:
                response = requests.get(f"{server_url}/health", timeout=10)
                if response.status_code == 200:
                    logger.info("✅ Server health check passed")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            # Try basic completion endpoint
            try:
                test_payload = {
                    "prompt": "Test",
                    "max_tokens": 1,
                    "temperature": 0.0
                }
                response = requests.post(
                    f"{server_url}/v1/completions",
                    json=test_payload,
                    timeout=10
                )
                if response.status_code in [200, 201]:
                    logger.info("✅ Server completion endpoint accessible")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            logger.warning(f"Server at {server_url} not responding to health checks")
            return False
            
        except Exception as e:
            logger.error(f"Server availability check failed: {e}")
            return False
    
    def export_environment_variables(self) -> Dict[str, str]:
        """Export current configuration as environment variables"""
        if not self.current_config:
            return {}
            
        env_vars = {
            'BENCHMARK_TEST_MODE': self.current_config.mode.value,
            'BENCHMARK_CHUNK_SIZE': str(self.current_config.chunk_size),
            'BENCHMARK_MAX_CONCURRENT': str(self.current_config.max_concurrent),
            'BENCHMARK_TIMEOUT': str(self.current_config.timeout_per_test),
            'BENCHMARK_MEMORY_LIMIT_MB': str(self.current_config.memory_limit_mb),
            'BENCHMARK_PROGRESS_INTERVAL': str(self.current_config.progress_interval),
            'BENCHMARK_ENABLE_COVERAGE': str(self.current_config.enable_coverage).lower(),
            'BENCHMARK_VERBOSE': str(self.current_config.verbose_output).lower(),
            'BENCHMARK_FORCE_SEQUENTIAL': str(self.current_config.force_sequential).lower(),
            'BENCHMARK_MIN_SUCCESS_RATE': str(self.current_config.min_success_rate),
            'BENCHMARK_CALIBRATION_THRESHOLD': str(self.current_config.calibration_threshold)
        }
        
        return env_vars

# Global test mode manager instance
_global_test_mode_manager: Optional[TestModeManager] = None

def get_test_mode_manager() -> TestModeManager:
    """Get or create global test mode manager"""
    global _global_test_mode_manager
    
    if _global_test_mode_manager is None:
        _global_test_mode_manager = TestModeManager()
    
    return _global_test_mode_manager

def initialize_test_mode(mode: Optional[TestMode] = None) -> TestModeConfiguration:
    """Initialize test mode (convenience function)"""
    manager = get_test_mode_manager()
    return manager.initialize_mode(mode)

def get_current_test_config() -> Optional[TestModeConfiguration]:
    """Get current test configuration (convenience function)"""
    manager = get_test_mode_manager()
    return manager.get_current_config()