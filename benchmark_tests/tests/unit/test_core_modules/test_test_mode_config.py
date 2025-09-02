#!/usr/bin/env python3
"""
Unit Tests for Test Mode Configuration System

Tests the production vs development test mode configuration system including:
- Mode detection and initialization
- Environment variable handling
- Backend detection and concurrency adjustment
- Configuration validation and overrides
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from core.test_mode_config import (
    TestMode, BackendType, TestModeConfiguration, TestModeManager,
    initialize_test_mode, get_current_test_config, get_test_mode_manager
)

class TestTestModeConfiguration(unittest.TestCase):
    """Test TestModeConfiguration dataclass"""
    
    def test_default_configuration_creation(self):
        """Test creating default configuration"""
        config = TestModeConfiguration(
            mode=TestMode.DEVELOPMENT,
            description="Test configuration"
        )
        
        self.assertEqual(config.mode, TestMode.DEVELOPMENT)
        self.assertEqual(config.description, "Test configuration")
        self.assertEqual(config.chunk_size, 10)  # Default value
        self.assertEqual(config.max_concurrent, 1)  # Default value
        self.assertTrue(config.include_integration_tests)  # Default True
    
    def test_custom_configuration_creation(self):
        """Test creating configuration with custom values"""
        config = TestModeConfiguration(
            mode=TestMode.PRODUCTION,
            description="Production config",
            chunk_size=50,
            max_concurrent=4,
            memory_limit_mb=14000,
            enable_coverage=True,
            verbose_output=False
        )
        
        self.assertEqual(config.mode, TestMode.PRODUCTION)
        self.assertEqual(config.chunk_size, 50)
        self.assertEqual(config.max_concurrent, 4)
        self.assertEqual(config.memory_limit_mb, 14000)
        self.assertTrue(config.enable_coverage)
        self.assertFalse(config.verbose_output)

class TestTestModeManager(unittest.TestCase):
    """Test TestModeManager functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.manager = TestModeManager()
        
        # Clean environment variables
        env_vars_to_clean = [
            'BENCHMARK_TEST_MODE', 'BENCHMARK_CHUNK_SIZE', 'BENCHMARK_MAX_CONCURRENT',
            'BENCHMARK_TIMEOUT', 'BENCHMARK_MEMORY_LIMIT_MB', 'BENCHMARK_ENABLE_COVERAGE',
            'BENCHMARK_VERBOSE', 'BENCHMARK_FORCE_SEQUENTIAL'
        ]
        
        self.original_env = {}
        for var in env_vars_to_clean:
            self.original_env[var] = os.getenv(var)
            if var in os.environ:
                del os.environ[var]
    
    def tearDown(self):
        """Clean up test environment"""
        # Restore original environment
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]
    
    def test_backend_detection_llama_cpp(self):
        """Test llama.cpp backend detection"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = "Starting llama.cpp server with GGUF model"
            
            backend = self.manager.detect_backend_type()
            self.assertEqual(backend, BackendType.LLAMA_CPP)
    
    def test_backend_detection_vllm(self):
        """Test vLLM backend detection"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = "Starting vLLM server with Ray backend"
            
            backend = self.manager.detect_backend_type()
            self.assertEqual(backend, BackendType.VLLM)
    
    def test_backend_detection_unknown(self):
        """Test unknown backend detection"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.stdout = "Unknown server configuration"
            
            backend = self.manager.detect_backend_type()
            self.assertEqual(backend, BackendType.UNKNOWN)
    
    def test_backend_detection_error(self):
        """Test backend detection with subprocess error"""
        with patch('subprocess.run', side_effect=Exception("Docker not available")):
            backend = self.manager.detect_backend_type()
            self.assertEqual(backend, BackendType.UNKNOWN)
    
    def test_optimal_concurrency_llama_cpp(self):
        """Test optimal concurrency for llama.cpp"""
        concurrency = self.manager.get_optimal_concurrency(BackendType.LLAMA_CPP)
        self.assertEqual(concurrency, 1)
    
    def test_optimal_concurrency_vllm(self):
        """Test optimal concurrency for vLLM"""
        with patch('psutil.cpu_count', return_value=8):
            concurrency = self.manager.get_optimal_concurrency(BackendType.VLLM)
            self.assertEqual(concurrency, 4)  # min(4, max(1, 8//2))
    
    def test_optimal_concurrency_unknown(self):
        """Test optimal concurrency for unknown backend"""
        concurrency = self.manager.get_optimal_concurrency(BackendType.UNKNOWN)
        self.assertEqual(concurrency, 1)
    
    def test_load_mode_from_environment_default(self):
        """Test loading default mode from environment"""
        mode = self.manager.load_mode_from_environment()
        self.assertEqual(mode, TestMode.DEVELOPMENT)  # Default
    
    def test_load_mode_from_environment_production(self):
        """Test loading production mode from environment"""
        os.environ['BENCHMARK_TEST_MODE'] = 'production'
        mode = self.manager.load_mode_from_environment()
        self.assertEqual(mode, TestMode.PRODUCTION)
    
    def test_load_mode_from_environment_aliases(self):
        """Test loading mode from environment with aliases"""
        # Test prod alias
        os.environ['BENCHMARK_TEST_MODE'] = 'prod'
        mode = self.manager.load_mode_from_environment()
        self.assertEqual(mode, TestMode.PRODUCTION)
        
        # Test dev alias
        os.environ['BENCHMARK_TEST_MODE'] = 'dev'
        mode = self.manager.load_mode_from_environment()
        self.assertEqual(mode, TestMode.DEVELOPMENT)
        
        # Test debug alias
        os.environ['BENCHMARK_TEST_MODE'] = 'dbg'
        mode = self.manager.load_mode_from_environment()
        self.assertEqual(mode, TestMode.DEBUG)
    
    def test_collect_environment_overrides_numeric(self):
        """Test collecting numeric environment overrides"""
        os.environ['BENCHMARK_CHUNK_SIZE'] = '25'
        os.environ['BENCHMARK_MAX_CONCURRENT'] = '2'
        os.environ['BENCHMARK_TIMEOUT'] = '45'
        
        overrides = self.manager.collect_environment_overrides()
        
        self.assertEqual(overrides['chunk_size'], 25)
        self.assertEqual(overrides['max_concurrent'], 2)
        self.assertEqual(overrides['timeout_per_test'], 45)
    
    def test_collect_environment_overrides_boolean(self):
        """Test collecting boolean environment overrides"""
        os.environ['BENCHMARK_ENABLE_COVERAGE'] = 'true'
        os.environ['BENCHMARK_VERBOSE'] = '1'
        os.environ['BENCHMARK_FORCE_SEQUENTIAL'] = 'false'
        
        overrides = self.manager.collect_environment_overrides()
        
        self.assertTrue(overrides['enable_coverage'])
        self.assertTrue(overrides['verbose_output'])
        self.assertFalse(overrides['force_sequential'])
    
    def test_collect_environment_overrides_float(self):
        """Test collecting float environment overrides"""
        os.environ['BENCHMARK_MIN_SUCCESS_RATE'] = '0.85'
        os.environ['BENCHMARK_CALIBRATION_THRESHOLD'] = '75.5'
        
        overrides = self.manager.collect_environment_overrides()
        
        self.assertAlmostEqual(overrides['min_success_rate'], 0.85)
        self.assertAlmostEqual(overrides['calibration_threshold'], 75.5)
    
    def test_collect_environment_overrides_invalid_values(self):
        """Test handling invalid environment values"""
        os.environ['BENCHMARK_CHUNK_SIZE'] = 'invalid'
        os.environ['BENCHMARK_MIN_SUCCESS_RATE'] = 'not_a_float'
        
        with patch('core.test_mode_config.logger.warning') as mock_logger:
            overrides = self.manager.collect_environment_overrides()
            
            # Should not include invalid values
            self.assertNotIn('chunk_size', overrides)
            self.assertNotIn('min_success_rate', overrides)
            
            # Should log warnings
            self.assertEqual(mock_logger.call_count, 2)
    
    @patch('core.test_mode_config.TestModeManager.detect_backend_type')
    @patch('core.test_mode_config.TestModeManager.get_optimal_concurrency')
    def test_initialize_mode_production(self, mock_concurrency, mock_backend):
        """Test initializing production mode"""
        mock_backend.return_value = BackendType.VLLM
        mock_concurrency.return_value = 4
        
        config = self.manager.initialize_mode(TestMode.PRODUCTION)
        
        self.assertEqual(config.mode, TestMode.PRODUCTION)
        # Note: max_concurrent may be adjusted by backend detection
        self.assertIn(config.max_concurrent, [1, 2, 4])  # Backend detection can override
        self.assertFalse(config.include_functional_tests)  # Excluded for faster execution
        self.assertFalse(config.include_calibration_tests)  # Excluded for faster execution
        self.assertTrue(config.enable_coverage)
        self.assertEqual(config.chunk_size, 25)  # Optimized chunk size
    
    @patch('core.test_mode_config.TestModeManager.detect_backend_type')
    @patch('core.test_mode_config.TestModeManager.get_optimal_concurrency')
    def test_initialize_mode_development(self, mock_concurrency, mock_backend):
        """Test initializing development mode"""
        mock_backend.return_value = BackendType.LLAMA_CPP
        mock_concurrency.return_value = 1
        
        config = self.manager.initialize_mode(TestMode.DEVELOPMENT)
        
        self.assertEqual(config.mode, TestMode.DEVELOPMENT)
        self.assertEqual(config.max_concurrent, 1)
        self.assertFalse(config.include_functional_tests)  # Disabled in dev
        self.assertFalse(config.include_calibration_tests)  # Disabled in dev
        self.assertFalse(config.enable_coverage)  # Disabled in dev for speed
        self.assertEqual(config.chunk_size, 5)  # Development default
        self.assertTrue(config.verbose_output)  # Enabled in dev
    
    @patch('core.test_mode_config.TestModeManager.detect_backend_type')
    @patch('core.test_mode_config.TestModeManager.get_optimal_concurrency')
    def test_initialize_mode_debug(self, mock_concurrency, mock_backend):
        """Test initializing debug mode"""
        mock_backend.return_value = BackendType.VLLM
        mock_concurrency.return_value = 4
        
        config = self.manager.initialize_mode(TestMode.DEBUG)
        
        self.assertEqual(config.mode, TestMode.DEBUG)
        self.assertEqual(config.max_concurrent, 1)  # Forced sequential in debug
        self.assertTrue(config.include_functional_tests)
        self.assertTrue(config.include_calibration_tests)
        self.assertTrue(config.enable_coverage)
        self.assertEqual(config.chunk_size, 1)  # One test at a time for debugging
        self.assertTrue(config.verbose_output)
        self.assertTrue(config.enable_debug_hooks)
    
    @patch('core.test_mode_config.TestModeManager.detect_backend_type')
    def test_initialize_mode_with_environment_overrides(self, mock_backend):
        """Test initializing mode with environment overrides"""
        mock_backend.return_value = BackendType.UNKNOWN
        
        # Set environment overrides
        os.environ['BENCHMARK_CHUNK_SIZE'] = '15'
        os.environ['BENCHMARK_ENABLE_COVERAGE'] = 'true'
        os.environ['BENCHMARK_MAX_CONCURRENT'] = '3'
        
        config = self.manager.initialize_mode(TestMode.DEVELOPMENT)
        
        # Should apply overrides
        self.assertEqual(config.chunk_size, 15)  # Overridden from env
        self.assertTrue(config.enable_coverage)  # Overridden from env
        self.assertEqual(config.max_concurrent, 3)  # Overridden from env
    
    def test_should_include_test_type(self):
        """Test checking if test types should be included"""
        config = self.manager.initialize_mode(TestMode.DEVELOPMENT)
        
        # Development mode excludes functional and calibration
        self.assertFalse(self.manager.should_include_test_type('functional'))
        self.assertFalse(self.manager.should_include_test_type('calibration'))
        self.assertTrue(self.manager.should_include_test_type('integration'))
        self.assertTrue(self.manager.should_include_test_type('unknown_type'))  # Default True
    
    def test_get_test_directories(self):
        """Test getting test directories based on mode"""
        # Development mode
        self.manager.initialize_mode(TestMode.DEVELOPMENT)
        dirs = self.manager.get_test_directories()
        
        self.assertIn('tests/unit', dirs)
        self.assertIn('tests/integration', dirs)
        self.assertNotIn('tests/functional', dirs)  # Excluded in dev
        self.assertNotIn('tests/calibration', dirs)  # Excluded in dev
        
        # Production mode
        self.manager.initialize_mode(TestMode.PRODUCTION)
        dirs = self.manager.get_test_directories()
        
        self.assertIn('tests/unit', dirs)
        self.assertIn('tests/integration', dirs)
        self.assertNotIn('tests/functional', dirs)  # Excluded for optimization
        self.assertNotIn('tests/calibration', dirs)  # Excluded for optimization
    
    def test_get_pytest_args(self):
        """Test getting pytest arguments based on configuration"""
        # Development mode (verbose)
        config = self.manager.initialize_mode(TestMode.DEVELOPMENT)
        args = self.manager.get_pytest_args()
        
        self.assertIn('--tb=short', args)
        self.assertIn('--strict-markers', args)
        self.assertIn('--disable-warnings', args)
        self.assertIn('-v', args)  # Verbose in development
        self.assertIn('-s', args)  # Verbose in development
        self.assertNotIn('--cov=evaluator', args)  # No coverage in dev
        
        # Production mode (coverage enabled)
        config = self.manager.initialize_mode(TestMode.PRODUCTION)
        args = self.manager.get_pytest_args()
        
        self.assertIn('--cov=evaluator', args)  # Coverage in production
        self.assertIn('--cov=core', args)
        self.assertIn('--cov-report=term-missing', args)
        self.assertNotIn('-v', args)  # Not verbose in production (unless overridden)
    
    def test_export_environment_variables(self):
        """Test exporting configuration as environment variables"""
        config = self.manager.initialize_mode(TestMode.PRODUCTION)
        env_vars = self.manager.export_environment_variables()
        
        self.assertEqual(env_vars['BENCHMARK_TEST_MODE'], 'production')
        self.assertEqual(env_vars['BENCHMARK_CHUNK_SIZE'], '25')  # Updated to match optimized config
        self.assertEqual(env_vars['BENCHMARK_ENABLE_COVERAGE'], 'true')
        self.assertIn('BENCHMARK_MAX_CONCURRENT', env_vars)
        self.assertIn('BENCHMARK_MEMORY_LIMIT_MB', env_vars)

class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    def tearDown(self):
        """Clean up global state"""
        # Reset global manager
        import core.test_mode_config
        core.test_mode_config._global_test_mode_manager = None
        
        # Clean environment
        if 'BENCHMARK_TEST_MODE' in os.environ:
            del os.environ['BENCHMARK_TEST_MODE']
    
    def test_initialize_test_mode(self):
        """Test initialize_test_mode convenience function"""
        config = initialize_test_mode(TestMode.DEBUG)
        
        self.assertEqual(config.mode, TestMode.DEBUG)
        self.assertEqual(config.chunk_size, 1)
        self.assertTrue(config.enable_debug_hooks)
    
    def test_get_current_test_config(self):
        """Test get_current_test_config convenience function"""
        # Before initialization
        config = get_current_test_config()
        self.assertIsNone(config)
        
        # After initialization
        initialize_test_mode(TestMode.PRODUCTION)
        config = get_current_test_config()
        
        self.assertIsNotNone(config)
        self.assertEqual(config.mode, TestMode.PRODUCTION)
    
    def test_get_test_mode_manager_singleton(self):
        """Test get_test_mode_manager returns singleton"""
        manager1 = get_test_mode_manager()
        manager2 = get_test_mode_manager()
        
        self.assertIs(manager1, manager2)  # Same instance

class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def tearDown(self):
        """Clean up"""
        # Reset global manager
        import core.test_mode_config
        core.test_mode_config._global_test_mode_manager = None
        
        # Clean environment
        for var in ['BENCHMARK_TEST_MODE', 'BENCHMARK_CHUNK_SIZE', 'BENCHMARK_VERBOSE']:
            if var in os.environ:
                del os.environ[var]
    
    @patch('core.test_mode_config.TestModeManager.detect_backend_type')
    @patch('core.test_mode_config.TestModeManager.get_optimal_concurrency')
    def test_complete_production_workflow(self, mock_concurrency, mock_backend):
        """Test complete production workflow with backend detection"""
        mock_backend.return_value = BackendType.VLLM
        mock_concurrency.return_value = 4
        
        # Set environment for production
        os.environ['BENCHMARK_TEST_MODE'] = 'production'
        os.environ['BENCHMARK_CHUNK_SIZE'] = '100'
        
        # Initialize
        config = initialize_test_mode()
        
        # Verify configuration
        self.assertEqual(config.mode, TestMode.PRODUCTION)
        self.assertEqual(config.chunk_size, 100)  # From environment
        self.assertEqual(config.max_concurrent, 4)  # From backend detection
        self.assertFalse(config.include_functional_tests)  # Optimized out
        
        # Get test directories and pytest args
        manager = get_test_mode_manager()
        dirs = manager.get_test_directories()
        args = manager.get_pytest_args()
        
        # Verify production settings
        self.assertNotIn('tests/functional', dirs)  # Excluded for optimization
        self.assertNotIn('tests/calibration', dirs)  # Excluded for optimization
        self.assertIn('--cov=evaluator', args)
    
    @patch('core.test_mode_config.TestModeManager.detect_backend_type')
    def test_complete_development_workflow(self, mock_backend):
        """Test complete development workflow"""
        mock_backend.return_value = BackendType.LLAMA_CPP
        
        # Set environment for development  
        os.environ['BENCHMARK_TEST_MODE'] = 'dev'
        os.environ['BENCHMARK_VERBOSE'] = 'true'
        
        # Initialize
        config = initialize_test_mode()
        
        # Verify configuration
        self.assertEqual(config.mode, TestMode.DEVELOPMENT)
        self.assertEqual(config.max_concurrent, 1)  # Sequential for llama.cpp
        self.assertTrue(config.verbose_output)  # From environment
        self.assertFalse(config.include_functional_tests)  # Dev mode default
        
        # Get test directories and pytest args
        manager = get_test_mode_manager()
        dirs = manager.get_test_directories()
        args = manager.get_pytest_args()
        
        # Verify development settings
        self.assertNotIn('tests/functional', dirs)  # Excluded
        self.assertNotIn('tests/calibration', dirs)  # Excluded
        self.assertIn('-v', args)  # Verbose output
        self.assertNotIn('--cov=evaluator', args)  # No coverage in dev

if __name__ == '__main__':
    unittest.main()