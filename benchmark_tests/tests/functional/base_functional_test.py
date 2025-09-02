#!/usr/bin/env python3
"""
Base Functional Test Infrastructure

Simple foundation for end-to-end functional testing of the benchmark system
using real API calls and actual domain test files.

Updated to use shared test infrastructure.
"""

import unittest
import subprocess
import tempfile
import shutil
import os
import json
import sys
from typing import List, Tuple, Dict, Any, Optional

# Import shared test infrastructure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from shared import TestSetupHelper, PathHelper, configure_functional_logging
except ImportError:
    # Fallback if shared module not available
    TestSetupHelper = None
    PathHelper = None
    configure_functional_logging = lambda: None

# Import enhanced server dependency management
from .server_dependency_manager import FunctionalTestHelper, ServerStatus


class BaseFunctionalTest(unittest.TestCase):
    """
    Base class for functional tests with common utilities.
    
    Provides:
    - CLI command execution via subprocess 
    - Temporary directory management
    - JSON file validation helpers
    - Simple assertion utilities
    """
    
    # Configuration constants
    CLI_BASE = ["python", "benchmark_runner.py"]
    LOCALHOST_ENDPOINT = "http://127.0.0.1:8004/v1/completions"
    DEFAULT_MODEL = "/app/models/hf/DeepSeek-R1-0528-Qwen3-8b"
    
    def setUp(self):
        """Set up test fixtures - create temporary output directory and check server status"""
        self.temp_output_dir = tempfile.mkdtemp(prefix="benchmark_test_")
        
        # Change to benchmark_tests directory for CLI execution
        self.original_cwd = os.getcwd()
        benchmark_tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        os.chdir(benchmark_tests_dir)
        
        # Initialize enhanced server dependency management
        self.test_helper = FunctionalTestHelper(self.LOCALHOST_ENDPOINT.split('/v1')[0])
        self.server_available, self.server_message = self.test_helper.setup_test_environment()
        
        # Configure functional logging if available
        if configure_functional_logging:
            configure_functional_logging()
    
    def tearDown(self):
        """Clean up test fixtures - remove temporary directories"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_output_dir, ignore_errors=True)
    
    def run_cli_command(self, args: List[str], timeout: int = 180) -> Tuple[str, str, int]:
        """
        Run CLI command using subprocess and return results.
        
        Args:
            args: Command line arguments (without 'python benchmark_runner.py')
            timeout: Command timeout in seconds
            
        Returns:
            Tuple of (stdout, stderr, exit_code)
        """
        # Build full command with output directory
        full_cmd = self.CLI_BASE + args + ["--output-dir", self.temp_output_dir]
        
        try:
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False  # Don't raise exception on non-zero exit
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", f"Command timed out after {timeout}s", -1
        except Exception as e:
            return "", f"Command execution failed: {str(e)}", -2
    
    def validate_json_file(self, file_path: str, required_keys: List[str] = None) -> Dict[str, Any]:
        """
        Validate JSON file exists and contains required keys.
        
        Args:
            file_path: Path to JSON file
            required_keys: List of required top-level keys
            
        Returns:
            Parsed JSON data
            
        Raises:
            AssertionError: If file doesn't exist or is invalid
        """
        self.assertTrue(os.path.exists(file_path), f"JSON file should exist: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            self.fail(f"Failed to parse JSON file {file_path}: {e}")
        
        if required_keys:
            for key in required_keys:
                self.assertIn(key, data, f"JSON file should contain key '{key}': {file_path}")
        
        return data
    
    def assert_file_exists(self, file_path: str, description: str = ""):
        """Assert that a file exists with helpful error message"""
        msg = f"File should exist: {file_path}"
        if description:
            msg = f"{description}: {msg}"
        self.assertTrue(os.path.exists(file_path), msg)
    
    def assert_command_success(self, stdout: str, stderr: str, exit_code: int, cmd_description: str = ""):
        """Assert that a CLI command executed successfully with enhanced error information"""
        msg = f"Command should succeed"
        if cmd_description:
            msg = f"{cmd_description} should succeed"
        
        if exit_code != 0:
            error_info = f"Exit code: {exit_code}\\nStdout: {stdout}\\nStderr: {stderr}"
            
            # Add server status information if available
            if hasattr(self, 'test_helper'):
                error_info += f"\\nServer status: {self.server_message}"
            
            self.fail(f"{msg}\\n{error_info}")
    
    def assert_command_failure(self, stdout: str, stderr: str, exit_code: int, cmd_description: str = ""):
        """Assert that a CLI command failed as expected"""
        msg = f"Command should fail"
        if cmd_description:
            msg = f"{cmd_description} should fail"
        
        self.assertNotEqual(exit_code, 0, f"{msg}\\nStdout: {stdout}\\nStderr: {stderr}")
    
    def get_test_output_files(self, pattern: str = "*") -> List[str]:
        """
        Get list of files in temp output directory matching pattern.
        
        Args:
            pattern: Glob pattern to match (default: all files)
            
        Returns:
            List of file paths
        """
        import glob
        search_pattern = os.path.join(self.temp_output_dir, pattern)
        return glob.glob(search_pattern)
    
    def count_json_files(self, suffix: str = "_result.json") -> int:
        """Count JSON files with specific suffix in output directory"""
        files = self.get_test_output_files(f"*{suffix}")
        return len(files)
    
    def find_file_by_pattern(self, pattern: str) -> Optional[str]:
        """Find first file matching pattern in output directory"""
        files = self.get_test_output_files(pattern)
        return files[0] if files else None
    
    def get_evaluation_file(self, test_id: str) -> Optional[str]:
        """Get evaluation results file path for a test ID"""
        # Try both patterns: test_id_evaluation.json and test_name_evaluation.json
        result = self.find_file_by_pattern(f"{test_id}_evaluation.json")
        if not result:
            result = self.find_file_by_pattern(f"*_evaluation.json")
        return result
    
    def get_result_file(self, test_id: str) -> Optional[str]:
        """Get result file path for a test ID"""
        # Try both patterns: test_id_result.json and test_name_result.json  
        result = self.find_file_by_pattern(f"{test_id}_result.json")
        if not result:
            result = self.find_file_by_pattern(f"*_result.json")
        return result
    
    def get_completion_file(self, test_id: str) -> Optional[str]:
        """Get completion text file path for a test ID"""
        # Try both patterns: test_id_completion.txt and test_name_completion.txt
        result = self.find_file_by_pattern(f"{test_id}_completion.txt")
        if not result:
            result = self.find_file_by_pattern(f"*_completion.txt")
        return result
    
    # Enhanced server dependency management methods
    def skip_if_server_unavailable(self, require_real_server: bool = True):
        """Skip test if server is unavailable and real server is required"""
        if hasattr(self, 'test_helper'):
            should_skip, skip_reason = self.test_helper.should_skip_test(require_real_server)
            if should_skip:
                self.skipTest(skip_reason)
    
    def get_resilient_endpoint(self) -> str:
        """Get endpoint that works with current server status"""
        if hasattr(self, 'test_helper'):
            return self.test_helper.get_test_endpoint()
        return self.LOCALHOST_ENDPOINT
    
    def is_mock_mode(self) -> bool:
        """Check if running in mock mode due to server unavailability"""
        return getattr(self.test_helper, 'mock_mode', False) if hasattr(self, 'test_helper') else False