"""
Shared Test Helper Functions

Common utilities for setting up benchmark tests, managing paths,
and standardizing test operations across all test suites.

"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, patch


class PathHelper:
    """Standardized path management for tests"""
    
    @staticmethod
    def get_project_root() -> Path:
        """Get the benchmark_tests project root directory"""
        current = Path(__file__).resolve()
        # Go up from tests/shared/test_helpers.py to benchmark_tests/
        return current.parent.parent.parent
    
    @staticmethod
    def get_test_data_dir() -> Path:
        """Get path to test data directory"""
        return PathHelper.get_project_root() / "tests" / "data"
    
    @staticmethod
    def get_domain_dir() -> Path:
        """Get path to domains directory"""
        return PathHelper.get_project_root() / "domains"
    
    @staticmethod
    def get_evaluator_dir() -> Path:
        """Get path to evaluator directory"""
        return PathHelper.get_project_root() / "evaluator"
    
    @staticmethod
    def ensure_in_path():
        """Ensure project root is in Python path for imports"""
        project_root = str(PathHelper.get_project_root())
        if project_root not in sys.path:
            sys.path.insert(0, project_root)


class TestSetupHelper:
    """Common test setup operations"""
    
    @staticmethod
    def setup_test_environment():
        """Standard test environment setup"""
        PathHelper.ensure_in_path()
        
        # Set environment variables for testing
        os.environ['EVALUATOR_EMBEDDING_STRATEGY'] = 'fallback'
        os.environ['EVALUATOR_FORCE_CPU'] = 'true'
        
        # Ensure we're in the right directory
        project_root = PathHelper.get_project_root()
        if os.getcwd() != str(project_root):
            os.chdir(str(project_root))
    
    @staticmethod
    def create_temp_test_file(content: Dict[str, Any], suffix: str = ".json") -> str:
        """Create temporary test file with given content"""
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix=suffix, 
            delete=False
        ) as f:
            json.dump(content, f, indent=2)
            return f.name
    
    @staticmethod
    def cleanup_temp_files(file_paths: List[str]):
        """Clean up temporary test files"""
        for path in file_paths:
            try:
                os.unlink(path)
            except (OSError, FileNotFoundError):
                pass  # File already removed or doesn't exist


class BenchmarkTestHelper:
    """Helper functions specific to benchmark testing"""
    
    @staticmethod
    def create_mock_test_data(domain: str = "reasoning", difficulty: str = "easy") -> Dict[str, Any]:
        """Create mock test data for testing"""
        return {
            "metadata": {
                "domain": domain,
                "difficulty": difficulty,
                "version": "1.0.0",
                "description": f"Mock {domain} tests for unit testing"
            },
            "tests": [
                {
                    "id": "mock_test_1",
                    "name": "Mock Test Case 1",
                    "prompt": "This is a mock test prompt",
                    "expected_patterns": ["test", "pattern"],
                    "cultural_context": "universal",
                    "scoring": {
                        "exact_match_weight": 0.3,
                        "partial_match_weight": 0.4, 
                        "semantic_similarity_weight": 0.3
                    }
                }
            ]
        }
    
    @staticmethod
    def create_mock_api_response(content: str = "Mock API response") -> Dict[str, Any]:
        """Create mock API response for testing"""
        return {
            "content": content,
            "model": "mock-model",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
    
    @staticmethod
    def mock_benchmark_runner() -> Mock:
        """Create mock BenchmarkTestRunner for testing"""
        mock_runner = Mock()
        mock_runner.run_test.return_value = {
            "response": "Mock response",
            "score": 75.0,
            "evaluation_details": {
                "exact_match_score": 0.0,
                "partial_match_score": 0.75,
                "semantic_similarity_score": 0.0
            }
        }
        return mock_runner
    
    @staticmethod
    def assert_valid_test_result(result: Dict[str, Any]):
        """Assert that a test result has expected structure"""
        assert "response" in result, "Test result missing 'response' field"
        assert "score" in result, "Test result missing 'score' field"
        assert isinstance(result["score"], (int, float)), "Score must be numeric"
        assert 0 <= result["score"] <= 100, "Score must be between 0 and 100"
        
        if "evaluation_details" in result:
            details = result["evaluation_details"]
            for score_type in ["exact_match_score", "partial_match_score", "semantic_similarity_score"]:
                if score_type in details:
                    assert 0 <= details[score_type] <= 1, f"{score_type} must be between 0 and 1"


class MockHelper:
    """Helper for creating consistent mocks across test suites"""
    
    @staticmethod
    def mock_evaluator_imports():
        """Mock evaluator imports for testing without dependencies"""
        mock_modules = [
            'evaluator.subjects.enhanced_universal_evaluator',
            'evaluator.advanced.semantic_coherence', 
            'evaluator.advanced.entropy_calculator',
            'evaluator.advanced.model_loader'
        ]
        
        patches = {}
        for module in mock_modules:
            patches[module] = patch.dict('sys.modules', {module: Mock()})
            patches[module].start()
        
        return patches
    
    @staticmethod
    def cleanup_patches(patches: Dict[str, Any]):
        """Clean up mock patches"""
        for patch_obj in patches.values():
            patch_obj.stop()


# Convenience functions
def setup_test():
    """Quick test setup function"""
    TestSetupHelper.setup_test_environment()

def get_project_root():
    """Quick access to project root"""
    return PathHelper.get_project_root()

def create_mock_test_data():
    """Quick mock test data creation"""
    return BenchmarkTestHelper.create_mock_test_data()