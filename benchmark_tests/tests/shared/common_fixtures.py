"""
Common Test Fixtures

Shared pytest fixtures for use across all test suites.
Provides consistent setup and teardown for benchmark tests.

"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, Generator

from .test_helpers import TestSetupHelper, BenchmarkTestHelper, PathHelper


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Session-wide test environment setup"""
    TestSetupHelper.setup_test_environment()
    yield
    # Session cleanup if needed


@pytest.fixture
def project_root():
    """Provide project root path"""
    return PathHelper.get_project_root()


@pytest.fixture
def temp_test_file():
    """Create and cleanup temporary test file"""
    temp_files = []
    
    def _create_temp_file(content: Dict[str, Any], suffix: str = ".json") -> str:
        file_path = TestSetupHelper.create_temp_test_file(content, suffix)
        temp_files.append(file_path)
        return file_path
    
    yield _create_temp_file
    
    # Cleanup
    TestSetupHelper.cleanup_temp_files(temp_files)


@pytest.fixture
def mock_test_data():
    """Provide mock test data"""
    return BenchmarkTestHelper.create_mock_test_data()


@pytest.fixture
def mock_api_response():
    """Provide mock API response"""
    return BenchmarkTestHelper.create_mock_api_response()


@pytest.fixture
def mock_benchmark_runner():
    """Provide mock BenchmarkTestRunner"""
    return BenchmarkTestHelper.mock_benchmark_runner()


@pytest.fixture
def evaluator_fallback_mode():
    """Force evaluator to use fallback mode (no embedding models)"""
    original_strategy = os.environ.get('EVALUATOR_EMBEDDING_STRATEGY', '')
    original_cpu = os.environ.get('EVALUATOR_FORCE_CPU', '')
    
    os.environ['EVALUATOR_EMBEDDING_STRATEGY'] = 'fallback'
    os.environ['EVALUATOR_FORCE_CPU'] = 'true'
    
    yield
    
    # Restore original values
    if original_strategy:
        os.environ['EVALUATOR_EMBEDDING_STRATEGY'] = original_strategy
    else:
        os.environ.pop('EVALUATOR_EMBEDDING_STRATEGY', None)
        
    if original_cpu:
        os.environ['EVALUATOR_FORCE_CPU'] = original_cpu
    else:
        os.environ.pop('EVALUATOR_FORCE_CPU', None)


@pytest.fixture
def sample_domain_files(project_root):
    """Provide paths to sample domain files for testing"""
    domain_dir = project_root / "domains"
    return {
        "reasoning_easy": domain_dir / "reasoning" / "base_models" / "easy.json",
        "reasoning_instruct": domain_dir / "reasoning" / "instruct_models" / "easy.json",
        "creativity_base": domain_dir / "creativity" / "base_models" / "easy.json",
    }


@pytest.fixture
def mock_llm_server():
    """Mock LLM server responses for testing"""
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = BenchmarkTestHelper.create_mock_api_response()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def mock_evaluator_system():
    """Mock the entire evaluator system for isolated testing"""
    with patch('sys.modules') as mock_modules:
        # Mock key evaluator modules
        mock_enhanced_evaluator = Mock()
        mock_enhanced_evaluator.EnhancedUniversalEvaluator.return_value.evaluate.return_value = {
            "exact_match_score": 0.0,
            "partial_match_score": 0.75,
            "semantic_similarity_score": 0.0,
            "overall_score": 75.0
        }
        
        mock_modules.update({
            'evaluator.subjects.enhanced_universal_evaluator': mock_enhanced_evaluator,
            'evaluator.advanced.semantic_coherence': Mock(),
            'evaluator.advanced.entropy_calculator': Mock(),
            'evaluator.advanced.model_loader': Mock()
        })
        
        yield mock_modules


@pytest.fixture
def captured_logs(caplog):
    """Convenient access to captured log records"""
    caplog.set_level("INFO")
    yield caplog


class TestResultValidator:
    """Validator for test results structure"""
    
    @staticmethod
    def validate_test_result(result: Dict[str, Any]):
        """Validate test result structure"""
        BenchmarkTestHelper.assert_valid_test_result(result)


@pytest.fixture
def result_validator():
    """Provide test result validator"""
    return TestResultValidator()