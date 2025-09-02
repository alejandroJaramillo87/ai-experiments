#!/usr/bin/env python3
"""
Unit Tests for Benchmarking Engine

Tests the multi-model comparative benchmarking system including pattern analysis,
statistical validation, and behavioral signature analysis.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "core"))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from core.benchmarking_engine import (
    MultiModelBenchmarking, ModelEndpoint, ComparativeResult, BenchmarkSummary
)


class TestModelEndpoint(unittest.TestCase):
    """Test ModelEndpoint dataclass functionality"""
    
    def test_model_endpoint_creation(self):
        """Test ModelEndpoint creation with required parameters"""
        endpoint = ModelEndpoint(
            name="Test-Model",
            endpoint_url="http://localhost:8004/v1/completions",
            model_path="/test/model/path.gguf"
        )
        
        self.assertEqual(endpoint.name, "Test-Model")
        self.assertEqual(endpoint.endpoint_url, "http://localhost:8004/v1/completions")
        self.assertEqual(endpoint.model_path, "/test/model/path.gguf")
        self.assertEqual(endpoint.max_context, 32000)  # Default value
    
    def test_model_endpoint_optimal_tokens(self):
        """Test proven token optimization strategy initialization"""
        endpoint = ModelEndpoint(
            name="Test-Model",
            endpoint_url="http://localhost:8004/v1/completions",
            model_path="/test/model/path.gguf"
        )
        
        # Should use proven token strategy
        expected_tokens = {'easy': 400, 'medium': 500, 'hard': 600}
        self.assertEqual(endpoint.optimal_tokens, expected_tokens)
    
    def test_model_endpoint_custom_tokens(self):
        """Test custom token configuration"""
        custom_tokens = {'easy': 300, 'medium': 450, 'hard': 550}
        endpoint = ModelEndpoint(
            name="Custom-Model",
            endpoint_url="http://localhost:8005/v1/completions",
            model_path="/test/custom.gguf",
            optimal_tokens=custom_tokens
        )
        
        self.assertEqual(endpoint.optimal_tokens, custom_tokens)


class TestMultiModelBenchmarking(unittest.TestCase):
    """Test MultiModelBenchmarking core functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.benchmarker = MultiModelBenchmarking()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_benchmarker_initialization(self):
        """Test MultiModelBenchmarking initialization"""
        self.assertIsNotNone(self.benchmarker.pattern_evaluator)
        self.assertIsInstance(self.benchmarker.model_endpoints, dict)
        self.assertIsInstance(self.benchmarker.comparative_results, list)
        self.assertIsInstance(self.benchmarker.calibration_criteria, dict)
        
        # Check default model endpoints
        self.assertIn('qwen3_30b', self.benchmarker.model_endpoints)
        
        # Check calibration criteria
        self.assertIn('min_quality_score', self.benchmarker.calibration_criteria)
        self.assertEqual(self.benchmarker.calibration_criteria['min_quality_score'], 65.0)
    
    def test_add_model_endpoint(self):
        """Test adding new model endpoints"""
        initial_count = len(self.benchmarker.model_endpoints)
        
        new_endpoint = ModelEndpoint(
            name="New-Test-Model",
            endpoint_url="http://localhost:8005/v1/completions", 
            model_path="/test/new-model.gguf"
        )
        
        self.benchmarker.add_model_endpoint(new_endpoint)
        
        self.assertEqual(len(self.benchmarker.model_endpoints), initial_count + 1)
        self.assertIn('new_test_model', self.benchmarker.model_endpoints)
        self.assertEqual(self.benchmarker.model_endpoints['new_test_model'].name, "New-Test-Model")
    
    @patch('core.benchmarking_engine.Path')
    def test_load_domain_tests(self, mock_path_class):
        """Test loading domain tests from filesystem"""
        # Mock domain file structure with proper Path behavior
        mock_domain_file = Mock()
        mock_domain_file.name = 'easy.json'  # Match actual filename pattern
        
        test_data = {
            'tests': [
                {'id': 'test_001', 'prompt': 'Test prompt 1'},
                {'id': 'test_002', 'prompt': 'Test prompt 2'},
                {'id': 'test_003', 'prompt': 'Test prompt 3'}
            ]
        }
        
        # Mock Path('domains') constructor and glob method
        mock_domains_path = Mock()
        mock_domains_path.glob.return_value = [mock_domain_file]
        mock_path_class.return_value = mock_domains_path
        
        # Mock file opening with proper context manager
        mock_file = Mock()
        mock_file_content = Mock()
        mock_file_content.read.return_value = json.dumps(test_data)
        mock_file.__enter__ = Mock(return_value=mock_file_content)
        mock_file.__exit__ = Mock(return_value=False)
        
        with patch('builtins.open', return_value=mock_file):
            domain_tests = self.benchmarker._load_domain_tests(['reasoning'], 2)
        
        self.assertIn('reasoning', domain_tests)
        self.assertEqual(len(domain_tests['reasoning']), 2)  # Limited to 2 tests
    
    def test_determine_test_difficulty(self):
        """Test test difficulty determination logic"""
        # Test explicit difficulty metadata
        test_data = {'difficulty': 'hard', 'prompt': 'Simple prompt'}
        difficulty = self.benchmarker._determine_test_difficulty(test_data)
        self.assertEqual(difficulty, 'hard')
        
        # Test heuristic difficulty - hard
        test_data = {'prompt': 'Complex analysis of multiple interconnected systems with detailed evaluation' * 3}
        difficulty = self.benchmarker._determine_test_difficulty(test_data)
        self.assertEqual(difficulty, 'hard')
        
        # Test heuristic difficulty - medium
        test_data = {'prompt': 'Please analyze and compare the following concepts with evaluation'}
        difficulty = self.benchmarker._determine_test_difficulty(test_data)
        self.assertEqual(difficulty, 'medium')
        
        # Test heuristic difficulty - easy
        test_data = {'prompt': 'Simple question'}
        difficulty = self.benchmarker._determine_test_difficulty(test_data)
        self.assertEqual(difficulty, 'easy')
    
    @patch('core.benchmarking_engine.statistics')
    def test_calculate_calibration_score(self, mock_statistics):
        """Test calibration score calculation methodology"""
        # Mock statistics functions
        mock_statistics.mean.return_value = 75.0
        mock_statistics.stdev.return_value = 5.0
        
        # Mock pattern result
        mock_pattern_result = Mock()
        mock_pattern_result.response_consistency = 0.8
        mock_pattern_result.pattern_adherence = 0.75
        mock_pattern_result.quality_indicators = {
            'coherence_score': 0.8,
            'fluency_score': 0.9,
            'engagement_score': 0.7
        }
        
        # Mock sample results
        sample_results = [
            {'completion_tokens': 400, 'total_tokens': 450},
            {'completion_tokens': 420, 'total_tokens': 470},
            {'completion_tokens': 410, 'total_tokens': 460}
        ]
        
        score = self.benchmarker._calculate_calibration_score(sample_results, mock_pattern_result)
        
        # Score should be between 0 and 100
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertIsInstance(score, float)
    
    def test_calculate_comparative_metrics(self):
        """Test comparative metrics calculation"""
        # Mock pattern result
        mock_pattern_result = Mock()
        mock_pattern_result.behavioral_signature = {
            'repetition_tendency': 0.1,
            'vocabulary_richness': 0.8,
            'response_style': 'analytical'
        }
        mock_pattern_result.quality_indicators = {
            'coherence_score': 0.8,
            'fluency_score': 0.9,
            'engagement_score': 0.7
        }
        
        # Mock sample results
        sample_results = [
            {'completion_tokens': 400, 'finish_reason': 'stop'},
            {'completion_tokens': 420, 'finish_reason': 'stop'},
            {'completion_tokens': 410, 'finish_reason': 'length'}
        ]
        
        metrics = self.benchmarker._calculate_comparative_metrics(
            sample_results, mock_pattern_result, "test-model"
        )
        
        # Verify metric structure
        expected_keys = [
            'avg_response_length', 'completion_consistency', 
            'repetition_control', 'vocabulary_diversity', 'overall_quality'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], (int, float))
    
    @patch('core.benchmarking_engine.requests.post')
    def test_make_api_request(self, mock_post):
        """Test API request functionality"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'choices': [{'text': 'Test response', 'finish_reason': 'stop'}],
            'usage': {'completion_tokens': 50, 'total_tokens': 75}
        }
        mock_post.return_value = mock_response
        
        model_config = ModelEndpoint(
            name="Test-Model",
            endpoint_url="http://localhost:8004/v1/completions",
            model_path="/test/model.gguf"
        )
        
        test_data = {'prompt': 'Test prompt', 'temperature': 0.0, 'top_p': 1.0}
        
        result = self.benchmarker._make_api_request(model_config, test_data, 400)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['response_text'], 'Test response')
        self.assertEqual(result['finish_reason'], 'stop')
        self.assertEqual(result['completion_tokens'], 50)
    
    @patch('core.benchmarking_engine.requests.post')
    def test_make_api_request_failure(self, mock_post):
        """Test API request failure handling"""
        # Mock API failure
        mock_post.side_effect = Exception("Connection error")
        
        model_config = ModelEndpoint(
            name="Test-Model", 
            endpoint_url="http://localhost:8004/v1/completions",
            model_path="/test/model.gguf"
        )
        
        test_data = {'prompt': 'Test prompt'}
        
        result = self.benchmarker._make_api_request(model_config, test_data, 400)
        
        self.assertIsNone(result)


class TestBenchmarkSummary(unittest.TestCase):
    """Test BenchmarkSummary dataclass and summary generation"""
    
    def test_benchmark_summary_creation(self):
        """Test BenchmarkSummary creation"""
        summary = BenchmarkSummary(
            total_models=2,
            total_tests=10,
            model_rankings={'model_a': 75.0, 'model_b': 68.0},
            behavioral_signatures={'model_a': {'style': 'analytical'}},
            domain_performance={'reasoning': {'model_a': 78.0, 'model_b': 65.0}},
            comparative_insights=['Model A performs better on reasoning tasks']
        )
        
        self.assertEqual(summary.total_models, 2)
        self.assertEqual(summary.total_tests, 10)
        self.assertIn('model_a', summary.model_rankings)
        self.assertIn('reasoning', summary.domain_performance)
    
    def test_generate_comparative_insights(self):
        """Test comparative insights generation"""
        benchmarker = MultiModelBenchmarking()
        
        rankings = {'model_a': 75.0, 'model_b': 68.0, 'model_c': 72.0}
        signatures = {
            'model_a': {'response_style': 'analytical', 'verbosity_level': 'concise'},
            'model_b': {'response_style': 'creative', 'verbosity_level': 'verbose'},
            'model_c': {'response_style': 'balanced', 'verbosity_level': 'medium'}
        }
        domain_perf = {
            'reasoning': {'model_a': 78.0, 'model_b': 65.0, 'model_c': 70.0},
            'creativity': {'model_a': 72.0, 'model_b': 75.0, 'model_c': 74.0}
        }
        
        insights = benchmarker._generate_comparative_insights(rankings, signatures, domain_perf)
        
        self.assertIsInstance(insights, list)
        self.assertGreater(len(insights), 0)
        
        # Should identify top performer
        top_performer_mentioned = any('model_a' in insight for insight in insights)
        self.assertTrue(top_performer_mentioned)


class TestBenchmarkingIntegration(unittest.TestCase):
    """Test benchmarking engine integration with other components"""
    
    def test_pattern_evaluator_integration(self):
        """Test integration with PatternBasedEvaluator"""
        benchmarker = MultiModelBenchmarking()
        
        # Should have pattern evaluator initialized
        self.assertIsNotNone(benchmarker.pattern_evaluator)
        
        # Should be able to call pattern evaluation methods
        self.assertTrue(hasattr(benchmarker.pattern_evaluator, 'evaluate_patterns'))
    
    def test_calibration_criteria_alignment(self):
        """Test that calibration criteria align with proven methodologies"""
        benchmarker = MultiModelBenchmarking()
        criteria = benchmarker.calibration_criteria
        
        # Should have quality thresholds based on 75% success validation
        self.assertGreaterEqual(criteria['min_quality_score'], 60.0)
        self.assertLessEqual(criteria['min_quality_score'], 70.0)
        
        # Should have statistical validation requirements
        self.assertGreaterEqual(criteria['samples_per_test'], 3)
        
        # Should have consistency requirements
        self.assertIn('min_consistency', criteria)
        self.assertGreaterEqual(criteria['min_consistency'], 0.7)


if __name__ == '__main__':
    print("🚀 Running Benchmarking Engine Unit Tests")
    print("=" * 50)
    
    unittest.main(verbosity=2)