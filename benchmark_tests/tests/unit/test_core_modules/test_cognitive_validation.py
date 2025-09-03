#!/usr/bin/env python3
"""
Unit Tests for EnhancedCognitiveValidator

Tests the cognitive validation functionality including domain testing, API interactions,
statistical analysis, and comprehensive cognitive profiling.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open, call
import json
import sys
import statistics
from pathlib import Path
from datetime import datetime

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "core"))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from core.cognitive_validation import EnhancedCognitiveValidator
from core.results_manager import CognitiveProfile
from core.cognitive_evaluation_pipeline import CognitiveEvaluationResult


class TestEnhancedCognitiveValidator(unittest.TestCase):
    """Test EnhancedCognitiveValidator core functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock the dependencies to avoid real API calls and file I/O
        with patch('core.cognitive_validation.TestResultsManager') as mock_results_manager, \
             patch('core.cognitive_validation.CognitiveEvaluationPipeline') as mock_evaluation_pipeline:
            
            self.mock_results_manager = mock_results_manager.return_value
            self.mock_evaluation_pipeline = mock_evaluation_pipeline.return_value
            
            self.validator = EnhancedCognitiveValidator()
    
    def test_initialization(self):
        """Test validator initialization with proper configuration"""
        # Test model configuration
        expected_config = {
            "name": "Qwen3-30B-A3B-UD-Q6_K_XL",
            "endpoint": "http://127.0.0.1:8004/v1/completions",
            "model_path": "/app/models/gguf/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-UD-Q6_K_XL.gguf",
            "max_context": 65536
        }
        self.assertEqual(self.validator.model_config, expected_config)
        
        # Test optimal tokens configuration
        expected_tokens = {'easy': 400, 'medium': 500, 'hard': 600}
        self.assertEqual(self.validator.optimal_tokens, expected_tokens)
        
        # Test dependencies initialization
        self.assertIsNotNone(self.validator.results_manager)
        self.assertIsNotNone(self.validator.evaluation_pipeline)
        self.assertIsNone(self.validator.current_run_dir)
    
    @patch('core.cognitive_validation.Path')
    def test_get_easy_domains(self, mock_path):
        """Test getting domains with easy.json files"""
        # Mock directory structure
        mock_domains_dir = Mock()
        mock_path.return_value = mock_domains_dir
        
        # Mock domain directories with MagicMock to support magic methods
        mock_reasoning = MagicMock()
        mock_reasoning.is_dir.return_value = True
        mock_reasoning.name = 'reasoning'
        
        mock_creativity = MagicMock()
        mock_creativity.is_dir.return_value = True
        mock_creativity.name = 'creativity'
        
        mock_invalid = MagicMock()
        mock_invalid.is_dir.return_value = False
        mock_invalid.name = 'invalid'
        
        mock_domains_dir.iterdir.return_value = [mock_reasoning, mock_creativity, mock_invalid]
        
        # Mock easy.json file existence - configure path division chain
        # For reasoning: domain_path / "base_models" / "easy.json" -> exists() = True
        mock_reasoning_easy_json = MagicMock()
        mock_reasoning_easy_json.exists.return_value = True
        mock_reasoning.__truediv__.return_value.__truediv__.return_value = mock_reasoning_easy_json
        
        # For creativity: domain_path / "base_models" / "easy.json" -> exists() = False  
        mock_creativity_easy_json = MagicMock()
        mock_creativity_easy_json.exists.return_value = False
        mock_creativity.__truediv__.return_value.__truediv__.return_value = mock_creativity_easy_json
        
        # Test method
        domains = self.validator.get_easy_domains()
        
        # Should only return reasoning (creativity has no easy.json)
        self.assertEqual(domains, ['reasoning'])
        
        # Verify path operations
        mock_path.assert_called_once_with("domains")
        mock_reasoning.__truediv__.assert_called_with("base_models")
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('core.cognitive_validation.Path')
    @patch('core.cognitive_validation.json.load')
    def test_load_domain_tests_success(self, mock_json_load, mock_path, mock_file):
        """Test successful loading of domain tests"""
        # Mock domain data
        test_data = {
            'tests': [
                {'id': 'test1', 'prompt': 'Test prompt 1'},
                {'id': 'test2', 'prompt': 'Test prompt 2'},
                {'id': 'test3', 'prompt': 'Test prompt 3'}
            ]
        }
        mock_json_load.return_value = test_data
        
        # Mock file path existence
        mock_easy_file = Mock()
        mock_easy_file.exists.return_value = True
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = mock_easy_file
        
        # Test method
        result = self.validator.load_domain_tests('reasoning', max_tests=2)
        
        # Should return first 2 tests
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['id'], 'test1')
        self.assertEqual(result[1]['id'], 'test2')
    
    @patch('core.cognitive_validation.Path')
    def test_load_domain_tests_file_not_found(self, mock_path):
        """Test loading domain tests when file doesn't exist"""
        # Mock file path non-existence
        mock_easy_file = Mock()
        mock_easy_file.exists.return_value = False
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = mock_easy_file
        
        # Test method
        result = self.validator.load_domain_tests('nonexistent')
        
        # Should return empty list
        self.assertEqual(result, [])
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('core.cognitive_validation.Path')
    @patch('core.cognitive_validation.json.load')
    def test_load_domain_tests_json_error(self, mock_json_load, mock_path, mock_file):
        """Test loading domain tests with JSON parsing error"""
        # Mock file exists but JSON load fails
        mock_easy_file = Mock()
        mock_easy_file.exists.return_value = True
        mock_path.return_value.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value = mock_easy_file
        mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        # Test method
        result = self.validator.load_domain_tests('reasoning')
        
        # Should return empty list on error
        self.assertEqual(result, [])
    
    @patch('core.cognitive_validation.requests.post')
    @patch('core.cognitive_validation.time.time')
    def test_make_api_request_success(self, mock_time, mock_post):
        """Test successful API request"""
        # Mock time for response time calculation
        mock_time.side_effect = [1000.0, 1001.5]  # 1.5 second response
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'choices': [{'text': 'Generated response', 'finish_reason': 'stop'}],
            'usage': {'completion_tokens': 50, 'total_tokens': 100}
        }
        mock_post.return_value = mock_response
        
        # Test data
        test_data = {
            'prompt': 'Test prompt',
            'temperature': 0.5,
            'top_p': 0.9
        }
        
        # Test method
        result = self.validator.make_api_request(test_data)
        
        # Verify result structure
        expected_result = {
            'response_text': 'Generated response',
            'finish_reason': 'stop',
            'completion_tokens': 50,
            'total_tokens': 100,
            'response_time_seconds': 1.5
        }
        self.assertEqual(result, expected_result)
        
        # Verify API call was made correctly
        expected_payload = {
            "model": "/app/models/gguf/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-UD-Q6_K_XL.gguf",
            "prompt": 'Test prompt',
            "max_tokens": 400,
            "temperature": 0.5,
            "top_p": 0.9,
            "stream": False
        }
        mock_post.assert_called_once_with(
            "http://127.0.0.1:8004/v1/completions",
            json=expected_payload,
            timeout=30
        )
    
    @patch('core.cognitive_validation.requests.post')
    def test_make_api_request_timeout(self, mock_post):
        """Test API request timeout handling"""
        # Mock timeout exception
        mock_post.side_effect = Exception("Connection timeout")
        
        test_data = {'prompt': 'Test prompt'}
        
        # Test method
        result = self.validator.make_api_request(test_data)
        
        # Should return None on error
        self.assertIsNone(result)
    
    @patch('core.cognitive_validation.requests.post')
    def test_make_api_request_http_error(self, mock_post):
        """Test API request HTTP error handling"""
        # Mock HTTP error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 500 Error")
        mock_post.return_value = mock_response
        
        test_data = {'prompt': 'Test prompt'}
        
        # Test method
        result = self.validator.make_api_request(test_data)
        
        # Should return None on error
        self.assertIsNone(result)
    
    @patch('core.cognitive_validation.time.sleep')
    @patch.object(EnhancedCognitiveValidator, 'make_api_request')
    @patch.object(EnhancedCognitiveValidator, 'load_domain_tests')
    @patch.object(EnhancedCognitiveValidator, '_generate_domain_insights')
    def test_test_domain_success(self, mock_insights, mock_load_tests, mock_api_request, mock_sleep):
        """Test successful domain testing"""
        # Mock test data
        mock_load_tests.return_value = [
            {'id': 'test1', 'prompt': 'Prompt 1'},
            {'id': 'test2', 'prompt': 'Prompt 2'}
        ]
        
        # Mock API responses
        mock_api_request.side_effect = [
            {'response_text': 'Response 1', 'finish_reason': 'stop', 'completion_tokens': 50, 'total_tokens': 100, 'response_time_seconds': 1.0},
            {'response_text': 'Response 2', 'finish_reason': 'stop', 'completion_tokens': 60, 'total_tokens': 110, 'response_time_seconds': 1.2}
        ]
        
        # Mock evaluation results - properly configure cognitive_subscores as actual dictionaries
        mock_eval_result1 = Mock(spec=CognitiveEvaluationResult)
        mock_eval_result1.overall_score = 75.0
        mock_eval_result1.confidence_score = 0.8
        mock_eval_result1.cognitive_domain = 'reasoning'
        # Set cognitive_subscores as actual dict so .items() works
        subscores1 = {'logical_reasoning': 80.0, 'pattern_recognition': 70.0}
        mock_eval_result1.cognitive_subscores = subscores1
        mock_eval_result1.__dict__ = {'overall_score': 75.0, 'confidence_score': 0.8, 'cognitive_subscores': subscores1}
        
        mock_eval_result2 = Mock(spec=CognitiveEvaluationResult)
        mock_eval_result2.overall_score = 80.0
        mock_eval_result2.confidence_score = 0.85
        mock_eval_result2.cognitive_domain = 'reasoning'
        # Set cognitive_subscores as actual dict so .items() works
        subscores2 = {'logical_reasoning': 85.0, 'pattern_recognition': 75.0}
        mock_eval_result2.cognitive_subscores = subscores2
        mock_eval_result2.__dict__ = {'overall_score': 80.0, 'confidence_score': 0.85, 'cognitive_subscores': subscores2}
        
        self.mock_evaluation_pipeline.evaluate_response.side_effect = [mock_eval_result1, mock_eval_result2]
        
        # Mock domain insights
        mock_insights.return_value = {'strengths': ['Strong reasoning'], 'concerns': []}
        
        # Set current run directory
        self.validator.current_run_dir = "/tmp/test_run"
        
        # Test method
        result = self.validator.test_domain('reasoning')
        
        # Verify result structure
        self.assertTrue(result['success'])
        self.assertEqual(result['domain'], 'reasoning')
        self.assertEqual(result['test_count'], 2)
        self.assertEqual(result['successful_tests'], 2)
        self.assertAlmostEqual(result['overall_score'], 77.5)  # Mean of 75 and 80
        self.assertAlmostEqual(result['confidence'], 0.825)  # Mean of 0.8 and 0.85
        
        # Verify calls were made
        mock_load_tests.assert_called_once_with('reasoning', max_tests=8)
        self.assertEqual(mock_api_request.call_count, 2)
        self.assertEqual(self.mock_evaluation_pipeline.evaluate_response.call_count, 2)
        self.assertEqual(self.mock_results_manager.save_test_response.call_count, 2)
    
    @patch.object(EnhancedCognitiveValidator, 'load_domain_tests')
    def test_test_domain_no_tests(self, mock_load_tests):
        """Test domain testing with no tests loaded"""
        mock_load_tests.return_value = []
        
        # Test method
        result = self.validator.test_domain('empty_domain')
        
        # Should return failure result
        self.assertFalse(result['success'])
        self.assertEqual(result['domain'], 'empty_domain')
        self.assertEqual(result['error'], 'No tests loaded')
    
    def test_generate_domain_insights(self):
        """Test domain insights generation"""
        # Mock evaluation results
        mock_result1 = Mock()
        mock_result1.behavioral_patterns = ['pattern1']
        mock_result1.pattern_strength = 0.8
        mock_result1.consistency_measure = 0.75
        mock_result1.cognitive_subscores = {'reasoning': 80.0, 'memory': 60.0}
        
        mock_result2 = Mock()
        mock_result2.behavioral_patterns = ['pattern2']
        mock_result2.pattern_strength = 0.7
        mock_result2.consistency_measure = 0.85
        mock_result2.cognitive_subscores = {'reasoning': 85.0, 'memory': 45.0}
        
        results = [mock_result1, mock_result2]
        
        # Test method
        insights = self.validator._generate_domain_insights(results)
        
        # Verify insights structure
        self.assertIn('behavioral_consistency', insights)
        self.assertAlmostEqual(insights['behavioral_consistency'], 0.8)  # Mean of 0.75 and 0.85
        
        self.assertIn('strengths', insights)
        self.assertIn('concerns', insights)
        self.assertIn('Reasoning: 82.5', insights['strengths'])  # Mean of 80 and 85 (≥75 threshold)
        # Memory average is 52.5, which is > 50, so it won't be in concerns (concerns are ≤50)
    
    def test_generate_domain_insights_empty(self):
        """Test domain insights generation with empty results"""
        insights = self.validator._generate_domain_insights([])
        
        # Should return default insights structure
        expected_insights = {
            'cognitive_patterns': [],
            'strengths': [],
            'concerns': [],
            'behavioral_consistency': 0.0
        }
        self.assertEqual(insights, expected_insights)
    
    @patch('builtins.print')
    @patch.object(EnhancedCognitiveValidator, '_save_validation_results')
    @patch.object(EnhancedCognitiveValidator, '_display_validation_summary')
    @patch.object(EnhancedCognitiveValidator, 'test_domain')
    @patch.object(EnhancedCognitiveValidator, 'get_easy_domains')
    @patch('core.cognitive_validation.time.sleep')
    def test_run_comprehensive_validation(self, mock_sleep, mock_get_domains, mock_test_domain, 
                                        mock_display, mock_save, mock_print):
        """Test comprehensive validation workflow"""
        # Mock domains
        mock_get_domains.return_value = ['reasoning', 'creativity']
        
        # Mock domain test results
        mock_test_domain.side_effect = [
            {'success': True, 'domain': 'reasoning', 'overall_score': 75.0},
            {'success': True, 'domain': 'creativity', 'overall_score': 80.0}
        ]
        
        # Mock cognitive profile
        mock_profile = Mock(spec=CognitiveProfile)
        self.mock_results_manager.analyze_cognitive_patterns.return_value = mock_profile
        self.mock_results_manager.create_run_directory.return_value = "/tmp/test_run_123"
        
        # Test method
        result = self.validator.run_comprehensive_validation()
        
        # Verify workflow execution
        mock_get_domains.assert_called_once()
        self.assertEqual(mock_test_domain.call_count, 2)
        mock_test_domain.assert_any_call('reasoning')
        mock_test_domain.assert_any_call('creativity')
        
        self.mock_results_manager.create_run_directory.assert_called_once()
        self.mock_results_manager.analyze_cognitive_patterns.assert_called_once_with("/tmp/test_run_123")
        
        mock_display.assert_called_once()
        mock_save.assert_called_once()
        
        # Should return cognitive profile
        self.assertEqual(result, mock_profile)
    
    @patch('builtins.print')
    def test_display_validation_summary(self, mock_print):
        """Test validation summary display"""
        # Mock domain results
        domain_results = [
            {'domain': 'reasoning', 'overall_score': 75.0, 'confidence': 0.8},
            {'domain': 'creativity', 'overall_score': 80.0, 'confidence': 0.85}
        ]
        
        # Mock cognitive profile
        mock_profile = Mock(spec=CognitiveProfile)
        mock_profile.model_name = "Test Model"
        mock_profile.reasoning_score = 77.5
        mock_profile.memory_score = 70.0
        mock_profile.creativity_score = 82.0
        mock_profile.social_score = 65.0
        mock_profile.integration_score = 75.0
        mock_profile.strengths = ["Strong reasoning"]
        mock_profile.weaknesses = ["Memory gaps"]
        mock_profile.blind_spots = ["Cultural bias"]
        
        # Test method
        self.validator._display_validation_summary(domain_results, mock_profile)
        
        # Should have made multiple print calls for formatting
        self.assertTrue(mock_print.called)
        # Verify some expected content was printed by checking all call arguments
        all_prints = []
        for call in mock_print.call_args_list:
            if call.args:
                all_prints.append(str(call.args[0]))
            elif call.kwargs:
                all_prints.extend(str(v) for v in call.kwargs.values())
        
        summary_text = " ".join(all_prints)
        # Check if any expected content appears in the print calls
        expected_phrases = ["COGNITIVE VALIDATION RESULTS", "Test Model", "reasoning", "creativity"]
        found_any = any(phrase in summary_text for phrase in expected_phrases)
        self.assertTrue(found_any, f"Expected content not found in print output: {summary_text}")
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('core.cognitive_validation.json.dump')
    @patch('core.cognitive_validation.datetime')
    @patch('core.cognitive_validation.Path')
    def test_save_validation_results(self, mock_path, mock_datetime, mock_json_dump, mock_file):
        """Test saving validation results to file"""
        # Mock datetime
        mock_datetime.now.return_value.strftime.return_value = "20231201_143000"
        mock_datetime.now.return_value.isoformat.return_value = "2023-12-01T14:30:00"
        
        # Mock path operations
        mock_summary_file = Mock()
        mock_path.return_value.__truediv__.return_value = mock_summary_file
        
        # Set current run directory
        self.validator.current_run_dir = "/tmp/test_run"
        
        # Mock data
        domain_results = [{'domain': 'reasoning', 'score': 75.0}]
        mock_profile = Mock()
        mock_profile.__dict__ = {'reasoning_score': 75.0}
        
        # Test method
        self.validator._save_validation_results(domain_results, mock_profile)
        
        # Verify file operations
        mock_file.assert_called_once_with(mock_summary_file, 'w')
        mock_json_dump.assert_called_once()
        
        # Verify JSON content structure
        args, kwargs = mock_json_dump.call_args
        saved_data = args[0]
        self.assertIn('timestamp', saved_data)
        self.assertIn('model_configuration', saved_data)
        self.assertIn('domain_results', saved_data)
        self.assertIn('cognitive_profile', saved_data)
        self.assertEqual(saved_data['domain_results'], domain_results)


if __name__ == '__main__':
    unittest.main()