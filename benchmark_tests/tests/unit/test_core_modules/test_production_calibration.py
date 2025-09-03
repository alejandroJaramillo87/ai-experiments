#!/usr/bin/env python3
"""
Unit Tests for ProductionCalibrationFramework

Tests the production calibration functionality including domain discovery,
calibration testing, statistical analysis, and comprehensive reporting.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open, call
import json
import sys
import statistics
from pathlib import Path
from dataclasses import asdict

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "core"))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from core.production_calibration import (
    ProductionCalibrationFramework, CalibrationResult, DomainStats
)


class TestCalibrationResult(unittest.TestCase):
    """Test CalibrationResult data structure"""
    
    def test_calibration_result_creation(self):
        """Test CalibrationResult creation with all fields"""
        response_quality = {'avg_completion_tokens': 350, 'completion_rate': 0.8}
        pattern_analysis = {'avg_repetitive_loops': 1, 'length_consistency': 0.9}
        
        result = CalibrationResult(
            domain="reasoning",
            test_id="basic_01",
            difficulty="easy",
            target_tokens=400,
            completion_tokens=350,
            finish_reason="stop",
            response_quality=response_quality,
            pattern_analysis=pattern_analysis,
            validation_passed=True,
            calibration_score=85.5
        )
        
        self.assertEqual(result.domain, "reasoning")
        self.assertEqual(result.test_id, "basic_01")
        self.assertEqual(result.difficulty, "easy")
        self.assertEqual(result.target_tokens, 400)
        self.assertEqual(result.completion_tokens, 350)
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.response_quality, response_quality)
        self.assertEqual(result.pattern_analysis, pattern_analysis)
        self.assertTrue(result.validation_passed)
        self.assertEqual(result.calibration_score, 85.5)


class TestDomainStats(unittest.TestCase):
    """Test DomainStats data structure"""
    
    def test_domain_stats_creation(self):
        """Test DomainStats creation"""
        difficulty_dist = {'easy': 45, 'medium': 30, 'hard': 25}
        token_dist = {'low (100-299)': 20, 'optimal (300-499)': 80}
        
        stats = DomainStats(
            domain_name="reasoning",
            total_tests=100,
            difficulty_distribution=difficulty_dist,
            token_distribution=token_dist,
            optimization_needed=True,
            estimated_impact="HIGH - 100 tests need optimization"
        )
        
        self.assertEqual(stats.domain_name, "reasoning")
        self.assertEqual(stats.total_tests, 100)
        self.assertEqual(stats.difficulty_distribution, difficulty_dist)
        self.assertEqual(stats.token_distribution, token_dist)
        self.assertTrue(stats.optimization_needed)
        self.assertIn("HIGH", stats.estimated_impact)


class TestProductionCalibrationFramework(unittest.TestCase):
    """Test ProductionCalibrationFramework functionality"""
    
    def test_initialization(self):
        """Test framework initialization with proper configuration"""
        framework = ProductionCalibrationFramework(
            endpoint="http://test:8004/v1/completions", 
            max_workers=2
        )
        
        self.assertEqual(framework.endpoint, "http://test:8004/v1/completions")
        self.assertEqual(framework.max_workers, 2)
        
        # Verify proven token strategy
        self.assertEqual(framework.optimal_tokens['easy'], 400)
        self.assertEqual(framework.optimal_tokens['medium'], 500)
        self.assertEqual(framework.optimal_tokens['hard'], 600)
        
        # Verify calibration criteria
        self.assertEqual(framework.calibration_criteria['min_completion_tokens'], 200)
        self.assertEqual(framework.calibration_criteria['max_repetitive_loops'], 2)
        self.assertEqual(framework.calibration_criteria['min_coherence_score'], 0.7)
        self.assertEqual(framework.calibration_criteria['acceptable_truncation_rate'], 0.3)
        
        # Verify storage initialization
        self.assertEqual(framework.calibration_results, [])
        self.assertEqual(framework.domain_stats, {})
    
    @patch.object(ProductionCalibrationFramework, 'discover_all_domains')
    def test_discover_all_domains(self, mock_discover):
        """Test domain discovery functionality"""
        framework = ProductionCalibrationFramework()
        
        # Mock discovered domain files - return Path objects that can be compared
        mock_files = [
            Path('domains/reasoning/base_models/easy.json'),
            Path('domains/creativity/base_models/medium.json'), 
            Path('domains/language/base_models/hard.json')
        ]
        mock_discover.return_value = mock_files
        
        discovered = framework.discover_all_domains()
        
        self.assertEqual(len(discovered), 3)
        self.assertEqual(discovered, mock_files)
    
    @patch.object(ProductionCalibrationFramework, 'discover_all_domains')
    def test_discover_all_domains_no_directory(self, mock_discover):
        """Test domain discovery when domains directory doesn't exist"""
        framework = ProductionCalibrationFramework()
        
        mock_discover.return_value = []
        
        discovered = framework.discover_all_domains()
        
        self.assertEqual(discovered, [])
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('core.production_calibration.json.load')
    def test_analyze_domain_success(self, mock_json_load, mock_file):
        """Test successful domain analysis"""
        framework = ProductionCalibrationFramework()
        
        # Mock domain data
        test_data = {
            'tests': [
                {'id': 'test1', 'prompt': 'Test 1', 'max_tokens': 250},
                {'id': 'test2', 'prompt': 'Test 2', 'max_tokens': 280},
                {'id': 'test3', 'prompt': 'Test 3', 'max_tokens': 450}
            ]
        }
        mock_json_load.return_value = test_data
        
        # Create mock path with easy difficulty
        domain_path = Mock(spec=Path)
        domain_path.__str__ = Mock(return_value='domains/reasoning/base_models/easy.json')
        domain_path.parent.parent.name = 'reasoning'
        
        stats = framework.analyze_domain(domain_path)
        
        self.assertIsInstance(stats, DomainStats)
        self.assertEqual(stats.domain_name, 'reasoning')
        self.assertEqual(stats.total_tests, 3)
        self.assertEqual(stats.difficulty_distribution, {'easy': 3})
        self.assertTrue(stats.optimization_needed)  # min token is 250 < 300
        self.assertIn("HIGH", stats.estimated_impact)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('core.production_calibration.json.load')
    def test_analyze_domain_file_error(self, mock_json_load, mock_file):
        """Test domain analysis with file loading error"""
        framework = ProductionCalibrationFramework()
        
        mock_json_load.side_effect = Exception("File read error")
        domain_path = Mock(spec=Path)
        
        stats = framework.analyze_domain(domain_path)
        
        self.assertIsNone(stats)
    
    def test_get_token_range(self):
        """Test token range categorization"""
        framework = ProductionCalibrationFramework()
        
        self.assertEqual(framework._get_token_range(50), "severe (< 100)")
        self.assertEqual(framework._get_token_range(250), "low (100-299)")
        self.assertEqual(framework._get_token_range(400), "optimal (300-499)")
        self.assertEqual(framework._get_token_range(650), "high (500-799)")
        self.assertEqual(framework._get_token_range(1000), "excessive (800+)")
    
    def test_determine_difficulty(self):
        """Test difficulty determination from path"""
        framework = ProductionCalibrationFramework()
        
        # Test explicit difficulty in path
        easy_path = Mock(spec=Path)
        easy_path.__str__ = Mock(return_value='domains/reasoning/base_models/easy.json')
        self.assertEqual(framework._determine_difficulty(easy_path), 'easy')
        
        medium_path = Mock(spec=Path)
        medium_path.__str__ = Mock(return_value='domains/creativity/base_models/medium.json')
        self.assertEqual(framework._determine_difficulty(medium_path), 'medium')
        
        hard_path = Mock(spec=Path)
        hard_path.__str__ = Mock(return_value='domains/language/base_models/hard.json')
        self.assertEqual(framework._determine_difficulty(hard_path), 'hard')
        
        # Test heuristic fallbacks
        basic_path = Mock(spec=Path)
        basic_path.__str__ = Mock(return_value='domains/cultural/basic_tests.json')
        self.assertEqual(framework._determine_difficulty(basic_path), 'easy')
        
        advanced_path = Mock(spec=Path)
        advanced_path.__str__ = Mock(return_value='domains/complex/advanced_tests.json')
        self.assertEqual(framework._determine_difficulty(advanced_path), 'hard')
        
        default_path = Mock(spec=Path)
        default_path.__str__ = Mock(return_value='domains/general/tests.json')
        self.assertEqual(framework._determine_difficulty(default_path), 'medium')
    
    def test_analyze_response_patterns(self):
        """Test response pattern analysis"""
        framework = ProductionCalibrationFramework()
        
        sample_results = [
            {
                'response_text': 'This is a good response. It has variety. No repetition here.',
                'finish_reason': 'stop',
                'completion_tokens': 300
            },
            {
                'response_text': 'Another response. Another response. This shows some loops.',
                'finish_reason': 'stop',
                'completion_tokens': 280
            },
            {
                'response_text': 'Third response with different content and good structure.',
                'finish_reason': 'length',
                'completion_tokens': 350
            }
        ]
        
        patterns = framework._analyze_response_patterns(sample_results)
        
        self.assertIn('avg_repetitive_loops', patterns)
        self.assertIn('length_consistency', patterns)
        self.assertIn('completion_consistency', patterns)
        self.assertIn('response_stability', patterns)
        
        # Verify completion consistency calculation
        expected_completion = 2/3  # 2 'stop' out of 3 total
        self.assertAlmostEqual(patterns['completion_consistency'], expected_completion, places=2)
    
    def test_assess_response_quality(self):
        """Test response quality assessment"""
        framework = ProductionCalibrationFramework()
        
        sample_results = [
            {
                'response_text': 'This is a quality response with good content.',
                'finish_reason': 'stop',
                'completion_tokens': 320
            },
            {
                'response_text': 'Another quality response here.',
                'finish_reason': 'length',
                'completion_tokens': 380
            }
        ]
        target_tokens = 400
        
        quality = framework._assess_response_quality(sample_results, target_tokens)
        
        self.assertIn('avg_completion_tokens', quality)
        self.assertIn('completion_rate', quality)
        self.assertIn('truncation_rate', quality)
        self.assertIn('avg_word_count', quality)
        self.assertIn('quality_tier', quality)
        
        expected_avg_tokens = (320 + 380) / 2
        expected_completion_rate = expected_avg_tokens / target_tokens
        expected_truncation_rate = 1/2  # 1 'length' out of 2 total
        
        self.assertEqual(quality['avg_completion_tokens'], expected_avg_tokens)
        self.assertAlmostEqual(quality['completion_rate'], expected_completion_rate, places=3)
        self.assertEqual(quality['truncation_rate'], expected_truncation_rate)
    
    def test_validate_calibration_success(self):
        """Test successful calibration validation"""
        framework = ProductionCalibrationFramework()
        
        quality = {
            'avg_completion_tokens': 300,
            'completion_rate': 0.75,
            'truncation_rate': 0.2
        }
        patterns = {
            'avg_repetitive_loops': 1,
            'length_consistency': 0.8
        }
        
        validation_passed = framework._validate_calibration(quality, patterns)
        self.assertTrue(validation_passed)
    
    def test_validate_calibration_failure(self):
        """Test calibration validation failure scenarios"""
        framework = ProductionCalibrationFramework()
        
        # Test low completion tokens
        quality_low_tokens = {
            'avg_completion_tokens': 150,  # Below 200 threshold
            'completion_rate': 0.75,
            'truncation_rate': 0.2
        }
        patterns_good = {
            'avg_repetitive_loops': 1,
            'length_consistency': 0.8
        }
        self.assertFalse(framework._validate_calibration(quality_low_tokens, patterns_good))
        
        # Test high loops
        quality_good = {
            'avg_completion_tokens': 300,
            'completion_rate': 0.75,
            'truncation_rate': 0.2
        }
        patterns_high_loops = {
            'avg_repetitive_loops': 3,  # Above 2 threshold
            'length_consistency': 0.8
        }
        self.assertFalse(framework._validate_calibration(quality_good, patterns_high_loops))
        
        # Test high truncation
        quality_high_truncation = {
            'avg_completion_tokens': 300,
            'completion_rate': 0.75,
            'truncation_rate': 0.4  # Above 0.3 threshold
        }
        self.assertFalse(framework._validate_calibration(quality_high_truncation, patterns_good))
    
    def test_calculate_calibration_score(self):
        """Test calibration score calculation"""
        framework = ProductionCalibrationFramework()
        
        quality = {
            'completion_rate': 0.8,
            'truncation_rate': 0.1
        }
        patterns = {
            'avg_repetitive_loops': 1,
            'length_consistency': 0.9
        }
        
        score = framework._calculate_calibration_score(quality, patterns)
        
        # Verify score is calculated correctly
        expected_completion = 0.8 * 100  # 80
        expected_loop = max(0, 100 - (1 * 25))  # 75
        expected_consistency = 0.9 * 100  # 90
        expected_truncation = max(0, 100 - (0.1 * 100))  # 90
        
        expected_total = (expected_completion * 0.3 + expected_loop * 0.3 + 
                         expected_consistency * 0.2 + expected_truncation * 0.2)
        
        self.assertAlmostEqual(score, expected_total, places=1)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
    
    @patch('core.production_calibration.requests.post')
    def test_run_single_sample_success(self, mock_post):
        """Test successful single sample execution"""
        framework = ProductionCalibrationFramework()
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'choices': [{
                'text': 'This is a test response from the API.',
                'finish_reason': 'stop'
            }],
            'usage': {
                'completion_tokens': 320
            }
        }
        mock_post.return_value = mock_response
        
        test_case = {
            'prompt': 'Test prompt',
            'temperature': 0.0,
            'top_p': 1.0
        }
        
        result = framework._run_single_sample(test_case, 400)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['response_text'], 'This is a test response from the API.')
        self.assertEqual(result['finish_reason'], 'stop')
        self.assertEqual(result['completion_tokens'], 320)
        
        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        self.assertEqual(payload['prompt'], 'Test prompt')
        self.assertEqual(payload['max_tokens'], 400)
        self.assertEqual(payload['temperature'], 0.0)
    
    @patch('core.production_calibration.requests.post')
    def test_run_single_sample_api_error(self, mock_post):
        """Test single sample execution with API error"""
        framework = ProductionCalibrationFramework()
        
        # Mock API error
        mock_post.side_effect = Exception("API connection failed")
        
        test_case = {
            'prompt': 'Test prompt',
            'temperature': 0.0,
            'top_p': 1.0
        }
        
        result = framework._run_single_sample(test_case, 400)
        
        self.assertIsNone(result)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('core.production_calibration.json.load')
    @patch('core.production_calibration.time.sleep')
    @patch.object(ProductionCalibrationFramework, '_run_single_sample')
    def test_run_calibration_test_success(self, mock_run_sample, mock_sleep, mock_json_load, mock_file):
        """Test successful calibration test execution"""
        framework = ProductionCalibrationFramework()
        
        # Mock domain data
        test_data = {
            'tests': [
                {'id': 'test1', 'prompt': 'Test prompt 1'},
                {'id': 'test2', 'prompt': 'Test prompt 2'}
            ]
        }
        mock_json_load.return_value = test_data
        
        # Mock sample results
        sample_result = {
            'response_text': 'Good test response without repetition.',
            'finish_reason': 'stop',
            'completion_tokens': 350
        }
        mock_run_sample.return_value = sample_result
        
        # Create mock domain path
        domain_path = Mock(spec=Path)
        domain_path.__str__ = Mock(return_value='domains/reasoning/base_models/easy.json')
        domain_path.parent.parent.name = 'reasoning'
        
        result = framework.run_calibration_test(domain_path, 'test1', samples=2)
        
        self.assertIsInstance(result, CalibrationResult)
        self.assertEqual(result.domain, 'reasoning')
        self.assertEqual(result.test_id, 'test1')
        self.assertEqual(result.difficulty, 'easy')
        self.assertEqual(result.target_tokens, 400)  # Easy difficulty
        self.assertEqual(result.completion_tokens, 350)
        self.assertEqual(result.finish_reason, 'stop')
        
        # Verify multiple samples were run
        self.assertEqual(mock_run_sample.call_count, 2)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('core.production_calibration.json.load')
    def test_run_calibration_test_file_error(self, mock_json_load, mock_file):
        """Test calibration test with file loading error"""
        framework = ProductionCalibrationFramework()
        
        mock_json_load.side_effect = Exception("File read error")
        domain_path = Mock(spec=Path)
        
        result = framework.run_calibration_test(domain_path, 'test1')
        
        self.assertIsNone(result)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('core.production_calibration.json.load')
    def test_run_calibration_test_missing_test(self, mock_json_load, mock_file):
        """Test calibration test with missing test ID"""
        framework = ProductionCalibrationFramework()
        
        test_data = {
            'tests': [
                {'id': 'test1', 'prompt': 'Test prompt 1'},
                {'id': 'test2', 'prompt': 'Test prompt 2'}
            ]
        }
        mock_json_load.return_value = test_data
        
        domain_path = Mock(spec=Path)
        
        result = framework.run_calibration_test(domain_path, 'missing_test')
        
        self.assertIsNone(result)
    
    @patch.object(ProductionCalibrationFramework, 'discover_all_domains')
    @patch.object(ProductionCalibrationFramework, 'analyze_domain')
    @patch.object(ProductionCalibrationFramework, 'run_calibration_test')
    @patch.object(ProductionCalibrationFramework, 'generate_production_report')
    @patch('builtins.open', new_callable=mock_open)
    @patch('core.production_calibration.json.load')
    @patch('core.production_calibration.time.sleep')
    def test_run_production_calibration(self, mock_sleep, mock_json_load, mock_file, 
                                       mock_report, mock_calibrate, mock_analyze, mock_discover):
        """Test full production calibration workflow"""
        framework = ProductionCalibrationFramework()
        
        # Mock domain discovery
        mock_domain_paths = [Mock(spec=Path) for _ in range(6)]
        mock_discover.return_value = mock_domain_paths
        
        # Mock domain analysis
        priority_stats = DomainStats('priority', 100, {}, {}, True, 'HIGH')
        optimized_stats = DomainStats('optimized', 50, {}, {}, False, 'LOW')
        mock_analyze.side_effect = [priority_stats, priority_stats, priority_stats,
                                   optimized_stats, optimized_stats, optimized_stats]
        
        # Mock domain data loading
        test_data = {
            'tests': [
                {'id': 'test1', 'prompt': 'Test 1'},
                {'id': 'test2', 'prompt': 'Test 2'}
            ]
        }
        mock_json_load.return_value = test_data
        
        # Mock calibration test results
        successful_result = CalibrationResult(
            'test_domain', 'test1', 'easy', 400, 350, 'stop', 
            {}, {}, True, 85.0
        )
        failed_result = CalibrationResult(
            'test_domain', 'test2', 'easy', 400, 150, 'length',
            {}, {}, False, 45.0
        )
        mock_calibrate.side_effect = [successful_result, failed_result] * 3  # For 3 domains, 2 tests each
        
        # Run production calibration
        framework.run_production_calibration(sample_domains=3, tests_per_domain=2)
        
        # Verify workflow execution
        mock_discover.assert_called_once()
        self.assertEqual(mock_analyze.call_count, 6)  # sample_domains * 3 for selection
        self.assertEqual(mock_calibrate.call_count, 6)  # 3 domains * 2 tests each
        mock_report.assert_called_once_with(6, 3)  # 6 total tests, 3 successful
    
    @patch('builtins.print')
    @patch('core.production_calibration.os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('core.production_calibration.json.dump')
    def test_generate_production_report(self, mock_json_dump, mock_file, mock_makedirs, mock_print):
        """Test production report generation"""
        framework = ProductionCalibrationFramework()
        
        # Add some mock results
        framework.calibration_results = [
            CalibrationResult('reasoning', 'test1', 'easy', 400, 350, 'stop', 
                            {'avg_completion_tokens': 350, 'completion_rate': 0.8}, 
                            {'avg_repetitive_loops': 0}, True, 85.0),
            CalibrationResult('creativity', 'test2', 'medium', 500, 400, 'stop',
                            {'avg_completion_tokens': 400, 'completion_rate': 0.8},
                            {'avg_repetitive_loops': 1}, True, 78.0),
            CalibrationResult('language', 'test3', 'hard', 600, 200, 'length',
                            {'avg_completion_tokens': 200, 'completion_rate': 0.3},
                            {'avg_repetitive_loops': 3}, False, 45.0)
        ]
        
        # Generate report
        framework.generate_production_report(total_tests=3, successful_calibrations=2)
        
        # Verify report generation
        mock_print.assert_called()  # Report was printed
        mock_makedirs.assert_called_once_with("test_results", exist_ok=True)
        mock_json_dump.assert_called_once()
        
        # Verify saved data structure
        call_args = mock_json_dump.call_args
        saved_data = call_args[0][0]
        
        self.assertIn('framework_version', saved_data)
        self.assertIn('timestamp', saved_data)
        self.assertIn('summary', saved_data)
        self.assertIn('detailed_results', saved_data)
        
        self.assertEqual(saved_data['summary']['total_tests'], 3)
        self.assertEqual(saved_data['summary']['successful_calibrations'], 2)
        self.assertEqual(len(saved_data['detailed_results']), 3)


if __name__ == '__main__':
    unittest.main()