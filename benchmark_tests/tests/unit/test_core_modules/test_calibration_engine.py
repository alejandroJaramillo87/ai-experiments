#!/usr/bin/env python3
"""
Unit Tests for SystematicBaseCalibrator

Tests the calibration engine functionality including domain progression,
statistical validation, quality assessment, and calibration workflows.
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

from core.calibration_engine import (
    SystematicBaseCalibrator, EvaluationCalibrationResult, SystematicCalibrationResult
)


class TestEvaluationCalibrationResult(unittest.TestCase):
    """Test EvaluationCalibrationResult data structure"""
    
    def test_evaluation_calibration_result_creation(self):
        """Test EvaluationCalibrationResult creation with all fields"""
        result = EvaluationCalibrationResult(
            domain="reasoning",
            test_id="basic_01",
            difficulty="easy",
            enhanced_score=75.5,
            target_range=(70.0, 85.0),
            validation_passed=True,
            calibration_status="✅ EXCELLENT CALIBRATION"
        )
        
        self.assertEqual(result.domain, "reasoning")
        self.assertEqual(result.test_id, "basic_01")
        self.assertEqual(result.difficulty, "easy")
        self.assertEqual(result.enhanced_score, 75.5)
        self.assertEqual(result.target_range, (70.0, 85.0))
        self.assertTrue(result.validation_passed)
        self.assertEqual(result.calibration_status, "✅ EXCELLENT CALIBRATION")


class TestSystematicCalibrationResult(unittest.TestCase):
    """Test SystematicCalibrationResult data structure"""
    
    def test_systematic_calibration_result_creation(self):
        """Test SystematicCalibrationResult creation"""
        progression = {
            'easy': {'avg_score': 75.0, 'success_rate': 0.8},
            'medium': {'avg_score': 68.0, 'success_rate': 0.7},
            'hard': None
        }
        
        result = SystematicCalibrationResult(
            domain="reasoning",
            difficulty_progression=progression,
            overall_success=False,
            progression_halted_at="medium",
            recommendation="Need to improve medium calibration"
        )
        
        self.assertEqual(result.domain, "reasoning")
        self.assertEqual(result.difficulty_progression, progression)
        self.assertFalse(result.overall_success)
        self.assertEqual(result.progression_halted_at, "medium")
        self.assertIn("improve", result.recommendation)


class TestSystematicBaseCalibrator(unittest.TestCase):
    """Test SystematicBaseCalibrator functionality"""
    
    @patch('core.calibration_engine.BenchmarkTestRunner')
    def test_initialization(self, mock_runner_class):
        """Test calibrator initialization with proper configuration"""
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        calibrator = SystematicBaseCalibrator()
        
        # Verify runner initialization
        mock_runner_class.assert_called_once()
        mock_runner.configure_api.assert_called_once_with(
            endpoint="http://localhost:8004/v1/completions",
            model="DeepSeek-R1-0528-Qwen3-8b",
            timeout=30
        )
        
        # Verify enhanced evaluation setup
        self.assertTrue(mock_runner.enhanced_evaluation)
        self.assertEqual(mock_runner.evaluation_mode, "full")
        
        # Verify configuration
        self.assertEqual(calibrator.target_ranges['easy'], (70, 85))
        self.assertEqual(calibrator.target_ranges['medium'], (60, 80))
        self.assertEqual(calibrator.target_ranges['hard'], (50, 75))
        
        expected_domains = ['reasoning', 'creativity', 'language', 'social', 'knowledge', 'integration']
        self.assertEqual(calibrator.core_domains, expected_domains)
    
    @patch.object(SystematicBaseCalibrator, 'get_available_domain_files')
    def test_get_available_domain_files(self, mock_get_files):
        """Test getting available domain files"""
        calibrator = SystematicBaseCalibrator()
        
        # Mock the method to return only easy and medium files
        mock_get_files.return_value = {
            'easy': Mock(spec=Path),
            'medium': Mock(spec=Path)
        }
        
        # Test method
        files = calibrator.get_available_domain_files('reasoning')
        
        # Should return existing files only
        self.assertIn('easy', files)
        self.assertIn('medium', files)
        self.assertNotIn('hard', files)
    
    def test_get_calibration_status(self):
        """Test calibration status determination"""
        calibrator = SystematicBaseCalibrator()
        
        # Test excellent calibration (within ±2 of center)
        status = calibrator._get_calibration_status(77.5, (70, 85))  # Center is 77.5
        self.assertEqual(status, "✅ EXCELLENT CALIBRATION")
        
        # Test good calibration (within range but not excellent)
        status = calibrator._get_calibration_status(72.0, (70, 85))
        self.assertEqual(status, "🟡 GOOD CALIBRATION")
        
        # Test needs calibration (outside range but within ±10 of center)
        # For range (70, 85), center is 77.5. 68.0 is within 10 points of center (9.5 points away)
        status = calibrator._get_calibration_status(68.0, (70, 85))
        self.assertEqual(status, "🟠 NEEDS CALIBRATION")
        
        # Test broken calibration (far from center)
        # For range (70, 85), center is 77.5. 50.0 is > 10 points away (27.5 points)
        status = calibrator._get_calibration_status(50.0, (70, 85))
        self.assertEqual(status, "❌ CALIBRATION BROKEN")
    
    def test_assess_calibration_quality_excellent(self):
        """Test calibration quality assessment - excellent level"""
        calibrator = SystematicBaseCalibrator()
        
        quality = calibrator._assess_calibration_quality(avg_score=82.0, success_rate=0.8)
        
        self.assertEqual(quality['level'], "EXCELLENT")
        self.assertTrue(quality['should_continue'])
        self.assertIn("Excellent calibration", quality['recommendation'])
    
    def test_assess_calibration_quality_good(self):
        """Test calibration quality assessment - good level"""
        calibrator = SystematicBaseCalibrator()
        
        quality = calibrator._assess_calibration_quality(avg_score=72.0, success_rate=0.8)
        
        self.assertEqual(quality['level'], "GOOD")
        self.assertTrue(quality['should_continue'])
        self.assertIn("Good calibration", quality['recommendation'])
    
    def test_assess_calibration_quality_needs_calibration(self):
        """Test calibration quality assessment - needs calibration"""
        calibrator = SystematicBaseCalibrator()
        
        quality = calibrator._assess_calibration_quality(avg_score=65.0, success_rate=0.8)
        
        self.assertEqual(quality['level'], "NEEDS_CALIBRATION")
        self.assertFalse(quality['should_continue'])
        self.assertIn("needs adjustment", quality['recommendation'])
    
    def test_assess_calibration_quality_broken(self):
        """Test calibration quality assessment - broken"""
        calibrator = SystematicBaseCalibrator()
        
        quality = calibrator._assess_calibration_quality(avg_score=45.0, success_rate=0.8)
        
        self.assertEqual(quality['level'], "BROKEN")
        self.assertFalse(quality['should_continue'])
        self.assertIn("broken", quality['recommendation'])
    
    def test_assess_calibration_quality_low_success_rate(self):
        """Test calibration quality with low success rate"""
        calibrator = SystematicBaseCalibrator()
        
        # High score but low success rate
        quality = calibrator._assess_calibration_quality(avg_score=85.0, success_rate=0.5)
        
        # Should not continue due to low success rate
        self.assertFalse(quality['should_continue'])
        self.assertIn("Success rate", quality['recommendation'])
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('core.calibration_engine.json.load')
    @patch('core.calibration_engine.time.sleep')
    def test_calibrate_difficulty_level_success(self, mock_sleep, mock_json_load, mock_file):
        """Test successful difficulty level calibration"""
        calibrator = SystematicBaseCalibrator()
        
        # Mock domain data
        test_data = {
            'tests': [
                {'id': 'test1', 'prompt': 'Test 1'},
                {'id': 'test2', 'prompt': 'Test 2'},
                {'id': 'test3', 'prompt': 'Test 3'}
            ]
        }
        mock_json_load.return_value = test_data
        
        # Mock benchmark runner results
        mock_result1 = Mock()
        mock_result1.enhanced_score = 75.0
        mock_result2 = Mock()
        mock_result2.enhanced_score = 77.0
        mock_result3 = Mock()
        mock_result3.enhanced_score = 76.0
        
        calibrator.benchmark_runner = Mock()
        calibrator.benchmark_runner.run_single_test.side_effect = [
            # 3 samples for test1
            mock_result1, mock_result2, mock_result3,
            # 3 samples for test2
            mock_result1, mock_result2, mock_result3,
            # 3 samples for test3
            mock_result1, mock_result2, mock_result3
        ]
        
        # Test method
        domain_file = Path('/test/path/easy.json')
        should_continue, results = calibrator.calibrate_difficulty_level(
            'reasoning', 'easy', domain_file, tests_per_level=3
        )
        
        # Verify results
        self.assertTrue(should_continue)
        self.assertEqual(results['domain'], 'reasoning')
        self.assertEqual(results['difficulty'], 'easy')
        self.assertEqual(results['tests_run'], 3)
        self.assertEqual(results['successful_calibrations'], 3)  # All should pass
        self.assertEqual(results['success_rate'], 1.0)
        self.assertAlmostEqual(results['avg_calibration_score'], 76.0)
        
        # Verify benchmark runner calls (3 tests × 3 samples each = 9 calls)
        self.assertEqual(calibrator.benchmark_runner.run_single_test.call_count, 9)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('core.calibration_engine.json.load')
    def test_calibrate_difficulty_level_file_load_error(self, mock_json_load, mock_file):
        """Test calibration with file loading error"""
        calibrator = SystematicBaseCalibrator()
        
        # Mock JSON load error
        mock_json_load.side_effect = Exception("File read error")
        
        domain_file = Path('/test/path/easy.json')
        should_continue, results = calibrator.calibrate_difficulty_level(
            'reasoning', 'easy', domain_file
        )
        
        # Should fail with error
        self.assertFalse(should_continue)
        self.assertIn('error', results)
        self.assertEqual(results['error'], "File read error")
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('core.calibration_engine.json.load')
    @patch('core.calibration_engine.time.sleep')
    def test_calibrate_difficulty_level_no_enhanced_scores(self, mock_sleep, mock_json_load, mock_file):
        """Test calibration when benchmark runner fails to provide enhanced scores"""
        calibrator = SystematicBaseCalibrator()
        
        # Mock domain data
        test_data = {'tests': [{'id': 'test1', 'prompt': 'Test 1'}]}
        mock_json_load.return_value = test_data
        
        # Mock benchmark runner to return None or result without enhanced_score
        calibrator.benchmark_runner = Mock()
        calibrator.benchmark_runner.run_single_test.return_value = None
        
        domain_file = Path('/test/path/easy.json')
        should_continue, results = calibrator.calibrate_difficulty_level(
            'reasoning', 'easy', domain_file, tests_per_level=1
        )
        
        # Should fail when no results available
        self.assertFalse(should_continue)
        self.assertIn('error', results)
        self.assertEqual(results['error'], 'all_tests_failed')
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('core.calibration_engine.json.load')
    @patch('core.calibration_engine.time.sleep')
    def test_calibrate_difficulty_level_partial_success(self, mock_sleep, mock_json_load, mock_file):
        """Test calibration with mixed results - some tests pass, some fail"""
        calibrator = SystematicBaseCalibrator()
        
        # Mock domain data
        test_data = {
            'tests': [
                {'id': 'test1', 'prompt': 'Test 1'},
                {'id': 'test2', 'prompt': 'Test 2'}
            ]
        }
        mock_json_load.return_value = test_data
        
        # Mock benchmark runner results - test1 passes (75), test2 fails (45)
        mock_result_pass = Mock()
        mock_result_pass.enhanced_score = 75.0  # Within easy range (70-85)
        mock_result_fail = Mock()
        mock_result_fail.enhanced_score = 45.0  # Below easy range
        
        calibrator.benchmark_runner = Mock()
        calibrator.benchmark_runner.run_single_test.side_effect = [
            # 3 samples for test1 (passing)
            mock_result_pass, mock_result_pass, mock_result_pass,
            # 3 samples for test2 (failing)
            mock_result_fail, mock_result_fail, mock_result_fail
        ]
        
        domain_file = Path('/test/path/easy.json')
        should_continue, results = calibrator.calibrate_difficulty_level(
            'reasoning', 'easy', domain_file, tests_per_level=2
        )
        
        # Should not continue due to low success rate (50% < 70%)
        self.assertFalse(should_continue)
        self.assertEqual(results['tests_run'], 2)
        self.assertEqual(results['successful_calibrations'], 1)  # Only test1 passed
        self.assertEqual(results['success_rate'], 0.5)
        self.assertAlmostEqual(results['avg_calibration_score'], 60.0)  # (75+45)/2
    
    @patch.object(SystematicBaseCalibrator, 'get_available_domain_files')
    @patch.object(SystematicBaseCalibrator, 'calibrate_difficulty_level')
    def test_calibrate_domain_progression_complete_success(self, mock_calibrate_level, mock_get_files):
        """Test complete domain progression success (easy→medium→hard)"""
        calibrator = SystematicBaseCalibrator()
        
        # Mock available files
        mock_files = {
            'easy': Path('/test/easy.json'),
            'medium': Path('/test/medium.json'),
            'hard': Path('/test/hard.json')
        }
        mock_get_files.return_value = mock_files
        
        # Mock successful calibration at all levels
        mock_calibrate_level.side_effect = [
            # Easy level - success
            (True, {'difficulty': 'easy', 'avg_calibration_score': 75.0, 'success_rate': 0.8}),
            # Medium level - success  
            (True, {'difficulty': 'medium', 'avg_calibration_score': 68.0, 'success_rate': 0.7}),
            # Hard level - success
            (True, {'difficulty': 'hard', 'avg_calibration_score': 62.0, 'success_rate': 0.7})
        ]
        
        # Test method
        result = calibrator.calibrate_domain_progression('reasoning')
        
        # Verify complete success
        self.assertIsInstance(result, SystematicCalibrationResult)
        self.assertEqual(result.domain, 'reasoning')
        self.assertTrue(result.overall_success)
        self.assertIsNone(result.progression_halted_at)
        self.assertIn("production", result.recommendation)
        
        # Verify all levels were attempted
        self.assertEqual(mock_calibrate_level.call_count, 3)
        # The method is called with default tests_per_level parameter
        mock_calibrate_level.assert_any_call('reasoning', 'easy', mock_files['easy'])
        mock_calibrate_level.assert_any_call('reasoning', 'medium', mock_files['medium'])
        mock_calibrate_level.assert_any_call('reasoning', 'hard', mock_files['hard'])
    
    @patch.object(SystematicBaseCalibrator, 'get_available_domain_files')
    @patch.object(SystematicBaseCalibrator, 'calibrate_difficulty_level')
    def test_calibrate_domain_progression_halt_at_medium(self, mock_calibrate_level, mock_get_files):
        """Test domain progression halting at medium level"""
        calibrator = SystematicBaseCalibrator()
        
        # Mock available files
        mock_files = {
            'easy': Path('/test/easy.json'),
            'medium': Path('/test/medium.json'),
            'hard': Path('/test/hard.json')
        }
        mock_get_files.return_value = mock_files
        
        # Mock calibration results - easy succeeds, medium fails
        mock_calibrate_level.side_effect = [
            # Easy level - success
            (True, {'difficulty': 'easy', 'avg_calibration_score': 75.0, 'success_rate': 0.8, 'calibration_quality': {'recommendation': 'Good enough'}}),
            # Medium level - failure
            (False, {'difficulty': 'medium', 'avg_calibration_score': 55.0, 'success_rate': 0.4, 'calibration_quality': {'recommendation': 'Needs work'}})
        ]
        
        # Test method
        result = calibrator.calibrate_domain_progression('reasoning')
        
        # Verify halted progression
        self.assertFalse(result.overall_success)
        self.assertEqual(result.progression_halted_at, 'medium')
        self.assertIn("medium", result.recommendation)
        
        # Verify hard level was not attempted
        self.assertEqual(mock_calibrate_level.call_count, 2)
    
    @patch.object(SystematicBaseCalibrator, 'get_available_domain_files')
    def test_calibrate_domain_progression_no_files(self, mock_get_files):
        """Test domain progression with no available files"""
        calibrator = SystematicBaseCalibrator()
        
        # Mock no available files
        mock_get_files.return_value = {}
        
        # Test method
        result = calibrator.calibrate_domain_progression('nonexistent')
        
        # Should fail with no files
        self.assertFalse(result.overall_success)
        self.assertIsNone(result.progression_halted_at)
        self.assertIn("not found", result.recommendation)
    
    @patch.object(SystematicBaseCalibrator, 'calibrate_domain_progression')
    @patch.object(SystematicBaseCalibrator, 'generate_systematic_report')
    def test_run_systematic_base_calibration(self, mock_report, mock_calibrate_domain):
        """Test running systematic base calibration across all core domains"""
        calibrator = SystematicBaseCalibrator()
        
        # Mock domain calibration results
        success_result = SystematicCalibrationResult(
            domain='reasoning', difficulty_progression={}, overall_success=True,
            progression_halted_at=None, recommendation="Complete success"
        )
        failure_result = SystematicCalibrationResult(
            domain='creativity', difficulty_progression={}, overall_success=False,
            progression_halted_at='medium', recommendation="Failed at medium"
        )
        
        mock_calibrate_domain.side_effect = [success_result, failure_result, 
                                              success_result, success_result, failure_result, success_result]
        
        # Test method
        results = calibrator.run_systematic_base_calibration()
        
        # Should return summary results
        self.assertIn('domains_processed', results)
        self.assertIn('successful_domains', results)
        self.assertIn('success_rate', results)
        self.assertIn('domain_results', results)
        self.assertEqual(results['domains_processed'], 6)  # All core domains
        
        # Verify all domains were processed
        self.assertEqual(mock_calibrate_domain.call_count, 6)
        mock_report.assert_called_once()
    
    @patch('builtins.print')
    @patch.object(SystematicBaseCalibrator, 'save_systematic_results')
    def test_generate_systematic_report(self, mock_save, mock_print):
        """Test generating systematic calibration report"""
        calibrator = SystematicBaseCalibrator()
        
        # Mock calibration results with proper structure
        results = [
            SystematicCalibrationResult(
                domain='reasoning',
                difficulty_progression={'easy': {'avg_calibration_score': 75.0, 'success_rate': 0.8, 'calibration_quality': {'level': 'GOOD'}}},
                overall_success=True,
                progression_halted_at=None,
                recommendation="Complete success"
            ),
            SystematicCalibrationResult(
                domain='creativity',
                difficulty_progression={'easy': {'avg_calibration_score': 65.0, 'success_rate': 0.6, 'calibration_quality': {'level': 'NEEDS_CALIBRATION'}}},
                overall_success=False,
                progression_halted_at='medium',
                recommendation="Failed at medium"
            )
        ]
        
        # Test method - should not raise exception
        calibrator.generate_systematic_report(results, 1, 120.0)
        
        # Verify print was called (report was generated)
        mock_print.assert_called()
        mock_save.assert_called_once_with(results, 1, 120.0)


if __name__ == '__main__':
    unittest.main()