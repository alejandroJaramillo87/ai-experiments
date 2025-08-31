#!/usr/bin/env python3
"""
Automated Calibration Validation Test Suite

Unit and integration tests for the calibration validation framework.
Provides automated testing of calibration components without requiring
full LLM inference runs.

Author: Claude Code
Version: 1.0.0 - Sequential Architecture
"""

import unittest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any

import sys
sys.path.append('.')
sys.path.append('..')

from calibration_validator import CalibrationValidator, CalibrationResult
from calibration_reporter import CalibrationReporter
from reference_test_cases import (
    REFERENCE_TEST_CASES, 
    get_active_test_cases, 
    validate_test_case_structure,
    get_calibration_summary
)

class TestReferenceTestCases(unittest.TestCase):
    """Test reference test cases configuration"""
    
    def test_reference_test_cases_structure(self):
        """Test that all reference test cases have valid structure"""
        for i, test_case in enumerate(REFERENCE_TEST_CASES):
            with self.subTest(test_case_index=i):
                self.assertTrue(
                    validate_test_case_structure(test_case),
                    f"Test case {i} failed validation: {test_case.get('name', 'Unknown')}"
                )
    
    def test_reference_test_cases_not_empty(self):
        """Test that we have reference test cases defined"""
        self.assertGreater(len(REFERENCE_TEST_CASES), 0)
    
    def test_get_active_test_cases(self):
        """Test getting active test cases"""
        active = get_active_test_cases()
        self.assertIsInstance(active, list)
        self.assertEqual(len(active), len(REFERENCE_TEST_CASES))
    
    def test_calibration_summary(self):
        """Test calibration summary generation"""
        summary = get_calibration_summary()
        
        required_fields = ['total_test_cases', 'categories', 'difficulties', 'target_range_summary']
        for field in required_fields:
            self.assertIn(field, summary)
        
        self.assertGreater(summary['total_test_cases'], 0)
        self.assertIsInstance(summary['categories'], list)
        self.assertIsInstance(summary['difficulties'], list)

class TestCalibrationValidator(unittest.TestCase):
    """Test calibration validator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = CalibrationValidator(sample_count=3)
    
    def test_calibration_validator_initialization(self):
        """Test calibration validator initializes correctly"""
        self.assertEqual(self.validator.sample_count, 3)
        self.assertIsNone(self.validator.benchmark_runner)
        self.assertIsNotNone(self.validator.reporter)
    
    def test_assess_calibration_status_perfect(self):
        """Test perfect calibration assessment"""
        status, points_off = self.validator._assess_calibration_status(80.0, 80.0, (75, 85))
        self.assertEqual(status, "✅ PERFECT CALIBRATION")
        self.assertEqual(points_off, 0.0)
    
    def test_assess_calibration_status_good(self):
        """Test good calibration assessment"""
        status, points_off = self.validator._assess_calibration_status(70.0, 80.0, (75, 85))
        self.assertEqual(status, "🟡 GOOD CALIBRATION")
        self.assertEqual(points_off, 5.0)
    
    def test_assess_calibration_status_needs_work(self):
        """Test needs calibration assessment"""
        status, points_off = self.validator._assess_calibration_status(65.0, 80.0, (75, 85))
        self.assertEqual(status, "🟠 NEEDS CALIBRATION")
        self.assertEqual(points_off, 10.0)
    
    def test_assess_calibration_status_broken(self):
        """Test broken calibration assessment"""
        status, points_off = self.validator._assess_calibration_status(50.0, 80.0, (75, 85))
        self.assertEqual(status, "❌ CALIBRATION BROKEN")
        self.assertEqual(points_off, 25.0)
    
    @patch('calibration_validator.BenchmarkTestRunner')
    def test_setup_benchmark_runner(self, mock_benchmark_runner):
        """Test benchmark runner setup"""
        mock_instance = Mock()
        mock_benchmark_runner.return_value = mock_instance
        
        success = self.validator.setup_benchmark_runner()
        
        self.assertTrue(success)
        self.assertIsNotNone(self.validator.benchmark_runner)
    
    def test_run_multi_sample_test_mock(self):
        """Test multi-sample test with mocked results"""
        # Mock test case
        test_case = {
            'name': 'Mock Test',
            'expected_range': (70, 80),
            'target_mean': 75.0
        }
        
        # Mock the run_single_test_sample method
        with patch.object(self.validator, 'run_single_test_sample') as mock_run:
            mock_run.side_effect = [72.0, 74.0, 76.0]  # Mock 3 scores
            
            result = self.validator.run_multi_sample_test(test_case)
            
            self.assertEqual(result.test_name, 'Mock Test')
            self.assertEqual(result.sample_count, 3)
            self.assertEqual(len(result.scores), 3)
            self.assertAlmostEqual(result.mean_score, 74.0, places=1)
            self.assertGreater(result.std_deviation, 0)

class TestCalibrationReporter(unittest.TestCase):
    """Test calibration reporter functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.reporter = CalibrationReporter()
        
        # Create mock calibration results
        self.mock_results = [
            CalibrationResult(
                test_name="Perfect Test",
                scores=[80.0, 82.0, 78.0],
                mean_score=80.0,
                std_deviation=2.0,
                target_range=(75, 85),
                target_mean=80.0,
                calibration_status="✅ PERFECT CALIBRATION",
                points_off_target=0.0,
                sample_count=3
            ),
            CalibrationResult(
                test_name="Good Test",
                scores=[70.0, 72.0, 68.0],
                mean_score=70.0,
                std_deviation=2.0,
                target_range=(75, 85),
                target_mean=80.0,
                calibration_status="🟡 GOOD CALIBRATION",
                points_off_target=5.0,
                sample_count=3
            ),
            CalibrationResult(
                test_name="Broken Test",
                scores=[30.0, 32.0, 28.0],
                mean_score=30.0,
                std_deviation=2.0,
                target_range=(75, 85),
                target_mean=80.0,
                calibration_status="❌ CALIBRATION BROKEN",
                points_off_target=45.0,
                sample_count=3
            )
        ]
    
    def test_count_calibration_statuses(self):
        """Test calibration status counting"""
        counts = self.reporter._count_calibration_statuses(self.mock_results)
        
        self.assertEqual(counts['perfect_calibration'], 1)
        self.assertEqual(counts['good_calibration'], 1)
        self.assertEqual(counts['needs_calibration'], 0)
        self.assertEqual(counts['calibration_broken'], 1)
    
    def test_determine_overall_status_excellent(self):
        """Test overall status determination - excellent"""
        status_counts = {
            'perfect_calibration': 4,
            'good_calibration': 1,
            'needs_calibration': 0,
            'calibration_broken': 0
        }
        
        overall = self.reporter._determine_overall_status(status_counts, 5)
        self.assertEqual(overall, "✅ CALIBRATION EXCELLENT")
    
    def test_determine_overall_status_broken(self):
        """Test overall status determination - broken"""
        status_counts = {
            'perfect_calibration': 1,
            'good_calibration': 0,
            'needs_calibration': 1,
            'calibration_broken': 3
        }
        
        overall = self.reporter._determine_overall_status(status_counts, 5)
        self.assertEqual(overall, "❌ CALIBRATION BROKEN")
    
    def test_generate_report(self):
        """Test comprehensive report generation"""
        report = self.reporter.generate_report(self.mock_results)
        
        # Check report structure
        required_sections = [
            'report_metadata', 'overall_assessment', 
            'calibration_status_breakdown', 'statistical_analysis',
            'test_details', 'recommendations'
        ]
        
        for section in required_sections:
            self.assertIn(section, report)
        
        # Check metadata
        metadata = report['report_metadata']
        self.assertEqual(metadata['total_tests'], 3)
        self.assertEqual(metadata['successful_tests'], 3)
        
        # Check overall assessment
        overall = report['overall_assessment']
        self.assertIn('overall_calibration_status', overall)
        self.assertIn('calibration_success_rate', overall)
    
    def test_save_detailed_report(self):
        """Test saving detailed report to file"""
        report = self.reporter.generate_report(self.mock_results)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            tmp_path = tmp.name
        
        try:
            success = self.reporter.save_detailed_report(report, tmp_path)
            self.assertTrue(success)
            
            # Verify file was created and contains valid JSON
            self.assertTrue(os.path.exists(tmp_path))
            
            with open(tmp_path, 'r') as f:
                loaded_report = json.load(f)
            
            self.assertEqual(loaded_report['report_metadata']['total_tests'], 3)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

class TestIntegration(unittest.TestCase):
    """Integration tests for calibration validation system"""
    
    def test_end_to_end_mock_workflow(self):
        """Test end-to-end workflow with mocked components"""
        
        # Initialize validator
        validator = CalibrationValidator(sample_count=2)
        
        # Mock benchmark runner setup
        with patch.object(validator, 'setup_benchmark_runner') as mock_setup:
            mock_setup.return_value = True
            
            # Mock single test runs
            with patch.object(validator, 'run_single_test_sample') as mock_run:
                mock_run.side_effect = [75.0, 77.0]  # Two sample scores
                
                # Get a reference test case
                test_cases = get_active_test_cases()
                if test_cases:
                    test_case = test_cases[0]
                    
                    # Run multi-sample test
                    result = validator.run_multi_sample_test(test_case)
                    
                    # Validate result
                    self.assertIsNotNone(result)
                    self.assertEqual(result.sample_count, 2)
                    self.assertEqual(len(result.scores), 2)
                    self.assertGreater(result.mean_score, 0)
    
    def test_reference_test_case_compatibility(self):
        """Test that reference test cases are compatible with validator"""
        validator = CalibrationValidator(sample_count=1)
        
        for test_case in REFERENCE_TEST_CASES:
            # Test that test case has required fields for validator
            required_fields = ['name', 'expected_range', 'target_mean']
            for field in required_fields:
                self.assertIn(field, test_case, 
                            f"Test case '{test_case.get('name')}' missing required field '{field}'")

class TestCalibrationFrameworkStability(unittest.TestCase):
    """Test framework stability and error handling"""
    
    def test_empty_results_handling(self):
        """Test handling of empty results"""
        reporter = CalibrationReporter()
        
        report = reporter.generate_report([])
        self.assertIn("error", report)
    
    def test_invalid_test_case_handling(self):
        """Test handling of invalid test cases"""
        invalid_test_case = {
            'name': 'Invalid Test',
            # Missing required fields
        }
        
        self.assertFalse(validate_test_case_structure(invalid_test_case))
    
    def test_single_sample_statistics(self):
        """Test statistics with single sample"""
        validator = CalibrationValidator(sample_count=1)
        
        with patch.object(validator, 'run_single_test_sample') as mock_run:
            mock_run.return_value = 75.0
            
            test_case = {
                'name': 'Single Sample Test',
                'expected_range': (70, 80),
                'target_mean': 75.0
            }
            
            result = validator.run_multi_sample_test(test_case)
            
            self.assertEqual(result.sample_count, 1)
            self.assertEqual(result.mean_score, 75.0)
            self.assertEqual(result.std_deviation, 0.0)

def run_calibration_unit_tests():
    """Run all calibration unit tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestReferenceTestCases,
        TestCalibrationValidator,
        TestCalibrationReporter,
        TestIntegration,
        TestCalibrationFrameworkStability
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("🧪 Running Calibration Validation Framework Unit Tests")
    print("=" * 60)
    
    success = run_calibration_unit_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All calibration framework unit tests passed!")
    else:
        print("❌ Some calibration framework unit tests failed!")
    
    exit(0 if success else 1)