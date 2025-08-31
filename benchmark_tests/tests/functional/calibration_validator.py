#!/usr/bin/env python3
"""
Programmatic Calibration Validation Framework

External client that validates Enhanced Universal Evaluator calibration using
statistical multi-sample testing. Implements KISS principles with clean separation
from evaluator logic.

Author: Claude Code
Version: 1.0.0 - Sequential Architecture for llama.cpp
"""

import sys
import json
import time
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Add the project root to Python path
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

from benchmark_runner import BenchmarkTestRunner
from reference_test_cases import REFERENCE_TEST_CASES
from calibration_reporter import CalibrationReporter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CalibrationResult:
    """Single calibration test result"""
    test_name: str
    scores: List[float]
    mean_score: float
    std_deviation: float
    target_range: Tuple[float, float]
    target_mean: float
    calibration_status: str
    points_off_target: float
    sample_count: int

class CalibrationValidator:
    """
    External validation framework for Enhanced Universal Evaluator calibration.
    
    Uses statistical multi-sample testing to validate evaluation quality
    without modifying core evaluator logic.
    """
    
    def __init__(self, sample_count: int = 3):
        """
        Initialize calibration validator
        
        Args:
            sample_count: Number of samples per test (3-5 for statistical validity)
        """
        self.sample_count = sample_count
        self.benchmark_runner = None
        self.reporter = CalibrationReporter()
        
        logger.info(f"Calibration Validator initialized with {sample_count} samples per test")
    
    def setup_benchmark_runner(self) -> bool:
        """
        Setup benchmark runner for sequential operation (llama.cpp compatibility)
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Initialize benchmark runner with sequential configuration
            self.benchmark_runner = BenchmarkTestRunner()
            
            # Disable concurrency for llama.cpp single request limitation
            if hasattr(self.benchmark_runner, 'max_concurrent_requests'):
                self.benchmark_runner.max_concurrent_requests = 1
                logger.info("Disabled concurrent requests for llama.cpp compatibility")
            
            logger.info("Benchmark runner setup successful")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup benchmark runner: {e}")
            return False
    
    def run_single_test_sample(self, test_case: Dict[str, Any]) -> Optional[float]:
        """
        Run a single test sample and extract enhanced evaluation score
        
        Args:
            test_case: Reference test case definition
            
        Returns:
            Enhanced evaluation score or None if failed
        """
        try:
            # Extract test parameters
            domain_path = test_case.get('domain_path')
            test_id = test_case.get('test_id')
            
            if not domain_path or not test_id:
                logger.error(f"Missing domain_path or test_id in test case: {test_case['name']}")
                return None
            
            # Run single test through benchmark runner
            # Use enhanced evaluation mode
            result = self.benchmark_runner.run_single_test(
                domain_path=domain_path,
                test_id=test_id,
                enhanced_evaluation=True
            )
            
            # Extract enhanced score
            if result and hasattr(result, 'enhanced_score'):
                score = float(result.enhanced_score)
                logger.debug(f"Test {test_case['name']} sample score: {score:.1f}")
                return score
            else:
                logger.warning(f"No enhanced score found in result for {test_case['name']}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to run test sample for {test_case['name']}: {e}")
            return None
    
    def run_multi_sample_test(self, test_case: Dict[str, Any]) -> CalibrationResult:
        """
        Run multi-sample test with statistical validation
        
        Args:
            test_case: Reference test case definition
            
        Returns:
            CalibrationResult with statistical analysis
        """
        test_name = test_case['name']
        logger.info(f"Running multi-sample test: {test_name} ({self.sample_count} samples)")
        
        scores = []
        
        # Run multiple samples
        for sample_num in range(1, self.sample_count + 1):
            logger.info(f"  Sample {sample_num}/{self.sample_count}")
            
            score = self.run_single_test_sample(test_case)
            if score is not None:
                scores.append(score)
                logger.info(f"    Score: {score:.1f}")
            else:
                logger.warning(f"    Sample failed")
            
            # Brief delay between samples for llama.cpp stability
            if sample_num < self.sample_count:
                time.sleep(1)
        
        # Calculate statistics
        if len(scores) >= 2:
            mean_score = statistics.mean(scores)
            std_deviation = statistics.stdev(scores)
        elif len(scores) == 1:
            mean_score = scores[0]
            std_deviation = 0.0
        else:
            logger.error(f"No valid scores for test {test_name}")
            mean_score = 0.0
            std_deviation = 0.0
        
        # Extract target parameters
        target_range = test_case.get('expected_range', (0, 100))
        target_mean = test_case.get('target_mean', sum(target_range) / 2)
        
        # Calculate calibration status
        calibration_status, points_off_target = self._assess_calibration_status(
            mean_score, target_mean, target_range
        )
        
        result = CalibrationResult(
            test_name=test_name,
            scores=scores,
            mean_score=mean_score,
            std_deviation=std_deviation,
            target_range=target_range,
            target_mean=target_mean,
            calibration_status=calibration_status,
            points_off_target=points_off_target,
            sample_count=len(scores)
        )
        
        logger.info(f"Test complete: {mean_score:.1f} ± {std_deviation:.1f} ({calibration_status})")
        return result
    
    def _assess_calibration_status(self, mean_score: float, target_mean: float, target_range: Tuple[float, float]) -> Tuple[str, float]:
        """
        Assess calibration status based on statistical framework
        
        Args:
            mean_score: Measured mean score
            target_mean: Expected mean score
            target_range: Expected score range
            
        Returns:
            Tuple of (status_string, points_off_target)
        """
        # Check if within target range first
        if target_range[0] <= mean_score <= target_range[1]:
            return "✅ PERFECT CALIBRATION", 0.0
        
        # Calculate distance from target range
        if mean_score < target_range[0]:
            points_off = target_range[0] - mean_score
        else:
            points_off = mean_score - target_range[1]
        
        # Apply calibration status levels
        if points_off <= 2.0:
            return "✅ PERFECT CALIBRATION", points_off
        elif points_off <= 5.0:
            return "🟡 GOOD CALIBRATION", points_off
        elif points_off <= 10.0:
            return "🟠 NEEDS CALIBRATION", points_off
        else:
            return "❌ CALIBRATION BROKEN", points_off
    
    def validate_all_reference_tests(self) -> List[CalibrationResult]:
        """
        Validate all reference test cases
        
        Returns:
            List of calibration results
        """
        logger.info(f"Starting calibration validation of {len(REFERENCE_TEST_CASES)} reference tests")
        
        results = []
        
        for i, test_case in enumerate(REFERENCE_TEST_CASES, 1):
            logger.info(f"\n=== Test {i}/{len(REFERENCE_TEST_CASES)}: {test_case['name']} ===")
            
            result = self.run_multi_sample_test(test_case)
            results.append(result)
            
            # Log immediate result
            self.reporter.log_test_result(result)
        
        logger.info(f"\nCalibration validation complete: {len(results)} tests")
        return results
    
    def generate_calibration_report(self, results: List[CalibrationResult], output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive calibration report
        
        Args:
            results: List of calibration results
            output_file: Optional JSON output file path
            
        Returns:
            Report dictionary
        """
        report = self.reporter.generate_report(results)
        
        # Save to JSON if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Calibration report saved to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save report to {output_file}: {e}")
        
        return report

def main():
    """Main calibration validation function"""
    logger.info("🔧 Programmatic Calibration Validation Framework")
    logger.info("=" * 60)
    
    try:
        # Initialize validator
        validator = CalibrationValidator(sample_count=3)
        
        # Setup benchmark runner
        if not validator.setup_benchmark_runner():
            logger.error("Failed to setup benchmark runner")
            return 1
        
        # Run calibration validation
        results = validator.validate_all_reference_tests()
        
        # Generate report
        report = validator.generate_calibration_report(
            results, 
            output_file="calibration_validation_results.json"
        )
        
        # Print summary
        validator.reporter.print_summary_report(report)
        
        # Return success/failure based on calibration results
        overall_success = report.get('overall_calibration_status', '❌') in ['✅', '🟡']
        return 0 if overall_success else 1
        
    except Exception as e:
        logger.error(f"Calibration validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())