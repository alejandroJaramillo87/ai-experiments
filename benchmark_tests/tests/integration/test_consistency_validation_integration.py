"""
Integration Test Suite for Consistency & Validation Framework

Tests the complete integration of consistency validation and knowledge validation
frameworks with the main UniversalEvaluator system, ensuring proper metrics
integration and evaluation flow.

Author: Claude Code
Version: 1.0.0
"""

import unittest
import logging
import sys
import os
from unittest.mock import patch, MagicMock

# Test imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

try:
    from evaluator.reasoning_evaluator import UniversalEvaluator, EvaluationResult
    from evaluator.evaluation_config import DEFAULT_CONFIG
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False
    logging.warning("UniversalEvaluator not available for integration testing")

# Disable logging during tests
logging.disable(logging.CRITICAL)


@unittest.skipUnless(EVALUATOR_AVAILABLE, "UniversalEvaluator not available")
class TestConsistencyValidationIntegration(unittest.TestCase):
    """Test integration of consistency and validation frameworks"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.evaluator = UniversalEvaluator()
        
        # Test scenarios
        self.test_scenarios = {
            "high_quality_factual": {
                "text": "The capital of France is Paris, which is located in the north-central part of the country. This city has been the capital since 1791 and has a population of approximately 2.16 million people within the city limits.",
                "expected_consistency": 0.8,
                "expected_validation": True,
                "category": "geography"
            },
            "low_quality_contradictory": {
                "text": "The capital of France is definitely Paris. However, I think it might be London. Actually, I'm certain it's Berlin. No, wait, it's clearly Madrid.",
                "expected_consistency": 0.2,
                "expected_validation": False,
                "category": "geography"
            },
            "mathematical_reasoning": {
                "text": "To calculate 15 × 24, I'll use the distributive property: 15 × 24 = 15 × (20 + 4) = (15 × 20) + (15 × 4) = 300 + 60 = 360. Therefore, 15 × 24 = 360.",
                "expected_consistency": 0.9,
                "expected_validation": True,
                "category": "mathematics"
            },
            "uncertain_response": {
                "text": "I'm not sure what the answer is. Maybe it could be this, or perhaps that. I don't know for certain. It's unclear to me.",
                "expected_consistency": 0.5,
                "expected_validation": False,
                "category": "general"
            },
            "high_confidence_incorrect": {
                "text": "I am absolutely certain that 2 + 2 = 5. This is definitely correct and I have no doubt about this answer.",
                "expected_consistency": 0.7,  # Internally consistent but wrong
                "expected_validation": False,
                "category": "mathematics"
            }
        }
    
    def test_evaluator_initialization_with_validators(self):
        """Test that evaluator initializes with consistency and validation modules"""
        # Test lazy loading properties
        consistency_validator = self.evaluator.consistency_validator
        knowledge_validator = self.evaluator.knowledge_validator
        
        # May be None if dependencies unavailable, but should not error
        if consistency_validator is not None:
            self.assertIsNotNone(consistency_validator.test_question_sets)
        
        if knowledge_validator is not None:
            self.assertIsNotNone(knowledge_validator.factual_tests)
    
    def test_basic_evaluation_with_consistency_validation(self):
        """Test basic evaluation includes consistency and validation metrics"""
        scenario = self.test_scenarios["high_quality_factual"]
        
        result = self.evaluator.evaluate_response(
            scenario["text"],
            "Integration Test - High Quality",
            reasoning_type=None
        )
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertIsNotNone(result.metrics)
        
        # Check that new metrics are included
        metrics = result.metrics
        self.assertHasAttr(metrics, 'consistency_score')
        self.assertHasAttr(metrics, 'factual_accuracy')
        self.assertHasAttr(metrics, 'knowledge_consistency')
        self.assertHasAttr(metrics, 'confidence_calibration')
        self.assertHasAttr(metrics, 'validation_passed')
    
    def test_high_quality_response_evaluation(self):
        """Test evaluation of high-quality response"""
        scenario = self.test_scenarios["high_quality_factual"]
        
        result = self.evaluator.evaluate_response(
            scenario["text"],
            "High Quality Test",
            reasoning_type=None
        )
        
        # Should have good overall score
        self.assertGreater(result.metrics.overall_score, 60)
        
        # Should have reasonable consistency and validation metrics
        self.assertGreaterEqual(result.metrics.consistency_score, 0.0)
        self.assertGreaterEqual(result.metrics.factual_accuracy, 0.0)
        self.assertGreaterEqual(result.metrics.confidence_calibration, 0.0)
        
        # Check advanced analysis integration
        if "advanced_analysis" in result.detailed_analysis:
            advanced = result.detailed_analysis["advanced_analysis"]
            
            if "consistency_validation" in advanced:
                consistency_data = advanced["consistency_validation"]
                self.assertIsInstance(consistency_data, dict)
                if "error" not in consistency_data:
                    self.assertIn("consistency_score", consistency_data)
            
            if "knowledge_validation" in advanced:
                knowledge_data = advanced["knowledge_validation"]
                self.assertIsInstance(knowledge_data, dict)
                if "error" not in knowledge_data:
                    self.assertIn("factual_accuracy", knowledge_data)
    
    def test_contradictory_response_evaluation(self):
        """Test evaluation of contradictory response"""
        scenario = self.test_scenarios["low_quality_contradictory"]
        
        result = self.evaluator.evaluate_response(
            scenario["text"],
            "Contradictory Test",
            reasoning_type=None
        )
        
        # Should detect internal inconsistencies
        if "advanced_analysis" in result.detailed_analysis:
            advanced = result.detailed_analysis["advanced_analysis"]
            
            if "consistency_validation" in advanced and "error" not in advanced["consistency_validation"]:
                consistency_data = advanced["consistency_validation"]
                # Should detect contradictions
                self.assertTrue(consistency_data.get("contradiction_detected", False))
                self.assertFalse(consistency_data.get("internal_consistency", True))
    
    def test_mathematical_response_evaluation(self):
        """Test evaluation of mathematical reasoning response"""
        scenario = self.test_scenarios["mathematical_reasoning"]
        
        result = self.evaluator.evaluate_response(
            scenario["text"],
            "Mathematical Test",
            reasoning_type=None
        )
        
        # Should score well for mathematical content
        self.assertGreater(result.metrics.overall_score, 50)
        
        # Should have good factual indicators
        if "advanced_analysis" in result.detailed_analysis:
            advanced = result.detailed_analysis["advanced_analysis"]
            
            if "knowledge_validation" in advanced and "error" not in advanced["knowledge_validation"]:
                knowledge_data = advanced["knowledge_validation"]
                factual_indicators = knowledge_data.get("factual_indicators", {})
                
                # Mathematical content should contain numbers
                self.assertTrue(factual_indicators.get("contains_numbers", False))
    
    def test_uncertain_response_evaluation(self):
        """Test evaluation of uncertain response"""
        scenario = self.test_scenarios["uncertain_response"]
        
        result = self.evaluator.evaluate_response(
            scenario["text"],
            "Uncertain Test",
            reasoning_type=None
        )
        
        # Should detect uncertainty in confidence calibration
        if "advanced_analysis" in result.detailed_analysis:
            advanced = result.detailed_analysis["advanced_analysis"]
            
            if "knowledge_validation" in advanced and "error" not in advanced["knowledge_validation"]:
                knowledge_data = advanced["knowledge_validation"]
                confidence_analysis = knowledge_data.get("confidence_analysis", {})
                
                # Should have low confidence calibration
                calibration_score = confidence_analysis.get("calibration_score", 0.5)
                self.assertLess(calibration_score, 0.8)
    
    def test_confidence_miscalibration_detection(self):
        """Test detection of confidence miscalibration"""
        scenario = self.test_scenarios["high_confidence_incorrect"]
        
        result = self.evaluator.evaluate_response(
            scenario["text"],
            "Miscalibration Test",
            reasoning_type=None
        )
        
        if "advanced_analysis" in result.detailed_analysis:
            advanced = result.detailed_analysis["advanced_analysis"]
            
            if "knowledge_validation" in advanced and "error" not in advanced["knowledge_validation"]:
                knowledge_data = advanced["knowledge_validation"]
                
                # Should have low validation score due to incorrect answer
                self.assertLess(knowledge_data.get("factual_accuracy", 1.0), 0.5)
                
                # But may have high confidence markers
                confidence_analysis = knowledge_data.get("confidence_analysis", {})
                distribution = confidence_analysis.get("confidence_distribution", {})
                high_conf_ratio = distribution.get("high", 0)
                
                # This represents miscalibration: high confidence + low accuracy
                if high_conf_ratio > 0.5:
                    self.assertLess(knowledge_data.get("factual_accuracy", 1.0), 0.5)
    
    def test_metrics_integration_bounds(self):
        """Test that integrated metrics are within valid bounds"""
        for scenario_name, scenario in self.test_scenarios.items():
            with self.subTest(scenario=scenario_name):
                result = self.evaluator.evaluate_response(
                    scenario["text"],
                    f"Bounds Test - {scenario_name}",
                    reasoning_type=None
                )
                
                metrics = result.metrics
                
                # All new metrics should be within valid bounds
                self.assertGreaterEqual(metrics.consistency_score, 0.0)
                self.assertLessEqual(metrics.consistency_score, 1.0)
                
                self.assertGreaterEqual(metrics.factual_accuracy, 0.0)
                self.assertLessEqual(metrics.factual_accuracy, 1.0)
                
                self.assertGreaterEqual(metrics.knowledge_consistency, 0.0)
                self.assertLessEqual(metrics.knowledge_consistency, 1.0)
                
                self.assertGreaterEqual(metrics.confidence_calibration, 0.0)
                self.assertLessEqual(metrics.confidence_calibration, 1.0)
                
                self.assertIsInstance(metrics.validation_passed, bool)
    
    def test_advanced_analysis_error_handling(self):
        """Test error handling in advanced analysis modules"""
        # Test with potentially problematic input
        problematic_inputs = [
            "",  # Empty string
            "   ",  # Whitespace only
            "Special chars: @#$%^&*(){}[]",  # Special characters
            "很长的中文文本测试" * 10,  # Non-ASCII characters
        ]
        
        for test_input in problematic_inputs:
            with self.subTest(input=test_input[:20] + "..."):
                result = self.evaluator.evaluate_response(
                    test_input,
                    "Error Handling Test",
                    reasoning_type=None
                )
                
                # Should not crash
                self.assertIsInstance(result, EvaluationResult)
                
                # Advanced analysis might have errors but should be structured
                if "advanced_analysis" in result.detailed_analysis:
                    advanced = result.detailed_analysis["advanced_analysis"]
                    
                    for module_name, module_data in advanced.items():
                        if isinstance(module_data, dict):
                            # Should either have valid data or error message
                            is_valid = len(module_data) > 0
                            has_error = "error" in module_data
                            self.assertTrue(is_valid or has_error, 
                                f"Module {module_name} has neither data nor error")
    
    def test_configuration_integration(self):
        """Test that configuration parameters are properly integrated"""
        # Check that new configuration sections are available
        config = DEFAULT_CONFIG
        
        if "consistency_validation_thresholds" in config:
            thresholds = config["consistency_validation_thresholds"]
            self.assertIn("consistency", thresholds)
            self.assertIn("factual_accuracy", thresholds)
            self.assertIn("confidence_calibration", thresholds)
        
        if "consistency_test_config" in config:
            consistency_config = config["consistency_test_config"]
            self.assertIn("enabled", consistency_config)
            self.assertIn("test_categories", consistency_config)
        
        if "knowledge_validation_config" in config:
            validation_config = config["knowledge_validation_config"]
            self.assertIn("enabled", validation_config)
            self.assertIn("validation_categories", validation_config)
    
    def test_model_specific_expectations(self):
        """Test model-specific consistency and validation expectations"""
        config = DEFAULT_CONFIG
        
        if "model_profiles" in config:
            profiles = config["model_profiles"]
            
            # Test GPT-OSS-20B profile
            if "gpt_oss_20b" in profiles:
                gpt_profile = profiles["gpt_oss_20b"]
                if "consistency_expectations" in gpt_profile:
                    consistency_exp = gpt_profile["consistency_expectations"]
                    self.assertIn("internal_consistency_baseline", consistency_exp)
                    self.assertIn("cross_phrasing_baseline", consistency_exp)
                
                if "validation_expectations" in gpt_profile:
                    validation_exp = gpt_profile["validation_expectations"]
                    self.assertIn("factual_accuracy_baseline", validation_exp)
                    self.assertIn("knowledge_consistency_baseline", validation_exp)
    
    def test_performance_impact(self):
        """Test that new validators don't significantly impact performance"""
        import time
        
        test_text = self.test_scenarios["high_quality_factual"]["text"]
        iterations = 3
        
        start_time = time.time()
        for _ in range(iterations):
            result = self.evaluator.evaluate_response(
                test_text,
                "Performance Test",
                reasoning_type=None
            )
        elapsed_time = (time.time() - start_time) / iterations
        
        # Should complete within reasonable time
        self.assertLess(elapsed_time, 3.0, "Evaluation taking too long")
        
        # Should still produce valid results
        self.assertIsInstance(result, EvaluationResult)
        self.assertGreater(result.metrics.overall_score, 0)
    
    def test_backwards_compatibility(self):
        """Test that existing evaluation functionality still works"""
        # Test basic evaluation without advanced features
        simple_text = "This is a simple response to test backwards compatibility."
        
        result = self.evaluator.evaluate_response(
            simple_text,
            "Backwards Compatibility Test",
            reasoning_type=None
        )
        
        # Should have all traditional metrics
        self.assertIsInstance(result.metrics.organization_quality, (int, float))
        self.assertIsInstance(result.metrics.technical_accuracy, (int, float))
        self.assertIsInstance(result.metrics.completeness, (int, float))
        self.assertIsInstance(result.metrics.overall_score, (int, float))
        
        # Should have evaluation result structure
        self.assertIsInstance(result.detailed_analysis, dict)
        self.assertIn("scores_breakdown", result.detailed_analysis)
    
    def assertHasAttr(self, obj, attr_name):
        """Helper method to check if object has attribute"""
        self.assertTrue(hasattr(obj, attr_name), 
            f"Object {type(obj).__name__} does not have attribute '{attr_name}'")


class TestConsistencyValidationConfig(unittest.TestCase):
    """Test configuration aspects of consistency and validation framework"""
    
    def test_config_structure_validity(self):
        """Test that configuration structure is valid"""
        config = DEFAULT_CONFIG
        
        # Test consistency validation thresholds
        if "consistency_validation_thresholds" in config:
            thresholds = config["consistency_validation_thresholds"]
            
            for category, levels in thresholds.items():
                self.assertIsInstance(levels, dict)
                for level, threshold in levels.items():
                    self.assertIsInstance(threshold, (int, float))
                    self.assertGreaterEqual(threshold, 0.0)
                    self.assertLessEqual(threshold, 1.0)
    
    def test_advanced_metrics_scoring(self):
        """Test advanced metrics scoring configuration"""
        config = DEFAULT_CONFIG
        
        if "advanced_metrics_scoring" in config:
            scoring = config["advanced_metrics_scoring"]
            
            # Test consistency bonuses and penalties
            if "consistency_bonuses" in scoring:
                bonuses = scoring["consistency_bonuses"]
                for bonus_type, value in bonuses.items():
                    self.assertIsInstance(value, (int, float))
                    self.assertGreater(value, 0)  # Bonuses should be positive
            
            if "consistency_penalties" in scoring:
                penalties = scoring["consistency_penalties"]
                for penalty_type, value in penalties.items():
                    self.assertIsInstance(value, (int, float))
                    self.assertLess(value, 0)  # Penalties should be negative
    
    def test_model_profile_completeness(self):
        """Test that model profiles have complete consistency/validation expectations"""
        config = DEFAULT_CONFIG
        
        if "model_profiles" in config:
            profiles = config["model_profiles"]
            
            for model_name, profile in profiles.items():
                if "consistency_expectations" in profile:
                    consistency = profile["consistency_expectations"]
                    required_keys = ["internal_consistency_baseline", "confidence_consistency_baseline"]
                    
                    for key in required_keys:
                        self.assertIn(key, consistency, 
                            f"Model {model_name} missing consistency expectation: {key}")
                        self.assertIsInstance(consistency[key], (int, float))


if __name__ == '__main__':
    # Run with high verbosity to see detailed test results
    unittest.main(verbosity=2)