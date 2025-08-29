"""
Test suite for Reasoning Evaluator

Tests for reasoning evaluation including universal metrics, reasoning type detection,
batch evaluation, and various reasoning patterns.

Author: Claude Code
Version: 1.0.0
"""

import unittest
import logging
import sys
import os
from datetime import datetime

# Test imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from evaluator.reasoning_evaluator import (
    UniversalEvaluator, ReasoningType, EvaluationMetrics, EvaluationResult,
    evaluate_reasoning
)

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestReasoningEvaluator(unittest.TestCase):
    """Test reasoning evaluator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = UniversalEvaluator()
        
        # Sample responses for different reasoning types
        self.chain_of_thought_response = """
        Let me think through this step by step.
        
        First, I need to identify the key components of the problem:
        - We have two variables: x and y
        - The constraint is x + y = 10
        - We want to maximize x * y
        
        Next, I'll use substitution to solve this:
        - If x + y = 10, then y = 10 - x
        - So I need to maximize x * (10 - x) = 10x - x²
        
        To find the maximum, I'll take the derivative:
        - d/dx(10x - x²) = 10 - 2x
        - Setting equal to 0: 10 - 2x = 0
        - Solving: x = 5
        
        Therefore, y = 10 - 5 = 5.
        
        The maximum value of x * y is 5 * 5 = 25.
        """
        
        self.mathematical_response = """
        Given: f(x) = 2x² + 3x - 1
        Find: f'(x)
        
        Using the power rule and sum rule:
        f'(x) = d/dx(2x²) + d/dx(3x) - d/dx(1)
        f'(x) = 2(2x) + 3 - 0
        f'(x) = 4x + 3
        
        Therefore, the derivative is 4x + 3.
        """
        
        self.multi_step_response = """
        To solve this problem, I'll break it down into several steps:
        
        Step 1: Gather the data
        We have sales figures for Q1: $10,000, Q2: $15,000, Q3: $12,000
        
        Step 2: Calculate the average
        Average = (10,000 + 15,000 + 12,000) / 3 = 37,000 / 3 = $12,333.33
        
        Step 3: Analyze the trend
        Q1 to Q2: +50% growth
        Q2 to Q3: -20% decline
        
        Step 4: Make a recommendation
        Based on this data, we should investigate what caused the Q3 decline
        and implement strategies to return to Q2 performance levels.
        """
        
        self.poor_reasoning_response = """
        The answer is 42. This is obviously correct because everyone knows that.
        No explanation needed. It's just true.
        """
        
        self.short_response = """
        Yes.
        """
        
        self.complex_reasoning_response = """
        This multifaceted problem requires careful analysis across several dimensions.
        
        From a theoretical perspective, we must consider the underlying principles
        that govern this system. The fundamental equations suggest that when we
        have variable A interacting with system B, the resulting dynamic C exhibits
        properties consistent with established models in the literature.
        
        Empirically, the data supports this theoretical framework. Analysis of
        the sample set (n=1,247) shows correlation coefficients of r=0.83 with
        p<0.001, indicating statistical significance. The confidence intervals
        [0.76, 0.91] provide reasonable bounds for our estimates.
        
        However, we must acknowledge several limitations: sampling bias may affect
        generalizability, temporal factors could introduce variability, and
        measurement precision remains a concern.
        
        In conclusion, while the evidence strongly supports the primary hypothesis,
        additional research with controlled conditions would strengthen our confidence
        in these findings and their practical applications.
        """
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        self.assertIsNotNone(self.evaluator)
        self.assertIsNotNone(self.evaluator.config)
        
        # Test initialization with custom config
        evaluator_custom = UniversalEvaluator(config_path=None)
        self.assertIsNotNone(evaluator_custom)
    
    def test_reasoning_type_enum(self):
        """Test ReasoningType enum"""
        # Test enum values
        self.assertEqual(ReasoningType.CHAIN_OF_THOUGHT.value, "chain_of_thought")
        self.assertEqual(ReasoningType.MATHEMATICAL.value, "mathematical")
        self.assertEqual(ReasoningType.MULTI_STEP.value, "multi_step")
        self.assertEqual(ReasoningType.GENERAL.value, "general")
        
        # Test all enum members exist
        expected_types = [
            "CHAIN_OF_THOUGHT", "MULTI_STEP", "VERIFICATION", "MATHEMATICAL",
            "MULTI_HOP", "SCAFFOLDED", "BACKWARD", "GENERAL"
        ]
        
        for type_name in expected_types:
            self.assertTrue(hasattr(ReasoningType, type_name))
    
    def test_evaluation_metrics_dataclass(self):
        """Test EvaluationMetrics dataclass"""
        metrics = EvaluationMetrics(
            organization_quality=0.8,
            technical_accuracy=0.7,
            completeness=0.9,
            thoroughness=0.6,
            reliability=0.8,
            scope_coverage=0.7,
            domain_appropriateness=0.8,
            overall_score=75.5,
            word_count=150,
            confidence_score=0.85
        )
        
        self.assertEqual(metrics.organization_quality, 0.8)
        self.assertEqual(metrics.overall_score, 75.5)
        self.assertEqual(metrics.word_count, 150)
        
        # Test default values for advanced metrics
        self.assertEqual(metrics.token_entropy, 0.0)
        self.assertEqual(metrics.cultural_authenticity, 0.0)
        self.assertEqual(metrics.validation_passed, False)
    
    def test_evaluate_response_basic(self):
        """Test basic response evaluation"""
        result = self.evaluator.evaluate_response(
            self.chain_of_thought_response,
            "test_chain_of_thought",
            ReasoningType.CHAIN_OF_THOUGHT
        )
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertIsInstance(result.metrics, EvaluationMetrics)
        self.assertEqual(result.reasoning_type, ReasoningType.CHAIN_OF_THOUGHT)
        self.assertIsInstance(result.detailed_analysis, dict)
        self.assertIsInstance(result.recommendations, list)
        self.assertIsInstance(result.timestamp, str)
        
        # Check metrics bounds
        metrics = result.metrics
        self.assertGreaterEqual(metrics.overall_score, 0.0)
        self.assertLessEqual(metrics.overall_score, 100.0)
        self.assertGreaterEqual(metrics.organization_quality, 0.0)
        self.assertLessEqual(metrics.organization_quality, 100.0)
        self.assertGreaterEqual(metrics.technical_accuracy, 0.0)
        self.assertLessEqual(metrics.technical_accuracy, 100.0)
        self.assertGreater(metrics.word_count, 0)
    
    def test_evaluate_response_mathematical(self):
        """Test mathematical reasoning evaluation"""
        result = self.evaluator.evaluate_response(
            self.mathematical_response,
            "test_mathematical",
            ReasoningType.MATHEMATICAL
        )
        
        self.assertEqual(result.reasoning_type, ReasoningType.MATHEMATICAL)
        self.assertGreater(result.metrics.overall_score, 10.0)  # Should get decent score
        self.assertGreater(result.metrics.word_count, 20)
    
    def test_evaluate_response_multi_step(self):
        """Test multi-step reasoning evaluation"""
        result = self.evaluator.evaluate_response(
            self.multi_step_response,
            "test_multi_step",
            ReasoningType.MULTI_STEP
        )
        
        self.assertEqual(result.reasoning_type, ReasoningType.MULTI_STEP)
        self.assertGreater(result.metrics.overall_score, 10.0)
        # Should have good organization due to explicit steps
        self.assertGreater(result.metrics.organization_quality, 10.0)
    
    def test_evaluate_response_poor_quality(self):
        """Test evaluation of poor quality response"""
        result = self.evaluator.evaluate_response(
            self.poor_reasoning_response,
            "test_poor",
            ReasoningType.GENERAL
        )
        
        # Poor response should get low scores
        self.assertLess(result.metrics.overall_score, 40.0)
        self.assertLess(result.metrics.completeness, 50.0)
        self.assertLess(result.metrics.thoroughness, 50.0)
    
    def test_evaluate_response_short_response(self):
        """Test evaluation of very short response"""
        result = self.evaluator.evaluate_response(
            self.short_response,
            "test_short",
            ReasoningType.GENERAL
        )
        
        # Short response should be handled gracefully
        self.assertIsInstance(result, EvaluationResult)
        self.assertLess(result.metrics.overall_score, 30.0)
        self.assertLess(result.metrics.thoroughness, 30.0)
        self.assertEqual(result.metrics.word_count, 1)
    
    def test_evaluate_response_auto_detect_type(self):
        """Test automatic reasoning type detection"""
        result = self.evaluator.evaluate_response(
            self.chain_of_thought_response,
            "test_auto_detect",
            reasoning_type=None  # Let it auto-detect
        )
        
        # Should auto-detect as some reasoning type
        self.assertIsInstance(result.reasoning_type, ReasoningType)
        self.assertIsInstance(result, EvaluationResult)
    
    def test_evaluate_response_with_category(self):
        """Test evaluation with test category"""
        result = self.evaluator.evaluate_response(
            self.mathematical_response,
            "test_category",
            ReasoningType.MATHEMATICAL,
            test_category="mathematics"
        )
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertEqual(result.reasoning_type, ReasoningType.MATHEMATICAL)
    
    def test_evaluate_batch(self):
        """Test batch evaluation"""
        responses = [
            (self.chain_of_thought_response, "test1"),
            (self.mathematical_response, "test2"),
            (self.multi_step_response, "test3")
        ]
        
        results = self.evaluator.evaluate_batch(responses)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, EvaluationResult)
            self.assertGreater(result.metrics.overall_score, 0.0)
    
    def test_evaluate_batch_with_types(self):
        """Test batch evaluation with specified reasoning types"""
        responses = [
            (self.chain_of_thought_response, "test1"),
            (self.mathematical_response, "test2")
        ]
        
        reasoning_types = [ReasoningType.CHAIN_OF_THOUGHT, ReasoningType.MATHEMATICAL]
        
        results = self.evaluator.evaluate_batch(responses, reasoning_types)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].reasoning_type, ReasoningType.CHAIN_OF_THOUGHT)
        self.assertEqual(results[1].reasoning_type, ReasoningType.MATHEMATICAL)
    
    def test_generate_summary_report(self):
        """Test summary report generation"""
        responses = [
            (self.chain_of_thought_response, "test1"),
            (self.mathematical_response, "test2"),
            (self.poor_reasoning_response, "test3")
        ]
        
        results = self.evaluator.evaluate_batch(responses)
        summary = self.evaluator.generate_summary_report(results)
        
        self.assertIsInstance(summary, dict)
        self.assertIn('summary_statistics', summary)
        self.assertIn('metric_averages', summary)
        self.assertIn('recommendations', summary)
        
        # Check summary statistics
        stats = summary['summary_statistics']
        self.assertIn('total_evaluations', stats)
        self.assertIn('average_score', stats)
        self.assertEqual(stats['total_evaluations'], 3)
        self.assertGreaterEqual(stats['average_score'], 0.0)
        self.assertLessEqual(stats['average_score'], 100.0)
    
    def test_standalone_evaluate_reasoning_function(self):
        """Test standalone evaluate_reasoning function"""
        result = evaluate_reasoning(
            self.chain_of_thought_response,
            "standalone_test"
        )
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertIsInstance(result.metrics, EvaluationMetrics)
        self.assertGreater(result.metrics.overall_score, 0.0)
    
    def test_empty_or_invalid_input(self):
        """Test handling of empty or invalid input"""
        # Empty string
        result = self.evaluator.evaluate_response(
            "",
            "empty_test",
            ReasoningType.GENERAL
        )
        self.assertIsInstance(result, EvaluationResult)
        # Should handle gracefully, likely with low score
        
        # Very short string
        result = self.evaluator.evaluate_response(
            "Hi",
            "short_test",
            ReasoningType.GENERAL
        )
        self.assertIsInstance(result, EvaluationResult)
        self.assertLess(result.metrics.overall_score, 30.0)
    
    def test_complex_reasoning_response(self):
        """Test evaluation of complex reasoning response"""
        result = self.evaluator.evaluate_response(
            self.complex_reasoning_response,
            "complex_test",
            ReasoningType.GENERAL
        )
        
        # Complex response should score well
        self.assertGreater(result.metrics.overall_score, 60.0)
        self.assertGreater(result.metrics.thoroughness, 0.6)
        self.assertGreater(result.metrics.completeness, 0.5)
        self.assertGreater(result.metrics.word_count, 100)
    
    def test_metric_score_bounds(self):
        """Test that all metric scores are within valid bounds"""
        test_responses = [
            self.chain_of_thought_response,
            self.mathematical_response,
            self.multi_step_response,
            self.poor_reasoning_response,
            self.short_response,
            self.complex_reasoning_response
        ]
        
        for response in test_responses:
            result = self.evaluator.evaluate_response(
                response,
                "bounds_test",
                ReasoningType.GENERAL
            )
            
            metrics = result.metrics
            
            # Check primary metric bounds (0.0 to 100.0 for main metrics)
            self.assertGreaterEqual(metrics.organization_quality, 0.0)
            self.assertLessEqual(metrics.organization_quality, 100.0)
            self.assertGreaterEqual(metrics.technical_accuracy, 0.0)
            self.assertLessEqual(metrics.technical_accuracy, 100.0)
            self.assertGreaterEqual(metrics.completeness, 0.0)
            self.assertLessEqual(metrics.completeness, 100.0)
            self.assertGreaterEqual(metrics.thoroughness, 0.0)
            self.assertLessEqual(metrics.thoroughness, 100.0)
            self.assertGreaterEqual(metrics.reliability, 0.0)
            self.assertLessEqual(metrics.reliability, 100.0)
            self.assertGreaterEqual(metrics.scope_coverage, 0.0)
            self.assertLessEqual(metrics.scope_coverage, 100.0)
            self.assertGreaterEqual(metrics.domain_appropriateness, 0.0)
            self.assertLessEqual(metrics.domain_appropriateness, 100.0)
            self.assertGreaterEqual(metrics.confidence_score, 0.0)
            self.assertLessEqual(metrics.confidence_score, 100.0)
            
            # Check overall score bounds (0.0 to 100.0)
            self.assertGreaterEqual(metrics.overall_score, 0.0)
            self.assertLessEqual(metrics.overall_score, 100.0)
            
            # Check word count is non-negative
            self.assertGreaterEqual(metrics.word_count, 0)
    
    def test_reasoning_type_string_conversion(self):
        """Test reasoning type string conversion"""
        # Test with string input
        result = self.evaluator.evaluate_response(
            self.chain_of_thought_response,
            "string_type_test",
            reasoning_type="chain_of_thought"
        )
        
        self.assertEqual(result.reasoning_type, ReasoningType.CHAIN_OF_THOUGHT)
    
    def test_detailed_analysis_structure(self):
        """Test structure of detailed analysis"""
        result = self.evaluator.evaluate_response(
            self.chain_of_thought_response,
            "analysis_test",
            ReasoningType.CHAIN_OF_THOUGHT
        )
        
        analysis = result.detailed_analysis
        self.assertIsInstance(analysis, dict)
        
        # Should contain key analysis components
        # Note: exact keys depend on implementation details
        self.assertGreater(len(analysis), 0)
        
    def test_recommendations_generation(self):
        """Test that recommendations are generated"""
        result = self.evaluator.evaluate_response(
            self.poor_reasoning_response,
            "recommendations_test",
            ReasoningType.GENERAL
        )
        
        self.assertIsInstance(result.recommendations, list)
        # Poor response should generate recommendations
        self.assertGreater(len(result.recommendations), 0)
    
    def test_timestamp_format(self):
        """Test that timestamp is properly formatted"""
        result = self.evaluator.evaluate_response(
            self.chain_of_thought_response,
            "timestamp_test",
            ReasoningType.GENERAL
        )
        
        # Should be able to parse the timestamp
        self.assertIsInstance(result.timestamp, str)
        self.assertGreater(len(result.timestamp), 0)
        
        # Try to parse it as datetime to verify format
        try:
            datetime.fromisoformat(result.timestamp.replace('Z', '+00:00'))
        except ValueError:
            # If ISO format fails, try other common formats
            # Just check it's a non-empty string
            self.assertIsInstance(result.timestamp, str)
            self.assertGreater(len(result.timestamp), 10)


class TestReasoningEvaluatorConfiguration(unittest.TestCase):
    """Test reasoning evaluator configuration handling"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        evaluator = UniversalEvaluator()
        self.assertIsNotNone(evaluator.config)
        
        # Config should be a dictionary
        self.assertIsInstance(evaluator.config, dict)
    
    def test_invalid_config_path(self):
        """Test handling of invalid config path"""
        # Should not raise an error, should use defaults
        evaluator = UniversalEvaluator(config_path="nonexistent_file.json")
        self.assertIsNotNone(evaluator)
        self.assertIsNotNone(evaluator.config)


class TestReasoningEvaluatorEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = UniversalEvaluator()
    
    def test_very_long_response(self):
        """Test handling of very long responses"""
        long_response = "This is a reasoning response. " * 1000  # ~30,000 characters
        
        result = self.evaluator.evaluate_response(
            long_response,
            "long_response_test",
            ReasoningType.GENERAL
        )
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertGreater(result.metrics.word_count, 5000)
        self.assertGreaterEqual(result.metrics.overall_score, 0.0)
    
    def test_special_characters_handling(self):
        """Test handling of special characters and unicode"""
        special_response = """
        Mathematical proof with symbols: ∀x∈ℝ, x² ≥ 0
        Greek letters: α, β, γ, δ, ε
        Special punctuation: "quotes", 'apostrophes', —em-dashes—
        Numbers: 3.14159, 2.71828, 1.41421
        """
        
        result = self.evaluator.evaluate_response(
            special_response,
            "special_chars_test",
            ReasoningType.MATHEMATICAL
        )
        
        self.assertIsInstance(result, EvaluationResult)
        self.assertGreaterEqual(result.metrics.overall_score, 0.0)
    
    def test_mixed_content_response(self):
        """Test response with mixed content types"""
        mixed_response = """
        Here's a step-by-step analysis:
        
        1. First, let's look at the data:
           - Sample A: 15.3
           - Sample B: 22.7
           - Sample C: 18.9
        
        2. Calculate the mean: (15.3 + 22.7 + 18.9) / 3 = 18.97
        
        3. The mathematical relationship can be expressed as:
           f(x) = ax² + bx + c
        
        4. Conclusion: The data suggests a quadratic relationship.
        """
        
        result = self.evaluator.evaluate_response(
            mixed_response,
            "mixed_content_test",
            ReasoningType.MULTI_STEP
        )
        
        # Should handle mixed content well
        self.assertGreater(result.metrics.overall_score, 10.0)
        self.assertGreater(result.metrics.organization_quality, 10.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)