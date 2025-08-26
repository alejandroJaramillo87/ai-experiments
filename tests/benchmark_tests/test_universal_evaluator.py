#!/usr/bin/env python3
"""
Comprehensive Test Suite for UniversalEvaluator and Evaluation Configuration

Tests the UniversalEvaluator system without mocking, using real functionality
to verify correct behavior across different test types.

Author: Claude Code
Version: 1.0.0
"""

import unittest
import sys
import os

# Add the benchmark_tests directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reasoning_evaluator import UniversalEvaluator, evaluate_reasoning, EvaluationMetrics, ReasoningType
from evaluation_config import DEFAULT_CONFIG, UniversalWeights


class TestEvaluationConfig(unittest.TestCase):
    """Test suite for evaluation configuration validation"""
    
    def test_universal_weights_sum_to_one(self):
        """Test that UniversalWeights class weights sum to 1.0"""
        total = (UniversalWeights.ORGANIZATION_QUALITY + 
                UniversalWeights.TECHNICAL_ACCURACY +
                UniversalWeights.COMPLETENESS +
                UniversalWeights.THOROUGHNESS +
                UniversalWeights.RELIABILITY +
                UniversalWeights.SCOPE_COVERAGE +
                UniversalWeights.DOMAIN_APPROPRIATENESS)
        
        self.assertAlmostEqual(total, 1.0, places=3, 
                              msg=f"UniversalWeights sum to {total}, not 1.0")
    
    def test_default_config_weights_sum_to_one(self):
        """Test that default configuration weights sum to 1.0"""
        weights = DEFAULT_CONFIG["weights"]
        total = sum(weights.values())
        
        self.assertAlmostEqual(total, 1.0, places=3,
                              msg=f"Default config weights sum to {total}, not 1.0")
    
    def test_all_universal_metrics_present_in_default(self):
        """Test that all universal metrics are present in default config"""
        required_metrics = [
            "organization_quality", "technical_accuracy", "completeness",
            "thoroughness", "reliability", "scope_coverage", "domain_appropriateness"
        ]
        
        weights = DEFAULT_CONFIG["weights"]
        for metric in required_metrics:
            self.assertIn(metric, weights, f"Missing metric in default config: {metric}")
    
    def test_test_type_configs_weights_sum_to_one(self):
        """Test that test type specific configurations have valid weights"""
        test_type_configs = DEFAULT_CONFIG.get("test_type_configs", {})
        
        for test_type, config in test_type_configs.items():
            if "weights" in config:
                total = sum(config["weights"].values())
                self.assertAlmostEqual(total, 1.0, places=3,
                                     msg=f"{test_type} config weights sum to {total}, not 1.0")
    
    def test_all_metrics_present_in_test_type_configs(self):
        """Test that all metrics are present in test type configurations"""
        required_metrics = [
            "organization_quality", "technical_accuracy", "completeness",
            "thoroughness", "reliability", "scope_coverage", "domain_appropriateness"
        ]
        
        test_type_configs = DEFAULT_CONFIG.get("test_type_configs", {})
        for test_type, config in test_type_configs.items():
            if "weights" in config:
                for metric in required_metrics:
                    self.assertIn(metric, config["weights"],
                                f"Missing {metric} in {test_type} config")


class TestUniversalEvaluator(unittest.TestCase):
    """Test suite for UniversalEvaluator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = UniversalEvaluator()
        
        # Sample responses for different test types
        self.linux_sample = """#!/bin/bash
# System health monitoring script

# Check CPU usage
cpu_usage=$(top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | sed 's/%us,//')
if [ ${cpu_usage%.*} -gt 80 ]; then
    echo "WARNING: CPU usage is high: ${cpu_usage}%"
    logger "High CPU usage detected: ${cpu_usage}%"
fi

# Check memory usage with error handling
memory_usage=$(free | grep Mem | awk '{printf "%.2f", $3/$2 * 100.0}')
if [ ${memory_usage%.*} -gt 90 ]; then
    echo "WARNING: Memory usage is high: ${memory_usage}%"
    logger "High memory usage detected: ${memory_usage}%"
fi

# Check disk usage
disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $disk_usage -gt 85 ]; then
    echo "WARNING: Disk usage is high: ${disk_usage}%"
    logger "High disk usage detected: ${disk_usage}%"
fi

echo "System health check completed successfully"
exit 0
"""

        self.creative_sample = """Here's an innovative approach to this creative challenge that explores multiple dimensions.

First, let me consider various perspectives:
1. Traditional approach: Following conventional methods
2. Lateral thinking: Approaching from unexpected angles  
3. Synthesis approach: Combining disparate elements creatively

For the traditional method, we would typically focus on established patterns. However, I want to explore more unconventional alternatives that could yield unique results.

The lateral thinking approach opens up fascinating possibilities. By considering the problem from completely different viewpoints, we can discover novel solutions that others might miss.

The synthesis approach is particularly exciting because it allows us to combine elements that don't normally go together, creating something truly original and innovative.

This comprehensive exploration ensures we address all requirements while maintaining creative integrity and pushing boundaries.
"""

        self.reasoning_sample = """Let me analyze this step-by-step using careful logical reasoning.

First, I need to examine the evidence provided in the scenario. Based on the data presented, I can identify several key patterns that will inform my analysis.

The evidence shows that there are three main factors to consider:
1. The initial conditions clearly indicate a specific trend
2. The intermediate data points support this trend consistently  
3. The final measurements confirm the pattern holds

Therefore, I can conclude that the hypothesis is well-supported by the evidence. The logical progression from the premises to this conclusion follows a clear chain of reasoning.

To verify this conclusion, let me check my reasoning: if the initial premise is true, and the supporting evidence is valid, then the conclusion must logically follow. This verification step confirms that my analysis is sound.

Hence, the final conclusion is that the evidence strongly supports the proposed hypothesis through multiple lines of convergent reasoning.
"""

        self.poor_linux_sample = "maybe do something with systemctl nginx restart or whatever"
        
        self.poor_reasoning_sample = "I think the answer is probably yes because it seems right."


class TestTypeDetection(TestUniversalEvaluator):
    """Test test type detection functionality"""
    
    def test_linux_category_detection(self):
        """Test that Linux categories are correctly detected"""
        linux_categories = [
            "log_analysis", "containerization", "security", "monitoring",
            "backup", "service_management", "networking", "linux_monitoring"
        ]
        
        for category in linux_categories:
            test_type = self.evaluator._detect_test_type(category)
            self.assertEqual(test_type, "linux", 
                           f"Category '{category}' should detect as 'linux', got '{test_type}'")
    
    def test_creative_category_detection(self):
        """Test that creative categories are correctly detected"""
        creative_categories = [
            "creative_thinking", "strategic_thinking", "ambiguity_handling",
            "metacognitive_reasoning", "constraint_following"
        ]
        
        for category in creative_categories:
            test_type = self.evaluator._detect_test_type(category)
            self.assertEqual(test_type, "creative",
                           f"Category '{category}' should detect as 'creative', got '{test_type}'")
    
    def test_reasoning_category_detection(self):
        """Test that reasoning categories default correctly"""
        reasoning_categories = [
            "complex_synthesis", "chain_of_thought", "mathematical_reasoning",
            "multi_hop_inference", "unknown_category", None
        ]
        
        for category in reasoning_categories:
            test_type = self.evaluator._detect_test_type(category)
            self.assertEqual(test_type, "reasoning",
                           f"Category '{category}' should detect as 'reasoning', got '{test_type}'")


class TestMetricCalculations(TestUniversalEvaluator):
    """Test individual metric calculations"""
    
    def test_linux_technical_accuracy_good_command(self):
        """Test Linux technical accuracy with good commands"""
        good_linux = "sudo systemctl restart nginx && systemctl status nginx"
        score = self.evaluator._calculate_technical_accuracy(good_linux, "linux")
        
        self.assertGreater(score, 15, "Good Linux command should score reasonably well")
        self.assertLessEqual(score, 100, "Score should not exceed maximum")
    
    def test_linux_technical_accuracy_dangerous_command(self):
        """Test Linux technical accuracy penalizes dangerous commands"""
        dangerous_linux = "rm -rf / && chmod 777 /etc/passwd"
        score = self.evaluator._calculate_technical_accuracy(dangerous_linux, "linux")
        
        self.assertLess(score, 50, "Dangerous commands should score poorly")
    
    def test_creative_organization_quality(self):
        """Test creative organization quality scoring"""
        organized_creative = """First, let me explore this creatively.
        
However, there are alternative approaches to consider.
        
Therefore, I propose this innovative solution.
        
In conclusion, this creative approach offers unique advantages."""
        
        score = self.evaluator._calculate_organization_quality(organized_creative, "creative")
        self.assertGreater(score, 5, "Well-organized creative text should score well")
    
    def test_reasoning_completeness(self):
        """Test reasoning completeness scoring"""
        complete_reasoning = "Based on the evidence provided, according to research data, analysis shows clear patterns"
        score = self.evaluator._calculate_completeness(complete_reasoning, "reasoning")
        
        self.assertGreater(score, 20, "Complete reasoning should score well")
    
    def test_metric_score_bounds(self):
        """Test that all metrics return scores within valid bounds"""
        test_text = "This is a sample text for testing metric bounds"
        
        for test_type in ["linux", "creative", "reasoning"]:
            org_score = self.evaluator._calculate_organization_quality(test_text, test_type)
            tech_score = self.evaluator._calculate_technical_accuracy(test_text, test_type)
            comp_score = self.evaluator._calculate_completeness(test_text, test_type)
            
            # All scores should be between 0 and 100
            for score, metric in [(org_score, "organization_quality"), 
                                (tech_score, "technical_accuracy"), 
                                (comp_score, "completeness")]:
                self.assertGreaterEqual(score, 0, 
                                      f"{metric} for {test_type} should be >= 0, got {score}")
                self.assertLessEqual(score, 100, 
                                   f"{metric} for {test_type} should be <= 100, got {score}")


class TestEndToEndIntegration(TestUniversalEvaluator):
    """Test complete evaluation pipeline"""
    
    def test_linux_evaluation_pipeline(self):
        """Test complete Linux evaluation pipeline"""
        result = evaluate_reasoning(self.linux_sample, "Linux Health Check", test_category="monitoring")
        
        # Test structure
        self.assertIsInstance(result.metrics, EvaluationMetrics)
        self.assertIsInstance(result.reasoning_type, ReasoningType)
        
        # Test score bounds
        self.assertGreaterEqual(result.metrics.overall_score, 0)
        self.assertLessEqual(result.metrics.overall_score, 100)
        
        # Test metrics presence
        self.assertGreater(result.metrics.word_count, 0)
        self.assertGreaterEqual(result.metrics.confidence_score, 0)
        
        # Linux tests should prioritize technical accuracy
        self.assertIsInstance(result.metrics.technical_accuracy, (int, float))
        self.assertIsInstance(result.metrics.organization_quality, (int, float))
    
    def test_creative_evaluation_pipeline(self):
        """Test complete creative evaluation pipeline"""  
        result = evaluate_reasoning(self.creative_sample, "Creative Challenge", test_category="creative_thinking")
        
        # Test structure
        self.assertIsInstance(result.metrics, EvaluationMetrics)
        self.assertGreaterEqual(result.metrics.overall_score, 0)
        self.assertLessEqual(result.metrics.overall_score, 100)
        
        # Creative tests should have reasonable thoroughness scores
        self.assertIsInstance(result.metrics.thoroughness, (int, float))
        self.assertIsInstance(result.metrics.completeness, (int, float))
    
    def test_reasoning_evaluation_pipeline(self):
        """Test complete reasoning evaluation pipeline"""
        result = evaluate_reasoning(self.reasoning_sample, "Logical Analysis", test_category="complex_synthesis")
        
        # Test structure  
        self.assertIsInstance(result.metrics, EvaluationMetrics)
        self.assertGreaterEqual(result.metrics.overall_score, 0)
        self.assertLessEqual(result.metrics.overall_score, 100)
        
        # Reasoning tests should have balanced metrics
        self.assertIsInstance(result.metrics.technical_accuracy, (int, float))
        self.assertIsInstance(result.metrics.domain_appropriateness, (int, float))


class TestEdgeCases(TestUniversalEvaluator):
    """Test edge cases and error handling"""
    
    def test_empty_response(self):
        """Test evaluation of empty response"""
        result = evaluate_reasoning("", "Empty Test")
        
        self.assertEqual(result.metrics.overall_score, 0)
        self.assertEqual(result.metrics.word_count, 0)
        self.assertIn("error", result.detailed_analysis)
    
    def test_very_short_response(self):
        """Test evaluation of very short response"""
        result = evaluate_reasoning("Yes", "Short Test")
        
        self.assertEqual(result.metrics.word_count, 1)
        self.assertGreaterEqual(result.metrics.overall_score, 0)
        self.assertLessEqual(result.metrics.overall_score, 100)
    
    def test_very_long_response(self):
        """Test evaluation of very long response"""
        long_text = "This is a test word. " * 200  # 1000 words
        result = evaluate_reasoning(long_text, "Long Test")
        
        self.assertEqual(result.metrics.word_count, 1000)
        self.assertGreaterEqual(result.metrics.overall_score, 0)
        self.assertLessEqual(result.metrics.overall_score, 100)
    
    def test_none_category_handling(self):
        """Test handling of None category"""
        result = evaluate_reasoning("Test response", "Test", test_category=None)
        
        # Should default to reasoning type
        self.assertIsInstance(result.metrics, EvaluationMetrics)
        self.assertGreaterEqual(result.metrics.overall_score, 0)
    
    def test_unknown_category_handling(self):
        """Test handling of unknown category"""
        result = evaluate_reasoning("Test response", "Test", test_category="unknown_category_12345")
        
        # Should default to reasoning type
        self.assertIsInstance(result.metrics, EvaluationMetrics)
        self.assertGreaterEqual(result.metrics.overall_score, 0)


class TestScoreReasonableness(TestUniversalEvaluator):
    """Test that scores make intuitive sense"""
    
    def test_better_linux_scores_higher(self):
        """Test that better Linux responses score higher"""
        good_result = evaluate_reasoning(self.linux_sample, "Linux Test", test_category="monitoring")
        poor_result = evaluate_reasoning(self.poor_linux_sample, "Linux Test", test_category="monitoring")
        
        self.assertGreater(good_result.metrics.overall_score, poor_result.metrics.overall_score,
                          "Better Linux response should score higher overall")
        self.assertGreater(good_result.metrics.technical_accuracy, poor_result.metrics.technical_accuracy,
                          "Better Linux response should have higher technical accuracy")
    
    def test_better_reasoning_scores_higher(self):
        """Test that better reasoning responses score higher"""
        good_result = evaluate_reasoning(self.reasoning_sample, "Reasoning Test", test_category="complex_synthesis")
        poor_result = evaluate_reasoning(self.poor_reasoning_sample, "Reasoning Test", test_category="complex_synthesis")
        
        self.assertGreater(good_result.metrics.overall_score, poor_result.metrics.overall_score,
                          "Better reasoning response should score higher overall")
        self.assertGreater(good_result.metrics.organization_quality, poor_result.metrics.organization_quality,
                          "Better reasoning response should have better organization")
    
    def test_creative_response_appropriateness(self):
        """Test that creative responses are scored appropriately"""
        creative_result = evaluate_reasoning(self.creative_sample, "Creative Test", test_category="creative_thinking")
        
        # Creative responses should score reasonably well on thoroughness
        self.assertGreater(creative_result.metrics.thoroughness, 30,
                          "Good creative response should score well on thoroughness")
        self.assertGreater(creative_result.metrics.overall_score, 40,
                          "Good creative response should have decent overall score")
    
    def test_score_consistency(self):
        """Test that repeated evaluations are consistent"""
        result1 = evaluate_reasoning(self.linux_sample, "Consistency Test", test_category="monitoring")
        result2 = evaluate_reasoning(self.linux_sample, "Consistency Test", test_category="monitoring")
        
        # Scores should be identical for identical input
        self.assertEqual(result1.metrics.overall_score, result2.metrics.overall_score)
        self.assertEqual(result1.metrics.technical_accuracy, result2.metrics.technical_accuracy)
    
    def test_different_categories_different_scores(self):
        """Test that same text evaluated as different categories produces different scores"""
        text = "systemctl restart nginx && echo 'Service restarted'"
        
        linux_result = evaluate_reasoning(text, "Test", test_category="monitoring")
        creative_result = evaluate_reasoning(text, "Test", test_category="creative_thinking")
        
        # Scores should differ because evaluation logic is different
        # Linux should score better on technical accuracy for this text
        self.assertGreater(linux_result.metrics.technical_accuracy, 
                          creative_result.metrics.technical_accuracy,
                          "Same text should score differently for different test types")


if __name__ == "__main__":
    # Configure test runner for verbose output
    unittest.main(verbosity=2, buffer=True)