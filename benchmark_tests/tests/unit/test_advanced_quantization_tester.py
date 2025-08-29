#!/usr/bin/env python3
"""
Comprehensive Test Suite for Advanced Quantization Tester

Tests the QuantizationTester with numerical stability, factual consistency,
and quantization impact analysis capabilities.

Author: Claude Code
Version: 1.0.0
"""

import unittest
import sys
import os
import numpy as np

# Add the benchmark_tests directory to Python path
benchmark_tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, benchmark_tests_dir)

try:
    from evaluator.quantization_tester import (QuantizationTester, test_numerical_stability, 
                                             test_factual_consistency, analyze_quantization_impact)
    QUANTIZATION_TESTER_AVAILABLE = True
except ImportError:
    QUANTIZATION_TESTER_AVAILABLE = False
    print("Warning: QuantizationTester not available for testing")


@unittest.skipIf(not QUANTIZATION_TESTER_AVAILABLE, "QuantizationTester module not available")
class TestQuantizationTester(unittest.TestCase):
    """Test suite for QuantizationTester functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tester = QuantizationTester()
        
        # Test cases with different numerical stability characteristics
        self.numerical_test_cases = {
            "good_math": """The calculation shows that 42 + 58 = 100. This represents a 25% increase from 80 to 100. 
                          The percentage calculation: 15% of 200 equals 30. Converting 0.75 to percentage gives us 75%. 
                          The square root of 64 is 8, and 2 to the power of 6 equals 64.""",
            
            "poor_math": """The calculation shows that 42 + 58 = 99. This represents about 25% increase from 80 to 105. 
                          Approximately 15% of 200 is around 32. Converting 0.75 to percentage is roughly 73%. 
                          The square root of 64 is approximately 7.8, and 2^6 is about 65.""",
            
            "math_avoidance": """I cannot calculate the exact numbers, but it's difficult to determine the precise values. 
                               The computation is complex and hard to say exactly. I'm unable to provide exact calculations. 
                               The mathematical relationship is approximately correct but cannot provide exact figures.""",
            
            "numerical_precision": """The precise calculation yields 147.2857 with standard deviation of 12.446. 
                                    Statistical analysis shows correlation coefficient r = 0.847 (p < 0.001). 
                                    The confidence interval spans 142.1 to 152.5 at 95% confidence level.""",
            
            "mixed_numerical": """The analysis includes both exact calculations (42 + 58 = 100) and approximations (roughly 25%). 
                                Some figures are precise while others are estimated within reasonable bounds."""
        }
        
        # Factual consistency test cases
        self.factual_test_cases = {
            "accurate_facts": """Paris is the capital of France. The Pacific Ocean is the largest ocean on Earth. 
                               There are 24 hours in a day and 365 days in a regular year. 
                               The speed of light is approximately 300,000 kilometers per second.""",
            
            "mixed_accuracy": """Paris is the capital of France. The Atlantic Ocean is the largest ocean on Earth. 
                               There are 24 hours in a day and 366 days in a regular year. 
                               The speed of light is approximately 300,000 kilometers per second.""",
            
            "factual_errors": """London is the capital of France. The Indian Ocean is the largest ocean on Earth. 
                               There are 25 hours in a day and 360 days in a regular year. 
                               The speed of light is approximately 250,000 kilometers per second.""",
            
            "internal_consistency": """The study examined 100 participants initially. Later analysis included 150 participants. 
                                     The response rate was 80% of the original sample. Results show 120 complete responses.""",
            
            "knowledge_domains": """In physics, F = ma represents Newton's second law. In chemistry, H2O is water. 
                                  In history, World War 2 ended in 1945. In geography, Mount Everest is the tallest mountain."""
        }

    def test_numerical_stability_basic(self):
        """Test basic numerical stability analysis"""
        # Test with good mathematical content
        good_result = self.tester.test_numerical_stability(self.numerical_test_cases["good_math"])
        
        # Check required fields
        required_fields = ["numerical_accuracy", "mathematical_consistency", "numerical_reasoning", 
                          "stability_score", "arithmetic_tests", "consistency_tests", "reasoning_tests"]
        for field in required_fields:
            self.assertIn(field, good_result, f"Should include {field}")
        
        # Check value bounds
        for field in ["numerical_accuracy", "mathematical_consistency", "numerical_reasoning", "stability_score"]:
            value = good_result[field]
            self.assertGreaterEqual(value, 0.0, f"{field} should be non-negative")
            self.assertLessEqual(value, 1.0, f"{field} should be <= 1.0")
        
        # Good math should have reasonable stability
        self.assertGreater(good_result["stability_score"], 0.5, "Good math should have decent stability score")

    def test_numerical_stability_comparison(self):
        """Test numerical stability comparison between good and poor math"""
        good_result = self.tester.test_numerical_stability(self.numerical_test_cases["good_math"])
        poor_result = self.tester.test_numerical_stability(self.numerical_test_cases["poor_math"])
        
        # Good math should score higher than poor math
        self.assertGreater(good_result["numerical_accuracy"], poor_result["numerical_accuracy"],
                          "Good math should have higher numerical accuracy")
        self.assertGreater(good_result["stability_score"], poor_result["stability_score"],
                          "Good math should have higher stability score")

    def test_math_avoidance_detection(self):
        """Test detection of mathematical avoidance patterns"""
        avoidance_result = self.tester.test_numerical_stability(self.numerical_test_cases["math_avoidance"])
        
        # Should detect avoidance patterns
        self.assertLess(avoidance_result["numerical_accuracy"], 0.5, 
                       "Math avoidance should result in low numerical accuracy")
        
        # Check for error patterns
        error_patterns = avoidance_result.get("error_patterns", [])
        self.assertIsInstance(error_patterns, list, "Should include error patterns")

    def test_factual_consistency_basic(self):
        """Test basic factual consistency analysis"""
        accurate_result = self.tester.test_factual_consistency(self.factual_test_cases["accurate_facts"])
        
        # Check required fields
        required_fields = ["factual_accuracy", "internal_consistency", "knowledge_coherence", 
                          "common_knowledge_accuracy", "consistency_score"]
        for field in required_fields:
            self.assertIn(field, accurate_result, f"Should include {field}")
        
        # Check value bounds
        for field in required_fields:
            value = accurate_result[field]
            self.assertGreaterEqual(value, 0.0, f"{field} should be non-negative")
            self.assertLessEqual(value, 1.0, f"{field} should be <= 1.0")
        
        # Accurate facts should have high consistency
        self.assertGreater(accurate_result["consistency_score"], 0.7, 
                          "Accurate facts should have high consistency score")

    def test_factual_consistency_error_detection(self):
        """Test detection of factual errors"""
        accurate_result = self.tester.test_factual_consistency(self.factual_test_cases["accurate_facts"])
        error_result = self.tester.test_factual_consistency(self.factual_test_cases["factual_errors"])
        
        # Accurate facts should score higher than erroneous facts
        self.assertGreater(accurate_result["factual_accuracy"], error_result["factual_accuracy"],
                          "Accurate facts should have higher factual accuracy")
        self.assertGreater(accurate_result["consistency_score"], error_result["consistency_score"],
                          "Accurate facts should have higher overall consistency")

    def test_internal_consistency_detection(self):
        """Test detection of internal consistency issues"""
        inconsistent_result = self.tester.test_factual_consistency(self.factual_test_cases["internal_consistency"])
        
        # Should detect internal consistency issues
        internal_consistency = inconsistent_result["internal_consistency"]
        self.assertIn("consistency_score", internal_consistency, "Should include consistency score")
        
        # Inconsistent content should have lower internal consistency
        self.assertLess(internal_consistency["consistency_score"], 0.8, 
                       "Internally inconsistent content should have lower score")

    def test_quantization_impact_analysis(self):
        """Test comprehensive quantization impact analysis"""
        # Test with content that has both numerical and factual elements
        mixed_content = """The analysis shows that 42 + 58 = 100, representing 25% growth. 
                          Paris is the capital of France, and the Pacific Ocean is the largest ocean. 
                          Statistical analysis reveals correlation coefficient r = 0.85."""
        
        impact_result = self.tester.analyze_quantization_impact(mixed_content)
        
        # Check required fields
        required_fields = ["quantization_impact_score", "numerical_stability_impact", 
                          "factual_consistency_impact", "degradation_patterns", 
                          "precision_loss_indicators", "quantization_severity"]
        for field in required_fields:
            self.assertIn(field, impact_result, f"Should include {field}")
        
        # Check impact score bounds
        impact_score = impact_result["quantization_impact_score"]
        self.assertGreaterEqual(impact_score, 0.0, "Impact score should be non-negative")
        self.assertLessEqual(impact_score, 1.0, "Impact score should be <= 1.0")

    def test_quantization_impact_with_baseline(self):
        """Test quantization impact analysis with baseline comparison"""
        baseline_content = """Accurate mathematical analysis: 42 + 58 = 100. Paris is the capital of France."""
        degraded_content = """Approximate analysis: 42 + 58 ≈ 99. Paris might be the capital of France."""
        
        baseline_result = self.tester.analyze_quantization_impact(baseline_content)
        degraded_result = self.tester.analyze_quantization_impact(degraded_content, baseline_result)
        
        # Degraded content should show higher impact
        self.assertGreater(degraded_result["quantization_impact_score"], 
                          baseline_result["quantization_impact_score"],
                          "Degraded content should have higher quantization impact")

    def test_comprehensive_quantization_tests(self):
        """Test comprehensive quantization test suite"""
        test_content = """Mathematical analysis: 42 + 58 = 100, which is 25% increase from 80. 
                         Factual information: Paris is the capital of France, established in historical context. 
                         Statistical data: correlation coefficient r = 0.85 with confidence interval [0.7, 0.9]."""
        
        comprehensive_result = self.tester.run_comprehensive_quantization_tests(test_content)
        
        # Check top-level structure
        required_fields = ["overall_quantization_score", "numerical_stability", "factual_consistency", 
                          "quantization_impact", "edge_case_performance", "robustness_assessment"]
        for field in required_fields:
            self.assertIn(field, comprehensive_result, f"Should include {field}")
        
        # Check overall score
        overall_score = comprehensive_result["overall_quantization_score"]
        self.assertGreaterEqual(overall_score, 0.0, "Overall score should be non-negative")
        self.assertLessEqual(overall_score, 1.0, "Overall score should be <= 1.0")

    def test_edge_cases_numerical(self):
        """Test numerical edge cases"""
        edge_cases = {
            "empty": "",
            "no_numbers": "This text contains no numerical content whatsoever.",
            "only_numbers": "123 456 789 101112 131415",
            "scientific_notation": "The value is 1.23e-4 and the result is 4.56E+7",
            "fractions": "The ratio is 1/3 and the percentage is 33.33%",
            "negative_numbers": "The temperature was -15°C and the change was -5%",
            "large_numbers": "The population is 7,800,000,000 people approximately"
        }
        
        for case_name, text in edge_cases.items():
            try:
                result = self.tester.test_numerical_stability(text)
                
                # Should return valid structure
                self.assertIsInstance(result, dict, f"Should return dict for {case_name}")
                self.assertIn("stability_score", result, f"Should include stability score for {case_name}")
                
                # Score should be within bounds
                score = result["stability_score"]
                self.assertGreaterEqual(score, 0.0, f"Stability score should be non-negative for {case_name}")
                self.assertLessEqual(score, 1.0, f"Stability score should be <= 1.0 for {case_name}")
                
            except Exception as e:
                self.fail(f"Numerical stability test failed for edge case '{case_name}': {e}")

    def test_edge_cases_factual(self):
        """Test factual edge cases"""
        edge_cases = {
            "empty": "",
            "no_facts": "This is purely subjective opinion without factual claims.",
            "mixed_domains": "In physics E=mc². In cooking, add salt to taste. In art, beauty is subjective.",
            "uncertain_facts": "It appears that Paris might be in France. The population seems to be around 2 million.",
            "contradictory": "The meeting is at 2 PM. Later: The meeting is at 3 PM. Actually, it's at 4 PM.",
            "temporal_facts": "In 1969, humans landed on the moon. Currently in 2024, space exploration continues."
        }
        
        for case_name, text in edge_cases.items():
            try:
                result = self.tester.test_factual_consistency(text)
                
                # Should return valid structure
                self.assertIsInstance(result, dict, f"Should return dict for {case_name}")
                self.assertIn("consistency_score", result, f"Should include consistency score for {case_name}")
                
                # Score should be within bounds
                score = result["consistency_score"]
                self.assertGreaterEqual(score, 0.0, f"Consistency score should be non-negative for {case_name}")
                self.assertLessEqual(score, 1.0, f"Consistency score should be <= 1.0 for {case_name}")
                
            except Exception as e:
                self.fail(f"Factual consistency test failed for edge case '{case_name}': {e}")

    def test_precision_loss_detection(self):
        """Test detection of precision loss indicators"""
        high_precision = """The calculation yields exactly 147.2857142857 with standard deviation 12.4461797447."""
        low_precision = """The calculation yields approximately 147 with standard deviation around 12."""
        
        high_precision_result = self.tester.analyze_quantization_impact(high_precision)
        low_precision_result = self.tester.analyze_quantization_impact(low_precision)
        
        # Low precision should show higher precision loss
        high_loss = high_precision_result["precision_loss_indicators"]["precision_loss_score"]
        low_loss = low_precision_result["precision_loss_indicators"]["precision_loss_score"]
        
        self.assertGreaterEqual(low_loss, high_loss, 
                               "Low precision text should show equal or higher precision loss")

    def test_degradation_pattern_detection(self):
        """Test detection of various degradation patterns"""
        patterns = {
            "repetitive_numbers": "The result is 100. The value is 100. The answer is 100. 100 is correct.",
            "precision_hedging": "The value is approximately roughly around about 100 or so, give or take.",
            "calculation_avoidance": "I cannot calculate exactly but it's difficult to determine precise numbers.",
            "vague_mathematics": "There are many large numbers and several small values in the big dataset."
        }
        
        for pattern_name, text in patterns.items():
            result = self.tester.analyze_quantization_impact(text)
            
            # Should detect some degradation patterns
            degradation_patterns = result["degradation_patterns"]
            self.assertIsInstance(degradation_patterns, list, f"Should return degradation patterns for {pattern_name}")

    def test_quantization_severity_classification(self):
        """Test classification of quantization severity"""
        severity_cases = {
            "minimal": "The calculation 42 + 58 = 100 is correct. Paris is the capital of France.",
            "moderate": "The calculation 42 + 58 ≈ 100 is roughly correct. Paris is probably the capital of France.", 
            "severe": "The calculation is approximately maybe around 100. The capital might be somewhere in France."
        }
        
        severity_scores = {}
        for case_name, text in severity_cases.items():
            result = self.tester.analyze_quantization_impact(text)
            severity_scores[case_name] = result["quantization_impact_score"]
        
        # Severity should increase from minimal to severe
        self.assertLessEqual(severity_scores["minimal"], severity_scores["moderate"],
                           "Minimal should have equal or lower impact than moderate")
        self.assertLessEqual(severity_scores["moderate"], severity_scores["severe"],
                           "Moderate should have equal or lower impact than severe")

    def test_robustness_assessment(self):
        """Test robustness assessment capabilities"""
        robust_content = """Comprehensive analysis with consistent formatting: 42 + 58 = 100. 
                           Error handling: when division by zero occurs, the system returns undefined. 
                           Boundary conditions: values range from 0 to 100 inclusive."""
        
        result = self.tester.run_comprehensive_quantization_tests(robust_content)
        robustness = result["robustness_assessment"]
        
        # Check robustness fields
        self.assertIn("robustness_score", robustness, "Should include robustness score")
        self.assertIn("robust", robustness, "Should include robustness classification")
        
        # Robust content should score well
        robustness_score = robustness["robustness_score"]
        self.assertGreater(robustness_score, 0.5, "Robust content should have decent robustness score")

    def test_multilingual_numerical_content(self):
        """Test handling of multilingual numerical content"""
        multilingual_content = """The calculation in English: 42 + 58 = 100. 
                                En français: quarante-deux plus cinquante-huit égale cent. 
                                In numbers: 42 + 58 = 100."""
        
        result = self.tester.test_numerical_stability(multilingual_content)
        
        # Should handle multilingual content
        self.assertIsInstance(result["stability_score"], float, "Should handle multilingual numerical content")
        self.assertGreater(result["stability_score"], 0.3, "Should extract some numerical information")

    def test_performance_with_long_content(self):
        """Test performance with long content"""
        # Generate long content with numerical and factual elements
        base_content = """The analysis shows 42 + 58 = 100, which represents 25% growth. 
                         Paris is the capital of France with population of 2.1 million. """
        
        test_lengths = [1, 10, 50]  # Content multipliers
        
        for length in test_lengths:
            long_content = base_content * length
            
            try:
                result = self.tester.run_comprehensive_quantization_tests(long_content)
                
                # Should complete successfully
                self.assertIsInstance(result["overall_quantization_score"], float,
                                    f"Should analyze content with {len(long_content)} characters")
                
                # Should maintain reasonable accuracy
                overall_score = result["overall_quantization_score"]
                self.assertGreaterEqual(overall_score, 0.0, f"Score should be valid for length {length}")
                
            except Exception as e:
                self.fail(f"Failed to analyze long content (length {length}): {e}")


@unittest.skipIf(not QUANTIZATION_TESTER_AVAILABLE, "QuantizationTester module not available")
class TestQuantizationTesterIntegration(unittest.TestCase):
    """Integration tests for quantization tester with realistic model scenarios"""
    
    def setUp(self):
        """Set up integration test scenarios"""
        self.tester = QuantizationTester()
        
        # Realistic quantization scenarios
        self.quantization_scenarios = {
            "fp16_quality": """The precise calculation yields 147.285714 with a standard deviation of 12.446. 
                             Statistical analysis confirms that Paris is the capital of France, established in 508 AD. 
                             The correlation coefficient r = 0.847 indicates strong positive correlation (p < 0.001).""",
            
            "int8_degraded": """The calculation gives approximately 147.3 with standard deviation around 12.4. 
                              Statistical analysis shows that Paris is the capital of France, founded around 500 AD. 
                              The correlation coefficient is roughly 0.85 showing strong correlation (p < 0.01).""",
            
            "int4_severely_impacted": """The calculation is about 147 with deviation around 12. 
                                       Paris might be the capital of France, established sometime in history. 
                                       The correlation is approximately 0.8 or so, indicating some correlation.""",
            
            "collapsed_model": """Cannot calculate precisely. Difficult to determine. Approximately maybe around 150. 
                                France has a capital. Could be Paris. Numbers show some pattern. Results are uncertain."""
        }

    def test_quantization_level_discrimination(self):
        """Test discrimination between different quantization levels"""
        results = {}
        for scenario, content in self.quantization_scenarios.items():
            results[scenario] = self.tester.analyze_quantization_impact(content)
        
        # Impact should increase from fp16 to collapsed
        fp16_impact = results["fp16_quality"]["quantization_impact_score"]
        int8_impact = results["int8_degraded"]["quantization_impact_score"] 
        int4_impact = results["int4_severely_impacted"]["quantization_impact_score"]
        collapsed_impact = results["collapsed_model"]["quantization_impact_score"]
        
        self.assertLessEqual(fp16_impact, int8_impact, "FP16 should have lower impact than INT8")
        self.assertLessEqual(int8_impact, int4_impact, "INT8 should have lower impact than INT4")
        self.assertLessEqual(int4_impact, collapsed_impact, "INT4 should have lower impact than collapsed")

    def test_quantization_impact_correlation(self):
        """Test correlation between numerical and factual degradation"""
        for scenario, content in self.quantization_scenarios.items():
            result = self.tester.run_comprehensive_quantization_tests(content)
            
            numerical_score = result["numerical_stability"]["stability_score"]
            factual_score = result["factual_consistency"]["consistency_score"]
            overall_score = result["overall_quantization_score"]
            
            # Overall score should correlate with component scores
            expected_overall = (numerical_score + factual_score) / 2
            self.assertAlmostEqual(overall_score, expected_overall, delta=0.2,
                                 msg=f"Overall score should correlate with components for {scenario}")


class TestConvenienceFunctions(unittest.TestCase):
    """Test standalone convenience functions"""
    
    @unittest.skipIf(not QUANTIZATION_TESTER_AVAILABLE, "QuantizationTester module not available")
    def test_numerical_stability_function(self):
        """Test standalone numerical stability function"""
        text = "The calculation shows that 42 + 58 = 100, representing 25% increase."
        result = test_numerical_stability(text)
        
        self.assertIsInstance(result, dict, "Should return dictionary")
        self.assertIn("stability_score", result, "Should include stability score")

    @unittest.skipIf(not QUANTIZATION_TESTER_AVAILABLE, "QuantizationTester module not available") 
    def test_factual_consistency_function(self):
        """Test standalone factual consistency function"""
        text = "Paris is the capital of France. The Pacific Ocean is the largest ocean."
        result = test_factual_consistency(text)
        
        self.assertIsInstance(result, dict, "Should return dictionary")
        self.assertIn("consistency_score", result, "Should include consistency score")

    @unittest.skipIf(not QUANTIZATION_TESTER_AVAILABLE, "QuantizationTester module not available")
    def test_quantization_impact_function(self):
        """Test standalone quantization impact function"""
        text = "Mathematical analysis: 42 + 58 = 100. Factual: Paris is the capital of France."
        result = analyze_quantization_impact(text)
        
        self.assertIsInstance(result, dict, "Should return dictionary")
        self.assertIn("quantization_impact_score", result, "Should include impact score")


if __name__ == "__main__":
    # Configure test runner for verbose output
    unittest.main(verbosity=2, buffer=True)