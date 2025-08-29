#!/usr/bin/env python3
"""
Comprehensive Test Suite for Advanced Entropy Calculator

Tests the EntropyCalculator system with rigorous validation of Shannon entropy,
semantic entropy, and entropy pattern analysis capabilities.

Author: Claude Code
Version: 1.0.0
"""

import unittest
import sys
import os
import numpy as np
from collections import Counter

# Add the benchmark_tests directory to Python path
benchmark_tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, benchmark_tests_dir)

try:
    from evaluator.entropy_calculator import EntropyCalculator, calculate_shannon_entropy, analyze_text_entropy
    ENTROPY_CALCULATOR_AVAILABLE = True
except ImportError:
    ENTROPY_CALCULATOR_AVAILABLE = False
    print("Warning: EntropyCalculator not available for testing")


@unittest.skipIf(not ENTROPY_CALCULATOR_AVAILABLE, "EntropyCalculator module not available")
class TestEntropyCalculator(unittest.TestCase):
    """Test suite for EntropyCalculator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = EntropyCalculator()
        
        # Test cases with known entropy characteristics
        self.test_cases = {
            "empty": "",
            "single_word": "Hello",
            "repetitive_low_entropy": "the the the the the the",
            "diverse_high_entropy": "The quick brown fox jumps over the lazy dog with remarkable agility and grace.",
            "technical_content": "Algorithm optimization requires careful analysis of computational complexity, memory allocation, and performance bottlenecks in distributed systems.",
            "creative_content": "In the ethereal twilight, shadows danced across cobblestone streets while mysterious figures whispered secrets beneath ancient oak trees.",
            "mathematical_content": "The calculation shows that 42 + 58 = 100, which represents approximately 25% increase from the baseline of 80 units.",
            "repetitive_pattern": "I am analyzing this problem. I am analyzing this problem. I am analyzing this problem repeatedly.",
            "degrading_quality": "This is a comprehensive analysis of the problem at hand. The solution requires careful consideration. But then it starts repeating. Repeating again. Repeating. Repeat.",
            "mixed_languages": "Hello world, this is English. Hola mundo, esto es español. 世界你好, 这是中文。"
        }

    def test_shannon_entropy_basic_cases(self):
        """Test Shannon entropy calculation for basic cases"""
        # Empty text should return 0
        entropy_empty = self.calculator.calculate_shannon_entropy("")
        self.assertEqual(entropy_empty, 0.0, "Empty text should have 0 entropy")
        
        # Single word should have 0 entropy (all tokens identical)
        entropy_single = self.calculator.calculate_shannon_entropy("hello")
        self.assertGreaterEqual(entropy_single, 0.0, "Single word should have non-negative entropy")
        
        # Repetitive text should have low entropy
        entropy_repetitive = self.calculator.calculate_shannon_entropy(self.test_cases["repetitive_low_entropy"])
        self.assertLess(entropy_repetitive, 2.0, "Repetitive text should have low entropy")
        
        # Diverse text should have higher entropy
        entropy_diverse = self.calculator.calculate_shannon_entropy(self.test_cases["diverse_high_entropy"])
        self.assertGreater(entropy_diverse, entropy_repetitive, "Diverse text should have higher entropy than repetitive text")

    def test_shannon_entropy_mathematical_properties(self):
        """Test mathematical properties of Shannon entropy"""
        # Test monotonicity: adding unique words should increase or maintain entropy
        text1 = "the quick brown fox"
        text2 = "the quick brown fox jumps over lazy dog"
        
        entropy1 = self.calculator.calculate_shannon_entropy(text1)
        entropy2 = self.calculator.calculate_shannon_entropy(text2)
        
        self.assertGreaterEqual(entropy2, entropy1 - 0.1, "Adding diverse words should not significantly decrease entropy")
        
        # Test bounds: entropy should be between 0 and log2(vocabulary_size)
        for text_key, text in self.test_cases.items():
            if text:  # Skip empty text
                entropy = self.calculator.calculate_shannon_entropy(text)
                self.assertGreaterEqual(entropy, 0.0, f"Entropy should be non-negative for {text_key}")
                
                # Calculate theoretical maximum
                words = text.lower().split()
                unique_words = len(set(words))
                if unique_words > 1:
                    max_entropy = np.log2(unique_words)
                    self.assertLessEqual(entropy, max_entropy + 0.1, f"Entropy should not exceed theoretical maximum for {text_key}")

    def test_semantic_entropy_analysis(self):
        """Test semantic entropy calculation"""
        # Test with different content types
        technical_semantic = self.calculator.calculate_semantic_entropy(self.test_cases["technical_content"])
        creative_semantic = self.calculator.calculate_semantic_entropy(self.test_cases["creative_content"])
        
        # Both should return valid semantic metrics
        for semantic_result in [technical_semantic, creative_semantic]:
            self.assertIn("semantic_entropy", semantic_result, "Should return semantic entropy")
            self.assertIn("semantic_diversity", semantic_result, "Should return semantic diversity")
            self.assertIn("embedding_variance", semantic_result, "Should return embedding variance")
            
            # Values should be within reasonable bounds
            self.assertGreaterEqual(semantic_result["semantic_entropy"], 0.0, "Semantic entropy should be non-negative")
            self.assertGreaterEqual(semantic_result["semantic_diversity"], 0.0, "Semantic diversity should be non-negative")
            self.assertLessEqual(semantic_result["semantic_diversity"], 1.0, "Semantic diversity should be <= 1.0")

    def test_ngram_entropy_patterns(self):
        """Test n-gram entropy analysis"""
        # Test bigram entropy
        bigram_entropy = self.calculator.calculate_ngram_entropy(self.test_cases["diverse_high_entropy"], n=2)
        self.assertGreater(bigram_entropy, 0, "Bigram entropy should be positive for diverse text")
        
        # Test trigram entropy
        trigram_entropy = self.calculator.calculate_ngram_entropy(self.test_cases["diverse_high_entropy"], n=3)
        self.assertGreater(trigram_entropy, 0, "Trigram entropy should be positive for diverse text")
        
        # Repetitive text should have lower n-gram entropy
        repetitive_bigram = self.calculator.calculate_ngram_entropy(self.test_cases["repetitive_low_entropy"], n=2)
        self.assertLess(repetitive_bigram, bigram_entropy, "Repetitive text should have lower n-gram entropy")

    def test_vocabulary_entropy_metrics(self):
        """Test vocabulary-based entropy metrics"""
        vocab_metrics = self.calculator.calculate_vocabulary_entropy(self.test_cases["diverse_high_entropy"])
        
        # Check required fields
        required_fields = ["vocab_entropy", "vocab_diversity", "unique_ratio", "unique_words", "total_words"]
        for field in required_fields:
            self.assertIn(field, vocab_metrics, f"Should include {field} in vocabulary metrics")
        
        # Check value ranges
        self.assertGreaterEqual(vocab_metrics["vocab_diversity"], 0.0, "Vocabulary diversity should be non-negative")
        self.assertLessEqual(vocab_metrics["vocab_diversity"], 1.0, "Vocabulary diversity should be <= 1.0")
        self.assertGreaterEqual(vocab_metrics["unique_ratio"], 0.0, "Unique ratio should be non-negative")
        self.assertLessEqual(vocab_metrics["unique_ratio"], 1.0, "Unique ratio should be <= 1.0")

    def test_comprehensive_entropy_profile(self):
        """Test comprehensive entropy analysis"""
        for text_key, text in self.test_cases.items():
            if not text:  # Skip empty text
                continue
                
            profile = self.calculator.analyze_entropy_profile(text)
            
            # Check all expected fields are present
            expected_fields = [
                "token_entropy", "word_entropy", "entropy_quality_ratio",
                "semantic_entropy", "semantic_diversity", "embedding_variance",
                "bigram_entropy", "trigram_entropy", "vocab_entropy",
                "vocab_diversity", "unique_ratio", "entropy_patterns"
            ]
            
            for field in expected_fields:
                self.assertIn(field, profile, f"Profile should include {field} for {text_key}")
            
            # Validate numeric bounds
            numeric_fields = [
                "token_entropy", "word_entropy", "entropy_quality_ratio",
                "semantic_entropy", "semantic_diversity", "bigram_entropy",
                "trigram_entropy", "vocab_entropy", "vocab_diversity", "unique_ratio"
            ]
            
            for field in numeric_fields:
                self.assertGreaterEqual(profile[field], 0.0, f"{field} should be non-negative for {text_key}")

    def test_entropy_pattern_detection(self):
        """Test entropy pattern analysis"""
        # Test repetitive pattern detection
        repetitive_profile = self.calculator.analyze_entropy_profile(self.test_cases["repetitive_pattern"])
        patterns = repetitive_profile["entropy_patterns"]
        
        self.assertIn("has_repetitive_patterns", patterns, "Should detect repetitive patterns")
        self.assertTrue(patterns.get("has_repetitive_patterns", False), "Should flag repetitive patterns as True")
        
        # Test degrading quality pattern
        degrading_profile = self.calculator.analyze_entropy_profile(self.test_cases["degrading_quality"])
        degrading_patterns = degrading_profile["entropy_patterns"]
        
        self.assertIn("entropy_trend", degrading_patterns, "Should analyze entropy trend")
        # Could be "decreasing" or "stable" depending on the exact text
        self.assertIn(degrading_patterns.get("entropy_trend"), ["decreasing", "stable", "increasing"], "Should have valid trend classification")

    def test_entropy_comparison(self):
        """Test entropy comparison between different texts"""
        high_entropy_text = self.test_cases["diverse_high_entropy"]
        low_entropy_text = self.test_cases["repetitive_low_entropy"]
        
        comparison = self.calculator.compare_entropy_profiles(high_entropy_text, low_entropy_text)
        
        # Check comparison fields
        expected_fields = [
            "token_entropy_diff", "semantic_entropy_diff", "vocab_entropy_diff",
            "semantic_diversity_diff", "entropy_similarity"
        ]
        
        for field in expected_fields:
            self.assertIn(field, comparison, f"Comparison should include {field}")
        
        # High entropy text should have higher token entropy
        self.assertGreater(comparison["token_entropy_diff"], 0, "High entropy text should have higher token entropy")
        
        # Entropy similarity should be between 0 and 1
        self.assertGreaterEqual(comparison["entropy_similarity"], 0.0, "Entropy similarity should be non-negative")
        self.assertLessEqual(comparison["entropy_similarity"], 1.0, "Entropy similarity should be <= 1.0")

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling"""
        # Test with very short texts
        short_profiles = [
            self.calculator.analyze_entropy_profile("A"),
            self.calculator.analyze_entropy_profile("A B"),
            self.calculator.analyze_entropy_profile("A B C")
        ]
        
        for profile in short_profiles:
            # Should not crash and should return valid structure
            self.assertIsInstance(profile, dict, "Should return dictionary for short texts")
            self.assertIn("token_entropy", profile, "Should include token_entropy even for short texts")
        
        # Test with special characters
        special_char_text = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        special_profile = self.calculator.analyze_entropy_profile(special_char_text)
        self.assertIsInstance(special_profile, dict, "Should handle special characters")
        
        # Test with numbers and mixed content
        mixed_content = "123 ABC 456 def 789 GHI !@# xyz"
        mixed_profile = self.calculator.analyze_entropy_profile(mixed_content)
        self.assertGreater(mixed_profile["token_entropy"], 0, "Mixed content should have positive entropy")

    def test_mathematical_content_analysis(self):
        """Test entropy analysis of mathematical content"""
        math_profile = self.calculator.analyze_entropy_profile(self.test_cases["mathematical_content"])
        
        # Mathematical content should have reasonable entropy
        self.assertGreater(math_profile["token_entropy"], 2.0, "Mathematical content should have reasonable token entropy")
        
        # Should detect numbers and mathematical expressions
        self.assertGreater(math_profile["unique_ratio"], 0.5, "Mathematical content should have good vocabulary diversity")

    def test_multilingual_content_handling(self):
        """Test handling of multilingual content"""
        if "mixed_languages" in self.test_cases:
            multilingual_profile = self.calculator.analyze_entropy_profile(self.test_cases["mixed_languages"])
            
            # Multilingual content should have high entropy due to diverse character sets
            self.assertGreater(multilingual_profile["token_entropy"], 3.0, "Multilingual content should have high entropy")
            
            # Should not crash with different scripts
            self.assertIsInstance(multilingual_profile["entropy_patterns"], dict, "Should handle multilingual patterns")

    def test_performance_and_scalability(self):
        """Test performance with different text lengths"""
        # Generate texts of different lengths
        base_text = "The quick brown fox jumps over the lazy dog. "
        test_lengths = [1, 10, 100, 500]  # Multiples of base text
        
        for length in test_lengths:
            long_text = base_text * length
            
            # Should complete within reasonable time and not crash
            try:
                profile = self.calculator.analyze_entropy_profile(long_text)
                self.assertIsInstance(profile, dict, f"Should handle text of length {len(long_text)}")
                self.assertGreater(profile["word_count"], length * 8, f"Should count words correctly for length {length}")
            except Exception as e:
                self.fail(f"Failed to process text of length {len(long_text)}: {e}")

    def test_entropy_quality_correlation(self):
        """Test correlation between entropy and perceived quality"""
        # High-quality text should have balanced entropy
        high_quality = self.test_cases["technical_content"]
        low_quality = self.test_cases["repetitive_pattern"]
        
        high_profile = self.calculator.analyze_entropy_profile(high_quality)
        low_profile = self.calculator.analyze_entropy_profile(low_quality)
        
        # High-quality text should have better entropy-quality ratio
        self.assertGreater(high_profile["entropy_quality_ratio"], low_profile["entropy_quality_ratio"],
                          "High-quality text should have better entropy-quality ratio")
        
        # High-quality text should have fewer repetitive patterns
        high_patterns = high_profile["entropy_patterns"]
        low_patterns = low_profile["entropy_patterns"]
        
        self.assertLess(high_patterns.get("local_entropy_drops", 0), 
                       low_patterns.get("local_entropy_drops", 0),
                       "High-quality text should have fewer local entropy drops")


class TestConvenienceFunctions(unittest.TestCase):
    """Test standalone convenience functions"""
    
    @unittest.skipIf(not ENTROPY_CALCULATOR_AVAILABLE, "EntropyCalculator module not available")
    def test_calculate_shannon_entropy_function(self):
        """Test standalone Shannon entropy function"""
        text = "The quick brown fox jumps over the lazy dog"
        entropy = calculate_shannon_entropy(text)
        
        self.assertGreater(entropy, 0, "Standalone function should return positive entropy")
        self.assertIsInstance(entropy, float, "Should return float value")

    @unittest.skipIf(not ENTROPY_CALCULATOR_AVAILABLE, "EntropyCalculator module not available")
    def test_analyze_text_entropy_function(self):
        """Test standalone text entropy analysis function"""
        text = "The quick brown fox jumps over the lazy dog"
        analysis = analyze_text_entropy(text)
        
        self.assertIsInstance(analysis, dict, "Should return dictionary")
        self.assertIn("token_entropy", analysis, "Should include token entropy")
        self.assertIn("semantic_diversity", analysis, "Should include semantic diversity")


@unittest.skipIf(not ENTROPY_CALCULATOR_AVAILABLE, "EntropyCalculator module not available")
class TestEntropyCalculatorIntegration(unittest.TestCase):
    """Integration tests for entropy calculator with different scenarios"""
    
    def setUp(self):
        """Set up integration test scenarios"""
        self.calculator = EntropyCalculator()
        
        # Realistic model output scenarios
        self.model_outputs = {
            "gpt_oss_20b_good": """The economic implications of this policy are multifaceted. 
                                 First, we must consider the immediate impact on market liquidity. 
                                 Second, the long-term effects on consumer behavior require analysis. 
                                 Finally, regulatory compliance costs will affect implementation.""",
            
            "gpt_oss_20b_degraded": """The system works well in most cases. The system provides good results. 
                                     The system handles various inputs. The system processes data efficiently. 
                                     The system maintains performance. The system delivers output consistently.""",
            
            "gpt_oss_20b_collapsed": """Error error error. Cannot process. Cannot process. Cannot process. 
                                      System failure. System failure. System failure. Retry needed. 
                                      Retry needed. Retry needed repeatedly.""",
            
            "claude_high_quality": """This comprehensive analysis examines the multidimensional aspects 
                                    of the proposed framework. By integrating diverse methodological approaches, 
                                    we can establish robust conclusions supported by empirical evidence. 
                                    The synthesis reveals nuanced patterns that inform strategic decision-making."""
        }

    def test_model_output_entropy_signatures(self):
        """Test that different model outputs have characteristic entropy signatures"""
        profiles = {}
        
        for model_name, output in self.model_outputs.items():
            profiles[model_name] = self.calculator.analyze_entropy_profile(output)
        
        # Good output should have higher entropy than degraded
        self.assertGreater(profiles["gpt_oss_20b_good"]["token_entropy"],
                          profiles["gpt_oss_20b_degraded"]["token_entropy"],
                          "Good output should have higher token entropy than degraded")
        
        # Collapsed output should have lowest entropy
        self.assertLess(profiles["gpt_oss_20b_collapsed"]["token_entropy"],
                       profiles["gpt_oss_20b_degraded"]["token_entropy"],
                       "Collapsed output should have lowest entropy")
        
        # Claude high-quality should have highest semantic diversity
        self.assertGreater(profiles["claude_high_quality"]["semantic_diversity"],
                          profiles["gpt_oss_20b_good"]["semantic_diversity"],
                          "High-quality output should have higher semantic diversity")

    def test_entropy_based_quality_detection(self):
        """Test using entropy for quality detection"""
        good_profile = self.calculator.analyze_entropy_profile(self.model_outputs["gpt_oss_20b_good"])
        collapsed_profile = self.calculator.analyze_entropy_profile(self.model_outputs["gpt_oss_20b_collapsed"])
        
        # Entropy patterns should detect quality issues
        good_patterns = good_profile["entropy_patterns"]
        collapsed_patterns = collapsed_profile["entropy_patterns"]
        
        # Collapsed output should have repetitive patterns
        self.assertTrue(collapsed_patterns.get("has_repetitive_patterns", False),
                       "Collapsed output should be detected as repetitive")
        
        # Good output should not be flagged as repetitive
        self.assertFalse(good_patterns.get("has_repetitive_patterns", False),
                        "Good output should not be flagged as repetitive")


if __name__ == "__main__":
    # Configure test runner for verbose output
    unittest.main(verbosity=2, buffer=True)