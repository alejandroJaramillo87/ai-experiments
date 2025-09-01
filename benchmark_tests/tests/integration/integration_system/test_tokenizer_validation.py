"""
Direct validation of tokenizer improvements for target models

Tests that the updated tokenizer mappings work correctly for gpt-oss-20b
and Qwen3-30B without complex integration dependencies.

Author: Claude Code  
Version: 1.0.0
"""

import unittest
import logging
import sys
import os

# Test imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from evaluator.advanced import EntropyCalculator
from evaluator.advanced import ContextWindowAnalyzer

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestTokenizerValidation(unittest.TestCase):
    """Direct validation of tokenizer improvements"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_text = """
        Machine learning involves algorithms that improve through experience. 
        Neural networks use backpropagation for training. Deep learning models
        can process complex patterns in data like images and natural language.
        """
        
        self.target_models = [
            "gpt-oss-20b",
            "qwen3-30b-a3b-base", 
            "Qwen/Qwen3-30B-A3B-Base",  # HuggingFace format
            "openai/gpt-oss-20b"        # HuggingFace format
        ]
        
    def test_gpt_oss_20b_entropy_calculator(self):
        """Test entropy calculator with GPT-OSS-20B tokenizer"""
        try:
            calculator = EntropyCalculator(model_name="gpt-oss-20b")
            
            # Should initialize without errors
            self.assertIsNotNone(calculator)
            
            # Test entropy calculation
            entropy = calculator.calculate_shannon_entropy(self.test_text)
            self.assertIsInstance(entropy, float)
            self.assertGreater(entropy, 0)
            
            # If tokenizer available, test token-based calculation
            if calculator.tokenizer is not None:
                token_entropy = calculator.calculate_shannon_entropy(self.test_text, use_tokens=True)
                char_entropy = calculator.calculate_shannon_entropy(self.test_text, use_tokens=False)
                
                # Token and character entropy should differ
                self.assertNotEqual(token_entropy, char_entropy)
                
                # Token entropy should be reasonable (not extreme values)
                self.assertGreater(token_entropy, 1.0)
                self.assertLess(token_entropy, 10.0)
                
            print(f"✓ GPT-OSS-20B entropy calculation successful: {entropy:.3f}")
            
        except Exception as e:
            self.fail(f"GPT-OSS-20B entropy calculation failed: {e}")
    
    def test_qwen3_30b_entropy_calculator(self):
        """Test entropy calculator with Qwen3-30B tokenizer"""
        try:
            calculator = EntropyCalculator(model_name="qwen3-30b-a3b-base")
            
            # Should initialize without errors
            self.assertIsNotNone(calculator)
            
            # Test entropy calculation
            entropy = calculator.calculate_shannon_entropy(self.test_text)
            self.assertIsInstance(entropy, float)
            self.assertGreater(entropy, 0)
            
            print(f"✓ Qwen3-30B entropy calculation successful: {entropy:.3f}")
            
        except Exception as e:
            self.fail(f"Qwen3-30B entropy calculation failed: {e}")
    
    def test_gpt_oss_20b_context_analyzer(self):
        """Test context analyzer with GPT-OSS-20B tokenizer"""
        try:
            analyzer = ContextWindowAnalyzer(tokenizer_model="gpt-oss-20b")
            
            # Should initialize without errors
            self.assertIsNotNone(analyzer)
            
            # Test context saturation detection
            repetitive_text = "The same idea repeated. " * 15
            saturation = analyzer.detect_context_saturation(repetitive_text)
            
            self.assertIsInstance(saturation, dict)
            self.assertIn('saturation_detected', saturation)
            
            print(f"✓ GPT-OSS-20B context analysis successful")
            
        except Exception as e:
            self.fail(f"GPT-OSS-20B context analysis failed: {e}")
    
    def test_qwen3_30b_context_analyzer(self):
        """Test context analyzer with Qwen3-30B tokenizer"""
        try:
            analyzer = ContextWindowAnalyzer(tokenizer_model="qwen3-30b-a3b-base")
            
            # Should initialize without errors
            self.assertIsNotNone(analyzer)
            
            # Test basic functionality
            multilingual_text = """
            English text followed by Chinese: 你好世界。
            Then German: Guten Tag. And French: Bonjour le monde.
            Mixed languages test tokenization capabilities.
            """
            
            saturation = analyzer.detect_context_saturation(multilingual_text)
            
            self.assertIsInstance(saturation, dict)
            self.assertIn('saturation_detected', saturation)
            
            print(f"✓ Qwen3-30B multilingual context analysis successful")
            
        except Exception as e:
            self.fail(f"Qwen3-30B context analysis failed: {e}")
    
    def test_huggingface_model_path_handling(self):
        """Test that HuggingFace model paths work correctly"""
        hf_models = [
            "openai/gpt-oss-20b",
            "Qwen/Qwen3-30B-A3B-Base"
        ]
        
        for model_path in hf_models:
            with self.subTest(model=model_path):
                try:
                    calculator = EntropyCalculator(model_name=model_path)
                    entropy = calculator.calculate_shannon_entropy("Test text for HF models.")
                    
                    self.assertIsInstance(entropy, float)
                    self.assertGreater(entropy, 0)
                    
                    print(f"✓ HuggingFace model path {model_path} handled successfully")
                    
                except Exception as e:
                    self.fail(f"HuggingFace model path {model_path} failed: {e}")
    
    def test_tokenizer_consistency(self):
        """Test that same text produces consistent results across model variants"""
        test_cases = [
            ("gpt-oss-20b", "openai/gpt-oss-20b"),
            ("qwen3-30b-a3b-base", "Qwen/Qwen3-30B-A3B-Base")
        ]
        
        for base_model, hf_model in test_cases:
            with self.subTest(base=base_model, hf=hf_model):
                try:
                    calc1 = EntropyCalculator(model_name=base_model)
                    calc2 = EntropyCalculator(model_name=hf_model)
                    
                    entropy1 = calc1.calculate_shannon_entropy(self.test_text)
                    entropy2 = calc2.calculate_shannon_entropy(self.test_text)
                    
                    # Should be identical since they map to same tokenizer
                    self.assertAlmostEqual(entropy1, entropy2, places=5)
                    
                    print(f"✓ Consistent results for {base_model} vs {hf_model}")
                    
                except Exception as e:
                    self.fail(f"Consistency test failed for {base_model}/{hf_model}: {e}")
    
    def test_fallback_robustness(self):
        """Test that unknown models fall back gracefully"""
        unknown_models = [
            "unknown-model-123",
            "some-future-model",
            "non-existent/model-path"
        ]
        
        for model_name in unknown_models:
            with self.subTest(model=model_name):
                try:
                    calculator = EntropyCalculator(model_name=model_name)
                    
                    # Should still work with fallback tokenization
                    entropy = calculator.calculate_shannon_entropy(self.test_text)
                    self.assertIsInstance(entropy, float)
                    self.assertGreater(entropy, 0)
                    
                    print(f"✓ Fallback handling successful for {model_name}")
                    
                except Exception as e:
                    self.fail(f"Fallback failed for {model_name}: {e}")
    
    def test_performance_impact(self):
        """Test that tokenizer improvements don't significantly impact performance"""
        import time
        
        # Test with multiple models
        models = ["gpt-oss-20b", "qwen3-30b-a3b-base", "gpt-4"]
        times = []
        
        for model in models:
            start_time = time.time()
            
            calculator = EntropyCalculator(model_name=model)
            for _ in range(5):  # Multiple iterations
                entropy = calculator.calculate_shannon_entropy(self.test_text)
                
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            # Should complete quickly
            self.assertLess(elapsed, 1.0, f"Performance too slow for {model}")
            
        # Performance should be consistent across models  
        max_time = max(times)
        min_time = min(times)
        
        # Variance shouldn't be extreme
        if min_time > 0:
            ratio = max_time / min_time
            self.assertLess(ratio, 2000.0, "Performance variance extremely high across models")
        
        print(f"✓ Performance test passed - times: {[f'{t:.3f}s' for t in times]}")


if __name__ == '__main__':
    unittest.main(verbosity=2)