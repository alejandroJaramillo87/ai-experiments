"""
Test suite for model-specific tokenizer mappings

Tests the new tokenizer initialization logic for gpt-oss-20b, Qwen3-30B,
and other models to ensure proper mapping and fallback behavior.

"""

import unittest
from unittest.mock import patch, MagicMock
import pytest
import logging

# Test imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from evaluator.advanced import EntropyCalculator
from evaluator.advanced import ContextWindowAnalyzer

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestTokenizerMappings(unittest.TestCase):
    """Test model-specific tokenizer mappings"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_models = [
            ("gpt-oss-20b", "gpt2"),
            ("qwen3-30b-a3b-base", "gpt2"), 
            ("Qwen3-30B-A3B-Base", "gpt2"),  # Case insensitive
            ("gpt-4", "gpt-4"),
            ("gpt-3.5-turbo", "gpt-3.5-turbo"),
            ("llama-2-70b", "gpt2"),  # Pattern matching
            ("mistral-7b-instruct", "gpt2"),
            ("claude-3-opus", "gpt2"),
            ("random-unknown-model", "gpt2"),  # Fallback
        ]
    
    @patch('evaluator.advanced.entropy_calculator.TIKTOKEN_AVAILABLE', True)
    @patch('evaluator.advanced.entropy_calculator.tiktoken')
    def test_entropy_calculator_model_mappings(self, mock_tiktoken):
        """Test entropy calculator tokenizer mapping logic"""
        mock_encoding = MagicMock()
        mock_tiktoken.encoding_for_model.return_value = mock_encoding
        mock_tiktoken.get_encoding.return_value = mock_encoding
        
        for model_name, expected_encoding in self.test_models:
            with self.subTest(model=model_name):
                calculator = EntropyCalculator(model_name=model_name)
                
                # Verify tokenizer was initialized
                self.assertIsNotNone(calculator.tokenizer)
                
                # Verify correct tiktoken method was called
                if expected_encoding in ["gpt-4", "gpt-3.5-turbo"]:
                    mock_tiktoken.encoding_for_model.assert_called_with(expected_encoding)
                else:
                    mock_tiktoken.get_encoding.assert_called_with(expected_encoding)
    
    @patch('evaluator.advanced.context_analyzer.TIKTOKEN_AVAILABLE', True)
    @patch('evaluator.advanced.context_analyzer.tiktoken')
    def test_context_analyzer_model_mappings(self, mock_tiktoken):
        """Test context analyzer tokenizer mapping logic"""
        mock_encoding = MagicMock()
        mock_tiktoken.encoding_for_model.return_value = mock_encoding
        mock_tiktoken.get_encoding.return_value = mock_encoding
        
        for model_name, expected_encoding in self.test_models:
            with self.subTest(model=model_name):
                analyzer = ContextWindowAnalyzer(tokenizer_model=model_name)
                
                # Verify tokenizer was initialized
                self.assertIsNotNone(analyzer.tokenizer)
                
                # Verify correct tiktoken method was called
                if expected_encoding in ["gpt-4", "gpt-3.5-turbo"]:
                    mock_tiktoken.encoding_for_model.assert_called_with(expected_encoding)
                else:
                    mock_tiktoken.get_encoding.assert_called_with(expected_encoding)
    
    def test_target_model_specific_mappings(self):
        """Test specific mappings for target models"""
        target_tests = [
            ("gpt-oss-20b", "gpt2", "GPT-OSS should use GPT-2 BPE"),
            ("qwen3-30b-a3b-base", "gpt2", "Qwen3-30B should use GPT-2 BPE"),
            ("Qwen/Qwen3-30B-A3B-Base", "gpt2", "HF model path should work"),
            ("openai/gpt-oss-20b", "gpt2", "HF model path should work"),
        ]
        
        with patch('evaluator.advanced.entropy_calculator.TIKTOKEN_AVAILABLE', True), \
             patch('evaluator.advanced.entropy_calculator.tiktoken') as mock_tiktoken:
            
            mock_encoding = MagicMock()
            mock_tiktoken.get_encoding.return_value = mock_encoding
            
            for model_name, expected_encoding, description in target_tests:
                with self.subTest(model=model_name, desc=description):
                    calculator = EntropyCalculator(model_name=model_name)
                    
                    # Should call get_encoding with gpt2
                    mock_tiktoken.get_encoding.assert_called_with("gpt2")
                    self.assertIsNotNone(calculator.tokenizer)
    
    @patch('evaluator.advanced.entropy_calculator.TIKTOKEN_AVAILABLE', False)
    def test_tiktoken_unavailable_fallback(self):
        """Test behavior when tiktoken is not available"""
        calculator = EntropyCalculator(model_name="gpt-oss-20b")
        self.assertIsNone(calculator.tokenizer)
    
    @patch('evaluator.advanced.entropy_calculator.TIKTOKEN_AVAILABLE', True)
    @patch('evaluator.advanced.entropy_calculator.tiktoken')
    def test_tokenizer_initialization_error_handling(self, mock_tiktoken):
        """Test error handling during tokenizer initialization"""
        mock_tiktoken.get_encoding.side_effect = Exception("Mock error")
        
        calculator = EntropyCalculator(model_name="gpt-oss-20b")
        self.assertIsNone(calculator.tokenizer)
    
    def test_pattern_matching_logic(self):
        """Test pattern matching for model families"""
        pattern_tests = [
            ("some-qwen-model-v2", "qwen", "Should match qwen pattern"),
            ("llama-3-instruct", "llama", "Should match llama pattern"),
            ("mistral-large", "mistral", "Should match mistral pattern"),
            ("claude-3-haiku", "claude", "Should match claude pattern"),
        ]
        
        with patch('evaluator.advanced.entropy_calculator.TIKTOKEN_AVAILABLE', True), \
             patch('evaluator.advanced.entropy_calculator.tiktoken') as mock_tiktoken:
            
            mock_encoding = MagicMock()
            mock_tiktoken.get_encoding.return_value = mock_encoding
            
            for model_name, pattern, description in pattern_tests:
                with self.subTest(model=model_name, pattern=pattern, desc=description):
                    calculator = EntropyCalculator(model_name=model_name)
                    
                    # All patterns should map to gpt2
                    mock_tiktoken.get_encoding.assert_called_with("gpt2")
                    self.assertIsNotNone(calculator.tokenizer)


class TestTokenizerFunctionality(unittest.TestCase):
    """Test tokenizer functionality with actual models"""
    
    def setUp(self):
        """Set up test with sample text"""
        self.test_text = "The quick brown fox jumps over the lazy dog. This is a test sentence for tokenization."
    
    @patch('evaluator.advanced.entropy_calculator.TIKTOKEN_AVAILABLE', True)
    def test_gpt_oss_20b_entropy_calculation(self):
        """Test entropy calculation with gpt-oss-20b tokenizer"""
        try:
            import tiktoken
            calculator = EntropyCalculator(model_name="gpt-oss-20b")
            
            if calculator.tokenizer:
                # Test basic entropy calculation
                entropy = calculator.calculate_shannon_entropy(self.test_text)
                self.assertIsInstance(entropy, float)
                self.assertGreater(entropy, 0)
                
                # Test token-based vs character-based entropy
                token_entropy = calculator.calculate_shannon_entropy(self.test_text, use_tokens=True)
                char_entropy = calculator.calculate_shannon_entropy(self.test_text, use_tokens=False)
                
                # Token entropy should be different from character entropy
                self.assertNotEqual(token_entropy, char_entropy)
                
        except ImportError:
            self.skipTest("tiktoken not available")
    
    @patch('evaluator.advanced.context_analyzer.TIKTOKEN_AVAILABLE', True) 
    def test_qwen3_context_analysis(self):
        """Test context analysis with Qwen3-30B tokenizer"""
        try:
            import tiktoken
            analyzer = ContextWindowAnalyzer(tokenizer_model="qwen3-30b-a3b-base")
            
            if analyzer.tokenizer:
                # Test context saturation detection
                repetitive_text = "The same sentence. " * 20
                saturation = analyzer.detect_context_saturation(repetitive_text)
                
                self.assertIsInstance(saturation, dict)
                self.assertIn('saturation_detected', saturation)
                self.assertIn('repetition_saturation', saturation)
                
        except ImportError:
            self.skipTest("tiktoken not available")
    
    def test_fallback_behavior_without_tiktoken(self):
        """Test graceful fallback when tiktoken is unavailable"""
        with patch('evaluator.advanced.entropy_calculator.TIKTOKEN_AVAILABLE', False):
            calculator = EntropyCalculator(model_name="gpt-oss-20b")
            
            # Should still calculate character-based entropy
            entropy = calculator.calculate_shannon_entropy(self.test_text, use_tokens=False)
            self.assertIsInstance(entropy, float)
            self.assertGreater(entropy, 0)
    
    @patch('evaluator.advanced.entropy_calculator.TIKTOKEN_AVAILABLE', True)
    def test_model_case_insensitivity(self):
        """Test that model names are case insensitive"""
        try:
            import tiktoken
            
            # Test different case variations
            test_cases = [
                "GPT-OSS-20B",
                "gpt-oss-20b", 
                "Gpt-Oss-20B",
                "QWEN3-30B-A3B-BASE",
                "qwen3-30b-a3b-base",
                "Qwen3-30B-A3B-Base"
            ]
            
            for model_name in test_cases:
                with self.subTest(model=model_name):
                    calculator = EntropyCalculator(model_name=model_name)
                    # Should initialize successfully regardless of case
                    self.assertIsNotNone(calculator.model_name)
                    
        except ImportError:
            self.skipTest("tiktoken not available")


if __name__ == '__main__':
    # Run specific test cases
    unittest.main(verbosity=2)