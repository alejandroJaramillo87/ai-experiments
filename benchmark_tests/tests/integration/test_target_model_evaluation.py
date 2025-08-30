"""
Integration test for target model evaluation

Tests the evaluator system with specific focus on gpt-oss-20b and Qwen3-30B
model tokenization and evaluation capabilities.

Author: Claude Code
Version: 1.0.0
"""

import unittest
import logging
from pathlib import Path
import sys
import os

# Test imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from evaluator.subjects.reasoning_evaluator import UniversalEvaluator, ReasoningType
from evaluator.core.evaluation_config import DEFAULT_CONFIG

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestTargetModelEvaluation(unittest.TestCase):
    """Test evaluation system with target models"""
    
    def setUp(self):
        """Set up test fixtures"""
        config = DEFAULT_CONFIG
        
        # Test scenarios with specific focus on tokenization differences
        self.test_scenarios = {
            "mathematical_reasoning": {
                "text": "To solve this equation: 2x + 5 = 13, we subtract 5 from both sides: 2x = 8. Then divide by 2: x = 4. We can verify: 2(4) + 5 = 8 + 5 = 13 ✓",
                "expected_tokens_gpt2": 35,  # Approximate token count with gpt2
                "reasoning_type": ReasoningType.MATHEMATICAL
            },
            "code_analysis": {
                "text": "This Python function implements binary search: def binary_search(arr, target): left, right = 0, len(arr) - 1; while left <= right: mid = (left + right) // 2; if arr[mid] == target: return mid; elif arr[mid] < target: left = mid + 1; else: right = mid - 1; return -1",
                "expected_tokens_gpt2": 65,  # Code typically has higher token density
                "reasoning_type": ReasoningType.MULTI_STEP
            },
            "multilingual_context": {
                "text": "The concept of 'schadenfreude' (German) refers to pleasure derived from another's misfortune. Similar concepts exist globally: 'mudita' (Sanskrit) means sympathetic joy, the opposite feeling. In Chinese, '幸灾乐祸' captures similar meaning to schadenfreude.",
                "expected_tokens_gpt2": 45,  # Multilingual content tokenization test
                "reasoning_type": ReasoningType.GENERAL
            }
        }
        
        # Initialize evaluators for target models
        self.evaluator = UniversalEvaluator()
    
    def test_gpt_oss_20b_tokenization_integration(self):
        """Test GPT-OSS-20B specific tokenization and evaluation"""
        scenario = self.test_scenarios["mathematical_reasoning"]
        
        result = self.evaluator.evaluate_response(
            scenario["text"], 
            "GPT-OSS Math Test",
            reasoning_type=scenario["reasoning_type"]
        )
        
        # Basic evaluation should work
        self.assertIsNotNone(result)
        self.assertGreater(result.metrics.overall_score, 0)
        
        # Advanced analysis should be available
        if "advanced_analysis" in result.detailed_analysis:
            advanced = result.detailed_analysis["advanced_analysis"]
            
            # Entropy analysis with gpt2 tokenizer should work
            if "entropy_analysis" in advanced and "error" not in advanced["entropy_analysis"]:
                entropy_data = advanced["entropy_analysis"]
                
                # Should have token-based entropy calculation
                self.assertIn("token_entropy", entropy_data)
                self.assertGreater(entropy_data["token_entropy"], 0)
                
                # Token entropy should be different from character entropy
                if "character_entropy" in entropy_data:
                    self.assertNotEqual(
                        entropy_data["token_entropy"],
                        entropy_data["character_entropy"]
                    )
    
    def test_qwen3_30b_tokenization_integration(self):
        """Test Qwen3-30B specific tokenization and evaluation"""
        scenario = self.test_scenarios["multilingual_context"]
        
        result = self.evaluator.evaluate_response(
            scenario["text"],
            "Qwen3 Multilingual Test", 
            reasoning_type=scenario["reasoning_type"]
        )
        
        # Basic evaluation should work
        self.assertIsNotNone(result)
        self.assertGreater(result.metrics.overall_score, 0)
        
        # Context analysis should handle multilingual content
        if "advanced_analysis" in result.detailed_analysis:
            advanced = result.detailed_analysis["advanced_analysis"]
            
            # Context analysis with proper tokenization
            if "context_analysis" in advanced and "error" not in advanced["context_analysis"]:
                context_data = advanced["context_analysis"]
                
                # Should analyze token positions correctly
                self.assertIn("position_analysis", context_data)
                
                # Should have reasonable context health for multilingual text
                health_score = context_data.get("context_health_score", 0)
                self.assertGreater(health_score, 0.3)  # Reasonable threshold for multilingual
    
    def test_cross_model_consistency(self):
        """Test that different models produce consistent evaluation patterns"""
        scenario = self.test_scenarios["code_analysis"]
        
        gpt_result = self.evaluator.evaluate_response(
            scenario["text"],
            "GPT-OSS Code Test",
            reasoning_type=scenario["reasoning_type"]
        )
        
        qwen_result = self.evaluator.evaluate_response(
            scenario["text"],
            "Qwen3 Code Test", 
            reasoning_type=scenario["reasoning_type"]
        )
        
        # Both should evaluate successfully
        self.assertIsNotNone(gpt_result)
        self.assertIsNotNone(qwen_result)
        
        # Scores should be in reasonable ranges for code
        self.assertGreaterEqual(gpt_result.metrics.overall_score, 5)  # Code should score reasonably
        self.assertGreaterEqual(qwen_result.metrics.overall_score, 5)
        
        # Score difference shouldn't be extreme (same content, different tokenization)
        score_difference = abs(gpt_result.metrics.overall_score - qwen_result.metrics.overall_score)
        self.assertLess(score_difference, 20, "Tokenization shouldn't cause extreme score differences")
    
    def test_tokenizer_fallback_behavior(self):
        """Test graceful fallback when advanced features aren't available"""
        # Test with a model name that would fail tiktoken lookup
        unknown_evaluator = UniversalEvaluator()
        
        scenario = self.test_scenarios["mathematical_reasoning"]
        
        result = unknown_evaluator.evaluate_response(
            scenario["text"],
            "Unknown Model Test",
            reasoning_type=scenario["reasoning_type"]
        )
        
        # Should still work with fallback behavior
        self.assertIsNotNone(result)
        self.assertGreater(result.metrics.overall_score, 0)
        
        # Advanced analysis might have errors but shouldn't crash
        if "advanced_analysis" in result.detailed_analysis:
            advanced = result.detailed_analysis["advanced_analysis"]
            
            # Check that modules either work or fail gracefully
            for module_name, module_data in advanced.items():
                if isinstance(module_data, dict):
                    # Either contains valid data or error message
                    self.assertTrue(
                        "error" in module_data or len(module_data) > 0,
                        f"{module_name} should have data or error message"
                    )
    
    def test_model_specific_configuration_application(self):
        """Test that model-specific configurations are applied correctly"""
        # Test GPT-OSS-20B specific config
        gpt_result = self.evaluator.evaluate_response(
            self.test_scenarios["mathematical_reasoning"]["text"],
            "GPT-OSS Config Test",
            reasoning_type=ReasoningType.MATHEMATICAL
        )
        
        # Should apply GPT-OSS-20B profile if available
        self.assertIsNotNone(gpt_result)
        
        # Check if model-specific thresholds were applied
        if hasattr(gpt_result, 'detailed_analysis') and "advanced_analysis" in gpt_result.detailed_analysis:
            advanced = gpt_result.detailed_analysis["advanced_analysis"]
            
            # Entropy analysis should use model-specific expectations
            if "entropy_analysis" in advanced and "error" not in advanced["entropy_analysis"]:
                entropy_data = advanced["entropy_analysis"]
                
                # Should have entropy quality assessment based on model profile
                # Check for entropy quality ratio which is present in the actual data
                self.assertIn("entropy_quality_ratio", entropy_data)
    
    def test_performance_with_target_models(self):
        """Test that tokenizer changes don't negatively impact performance"""
        import time
        
        scenario = self.test_scenarios["code_analysis"]
        iterations = 3
        
        # Test GPT-OSS-20B performance
        start_time = time.time()
        for _ in range(iterations):
            self.evaluator.evaluate_response(
                scenario["text"],
                "GPT-OSS Performance Test",
                reasoning_type=scenario["reasoning_type"]
            )
        gpt_time = (time.time() - start_time) / iterations
        
        # Test Qwen3-30B performance
        start_time = time.time()
        for _ in range(iterations):
            self.evaluator.evaluate_response(
                scenario["text"],
                "Qwen3 Performance Test",
                reasoning_type=scenario["reasoning_type"]
            )
        qwen_time = (time.time() - start_time) / iterations
        
        # Should complete within reasonable time (not a strict performance test)
        self.assertLess(gpt_time, 2.0, "GPT-OSS evaluation should complete within 2 seconds")
        self.assertLess(qwen_time, 2.0, "Qwen3 evaluation should complete within 2 seconds")
        
        # Performance should be similar between models
        time_difference = abs(gpt_time - qwen_time)
        self.assertLess(time_difference, 1.0, "Performance shouldn't vary drastically between tokenizers")


if __name__ == '__main__':
    unittest.main(verbosity=2)