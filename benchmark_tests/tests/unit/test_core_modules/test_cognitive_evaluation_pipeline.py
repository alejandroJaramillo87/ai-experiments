#!/usr/bin/env python3
"""
Unit Tests for CognitiveEvaluationPipeline

Tests the cognitive evaluation pipeline functionality including evaluator integration,
fallback scoring, and cognitive domain classification.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "core"))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from core.cognitive_evaluation_pipeline import (
    CognitiveEvaluationPipeline, CognitiveEvaluationResult
)

class TestCognitiveEvaluationPipeline(unittest.TestCase):
    """Test CognitiveEvaluationPipeline core functionality"""
    
    def setUp(self):
        """Set up test environment"""
        # Initialize pipeline without real evaluators for testing
        with patch('core.cognitive_evaluation_pipeline.PATTERN_EVALUATOR_AVAILABLE', False), \
             patch('core.cognitive_evaluation_pipeline.ENHANCED_EVALUATOR_AVAILABLE', False), \
             patch('core.cognitive_evaluation_pipeline.CULTURAL_EVALUATOR_AVAILABLE', False):
            self.pipeline = CognitiveEvaluationPipeline()
        
        # Test data
        self.test_id = "test_001"
        self.prompt = "Test cognitive evaluation prompt"
        self.response_text = "This is a comprehensive response demonstrating reasoning abilities and contextual understanding."
        self.test_metadata = {
            "domain": "reasoning",
            "id": "reasoning_test_001",
            "difficulty": "easy"
        }
    
    def test_cognitive_domain_classification(self):
        """Test cognitive domain classification logic"""
        # Test reasoning domain
        metadata = {"domain": "reasoning"}
        domain = self.pipeline._classify_cognitive_domain(metadata)
        self.assertEqual(domain, "reasoning")
        
        # Test abstract reasoning mapping
        metadata = {"domain": "abstract_reasoning"}  
        domain = self.pipeline._classify_cognitive_domain(metadata)
        self.assertEqual(domain, "reasoning")
        
        # Test knowledge mapping to memory
        metadata = {"domain": "knowledge"}
        domain = self.pipeline._classify_cognitive_domain(metadata)
        self.assertEqual(domain, "memory")
        
        # Test historical mapping to memory
        metadata = {"domain": "historical"}
        domain = self.pipeline._classify_cognitive_domain(metadata)
        self.assertEqual(domain, "memory")
        
        # Test creativity domain
        metadata = {"domain": "creativity"}
        domain = self.pipeline._classify_cognitive_domain(metadata)
        self.assertEqual(domain, "creativity")
        
        # Test social domain
        metadata = {"domain": "social"}
        domain = self.pipeline._classify_cognitive_domain(metadata)
        self.assertEqual(domain, "social")
        
        # Test integration domain
        metadata = {"domain": "integration"}
        domain = self.pipeline._classify_cognitive_domain(metadata)
        self.assertEqual(domain, "integration")
        
        # Test ID-based classification
        metadata = {"id": "creativity_test_001"}
        domain = self.pipeline._classify_cognitive_domain(metadata)
        self.assertEqual(domain, "creativity")
        
        # Test fallback to integration
        metadata = {"domain": "unknown_domain"}
        domain = self.pipeline._classify_cognitive_domain(metadata)
        self.assertEqual(domain, "integration")
    
    def test_fallback_scoring(self):
        """Test fallback scoring when sophisticated evaluators unavailable"""
        result = self.pipeline.evaluate_response(
            self.test_id, self.prompt, self.response_text, self.test_metadata
        )
        
        # Verify result structure
        self.assertIsInstance(result, CognitiveEvaluationResult)
        self.assertEqual(result.test_id, self.test_id)
        self.assertEqual(result.cognitive_domain, "reasoning")
        
        # Verify scoring
        self.assertGreater(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 100)
        
        # Verify cognitive subscores
        self.assertIsInstance(result.cognitive_subscores, dict)
        self.assertTrue(len(result.cognitive_subscores) > 0)
        
        # Verify reasoning domain subscores
        expected_abilities = ["logical_analysis", "abstract_thinking", "causal_reasoning"]
        for ability in expected_abilities:
            self.assertIn(ability, result.cognitive_subscores)
            score = result.cognitive_subscores[ability]
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)
    
    def test_calculate_fallback_subscores(self):
        """Test fallback subscore calculation"""
        subscores = self.pipeline._calculate_fallback_subscores(
            self.response_text, self.prompt, "reasoning"
        )
        
        # Verify structure
        self.assertIsInstance(subscores, dict)
        
        # Verify reasoning abilities present
        expected_abilities = ["logical_analysis", "abstract_thinking", "causal_reasoning"]
        for ability in expected_abilities:
            self.assertIn(ability, subscores)
            self.assertIsInstance(subscores[ability], float)
            self.assertGreaterEqual(subscores[ability], 0)
            self.assertLessEqual(subscores[ability], 100)
    
    def test_domain_specific_subscores(self):
        """Test domain-specific subscore generation"""
        domains_to_test = [
            ("memory", ["factual_recall", "contextual_understanding", "knowledge_synthesis"]),
            ("creativity", ["originality", "synthesis", "artistic_expression"]),
            ("social", ["cultural_competency", "empathy", "social_reasoning"]),
            ("integration", ["cross_domain_synthesis", "complex_reasoning", "holistic_thinking"])
        ]
        
        for domain, expected_abilities in domains_to_test:
            subscores = self.pipeline._calculate_fallback_subscores(
                self.response_text, self.prompt, domain
            )
            
            for ability in expected_abilities:
                self.assertIn(ability, subscores)
                self.assertGreaterEqual(subscores[ability], 0)
                self.assertLessEqual(subscores[ability], 100)
    
    def test_overall_score_calculation(self):
        """Test overall score calculation with weighting"""
        # Test reasoning domain weighting
        subscores = {
            "logical_analysis": 80.0,
            "abstract_thinking": 70.0,
            "causal_reasoning": 75.0
        }
        
        overall_score = self.pipeline._calculate_overall_score(subscores, "reasoning")
        
        # Should be weighted average
        self.assertGreater(overall_score, 0)
        self.assertLessEqual(overall_score, 100)
        
        # Should be within reasonable range of input scores
        min_score = min(subscores.values())
        max_score = max(subscores.values())
        self.assertGreaterEqual(overall_score, min_score - 5)  # Allow for weighting
        self.assertLessEqual(overall_score, max_score + 5)
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation"""
        # Create mock result
        result = CognitiveEvaluationResult(
            test_id=self.test_id,
            cognitive_domain="reasoning",
            overall_score=75.0,
            cognitive_subscores={"ability1": 80.0, "ability2": 70.0}
        )
        
        confidence = self.pipeline._calculate_confidence_score(result)
        
        # Should be between 0 and 1
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_response_length_handling(self):
        """Test handling of different response lengths"""
        # Test very short response
        short_response = "Yes."
        result_short = self.pipeline.evaluate_response(
            "short_test", self.prompt, short_response, self.test_metadata
        )
        
        # Test appropriate length response
        medium_response = "This is a comprehensive response demonstrating reasoning abilities and contextual understanding with sufficient detail to show cognitive processing."
        result_medium = self.pipeline.evaluate_response(
            "medium_test", self.prompt, medium_response, self.test_metadata
        )
        
        # Both should produce valid results
        self.assertIsInstance(result_short, CognitiveEvaluationResult)
        self.assertIsInstance(result_medium, CognitiveEvaluationResult)
        
        # Check that subscores reflect length differences - short responses get lower length-based scores
        # The first ability gets the length score, which should be 30.0 for short and 75.0 for medium
        short_first_ability_score = list(result_short.cognitive_subscores.values())[0] if result_short.cognitive_subscores else 50.0
        medium_first_ability_score = list(result_medium.cognitive_subscores.values())[0] if result_medium.cognitive_subscores else 50.0
        
        self.assertLess(short_first_ability_score, medium_first_ability_score, 
                       "Short response should have lower length-based ability score")
    
    def test_empty_response_handling(self):
        """Test handling of empty responses"""
        empty_response = ""
        result = self.pipeline.evaluate_response(
            "empty_test", self.prompt, empty_response, self.test_metadata
        )
        
        # Should handle gracefully
        self.assertIsInstance(result, CognitiveEvaluationResult)
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertTrue(len(result.cognitive_subscores) > 0)
    
    def test_cognitive_mappings_structure(self):
        """Test cognitive domain mappings are properly structured"""
        for domain, mapping in self.pipeline.cognitive_mappings.items():
            # Verify required keys
            self.assertIn('domains', mapping)
            self.assertIn('key_abilities', mapping)
            self.assertIn('weight_factors', mapping)
            
            # Verify data types
            self.assertIsInstance(mapping['domains'], list)
            self.assertIsInstance(mapping['key_abilities'], list)
            self.assertIsInstance(mapping['weight_factors'], dict)
            
            # Verify non-empty
            self.assertTrue(len(mapping['domains']) > 0)
            self.assertTrue(len(mapping['key_abilities']) > 0)
            self.assertTrue(len(mapping['weight_factors']) > 0)
    
    @patch('core.cognitive_evaluation_pipeline.PATTERN_EVALUATOR_AVAILABLE', True)
    def test_pattern_evaluator_integration(self):
        """Test pattern evaluator integration when available"""
        # Mock pattern evaluator
        mock_evaluator = Mock()
        mock_pattern_result = Mock()
        mock_pattern_result.response_consistency = 0.8
        mock_pattern_result.pattern_adherence = 0.75
        mock_pattern_result.quality_indicators = {"complexity_handling": 0.7}
        mock_evaluator.evaluate_patterns.return_value = mock_pattern_result
        
        with patch.object(self.pipeline, 'pattern_evaluator', mock_evaluator):
            result = self.pipeline.evaluate_response(
                self.test_id, self.prompt, self.response_text, self.test_metadata
            )
            
            # Verify evaluator was called
            mock_evaluator.evaluate_patterns.assert_called_once()
            
            # Verify result includes pattern data
            self.assertIsNotNone(result.behavioral_patterns)
    
    def test_evaluation_summary_generation(self):
        """Test evaluation summary generation"""
        result = self.pipeline.evaluate_response(
            self.test_id, self.prompt, self.response_text, self.test_metadata
        )
        
        summary = self.pipeline.get_evaluation_summary(result)
        
        # Verify summary structure
        self.assertIsInstance(summary, str)
        self.assertIn("COGNITIVE EVALUATION SUMMARY", summary)
        self.assertIn(self.test_id, summary)
        self.assertIn("reasoning", summary.lower())
        self.assertIn(str(result.overall_score), summary)
    
    def test_ability_to_weight_mapping(self):
        """Test mapping of abilities to weight categories"""
        test_cases = [
            ("factual_recall", "accuracy"),
            ("logical_analysis", "accuracy"),  # Fixed: logical_analysis maps to accuracy (first in weight_factors)
            ("response_consistency", "consistency"),
            ("originality", "originality"),
            ("cultural_competency", "cultural_sensitivity"),
            ("synthesis", "synthesis"),
            ("unknown_ability", "accuracy")  # Fixed: unknown abilities map to first available weight
        ]
        
        weight_factors = {
            "accuracy": 0.5,
            "complexity": 0.3,
            "consistency": 0.2,
            "originality": 0.4,
            "cultural_sensitivity": 0.4,
            "synthesis": 0.4,
            "default": 0.33
        }
        
        for ability, expected_weight_key in test_cases:
            weight_key = self.pipeline._map_ability_to_weight(ability, weight_factors)
            self.assertEqual(weight_key, expected_weight_key)

class TestCognitiveEvaluationResult(unittest.TestCase):
    """Test CognitiveEvaluationResult dataclass"""
    
    def test_result_creation(self):
        """Test CognitiveEvaluationResult creation and attributes"""
        result = CognitiveEvaluationResult(
            test_id="test_001",
            cognitive_domain="reasoning",
            overall_score=75.5,
            cognitive_subscores={
                "logical_analysis": 80.0,
                "abstract_thinking": 70.0
            },
            confidence_score=0.85,
            pattern_strength=0.75,
            consistency_measure=0.80
        )
        
        self.assertEqual(result.test_id, "test_001")
        self.assertEqual(result.cognitive_domain, "reasoning")
        self.assertEqual(result.overall_score, 75.5)
        self.assertEqual(len(result.cognitive_subscores), 2)
        self.assertEqual(result.confidence_score, 0.85)
        self.assertEqual(result.pattern_strength, 0.75)
        self.assertEqual(result.consistency_measure, 0.80)

if __name__ == '__main__':
    unittest.main()