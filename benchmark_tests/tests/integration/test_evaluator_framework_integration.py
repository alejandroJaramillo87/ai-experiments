#!/usr/bin/env python3
"""
Comprehensive Evaluator Framework Integration Tests

Tests the integration between all evaluator components including:
- PatternBasedEvaluator + EnhancedUniversalEvaluator integration
- CognitiveEvaluationPipeline orchestration
- Cross-evaluator consistency validation
- Fallback behavior when dependencies unavailable
- Statistical validation of multi-evaluator results

Addresses CLAUDE.md requirements for evaluator integration testing.
"""

import unittest
import tempfile
import shutil
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent.parent / "core"))
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.cognitive_evaluation_pipeline import CognitiveEvaluationPipeline, CognitiveEvaluationResult


class TestEvaluatorFrameworkIntegration(unittest.TestCase):
    """Test comprehensive evaluator framework integration"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = CognitiveEvaluationPipeline()
        
        # Test data for evaluation
        self.test_data_reasoning = {
            'id': 'reasoning_integration_test',
            'prompt': 'If A implies B, and B implies C, what can we conclude about A and C?',
            'domain': 'reasoning',
            'difficulty': 'medium'
        }
        
        self.test_data_creativity = {
            'id': 'creativity_integration_test', 
            'prompt': 'Create a short poem about quantum mechanics using everyday metaphors',
            'domain': 'creativity',
            'difficulty': 'hard'
        }
        
        self.test_data_social = {
            'id': 'social_integration_test',
            'prompt': 'How would you mediate a conflict between team members with different cultural backgrounds?',
            'domain': 'social',
            'difficulty': 'hard'
        }
        
        self.sample_responses = {
            'reasoning': 'Based on the logical principle of transitivity, if A implies B and B implies C, then A implies C. This is a fundamental rule in propositional logic.',
            'creativity': 'Electrons dance like children at play,\nOrbiting atoms in quantum ballet,\nWave-particles spinning in probability\'s way,\nUncertain yet certain, night becoming day.',
            'social': 'I would first listen to each person individually to understand their perspectives, then facilitate a group discussion focusing on shared goals and values while respecting cultural differences.'
        }
    
    def tearDown(self):
        """Clean up integration test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_initialization_with_available_evaluators(self):
        """Test pipeline initializes correctly with available evaluators"""
        pipeline = CognitiveEvaluationPipeline()
        
        # Check that pipeline attempts to initialize evaluators
        self.assertIsNotNone(pipeline.cognitive_mappings)
        self.assertIn('reasoning', pipeline.cognitive_mappings)
        self.assertIn('creativity', pipeline.cognitive_mappings) 
        self.assertIn('social', pipeline.cognitive_mappings)
        
        # Check evaluator initialization
        # Note: evaluators may not be available in test environment
        self.assertTrue(hasattr(pipeline, 'pattern_evaluator'))
        self.assertTrue(hasattr(pipeline, 'enhanced_evaluator'))
        self.assertTrue(hasattr(pipeline, 'cultural_evaluator'))
    
    def test_multi_evaluator_response_analysis(self):
        """Test integration between multiple evaluators on same response"""
        # Test with reasoning response
        result = self.pipeline.evaluate_response(
            test_id=self.test_data_reasoning['id'],
            prompt=self.test_data_reasoning['prompt'],
            response_text=self.sample_responses['reasoning'],
            test_metadata=self.test_data_reasoning
        )
        
        self.assertIsNotNone(result)
        self.assertIsInstance(result, CognitiveEvaluationResult)
        self.assertEqual(result.test_id, 'reasoning_integration_test')
        self.assertEqual(result.cognitive_domain, 'reasoning')
        
        # Verify core scores are present
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 100)
        self.assertIsInstance(result.cognitive_subscores, dict)
        
        # Test pattern analysis integration
        if result.behavioral_patterns:
            self.assertIsInstance(result.behavioral_patterns.response_consistency, float)
            self.assertIsInstance(result.behavioral_patterns.behavioral_signature, dict)
    
    def test_cross_domain_evaluator_consistency(self):
        """Test evaluator consistency across different cognitive domains"""
        results = {}
        
        test_cases = [
            (self.test_data_reasoning, self.sample_responses['reasoning']),
            (self.test_data_creativity, self.sample_responses['creativity']),
            (self.test_data_social, self.sample_responses['social'])
        ]
        
        for test_data, response in test_cases:
            result = self.pipeline.evaluate_response(
                test_id=test_data['id'],
                prompt=test_data['prompt'],
                response_text=response,
                test_metadata=test_data
            )
            results[test_data['domain']] = result
        
        # Check that all evaluations completed successfully
        self.assertEqual(len(results), 3)
        
        # Verify score consistency (scores should be reasonable)
        for domain, result in results.items():
            self.assertGreaterEqual(result.overall_score, 0,
                                  f"Score for {domain} should be non-negative")
            self.assertLessEqual(result.overall_score, 100,
                                f"Score for {domain} should not exceed 100")
            self.assertGreater(result.confidence_score, 0,
                              f"Confidence for {domain} should be positive")
    
    def test_evaluator_pipeline_error_handling(self):
        """Test pipeline behavior when individual evaluators fail"""
        with patch('core.cognitive_evaluation_pipeline.PATTERN_EVALUATOR_AVAILABLE', False):
            pipeline = CognitiveEvaluationPipeline()
            
            # Pipeline should still work even if pattern evaluator unavailable
            result = pipeline.evaluate_response(
                test_id=self.test_data_reasoning['id'],
                prompt=self.test_data_reasoning['prompt'],
                response_text=self.sample_responses['reasoning'],
                test_metadata=self.test_data_reasoning
            )
            
            self.assertIsNotNone(result)
            self.assertIsInstance(result, CognitiveEvaluationResult)
            # Should have graceful degradation
            self.assertGreaterEqual(result.overall_score, 0)
    
    def test_statistical_validation_across_evaluators(self):
        """Test statistical consistency of evaluations"""
        # Run multiple evaluations of the same response
        scores = []
        
        for i in range(5):  # Multiple evaluations for statistical analysis
            result = self.pipeline.evaluate_response(
                test_id=f"{self.test_data_reasoning['id']}_stat_{i}",
                prompt=self.test_data_reasoning['prompt'],
                response_text=self.sample_responses['reasoning'],
                test_metadata=self.test_data_reasoning
            )
            scores.append(result.overall_score)
        
        # Verify statistical properties
        self.assertEqual(len(scores), 5)
        
        # Scores should be consistent (standard deviation should be reasonable)
        import statistics
        if len(set(scores)) > 1:  # Only if scores vary
            std_dev = statistics.stdev(scores)
            mean_score = statistics.mean(scores)
            
            # Coefficient of variation should be reasonable (< 20%)
            cv = std_dev / mean_score if mean_score > 0 else 0
            self.assertLess(cv, 0.2, 
                           f"Score variation too high: std={std_dev}, mean={mean_score}, cv={cv}")
    
    def test_cognitive_domain_mapping_accuracy(self):
        """Test that cognitive domain mapping works correctly"""
        # Test reasoning domain mapping
        result = self.pipeline.evaluate_response(
            test_id=self.test_data_reasoning['id'],
            prompt=self.test_data_reasoning['prompt'],
            response_text=self.sample_responses['reasoning'], 
            test_metadata=self.test_data_reasoning
        )
        
        self.assertEqual(result.cognitive_domain, 'reasoning')
        self.assertIn('logical_analysis', result.cognitive_subscores)
        
        # Test creativity domain mapping
        result = self.pipeline.evaluate_response(
            test_id=self.test_data_creativity['id'],
            prompt=self.test_data_creativity['prompt'],
            response_text=self.sample_responses['creativity'],
            test_metadata=self.test_data_creativity
        )
        
        self.assertEqual(result.cognitive_domain, 'creativity')
        
        # Test social domain mapping  
        result = self.pipeline.evaluate_response(
            test_id=self.test_data_social['id'],
            prompt=self.test_data_social['prompt'],
            response_text=self.sample_responses['social'],
            test_metadata=self.test_data_social
        )
        
        self.assertEqual(result.cognitive_domain, 'social')
    
    def test_evaluation_aggregation_logic(self):
        """Test that multiple evaluator results are properly aggregated"""
        result = self.pipeline.evaluate_response(
            test_id=self.test_data_reasoning['id'],
            prompt=self.test_data_reasoning['prompt'],
            response_text=self.sample_responses['reasoning'],
            test_metadata=self.test_data_reasoning
        )
        
        # Verify aggregation produces complete result
        self.assertIsNotNone(result.overall_score)
        self.assertIsNotNone(result.cognitive_subscores)
        self.assertIsInstance(result.evaluation_details, dict)
        
        # Check that raw scores from different evaluators are preserved
        if result.raw_scores:
            self.assertIsInstance(result.raw_scores, dict)
            self.assertGreater(len(result.raw_scores), 0)


class TestEvaluatorFallbackBehavior(unittest.TestCase):
    """Test evaluator fallback behavior when dependencies unavailable"""
    
    def setUp(self):
        """Set up fallback testing environment"""
        self.test_data = {
            'id': 'fallback_test',
            'prompt': 'Test prompt for fallback behavior',
            'domain': 'reasoning'
        }
        self.test_response = "Test response for fallback evaluation"
    
    def test_pattern_evaluator_fallback(self):
        """Test pipeline behavior when PatternBasedEvaluator unavailable"""
        with patch('core.cognitive_evaluation_pipeline.PATTERN_EVALUATOR_AVAILABLE', False):
            pipeline = CognitiveEvaluationPipeline()
            
            result = pipeline.evaluate_response(
                test_id=self.test_data['id'],
                prompt=self.test_data['prompt'],
                response_text=self.test_response,
                test_metadata=self.test_data
            )
            
            # Should still return valid result
            self.assertIsNotNone(result)
            self.assertIsInstance(result, CognitiveEvaluationResult)
            self.assertIsNone(result.behavioral_patterns)  # Pattern analysis unavailable
            self.assertGreaterEqual(result.overall_score, 0)
    
    def test_enhanced_evaluator_fallback(self):
        """Test pipeline behavior when EnhancedUniversalEvaluator unavailable"""
        with patch('core.cognitive_evaluation_pipeline.ENHANCED_EVALUATOR_AVAILABLE', False):
            pipeline = CognitiveEvaluationPipeline()
            
            result = pipeline.evaluate_response(
                test_id=self.test_data['id'],
                prompt=self.test_data['prompt'],
                response_text=self.test_response,
                test_metadata=self.test_data
            )
            
            # Should gracefully degrade
            self.assertIsNotNone(result)
            self.assertGreaterEqual(result.overall_score, 0)
    
    def test_cultural_evaluator_fallback(self):
        """Test pipeline behavior when CulturalAuthenticityAnalyzer unavailable"""
        with patch('core.cognitive_evaluation_pipeline.CULTURAL_EVALUATOR_AVAILABLE', False):
            pipeline = CognitiveEvaluationPipeline()
            
            result = pipeline.evaluate_response(
                test_id=self.test_data['id'],
                prompt=self.test_data['prompt'],
                response_text=self.test_response,
                test_metadata=self.test_data
            )
            
            # Should still work for non-cultural domains
            self.assertIsNotNone(result)
            self.assertIsNone(result.cultural_analysis)  # Cultural analysis unavailable
    
    def test_all_evaluators_unavailable_fallback(self):
        """Test pipeline behavior when all advanced evaluators unavailable"""
        with patch('core.cognitive_evaluation_pipeline.PATTERN_EVALUATOR_AVAILABLE', False), \
             patch('core.cognitive_evaluation_pipeline.ENHANCED_EVALUATOR_AVAILABLE', False), \
             patch('core.cognitive_evaluation_pipeline.CULTURAL_EVALUATOR_AVAILABLE', False), \
             patch('core.cognitive_evaluation_pipeline.BIAS_ANALYSIS_AVAILABLE', False):
            
            pipeline = CognitiveEvaluationPipeline()
            
            result = pipeline.evaluate_response(
                test_id=self.test_data['id'],
                prompt=self.test_data['prompt'],
                response_text=self.test_response,
                test_metadata=self.test_data
            )
            
            # Should still return basic evaluation
            self.assertIsNotNone(result)
            self.assertIsInstance(result, CognitiveEvaluationResult)
            self.assertGreaterEqual(result.overall_score, 0)
            
            # Advanced features should be None/empty
            self.assertIsNone(result.behavioral_patterns)
            self.assertIsNone(result.cultural_analysis)
            self.assertIsNone(result.bias_indicators)


class TestEvaluatorPerformanceCharacteristics(unittest.TestCase):
    """Test performance characteristics of integrated evaluator framework"""
    
    def setUp(self):
        """Set up performance testing environment"""
        self.pipeline = CognitiveEvaluationPipeline()
        self.test_cases = [
            {
                'id': f'perf_test_{i}',
                'prompt': f'Performance test prompt {i} ' * 10,
                'domain': ['reasoning', 'creativity', 'social'][i % 3]
            }
            for i in range(20)  # 20 test cases for performance analysis
        ]
    
    def test_evaluation_throughput_performance(self):
        """Test evaluation throughput across multiple tests"""
        import time
        
        start_time = time.time()
        results = []
        
        for test_case in self.test_cases:
            response_text = f"Performance test response for {test_case['id']}"
            
            result = self.pipeline.evaluate_response(
                test_id=test_case['id'],
                prompt=test_case['prompt'],
                response_text=response_text,
                test_metadata=test_case
            )
            
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = len(self.test_cases) / total_time
        
        # Performance assertions
        self.assertEqual(len(results), len(self.test_cases))
        self.assertGreater(throughput, 2.0,  # Should process at least 2 evaluations per second
                          f"Throughput too low: {throughput:.2f} evaluations/second")
        
        print(f"✅ Evaluator Integration Performance:")
        print(f"   Total evaluations: {len(results)}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Throughput: {throughput:.2f} evaluations/second")
    
    def test_memory_usage_under_load(self):
        """Test memory usage during sustained evaluation"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many evaluations
        for i in range(50):
            test_data = {
                'id': f'memory_test_{i}',
                'prompt': f'Memory test prompt {i}',
                'domain': 'reasoning'
            }
            
            result = self.pipeline.evaluate_response(
                test_id=test_data['id'],
                prompt=test_data['prompt'],
                response_text=f"Memory test response {i}",
                test_metadata=test_data
            )
            
            self.assertIsNotNone(result)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        # Memory should not increase excessively
        self.assertLess(memory_increase, 50,  # Less than 50MB increase
                       f"Memory increase too high: {memory_increase:.2f}MB")
        
        print(f"✅ Memory Usage Test:")
        print(f"   Baseline memory: {baseline_memory:.2f}MB")
        print(f"   Final memory: {final_memory:.2f}MB")
        print(f"   Memory increase: {memory_increase:.2f}MB")


if __name__ == '__main__':
    print("🧪 Running Comprehensive Evaluator Framework Integration Tests")
    print("=" * 70)
    
    # Set up logging to capture integration behavior
    logging.basicConfig(level=logging.INFO)
    
    # Run tests with detailed output
    unittest.main(verbosity=2)