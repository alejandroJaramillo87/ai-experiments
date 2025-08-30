#!/usr/bin/env python3
"""
Basic Concurrency Robustness Tests

Tests that concurrent execution works reliably with multiple workers.
Keeps it simple - only tests basic concurrent execution without over-engineering.
"""

import unittest
import sys
import os
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor

# Add the benchmark_tests directory to Python path
benchmark_tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, benchmark_tests_dir)

from evaluator.subjects.reasoning_evaluator import UniversalEvaluator, evaluate_reasoning


class TestConcurrencyBasic(unittest.TestCase):
    """Test basic concurrent evaluation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = UniversalEvaluator()
        
        # Simple test responses for concurrent evaluation
        self.test_responses = [
            "This is a basic response for concurrent testing purposes.",
            "Another simple response to test concurrent evaluation.",
            "A third response to verify concurrent processing works.",
            "Final test response for concurrent execution validation."
        ]
    
    def evaluate_single_response(self, response_text):
        """Helper method to evaluate a single response"""
        try:
            result = evaluate_reasoning(
                response_text, 
                "Concurrent Test", 
                test_category="complex_synthesis"
            )
            return {
                'success': True,
                'score': result.metrics.overall_score,
                'word_count': result.metrics.word_count
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def test_concurrent_evaluation_basic(self):
        """Test that concurrent evaluation produces consistent results"""
        
        # Run evaluations concurrently with 2 workers (simple, not over-engineered)
        with ThreadPoolExecutor(max_workers=2) as executor:
            concurrent_results = list(executor.map(
                self.evaluate_single_response, 
                self.test_responses
            ))
        
        # Run evaluations sequentially for comparison
        sequential_results = [
            self.evaluate_single_response(response) 
            for response in self.test_responses
        ]
        
        # All evaluations should succeed
        for i, (concurrent, sequential) in enumerate(zip(concurrent_results, sequential_results)):
            self.assertTrue(concurrent['success'], 
                          f"Concurrent evaluation {i} should succeed")
            self.assertTrue(sequential['success'], 
                          f"Sequential evaluation {i} should succeed")
            
            # Scores should be identical (evaluation is deterministic)
            self.assertEqual(concurrent['score'], sequential['score'],
                           f"Concurrent and sequential scores should match for response {i}")
    
    def test_concurrent_evaluation_no_resource_conflicts(self):
        """Test that concurrent evaluations don't interfere with each other"""
        
        # Use the same response multiple times to test for resource conflicts
        same_response = "This is the same response text repeated for testing."
        responses = [same_response] * 3
        
        # Run concurrently with 3 workers
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(self.evaluate_single_response, responses))
        
        # All should succeed and have identical results
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertTrue(result['success'], f"Result {i} should succeed")
        
        # All scores should be identical since input is identical
        scores = [r['score'] for r in results]
        self.assertEqual(len(set(scores)), 1, "All identical inputs should produce identical scores")


if __name__ == "__main__":
    unittest.main(verbosity=2)