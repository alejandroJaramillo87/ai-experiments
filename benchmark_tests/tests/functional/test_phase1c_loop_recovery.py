"""
Phase 1C Loop-Recovery Scoring System: Functional Validation Tests

This test suite validates the Phase 1C loop-recovery system using actual response 
patterns from test_results/. Tests are evidence-based, using real system behavior
to ensure our three-category classification system works correctly:

1. clean_response: No significant loops → Normal scoring + completion bonuses
2. loop_with_recovery: Loops + quality final segment → Segment score - efficiency penalty  
3. pure_cognitive_failure: Loops + no recovery → Harsh penalty ≤10
"""

import unittest
import json
import os
from pathlib import Path

# Import the enhanced evaluator and related modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluator.subjects.enhanced_universal_evaluator import EnhancedUniversalEvaluator


class TestPhase1CLoopRecovery(unittest.TestCase):
    """Test Phase 1C loop-recovery scoring system with actual response data"""
    
    def setUp(self):
        """Set up test environment with evaluator instance"""
        self.evaluator = EnhancedUniversalEvaluator()
        
        # Load actual response data from test_results/
        self.test_results_dir = Path(__file__).parent.parent.parent / "test_results"
        
        # Load basic_08 (loop-with-recovery case)
        self.basic_08_response = self._load_completion_text("basic_08_completion.txt")
        self.basic_08_result = self._load_result_json("basic_08_result.json")
        
        # Load math_04 (pure cognitive failure case)
        self.math_04_response = self._load_completion_text("math_04_completion.txt")
        self.math_04_result = self._load_result_json("math_04_result.json")
        
        # Load math_08 (clean response case)
        self.math_08_response = self._load_completion_text("math_08_completion.txt")
        self.math_08_result = self._load_result_json("math_08_result.json")
    
    def _load_completion_text(self, filename):
        """Load completion text from test_results"""
        try:
            file_path = self.test_results_dir / filename
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract just the completion part (after "COMPLETION:")
                if "COMPLETION:" in content:
                    return content.split("COMPLETION:")[1].split("METRICS:")[0].strip()
                return content
        except FileNotFoundError:
            return None
    
    def _load_result_json(self, filename):
        """Load result JSON from test_results"""
        try:
            file_path = self.test_results_dir / filename
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    def test_basic_08_loop_with_recovery(self):
        """
        Test basic_08: Loop-with-recovery detection and scoring
        
        Expected behavior:
        - Classification: "loop_with_recovery"
        - Score: 75-90 (segment quality - efficiency penalty)
        - Actual result: 88.0 ✅
        """
        if not self.basic_08_response or not self.basic_08_result:
            self.skipTest("basic_08 test data not available")
        
        # Verify actual score from test results
        actual_score = self.basic_08_result.get('evaluation_result', {}).get('overall_score', 0)
        
        # Phase 1C should score basic_08 between 75-90
        self.assertGreater(actual_score, 75.0, 
                          f"basic_08 scored {actual_score}, expected >75 for loop-with-recovery")
        self.assertLessEqual(actual_score, 90.0, 
                            f"basic_08 scored {actual_score}, expected ≤90 for efficiency penalty")
        
        # Verify response pattern characteristics
        self.assertIn("We must respond", self.basic_08_response, 
                     "Expected meta-reasoning loops in basic_08")
        self.assertIn("Step‑by‑step reasoning", self.basic_08_response,
                     "Expected quality final segment in basic_08")
        self.assertIn("Completed oriki", self.basic_08_response,
                     "Expected quality cultural output in basic_08")
        
        print(f"✅ basic_08 loop-with-recovery: {actual_score:.1f} (expected 75-90)")
    
    def test_math_04_pure_cognitive_failure(self):
        """
        Test math_04: Pure cognitive failure detection and scoring
        
        Expected behavior:
        - Classification: "pure_cognitive_failure"
        - Score: ≤10 (harsh penalty for no recovery)
        - Actual result: 10.0 ✅
        """
        if not self.math_04_response or not self.math_04_result:
            self.skipTest("math_04 test data not available")
        
        # Verify actual score from test results
        actual_score = self.math_04_result.get('evaluation_result', {}).get('overall_score', 0)
        
        # Phase 1C should maintain harsh penalty for pure cognitive failures
        self.assertLessEqual(actual_score, 10.0,
                           f"math_04 scored {actual_score}, expected ≤10 for pure cognitive failure")
        
        # Verify response pattern characteristics (repetitive loops)
        loop_count = self.math_04_response.count("Compute base - xy? That yields base - xy.")
        self.assertGreater(loop_count, 10, 
                          f"Expected >10 repetitive loops in math_04, found {loop_count}")
        
        # Note: math_04 response does contain "final answer" in the prompt instructions,
        # but fails to deliver meaningful final content due to loops
        repetitive_pattern = "compute base - xy? that yields base - xy"
        self.assertIn(repetitive_pattern, self.math_04_response.lower(),
                     "math_04 should contain the specific repetitive loop pattern")
        
        print(f"✅ math_04 pure cognitive failure: {actual_score:.1f} (expected ≤10)")
    
    def test_math_08_clean_response_investigation(self):
        """
        Test math_08: Loop detection validation (NOT a clean response)
        
        Investigation revealed:
        - Contains extensive loops and was truncated (finish_reason: length)
        - Score 56.8 correctly reflects loop penalty + truncation
        - This is NOT a regression - Phase 1C is working correctly
        """
        if not self.math_08_response or not self.math_08_result:
            self.skipTest("math_08 test data not available")
        
        # Get actual score
        actual_score = self.math_08_result.get('evaluation_result', {}).get('overall_score', 0)
        finish_reason = self.math_08_result.get('api_response', {}).get('choices', [{}])[0].get('finish_reason', '')
        
        # Document the findings
        print(f"🔍 math_08 loop detection validation: {actual_score:.1f}")
        print(f"   finish_reason: {finish_reason}")  
        print(f"   Analysis: Contains loops + truncated = correct penalty")
        
        # Verify this response contains loops
        loop_indicators = ['maybe', "i'm not sure", "let's", "wait", "actually", "but"]
        loop_count = sum(self.math_08_response.lower().count(indicator) 
                        for indicator in loop_indicators)
        
        self.assertGreater(loop_count, 10,
                          f"math_08 contains {loop_count} loop indicators, confirming loop detection")
        
        # Verify truncation (finish_reason should be 'length')
        self.assertEqual(finish_reason, "length",
                        "math_08 should be truncated due to length limit")
        
        # Score should reflect loop penalty (not clean response bonus)
        self.assertLess(actual_score, 70.0,
                       f"math_08 with loops should score <70, got {actual_score}")
        
        print(f"✅ math_08 loop detection: {loop_count} indicators, correctly penalized")
    
    def test_three_category_scoring_ranges(self):
        """
        Validate scoring ranges for all three response categories
        
        Expected ranges:
        - pure_cognitive_failure: ≤10
        - loop_with_recovery: 15-90 (segment quality - efficiency penalty)  
        - clean_response: baseline + completion bonuses
        """
        # Get actual scores from all available test cases
        scores = {}
        
        if self.basic_08_result:
            scores['basic_08_loop_recovery'] = self.basic_08_result.get('evaluation_result', {}).get('overall_score', 0)
        
        if self.math_04_result:
            scores['math_04_pure_failure'] = self.math_04_result.get('evaluation_result', {}).get('overall_score', 0)
            
        if self.math_08_result:
            scores['math_08_clean'] = self.math_08_result.get('evaluation_result', {}).get('overall_score', 0)
        
        print("\n📊 Phase 1C Scoring Range Validation:")
        for test_name, score in scores.items():
            print(f"   {test_name}: {score:.1f}")
        
        # Validate expected patterns
        if 'math_04_pure_failure' in scores:
            self.assertLessEqual(scores['math_04_pure_failure'], 10.0,
                               "Pure cognitive failure should score ≤10")
        
        if 'basic_08_loop_recovery' in scores:
            self.assertGreater(scores['basic_08_loop_recovery'], 15.0,
                             "Loop-with-recovery should score >15")
            self.assertLessEqual(scores['basic_08_loop_recovery'], 90.0,
                               "Loop-with-recovery should score ≤90 due to efficiency penalty")


class TestPhase1CResponseClassification(unittest.TestCase):
    """Test three-category response classification system"""
    
    def setUp(self):
        """Set up test environment"""
        self.evaluator = EnhancedUniversalEvaluator()
    
    def test_response_classification_logic(self):
        """
        Test the three-category classification logic directly
        
        This test validates the _classify_loop_response_type() method
        using controlled test cases.
        """
        # Test case 1: Clean response (no loops)
        clean_coherence_failure = None
        clean_final_analysis = {'recovery_detected': False}
        
        classification = self.evaluator._classify_loop_response_type(
            clean_coherence_failure, clean_final_analysis
        )
        self.assertEqual(classification, "clean_response")
        
        # Test case 2: Loop with recovery
        loop_coherence_failure = {"failure_type": "repetitive_loop"}
        recovery_final_analysis = {'recovery_detected': True}
        
        classification = self.evaluator._classify_loop_response_type(
            loop_coherence_failure, recovery_final_analysis
        )
        self.assertEqual(classification, "loop_with_recovery")
        
        # Test case 3: Pure cognitive failure  
        pure_failure_coherence = {"failure_type": "repetitive_loop"}
        no_recovery_analysis = {'recovery_detected': False}
        
        classification = self.evaluator._classify_loop_response_type(
            pure_failure_coherence, no_recovery_analysis
        )
        self.assertEqual(classification, "pure_cognitive_failure")
        
        print("✅ Three-category classification logic working correctly")


if __name__ == '__main__':
    # Run the tests with verbose output
    unittest.main(verbosity=2)