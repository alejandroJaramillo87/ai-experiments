"""
Unit Tests for Phase 1C Loop Classification Logic

Tests the three-category response classification system:
- _classify_loop_response_type()
- Classification accuracy for all three categories
"""

import unittest
from pathlib import Path

# Import the enhanced evaluator
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluator.subjects.enhanced_universal_evaluator import EnhancedUniversalEvaluator


class TestLoopResponseClassification(unittest.TestCase):
    """Test the _classify_loop_response_type() method"""
    
    def setUp(self):
        """Set up test environment"""
        self.evaluator = EnhancedUniversalEvaluator()
    
    def test_classify_clean_response(self):
        """Test classification of clean responses (no loops)"""
        # Test case 1: No coherence failure at all
        coherence_failure = None
        final_segment_analysis = {'recovery_detected': False}
        
        classification = self.evaluator._classify_loop_response_type(
            coherence_failure, final_segment_analysis
        )
        
        self.assertEqual(classification, "clean_response",
                        "No coherence failure should be classified as clean_response")
        
        # Test case 2: Coherence failure but not repetitive loop
        coherence_failure = {"failure_type": "incomplete_response"}
        final_segment_analysis = {'recovery_detected': False}
        
        classification = self.evaluator._classify_loop_response_type(
            coherence_failure, final_segment_analysis
        )
        
        self.assertEqual(classification, "clean_response",
                        "Non-loop coherence failure should be classified as clean_response")
        
        # Test case 3: Empty coherence failure
        coherence_failure = {}
        final_segment_analysis = {'recovery_detected': False}
        
        classification = self.evaluator._classify_loop_response_type(
            coherence_failure, final_segment_analysis
        )
        
        self.assertEqual(classification, "clean_response",
                        "Empty coherence failure should be classified as clean_response")
        
        print("✅ Clean response classification working correctly")
    
    def test_classify_loop_with_recovery(self):
        """Test classification of loop-with-recovery responses"""
        # Test case: Repetitive loop with recovery detected
        coherence_failure = {"failure_type": "repetitive_loop"}
        final_segment_analysis = {'recovery_detected': True}
        
        classification = self.evaluator._classify_loop_response_type(
            coherence_failure, final_segment_analysis
        )
        
        self.assertEqual(classification, "loop_with_recovery",
                        "Repetitive loop with recovery should be classified as loop_with_recovery")
        
        print("✅ Loop-with-recovery classification working correctly")
    
    def test_classify_pure_cognitive_failure(self):
        """Test classification of pure cognitive failure responses"""
        # Test case: Repetitive loop without recovery
        coherence_failure = {"failure_type": "repetitive_loop"}
        final_segment_analysis = {'recovery_detected': False}
        
        classification = self.evaluator._classify_loop_response_type(
            coherence_failure, final_segment_analysis
        )
        
        self.assertEqual(classification, "pure_cognitive_failure",
                        "Repetitive loop without recovery should be classified as pure_cognitive_failure")
        
        print("✅ Pure cognitive failure classification working correctly")
    
    def test_classification_comprehensive_scenarios(self):
        """Test comprehensive classification scenarios"""
        # Test scenarios covering all combinations
        test_scenarios = [
            # (coherence_failure, recovery_detected, expected_classification)
            (None, False, "clean_response"),
            (None, True, "clean_response"),  # Recovery irrelevant if no loops
            ({"failure_type": "incomplete_response"}, False, "clean_response"),
            ({"failure_type": "incomplete_response"}, True, "clean_response"),
            ({"failure_type": "repetitive_loop"}, True, "loop_with_recovery"),
            ({"failure_type": "repetitive_loop"}, False, "pure_cognitive_failure"),
        ]
        
        print("\n🧪 Comprehensive Classification Test:")
        for i, (coherence_failure, recovery_detected, expected) in enumerate(test_scenarios, 1):
            with self.subTest(scenario=i):
                final_segment_analysis = {'recovery_detected': recovery_detected}
                
                classification = self.evaluator._classify_loop_response_type(
                    coherence_failure, final_segment_analysis
                )
                
                self.assertEqual(classification, expected,
                               f"Scenario {i}: Expected {expected}, got {classification}")
                
                failure_type = coherence_failure.get("failure_type") if coherence_failure else "None"
                print(f"   Scenario {i}: failure_type='{failure_type}', "
                      f"recovery={recovery_detected} → {classification}")
        
        print("✅ All classification scenarios working correctly")


class TestClassificationIntegration(unittest.TestCase):
    """Test integration of classification with actual response patterns"""
    
    def setUp(self):
        """Set up test environment"""
        self.evaluator = EnhancedUniversalEvaluator()
    
    def test_end_to_end_classification_basic_08_pattern(self):
        """Test end-to-end classification for basic_08 pattern (loop-with-recovery)"""
        # Simulate basic_08 response pattern: loops → quality recovery
        basic_08_pattern = '''
        We must respond to the final answer as per instructions...
        Let me think about this more...
        Actually, let me reconsider...
        We need to fill blanks with appropriate Yoruba oriki style...
        
        **Step‑by‑step reasoning**
        
        1. **Identify the pattern**  
           Each line of the oriki adds a new attribute or metaphor.
        
        2. **Choose attributes that fit**  
           Protector, Guardian, Strong one with cultural significance.
        
        **Completed oriki**
        
        > Protector of the village's children,  
        > Guardian who keeps the river's current steady,  
        > Strong one whose roots run as deep as the baobab.
        '''
        
        # Analyze final segment
        final_segment_analysis = self.evaluator._analyze_final_segment_quality(basic_08_pattern)
        
        # Simulate coherence failure detection (would come from reasoning evaluator)
        coherence_failure = {"failure_type": "repetitive_loop"}  # Meta-reasoning loops detected
        
        # Classify response type
        classification = self.evaluator._classify_loop_response_type(
            coherence_failure, final_segment_analysis
        )
        
        # Should be classified as loop-with-recovery
        self.assertEqual(classification, "loop_with_recovery",
                        "basic_08 pattern should be classified as loop_with_recovery")
        
        # Verify recovery was detected in final segment
        self.assertTrue(final_segment_analysis.get('recovery_detected', False),
                       "basic_08 pattern should show recovery in final segment")
        
        print(f"✅ basic_08 end-to-end: {classification}, "
              f"recovery_detected={final_segment_analysis.get('recovery_detected')}")
    
    def test_end_to_end_classification_math_04_pattern(self):
        """Test end-to-end classification for math_04 pattern (pure cognitive failure)"""
        # Simulate math_04 response pattern: pure repetitive loops
        math_04_pattern = '''
        Compute base - xy? That yields base - xy. But we need to compute base^2 - base*(x+y) + xy. But we can compute as:
        Compute base - xy? That yields base - xy. But we need to compute base^2 - base*(x+y) + xy. But we can compute as:
        Compute base - xy? That yields base - xy. But we need to compute base^2 - base*(x+y) + xy. But we can compute as:
        Compute base - xy? That yields base - xy. But we need to compute base^2 - base*(x+y) + xy. But we can compute as:
        Compute base - xy? That yields base - xy. But we need to compute base^2 - base*(x+y) + xy. But we can compute as:
        '''
        
        # Analyze final segment
        final_segment_analysis = self.evaluator._analyze_final_segment_quality(math_04_pattern)
        
        # Simulate coherence failure detection
        coherence_failure = {"failure_type": "repetitive_loop"}  # Severe loops detected
        
        # Classify response type
        classification = self.evaluator._classify_loop_response_type(
            coherence_failure, final_segment_analysis
        )
        
        # Should be classified as pure cognitive failure
        self.assertEqual(classification, "pure_cognitive_failure",
                        "math_04 pattern should be classified as pure_cognitive_failure")
        
        # Verify no recovery was detected
        self.assertFalse(final_segment_analysis.get('recovery_detected', False),
                        "math_04 pattern should not show recovery")
        
        print(f"✅ math_04 end-to-end: {classification}, "
              f"recovery_detected={final_segment_analysis.get('recovery_detected')}")
    
    def test_end_to_end_classification_clean_response_pattern(self):
        """Test end-to-end classification for clean response pattern"""
        # Simulate clean mathematical response
        clean_pattern = '''
        To solve this Sangaku geometry problem, I'll analyze the triangle configuration.
        
        Given the circle inscribed in triangle ABC with radius r, and the specific measurements,
        I can apply the relationship between the inradius and the triangle's sides.
        
        Using the formula: A = rs, where A is area, r is inradius, s is semiperimeter:
        - Semiperimeter s = (a + b + c)/2 = (8 + 15 + 17)/2 = 20
        - Area using Heron's formula: A = √[s(s-a)(s-b)(s-c)] = √[20×12×5×3] = 60
        - Therefore: r = A/s = 60/20 = 3
        
        The inradius of the triangle is 3 units.
        '''
        
        # Analyze final segment
        final_segment_analysis = self.evaluator._analyze_final_segment_quality(clean_pattern)
        
        # No coherence failure for clean response
        coherence_failure = None
        
        # Classify response type
        classification = self.evaluator._classify_loop_response_type(
            coherence_failure, final_segment_analysis
        )
        
        # Should be classified as clean response
        self.assertEqual(classification, "clean_response",
                        "Clean mathematical response should be classified as clean_response")
        
        print(f"✅ Clean response end-to-end: {classification}")


class TestClassificationEdgeCases(unittest.TestCase):
    """Test edge cases for loop classification"""
    
    def setUp(self):
        """Set up test environment"""
        self.evaluator = EnhancedUniversalEvaluator()
    
    def test_classification_edge_cases(self):
        """Test various edge cases for classification"""
        # Edge case 1: Recovery detected but no loops (should be clean)
        coherence_failure = None
        high_quality_analysis = {
            'recovery_detected': True,
            'quality_score': 85.0,
            'has_structure': True
        }
        
        classification = self.evaluator._classify_loop_response_type(
            coherence_failure, high_quality_analysis
        )
        
        self.assertEqual(classification, "clean_response",
                        "High quality without loops should be clean_response regardless of recovery flag")
        
        # Edge case 2: Loops with marginal recovery (right at threshold)
        coherence_failure = {"failure_type": "repetitive_loop"}
        marginal_analysis = {
            'recovery_detected': True,  # Just barely meets threshold
            'quality_score': 70.1,
            'has_structure': True
        }
        
        classification = self.evaluator._classify_loop_response_type(
            coherence_failure, marginal_analysis
        )
        
        self.assertEqual(classification, "loop_with_recovery",
                        "Marginal recovery should still be classified as loop_with_recovery")
        
        # Edge case 3: Loops with quality just below threshold
        coherence_failure = {"failure_type": "repetitive_loop"}
        below_threshold_analysis = {
            'recovery_detected': False,  # Below recovery threshold
            'quality_score': 69.9,
            'has_structure': False
        }
        
        classification = self.evaluator._classify_loop_response_type(
            coherence_failure, below_threshold_analysis
        )
        
        self.assertEqual(classification, "pure_cognitive_failure",
                        "Quality below threshold should be pure_cognitive_failure")
        
        print("✅ Edge cases handled correctly")
    
    def test_classification_robustness(self):
        """Test classification robustness with malformed inputs"""
        # Test with missing keys in analysis
        coherence_failure = {"failure_type": "repetitive_loop"}
        incomplete_analysis = {}  # Missing recovery_detected key
        
        # Should not crash and should default to pure_cognitive_failure
        classification = self.evaluator._classify_loop_response_type(
            coherence_failure, incomplete_analysis
        )
        
        self.assertEqual(classification, "pure_cognitive_failure",
                        "Missing recovery key should default to pure_cognitive_failure")
        
        # Test with unexpected failure types
        weird_failure = {"failure_type": "unknown_failure_type"}
        analysis = {'recovery_detected': True}
        
        classification = self.evaluator._classify_loop_response_type(
            weird_failure, analysis
        )
        
        self.assertEqual(classification, "clean_response",
                        "Unknown failure types should be treated as clean_response")
        
        print("✅ Robustness tests passed")


if __name__ == '__main__':
    # Run the tests with verbose output
    unittest.main(verbosity=2)