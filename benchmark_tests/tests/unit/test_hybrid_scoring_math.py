"""
Unit Tests for Phase 1C Hybrid Scoring Mathematics

Tests the _apply_loop_recovery_scoring() method and scoring calculations:
- Pure cognitive failure scoring (≤10)
- Loop-with-recovery scoring (segment quality - efficiency penalty) 
- Clean response scoring (unchanged)
- Scoring mathematics validation
"""

import unittest
from unittest.mock import Mock
from pathlib import Path

# Import the enhanced evaluator
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluator.subjects.enhanced_universal_evaluator import EnhancedUniversalEvaluator


class TestHybridScoringMath(unittest.TestCase):
    """Test the _apply_loop_recovery_scoring() method"""
    
    def setUp(self):
        """Set up test environment"""
        self.evaluator = EnhancedUniversalEvaluator()
    
    def test_pure_cognitive_failure_scoring(self):
        """Test scoring for pure cognitive failure responses"""
        # Mock enhanced metrics with various initial scores
        test_scores = [95.0, 75.0, 50.0, 25.0, 5.0]
        
        for initial_score in test_scores:
            with self.subTest(initial_score=initial_score):
                # Create mock metrics object
                mock_metrics = Mock()
                mock_metrics.overall_score = initial_score
                
                # Apply pure cognitive failure scoring
                self.evaluator._apply_loop_recovery_scoring(
                    enhanced_metrics=mock_metrics,
                    loop_type="pure_cognitive_failure",
                    final_segment_analysis={'quality_score': 0.0},
                    test_name="test_case"
                )
                
                # Should be capped at 10.0 regardless of initial score
                self.assertLessEqual(mock_metrics.overall_score, 10.0,
                                   f"Pure cognitive failure should cap score at 10.0, got {mock_metrics.overall_score}")
                
                # Should not increase score if initially lower
                expected_score = min(initial_score, 10.0)
                self.assertEqual(mock_metrics.overall_score, expected_score,
                               f"Score should be min({initial_score}, 10.0) = {expected_score}")
        
        print("✅ Pure cognitive failure scoring: All scores correctly capped at ≤10.0")
    
    def test_loop_with_recovery_scoring_math(self):
        """Test scoring mathematics for loop-with-recovery responses"""
        # Test cases with different segment quality scores
        test_cases = [
            {'segment_quality': 95.0, 'expected_range': (83.0, 85.0)},  # 95 - 12 = 83
            {'segment_quality': 85.0, 'expected_range': (73.0, 75.0)},  # 85 - 12 = 73
            {'segment_quality': 75.0, 'expected_range': (63.0, 65.0)},  # 75 - 12 = 63
            {'segment_quality': 50.0, 'expected_range': (38.0, 40.0)},  # 50 - 12 = 38
            {'segment_quality': 25.0, 'expected_range': (15.0, 15.0)},  # 25 - 12 = 13, floored at 15
            {'segment_quality': 10.0, 'expected_range': (15.0, 15.0)},  # 10 - 12 = -2, floored at 15
        ]
        
        print("🧮 Loop-with-recovery scoring mathematics:")
        for case in test_cases:
            segment_quality = case['segment_quality']
            expected_min, expected_max = case['expected_range']
            
            with self.subTest(segment_quality=segment_quality):
                # Create mock metrics (initial score shouldn't matter for recovery scoring)
                mock_metrics = Mock()
                mock_metrics.overall_score = 100.0  # Will be replaced
                
                # Mock final segment analysis
                final_segment_analysis = {'quality_score': segment_quality}
                
                # Apply loop-with-recovery scoring
                self.evaluator._apply_loop_recovery_scoring(
                    enhanced_metrics=mock_metrics,
                    loop_type="loop_with_recovery",
                    final_segment_analysis=final_segment_analysis,
                    test_name="test_case"
                )
                
                # Verify scoring mathematics
                self.assertGreaterEqual(mock_metrics.overall_score, expected_min,
                                      f"Score should be ≥{expected_min} for segment quality {segment_quality}")
                self.assertLessEqual(mock_metrics.overall_score, expected_max,
                                   f"Score should be ≤{expected_max} for segment quality {segment_quality}")
                
                print(f"   Segment quality {segment_quality:5.1f} → Final score {mock_metrics.overall_score:5.1f}")
        
        print("✅ Loop-with-recovery scoring mathematics working correctly")
    
    def test_clean_response_scoring_unchanged(self):
        """Test that clean responses maintain their original scoring"""
        # Test various initial scores for clean responses
        test_scores = [95.0, 85.0, 75.0, 65.0, 45.0]
        
        for initial_score in test_scores:
            with self.subTest(initial_score=initial_score):
                # Create mock metrics
                mock_metrics = Mock()
                mock_metrics.overall_score = initial_score
                
                # Apply clean response scoring (should be unchanged)
                self.evaluator._apply_loop_recovery_scoring(
                    enhanced_metrics=mock_metrics,
                    loop_type="clean_response",
                    final_segment_analysis={'quality_score': 80.0},  # Shouldn't matter
                    test_name="test_case"
                )
                
                # Score should remain unchanged
                self.assertEqual(mock_metrics.overall_score, initial_score,
                               f"Clean response score should remain unchanged at {initial_score}")
        
        print("✅ Clean response scoring: All scores remain unchanged")
    
    def test_efficiency_penalty_calculation(self):
        """Test the efficiency penalty calculation specifically"""
        EFFICIENCY_PENALTY = 12.0  # From implementation
        
        # Test that penalty is applied consistently
        segment_qualities = [90.0, 80.0, 70.0, 60.0, 50.0]
        
        print(f"🎯 Efficiency penalty validation (penalty = {EFFICIENCY_PENALTY}):")
        for quality in segment_qualities:
            mock_metrics = Mock()
            mock_metrics.overall_score = 100.0
            
            final_segment_analysis = {'quality_score': quality}
            
            self.evaluator._apply_loop_recovery_scoring(
                mock_metrics, "loop_with_recovery", final_segment_analysis, "test"
            )
            
            expected_before_floor = quality - EFFICIENCY_PENALTY
            expected_after_floor = max(expected_before_floor, 15.0)
            
            self.assertEqual(mock_metrics.overall_score, expected_after_floor,
                           f"Quality {quality} - penalty {EFFICIENCY_PENALTY} should = {expected_after_floor}")
            
            print(f"   Quality {quality:5.1f} - penalty {EFFICIENCY_PENALTY:4.1f} = {expected_before_floor:5.1f} → {expected_after_floor:5.1f}")
        
        print("✅ Efficiency penalty calculation validated")
    
    def test_minimum_score_floor(self):
        """Test the minimum score floor for loop-with-recovery (15.0)"""
        MIN_RECOVERY_SCORE = 15.0
        
        # Test very low segment quality scores
        low_qualities = [20.0, 15.0, 10.0, 5.0, 0.0]
        
        print(f"🔻 Minimum score floor validation (floor = {MIN_RECOVERY_SCORE}):")
        for quality in low_qualities:
            mock_metrics = Mock()
            mock_metrics.overall_score = 100.0
            
            final_segment_analysis = {'quality_score': quality}
            
            self.evaluator._apply_loop_recovery_scoring(
                mock_metrics, "loop_with_recovery", final_segment_analysis, "test"
            )
            
            # Should never go below minimum floor
            self.assertGreaterEqual(mock_metrics.overall_score, MIN_RECOVERY_SCORE,
                                  f"Score should never go below {MIN_RECOVERY_SCORE}, got {mock_metrics.overall_score}")
            
            # If calculation would be below floor, should equal floor
            calculated = quality - 12.0
            expected = max(calculated, MIN_RECOVERY_SCORE)
            self.assertEqual(mock_metrics.overall_score, expected,
                           f"Score should be max({calculated:.1f}, {MIN_RECOVERY_SCORE}) = {expected}")
            
            print(f"   Quality {quality:5.1f} → Score {mock_metrics.overall_score:5.1f} (floored at {MIN_RECOVERY_SCORE})")
        
        print("✅ Minimum score floor working correctly")


class TestScoringBoundaryConditions(unittest.TestCase):
    """Test boundary conditions and edge cases for scoring"""
    
    def setUp(self):
        """Set up test environment"""
        self.evaluator = EnhancedUniversalEvaluator()
    
    def test_scoring_boundary_conditions(self):
        """Test boundary conditions for all scoring types"""
        # Test maximum possible scores don't exceed 100
        mock_metrics = Mock()
        mock_metrics.overall_score = 100.0
        
        # Test loop-with-recovery with very high segment quality
        final_segment_analysis = {'quality_score': 200.0}  # Unrealistically high
        
        self.evaluator._apply_loop_recovery_scoring(
            mock_metrics, "loop_with_recovery", final_segment_analysis, "boundary_test"
        )
        
        # Should not exceed 100 (implementation uses min(recovery_score, 100.0))
        self.assertLessEqual(mock_metrics.overall_score, 100.0,
                           "Score should never exceed 100.0")
        
        print(f"✅ Maximum score boundary: Capped at {mock_metrics.overall_score}")
        
        # Test that negative segment qualities are handled gracefully
        mock_metrics.overall_score = 50.0
        final_segment_analysis = {'quality_score': -10.0}  # Negative quality
        
        self.evaluator._apply_loop_recovery_scoring(
            mock_metrics, "loop_with_recovery", final_segment_analysis, "negative_test"
        )
        
        # Should still respect minimum floor
        self.assertGreaterEqual(mock_metrics.overall_score, 15.0,
                              "Negative segment quality should still respect minimum floor")
        
        print(f"✅ Negative quality handling: Score {mock_metrics.overall_score}")
    
    def test_score_precision(self):
        """Test that scores maintain reasonable precision"""
        # Test with decimal segment qualities
        decimal_qualities = [87.3, 74.7, 61.2, 48.9]
        
        print("🔢 Score precision validation:")
        for quality in decimal_qualities:
            mock_metrics = Mock()
            mock_metrics.overall_score = 100.0
            
            final_segment_analysis = {'quality_score': quality}
            
            self.evaluator._apply_loop_recovery_scoring(
                mock_metrics, "loop_with_recovery", final_segment_analysis, "precision_test"
            )
            
            expected = max(quality - 12.0, 15.0)
            
            # Allow for floating point precision
            self.assertAlmostEqual(mock_metrics.overall_score, expected, places=1,
                                 msg=f"Score precision issue: {quality} → {mock_metrics.overall_score}, expected {expected}")
            
            print(f"   Quality {quality:5.1f} → Score {mock_metrics.overall_score:5.1f}")
        
        print("✅ Score precision maintained")


class TestScoringIntegrationScenarios(unittest.TestCase):
    """Test scoring integration with realistic scenarios"""
    
    def setUp(self):
        """Set up test environment"""
        self.evaluator = EnhancedUniversalEvaluator()
    
    def test_basic_08_scoring_scenario(self):
        """Test scoring scenario matching basic_08 (expected ~88.0)"""
        # basic_08 achieved 88.0 score in actual results
        # Reverse-engineer what segment quality would produce this score
        
        # If final score is 88.0 and efficiency penalty is 12.0:
        # segment_quality - 12.0 = 88.0
        # segment_quality = 100.0
        
        mock_metrics = Mock()
        mock_metrics.overall_score = 50.0  # Initial score (will be replaced)
        
        final_segment_analysis = {'quality_score': 100.0}  # Perfect segment quality
        
        self.evaluator._apply_loop_recovery_scoring(
            mock_metrics, "loop_with_recovery", final_segment_analysis, "basic_08_simulation"
        )
        
        # Should produce score close to 88.0
        expected_score = 100.0 - 12.0  # 88.0
        self.assertEqual(mock_metrics.overall_score, expected_score,
                        f"basic_08 scenario should produce {expected_score}")
        
        print(f"✅ basic_08 scenario: Segment quality 100.0 → Final score {mock_metrics.overall_score}")
    
    def test_math_04_scoring_scenario(self):
        """Test scoring scenario matching math_04 (expected ≤10.0)"""
        # math_04 maintained 10.0 score (pure cognitive failure)
        
        initial_scores = [95.0, 75.0, 50.0, 25.0, 10.0, 5.0]
        
        print("🧮 math_04 scenario validation:")
        for initial_score in initial_scores:
            mock_metrics = Mock()
            mock_metrics.overall_score = initial_score
            
            # Pure cognitive failure should cap at 10.0 regardless of initial score
            self.evaluator._apply_loop_recovery_scoring(
                mock_metrics, "pure_cognitive_failure", 
                {'quality_score': 0.0}, "math_04_simulation"
            )
            
            expected = min(initial_score, 10.0)
            self.assertEqual(mock_metrics.overall_score, expected,
                           f"math_04 scenario: {initial_score} → {expected}")
            
            print(f"   Initial {initial_score:5.1f} → Final {mock_metrics.overall_score:5.1f}")
        
        print("✅ math_04 scenario: All scores correctly capped")
    
    def test_scoring_distribution_analysis(self):
        """Analyze the scoring distribution across different scenarios"""
        scenarios = []
        
        # Generate various scoring scenarios
        segment_qualities = range(0, 101, 10)  # 0, 10, 20, ..., 100
        
        print("📊 Scoring Distribution Analysis:")
        print("   Segment Quality → Recovery Score")
        
        for quality in segment_qualities:
            mock_metrics = Mock()
            mock_metrics.overall_score = 100.0
            
            final_segment_analysis = {'quality_score': float(quality)}
            
            self.evaluator._apply_loop_recovery_scoring(
                mock_metrics, "loop_with_recovery", final_segment_analysis, "distribution_test"
            )
            
            scenarios.append((quality, mock_metrics.overall_score))
            print(f"   {quality:3d} → {mock_metrics.overall_score:5.1f}")
        
        # Verify scoring is monotonic (higher segment quality → higher or equal final score)
        for i in range(1, len(scenarios)):
            prev_segment, prev_score = scenarios[i-1]
            curr_segment, curr_score = scenarios[i]
            
            self.assertGreaterEqual(curr_score, prev_score,
                                  f"Scoring should be monotonic: {prev_segment}→{prev_score} vs {curr_segment}→{curr_score}")
        
        print("✅ Scoring distribution is monotonic and well-behaved")


if __name__ == '__main__':
    # Run the tests with verbose output
    unittest.main(verbosity=2)