"""
Phase 1C Loop-Recovery Calibration Tests

This test suite investigates specific calibration issues discovered in Phase 1C:
1. math_08 regression: Clean response scoring lower (64.8 → 56.8)
2. basic_03 unexpected recovery: Loop response now scoring 88.0 (was ≤10)

Uses actual response data from test_results/ for evidence-based debugging.
"""

import unittest
import json
from pathlib import Path
from unittest.mock import Mock

# Import the enhanced evaluator and related modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluator.subjects.enhanced_universal_evaluator import EnhancedUniversalEvaluator


class TestMath08CleanResponseRegression(unittest.TestCase):
    """Investigate math_08 clean response scoring regression"""
    
    def setUp(self):
        """Set up test environment"""
        self.evaluator = EnhancedUniversalEvaluator()
        self.test_results_dir = Path(__file__).parent.parent.parent / "test_results"
        
        # Load math_08 data
        self.math_08_response = self._load_completion_text("math_08_completion.txt")
        self.math_08_result = self._load_result_json("math_08_result.json")
    
    def _load_completion_text(self, filename):
        """Load completion text from test_results"""
        try:
            file_path = self.test_results_dir / filename
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
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
    
    def test_math_08_completion_bonus_logic(self):
        """
        Test completion bonus application for length-limited responses
        
        Issue: math_08 finish_reason="length" but should still get appropriate scoring
        Question: Should clean responses get bonuses even with finish_reason="length"?
        """
        if not self.math_08_result:
            self.skipTest("math_08 test data not available")
        
        # Analyze math_08 completion characteristics
        actual_score = self.math_08_result.get('score', 0)
        finish_reason = self.math_08_result.get('finish_reason', '')
        completion_tokens = self.math_08_result.get('completion_tokens', 0)
        
        print(f"🔍 math_08 Completion Bonus Analysis:")
        print(f"   Score: {actual_score:.1f} (was 64.8 before Phase 1C)")
        print(f"   finish_reason: '{finish_reason}'")
        print(f"   completion_tokens: {completion_tokens}")
        
        # Check if this is truly a clean response (no loops)
        if self.math_08_response:
            loop_indicators = ['maybe', "i'm not sure", "let's think", "wait", "actually"]
            loop_count = sum(self.math_08_response.lower().count(indicator) 
                            for indicator in loop_indicators)
            print(f"   loop_indicators: {loop_count} (should be minimal for clean response)")
            
            # Check for mathematical quality indicators
            math_indicators = ['calculate', 'solve', 'equation', 'answer', 'result', '=']
            math_count = sum(1 for indicator in math_indicators 
                           if indicator in self.math_08_response.lower())
            print(f"   math_indicators: {math_count} (should be high for quality math)")
        
        # Analysis: Should length-limited clean responses get completion bonuses?
        if finish_reason == "length" and actual_score < 65.0:
            print(f"⚠️  ISSUE: Clean mathematical response with finish_reason='length'")
            print(f"   Expected: Completion bonuses should focus on content quality, not just finish_reason='stop'")
            print(f"   Recommendation: Modify completion bonus logic for high-quality truncated responses")
    
    def test_math_08_baseline_scoring_comparison(self):
        """
        Compare math_08 scoring before/after Phase 1C changes
        
        Isolate whether Phase 1C caused the regression by examining scoring components
        """
        if not self.math_08_result:
            self.skipTest("math_08 test data not available")
        
        # Document the regression
        previous_score = 64.8  # Score before Phase 1C
        current_score = self.math_08_result.get('score', 0)
        score_change = current_score - previous_score
        
        print(f"📊 math_08 Baseline Scoring Comparison:")
        print(f"   Before Phase 1C: {previous_score:.1f}")
        print(f"   After Phase 1C: {current_score:.1f}")
        print(f"   Change: {score_change:+.1f}")
        
        if score_change < -5.0:  # Significant decrease
            print(f"⚠️  SIGNIFICANT REGRESSION: Score decreased by {abs(score_change):.1f} points")
            print(f"   Possible causes:")
            print(f"   1. Phase 1C changes affected baseline scoring for clean responses")
            print(f"   2. Completion bonus logic not working correctly for length-limited responses")
            print(f"   3. Quality gate thresholds too restrictive for mathematical content")
        
        # This test documents the issue for further investigation
        self.assertGreater(current_score, 50.0, 
                          f"math_08 score {current_score:.1f} too low for quality mathematical response")


class TestBasic03UnexpectedRecovery(unittest.TestCase):
    """Investigate basic_03 unexpected recovery detection"""
    
    def setUp(self):
        """Set up test environment"""
        self.evaluator = EnhancedUniversalEvaluator()
        self.test_results_dir = Path(__file__).parent.parent.parent / "test_results"
        
        # Load basic_03 data (if available)
        self.basic_03_response = self._load_completion_text("basic_03_completion.txt")
        self.basic_03_result = self._load_result_json("basic_03_result.json")
    
    def _load_completion_text(self, filename):
        """Load completion text from test_results"""
        try:
            file_path = self.test_results_dir / filename
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
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
    
    def test_basic_03_final_segment_quality_analysis(self):
        """
        Analyze basic_03 final segment for actual quality indicators
        
        Question: Does the final segment actually show recovery worthy of 88.0 score?
        """
        # Note: basic_03 data may not be in test_results/ yet
        # This test provides the framework for analysis when data is available
        
        if not self.basic_03_response or not self.basic_03_result:
            print("ℹ️  basic_03 test data not available - test framework ready for analysis")
            self.skipTest("basic_03 test data not available")
        
        actual_score = self.basic_03_result.get('score', 0)
        
        print(f"🔍 basic_03 Final Segment Analysis:")
        print(f"   Score: {actual_score:.1f} (was ≤10 before Phase 1C)")
        
        # Analyze final segment (last 25% of response)
        if self.basic_03_response:
            lines = self.basic_03_response.split('\n')
            total_lines = len(lines)
            final_segment_start = int(total_lines * 0.75)
            final_segment = '\n'.join(lines[final_segment_start:])
            
            print(f"   Total lines: {total_lines}")
            print(f"   Final segment lines: {total_lines - final_segment_start}")
            
            # Check for quality indicators in final segment
            quality_indicators = [
                '**', '##', '1.', '2.', '3.', '- ', '---',
                'step', 'analysis', 'conclusion', 'answer', 'result'
            ]
            quality_count = sum(1 for indicator in quality_indicators 
                              if indicator.lower() in final_segment.lower())
            print(f"   Quality indicators in final segment: {quality_count}")
            
            # Check for completion indicators
            completion_indicators = [
                'therefore', 'thus', 'final', 'complete', 'answer', 'result',
                'conclusion', 'solution', 'translation', 'summary'
            ]
            completion_count = sum(1 for indicator in completion_indicators
                                 if indicator in final_segment.lower())
            print(f"   Completion indicators: {completion_count}")
            
            # Determine if recovery is justified
            if actual_score > 80.0:
                print(f"✅ High score justified if final segment shows:")
                print(f"   - Quality indicators: {quality_count} (need ≥2)")
                print(f"   - Completion indicators: {completion_count} (need ≥1)")
                print(f"   - Substantial content delivery")
            else:
                print(f"⚠️  Score {actual_score:.1f} indicates moderate recovery detection")
    
    def test_recovery_detection_threshold_validation(self):
        """
        Validate recovery detection threshold accuracy
        
        Question: Are we detecting recovery where none exists?
        """
        # Test recovery detection thresholds with controlled examples
        
        # Example 1: True recovery (basic_08 pattern)
        recovery_text = """
        **Step‑by‑step reasoning**
        
        1. **Identify the pattern**  
           Each line adds new attributes with nature imagery.
        
        2. **Choose appropriate attributes**
           Protector, Guardian, Strong one with cultural relevance.
           
        **Completed oriki**
        > Protector of the village's children,
        > Guardian who keeps the river's current steady,
        > Strong one whose roots run as deep as the baobab.
        """
        
        # Test final segment analysis on recovery text
        final_segment_analysis = self.evaluator._analyze_final_segment_quality(recovery_text)
        
        print(f"🧪 Recovery Detection Threshold Test:")
        print(f"   Quality score: {final_segment_analysis.get('quality_score', 0):.1f}")
        print(f"   Has structure: {final_segment_analysis.get('has_structure', False)}")
        print(f"   Is coherent: {final_segment_analysis.get('is_coherent', False)}")
        print(f"   Delivers content: {final_segment_analysis.get('delivers_content', False)}")
        print(f"   Recovery detected: {final_segment_analysis.get('recovery_detected', False)}")
        
        # This should detect recovery
        self.assertTrue(final_segment_analysis.get('recovery_detected', False),
                       "High-quality structured final segment should be detected as recovery")
        
        # Example 2: No recovery (pure loop)
        loop_text = """
        Compute base - xy? That yields base - xy. But we need to compute base^2 - base*(x+y) + xy.
        Compute base - xy? That yields base - xy. But we need to compute base^2 - base*(x+y) + xy.
        Compute base - xy? That yields base - xy. But we need to compute base^2 - base*(x+y) + xy.
        """
        
        no_recovery_analysis = self.evaluator._analyze_final_segment_quality(loop_text)
        print(f"\n   No Recovery Test:")
        print(f"   Quality score: {no_recovery_analysis.get('quality_score', 0):.1f}")
        print(f"   Recovery detected: {no_recovery_analysis.get('recovery_detected', False)}")
        
        # This should NOT detect recovery
        self.assertFalse(no_recovery_analysis.get('recovery_detected', False),
                        "Pure repetitive loops should not be detected as recovery")


class TestCompletionBonusLogic(unittest.TestCase):
    """Test completion bonus logic for different finish_reason scenarios"""
    
    def setUp(self):
        """Set up test environment"""
        self.evaluator = EnhancedUniversalEvaluator()
    
    def test_completion_bonus_scenarios(self):
        """
        Test completion bonus application across different scenarios
        
        Current logic from spec:
        if finish_reason == "stop" and quality_metrics.coherence_score > 0.7:
            return min(100.0, score + 3.0)
        
        Issue: This excludes high-quality responses with finish_reason="length"
        """
        # Test scenarios with different finish_reason values
        test_scenarios = [
            {
                'name': 'Natural completion',
                'finish_reason': 'stop',
                'quality_score': 0.8,
                'expected_bonus': True
            },
            {
                'name': 'High-quality but truncated',
                'finish_reason': 'length',
                'quality_score': 0.9,
                'expected_bonus': False  # Current logic gives no bonus
            },
            {
                'name': 'Low-quality natural completion',
                'finish_reason': 'stop',
                'quality_score': 0.5,
                'expected_bonus': False
            }
        ]
        
        print("🧪 Completion Bonus Logic Test:")
        for scenario in test_scenarios:
            name = scenario['name']
            finish_reason = scenario['finish_reason']
            quality_score = scenario['quality_score']
            expected_bonus = scenario['expected_bonus']
            
            # Simulate current bonus logic
            gets_bonus = (finish_reason == "stop" and quality_score > 0.7)
            
            print(f"   {name}:")
            print(f"     finish_reason='{finish_reason}', quality={quality_score:.1f}")
            print(f"     Current logic gives bonus: {gets_bonus}")
            print(f"     Expected bonus: {expected_bonus}")
            
            if gets_bonus != expected_bonus:
                print(f"     ⚠️  LOGIC ISSUE: May need to adjust completion bonus criteria")
        
        # Recommendation for improved logic
        print(f"\n💡 Recommended improved completion bonus logic:")
        print(f"   if (finish_reason == 'stop' and quality > 0.7) or")
        print(f"      (finish_reason == 'length' and quality > 0.8):  # Higher threshold for truncated")
        print(f"       return min(100.0, score + bonus)")


if __name__ == '__main__':
    # Run the tests with verbose output
    unittest.main(verbosity=2)