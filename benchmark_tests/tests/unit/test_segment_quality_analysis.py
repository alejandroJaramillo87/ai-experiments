"""
Unit Tests for Phase 1C Final Segment Quality Analysis

Tests individual Phase 1C functions:
- _analyze_final_segment_quality()
- _check_structured_format()
- _check_coherence_final_segment()
- _check_content_delivery()
- _calculate_segment_quality()
"""

import unittest
from unittest.mock import Mock
from pathlib import Path

# Import the enhanced evaluator
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluator.subjects.enhanced_universal_evaluator import EnhancedUniversalEvaluator


class TestFinalSegmentQualityAnalysis(unittest.TestCase):
    """Test the _analyze_final_segment_quality() method"""
    
    def setUp(self):
        """Set up test environment"""
        self.evaluator = EnhancedUniversalEvaluator()
    
    def test_analyze_final_segment_quality_basic_08_pattern(self):
        """Test final segment analysis with basic_08 recovery pattern"""
        # Simulate basic_08 final segment (high-quality cultural output)
        recovery_response = '''
        We must respond to the final answer...
        Let me think about this...
        Actually, let me approach this differently...
        
        **Step‑by‑step reasoning**
        
        1. **Identify the pattern**  
           Each line of the oriki adds a new attribute or metaphor.  
           The language is rich in nature imagery and action verbs.
        
        2. **Choose attributes that fit**  
           Protector, Guardian, Strong one with cultural significance.
        
        **Completed oriki**
        
        > Protector of the village's children,  
        > Guardian who keeps the river's current steady,  
        > Strong one whose roots run as deep as the baobab.
        '''
        
        # Analyze final segment
        analysis = self.evaluator._analyze_final_segment_quality(recovery_response)
        
        # Verify analysis results
        self.assertIsInstance(analysis, dict, "Should return dictionary")
        self.assertIn('quality_score', analysis)
        self.assertIn('has_structure', analysis) 
        self.assertIn('is_coherent', analysis)
        self.assertIn('delivers_content', analysis)
        self.assertIn('recovery_detected', analysis)
        
        # This should be detected as recovery
        self.assertTrue(analysis['has_structure'], "Should detect structured formatting")
        self.assertTrue(analysis['is_coherent'], "Should detect coherent final segment")  
        self.assertTrue(analysis['delivers_content'], "Should detect content delivery")
        self.assertGreater(analysis['quality_score'], 70.0, "Should have high quality score")
        self.assertTrue(analysis['recovery_detected'], "Should detect recovery")
        
        print(f"✅ basic_08 pattern analysis: quality={analysis['quality_score']:.1f}, recovery={analysis['recovery_detected']}")
    
    def test_analyze_final_segment_quality_math_04_pattern(self):
        """Test final segment analysis with math_04 pure loop pattern"""
        # Simulate math_04 final segment (pure repetitive loops)
        loop_response = '''
        Compute base - xy? That yields base - xy. But we need to compute base^2 - base*(x+y) + xy. But we can compute as:
        Compute base - xy? That yields base - xy. But we need to compute base^2 - base*(x+y) + xy. But we can compute as:
        Compute base - xy? That yields base - xy. But we need to compute base^2 - base*(x+y) + xy. But we can compute as:
        Compute base - xy? That yields base - xy. But we need to compute base^2 - base*(x+y) + xy. But we can compute as:
        '''
        
        # Analyze final segment
        analysis = self.evaluator._analyze_final_segment_quality(loop_response)
        
        # This should NOT be detected as recovery
        self.assertFalse(analysis['has_structure'], "Should not detect structure in repetitive loops")
        self.assertFalse(analysis['is_coherent'], "Should not detect coherence in loops")
        self.assertFalse(analysis['delivers_content'], "Should not detect content in pure loops")
        self.assertLess(analysis['quality_score'], 50.0, "Should have low quality score")
        self.assertFalse(analysis['recovery_detected'], "Should not detect recovery")
        
        print(f"✅ math_04 pattern analysis: quality={analysis['quality_score']:.1f}, recovery={analysis['recovery_detected']}")
    
    def test_analyze_final_segment_quality_edge_cases(self):
        """Test edge cases for final segment analysis"""
        # Test 1: Very short response
        short_response = "Yes."
        analysis = self.evaluator._analyze_final_segment_quality(short_response)
        self.assertFalse(analysis['recovery_detected'], "Short response should not show recovery")
        
        # Test 2: Empty response
        empty_response = ""
        analysis = self.evaluator._analyze_final_segment_quality(empty_response)
        self.assertFalse(analysis['recovery_detected'], "Empty response should not show recovery")
        
        # Test 3: Just meta-reasoning, no final output
        meta_only_response = '''
        Let me think about this...
        Actually, I'm not sure...
        Maybe I should reconsider...
        Wait, let me try again...
        '''
        analysis = self.evaluator._analyze_final_segment_quality(meta_only_response)
        self.assertFalse(analysis['recovery_detected'], "Pure meta-reasoning should not show recovery")
        
        print("✅ Edge cases handled correctly")


class TestStructuredFormatDetection(unittest.TestCase):
    """Test the _check_structured_format() method"""
    
    def setUp(self):
        """Set up test environment"""
        self.evaluator = EnhancedUniversalEvaluator()
    
    def test_check_structured_format_positive_cases(self):
        """Test cases that should be detected as structured"""
        structured_texts = [
            "**Step 1**: Analysis\n**Step 2**: Solution",  # Bold headers
            "## Analysis\n## Solution",  # Markdown headers
            "1. First point\n2. Second point\n3. Third point",  # Numbered list
            "- Point A\n- Point B\n- Point C",  # Bullet points
            "---\nSection 1\n---\nSection 2",  # Separators
            "Step 1: Analysis\nStep 2: Solution"  # Step indicators
        ]
        
        for text in structured_texts:
            with self.subTest(text=text[:30]):
                result = self.evaluator._check_structured_format(text)
                self.assertTrue(result, f"Should detect structure in: {text[:30]}...")
        
        print("✅ Structured format detection working for positive cases")
    
    def test_check_structured_format_negative_cases(self):
        """Test cases that should NOT be detected as structured"""
        unstructured_texts = [
            "This is just plain text without any formatting.",
            "Some text here and some more text there.",
            "Maybe we should think about this more carefully.",
            "I'm not sure what the answer is to be honest."
        ]
        
        for text in unstructured_texts:
            with self.subTest(text=text[:30]):
                result = self.evaluator._check_structured_format(text)
                self.assertFalse(result, f"Should not detect structure in: {text[:30]}...")
        
        print("✅ Structured format detection working for negative cases")


class TestCoherenceFinalSegment(unittest.TestCase):
    """Test the _check_coherence_final_segment() method"""
    
    def setUp(self):
        """Set up test environment"""
        self.evaluator = EnhancedUniversalEvaluator()
    
    def test_check_coherence_final_segment_positive_cases(self):
        """Test cases that should be detected as coherent"""
        coherent_texts = [
            '''Therefore, the final answer is based on careful analysis.
            The solution involves three main steps.
            Thus, we can conclude the oriki pattern is complete.''',
            
            '''The analysis shows clear results.
            Final calculations confirm our approach.
            In conclusion, the cultural elements are properly integrated.''',
            
            '''This translation captures the essence of the original.
            The cultural context supports this interpretation.  
            Therefore, the completed work maintains authenticity.'''
        ]
        
        for text in coherent_texts:
            with self.subTest(text=text[:40]):
                result = self.evaluator._check_coherence_final_segment(text)
                self.assertTrue(result, f"Should detect coherence in: {text[:40]}...")
        
        print("✅ Coherence detection working for positive cases")
    
    def test_check_coherence_final_segment_negative_cases(self):
        """Test cases that should NOT be detected as coherent"""
        incoherent_texts = [
            "Maybe we should... I'm not sure... Let's think... Actually wait...",
            "Hmm, let me reconsider... Actually, maybe not... Wait, let's try...",
            "I'm not sure about this... Let's think again... Maybe we can...",
            "This is short."  # Too short to be meaningfully coherent
        ]
        
        for text in incoherent_texts:
            with self.subTest(text=text[:40]):
                result = self.evaluator._check_coherence_final_segment(text)
                self.assertFalse(result, f"Should not detect coherence in: {text[:40]}...")
        
        print("✅ Coherence detection working for negative cases")


class TestContentDelivery(unittest.TestCase):
    """Test the _check_content_delivery() method"""
    
    def setUp(self):
        """Set up test environment"""
        self.evaluator = EnhancedUniversalEvaluator()
    
    def test_check_content_delivery_positive_cases(self):
        """Test cases that should be detected as delivering content"""
        content_delivery_texts = [
            '''Protector of the village's children,
            Guardian who keeps the river's current steady,
            Strong one whose roots run as deep as the baobab.''',  # Cultural content
            
            '''The equation yields: x = 5
            Therefore, the final answer is 25.
            This solves the mathematical problem completely.''',  # Mathematical content
            
            '''Analysis shows three key points:
            1. Cultural authenticity is maintained
            2. Traditional patterns are followed
            3. The translation preserves meaning''',  # Analytical content
            
            '''"Ògún, master of iron" translates to the praise of the deity.
            This example shows proper oriki structure.
            The description follows Yoruba traditions.'''  # Translation content
        ]
        
        for text in content_delivery_texts:
            with self.subTest(text=text[:40]):
                result = self.evaluator._check_content_delivery(text)
                self.assertTrue(result, f"Should detect content delivery in: {text[:40]}...")
        
        print("✅ Content delivery detection working for positive cases")
    
    def test_check_content_delivery_negative_cases(self):
        """Test cases that should NOT be detected as delivering content"""
        no_content_texts = [
            "Let me think about this more...",  # Pure meta-reasoning
            "I'm not sure how to proceed.",  # Uncertainty without delivery
            "Maybe we should try again.",  # Vague suggestion
            "Hmm."  # Too short
        ]
        
        for text in no_content_texts:
            with self.subTest(text=text[:40]):
                result = self.evaluator._check_content_delivery(text)
                self.assertFalse(result, f"Should not detect content delivery in: {text[:40]}...")
        
        print("✅ Content delivery detection working for negative cases")


class TestSegmentQualityCalculation(unittest.TestCase):
    """Test the _calculate_segment_quality() method"""
    
    def setUp(self):
        """Set up test environment"""
        self.evaluator = EnhancedUniversalEvaluator()
    
    def test_calculate_segment_quality_scoring_logic(self):
        """Test the quality scoring calculation logic"""
        # Test maximum quality (all indicators true, substantial length)
        substantial_text = "This is a substantial response with meaningful content that demonstrates quality and completeness in its delivery and analysis. It provides thorough explanation with detailed reasoning and comprehensive coverage of the topic at hand, ensuring that all requirements are met with precision and clarity."
        
        max_score = self.evaluator._calculate_segment_quality(
            substantial_text, has_structure=True, is_coherent=True, delivers_content=True
        )
        
        # Should get base (40) + structure (25) + coherence (20) + content (15) + length bonus
        self.assertGreater(max_score, 90.0, "Maximum quality should score >90")
        self.assertLessEqual(max_score, 100.0, "Score should not exceed 100")
        
        # Test minimum quality (no indicators, very short)
        short_text = "No."
        min_score = self.evaluator._calculate_segment_quality(
            short_text, has_structure=False, is_coherent=False, delivers_content=False
        )
        
        # Should get base (40) - length penalty (10) = 30
        self.assertLess(min_score, 40.0, "Minimum quality should score <40")
        self.assertGreaterEqual(min_score, 0.0, "Score should not be negative")
        
        # Test partial quality
        medium_text = "This is a medium length response that has some structure and content. It provides adequate detail and explanation without being overly comprehensive."
        partial_score = self.evaluator._calculate_segment_quality(
            medium_text, has_structure=True, is_coherent=False, delivers_content=True
        )
        
        # Should get base (40) + structure (25) + content (15) + small length bonus
        self.assertGreater(partial_score, 70.0, "Partial quality should score >70")
        self.assertLess(partial_score, 90.0, "Partial quality should score <90")
        
        print(f"✅ Quality calculation: max={max_score:.1f}, partial={partial_score:.1f}, min={min_score:.1f}")
    
    def test_calculate_segment_quality_recovery_threshold(self):
        """Test that quality scores align with recovery detection threshold (>70)"""
        # Text that should trigger recovery detection
        recovery_text = '''
        **Step-by-step reasoning**
        
        1. Analyze the pattern carefully
        2. Choose appropriate cultural elements
        3. Deliver the completed response
        
        **Final Answer**
        Protector of the village's children,
        Guardian who keeps the river's current steady.
        '''
        
        recovery_score = self.evaluator._calculate_segment_quality(
            recovery_text, has_structure=True, is_coherent=True, delivers_content=True
        )
        
        # Should exceed recovery threshold
        self.assertGreater(recovery_score, 70.0, 
                          "Recovery-worthy content should score >70 for threshold")
        
        # Text that should NOT trigger recovery detection
        no_recovery_text = "Maybe we should think about this more... I'm not sure..."
        
        no_recovery_score = self.evaluator._calculate_segment_quality(
            no_recovery_text, has_structure=False, is_coherent=False, delivers_content=False
        )
        
        # Should NOT exceed recovery threshold
        self.assertLessEqual(no_recovery_score, 70.0,
                           "Non-recovery content should score ≤70")
        
        print(f"✅ Recovery threshold alignment: recovery={recovery_score:.1f}, no-recovery={no_recovery_score:.1f}")


if __name__ == '__main__':
    # Run the tests with verbose output
    unittest.main(verbosity=2)