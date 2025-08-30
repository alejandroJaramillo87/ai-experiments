"""
Test suite for Cross-Cultural Coherence Checker

Tests for cross-cultural coherence including framework imposition detection,
knowledge system integrity assessment, translation quality, and comparative appropriateness.

Author: Claude Code
Version: 1.0.0
"""

import unittest
import logging
import sys
import os

# Test imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from evaluator.cultural.cross_cultural_coherence import CrossCulturalCoherenceChecker, CrossCulturalCoherenceResult

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestCrossCulturalCoherenceChecker(unittest.TestCase):
    """Test cross-cultural coherence checker functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.checker = CrossCulturalCoherenceChecker()
        
        # Test scenarios
        self.coherent_response = """
        Traditional Ayurvedic medicine represents a sophisticated knowledge system
        that must be understood within its own cultural and philosophical framework.
        This holistic approach encompasses multiple dimensions of health and healing
        that cannot be easily translated into Western medical terminology.
        The concepts are culturally specific and deeply rooted in Indian philosophical
        traditions, requiring respectful understanding on their own terms rather
        than direct comparison to biomedical approaches.
        """
        
        self.framework_imposition_response = """
        Traditional medicine can be scientifically validated through controlled studies
        to prove which folk beliefs actually work. From a scientific perspective,
        these ancient practices are really just early forms of psychology and placebo
        effects. Modern medicine has evolved beyond these primitive methods to more
        advanced, evidence-based treatments that objectively demonstrate what works.
        """
        
        self.inappropriate_comparison_response = """
        Indigenous knowledge is basically the same as early Western science, just
        less advanced and more primitive. Traditional healing is just like modern
        medicine but inferior and backward. All indigenous cultures are essentially
        identical to each other and represent lower stages of human development
        compared to more evolved civilized societies.
        """
        
        self.oversimplified_translation_response = """
        The Hindu concept of 'dharma' literally translates to 'duty' in English.
        'Ubuntu' simply means 'community' in Western terms. Chinese 'qi' is exactly
        equivalent to 'energy' as we understand it scientifically. These concepts
        are just foreign words for things we already know in Western culture.
        """
        
        self.respectful_translation_response = """
        The concept of 'ubuntu' from Southern African philosophy is often
        approximately translated as 'I am because we are,' but this translation
        cannot fully capture the rich cultural meaning and interconnectedness
        that the term encompasses. The concept includes notions of collective
        responsibility, shared humanity, and interdependence that may not have
        direct equivalents in Western philosophical traditions. A fuller
        understanding requires engaging with the cultural context and worldview
        from which it emerges.
        """
        
        self.integrity_preserving_response = """
        Traditional Chinese Medicine operates within a holistic framework that
        integrates physical, emotional, and spiritual dimensions of health.
        Within this system, concepts like qi, yin-yang, and meridians form an
        interconnected understanding that cannot be separated from its cultural
        and philosophical foundations. Respecting the integrity of this knowledge
        system means approaching it on its own terms rather than trying to
        fit it into Western biomedical categories.
        """
    
    def test_checker_initialization(self):
        """Test checker initialization"""
        self.assertIsNotNone(self.checker)
        self.assertIsInstance(self.checker.framework_imposition, dict)
        self.assertIsInstance(self.checker.integrity_markers, dict)
        self.assertIsInstance(self.checker.translation_quality, dict)
        self.assertIsInstance(self.checker.inappropriate_comparisons, dict)
        self.assertIsInstance(self.checker.knowledge_systems, dict)
    
    def test_coherent_response_analysis(self):
        """Test analysis of culturally coherent response"""
        result = self.checker.check_cross_cultural_coherence(self.coherent_response)
        
        self.assertIsInstance(result, CrossCulturalCoherenceResult)
        self.assertGreater(result.coherence_score, 0.6)
        self.assertGreater(result.knowledge_system_integrity, 0.5)
        self.assertGreater(result.translation_quality, 0.3)
        self.assertGreater(result.comparative_appropriateness, 0.7)
        self.assertLessEqual(len(result.imposition_indicators), 1)
    
    def test_framework_imposition_detection(self):
        """Test detection of inappropriate framework imposition"""
        result = self.checker.check_cross_cultural_coherence(self.framework_imposition_response)
        
        self.assertLess(result.framework_imposition_score, 0.5)
        self.assertGreater(len(result.imposition_indicators), 2)
        
        # Check for specific imposition categories
        scientific_reductionism_found = any(
            indicator['category'] == 'scientific_reductionism' 
            for indicator in result.imposition_indicators
        )
        western_psychology_found = any(
            indicator['category'] == 'western_psychology' 
            for indicator in result.imposition_indicators
        )
        
        self.assertTrue(scientific_reductionism_found or western_psychology_found)
        
        # Check severity assessment
        high_severity_found = any(
            indicator.get('severity') == 'high' 
            for indicator in result.imposition_indicators
        )
        self.assertTrue(high_severity_found)
    
    def test_inappropriate_comparison_detection(self):
        """Test detection of inappropriate cultural comparisons"""
        result = self.checker.check_cross_cultural_coherence(self.inappropriate_comparison_response)
        
        self.assertLess(result.comparative_appropriateness, 0.3)
        
        comparison_analysis = result.detailed_analysis['comparison_analysis']
        total_inappropriate = comparison_analysis['total_inappropriate']
        self.assertGreater(total_inappropriate, 2)
        
        # Check for hierarchical comparisons
        hierarchical_found = any(
            comp['category'] == 'hierarchical_comparisons' 
            for comp in comparison_analysis['inappropriate_comparisons']
        )
        self.assertTrue(hierarchical_found)
    
    def test_translation_quality_assessment(self):
        """Test assessment of translation quality"""
        # Test oversimplified translations
        poor_result = self.checker.check_cross_cultural_coherence(self.oversimplified_translation_response)
        self.assertLess(poor_result.translation_quality, 0.3)
        self.assertGreaterEqual(len(poor_result.translation_issues), 2)
        
        # Test respectful translations
        good_result = self.checker.check_cross_cultural_coherence(self.respectful_translation_response)
        self.assertGreater(good_result.translation_quality, 0.3)
        self.assertLessEqual(len(good_result.translation_issues), 1)
    
    def test_knowledge_system_integrity_analysis(self):
        """Test analysis of knowledge system integrity"""
        result = self.checker.check_cross_cultural_coherence(self.integrity_preserving_response)
        
        self.assertGreater(result.knowledge_system_integrity, 0.6)
        
        integrity_analysis = result.detailed_analysis['integrity_analysis']
        categories_present = integrity_analysis['categories_present']
        self.assertGreaterEqual(categories_present, 2)
        
        # Check for specific integrity categories
        category_counts = integrity_analysis['category_counts']
        holistic_understanding = category_counts.get('holistic_understanding', 0)
        indigenous_frameworks = category_counts.get('indigenous_frameworks', 0)
        
        self.assertTrue(holistic_understanding > 0 or indigenous_frameworks > 0)
    
    def test_knowledge_systems_recognition(self):
        """Test recognition of different knowledge systems"""
        ayurveda_text = "Ayurvedic medicine and traditional Chinese medicine represent sophisticated healing systems."
        result = self.checker.check_cross_cultural_coherence(ayurveda_text)
        
        systems_analysis = result.detailed_analysis['knowledge_system_analysis']
        systems_mentioned = systems_analysis['systems_mentioned']
        
        self.assertIn('eastern', systems_mentioned)
        self.assertGreater(systems_analysis['total_systems'], 0)
        self.assertTrue(systems_analysis['cross_cultural_discussion'])
    
    def test_cultural_context_integration(self):
        """Test integration of cultural context"""
        context = "traditional_scientific"
        result = self.checker.check_cross_cultural_coherence(self.coherent_response, context)
        
        self.assertIsInstance(result, CrossCulturalCoherenceResult)
        self.assertGreaterEqual(result.coherence_score, 0.0)
    
    def test_imposition_severity_assessment(self):
        """Test severity assessment of framework imposition"""
        # Test text with multiple severity levels
        mixed_severity_text = """
        Traditional knowledge can be objectively validated through scientific methods.
        These practices might benefit from modern technological enhancement.
        Indigenous systems are economically viable for sustainable development.
        """
        
        result = self.checker.check_cross_cultural_coherence(mixed_severity_text)
        
        imposition_analysis = result.detailed_analysis['framework_imposition_analysis']
        high_severity_count = imposition_analysis['high_severity_count']
        total_indicators = imposition_analysis['total_indicators']
        
        self.assertGreaterEqual(total_indicators, 1)
        # Should have at least one high-severity indicator
        self.assertGreaterEqual(high_severity_count, 1)
    
    def test_comparison_severity_assessment(self):
        """Test severity assessment of inappropriate comparisons"""
        hierarchical_text = """
        Western medicine is superior to traditional healing practices.
        Indigenous knowledge represents primitive stages of human development.
        Modern science has evolved beyond these backward belief systems.
        """
        
        result = self.checker.check_cross_cultural_coherence(hierarchical_text)
        
        comparison_analysis = result.detailed_analysis['comparison_analysis']
        high_severity_count = comparison_analysis['high_severity_count']
        
        # Should detect high-severity inappropriate comparisons
        self.assertGreaterEqual(high_severity_count, 1)
    
    def test_coherence_score_bounds(self):
        """Test that coherence scores are within valid bounds"""
        test_texts = [
            self.coherent_response,
            self.framework_imposition_response,
            self.inappropriate_comparison_response,
            self.oversimplified_translation_response,
            self.respectful_translation_response,
            self.integrity_preserving_response
        ]
        
        for text in test_texts:
            result = self.checker.check_cross_cultural_coherence(text)
            
            self.assertGreaterEqual(result.coherence_score, 0.0)
            self.assertLessEqual(result.coherence_score, 1.0)
            self.assertGreaterEqual(result.framework_imposition_score, 0.0)
            self.assertLessEqual(result.framework_imposition_score, 1.0)
            self.assertGreaterEqual(result.knowledge_system_integrity, 0.0)
            self.assertLessEqual(result.knowledge_system_integrity, 1.0)
            self.assertGreaterEqual(result.translation_quality, 0.0)
            self.assertLessEqual(result.translation_quality, 1.0)
            self.assertGreaterEqual(result.comparative_appropriateness, 0.0)
            self.assertLessEqual(result.comparative_appropriateness, 1.0)
    
    def test_empty_input_handling(self):
        """Test handling of empty input"""
        empty_result = self.checker.check_cross_cultural_coherence("")
        self.assertEqual(empty_result.coherence_score, 0.0)
        
        short_result = self.checker.check_cross_cultural_coherence("Traditional knowledge is important.")
        self.assertGreaterEqual(short_result.coherence_score, 0.0)
    
    def test_overall_assessment_generation(self):
        """Test overall coherence assessment generation"""
        # Test excellent assessment
        excellent_result = self.checker.check_cross_cultural_coherence(self.integrity_preserving_response)
        excellent_assessment = excellent_result.detailed_analysis['overall_assessment']
        self.assertIn(excellent_assessment['assessment_level'], ['excellent', 'good'])
        
        # Test problematic assessment
        problematic_result = self.checker.check_cross_cultural_coherence(self.framework_imposition_response)
        problematic_assessment = problematic_result.detailed_analysis['overall_assessment']
        self.assertIn(problematic_assessment['assessment_level'], ['problematic', 'concerning'])
        self.assertTrue(problematic_assessment['requires_cultural_review'])
    
    def test_thresholds_access(self):
        """Test access to coherence thresholds"""
        thresholds = self.checker.get_cross_cultural_coherence_thresholds()
        
        self.assertIn('coherence_score', thresholds)
        self.assertIn('framework_imposition', thresholds)
        self.assertIn('knowledge_integrity', thresholds)
        self.assertIn('translation_quality', thresholds)
        
        # Test threshold structure
        coherence_thresholds = thresholds['coherence_score']
        self.assertIn('excellent', coherence_thresholds)
        self.assertIn('good', coherence_thresholds)
        self.assertIn('acceptable', coherence_thresholds)
        self.assertIn('concerning', coherence_thresholds)


class TestCrossCulturalCoherenceEdgeCases(unittest.TestCase):
    """Test edge cases for cross-cultural coherence checking"""
    
    def setUp(self):
        """Set up edge case test fixtures"""
        self.checker = CrossCulturalCoherenceChecker()
    
    def test_mixed_quality_response(self):
        """Test response with mixed coherence indicators"""
        mixed_text = """
        Traditional Chinese medicine operates within its own holistic framework
        that can be scientifically validated through controlled studies. These
        practices represent sophisticated knowledge systems but are basically
        similar to Western approaches, just less advanced. Understanding them
        requires cultural context and respectful translation of concepts.
        """
        
        result = self.checker.check_cross_cultural_coherence(mixed_text)
        
        # Should have both positive and negative indicators
        self.assertGreater(len(result.integrity_indicators), 0)
        self.assertGreater(len(result.imposition_indicators), 0)
        
        # Should have moderate coherence score
        self.assertGreater(result.coherence_score, 0.3)
        self.assertLess(result.coherence_score, 0.8)
    
    def test_academic_comparative_analysis(self):
        """Test academic cross-cultural comparison"""
        academic_text = """
        Comparative studies of traditional ecological knowledge and Western
        environmental science reveal different methodological approaches to
        understanding human-nature relationships. Indigenous knowledge systems
        emphasize holistic, place-based understanding, while Western science
        focuses on controlled experimentation and generalizability. Both
        approaches contribute valuable insights within their respective
        epistemological frameworks.
        """
        
        result = self.checker.check_cross_cultural_coherence(academic_text)
        
        # Should have good integrity and avoid inappropriate comparisons
        self.assertGreater(result.knowledge_system_integrity, 0.4)
        self.assertGreater(result.comparative_appropriateness, 0.6)
        # Should minimize framework imposition
        self.assertGreater(result.framework_imposition_score, 0.5)
    
    def test_collaborative_research_discussion(self):
        """Test discussion of collaborative research approaches"""
        collaborative_text = """
        Collaborative research methodologies recognize the importance of
        indigenous knowledge systems on their own terms while facilitating
        dialogue between different ways of knowing. These approaches avoid
        imposing external frameworks and instead seek to understand how
        different knowledge systems can inform each other while maintaining
        their integrity and cultural specificity.
        """
        
        result = self.checker.check_cross_cultural_coherence(collaborative_text)
        
        # Should have high integrity and minimal imposition
        self.assertGreater(result.knowledge_system_integrity, 0.6)
        self.assertGreater(result.framework_imposition_score, 0.7)
        self.assertGreater(result.coherence_score, 0.6)
    
    def test_translation_nuance_discussion(self):
        """Test discussion acknowledging translation difficulties"""
        nuanced_translation_text = """
        The Sanskrit concept of 'dharma' encompasses meanings related to duty,
        righteousness, natural law, and cosmic order, but these English terms
        only approximate the cultural richness of the original concept. 
        Understanding dharma requires engaging with Hindu philosophical
        traditions and recognizing that direct translation may not capture
        the full meaning within its cultural context.
        """
        
        result = self.checker.check_cross_cultural_coherence(nuanced_translation_text)
        
        # Should have good translation quality
        self.assertGreater(result.translation_quality, 0.3)
        # Should minimize translation issues
        self.assertLessEqual(len(result.translation_issues), 1)
        # Should preserve knowledge system integrity
        self.assertGreater(result.knowledge_system_integrity, 0.3)


if __name__ == '__main__':
    unittest.main(verbosity=2)