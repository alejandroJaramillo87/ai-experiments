"""
Test suite for Cultural Authenticity Analyzer

Tests for cultural authenticity analysis including stereotype detection,
appropriation markers, bias detection, and respectful language assessment.

Author: Claude Code
Version: 1.0.0
"""

import unittest
import logging
import sys
import os

# Test imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from evaluator.cultural.cultural_authenticity import CulturalAuthenticityAnalyzer, CulturalAuthenticityResult

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestCulturalAuthenticityAnalyzer(unittest.TestCase):
    """Test cultural authenticity analyzer functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = CulturalAuthenticityAnalyzer()
        
        # Test scenarios
        self.respectful_response = """
        According to indigenous communities in the Pacific Northwest, traditional ecological knowledge 
        has been passed down through generations of elders. Within their cultural framework, 
        the relationship between humans and nature is understood as interconnected and reciprocal.
        This complex system of knowledge encompasses traditional medicine practices, sustainable 
        harvesting methods, and spiritual connections to the land. It's important to note that 
        this is a limited understanding from an outsider's perspective, and the full depth 
        of these traditions requires learning directly from community members and elders.
        """
        
        self.problematic_response = """
        All indigenous people believe in mystical connections to nature and use primitive healing methods.
        These exotic practices are basically just folk medicine that can be explained scientifically.
        You can buy dreamcatchers and medicine wheels online to harness their ancient mystical powers.
        Traditional shamans are just early psychologists who didn't understand modern science.
        These backward cultures need to modernize their outdated beliefs with scientific knowledge.
        """
        
        self.stereotyping_response = """
        Indigenous cultures always live in harmony with nature and never harm the environment.
        They typically use strange rituals and unusual ceremonies for healing. All tribal people
        share the same mystical beliefs about spirit animals and ancient secrets. Every native
        community practices the same traditional customs and holds identical spiritual views.
        """
        
        self.appropriation_response = """
        You can easily practice smudging at home by burning sage and try vision quests in your backyard.
        DIY spiritual healing using traditional medicine wheels is available for purchase online.
        Anyone can become a shaman by learning these simple techniques and borrowing from indigenous traditions.
        These ancient practices are perfect for modern wellness and can be commercialized for profit.
        """
        
        self.bias_response = """
        Modern science proves that traditional indigenous knowledge is really just superstition.
        From a scientific perspective, these folk beliefs can be explained by psychology and placebo effects.
        Western medicine has evolved beyond these primitive healing methods to more advanced treatments.
        Objective scientific truth transcends cultural beliefs and shows us what really works.
        """
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        self.assertIsNotNone(self.analyzer)
        self.assertIsInstance(self.analyzer.stereotype_patterns, dict)
        self.assertIsInstance(self.analyzer.appropriation_patterns, dict)
        self.assertIsInstance(self.analyzer.bias_patterns, dict)
        self.assertIsInstance(self.analyzer.respectful_markers, dict)
    
    def test_respectful_response_analysis(self):
        """Test analysis of respectful response"""
        result = self.analyzer.analyze_cultural_authenticity(self.respectful_response)
        
        self.assertIsInstance(result, CulturalAuthenticityResult)
        self.assertGreater(result.authenticity_score, 0.7)
        self.assertGreater(result.respectful_language_score, 0.5)
        self.assertGreater(result.cultural_sensitivity_score, 0.5)
        self.assertEqual(len(result.stereotype_indicators), 0)
        self.assertEqual(len(result.appropriation_markers), 0)
        self.assertLessEqual(len(result.bias_indicators), 1)  # Allow minimal bias
    
    def test_problematic_response_analysis(self):
        """Test analysis of problematic response with multiple issues"""
        result = self.analyzer.analyze_cultural_authenticity(self.problematic_response)
        
        self.assertIsInstance(result, CulturalAuthenticityResult)
        self.assertLess(result.authenticity_score, 0.4)
        self.assertGreater(len(result.stereotype_indicators), 2)
        self.assertGreater(len(result.appropriation_markers), 1)
        self.assertGreater(len(result.bias_indicators), 2)
        
        # Check that critical issues are detected
        assessment = result.detailed_analysis.get('overall_assessment', {})
        self.assertIn(assessment.get('cultural_risk_level'), ['high', 'medium'])
        self.assertTrue(assessment.get('requires_review', False))
    
    def test_stereotype_detection(self):
        """Test stereotype detection functionality"""
        result = self.analyzer.analyze_cultural_authenticity(self.stereotyping_response)
        
        self.assertGreater(len(result.stereotype_indicators), 3)
        
        # Check specific stereotype categories
        essentializing_found = any(
            indicator['category'] == 'essentializing' 
            for indicator in result.stereotype_indicators
        )
        othering_found = any(
            indicator['category'] == 'othering' 
            for indicator in result.stereotype_indicators
        )
        romanticizing_found = any(
            indicator['category'] == 'romanticizing' 
            for indicator in result.stereotype_indicators
        )
        
        self.assertTrue(essentializing_found or othering_found or romanticizing_found)
    
    def test_appropriation_marker_detection(self):
        """Test cultural appropriation marker detection"""
        result = self.analyzer.analyze_cultural_authenticity(self.appropriation_response)
        
        self.assertGreater(len(result.appropriation_markers), 2)
        
        # Check for specific appropriation categories
        sacred_elements_found = any(
            marker['category'] == 'sacred_elements' 
            for marker in result.appropriation_markers
        )
        ceremonial_context_found = any(
            marker['category'] == 'ceremonial_context' 
            for marker in result.appropriation_markers
        )
        
        self.assertTrue(sacred_elements_found or ceremonial_context_found)
        
        # Check severity assessment
        high_severity_found = any(
            marker.get('severity') == 'high' 
            for marker in result.appropriation_markers
        )
        self.assertTrue(high_severity_found)
    
    def test_bias_indicator_detection(self):
        """Test cultural bias indicator detection"""
        result = self.analyzer.analyze_cultural_authenticity(self.bias_response)
        
        self.assertGreater(len(result.bias_indicators), 2)
        
        # Check for specific bias categories
        superiority_found = any(
            indicator['category'] == 'western_superiority' 
            for indicator in result.bias_indicators
        )
        progress_narrative_found = any(
            indicator['category'] == 'progress_narrative' 
            for indicator in result.bias_indicators
        )
        universalizing_found = any(
            indicator['category'] == 'universalizing' 
            for indicator in result.bias_indicators
        )
        
        self.assertTrue(superiority_found or progress_narrative_found or universalizing_found)
    
    def test_respectful_language_analysis(self):
        """Test respectful language marker detection"""
        result = self.analyzer.analyze_cultural_authenticity(self.respectful_response)
        
        # Should find attribution markers
        attribution_count = result.detailed_analysis['respectful_language_analysis']['category_counts'].get('attribution', 0)
        self.assertGreater(attribution_count, 0)
        
        # Should find contextualization markers
        contextualization_count = result.detailed_analysis['respectful_language_analysis']['category_counts'].get('contextualization', 0)
        self.assertGreater(contextualization_count, 0)
        
        # Should find humility markers
        humility_count = result.detailed_analysis['respectful_language_analysis']['category_counts'].get('humility', 0)
        self.assertGreater(humility_count, 0)
    
    def test_cultural_sensitivity_analysis(self):
        """Test cultural sensitivity analysis"""
        result = self.analyzer.analyze_cultural_authenticity(self.respectful_response)
        
        sensitivity_analysis = result.detailed_analysis['sensitivity_analysis']
        
        # Should detect positive sensitivity indicators
        indicators = sensitivity_analysis['sensitivity_indicators']
        self.assertTrue(indicators['shows_humility'])
        self.assertTrue(indicators['respects_complexity'])
        
        # Should have reasonable sensitivity score
        self.assertGreater(sensitivity_analysis['score'], 0.3)
    
    def test_cultural_group_recognition(self):
        """Test cultural group mention recognition"""
        test_text = "Indigenous communities and native peoples have developed sophisticated traditional knowledge systems."
        result = self.analyzer.analyze_cultural_authenticity(test_text)
        
        sensitivity_analysis = result.detailed_analysis['sensitivity_analysis']
        group_mentions = sensitivity_analysis['group_mentions']
        
        self.assertGreater(group_mentions['indigenous'], 0)
        self.assertGreater(sensitivity_analysis['total_cultural_mentions'], 1)
    
    def test_authenticity_score_bounds(self):
        """Test that authenticity scores are within valid bounds"""
        test_texts = [
            self.respectful_response,
            self.problematic_response,
            self.stereotyping_response,
            self.appropriation_response,
            self.bias_response
        ]
        
        for text in test_texts:
            result = self.analyzer.analyze_cultural_authenticity(text)
            
            self.assertGreaterEqual(result.authenticity_score, 0.0)
            self.assertLessEqual(result.authenticity_score, 1.0)
            self.assertGreaterEqual(result.respectful_language_score, 0.0)
            self.assertLessEqual(result.respectful_language_score, 1.0)
            self.assertGreaterEqual(result.cultural_sensitivity_score, 0.0)
            self.assertLessEqual(result.cultural_sensitivity_score, 1.0)
    
    def test_empty_input_handling(self):
        """Test handling of empty or minimal input"""
        empty_result = self.analyzer.analyze_cultural_authenticity("")
        self.assertEqual(empty_result.authenticity_score, 0.0)
        
        short_result = self.analyzer.analyze_cultural_authenticity("Yes.")
        self.assertGreaterEqual(short_result.authenticity_score, 0.0)
    
    def test_special_characters_handling(self):
        """Test handling of special characters and non-ASCII text"""
        special_text = """
        Traditional practices include ceremonies with specific symbols: ☮ ☯ ॐ
        Indigenous languages contain concepts like 'ubuntu' that don't translate directly.
        Numbers like π and mathematical concepts vary across cultures.
        """
        
        result = self.analyzer.analyze_cultural_authenticity(special_text)
        self.assertIsInstance(result, CulturalAuthenticityResult)
        self.assertGreaterEqual(result.authenticity_score, 0.0)
    
    def test_cultural_context_integration(self):
        """Test cultural context integration"""
        context = "traditional_scientific"
        result = self.analyzer.analyze_cultural_authenticity(self.respectful_response, context)
        
        self.assertIsInstance(result, CulturalAuthenticityResult)
        self.assertGreaterEqual(result.authenticity_score, 0.0)
    
    def test_thresholds_access(self):
        """Test access to cultural authenticity thresholds"""
        thresholds = self.analyzer.get_cultural_authenticity_thresholds()
        
        self.assertIn('authenticity_score', thresholds)
        self.assertIn('stereotype_severity', thresholds)
        self.assertIn('appropriation_risk', thresholds)
        self.assertIn('bias_score', thresholds)
        
        # Test threshold structure
        auth_thresholds = thresholds['authenticity_score']
        self.assertIn('excellent', auth_thresholds)
        self.assertIn('good', auth_thresholds)
        self.assertIn('acceptable', auth_thresholds)
        self.assertIn('poor', auth_thresholds)


class TestCulturalAuthenticityEdgeCases(unittest.TestCase):
    """Test edge cases for cultural authenticity analysis"""
    
    def setUp(self):
        """Set up edge case test fixtures"""
        self.analyzer = CulturalAuthenticityAnalyzer()
    
    def test_mixed_quality_response(self):
        """Test response with both respectful and problematic elements"""
        mixed_text = """
        According to traditional knowledge systems, indigenous communities have developed
        sophisticated ecological practices. However, these primitive methods need to be 
        validated by modern science to prove their mystical effectiveness. You can buy 
        traditional dreamcatchers online while learning from community elders.
        """
        
        result = self.analyzer.analyze_cultural_authenticity(mixed_text)
        
        # Should detect both positive and negative indicators
        self.assertGreater(len(result.stereotype_indicators), 0)
        self.assertGreater(len(result.appropriation_markers), 0)
        self.assertGreater(result.respectful_language_score, 0.0)
        
        # Overall score should be moderate
        self.assertGreater(result.authenticity_score, 0.2)
        self.assertLess(result.authenticity_score, 0.7)
    
    def test_academic_discussion_response(self):
        """Test academic discussion of cultural topics"""
        academic_text = """
        Research on traditional ecological knowledge (TEK) systems demonstrates
        the importance of contextualizing indigenous practices within their
        cultural frameworks. Scholars emphasize the need for respectful
        collaboration with community members and proper attribution to
        knowledge holders. This requires acknowledging the limitations of
        external perspectives and the complexity of cultural knowledge systems.
        """
        
        result = self.analyzer.analyze_cultural_authenticity(academic_text)
        
        # Should have good respectful language indicators
        self.assertGreater(result.respectful_language_score, 0.4)
        self.assertGreater(result.cultural_sensitivity_score, 0.3)
        # Should have minimal problematic indicators
        self.assertLessEqual(len(result.stereotype_indicators), 1)
        self.assertEqual(len(result.appropriation_markers), 0)
    
    def test_personal_experience_sharing(self):
        """Test response sharing personal cultural experience"""
        personal_text = """
        In my community, our elders teach us about sustainable farming practices
        that have been used for generations. This knowledge is rooted in our
        cultural understanding of land stewardship and community responsibility.
        While I can share some general principles, the full depth of these
        practices is complex and varies within our community.
        """
        
        result = self.analyzer.analyze_cultural_authenticity(personal_text)
        
        # Should be respectful with community attribution
        self.assertGreater(result.respectful_language_score, 0.3)
        self.assertEqual(len(result.appropriation_markers), 0)
        self.assertLessEqual(len(result.stereotype_indicators), 1)
    
    def test_comparative_cultural_discussion(self):
        """Test discussion comparing cultural systems appropriately"""
        comparative_text = """
        Different cultural knowledge systems approach environmental management
        in diverse ways. While Western scientific methods emphasize controlled
        studies and peer review, traditional ecological knowledge systems often
        integrate spiritual, social, and ecological understanding holistically.
        Both approaches offer valuable insights within their respective contexts,
        and neither should be considered superior to the other.
        """
        
        result = self.analyzer.analyze_cultural_authenticity(comparative_text)
        
        # Should avoid hierarchical comparisons and framework imposition
        self.assertLessEqual(len(result.bias_indicators), 2)
        self.assertGreater(result.cultural_sensitivity_score, 0.2)
        
        # Should acknowledge diversity and complexity
        sensitivity_indicators = result.detailed_analysis['sensitivity_analysis']['sensitivity_indicators']
        self.assertTrue(sensitivity_indicators.get('acknowledges_diversity', False))
    
    def test_very_long_response(self):
        """Test handling of very long responses"""
        long_text = self.analyzer._init_pattern_databases  # This is a method, will cause AttributeError
        # Actually create a proper long text
        long_text = """
        Traditional knowledge systems represent sophisticated ways of understanding
        and interacting with the world that have developed over generations.
        """ * 50  # Repeat 50 times
        
        result = self.analyzer.analyze_cultural_authenticity(long_text)
        self.assertIsInstance(result, CulturalAuthenticityResult)
        self.assertGreaterEqual(result.authenticity_score, 0.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)