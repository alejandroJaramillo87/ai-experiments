"""
Test suite for Traditional Knowledge Validator

Tests for traditional knowledge validation including contextualization,
sacred knowledge respect, community attribution, and living tradition recognition.

"""

import unittest
import logging
import sys
import os

# Test imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from evaluator.cultural.tradition_validator import TraditionalKnowledgeValidator, TraditionValidationResult

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestTraditionalKnowledgeValidator(unittest.TestCase):
    """Test traditional knowledge validator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = TraditionalKnowledgeValidator()
        
        # Test scenarios
        self.respectful_response = """
        According to elders in the Pacific Northwest indigenous communities, 
        traditional ecological knowledge encompasses a holistic understanding 
        of the relationship between humans, animals, and the natural world.
        This knowledge system, maintained by the community for generations,
        continues to evolve and adapt while preserving its core principles.
        The practices are rooted in cultural and spiritual contexts that
        cannot be separated from their community origins. Traditional
        practitioners and community members emphasize that this knowledge
        is still actively practiced and relevant in contemporary times.
        """
        
        self.sacred_violation_response = """
        The secret vision quest ceremony involves specific steps that anyone can follow:
        First, you must go to a sacred mountain and perform the ritual exactly as described.
        The private healing ceremonies include these sacred medicines and secret chants
        that I will now share with you. You can buy the ceremonial objects online
        and practice these initiation rites at home for personal spiritual growth.
        """
        
        self.commercialization_response = """
        Traditional sacred items are now available for purchase in our online store!
        You can buy authentic medicine wheels, ceremonial sage, and sacred drums
        to mass-produce for your spiritual business. These traditional ceremonial
        tools can be commercialized and marketed for profit to anyone seeking
        mystical experiences and ancient wisdom.
        """
        
        self.decontextualized_response = """
        You can use sacred smudging practices without any cultural context for your
        personal wellness routine. Anyone can practice traditional ceremonies outside
        of their cultural framework for stress relief. These ancient rituals work
        great when mixed with modern meditation and yoga practices for personal growth.
        """
        
        self.past_tense_response = """
        Indigenous peoples used to believe in these ancient traditions that were
        once practiced in historical times. These old customs were traditional ways
        that were formerly held by primitive cultures. These abandoned practices
        are no longer relevant and have been replaced by modern scientific understanding.
        """
        
        self.well_contextualized_response = """
        Within the cultural framework of Plains indigenous communities, traditional
        knowledge about sustainable hunting practices has been developed over millennia.
        According to community elders, these practices are grounded in spiritual
        relationships with animal spirits and environmental stewardship principles.
        The knowledge is understood within their traditional worldview and continues
        to be practiced by contemporary community members. This represents a living
        tradition that has evolved while maintaining its cultural integrity and
        connection to ancestral teachings passed down through generations.
        """
    
    def test_validator_initialization(self):
        """Test validator initialization"""
        self.assertIsNotNone(self.validator)
        self.assertIsInstance(self.validator.sacred_violations, dict)
        self.assertIsInstance(self.validator.contextualization_markers, dict)
        self.assertIsInstance(self.validator.attribution_patterns, dict)
        self.assertIsInstance(self.validator.living_tradition_markers, dict)
        self.assertIsInstance(self.validator.knowledge_domains, dict)
    
    def test_respectful_response_validation(self):
        """Test validation of respectful response"""
        result = self.validator.validate_traditional_knowledge(self.respectful_response)
        
        self.assertIsInstance(result, TraditionValidationResult)
        self.assertGreater(result.tradition_respect_score, 0.6)
        self.assertGreater(result.contextualization_quality, 0.4)
        self.assertGreater(result.community_attribution_score, 0.3)
        self.assertGreater(result.living_tradition_recognition, 0.2)
        self.assertEqual(len(result.violation_indicators), 0)
        self.assertGreater(len(result.positive_indicators), 1)
    
    def test_sacred_knowledge_violations(self):
        """Test detection of sacred knowledge violations"""
        result = self.validator.validate_traditional_knowledge(self.sacred_violation_response)
        
        self.assertLess(result.tradition_respect_score, 0.5)
        self.assertLess(result.sacred_knowledge_respect, 0.3)
        self.assertGreater(len(result.violation_indicators), 2)
        
        # Check for critical violations
        critical_violations = [v for v in result.violation_indicators if v.get('severity') == 'critical']
        self.assertGreater(len(critical_violations), 0)
        
        # Check specific violation types
        violation_types = [v['type'] for v in result.violation_indicators]
        self.assertIn('sacred_knowledge_violation', violation_types)
    
    def test_commercialization_detection(self):
        """Test detection of inappropriate commercialization"""
        result = self.validator.validate_traditional_knowledge(self.commercialization_response)
        
        self.assertLess(result.tradition_respect_score, 0.4)
        self.assertGreater(len(result.violation_indicators), 1)
        
        # Should detect commercialization violations
        commercialization_found = any(
            'commercialization' in v.get('category', '') 
            for v in result.violation_indicators
        )
        self.assertTrue(commercialization_found)
    
    def test_decontextualization_detection(self):
        """Test detection of decontextualized usage"""
        result = self.validator.validate_traditional_knowledge(self.decontextualized_response)
        
        self.assertLess(result.contextualization_quality, 0.3)
        self.assertGreater(len(result.violation_indicators), 0)
        
        # Should detect decontextualization
        decontextualization_found = any(
            'decontextualization' in v.get('category', '') 
            for v in result.violation_indicators
        )
        self.assertTrue(decontextualization_found)
    
    def test_contextualization_quality_assessment(self):
        """Test assessment of contextualization quality"""
        result = self.validator.validate_traditional_knowledge(self.well_contextualized_response)
        
        self.assertGreater(result.contextualization_quality, 0.6)
        
        contextualization_analysis = result.detailed_analysis['contextualization_analysis']
        categories_present = contextualization_analysis['categories_present']
        
        # Should have multiple contextualization categories
        self.assertGreaterEqual(categories_present, 2)
        
        category_counts = contextualization_analysis['category_counts']
        self.assertGreater(category_counts.get('cultural_context', 0), 0)
    
    def test_community_attribution_assessment(self):
        """Test assessment of community attribution"""
        result = self.validator.validate_traditional_knowledge(self.respectful_response)
        
        self.assertGreater(result.community_attribution_score, 0.2)
        
        attribution_analysis = result.detailed_analysis['attribution_analysis']
        
        # Should find elder attribution
        elder_attribution = attribution_analysis['category_counts'].get('elder_attribution', 0)
        community_attribution = attribution_analysis['category_counts'].get('community_attribution', 0)
        
        self.assertTrue(elder_attribution > 0 or community_attribution > 0)
    
    def test_living_tradition_recognition(self):
        """Test recognition of traditions as living systems"""
        result = self.validator.validate_traditional_knowledge(self.well_contextualized_response)
        
        self.assertGreater(result.living_tradition_recognition, 0.3)
        
        living_tradition_analysis = result.detailed_analysis['living_tradition_analysis']
        
        # Should have present tense markers
        present_tense_count = living_tradition_analysis['category_counts'].get('present_tense', 0)
        self.assertGreater(present_tense_count, 0)
        
        # Should have low past-only markers
        past_only_count = living_tradition_analysis['past_only_count']
        self.assertLessEqual(past_only_count, 2)
    
    def test_past_tense_only_penalty(self):
        """Test penalty for past-tense-only language"""
        result = self.validator.validate_traditional_knowledge(self.past_tense_response)
        
        self.assertLess(result.living_tradition_recognition, 0.2)
        
        living_tradition_analysis = result.detailed_analysis['living_tradition_analysis']
        past_only_count = living_tradition_analysis['past_only_count']
        
        # Should detect multiple past-only markers
        self.assertGreater(past_only_count, 2)
    
    def test_knowledge_domain_analysis(self):
        """Test knowledge domain analysis"""
        spiritual_text = "Traditional ceremonies and sacred rituals connect communities to spiritual understanding."
        result = self.validator.validate_traditional_knowledge(spiritual_text)
        
        domain_analysis = result.detailed_analysis['domain_analysis']
        
        # Should identify spiritual domain
        self.assertEqual(domain_analysis['primary_domain'], 'spiritual')
        self.assertGreater(domain_analysis['domain_confidence'], 0.0)
    
    def test_domain_hint_integration(self):
        """Test integration of domain hints"""
        healing_text = "Traditional medicine practices involve herbal treatments and holistic healing approaches."
        result = self.validator.validate_traditional_knowledge(healing_text, domain_hint="healing")
        
        domain_analysis = result.detailed_analysis['domain_analysis']
        self.assertEqual(domain_analysis['hint_provided'], "healing")
        self.assertTrue(domain_analysis.get('hint_matches_analysis'))
    
    def test_positive_indicators_detection(self):
        """Test detection of positive indicators"""
        result = self.validator.validate_traditional_knowledge(self.well_contextualized_response)
        
        self.assertGreater(len(result.positive_indicators), 2)
        
        positive_types = [p['type'] for p in result.positive_indicators]
        self.assertIn('comprehensive_contextualization', positive_types)
        
        # Check for high strength indicators
        high_strength_found = any(
            p.get('strength') == 'high' 
            for p in result.positive_indicators
        )
        self.assertTrue(high_strength_found)
    
    def test_tradition_respect_score_calculation(self):
        """Test overall tradition respect score calculation"""
        # Test high respect scenario
        high_respect_result = self.validator.validate_traditional_knowledge(self.well_contextualized_response)
        self.assertGreater(high_respect_result.tradition_respect_score, 0.6)
        
        # Test low respect scenario
        low_respect_result = self.validator.validate_traditional_knowledge(self.sacred_violation_response)
        self.assertLess(low_respect_result.tradition_respect_score, 0.5)
    
    def test_validation_bounds(self):
        """Test that all scores are within valid bounds"""
        test_texts = [
            self.respectful_response,
            self.sacred_violation_response,
            self.commercialization_response,
            self.decontextualized_response,
            self.past_tense_response,
            self.well_contextualized_response
        ]
        
        for text in test_texts:
            result = self.validator.validate_traditional_knowledge(text)
            
            self.assertGreaterEqual(result.tradition_respect_score, 0.0)
            self.assertLessEqual(result.tradition_respect_score, 1.0)
            self.assertGreaterEqual(result.contextualization_quality, 0.0)
            self.assertLessEqual(result.contextualization_quality, 1.0)
            self.assertGreaterEqual(result.sacred_knowledge_respect, 0.0)
            self.assertLessEqual(result.sacred_knowledge_respect, 1.0)
            self.assertGreaterEqual(result.community_attribution_score, 0.0)
            self.assertLessEqual(result.community_attribution_score, 1.0)
            self.assertGreaterEqual(result.living_tradition_recognition, 0.0)
            self.assertLessEqual(result.living_tradition_recognition, 1.0)
    
    def test_empty_input_handling(self):
        """Test handling of empty input"""
        empty_result = self.validator.validate_traditional_knowledge("")
        self.assertEqual(empty_result.tradition_respect_score, 0.0)
        
        short_result = self.validator.validate_traditional_knowledge("Traditional knowledge.")
        self.assertGreaterEqual(short_result.tradition_respect_score, 0.0)
    
    def test_overall_assessment_generation(self):
        """Test overall assessment generation"""
        # Test excellent assessment
        excellent_result = self.validator.validate_traditional_knowledge(self.well_contextualized_response)
        excellent_assessment = excellent_result.detailed_analysis['overall_assessment']
        self.assertIn(excellent_assessment['assessment_level'], ['excellent', 'good'])
        
        # Test problematic assessment
        problematic_result = self.validator.validate_traditional_knowledge(self.sacred_violation_response)
        problematic_assessment = problematic_result.detailed_analysis['overall_assessment']
        self.assertIn(problematic_assessment['assessment_level'], ['critical_issues', 'problematic', 'concerning'])
        self.assertTrue(problematic_assessment['requires_cultural_review'])
    
    def test_thresholds_access(self):
        """Test access to validation thresholds"""
        thresholds = self.validator.get_tradition_validation_thresholds()
        
        self.assertIn('tradition_respect', thresholds)
        self.assertIn('contextualization_quality', thresholds)
        self.assertIn('sacred_respect', thresholds)
        self.assertIn('attribution', thresholds)
        
        # Test threshold structure
        respect_thresholds = thresholds['tradition_respect']
        self.assertIn('excellent', respect_thresholds)
        self.assertIn('good', respect_thresholds)
        self.assertIn('acceptable', respect_thresholds)
        self.assertIn('concerning', respect_thresholds)


class TestTraditionalKnowledgeValidatorEdgeCases(unittest.TestCase):
    """Test edge cases for traditional knowledge validation"""
    
    def setUp(self):
        """Set up edge case test fixtures"""
        self.validator = TraditionalKnowledgeValidator()
    
    def test_mixed_quality_response(self):
        """Test response with mixed quality indicators"""
        mixed_text = """
        According to community elders, traditional healing practices involve
        sacred ceremonies that you can buy online and practice at home.
        These living traditions continue to evolve while maintaining their
        secret knowledge that anyone can learn from books.
        """
        
        result = self.validator.validate_traditional_knowledge(mixed_text)
        
        # Should have both positive and negative indicators
        self.assertGreater(len(result.positive_indicators), 0)
        self.assertGreater(len(result.violation_indicators), 0)
        
        # Should have moderate scores
        self.assertGreater(result.tradition_respect_score, 0.2)
        self.assertLess(result.tradition_respect_score, 0.8)
    
    def test_academic_discussion(self):
        """Test academic discussion of traditional knowledge"""
        academic_text = """
        Anthropological studies of traditional ecological knowledge systems
        examine the complex relationships between cultural practices and
        environmental management. Researchers emphasize the importance of
        collaborative methodologies and community-based participatory research
        when studying indigenous knowledge systems. This work requires
        respectful engagement with knowledge holders and recognition of
        intellectual sovereignty.
        """
        
        result = self.validator.validate_traditional_knowledge(academic_text)
        
        # Should have good contextualization without violations
        self.assertGreater(result.contextualization_quality, 0.3)
        self.assertEqual(len(result.violation_indicators), 0)
        self.assertGreater(result.community_attribution_score, 0.1)
    
    def test_cultural_revitalization_discussion(self):
        """Test discussion of cultural revitalization"""
        revitalization_text = """
        Language revitalization programs are helping communities reclaim
        traditional knowledge that was disrupted by historical policies.
        Young people are learning from elders to continue cultural practices
        while adapting them for contemporary contexts. These efforts recognize
        that traditional knowledge is not static but continues to evolve
        within community frameworks.
        """
        
        result = self.validator.validate_traditional_knowledge(revitalization_text)
        
        # Should recognize living tradition aspects
        self.assertGreater(result.living_tradition_recognition, 0.4)
        # Should have good contextualization
        self.assertGreater(result.contextualization_quality, 0.3)
        # Should recognize community involvement
        self.assertGreater(result.community_attribution_score, 0.2)
    
    def test_comparative_knowledge_systems(self):
        """Test discussion comparing knowledge systems"""
        comparative_text = """
        Traditional knowledge systems and Western scientific approaches
        both contribute valuable insights to environmental understanding.
        Indigenous knowledge holders emphasize holistic relationships,
        while scientific methods focus on controlled experimentation.
        Both systems have their own validity within their cultural contexts
        and should be respected as different ways of knowing.
        """
        
        result = self.validator.validate_traditional_knowledge(comparative_text)
        
        # Should have reasonable contextualization
        self.assertGreater(result.contextualization_quality, 0.2)
        # Should avoid major violations
        self.assertLessEqual(len(result.violation_indicators), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)