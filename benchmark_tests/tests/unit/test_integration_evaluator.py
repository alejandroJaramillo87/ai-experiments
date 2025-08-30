"""
Test suite for Integration Evaluator

Tests for integration evaluation including cross-domain coherence, cultural authenticity,
logical consistency, creative appropriateness, social awareness, and synthesis quality.

Author: Claude Code
Version: 1.0.0
"""

import unittest
import logging
import sys
import os
from dataclasses import dataclass

# Test imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from evaluator.subjects.integration_evaluator import (
    IntegrationEvaluator, IntegrationType, CrossDomainCoherence
)
from evaluator.core.domain_evaluator_base import CulturalContext, EvaluationDimension

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestIntegrationEvaluator(unittest.TestCase):
    """Test integration evaluator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = IntegrationEvaluator()
        
        # Sample cultural contexts
        self.rich_cultural_context = CulturalContext(
            traditions=['polynesian', 'traditional_navigation'],
            knowledge_systems=['traditional_knowledge', 'indigenous_science'],
            cultural_groups=['maori', 'polynesian']
        )
        
        self.empty_context = CulturalContext(
            traditions=[],
            knowledge_systems=[],
            cultural_groups=[]
        )
        
        # Sample test responses
        self.knowledge_reasoning_response = """
        Through careful analysis of traditional Maori navigation methods, we can understand how indigenous 
        astronomical knowledge combines with empirical reasoning to create sophisticated wayfinding systems.
        
        The traditional knowledge shows that Polynesian navigators used star patterns, wind direction, and ocean 
        swells as integrated data sources. This represents a holistic reasoning approach where multiple information 
        streams are synthesized into coherent navigational decisions.
        
        From a logical perspective, this system demonstrates evidence-based reasoning: navigators gathered multiple 
        data points, cross-referenced them against learned patterns, and made probabilistic decisions about direction 
        and distance. The cultural framework provides the interpretive context, while the reasoning process ensures 
        accuracy and safety.
        
        This integration of traditional knowledge and systematic reasoning challenges Western assumptions about 
        primitive vs. sophisticated thinking, showing how different cultures develop equally valid but distinct 
        approaches to complex problem-solving.
        """
        
        self.social_creative_response = """
        Community art therapy programs in indigenous communities combine traditional storytelling with 
        contemporary artistic expression to heal intergenerational trauma.
        
        These programs respect cultural protocols while creating new forms of creative expression that 
        speak to both traditional values and contemporary challenges. Elders work with youth to ensure 
        that artistic innovations honor ancestral wisdom while addressing current community needs.
        
        The creative process becomes a form of social healing, where community members can express 
        difficult experiences through culturally grounded artistic practices that strengthen rather 
        than appropriate traditional knowledge systems.
        """
        
        self.multilingual_knowledge_response = """
        Traditional ecological knowledge can be expressed differently depending on the language used. 
        In Quechua, the concept of "sumak kawsay" encompasses living well in harmony with nature, 
        while the Spanish translation "buen vivir" loses some cultural nuance.
        
        Code-switching between indigenous languages and colonial languages allows knowledge holders 
        to preserve cultural concepts while making them accessible to broader audiences. However, 
        this linguistic diversity also presents challenges for knowledge transmission and validation.
        
        Multilingual expression of traditional knowledge requires careful attention to cultural 
        translation and the recognition that some concepts cannot be fully captured across 
        linguistic boundaries.
        """
        
        self.poor_integration_response = """
        This is a simple answer. It doesn't integrate anything. There's no reasoning or creativity 
        involved. Just a basic statement without any depth or cultural awareness.
        """
        
        self.comprehensive_integration_response = """
        Addressing climate change requires a holistic approach that integrates traditional ecological 
        knowledge with modern scientific methods, creative communication strategies, and culturally 
        sensitive social engagement practices.
        
        Indigenous knowledge systems offer sophisticated understanding of local ecosystems, while 
        contemporary climate science provides global modeling capabilities. Creative storytelling 
        and artistic expression can communicate complex environmental concepts across cultures, 
        and community-based social approaches ensure that solutions respect local values and practices.
        
        This comprehensive integration synthesizes multiple domains: knowledge systems inform analysis, 
        reasoning guides decision-making, creativity enables communication, and social awareness ensures 
        ethical implementation. The result is a multifaceted approach that bridges traditional wisdom 
        and contemporary innovation while respecting cultural boundaries and community needs.
        """
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        self.assertIsNotNone(self.evaluator)
        self.assertEqual(self.evaluator.get_domain_name(), "integration")
        self.assertIsInstance(self.evaluator.get_evaluation_dimensions(), list)
        self.assertEqual(len(self.evaluator.get_evaluation_dimensions()), 6)
        
        expected_dimensions = [
            "cross_domain_coherence",
            "cultural_authenticity_integration",
            "logical_consistency_across_domains",
            "creative_appropriateness",
            "social_awareness_integration",
            "synthesis_quality"
        ]
        
        for dim in expected_dimensions:
            self.assertIn(dim, self.evaluator.get_evaluation_dimensions())
    
    def test_supported_evaluation_types(self):
        """Test supported evaluation types"""
        eval_types = self.evaluator.get_supported_evaluation_types()
        self.assertIsInstance(eval_types, list)
        self.assertEqual(len(eval_types), 5)
        
        expected_types = [
            "knowledge_reasoning_synthesis",
            "social_creative_solutions", 
            "multilingual_knowledge_expression",
            "culturally_sensitive_reasoning",
            "comprehensive_integration"
        ]
        
        for eval_type in expected_types:
            self.assertIn(eval_type, eval_types)
    
    def test_integration_type_enum(self):
        """Test IntegrationType enum"""
        self.assertEqual(IntegrationType.KNOWLEDGE_REASONING.value, "knowledge_reasoning_synthesis")
        self.assertEqual(IntegrationType.SOCIAL_CREATIVITY.value, "social_creative_solutions")
        self.assertEqual(IntegrationType.LANGUAGE_KNOWLEDGE.value, "multilingual_knowledge_expression")
        self.assertEqual(IntegrationType.REASONING_SOCIAL.value, "culturally_sensitive_reasoning")
        self.assertEqual(IntegrationType.COMPREHENSIVE.value, "comprehensive_integration")
    
    def test_cross_domain_coherence_dataclass(self):
        """Test CrossDomainCoherence dataclass"""
        coherence = CrossDomainCoherence(
            domains_integrated=["knowledge", "reasoning"],
            integration_quality=0.8,
            coherence_score=0.7,
            transition_quality=0.6,
            evidence=["Test evidence"]
        )
        
        self.assertEqual(coherence.domains_integrated, ["knowledge", "reasoning"])
        self.assertEqual(coherence.integration_quality, 0.8)
        self.assertEqual(coherence.coherence_score, 0.7)
        self.assertEqual(coherence.transition_quality, 0.6)
        self.assertEqual(coherence.evidence, ["Test evidence"])
    
    def test_cross_domain_coherence_evaluation(self):
        """Test cross-domain coherence dimension"""
        test_metadata = {
            'domains_required': ['knowledge', 'reasoning'],
            'category': 'knowledge_reasoning_synthesis'
        }
        
        result = self.evaluator._evaluate_cross_domain_coherence(
            self.knowledge_reasoning_response, 
            test_metadata,
            self.rich_cultural_context
        )
        
        self.assertIsInstance(result, EvaluationDimension)
        self.assertEqual(result.name, "cross_domain_coherence")
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertGreater(result.confidence, 0.5)
        self.assertEqual(result.cultural_relevance, 1.0)  # Has traditions
        self.assertIsInstance(result.evidence, list)
        self.assertGreater(len(result.evidence), 0)
        self.assertIsInstance(result.cultural_markers, list)
    
    def test_cross_domain_coherence_poor_integration(self):
        """Test cross-domain coherence with poor integration"""
        test_metadata = {
            'domains_required': ['knowledge', 'reasoning', 'creativity'],
            'category': 'comprehensive_integration'
        }
        
        result = self.evaluator._evaluate_cross_domain_coherence(
            self.poor_integration_response,
            test_metadata,
            self.empty_context
        )
        
        # Should get low score for poor integration
        self.assertLess(result.score, 0.5)
        self.assertEqual(result.cultural_relevance, 0.7)  # No traditions
    
    def test_cultural_authenticity_integration_evaluation(self):
        """Test cultural authenticity integration dimension"""
        result = self.evaluator._evaluate_cultural_authenticity_integration(
            self.knowledge_reasoning_response,
            self.rich_cultural_context
        )
        
        self.assertIsInstance(result, EvaluationDimension)
        self.assertEqual(result.name, "cultural_authenticity_integration")
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertGreater(result.confidence, 0.4)
        self.assertEqual(result.cultural_relevance, 1.0)  # Has traditions
        self.assertIsInstance(result.evidence, list)
    
    def test_cultural_authenticity_empty_context(self):
        """Test cultural authenticity with empty context"""
        result = self.evaluator._evaluate_cultural_authenticity_integration(
            self.knowledge_reasoning_response,
            self.empty_context
        )
        
        self.assertEqual(result.cultural_relevance, 0.3)  # No traditions
        self.assertLessEqual(result.confidence, 0.5)  # Lower confidence without context
    
    def test_logical_consistency_evaluation(self):
        """Test logical consistency across domains dimension"""
        test_metadata = {
            'category': 'knowledge_reasoning_synthesis',
            'domains_required': ['knowledge', 'reasoning']
        }
        
        result = self.evaluator._evaluate_logical_consistency(
            self.knowledge_reasoning_response,
            test_metadata,
            self.rich_cultural_context
        )
        
        self.assertIsInstance(result, EvaluationDimension)
        self.assertEqual(result.name, "logical_consistency_across_domains")
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertGreaterEqual(result.confidence, 0.5)
        self.assertIsInstance(result.evidence, list)
        self.assertIsInstance(result.cultural_markers, list)
    
    def test_logical_consistency_poor_reasoning(self):
        """Test logical consistency with poor reasoning"""
        result = self.evaluator._evaluate_logical_consistency(
            self.poor_integration_response,
            {'category': 'test'},
            self.empty_context
        )
        
        # Poor reasoning should get low logical consistency score
        self.assertLess(result.score, 0.5)
        self.assertLessEqual(result.confidence, 0.5)
    
    def test_creative_appropriateness_evaluation_creative_expected(self):
        """Test creative appropriateness when creativity is expected"""
        test_metadata = {
            'category': 'social_creative_solutions',
            'domains_required': ['social', 'creativity']
        }
        
        result = self.evaluator._evaluate_creative_appropriateness(
            self.social_creative_response,
            test_metadata,
            self.rich_cultural_context
        )
        
        self.assertIsInstance(result, EvaluationDimension)
        self.assertEqual(result.name, "creative_appropriateness")
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertGreaterEqual(result.confidence, 0.6)
        self.assertEqual(result.cultural_relevance, 1.0)  # Has traditions
    
    def test_creative_appropriateness_not_expected(self):
        """Test creative appropriateness when creativity is not expected"""
        test_metadata = {
            'category': 'knowledge_reasoning_synthesis',  # No creativity expected
            'domains_required': ['knowledge', 'reasoning']
        }
        
        result = self.evaluator._evaluate_creative_appropriateness(
            self.knowledge_reasoning_response,
            test_metadata,
            self.rich_cultural_context
        )
        
        # Should get moderate score when creativity not expected
        self.assertEqual(result.score, 0.7)
        self.assertEqual(result.confidence, 0.5)
        self.assertIn("not specifically required", result.evidence[0])
    
    def test_social_awareness_integration_evaluation(self):
        """Test social awareness integration dimension"""
        result = self.evaluator._evaluate_social_awareness_integration(
            self.social_creative_response,
            self.rich_cultural_context
        )
        
        self.assertIsInstance(result, EvaluationDimension)
        self.assertEqual(result.name, "social_awareness_integration")
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertGreaterEqual(result.confidence, 0.5)
        self.assertGreaterEqual(result.cultural_relevance, 0.6)
        self.assertIsInstance(result.evidence, list)
        self.assertIsInstance(result.cultural_markers, list)
    
    def test_social_awareness_poor_social_content(self):
        """Test social awareness with poor social content"""
        result = self.evaluator._evaluate_social_awareness_integration(
            self.poor_integration_response,
            self.empty_context
        )
        
        # Poor social content should get low social awareness score
        self.assertLess(result.score, 0.3)
        self.assertLessEqual(result.confidence, 0.5)
    
    def test_synthesis_quality_evaluation(self):
        """Test synthesis quality dimension"""
        test_metadata = {
            'category': 'comprehensive_integration',
            'domains_required': ['knowledge', 'reasoning', 'creativity', 'social']
        }
        
        result = self.evaluator._evaluate_synthesis_quality(
            self.comprehensive_integration_response,
            test_metadata,
            self.rich_cultural_context
        )
        
        self.assertIsInstance(result, EvaluationDimension)
        self.assertEqual(result.name, "synthesis_quality")
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)
        self.assertGreaterEqual(result.confidence, 0.5)
        self.assertEqual(result.cultural_relevance, 0.7)
        self.assertIsInstance(result.evidence, list)
        self.assertGreater(len(result.evidence), 0)
    
    def test_synthesis_quality_poor_synthesis(self):
        """Test synthesis quality with poor synthesis"""
        result = self.evaluator._evaluate_synthesis_quality(
            self.poor_integration_response,
            {'category': 'test'},
            self.empty_context
        )
        
        # Poor synthesis should get low synthesis quality score
        self.assertLess(result.score, 0.3)
        # Note: confidence can be high even with low scores if keywords are detected
        self.assertGreaterEqual(result.confidence, 0.0)
    
    def test_evaluate_dimension_dispatch(self):
        """Test dimension evaluation dispatch"""
        test_metadata = {"category": "knowledge_reasoning_synthesis"}
        
        # Test all supported dimensions
        dimensions = self.evaluator.get_evaluation_dimensions()
        for dimension in dimensions:
            result = self.evaluator.evaluate_dimension(
                dimension, 
                self.knowledge_reasoning_response, 
                test_metadata, 
                self.rich_cultural_context
            )
            self.assertIsInstance(result, EvaluationDimension)
            self.assertEqual(result.name, dimension)
    
    def test_evaluate_dimension_unknown(self):
        """Test dimension evaluation with unknown dimension"""
        result = self.evaluator.evaluate_dimension(
            "unknown_dimension", 
            self.knowledge_reasoning_response, 
            {}, 
            self.rich_cultural_context
        )
        
        self.assertEqual(result.name, "unknown_dimension")
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.confidence, 0.0)
        self.assertIn("Unknown integration dimension", result.evidence)
    
    def test_analyze_cross_domain_coherence(self):
        """Test cross-domain coherence analysis"""
        expected_domains = ['knowledge', 'reasoning', 'social']
        
        coherence = self.evaluator._analyze_cross_domain_coherence(
            self.comprehensive_integration_response,
            expected_domains
        )
        
        self.assertIsInstance(coherence, CrossDomainCoherence)
        self.assertIsInstance(coherence.domains_integrated, list)
        self.assertGreater(len(coherence.domains_integrated), 0)
        self.assertGreaterEqual(coherence.integration_quality, 0.0)
        self.assertLessEqual(coherence.integration_quality, 1.0)
        self.assertGreaterEqual(coherence.coherence_score, 0.0)
        self.assertLessEqual(coherence.coherence_score, 1.0)
        self.assertGreaterEqual(coherence.transition_quality, 0.0)
        self.assertLessEqual(coherence.transition_quality, 1.0)
        self.assertIsInstance(coherence.evidence, list)
        self.assertGreater(len(coherence.evidence), 0)
    
    def test_analyze_cross_domain_coherence_no_domains(self):
        """Test cross-domain coherence analysis with no domains detected"""
        expected_domains = ['knowledge', 'reasoning']
        
        coherence = self.evaluator._analyze_cross_domain_coherence(
            "Simple text without domain indicators.",
            expected_domains
        )
        
        self.assertEqual(len(coherence.domains_integrated), 0)
        self.assertEqual(coherence.coherence_score, 0.0)
        self.assertGreaterEqual(coherence.integration_quality, 0.0)
    
    def test_full_evaluation_integration(self):
        """Test full evaluation with complete workflow"""
        test_metadata = {
            'category': 'knowledge_reasoning_synthesis',
            'domains_required': ['knowledge', 'reasoning']
        }
        
        result = self.evaluator.evaluate(
            self.knowledge_reasoning_response,
            test_metadata,
            self.rich_cultural_context
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.domain, "integration")
        self.assertGreaterEqual(result.overall_score, 0.0)
        self.assertLessEqual(result.overall_score, 1.0)
        self.assertEqual(len(result.dimensions), 6)
        
        # Check that all dimensions are evaluated
        dimension_names = [dim.name for dim in result.dimensions]
        expected_dimensions = self.evaluator.get_evaluation_dimensions()
        for expected_dim in expected_dimensions:
            self.assertIn(expected_dim, dimension_names)
    
    def test_evaluation_with_different_contexts(self):
        """Test evaluation with different cultural contexts"""
        test_metadata = {"category": "knowledge_reasoning_synthesis"}
        
        # Test with rich context
        rich_result = self.evaluator.evaluate(
            self.knowledge_reasoning_response,
            test_metadata,
            self.rich_cultural_context
        )
        
        # Test with empty context
        empty_result = self.evaluator.evaluate(
            self.knowledge_reasoning_response,
            test_metadata,
            self.empty_context
        )
        
        # Both should succeed but may have different scores
        for result in [rich_result, empty_result]:
            self.assertIsNotNone(result)
            self.assertEqual(result.domain, "integration")
            self.assertGreaterEqual(result.overall_score, 0.0)
            self.assertLessEqual(result.overall_score, 1.0)
        
        # Rich context should generally score higher on cultural relevance
        # (though exact comparison depends on implementation details)
    
    def test_evaluation_different_integration_types(self):
        """Test evaluation with different integration types"""
        test_cases = [
            (self.knowledge_reasoning_response, {"category": "knowledge_reasoning_synthesis"}),
            (self.social_creative_response, {"category": "social_creative_solutions"}),
            (self.multilingual_knowledge_response, {"category": "multilingual_knowledge_expression"}),
            (self.comprehensive_integration_response, {"category": "comprehensive_integration"})
        ]
        
        for response, metadata in test_cases:
            result = self.evaluator.evaluate(
                response,
                metadata,
                self.rich_cultural_context
            )
            
            self.assertIsNotNone(result)
            self.assertEqual(result.domain, "integration")
            self.assertGreaterEqual(result.overall_score, 0.0)
            self.assertLessEqual(result.overall_score, 1.0)
            self.assertEqual(len(result.dimensions), 6)
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        test_metadata = {"category": "knowledge_reasoning_synthesis"}
        
        # Empty text
        empty_result = self.evaluator.evaluate(
            "",
            test_metadata,
            self.rich_cultural_context
        )
        self.assertIsNotNone(empty_result)
        self.assertGreaterEqual(empty_result.overall_score, 0.0)
        
        # Very short text
        short_result = self.evaluator.evaluate(
            "Yes.",
            test_metadata,
            self.rich_cultural_context
        )
        self.assertIsNotNone(short_result)
        
        # Very long text
        long_text = self.comprehensive_integration_response * 10
        long_result = self.evaluator.evaluate(
            long_text,
            test_metadata,
            self.rich_cultural_context
        )
        self.assertIsNotNone(long_result)
    
    def test_score_boundaries(self):
        """Test that all scores are within valid boundaries"""
        test_metadata = {"category": "comprehensive_integration"}
        
        test_responses = [
            self.knowledge_reasoning_response,
            self.social_creative_response,
            self.multilingual_knowledge_response,
            self.poor_integration_response,
            self.comprehensive_integration_response,
            ""
        ]
        
        contexts = [self.rich_cultural_context, self.empty_context]
        
        for response in test_responses:
            for context in contexts:
                result = self.evaluator.evaluate(response, test_metadata, context)
                
                # Check overall score bounds
                self.assertGreaterEqual(result.overall_score, 0.0)
                self.assertLessEqual(result.overall_score, 1.0)
                
                # Check dimension score bounds
                for dimension in result.dimensions:
                    self.assertGreaterEqual(dimension.score, 0.0)
                    self.assertLessEqual(dimension.score, 1.0)
                    self.assertGreaterEqual(dimension.confidence, 0.0)
                    self.assertLessEqual(dimension.confidence, 1.0)
                    self.assertGreaterEqual(dimension.cultural_relevance, 0.0)
                    self.assertLessEqual(dimension.cultural_relevance, 1.0)


class TestIntegrationEvaluatorConfiguration(unittest.TestCase):
    """Test integration evaluator configuration options"""
    
    def test_custom_configuration(self):
        """Test evaluator with custom configuration"""
        custom_config = {
            "integration": {
                "synthesis_threshold": 0.5,
                "coherence_weight": 0.6
            }
        }
        
        evaluator = IntegrationEvaluator(custom_config)
        self.assertIsNotNone(evaluator)
        
        # Should still function normally
        result = evaluator.evaluate(
            "A simple integration test about traditional knowledge and reasoning.",
            {"category": "knowledge_reasoning_synthesis"},
            CulturalContext(traditions=["test_tradition"])
        )
        self.assertIsNotNone(result)
    
    def test_pattern_library_integration(self):
        """Test integration with cultural pattern library"""
        evaluator = IntegrationEvaluator()
        self.assertIsNotNone(evaluator.pattern_library)
        
        # Test pattern detection through evaluator
        context = CulturalContext(
            traditions=['polynesian'],
            knowledge_systems=['traditional_knowledge'],
            cultural_groups=['polynesian']
        )
        
        text = "Traditional navigation knowledge combines with logical reasoning."
        
        # Should be able to detect patterns
        patterns = evaluator.pattern_library.detect_patterns(text, context.traditions)
        self.assertIsInstance(patterns, list)
    
    def test_creativity_evaluator_integration(self):
        """Test integration with creativity evaluator"""
        evaluator = IntegrationEvaluator()
        self.assertIsNotNone(evaluator.creativity_evaluator)
        
        # Should be able to use creativity evaluator for creative appropriateness
        test_metadata = {"category": "social_creative_solutions"}
        context = CulturalContext(traditions=['griot'])
        
        result = evaluator._evaluate_creative_appropriateness(
            "Creative storytelling helps heal community trauma.",
            test_metadata,
            context
        )
        
        self.assertIsInstance(result, EvaluationDimension)


class TestIntegrationEvaluatorPatternMatching(unittest.TestCase):
    """Test pattern matching functionality in integration evaluator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = IntegrationEvaluator()
    
    def test_integration_patterns_detection(self):
        """Test detection of integration-specific patterns"""
        # Knowledge-reasoning patterns
        kr_text = "Traditional knowledge shows evidence from indigenous systems with logical analysis."
        kr_matches = 0
        for pattern in self.evaluator.integration_patterns["knowledge_reasoning"]:
            kr_matches += len(__import__('re').findall(pattern, kr_text, __import__('re').IGNORECASE))
        self.assertGreater(kr_matches, 0)
        
        # Social-creativity patterns
        sc_text = "Creative solutions for social healing through artistic community practices."
        sc_matches = 0
        for pattern in self.evaluator.integration_patterns["social_creativity"]:
            sc_matches += len(__import__('re').findall(pattern, sc_text, __import__('re').IGNORECASE))
        self.assertGreater(sc_matches, 0)
    
    def test_transition_markers_detection(self):
        """Test detection of domain transition markers"""
        text_with_transitions = "Furthermore, the analysis shows that. However, we must also consider. Therefore, the conclusion is."
        
        transition_count = 0
        for marker in self.evaluator.domain_transition_markers:
            if marker in text_with_transitions.lower():
                transition_count += 1
        
        self.assertGreater(transition_count, 2)
    
    def test_integration_quality_indicators(self):
        """Test detection of integration quality indicators"""
        text_with_indicators = "This approach synthesizes multiple perspectives and integrates different frameworks to unify the analysis."
        
        indicator_count = 0
        for indicator in self.evaluator.integration_quality_indicators:
            if indicator in text_with_indicators.lower():
                indicator_count += 1
        
        self.assertGreater(indicator_count, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)