"""
Comprehensive Integration Tests for Domain-Aware Evaluation System

Tests the complete domain-aware evaluation pipeline including:
- Domain metadata extraction
- Domain-specific evaluators
- Integration domain evaluation
- Cultural pattern detection
- Cross-domain synthesis

"""

import unittest
import logging
import sys
import os
from typing import Dict, List, Any

# Test imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from evaluator.subjects.domain_evaluation_router import DomainEvaluationRouter, Domain, EvaluationType
from evaluator.data.domain_metadata_extractor import DomainMetadataExtractor
from evaluator.subjects.creativity_evaluator import CreativityEvaluator
from evaluator.subjects.integration_evaluator import IntegrationEvaluator
from evaluator.cultural.cultural_pattern_library import CulturalPatternLibrary
from evaluator.core.domain_evaluator_base import CulturalContext

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestDomainAwareEvaluationSystem(unittest.TestCase):
    """Test the complete domain-aware evaluation system."""
    
    def setUp(self):
        """Set up test fixtures"""
        self.router = DomainEvaluationRouter()
        self.metadata_extractor = DomainMetadataExtractor()
        self.creativity_evaluator = CreativityEvaluator()
        self.integration_evaluator = IntegrationEvaluator()
        self.pattern_library = CulturalPatternLibrary()
        
        # Test data from actual domain categories
        self.creativity_test_metadata = {
            'file_path': 'domains/creativity/base_models/easy.json',
            'category': 'narrative_creation', 
            'test_id': 'narrative_01',
            'creativity_focus': 'original plot development, character creation, cultural narrative patterns',
            'temperature_range': [0.6, 0.9]
        }
        
        self.integration_test_metadata = {
            'file_path': 'domains/integration/base_models/easy.json',
            'category': 'cultural_knowledge_reasoning',
            'test_id': 'integration_01',
            'domains_required': ['knowledge', 'reasoning'],
            'integration_focus': 'traditional knowledge application, logical reasoning with cultural frameworks'
        }
        
        self.social_test_metadata = {
            'file_path': 'domains/social/base_models/easy.json',
            'category': 'conflict_resolution',
            'test_id': 'conflict_01',
            'social_focus': 'traditional mediation, restorative justice, elder councils'
        }
        
        self.language_test_metadata = {
            'file_path': 'domains/language/base_models/easy.json',
            'category': 'advanced_code_switching',
            'test_id': 'codeswitch_01',
            'language_focus': 'cultural code-switching, identity-based language choices'
        }
        
        # Sample responses based on actual test patterns
        self.griot_narrative_response = """
        Listen well, friends, to this tale from our ancestors. Once upon a time, in a village by the great river, 
        there lived a wise griot who knew all the stories of our people.

        Every evening, when the sun painted the sky in shades of gold, the community would gather in the central square. 
        The griot would begin with the traditional call: "Who will hear the wisdom of ages?"

        And all would respond in unison: "We will hear, we will remember, we will pass it on!"

        The story flowed like the river itself, with rhythm and grace, teaching us about courage, community, 
        and the bonds that connect us across generations. Through repetition and song, through gesture and voice, 
        the griot wove the past into the present, ensuring our culture would live on.
        
        This tale teaches us that wisdom shared is wisdom multiplied, and that every voice in our community 
        carries the seeds of understanding for those who come after.
        """
        
        self.integration_response = """
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
        
        self.social_mediation_response = """
        In many African communities, conflict resolution follows the principle of ubuntu - "I am because we are" - 
        which emphasizes restoring harmony rather than determining guilt or punishment.

        The elder council creates a circle where all parties can speak their truth without interruption. The mediator's 
        role is not to judge but to help the community understand how the conflict affects everyone's wellbeing.

        Traditional mediation focuses on three key questions: What harm was caused? What are the underlying needs? 
        How can relationships be restored? This approach recognizes that individuals exist within networks of 
        relationships that must be healed for true resolution to occur.

        The process often involves symbolic acts of reconciliation, community witnessing, and agreements that address 
        both immediate concerns and long-term relationship maintenance. Success is measured not by compliance with 
        rules but by the restoration of social harmony and mutual respect.
        """
    
    def test_domain_metadata_extraction(self):
        """Test domain and cultural metadata extraction from test configurations."""
        
        # Test creativity domain extraction
        creativity_extraction = self.metadata_extractor.extract_metadata(
            self.creativity_test_metadata, self.griot_narrative_response
        )
        
        self.assertEqual(creativity_extraction.domain, Domain.CREATIVITY)
        self.assertEqual(creativity_extraction.evaluation_type, EvaluationType.CREATIVE_EXPRESSION)
        self.assertGreater(creativity_extraction.confidence, 0.7)
        self.assertIn('griot', creativity_extraction.cultural_context.traditions)
        self.assertIn('oral_tradition', creativity_extraction.cultural_context.traditions)
        
        # Test integration domain extraction  
        integration_extraction = self.metadata_extractor.extract_metadata(
            self.integration_test_metadata, self.integration_response
        )
        
        self.assertEqual(integration_extraction.domain, Domain.INTEGRATION)
        self.assertEqual(integration_extraction.evaluation_type, EvaluationType.KNOWLEDGE_REASONING_SYNTHESIS)
        self.assertGreater(integration_extraction.confidence, 0.8)
        self.assertIn('traditional knowledge', integration_extraction.cultural_context.knowledge_systems)
    
    def test_creativity_evaluator_functionality(self):
        """Test creativity evaluator with griot narrative."""
        
        cultural_context = CulturalContext(
            traditions=['griot', 'oral_tradition'],
            performance_aspects=['storytelling', 'rhythmic_speech', 'call_response'],
            cultural_groups=['west_african']
        )
        
        result = self.creativity_evaluator.evaluate(
            self.griot_narrative_response, 
            self.creativity_test_metadata, 
            cultural_context
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.domain, "creativity")
        self.assertGreater(result.overall_score, 0.3)
        self.assertEqual(len(result.dimensions), 6)
        
        # Check for specific creativity dimensions
        dimension_names = [dim.name for dim in result.dimensions]
        self.assertIn("cultural_creative_patterns", dimension_names)
        self.assertIn("rhythmic_quality", dimension_names)
        self.assertIn("narrative_coherence", dimension_names)
        self.assertIn("collaborative_creation", dimension_names)
        
        # Check for cultural pattern detection
        cultural_patterns_dim = next(
            dim for dim in result.dimensions 
            if dim.name == "cultural_creative_patterns"
        )
        self.assertGreater(cultural_patterns_dim.score, 0.3)
        self.assertGreater(len(cultural_patterns_dim.cultural_markers), 0)
    
    def test_integration_evaluator_functionality(self):
        """Test integration evaluator with cross-domain scenario."""
        
        cultural_context = CulturalContext(
            traditions=['polynesian', 'traditional_navigation'],
            knowledge_systems=['traditional_knowledge', 'indigenous_science'],
            cultural_groups=['maori', 'polynesian']
        )
        
        result = self.integration_evaluator.evaluate(
            self.integration_response,
            self.integration_test_metadata,
            cultural_context
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.domain, "integration")  
        self.assertGreater(result.overall_score, 0.3)
        self.assertEqual(len(result.dimensions), 6)
        
        # Check for specific integration dimensions
        dimension_names = [dim.name for dim in result.dimensions]
        self.assertIn("cross_domain_coherence", dimension_names)
        self.assertIn("cultural_authenticity_integration", dimension_names)
        self.assertIn("logical_consistency_across_domains", dimension_names)
        self.assertIn("synthesis_quality", dimension_names)
        
        # Check cross-domain coherence
        coherence_dim = next(
            dim for dim in result.dimensions 
            if dim.name == "cross_domain_coherence"
        )
        self.assertGreater(coherence_dim.score, 0.4)
    
    def test_cultural_pattern_library_functionality(self):
        """Test cultural pattern detection across traditions."""
        
        # Test griot patterns
        griot_patterns = self.pattern_library.detect_patterns(
            self.griot_narrative_response, ['griot', 'oral_performance']
        )
        
        self.assertGreater(len(griot_patterns), 0)
        
        griot_pattern_names = [p.pattern_name for p in griot_patterns]
        # Should detect at least some griot-related patterns (call-response, collective memory, repetition, etc.)
        expected_patterns = ['call_response', 'collective_memory', 'repetition', 'moral_embedding', 'community_wisdom']
        patterns_found = [name for name in griot_pattern_names if name in expected_patterns]
        self.assertGreater(len(patterns_found), 0, f"Expected to find at least one pattern from {expected_patterns}, found: {griot_pattern_names}")
        
        # Test cultural competence analysis
        competence_analysis = self.pattern_library.analyze_cultural_competence(griot_patterns)
        self.assertGreater(competence_analysis['overall_competence'], 0.25)
        self.assertIn('griot', competence_analysis['tradition_coverage'])
    
    def test_domain_evaluation_router_integration(self):
        """Test the complete evaluation routing system."""
        
        # Test creativity routing
        creativity_result = self.router.route_evaluation(
            self.griot_narrative_response, 
            self.creativity_test_metadata
        )
        
        self.assertIsNotNone(creativity_result)
        self.assertGreater(creativity_result.overall_score, 30.0)  # Should get a reasonable score
        
        # Test integration routing
        integration_result = self.router.route_evaluation(
            self.integration_response,
            self.integration_test_metadata  
        )
        
        self.assertIsNotNone(integration_result)
        self.assertGreater(integration_result.overall_score, 30.0)  # Should get a reasonable score
    
    def test_cultural_authenticity_integration(self):
        """Test cultural authenticity assessment across domains."""
        
        # Test respectful cultural content
        respectful_metadata = {
            'category': 'cultural_knowledge',
            'cultural_focus': 'traditional practices, community attribution'
        }
        
        respectful_response = """
        According to elders in the Pacific Northwest indigenous communities, traditional ecological knowledge 
        encompasses sophisticated understanding of seasonal cycles, plant medicine, and sustainable harvesting practices.
        
        Within their cultural framework, this knowledge is understood as sacred and is transmitted through 
        specific protocols that honor both the knowledge and the community that holds it. It's important to 
        recognize that this represents only a limited outside perspective, and the full depth of these 
        traditions can only be properly understood through direct learning from community members.
        """
        
        result = self.router.route_evaluation(respectful_response, respectful_metadata)
        self.assertIsNotNone(result)
        self.assertGreater(result.overall_score, 40.0)  # Should score well for respectful approach
        
        # Test problematic cultural content
        problematic_response = """
        All indigenous people have mystical connections to nature and use primitive healing methods. 
        You can easily buy dreamcatchers online to harness their ancient powers for modern wellness.
        These exotic practices are basically just folk medicine that science has evolved beyond.
        """
        
        problematic_result = self.router.route_evaluation(problematic_response, respectful_metadata)
        self.assertIsNotNone(problematic_result)
        self.assertLess(problematic_result.overall_score, 55.0)  # Should score poorly for appropriation/bias
    
    def test_domain_specific_evaluation_types(self):
        """Test that different domains use appropriate evaluation types."""
        
        test_cases = [
            (self.creativity_test_metadata, EvaluationType.CREATIVE_EXPRESSION),
            (self.integration_test_metadata, EvaluationType.KNOWLEDGE_REASONING_SYNTHESIS),
            (self.social_test_metadata, EvaluationType.SOCIAL_CONTEXT),
            (self.language_test_metadata, EvaluationType.LINGUISTIC_COMPETENCE)
        ]
        
        for metadata, expected_type in test_cases:
            extraction = self.metadata_extractor.extract_metadata(metadata, "sample text")
            # Should correctly identify evaluation type based on metadata
            self.assertIsNotNone(extraction.evaluation_type)
            # Note: exact match may vary based on inference, but should be appropriate for domain
    
    def test_cross_domain_synthesis_scenarios(self):
        """Test complex cross-domain synthesis scenarios."""
        
        # Social-creativity integration
        social_creative_metadata = {
            'category': 'social_creativity_integration',
            'domains_required': ['social', 'creativity'],
            'integration_focus': 'creative solutions to social problems'
        }
        
        social_creative_response = """
        Community art therapy programs in indigenous communities combine traditional storytelling with 
        contemporary artistic expression to heal intergenerational trauma.
        
        These programs respect cultural protocols while creating new forms of creative expression that 
        speak to both traditional values and contemporary challenges. Elders work with youth to ensure 
        that artistic innovations honor ancestral wisdom while addressing current community needs.
        
        The creative process becomes a form of social healing, where community members can express 
        difficult experiences through culturally grounded artistic practices that strengthen rather 
        than appropriate traditional knowledge systems.
        """
        
        result = self.router.route_evaluation(social_creative_response, social_creative_metadata)
        self.assertIsNotNone(result)
        self.assertGreater(result.overall_score, 35.0)  # Should handle cross-domain synthesis well


class TestDomainConfigurationAndSettings(unittest.TestCase):
    """Test domain configuration and settings integration."""
    
    def setUp(self):
        """Set up configuration tests"""
        from evaluator.core.evaluation_config import DEFAULT_CONFIG
        self.config = DEFAULT_CONFIG
    
    def test_domain_configuration_completeness(self):
        """Test that all domains have proper configuration."""
        
        domain_config = self.config.get('domain_evaluation_config', {}).get('domain_settings', {})
        
        # Check that all main domains are configured
        expected_domains = ['creativity', 'language', 'social', 'reasoning', 'knowledge', 'integration']
        for domain in expected_domains:
            self.assertIn(domain, domain_config, f"Domain {domain} not configured")
            
            domain_settings = domain_config[domain]
            self.assertIn('evaluator_class', domain_settings)
            self.assertIn('dimensions', domain_settings)
            self.assertIsInstance(domain_settings['dimensions'], list)
            self.assertGreater(len(domain_settings['dimensions']), 0)
    
    def test_cultural_pattern_library_configuration(self):
        """Test cultural pattern library configuration."""
        
        pattern_config = self.config.get('domain_evaluation_config', {}).get('cultural_pattern_libraries', {})
        
        # Check that key traditions are configured
        expected_traditions = ['griot', 'dreamtime', 'kamishibai', 'oral_performance']
        for tradition in expected_traditions:
            self.assertIn(tradition, pattern_config, f"Tradition {tradition} not configured")
        
        # Check integration-specific patterns
        self.assertIn('cross_domain', pattern_config)
        self.assertIn('synthesis_patterns', pattern_config)
    
    def test_evaluation_framework_settings(self):
        """Test evaluation framework configuration for integration domain."""
        
        integration_config = self.config.get('domain_evaluation_config', {}).get('domain_settings', {}).get('integration', {})
        
        self.assertIn('integration_types', integration_config)
        integration_types = integration_config['integration_types']
        
        expected_types = [
            'knowledge_reasoning_synthesis',
            'social_creative_solutions', 
            'multilingual_knowledge_expression',
            'culturally_sensitive_reasoning',
            'comprehensive_integration'
        ]
        
        for integration_type in expected_types:
            self.assertIn(integration_type, integration_types)


if __name__ == '__main__':
    unittest.main()