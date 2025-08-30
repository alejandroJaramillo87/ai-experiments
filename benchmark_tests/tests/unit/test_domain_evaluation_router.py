"""
Unit tests for DomainEvaluationRouter and DomainMetadataExtractor.

Tests the routing logic, domain detection, evaluation type extraction,
cultural context analysis, and result integration.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import logging

from evaluator.subjects.domain_evaluation_router import (
    Domain,
    EvaluationType, 
    DomainEvaluationResult,
    IntegratedEvaluationResult,
    DomainMetadataExtractor,
    DomainEvaluationRouter
)


class TestDomainMetadataExtractor(unittest.TestCase):
    """Test cases for DomainMetadataExtractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = DomainMetadataExtractor()
    
    def test_init_domain_patterns(self):
        """Test initialization of domain patterns."""
        self.assertIsNotNone(self.extractor.domain_patterns)
        self.assertIsNotNone(self.extractor.evaluation_type_patterns)
        
        # Check most domains have patterns (Integration may not be in patterns)
        expected_domains = [Domain.CREATIVITY, Domain.KNOWLEDGE, Domain.LANGUAGE, 
                           Domain.REASONING, Domain.SOCIAL]
        for domain in expected_domains:
            self.assertIn(domain, self.extractor.domain_patterns)
            self.assertIsInstance(self.extractor.domain_patterns[domain], list)
            self.assertGreater(len(self.extractor.domain_patterns[domain]), 0)
    
    def test_extract_domain_direct_specification(self):
        """Test domain extraction with direct domain specification."""
        test_metadata = {'domain': 'creativity'}
        domain = self.extractor.extract_domain(test_metadata)
        self.assertEqual(domain, Domain.CREATIVITY)
        
        test_metadata = {'domain': 'REASONING'}
        domain = self.extractor.extract_domain(test_metadata)
        self.assertEqual(domain, Domain.REASONING)
    
    def test_extract_domain_from_content(self):
        """Test domain extraction from test content."""
        # Creativity domain
        test_metadata = {
            'name': 'narrative storytelling test',
            'category': 'creative expression',
            'description': 'griot tradition performance'
        }
        domain = self.extractor.extract_domain(test_metadata)
        self.assertEqual(domain, Domain.CREATIVITY)
        
        # Reasoning domain
        test_metadata = {
            'name': 'logic reasoning test',
            'category': 'basic_logic',
            'description': 'chain of thought inference'
        }
        domain = self.extractor.extract_domain(test_metadata)
        self.assertEqual(domain, Domain.REASONING)
        
        # Language domain
        test_metadata = {
            'name': 'multilingual competence',
            'category': 'linguistic test',
            'description': 'code switching pragmatic'
        }
        domain = self.extractor.extract_domain(test_metadata)
        self.assertEqual(domain, Domain.LANGUAGE)
    
    def test_extract_domain_no_match(self):
        """Test domain extraction with no matching patterns."""
        test_metadata = {
            'name': 'unknown test',
            'category': 'random category',
            'description': 'unrelated content'
        }
        domain = self.extractor.extract_domain(test_metadata)
        self.assertIsNone(domain)
    
    def test_extract_evaluation_type_from_content(self):
        """Test evaluation type extraction from content."""
        # Creative expression
        test_metadata = {
            'name': 'narrative performance test',
            'description': 'storytelling collaborative'
        }
        eval_type = self.extractor.extract_evaluation_type(test_metadata, Domain.CREATIVITY)
        self.assertEqual(eval_type, EvaluationType.CREATIVE_EXPRESSION)
        
        # Linguistic competence  
        test_metadata = {
            'name': 'multilingual code switching',
            'description': 'register pragmatic communication'
        }
        eval_type = self.extractor.extract_evaluation_type(test_metadata, Domain.LANGUAGE)
        self.assertEqual(eval_type, EvaluationType.LINGUISTIC_COMPETENCE)
    
    def test_extract_evaluation_type_default_mapping(self):
        """Test evaluation type extraction using default domain mapping."""
        test_metadata = {'name': 'generic test', 'description': 'no specific patterns'}
        
        # Test default mappings
        self.assertEqual(
            self.extractor.extract_evaluation_type(test_metadata, Domain.CREATIVITY),
            EvaluationType.CREATIVE_EXPRESSION
        )
        self.assertEqual(
            self.extractor.extract_evaluation_type(test_metadata, Domain.LANGUAGE),
            EvaluationType.LINGUISTIC_COMPETENCE
        )
        self.assertEqual(
            self.extractor.extract_evaluation_type(test_metadata, Domain.SOCIAL),
            EvaluationType.SOCIAL_CONTEXT
        )
        self.assertEqual(
            self.extractor.extract_evaluation_type(test_metadata, Domain.REASONING),
            EvaluationType.GENERAL_REASONING
        )
        self.assertEqual(
            self.extractor.extract_evaluation_type(test_metadata, Domain.KNOWLEDGE),
            EvaluationType.TRADITIONAL_KNOWLEDGE
        )
    
    def test_extract_cultural_context_traditions(self):
        """Test cultural context extraction for traditions."""
        test_metadata = {
            'name': 'griot storytelling test',
            'description': 'african oral tradition narrative',
            'prompt': 'Tell a story using griot techniques'
        }
        
        cultural_context = self.extractor.extract_cultural_context(test_metadata)
        
        self.assertIn('traditions', cultural_context)
        self.assertIn('african', cultural_context['traditions'])
    
    def test_extract_cultural_context_knowledge_systems(self):
        """Test cultural context extraction for knowledge systems."""
        test_metadata = {
            'name': 'traditional knowledge test',
            'description': 'oral storytelling ceremonial practices',
            'prompt': 'Explain traditional folk wisdom'
        }
        
        cultural_context = self.extractor.extract_cultural_context(test_metadata)
        
        self.assertIn('knowledge_systems', cultural_context)
        self.assertIn('traditional', cultural_context['knowledge_systems'])
        self.assertIn('oral_tradition', cultural_context['knowledge_systems'])
        self.assertIn('ceremonial', cultural_context['knowledge_systems'])
    
    def test_extract_cultural_context_performance_aspects(self):
        """Test cultural context extraction for performance aspects."""
        test_metadata = {
            'name': 'interactive performance',
            'description': 'call and response rhythmic collaborative',
            'prompt': 'Create a community engagement piece'
        }
        
        cultural_context = self.extractor.extract_cultural_context(test_metadata)
        
        self.assertIn('performance_aspects', cultural_context)
        self.assertIn('interactive', cultural_context['performance_aspects'])
        self.assertIn('rhythmic', cultural_context['performance_aspects'])
        self.assertIn('collaborative', cultural_context['performance_aspects'])
    
    def test_extract_cultural_context_multiple_traditions(self):
        """Test cultural context extraction with multiple traditions."""
        test_metadata = {
            'name': 'multicultural test',
            'description': 'griot and kamishibai storytelling dreamtime',
            'prompt': 'Combine african, japanese, and aboriginal traditions'
        }
        
        cultural_context = self.extractor.extract_cultural_context(test_metadata)
        
        traditions = cultural_context['traditions']
        self.assertIn('african', traditions)
        self.assertIn('east_asian', traditions)
        self.assertIn('indigenous_australian', traditions)


class TestDomainEvaluationRouter(unittest.TestCase):
    """Test cases for DomainEvaluationRouter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.router = DomainEvaluationRouter()
        
        # Mock evaluator results
        self.mock_domain_result = DomainEvaluationResult(
            domain=Domain.CREATIVITY,
            evaluation_type=EvaluationType.CREATIVE_EXPRESSION,
            primary_score=0.75,
            detailed_metrics={'creativity': 0.8, 'cultural_authenticity': 0.7},
            cultural_indicators=[{'tradition': 'griot', 'score': 0.8}],
            confidence=0.85,
            analysis_details={'strengths': ['good narrative flow']}
        )
    
    def test_router_initialization(self):
        """Test router initialization."""
        self.assertIsNotNone(self.router.metadata_extractor)
        self.assertIsInstance(self.router.metadata_extractor, DomainMetadataExtractor)
        
        # Check evaluators are initially None (lazy loading)
        self.assertIsNone(self.router._creative_evaluator)
        self.assertIsNone(self.router._linguistic_evaluator)
        self.assertIsNone(self.router._social_evaluator)
        self.assertIsNone(self.router._reasoning_evaluator)
        self.assertIsNone(self.router._knowledge_evaluator)
        self.assertIsNone(self.router._integration_evaluator)
    
    @patch('evaluator.subjects.domain_evaluation_router.logger')
    def test_get_creative_evaluator_import_error(self, mock_logger):
        """Test creative evaluator getter with import error."""
        with patch('builtins.__import__', side_effect=ImportError):
            evaluator = self.router.get_creative_evaluator()
            self.assertIsNone(evaluator)
            mock_logger.warning.assert_called_once_with("CreativeExpressionEvaluator not available")
    
    def test_get_creative_evaluator_success(self):
        """Test creative evaluator getter with successful import."""
        # Mock the evaluator instance directly
        mock_evaluator_instance = Mock()
        
        # Since we can't easily mock the dynamic import, let's just test the caching behavior
        # by setting the evaluator directly
        self.router._creative_evaluator = mock_evaluator_instance
        
        evaluator = self.router.get_creative_evaluator()
        self.assertEqual(evaluator, mock_evaluator_instance)
        
        # Test caching - should return same instance
        evaluator2 = self.router.get_creative_evaluator()
        self.assertEqual(evaluator, evaluator2)
    
    def test_get_domain_weight(self):
        """Test domain weight calculation."""
        # Primary domain gets highest weight
        weight = self.router._get_domain_weight(
            Domain.CREATIVITY, 
            EvaluationType.CREATIVE_EXPRESSION, 
            Domain.CREATIVITY
        )
        self.assertEqual(weight, 0.7)
        
        # Cultural evaluation for cultural domain
        weight = self.router._get_domain_weight(
            Domain.KNOWLEDGE,
            EvaluationType.TRADITIONAL_KNOWLEDGE,
            Domain.CREATIVITY
        )
        self.assertEqual(weight, 0.5)
        
        # Secondary domain
        weight = self.router._get_domain_weight(
            Domain.LANGUAGE,
            EvaluationType.LINGUISTIC_COMPETENCE,
            Domain.REASONING
        )
        self.assertEqual(weight, 0.3)
    
    def test_identify_strengths(self):
        """Test strength identification from domain results."""
        high_score_result = DomainEvaluationResult(
            domain=Domain.CREATIVITY,
            evaluation_type=EvaluationType.CREATIVE_EXPRESSION,
            primary_score=0.85,  # High score
            detailed_metrics={'narrative_flow': 0.9, 'cultural_authenticity': 0.7},
            cultural_indicators=[],
            confidence=0.8,
            analysis_details={}
        )
        
        strengths = self.router._identify_strengths([high_score_result])
        
        self.assertIn('Strong Creativity competency', strengths)
        self.assertIn('Excellent Narrative Flow', strengths)
        self.assertLessEqual(len(strengths), 5)
    
    def test_identify_improvements(self):
        """Test improvement identification from domain results."""
        low_score_result = DomainEvaluationResult(
            domain=Domain.REASONING,
            evaluation_type=EvaluationType.GENERAL_REASONING,
            primary_score=0.4,  # Low score
            detailed_metrics={'logical_consistency': 0.3, 'inference_quality': 0.5},
            cultural_indicators=[],
            confidence=0.6,
            analysis_details={}
        )
        
        improvements = self.router._identify_improvements([low_score_result])
        
        self.assertIn('Enhance Reasoning competency', improvements)
        self.assertIn('Improve Logical Consistency', improvements)
        self.assertLessEqual(len(improvements), 5)
    
    def test_generate_recommendations_cultural(self):
        """Test recommendation generation for cultural content."""
        cultural_result = DomainEvaluationResult(
            domain=Domain.CREATIVITY,
            evaluation_type=EvaluationType.CREATIVE_EXPRESSION,
            primary_score=0.4,  # Low score to trigger tradition-specific recommendation
            detailed_metrics={'cultural_authenticity': 0.4},  # Low cultural score
            cultural_indicators=[],
            confidence=0.7,
            analysis_details={}
        )
        
        cultural_context = {'traditions': ['african', 'east_asian']}
        recommendations = self.router._generate_recommendations([cultural_result], cultural_context)
        
        self.assertTrue(any('culturally authentic' in rec for rec in recommendations))
        self.assertTrue(any('african, east_asian' in rec for rec in recommendations))
        self.assertLessEqual(len(recommendations), 7)
    
    def test_generate_recommendations_domain_specific(self):
        """Test domain-specific recommendation generation."""
        creative_result = DomainEvaluationResult(
            domain=Domain.CREATIVITY,
            evaluation_type=EvaluationType.CREATIVE_EXPRESSION,
            primary_score=0.5,  # Low score
            detailed_metrics={},
            cultural_indicators=[],
            confidence=0.7,
            analysis_details={}
        )
        
        recommendations = self.router._generate_recommendations([creative_result], {})
        
        self.assertTrue(any('creative expression' in rec for rec in recommendations))
    
    def test_integrate_domain_results_empty(self):
        """Test domain result integration with empty results."""
        result = self.router._integrate_domain_results([], Domain.CREATIVITY, {})
        
        self.assertIsInstance(result, IntegratedEvaluationResult)
        self.assertEqual(result.overall_score, 0.0)
        self.assertEqual(result.cultural_authenticity_score, 0.0)
        self.assertEqual(len(result.domain_scores), 0)
        self.assertIn('error', result.synthesis_analysis)
    
    def test_integrate_domain_results_single(self):
        """Test domain result integration with single result."""
        domain_results = [self.mock_domain_result]
        
        result = self.router._integrate_domain_results(
            domain_results, 
            Domain.CREATIVITY, 
            {'traditions': ['african']}
        )
        
        self.assertIsInstance(result, IntegratedEvaluationResult)
        self.assertAlmostEqual(result.overall_score, 75.0, places=1)  # 0.75 * 100
        # Cultural authenticity score comes from results with 'cultural_authenticity' in detailed_metrics
        self.assertEqual(result.cultural_authenticity_score, 0.7)  # Should match our mock
        self.assertIn(Domain.CREATIVITY, result.domain_scores)
        self.assertEqual(result.domain_scores[Domain.CREATIVITY], 75.0)
        
        # Check synthesis analysis
        self.assertEqual(result.synthesis_analysis['primary_domain'], 'creativity')
        self.assertEqual(result.synthesis_analysis['domain_count'], 1)
        self.assertIn('cultural_context', result.synthesis_analysis)
        self.assertIsInstance(result.recommendations, list)
    
    def test_integrate_domain_results_multiple(self):
        """Test domain result integration with multiple results."""
        reasoning_result = DomainEvaluationResult(
            domain=Domain.REASONING,
            evaluation_type=EvaluationType.GENERAL_REASONING,
            primary_score=0.6,
            detailed_metrics={'logical_consistency': 0.65},
            cultural_indicators=[],
            confidence=0.75,
            analysis_details={}
        )
        
        domain_results = [self.mock_domain_result, reasoning_result]
        
        result = self.router._integrate_domain_results(
            domain_results,
            Domain.CREATIVITY,
            {}
        )
        
        # Should have weighted average based on domain weights
        # Creativity (primary): 0.75 * 0.7 = 0.525
        # Reasoning (secondary): 0.6 * 0.3 = 0.18
        # Total: 0.705, Weight: 1.0
        expected_score = ((0.75 * 0.7) + (0.6 * 0.3)) * 100
        self.assertAlmostEqual(result.overall_score, expected_score, places=1)
        
        self.assertEqual(len(result.domain_scores), 2)
        self.assertIn(Domain.CREATIVITY, result.domain_scores)
        self.assertIn(Domain.REASONING, result.domain_scores)
    
    @patch.object(DomainEvaluationRouter, 'get_creative_evaluator')
    def test_route_evaluation_creative(self, mock_get_evaluator):
        """Test evaluation routing for creative domain."""
        # Mock evaluator and its response
        mock_evaluator = Mock()
        mock_evaluator.evaluate.return_value = self.mock_domain_result
        mock_get_evaluator.return_value = mock_evaluator
        
        test_metadata = {
            'domain': 'creativity',
            'name': 'narrative test',
            'description': 'storytelling griot'
        }
        
        result = self.router.route_evaluation("Test response text", test_metadata)
        
        self.assertIsInstance(result, IntegratedEvaluationResult)
        mock_evaluator.evaluate.assert_called_once()
        
        # Check that cultural context was extracted
        call_args = mock_evaluator.evaluate.call_args
        self.assertEqual(call_args[0][0], "Test response text")
        self.assertEqual(call_args[0][1], test_metadata)
        self.assertIsInstance(call_args[0][2], dict)  # Cultural context
    
    @patch.object(DomainEvaluationRouter, 'get_integration_evaluator')
    def test_route_evaluation_integration(self, mock_get_evaluator):
        """Test evaluation routing for integration domain."""
        mock_evaluator = Mock()
        integration_result = DomainEvaluationResult(
            domain=Domain.INTEGRATION,
            evaluation_type=EvaluationType.COMPREHENSIVE_INTEGRATION,
            primary_score=0.8,
            detailed_metrics={},
            cultural_indicators=[],
            confidence=0.9,
            analysis_details={}
        )
        mock_evaluator.evaluate.return_value = integration_result
        mock_get_evaluator.return_value = mock_evaluator
        
        # Use direct domain specification to force integration domain
        test_metadata = {
            'domain': 'integration',
            'name': 'comprehensive integration test',
            'description': 'knowledge reasoning synthesis evaluation'
        }
        
        # Mock the metadata extractor to return integration evaluation type
        with patch.object(self.router.metadata_extractor, 'extract_evaluation_type', 
                         return_value=EvaluationType.COMPREHENSIVE_INTEGRATION):
            result = self.router.route_evaluation("Test response", test_metadata)
        
        self.assertIsInstance(result, IntegratedEvaluationResult)
        mock_evaluator.evaluate.assert_called_once()
    
    def test_route_evaluation_no_evaluator_available(self):
        """Test evaluation routing when no evaluators are available."""
        test_metadata = {
            'name': 'unknown test',
            'description': 'no specific domain patterns'
        }
        
        with patch.object(self.router, 'get_reasoning_evaluator', return_value=None):
            result = self.router.route_evaluation("Test response", test_metadata)
            
            self.assertIsInstance(result, IntegratedEvaluationResult)
            self.assertEqual(result.overall_score, 50.0)  # Fallback score
            
            # Should have one fallback result
            self.assertEqual(len(result.domain_results), 1)
            fallback_result = result.domain_results[0]
            self.assertEqual(fallback_result.primary_score, 0.5)
            self.assertEqual(fallback_result.confidence, 0.3)
            self.assertIn('No domain-specific evaluator available', 
                         fallback_result.analysis_details['note'])
    
    def test_route_evaluation_default_domain(self):
        """Test evaluation routing with default domain fallback."""
        test_metadata = {
            'name': 'generic test',
            'description': 'no domain-specific patterns'
        }
        
        # With no domain-specific evaluator available, should create fallback result
        with patch.object(self.router, 'get_reasoning_evaluator', return_value=None):
            result = self.router.route_evaluation("Test response", test_metadata)
            
            self.assertIsInstance(result, IntegratedEvaluationResult)
            # Should have at least one domain result (the fallback)
            self.assertGreaterEqual(len(result.domain_results), 1)
    
    def test_route_evaluation_with_cultural_analysis(self):
        """Test evaluation routing with cultural analysis for cultural domains."""
        # Test with creativity domain (should attempt cultural analysis)
        test_metadata = {
            'domain': 'creativity',
            'name': 'griot storytelling',
            'description': 'african oral tradition'
        }
        
        with patch.object(self.router, 'get_creative_evaluator') as mock_get_creative:
            mock_evaluator = Mock()
            mock_evaluator.evaluate.return_value = self.mock_domain_result
            mock_get_creative.return_value = mock_evaluator
            
            result = self.router.route_evaluation("Test response", test_metadata)
            
            # Should have domain evaluation result
            self.assertIsInstance(result, IntegratedEvaluationResult)
            self.assertGreaterEqual(len(result.domain_results), 1)
            
            # Should have called the creative evaluator
            mock_evaluator.evaluate.assert_called_once()
    
    @patch('evaluator.subjects.domain_evaluation_router.logger')
    def test_route_evaluation_cultural_import_error(self, mock_logger):
        """Test evaluation routing with cultural analysis import error."""
        test_metadata = {
            'domain': 'creativity',
            'name': 'cultural storytelling test'
        }
        
        with patch.object(self.router, 'get_creative_evaluator') as mock_get_creative:
            mock_evaluator = Mock()
            mock_evaluator.evaluate.return_value = self.mock_domain_result
            mock_get_creative.return_value = mock_evaluator
            
            # Mock import error for cultural modules
            with patch('builtins.__import__', side_effect=ImportError):
                result = self.router.route_evaluation("Test response", test_metadata)
                
                self.assertIsInstance(result, IntegratedEvaluationResult)
                mock_logger.warning.assert_called_with("Cultural evaluation modules not available")


if __name__ == '__main__':
    unittest.main()