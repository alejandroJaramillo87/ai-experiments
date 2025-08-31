"""
Validation Tests for Enhanced Universal Evaluator

Tests the Phase 1 enhancements while ensuring backward compatibility
and improved evaluation quality.

Author: Claude Code
Version: 1.0.0
"""

import unittest
import json
import logging
from typing import Dict, Any
from pathlib import Path

# Import both evaluators for comparison
from evaluator.subjects.reasoning_evaluator import UniversalEvaluator, ReasoningType
from evaluator.subjects.enhanced_universal_evaluator import EnhancedUniversalEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestEnhancedUniversalEvaluator(unittest.TestCase):
    """Test enhanced evaluator capabilities and backward compatibility"""
    
    def setUp(self):
        """Set up test environment"""
        self.base_evaluator = UniversalEvaluator()
        self.enhanced_evaluator = EnhancedUniversalEvaluator()
        
        # Load test data
        self.reasoning_tests = self._load_reasoning_tests()
        
    def _load_reasoning_tests(self) -> Dict[str, Any]:
        """Load reasoning test data for validation"""
        test_files = [
            "domains/reasoning/base_models/easy.json",
            "domains/reasoning/instruct_models/easy.json"
        ]
        
        all_tests = {"base_tests": [], "instruct_tests": []}
        
        for test_file in test_files:
            try:
                with open(test_file, 'r') as f:
                    test_data = json.load(f)
                    tests = test_data.get('tests', [])
                    
                    if 'base_models' in test_file:
                        all_tests['base_tests'].extend(tests[:5])  # Sample 5 tests
                    else:
                        all_tests['instruct_tests'].extend(tests[:5])  # Sample 5 tests
                        
            except FileNotFoundError:
                logger.warning(f"Test file not found: {test_file}")
                continue
                
        return all_tests
    
    def test_backward_compatibility(self):
        """Test that enhanced evaluator maintains backward compatibility"""
        test_response = "This is a thoughtful analysis of the cultural pattern in Japanese haiku, following the 5-7-5 syllable structure while incorporating traditional seasonal imagery and creating poetic closure."
        test_name = "haiku_pattern_test"
        
        # Test both evaluators with same input
        base_result = self.base_evaluator.evaluate_response(
            test_response, test_name, ReasoningType.GENERAL
        )
        
        enhanced_result = self.enhanced_evaluator.evaluate_response(
            test_response, test_name, ReasoningType.GENERAL
        )
        
        # Verify same interface
        self.assertIsNotNone(base_result.metrics)
        self.assertIsNotNone(enhanced_result.metrics)
        self.assertIsInstance(base_result.reasoning_type, ReasoningType)
        self.assertIsInstance(enhanced_result.reasoning_type, ReasoningType)
        
        # Verify basic functionality preserved
        self.assertGreater(base_result.metrics.overall_score, 0.0)
        self.assertGreater(enhanced_result.metrics.overall_score, 0.0)
        
        logger.info(f"Base evaluator score: {base_result.metrics.overall_score:.3f}")
        logger.info(f"Enhanced evaluator score: {enhanced_result.metrics.overall_score:.3f}")
        
    def test_multi_tier_scoring(self):
        """Test multi-tier scoring with test definition"""
        test_definition = {
            'id': 'test_multi_tier',
            'name': 'Multi-tier Scoring Test',
            'category': 'pattern_recognition',
            'expected_patterns': ['seasonal imagery', 'poetic closure', '5-7-5'],
            'scoring': {
                'exact_match': 1.0,
                'partial_match': 0.5,
                'semantic_similarity': 0.3
            },
            'metadata': {
                'concepts_tested': ['haiku_structure', 'cultural_authenticity', 'poetic_form']
            }
        }
        
        test_response = "Gentle petals drift to earth - a perfect example of seasonal imagery with poetic closure following traditional 5-7-5 syllable structure."
        
        # Test enhanced evaluation
        result = self.enhanced_evaluator.evaluate_response_enhanced(
            test_response, test_definition
        )
        
        # Verify enhanced metrics exist
        self.assertIsNotNone(result.enhanced_metrics)
        self.assertIsNotNone(result.scoring_breakdown)
        
        # Verify multi-tier scores
        enhanced_metrics = result.enhanced_metrics
        self.assertGreaterEqual(enhanced_metrics.exact_match_score, 0.0)
        self.assertLessEqual(enhanced_metrics.exact_match_score, 1.0)
        self.assertGreaterEqual(enhanced_metrics.partial_match_score, 0.0)
        self.assertLessEqual(enhanced_metrics.partial_match_score, 1.0)
        self.assertGreaterEqual(enhanced_metrics.semantic_similarity_score, 0.0)
        self.assertLessEqual(enhanced_metrics.semantic_similarity_score, 1.0)
        
        logger.info(f"Exact match score: {enhanced_metrics.exact_match_score:.3f}")
        logger.info(f"Partial match score: {enhanced_metrics.partial_match_score:.3f}")
        logger.info(f"Semantic similarity score: {enhanced_metrics.semantic_similarity_score:.3f}")
    
    def test_cross_domain_integration(self):
        """Test cross-domain integration assessment"""
        test_definition = {
            'id': 'quantum_philosophy_test',
            'name': 'Quantum Philosophy Integration',
            'category': 'cross_domain_synthesis',
            'metadata': {
                'domains_integrated': ['quantum_mechanics', 'philosophy', 'epistemology'],
                'concepts_tested': ['observer_effect', 'measurement_theory', 'consensus_reality']
            }
        }
        
        test_response = "The quantum measurement problem demonstrates how observation affects reality, which connects to philosophical questions about epistemology and what constitutes knowledge. When multiple observers reach consensus, we see the collapse from quantum superposition to agreed-upon reality."
        
        result = self.enhanced_evaluator.evaluate_response_enhanced(
            test_response, test_definition
        )
        
        # Verify integration analysis
        integration = result.integration_analysis
        self.assertTrue(integration['is_multi_domain'])
        self.assertEqual(len(integration['domains_integrated']), 3)
        self.assertGreater(integration['integration_quality'], 0.0)
        self.assertGreater(integration['synthesis_coherence'], 0.0)
        
        logger.info(f"Integration quality: {integration['integration_quality']:.3f}")
        logger.info(f"Synthesis coherence: {integration['synthesis_coherence']:.3f}")
    
    def test_cultural_authenticity_enhancement(self):
        """Test enhanced cultural authenticity analysis"""
        test_definition = {
            'id': 'cultural_test',
            'name': 'Cultural Authenticity Test',
            'category': 'cultural_reasoning',
            'cultural_context': {
                'traditions': ['japanese_haiku', 'seasonal_awareness']
            },
            'metadata': {
                'concepts_tested': ['cultural_authenticity', 'traditional_wisdom']
            }
        }
        
        test_response = "Traditional Japanese haiku honors the ancient wisdom of seasonal awareness, respecting the cultural heritage of observing nature's subtle changes with deep reverence for the natural world."
        
        result = self.enhanced_evaluator.evaluate_response_enhanced(
            test_response, test_definition
        )
        
        # Verify cultural enhancement
        enhanced_metrics = result.enhanced_metrics
        self.assertGreaterEqual(enhanced_metrics.cultural_depth_score, 0.0)
        self.assertGreaterEqual(enhanced_metrics.tradition_accuracy_score, 0.0)
        self.assertGreaterEqual(enhanced_metrics.cross_cultural_sensitivity, 0.0)
        
        logger.info(f"Cultural depth: {enhanced_metrics.cultural_depth_score:.3f}")
        logger.info(f"Tradition accuracy: {enhanced_metrics.tradition_accuracy_score:.3f}")
        logger.info(f"Cultural sensitivity: {enhanced_metrics.cross_cultural_sensitivity:.3f}")
    
    def test_with_actual_reasoning_tests(self):
        """Test with actual reasoning test data"""
        if not self.reasoning_tests['base_tests']:
            self.skipTest("No reasoning test data available")
        
        # Test with a base model test
        base_test = self.reasoning_tests['base_tests'][0]
        test_response = "Falling softly down - this completes the haiku while maintaining the 5-7-5 syllable pattern and connecting to the spring cherry blossom theme with appropriate seasonal imagery and poetic closure."
        
        # Enhanced evaluation
        result = self.enhanced_evaluator.evaluate_response_enhanced(
            test_response, base_test
        )
        
        # Verify comprehensive evaluation
        self.assertIsNotNone(result.enhanced_metrics)
        self.assertIsNotNone(result.scoring_breakdown)
        self.assertGreater(result.metrics.overall_score, 0.0)
        
        logger.info(f"Actual test evaluation score: {result.metrics.overall_score:.3f}")
        logger.info(f"Enhanced scoring breakdown: {result.scoring_breakdown}")
    
    def test_epistemological_collapse_style_scoring(self):
        """Test scoring similar to epistemological_collapse domain tests"""
        test_definition = {
            'id': 'ec_test_style',
            'category': 'observer_dependent_reality',
            'prompt': 'Facts that changed based on who observed them reached consensus only when all observers agreed to',
            'expected_patterns': ['not observe', 'average', 'vote', 'forget', 'pretend'],
            'scoring': {
                'exact_match': 1.0,
                'partial_match': 0.5,
                'semantic_similarity': 0.3
            },
            'metadata': {
                'reasoning_steps': 3,
                'concepts_tested': ['observer_effect', 'consensus_reality', 'epistemology'],
                'domains_integrated': ['quantum_mechanics', 'philosophy', 'sociology']
            }
        }
        
        test_response = "not observe the phenomenon, allowing quantum superposition to persist until measurement collapses the wavefunction into consensus reality."
        
        result = self.enhanced_evaluator.evaluate_response_enhanced(
            test_response, test_definition
        )
        
        # Verify sophisticated scoring
        enhanced_metrics = result.enhanced_metrics
        integration = result.integration_analysis
        
        # Should detect exact match with expected pattern
        self.assertGreater(enhanced_metrics.exact_match_score, 0.0)
        
        # Should detect multi-domain integration
        self.assertTrue(integration['is_multi_domain'])
        self.assertEqual(len(integration['domains_integrated']), 3)
        
        # Should assess domain synthesis quality
        self.assertGreater(enhanced_metrics.domain_synthesis_score, 0.0)
        
        logger.info(f"Epistemological test exact match: {enhanced_metrics.exact_match_score:.3f}")
        logger.info(f"Domain synthesis quality: {enhanced_metrics.domain_synthesis_score:.3f}")
    
    def test_performance_comparison(self):
        """Compare performance between base and enhanced evaluators"""
        test_response = "This demonstrates sophisticated reasoning across multiple domains, integrating quantum mechanical concepts with philosophical epistemology while maintaining cultural authenticity in its traditional knowledge representation."
        
        test_definition = {
            'id': 'performance_test',
            'name': 'Performance Comparison Test',
            'category': 'comprehensive_evaluation',
            'metadata': {
                'domains_integrated': ['quantum_mechanics', 'philosophy', 'cultural_studies'],
                'concepts_tested': ['integration', 'authenticity', 'sophistication']
            }
        }
        
        # Base evaluation
        base_result = self.base_evaluator.evaluate_response(
            test_response, 'performance_test', ReasoningType.GENERAL
        )
        
        # Enhanced evaluation
        enhanced_result = self.enhanced_evaluator.evaluate_response_enhanced(
            test_response, test_definition
        )
        
        # Compare overall quality
        logger.info("=== Performance Comparison ===")
        logger.info(f"Base evaluator overall score: {base_result.metrics.overall_score:.3f}")
        logger.info(f"Enhanced evaluator overall score: {enhanced_result.metrics.overall_score:.3f}")
        
        # Enhanced evaluator should provide additional insights
        self.assertIsNotNone(enhanced_result.integration_analysis)
        self.assertIsNotNone(enhanced_result.scoring_breakdown)
        
        # Verify enhanced capabilities
        enhanced_metrics = enhanced_result.enhanced_metrics
        self.assertGreaterEqual(enhanced_metrics.domain_synthesis_score, 0.0)
        self.assertGreaterEqual(enhanced_metrics.conceptual_creativity_score, 0.0)

if __name__ == '__main__':
    # Create necessary directories
    Path('tests/validation').mkdir(parents=True, exist_ok=True)
    
    # Run validation tests
    unittest.main(verbosity=2)