"""
Enhanced Universal Evaluator

Phase 1 enhancement of the existing UniversalEvaluator with multi-tier scoring,
advanced analytics integration, and cross-domain synthesis assessment.

Maintains full backward compatibility while adding sophisticated evaluation capabilities
needed for advanced domain content like quantum philosophy.

Author: Claude Code
Version: 1.1.0 (Enhanced)
"""

import re
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Import the base evaluator and all its components
from .reasoning_evaluator import (
    UniversalEvaluator, 
    EvaluationMetrics, 
    EvaluationResult, 
    ReasoningType
)

logger = logging.getLogger(__name__)

@dataclass
class EnhancedEvaluationMetrics(EvaluationMetrics):
    """Enhanced metrics with multi-tier scoring capabilities"""
    # Multi-tier scoring metrics  
    exact_match_score: float = 0.0
    partial_match_score: float = 0.0
    semantic_similarity_score: float = 0.0
    domain_synthesis_score: float = 0.0
    conceptual_creativity_score: float = 0.0
    
    # Cross-domain integration metrics
    integration_quality: float = 0.0
    domain_coverage: int = 0
    synthesis_coherence: float = 0.0
    
    # Enhanced cultural metrics
    cultural_depth_score: float = 0.0
    tradition_accuracy_score: float = 0.0
    cross_cultural_sensitivity: float = 0.0

@dataclass  
class EnhancedEvaluationResult(EvaluationResult):
    """Enhanced evaluation result with multi-tier scoring details"""
    enhanced_metrics: EnhancedEvaluationMetrics
    scoring_breakdown: Dict[str, float]
    integration_analysis: Dict[str, Any]
    
class EnhancedUniversalEvaluator(UniversalEvaluator):
    """
    Enhanced version of UniversalEvaluator with Phase 1 improvements:
    
    1. Multi-tier scoring system (exact_match, partial_match, semantic_similarity)  
    2. Cross-domain synthesis assessment
    3. Test-definition-driven scoring configuration
    4. Advanced analytics integration
    5. Enhanced cultural authenticity assessment
    
    Maintains full backward compatibility with existing interface.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize enhanced evaluator with all base capabilities"""
        super().__init__(config_path)
        
        # Initialize semantic similarity components
        self._semantic_analyzer = None
        self._domain_integrator = None
        
        # Enhanced cultural components
        self._cultural_depth_analyzer = None
        
        logger.info("EnhancedUniversalEvaluator initialized with multi-tier scoring")
    
    def evaluate_response_enhanced(self, 
                                 response_text: str, 
                                 test_definition: Dict[str, Any],
                                 test_name: Optional[str] = None,
                                 reasoning_type: Optional[Union[str, ReasoningType]] = None,
                                 use_llm_evaluation: bool = False) -> EnhancedEvaluationResult:
        """
        Enhanced evaluation method with multi-tier scoring and test-definition integration
        
        Args:
            response_text: The model's response to evaluate
            test_definition: Complete test definition with scoring configuration
            test_name: Name of the test case (extracted from test_definition if None)
            reasoning_type: Type of reasoning expected (auto-detected if None)  
            use_llm_evaluation: Whether to include LLM-based evaluation
            
        Returns:
            EnhancedEvaluationResult with multi-tier scoring and analysis
        """
        # Extract test name and metadata from test definition
        test_name = test_name or test_definition.get('name', test_definition.get('id', 'unknown'))
        test_category = test_definition.get('category', 'general')
        
        # Get base evaluation using existing sophisticated logic
        base_result = self.evaluate_response(
            response_text=response_text,
            test_name=test_name, 
            reasoning_type=reasoning_type,
            test_category=test_category,
            use_llm_evaluation=use_llm_evaluation
        )
        
        # Add enhanced multi-tier scoring
        enhanced_scores = self._compute_multi_tier_scores(response_text, test_definition)
        
        # Assess cross-domain integration
        integration_analysis = self._assess_cross_domain_integration(
            response_text, test_definition
        )
        
        # Enhanced cultural analysis
        cultural_enhancement = self._perform_enhanced_cultural_analysis(
            response_text, test_definition
        )
        
        # Combine base metrics with enhanced metrics
        enhanced_metrics = self._create_enhanced_metrics(
            base_result.metrics, enhanced_scores, integration_analysis, cultural_enhancement
        )
        
        # Create scoring breakdown for transparency
        scoring_breakdown = self._create_scoring_breakdown(
            enhanced_scores, test_definition
        )
        
        return EnhancedEvaluationResult(
            # Preserve base result structure
            metrics=base_result.metrics,
            reasoning_type=base_result.reasoning_type,
            detailed_analysis=base_result.detailed_analysis,
            recommendations=base_result.recommendations,  
            timestamp=base_result.timestamp,
            
            # Add enhanced components
            enhanced_metrics=enhanced_metrics,
            scoring_breakdown=scoring_breakdown,
            integration_analysis=integration_analysis
        )
    
    def _compute_multi_tier_scores(self, response_text: str, test_definition: Dict[str, Any]) -> Dict[str, float]:
        """Compute multi-tier scores: exact_match, partial_match, semantic_similarity"""
        scores = {
            'exact_match_score': 0.0,
            'partial_match_score': 0.0, 
            'semantic_similarity_score': 0.0,
            'domain_synthesis_score': 0.0,
            'conceptual_creativity_score': 0.0
        }
        
        # Get scoring configuration from test definition
        scoring_config = test_definition.get('scoring', {})
        expected_patterns = test_definition.get('expected_patterns', [])
        
        # Exact match assessment
        if expected_patterns:
            scores['exact_match_score'] = self._assess_exact_match(
                response_text, expected_patterns
            )
        
        # Partial match assessment  
        scores['partial_match_score'] = self._assess_partial_match(
            response_text, expected_patterns, test_definition
        )
        
        # Semantic similarity assessment
        scores['semantic_similarity_score'] = self._assess_semantic_similarity(
            response_text, test_definition
        )
        
        # Domain synthesis assessment for complex tests
        if self._is_multi_domain_test(test_definition):
            scores['domain_synthesis_score'] = self._assess_domain_synthesis(
                response_text, test_definition
            )
        
        # Conceptual creativity assessment
        scores['conceptual_creativity_score'] = self._assess_conceptual_creativity(
            response_text, test_definition
        )
        
        return scores
    
    def _assess_exact_match(self, response_text: str, expected_patterns: List[str]) -> float:
        """Assess exact match against expected patterns"""
        if not expected_patterns:
            return 0.0
            
        response_lower = response_text.lower()
        matches = 0
        
        for pattern in expected_patterns:
            if pattern.lower() in response_lower:
                matches += 1
        
        return matches / len(expected_patterns) if expected_patterns else 0.0
    
    def _assess_partial_match(self, response_text: str, expected_patterns: List[str], test_definition: Dict[str, Any]) -> float:
        """Assess partial match using fuzzy matching and context analysis"""
        if not expected_patterns:
            # Fallback to concept-based assessment
            return self._assess_concept_coverage(response_text, test_definition)
            
        response_words = set(response_text.lower().split())
        total_pattern_words = 0
        matched_words = 0
        
        for pattern in expected_patterns:
            pattern_words = set(pattern.lower().split())
            total_pattern_words += len(pattern_words)
            matched_words += len(pattern_words.intersection(response_words))
        
        return matched_words / total_pattern_words if total_pattern_words > 0 else 0.0
    
    def _assess_semantic_similarity(self, response_text: str, test_definition: Dict[str, Any]) -> float:
        """Assess semantic similarity to test concepts and expected reasoning"""
        # Initialize semantic analyzer if needed
        if self._semantic_analyzer is None:
            self._semantic_analyzer = self._initialize_semantic_analyzer()
        
        if self._semantic_analyzer is None:
            # Fallback to keyword-based assessment
            return self._assess_keyword_semantic_similarity(response_text, test_definition)
        
        # Use advanced semantic analysis with proper method
        prompt = test_definition.get('prompt', test_definition.get('description', ''))
        coherence_analysis = self._semantic_analyzer.comprehensive_coherence_analysis(
            response_text, prompt
        )
        
        # Extract semantic similarity score from coherence analysis
        return coherence_analysis.get('overall_coherence_score', 0.0)
    
    def _assess_keyword_semantic_similarity(self, response_text: str, test_definition: Dict[str, Any]) -> float:
        """Fallback semantic assessment using keyword analysis"""
        # Extract conceptual keywords from test
        concepts_tested = test_definition.get('metadata', {}).get('concepts_tested', [])
        description = test_definition.get('description', '')
        
        all_concepts = concepts_tested + [description]
        concept_keywords = set()
        
        for concept in all_concepts:
            if isinstance(concept, str):
                concept_keywords.update(concept.lower().split('_'))
                concept_keywords.update(concept.lower().split())
        
        # Check concept coverage in response
        response_words = set(response_text.lower().split())
        matches = len(concept_keywords.intersection(response_words))
        
        return min(matches / len(concept_keywords) if concept_keywords else 0.0, 1.0)
    
    def _assess_cross_domain_integration(self, response_text: str, test_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Assess cross-domain integration quality for multi-domain tests"""
        integration_analysis = {
            'is_multi_domain': False,
            'domains_integrated': [],
            'integration_quality': 0.0,
            'domain_coverage': 0,
            'synthesis_coherence': 0.0
        }
        
        # Check if this is a multi-domain test
        domains_integrated = test_definition.get('metadata', {}).get('domains_integrated', [])
        
        if domains_integrated:
            integration_analysis['is_multi_domain'] = True
            integration_analysis['domains_integrated'] = domains_integrated
            integration_analysis['domain_coverage'] = len(domains_integrated)
            
            # Assess integration quality
            integration_analysis['integration_quality'] = self._compute_integration_quality(
                response_text, domains_integrated
            )
            
            # Assess synthesis coherence
            integration_analysis['synthesis_coherence'] = self._assess_synthesis_coherence(
                response_text, domains_integrated
            )
        
        return integration_analysis
    
    def _compute_integration_quality(self, response_text: str, domains: List[str]) -> float:
        """Compute the quality of multi-domain integration"""
        if len(domains) < 2:
            return 0.0
        
        # Domain-specific keywords for assessment
        domain_keywords = {
            'quantum_mechanics': ['quantum', 'superposition', 'measurement', 'observer', 'wave', 'particle'],
            'philosophy': ['epistemology', 'metaphysics', 'reality', 'knowledge', 'existence', 'truth'],
            'sociology': ['social', 'community', 'collective', 'consensus', 'society', 'group'],
            'physics': ['energy', 'matter', 'force', 'field', 'theory', 'law'],
            'mathematics': ['equation', 'formula', 'theorem', 'proof', 'logic', 'set'],
            'linguistics': ['language', 'meaning', 'semantic', 'syntax', 'grammar', 'communication']
        }
        
        response_lower = response_text.lower()
        domain_presence = {}
        
        for domain in domains:
            keywords = domain_keywords.get(domain, domain.split('_'))
            presence = sum(1 for keyword in keywords if keyword in response_lower)
            domain_presence[domain] = min(presence / len(keywords) if keywords else 0.0, 1.0)
        
        # Integration quality is the minimum domain coverage (weakest link)
        # This ensures all domains are meaningfully integrated
        return min(domain_presence.values()) if domain_presence else 0.0
    
    def _assess_synthesis_coherence(self, response_text: str, domains: List[str]) -> float:
        """Assess how coherently the response synthesizes multiple domains"""
        # Look for integration indicators
        integration_indicators = [
            'because', 'therefore', 'thus', 'consequently', 'as a result',
            'this means', 'implies', 'suggests', 'demonstrates', 'shows',
            'connects to', 'relates to', 'similar to', 'analogous to',
            'bridges', 'integrates', 'synthesizes', 'combines'
        ]
        
        response_lower = response_text.lower()
        integration_signals = sum(1 for indicator in integration_indicators if indicator in response_lower)
        
        # Normalize by response length and expected integration complexity
        words = len(response_text.split())
        expected_integrations = len(domains) - 1  # n domains = n-1 integration points
        
        return min(integration_signals / max(expected_integrations, 1), 1.0)
    
    def _perform_enhanced_cultural_analysis(self, response_text: str, test_definition: Dict[str, Any]) -> Dict[str, float]:
        """Perform enhanced cultural authenticity and sensitivity analysis"""
        cultural_scores = {
            'cultural_depth_score': 0.0,
            'tradition_accuracy_score': 0.0, 
            'cross_cultural_sensitivity': 0.0
        }
        
        # Extract cultural context from test
        cultural_context = test_definition.get('cultural_context', {})
        traditions = cultural_context.get('traditions', [])
        
        if not traditions and not self._has_cultural_content(test_definition):
            # No cultural content to analyze
            return cultural_scores
        
        # Use existing cultural analysis components
        if hasattr(self, '_cultural_authenticity_analyzer') and self._cultural_authenticity_analyzer:
            try:
                cultural_analysis = self._cultural_authenticity_analyzer.analyze(response_text, cultural_context)
                cultural_scores['cultural_depth_score'] = cultural_analysis.get('authenticity_score', 0.0)
            except Exception as e:
                logger.warning(f"Cultural analysis failed: {e}")
        
        # Assess tradition accuracy
        cultural_scores['tradition_accuracy_score'] = self._assess_tradition_accuracy(
            response_text, traditions, test_definition
        )
        
        # Assess cross-cultural sensitivity
        cultural_scores['cross_cultural_sensitivity'] = self._assess_cultural_sensitivity(
            response_text, test_definition
        )
        
        return cultural_scores
    
    def _has_cultural_content(self, test_definition: Dict[str, Any]) -> bool:
        """Check if test has cultural content requiring analysis"""
        cultural_indicators = [
            'cultural', 'tradition', 'heritage', 'indigenous', 'folk',
            'japanese', 'african', 'arabic', 'chinese', 'indian', 'european',
            'haiku', 'proverb', 'story', 'wisdom', 'ancestral'
        ]
        
        test_text = json.dumps(test_definition).lower()
        return any(indicator in test_text for indicator in cultural_indicators)
    
    def _assess_tradition_accuracy(self, response_text: str, traditions: List[str], test_definition: Dict[str, Any]) -> float:
        """Assess accuracy in representing cultural traditions"""
        if not traditions:
            return 0.0
        
        # Look for respectful and accurate representation
        accuracy_indicators = [
            'traditional', 'ancient', 'respected', 'honored', 'sacred',
            'wisdom', 'teaching', 'practice', 'custom', 'heritage'
        ]
        
        response_lower = response_text.lower()
        accuracy_signals = sum(1 for indicator in accuracy_indicators if indicator in response_lower)
        
        # Check for inappropriate or inaccurate representations (penalties)
        inappropriate_terms = [
            'primitive', 'backward', 'superstitious', 'outdated', 'silly'
        ]
        
        penalties = sum(1 for term in inappropriate_terms if term in response_lower)
        
        return max((accuracy_signals - penalties) / len(accuracy_indicators), 0.0)
    
    def _assess_cultural_sensitivity(self, response_text: str, test_definition: Dict[str, Any]) -> float:
        """Assess cultural sensitivity and appropriateness"""
        # Look for respectful language and appropriate cultural context
        sensitivity_indicators = [
            'respect', 'honor', 'appreciate', 'understand', 'acknowledge',
            'cultural context', 'traditional wisdom', 'heritage', 'ancestors'
        ]
        
        response_lower = response_text.lower()
        sensitivity_score = sum(1 for indicator in sensitivity_indicators if indicator in response_lower)
        
        return min(sensitivity_score / 3.0, 1.0)  # Normalize to 0-1 scale
    
    def _create_enhanced_metrics(self, base_metrics: EvaluationMetrics, 
                                enhanced_scores: Dict[str, float],
                                integration_analysis: Dict[str, Any],
                                cultural_enhancement: Dict[str, float]) -> EnhancedEvaluationMetrics:
        """Create enhanced metrics combining base and new scoring"""
        
        # Convert base metrics to dict and create enhanced version
        base_dict = asdict(base_metrics)
        
        enhanced_metrics = EnhancedEvaluationMetrics(
            **base_dict,
            **enhanced_scores,
            integration_quality=integration_analysis.get('integration_quality', 0.0),
            domain_coverage=integration_analysis.get('domain_coverage', 0),
            synthesis_coherence=integration_analysis.get('synthesis_coherence', 0.0),
            **cultural_enhancement
        )
        
        return enhanced_metrics
    
    def _create_scoring_breakdown(self, enhanced_scores: Dict[str, float], test_definition: Dict[str, Any]) -> Dict[str, float]:
        """Create transparent scoring breakdown for analysis"""
        scoring_config = test_definition.get('scoring', {})
        
        breakdown = {
            'base_evaluation_weight': 0.6,  # 60% from base sophisticated evaluation
            'enhanced_scoring_weight': 0.4,  # 40% from enhanced multi-tier scoring
            **enhanced_scores
        }
        
        # Add test-specific scoring weights if available
        if scoring_config:
            breakdown['test_specific_config'] = scoring_config
        
        return breakdown
    
    def _initialize_semantic_analyzer(self):
        """Initialize semantic analyzer with fallback handling"""
        try:
            # Try to import and initialize advanced semantic analyzer
            from ..advanced.semantic_coherence import SemanticCoherenceAnalyzer
            return SemanticCoherenceAnalyzer()
        except ImportError:
            logger.warning("Advanced semantic analyzer not available, using fallback methods")
            return None
    
    def _is_multi_domain_test(self, test_definition: Dict[str, Any]) -> bool:
        """Check if this is a multi-domain integration test"""
        domains = test_definition.get('metadata', {}).get('domains_integrated', [])
        return len(domains) > 1
    
    def _assess_concept_coverage(self, response_text: str, test_definition: Dict[str, Any]) -> float:
        """Assess coverage of test concepts when no specific patterns available"""
        concepts = test_definition.get('metadata', {}).get('concepts_tested', [])
        if not concepts:
            return 0.5  # Neutral score when no concepts specified
        
        response_lower = response_text.lower()
        covered_concepts = 0
        
        for concept in concepts:
            concept_words = concept.lower().replace('_', ' ').split()
            if any(word in response_lower for word in concept_words):
                covered_concepts += 1
        
        return covered_concepts / len(concepts) if concepts else 0.0
    
    def _assess_domain_synthesis(self, response_text: str, test_definition: Dict[str, Any]) -> float:
        """Assess quality of domain synthesis in response"""
        domains = test_definition.get('metadata', {}).get('domains_integrated', [])
        if len(domains) < 2:
            return 0.0
        
        return self._compute_integration_quality(response_text, domains)
    
    def _assess_conceptual_creativity(self, response_text: str, test_definition: Dict[str, Any]) -> float:
        """Assess conceptual creativity and novel insights"""
        creativity_indicators = [
            'novel', 'innovative', 'unique', 'original', 'creative',
            'new perspective', 'fresh approach', 'different way',
            'imagine', 'envision', 'conceive', 'insight', 'breakthrough'
        ]
        
        response_lower = response_text.lower()
        creativity_signals = sum(1 for indicator in creativity_indicators if indicator in response_lower)
        
        # Normalize by response length
        words = len(response_text.split())
        creativity_density = creativity_signals / max(words / 50, 1)  # Per ~50 words
        
        return min(creativity_density, 1.0)

# Backward compatibility: maintain existing interface
def evaluate_reasoning(response_text: str, test_name: str, reasoning_type: Optional[Union[str, ReasoningType]] = None) -> EvaluationResult:
    """Backward compatible function for existing code"""
    evaluator = EnhancedUniversalEvaluator()
    return evaluator.evaluate_response(response_text, test_name, reasoning_type)