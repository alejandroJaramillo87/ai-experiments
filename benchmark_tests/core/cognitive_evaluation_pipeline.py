#!/usr/bin/env python3
"""
Cognitive Evaluation Pipeline

Integrates sophisticated evaluators for comprehensive cognitive pattern detection.
Replaces basic scoring with advanced pattern analysis using:
- PatternBasedEvaluator for behavioral consistency 
- CulturalAuthenticity for social/cultural cognitive patterns
- EnhancedUniversalEvaluator for multi-tier assessment
- Statistical bias detection for systematic blind spots

Maps to cognitive dimensions: reasoning, memory, creativity, social, integration
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import statistics
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import sophisticated evaluator framework
try:
    from evaluator.subjects.pattern_based_evaluator import PatternBasedEvaluator, PatternAnalysisResult
    PATTERN_EVALUATOR_AVAILABLE = True
except ImportError as e:
    PATTERN_EVALUATOR_AVAILABLE = False
    logging.warning(f"PatternBasedEvaluator not available: {e}")

try:
    from evaluator.subjects.enhanced_universal_evaluator import EnhancedUniversalEvaluator
    ENHANCED_EVALUATOR_AVAILABLE = True
except ImportError as e:
    ENHANCED_EVALUATOR_AVAILABLE = False
    logging.warning(f"EnhancedUniversalEvaluator not available: {e}")

try:
    from evaluator.cultural.cultural_authenticity import CulturalAuthenticityAnalyzer
    CULTURAL_EVALUATOR_AVAILABLE = True
except ImportError as e:
    CULTURAL_EVALUATOR_AVAILABLE = False  
    logging.warning(f"CulturalAuthenticityAnalyzer not available: {e}")

try:
    from evaluator.core.evaluation_aggregator import EvaluationAggregator, BiasAnalysis
    BIAS_ANALYSIS_AVAILABLE = True
except ImportError as e:
    BIAS_ANALYSIS_AVAILABLE = False
    logging.warning(f"BiasAnalysis not available: {e}")

logger = logging.getLogger(__name__)

@dataclass
class CognitiveEvaluationResult:
    """Comprehensive evaluation result with cognitive pattern analysis"""
    test_id: str
    cognitive_domain: str
    
    # Core scores
    overall_score: float           # 0-100 overall performance  
    cognitive_subscores: Dict[str, float]  # Specific cognitive abilities
    
    # Pattern analysis results
    behavioral_patterns: Optional[PatternAnalysisResult] = None
    cultural_analysis: Optional[Dict[str, Any]] = None
    bias_indicators: Optional[Dict[str, Any]] = None
    
    # Statistical measures  
    confidence_score: float = 0.0
    pattern_strength: float = 0.0
    consistency_measure: float = 0.0
    
    # Evidence and details
    evaluation_details: Dict[str, Any] = None
    raw_scores: Dict[str, float] = None

class CognitiveEvaluationPipeline:
    """
    Comprehensive evaluation pipeline using sophisticated evaluator framework
    """
    
    def __init__(self):
        self.pattern_evaluator = None
        self.enhanced_evaluator = None  
        self.cultural_evaluator = None
        self.bias_aggregator = None
        
        # Initialize available evaluators
        self._initialize_evaluators()
        
        # Cognitive domain mapping
        self.cognitive_mappings = {
            'reasoning': {
                'domains': ['reasoning', 'abstract_reasoning', 'logical', 'inference'],
                'key_abilities': ['logical_analysis', 'abstract_thinking', 'causal_reasoning'],
                'weight_factors': {'consistency': 0.3, 'complexity': 0.4, 'accuracy': 0.3}
            },
            'memory': {
                'domains': ['knowledge', 'historical', 'factual', 'contextual'],
                'key_abilities': ['factual_recall', 'contextual_understanding', 'knowledge_synthesis'],
                'weight_factors': {'accuracy': 0.5, 'completeness': 0.3, 'context_awareness': 0.2}
            },
            'creativity': {
                'domains': ['creativity', 'narrative', 'artistic', 'innovation'],
                'key_abilities': ['originality', 'synthesis', 'artistic_expression', 'novel_combinations'],
                'weight_factors': {'originality': 0.4, 'coherence': 0.3, 'expressiveness': 0.3}
            },
            'social': {
                'domains': ['social', 'cultural', 'empathy', 'interpersonal'],
                'key_abilities': ['cultural_competency', 'empathy', 'social_reasoning', 'conflict_resolution'],
                'weight_factors': {'cultural_sensitivity': 0.4, 'empathy': 0.3, 'social_reasoning': 0.3}
            },
            'integration': {
                'domains': ['integration', 'cross_domain', 'complex', 'synthesis'],
                'key_abilities': ['cross_domain_synthesis', 'complex_reasoning', 'holistic_thinking'],
                'weight_factors': {'synthesis': 0.4, 'complexity_handling': 0.3, 'coherence': 0.3}
            }
        }
    
    def _initialize_evaluators(self):
        """Initialize available sophisticated evaluators"""
        
        if PATTERN_EVALUATOR_AVAILABLE:
            try:
                self.pattern_evaluator = PatternBasedEvaluator()
                logger.info("PatternBasedEvaluator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize PatternBasedEvaluator: {e}")
        
        if ENHANCED_EVALUATOR_AVAILABLE:
            try:
                self.enhanced_evaluator = EnhancedUniversalEvaluator()
                logger.info("EnhancedUniversalEvaluator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize EnhancedUniversalEvaluator: {e}")
        
        if CULTURAL_EVALUATOR_AVAILABLE:
            try:
                self.cultural_evaluator = CulturalAuthenticityAnalyzer()
                logger.info("CulturalAuthenticityAnalyzer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize CulturalAuthenticityAnalyzer: {e}")
                
        if BIAS_ANALYSIS_AVAILABLE:
            try:
                self.bias_aggregator = EvaluationAggregator()
                logger.info("EvaluationAggregator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize EvaluationAggregator: {e}")
    
    def evaluate_response(self,
                         test_id: str,
                         prompt: str, 
                         response_text: str,
                         test_metadata: Dict[str, Any]) -> CognitiveEvaluationResult:
        """
        Comprehensive cognitive evaluation of model response
        
        Args:
            test_id: Unique test identifier
            prompt: Original test prompt
            response_text: Model's response
            test_metadata: Test configuration and metadata
            
        Returns:
            CognitiveEvaluationResult with comprehensive analysis
        """
        
        # Determine cognitive domain
        cognitive_domain = self._classify_cognitive_domain(test_metadata)
        
        logger.debug(f"Evaluating {test_id} in {cognitive_domain} domain")
        
        # Initialize result structure
        result = CognitiveEvaluationResult(
            test_id=test_id,
            cognitive_domain=cognitive_domain,
            overall_score=0.0,
            cognitive_subscores={},
            evaluation_details={},
            raw_scores={}
        )
        
        # Run pattern-based evaluation (with fallback)
        if self.pattern_evaluator:
            try:
                result.behavioral_patterns = self._run_pattern_evaluation(
                    response_text, prompt, test_metadata
                )
                logger.debug(f"Pattern evaluation successful for {test_id}")
            except Exception as e:
                logger.warning(f"Pattern evaluation failed for {test_id}: {e}")
                result.behavioral_patterns = None
        
        # Run enhanced universal evaluation  
        if self.enhanced_evaluator:
            enhanced_results = self._run_enhanced_evaluation(
                response_text, prompt, test_metadata
            )
            result.evaluation_details['enhanced'] = enhanced_results
        
        # Run cultural/social evaluation for social domain
        if self.cultural_evaluator and cognitive_domain == 'social':
            result.cultural_analysis = self._run_cultural_evaluation(
                response_text, test_metadata
            )
        
        # Calculate cognitive-specific scores
        result.cognitive_subscores = self._calculate_cognitive_subscores(
            cognitive_domain, result.behavioral_patterns, 
            result.evaluation_details, result.cultural_analysis
        )
        
        # Fallback scoring if sophisticated evaluators failed
        if not result.cognitive_subscores:
            result.cognitive_subscores = self._calculate_fallback_subscores(
                response_text, prompt, cognitive_domain
            )
        
        # Calculate overall score
        result.overall_score = self._calculate_overall_score(
            result.cognitive_subscores, cognitive_domain
        )
        
        # Statistical measures
        result.confidence_score = self._calculate_confidence_score(result)
        result.pattern_strength = self._calculate_pattern_strength(result.behavioral_patterns)
        result.consistency_measure = self._calculate_consistency_measure(result.behavioral_patterns)
        
        logger.debug(f"Evaluation complete: {test_id} = {result.overall_score:.1f}")
        return result
    
    def _classify_cognitive_domain(self, test_metadata: Dict[str, Any]) -> str:
        """Classify test into cognitive domain"""
        
        # Check explicit domain in metadata
        if 'domain' in test_metadata:
            domain_name = test_metadata['domain'].lower()
            for cognitive_domain, mapping in self.cognitive_mappings.items():
                if any(keyword in domain_name for keyword in mapping['domains']):
                    return cognitive_domain
        
        # Check test ID for domain clues
        test_id = test_metadata.get('id', '').lower()
        for cognitive_domain, mapping in self.cognitive_mappings.items():
            if any(keyword in test_id for keyword in mapping['domains']):
                return cognitive_domain
                
        # Default to integration for unclear cases
        return 'integration'
    
    def _run_pattern_evaluation(self, 
                               response_text: str, 
                               prompt: str, 
                               test_metadata: Dict[str, Any]) -> Optional[PatternAnalysisResult]:
        """Run pattern-based evaluation"""
        
        try:
            return self.pattern_evaluator.evaluate_patterns(
                response_text=response_text,
                prompt=prompt, 
                test_metadata=test_metadata,
                model_id="current"
            )
        except Exception as e:
            logger.error(f"Pattern evaluation failed: {e}")
            return None
    
    def _run_enhanced_evaluation(self,
                                response_text: str,
                                prompt: str, 
                                test_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Run enhanced universal evaluation"""
        
        try:
            # This would depend on the specific API of EnhancedUniversalEvaluator
            # For now, return placeholder structure
            return {
                'multi_tier_scores': {
                    'exact_match': 0.0,
                    'partial_match': 0.0, 
                    'semantic_similarity': 0.0
                },
                'reasoning_analysis': {},
                'quality_metrics': {}
            }
        except Exception as e:
            logger.error(f"Enhanced evaluation failed: {e}")
            return {}
    
    def _run_cultural_evaluation(self, 
                                response_text: str,
                                test_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Run cultural authenticity evaluation"""
        
        try:
            # This would depend on the specific API of CulturalAuthenticity
            # For now, return placeholder structure
            return {
                'authenticity_score': 0.0,
                'bias_indicators': [],
                'cultural_competency': 0.0,
                'respectful_language': 0.0
            }
        except Exception as e:
            logger.error(f"Cultural evaluation failed: {e}")
            return {}
    
    def _calculate_cognitive_subscores(self,
                                      cognitive_domain: str,
                                      behavioral_patterns: Optional[PatternAnalysisResult],
                                      evaluation_details: Dict[str, Any],
                                      cultural_analysis: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate domain-specific cognitive subscores"""
        
        subscores = {}
        domain_config = self.cognitive_mappings.get(cognitive_domain, {})
        key_abilities = domain_config.get('key_abilities', [])
        
        # Extract scores from behavioral patterns
        if behavioral_patterns:
            quality_indicators = behavioral_patterns.quality_indicators
            
            # Map to cognitive abilities based on domain
            if cognitive_domain == 'reasoning':
                subscores['logical_analysis'] = behavioral_patterns.pattern_adherence * 100
                subscores['abstract_thinking'] = quality_indicators.get('complexity_handling', 0.5) * 100
                subscores['causal_reasoning'] = behavioral_patterns.response_consistency * 100
                
            elif cognitive_domain == 'memory':
                subscores['factual_recall'] = quality_indicators.get('accuracy', 0.5) * 100
                subscores['contextual_understanding'] = behavioral_patterns.response_consistency * 100
                subscores['knowledge_synthesis'] = quality_indicators.get('completeness', 0.5) * 100
                
            elif cognitive_domain == 'creativity':
                subscores['originality'] = quality_indicators.get('novelty', 0.5) * 100
                subscores['synthesis'] = behavioral_patterns.pattern_adherence * 100
                subscores['artistic_expression'] = quality_indicators.get('expressiveness', 0.5) * 100
                
            elif cognitive_domain == 'social':
                if cultural_analysis:
                    subscores['cultural_competency'] = cultural_analysis.get('cultural_competency', 0) * 100
                    subscores['empathy'] = cultural_analysis.get('respectful_language', 0) * 100
                else:
                    subscores['cultural_competency'] = behavioral_patterns.response_consistency * 100
                    subscores['empathy'] = quality_indicators.get('social_awareness', 0.5) * 100
                subscores['social_reasoning'] = behavioral_patterns.pattern_adherence * 100
                
            elif cognitive_domain == 'integration':
                subscores['cross_domain_synthesis'] = behavioral_patterns.pattern_adherence * 100
                subscores['complex_reasoning'] = quality_indicators.get('complexity_handling', 0.5) * 100
                subscores['holistic_thinking'] = behavioral_patterns.response_consistency * 100
        
        # Return empty dict if no patterns available - let main fallback handle scoring
        # The main evaluate_response method will call _calculate_fallback_subscores
        
        return subscores
    
    def _calculate_overall_score(self, 
                                cognitive_subscores: Dict[str, float],
                                cognitive_domain: str) -> float:
        """Calculate weighted overall score for cognitive domain"""
        
        if not cognitive_subscores:
            return 0.0
            
        domain_config = self.cognitive_mappings.get(cognitive_domain, {})
        weight_factors = domain_config.get('weight_factors', {})
        
        # If no weights defined, use equal weighting
        if not weight_factors:
            return statistics.mean(cognitive_subscores.values())
        
        # Apply domain-specific weighting
        weighted_sum = 0.0
        total_weight = 0.0
        
        for ability, score in cognitive_subscores.items():
            # Map ability to weight category (simplified mapping)
            weight_key = self._map_ability_to_weight(ability, weight_factors)
            weight = weight_factors.get(weight_key, 0.33)  # Default equal weight
            
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / max(total_weight, 1.0)
    
    def _map_ability_to_weight(self, ability: str, weight_factors: Dict[str, float]) -> str:
        """Map cognitive ability to weight factor category"""
        
        # Simple mapping based on ability keywords
        ability_lower = ability.lower()
        
        if 'accuracy' in ability_lower or 'recall' in ability_lower:
            return 'accuracy'
        elif 'consistency' in ability_lower or 'coherence' in ability_lower:  
            return 'consistency'
        elif 'complexity' in ability_lower or 'reasoning' in ability_lower:
            return 'complexity'
        elif 'originality' in ability_lower or 'novel' in ability_lower:
            return 'originality'
        elif 'cultural' in ability_lower or 'empathy' in ability_lower:
            return 'cultural_sensitivity'
        elif 'synthesis' in ability_lower or 'integration' in ability_lower:
            return 'synthesis'
        else:
            # Default to first available weight
            return list(weight_factors.keys())[0] if weight_factors else 'default'
    
    def _calculate_fallback_quality_score(self, evaluation_details: Dict[str, Any]) -> float:
        """Calculate fallback quality score when sophisticated evaluators unavailable"""
        
        # Simple heuristics - this is the fallback when evaluators fail
        response_length_score = 50.0  # Neutral baseline
        
        # Try to extract any scores from evaluation details
        if evaluation_details:
            for key, value in evaluation_details.items():
                if isinstance(value, (int, float)) and 0 <= value <= 100:
                    return float(value)
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float)) and 0 <= subvalue <= 100:
                            return float(subvalue)
        
        return response_length_score
    
    def _calculate_confidence_score(self, result: CognitiveEvaluationResult) -> float:
        """Calculate confidence in evaluation results"""
        
        confidence_factors = []
        
        # Factor 1: Pattern analysis availability
        if result.behavioral_patterns:
            confidence_factors.append(0.8)  # High confidence with patterns
        else:
            confidence_factors.append(0.3)  # Low confidence without patterns
        
        # Factor 2: Number of subscores available
        if result.cognitive_subscores:
            score_completeness = len(result.cognitive_subscores) / 3  # Assuming 3 key abilities
            confidence_factors.append(min(score_completeness, 1.0))
        else:
            confidence_factors.append(0.2)
        
        # Factor 3: Cultural analysis for social domain
        if result.cognitive_domain == 'social' and result.cultural_analysis:
            confidence_factors.append(0.9)
        elif result.cognitive_domain != 'social':
            confidence_factors.append(0.7)  # Not applicable
        else:
            confidence_factors.append(0.4)  # Missing cultural analysis for social domain
        
        return statistics.mean(confidence_factors)
    
    def _calculate_pattern_strength(self, behavioral_patterns: Optional[PatternAnalysisResult]) -> float:
        """Calculate strength of detected behavioral patterns"""
        
        if not behavioral_patterns:
            return 0.0
        
        # Use response consistency as proxy for pattern strength
        return behavioral_patterns.response_consistency
    
    def _calculate_consistency_measure(self, behavioral_patterns: Optional[PatternAnalysisResult]) -> float:
        """Calculate behavioral consistency measure"""
        
        if not behavioral_patterns:
            return 0.0
        
        return behavioral_patterns.response_consistency
    
    def get_evaluation_summary(self, result: CognitiveEvaluationResult) -> str:
        """Generate human-readable evaluation summary"""
        
        summary = f"""
🧠 COGNITIVE EVALUATION SUMMARY
===============================
Test ID: {result.test_id}
Cognitive Domain: {result.cognitive_domain.title()}
Overall Score: {result.overall_score:.1f}/100
Confidence: {result.confidence_score:.2f}

📊 COGNITIVE SUBSCORES:
"""
        
        for ability, score in result.cognitive_subscores.items():
            ability_name = ability.replace('_', ' ').title()
            summary += f"    {ability_name}: {score:.1f}/100\n"
        
        if result.behavioral_patterns:
            summary += f"""
🔍 BEHAVIORAL PATTERNS:
    Response Consistency: {result.behavioral_patterns.response_consistency:.2f}
    Pattern Adherence: {result.behavioral_patterns.pattern_adherence:.2f}
    Pattern Strength: {result.pattern_strength:.2f}
"""
        
        if result.cultural_analysis and result.cognitive_domain == 'social':
            summary += f"""
🌍 CULTURAL ANALYSIS:
    Cultural Competency: {result.cultural_analysis.get('cultural_competency', 0)*100:.1f}/100
    Respectful Language: {result.cultural_analysis.get('respectful_language', 0)*100:.1f}/100
"""
        
        return summary
    
    def _calculate_fallback_subscores(self, 
                                     response_text: str, 
                                     prompt: str,
                                     cognitive_domain: str) -> Dict[str, float]:
        """Calculate fallback subscores when sophisticated evaluators fail"""
        
        # Simple heuristic scoring based on response characteristics
        response_length = len(response_text.strip())
        word_count = len(response_text.split())
        
        # Length appropriateness score (40-80 good range based on 400 tokens)
        if 100 <= response_length <= 2000:
            length_score = 75.0
        elif response_length < 50:
            length_score = 30.0  # Too short
        else:
            length_score = 60.0  # Acceptable
        
        # Basic content quality score
        unique_words = len(set(response_text.lower().split()))
        word_diversity = unique_words / max(word_count, 1) if word_count > 0 else 0.3
        content_score = min(word_diversity * 120, 85.0)  # Cap at 85
        
        # Prompt relevance score
        prompt_words = set(prompt.lower().split())
        response_words = set(response_text.lower().split())
        relevance_ratio = len(prompt_words & response_words) / max(len(prompt_words), 1)
        relevance_score = min(relevance_ratio * 100, 80.0)
        
        # Create domain-appropriate subscores
        domain_config = self.cognitive_mappings.get(cognitive_domain, {})
        key_abilities = domain_config.get('key_abilities', ['general_ability'])
        
        # Assign scores based on different heuristics for variety
        subscores = {}
        if len(key_abilities) >= 1:
            subscores[key_abilities[0]] = length_score
        if len(key_abilities) >= 2:
            subscores[key_abilities[1]] = content_score
        if len(key_abilities) >= 3:
            subscores[key_abilities[2]] = relevance_score
            
        # Fill remaining abilities with average
        avg_score = (length_score + content_score + relevance_score) / 3
        for ability in key_abilities[3:]:
            subscores[ability] = avg_score
        
        logger.debug(f"Fallback scoring for {cognitive_domain}: {subscores}")
        return subscores