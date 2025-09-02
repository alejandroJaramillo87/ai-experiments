"""
Multi-Source Fact Validation Ensemble System

Combines multiple fact validation sources to create a robust, culturally-aware fact-checking system.
Integrates with existing ensemble disagreement detection framework.

Sources:
- Internal Knowledge Validator (existing factual tests)
- Wikipedia Fact Checker (external validation) 
- Cultural Authenticity Analysis (cultural perspective validation)
- Future: Wikidata, academic databases, cultural knowledge bases

Features:
- Ensemble disagreement detection across validation sources
- Cultural bias detection and adjustment
- Confidence calibration using multiple validation strategies
- Graceful degradation when sources are unavailable

"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics

from .wikipedia_fact_checker import WikipediaFactChecker, FactCheckingResult, integrate_with_ensemble_evaluation
from .knowledge_validator import KnowledgeValidator

# Set up logging
logging.basicConfig(level=logging.INFO)  
logger = logging.getLogger(__name__)


class ValidationSource(Enum):
    """Sources of fact validation"""
    INTERNAL_KNOWLEDGE = "internal_knowledge"
    WIKIPEDIA_EXTERNAL = "wikipedia_external" 
    CULTURAL_AUTHENTICITY = "cultural_authenticity"
    WIKIDATA_STRUCTURED = "wikidata_structured"  # Future enhancement
    ACADEMIC_DATABASES = "academic_databases"    # Future enhancement


@dataclass
class SourceValidationResult:
    """Result from a single validation source"""
    source: ValidationSource
    confidence: float              # 0-1 confidence in factual accuracy
    cultural_sensitivity: float   # 0-1 cultural sensitivity score
    evidence_quality: float       # 0-1 quality of evidence/sources
    uncertainty_factors: List[str] # Factors that increase uncertainty
    details: Dict[str, Any]       # Source-specific details
    processing_time: float        # Time taken for validation
    available: bool = True        # Whether source was available


@dataclass
class EnsembleFactValidationResult:
    """Comprehensive fact validation result from multiple sources"""
    original_text: str
    source_results: List[SourceValidationResult]
    
    # Ensemble metrics
    ensemble_confidence: float      # Weighted confidence across sources
    ensemble_disagreement: float    # Disagreement between sources (0-1)
    confidence_reliability: float   # How reliable the confidence estimate is
    
    # Cultural analysis
    cultural_sensitivity_score: float  # Overall cultural sensitivity
    cultural_bias_detected: bool      # Whether significant bias was detected
    cultural_perspectives: List[str]   # Different cultural perspectives found
    
    # Recommendations
    recommendations: List[str]         # Actionable recommendations
    high_confidence_claims: List[str]  # Claims with high validation confidence
    disputed_claims: List[str]         # Claims with disagreement between sources
    
    # Integration with existing system
    integration_notes: List[str]       # Notes on integration with linguistic confidence
    overall_assessment: str           # Summary assessment
    

class MultiSourceFactValidator:
    """
    Multi-source ensemble fact validation system.
    
    Combines multiple validation sources with disagreement detection and cultural
    bias awareness to provide robust fact-checking capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the multi-source fact validator"""
        self.config = config or self._get_default_config()
        
        # Initialize validation sources
        self.sources = {}
        self._initialize_validation_sources()
        
        # Ensemble configuration
        self.source_weights = self.config.get('source_weights', {
            ValidationSource.INTERNAL_KNOWLEDGE: 0.3,
            ValidationSource.WIKIPEDIA_EXTERNAL: 0.4,
            ValidationSource.CULTURAL_AUTHENTICITY: 0.3
        })
        
        # Thresholds
        self.disagreement_threshold = self.config.get('disagreement_threshold', 0.4)
        self.cultural_bias_threshold = self.config.get('cultural_bias_threshold', 0.6)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
    
    def validate_factual_content(self, 
                                text: str,
                                domain_context: Optional[str] = None,
                                cultural_context: Optional[Dict[str, Any]] = None,
                                internal_confidence: Optional[float] = None) -> EnsembleFactValidationResult:
        """
        Main entry point for multi-source fact validation.
        
        Args:
            text: Text to validate
            domain_context: Optional domain context  
            cultural_context: Optional cultural context information
            internal_confidence: Optional confidence from linguistic analysis
            
        Returns:
            EnsembleFactValidationResult with comprehensive analysis
        """
        try:
            import time
            start_time = time.time()
            
            # Run validation across all available sources
            source_results = []
            
            # Internal Knowledge Validation
            if ValidationSource.INTERNAL_KNOWLEDGE in self.sources:
                internal_result = self._validate_with_internal_knowledge(text, domain_context)
                if internal_result:
                    source_results.append(internal_result)
            
            # Wikipedia External Validation  
            if ValidationSource.WIKIPEDIA_EXTERNAL in self.sources:
                wikipedia_result = self._validate_with_wikipedia(text, domain_context)
                if wikipedia_result:
                    source_results.append(wikipedia_result)
            
            # Cultural Authenticity Validation
            if ValidationSource.CULTURAL_AUTHENTICITY in self.sources:
                cultural_result = self._validate_cultural_authenticity(text, cultural_context)
                if cultural_result:
                    source_results.append(cultural_result)
            
            # Calculate ensemble metrics
            ensemble_confidence = self._calculate_ensemble_confidence(source_results)
            disagreement = self._calculate_ensemble_disagreement(source_results)
            reliability = self._calculate_confidence_reliability(source_results, disagreement)
            
            # Cultural analysis
            cultural_sensitivity = self._analyze_cultural_sensitivity(source_results)
            bias_detected = self._detect_cultural_bias(source_results)
            perspectives = self._extract_cultural_perspectives(source_results)
            
            # Generate recommendations
            recommendations = self._generate_ensemble_recommendations(source_results, disagreement, bias_detected)
            high_confidence_claims = self._extract_high_confidence_claims(source_results)
            disputed_claims = self._extract_disputed_claims(source_results)
            
            # Integration with existing linguistic confidence
            integration_notes = []
            if internal_confidence is not None:
                integration_notes = self._generate_integration_notes(
                    internal_confidence, ensemble_confidence, disagreement
                )
            
            # Overall assessment
            assessment = self._generate_overall_assessment(
                ensemble_confidence, disagreement, bias_detected, len(source_results)
            )
            
            total_time = time.time() - start_time
            logger.info(f"Multi-source validation completed in {total_time:.2f}s with {len(source_results)} sources")
            
            return EnsembleFactValidationResult(
                original_text=text,
                source_results=source_results,
                ensemble_confidence=ensemble_confidence,
                ensemble_disagreement=disagreement,
                confidence_reliability=reliability,
                cultural_sensitivity_score=cultural_sensitivity,
                cultural_bias_detected=bias_detected,
                cultural_perspectives=perspectives,
                recommendations=recommendations,
                high_confidence_claims=high_confidence_claims,
                disputed_claims=disputed_claims,
                integration_notes=integration_notes,
                overall_assessment=assessment
            )
            
        except Exception as e:
            logger.error(f"Multi-source fact validation failed: {str(e)}")
            return self._create_error_result(text, str(e))
    
    def _initialize_validation_sources(self):
        """Initialize available validation sources"""
        try:
            # Internal Knowledge Validator
            self.sources[ValidationSource.INTERNAL_KNOWLEDGE] = KnowledgeValidator()
            logger.info("Internal Knowledge Validator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Internal Knowledge Validator: {e}")
        
        try:
            # Wikipedia Fact Checker
            wikipedia_config = self.config.get('wikipedia', {})
            self.sources[ValidationSource.WIKIPEDIA_EXTERNAL] = WikipediaFactChecker(wikipedia_config)
            logger.info("Wikipedia Fact Checker initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Wikipedia Fact Checker: {e}")
        
        # Cultural Authenticity (placeholder - would integrate with existing cultural modules)
        try:
            # This would integrate with existing cultural authenticity analyzer
            # For now, using a placeholder
            self.sources[ValidationSource.CULTURAL_AUTHENTICITY] = "placeholder"
            logger.info("Cultural Authenticity Validator placeholder initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Cultural Authenticity Validator: {e}")
    
    def _validate_with_internal_knowledge(self, text: str, domain_context: Optional[str] = None) -> Optional[SourceValidationResult]:
        """Validate using internal knowledge validator"""
        try:
            validator = self.sources[ValidationSource.INTERNAL_KNOWLEDGE]
            
            import time
            start_time = time.time()
            
            # Run internal validation with simplified adapter
            # Note: KnowledgeValidator is designed for interactive testing
            # For text-based validation, we'll use a simplified approach
            def simple_model_func(question: str) -> str:
                # Simple heuristic: check if the text contains relevant information
                if any(keyword.lower() in text.lower() for keyword in question.split()[:3]):
                    return text[:200]  # Return relevant portion
                return "I don't know"
            
            result = validator.validate_factual_knowledge(simple_model_func, [domain_context] if domain_context else None, 2)
            processing_time = time.time() - start_time
            
            # Convert to standard format
            confidence = result.overall_accuracy if hasattr(result, 'overall_accuracy') else 0.5
            cultural_sensitivity = 0.8  # Default cultural sensitivity for internal validation
            evidence_quality = result.confidence_score if hasattr(result, 'confidence_score') else 0.6
            
            uncertainty_factors = []
            if hasattr(result, 'inconsistency_flags'):
                uncertainty_factors.extend([f"Internal inconsistency: {flag}" for flag in result.inconsistency_flags])
            
            return SourceValidationResult(
                source=ValidationSource.INTERNAL_KNOWLEDGE,
                confidence=confidence,
                cultural_sensitivity=cultural_sensitivity,
                evidence_quality=evidence_quality,
                uncertainty_factors=uncertainty_factors,
                details={'internal_result': result},
                processing_time=processing_time,
                available=True
            )
            
        except Exception as e:
            logger.error(f"Internal knowledge validation failed: {str(e)}")
            return SourceValidationResult(
                source=ValidationSource.INTERNAL_KNOWLEDGE,
                confidence=0.0,
                cultural_sensitivity=0.5,
                evidence_quality=0.0,
                uncertainty_factors=[f"Internal validation failed: {str(e)}"],
                details={'error': str(e)},
                processing_time=0.0,
                available=False
            )
    
    def _validate_with_wikipedia(self, text: str, domain_context: Optional[str] = None) -> Optional[SourceValidationResult]:
        """Validate using Wikipedia fact checker"""
        try:
            fact_checker = self.sources[ValidationSource.WIKIPEDIA_EXTERNAL]
            
            import time
            start_time = time.time()
            
            # Run Wikipedia validation
            result = fact_checker.check_factual_claims(text, domain_context)
            processing_time = time.time() - start_time
            
            # Extract metrics
            confidence = result.overall_factual_confidence
            cultural_sensitivity = result.cultural_sensitivity_score
            evidence_quality = self._assess_wikipedia_evidence_quality(result)
            
            uncertainty_factors = []
            for validation_result in result.validation_results:
                uncertainty_factors.extend(validation_result.uncertainty_factors)
            
            return SourceValidationResult(
                source=ValidationSource.WIKIPEDIA_EXTERNAL,
                confidence=confidence,
                cultural_sensitivity=cultural_sensitivity,
                evidence_quality=evidence_quality,
                uncertainty_factors=uncertainty_factors,
                details={'wikipedia_result': result},
                processing_time=processing_time,
                available=True
            )
            
        except Exception as e:
            logger.error(f"Wikipedia validation failed: {str(e)}")
            return SourceValidationResult(
                source=ValidationSource.WIKIPEDIA_EXTERNAL,
                confidence=0.0,
                cultural_sensitivity=0.5,
                evidence_quality=0.0,
                uncertainty_factors=[f"Wikipedia validation failed: {str(e)}"],
                details={'error': str(e)},
                processing_time=0.0,
                available=False
            )
    
    def _validate_cultural_authenticity(self, text: str, cultural_context: Optional[Dict[str, Any]] = None) -> Optional[SourceValidationResult]:
        """Validate cultural authenticity (placeholder implementation)"""
        try:
            # This is a placeholder implementation
            # In the full system, this would integrate with existing cultural authenticity analyzer
            
            import time
            start_time = time.time()
            
            # Placeholder cultural validation
            confidence = 0.7  # Placeholder confidence
            cultural_sensitivity = 0.9  # High cultural sensitivity from cultural source
            evidence_quality = 0.6  # Moderate evidence quality
            
            uncertainty_factors = []
            if not cultural_context:
                uncertainty_factors.append("No cultural context provided for cultural validation")
            
            processing_time = time.time() - start_time
            
            return SourceValidationResult(
                source=ValidationSource.CULTURAL_AUTHENTICITY,
                confidence=confidence,
                cultural_sensitivity=cultural_sensitivity,
                evidence_quality=evidence_quality,
                uncertainty_factors=uncertainty_factors,
                details={'cultural_context': cultural_context, 'placeholder': True},
                processing_time=processing_time,
                available=True
            )
            
        except Exception as e:
            logger.error(f"Cultural authenticity validation failed: {str(e)}")
            return SourceValidationResult(
                source=ValidationSource.CULTURAL_AUTHENTICITY,
                confidence=0.0,
                cultural_sensitivity=0.5,
                evidence_quality=0.0,
                uncertainty_factors=[f"Cultural validation failed: {str(e)}"],
                details={'error': str(e)},
                processing_time=0.0,
                available=False
            )
    
    def _assess_wikipedia_evidence_quality(self, result: FactCheckingResult) -> float:
        """Assess the quality of Wikipedia evidence"""
        if not result.validation_results:
            return 0.0
        
        quality_scores = []
        for validation_result in result.validation_results:
            # Base quality on sources quality and number of supporting articles
            source_quality = validation_result.sources_quality
            support_count = len(validation_result.supporting_articles)
            contradict_count = len(validation_result.contradicting_evidence)
            
            # Higher quality for more supporting evidence, lower for contradictions
            evidence_balance = support_count / max(1, support_count + contradict_count)
            
            quality = source_quality * 0.7 + evidence_balance * 0.3
            quality_scores.append(quality)
        
        return statistics.mean(quality_scores) if quality_scores else 0.0
    
    def _calculate_ensemble_confidence(self, source_results: List[SourceValidationResult]) -> float:
        """Calculate weighted ensemble confidence"""
        if not source_results:
            return 0.0
        
        available_results = [r for r in source_results if r.available]
        if not available_results:
            return 0.0
        
        # Calculate weighted confidence
        total_weight = 0.0
        weighted_confidence = 0.0
        
        for result in available_results:
            weight = self.source_weights.get(result.source, 1.0)
            # Adjust weight by evidence quality
            adjusted_weight = weight * result.evidence_quality
            
            weighted_confidence += result.confidence * adjusted_weight
            total_weight += adjusted_weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _calculate_ensemble_disagreement(self, source_results: List[SourceValidationResult]) -> float:
        """Calculate disagreement between validation sources"""
        available_results = [r for r in source_results if r.available]
        if len(available_results) < 2:
            return 0.0
        
        confidences = [r.confidence for r in available_results]
        mean_confidence = statistics.mean(confidences)
        
        # Calculate variance (disagreement)
        variance = statistics.variance(confidences) if len(confidences) > 1 else 0.0
        
        # Normalize to [0,1] scale
        return min(1.0, variance * 4)
    
    def _calculate_confidence_reliability(self, source_results: List[SourceValidationResult], disagreement: float) -> float:
        """Calculate how reliable the confidence estimate is"""
        if not source_results:
            return 0.0
        
        # Base reliability on inverse of disagreement
        disagreement_factor = 1.0 - disagreement
        
        # Adjust by number of available sources (more sources = higher reliability)
        available_count = len([r for r in source_results if r.available])
        source_factor = min(1.0, available_count / 3.0)  # Normalize by expected 3 sources
        
        # Adjust by evidence quality
        evidence_qualities = [r.evidence_quality for r in source_results if r.available]
        evidence_factor = statistics.mean(evidence_qualities) if evidence_qualities else 0.0
        
        return (disagreement_factor * 0.5 + source_factor * 0.3 + evidence_factor * 0.2)
    
    def _analyze_cultural_sensitivity(self, source_results: List[SourceValidationResult]) -> float:
        """Analyze overall cultural sensitivity"""
        if not source_results:
            return 0.5
        
        available_results = [r for r in source_results if r.available]
        if not available_results:
            return 0.5
        
        # Weight cultural sensitivity by source reliability
        sensitivity_scores = [r.cultural_sensitivity for r in available_results]
        return statistics.mean(sensitivity_scores)
    
    def _detect_cultural_bias(self, source_results: List[SourceValidationResult]) -> bool:
        """Detect if significant cultural bias is present"""
        for result in source_results:
            if not result.available:
                continue
                
            # Check for bias indicators in Wikipedia results
            if result.source == ValidationSource.WIKIPEDIA_EXTERNAL:
                wikipedia_result = result.details.get('wikipedia_result')
                if wikipedia_result:
                    for validation_result in wikipedia_result.validation_results:
                        if validation_result.cultural_bias_score > self.cultural_bias_threshold:
                            return True
            
            # Check cultural sensitivity threshold
            if result.cultural_sensitivity < (1.0 - self.cultural_bias_threshold):
                return True
        
        return False
    
    def _extract_cultural_perspectives(self, source_results: List[SourceValidationResult]) -> List[str]:
        """Extract different cultural perspectives found"""
        perspectives = []
        
        for result in source_results:
            if not result.available:
                continue
                
            if result.source == ValidationSource.WIKIPEDIA_EXTERNAL:
                wikipedia_result = result.details.get('wikipedia_result')
                if wikipedia_result:
                    for validation_result in wikipedia_result.validation_results:
                        if validation_result.claim.cultural_context:
                            perspectives.append(f"Wikipedia: {validation_result.claim.cultural_context}")
            
            elif result.source == ValidationSource.CULTURAL_AUTHENTICITY:
                cultural_context = result.details.get('cultural_context')
                if cultural_context:
                    perspectives.append(f"Cultural: {cultural_context}")
        
        return list(set(perspectives))  # Remove duplicates
    
    def _generate_ensemble_recommendations(self, 
                                         source_results: List[SourceValidationResult],
                                         disagreement: float,
                                         bias_detected: bool) -> List[str]:
        """Generate recommendations based on ensemble analysis"""
        recommendations = []
        
        available_sources = len([r for r in source_results if r.available])
        
        # Source availability recommendations
        if available_sources < 2:
            recommendations.append("Limited validation sources available - consider manual fact-checking")
        
        # Disagreement recommendations
        if disagreement > self.disagreement_threshold:
            recommendations.append("High disagreement between validation sources detected - verify claims manually")
        
        # Cultural bias recommendations
        if bias_detected:
            recommendations.append("Cultural bias detected in validation sources - consider alternative cultural perspectives")
        
        # Source-specific recommendations
        for result in source_results:
            if result.available and result.uncertainty_factors:
                recommendations.append(f"{result.source.value}: {result.uncertainty_factors[0]}")
        
        # Evidence quality recommendations
        low_quality_sources = [r for r in source_results if r.available and r.evidence_quality < 0.3]
        if low_quality_sources:
            recommendations.append("Some validation sources have low evidence quality - cross-check with authoritative sources")
        
        return recommendations
    
    def _extract_high_confidence_claims(self, source_results: List[SourceValidationResult]) -> List[str]:
        """Extract claims with high validation confidence"""
        high_confidence_claims = []
        
        for result in source_results:
            if not result.available or result.confidence < self.confidence_threshold:
                continue
            
            if result.source == ValidationSource.WIKIPEDIA_EXTERNAL:
                wikipedia_result = result.details.get('wikipedia_result')
                if wikipedia_result:
                    for validation_result in wikipedia_result.validation_results:
                        if validation_result.wikipedia_confidence > self.confidence_threshold:
                            high_confidence_claims.append(validation_result.claim.text)
        
        return high_confidence_claims
    
    def _extract_disputed_claims(self, source_results: List[SourceValidationResult]) -> List[str]:
        """Extract claims with disagreement between sources"""
        disputed_claims = []
        
        # This would require more sophisticated claim matching across sources
        # For now, use uncertainty factors as proxy for disputed claims
        for result in source_results:
            if result.available and result.uncertainty_factors:
                if result.source == ValidationSource.WIKIPEDIA_EXTERNAL:
                    wikipedia_result = result.details.get('wikipedia_result')
                    if wikipedia_result:
                        for validation_result in wikipedia_result.validation_results:
                            if validation_result.contradicting_evidence:
                                disputed_claims.append(validation_result.claim.text)
        
        return disputed_claims
    
    def _generate_integration_notes(self, 
                                   internal_confidence: float,
                                   ensemble_confidence: float, 
                                   disagreement: float) -> List[str]:
        """Generate notes on integration with linguistic confidence"""
        notes = []
        
        confidence_gap = abs(internal_confidence - ensemble_confidence)
        
        if confidence_gap > 0.3:
            if internal_confidence > ensemble_confidence:
                notes.append("Linguistic confidence higher than fact validation - possible overconfidence")
            else:
                notes.append("Fact validation confidence higher than linguistic - claims well-supported externally")
        
        if disagreement > 0.4:
            notes.append("High disagreement between validation sources increases overall uncertainty")
        
        if ensemble_confidence > 0.8 and internal_confidence > 0.8:
            notes.append("High agreement between linguistic and factual validation - high overall confidence")
        
        return notes
    
    def _generate_overall_assessment(self, 
                                   confidence: float,
                                   disagreement: float,
                                   bias_detected: bool,
                                   source_count: int) -> str:
        """Generate overall assessment summary"""
        
        if source_count == 0:
            return "No validation sources available"
        
        confidence_level = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
        disagreement_level = "high" if disagreement > 0.6 else "medium" if disagreement > 0.3 else "low"
        
        assessment = f"Factual confidence: {confidence_level} ({confidence:.2f})"
        assessment += f" | Source disagreement: {disagreement_level} ({disagreement:.2f})"
        assessment += f" | Sources: {source_count}"
        
        if bias_detected:
            assessment += " | Cultural bias detected"
        
        return assessment
    
    def _create_error_result(self, text: str, error_message: str) -> EnsembleFactValidationResult:
        """Create error result when validation fails"""
        return EnsembleFactValidationResult(
            original_text=text,
            source_results=[],
            ensemble_confidence=0.0,
            ensemble_disagreement=1.0,
            confidence_reliability=0.0,
            cultural_sensitivity_score=0.5,
            cultural_bias_detected=False,
            cultural_perspectives=[],
            recommendations=[f"Multi-source validation failed: {error_message}"],
            high_confidence_claims=[],
            disputed_claims=[],
            integration_notes=[],
            overall_assessment="Validation failed"
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'source_weights': {
                ValidationSource.INTERNAL_KNOWLEDGE: 0.3,
                ValidationSource.WIKIPEDIA_EXTERNAL: 0.4,
                ValidationSource.CULTURAL_AUTHENTICITY: 0.3
            },
            'disagreement_threshold': 0.4,
            'cultural_bias_threshold': 0.6,
            'confidence_threshold': 0.7,
            'wikipedia': {
                'min_request_interval': 0.5,
                'claim_extraction_threshold': 0.3,
                'max_claims_per_text': 10
            }
        }


# Integration function with existing evaluation system
def integrate_multi_source_validation(text: str,
                                     internal_confidence: float,
                                     ensemble_results: List[Dict],
                                     domain_context: Optional[str] = None,
                                     cultural_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Integrate multi-source fact validation with existing ensemble evaluation system.
    
    This is the main integration point for incorporating external fact validation
    into your existing sophisticated evaluation framework.
    """
    
    # Initialize multi-source validator
    validator = MultiSourceFactValidator()
    
    # Run multi-source validation
    fact_validation_result = validator.validate_factual_content(
        text=text,
        domain_context=domain_context,
        cultural_context=cultural_context,
        internal_confidence=internal_confidence
    )
    
    # Integrate with existing ensemble results
    enhanced_ensemble_results = []
    for existing_result in ensemble_results:
        enhanced_result = existing_result.copy()
        enhanced_result['fact_validation_confidence'] = fact_validation_result.ensemble_confidence
        enhanced_result['fact_validation_disagreement'] = fact_validation_result.ensemble_disagreement
        enhanced_result['cultural_sensitivity'] = fact_validation_result.cultural_sensitivity_score
        enhanced_ensemble_results.append(enhanced_result)
    
    # Calculate comprehensive confidence that includes fact validation
    all_confidences = [internal_confidence, fact_validation_result.ensemble_confidence]
    comprehensive_confidence = statistics.mean(all_confidences)
    
    # Calculate comprehensive disagreement including fact validation
    comprehensive_disagreement = max(
        fact_validation_result.ensemble_disagreement,
        abs(internal_confidence - fact_validation_result.ensemble_confidence)
    )
    
    return {
        'comprehensive_confidence': comprehensive_confidence,
        'comprehensive_disagreement': comprehensive_disagreement,
        'confidence_reliability': fact_validation_result.confidence_reliability,
        'fact_validation_details': fact_validation_result,
        'enhanced_ensemble_results': enhanced_ensemble_results,
        'integration_assessment': fact_validation_result.overall_assessment,
        'recommendations': fact_validation_result.recommendations,
        'cultural_analysis': {
            'sensitivity_score': fact_validation_result.cultural_sensitivity_score,
            'bias_detected': fact_validation_result.cultural_bias_detected,
            'perspectives': fact_validation_result.cultural_perspectives
        }
    }