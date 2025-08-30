"""
Domain-Aware Evaluation Router

Routes evaluation requests to appropriate domain-specific evaluators based on test metadata.
Integrates multiple domain evaluations with cross-domain cultural authenticity validation.

Author: Claude Code
Version: 1.0.0
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class Domain(Enum):
    """Evaluation domains"""
    CREATIVITY = "creativity"
    KNOWLEDGE = "knowledge"
    LANGUAGE = "language"
    REASONING = "reasoning"
    SOCIAL = "social"
    INTEGRATION = "integration"


class EvaluationType(Enum):
    """Types of evaluation approaches"""
    CREATIVE_EXPRESSION = "creative_expression"
    LINGUISTIC_COMPETENCE = "linguistic_competence" 
    SOCIAL_CONTEXT = "social_context"
    ALTERNATIVE_LOGIC = "alternative_logic"
    TRADITIONAL_KNOWLEDGE = "traditional_knowledge"
    GENERAL_REASONING = "general_reasoning"
    # Integration-specific evaluation types
    KNOWLEDGE_REASONING_SYNTHESIS = "knowledge_reasoning_synthesis"
    SOCIAL_CREATIVE_SOLUTIONS = "social_creative_solutions"
    MULTILINGUAL_KNOWLEDGE_EXPRESSION = "multilingual_knowledge_expression"
    CULTURALLY_SENSITIVE_REASONING = "culturally_sensitive_reasoning"
    COMPREHENSIVE_INTEGRATION = "comprehensive_integration"


@dataclass
class DomainEvaluationResult:
    """Result from a domain-specific evaluator"""
    domain: Domain
    evaluation_type: EvaluationType
    primary_score: float
    detailed_metrics: Dict[str, float]
    cultural_indicators: List[Dict[str, Any]]
    confidence: float
    analysis_details: Dict[str, Any]


@dataclass
class IntegratedEvaluationResult:
    """Integrated result from multiple domain evaluators"""
    overall_score: float
    domain_scores: Dict[Domain, float]
    cultural_authenticity_score: float
    domain_results: List[DomainEvaluationResult]
    synthesis_analysis: Dict[str, Any]
    recommendations: List[str]


class DomainMetadataExtractor:
    """Extracts domain and category information from test metadata"""
    
    def __init__(self):
        """Initialize domain metadata extractor"""
        self._init_domain_patterns()
        
    def _init_domain_patterns(self):
        """Initialize patterns for domain detection"""
        self.domain_patterns = {
            Domain.CREATIVITY: [
                'narrative', 'performance', 'descriptive', 'musical', 'adaptive', 
                'problem', 'interpretation', 'collaborative', 'authenticity',
                'griot', 'dreamtime', 'kamishibai', 'storytelling', 'creative'
            ],
            Domain.KNOWLEDGE: [
                'traditional', 'historical', 'geographic', 'mathematical',
                'social_systems', 'material', 'conflict', 'knowledge_systems',
                'wayfinding', 'tcm', 'astronomy', 'indigenous', 'cultural'
            ],
            Domain.LANGUAGE: [
                'historical_linguistics', 'writing_systems', 'multilingual',
                'register', 'cultural_communication', 'dialectal', 'pragmatic',
                'evolution', 'code_switching', 'linguistic'
            ],
            Domain.REASONING: [
                'basic_logic', 'chain_of_thought', 'multi_step', 'elementary_math',
                'cultural_reasoning', 'self_verification', 'logic_systems',
                'reasoning', 'inference', 'verification'
            ],
            Domain.SOCIAL: [
                'conflict_resolution', 'consensus', 'hierarchy', 'communication',
                'leadership', 'relationship', 'social', 'community', 'cultural'
            ]
        }
        
        self.evaluation_type_patterns = {
            EvaluationType.CREATIVE_EXPRESSION: [
                'narrative', 'performance', 'descriptive', 'musical', 'adaptive',
                'collaborative', 'authenticity', 'creative', 'storytelling'
            ],
            EvaluationType.LINGUISTIC_COMPETENCE: [
                'multilingual', 'register', 'dialectal', 'pragmatic', 'writing',
                'code_switching', 'linguistic', 'communication'
            ],
            EvaluationType.SOCIAL_CONTEXT: [
                'conflict_resolution', 'consensus', 'hierarchy', 'leadership',
                'relationship', 'social', 'community', 'cultural_communication'
            ],
            EvaluationType.ALTERNATIVE_LOGIC: [
                'logic_systems', 'cultural_reasoning', 'self_verification',
                'comparative', 'circular', 'dialectical', 'holistic'
            ],
            EvaluationType.TRADITIONAL_KNOWLEDGE: [
                'traditional', 'indigenous', 'knowledge_systems', 'material',
                'historical', 'geographic', 'mathematical_traditions'
            ],
            EvaluationType.GENERAL_REASONING: [
                'basic_logic', 'chain_of_thought', 'multi_step', 'elementary_math',
                'inference', 'verification', 'reasoning'
            ]
        }
    
    def extract_domain(self, test_metadata: Dict[str, Any]) -> Optional[Domain]:
        """Extract domain from test metadata"""
        
        # Direct domain specification
        if 'domain' in test_metadata:
            domain_str = test_metadata['domain'].lower()
            for domain in Domain:
                if domain.value == domain_str:
                    return domain
        
        # Extract from test name or category
        test_text = ' '.join([
            test_metadata.get('name', ''),
            test_metadata.get('category', ''),
            test_metadata.get('id', ''),
            test_metadata.get('description', '')
        ]).lower()
        
        domain_scores = {}
        for domain, patterns in self.domain_patterns.items():
            score = sum(1 for pattern in patterns if pattern in test_text)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return None
    
    def extract_evaluation_type(self, test_metadata: Dict[str, Any], domain: Domain) -> EvaluationType:
        """Extract evaluation type from test metadata and domain"""
        
        test_text = ' '.join([
            test_metadata.get('name', ''),
            test_metadata.get('category', ''),
            test_metadata.get('id', ''),
            test_metadata.get('description', '')
        ]).lower()
        
        type_scores = {}
        for eval_type, patterns in self.evaluation_type_patterns.items():
            score = sum(1 for pattern in patterns if pattern in test_text)
            if score > 0:
                type_scores[eval_type] = score
        
        if type_scores:
            return max(type_scores, key=type_scores.get)
        
        # Default mapping based on domain
        domain_defaults = {
            Domain.CREATIVITY: EvaluationType.CREATIVE_EXPRESSION,
            Domain.LANGUAGE: EvaluationType.LINGUISTIC_COMPETENCE,
            Domain.SOCIAL: EvaluationType.SOCIAL_CONTEXT,
            Domain.REASONING: EvaluationType.GENERAL_REASONING,
            Domain.KNOWLEDGE: EvaluationType.TRADITIONAL_KNOWLEDGE
        }
        
        return domain_defaults.get(domain, EvaluationType.GENERAL_REASONING)
    
    def extract_cultural_context(self, test_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract cultural context information"""
        
        cultural_indicators = {}
        test_text = ' '.join([
            test_metadata.get('name', ''),
            test_metadata.get('description', ''),
            test_metadata.get('prompt', '')
        ]).lower()
        
        # Extract cultural traditions mentioned
        cultural_traditions = {
            'african': ['african', 'griot', 'ubuntu', 'bantu'],
            'indigenous_australian': ['aboriginal', 'dreamtime', 'songlines'],
            'polynesian': ['polynesian', 'wayfinding', 'navigator'],
            'east_asian': ['chinese', 'japanese', 'kamishibai', 'tcm', 'confucian', 'taoist'],
            'indigenous_american': ['native', 'indigenous', 'tribal', 'elder'],
            'south_asian': ['ayurveda', 'vedic', 'sanskrit', 'hindu', 'buddhist'],
            'middle_eastern': ['arabic', 'islamic', 'persian', 'sufi'],
            'european': ['celtic', 'norse', 'slavic', 'mediterranean']
        }
        
        traditions_found = []
        for tradition, keywords in cultural_traditions.items():
            if any(keyword in test_text for keyword in keywords):
                traditions_found.append(tradition)
        
        cultural_indicators['traditions'] = traditions_found
        
        # Extract knowledge systems
        knowledge_systems = []
        if any(word in test_text for word in ['traditional', 'indigenous', 'folk', 'ancestral']):
            knowledge_systems.append('traditional')
        if any(word in test_text for word in ['oral', 'storytelling', 'narrative', 'spoken']):
            knowledge_systems.append('oral_tradition')
        if any(word in test_text for word in ['ceremonial', 'ritual', 'sacred', 'spiritual']):
            knowledge_systems.append('ceremonial')
            
        cultural_indicators['knowledge_systems'] = knowledge_systems
        
        # Extract performance aspects
        performance_aspects = []
        if any(word in test_text for word in ['call', 'response', 'audience', 'engagement']):
            performance_aspects.append('interactive')
        if any(word in test_text for word in ['rhythm', 'musical', 'sound', 'beat']):
            performance_aspects.append('rhythmic')
        if any(word in test_text for word in ['collaborative', 'community', 'collective']):
            performance_aspects.append('collaborative')
            
        cultural_indicators['performance_aspects'] = performance_aspects
        
        return cultural_indicators


class DomainEvaluationRouter:
    """Routes evaluation requests to appropriate domain-specific evaluators"""
    
    def __init__(self):
        """Initialize the domain evaluation router"""
        self.metadata_extractor = DomainMetadataExtractor()
        self.domain_evaluators = {}
        self._init_evaluators()
        
    def _init_evaluators(self):
        """Initialize domain-specific evaluators (lazy loading)"""
        # These will be loaded on demand
        self._creative_evaluator = None
        self._linguistic_evaluator = None
        self._social_evaluator = None
        self._reasoning_evaluator = None
        self._knowledge_evaluator = None
        self._integration_evaluator = None
        
    def get_creative_evaluator(self):
        """Get or create creative expression evaluator"""
        if self._creative_evaluator is None:
            try:
                from .creative_expression_evaluator import CreativeExpressionEvaluator
                self._creative_evaluator = CreativeExpressionEvaluator()
            except ImportError:
                logger.warning("CreativeExpressionEvaluator not available")
                self._creative_evaluator = None
        return self._creative_evaluator
    
    def get_linguistic_evaluator(self):
        """Get or create linguistic competence evaluator"""
        if self._linguistic_evaluator is None:
            try:
                from .linguistic_competence_evaluator import LinguisticCompetenceEvaluator
                self._linguistic_evaluator = LinguisticCompetenceEvaluator()
            except ImportError:
                logger.warning("LinguisticCompetenceEvaluator not available")
                self._linguistic_evaluator = None
        return self._linguistic_evaluator
    
    def get_social_evaluator(self):
        """Get or create social context evaluator"""
        if self._social_evaluator is None:
            try:
                from .social_context_evaluator import SocialContextEvaluator
                self._social_evaluator = SocialContextEvaluator()
            except ImportError:
                logger.warning("SocialContextEvaluator not available")
                self._social_evaluator = None
        return self._social_evaluator
    
    def get_reasoning_evaluator(self):
        """Get or create alternative reasoning evaluator"""
        if self._reasoning_evaluator is None:
            try:
                from .alternative_reasoning_evaluator import AlternativeReasoningEvaluator
                self._reasoning_evaluator = AlternativeReasoningEvaluator()
            except ImportError:
                logger.warning("AlternativeReasoningEvaluator not available")
                self._reasoning_evaluator = None
        return self._reasoning_evaluator
    
    def get_knowledge_evaluator(self):
        """Get or create traditional knowledge evaluator"""
        if self._knowledge_evaluator is None:
            try:
                from .traditional_knowledge_evaluator import TraditionalKnowledgeEvaluator
                self._knowledge_evaluator = TraditionalKnowledgeEvaluator()
            except ImportError:
                logger.warning("TraditionalKnowledgeEvaluator not available")
                self._knowledge_evaluator = None
        return self._knowledge_evaluator
    
    def get_integration_evaluator(self):
        """Get or create integration evaluator"""
        if self._integration_evaluator is None:
            try:
                from .integration_evaluator import IntegrationEvaluator
                self._integration_evaluator = IntegrationEvaluator()
            except ImportError:
                logger.warning("IntegrationEvaluator not available")
                self._integration_evaluator = None
        return self._integration_evaluator
    
    def route_evaluation(self, response_text: str, test_metadata: Dict[str, Any]) -> IntegratedEvaluationResult:
        """Route evaluation to appropriate domain-specific evaluators"""
        
        # Extract domain and evaluation type
        domain = self.metadata_extractor.extract_domain(test_metadata)
        if domain is None:
            domain = Domain.REASONING  # Default fallback
            
        evaluation_type = self.metadata_extractor.extract_evaluation_type(test_metadata, domain)
        cultural_context = self.metadata_extractor.extract_cultural_context(test_metadata)
        
        # Get domain-specific evaluators
        domain_results = []
        
        # Route to appropriate evaluators
        if evaluation_type == EvaluationType.CREATIVE_EXPRESSION:
            evaluator = self.get_creative_evaluator()
            if evaluator:
                result = evaluator.evaluate(response_text, test_metadata, cultural_context)
                domain_results.append(result)
                
        elif evaluation_type == EvaluationType.LINGUISTIC_COMPETENCE:
            evaluator = self.get_linguistic_evaluator()
            if evaluator:
                result = evaluator.evaluate(response_text, test_metadata, cultural_context)
                domain_results.append(result)
                
        elif evaluation_type == EvaluationType.SOCIAL_CONTEXT:
            evaluator = self.get_social_evaluator()
            if evaluator:
                result = evaluator.evaluate(response_text, test_metadata, cultural_context)
                domain_results.append(result)
                
        elif evaluation_type == EvaluationType.ALTERNATIVE_LOGIC:
            evaluator = self.get_reasoning_evaluator()
            if evaluator:
                result = evaluator.evaluate(response_text, test_metadata, cultural_context)
                domain_results.append(result)
                
        elif evaluation_type == EvaluationType.TRADITIONAL_KNOWLEDGE:
            evaluator = self.get_knowledge_evaluator()
            if evaluator:
                result = evaluator.evaluate(response_text, test_metadata, cultural_context)
                domain_results.append(result)
                
        # Integration domain evaluations
        elif evaluation_type in [EvaluationType.KNOWLEDGE_REASONING_SYNTHESIS,
                               EvaluationType.SOCIAL_CREATIVE_SOLUTIONS,
                               EvaluationType.MULTILINGUAL_KNOWLEDGE_EXPRESSION,
                               EvaluationType.CULTURALLY_SENSITIVE_REASONING,
                               EvaluationType.COMPREHENSIVE_INTEGRATION]:
            evaluator = self.get_integration_evaluator()
            if evaluator:
                result = evaluator.evaluate(response_text, test_metadata, cultural_context)
                domain_results.append(result)
        
        # Always include cultural authenticity evaluation for cultural domains
        if domain in [Domain.CREATIVITY, Domain.KNOWLEDGE, Domain.SOCIAL, Domain.INTEGRATION] or cultural_context.get('traditions'):
            try:
                from .cultural_authenticity import CulturalAuthenticityAnalyzer
                from .tradition_validator import TraditionalKnowledgeValidator
                from .cross_cultural_coherence import CrossCulturalCoherenceChecker
                
                # Run cultural evaluations
                cultural_analyzer = CulturalAuthenticityAnalyzer()
                tradition_validator = TraditionalKnowledgeValidator()
                coherence_checker = CrossCulturalCoherenceChecker()
                
                cultural_result = cultural_analyzer.analyze_cultural_authenticity(
                    response_text, cultural_context.get('primary_tradition')
                )
                tradition_result = tradition_validator.validate_traditional_knowledge(
                    response_text, cultural_context.get('knowledge_domain')
                )
                coherence_result = coherence_checker.check_cross_cultural_coherence(
                    response_text, cultural_context.get('primary_tradition')
                )
                
                # Create cultural domain result
                cultural_domain_result = DomainEvaluationResult(
                    domain=domain,
                    evaluation_type=EvaluationType.CREATIVE_EXPRESSION,  # Placeholder
                    primary_score=(cultural_result.authenticity_score + tradition_result.tradition_respect_score + coherence_result.coherence_score) / 3,
                    detailed_metrics={
                        'cultural_authenticity': cultural_result.authenticity_score,
                        'tradition_respect': tradition_result.tradition_respect_score,
                        'cross_cultural_coherence': coherence_result.coherence_score
                    },
                    cultural_indicators=[],
                    confidence=0.8,  # Base confidence for cultural evaluation
                    analysis_details={
                        'cultural_authenticity': cultural_result.detailed_analysis,
                        'tradition_validation': tradition_result.detailed_analysis,
                        'coherence_analysis': coherence_result.detailed_analysis
                    }
                )
                domain_results.append(cultural_domain_result)
                
            except ImportError:
                logger.warning("Cultural evaluation modules not available")
        
        # If no domain-specific evaluators available, create fallback result
        if not domain_results:
            fallback_result = DomainEvaluationResult(
                domain=domain,
                evaluation_type=evaluation_type,
                primary_score=0.5,  # Neutral score
                detailed_metrics={'general_score': 0.5},
                cultural_indicators=[],
                confidence=0.3,  # Low confidence for fallback
                analysis_details={'note': 'No domain-specific evaluator available'}
            )
            domain_results.append(fallback_result)
        
        # Integrate results
        return self._integrate_domain_results(domain_results, domain, cultural_context)
    
    def _integrate_domain_results(self, domain_results: List[DomainEvaluationResult], 
                                 primary_domain: Domain, cultural_context: Dict[str, Any]) -> IntegratedEvaluationResult:
        """Integrate results from multiple domain evaluators"""
        
        if not domain_results:
            return IntegratedEvaluationResult(
                overall_score=0.0,
                domain_scores={},
                cultural_authenticity_score=0.0,
                domain_results=[],
                synthesis_analysis={'error': 'No evaluation results'},
                recommendations=[]
            )
        
        # Calculate weighted overall score
        total_weight = 0
        weighted_score = 0
        domain_scores = {}
        
        for result in domain_results:
            weight = self._get_domain_weight(result.domain, result.evaluation_type, primary_domain)
            weighted_score += result.primary_score * weight
            total_weight += weight
            domain_scores[result.domain] = result.primary_score
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Extract cultural authenticity score
        cultural_authenticity_score = 0.0
        cultural_results = [r for r in domain_results if 'cultural_authenticity' in r.detailed_metrics]
        if cultural_results:
            cultural_authenticity_score = sum(r.detailed_metrics['cultural_authenticity'] for r in cultural_results) / len(cultural_results)
        
        # Generate synthesis analysis
        synthesis_analysis = {
            'primary_domain': primary_domain.value,
            'evaluation_types_used': [r.evaluation_type.value for r in domain_results],
            'cultural_context': cultural_context,
            'domain_count': len(set(r.domain for r in domain_results)),
            'average_confidence': sum(r.confidence for r in domain_results) / len(domain_results),
            'strengths': self._identify_strengths(domain_results),
            'areas_for_improvement': self._identify_improvements(domain_results)
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(domain_results, cultural_context)
        
        return IntegratedEvaluationResult(
            overall_score=overall_score * 100,  # Convert to 0-100 scale
            domain_scores={d: s * 100 for d, s in domain_scores.items()},
            cultural_authenticity_score=cultural_authenticity_score,
            domain_results=domain_results,
            synthesis_analysis=synthesis_analysis,
            recommendations=recommendations
        )
    
    def _get_domain_weight(self, result_domain: Domain, evaluation_type: EvaluationType, primary_domain: Domain) -> float:
        """Get weighting for domain result based on primary domain"""
        
        if result_domain == primary_domain:
            return 0.7  # Primary domain gets highest weight
        
        # Cultural evaluation always gets significant weight for cultural domains
        if evaluation_type in [EvaluationType.CREATIVE_EXPRESSION, EvaluationType.TRADITIONAL_KNOWLEDGE]:
            if primary_domain in [Domain.CREATIVITY, Domain.KNOWLEDGE, Domain.SOCIAL]:
                return 0.5
        
        return 0.3  # Secondary domains get lower weight
    
    def _identify_strengths(self, domain_results: List[DomainEvaluationResult]) -> List[str]:
        """Identify strengths from domain evaluation results"""
        strengths = []
        
        for result in domain_results:
            if result.primary_score > 0.7:
                domain_name = result.domain.value.replace('_', ' ').title()
                strengths.append(f"Strong {domain_name} competency")
                
            # Look for specific high-scoring metrics
            for metric, score in result.detailed_metrics.items():
                if score > 0.8:
                    metric_name = metric.replace('_', ' ').title()
                    strengths.append(f"Excellent {metric_name}")
        
        return strengths[:5]  # Limit to top 5 strengths
    
    def _identify_improvements(self, domain_results: List[DomainEvaluationResult]) -> List[str]:
        """Identify areas for improvement from domain evaluation results"""
        improvements = []
        
        for result in domain_results:
            if result.primary_score < 0.5:
                domain_name = result.domain.value.replace('_', ' ').title()
                improvements.append(f"Enhance {domain_name} competency")
                
            # Look for specific low-scoring metrics
            for metric, score in result.detailed_metrics.items():
                if score < 0.4:
                    metric_name = metric.replace('_', ' ').title()
                    improvements.append(f"Improve {metric_name}")
        
        return improvements[:5]  # Limit to top 5 improvements
    
    def _generate_recommendations(self, domain_results: List[DomainEvaluationResult], 
                                cultural_context: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Cultural authenticity recommendations
        cultural_results = [r for r in domain_results if 'cultural_authenticity' in r.detailed_metrics]
        if cultural_results:
            avg_cultural_score = sum(r.detailed_metrics['cultural_authenticity'] for r in cultural_results) / len(cultural_results)
            if avg_cultural_score < 0.6:
                recommendations.append("Consider more culturally authentic representation")
                recommendations.append("Include proper cultural context and attribution")
        
        # Domain-specific recommendations
        for result in domain_results:
            if result.evaluation_type == EvaluationType.CREATIVE_EXPRESSION and result.primary_score < 0.6:
                recommendations.append("Enhance creative expression with cultural storytelling patterns")
            elif result.evaluation_type == EvaluationType.LINGUISTIC_COMPETENCE and result.primary_score < 0.6:
                recommendations.append("Improve linguistic cultural competence and register awareness")
            elif result.evaluation_type == EvaluationType.SOCIAL_CONTEXT and result.primary_score < 0.6:
                recommendations.append("Develop better understanding of social cultural dynamics")
        
        # Cultural tradition-specific recommendations
        traditions = cultural_context.get('traditions', [])
        if traditions and any(r.primary_score < 0.5 for r in domain_results):
            recommendations.append(f"Study {', '.join(traditions)} cultural traditions more deeply")
        
        return recommendations[:7]  # Limit to top 7 recommendations