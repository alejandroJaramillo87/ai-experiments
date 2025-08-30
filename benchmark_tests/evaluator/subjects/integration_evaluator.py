from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import re
import statistics

from ..core.domain_evaluator_base import (
    MultiDimensionalEvaluator, EvaluationDimension, DomainEvaluationResult, CulturalContext
)
from .creativity_evaluator import CreativityEvaluator
from ..cultural.cultural_pattern_library import CulturalPatternLibrary


class IntegrationType(Enum):
    """Types of integration evaluation."""
    KNOWLEDGE_REASONING = "knowledge_reasoning_synthesis"
    SOCIAL_CREATIVITY = "social_creative_solutions"
    LANGUAGE_KNOWLEDGE = "multilingual_knowledge_expression"
    REASONING_SOCIAL = "culturally_sensitive_reasoning"
    COMPREHENSIVE = "comprehensive_integration"


@dataclass
class CrossDomainCoherence:
    """Analysis of coherence across multiple domains."""
    domains_integrated: List[str]
    integration_quality: float  # 0.0 to 1.0
    coherence_score: float  # 0.0 to 1.0
    transition_quality: float  # 0.0 to 1.0
    evidence: List[str]


class IntegrationEvaluator(MultiDimensionalEvaluator):
    """Evaluates cross-domain integration capabilities and comprehensive competence."""
    
    VERSION = "1.0.0"
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.pattern_library = CulturalPatternLibrary(config)
        self.creativity_evaluator = CreativityEvaluator(config)
        
    def _initialize_evaluator(self):
        """Initialize integration-specific components."""
        self.integration_patterns = {
            "knowledge_reasoning": [
                r"(?:traditional\s+knowledge|indigenous\s+wisdom)\s+(?:shows|demonstrates|proves|indicates)",
                r"(?:cultural\s+understanding|ancestral\s+wisdom)\s+(?:combined\s+with|integrated\s+with|supports)",
                r"(?:logical\s+analysis|reasoning)\s+(?:of|about|within)\s+(?:cultural|traditional|indigenous)",
                r"(?:evidence|data|facts)\s+(?:from|within)\s+(?:traditional|cultural|indigenous)\s+(?:systems|knowledge)"
            ],
            "social_creativity": [
                r"(?:creative\s+solution|innovative\s+approach|artistic\s+response)\s+(?:to|for)\s+(?:social|community|cultural)",
                r"(?:community|social)\s+(?:healing|building|strengthening)\s+(?:through|via|using)\s+(?:art|creativity|story)",
                r"(?:cultural\s+traditions|artistic\s+practices)\s+(?:address|solve|help\s+with)\s+(?:social|community)",
                r"(?:storytelling|performance|ritual)\s+(?:builds|creates|fosters)\s+(?:community|social|collective)"
            ],
            "language_knowledge": [
                r"(?:code-switching|multilingual|bilingual)\s+(?:expression|communication)\s+(?:of|about|regarding)",
                r"(?:traditional\s+knowledge|cultural\s+concepts)\s+(?:expressed|communicated|shared)\s+(?:in|through|via)",
                r"(?:multiple\s+languages|linguistic\s+variety|cultural\s+translation)\s+(?:conveys|expresses|transmits)",
                r"(?:language\s+choice|register\s+selection|code\s+selection)\s+(?:reflects|demonstrates|shows)\s+(?:cultural|knowledge)"
            ],
            "reasoning_social": [
                r"(?:cultural\s+sensitivity|social\s+awareness)\s+(?:in|within|during)\s+(?:reasoning|analysis|thinking)",
                r"(?:ethical\s+reasoning|moral\s+analysis)\s+(?:considers|includes|incorporates)\s+(?:cultural|social|community)",
                r"(?:multiple\s+perspectives|diverse\s+viewpoints|various\s+stakeholders)\s+(?:in|within)\s+(?:analysis|reasoning)",
                r"(?:logical\s+analysis|reasoning\s+process)\s+(?:respects|honors|considers)\s+(?:cultural|social|traditional)"
            ],
            "comprehensive": [
                r"(?:holistic|comprehensive|integrated|unified)\s+(?:approach|solution|analysis|understanding)",
                r"(?:multiple\s+domains|various\s+areas|different\s+aspects)\s+(?:combined|integrated|unified|synthesized)",
                r"(?:creative|linguistic|social|reasoning|knowledge)\s+(?:and|with|\+)\s+(?:creative|linguistic|social|reasoning|knowledge)",
                r"(?:cross-cultural|interdisciplinary|multi-faceted)\s+(?:competence|understanding|analysis|approach)"
            ]
        }
        
        self.domain_transition_markers = [
            "furthermore", "additionally", "moreover", "in addition", "also",
            "however", "nevertheless", "on the other hand", "conversely",
            "therefore", "thus", "consequently", "as a result",
            "similarly", "likewise", "in contrast", "meanwhile"
        ]
        
        self.integration_quality_indicators = [
            "synthesize", "integrate", "combine", "merge", "unify",
            "bridge", "connect", "link", "relate", "interweave",
            "coordinate", "harmonize", "balance", "reconcile"
        ]
    
    def get_supported_evaluation_types(self) -> List[str]:
        """Return supported evaluation types."""
        return [
            "knowledge_reasoning_synthesis",
            "social_creative_solutions", 
            "multilingual_knowledge_expression",
            "culturally_sensitive_reasoning",
            "comprehensive_integration"
        ]
    
    def get_evaluation_dimensions(self) -> List[str]:
        """Return integration evaluation dimensions."""
        return [
            "cross_domain_coherence",
            "cultural_authenticity_integration",
            "logical_consistency_across_domains",
            "creative_appropriateness",
            "social_awareness_integration",
            "synthesis_quality"
        ]
    
    def get_domain_name(self) -> str:
        """Return domain name."""
        return "integration"
    
    def evaluate_dimension(self, 
                          dimension: str,
                          response_text: str, 
                          test_metadata: Dict[str, Any], 
                          cultural_context: CulturalContext) -> EvaluationDimension:
        """Evaluate specific integration dimension."""
        
        if dimension == "cross_domain_coherence":
            return self._evaluate_cross_domain_coherence(response_text, test_metadata, cultural_context)
        elif dimension == "cultural_authenticity_integration":
            return self._evaluate_cultural_authenticity_integration(response_text, cultural_context)
        elif dimension == "logical_consistency_across_domains":
            return self._evaluate_logical_consistency(response_text, test_metadata, cultural_context)
        elif dimension == "creative_appropriateness":
            return self._evaluate_creative_appropriateness(response_text, test_metadata, cultural_context)
        elif dimension == "social_awareness_integration":
            return self._evaluate_social_awareness_integration(response_text, cultural_context)
        elif dimension == "synthesis_quality":
            return self._evaluate_synthesis_quality(response_text, test_metadata, cultural_context)
        else:
            # Fallback dimension
            return EvaluationDimension(
                name=dimension,
                score=0.0,
                confidence=0.0,
                cultural_relevance=0.0,
                evidence=["Unknown integration dimension"],
                cultural_markers=[]
            )
    
    def _evaluate_cross_domain_coherence(self, text: str, test_metadata: Dict[str, Any],
                                       cultural_context: CulturalContext) -> EvaluationDimension:
        """Evaluate how well response integrates multiple domain capabilities."""
        
        evidence = []
        cultural_markers = []
        
        # Determine expected domains from test metadata
        expected_domains = test_metadata.get('domains_required', [])
        if not expected_domains:
            # Infer from category
            category = test_metadata.get('category', '')
            if 'knowledge_reasoning' in category:
                expected_domains = ['knowledge', 'reasoning']
            elif 'social_creativity' in category:
                expected_domains = ['social', 'creativity']
            elif 'language_knowledge' in category:
                expected_domains = ['language', 'knowledge']
            elif 'reasoning_social' in category:
                expected_domains = ['reasoning', 'social']
            elif 'cross_domain' in category:
                expected_domains = ['knowledge', 'reasoning', 'social', 'language', 'creativity']
        
        # Analyze cross-domain coherence
        coherence_analysis = self._analyze_cross_domain_coherence(text, expected_domains)
        
        # Calculate score based on coherence quality
        coherence_score = coherence_analysis.coherence_score
        integration_score = coherence_analysis.integration_quality
        transition_score = coherence_analysis.transition_quality
        
        overall_score = (coherence_score * 0.4 + integration_score * 0.4 + transition_score * 0.2)
        
        evidence.extend(coherence_analysis.evidence)
        evidence.append(f"Domains integrated: {len(coherence_analysis.domains_integrated)}/{len(expected_domains)}")
        
        # Add cultural markers based on detected integrations
        for domain in coherence_analysis.domains_integrated:
            cultural_markers.append(f"integration:{domain}")
        
        # Assess confidence and cultural relevance
        confidence = 0.9 if len(coherence_analysis.domains_integrated) >= len(expected_domains) else 0.6
        cultural_relevance = 1.0 if cultural_context.traditions else 0.7
        
        return EvaluationDimension(
            name="cross_domain_coherence",
            score=overall_score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_cultural_authenticity_integration(self, text: str, 
                                                  cultural_context: CulturalContext) -> EvaluationDimension:
        """Evaluate cultural authenticity across integrated domains."""
        
        evidence = []
        cultural_markers = []
        
        # Detect cultural patterns across domains
        detected_patterns = self.pattern_library.detect_patterns(text, cultural_context.traditions)
        
        authenticity_score = 0.0
        if detected_patterns:
            # Calculate authenticity based on pattern confidence and variety
            pattern_types = set(p.pattern_type for p in detected_patterns)
            avg_confidence = statistics.mean([p.confidence for p in detected_patterns])
            type_diversity = len(pattern_types) / 6.0  # 6 total pattern types
            
            authenticity_score = (avg_confidence * 0.7 + type_diversity * 0.3)
            evidence.append(f"Detected {len(detected_patterns)} cultural patterns across {len(pattern_types)} types")
            
            for pattern in detected_patterns[:3]:  # Limit evidence
                cultural_markers.append(f"cultural:{pattern.tradition}:{pattern.pattern_name}")
        
        # Check for appropriation warnings across domains
        appropriation_warnings = self.pattern_library.check_appropriation_warnings(detected_patterns, text)
        appropriation_penalty = min(0.3, len(appropriation_warnings) * 0.1)
        
        if appropriation_warnings:
            evidence.extend([f"Appropriation concern: {w}" for w in appropriation_warnings[:2]])
        
        # Check for sacred boundary violations
        boundary_violations = self.pattern_library.check_sacred_boundaries(detected_patterns, text)
        boundary_penalty = min(0.5, len(boundary_violations) * 0.2)
        
        if boundary_violations:
            evidence.extend([f"Sacred boundary concern: {v}" for v in boundary_violations[:2]])
        
        # Calculate final authenticity score
        final_score = max(0.0, authenticity_score - appropriation_penalty - boundary_penalty)
        
        # Assess confidence and cultural relevance
        confidence = 0.8 if detected_patterns else 0.5
        cultural_relevance = 1.0 if cultural_context.traditions else 0.3
        
        if not evidence:
            evidence.append("Limited cultural authenticity detected across domains")
        
        return EvaluationDimension(
            name="cultural_authenticity_integration",
            score=final_score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_logical_consistency(self, text: str, test_metadata: Dict[str, Any],
                                    cultural_context: CulturalContext) -> EvaluationDimension:
        """Evaluate reasoning coherence across different knowledge systems."""
        
        evidence = []
        cultural_markers = []
        
        # Look for reasoning patterns
        reasoning_patterns = [
            r"(?:because|since|given\s+that|due\s+to)\s+.*?\s+(?:therefore|thus|hence|consequently)",
            r"(?:if|when|assuming)\s+.*?\s+(?:then|consequently|therefore)",
            r"(?:evidence|data|facts)\s+(?:shows|demonstrates|indicates|suggests)",
            r"(?:analysis|reasoning|logic)\s+(?:reveals|shows|indicates|demonstrates)"
        ]
        
        reasoning_score = 0.0
        reasoning_matches = 0
        for pattern in reasoning_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            reasoning_matches += matches
            if matches > 0:
                evidence.append(f"Logical structure: {pattern}")
        
        reasoning_score = min(1.0, reasoning_matches / 3.0)
        
        # Check for cultural logic integration
        cultural_logic_score = 0.0
        cultural_logic_patterns = [
            r"(?:traditional|cultural|indigenous)\s+(?:logic|reasoning|understanding|knowledge)",
            r"(?:multiple|different|various)\s+(?:perspectives|viewpoints|approaches|frameworks)",
            r"(?:holistic|cyclical|relational)\s+(?:thinking|reasoning|approach|understanding)",
            r"(?:western|eastern|indigenous|traditional)\s+(?:logic|reasoning)\s+(?:and|with|combined)"
        ]
        
        cultural_logic_matches = 0
        for pattern in cultural_logic_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            cultural_logic_matches += matches
            if matches > 0:
                evidence.append(f"Cultural logic integration: {pattern}")
                cultural_markers.append(f"logic:cultural_integration")
        
        cultural_logic_score = min(1.0, cultural_logic_matches / 2.0)
        
        # Check for contradictions or inconsistencies
        contradiction_patterns = [
            r"(?:however|but|nevertheless|on\s+the\s+other\s+hand)\s+.*?\s+(?:contradicts|conflicts|opposes)",
            r"(?:this|that)\s+(?:contradicts|conflicts\s+with|opposes)\s+(?:the|our|my)",
            r"(?:inconsistent|incompatible)\s+(?:with|to)\s+"
        ]
        
        contradiction_penalty = 0.0
        for pattern in contradiction_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 0:
                contradiction_penalty += matches * 0.1
                evidence.append(f"Potential contradiction detected: {matches} instances")
        
        contradiction_penalty = min(0.4, contradiction_penalty)
        
        # Calculate overall logical consistency
        overall_score = max(0.0, (reasoning_score * 0.5 + cultural_logic_score * 0.5) - contradiction_penalty)
        
        # Assess confidence and cultural relevance
        confidence = 0.8 if reasoning_matches > 0 or cultural_logic_matches > 0 else 0.5
        cultural_relevance = 1.0 if cultural_logic_matches > 0 else 0.6
        
        if not evidence:
            evidence.append("Limited logical consistency across domains detected")
        
        return EvaluationDimension(
            name="logical_consistency_across_domains",
            score=overall_score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_creative_appropriateness(self, text: str, test_metadata: Dict[str, Any],
                                         cultural_context: CulturalContext) -> EvaluationDimension:
        """Evaluate creative solutions that respect cultural boundaries."""
        
        evidence = []
        cultural_markers = []
        
        # Check if creativity is expected in this integration
        category = test_metadata.get('category', '')
        creativity_expected = any(term in category for term in ['creativity', 'creative', 'artistic'])
        
        if not creativity_expected:
            # If creativity isn't expected, give moderate score for general appropriateness
            return EvaluationDimension(
                name="creative_appropriateness",
                score=0.7,
                confidence=0.5,
                cultural_relevance=0.5,
                evidence=["Creativity not specifically required for this integration"],
                cultural_markers=[]
            )
        
        # Use creativity evaluator for creative pattern detection
        creative_patterns = self.pattern_library.detect_patterns(text, cultural_context.traditions)
        creative_patterns = [p for p in creative_patterns 
                           if 'creative' in p.pattern_name or 'narrative' in p.pattern_name or 'performance' in p.pattern_name]
        
        creativity_score = 0.0
        if creative_patterns:
            creativity_score = min(1.0, len(creative_patterns) / 3.0)
            evidence.append(f"Creative patterns detected: {len(creative_patterns)}")
            for pattern in creative_patterns[:2]:
                cultural_markers.append(f"creative:{pattern.tradition}:{pattern.pattern_name}")
        
        # Check for appropriate creative solutions to problems
        solution_patterns = [
            r"(?:creative|innovative|artistic|imaginative)\s+(?:solution|approach|method|way)",
            r"(?:through|via|using)\s+(?:art|creativity|story|performance|ritual)",
            r"(?:artistic|creative)\s+(?:expression|practice|tradition)\s+(?:addresses|solves|helps)"
        ]
        
        solution_score = 0.0
        solution_matches = 0
        for pattern in solution_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            solution_matches += matches
            if matches > 0:
                evidence.append(f"Creative solution approach: {pattern}")
        
        solution_score = min(1.0, solution_matches / 2.0)
        
        # Check for cultural appropriateness in creative expression
        appropriateness_penalty = 0.0
        inappropriate_patterns = [
            r"(?:stereotype|cliché|generic)\s+(?:representation|portrayal|depiction)",
            r"(?:inappropriate|disrespectful|insensitive)\s+(?:use|appropriation|borrowing)",
            r"(?:superficial|shallow|tokenistic)\s+(?:cultural|traditional|ethnic)"
        ]
        
        for pattern in inappropriate_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 0:
                appropriateness_penalty += matches * 0.2
                evidence.append(f"Appropriateness concern: {matches} instances")
        
        appropriateness_penalty = min(0.5, appropriateness_penalty)
        
        # Calculate overall creative appropriateness
        overall_score = max(0.0, (creativity_score * 0.4 + solution_score * 0.6) - appropriateness_penalty)
        
        # Assess confidence and cultural relevance
        confidence = 0.8 if creative_patterns or solution_matches > 0 else 0.6
        cultural_relevance = 1.0 if cultural_context.traditions else 0.5
        
        if not evidence:
            evidence.append("Limited creative appropriateness detected")
        
        return EvaluationDimension(
            name="creative_appropriateness",
            score=overall_score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_social_awareness_integration(self, text: str, 
                                             cultural_context: CulturalContext) -> EvaluationDimension:
        """Evaluate understanding of social dynamics and implications."""
        
        evidence = []
        cultural_markers = []
        
        # Check for social awareness indicators
        social_patterns = [
            r"(?:community|social|collective)\s+(?:impact|implications|effects|consequences)",
            r"(?:cultural\s+sensitivity|social\s+awareness|community\s+respect)",
            r"(?:stakeholders|community\s+members|affected\s+parties|social\s+groups)",
            r"(?:power\s+dynamics|social\s+hierarchy|cultural\s+protocol|social\s+norms)",
            r"(?:inclusive|equitable|respectful|culturally\s+appropriate)\s+(?:approach|solution|method)"
        ]
        
        social_score = 0.0
        social_matches = 0
        for pattern in social_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            social_matches += matches
            if matches > 0:
                evidence.append(f"Social awareness: {pattern}")
                cultural_markers.append("social:awareness")
        
        social_score = min(1.0, social_matches / 4.0)
        
        # Check for ethical considerations
        ethical_patterns = [
            r"(?:ethical|moral)\s+(?:consideration|implication|concern|responsibility)",
            r"(?:right|wrong|appropriate|inappropriate)\s+(?:to|for|in)",
            r"(?:consent|permission|authorization)\s+(?:from|of|by)",
            r"(?:harm|benefit|impact)\s+(?:to|on|for)\s+(?:community|culture|people)"
        ]
        
        ethical_score = 0.0
        ethical_matches = 0
        for pattern in ethical_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            ethical_matches += matches
            if matches > 0:
                evidence.append(f"Ethical consideration: {pattern}")
                cultural_markers.append("social:ethics")
        
        ethical_score = min(1.0, ethical_matches / 3.0)
        
        # Check for multicultural awareness
        multicultural_patterns = [
            r"(?:cross-cultural|intercultural|multicultural)\s+(?:understanding|competence|awareness)",
            r"(?:different|diverse|various)\s+(?:cultures|traditions|perspectives|worldviews)",
            r"(?:cultural\s+differences|cultural\s+diversity|cultural\s+variations)",
            r"(?:global|international|worldwide)\s+(?:perspective|understanding|awareness)"
        ]
        
        multicultural_score = 0.0
        multicultural_matches = 0
        for pattern in multicultural_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            multicultural_matches += matches
            if matches > 0:
                evidence.append(f"Multicultural awareness: {pattern}")
                cultural_markers.append("social:multicultural")
        
        multicultural_score = min(1.0, multicultural_matches / 2.0)
        
        # Calculate overall social awareness integration
        overall_score = (social_score * 0.4 + ethical_score * 0.3 + multicultural_score * 0.3)
        
        # Assess confidence and cultural relevance
        confidence = 0.8 if social_matches > 0 or ethical_matches > 0 else 0.5
        cultural_relevance = 1.0 if multicultural_matches > 0 or cultural_context.cultural_groups else 0.6
        
        if not evidence:
            evidence.append("Limited social awareness integration detected")
        
        return EvaluationDimension(
            name="social_awareness_integration",
            score=overall_score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_synthesis_quality(self, text: str, test_metadata: Dict[str, Any],
                                  cultural_context: CulturalContext) -> EvaluationDimension:
        """Evaluate quality of synthesis across domains."""
        
        evidence = []
        cultural_markers = []
        
        # Check for synthesis indicators
        synthesis_matches = 0
        for indicator in self.integration_quality_indicators:
            matches = len(re.findall(rf"\b{indicator}\b", text, re.IGNORECASE))
            synthesis_matches += matches
            if matches > 0:
                evidence.append(f"Synthesis indicator: '{indicator}' ({matches} times)")
        
        synthesis_score = min(1.0, synthesis_matches / 5.0)
        
        # Check for transition quality
        transition_matches = 0
        for marker in self.domain_transition_markers:
            matches = len(re.findall(rf"\b{marker}\b", text, re.IGNORECASE))
            transition_matches += matches
        
        transition_score = min(1.0, transition_matches / 4.0)
        if transition_score > 0.3:
            evidence.append(f"Good domain transitions ({transition_matches} markers)")
        
        # Check for comprehensive understanding
        comprehensive_patterns = [
            r"(?:holistic|comprehensive|complete|thorough)\s+(?:understanding|analysis|approach)",
            r"(?:integrates|combines|synthesizes|unifies)\s+(?:multiple|various|different|diverse)",
            r"(?:brings\s+together|connects|links|relates)\s+(?:different|various|multiple)",
            r"(?:multifaceted|complex|nuanced|sophisticated)\s+(?:understanding|approach|analysis)"
        ]
        
        comprehensive_score = 0.0
        comprehensive_matches = 0
        for pattern in comprehensive_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            comprehensive_matches += matches
            if matches > 0:
                evidence.append(f"Comprehensive integration: {pattern}")
        
        comprehensive_score = min(1.0, comprehensive_matches / 3.0)
        
        # Calculate overall synthesis quality
        overall_score = (synthesis_score * 0.4 + transition_score * 0.3 + comprehensive_score * 0.3)
        
        # Assess confidence and cultural relevance
        confidence = 0.8 if synthesis_matches > 0 or comprehensive_matches > 0 else 0.5
        cultural_relevance = 0.7  # Synthesis quality is moderately culture-dependent
        
        if not evidence:
            evidence.append("Limited synthesis quality detected")
        
        return EvaluationDimension(
            name="synthesis_quality",
            score=overall_score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _analyze_cross_domain_coherence(self, text: str, expected_domains: List[str]) -> CrossDomainCoherence:
        """Analyze coherence across multiple domains."""
        
        # Domain detection patterns
        domain_indicators = {
            'creativity': ['creative', 'story', 'narrative', 'artistic', 'imaginative', 'original'],
            'knowledge': ['knowledge', 'traditional', 'wisdom', 'information', 'cultural', 'indigenous'],
            'language': ['language', 'linguistic', 'multilingual', 'code-switching', 'dialect', 'translation'],
            'reasoning': ['reasoning', 'logic', 'analysis', 'evidence', 'conclude', 'infer'],
            'social': ['social', 'community', 'cultural', 'relationship', 'interaction', 'society']
        }
        
        detected_domains = []
        domain_evidence = {}
        
        text_lower = text.lower()
        for domain, indicators in domain_indicators.items():
            domain_score = sum(1 for indicator in indicators if indicator in text_lower)
            if domain_score >= 2:  # At least 2 indicators
                detected_domains.append(domain)
                domain_evidence[domain] = domain_score
        
        # Calculate integration quality
        integration_quality = len(detected_domains) / max(1, len(expected_domains))
        integration_quality = min(1.0, integration_quality)
        
        # Calculate coherence score based on balance
        if detected_domains:
            domain_scores = list(domain_evidence.values())
            coherence_score = 1.0 - (statistics.stdev(domain_scores) / statistics.mean(domain_scores)) if len(domain_scores) > 1 else 1.0
            coherence_score = max(0.0, min(1.0, coherence_score))
        else:
            coherence_score = 0.0
        
        # Calculate transition quality
        transition_count = sum(1 for marker in self.domain_transition_markers if marker in text_lower)
        transition_quality = min(1.0, transition_count / 3.0)
        
        evidence = [
            f"Detected domains: {detected_domains}",
            f"Expected domains: {expected_domains}",
            f"Integration coverage: {len(detected_domains)}/{len(expected_domains)}"
        ]
        
        return CrossDomainCoherence(
            domains_integrated=detected_domains,
            integration_quality=integration_quality,
            coherence_score=coherence_score,
            transition_quality=transition_quality,
            evidence=evidence
        )