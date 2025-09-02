"""
Intercultural Competence Assessor

Specialized evaluator for assessing intercultural competence across multiple dimensions
including cultural awareness, cultural sensitivity, cross-cultural communication,
adaptation skills, global mindset, and intercultural relationship building.

"""

from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import re
from collections import defaultdict, Counter

from ..core.domain_evaluator_base import (
    MultiDimensionalEvaluator, 
    EvaluationDimension, 
    DomainEvaluationResult,
    CulturalContext
)


class InterculturalCompetenceType(Enum):
    """Types of intercultural competence assessment."""
    CULTURAL_AWARENESS = "cultural_awareness"
    CULTURAL_SENSITIVITY = "cultural_sensitivity"
    CROSS_CULTURAL_COMMUNICATION = "cross_cultural_communication"
    ADAPTATION_SKILLS = "adaptation_skills"
    GLOBAL_MINDSET = "global_mindset"
    INTERCULTURAL_EMPATHY = "intercultural_empathy"
    CULTURAL_BRIDGE_BUILDING = "cultural_bridge_building"


class InterculturalCompetenceAssessor(MultiDimensionalEvaluator):
    """Evaluates intercultural competence across multiple dimensions."""
    
    VERSION = "1.0.0"
    
    def _initialize_evaluator(self):
        """Initialize intercultural competence evaluation components."""
        
        # Cultural awareness indicators
        self.cultural_awareness_patterns = {
            "cultural_knowledge": [
                "cultural differences", "cultural values", "cultural norms", 
                "traditions", "customs", "beliefs", "practices", "rituals",
                "cultural context", "cultural background", "heritage", "ancestry"
            ],
            "cultural_dimensions": [
                "individualism", "collectivism", "power distance", "uncertainty avoidance",
                "masculinity", "femininity", "long-term orientation", "short-term orientation",
                "high-context", "low-context", "direct communication", "indirect communication"
            ],
            "cultural_frameworks": [
                "hofstede", "trompenaars", "kluckhohn", "cultural iceberg",
                "cultural intelligence", "cultural competence", "cross-cultural"
            ]
        }
        
        # Cultural sensitivity markers
        self.cultural_sensitivity_patterns = {
            "respectful_language": [
                "with respect", "respectfully", "honor", "appreciate", "acknowledge",
                "recognize", "value", "cherish", "sacred", "meaningful",
                "important to", "significant for", "deeply valued"
            ],
            "inclusive_language": [
                "diverse perspectives", "different viewpoints", "various approaches",
                "multiple ways", "different experiences", "varied backgrounds",
                "inclusive", "welcoming", "embracing diversity"
            ],
            "avoiding_stereotypes": [
                "not all", "some may", "individuals vary", "personal experience",
                "avoid generalizing", "stereotype", "assumption", "individual differences",
                "unique circumstances", "context-dependent"
            ],
            "cultural_humility": [
                "I don't fully understand", "I'm still learning", "help me understand",
                "correct me if", "I may be wrong", "from my limited perspective",
                "open to learning", "cultural humility", "acknowledge ignorance"
            ]
        }
        
        # Cross-cultural communication skills
        self.communication_patterns = {
            "clarification_seeking": [
                "could you explain", "what do you mean by", "help me understand",
                "can you clarify", "I want to make sure", "let me check my understanding",
                "correct me if I'm wrong", "am I understanding correctly"
            ],
            "perspective_taking": [
                "from your perspective", "in your experience", "how do you see",
                "your point of view", "your cultural lens", "from where you stand",
                "in your context", "given your background"
            ],
            "cultural_translation": [
                "in my culture", "where I come from", "the equivalent would be",
                "similar concept", "cultural translation", "cultural bridge",
                "comparable experience", "analogous situation"
            ],
            "code_switching": [
                "adjusting communication", "adapting style", "modifying approach",
                "shifting register", "changing tone", "contextual communication"
            ]
        }
        
        # Adaptation skills indicators
        self.adaptation_patterns = {
            "behavioral_flexibility": [
                "adapt", "adjust", "modify", "flexible", "adaptable",
                "changing approach", "different strategy", "alternative method",
                "when in Rome", "local customs", "contextual behavior"
            ],
            "learning_orientation": [
                "learning from", "observing", "noticing patterns", "picking up",
                "studying behavior", "watching how", "learning curve",
                "trial and error", "feedback", "continuous learning"
            ],
            "discomfort_tolerance": [
                "uncomfortable situation", "challenging experience", "difficult moment",
                "outside comfort zone", "unfamiliar territory", "uncertainty",
                "ambiguity", "unknown", "unpredictable"
            ],
            "resilience": [
                "bounce back", "recover from", "overcome challenges", "persist",
                "persevere", "keep trying", "learn from mistakes", "growth mindset"
            ]
        }
        
        # Global mindset indicators
        self.global_mindset_patterns = {
            "global_awareness": [
                "global perspective", "worldwide", "international", "across cultures",
                "around the world", "globally", "universal", "transnational",
                "cross-border", "multinational", "worldwide patterns"
            ],
            "interconnectedness": [
                "connected", "interdependent", "interrelated", "global community",
                "shared humanity", "common challenges", "mutual impact",
                "ripple effects", "global citizenship"
            ],
            "complexity_thinking": [
                "complex situation", "multiple factors", "interconnected issues",
                "systems thinking", "holistic view", "nuanced understanding",
                "multifaceted", "layered complexity", "dynamic relationship"
            ],
            "future_orientation": [
                "long-term thinking", "future generations", "sustainable",
                "forward-thinking", "anticipating", "preparing for",
                "future implications", "next generation"
            ]
        }
        
        # Intercultural empathy patterns
        self.empathy_patterns = {
            "emotional_understanding": [
                "I can imagine", "must feel", "understand the emotion", 
                "empathize with", "feel for", "emotional impact",
                "puts into perspective", "touches my heart"
            ],
            "perspective_validation": [
                "valid concern", "understandable reaction", "makes sense",
                "reasonable response", "legitimate feeling", "justified emotion",
                "appropriate reaction", "natural response"
            ],
            "shared_humanity": [
                "we all", "human experience", "common ground", "shared struggles",
                "universal feelings", "human nature", "we share", "similar experiences"
            ],
            "cultural_context_empathy": [
                "in your situation", "given your background", "considering your culture",
                "within your context", "from your position", "your lived experience"
            ]
        }
        
        # Cultural bridge building skills
        self.bridge_building_patterns = {
            "finding_commonalities": [
                "common ground", "shared values", "similar experiences", "mutual interests",
                "both cultures", "universal themes", "connecting points",
                "bridging differences", "what we share"
            ],
            "translation_skills": [
                "let me explain", "in other words", "think of it like", "similar to",
                "cultural equivalent", "comparable concept", "like when you",
                "translation", "interpretation", "making connections"
            ],
            "mediation_skills": [
                "different perspectives", "both sides", "each viewpoint", "various angles",
                "middle ground", "compromise", "finding balance", "reconciling differences"
            ],
            "integration_skills": [
                "combining approaches", "blending traditions", "integrating practices",
                "hybrid solution", "best of both", "creative combination",
                "synergy", "fusion", "merged approach"
            ]
        }
        
        # Cultural intelligence dimensions
        self.cultural_intelligence = {
            "cq_drive": [
                "curious about", "interested in learning", "motivated to understand",
                "eager to explore", "fascinated by", "drawn to", "passionate about learning"
            ],
            "cq_knowledge": [
                "cultural systems", "cultural values", "cultural practices",
                "historical context", "cultural background", "traditions",
                "social structures", "communication patterns"
            ],
            "cq_strategy": [
                "planning approach", "strategic thinking", "considering context",
                "preparing for", "anticipating differences", "checking assumptions",
                "monitoring interactions", "adjusting strategy"
            ],
            "cq_action": [
                "modifying behavior", "adapting actions", "changing approach",
                "flexible response", "contextual behavior", "situation-appropriate"
            ]
        }
        
        # Bias awareness and mitigation
        self.bias_awareness = {
            "recognition": [
                "my bias", "my assumption", "stereotyping", "prejudice",
                "preconceived notion", "unconscious bias", "implicit bias",
                "cultural lens", "my perspective is limited"
            ],
            "checking": [
                "question my assumptions", "challenge my thinking", "step back",
                "examine my reaction", "reflect on", "consider alternative",
                "check my bias", "aware of prejudice"
            ],
            "mitigation": [
                "try to be objective", "seek different perspectives", "listen more",
                "suspend judgment", "remain open", "withhold conclusion",
                "gathering more information", "avoiding stereotypes"
            ]
        }
    
    def get_domain_name(self) -> str:
        """Return the domain name this evaluator handles."""
        return "intercultural_competence"
    
    def get_supported_evaluation_types(self) -> List[str]:
        """Return list of evaluation types this evaluator supports."""
        return [evaluation_type.value for evaluation_type in InterculturalCompetenceType]
    
    def get_evaluation_dimensions(self) -> List[str]:
        """Return list of dimensions this evaluator assesses."""
        return [
            "cultural_awareness",
            "cultural_sensitivity",
            "cross_cultural_communication",
            "adaptation_skills",
            "global_mindset",
            "intercultural_empathy",
            "cultural_bridge_building",
            "bias_awareness_mitigation"
        ]
    
    def evaluate_dimension(self, 
                          dimension: str,
                          response_text: str, 
                          test_metadata: Dict[str, Any], 
                          cultural_context: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate a specific dimension."""
        cultural_ctx = self._create_cultural_context(cultural_context)
        
        if dimension == "cultural_awareness":
            return self._evaluate_cultural_awareness(response_text, cultural_ctx, test_metadata)
        elif dimension == "cultural_sensitivity":
            return self._evaluate_cultural_sensitivity(response_text, cultural_ctx, test_metadata)
        elif dimension == "cross_cultural_communication":
            return self._evaluate_cross_cultural_communication(response_text, cultural_ctx, test_metadata)
        elif dimension == "adaptation_skills":
            return self._evaluate_adaptation_skills(response_text, cultural_ctx, test_metadata)
        elif dimension == "global_mindset":
            return self._evaluate_global_mindset(response_text, cultural_ctx, test_metadata)
        elif dimension == "intercultural_empathy":
            return self._evaluate_intercultural_empathy(response_text, cultural_ctx, test_metadata)
        elif dimension == "cultural_bridge_building":
            return self._evaluate_cultural_bridge_building(response_text, cultural_ctx, test_metadata)
        elif dimension == "bias_awareness_mitigation":
            return self._evaluate_bias_awareness_mitigation(response_text, cultural_ctx, test_metadata)
        else:
            return EvaluationDimension(
                name=dimension,
                score=0.0,
                confidence=0.0,
                cultural_relevance=0.0,
                evidence=[f"Unknown dimension: {dimension}"],
                cultural_markers=[]
            )
    
    def _evaluate_cultural_awareness(self, response: str, cultural_context: CulturalContext,
                                   test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate cultural awareness and knowledge."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count cultural knowledge indicators
        knowledge_count = sum(1 for indicator in self.cultural_awareness_patterns["cultural_knowledge"]
                             if indicator in response_lower)
        dimensions_count = sum(1 for dimension in self.cultural_awareness_patterns["cultural_dimensions"]
                              if dimension in response_lower)
        frameworks_count = sum(1 for framework in self.cultural_awareness_patterns["cultural_frameworks"]
                              if framework in response_lower)
        
        if knowledge_count > 0:
            evidence.append(f"Cultural knowledge indicators: {knowledge_count}")
            cultural_markers.append("cultural_knowledge")
        
        if dimensions_count > 0:
            evidence.append(f"Cultural dimensions awareness: {dimensions_count}")
            cultural_markers.append("dimensional_thinking")
        
        if frameworks_count > 0:
            evidence.append(f"Cultural frameworks referenced: {frameworks_count}")
            cultural_markers.append("theoretical_awareness")
        
        # Bonus for multicultural context awareness
        multicultural_bonus = 0.0
        if len(cultural_context.cultural_groups) > 1:
            multicultural_bonus = 0.1
            cultural_markers.append("multicultural_awareness")
        
        total_score = (knowledge_count * 0.4 + dimensions_count * 0.35 + frameworks_count * 0.25)
        score = min(1.0, (total_score * 0.1) + multicultural_bonus)
        confidence = min(1.0, (knowledge_count + dimensions_count + frameworks_count) * 0.08)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="cultural_awareness",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_cultural_sensitivity(self, response: str, cultural_context: CulturalContext,
                                     test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate cultural sensitivity and respectful communication."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count sensitivity indicators
        respectful_count = sum(1 for phrase in self.cultural_sensitivity_patterns["respectful_language"]
                              if phrase in response_lower)
        inclusive_count = sum(1 for phrase in self.cultural_sensitivity_patterns["inclusive_language"]
                             if phrase in response_lower)
        avoiding_stereotypes_count = sum(1 for phrase in self.cultural_sensitivity_patterns["avoiding_stereotypes"]
                                        if phrase in response_lower)
        humility_count = sum(1 for phrase in self.cultural_sensitivity_patterns["cultural_humility"]
                            if phrase in response_lower)
        
        if respectful_count > 0:
            evidence.append(f"Respectful language: {respectful_count} instances")
            cultural_markers.append("respectful_communication")
        
        if inclusive_count > 0:
            evidence.append(f"Inclusive language: {inclusive_count} instances")
            cultural_markers.append("inclusive_mindset")
        
        if avoiding_stereotypes_count > 0:
            evidence.append(f"Stereotype avoidance: {avoiding_stereotypes_count} instances")
            cultural_markers.append("stereotype_awareness")
        
        if humility_count > 0:
            evidence.append(f"Cultural humility: {humility_count} instances")
            cultural_markers.append("cultural_humility")
        
        total_score = (respectful_count * 0.3 + inclusive_count * 0.25 + 
                      avoiding_stereotypes_count * 0.25 + humility_count * 0.2)
        
        score = min(1.0, total_score * 0.12)
        confidence = min(1.0, (respectful_count + inclusive_count + 
                              avoiding_stereotypes_count + humility_count) * 0.08)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="cultural_sensitivity",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_cross_cultural_communication(self, response: str, cultural_context: CulturalContext,
                                             test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate cross-cultural communication skills."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count communication skill indicators
        clarification_count = sum(1 for phrase in self.communication_patterns["clarification_seeking"]
                                 if phrase in response_lower)
        perspective_count = sum(1 for phrase in self.communication_patterns["perspective_taking"]
                               if phrase in response_lower)
        translation_count = sum(1 for phrase in self.communication_patterns["cultural_translation"]
                               if phrase in response_lower)
        code_switching_count = sum(1 for phrase in self.communication_patterns["code_switching"]
                                  if phrase in response_lower)
        
        if clarification_count > 0:
            evidence.append(f"Clarification seeking: {clarification_count} instances")
            cultural_markers.append("clarification_competence")
        
        if perspective_count > 0:
            evidence.append(f"Perspective taking: {perspective_count} instances")
            cultural_markers.append("perspective_taking_skill")
        
        if translation_count > 0:
            evidence.append(f"Cultural translation: {translation_count} instances")
            cultural_markers.append("translation_competence")
        
        if code_switching_count > 0:
            evidence.append(f"Communication adaptation: {code_switching_count} instances")
            cultural_markers.append("adaptive_communication")
        
        total_score = (clarification_count * 0.3 + perspective_count * 0.3 + 
                      translation_count * 0.25 + code_switching_count * 0.15)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (clarification_count + perspective_count + 
                              translation_count + code_switching_count) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="cross_cultural_communication",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_adaptation_skills(self, response: str, cultural_context: CulturalContext,
                                  test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate cultural adaptation and flexibility skills."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count adaptation skill indicators
        flexibility_count = sum(1 for phrase in self.adaptation_patterns["behavioral_flexibility"]
                               if phrase in response_lower)
        learning_count = sum(1 for phrase in self.adaptation_patterns["learning_orientation"]
                            if phrase in response_lower)
        tolerance_count = sum(1 for phrase in self.adaptation_patterns["discomfort_tolerance"]
                             if phrase in response_lower)
        resilience_count = sum(1 for phrase in self.adaptation_patterns["resilience"]
                              if phrase in response_lower)
        
        if flexibility_count > 0:
            evidence.append(f"Behavioral flexibility: {flexibility_count} instances")
            cultural_markers.append("adaptive_behavior")
        
        if learning_count > 0:
            evidence.append(f"Learning orientation: {learning_count} instances")
            cultural_markers.append("continuous_learning")
        
        if tolerance_count > 0:
            evidence.append(f"Discomfort tolerance: {tolerance_count} instances")
            cultural_markers.append("ambiguity_tolerance")
        
        if resilience_count > 0:
            evidence.append(f"Resilience indicators: {resilience_count} instances")
            cultural_markers.append("cultural_resilience")
        
        total_score = (flexibility_count * 0.3 + learning_count * 0.25 + 
                      tolerance_count * 0.25 + resilience_count * 0.2)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (flexibility_count + learning_count + 
                              tolerance_count + resilience_count) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="adaptation_skills",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_global_mindset(self, response: str, cultural_context: CulturalContext,
                               test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate global mindset and systems thinking."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count global mindset indicators
        awareness_count = sum(1 for phrase in self.global_mindset_patterns["global_awareness"]
                             if phrase in response_lower)
        interconnect_count = sum(1 for phrase in self.global_mindset_patterns["interconnectedness"]
                               if phrase in response_lower)
        complexity_count = sum(1 for phrase in self.global_mindset_patterns["complexity_thinking"]
                              if phrase in response_lower)
        future_count = sum(1 for phrase in self.global_mindset_patterns["future_orientation"]
                          if phrase in response_lower)
        
        if awareness_count > 0:
            evidence.append(f"Global awareness: {awareness_count} instances")
            cultural_markers.append("global_perspective")
        
        if interconnect_count > 0:
            evidence.append(f"Interconnectedness thinking: {interconnect_count} instances")
            cultural_markers.append("systems_thinking")
        
        if complexity_count > 0:
            evidence.append(f"Complexity thinking: {complexity_count} instances")
            cultural_markers.append("nuanced_thinking")
        
        if future_count > 0:
            evidence.append(f"Future orientation: {future_count} instances")
            cultural_markers.append("forward_thinking")
        
        total_score = (awareness_count * 0.3 + interconnect_count * 0.25 + 
                      complexity_count * 0.25 + future_count * 0.2)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (awareness_count + interconnect_count + 
                              complexity_count + future_count) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="global_mindset",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_intercultural_empathy(self, response: str, cultural_context: CulturalContext,
                                      test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate intercultural empathy and emotional intelligence."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count empathy indicators
        emotional_count = sum(1 for phrase in self.empathy_patterns["emotional_understanding"]
                             if phrase in response_lower)
        validation_count = sum(1 for phrase in self.empathy_patterns["perspective_validation"]
                              if phrase in response_lower)
        humanity_count = sum(1 for phrase in self.empathy_patterns["shared_humanity"]
                            if phrase in response_lower)
        context_count = sum(1 for phrase in self.empathy_patterns["cultural_context_empathy"]
                           if phrase in response_lower)
        
        if emotional_count > 0:
            evidence.append(f"Emotional understanding: {emotional_count} instances")
            cultural_markers.append("emotional_intelligence")
        
        if validation_count > 0:
            evidence.append(f"Perspective validation: {validation_count} instances")
            cultural_markers.append("validating_empathy")
        
        if humanity_count > 0:
            evidence.append(f"Shared humanity recognition: {humanity_count} instances")
            cultural_markers.append("universal_empathy")
        
        if context_count > 0:
            evidence.append(f"Cultural context empathy: {context_count} instances")
            cultural_markers.append("contextual_empathy")
        
        total_score = (emotional_count * 0.3 + validation_count * 0.25 + 
                      humanity_count * 0.25 + context_count * 0.2)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (emotional_count + validation_count + 
                              humanity_count + context_count) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="intercultural_empathy",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_cultural_bridge_building(self, response: str, cultural_context: CulturalContext,
                                         test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate cultural bridge building and integration skills."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count bridge building indicators
        commonality_count = sum(1 for phrase in self.bridge_building_patterns["finding_commonalities"]
                               if phrase in response_lower)
        translation_count = sum(1 for phrase in self.bridge_building_patterns["translation_skills"]
                               if phrase in response_lower)
        mediation_count = sum(1 for phrase in self.bridge_building_patterns["mediation_skills"]
                             if phrase in response_lower)
        integration_count = sum(1 for phrase in self.bridge_building_patterns["integration_skills"]
                               if phrase in response_lower)
        
        if commonality_count > 0:
            evidence.append(f"Finding commonalities: {commonality_count} instances")
            cultural_markers.append("commonality_identification")
        
        if translation_count > 0:
            evidence.append(f"Cultural translation: {translation_count} instances")
            cultural_markers.append("cultural_translation")
        
        if mediation_count > 0:
            evidence.append(f"Cultural mediation: {mediation_count} instances")
            cultural_markers.append("cultural_mediation")
        
        if integration_count > 0:
            evidence.append(f"Cultural integration: {integration_count} instances")
            cultural_markers.append("cultural_synthesis")
        
        # Bonus for multicultural contexts
        multicultural_bonus = 0.0
        if len(cultural_context.cultural_groups) > 1:
            multicultural_bonus = 0.15
            cultural_markers.append("multicultural_bridging")
        
        total_score = (commonality_count * 0.3 + translation_count * 0.25 + 
                      mediation_count * 0.25 + integration_count * 0.2)
        
        score = min(1.0, (total_score * 0.12) + multicultural_bonus)
        confidence = min(1.0, (commonality_count + translation_count + 
                              mediation_count + integration_count) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="cultural_bridge_building",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_bias_awareness_mitigation(self, response: str, cultural_context: CulturalContext,
                                          test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate bias awareness and mitigation strategies."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count bias awareness indicators
        recognition_count = sum(1 for phrase in self.bias_awareness["recognition"]
                               if phrase in response_lower)
        checking_count = sum(1 for phrase in self.bias_awareness["checking"]
                            if phrase in response_lower)
        mitigation_count = sum(1 for phrase in self.bias_awareness["mitigation"]
                              if phrase in response_lower)
        
        if recognition_count > 0:
            evidence.append(f"Bias recognition: {recognition_count} instances")
            cultural_markers.append("bias_awareness")
        
        if checking_count > 0:
            evidence.append(f"Assumption checking: {checking_count} instances")
            cultural_markers.append("self_reflection")
        
        if mitigation_count > 0:
            evidence.append(f"Bias mitigation: {mitigation_count} instances")
            cultural_markers.append("bias_correction")
        
        total_score = (recognition_count * 0.4 + checking_count * 0.35 + mitigation_count * 0.25)
        
        score = min(1.0, total_score * 0.2)
        confidence = min(1.0, (recognition_count + checking_count + mitigation_count) * 0.15)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="bias_awareness_mitigation",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _create_cultural_context(self, cultural_context: Dict[str, Any]) -> CulturalContext:
        """Create CulturalContext object from dictionary input."""
        if isinstance(cultural_context, CulturalContext):
            return cultural_context
        
        return CulturalContext(
            traditions=cultural_context.get('traditions', []),
            knowledge_systems=cultural_context.get('knowledge_systems', []),
            performance_aspects=cultural_context.get('performance_aspects', []),
            cultural_groups=cultural_context.get('cultural_groups', []),
            linguistic_varieties=cultural_context.get('linguistic_varieties', [])
        )
    
    def _calculate_cultural_relevance(self, cultural_markers: List[str], 
                                    cultural_context: CulturalContext) -> float:
        """Calculate cultural relevance score based on detected markers and context."""
        if not cultural_markers:
            return 0.5  # Default relevance when no markers detected
        
        total_relevance = 0.0
        marker_count = 0
        
        # Check alignment with cultural context
        for marker in cultural_markers:
            marker_type = marker.split('_')[0] if '_' in marker else marker
            relevance_score = 0.5  # Default relevance
            
            # Higher relevance for markers aligned with multicultural contexts
            if marker_type in ["multicultural", "intercultural", "global"] and len(cultural_context.cultural_groups) > 1:
                relevance_score = 0.95
            elif marker_type in ["cultural", "cross"] and len(cultural_context.cultural_groups) > 0:
                relevance_score = 0.9
            elif marker_type in ["empathy", "sensitivity", "awareness"]:
                relevance_score = 0.85
            elif marker_type in ["adaptation", "flexibility", "learning"]:
                relevance_score = 0.8
            elif marker_type in ["communication", "translation", "bridge"]:
                relevance_score = 0.8
            elif marker_type in ["bias", "stereotype", "assumption"]:
                relevance_score = 0.75
            elif marker_type in ["respectful", "inclusive", "humility"]:
                relevance_score = 0.85
            
            total_relevance += relevance_score
            marker_count += 1
        
        return min(1.0, total_relevance / marker_count) if marker_count > 0 else 0.5