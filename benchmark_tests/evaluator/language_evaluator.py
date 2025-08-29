"""
Language Domain Evaluator

Comprehensive evaluator for linguistic competence across multiple dimensions including
register appropriateness, code-switching quality, pragmatic competence, multilingual patterns,
dialectal competence, sociolinguistic awareness, historical linguistics, narrative structure,
and semantic sophistication.

Author: Claude Code
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import re
from collections import defaultdict, Counter

from evaluator.domain_evaluator_base import (
    MultiDimensionalEvaluator, 
    EvaluationDimension, 
    DomainEvaluationResult,
    CulturalContext
)


class LanguageEvaluationType(Enum):
    """Types of language evaluation supported."""
    REGISTER_VARIATION = "register_variation"
    CODE_SWITCHING = "code_switching"
    PRAGMATIC_COMPETENCE = "pragmatic_competence"
    MULTILINGUAL_PATTERNS = "multilingual_patterns"
    DIALECTAL_COMPETENCE = "dialectal_competence"
    SOCIOLINGUISTIC_AWARENESS = "sociolinguistic_awareness"
    HISTORICAL_LINGUISTICS = "historical_linguistics"
    NARRATIVE_STRUCTURE = "narrative_structure"
    SEMANTIC_SOPHISTICATION = "semantic_sophistication"


class LanguageEvaluator(MultiDimensionalEvaluator):
    """Evaluates language competence across multiple linguistic dimensions."""
    
    VERSION = "1.0.0"
    
    def _initialize_evaluator(self):
        """Initialize language-specific evaluation components."""
        
        # Register patterns for different formality levels
        self.register_patterns = {
            "formal": {
                "markers": [
                    "therefore", "furthermore", "consequently", "notwithstanding",
                    "in consideration of", "pursuant to", "with respect to",
                    "accordingly", "inasmuch as", "whereupon", "heretofore"
                ],
                "pronouns": ["one", "oneself"],
                "contractions_penalty": ["can't", "won't", "don't", "isn't", "aren't"]
            },
            "academic": {
                "markers": [
                    "research indicates", "studies show", "evidence suggests",
                    "theoretical framework", "methodology", "empirical data",
                    "significant correlation", "hypothesis", "paradigm", "discourse"
                ],
                "hedging": ["may suggest", "appears to", "potentially", "presumably"]
            },
            "informal": {
                "markers": [
                    "like", "you know", "sort of", "kind of", "basically",
                    "anyway", "whatever", "totally", "awesome", "cool"
                ],
                "contractions": ["can't", "won't", "don't", "isn't", "gonna", "wanna"]
            },
            "conversational": {
                "markers": [
                    "well", "so", "yeah", "okay", "right", "I mean",
                    "you see", "actually", "honestly", "frankly"
                ],
                "discourse_markers": ["by the way", "speaking of", "come to think of it"]
            }
        }
        
        # Code-switching patterns across languages
        self.code_switching_patterns = {
            "spanish_english": {
                "common_switches": [
                    "sí", "no", "gracias", "por favor", "familia", "casa",
                    "trabajo", "amigo", "hermano", "madre", "padre"
                ],
                "functional_switches": [
                    "¿verdad?", "¿no?", "mira", "oye", "bueno", "entonces"
                ]
            },
            "french_english": {
                "common_switches": [
                    "oui", "non", "merci", "bonjour", "au revoir", "famille",
                    "maison", "travail", "ami", "c'est", "très", "bien"
                ],
                "functional_switches": [
                    "n'est-ce pas?", "voilà", "alors", "bon", "écoute"
                ]
            },
            "mandarin_english": {
                "romanized_switches": [
                    "shi", "bu shi", "xie xie", "ni hao", "zai jian",
                    "jia", "gong zuo", "peng you", "hen hao"
                ]
            },
            "arabic_english": {
                "romanized_switches": [
                    "shukran", "marhaba", "ma salama", "ahlan", "yalla",
                    "habibi", "khalas", "inshallah", "wallah"
                ]
            }
        }
        
        # Pragmatic competence markers
        self.pragmatic_markers = {
            "politeness_strategies": {
                "positive_politeness": [
                    "we", "us", "together", "shared", "common", "understand",
                    "appreciate your", "I can see", "that's interesting"
                ],
                "negative_politeness": [
                    "if you don't mind", "sorry to bother", "could you possibly",
                    "would it be possible", "I hope I'm not", "excuse me"
                ],
                "indirect_requests": [
                    "I wonder if", "might you", "would you happen to",
                    "could you help me", "any chance", "if it's not too much trouble"
                ]
            },
            "speech_acts": {
                "commissives": ["I promise", "I guarantee", "I commit to", "I will ensure"],
                "directives": ["please", "could you", "would you mind", "I need you to"],
                "expressives": ["thank you", "I'm sorry", "congratulations", "I appreciate"],
                "representatives": ["I believe", "in my opinion", "I think", "it seems to me"]
            },
            "implicature": [
                "between the lines", "what I'm really saying", "if you catch my drift",
                "you know what I mean", "hint hint", "read between"
            ]
        }
        
        # Multilingual competence indicators
        self.multilingual_indicators = {
            "metalinguistic_awareness": [
                "in my language", "we say", "the equivalent is", "translates to",
                "means roughly", "doesn't translate well", "cultural context",
                "linguistic nuance", "idiomatic expression"
            ],
            "cross_linguistic_transfer": [
                "similar to", "unlike in", "whereas in", "compared to",
                "cognate", "false friend", "interference", "transfer"
            ],
            "language_mixing": [
                "bilingual", "multilingual", "heritage language", "dominant language",
                "L1", "L2", "native speaker", "second language acquisition"
            ]
        }
        
        # Dialectal variation patterns
        self.dialectal_patterns = {
            "phonological_variation": {
                "r_dropping": ["r-dropping", "non-rhotic", "car -> cah", "park -> pahk"],
                "vowel_shifts": ["pin/pen merger", "cot/caught merger", "vowel shift"],
                "consonant_changes": ["th-stopping", "h-dropping", "consonant variation"]
            },
            "grammatical_variation": {
                "double_modal": ["might could", "used to could", "might should"],
                "habitual_be": ["he be working", "she be singing"],
                "for_to": ["I want for to go", "need for to"]
            },
            "lexical_variation": {
                "regional_terms": [
                    "soda vs pop vs soft drink", "bag vs sack",
                    "shopping cart vs buggy", "sandwich vs sub vs hoagie"
                ]
            }
        }
        
        # Historical linguistics markers
        self.historical_linguistics_patterns = {
            "etymology_awareness": [
                "etymology", "derives from", "comes from", "borrowed from",
                "cognate", "root word", "ancestor language", "proto-language"
            ],
            "language_change": [
                "sound change", "semantic shift", "grammaticalization",
                "lexicalization", "borrowing", "calque", "substrate", "superstrate"
            ],
            "comparative_linguistics": [
                "language family", "genetic relationship", "comparative method",
                "reconstruction", "regular correspondence", "cognate set"
            ]
        }
        
        # Narrative structure patterns
        self.narrative_patterns = {
            "story_structure": {
                "orientation": ["once upon", "there was", "in the beginning", "long ago"],
                "complication": ["but then", "suddenly", "however", "unfortunately"],
                "resolution": ["finally", "in the end", "eventually", "at last"],
                "evaluation": ["amazing", "incredible", "unbelievable", "remarkable"]
            },
            "discourse_markers": {
                "temporal": ["first", "then", "next", "afterwards", "meanwhile"],
                "causal": ["because", "since", "therefore", "as a result"],
                "additive": ["and", "also", "furthermore", "in addition"],
                "contrastive": ["but", "however", "on the other hand", "nevertheless"]
            },
            "cohesion_devices": [
                "this", "that", "these", "those", "such", "the former", "the latter"
            ]
        }
        
        # Semantic sophistication indicators
        self.semantic_sophistication = {
            "lexical_diversity": {
                "high_frequency": ["good", "bad", "big", "small", "nice", "get", "make", "do"],
                "sophisticated_alternatives": [
                    "exceptional", "detrimental", "substantial", "minute", "exquisite",
                    "obtain", "construct", "execute"
                ]
            },
            "metaphorical_language": [
                "metaphor", "like", "as if", "reminds me of", "similar to",
                "analogy", "parallel", "comparison", "symbolizes"
            ],
            "abstract_concepts": [
                "concept", "notion", "principle", "theory", "framework",
                "paradigm", "ideology", "philosophy", "methodology"
            ],
            "semantic_fields": [
                "emotion", "cognition", "perception", "social relations",
                "temporal concepts", "spatial concepts", "causation"
            ]
        }
    
    def get_domain_name(self) -> str:
        """Return the domain name this evaluator handles."""
        return "language"
    
    def get_supported_evaluation_types(self) -> List[str]:
        """Return list of evaluation types this evaluator supports."""
        return [evaluation_type.value for evaluation_type in LanguageEvaluationType]
    
    def get_evaluation_dimensions(self) -> List[str]:
        """Return list of dimensions this evaluator assesses."""
        return [
            "register_appropriateness",
            "code_switching_quality",
            "pragmatic_competence",
            "multilingual_patterns",
            "dialectal_competence",
            "sociolinguistic_awareness",
            "historical_linguistics",
            "narrative_structure",
            "semantic_sophistication"
        ]
    
    def evaluate_dimension(self, 
                          dimension: str,
                          response_text: str, 
                          test_metadata: Dict[str, Any], 
                          cultural_context: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate a specific dimension."""
        cultural_ctx = self._create_cultural_context(cultural_context)
        
        if dimension == "register_appropriateness":
            return self._evaluate_register_appropriateness(response_text, cultural_ctx, test_metadata)
        elif dimension == "code_switching_quality":
            return self._evaluate_code_switching_quality(response_text, cultural_ctx, test_metadata)
        elif dimension == "pragmatic_competence":
            return self._evaluate_pragmatic_competence(response_text, cultural_ctx, test_metadata)
        elif dimension == "multilingual_patterns":
            return self._evaluate_multilingual_patterns(response_text, cultural_ctx, test_metadata)
        elif dimension == "dialectal_competence":
            return self._evaluate_dialectal_competence(response_text, cultural_ctx, test_metadata)
        elif dimension == "sociolinguistic_awareness":
            return self._evaluate_sociolinguistic_awareness(response_text, cultural_ctx, test_metadata)
        elif dimension == "historical_linguistics":
            return self._evaluate_historical_linguistics(response_text, cultural_ctx, test_metadata)
        elif dimension == "narrative_structure":
            return self._evaluate_narrative_structure(response_text, cultural_ctx, test_metadata)
        elif dimension == "semantic_sophistication":
            return self._evaluate_semantic_sophistication(response_text, cultural_ctx, test_metadata)
        else:
            # Return empty dimension for unknown types
            return EvaluationDimension(
                name=dimension,
                score=0.0,
                confidence=0.0,
                cultural_relevance=0.0,
                evidence=[f"Unknown dimension: {dimension}"],
                cultural_markers=[]
            )
    
    def _evaluate_register_appropriateness(self, response: str, cultural_context: CulturalContext,
                                         test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate appropriate register usage for context."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Determine expected register from metadata
        expected_register = test_metadata.get("register", "formal")
        context_formality = test_metadata.get("formality_level", "medium")
        
        # Count register markers
        formal_markers = sum(1 for marker in self.register_patterns["formal"]["markers"]
                           if marker in response_lower)
        academic_markers = sum(1 for marker in self.register_patterns["academic"]["markers"]
                             if marker in response_lower)
        informal_markers = sum(1 for marker in self.register_patterns["informal"]["markers"]
                             if marker in response_lower)
        conversational_markers = sum(1 for marker in self.register_patterns["conversational"]["markers"]
                                   if marker in response_lower)
        
        # Check for contractions (formality penalty)
        contractions = sum(1 for contraction in self.register_patterns["formal"]["contractions_penalty"]
                          if contraction in response_lower)
        
        # Calculate register alignment score
        if expected_register in ["formal", "academic"]:
            positive_markers = formal_markers + academic_markers
            negative_markers = informal_markers + conversational_markers + contractions
            evidence.append(f"Formal/academic markers: {positive_markers}")
            if contractions > 0:
                evidence.append(f"Inappropriate contractions: {contractions}")
                cultural_markers.append("inappropriate_informality")
        else:
            positive_markers = informal_markers + conversational_markers
            negative_markers = formal_markers * 0.5  # Formal in informal context less penalized
            evidence.append(f"Informal/conversational markers: {positive_markers}")
        
        # Cultural context adjustment
        if "academic" in cultural_context.performance_aspects:
            academic_bonus = academic_markers * 0.1
            positive_markers += academic_bonus
            cultural_markers.append("academic_discourse")
        
        base_score = min(1.0, positive_markers * 0.1 - negative_markers * 0.05)
        score = max(0.0, base_score)
        confidence = min(1.0, (positive_markers + negative_markers) * 0.08)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="register_appropriateness",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_code_switching_quality(self, response: str, cultural_context: CulturalContext,
                                       test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate quality of code-switching between languages."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Check for different types of code-switching
        spanish_switches = sum(1 for switch in self.code_switching_patterns["spanish_english"]["common_switches"]
                             if switch in response_lower)
        french_switches = sum(1 for switch in self.code_switching_patterns["french_english"]["common_switches"]
                            if switch in response_lower)
        mandarin_switches = sum(1 for switch in self.code_switching_patterns["mandarin_english"]["romanized_switches"]
                              if switch in response_lower)
        arabic_switches = sum(1 for switch in self.code_switching_patterns["arabic_english"]["romanized_switches"]
                            if switch in response_lower)
        
        total_switches = spanish_switches + french_switches + mandarin_switches + arabic_switches
        
        # Check for functional code-switching
        functional_switches = 0
        for lang_pattern in self.code_switching_patterns.values():
            if "functional_switches" in lang_pattern:
                functional_switches += sum(1 for switch in lang_pattern["functional_switches"]
                                         if switch in response_lower)
        
        if total_switches > 0:
            evidence.append(f"Code-switching instances: {total_switches}")
            cultural_markers.append("multilingual_competence")
            
            if spanish_switches > 0:
                cultural_markers.append("spanish_english_switching")
            if french_switches > 0:
                cultural_markers.append("french_english_switching")
            if mandarin_switches > 0:
                cultural_markers.append("mandarin_english_switching")
            if arabic_switches > 0:
                cultural_markers.append("arabic_english_switching")
        
        if functional_switches > 0:
            evidence.append(f"Functional code-switching: {functional_switches}")
            cultural_markers.append("functional_switching")
        
        # Cultural context bonus
        cultural_bonus = 0.0
        if "multilingual" in cultural_context.linguistic_varieties:
            cultural_bonus = 0.2
            cultural_markers.append("multilingual_context")
        
        score = min(1.0, (total_switches * 0.15 + functional_switches * 0.2) + cultural_bonus)
        confidence = min(1.0, total_switches * 0.1 + functional_switches * 0.15)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="code_switching_quality",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_pragmatic_competence(self, response: str, cultural_context: CulturalContext,
                                     test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate pragmatic language competence."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count politeness strategies
        positive_politeness = sum(1 for marker in self.pragmatic_markers["politeness_strategies"]["positive_politeness"]
                                if marker in response_lower)
        negative_politeness = sum(1 for marker in self.pragmatic_markers["politeness_strategies"]["negative_politeness"]
                                if marker in response_lower)
        indirect_requests = sum(1 for marker in self.pragmatic_markers["politeness_strategies"]["indirect_requests"]
                              if marker in response_lower)
        
        # Count speech acts
        commissives = sum(1 for marker in self.pragmatic_markers["speech_acts"]["commissives"]
                        if marker in response_lower)
        directives = sum(1 for marker in self.pragmatic_markers["speech_acts"]["directives"]
                       if marker in response_lower)
        expressives = sum(1 for marker in self.pragmatic_markers["speech_acts"]["expressives"]
                        if marker in response_lower)
        representatives = sum(1 for marker in self.pragmatic_markers["speech_acts"]["representatives"]
                            if marker in response_lower)
        
        # Count implicature markers
        implicature_markers = sum(1 for marker in self.pragmatic_markers["implicature"]
                                if marker in response_lower)
        
        # Build evidence
        if positive_politeness > 0:
            evidence.append(f"Positive politeness strategies: {positive_politeness}")
            cultural_markers.append("positive_politeness")
        
        if negative_politeness > 0:
            evidence.append(f"Negative politeness strategies: {negative_politeness}")
            cultural_markers.append("negative_politeness")
        
        if indirect_requests > 0:
            evidence.append(f"Indirect requests: {indirect_requests}")
            cultural_markers.append("indirectness")
        
        speech_act_total = commissives + directives + expressives + representatives
        if speech_act_total > 0:
            evidence.append(f"Speech act diversity: {speech_act_total}")
            cultural_markers.append("speech_act_competence")
        
        if implicature_markers > 0:
            evidence.append(f"Implicature markers: {implicature_markers}")
            cultural_markers.append("implicature_competence")
        
        # Calculate score
        politeness_score = (positive_politeness + negative_politeness + indirect_requests) * 0.1
        speech_act_score = speech_act_total * 0.08
        implicature_score = implicature_markers * 0.2
        
        score = min(1.0, politeness_score + speech_act_score + implicature_score)
        confidence = min(1.0, (positive_politeness + negative_politeness + indirect_requests + 
                              speech_act_total + implicature_markers) * 0.05)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="pragmatic_competence",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_multilingual_patterns(self, response: str, cultural_context: CulturalContext,
                                      test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate multilingual competence patterns."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count metalinguistic awareness
        metalinguistic = sum(1 for marker in self.multilingual_indicators["metalinguistic_awareness"]
                           if marker in response_lower)
        
        # Count cross-linguistic transfer awareness
        cross_linguistic = sum(1 for marker in self.multilingual_indicators["cross_linguistic_transfer"]
                             if marker in response_lower)
        
        # Count language mixing competence
        language_mixing = sum(1 for marker in self.multilingual_indicators["language_mixing"]
                            if marker in response_lower)
        
        if metalinguistic > 0:
            evidence.append(f"Metalinguistic awareness: {metalinguistic}")
            cultural_markers.append("metalinguistic_competence")
        
        if cross_linguistic > 0:
            evidence.append(f"Cross-linguistic awareness: {cross_linguistic}")
            cultural_markers.append("cross_linguistic_competence")
        
        if language_mixing > 0:
            evidence.append(f"Language mixing competence: {language_mixing}")
            cultural_markers.append("language_mixing_competence")
        
        # Cultural context consideration
        multilingual_bonus = 0.0
        if len(cultural_context.linguistic_varieties) > 1:
            multilingual_bonus = 0.2
            cultural_markers.append("multilingual_environment")
        
        score = min(1.0, (metalinguistic * 0.15 + cross_linguistic * 0.15 + 
                         language_mixing * 0.1) + multilingual_bonus)
        confidence = min(1.0, (metalinguistic + cross_linguistic + language_mixing) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="multilingual_patterns",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_dialectal_competence(self, response: str, cultural_context: CulturalContext,
                                     test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate dialectal variation competence."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Check for phonological variation awareness
        phonological_markers = 0
        for variation_type, patterns in self.dialectal_patterns["phonological_variation"].items():
            if isinstance(patterns, list):
                phonological_markers += sum(1 for pattern in patterns if pattern in response_lower)
        
        # Check for grammatical variation
        grammatical_markers = 0
        for variation_type, patterns in self.dialectal_patterns["grammatical_variation"].items():
            grammatical_markers += sum(1 for pattern in patterns if pattern in response_lower)
        
        # Check for lexical variation awareness
        lexical_markers = 0
        for variation_type, patterns in self.dialectal_patterns["lexical_variation"].items():
            if isinstance(patterns, list):
                for pattern in patterns:
                    if any(term in response_lower for term in pattern.split(" vs ")):
                        lexical_markers += 1
        
        if phonological_markers > 0:
            evidence.append(f"Phonological variation awareness: {phonological_markers}")
            cultural_markers.append("phonological_competence")
        
        if grammatical_markers > 0:
            evidence.append(f"Grammatical variation awareness: {grammatical_markers}")
            cultural_markers.append("grammatical_competence")
        
        if lexical_markers > 0:
            evidence.append(f"Lexical variation awareness: {lexical_markers}")
            cultural_markers.append("lexical_competence")
        
        score = min(1.0, (phonological_markers * 0.2 + grammatical_markers * 0.2 + lexical_markers * 0.15))
        confidence = min(1.0, (phonological_markers + grammatical_markers + lexical_markers) * 0.12)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="dialectal_competence",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_sociolinguistic_awareness(self, response: str, cultural_context: CulturalContext,
                                          test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate sociolinguistic awareness and competence."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Combine multiple linguistic competences for sociolinguistic awareness
        social_context_terms = [
            "social class", "socioeconomic", "education level", "regional dialect",
            "prestige variety", "standard language", "vernacular", "language attitude",
            "language ideology", "linguistic discrimination", "code-switching", "style-shifting"
        ]
        
        power_dynamics_terms = [
            "power relations", "authority", "hierarchy", "status", "prestige",
            "dominant language", "minority language", "language maintenance", "language shift"
        ]
        
        identity_terms = [
            "language identity", "cultural identity", "heritage language",
            "mother tongue", "native speaker", "linguistic rights", "language loyalty"
        ]
        
        social_context = sum(1 for term in social_context_terms if term in response_lower)
        power_dynamics = sum(1 for term in power_dynamics_terms if term in response_lower)
        identity_markers = sum(1 for term in identity_terms if term in response_lower)
        
        if social_context > 0:
            evidence.append(f"Social context awareness: {social_context}")
            cultural_markers.append("social_context_competence")
        
        if power_dynamics > 0:
            evidence.append(f"Power dynamics awareness: {power_dynamics}")
            cultural_markers.append("power_dynamics_competence")
        
        if identity_markers > 0:
            evidence.append(f"Language identity awareness: {identity_markers}")
            cultural_markers.append("identity_competence")
        
        # Cultural context bonus
        sociocultural_bonus = 0.0
        if len(cultural_context.cultural_groups) > 1:
            sociocultural_bonus = 0.15
            cultural_markers.append("multicultural_competence")
        
        score = min(1.0, (social_context * 0.15 + power_dynamics * 0.15 + 
                         identity_markers * 0.1) + sociocultural_bonus)
        confidence = min(1.0, (social_context + power_dynamics + identity_markers) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="sociolinguistic_awareness",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_historical_linguistics(self, response: str, cultural_context: CulturalContext,
                                        test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate historical linguistics competence."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count etymology awareness
        etymology_markers = sum(1 for marker in self.historical_linguistics_patterns["etymology_awareness"]
                              if marker in response_lower)
        
        # Count language change awareness
        change_markers = sum(1 for marker in self.historical_linguistics_patterns["language_change"]
                           if marker in response_lower)
        
        # Count comparative linguistics awareness
        comparative_markers = sum(1 for marker in self.historical_linguistics_patterns["comparative_linguistics"]
                                if marker in response_lower)
        
        if etymology_markers > 0:
            evidence.append(f"Etymology awareness: {etymology_markers}")
            cultural_markers.append("etymology_competence")
        
        if change_markers > 0:
            evidence.append(f"Language change awareness: {change_markers}")
            cultural_markers.append("diachronic_competence")
        
        if comparative_markers > 0:
            evidence.append(f"Comparative linguistics awareness: {comparative_markers}")
            cultural_markers.append("comparative_competence")
        
        score = min(1.0, etymology_markers * 0.2 + change_markers * 0.15 + comparative_markers * 0.2)
        confidence = min(1.0, (etymology_markers + change_markers + comparative_markers) * 0.15)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="historical_linguistics",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_narrative_structure(self, response: str, cultural_context: CulturalContext,
                                    test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate narrative structure and discourse organization."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count story structure elements
        orientation_markers = sum(1 for marker in self.narrative_patterns["story_structure"]["orientation"]
                                if marker in response_lower)
        complication_markers = sum(1 for marker in self.narrative_patterns["story_structure"]["complication"]
                                 if marker in response_lower)
        resolution_markers = sum(1 for marker in self.narrative_patterns["story_structure"]["resolution"]
                               if marker in response_lower)
        evaluation_markers = sum(1 for marker in self.narrative_patterns["story_structure"]["evaluation"]
                               if marker in response_lower)
        
        # Count discourse markers
        temporal_markers = sum(1 for marker in self.narrative_patterns["discourse_markers"]["temporal"]
                             if marker in response_lower)
        causal_markers = sum(1 for marker in self.narrative_patterns["discourse_markers"]["causal"]
                           if marker in response_lower)
        additive_markers = sum(1 for marker in self.narrative_patterns["discourse_markers"]["additive"]
                             if marker in response_lower)
        contrastive_markers = sum(1 for marker in self.narrative_patterns["discourse_markers"]["contrastive"]
                                if marker in response_lower)
        
        # Count cohesion devices
        cohesion_markers = sum(1 for marker in self.narrative_patterns["cohesion_devices"]
                             if marker in response_lower)
        
        story_structure_total = orientation_markers + complication_markers + resolution_markers + evaluation_markers
        discourse_marker_total = temporal_markers + causal_markers + additive_markers + contrastive_markers
        
        if story_structure_total > 0:
            evidence.append(f"Narrative structure elements: {story_structure_total}")
            cultural_markers.append("narrative_competence")
        
        if discourse_marker_total > 0:
            evidence.append(f"Discourse markers: {discourse_marker_total}")
            cultural_markers.append("discourse_competence")
        
        if cohesion_markers > 0:
            evidence.append(f"Cohesion devices: {cohesion_markers}")
            cultural_markers.append("cohesion_competence")
        
        score = min(1.0, (story_structure_total * 0.1 + discourse_marker_total * 0.08 + cohesion_markers * 0.12))
        confidence = min(1.0, (story_structure_total + discourse_marker_total + cohesion_markers) * 0.05)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="narrative_structure",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_semantic_sophistication(self, response: str, cultural_context: CulturalContext,
                                        test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate semantic sophistication and lexical diversity."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        words = response_lower.split()
        
        # Check for sophisticated vocabulary
        high_frequency_words = sum(1 for word in words 
                                 if word in self.semantic_sophistication["lexical_diversity"]["high_frequency"])
        sophisticated_alternatives = sum(1 for word in words 
                                       if word in self.semantic_sophistication["lexical_diversity"]["sophisticated_alternatives"])
        
        # Count metaphorical language
        metaphorical_markers = sum(1 for marker in self.semantic_sophistication["metaphorical_language"]
                                 if marker in response_lower)
        
        # Count abstract concepts
        abstract_concepts = sum(1 for concept in self.semantic_sophistication["abstract_concepts"]
                              if concept in response_lower)
        
        # Count semantic field diversity
        semantic_fields = sum(1 for field in self.semantic_sophistication["semantic_fields"]
                            if field in response_lower)
        
        # Calculate lexical sophistication ratio
        total_words = len(words)
        if total_words > 0:
            sophistication_ratio = sophisticated_alternatives / total_words
            high_frequency_ratio = high_frequency_words / total_words
            lexical_sophistication = max(0.0, sophistication_ratio - (high_frequency_ratio * 0.5))
        else:
            lexical_sophistication = 0.0
        
        if sophisticated_alternatives > 0:
            evidence.append(f"Sophisticated vocabulary: {sophisticated_alternatives}")
            cultural_markers.append("lexical_sophistication")
        
        if metaphorical_markers > 0:
            evidence.append(f"Metaphorical language: {metaphorical_markers}")
            cultural_markers.append("metaphorical_competence")
        
        if abstract_concepts > 0:
            evidence.append(f"Abstract concepts: {abstract_concepts}")
            cultural_markers.append("abstract_thinking")
        
        if semantic_fields > 0:
            evidence.append(f"Semantic field diversity: {semantic_fields}")
            cultural_markers.append("semantic_diversity")
        
        score = min(1.0, (lexical_sophistication * 2.0 + metaphorical_markers * 0.1 + 
                         abstract_concepts * 0.1 + semantic_fields * 0.08))
        confidence = min(1.0, (sophisticated_alternatives + metaphorical_markers + 
                              abstract_concepts + semantic_fields) * 0.08)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="semantic_sophistication",
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
            marker_type = marker.split(':')[0] if ':' in marker else marker
            relevance_score = 0.5  # Default relevance
            
            # Higher relevance for markers aligned with linguistic context
            if marker_type in ["multilingual", "code_switching"] and len(cultural_context.linguistic_varieties) > 1:
                relevance_score = 0.9
            elif marker_type in ["academic", "formal"] and "academic" in cultural_context.performance_aspects:
                relevance_score = 0.9
            elif marker_type in ["pragmatic", "social_context"] and len(cultural_context.cultural_groups) > 0:
                relevance_score = 0.8
            elif marker_type in ["narrative", "discourse"] and "oral_tradition" in cultural_context.traditions:
                relevance_score = 0.9
            elif marker_type in ["semantic", "sophistication"] and "literary" in cultural_context.performance_aspects:
                relevance_score = 0.8
            elif marker_type in ["historical", "etymology"] and "linguistic_heritage" in cultural_context.knowledge_systems:
                relevance_score = 0.9
            elif marker_type in ["dialectal", "variation"] and len(cultural_context.linguistic_varieties) > 0:
                relevance_score = 0.8
            
            total_relevance += relevance_score
            marker_count += 1
        
        return min(1.0, total_relevance / marker_count) if marker_count > 0 else 0.5