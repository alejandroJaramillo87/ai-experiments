"""
Multilingual Code-Switching Evaluator

Specialized evaluator for assessing multilingual code-switching competence,
including intrasentential switching, intersentential switching, tag-switching,
and cultural appropriateness of language mixing across diverse linguistic contexts.

Author: Claude Code
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
import re
from collections import defaultdict, Counter

from evaluator.domain_evaluator_base import (
    MultiDimensionalEvaluator, 
    EvaluationDimension, 
    DomainEvaluationResult,
    CulturalContext
)


class CodeSwitchingType(Enum):
    """Types of code-switching patterns."""
    INTRASENTENTIAL = "intrasentential"  # Within sentence
    INTERSENTENTIAL = "intersentential"  # Between sentences
    TAG_SWITCHING = "tag_switching"       # Tags and exclamations
    EMBLEMATIC_SWITCHING = "emblematic_switching"  # Cultural markers
    METAPHORICAL_SWITCHING = "metaphorical_switching"  # Situational switching


class MultilingualCodeSwitchingEvaluator(MultiDimensionalEvaluator):
    """Evaluates multilingual code-switching competence and cultural appropriateness."""
    
    VERSION = "1.0.0"
    
    def _initialize_evaluator(self):
        """Initialize multilingual code-switching evaluation components."""
        
        # Language pair patterns (with common code-switching patterns)
        self.language_pairs = {
            "spanish_english": {
                "common_switches": {
                    "nouns": ["familia", "casa", "trabajo", "escuela", "dinero", "comida", "tiempo"],
                    "adjectives": ["bueno", "malo", "grande", "pequeño", "bonito", "feo"],
                    "discourse_markers": ["pero", "entonces", "porque", "cuando", "donde"],
                    "tags": ["¿verdad?", "¿no?", "¿sabes qué?", "oye", "mira"],
                    "exclamations": ["¡ay!", "¡órale!", "¡qué bueno!", "¡no way!", "¡wow!"],
                    "cultural_terms": ["quinceañera", "piñata", "tamales", "abuela", "compadre"]
                },
                "functional_patterns": {
                    "emphasis": ["muy", "bien", "súper", "re-"],
                    "quotatives": ["like", "como", "que"],
                    "fillers": ["este", "bueno", "so", "like"]
                }
            },
            "french_english": {
                "common_switches": {
                    "nouns": ["famille", "maison", "travail", "école", "argent", "temps"],
                    "adjectives": ["bon", "mauvais", "grand", "petit", "beau", "joli"],
                    "discourse_markers": ["mais", "alors", "parce que", "quand", "où"],
                    "tags": ["n'est-ce pas?", "tu vois?", "écoute", "bon"],
                    "exclamations": ["oh là là!", "mon dieu!", "c'est ça!", "voilà!"],
                    "cultural_terms": ["baguette", "croissant", "café", "bonjour", "merci"]
                },
                "functional_patterns": {
                    "emphasis": ["très", "bien", "super"],
                    "quotatives": ["comme", "genre"],
                    "fillers": ["euh", "bon", "well", "so"]
                }
            },
            "mandarin_english": {
                "common_switches": {
                    "nouns": ["jiā" "家", "gōngzuò" "工作", "xuéxiào" "学校", "qián" "钱"],
                    "adjectives": ["hǎo" "好", "bù hǎo" "不好", "dà" "大", "xiǎo" "小"],
                    "discourse_markers": ["dànshì" "但是", "suǒyǐ" "所以", "yīnwèi" "因为"],
                    "tags": ["duì ma?" "对吗?", "shì ba?" "是吧?", "nǐ zhīdào ma?" "你知道吗?"],
                    "cultural_terms": ["dumplings", "nǎi nai" "奶奶", "chūn jié" "春节"]
                },
                "functional_patterns": {
                    "emphasis": ["hěn" "很", "zhēn" "真"],
                    "fillers": ["jiù shì" "就是", "nà ge" "那个", "well", "so"]
                }
            },
            "arabic_english": {
                "common_switches": {
                    "nouns": ["bayt", "shughl", "madrasa", "flus", "akl", "waqt"],
                    "adjectives": ["kwayyes", "mish kwayyes", "kbir", "zghir", "helw"],
                    "discourse_markers": ["bass", "w", "la2an", "lamma", "ween"],
                    "tags": ["mish heek?", "wallah", "yalla"],
                    "exclamations": ["wallah!", "yalla!", "habibi!", "ya allah!"],
                    "cultural_terms": ["habibi", "yalla", "inshallah", "mashallah", "khalas"]
                },
                "functional_patterns": {
                    "emphasis": ["ktir", "hafez", "very"],
                    "fillers": ["ya3ni", "well", "so", "like"]
                }
            },
            "hindi_english": {
                "common_switches": {
                    "nouns": ["ghar", "kaam", "school", "paisa", "khaana", "time"],
                    "adjectives": ["accha", "bura", "bada", "chhota", "sundar"],
                    "discourse_markers": ["lekin", "phir", "kyunki", "jab", "kahan"],
                    "tags": ["na?", "hai na?", "yaar"],
                    "exclamations": ["arे!", "wah!", "haan!", "achha!"],
                    "cultural_terms": ["namaste", "ji", "beta", "didi", "bhai"]
                },
                "functional_patterns": {
                    "emphasis": ["bahut", "bilkul", "really"],
                    "fillers": ["matlab", "woh", "like", "so"]
                }
            }
        }
        
        # Code-switching functions and motivations
        self.switching_functions = {
            "referential": [
                "lack of facility", "lexical need", "real lexical need",
                "semantic precision", "technical terms", "specific concept"
            ],
            "expressive": [
                "emotional expression", "emphasis", "personal feelings",
                "intimacy", "solidarity", "group identity"
            ],
            "directive": [
                "attention getting", "command", "request",
                "instruction", "persuasion"
            ],
            "metalinguistic": [
                "language play", "quotation", "reported speech",
                "translation", "clarification", "explanation"
            ],
            "poetic": [
                "rhythmic effect", "wordplay", "humor",
                "pun", "artistic expression", "aesthetic effect"
            ]
        }
        
        # Grammatical constraints and patterns
        self.grammatical_constraints = {
            "noun_phrase_switching": [
                "el", "la", "los", "las",  # Spanish determiners with English nouns
                "le", "la", "les",         # French determiners
                "the", "a", "an"          # English determiners with other language nouns
            ],
            "verb_phrase_constraints": [
                "auxiliary_main", "modal_infinitive", "copula_predicate"
            ],
            "embedded_language_islands": [
                "complete_phrases", "idiomatic_expressions", "fixed_expressions"
            ]
        }
        
        # Sociolinguistic factors
        self.sociolinguistic_factors = {
            "participant_factors": [
                "age", "gender", "education", "social_class",
                "linguistic_background", "generation", "identity"
            ],
            "situational_factors": [
                "setting", "topic", "audience", "formality",
                "power_relations", "solidarity", "accommodation"
            ],
            "community_factors": [
                "language_vitality", "language_attitudes", "prestige",
                "community_norms", "language_maintenance", "shift"
            ]
        }
        
        # Pragmatic functions of code-switching
        self.pragmatic_functions = {
            "contextualization_cues": [
                "frame_shift", "topic_shift", "participant_shift",
                "activity_shift", "register_shift"
            ],
            "conversational_strategies": [
                "floor_holding", "turn_taking", "repair",
                "clarification", "elaboration", "reformulation"
            ],
            "identity_construction": [
                "ethnic_identity", "professional_identity", "generational_identity",
                "gender_identity", "class_identity", "regional_identity"
            ]
        }
        
        # Cultural appropriateness markers
        self.appropriateness_markers = {
            "community_acceptance": [
                "natural_switching", "community_norm", "accepted_pattern",
                "cultural_competence", "linguistic_authenticity"
            ],
            "contextual_sensitivity": [
                "audience_awareness", "setting_appropriate", "register_matching",
                "cultural_respect", "linguistic_etiquette"
            ],
            "competence_indicators": [
                "bilingual_competence", "cultural_knowledge", "pragmatic_awareness",
                "sociolinguistic_competence", "strategic_competence"
            ]
        }
    
    def get_domain_name(self) -> str:
        """Return the domain name this evaluator handles."""
        return "multilingual_code_switching"
    
    def get_supported_evaluation_types(self) -> List[str]:
        """Return list of evaluation types this evaluator supports."""
        return [evaluation_type.value for evaluation_type in CodeSwitchingType]
    
    def get_evaluation_dimensions(self) -> List[str]:
        """Return list of dimensions this evaluator assesses."""
        return [
            "intrasentential_switching",
            "intersentential_switching", 
            "tag_switching",
            "functional_appropriateness",
            "grammatical_constraints",
            "sociolinguistic_competence",
            "pragmatic_functions",
            "cultural_authenticity"
        ]
    
    def evaluate_dimension(self, 
                          dimension: str,
                          response_text: str, 
                          test_metadata: Dict[str, Any], 
                          cultural_context: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate a specific dimension."""
        cultural_ctx = self._create_cultural_context(cultural_context)
        
        if dimension == "intrasentential_switching":
            return self._evaluate_intrasentential_switching(response_text, cultural_ctx, test_metadata)
        elif dimension == "intersentential_switching":
            return self._evaluate_intersentential_switching(response_text, cultural_ctx, test_metadata)
        elif dimension == "tag_switching":
            return self._evaluate_tag_switching(response_text, cultural_ctx, test_metadata)
        elif dimension == "functional_appropriateness":
            return self._evaluate_functional_appropriateness(response_text, cultural_ctx, test_metadata)
        elif dimension == "grammatical_constraints":
            return self._evaluate_grammatical_constraints(response_text, cultural_ctx, test_metadata)
        elif dimension == "sociolinguistic_competence":
            return self._evaluate_sociolinguistic_competence(response_text, cultural_ctx, test_metadata)
        elif dimension == "pragmatic_functions":
            return self._evaluate_pragmatic_functions(response_text, cultural_ctx, test_metadata)
        elif dimension == "cultural_authenticity":
            return self._evaluate_cultural_authenticity(response_text, cultural_ctx, test_metadata)
        else:
            return EvaluationDimension(
                name=dimension,
                score=0.0,
                confidence=0.0,
                cultural_relevance=0.0,
                evidence=[f"Unknown dimension: {dimension}"],
                cultural_markers=[]
            )
    
    def _detect_language_pairs(self, text: str) -> List[str]:
        """Detect which language pairs are present in the text."""
        detected_pairs = []
        text_lower = text.lower()
        
        for pair, patterns in self.language_pairs.items():
            switches_found = 0
            for category, words in patterns["common_switches"].items():
                switches_found += sum(1 for word in words if word.lower() in text_lower)
            
            if switches_found >= 2:  # Minimum threshold for pair detection
                detected_pairs.append(pair)
        
        return detected_pairs
    
    def _evaluate_intrasentential_switching(self, response: str, cultural_context: CulturalContext,
                                          test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate intrasentential (within-sentence) code-switching patterns."""
        evidence = []
        cultural_markers = []
        
        # Detect language pairs present
        detected_pairs = self._detect_language_pairs(response)
        
        if not detected_pairs:
            return EvaluationDimension(
                name="intrasentential_switching",
                score=0.0,
                confidence=0.0,
                cultural_relevance=0.5,
                evidence=["No multilingual code-switching detected"],
                cultural_markers=[]
            )
        
        # Count intrasentential switches for each detected pair
        total_switches = 0
        sentences = response.split('.')
        
        for pair in detected_pairs:
            patterns = self.language_pairs[pair]["common_switches"]
            cultural_markers.append(f"{pair}_switching")
            
            for sentence in sentences:
                sentence_lower = sentence.lower().strip()
                if len(sentence_lower) > 10:  # Ignore very short sentences
                    switches_in_sentence = 0
                    for category, words in patterns.items():
                        switches_in_sentence += sum(1 for word in words if word.lower() in sentence_lower)
                    
                    if switches_in_sentence >= 2:  # At least 2 switches within sentence
                        total_switches += 1
                        evidence.append(f"Intrasentential switching in: '{sentence.strip()[:50]}...'")
        
        # Evaluate grammatical naturalness
        naturalness_score = self._assess_switching_naturalness(response, detected_pairs)
        
        if total_switches > 0:
            evidence.append(f"Intrasentential switches: {total_switches}")
            cultural_markers.append("within_sentence_mixing")
        
        # Calculate score based on frequency and naturalness
        frequency_score = min(1.0, total_switches * 0.2)
        score = (frequency_score * 0.7) + (naturalness_score * 0.3)
        confidence = min(1.0, total_switches * 0.15)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="intrasentential_switching",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_intersentential_switching(self, response: str, cultural_context: CulturalContext,
                                          test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate intersentential (between-sentence) code-switching patterns."""
        evidence = []
        cultural_markers = []
        
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) < 2:
            return EvaluationDimension(
                name="intersentential_switching",
                score=0.0,
                confidence=0.0,
                cultural_relevance=0.5,
                evidence=["Insufficient text for intersentential analysis"],
                cultural_markers=[]
            )
        
        detected_pairs = self._detect_language_pairs(response)
        
        # Analyze sentence-to-sentence switches
        sentence_switches = 0
        for i in range(len(sentences) - 1):
            current_sentence = sentences[i].lower()
            next_sentence = sentences[i + 1].lower()
            
            # Check if consecutive sentences show different language dominance
            for pair in detected_pairs:
                patterns = self.language_pairs[pair]["common_switches"]
                
                current_switches = sum(sum(1 for word in words if word in current_sentence) 
                                     for words in patterns.values())
                next_switches = sum(sum(1 for word in words if word in next_sentence) 
                                  for words in patterns.values())
                
                if (current_switches > 0 and next_switches == 0) or \
                   (current_switches == 0 and next_switches > 0):
                    sentence_switches += 1
                    evidence.append(f"Language switch between sentences {i+1} and {i+2}")
                    cultural_markers.append("sentence_boundary_switching")
                    break
        
        if sentence_switches > 0:
            evidence.append(f"Intersentential switches: {sentence_switches}")
            cultural_markers.append("between_sentence_switching")
        
        score = min(1.0, sentence_switches * 0.25)
        confidence = min(1.0, sentence_switches * 0.2)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="intersentential_switching",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_tag_switching(self, response: str, cultural_context: CulturalContext,
                              test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate tag-switching (discourse markers, exclamations, tags)."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        tag_switches = 0
        
        detected_pairs = self._detect_language_pairs(response)
        
        for pair in detected_pairs:
            patterns = self.language_pairs[pair]["common_switches"]
            
            # Count tags and exclamations
            tags_found = sum(1 for tag in patterns.get("tags", []) if tag.lower() in response_lower)
            exclamations_found = sum(1 for exc in patterns.get("exclamations", []) if exc.lower() in response_lower)
            discourse_markers_found = sum(1 for dm in patterns.get("discourse_markers", []) if dm.lower() in response_lower)
            
            total_found = tags_found + exclamations_found + discourse_markers_found
            
            if total_found > 0:
                evidence.append(f"{pair} tags/exclamations: {total_found}")
                cultural_markers.append(f"{pair}_tags")
                tag_switches += total_found
        
        if tag_switches > 0:
            evidence.append(f"Total tag switches: {tag_switches}")
            cultural_markers.append("tag_switching_competence")
        
        score = min(1.0, tag_switches * 0.2)
        confidence = min(1.0, tag_switches * 0.15)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="tag_switching",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_functional_appropriateness(self, response: str, cultural_context: CulturalContext,
                                           test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate functional appropriateness of code-switching."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        functional_switches = 0
        
        # Check for different functions of code-switching
        for function, indicators in self.switching_functions.items():
            function_count = sum(1 for indicator in indicators if indicator in response_lower)
            if function_count > 0:
                evidence.append(f"{function.capitalize()} function: {function_count} instances")
                cultural_markers.append(f"{function}_switching")
                functional_switches += function_count
        
        # Check for pragmatic functions
        for function, indicators in self.pragmatic_functions.items():
            function_count = sum(1 for indicator in indicators if indicator in response_lower)
            if function_count > 0:
                evidence.append(f"Pragmatic {function}: {function_count} instances")
                cultural_markers.append(f"pragmatic_{function}")
                functional_switches += function_count
        
        if functional_switches > 0:
            evidence.append(f"Total functional switches: {functional_switches}")
            cultural_markers.append("functional_competence")
        
        score = min(1.0, functional_switches * 0.15)
        confidence = min(1.0, functional_switches * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="functional_appropriateness",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_grammatical_constraints(self, response: str, cultural_context: CulturalContext,
                                        test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate adherence to grammatical constraints in code-switching."""
        evidence = []
        cultural_markers = []
        
        # This is a simplified evaluation - would need more sophisticated parsing in practice
        constraint_adherence = 0
        
        # Check for determiner-noun agreement across languages
        det_noun_patterns = 0
        for det in self.grammatical_constraints["noun_phrase_switching"]:
            if det in response.lower():
                det_noun_patterns += 1
        
        if det_noun_patterns > 0:
            evidence.append(f"Cross-linguistic determiner patterns: {det_noun_patterns}")
            cultural_markers.append("determiner_noun_switching")
            constraint_adherence += det_noun_patterns
        
        # Look for embedded language islands (complete phrases in other languages)
        sentences = response.split('.')
        embedded_islands = 0
        
        for sentence in sentences:
            detected_pairs = self._detect_language_pairs(sentence)
            if detected_pairs:
                # Simple heuristic for embedded islands
                words = sentence.split()
                consecutive_switches = 0
                max_consecutive = 0
                
                for word in words:
                    if any(word.lower() in sum(patterns["common_switches"].values(), []) 
                          for patterns in self.language_pairs.values()):
                        consecutive_switches += 1
                        max_consecutive = max(max_consecutive, consecutive_switches)
                    else:
                        consecutive_switches = 0
                
                if max_consecutive >= 3:  # 3+ consecutive words from other language
                    embedded_islands += 1
        
        if embedded_islands > 0:
            evidence.append(f"Embedded language islands: {embedded_islands}")
            cultural_markers.append("embedded_phrases")
            constraint_adherence += embedded_islands
        
        if constraint_adherence > 0:
            evidence.append(f"Grammatical constraint adherence: {constraint_adherence}")
            cultural_markers.append("grammatical_competence")
        
        score = min(1.0, constraint_adherence * 0.2)
        confidence = min(1.0, constraint_adherence * 0.15)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="grammatical_constraints",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_sociolinguistic_competence(self, response: str, cultural_context: CulturalContext,
                                           test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate sociolinguistic awareness in code-switching."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        sociolinguistic_awareness = 0
        
        # Check for awareness of sociolinguistic factors
        for factor_type, factors in self.sociolinguistic_factors.items():
            factor_count = sum(1 for factor in factors if factor in response_lower)
            if factor_count > 0:
                evidence.append(f"{factor_type.replace('_', ' ').title()}: {factor_count} instances")
                cultural_markers.append(f"{factor_type}_awareness")
                sociolinguistic_awareness += factor_count
        
        # Bonus for multilingual context awareness
        multilingual_bonus = 0.0
        if len(cultural_context.linguistic_varieties) > 1:
            multilingual_bonus = 0.2
            cultural_markers.append("multilingual_context_awareness")
        
        if sociolinguistic_awareness > 0:
            evidence.append(f"Sociolinguistic awareness indicators: {sociolinguistic_awareness}")
            cultural_markers.append("sociolinguistic_competence")
        
        score = min(1.0, (sociolinguistic_awareness * 0.1) + multilingual_bonus)
        confidence = min(1.0, sociolinguistic_awareness * 0.08)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="sociolinguistic_competence",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_pragmatic_functions(self, response: str, cultural_context: CulturalContext,
                                    test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate pragmatic functions of code-switching."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        pragmatic_functions = 0
        
        # Check for pragmatic functions
        for function_type, functions in self.pragmatic_functions.items():
            function_count = sum(1 for func in functions if func in response_lower)
            if function_count > 0:
                evidence.append(f"{function_type.replace('_', ' ').title()}: {function_count} instances")
                cultural_markers.append(f"{function_type}")
                pragmatic_functions += function_count
        
        if pragmatic_functions > 0:
            evidence.append(f"Total pragmatic functions: {pragmatic_functions}")
            cultural_markers.append("pragmatic_switching")
        
        score = min(1.0, pragmatic_functions * 0.15)
        confidence = min(1.0, pragmatic_functions * 0.12)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="pragmatic_functions",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_cultural_authenticity(self, response: str, cultural_context: CulturalContext,
                                      test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate cultural authenticity and appropriateness of code-switching."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        authenticity_score = 0
        
        # Check for appropriateness markers
        for marker_type, markers in self.appropriateness_markers.items():
            marker_count = sum(1 for marker in markers if marker in response_lower)
            if marker_count > 0:
                evidence.append(f"{marker_type.replace('_', ' ').title()}: {marker_count} instances")
                cultural_markers.append(f"{marker_type}")
                authenticity_score += marker_count
        
        # Cultural context bonus
        cultural_bonus = 0.0
        detected_pairs = self._detect_language_pairs(response)
        
        for pair in detected_pairs:
            # Check if detected language pair aligns with cultural context
            if pair == "spanish_english" and any(group in ["hispanic", "latino", "chicano"] 
                                               for group in cultural_context.cultural_groups):
                cultural_bonus += 0.15
            elif pair == "french_english" and any(group in ["francophone", "quebec", "cajun"] 
                                                for group in cultural_context.cultural_groups):
                cultural_bonus += 0.15
            elif pair == "arabic_english" and any(group in ["arab", "middle_eastern"] 
                                                for group in cultural_context.cultural_groups):
                cultural_bonus += 0.15
            # Add similar checks for other language pairs
        
        if cultural_bonus > 0:
            cultural_markers.append("cultural_alignment")
            evidence.append(f"Cultural-linguistic alignment detected")
        
        if authenticity_score > 0:
            evidence.append(f"Cultural authenticity indicators: {authenticity_score}")
            cultural_markers.append("authentic_switching")
        
        score = min(1.0, (authenticity_score * 0.1) + cultural_bonus)
        confidence = min(1.0, authenticity_score * 0.08)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="cultural_authenticity",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _assess_switching_naturalness(self, text: str, detected_pairs: List[str]) -> float:
        """Assess the naturalness of code-switching patterns."""
        # Simplified naturalness assessment
        # In practice, this would involve more sophisticated linguistic analysis
        
        naturalness_score = 0.5  # Base naturalness
        
        # Check for common natural switching patterns
        for pair in detected_pairs:
            if pair in self.language_pairs:
                functional_patterns = self.language_pairs[pair].get("functional_patterns", {})
                for pattern_type, patterns in functional_patterns.items():
                    if any(pattern in text.lower() for pattern in patterns):
                        naturalness_score += 0.1
        
        return min(1.0, naturalness_score)
    
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
            
            # Higher relevance for markers aligned with multilingual context
            if marker_type in ["spanish", "french", "mandarin", "arabic", "hindi"]:
                language = marker_type
                if language in [variety.lower() for variety in cultural_context.linguistic_varieties]:
                    relevance_score = 0.95
                else:
                    relevance_score = 0.6
            elif marker_type in ["multilingual", "bilingual", "switching"]:
                if len(cultural_context.linguistic_varieties) > 1:
                    relevance_score = 0.9
                else:
                    relevance_score = 0.3
            elif marker_type in ["cultural", "authentic", "competence"]:
                relevance_score = 0.8
            elif marker_type in ["pragmatic", "functional", "sociolinguistic"]:
                relevance_score = 0.75
            elif marker_type in ["grammatical", "constraint"]:
                relevance_score = 0.7
            
            total_relevance += relevance_score
            marker_count += 1
        
        return min(1.0, total_relevance / marker_count) if marker_count > 0 else 0.5