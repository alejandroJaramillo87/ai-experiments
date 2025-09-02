"""
Pragmatic Meaning Evaluator

Specialized evaluator for assessing pragmatic language competence including
conversational implicature, speech acts, context interpretation, presupposition,
deixis, conversational maxims, and pragmatic inference skills.

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


class PragmaticMeaningType(Enum):
    """Types of pragmatic meaning evaluation."""
    IMPLICATURE = "implicature"
    SPEECH_ACTS = "speech_acts"
    CONTEXT_INTERPRETATION = "context_interpretation"
    PRESUPPOSITION = "presupposition"
    DEIXIS = "deixis"
    CONVERSATIONAL_MAXIMS = "conversational_maxims"
    PRAGMATIC_INFERENCE = "pragmatic_inference"


class PragmaticMeaningEvaluator(MultiDimensionalEvaluator):
    """Evaluates pragmatic language competence and meaning interpretation."""
    
    VERSION = "1.0.0"
    
    def _initialize_evaluator(self):
        """Initialize pragmatic meaning evaluation components."""
        
        # Conversational implicature patterns
        self.implicature_patterns = {
            "conversational_implicature": [
                "implied meaning", "reading between the lines", "what they really meant",
                "subtext", "unspoken message", "indirect meaning", "hint",
                "suggestion", "inference", "implication", "underlying message"
            ],
            "conventional_implicature": [
                "but", "however", "even", "still", "yet", "although",
                "therefore", "consequently", "thus", "hence", "so"
            ],
            "scalar_implicature": [
                "some", "many", "most", "all", "possible", "certain",
                "sometimes", "often", "always", "never", "good", "excellent"
            ],
            "particularized_implicature": [
                "context-dependent", "situation-specific", "particular context",
                "given the circumstances", "in this case", "contextual meaning"
            ]
        }
        
        # Speech acts classification
        self.speech_acts_patterns = {
            "assertives": [
                "I state", "I claim", "I assert", "I declare", "I announce",
                "I report", "I inform", "I tell", "I mention", "I describe",
                "I explain", "I suggest", "I believe", "I think"
            ],
            "directives": [
                "I request", "I ask", "I order", "I command", "I instruct",
                "I direct", "I advise", "I recommend", "I urge", "I beg",
                "please", "could you", "would you", "can you"
            ],
            "commissives": [
                "I promise", "I pledge", "I vow", "I swear", "I guarantee",
                "I commit", "I agree", "I consent", "I undertake", "I will"
            ],
            "expressives": [
                "I apologize", "I thank", "I congratulate", "I welcome",
                "I sympathize", "I condole", "I regret", "I'm sorry",
                "thank you", "congratulations", "welcome"
            ],
            "declarations": [
                "I pronounce", "I declare", "I name", "I christen",
                "I sentence", "I find guilty", "I now pronounce",
                "you're fired", "meeting adjourned", "court dismissed"
            ]
        }
        
        # Context interpretation markers
        self.context_interpretation_patterns = {
            "situational_context": [
                "given the situation", "in this context", "considering the circumstances",
                "situational awareness", "context clues", "environmental factors",
                "setting", "occasion", "circumstances", "situation"
            ],
            "linguistic_context": [
                "previous utterance", "earlier statement", "what was said before",
                "linguistic context", "co-text", "surrounding text",
                "discourse context", "textual context", "verbal context"
            ],
            "cultural_context": [
                "cultural background", "cultural norms", "cultural expectations",
                "social conventions", "cultural knowledge", "cultural frame",
                "cultural assumptions", "cultural context", "social context"
            ],
            "shared_knowledge": [
                "common ground", "shared assumptions", "mutual knowledge",
                "background knowledge", "shared understanding", "common knowledge",
                "presumed knowledge", "assumed knowledge", "shared beliefs"
            ]
        }
        
        # Presupposition patterns
        self.presupposition_patterns = {
            "existential_presupposition": [
                "the king of france", "my brother", "john's car", "the winner",
                "the person who", "the one that", "definite reference"
            ],
            "factive_presupposition": [
                "know that", "realize that", "regret that", "be aware that",
                "understand that", "recognize that", "discover that"
            ],
            "lexical_presupposition": [
                "stop", "start", "continue", "again", "still", "yet",
                "manage to", "fail to", "succeed in", "attempt to"
            ],
            "structural_presupposition": [
                "wh-questions", "it-clefts", "pseudo-clefts", "what bothers me is",
                "the problem is", "what happened was", "where I went was"
            ]
        }
        
        # Deictic expressions
        self.deixis_patterns = {
            "person_deixis": [
                "I", "you", "we", "they", "he", "she", "it",
                "my", "your", "our", "their", "his", "her", "its",
                "myself", "yourself", "ourselves", "themselves"
            ],
            "spatial_deixis": [
                "here", "there", "this", "that", "these", "those",
                "come", "go", "bring", "take", "up", "down",
                "left", "right", "near", "far", "close", "distant"
            ],
            "temporal_deixis": [
                "now", "then", "today", "yesterday", "tomorrow",
                "this week", "last month", "next year", "recently",
                "soon", "later", "before", "after", "currently"
            ],
            "discourse_deixis": [
                "this point", "that argument", "the following", "the preceding",
                "aforementioned", "above-mentioned", "as mentioned", "as stated"
            ]
        }
        
        # Conversational maxims (Grice)
        self.maxims_patterns = {
            "quantity_maxim": [
                "informative enough", "sufficient information", "just enough detail",
                "not too much information", "appropriate amount", "right level of detail",
                "adequate information", "necessary information", "relevant amount"
            ],
            "quality_maxim": [
                "truthful", "accurate", "honest", "reliable", "factual",
                "evidence-based", "verified", "true", "false", "lie",
                "misleading", "deceptive", "truthfulness", "accuracy"
            ],
            "relation_maxim": [
                "relevant", "on topic", "pertinent", "related", "connected",
                "applicable", "irrelevant", "off-topic", "unrelated",
                "beside the point", "relevance", "relevancy"
            ],
            "manner_maxim": [
                "clear", "brief", "orderly", "unambiguous", "perspicuous",
                "obscure", "ambiguous", "verbose", "unclear", "confusing",
                "clarity", "brevity", "organization", "ambiguity"
            ]
        }
        
        # Pragmatic inference indicators
        self.pragmatic_inference_patterns = {
            "causal_inference": [
                "because of this", "as a result", "therefore", "consequently",
                "leads to", "causes", "results in", "due to", "owing to"
            ],
            "temporal_inference": [
                "sequence of events", "chronological order", "time progression",
                "before this", "after that", "temporal relationship",
                "sequence", "order", "timing", "temporal connection"
            ],
            "conditional_inference": [
                "if then", "conditional relationship", "hypothetical", "suppose",
                "assume", "given that", "provided that", "unless", "otherwise"
            ],
            "comparative_inference": [
                "comparison", "contrast", "similarity", "difference",
                "like", "unlike", "similar to", "different from",
                "in comparison", "by contrast", "relative to"
            ]
        }
        
        # Pragmatic failure patterns (what to avoid)
        self.pragmatic_failure_patterns = {
            "literal_interpretation": [
                "taking literally", "literal meaning", "word-for-word",
                "surface meaning", "explicit meaning", "direct interpretation"
            ],
            "context_ignorance": [
                "ignoring context", "context-blind", "decontextualized",
                "without considering", "isolated interpretation"
            ],
            "cultural_misunderstanding": [
                "cultural misinterpretation", "cultural blindness", "ethnocentric",
                "cultural insensitivity", "missing cultural cues"
            ]
        }
        
        # Politeness strategies (Brown & Levinson)
        self.politeness_patterns = {
            "positive_politeness": [
                "we", "us", "together", "solidarity", "common ground",
                "shared interest", "friendship", "camaraderie", "in-group"
            ],
            "negative_politeness": [
                "if you don't mind", "sorry to bother", "could you possibly",
                "would it be possible", "I hope", "perhaps", "maybe"
            ],
            "off_record": [
                "hint", "indirect", "metaphorical", "rhetorical question",
                "irony", "sarcasm", "understatement", "tautology"
            ]
        }
    
    def get_domain_name(self) -> str:
        """Return the domain name this evaluator handles."""
        return "pragmatic_meaning"
    
    def get_supported_evaluation_types(self) -> List[str]:
        """Return list of evaluation types this evaluator supports."""
        return [evaluation_type.value for evaluation_type in PragmaticMeaningType]
    
    def get_evaluation_dimensions(self) -> List[str]:
        """Return list of dimensions this evaluator assesses."""
        return [
            "implicature_understanding",
            "speech_act_recognition",
            "context_interpretation",
            "presupposition_handling",
            "deixis_resolution",
            "maxim_adherence",
            "pragmatic_inference",
            "politeness_strategies"
        ]
    
    def evaluate_dimension(self, 
                          dimension: str,
                          response_text: str, 
                          test_metadata: Dict[str, Any], 
                          cultural_context: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate a specific dimension."""
        cultural_ctx = self._create_cultural_context(cultural_context)
        
        if dimension == "implicature_understanding":
            return self._evaluate_implicature_understanding(response_text, cultural_ctx, test_metadata)
        elif dimension == "speech_act_recognition":
            return self._evaluate_speech_act_recognition(response_text, cultural_ctx, test_metadata)
        elif dimension == "context_interpretation":
            return self._evaluate_context_interpretation(response_text, cultural_ctx, test_metadata)
        elif dimension == "presupposition_handling":
            return self._evaluate_presupposition_handling(response_text, cultural_ctx, test_metadata)
        elif dimension == "deixis_resolution":
            return self._evaluate_deixis_resolution(response_text, cultural_ctx, test_metadata)
        elif dimension == "maxim_adherence":
            return self._evaluate_maxim_adherence(response_text, cultural_ctx, test_metadata)
        elif dimension == "pragmatic_inference":
            return self._evaluate_pragmatic_inference(response_text, cultural_ctx, test_metadata)
        elif dimension == "politeness_strategies":
            return self._evaluate_politeness_strategies(response_text, cultural_ctx, test_metadata)
        else:
            return EvaluationDimension(
                name=dimension,
                score=0.0,
                confidence=0.0,
                cultural_relevance=0.0,
                evidence=[f"Unknown dimension: {dimension}"],
                cultural_markers=[]
            )
    
    def _evaluate_implicature_understanding(self, response: str, cultural_context: CulturalContext,
                                          test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate understanding of conversational implicature."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count implicature indicators
        conversational_impl = sum(1 for impl in self.implicature_patterns["conversational_implicature"]
                                 if impl in response_lower)
        conventional_impl = sum(1 for impl in self.implicature_patterns["conventional_implicature"]
                               if impl in response_lower)
        scalar_impl = sum(1 for impl in self.implicature_patterns["scalar_implicature"]
                         if impl in response_lower)
        particularized_impl = sum(1 for impl in self.implicature_patterns["particularized_implicature"]
                                 if impl in response_lower)
        
        # Check for pragmatic failure (literal interpretation)
        literal_interp = sum(1 for literal in self.pragmatic_failure_patterns["literal_interpretation"]
                            if literal in response_lower)
        
        if conversational_impl > 0:
            evidence.append(f"Conversational implicature awareness: {conversational_impl} instances")
            cultural_markers.append("conversational_implicature_competence")
        
        if conventional_impl > 0:
            evidence.append(f"Conventional implicature markers: {conventional_impl} instances")
            cultural_markers.append("conventional_implicature_competence")
        
        if scalar_impl > 0:
            evidence.append(f"Scalar implicature usage: {scalar_impl} instances")
            cultural_markers.append("scalar_implicature_competence")
        
        if particularized_impl > 0:
            evidence.append(f"Particularized implicature: {particularized_impl} instances")
            cultural_markers.append("context_dependent_implicature")
        
        # Penalty for overly literal interpretation
        literal_penalty = literal_interp * 0.1
        
        if literal_penalty > 0:
            evidence.append(f"Literal interpretation tendency: {literal_interp} instances")
            cultural_markers.append("literal_interpretation_risk")
        
        total_score = (conversational_impl * 0.4 + conventional_impl * 0.25 + 
                      scalar_impl * 0.2 + particularized_impl * 0.15)
        
        score = max(0.0, min(1.0, (total_score * 0.15) - literal_penalty))
        confidence = min(1.0, (conversational_impl + conventional_impl + 
                              scalar_impl + particularized_impl) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="implicature_understanding",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_speech_act_recognition(self, response: str, cultural_context: CulturalContext,
                                       test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate recognition and understanding of speech acts."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count speech act types
        assertives = sum(1 for act in self.speech_acts_patterns["assertives"]
                        if act in response_lower)
        directives = sum(1 for act in self.speech_acts_patterns["directives"]
                        if act in response_lower)
        commissives = sum(1 for act in self.speech_acts_patterns["commissives"]
                         if act in response_lower)
        expressives = sum(1 for act in self.speech_acts_patterns["expressives"]
                         if act in response_lower)
        declarations = sum(1 for act in self.speech_acts_patterns["declarations"]
                          if act in response_lower)
        
        if assertives > 0:
            evidence.append(f"Assertive speech acts: {assertives} instances")
            cultural_markers.append("assertive_speech_competence")
        
        if directives > 0:
            evidence.append(f"Directive speech acts: {directives} instances")
            cultural_markers.append("directive_speech_competence")
        
        if commissives > 0:
            evidence.append(f"Commissive speech acts: {commissives} instances")
            cultural_markers.append("commissive_speech_competence")
        
        if expressives > 0:
            evidence.append(f"Expressive speech acts: {expressives} instances")
            cultural_markers.append("expressive_speech_competence")
        
        if declarations > 0:
            evidence.append(f"Declarative speech acts: {declarations} instances")
            cultural_markers.append("declarative_speech_competence")
        
        # Diversity bonus for using multiple speech act types
        speech_act_types = sum([1 for count in [assertives, directives, commissives, expressives, declarations] if count > 0])
        diversity_bonus = (speech_act_types - 1) * 0.05  # Bonus for variety
        
        total_score = (assertives * 0.25 + directives * 0.25 + commissives * 0.2 + 
                      expressives * 0.2 + declarations * 0.1)
        
        score = min(1.0, (total_score * 0.1) + diversity_bonus)
        confidence = min(1.0, (assertives + directives + commissives + expressives + declarations) * 0.08)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="speech_act_recognition",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_context_interpretation(self, response: str, cultural_context: CulturalContext,
                                       test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate ability to interpret meaning based on context."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count context interpretation indicators
        situational_ctx = sum(1 for ctx in self.context_interpretation_patterns["situational_context"]
                             if ctx in response_lower)
        linguistic_ctx = sum(1 for ctx in self.context_interpretation_patterns["linguistic_context"]
                            if ctx in response_lower)
        cultural_ctx = sum(1 for ctx in self.context_interpretation_patterns["cultural_context"]
                          if ctx in response_lower)
        shared_knowledge = sum(1 for knowledge in self.context_interpretation_patterns["shared_knowledge"]
                              if knowledge in response_lower)
        
        # Check for context ignorance (negative indicator)
        context_ignorance = sum(1 for ignore in self.pragmatic_failure_patterns["context_ignorance"]
                               if ignore in response_lower)
        
        if situational_ctx > 0:
            evidence.append(f"Situational context awareness: {situational_ctx} instances")
            cultural_markers.append("situational_context_competence")
        
        if linguistic_ctx > 0:
            evidence.append(f"Linguistic context awareness: {linguistic_ctx} instances")
            cultural_markers.append("linguistic_context_competence")
        
        if cultural_ctx > 0:
            evidence.append(f"Cultural context awareness: {cultural_ctx} instances")
            cultural_markers.append("cultural_context_competence")
        
        if shared_knowledge > 0:
            evidence.append(f"Shared knowledge awareness: {shared_knowledge} instances")
            cultural_markers.append("shared_knowledge_competence")
        
        # Penalty for context ignorance
        ignorance_penalty = context_ignorance * 0.1
        
        if ignorance_penalty > 0:
            evidence.append(f"Context ignorance indicators: {context_ignorance} instances")
            cultural_markers.append("context_ignorance_risk")
        
        total_score = (situational_ctx * 0.3 + linguistic_ctx * 0.25 + 
                      cultural_ctx * 0.25 + shared_knowledge * 0.2)
        
        score = max(0.0, min(1.0, (total_score * 0.15) - ignorance_penalty))
        confidence = min(1.0, (situational_ctx + linguistic_ctx + cultural_ctx + shared_knowledge) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="context_interpretation",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_presupposition_handling(self, response: str, cultural_context: CulturalContext,
                                        test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate handling of presuppositions in communication."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count presupposition types
        existential = sum(1 for pres in self.presupposition_patterns["existential_presupposition"]
                         if pres in response_lower)
        factive = sum(1 for pres in self.presupposition_patterns["factive_presupposition"]
                     if pres in response_lower)
        lexical = sum(1 for pres in self.presupposition_patterns["lexical_presupposition"]
                     if pres in response_lower)
        structural = sum(1 for pres in self.presupposition_patterns["structural_presupposition"]
                        if pres in response_lower)
        
        if existential > 0:
            evidence.append(f"Existential presuppositions: {existential} instances")
            cultural_markers.append("existential_presupposition_competence")
        
        if factive > 0:
            evidence.append(f"Factive presuppositions: {factive} instances")
            cultural_markers.append("factive_presupposition_competence")
        
        if lexical > 0:
            evidence.append(f"Lexical presuppositions: {lexical} instances")
            cultural_markers.append("lexical_presupposition_competence")
        
        if structural > 0:
            evidence.append(f"Structural presuppositions: {structural} instances")
            cultural_markers.append("structural_presupposition_competence")
        
        total_score = (existential * 0.25 + factive * 0.3 + lexical * 0.25 + structural * 0.2)
        
        score = min(1.0, total_score * 0.2)
        confidence = min(1.0, (existential + factive + lexical + structural) * 0.15)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="presupposition_handling",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_deixis_resolution(self, response: str, cultural_context: CulturalContext,
                                  test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate resolution of deictic expressions."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count deixis types
        person_deixis = sum(1 for deixis in self.deixis_patterns["person_deixis"]
                           if deixis in response_lower)
        spatial_deixis = sum(1 for deixis in self.deixis_patterns["spatial_deixis"]
                            if deixis in response_lower)
        temporal_deixis = sum(1 for deixis in self.deixis_patterns["temporal_deixis"]
                             if deixis in response_lower)
        discourse_deixis = sum(1 for deixis in self.deixis_patterns["discourse_deixis"]
                              if deixis in response_lower)
        
        if person_deixis > 0:
            evidence.append(f"Person deixis usage: {person_deixis} instances")
            cultural_markers.append("person_deixis_competence")
        
        if spatial_deixis > 0:
            evidence.append(f"Spatial deixis usage: {spatial_deixis} instances")
            cultural_markers.append("spatial_deixis_competence")
        
        if temporal_deixis > 0:
            evidence.append(f"Temporal deixis usage: {temporal_deixis} instances")
            cultural_markers.append("temporal_deixis_competence")
        
        if discourse_deixis > 0:
            evidence.append(f"Discourse deixis usage: {discourse_deixis} instances")
            cultural_markers.append("discourse_deixis_competence")
        
        total_score = (person_deixis * 0.3 + spatial_deixis * 0.25 + 
                      temporal_deixis * 0.25 + discourse_deixis * 0.2)
        
        score = min(1.0, total_score * 0.05)  # Lower multiplier as deixis is very common
        confidence = min(1.0, (person_deixis + spatial_deixis + temporal_deixis + discourse_deixis) * 0.02)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="deixis_resolution",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_maxim_adherence(self, response: str, cultural_context: CulturalContext,
                                test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate adherence to conversational maxims."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count maxim-related indicators
        quantity = sum(1 for maxim in self.maxims_patterns["quantity_maxim"]
                      if maxim in response_lower)
        quality = sum(1 for maxim in self.maxims_patterns["quality_maxim"]
                     if maxim in response_lower)
        relation = sum(1 for maxim in self.maxims_patterns["relation_maxim"]
                      if maxim in response_lower)
        manner = sum(1 for maxim in self.maxims_patterns["manner_maxim"]
                    if maxim in response_lower)
        
        if quantity > 0:
            evidence.append(f"Quantity maxim awareness: {quantity} instances")
            cultural_markers.append("quantity_maxim_competence")
        
        if quality > 0:
            evidence.append(f"Quality maxim awareness: {quality} instances")
            cultural_markers.append("quality_maxim_competence")
        
        if relation > 0:
            evidence.append(f"Relation maxim awareness: {relation} instances")
            cultural_markers.append("relation_maxim_competence")
        
        if manner > 0:
            evidence.append(f"Manner maxim awareness: {manner} instances")
            cultural_markers.append("manner_maxim_competence")
        
        total_score = (quantity * 0.25 + quality * 0.25 + relation * 0.25 + manner * 0.25)
        
        score = min(1.0, total_score * 0.2)
        confidence = min(1.0, (quantity + quality + relation + manner) * 0.15)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="maxim_adherence",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_pragmatic_inference(self, response: str, cultural_context: CulturalContext,
                                    test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate pragmatic inference abilities."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count inference types
        causal = sum(1 for inf in self.pragmatic_inference_patterns["causal_inference"]
                    if inf in response_lower)
        temporal = sum(1 for inf in self.pragmatic_inference_patterns["temporal_inference"]
                      if inf in response_lower)
        conditional = sum(1 for inf in self.pragmatic_inference_patterns["conditional_inference"]
                         if inf in response_lower)
        comparative = sum(1 for inf in self.pragmatic_inference_patterns["comparative_inference"]
                         if inf in response_lower)
        
        if causal > 0:
            evidence.append(f"Causal inference: {causal} instances")
            cultural_markers.append("causal_inference_competence")
        
        if temporal > 0:
            evidence.append(f"Temporal inference: {temporal} instances")
            cultural_markers.append("temporal_inference_competence")
        
        if conditional > 0:
            evidence.append(f"Conditional inference: {conditional} instances")
            cultural_markers.append("conditional_inference_competence")
        
        if comparative > 0:
            evidence.append(f"Comparative inference: {comparative} instances")
            cultural_markers.append("comparative_inference_competence")
        
        total_score = (causal * 0.3 + temporal * 0.25 + conditional * 0.25 + comparative * 0.2)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (causal + temporal + conditional + comparative) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="pragmatic_inference",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_politeness_strategies(self, response: str, cultural_context: CulturalContext,
                                      test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate use of politeness strategies."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count politeness strategy types
        positive_politeness = sum(1 for pol in self.politeness_patterns["positive_politeness"]
                                 if pol in response_lower)
        negative_politeness = sum(1 for pol in self.politeness_patterns["negative_politeness"]
                                 if pol in response_lower)
        off_record = sum(1 for pol in self.politeness_patterns["off_record"]
                        if pol in response_lower)
        
        if positive_politeness > 0:
            evidence.append(f"Positive politeness strategies: {positive_politeness} instances")
            cultural_markers.append("positive_politeness_competence")
        
        if negative_politeness > 0:
            evidence.append(f"Negative politeness strategies: {negative_politeness} instances")
            cultural_markers.append("negative_politeness_competence")
        
        if off_record > 0:
            evidence.append(f"Off-record strategies: {off_record} instances")
            cultural_markers.append("off_record_competence")
        
        # Cultural context bonus for appropriate politeness
        cultural_bonus = 0.0
        if any(group in ["east_asian", "japanese"] for group in cultural_context.cultural_groups):
            if negative_politeness > 0:
                cultural_bonus = 0.1
                cultural_markers.append("cultural_politeness_alignment")
        
        total_score = (positive_politeness * 0.35 + negative_politeness * 0.35 + off_record * 0.3)
        
        score = min(1.0, (total_score * 0.15) + cultural_bonus)
        confidence = min(1.0, (positive_politeness + negative_politeness + off_record) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="politeness_strategies",
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
            
            # Higher relevance for context-sensitive pragmatic competencies
            if marker_type in ["conversational", "implicature", "context"] and len(cultural_context.cultural_groups) > 0:
                relevance_score = 0.9
            elif marker_type in ["cultural", "shared"] and len(cultural_context.cultural_groups) > 0:
                relevance_score = 0.95
            elif marker_type in ["politeness", "negative", "positive"] and "politeness_focus" in cultural_context.performance_aspects:
                relevance_score = 0.9
            elif marker_type in ["speech", "directive", "commissive", "expressive"]:
                relevance_score = 0.8
            elif marker_type in ["pragmatic", "inference", "causal", "temporal"]:
                relevance_score = 0.85
            elif marker_type in ["deixis", "person", "spatial", "temporal"]:
                relevance_score = 0.7
            elif marker_type in ["presupposition", "factive", "existential"]:
                relevance_score = 0.8
            elif marker_type in ["maxim", "quantity", "quality", "relation", "manner"]:
                relevance_score = 0.8
            elif marker_type in ["literal", "ignorance"] and len(cultural_context.cultural_groups) > 0:
                relevance_score = 0.9  # High relevance for awareness of pragmatic failures
            
            total_relevance += relevance_score
            marker_count += 1
        
        return min(1.0, total_relevance / marker_count) if marker_count > 0 else 0.5