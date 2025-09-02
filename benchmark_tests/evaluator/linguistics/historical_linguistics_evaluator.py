"""
Historical Linguistics Evaluator

Specialized evaluator for assessing historical linguistics competence including
diachronic analysis, etymology, sound changes, grammaticalization, language families,
comparative method, and reconstruction skills.

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


class HistoricalLinguisticsType(Enum):
    """Types of historical linguistics evaluation."""
    ETYMOLOGY = "etymology"
    SOUND_CHANGE = "sound_change"
    GRAMMATICALIZATION = "grammaticalization"
    COMPARATIVE_METHOD = "comparative_method"
    LANGUAGE_FAMILIES = "language_families"
    RECONSTRUCTION = "reconstruction"
    BORROWING_CONTACT = "borrowing_contact"


class HistoricalLinguisticsEvaluator(MultiDimensionalEvaluator):
    """Evaluates historical linguistics knowledge and analytical skills."""
    
    VERSION = "1.0.0"
    
    def _initialize_evaluator(self):
        """Initialize historical linguistics evaluation components."""
        
        # Etymology and word origins
        self.etymology_patterns = {
            "etymological_awareness": [
                "etymology", "derives from", "comes from", "originates from",
                "borrowed from", "root word", "cognate", "related to",
                "descended from", "evolved from", "traces back to"
            ],
            "source_languages": [
                "latin", "greek", "germanic", "sanskrit", "proto-indo-european",
                "old english", "middle english", "old french", "vulgar latin",
                "arabic", "hebrew", "celtic", "slavic", "romance"
            ],
            "word_formation": [
                "compound", "prefix", "suffix", "root", "stem", "morpheme",
                "derivation", "affixation", "compounding", "blending"
            ],
            "semantic_change": [
                "semantic shift", "narrowing", "broadening", "amelioration",
                "pejoration", "metaphorical extension", "metonymy",
                "generalization", "specialization", "semantic bleaching"
            ]
        }
        
        # Sound changes and phonological evolution
        self.sound_change_patterns = {
            "regular_sound_changes": [
                "sound change", "phonetic change", "sound law", "regular correspondence",
                "grimm's law", "verner's law", "rhotacism", "lenition",
                "palatalization", "umlaut", "vowel shift", "consonant shift"
            ],
            "phonological_processes": [
                "assimilation", "dissimilation", "epenthesis", "syncope",
                "apocope", "metathesis", "haplology", "analogy",
                "leveling", "conditioned change", "unconditioned change"
            ],
            "historical_phonology": [
                "proto-form", "reflex", "correspondence set", "reconstruction",
                "internal reconstruction", "comparative reconstruction",
                "phoneme merger", "phoneme split", "chain shift"
            ]
        }
        
        # Grammaticalization and morphosyntactic change
        self.grammaticalization_patterns = {
            "grammaticalization_processes": [
                "grammaticalization", "lexicalization", "degrammaticalization",
                "reanalysis", "extension", "decategorialization",
                "erosion", "cliticization", "auxiliation", "univerbation"
            ],
            "grammaticalization_paths": [
                "demonstrative to article", "verb to auxiliary", "noun to preposition",
                "body part to spatial", "temporal to causal", "concrete to abstract",
                "lexical to functional", "content word to function word"
            ],
            "morphosyntactic_change": [
                "word order change", "case loss", "agreement loss",
                "verbal inflection", "nominal inflection", "syntactic change",
                "clause structure", "constituent order", "head direction"
            ]
        }
        
        # Comparative method and reconstruction
        self.comparative_method_patterns = {
            "comparative_principles": [
                "comparative method", "regular correspondence", "cognate set",
                "systematic comparison", "proto-language", "common ancestor",
                "family tree", "genetic relationship", "shared innovation"
            ],
            "reconstruction_terminology": [
                "reconstruct", "proto-form", "asterisk form", "hypothetical form",
                "ancestral form", "parent language", "daughter language",
                "innovation", "retention", "archaic feature"
            ],
            "comparative_evidence": [
                "cognate", "correspondence", "systematic relationship",
                "regular pattern", "sound correspondence", "morphological correspondence",
                "lexical correspondence", "phoneme inventory", "sound system"
            ]
        }
        
        # Language families and classification
        self.language_family_patterns = {
            "major_families": [
                "indo-european", "sino-tibetan", "niger-congo", "austronesian",
                "trans-new guinea", "austroasiatic", "tai-kadai", "dravidian",
                "afroasiatic", "nilo-saharan", "khoe-kwadi", "atlantic-congo"
            ],
            "subfamily_knowledge": [
                "germanic", "romance", "slavic", "celtic", "indo-iranian",
                "italic", "hellenic", "balto-slavic", "anatolian", "tocharian",
                "mandarin", "bantu", "malayo-polynesian", "semitic", "cushitic"
            ],
            "classification_terminology": [
                "language family", "subfamily", "branch", "group",
                "genetic classification", "phylogenetic", "cladistic",
                "subgrouping", "innovation-based", "isogloss"
            ]
        }
        
        # Language contact and borrowing
        self.contact_patterns = {
            "borrowing_types": [
                "loanword", "borrowing", "calque", "loan translation",
                "substrate", "superstrate", "adstrate", "interference",
                "code-switching", "bilingual", "contact language"
            ],
            "contact_phenomena": [
                "pidgin", "creole", "mixed language", "language death",
                "language shift", "convergence", "areal feature",
                "sprachbund", "linguistic area", "contact-induced change"
            ],
            "borrowing_domains": [
                "cultural borrowing", "core vocabulary", "basic vocabulary",
                "swadesh list", "resistant vocabulary", "cultural terms",
                "technical vocabulary", "religious vocabulary"
            ]
        }
        
        # Diachronic analysis methods
        self.diachronic_methods = {
            "dating_methods": [
                "glottochronology", "lexicostatistics", "archaeological correlation",
                "historical records", "textual evidence", "epigraphic evidence",
                "relative chronology", "absolute chronology"
            ],
            "analytical_approaches": [
                "internal reconstruction", "comparative reconstruction",
                "philological method", "textual analysis", "corpus analysis",
                "variationist approach", "sociolinguistic reconstruction"
            ],
            "evidence_types": [
                "written records", "inscriptions", "manuscripts", "glosses",
                "place names", "personal names", "toponymy", "anthroponymy",
                "archaeological evidence", "genetic evidence"
            ]
        }
        
        # Technical terminology and concepts
        self.technical_concepts = {
            "linguistic_concepts": [
                "diachronic", "synchronic", "panchronic", "historical",
                "comparative", "genetic", "areal", "typological",
                "universal", "markedness", "naturalness", "frequency"
            ],
            "change_mechanisms": [
                "analogy", "reanalysis", "borrowing", "sound change",
                "semantic change", "syntactic change", "morphological change",
                "contact", "drift", "substratum", "innovation", "diffusion"
            ],
            "methodological_terms": [
                "regularity hypothesis", "neogrammarian", "family tree model",
                "wave model", "punctuated equilibrium", "uniformitarian principle",
                "actualism", "parsimony", "shared innovation"
            ]
        }
    
    def get_domain_name(self) -> str:
        """Return the domain name this evaluator handles."""
        return "historical_linguistics"
    
    def get_supported_evaluation_types(self) -> List[str]:
        """Return list of evaluation types this evaluator supports."""
        return [evaluation_type.value for evaluation_type in HistoricalLinguisticsType]
    
    def get_evaluation_dimensions(self) -> List[str]:
        """Return list of dimensions this evaluator assesses."""
        return [
            "etymological_analysis",
            "sound_change_knowledge",
            "grammaticalization_understanding",
            "comparative_method_application",
            "language_family_classification",
            "reconstruction_skills",
            "contact_linguistics",
            "diachronic_methodology"
        ]
    
    def evaluate_dimension(self, 
                          dimension: str,
                          response_text: str, 
                          test_metadata: Dict[str, Any], 
                          cultural_context: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate a specific dimension."""
        cultural_ctx = self._create_cultural_context(cultural_context)
        
        if dimension == "etymological_analysis":
            return self._evaluate_etymological_analysis(response_text, cultural_ctx, test_metadata)
        elif dimension == "sound_change_knowledge":
            return self._evaluate_sound_change_knowledge(response_text, cultural_ctx, test_metadata)
        elif dimension == "grammaticalization_understanding":
            return self._evaluate_grammaticalization_understanding(response_text, cultural_ctx, test_metadata)
        elif dimension == "comparative_method_application":
            return self._evaluate_comparative_method_application(response_text, cultural_ctx, test_metadata)
        elif dimension == "language_family_classification":
            return self._evaluate_language_family_classification(response_text, cultural_ctx, test_metadata)
        elif dimension == "reconstruction_skills":
            return self._evaluate_reconstruction_skills(response_text, cultural_ctx, test_metadata)
        elif dimension == "contact_linguistics":
            return self._evaluate_contact_linguistics(response_text, cultural_ctx, test_metadata)
        elif dimension == "diachronic_methodology":
            return self._evaluate_diachronic_methodology(response_text, cultural_ctx, test_metadata)
        else:
            return EvaluationDimension(
                name=dimension,
                score=0.0,
                confidence=0.0,
                cultural_relevance=0.0,
                evidence=[f"Unknown dimension: {dimension}"],
                cultural_markers=[]
            )
    
    def _evaluate_etymological_analysis(self, response: str, cultural_context: CulturalContext,
                                      test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate etymological analysis skills and knowledge."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count etymological indicators
        etymology_awareness = sum(1 for term in self.etymology_patterns["etymological_awareness"]
                                 if term in response_lower)
        source_languages = sum(1 for lang in self.etymology_patterns["source_languages"]
                              if lang in response_lower)
        word_formation = sum(1 for process in self.etymology_patterns["word_formation"]
                            if process in response_lower)
        semantic_change = sum(1 for change in self.etymology_patterns["semantic_change"]
                             if change in response_lower)
        
        if etymology_awareness > 0:
            evidence.append(f"Etymological awareness: {etymology_awareness} instances")
            cultural_markers.append("etymological_competence")
        
        if source_languages > 0:
            evidence.append(f"Source language knowledge: {source_languages} instances")
            cultural_markers.append("historical_language_knowledge")
        
        if word_formation > 0:
            evidence.append(f"Word formation processes: {word_formation} instances")
            cultural_markers.append("morphological_analysis")
        
        if semantic_change > 0:
            evidence.append(f"Semantic change awareness: {semantic_change} instances")
            cultural_markers.append("semantic_evolution_knowledge")
        
        # Cultural context bonus for heritage languages
        heritage_bonus = 0.0
        if "linguistic_heritage" in cultural_context.knowledge_systems:
            heritage_bonus = 0.1
            cultural_markers.append("heritage_language_etymology")
        
        total_score = (etymology_awareness * 0.3 + source_languages * 0.25 + 
                      word_formation * 0.25 + semantic_change * 0.2)
        
        score = min(1.0, (total_score * 0.1) + heritage_bonus)
        confidence = min(1.0, (etymology_awareness + source_languages + 
                              word_formation + semantic_change) * 0.08)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="etymological_analysis",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_sound_change_knowledge(self, response: str, cultural_context: CulturalContext,
                                       test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate understanding of sound changes and historical phonology."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count sound change indicators
        regular_changes = sum(1 for change in self.sound_change_patterns["regular_sound_changes"]
                             if change in response_lower)
        phonological_processes = sum(1 for process in self.sound_change_patterns["phonological_processes"]
                                   if process in response_lower)
        historical_phonology = sum(1 for concept in self.sound_change_patterns["historical_phonology"]
                                  if concept in response_lower)
        
        if regular_changes > 0:
            evidence.append(f"Regular sound changes: {regular_changes} instances")
            cultural_markers.append("sound_law_knowledge")
        
        if phonological_processes > 0:
            evidence.append(f"Phonological processes: {phonological_processes} instances")
            cultural_markers.append("phonological_analysis")
        
        if historical_phonology > 0:
            evidence.append(f"Historical phonology concepts: {historical_phonology} instances")
            cultural_markers.append("diachronic_phonology")
        
        total_score = (regular_changes * 0.4 + phonological_processes * 0.35 + 
                      historical_phonology * 0.25)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (regular_changes + phonological_processes + historical_phonology) * 0.12)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="sound_change_knowledge",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_grammaticalization_understanding(self, response: str, cultural_context: CulturalContext,
                                                 test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate understanding of grammaticalization and morphosyntactic change."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count grammaticalization indicators
        processes = sum(1 for process in self.grammaticalization_patterns["grammaticalization_processes"]
                       if process in response_lower)
        paths = sum(1 for path in self.grammaticalization_patterns["grammaticalization_paths"]
                   if path in response_lower)
        morphosyntactic = sum(1 for change in self.grammaticalization_patterns["morphosyntactic_change"]
                             if change in response_lower)
        
        if processes > 0:
            evidence.append(f"Grammaticalization processes: {processes} instances")
            cultural_markers.append("grammaticalization_competence")
        
        if paths > 0:
            evidence.append(f"Grammaticalization paths: {paths} instances")
            cultural_markers.append("pathway_knowledge")
        
        if morphosyntactic > 0:
            evidence.append(f"Morphosyntactic change: {morphosyntactic} instances")
            cultural_markers.append("syntactic_evolution")
        
        total_score = (processes * 0.4 + paths * 0.35 + morphosyntactic * 0.25)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (processes + paths + morphosyntactic) * 0.12)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="grammaticalization_understanding",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_comparative_method_application(self, response: str, cultural_context: CulturalContext,
                                               test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate application of comparative method principles."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count comparative method indicators
        principles = sum(1 for principle in self.comparative_method_patterns["comparative_principles"]
                        if principle in response_lower)
        reconstruction = sum(1 for term in self.comparative_method_patterns["reconstruction_terminology"]
                            if term in response_lower)
        evidence_types = sum(1 for evidence_type in self.comparative_method_patterns["comparative_evidence"]
                            if evidence_type in response_lower)
        
        if principles > 0:
            evidence.append(f"Comparative principles: {principles} instances")
            cultural_markers.append("comparative_competence")
        
        if reconstruction > 0:
            evidence.append(f"Reconstruction terminology: {reconstruction} instances")
            cultural_markers.append("reconstruction_knowledge")
        
        if evidence_types > 0:
            evidence.append(f"Comparative evidence types: {evidence_types} instances")
            cultural_markers.append("evidence_evaluation")
        
        total_score = (principles * 0.4 + reconstruction * 0.35 + evidence_types * 0.25)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (principles + reconstruction + evidence_types) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="comparative_method_application",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_language_family_classification(self, response: str, cultural_context: CulturalContext,
                                               test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate knowledge of language families and classification."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count language family indicators
        major_families = sum(1 for family in self.language_family_patterns["major_families"]
                            if family in response_lower)
        subfamilies = sum(1 for subfamily in self.language_family_patterns["subfamily_knowledge"]
                         if subfamily in response_lower)
        classification_terms = sum(1 for term in self.language_family_patterns["classification_terminology"]
                                  if term in response_lower)
        
        if major_families > 0:
            evidence.append(f"Major language families: {major_families} instances")
            cultural_markers.append("family_classification_knowledge")
        
        if subfamilies > 0:
            evidence.append(f"Subfamily knowledge: {subfamilies} instances")
            cultural_markers.append("detailed_classification")
        
        if classification_terms > 0:
            evidence.append(f"Classification terminology: {classification_terms} instances")
            cultural_markers.append("classification_methodology")
        
        # Cultural context bonus
        heritage_family_bonus = 0.0
        cultural_languages = [variety.lower() for variety in cultural_context.linguistic_varieties]
        for lang in cultural_languages:
            if any(lang in family or family in lang 
                  for family_list in self.language_family_patterns.values() 
                  for family in family_list):
                heritage_family_bonus = 0.15
                cultural_markers.append("heritage_language_family")
                break
        
        total_score = (major_families * 0.4 + subfamilies * 0.35 + classification_terms * 0.25)
        
        score = min(1.0, (total_score * 0.1) + heritage_family_bonus)
        confidence = min(1.0, (major_families + subfamilies + classification_terms) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="language_family_classification",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_reconstruction_skills(self, response: str, cultural_context: CulturalContext,
                                      test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate reconstruction skills and proto-language analysis."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count reconstruction indicators
        reconstruction_terms = sum(1 for term in self.comparative_method_patterns["reconstruction_terminology"]
                                  if term in response_lower)
        historical_phonology = sum(1 for concept in self.sound_change_patterns["historical_phonology"]
                                  if concept in response_lower)
        comparative_evidence = sum(1 for evidence_type in self.comparative_method_patterns["comparative_evidence"]
                                  if evidence_type in response_lower)
        
        # Look for asterisk notation (proto-forms)
        asterisk_forms = len(re.findall(r'\*\w+', response))
        
        if reconstruction_terms > 0:
            evidence.append(f"Reconstruction terminology: {reconstruction_terms} instances")
            cultural_markers.append("reconstruction_competence")
        
        if historical_phonology > 0:
            evidence.append(f"Historical phonology: {historical_phonology} instances")
            cultural_markers.append("phonological_reconstruction")
        
        if comparative_evidence > 0:
            evidence.append(f"Comparative evidence: {comparative_evidence} instances")
            cultural_markers.append("evidence_based_reconstruction")
        
        if asterisk_forms > 0:
            evidence.append(f"Proto-form notation: {asterisk_forms} instances")
            cultural_markers.append("proto_form_notation")
        
        total_score = (reconstruction_terms * 0.3 + historical_phonology * 0.25 + 
                      comparative_evidence * 0.25 + asterisk_forms * 0.2)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (reconstruction_terms + historical_phonology + 
                              comparative_evidence + asterisk_forms) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="reconstruction_skills",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_contact_linguistics(self, response: str, cultural_context: CulturalContext,
                                    test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate understanding of language contact and borrowing."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count contact linguistics indicators
        borrowing_types = sum(1 for btype in self.contact_patterns["borrowing_types"]
                             if btype in response_lower)
        contact_phenomena = sum(1 for phenomenon in self.contact_patterns["contact_phenomena"]
                               if phenomenon in response_lower)
        borrowing_domains = sum(1 for domain in self.contact_patterns["borrowing_domains"]
                               if domain in response_lower)
        
        if borrowing_types > 0:
            evidence.append(f"Borrowing types: {borrowing_types} instances")
            cultural_markers.append("borrowing_competence")
        
        if contact_phenomena > 0:
            evidence.append(f"Contact phenomena: {contact_phenomena} instances")
            cultural_markers.append("contact_linguistics_knowledge")
        
        if borrowing_domains > 0:
            evidence.append(f"Borrowing domains: {borrowing_domains} instances")
            cultural_markers.append("domain_specific_borrowing")
        
        # Bonus for multilingual contexts
        multilingual_bonus = 0.0
        if len(cultural_context.linguistic_varieties) > 1:
            multilingual_bonus = 0.15
            cultural_markers.append("multilingual_contact_awareness")
        
        total_score = (borrowing_types * 0.4 + contact_phenomena * 0.35 + borrowing_domains * 0.25)
        
        score = min(1.0, (total_score * 0.12) + multilingual_bonus)
        confidence = min(1.0, (borrowing_types + contact_phenomena + borrowing_domains) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="contact_linguistics",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_diachronic_methodology(self, response: str, cultural_context: CulturalContext,
                                       test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate understanding of diachronic methods and approaches."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count methodology indicators
        dating_methods = sum(1 for method in self.diachronic_methods["dating_methods"]
                            if method in response_lower)
        analytical_approaches = sum(1 for approach in self.diachronic_methods["analytical_approaches"]
                                   if approach in response_lower)
        evidence_types = sum(1 for etype in self.diachronic_methods["evidence_types"]
                            if etype in response_lower)
        technical_concepts = sum(1 for concept in self.technical_concepts["methodological_terms"]
                                if concept in response_lower)
        
        if dating_methods > 0:
            evidence.append(f"Dating methods: {dating_methods} instances")
            cultural_markers.append("chronological_methods")
        
        if analytical_approaches > 0:
            evidence.append(f"Analytical approaches: {analytical_approaches} instances")
            cultural_markers.append("methodological_competence")
        
        if evidence_types > 0:
            evidence.append(f"Evidence types: {evidence_types} instances")
            cultural_markers.append("evidence_awareness")
        
        if technical_concepts > 0:
            evidence.append(f"Technical concepts: {technical_concepts} instances")
            cultural_markers.append("theoretical_knowledge")
        
        total_score = (dating_methods * 0.25 + analytical_approaches * 0.3 + 
                      evidence_types * 0.25 + technical_concepts * 0.2)
        
        score = min(1.0, total_score * 0.12)
        confidence = min(1.0, (dating_methods + analytical_approaches + 
                              evidence_types + technical_concepts) * 0.08)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="diachronic_methodology",
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
            
            # Higher relevance for markers aligned with linguistic heritage
            if marker_type in ["heritage", "linguistic"] and "linguistic_heritage" in cultural_context.knowledge_systems:
                relevance_score = 0.95
            elif marker_type in ["etymological", "historical"] and len(cultural_context.linguistic_varieties) > 0:
                relevance_score = 0.9
            elif marker_type in ["multilingual", "contact"] and len(cultural_context.linguistic_varieties) > 1:
                relevance_score = 0.9
            elif marker_type in ["comparative", "reconstruction", "family"]:
                relevance_score = 0.85
            elif marker_type in ["sound", "phonological", "grammaticalization"]:
                relevance_score = 0.8
            elif marker_type in ["methodological", "theoretical", "analytical"]:
                relevance_score = 0.75
            elif marker_type in ["borrowing", "contact", "evolution"]:
                relevance_score = 0.8
            
            total_relevance += relevance_score
            marker_count += 1
        
        return min(1.0, total_relevance / marker_count) if marker_count > 0 else 0.5