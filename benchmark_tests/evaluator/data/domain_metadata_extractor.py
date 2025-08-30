from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import re
import json
import requests
import time
from enum import Enum

from ..subjects.domain_evaluation_router import Domain, EvaluationType
from ..core.domain_evaluator_base import CulturalContext


class MetadataSource(Enum):
    """Sources of metadata information."""
    TEST_METADATA = "test_metadata"
    FILE_PATH = "file_path"
    CATEGORY_CONFIG = "category_config"
    CONTENT_ANALYSIS = "content_analysis"
    INFERRED = "inferred"


@dataclass
class MetadataExtraction:
    """Result of metadata extraction."""
    domain: Optional[Domain]
    evaluation_type: Optional[EvaluationType]
    cultural_context: CulturalContext
    confidence: float  # 0.0 to 1.0
    extraction_sources: Dict[MetadataSource, List[str]]
    processing_notes: List[str]


@dataclass
class CulturalValidationResult:
    """Result of cultural context validation."""
    validated_traditions: List[str]
    validated_groups: List[str]
    validated_knowledge_systems: List[str]
    validation_confidence: float  # 0.0 to 1.0
    wikipedia_matches: Dict[str, str]  # term -> wikipedia_url
    validation_flags: List[str]  # Issues found
    external_confirmations: Dict[str, Any]  # External API confirmations


@dataclass
class WikipediaSearchResult:
    """Result from Wikipedia search."""
    title: str
    url: str
    extract: str
    confidence: float
    categories: List[str]


class DomainMetadataExtractor:
    """Extracts domain and cultural information from test metadata and content."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.domain_patterns = self._initialize_domain_patterns()
        self.cultural_keywords = self._initialize_cultural_keywords()
        self.tradition_patterns = self._initialize_tradition_patterns()
    
    def _initialize_domain_patterns(self) -> Dict[Domain, Dict[str, List[str]]]:
        """Initialize patterns for domain detection."""
        return {
            Domain.CREATIVITY: {
                "keywords": [
                    "creative", "story", "narrative", "poem", "song", "performance", 
                    "artistic", "imagination", "original", "innovative", "expressive"
                ],
                "patterns": [
                    r"(?:create|write|compose)\s+(?:a|an|the)\s+(?:story|poem|song|narrative)",
                    r"(?:creative|artistic|imaginative)\s+(?:response|work|piece|expression)",
                    r"(?:storytelling|narrative|performance)\s+(?:tradition|style|approach)",
                    r"(?:cultural|traditional)\s+(?:storytelling|performance|creative)"
                ],
                "file_indicators": ["creativity", "creative", "narrative", "story", "performance"]
            },
            Domain.LANGUAGE: {
                "keywords": [
                    "language", "linguistic", "dialect", "register", "code-switching",
                    "multilingual", "translation", "communication", "discourse", "pragmatic",
                    "intercultural", "historical linguistics", "etymology", "implicature"
                ],
                "patterns": [
                    r"(?:language|linguistic)\s+(?:variation|competence|ability|skill)",
                    r"(?:dialect|register|variety)\s+(?:switching|variation|use)",
                    r"(?:code-switching|multilingual)\s+(?:competence|ability|communication)",
                    r"(?:pragmatic|discourse)\s+(?:competence|analysis|patterns)"
                ],
                "file_indicators": ["language", "linguistic", "multilingual", "dialect", "pragmatic", "intercultural", "historical"]
            },
            Domain.SOCIAL: {
                "keywords": [
                    "social", "cultural", "community", "relationship", "hierarchy", 
                    "etiquette", "politeness", "interaction", "dynamics", "appropriateness",
                    "conflict resolution", "mediation", "consensus", "navigation", "leadership"
                ],
                "patterns": [
                    r"(?:social|cultural)\s+(?:interaction|competence|appropriateness|dynamics)",
                    r"(?:community|relationship)\s+(?:dynamics|maintenance|building|patterns)",
                    r"(?:hierarchy|power)\s+(?:navigation|dynamics|relationships|structures)",
                    r"(?:etiquette|politeness)\s+(?:patterns|rules|conventions|norms)"
                ],
                "file_indicators": ["social", "community", "relationship", "etiquette", "hierarchy", "conflict", "leadership"]
            },
            Domain.REASONING: {
                "keywords": [
                    "reasoning", "logic", "analysis", "inference", "deduction", 
                    "problem-solving", "critical", "analytical", "systematic", "logical"
                ],
                "patterns": [
                    r"(?:reasoning|logic|logical)\s+(?:patterns|frameworks|systems|approaches)",
                    r"(?:analytical|critical)\s+(?:thinking|analysis|reasoning|approach)",
                    r"(?:problem-solving|inference)\s+(?:strategies|methods|approaches|patterns)",
                    r"(?:cultural|traditional)\s+(?:logic|reasoning|thinking|analysis)"
                ],
                "file_indicators": ["reasoning", "logic", "analysis", "inference", "critical"]
            },
            Domain.KNOWLEDGE: {
                "keywords": [
                    "knowledge", "information", "facts", "traditional", "indigenous", 
                    "cultural", "historical", "scientific", "educational", "wisdom"
                ],
                "patterns": [
                    r"(?:knowledge|wisdom)\s+(?:systems|traditions|frameworks|bases)",
                    r"(?:traditional|indigenous)\s+(?:knowledge|wisdom|science|practices)",
                    r"(?:cultural|historical)\s+(?:knowledge|information|understanding|context)",
                    r"(?:factual|scientific)\s+(?:knowledge|information|accuracy|understanding)"
                ],
                "file_indicators": ["knowledge", "traditional", "indigenous", "cultural", "historical"]
            },
            Domain.INTEGRATION: {
                "keywords": [
                    "integration", "cross-domain", "synthesis", "interdisciplinary", "holistic",
                    "comprehensive", "multi-domain", "combined", "unified", "convergent"
                ],
                "patterns": [
                    r"(?:cross|multi)-(?:domain|disciplinary|cultural)",
                    r"(?:integration|synthesis|convergence)\s+(?:of|across|between)",
                    r"(?:comprehensive|holistic|unified)\s+(?:approach|solution|analysis)",
                    r"(?:combine|merge|integrate)\s+(?:domains|approaches|perspectives)"
                ],
                "file_indicators": ["integration", "cross-domain", "multi-domain", "comprehensive", "synthesis"]
            }
        }
    
    def _initialize_cultural_keywords(self) -> Dict[str, List[str]]:
        """Initialize keywords for cultural context detection."""
        return {
            "traditions": [
                "griot", "dreamtime", "kamishibai", "oral tradition", "storytelling tradition",
                "folktale", "folklore", "legend", "myth", "traditional story", "cultural story",
                "ancestral story", "tribal story", "indigenous story", "community story",
                "ceremonial", "ritual", "spiritual", "sacred", "traditional practice",
                "conflict resolution", "mediation tradition", "consensus building",
                "traditional governance", "community leadership", "restorative justice",
                "cultural preservation", "hierarchical organization", "ubuntu philosophy"
            ],
            "cultural_groups": [
                "african", "west african", "aboriginal", "indigenous", "japanese", "chinese", 
                "native american", "first nations", "maori", "inuit", "celtic", "nordic",
                "mediterranean", "south american", "polynesian", "melanesian", "micronesian",
                "asian", "european", "middle eastern", "caribbean", "pacific islander"
            ],
            "knowledge_systems": [
                "traditional knowledge", "traditional medicine", "herbal knowledge", "ecological wisdom", 
                "agricultural practice", "indigenous knowledge", "ancestral wisdom", "cultural knowledge",
                "navigation system", "astronomical knowledge", "weather prediction", "seasonal cycles",
                "spiritual practice", "ceremonial knowledge", "ritual understanding", "sacred geography",
                "kinship system", "social organization", "governance system", "legal tradition",
                "linguistic heritage", "comparative linguistics", "intercultural competence",
                "pragmatic competence", "multilingual competence", "bilingual competence",
                "conflict resolution", "community empowerment", "cultural intelligence",
                "organizational hierarchy", "power dynamics", "consensus methods"
            ],
            "performance_aspects": [
                "oral performance", "storytelling", "singing", "chanting", "recitation", "drama",
                "dance", "gesture", "body language", "vocal technique", "rhythm", "timing",
                "audience interaction", "call and response", "participation", "improvisation",
                "theatrical", "dramatic", "expressive", "performative", "embodied",
                "mediation", "facilitation", "negotiation", "consensus building",
                "cross-cultural communication", "intercultural dialogue", "conflict mediation",
                "community organizing", "collaborative governance", "visionary leadership",
                "formal communication", "academic discourse", "professional communication",
                "politeness focus", "pragmatic communication", "multilingual discourse"
            ],
            "linguistic_varieties": [
                "dialect", "creole", "pidgin", "vernacular", "colloquial", "formal", "informal",
                "academic", "professional", "ceremonial", "ritual", "sacred language", "liturgical",
                "regional variety", "social variety", "ethnic variety", "generational variety",
                "code-switching", "multilingual", "bilingual", "polyglot", "lingua franca",
                "spanish", "english", "spanglish", "french", "mandarin", "arabic", "hindi",
                "germanic", "romance", "indo_european_languages", "standard_english",
                "academic_english", "formal_register", "multilingual_context", "community_discourse"
            ]
        }
    
    def _initialize_tradition_patterns(self) -> Dict[str, List[str]]:
        """Initialize specific tradition detection patterns."""
        return {
            "griot": [
                r"(?:griot|djeli|jali)\s+(?:tradition|storytelling|performance|culture)",
                r"(?:west\s+african|mali|senegal|guinea)\s+(?:storytelling|oral|tradition)",
                r"(?:mandinka|wolof|fulani)\s+(?:tradition|culture|storytelling)",
                r"(?:oral\s+history|genealogy|praise\s+singing)\s+(?:tradition|practice)"
            ],
            "dreamtime": [
                r"(?:dreamtime|dreaming|aboriginal)\s+(?:story|tradition|culture|knowledge)",
                r"(?:indigenous\s+australian|aboriginal\s+australian)\s+(?:tradition|storytelling|culture)",
                r"(?:songlines|country|land)\s+(?:connection|knowledge|tradition)",
                r"(?:ancestor|spirit)\s+(?:beings|stories|tradition|knowledge)"
            ],
            "kamishibai": [
                r"(?:kamishibai|paper\s+theater)\s+(?:storytelling|performance|tradition)",
                r"(?:japanese)\s+(?:storytelling|visual\s+storytelling|performance)\s+(?:tradition|art)",
                r"(?:visual\s+narrative|picture\s+story)\s+(?:performance|telling|tradition)",
                r"(?:street\s+performance|traveling\s+theater)\s+(?:tradition|storytelling)"
            ],
            "oral_tradition": [
                r"(?:oral\s+tradition|oral\s+culture|oral\s+literature)\s+(?:storytelling|performance)",
                r"(?:traditional\s+storytelling|folk\s+narrative|cultural\s+narrative)",
                r"(?:community\s+storytelling|collective\s+memory|cultural\s+transmission)",
                r"(?:intergenerational|ancestral)\s+(?:knowledge|wisdom|storytelling|tradition)"
            ]
        }
    
    def extract_metadata(self, test_metadata: Dict[str, Any], 
                        content: str = "") -> MetadataExtraction:
        """
        Extract domain and cultural metadata from test information.
        
        Args:
            test_metadata: Test metadata dictionary
            content: Optional content text for analysis
            
        Returns:
            MetadataExtraction with domain, evaluation type, and cultural context
        """
        extraction_sources = {source: [] for source in MetadataSource}
        processing_notes = []
        
        # Extract domain
        domain, domain_confidence, domain_sources = self._extract_domain(test_metadata, content)
        for source, evidence in domain_sources.items():
            extraction_sources[source].extend(evidence)
        
        # Extract evaluation type
        eval_type, eval_confidence, eval_sources = self._extract_evaluation_type(
            test_metadata, content, domain
        )
        for source, evidence in eval_sources.items():
            extraction_sources[source].extend(evidence)
        
        # Extract cultural context
        cultural_context, cultural_confidence, cultural_sources = self._extract_cultural_context(
            test_metadata, content
        )
        for source, evidence in cultural_sources.items():
            extraction_sources[source].extend(evidence)
        
        # Calculate overall confidence
        overall_confidence = (domain_confidence + eval_confidence + cultural_confidence) / 3.0
        
        # Generate processing notes
        processing_notes.extend([
            f"Domain detection confidence: {domain_confidence:.2f}",
            f"Evaluation type confidence: {eval_confidence:.2f}",
            f"Cultural context confidence: {cultural_confidence:.2f}"
        ])
        
        if domain is None:
            processing_notes.append("Warning: Could not determine domain")
        if eval_type is None:
            processing_notes.append("Warning: Could not determine evaluation type")
        if not any([cultural_context.traditions, cultural_context.cultural_groups, 
                   cultural_context.knowledge_systems]):
            processing_notes.append("Warning: Limited cultural context detected")
        
        return MetadataExtraction(
            domain=domain,
            evaluation_type=eval_type,
            cultural_context=cultural_context,
            confidence=overall_confidence,
            extraction_sources=extraction_sources,
            processing_notes=processing_notes
        )
    
    def _extract_domain(self, test_metadata: Dict[str, Any], 
                       content: str) -> Tuple[Optional[Domain], float, Dict[MetadataSource, List[str]]]:
        """Extract domain from metadata and content."""
        sources = {source: [] for source in MetadataSource}
        domain_scores = {domain: 0.0 for domain in Domain}
        
        # Check file path
        file_path = test_metadata.get('file_path', '')
        if file_path:
            for domain, patterns in self.domain_patterns.items():
                for indicator in patterns['file_indicators']:
                    if indicator in file_path.lower():
                        domain_scores[domain] += 0.8
                        sources[MetadataSource.FILE_PATH].append(f"Path contains '{indicator}'")
        
        # Check explicit domain in metadata
        if 'domain' in test_metadata:
            domain_name = test_metadata['domain'].lower()
            for domain in Domain:
                if domain.value.lower() == domain_name:
                    domain_scores[domain] += 1.0
                    sources[MetadataSource.TEST_METADATA].append(f"Explicit domain: {domain_name}")
        
        # Check category information
        category = test_metadata.get('category', '')
        test_id = test_metadata.get('test_id', '')
        if category or test_id:
            category_text = f"{category} {test_id}".lower()
            for domain, patterns in self.domain_patterns.items():
                for keyword in patterns['keywords']:
                    if keyword.lower() in category_text:
                        domain_scores[domain] += 0.6
                        sources[MetadataSource.CATEGORY_CONFIG].append(f"Category contains '{keyword}'")
        
        # Special handling for integration domain indicators
        domains_required = test_metadata.get('domains_required', [])
        integration_focus = test_metadata.get('integration_focus', '')
        if domains_required and len(domains_required) > 1:
            domain_scores[Domain.INTEGRATION] += 1.2  # Strong indicator
            sources[MetadataSource.TEST_METADATA].append(f"Multiple domains required: {domains_required}")
        
        if integration_focus:
            for keyword in self.domain_patterns[Domain.INTEGRATION]['keywords']:
                if keyword.lower() in integration_focus.lower():
                    domain_scores[Domain.INTEGRATION] += 0.8
                    sources[MetadataSource.TEST_METADATA].append(f"Integration focus contains '{keyword}'")
        
        # Analyze content if provided
        if content:
            content_lower = content.lower()
            for domain, patterns in self.domain_patterns.items():
                # Keyword matching
                for keyword in patterns['keywords']:
                    if keyword.lower() in content_lower:
                        domain_scores[domain] += 0.3
                        sources[MetadataSource.CONTENT_ANALYSIS].append(f"Content contains '{keyword}'")
                
                # Pattern matching
                for pattern in patterns['patterns']:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        domain_scores[domain] += 0.4 * len(matches)
                        sources[MetadataSource.CONTENT_ANALYSIS].append(f"Pattern match: {pattern}")
        
        # Determine best domain
        if not any(score > 0 for score in domain_scores.values()):
            return None, 0.0, sources
        
        best_domain = max(domain_scores, key=domain_scores.get)
        confidence = min(1.0, domain_scores[best_domain])
        
        return best_domain, confidence, sources
    
    def _extract_evaluation_type(self, test_metadata: Dict[str, Any], content: str,
                                domain: Optional[Domain]) -> Tuple[Optional[EvaluationType], float, Dict[MetadataSource, List[str]]]:
        """Extract evaluation type from metadata and content."""
        sources = {source: [] for source in MetadataSource}
        
        # Check explicit evaluation type
        if 'evaluation_type' in test_metadata:
            eval_type_name = test_metadata['evaluation_type'].lower()
            for eval_type in EvaluationType:
                if eval_type.value.lower() == eval_type_name:
                    sources[MetadataSource.TEST_METADATA].append(f"Explicit type: {eval_type_name}")
                    return eval_type, 1.0, sources
        
        # Infer from domain
        if domain:
            if domain == Domain.CREATIVITY:
                sources[MetadataSource.INFERRED].append("Inferred from creativity domain")
                return EvaluationType.CREATIVE_EXPRESSION, 0.8, sources
            elif domain == Domain.LANGUAGE:
                sources[MetadataSource.INFERRED].append("Inferred from language domain")
                return EvaluationType.LINGUISTIC_COMPETENCE, 0.8, sources
            elif domain == Domain.SOCIAL:
                sources[MetadataSource.INFERRED].append("Inferred from social domain")
                return EvaluationType.SOCIAL_CONTEXT, 0.8, sources
            elif domain == Domain.REASONING:
                sources[MetadataSource.INFERRED].append("Inferred from reasoning domain")
                return EvaluationType.ALTERNATIVE_LOGIC, 0.8, sources
            elif domain == Domain.KNOWLEDGE:
                sources[MetadataSource.INFERRED].append("Inferred from knowledge domain")
                return EvaluationType.TRADITIONAL_KNOWLEDGE, 0.8, sources
            elif domain == Domain.INTEGRATION:
                # For integration domain, check category for specific type
                category = test_metadata.get('category', '')
                if 'knowledge_reasoning' in category:
                    sources[MetadataSource.INFERRED].append("Inferred knowledge-reasoning integration")
                    return EvaluationType.KNOWLEDGE_REASONING_SYNTHESIS, 0.9, sources
                elif 'social_creativity' in category:
                    sources[MetadataSource.INFERRED].append("Inferred social-creativity integration")
                    return EvaluationType.SOCIAL_CREATIVE_SOLUTIONS, 0.9, sources
                elif 'language_knowledge' in category:
                    sources[MetadataSource.INFERRED].append("Inferred language-knowledge integration")
                    return EvaluationType.MULTILINGUAL_KNOWLEDGE_EXPRESSION, 0.9, sources
                elif 'reasoning_social' in category:
                    sources[MetadataSource.INFERRED].append("Inferred reasoning-social integration")
                    return EvaluationType.CULTURALLY_SENSITIVE_REASONING, 0.9, sources
                elif 'cross_domain' in category:
                    sources[MetadataSource.INFERRED].append("Inferred comprehensive integration")
                    return EvaluationType.COMPREHENSIVE_INTEGRATION, 0.9, sources
                else:
                    sources[MetadataSource.INFERRED].append("Default integration type")
                    return EvaluationType.COMPREHENSIVE_INTEGRATION, 0.7, sources
        
        # Content-based inference
        if content:
            content_lower = content.lower()
            if any(word in content_lower for word in ["create", "story", "poem", "narrative"]):
                sources[MetadataSource.CONTENT_ANALYSIS].append("Creative language detected")
                return EvaluationType.CREATIVE_EXPRESSION, 0.6, sources
            elif any(word in content_lower for word in ["dialect", "language", "register"]):
                sources[MetadataSource.CONTENT_ANALYSIS].append("Linguistic language detected")
                return EvaluationType.LINGUISTIC_COMPETENCE, 0.6, sources
            elif any(word in content_lower for word in ["social", "community", "relationship"]):
                sources[MetadataSource.CONTENT_ANALYSIS].append("Social language detected")
                return EvaluationType.SOCIAL_CONTEXT, 0.6, sources
        
        # Default fallback
        sources[MetadataSource.INFERRED].append("Default fallback to general reasoning")
        return EvaluationType.GENERAL_REASONING, 0.3, sources
    
    def _extract_cultural_context(self, test_metadata: Dict[str, Any], 
                                 content: str) -> Tuple[CulturalContext, float, Dict[MetadataSource, List[str]]]:
        """Extract cultural context information."""
        sources = {source: [] for source in MetadataSource}
        
        # Initialize context lists
        traditions = []
        knowledge_systems = []
        performance_aspects = []
        cultural_groups = []
        linguistic_varieties = []
        
        # Combine metadata and content for analysis
        analysis_text = ""
        
        # Add metadata text
        for key, value in test_metadata.items():
            if isinstance(value, str):
                analysis_text += f" {value}"
        
        # Add content
        if content:
            analysis_text += f" {content}"
        
        analysis_text = analysis_text.lower()
        
        confidence_scores = []
        
        # Extract traditions
        for tradition, patterns in self.tradition_patterns.items():
            for pattern in patterns:
                if re.search(pattern, analysis_text, re.IGNORECASE):
                    traditions.append(tradition)
                    confidence_scores.append(0.9)
                    sources[MetadataSource.CONTENT_ANALYSIS].append(f"Tradition detected: {tradition}")
                    break
        
        # Extract cultural keywords
        for category, keywords in self.cultural_keywords.items():
            detected_items = []
            for keyword in keywords:
                if keyword.lower() in analysis_text:
                    detected_items.append(keyword)
                    confidence_scores.append(0.7)
                    sources[MetadataSource.CONTENT_ANALYSIS].append(f"{category}: {keyword}")
            
            # Assign to appropriate context category
            if category == "traditions" and detected_items:
                traditions.extend(detected_items)
            elif category == "cultural_groups" and detected_items:
                cultural_groups.extend(detected_items)
            elif category == "knowledge_systems" and detected_items:
                knowledge_systems.extend(detected_items)
            elif category == "performance_aspects" and detected_items:
                performance_aspects.extend(detected_items)
            elif category == "linguistic_varieties" and detected_items:
                linguistic_varieties.extend(detected_items)
        
        # Remove duplicates
        traditions = list(set(traditions))
        knowledge_systems = list(set(knowledge_systems))
        performance_aspects = list(set(performance_aspects))
        cultural_groups = list(set(cultural_groups))
        linguistic_varieties = list(set(linguistic_varieties))
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        overall_confidence = min(1.0, overall_confidence)
        
        cultural_context = CulturalContext(
            traditions=traditions,
            knowledge_systems=knowledge_systems,
            performance_aspects=performance_aspects,
            cultural_groups=cultural_groups,
            linguistic_varieties=linguistic_varieties
        )
        
        if not any([traditions, knowledge_systems, performance_aspects, cultural_groups, linguistic_varieties]):
            sources[MetadataSource.INFERRED].append("No specific cultural context detected")
            overall_confidence = 0.1
        
        return cultural_context, overall_confidence, sources
    
    def enhance_cultural_context(self, cultural_context: CulturalContext, 
                                domain: Optional[Domain]) -> CulturalContext:
        """Enhance cultural context based on domain knowledge."""
        if not domain:
            return cultural_context
        
        # Domain-specific enhancements
        enhanced_traditions = list(cultural_context.traditions)
        enhanced_knowledge_systems = list(cultural_context.knowledge_systems)
        enhanced_performance_aspects = list(cultural_context.performance_aspects)
        
        if domain == Domain.CREATIVITY:
            if not enhanced_traditions:
                enhanced_traditions.extend(["oral_tradition", "storytelling_tradition"])
            if not enhanced_performance_aspects:
                enhanced_performance_aspects.extend(["storytelling", "oral_performance"])
        
        elif domain == Domain.LANGUAGE:
            if not cultural_context.linguistic_varieties:
                enhanced_linguistic_varieties = ["vernacular", "formal", "colloquial"]
            else:
                enhanced_linguistic_varieties = cultural_context.linguistic_varieties
            
            return CulturalContext(
                traditions=enhanced_traditions,
                knowledge_systems=enhanced_knowledge_systems,
                performance_aspects=enhanced_performance_aspects,
                cultural_groups=cultural_context.cultural_groups,
                linguistic_varieties=enhanced_linguistic_varieties
            )
        
        elif domain == Domain.SOCIAL:
            if not enhanced_knowledge_systems:
                enhanced_knowledge_systems.extend(["social_organization", "kinship_system"])
        
        elif domain == Domain.REASONING:
            if not enhanced_knowledge_systems:
                enhanced_knowledge_systems.extend(["logic_tradition", "reasoning_system"])
        
        elif domain == Domain.KNOWLEDGE:
            if not enhanced_knowledge_systems:
                enhanced_knowledge_systems.extend(["traditional_knowledge", "indigenous_knowledge"])
        
        return CulturalContext(
            traditions=enhanced_traditions,
            knowledge_systems=enhanced_knowledge_systems,
            performance_aspects=enhanced_performance_aspects,
            cultural_groups=cultural_context.cultural_groups,
            linguistic_varieties=cultural_context.linguistic_varieties
        )
    
    def validate_extraction(self, extraction: MetadataExtraction) -> List[str]:
        """Validate extraction results and return any issues."""
        issues = []
        
        if extraction.confidence < 0.5:
            issues.append("Low overall extraction confidence")
        
        if extraction.domain is None:
            issues.append("Could not determine domain")
        
        if extraction.evaluation_type is None:
            issues.append("Could not determine evaluation type")
        
        context = extraction.cultural_context
        if not any([context.traditions, context.cultural_groups, context.knowledge_systems]):
            issues.append("Very limited cultural context detected")
        
        # Check for conflicting information
        if (extraction.domain == Domain.CREATIVITY and 
            extraction.evaluation_type not in [EvaluationType.CREATIVE_EXPRESSION, EvaluationType.GENERAL_REASONING]):
            issues.append("Domain and evaluation type mismatch")
        
        return issues
    
    def validate_cultural_context(self, cultural_context: CulturalContext,
                                 use_wikipedia: bool = True,
                                 use_external_apis: bool = False) -> CulturalValidationResult:
        """
        Validate cultural context information against external sources.
        
        Args:
            cultural_context: Cultural context to validate
            use_wikipedia: Whether to use Wikipedia for validation
            use_external_apis: Whether to use external cultural knowledge APIs
            
        Returns:
            CulturalValidationResult with validation results
        """
        validated_traditions = []
        validated_groups = []
        validated_knowledge_systems = []
        wikipedia_matches = {}
        validation_flags = []
        external_confirmations = {}
        
        # Wikipedia validation
        if use_wikipedia:
            # Validate traditions
            for tradition in cultural_context.traditions:
                wiki_result = self._search_wikipedia(tradition)
                if wiki_result and wiki_result.confidence > 0.6:
                    validated_traditions.append(tradition)
                    wikipedia_matches[tradition] = wiki_result.url
                else:
                    validation_flags.append(f"Could not validate tradition: {tradition}")
            
            # Validate cultural groups
            for group in cultural_context.cultural_groups:
                wiki_result = self._search_wikipedia(f"{group} culture")
                if wiki_result and wiki_result.confidence > 0.5:
                    validated_groups.append(group)
                    wikipedia_matches[group] = wiki_result.url
                else:
                    validation_flags.append(f"Could not validate cultural group: {group}")
            
            # Validate knowledge systems
            for knowledge_sys in cultural_context.knowledge_systems:
                wiki_result = self._search_wikipedia(knowledge_sys)
                if wiki_result and wiki_result.confidence > 0.4:
                    validated_knowledge_systems.append(knowledge_sys)
                    wikipedia_matches[knowledge_sys] = wiki_result.url
                else:
                    validation_flags.append(f"Could not validate knowledge system: {knowledge_sys}")
        
        # External API validation (placeholder for future implementation)
        if use_external_apis:
            external_confirmations = self._validate_with_external_apis(cultural_context)
        
        # Calculate validation confidence
        total_items = (len(cultural_context.traditions) + 
                      len(cultural_context.cultural_groups) + 
                      len(cultural_context.knowledge_systems))
        validated_items = (len(validated_traditions) + 
                          len(validated_groups) + 
                          len(validated_knowledge_systems))
        
        validation_confidence = validated_items / total_items if total_items > 0 else 0.0
        
        return CulturalValidationResult(
            validated_traditions=validated_traditions,
            validated_groups=validated_groups,
            validated_knowledge_systems=validated_knowledge_systems,
            validation_confidence=validation_confidence,
            wikipedia_matches=wikipedia_matches,
            validation_flags=validation_flags,
            external_confirmations=external_confirmations
        )
    
    def _search_wikipedia(self, search_term: str, language: str = "en") -> Optional[WikipediaSearchResult]:
        """
        Search Wikipedia for cultural information.
        
        Args:
            search_term: Term to search for
            language: Wikipedia language code
            
        Returns:
            WikipediaSearchResult or None if no good match found
        """
        try:
            # Wikipedia API endpoint
            api_url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{search_term}"
            
            # Add rate limiting
            time.sleep(0.1)  # 100ms delay to be respectful to Wikipedia
            
            response = requests.get(api_url, timeout=5, headers={
                'User-Agent': 'BenchmarkTests/1.0 (Cultural Validation; educational use)'
            })
            
            if response.status_code == 200:
                data = response.json()
                
                # Calculate confidence based on response quality
                confidence = self._calculate_wikipedia_confidence(data, search_term)
                
                # Get categories (requires additional API call)
                categories = self._get_wikipedia_categories(data.get('title', ''), language)
                
                return WikipediaSearchResult(
                    title=data.get('title', ''),
                    url=data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    extract=data.get('extract', ''),
                    confidence=confidence,
                    categories=categories
                )
            
            elif response.status_code == 404:
                # Try alternative search
                return self._alternative_wikipedia_search(search_term, language)
                
        except requests.RequestException as e:
            # Network error - return None to indicate failure
            pass
        except Exception as e:
            # Other errors - return None
            pass
        
        return None
    
    def _calculate_wikipedia_confidence(self, wikipedia_data: Dict[str, Any], search_term: str) -> float:
        """Calculate confidence score for Wikipedia match."""
        confidence = 0.0
        
        title = wikipedia_data.get('title', '').lower()
        extract = wikipedia_data.get('extract', '').lower()
        search_lower = search_term.lower()
        
        # Title match
        if search_lower in title:
            confidence += 0.4
        elif any(word in title for word in search_lower.split()):
            confidence += 0.2
        
        # Extract relevance
        if search_lower in extract:
            confidence += 0.3
        elif any(word in extract for word in search_lower.split() if len(word) > 3):
            confidence += 0.2
        
        # Page type indicators
        if any(indicator in extract.lower() for indicator in [
            'culture', 'tradition', 'cultural', 'ethnic', 'indigenous', 'folk'
        ]):
            confidence += 0.3
        
        # Disambiguation pages get lower confidence
        if 'disambiguation' in title:
            confidence *= 0.5
        
        return min(1.0, confidence)
    
    def _get_wikipedia_categories(self, title: str, language: str = "en") -> List[str]:
        """Get Wikipedia page categories."""
        try:
            categories_url = f"https://{language}.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'categories',
                'cllimit': 10
            }
            
            time.sleep(0.1)  # Rate limiting
            
            response = requests.get(categories_url, params=params, timeout=5, headers={
                'User-Agent': 'BenchmarkTests/1.0 (Cultural Validation; educational use)'
            })
            
            if response.status_code == 200:
                data = response.json()
                pages = data.get('query', {}).get('pages', {})
                
                categories = []
                for page_data in pages.values():
                    page_categories = page_data.get('categories', [])
                    for cat in page_categories:
                        cat_title = cat.get('title', '')
                        if cat_title.startswith('Category:'):
                            categories.append(cat_title[9:])  # Remove 'Category:' prefix
                
                return categories
                
        except Exception as e:
            pass
        
        return []
    
    def _alternative_wikipedia_search(self, search_term: str, language: str = "en") -> Optional[WikipediaSearchResult]:
        """Try alternative Wikipedia search strategies."""
        try:
            # Use Wikipedia search API
            search_url = f"https://{language}.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': search_term,
                'srlimit': 1
            }
            
            time.sleep(0.1)  # Rate limiting
            
            response = requests.get(search_url, params=params, timeout=5, headers={
                'User-Agent': 'BenchmarkTests/1.0 (Cultural Validation; educational use)'
            })
            
            if response.status_code == 200:
                data = response.json()
                search_results = data.get('query', {}).get('search', [])
                
                if search_results:
                    first_result = search_results[0]
                    title = first_result.get('title', '')
                    
                    # Get the summary for this title
                    return self._search_wikipedia(title, language)
            
        except Exception as e:
            pass
        
        return None
    
    def _validate_with_external_apis(self, cultural_context: CulturalContext) -> Dict[str, Any]:
        """
        Validate cultural context with external APIs.
        
        This is a placeholder for future implementation with cultural knowledge APIs.
        """
        confirmations = {}
        
        # TODO: Implement validation with external cultural APIs
        # - UNESCO databases
        # - Academic cultural corpora
        # - Cultural institution APIs
        
        confirmations['note'] = "External API validation not yet implemented"
        
        return confirmations
    
    def cross_reference_cultural_claims(self, 
                                       evaluation_claims: List[str],
                                       cultural_context: CulturalContext) -> Dict[str, Any]:
        """
        Cross-reference evaluation claims against cultural context.
        
        Args:
            evaluation_claims: Claims made by evaluators about cultural authenticity
            cultural_context: Cultural context information
            
        Returns:
            Dict with cross-reference results and validation flags
        """
        cross_reference_results = {
            'validated_claims': [],
            'questionable_claims': [],
            'unsupported_claims': [],
            'wikipedia_evidence': {},
            'confidence_scores': {}
        }
        
        for claim in evaluation_claims:
            claim_lower = claim.lower()
            
            # Check if claim relates to known cultural elements
            relates_to_traditions = any(tradition.lower() in claim_lower 
                                      for tradition in cultural_context.traditions)
            relates_to_groups = any(group.lower() in claim_lower 
                                  for group in cultural_context.cultural_groups)
            relates_to_knowledge = any(knowledge.lower() in claim_lower 
                                     for knowledge in cultural_context.knowledge_systems)
            
            if relates_to_traditions or relates_to_groups or relates_to_knowledge:
                # Try to validate with Wikipedia
                wiki_result = self._search_wikipedia(claim)
                
                if wiki_result and wiki_result.confidence > 0.6:
                    cross_reference_results['validated_claims'].append(claim)
                    cross_reference_results['wikipedia_evidence'][claim] = wiki_result.url
                    cross_reference_results['confidence_scores'][claim] = wiki_result.confidence
                elif wiki_result and wiki_result.confidence > 0.3:
                    cross_reference_results['questionable_claims'].append(claim)
                    cross_reference_results['wikipedia_evidence'][claim] = wiki_result.url
                    cross_reference_results['confidence_scores'][claim] = wiki_result.confidence
                else:
                    cross_reference_results['unsupported_claims'].append(claim)
                    cross_reference_results['confidence_scores'][claim] = 0.0
            else:
                cross_reference_results['unsupported_claims'].append(claim)
                cross_reference_results['confidence_scores'][claim] = 0.0
        
        return cross_reference_results
    
    def enhance_cultural_context_with_validation(self, 
                                                cultural_context: CulturalContext,
                                                validation_result: CulturalValidationResult) -> CulturalContext:
        """
        Enhance cultural context using validation results.
        
        Args:
            cultural_context: Original cultural context
            validation_result: Results from cultural validation
            
        Returns:
            Enhanced CulturalContext with validated information
        """
        # Only keep validated items with high confidence
        enhanced_traditions = []
        enhanced_groups = []
        enhanced_knowledge_systems = []
        
        # Add validated traditions
        for tradition in cultural_context.traditions:
            if tradition in validation_result.validated_traditions:
                enhanced_traditions.append(tradition)
        
        # Add validated cultural groups
        for group in cultural_context.cultural_groups:
            if group in validation_result.validated_groups:
                enhanced_groups.append(group)
        
        # Add validated knowledge systems
        for knowledge_sys in cultural_context.knowledge_systems:
            if knowledge_sys in validation_result.validated_knowledge_systems:
                enhanced_knowledge_systems.append(knowledge_sys)
        
        # Keep other aspects unchanged as they're harder to validate
        return CulturalContext(
            traditions=enhanced_traditions,
            knowledge_systems=enhanced_knowledge_systems,
            performance_aspects=cultural_context.performance_aspects,  # Keep as-is
            cultural_groups=enhanced_groups,
            linguistic_varieties=cultural_context.linguistic_varieties  # Keep as-is
        )