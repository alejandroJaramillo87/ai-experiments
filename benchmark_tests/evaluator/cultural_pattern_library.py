from typing import Dict, List, Any, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
from collections import defaultdict


class PatternType(Enum):
    """Types of cultural patterns that can be detected."""
    STORYTELLING_STRUCTURE = "storytelling_structure"
    PERFORMANCE_MARKER = "performance_marker"
    CULTURAL_VALUE = "cultural_value"
    RHYTHMIC_PATTERN = "rhythmic_pattern"
    KNOWLEDGE_INTEGRATION = "knowledge_integration"
    STORYTELLING_ETHICS = "storytelling_ethics"
    NARRATIVE_STRUCTURE = "narrative_structure"
    VISUAL_NARRATIVE = "visual_narrative"
    EDUCATIONAL_PURPOSE = "educational_purpose"
    PERFORMANCE_STYLE = "performance_style"
    MEMORY_AID = "memory_aid"
    AUDIENCE_INTERACTION = "audience_interaction"


@dataclass
class CulturalPattern:
    """Represents a detected cultural pattern."""
    pattern_type: PatternType
    tradition: str
    pattern_name: str
    confidence: float  # 0.0 to 1.0
    evidence: List[str]  # Text segments that support this pattern
    cultural_significance: str  # Description of cultural meaning
    authenticity_indicators: List[str]  # Features that support authenticity


@dataclass
class PatternLibraryEntry:
    """Entry in the cultural pattern library."""
    pattern_name: str
    pattern_type: PatternType
    tradition: str
    detection_keywords: List[str]
    detection_patterns: List[str]  # Regex patterns
    cultural_context: str
    authenticity_markers: List[str]
    appropriation_warnings: List[str]  # Signs of potential appropriation
    sacred_boundaries: List[str]  # Sacred elements to respect


class CulturalPatternLibrary:
    """Library of cultural patterns for detection and analysis."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.patterns: Dict[str, List[PatternLibraryEntry]] = {}
        self.tradition_patterns: Dict[str, List[PatternLibraryEntry]] = defaultdict(list)
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize the cultural pattern library with predefined patterns."""
        self._add_griot_patterns()
        self._add_dreamtime_patterns()
        self._add_kamishibai_patterns()
        self._add_oral_performance_patterns()
        self._add_general_cultural_patterns()
        self._build_indexes()
    
    def _add_griot_patterns(self):
        """Add West African griot storytelling patterns."""
        griot_patterns = [
            PatternLibraryEntry(
                pattern_name="call_response",
                pattern_type=PatternType.STORYTELLING_STRUCTURE,
                tradition="griot",
                detection_keywords=["listen", "hear", "respond", "answer", "unison", "call"],
                detection_patterns=[
                    r"(?:listen|hear)\s+(?:well|now|friends)",
                    r"(?:respond|answer)\s+(?:in\s+)?(?:unison|together)",
                    r"(?:say|tell)\s+(?:it|us|all)",
                    r"(?:repeat|sing)\s+(?:after|with)\s+me",
                    r"(?:audience|people|community)\s+(?:responds|answers|calls)",
                    r"(?:call|voice)\s+(?:and|or)\s+(?:response|answer)"
                ],
                cultural_context="Interactive storytelling where audience participates",
                authenticity_markers=["community engagement", "rhythmic speech", "repeated phrases"],
                appropriation_warnings=["superficial call-response", "missing cultural context"],
                sacred_boundaries=["spiritual invocations", "ancestor calling"]
            ),
            PatternLibraryEntry(
                pattern_name="moral_embedding",
                pattern_type=PatternType.STORYTELLING_STRUCTURE,
                tradition="griot",
                detection_keywords=["wisdom", "lesson", "teaches us", "learn from", "remember"],
                detection_patterns=[
                    r"(?:wisdom|lesson)\s+(?:of|from)\s+(?:ancestors|elders)",
                    r"(?:teaches|shows)\s+us\s+(?:that|how)",
                    r"(?:remember|learn)\s+(?:this|from)",
                    r"(?:moral|lesson)\s+(?:of|in)\s+(?:this|the)\s+story"
                ],
                cultural_context="Stories embed community values and moral teachings",
                authenticity_markers=["ancestral wisdom", "community values", "moral lessons"],
                appropriation_warnings=["simplistic morals", "missing cultural depth"],
                sacred_boundaries=["sacred teachings", "initiation stories"]
            ),
            PatternLibraryEntry(
                pattern_name="historical_weaving",
                pattern_type=PatternType.STORYTELLING_STRUCTURE,
                tradition="griot",
                detection_keywords=["generations", "ancestors", "history", "lineage", "tradition"],
                detection_patterns=[
                    r"(?:generations|centuries)\s+(?:ago|past)",
                    r"(?:ancestors|forefathers)\s+(?:did|told|taught)",
                    r"(?:history|tradition)\s+(?:tells|shows|says)",
                    r"(?:lineage|family)\s+(?:of|from)"
                ],
                cultural_context="Historical events woven into storytelling",
                authenticity_markers=["historical accuracy", "genealogical knowledge", "cultural continuity"],
                appropriation_warnings=["inaccurate history", "misrepresented lineages"],
                sacred_boundaries=["family secrets", "sacred histories"]
            )
        ]
        
        self.patterns["griot"] = griot_patterns
    
    def _add_dreamtime_patterns(self):
        """Add Aboriginal Dreamtime storytelling patterns."""
        dreamtime_patterns = [
            PatternLibraryEntry(
                pattern_name="landscape_embodiment",
                pattern_type=PatternType.NARRATIVE_STRUCTURE,
                tradition="dreamtime",
                detection_keywords=["land", "country", "rock", "water", "tree", "mountain"],
                detection_patterns=[
                    r"(?:land|country)\s+(?:speaks|tells|remembers)",
                    r"(?:rock|stone|mountain)\s+(?:became|transformed|formed)",
                    r"(?:water|river|spring)\s+(?:flows|carries|holds)",
                    r"(?:tree|plant)\s+(?:grows|stands|watches)"
                ],
                cultural_context="Landscape features as active participants in stories",
                authenticity_markers=["geographical specificity", "ecological knowledge", "spiritual connection"],
                appropriation_warnings=["generic landscapes", "missing ecological context"],
                sacred_boundaries=["sacred sites", "men's/women's sites", "initiation sites"]
            ),
            PatternLibraryEntry(
                pattern_name="ancestor_presence",
                pattern_type=PatternType.NARRATIVE_STRUCTURE,
                tradition="dreamtime",
                detection_keywords=["ancestor", "spirit", "dreaming", "creation", "eternal"],
                detection_patterns=[
                    r"(?:ancestor|spirit)\s+(?:beings|spirits)\s+(?:walked|traveled|created)",
                    r"(?:dreaming|creation)\s+(?:time|beings|spirits)",
                    r"(?:eternal|forever)\s+(?:present|here|watching)",
                    r"(?:spirit|ancestor)\s+(?:lives|dwells|remains)\s+(?:in|within)"
                ],
                cultural_context="Ancestral beings as continuing presence in landscape",
                authenticity_markers=["spiritual continuity", "ancestral respect", "ongoing presence"],
                appropriation_warnings=["romanticized spirits", "disconnected from culture"],
                sacred_boundaries=["sacred ancestor names", "restricted knowledge", "gender-specific stories"]
            ),
            PatternLibraryEntry(
                pattern_name="cyclical_time",
                pattern_type=PatternType.NARRATIVE_STRUCTURE,
                tradition="dreamtime",
                detection_keywords=["cycle", "return", "eternal", "always", "beginning"],
                detection_patterns=[
                    r"(?:cycle|circle)\s+(?:of|continues|returns)",
                    r"(?:eternal|forever|always)\s+(?:present|here|happening)",
                    r"(?:beginning|end)\s+(?:and|is|becomes)\s+(?:end|beginning)",
                    r"(?:time|story)\s+(?:circles|returns|cycles)"
                ],
                cultural_context="Non-linear time conception in storytelling",
                authenticity_markers=["cyclical understanding", "eternal present", "continuous creation"],
                appropriation_warnings=["linear time concepts", "finite narratives"],
                sacred_boundaries=["creation secrets", "time/space concepts"]
            )
        ]
        
        self.patterns["dreamtime"] = dreamtime_patterns
    
    def _add_kamishibai_patterns(self):
        """Add Japanese kamishibai storytelling patterns."""
        kamishibai_patterns = [
            PatternLibraryEntry(
                pattern_name="image_text_harmony",
                pattern_type=PatternType.VISUAL_NARRATIVE,
                tradition="kamishibai",
                detection_keywords=["picture", "image", "shows", "see", "visual"],
                detection_patterns=[
                    r"(?:picture|image)\s+(?:shows|reveals|depicts)",
                    r"(?:see|look)\s+(?:how|at|the)",
                    r"(?:visual|scene)\s+(?:tells|shows|depicts)",
                    r"(?:drawing|illustration)\s+(?:captures|shows)"
                ],
                cultural_context="Visual and textual elements work together",
                authenticity_markers=["visual-text integration", "scene description", "visual storytelling"],
                appropriation_warnings=["text-only approach", "missing visual elements"],
                sacred_boundaries=["religious imagery", "cultural symbols"]
            ),
            PatternLibraryEntry(
                pattern_name="dramatic_pacing",
                pattern_type=PatternType.PERFORMANCE_STYLE,
                tradition="kamishibai",
                detection_keywords=["pause", "moment", "suddenly", "slowly", "dramatic"],
                detection_patterns=[
                    r"(?:pause|wait)\s+(?:for|a)\s+(?:moment|beat)",
                    r"(?:suddenly|quickly|slowly)\s+(?:the|something|everything)",
                    r"(?:dramatic|tense)\s+(?:moment|pause|silence)",
                    r"(?:timing|pace)\s+(?:of|in)\s+(?:the|this)\s+story"
                ],
                cultural_context="Controlled pacing for dramatic effect",
                authenticity_markers=["controlled timing", "dramatic pauses", "pacing awareness"],
                appropriation_warnings=["rushed delivery", "missing dramatic elements"],
                sacred_boundaries=["ritual timing", "ceremonial pacing"]
            ),
            PatternLibraryEntry(
                pattern_name="audience_participation",
                pattern_type=PatternType.AUDIENCE_INTERACTION,
                tradition="kamishibai",
                detection_keywords=["children", "audience", "clap", "cheer", "participate"],
                detection_patterns=[
                    r"(?:children|audience)\s+(?:clap|cheer|laugh|gasp)",
                    r"(?:participate|join)\s+(?:in|with)\s+(?:the|this)",
                    r"(?:everyone|all)\s+(?:says|sings|calls)",
                    r"(?:response|reaction)\s+(?:from|of)\s+(?:audience|children)"
                ],
                cultural_context="Interactive storytelling with audience engagement",
                authenticity_markers=["audience engagement", "interactive elements", "participatory storytelling"],
                appropriation_warnings=["passive audience", "missing interaction"],
                sacred_boundaries=["adult-only stories", "restricted participation"]
            )
        ]
        
        self.patterns["kamishibai"] = kamishibai_patterns
    
    def _add_oral_performance_patterns(self):
        """Add general oral performance tradition patterns."""
        oral_patterns = [
            PatternLibraryEntry(
                pattern_name="meter_consistency",
                pattern_type=PatternType.RHYTHMIC_PATTERN,
                tradition="oral_performance",
                detection_keywords=["rhythm", "beat", "meter", "flow", "cadence"],
                detection_patterns=[
                    r"(?:rhythm|beat|meter)\s+(?:of|in)\s+(?:the|this|words)",
                    r"(?:flow|cadence)\s+(?:of|in)\s+(?:speech|language|words)",
                    r"(?:steady|consistent)\s+(?:rhythm|beat|flow)",
                    r"(?:musical|rhythmic)\s+(?:quality|pattern|structure)"
                ],
                cultural_context="Consistent rhythmic patterns in oral delivery",
                authenticity_markers=["rhythmic consistency", "musical quality", "oral flow"],
                appropriation_warnings=["irregular rhythm", "missing musical quality"],
                sacred_boundaries=["ritual rhythms", "ceremonial meters"]
            ),
            PatternLibraryEntry(
                pattern_name="repetition",
                pattern_type=PatternType.MEMORY_AID,
                tradition="oral_performance",
                detection_keywords=["repeat", "repetition", "again", "once more", "refrain", "echo"],
                detection_patterns=[
                    r"(?:repeat|again|once\s+more)",
                    r"(?:through|via|using)\s+(?:repetition|repeat)",
                    r"(?:refrain|chorus)\s+(?:of|that|goes)",
                    r"(?:echo|echoes)\s+(?:through|in|across)",
                    r"(?:same|repeated)\s+(?:phrase|words|line)"
                ],
                cultural_context="Repetition as memory aid and emphasis",
                authenticity_markers=["meaningful repetition", "memory structure", "emphasis patterns"],
                appropriation_warnings=["empty repetition", "meaningless echoes"],
                sacred_boundaries=["sacred phrases", "ritual repetitions"]
            ),
            PatternLibraryEntry(
                pattern_name="collective_memory",
                pattern_type=PatternType.AUDIENCE_INTERACTION,
                tradition="oral_performance",
                detection_keywords=["remember", "memory", "collective", "together", "community"],
                detection_patterns=[
                    r"(?:remember|recall)\s+(?:together|as\s+one|collectively)",
                    r"(?:collective|shared)\s+(?:memory|remembrance|knowledge)",
                    r"(?:community|group)\s+(?:remembers|holds|keeps)",
                    r"(?:together|united)\s+(?:in|through)\s+(?:memory|story)"
                ],
                cultural_context="Shared community memory through storytelling",
                authenticity_markers=["community knowledge", "shared memory", "collective participation"],
                appropriation_warnings=["individual focus", "missing community aspect"],
                sacred_boundaries=["community secrets", "restricted memories"]
            )
        ]
        
        self.patterns["oral_performance"] = oral_patterns
    
    def _add_general_cultural_patterns(self):
        """Add general cultural patterns applicable across traditions."""
        general_patterns = [
            PatternLibraryEntry(
                pattern_name="cultural_respect",
                pattern_type=PatternType.CULTURAL_VALUE,
                tradition="general",
                detection_keywords=["respect", "honor", "reverence", "sacred", "traditional"],
                detection_patterns=[
                    r"(?:respect|honor|reverence)\s+(?:for|to|the)",
                    r"(?:sacred|holy|blessed)\s+(?:tradition|knowledge|practice)",
                    r"(?:traditional|ancestral)\s+(?:ways|wisdom|knowledge)",
                    r"(?:cultural|community)\s+(?:values|traditions|practices)"
                ],
                cultural_context="Demonstration of cultural respect and awareness",
                authenticity_markers=["respectful language", "cultural awareness", "appropriate reverence"],
                appropriation_warnings=["superficial respect", "tokenistic language"],
                sacred_boundaries=["sacred terminology", "restricted concepts"]
            ),
            PatternLibraryEntry(
                pattern_name="community_focus",
                pattern_type=PatternType.CULTURAL_VALUE,
                tradition="general",
                detection_keywords=["community", "together", "collective", "shared", "unity"],
                detection_patterns=[
                    r"(?:community|collective|group)\s+(?:strength|wisdom|knowledge)",
                    r"(?:together|united|shared)\s+(?:we|in|through)",
                    r"(?:collective|community)\s+(?:responsibility|effort|action)",
                    r"(?:unity|solidarity)\s+(?:in|through|of)"
                ],
                cultural_context="Emphasis on community over individual",
                authenticity_markers=["community emphasis", "collective focus", "shared responsibility"],
                appropriation_warnings=["individualistic focus", "missing community aspect"],
                sacred_boundaries=["community secrets", "collective sacred knowledge"]
            )
        ]
        
        self.patterns["general"] = general_patterns
    
    def _build_indexes(self):
        """Build indexes for efficient pattern lookup."""
        for tradition, pattern_list in self.patterns.items():
            for pattern in pattern_list:
                self.tradition_patterns[tradition].append(pattern)
    
    def detect_patterns(self, text: str, traditions: List[str] = None) -> List[CulturalPattern]:
        """
        Detect cultural patterns in text.
        
        Args:
            text: Text to analyze
            traditions: Specific traditions to look for (default: all)
            
        Returns:
            List of detected cultural patterns
        """
        if traditions is None:
            traditions = list(self.patterns.keys())
        
        detected_patterns = []
        text_lower = text.lower()
        
        for tradition in traditions:
            if tradition not in self.patterns:
                continue
            
            for pattern_entry in self.patterns[tradition]:
                confidence, evidence = self._evaluate_pattern(text, text_lower, pattern_entry)
                
                if confidence > 0.3:  # Threshold for pattern detection
                    cultural_pattern = CulturalPattern(
                        pattern_type=pattern_entry.pattern_type,
                        tradition=tradition,
                        pattern_name=pattern_entry.pattern_name,
                        confidence=confidence,
                        evidence=evidence,
                        cultural_significance=pattern_entry.cultural_context,
                        authenticity_indicators=self._assess_authenticity(text, pattern_entry)
                    )
                    detected_patterns.append(cultural_pattern)
        
        return detected_patterns
    
    def _evaluate_pattern(self, text: str, text_lower: str, 
                         pattern_entry: PatternLibraryEntry) -> Tuple[float, List[str]]:
        """Evaluate if a pattern is present in text."""
        confidence = 0.0
        evidence = []
        
        # Keyword matching
        keyword_matches = 0
        for keyword in pattern_entry.detection_keywords:
            if keyword.lower() in text_lower:
                keyword_matches += 1
                evidence.append(f"Keyword: '{keyword}'")
        
        keyword_score = min(1.0, keyword_matches / len(pattern_entry.detection_keywords))
        
        # Pattern matching
        pattern_matches = 0
        for pattern in pattern_entry.detection_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                pattern_matches += len(matches)
                for match in matches[:2]:  # Limit evidence
                    if isinstance(match, tuple):
                        evidence.append(f"Pattern match: {' '.join(match)}")
                    else:
                        evidence.append(f"Pattern match: {match}")
        
        pattern_score = min(1.0, pattern_matches / len(pattern_entry.detection_patterns))
        
        # Combine scores
        confidence = (keyword_score * 0.4) + (pattern_score * 0.6)
        
        return confidence, evidence
    
    def _assess_authenticity(self, text: str, pattern_entry: PatternLibraryEntry) -> List[str]:
        """Assess authenticity indicators for detected pattern."""
        authenticity_indicators = []
        text_lower = text.lower()
        
        for marker in pattern_entry.authenticity_markers:
            if any(word in text_lower for word in marker.lower().split()):
                authenticity_indicators.append(marker)
        
        return authenticity_indicators
    
    def get_pattern_by_name(self, pattern_name: str, tradition: str = None) -> Optional[PatternLibraryEntry]:
        """Get pattern entry by name and optional tradition."""
        if tradition and tradition in self.patterns:
            for pattern in self.patterns[tradition]:
                if pattern.pattern_name == pattern_name:
                    return pattern
        else:
            # Search all traditions
            for tradition_patterns in self.patterns.values():
                for pattern in tradition_patterns:
                    if pattern.pattern_name == pattern_name:
                        return pattern
        return None
    
    def get_traditions(self) -> List[str]:
        """Get list of available traditions."""
        return list(self.patterns.keys())
    
    def get_patterns_for_tradition(self, tradition: str) -> List[PatternLibraryEntry]:
        """Get all patterns for a specific tradition."""
        return self.patterns.get(tradition, [])
    
    def check_appropriation_warnings(self, detected_patterns: List[CulturalPattern], 
                                   text: str) -> List[str]:
        """Check for potential cultural appropriation indicators."""
        warnings = []
        text_lower = text.lower()
        
        for pattern in detected_patterns:
            pattern_entry = self.get_pattern_by_name(pattern.pattern_name, pattern.tradition)
            if pattern_entry:
                for warning in pattern_entry.appropriation_warnings:
                    # Simple check - could be more sophisticated
                    warning_words = warning.lower().split()
                    if any(word in text_lower for word in warning_words):
                        warnings.append(f"{pattern.tradition}:{pattern.pattern_name} - {warning}")
        
        return warnings
    
    def check_sacred_boundaries(self, detected_patterns: List[CulturalPattern], 
                               text: str) -> List[str]:
        """Check for potential sacred boundary violations."""
        violations = []
        text_lower = text.lower()
        
        for pattern in detected_patterns:
            pattern_entry = self.get_pattern_by_name(pattern.pattern_name, pattern.tradition)
            if pattern_entry:
                for boundary in pattern_entry.sacred_boundaries:
                    boundary_words = boundary.lower().split()
                    if any(word in text_lower for word in boundary_words):
                        violations.append(f"{pattern.tradition}:{pattern.pattern_name} - {boundary}")
        
        return violations
    
    def analyze_cultural_competence(self, detected_patterns: List[CulturalPattern]) -> Dict[str, Any]:
        """Analyze cultural competence based on detected patterns."""
        if not detected_patterns:
            return {
                'overall_competence': 0.0,
                'tradition_coverage': {},
                'pattern_diversity': 0.0,
                'authenticity_score': 0.0,
                'recommendations': ['No cultural patterns detected']
            }
        
        # Calculate tradition coverage
        tradition_coverage = defaultdict(float)
        for pattern in detected_patterns:
            tradition_coverage[pattern.tradition] += pattern.confidence
        
        # Normalize by number of patterns per tradition
        for tradition in tradition_coverage:
            total_patterns = len(self.patterns.get(tradition, []))
            if total_patterns > 0:
                tradition_coverage[tradition] /= total_patterns
        
        # Calculate pattern diversity
        pattern_types = set(pattern.pattern_type for pattern in detected_patterns)
        pattern_diversity = len(pattern_types) / len(PatternType)
        
        # Calculate authenticity score
        authenticity_scores = []
        for pattern in detected_patterns:
            if pattern.authenticity_indicators:
                authenticity_scores.append(pattern.confidence)
        authenticity_score = sum(authenticity_scores) / len(authenticity_scores) if authenticity_scores else 0.0
        
        # Overall competence
        overall_competence = (
            sum(tradition_coverage.values()) * 0.4 +
            pattern_diversity * 0.3 +
            authenticity_score * 0.3
        )
        
        # Recommendations
        recommendations = []
        if pattern_diversity < 0.5:
            recommendations.append("Increase variety of cultural pattern types")
        if authenticity_score < 0.6:
            recommendations.append("Improve cultural authenticity markers")
        if len(tradition_coverage) < 2:
            recommendations.append("Consider incorporating multiple cultural traditions")
        
        return {
            'overall_competence': overall_competence,
            'tradition_coverage': dict(tradition_coverage),
            'pattern_diversity': pattern_diversity,
            'authenticity_score': authenticity_score,
            'recommendations': recommendations
        }