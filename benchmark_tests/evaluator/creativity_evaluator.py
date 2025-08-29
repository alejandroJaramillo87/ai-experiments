from typing import Dict, List, Any, Optional, Tuple
import re
import statistics
from collections import Counter, defaultdict

from .domain_evaluator_base import (
    MultiDimensionalEvaluator, EvaluationDimension, DomainEvaluationResult, CulturalContext
)
from .cultural_pattern_library import CulturalPatternLibrary, CulturalPattern, PatternType


class CreativityEvaluator(MultiDimensionalEvaluator):
    """Evaluates creative expression with cultural authenticity and competence."""
    
    VERSION = "1.0.0"
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.pattern_library = CulturalPatternLibrary(config)
        
    def _initialize_evaluator(self):
        """Initialize creativity-specific components."""
        self.creativity_patterns = {
            "narrative_structures": [
                r"(?:once\s+upon|long\s+ago|in\s+the\s+beginning|there\s+was)",
                r"(?:and\s+then|suddenly|meanwhile|finally|in\s+the\s+end)",
                r"(?:moral\s+of|lesson\s+learned|teaches\s+us|wisdom\s+tells)"
            ],
            "performance_markers": [
                r"(?:listen\s+well|hear\s+me|gather\s+round|come\s+close)",
                r"(?:rhythm|beat|cadence|flow|tempo|pace)",
                r"(?:voice|tone|whisper|shout|sing|chant)",
                r"(?:gesture|movement|dance|sway|step|motion)"
            ],
            "cultural_creativity": [
                r"(?:ancestor|elder|wisdom|tradition|heritage|legacy)",
                r"(?:community|village|tribe|clan|family|kinship)",
                r"(?:sacred|spiritual|blessed|divine|holy|reverent)",
                r"(?:ceremony|ritual|celebration|festival|gathering)"
            ],
            "originality_markers": [
                r"(?:unique|original|innovative|creative|imaginative|novel)",
                r"(?:new\s+perspective|different\s+approach|fresh\s+idea)",
                r"(?:twist|variation|adaptation|interpretation|reimagining)",
                r"(?:blend|fusion|combination|synthesis|integration)"
            ]
        }
        
        self.appropriation_warnings = [
            "superficial cultural references",
            "stereotypical representations", 
            "missing cultural context",
            "inauthentic traditions",
            "commercialized spirituality"
        ]
    
    def get_supported_evaluation_types(self) -> List[str]:
        """Return supported evaluation types."""
        return ["creative_expression", "cultural_creativity", "performative_creativity"]
    
    def get_evaluation_dimensions(self) -> List[str]:
        """Return creativity evaluation dimensions."""
        return [
            "cultural_creative_patterns",
            "rhythmic_quality", 
            "narrative_coherence",
            "originality_within_bounds",
            "performance_quality",
            "collaborative_creation"
        ]
    
    def get_domain_name(self) -> str:
        """Return domain name."""
        return "creativity"
    
    def evaluate_dimension(self, 
                          dimension: str,
                          response_text: str, 
                          test_metadata: Dict[str, Any], 
                          cultural_context: CulturalContext) -> EvaluationDimension:
        """Evaluate specific creativity dimension."""
        
        if dimension == "cultural_creative_patterns":
            return self._evaluate_cultural_creative_patterns(response_text, cultural_context)
        elif dimension == "rhythmic_quality":
            return self._evaluate_rhythmic_quality(response_text, cultural_context)
        elif dimension == "narrative_coherence":
            return self._evaluate_narrative_coherence(response_text, cultural_context)
        elif dimension == "originality_within_bounds":
            return self._evaluate_originality_within_bounds(response_text, cultural_context)
        elif dimension == "performance_quality":
            return self._evaluate_performance_quality(response_text, cultural_context)
        elif dimension == "collaborative_creation":
            return self._evaluate_collaborative_creation(response_text, cultural_context)
        else:
            # Fallback dimension
            return EvaluationDimension(
                name=dimension,
                score=0.0,
                confidence=0.0,
                cultural_relevance=0.0,
                evidence=["Unknown dimension"],
                cultural_markers=[]
            )
    
    def _evaluate_cultural_creative_patterns(self, text: str, 
                                           cultural_context: CulturalContext) -> EvaluationDimension:
        """Evaluate detection and use of cultural creative patterns."""
        
        # Detect cultural patterns using pattern library
        detected_patterns = self.pattern_library.detect_patterns(text, cultural_context.traditions)
        
        evidence = []
        cultural_markers = []
        
        # Score based on pattern detection
        pattern_score = 0.0
        if detected_patterns:
            pattern_types = set(p.pattern_type for p in detected_patterns)
            creative_patterns = [p for p in detected_patterns 
                               if p.pattern_type in [PatternType.STORYTELLING_STRUCTURE, 
                                                   PatternType.NARRATIVE_STRUCTURE,
                                                   PatternType.VISUAL_NARRATIVE]]
            
            pattern_score = min(1.0, len(creative_patterns) / 3.0)  # Up to 3 patterns for full score
            
            for pattern in creative_patterns[:3]:  # Limit evidence
                evidence.append(f"Detected {pattern.tradition} {pattern.pattern_name}")
                cultural_markers.append(f"{pattern.tradition}:{pattern.pattern_name}")
        
        # Check for cultural creativity keywords
        keyword_score = 0.0
        text_lower = text.lower()
        for category, patterns in self.creativity_patterns.items():
            if category == "cultural_creativity":
                matches = 0
                for pattern in patterns:
                    pattern_matches = len(re.findall(pattern, text, re.IGNORECASE))
                    matches += pattern_matches
                    if pattern_matches > 0:
                        evidence.append(f"Cultural creativity marker: {pattern}")
                
                keyword_score = min(1.0, matches / 4.0)
                break
        
        # Check for appropriation warnings
        appropriation_penalty = 0.0
        appropriation_warnings = self.pattern_library.check_appropriation_warnings(detected_patterns, text)
        if appropriation_warnings:
            appropriation_penalty = min(0.3, len(appropriation_warnings) * 0.1)
            evidence.extend([f"Appropriation concern: {w}" for w in appropriation_warnings[:2]])
        
        # Calculate overall score
        overall_score = max(0.0, (pattern_score * 0.6 + keyword_score * 0.4) - appropriation_penalty)
        
        # Determine confidence and cultural relevance
        confidence = 0.8 if detected_patterns else 0.6
        cultural_relevance = 1.0 if cultural_context.traditions else 0.5
        
        if not evidence:
            evidence.append("No cultural creative patterns detected")
        
        return EvaluationDimension(
            name="cultural_creative_patterns",
            score=overall_score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_rhythmic_quality(self, text: str, 
                                 cultural_context: CulturalContext) -> EvaluationDimension:
        """Evaluate rhythmic and oral performance qualities."""
        
        evidence = []
        cultural_markers = []
        
        # Detect rhythmic patterns
        rhythmic_patterns = self.pattern_library.detect_patterns(
            text, 
            ["oral_performance"] + cultural_context.traditions
        )
        rhythmic_patterns = [p for p in rhythmic_patterns 
                           if p.pattern_type == PatternType.RHYTHMIC_PATTERN]
        
        pattern_score = 0.0
        if rhythmic_patterns:
            pattern_score = min(1.0, len(rhythmic_patterns) / 2.0)
            for pattern in rhythmic_patterns[:2]:
                evidence.append(f"Rhythmic pattern: {pattern.pattern_name}")
                cultural_markers.append(f"rhythm:{pattern.pattern_name}")
        
        # Analyze text rhythm through repetition and structure
        repetition_score = self._analyze_repetition_patterns(text)
        if repetition_score > 0.3:
            evidence.append(f"Repetitive structure detected (score: {repetition_score:.2f})")
        
        # Check for performance-oriented language
        performance_score = 0.0
        performance_patterns = self.creativity_patterns.get("performance_markers", [])
        performance_matches = 0
        for pattern in performance_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            performance_matches += matches
            if matches > 0:
                evidence.append(f"Performance marker: {pattern}")
        
        performance_score = min(1.0, performance_matches / 3.0)
        
        # Calculate overall rhythmic quality
        overall_score = (pattern_score * 0.4 + repetition_score * 0.3 + performance_score * 0.3)
        
        # Assess cultural relevance
        cultural_relevance = 1.0 if "oral_performance" in cultural_context.performance_aspects else 0.6
        confidence = 0.7 if rhythmic_patterns or performance_matches > 0 else 0.5
        
        if not evidence:
            evidence.append("No rhythmic qualities detected")
        
        return EvaluationDimension(
            name="rhythmic_quality",
            score=overall_score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_narrative_coherence(self, text: str, 
                                    cultural_context: CulturalContext) -> EvaluationDimension:
        """Evaluate narrative structure and cultural storytelling coherence."""
        
        evidence = []
        cultural_markers = []
        
        # Detect narrative structure patterns
        narrative_patterns = []
        for pattern_list in self.creativity_patterns["narrative_structures"]:
            matches = re.findall(pattern_list, text, re.IGNORECASE)
            narrative_patterns.extend(matches)
        
        structure_score = min(1.0, len(narrative_patterns) / 3.0)  # Up to 3 structural elements
        if narrative_patterns:
            evidence.append(f"Narrative structure elements: {len(narrative_patterns)}")
        
        # Analyze story flow and coherence
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        flow_score = 0.0
        if len(sentences) >= 3:
            # Simple coherence check - look for connecting words
            connectors = ["then", "next", "after", "finally", "meanwhile", "however", "but", "and"]
            connector_count = 0
            for sentence in sentences[1:]:  # Skip first sentence
                if any(conn in sentence.lower() for conn in connectors):
                    connector_count += 1
            
            flow_score = min(1.0, connector_count / (len(sentences) - 1))
            if flow_score > 0.3:
                evidence.append(f"Good narrative flow with connectors ({flow_score:.2f})")
        
        # Check for cultural storytelling patterns
        cultural_story_score = 0.0
        detected_cultural_patterns = self.pattern_library.detect_patterns(text, cultural_context.traditions)
        story_patterns = [p for p in detected_cultural_patterns 
                         if p.pattern_type in [PatternType.STORYTELLING_STRUCTURE, 
                                             PatternType.NARRATIVE_STRUCTURE]]
        
        if story_patterns:
            cultural_story_score = min(1.0, len(story_patterns) / 2.0)
            for pattern in story_patterns[:2]:
                evidence.append(f"Cultural story pattern: {pattern.tradition} {pattern.pattern_name}")
                cultural_markers.append(f"story:{pattern.tradition}:{pattern.pattern_name}")
        
        # Calculate overall coherence
        overall_score = (structure_score * 0.3 + flow_score * 0.4 + cultural_story_score * 0.3)
        
        # Assess confidence and cultural relevance
        confidence = 0.8 if story_patterns else 0.6
        cultural_relevance = 1.0 if cultural_context.traditions else 0.5
        
        if not evidence:
            evidence.append("Limited narrative coherence detected")
        
        return EvaluationDimension(
            name="narrative_coherence",
            score=overall_score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_originality_within_bounds(self, text: str, 
                                          cultural_context: CulturalContext) -> EvaluationDimension:
        """Evaluate originality while respecting cultural boundaries."""
        
        evidence = []
        cultural_markers = []
        
        # Check for originality markers
        originality_score = 0.0
        originality_patterns = self.creativity_patterns.get("originality_markers", [])
        originality_matches = 0
        for pattern in originality_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            originality_matches += matches
            if matches > 0:
                evidence.append(f"Originality marker: {pattern}")
        
        originality_score = min(1.0, originality_matches / 3.0)
        
        # Check for cultural boundary respect
        boundary_violations = self.pattern_library.check_sacred_boundaries(
            self.pattern_library.detect_patterns(text, cultural_context.traditions), 
            text
        )
        
        boundary_penalty = 0.0
        if boundary_violations:
            boundary_penalty = min(0.5, len(boundary_violations) * 0.2)
            evidence.extend([f"Boundary concern: {v}" for v in boundary_violations[:2]])
        
        # Assess cultural adaptation vs appropriation
        adaptation_score = 0.0
        if cultural_context.traditions:
            # Look for respectful adaptation indicators
            respectful_language = ["inspired by", "drawing from", "honoring", "respecting", "learning from"]
            respect_indicators = 0
            text_lower = text.lower()
            for indicator in respectful_language:
                if indicator in text_lower:
                    respect_indicators += 1
                    evidence.append(f"Respectful adaptation: '{indicator}'")
            
            adaptation_score = min(1.0, respect_indicators / 2.0)
        
        # Calculate final score
        overall_score = max(0.0, (originality_score * 0.5 + adaptation_score * 0.5) - boundary_penalty)
        
        # Assess confidence and cultural relevance
        confidence = 0.7 if originality_matches > 0 or adaptation_score > 0 else 0.5
        cultural_relevance = 1.0 if cultural_context.traditions else 0.3
        
        if not evidence:
            evidence.append("Limited originality within cultural bounds detected")
        
        return EvaluationDimension(
            name="originality_within_bounds",
            score=overall_score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_performance_quality(self, text: str, 
                                    cultural_context: CulturalContext) -> EvaluationDimension:
        """Evaluate performance and theatrical qualities."""
        
        evidence = []
        cultural_markers = []
        
        # Detect performance patterns
        performance_patterns = self.pattern_library.detect_patterns(text, cultural_context.traditions)
        performance_patterns = [p for p in performance_patterns 
                              if p.pattern_type in [PatternType.PERFORMANCE_MARKER, 
                                                  PatternType.PERFORMANCE_STYLE]]
        
        pattern_score = 0.0
        if performance_patterns:
            pattern_score = min(1.0, len(performance_patterns) / 2.0)
            for pattern in performance_patterns[:2]:
                evidence.append(f"Performance pattern: {pattern.tradition} {pattern.pattern_name}")
                cultural_markers.append(f"performance:{pattern.tradition}:{pattern.pattern_name}")
        
        # Check for theatrical language
        theatrical_score = 0.0
        theatrical_markers = [
            "dramatic", "stage", "scene", "act", "curtain", "audience", 
            "applause", "spotlight", "gesture", "movement", "expression"
        ]
        theatrical_matches = 0
        text_lower = text.lower()
        for marker in theatrical_markers:
            if marker in text_lower:
                theatrical_matches += 1
                evidence.append(f"Theatrical element: '{marker}'")
        
        theatrical_score = min(1.0, theatrical_matches / 4.0)
        
        # Analyze dialogue and voice variation
        dialogue_score = self._analyze_dialogue_quality(text)
        if dialogue_score > 0.3:
            evidence.append(f"Dialogue quality detected (score: {dialogue_score:.2f})")
        
        # Calculate overall performance quality
        overall_score = (pattern_score * 0.4 + theatrical_score * 0.3 + dialogue_score * 0.3)
        
        # Assess confidence and cultural relevance
        confidence = 0.8 if performance_patterns else 0.6
        cultural_relevance = 1.0 if cultural_context.performance_aspects else 0.4
        
        if not evidence:
            evidence.append("Limited performance quality detected")
        
        return EvaluationDimension(
            name="performance_quality",
            score=overall_score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_collaborative_creation(self, text: str, 
                                       cultural_context: CulturalContext) -> EvaluationDimension:
        """Evaluate collaborative and community-oriented creative elements."""
        
        evidence = []
        cultural_markers = []
        
        # Detect collaborative patterns
        collaborative_patterns = self.pattern_library.detect_patterns(text, cultural_context.traditions)
        collaborative_patterns = [p for p in collaborative_patterns 
                                if p.pattern_type == PatternType.AUDIENCE_INTERACTION]
        
        pattern_score = 0.0
        if collaborative_patterns:
            pattern_score = min(1.0, len(collaborative_patterns) / 2.0)
            for pattern in collaborative_patterns[:2]:
                evidence.append(f"Collaborative pattern: {pattern.tradition} {pattern.pattern_name}")
                cultural_markers.append(f"collaborative:{pattern.tradition}:{pattern.pattern_name}")
        
        # Check for community-oriented language
        community_score = 0.0
        community_markers = [
            "together", "community", "everyone", "all of us", "we", "our", 
            "shared", "collective", "join", "participate", "respond", "sing along"
        ]
        community_matches = 0
        text_lower = text.lower()
        for marker in community_markers:
            if marker in text_lower:
                community_matches += 1
        
        community_score = min(1.0, community_matches / 4.0)
        if community_score > 0.3:
            evidence.append(f"Community orientation detected ({community_matches} markers)")
        
        # Check for interactive elements
        interactive_score = 0.0
        interactive_patterns = [
            r"(?:repeat|say|sing)\s+(?:after|with)\s+me",
            r"(?:all\s+together|everyone)\s+(?:now|say|sing)",
            r"(?:response|answer|reply)\s+(?:is|now|together)",
            r"(?:clap|cheer|shout)\s+(?:along|with|now)"
        ]
        
        interactive_matches = 0
        for pattern in interactive_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            interactive_matches += matches
            if matches > 0:
                evidence.append(f"Interactive element: {pattern}")
        
        interactive_score = min(1.0, interactive_matches / 2.0)
        
        # Calculate overall collaborative score
        overall_score = (pattern_score * 0.4 + community_score * 0.3 + interactive_score * 0.3)
        
        # Assess confidence and cultural relevance
        confidence = 0.8 if collaborative_patterns or interactive_matches > 0 else 0.5
        cultural_relevance = 1.0 if any("collaborative" in aspect for aspect in cultural_context.performance_aspects) else 0.6
        
        if not evidence:
            evidence.append("Limited collaborative creation detected")
        
        return EvaluationDimension(
            name="collaborative_creation",
            score=overall_score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _analyze_repetition_patterns(self, text: str) -> float:
        """Analyze repetitive patterns that suggest rhythmic quality."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.0
        
        # Look for repeated phrases or structures
        phrase_counts = Counter()
        for sentence in sentences:
            words = sentence.lower().split()
            if len(words) >= 3:
                # Check 3-word phrases
                for i in range(len(words) - 2):
                    phrase = " ".join(words[i:i+3])
                    phrase_counts[phrase] += 1
        
        # Score based on repetition
        repeated_phrases = [count for count in phrase_counts.values() if count > 1]
        repetition_score = min(1.0, len(repeated_phrases) / 3.0)
        
        return repetition_score
    
    def _analyze_dialogue_quality(self, text: str) -> float:
        """Analyze dialogue and character voice quality."""
        # Look for dialogue markers
        dialogue_markers = ['"', "'", "said", "asked", "replied", "whispered", "shouted"]
        dialogue_indicators = 0
        
        text_lower = text.lower()
        for marker in dialogue_markers:
            if marker in text:
                dialogue_indicators += 1
        
        # Check for voice variation indicators
        voice_markers = ["tone", "voice", "accent", "whisper", "shout", "murmur", "exclaim"]
        voice_indicators = 0
        for marker in voice_markers:
            if marker in text_lower:
                voice_indicators += 1
        
        # Calculate dialogue quality score
        dialogue_score = min(1.0, (dialogue_indicators + voice_indicators) / 6.0)
        return dialogue_score