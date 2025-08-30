from typing import Dict, List, Any, Optional, Tuple
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass

from ..core.domain_evaluator_base import CulturalContext


@dataclass
class RhythmicAnalysis:
    """Analysis results for rhythmic qualities in text."""
    meter_consistency: float  # 0.0 to 1.0
    stress_patterns: float  # 0.0 to 1.0
    repetition_quality: float  # 0.0 to 1.0
    alliteration_score: float  # 0.0 to 1.0
    breath_phrasing: float  # 0.0 to 1.0
    overall_rhythmic_quality: float  # 0.0 to 1.0
    evidence: List[str]
    cultural_markers: List[str]


@dataclass
class SyllablePattern:
    """Pattern of syllables in a text segment."""
    syllable_counts: List[int]
    stress_pattern: List[str]  # 'strong', 'weak', 'medium'
    consistency_score: float


class RhythmicQualityAnalyzer:
    """Analyzes rhythmic qualities in text for oral tradition evaluation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.vowel_sounds = 'aeiouAEIOU'
        self.stress_indicators = self._initialize_stress_patterns()
        self.rhythmic_traditions = self._initialize_rhythmic_traditions()
        
    def _initialize_stress_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns that indicate stressed syllables."""
        return {
            "strong_stress": [
                "LOUD", "STRONG", "EMPHASIZED", "STRESSED",
                # Common strong stress words
                "FIRST", "MAIN", "GREAT", "BIG", "STRONG", "LOUD", "DEEP",
                # Action words that tend to be stressed
                "STRIKE", "BEAT", "POUND", "SLAM", "CRASH", "BOOM"
            ],
            "weak_stress": [
                "the", "a", "an", "of", "to", "in", "for", "with", "by",
                "at", "on", "up", "as", "is", "are", "was", "were"
            ],
            "rhythmic_markers": [
                "rhythm", "beat", "pulse", "meter", "tempo", "cadence",
                "flow", "pace", "timing", "measure", "verse", "line"
            ]
        }
    
    def _initialize_rhythmic_traditions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize rhythmic patterns for different cultural traditions."""
        return {
            "griot": {
                "typical_patterns": ["strong-weak-strong", "strong-weak-weak-strong"],
                "breath_phrases": [8, 12, 16],  # Typical phrase lengths in syllables
                "repetition_emphasis": 0.8,
                "call_response_markers": ["listen", "hear", "say", "tell", "sing"]
            },
            "oral_performance": {
                "typical_patterns": ["strong-weak", "weak-strong", "strong-weak-weak"],
                "breath_phrases": [6, 8, 10, 12],
                "repetition_emphasis": 0.7,
                "memory_aids": ["repeat", "again", "echo", "refrain", "chorus"]
            },
            "kamishibai": {
                "typical_patterns": ["strong-weak-strong-weak"],
                "breath_phrases": [10, 14, 18],
                "repetition_emphasis": 0.6,
                "dramatic_markers": ["pause", "moment", "suddenly", "slowly"]
            },
            "dreamtime": {
                "typical_patterns": ["flowing", "cyclical"],
                "breath_phrases": [12, 16, 20],
                "repetition_emphasis": 0.9,
                "cyclical_markers": ["again", "returns", "circles", "eternal", "always"]
            }
        }
    
    def analyze_rhythmic_quality(self, text: str, 
                               cultural_context: CulturalContext) -> RhythmicAnalysis:
        """
        Analyze rhythmic qualities in text for oral tradition performance.
        
        Args:
            text: Text to analyze
            cultural_context: Cultural context for tradition-specific analysis
            
        Returns:
            RhythmicAnalysis with detailed rhythmic assessment
        """
        evidence = []
        cultural_markers = []
        
        # Analyze meter consistency
        meter_score, meter_evidence = self._analyze_meter_consistency(text, cultural_context)
        evidence.extend(meter_evidence)
        
        # Analyze stress patterns
        stress_score, stress_evidence = self._analyze_stress_patterns(text)
        evidence.extend(stress_evidence)
        
        # Analyze repetition quality
        repetition_score, repetition_evidence = self._analyze_repetition_quality(text, cultural_context)
        evidence.extend(repetition_evidence)
        
        # Analyze alliteration and sound patterns
        alliteration_score, alliteration_evidence = self._analyze_alliteration(text)
        evidence.extend(alliteration_evidence)
        
        # Analyze breath phrasing
        breathing_score, breathing_evidence = self._analyze_breath_phrasing(text, cultural_context)
        evidence.extend(breathing_evidence)
        
        # Extract cultural markers
        cultural_markers = self._extract_rhythmic_cultural_markers(text, cultural_context)
        
        # Calculate overall rhythmic quality
        overall_score = self._calculate_overall_rhythmic_score(
            meter_score, stress_score, repetition_score, 
            alliteration_score, breathing_score, cultural_context
        )
        
        return RhythmicAnalysis(
            meter_consistency=meter_score,
            stress_patterns=stress_score,
            repetition_quality=repetition_score,
            alliteration_score=alliteration_score,
            breath_phrasing=breathing_score,
            overall_rhythmic_quality=overall_score,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _analyze_meter_consistency(self, text: str, 
                                 cultural_context: CulturalContext) -> Tuple[float, List[str]]:
        """Analyze consistency of metrical patterns."""
        evidence = []
        
        # Split into lines (sentences or line breaks)
        lines = re.split(r'[.!?;\n]+', text)
        lines = [line.strip() for line in lines if line.strip()]
        
        if len(lines) < 2:
            return 0.0, ["Insufficient text for meter analysis"]
        
        # Analyze syllable patterns in each line
        syllable_counts = []
        for line in lines:
            syllables = self._estimate_syllables(line)
            syllable_counts.append(syllables)
        
        # Calculate consistency
        if len(syllable_counts) < 2:
            consistency_score = 0.0
        else:
            mean_syllables = statistics.mean(syllable_counts)
            syllable_variance = statistics.variance(syllable_counts) if len(syllable_counts) > 1 else 0
            
            # Lower variance indicates better consistency
            consistency_score = max(0.0, 1.0 - (syllable_variance / (mean_syllables * mean_syllables + 1)))
            
            if consistency_score > 0.7:
                evidence.append(f"Strong meter consistency (variance: {syllable_variance:.2f})")
            elif consistency_score > 0.4:
                evidence.append(f"Moderate meter consistency (variance: {syllable_variance:.2f})")
        
        # Check for tradition-specific patterns
        tradition_bonus = 0.0
        for tradition in cultural_context.traditions:
            if tradition in self.rhythmic_traditions:
                tradition_data = self.rhythmic_traditions[tradition]
                typical_lengths = tradition_data.get("breath_phrases", [])
                
                # Check if syllable counts match typical patterns
                matches = sum(1 for count in syllable_counts if any(abs(count - typical) <= 2 for typical in typical_lengths))
                if matches > 0:
                    tradition_bonus = min(0.3, matches / len(syllable_counts))
                    evidence.append(f"Matches {tradition} typical phrase lengths ({matches} lines)")
        
        final_score = min(1.0, consistency_score + tradition_bonus)
        return final_score, evidence
    
    def _analyze_stress_patterns(self, text: str) -> Tuple[float, List[str]]:
        """Analyze stress patterns in text."""
        evidence = []
        
        words = text.split()
        if len(words) < 4:
            return 0.0, ["Insufficient text for stress analysis"]
        
        # Simple stress pattern analysis
        strong_stress_count = 0
        weak_stress_count = 0
        
        for word in words:
            word_lower = word.lower().strip('.,!?;:"')
            if word_lower in self.stress_indicators["weak_stress"]:
                weak_stress_count += 1
            elif len(word) >= 3 and word.isupper():  # All caps might indicate emphasis
                strong_stress_count += 1
            elif len(word) >= 6:  # Longer words often have stress
                strong_stress_count += 1
        
        # Calculate stress pattern quality
        total_words = len(words)
        stress_ratio = (strong_stress_count + weak_stress_count) / total_words
        
        # Good stress patterns have varied stress (not all strong or all weak)
        if strong_stress_count > 0 and weak_stress_count > 0:
            balance = min(strong_stress_count, weak_stress_count) / max(strong_stress_count, weak_stress_count)
            stress_score = balance * stress_ratio
            evidence.append(f"Balanced stress pattern ({strong_stress_count} strong, {weak_stress_count} weak)")
        else:
            stress_score = stress_ratio * 0.5
            evidence.append(f"Unbalanced stress pattern ({strong_stress_count} strong, {weak_stress_count} weak)")
        
        return min(1.0, stress_score), evidence
    
    def _analyze_repetition_quality(self, text: str, 
                                  cultural_context: CulturalContext) -> Tuple[float, List[str]]:
        """Analyze quality and cultural appropriateness of repetition."""
        evidence = []
        
        # Find repeated phrases
        words = text.lower().split()
        if len(words) < 6:
            return 0.0, ["Insufficient text for repetition analysis"]
        
        # Check for repeated 2-3 word phrases
        phrase_counts = Counter()
        for i in range(len(words) - 1):
            if i < len(words) - 2:
                # 3-word phrases
                phrase = " ".join(words[i:i+3])
                phrase_counts[phrase] += 1
            
            # 2-word phrases
            phrase = " ".join(words[i:i+2])
            phrase_counts[phrase] += 1
        
        # Count meaningful repetitions (appearing more than once)
        repeated_phrases = [(phrase, count) for phrase, count in phrase_counts.items() if count > 1]
        
        if not repeated_phrases:
            return 0.0, ["No repetitive patterns detected"]
        
        # Analyze quality of repetitions
        repetition_score = 0.0
        meaningful_repetitions = 0
        
        for phrase, count in repeated_phrases:
            # Skip very common words/phrases
            if not any(word in self.stress_indicators["weak_stress"] for word in phrase.split()):
                meaningful_repetitions += 1
                repetition_score += min(0.3, count * 0.1)  # Diminishing returns for excessive repetition
        
        if meaningful_repetitions > 0:
            evidence.append(f"Found {meaningful_repetitions} meaningful repetitions")
        
        # Check for tradition-specific repetition patterns
        tradition_bonus = 0.0
        for tradition in cultural_context.traditions:
            if tradition in self.rhythmic_traditions:
                tradition_data = self.rhythmic_traditions[tradition]
                repetition_emphasis = tradition_data.get("repetition_emphasis", 0.5)
                
                # Check for tradition-specific markers
                tradition_markers = tradition_data.get("call_response_markers", []) + \
                                  tradition_data.get("memory_aids", []) + \
                                  tradition_data.get("cyclical_markers", [])
                
                found_markers = sum(1 for marker in tradition_markers if marker in text.lower())
                if found_markers > 0:
                    tradition_bonus = repetition_emphasis * min(0.4, found_markers * 0.1)
                    evidence.append(f"Found {found_markers} {tradition} repetition markers")
        
        final_score = min(1.0, repetition_score + tradition_bonus)
        return final_score, evidence
    
    def _analyze_alliteration(self, text: str) -> Tuple[float, List[str]]:
        """Analyze alliterative patterns."""
        evidence = []
        
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) < 4:
            return 0.0, ["Insufficient text for alliteration analysis"]
        
        # Group words by starting letter
        letter_groups = defaultdict(list)
        for word in words:
            if len(word) > 2:  # Skip very short words
                first_letter = word[0]
                letter_groups[first_letter].append(word)
        
        # Find alliterative sequences
        alliteration_score = 0.0
        alliterative_groups = 0
        
        for letter, word_list in letter_groups.items():
            if len(word_list) >= 2:
                # Check if words appear close together
                close_pairs = 0
                for i, word in enumerate(word_list[:-1]):
                    word_index = words.index(word)
                    next_word_index = words.index(word_list[i + 1], word_index + 1)
                    
                    if next_word_index - word_index <= 5:  # Within 5 words
                        close_pairs += 1
                
                if close_pairs > 0:
                    alliterative_groups += 1
                    alliteration_score += min(0.2, close_pairs * 0.05)
        
        if alliterative_groups > 0:
            evidence.append(f"Found {alliterative_groups} alliterative groups")
        
        return min(1.0, alliteration_score), evidence
    
    def _analyze_breath_phrasing(self, text: str, 
                               cultural_context: CulturalContext) -> Tuple[float, List[str]]:
        """Analyze natural breath phrasing for oral performance."""
        evidence = []
        
        # Split into potential breath phrases (by punctuation and conjunctions)
        phrases = re.split(r'[,.;!?]|\band\b|\bbut\b|\bor\b|\byet\b', text)
        phrases = [phrase.strip() for phrase in phrases if phrase.strip()]
        
        if len(phrases) < 2:
            return 0.0, ["Insufficient phrases for breath analysis"]
        
        # Analyze phrase lengths in syllables
        phrase_syllables = [self._estimate_syllables(phrase) for phrase in phrases]
        
        # Good breath phrases are typically 6-20 syllables
        optimal_phrases = sum(1 for count in phrase_syllables if 6 <= count <= 20)
        optimal_ratio = optimal_phrases / len(phrase_syllables)
        
        evidence.append(f"Optimal breath phrases: {optimal_phrases}/{len(phrase_syllables)}")
        
        # Check for tradition-specific breath patterns
        tradition_bonus = 0.0
        for tradition in cultural_context.traditions:
            if tradition in self.rhythmic_traditions:
                tradition_data = self.rhythmic_traditions[tradition]
                typical_lengths = tradition_data.get("breath_phrases", [])
                
                # Check how many phrases match typical lengths
                matches = sum(1 for count in phrase_syllables 
                             if any(abs(count - typical) <= 3 for typical in typical_lengths))
                
                if matches > 0:
                    tradition_bonus = min(0.3, matches / len(phrase_syllables))
                    evidence.append(f"Matches {tradition} breath patterns ({matches} phrases)")
        
        final_score = min(1.0, optimal_ratio + tradition_bonus)
        return final_score, evidence
    
    def _extract_rhythmic_cultural_markers(self, text: str, 
                                         cultural_context: CulturalContext) -> List[str]:
        """Extract cultural markers related to rhythm and oral performance."""
        markers = []
        text_lower = text.lower()
        
        # General rhythmic markers
        for marker in self.stress_indicators["rhythmic_markers"]:
            if marker in text_lower:
                markers.append(f"rhythmic:{marker}")
        
        # Tradition-specific markers
        for tradition in cultural_context.traditions:
            if tradition in self.rhythmic_traditions:
                tradition_data = self.rhythmic_traditions[tradition]
                
                # Check all marker types for this tradition
                for marker_type in ["call_response_markers", "memory_aids", "dramatic_markers", "cyclical_markers"]:
                    marker_list = tradition_data.get(marker_type, [])
                    for marker in marker_list:
                        if marker in text_lower:
                            markers.append(f"{tradition}:{marker_type}:{marker}")
        
        return list(set(markers))  # Remove duplicates
    
    def _calculate_overall_rhythmic_score(self, meter_score: float, stress_score: float,
                                        repetition_score: float, alliteration_score: float,
                                        breathing_score: float, 
                                        cultural_context: CulturalContext) -> float:
        """Calculate overall rhythmic quality score."""
        
        # Base weights for different aspects
        weights = {
            'meter': 0.25,
            'stress': 0.20,
            'repetition': 0.25,
            'alliteration': 0.15,
            'breathing': 0.15
        }
        
        # Adjust weights based on cultural context
        if any(tradition in self.rhythmic_traditions for tradition in cultural_context.traditions):
            # Increase importance of repetition for traditions that emphasize it
            for tradition in cultural_context.traditions:
                if tradition in self.rhythmic_traditions:
                    tradition_data = self.rhythmic_traditions[tradition]
                    if tradition_data.get("repetition_emphasis", 0) > 0.7:
                        weights['repetition'] += 0.1
                        weights['meter'] -= 0.05
                        weights['stress'] -= 0.05
        
        # Calculate weighted score
        overall_score = (
            meter_score * weights['meter'] +
            stress_score * weights['stress'] +
            repetition_score * weights['repetition'] +
            alliteration_score * weights['alliteration'] +
            breathing_score * weights['breathing']
        )
        
        return min(1.0, overall_score)
    
    def _estimate_syllables(self, text: str) -> int:
        """Estimate syllable count in text (simple approximation)."""
        # Remove punctuation and convert to lowercase
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = clean_text.split()
        
        total_syllables = 0
        for word in words:
            if not word:
                continue
                
            # Simple syllable estimation based on vowel groups
            vowel_groups = 0
            prev_was_vowel = False
            
            for char in word:
                is_vowel = char in self.vowel_sounds
                if is_vowel and not prev_was_vowel:
                    vowel_groups += 1
                prev_was_vowel = is_vowel
            
            # Minimum 1 syllable per word
            syllables = max(1, vowel_groups)
            
            # Adjust for silent 'e'
            if word.endswith('e') and len(word) > 3:
                syllables -= 1
                syllables = max(1, syllables)  # Don't go below 1
            
            total_syllables += syllables
        
        return total_syllables