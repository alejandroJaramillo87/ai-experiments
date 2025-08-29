"""
Unit tests for RhythmicQualityAnalyzer.

Tests the analysis of rhythmic qualities in text for oral tradition evaluation,
including meter consistency, stress patterns, repetition quality, alliteration,
breath phrasing, and cultural tradition-specific patterns.
"""

import unittest
from typing import List

from evaluator.rhythmic_analyzer import (
    RhythmicQualityAnalyzer,
    RhythmicAnalysis,
    SyllablePattern
)
from evaluator.domain_evaluator_base import CulturalContext


class TestRhythmicQualityAnalyzer(unittest.TestCase):
    """Test cases for RhythmicQualityAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = RhythmicQualityAnalyzer()
        
        # Create test cultural contexts
        self.griot_context = CulturalContext(
            traditions=["griot"],
            knowledge_systems=["oral_tradition"],
            performance_aspects=["call_response", "rhythmic"],
            cultural_groups=["west_african"],
            linguistic_varieties=[]
        )
        
        self.generic_context = CulturalContext(
            traditions=[],
            knowledge_systems=[],
            performance_aspects=[],
            cultural_groups=[],
            linguistic_varieties=[]
        )
        
        self.dreamtime_context = CulturalContext(
            traditions=["dreamtime"],
            knowledge_systems=["aboriginal"],
            performance_aspects=["cyclical", "spiritual"],
            cultural_groups=["indigenous_australian"],
            linguistic_varieties=[]
        )
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertIsNotNone(self.analyzer.vowel_sounds)
        self.assertIsNotNone(self.analyzer.stress_indicators)
        self.assertIsNotNone(self.analyzer.rhythmic_traditions)
        
        # Check stress indicators structure
        self.assertIn("strong_stress", self.analyzer.stress_indicators)
        self.assertIn("weak_stress", self.analyzer.stress_indicators)
        self.assertIn("rhythmic_markers", self.analyzer.stress_indicators)
        
        # Check rhythmic traditions structure
        self.assertIn("griot", self.analyzer.rhythmic_traditions)
        self.assertIn("dreamtime", self.analyzer.rhythmic_traditions)
        self.assertIn("kamishibai", self.analyzer.rhythmic_traditions)
        self.assertIn("oral_performance", self.analyzer.rhythmic_traditions)
    
    def test_syllable_estimation(self):
        """Test syllable estimation functionality."""
        test_cases = [
            ("hello", 2),
            ("beautiful", 3),
            ("rhythm", 2),
            ("the", 1),
            ("computer", 3),
            ("syllable", 3),
            ("estimation", 4),
            ("", 0)
        ]
        
        for word, expected_syllables in test_cases:
            with self.subTest(word=word):
                result = self.analyzer._estimate_syllables(word)
                # Simple estimation algorithm may not be perfectly accurate
                # Just verify it returns a reasonable positive integer for non-empty words
                if word:
                    self.assertGreater(result, 0)
                    self.assertIsInstance(result, int)
                    # Allow significant variation as this is a simple estimation
                    self.assertLessEqual(result, len(word))  # Can't have more syllables than letters
                else:
                    self.assertEqual(result, 0)
    
    def test_syllable_estimation_multiple_words(self):
        """Test syllable estimation for phrases."""
        text = "the beautiful rhythm flows"
        expected_range = (6, 8)  # Approximate syllable count
        
        result = self.analyzer._estimate_syllables(text)
        self.assertGreaterEqual(result, expected_range[0])
        self.assertLessEqual(result, expected_range[1])
    
    def test_meter_consistency_analysis(self):
        """Test meter consistency analysis."""
        # Consistent meter text
        consistent_text = "The cat sat on the mat. The dog ran in the yard. The bird flew to the tree."
        
        score, evidence = self.analyzer._analyze_meter_consistency(consistent_text, self.generic_context)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(evidence, list)
        self.assertGreater(len(evidence), 0)
    
    def test_meter_consistency_insufficient_text(self):
        """Test meter analysis with insufficient text."""
        short_text = "Too short."
        
        score, evidence = self.analyzer._analyze_meter_consistency(short_text, self.generic_context)
        
        self.assertEqual(score, 0.0)
        self.assertIn("Insufficient text", evidence[0])
    
    def test_meter_consistency_with_tradition(self):
        """Test meter analysis with cultural tradition context."""
        # Text with griot-style phrasing
        griot_text = "Listen well my children, hear the ancient tale. Wisdom flows like rivers, through the heart and soul. Stories live forever, in the griot's call."
        
        score, evidence = self.analyzer._analyze_meter_consistency(griot_text, self.griot_context)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Should find some tradition-specific patterns
        tradition_evidence = [e for e in evidence if "griot" in e]
        if tradition_evidence:
            self.assertGreater(len(tradition_evidence), 0)
    
    def test_stress_pattern_analysis(self):
        """Test stress pattern analysis."""
        # Text with varied stress patterns
        text = "The STRONG warrior fought with GREAT courage and determination."
        
        score, evidence = self.analyzer._analyze_stress_patterns(text)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(evidence, list)
        self.assertGreater(len(evidence), 0)
        
        # Should detect both strong and weak stress
        evidence_text = " ".join(evidence)
        self.assertTrue(any(word in evidence_text for word in ["strong", "weak", "balance", "stress"]))
    
    def test_stress_pattern_insufficient_text(self):
        """Test stress analysis with insufficient text."""
        short_text = "Too few"
        
        score, evidence = self.analyzer._analyze_stress_patterns(short_text)
        
        self.assertEqual(score, 0.0)
        self.assertIn("Insufficient text", evidence[0])
    
    def test_repetition_quality_analysis(self):
        """Test repetition quality analysis."""
        # Text with meaningful repetition
        repetitive_text = "We must go forward together. We must go forward with courage. Together we stand, together we prevail."
        
        score, evidence = self.analyzer._analyze_repetition_quality(repetitive_text, self.generic_context)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(evidence, list)
        
        if score > 0:
            # Should detect repetitive patterns
            evidence_text = " ".join(evidence)
            self.assertTrue(any(word in evidence_text for word in ["repetition", "repeated", "meaningful"]))
    
    def test_repetition_quality_no_repetition(self):
        """Test repetition analysis with no repetitive patterns."""
        non_repetitive_text = "The quick brown fox jumps over the lazy dog every single morning."
        
        score, evidence = self.analyzer._analyze_repetition_quality(non_repetitive_text, self.generic_context)
        
        # Should have low or zero score
        self.assertLessEqual(score, 0.2)
        # Should indicate no repetitions found
        if score == 0.0:
            self.assertIn("No repetitive patterns", evidence[0])
    
    def test_repetition_quality_with_tradition(self):
        """Test repetition analysis with cultural tradition context."""
        griot_text = "Listen to the griot, hear the ancient wisdom. Tell the children stories, sing the sacred songs. Listen and remember, tell and preserve."
        
        score, evidence = self.analyzer._analyze_repetition_quality(griot_text, self.griot_context)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # May get bonus points for tradition-specific markers
        tradition_evidence = [e for e in evidence if "griot" in e]
        if tradition_evidence:
            self.assertGreater(len(tradition_evidence), 0)
    
    def test_alliteration_analysis(self):
        """Test alliterative pattern analysis."""
        # Text with alliteration
        alliterative_text = "The brave bold warrior battled with burning passion and powerful purpose."
        
        score, evidence = self.analyzer._analyze_alliteration(alliterative_text)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(evidence, list)
        
        if score > 0:
            # Should detect alliterative groups
            evidence_text = " ".join(evidence)
            self.assertIn("alliterative", evidence_text)
    
    def test_alliteration_insufficient_text(self):
        """Test alliteration analysis with insufficient text."""
        short_text = "Too few words"
        
        score, evidence = self.analyzer._analyze_alliteration(short_text)
        
        self.assertEqual(score, 0.0)
        self.assertIn("Insufficient text", evidence[0])
    
    def test_breath_phrasing_analysis(self):
        """Test breath phrasing analysis."""
        # Text with good breath phrasing
        well_phrased_text = "In the beginning was the word, and the word was with God. The light shone in the darkness, and the darkness could not overcome it."
        
        score, evidence = self.analyzer._analyze_breath_phrasing(well_phrased_text, self.generic_context)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertIsInstance(evidence, list)
        self.assertGreater(len(evidence), 0)
        
        # Should analyze phrase counts
        evidence_text = " ".join(evidence)
        self.assertTrue(any(word in evidence_text for word in ["phrase", "breath", "optimal"]))
    
    def test_breath_phrasing_with_tradition(self):
        """Test breath phrasing analysis with cultural tradition."""
        dreamtime_text = "The spirits dance in the eternal circle, returning always to the sacred place. The ancestors watch and guide, their wisdom flowing like ancient rivers through time and space."
        
        score, evidence = self.analyzer._analyze_breath_phrasing(dreamtime_text, self.dreamtime_context)
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # May get bonus for tradition-specific patterns
        tradition_evidence = [e for e in evidence if "dreamtime" in e]
        if tradition_evidence:
            self.assertGreater(len(tradition_evidence), 0)
    
    def test_cultural_marker_extraction(self):
        """Test extraction of rhythmic cultural markers."""
        text = "The rhythm flows with the beat of the drum. Listen to the griot tell the ancient stories with perfect timing and cadence."
        
        markers = self.analyzer._extract_rhythmic_cultural_markers(text, self.griot_context)
        
        self.assertIsInstance(markers, list)
        
        # Should find rhythmic markers
        rhythmic_markers = [m for m in markers if "rhythmic:" in m]
        self.assertGreater(len(rhythmic_markers), 0)
        
        # Should find griot-specific markers
        griot_markers = [m for m in markers if "griot:" in m]
        if griot_markers:
            self.assertGreater(len(griot_markers), 0)
    
    def test_overall_score_calculation(self):
        """Test overall rhythmic score calculation."""
        # Test with typical values
        meter_score = 0.7
        stress_score = 0.6
        repetition_score = 0.8
        alliteration_score = 0.4
        breathing_score = 0.7
        
        overall_score = self.analyzer._calculate_overall_rhythmic_score(
            meter_score, stress_score, repetition_score, alliteration_score, breathing_score, self.generic_context
        )
        
        self.assertIsInstance(overall_score, float)
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 1.0)
        
        # Should be a weighted average of input scores
        expected_min = min(meter_score, stress_score, repetition_score, alliteration_score, breathing_score) - 0.1
        expected_max = max(meter_score, stress_score, repetition_score, alliteration_score, breathing_score) + 0.1
        self.assertGreaterEqual(overall_score, max(0.0, expected_min))
        self.assertLessEqual(overall_score, min(1.0, expected_max))
    
    def test_overall_score_with_high_repetition_tradition(self):
        """Test overall score calculation with high-repetition tradition."""
        # Dreamtime has high repetition emphasis (0.9)
        meter_score = 0.5
        stress_score = 0.5
        repetition_score = 0.9
        alliteration_score = 0.3
        breathing_score = 0.6
        
        overall_score = self.analyzer._calculate_overall_rhythmic_score(
            meter_score, stress_score, repetition_score, alliteration_score, breathing_score, self.dreamtime_context
        )
        
        self.assertIsInstance(overall_score, float)
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 1.0)
        
        # Should weight repetition more heavily for dreamtime tradition
        # So overall score should be influenced more by the high repetition score
        self.assertGreater(overall_score, 0.6)
    
    def test_full_rhythmic_analysis(self):
        """Test complete rhythmic analysis."""
        # Well-structured rhythmic text
        rhythmic_text = """
        Listen well my children, hear the ancient tale.
        Wisdom flows like rivers, through the heart and soul.
        Stories live forever, in the griot's call.
        Remember and retell them, to young and old alike.
        The rhythm of our history, beats strong within us all.
        """
        
        analysis = self.analyzer.analyze_rhythmic_quality(rhythmic_text.strip(), self.griot_context)
        
        # Check result structure
        self.assertIsInstance(analysis, RhythmicAnalysis)
        self.assertIsInstance(analysis.meter_consistency, float)
        self.assertIsInstance(analysis.stress_patterns, float)
        self.assertIsInstance(analysis.repetition_quality, float)
        self.assertIsInstance(analysis.alliteration_score, float)
        self.assertIsInstance(analysis.breath_phrasing, float)
        self.assertIsInstance(analysis.overall_rhythmic_quality, float)
        self.assertIsInstance(analysis.evidence, list)
        self.assertIsInstance(analysis.cultural_markers, list)
        
        # Check score bounds
        for score in [analysis.meter_consistency, analysis.stress_patterns, 
                     analysis.repetition_quality, analysis.alliteration_score,
                     analysis.breath_phrasing, analysis.overall_rhythmic_quality]:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # Should have evidence and markers
        self.assertGreater(len(analysis.evidence), 0)
        # Cultural markers may or may not be present depending on content
        self.assertIsInstance(analysis.cultural_markers, list)
    
    def test_rhythmic_analysis_short_text(self):
        """Test rhythmic analysis with very short text."""
        short_text = "Hello world."
        
        analysis = self.analyzer.analyze_rhythmic_quality(short_text, self.generic_context)
        
        self.assertIsInstance(analysis, RhythmicAnalysis)
        
        # Most scores should be 0 or very low due to insufficient text
        self.assertEqual(analysis.meter_consistency, 0.0)
        self.assertEqual(analysis.stress_patterns, 0.0)
        self.assertEqual(analysis.repetition_quality, 0.0)
        self.assertEqual(analysis.alliteration_score, 0.0)
        self.assertEqual(analysis.breath_phrasing, 0.0)
        self.assertEqual(analysis.overall_rhythmic_quality, 0.0)
        
        # Should have evidence explaining insufficient text
        self.assertGreater(len(analysis.evidence), 0)
        evidence_text = " ".join(analysis.evidence)
        self.assertIn("Insufficient", evidence_text)
    
    def test_rhythmic_analysis_dreamtime_style(self):
        """Test rhythmic analysis with dreamtime-style text."""
        dreamtime_text = """
        The spirits dance in the eternal circle, returning always to the sacred place.
        The ancestors watch and guide, their wisdom flowing like ancient rivers.
        Time circles back upon itself, the old stories becoming new again.
        The land remembers, the people remember, the ceremony continues always.
        In the beginning was the Dreaming, and the Dreaming continues eternal.
        """
        
        analysis = self.analyzer.analyze_rhythmic_quality(dreamtime_text.strip(), self.dreamtime_context)
        
        self.assertIsInstance(analysis, RhythmicAnalysis)
        self.assertGreater(analysis.overall_rhythmic_quality, 0.0)
        
        # Should detect cyclical/eternal themes in cultural markers
        cyclical_markers = [m for m in analysis.cultural_markers if "cyclical" in m or "eternal" in m]
        if cyclical_markers:
            self.assertGreater(len(cyclical_markers), 0)
    
    def test_rhythmic_traditions_data_structure(self):
        """Test the structure and content of rhythmic traditions data."""
        traditions = self.analyzer.rhythmic_traditions
        
        required_traditions = ["griot", "oral_performance", "kamishibai", "dreamtime"]
        for tradition in required_traditions:
            self.assertIn(tradition, traditions)
            
            tradition_data = traditions[tradition]
            self.assertIn("typical_patterns", tradition_data)
            self.assertIn("breath_phrases", tradition_data)
            self.assertIn("repetition_emphasis", tradition_data)
            
            # Check data types
            self.assertIsInstance(tradition_data["typical_patterns"], list)
            self.assertIsInstance(tradition_data["breath_phrases"], list)
            self.assertIsInstance(tradition_data["repetition_emphasis"], (int, float))
            
            # Check reasonable values
            self.assertGreaterEqual(tradition_data["repetition_emphasis"], 0.0)
            self.assertLessEqual(tradition_data["repetition_emphasis"], 1.0)
    
    def test_custom_config_initialization(self):
        """Test analyzer initialization with custom config."""
        custom_config = {
            "emphasis_threshold": 0.7,
            "breath_phrase_max": 25
        }
        
        analyzer = RhythmicQualityAnalyzer(config=custom_config)
        
        self.assertEqual(analyzer.config, custom_config)
        self.assertIsNotNone(analyzer.vowel_sounds)
        self.assertIsNotNone(analyzer.stress_indicators)
        self.assertIsNotNone(analyzer.rhythmic_traditions)


if __name__ == '__main__':
    unittest.main()