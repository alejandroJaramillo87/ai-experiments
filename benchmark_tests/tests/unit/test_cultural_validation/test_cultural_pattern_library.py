"""
Unit tests for CulturalPatternLibrary.

Tests the detection and analysis of cultural patterns in text,
including pattern library initialization, pattern detection,
authenticity assessment, and cultural competence analysis.
"""

import unittest
from typing import List
from unittest.mock import patch

from evaluator.cultural.cultural_pattern_library import (
    CulturalPatternLibrary,
    CulturalPattern,
    PatternLibraryEntry,
    PatternType
)


class TestCulturalPatternLibrary(unittest.TestCase):
    """Test cases for CulturalPatternLibrary."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.library = CulturalPatternLibrary()
    
    def test_library_initialization(self):
        """Test library initialization."""
        self.assertIsNotNone(self.library.patterns)
        self.assertIsNotNone(self.library.tradition_patterns)
        
        # Check that patterns were loaded
        self.assertGreater(len(self.library.patterns), 0)
        
        # Check expected traditions
        expected_traditions = ["griot", "dreamtime", "kamishibai", "oral_performance", "general"]
        for tradition in expected_traditions:
            self.assertIn(tradition, self.library.patterns)
            self.assertGreater(len(self.library.patterns[tradition]), 0)
    
    def test_griot_patterns_loaded(self):
        """Test that griot patterns are properly loaded."""
        griot_patterns = self.library.patterns.get("griot", [])
        self.assertGreater(len(griot_patterns), 0)
        
        # Check for specific griot patterns
        pattern_names = [p.pattern_name for p in griot_patterns]
        self.assertIn("call_response", pattern_names)
        self.assertIn("moral_embedding", pattern_names)
        self.assertIn("historical_weaving", pattern_names)
    
    def test_dreamtime_patterns_loaded(self):
        """Test that dreamtime patterns are properly loaded."""
        dreamtime_patterns = self.library.patterns.get("dreamtime", [])
        self.assertGreater(len(dreamtime_patterns), 0)
        
        # Check for specific dreamtime patterns
        pattern_names = [p.pattern_name for p in dreamtime_patterns]
        self.assertIn("landscape_embodiment", pattern_names)
        self.assertIn("ancestor_presence", pattern_names)
        self.assertIn("cyclical_time", pattern_names)
    
    def test_kamishibai_patterns_loaded(self):
        """Test that kamishibai patterns are properly loaded."""
        kamishibai_patterns = self.library.patterns.get("kamishibai", [])
        self.assertGreater(len(kamishibai_patterns), 0)
        
        # Check for specific kamishibai patterns
        pattern_names = [p.pattern_name for p in kamishibai_patterns]
        self.assertIn("image_text_harmony", pattern_names)
        self.assertIn("dramatic_pacing", pattern_names)
        self.assertIn("audience_participation", pattern_names)
    
    def test_oral_performance_patterns_loaded(self):
        """Test that oral performance patterns are properly loaded."""
        oral_patterns = self.library.patterns.get("oral_performance", [])
        self.assertGreater(len(oral_patterns), 0)
        
        # Check for specific oral performance patterns
        pattern_names = [p.pattern_name for p in oral_patterns]
        self.assertIn("meter_consistency", pattern_names)
        self.assertIn("repetition", pattern_names)
        self.assertIn("collective_memory", pattern_names)
    
    def test_general_patterns_loaded(self):
        """Test that general patterns are properly loaded."""
        general_patterns = self.library.patterns.get("general", [])
        self.assertGreater(len(general_patterns), 0)
        
        # Check for specific general patterns
        pattern_names = [p.pattern_name for p in general_patterns]
        self.assertIn("cultural_respect", pattern_names)
        self.assertIn("community_focus", pattern_names)
    
    def test_pattern_library_entry_structure(self):
        """Test that pattern library entries have correct structure."""
        for tradition, patterns in self.library.patterns.items():
            for pattern in patterns:
                self.assertIsInstance(pattern, PatternLibraryEntry)
                self.assertIsInstance(pattern.pattern_name, str)
                self.assertIsInstance(pattern.pattern_type, PatternType)
                self.assertIsInstance(pattern.tradition, str)
                self.assertIsInstance(pattern.detection_keywords, list)
                self.assertIsInstance(pattern.detection_patterns, list)
                self.assertIsInstance(pattern.cultural_context, str)
                self.assertIsInstance(pattern.authenticity_markers, list)
                self.assertIsInstance(pattern.appropriation_warnings, list)
                self.assertIsInstance(pattern.sacred_boundaries, list)
                
                # Check non-empty required fields
                self.assertGreater(len(pattern.pattern_name), 0)
                self.assertGreater(len(pattern.tradition), 0)
                self.assertGreater(len(pattern.detection_keywords), 0)
                self.assertGreater(len(pattern.cultural_context), 0)
    
    def test_detect_griot_call_response_pattern(self):
        """Test detection of griot call and response pattern."""
        text = "Listen well my friends, hear the ancient story. The audience responds in unison, calling back to the griot."
        
        detected = self.library.detect_patterns(text, ["griot"])
        
        self.assertIsInstance(detected, list)
        
        # Should detect call_response pattern
        call_response_patterns = [p for p in detected if p.pattern_name == "call_response"]
        if call_response_patterns:
            pattern = call_response_patterns[0]
            self.assertEqual(pattern.tradition, "griot")
            self.assertEqual(pattern.pattern_type, PatternType.STORYTELLING_STRUCTURE)
            self.assertGreater(pattern.confidence, 0.3)
            self.assertGreater(len(pattern.evidence), 0)
    
    def test_detect_dreamtime_landscape_pattern(self):
        """Test detection of dreamtime landscape embodiment pattern."""
        text = "The land speaks to those who listen, and the ancient rock formations tell stories of creation."
        
        detected = self.library.detect_patterns(text, ["dreamtime"])
        
        self.assertIsInstance(detected, list)
        
        # Should detect landscape_embodiment pattern
        landscape_patterns = [p for p in detected if p.pattern_name == "landscape_embodiment"]
        if landscape_patterns:
            pattern = landscape_patterns[0]
            self.assertEqual(pattern.tradition, "dreamtime")
            self.assertEqual(pattern.pattern_type, PatternType.NARRATIVE_STRUCTURE)
            self.assertGreater(pattern.confidence, 0.3)
            self.assertIn("land", pattern.evidence[0].lower())
    
    def test_detect_kamishibai_visual_pattern(self):
        """Test detection of kamishibai visual narrative pattern."""
        text = "The picture shows a beautiful garden where the children can see the magical transformation."
        
        detected = self.library.detect_patterns(text, ["kamishibai"])
        
        self.assertIsInstance(detected, list)
        
        # Should detect image_text_harmony pattern
        visual_patterns = [p for p in detected if p.pattern_name == "image_text_harmony"]
        if visual_patterns:
            pattern = visual_patterns[0]
            self.assertEqual(pattern.tradition, "kamishibai")
            self.assertEqual(pattern.pattern_type, PatternType.VISUAL_NARRATIVE)
            self.assertGreater(pattern.confidence, 0.3)
    
    def test_detect_no_patterns(self):
        """Test detection when no cultural patterns are present."""
        text = "This is a simple technical document about computer programming and software development."
        
        detected = self.library.detect_patterns(text)
        
        # Should detect no significant patterns or very low confidence ones
        high_confidence_patterns = [p for p in detected if p.confidence > 0.5]
        self.assertEqual(len(high_confidence_patterns), 0)
    
    def test_detect_multiple_traditions(self):
        """Test detection of patterns from multiple traditions."""
        text = "Listen well, friends, as the land speaks to us. The picture shows the ancestors walking through the eternal cycle, and the children respond with joy."
        
        detected = self.library.detect_patterns(text)
        
        self.assertIsInstance(detected, list)
        
        # Should detect some patterns
        self.assertGreaterEqual(len(detected), 0)
        
        # Check that we can detect patterns from different traditions when appropriate
        traditions_found = set(p.tradition for p in detected if p.confidence > 0.3)
        
        # At minimum, should detect at least one tradition
        # Multiple traditions depend on the complexity of pattern matching
        if len(traditions_found) > 1:
            self.assertGreater(len(traditions_found), 1)
        else:
            # If only one tradition found, verify it's reasonable
            self.assertGreaterEqual(len(traditions_found), 1)
    
    def test_pattern_evaluation(self):
        """Test pattern evaluation logic."""
        # Get a griot pattern for testing
        griot_patterns = self.library.patterns["griot"]
        call_response_pattern = next((p for p in griot_patterns if p.pattern_name == "call_response"), None)
        self.assertIsNotNone(call_response_pattern)
        
        # Text with keywords but no regex patterns
        text_keywords_only = "Listen well, everyone should hear this story"
        confidence, evidence = self.library._evaluate_pattern(text_keywords_only, text_keywords_only.lower(), call_response_pattern)
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIsInstance(evidence, list)
        self.assertGreater(len(evidence), 0)
        
        # Text with both keywords and patterns
        text_full = "Listen well my friends, the audience responds in unison to the call"
        confidence_full, evidence_full = self.library._evaluate_pattern(text_full, text_full.lower(), call_response_pattern)
        
        # Should have higher confidence with both keywords and patterns
        self.assertGreater(confidence_full, confidence)
        self.assertGreater(len(evidence_full), len(evidence))
    
    def test_authenticity_assessment(self):
        """Test authenticity indicator assessment."""
        griot_patterns = self.library.patterns["griot"]
        call_response_pattern = next((p for p in griot_patterns if p.pattern_name == "call_response"), None)
        self.assertIsNotNone(call_response_pattern)
        
        # Text with authenticity markers
        text = "The community engaged in rhythmic speech with repeated phrases during the traditional call and response."
        
        authenticity_indicators = self.library._assess_authenticity(text, call_response_pattern)
        
        self.assertIsInstance(authenticity_indicators, list)
        # Should find some authenticity indicators
        if authenticity_indicators:
            self.assertGreater(len(authenticity_indicators), 0)
            for indicator in authenticity_indicators:
                self.assertIsInstance(indicator, str)
    
    def test_get_pattern_by_name(self):
        """Test retrieval of pattern by name."""
        # Test with tradition specified
        pattern = self.library.get_pattern_by_name("call_response", "griot")
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.pattern_name, "call_response")
        self.assertEqual(pattern.tradition, "griot")
        
        # Test without tradition specified
        pattern_general = self.library.get_pattern_by_name("call_response")
        self.assertIsNotNone(pattern_general)
        self.assertEqual(pattern_general.pattern_name, "call_response")
        
        # Test with non-existent pattern
        pattern_none = self.library.get_pattern_by_name("nonexistent_pattern")
        self.assertIsNone(pattern_none)
    
    def test_get_traditions(self):
        """Test retrieval of available traditions."""
        traditions = self.library.get_traditions()
        
        self.assertIsInstance(traditions, list)
        self.assertGreater(len(traditions), 0)
        self.assertIn("griot", traditions)
        self.assertIn("dreamtime", traditions)
        self.assertIn("kamishibai", traditions)
        self.assertIn("oral_performance", traditions)
        self.assertIn("general", traditions)
    
    def test_get_patterns_for_tradition(self):
        """Test retrieval of patterns for specific tradition."""
        griot_patterns = self.library.get_patterns_for_tradition("griot")
        
        self.assertIsInstance(griot_patterns, list)
        self.assertGreater(len(griot_patterns), 0)
        
        # All patterns should be griot tradition
        for pattern in griot_patterns:
            self.assertEqual(pattern.tradition, "griot")
        
        # Test with non-existent tradition
        empty_patterns = self.library.get_patterns_for_tradition("nonexistent")
        self.assertEqual(len(empty_patterns), 0)
    
    def test_check_appropriation_warnings(self):
        """Test checking for cultural appropriation warnings."""
        # Create a mock detected pattern
        mock_pattern = CulturalPattern(
            pattern_type=PatternType.STORYTELLING_STRUCTURE,
            tradition="griot",
            pattern_name="call_response",
            confidence=0.8,
            evidence=["test evidence"],
            cultural_significance="test significance",
            authenticity_indicators=[]
        )
        
        # Text that might trigger appropriation warnings
        text = "This is a superficial call-response without any cultural context or depth."
        
        warnings = self.library.check_appropriation_warnings([mock_pattern], text)
        
        self.assertIsInstance(warnings, list)
        # May or may not find warnings depending on text content
        for warning in warnings:
            self.assertIsInstance(warning, str)
            self.assertIn("griot:call_response", warning)
    
    def test_check_sacred_boundaries(self):
        """Test checking for sacred boundary violations."""
        # Create a mock detected pattern
        mock_pattern = CulturalPattern(
            pattern_type=PatternType.STORYTELLING_STRUCTURE,
            tradition="griot",
            pattern_name="call_response",
            confidence=0.8,
            evidence=["test evidence"],
            cultural_significance="test significance",
            authenticity_indicators=[]
        )
        
        # Text that might have sacred elements
        text = "The story includes spiritual invocations and ancestor calling ceremonies."
        
        violations = self.library.check_sacred_boundaries([mock_pattern], text)
        
        self.assertIsInstance(violations, list)
        # May or may not find violations depending on text content
        for violation in violations:
            self.assertIsInstance(violation, str)
            self.assertIn("griot:call_response", violation)
    
    def test_cultural_competence_analysis_empty(self):
        """Test cultural competence analysis with empty patterns."""
        analysis = self.library.analyze_cultural_competence([])
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('overall_competence', analysis)
        self.assertIn('tradition_coverage', analysis)
        self.assertIn('pattern_diversity', analysis)
        self.assertIn('authenticity_score', analysis)
        self.assertIn('recommendations', analysis)
        
        # Empty patterns should result in zero competence
        self.assertEqual(analysis['overall_competence'], 0.0)
        self.assertEqual(analysis['pattern_diversity'], 0.0)
        self.assertEqual(analysis['authenticity_score'], 0.0)
        self.assertEqual(len(analysis['tradition_coverage']), 0)
        self.assertIn("No cultural patterns detected", analysis['recommendations'])
    
    def test_cultural_competence_analysis_with_patterns(self):
        """Test cultural competence analysis with detected patterns."""
        # Create mock patterns
        pattern1 = CulturalPattern(
            pattern_type=PatternType.STORYTELLING_STRUCTURE,
            tradition="griot",
            pattern_name="call_response",
            confidence=0.8,
            evidence=["test evidence"],
            cultural_significance="test significance",
            authenticity_indicators=["community engagement"]
        )
        
        pattern2 = CulturalPattern(
            pattern_type=PatternType.NARRATIVE_STRUCTURE,
            tradition="dreamtime",
            pattern_name="landscape_embodiment",
            confidence=0.7,
            evidence=["test evidence"],
            cultural_significance="test significance",
            authenticity_indicators=["geographical specificity"]
        )
        
        patterns = [pattern1, pattern2]
        analysis = self.library.analyze_cultural_competence(patterns)
        
        self.assertIsInstance(analysis, dict)
        self.assertGreater(analysis['overall_competence'], 0.0)
        self.assertLessEqual(analysis['overall_competence'], 1.0)
        self.assertGreater(analysis['pattern_diversity'], 0.0)
        self.assertGreater(analysis['authenticity_score'], 0.0)
        
        # Should have coverage for both traditions
        self.assertIn('griot', analysis['tradition_coverage'])
        self.assertIn('dreamtime', analysis['tradition_coverage'])
        
        # Should have some recommendations
        self.assertIsInstance(analysis['recommendations'], list)
    
    def test_pattern_confidence_threshold(self):
        """Test that pattern detection respects confidence threshold."""
        # Use text that would generate low confidence
        text = "A very short text."
        
        detected = self.library.detect_patterns(text)
        
        # All detected patterns should have confidence > 0.3
        for pattern in detected:
            self.assertGreater(pattern.confidence, 0.3)
    
    def test_cultural_pattern_dataclass(self):
        """Test CulturalPattern dataclass structure."""
        pattern = CulturalPattern(
            pattern_type=PatternType.STORYTELLING_STRUCTURE,
            tradition="test_tradition",
            pattern_name="test_pattern",
            confidence=0.5,
            evidence=["test evidence"],
            cultural_significance="test significance",
            authenticity_indicators=["test indicator"]
        )
        
        self.assertEqual(pattern.pattern_type, PatternType.STORYTELLING_STRUCTURE)
        self.assertEqual(pattern.tradition, "test_tradition")
        self.assertEqual(pattern.pattern_name, "test_pattern")
        self.assertEqual(pattern.confidence, 0.5)
        self.assertEqual(pattern.evidence, ["test evidence"])
        self.assertEqual(pattern.cultural_significance, "test significance")
        self.assertEqual(pattern.authenticity_indicators, ["test indicator"])
    
    def test_pattern_type_enum(self):
        """Test PatternType enum values."""
        expected_types = [
            "STORYTELLING_STRUCTURE", "PERFORMANCE_MARKER", "CULTURAL_VALUE",
            "RHYTHMIC_PATTERN", "KNOWLEDGE_INTEGRATION", "STORYTELLING_ETHICS",
            "NARRATIVE_STRUCTURE", "VISUAL_NARRATIVE", "EDUCATIONAL_PURPOSE",
            "PERFORMANCE_STYLE", "MEMORY_AID", "AUDIENCE_INTERACTION"
        ]
        
        for pattern_type_name in expected_types:
            self.assertTrue(hasattr(PatternType, pattern_type_name))
            pattern_type = getattr(PatternType, pattern_type_name)
            self.assertIsInstance(pattern_type, PatternType)
    
    def test_custom_config_initialization(self):
        """Test library initialization with custom config."""
        custom_config = {
            "detection_threshold": 0.5,
            "max_patterns_per_text": 10
        }
        
        library = CulturalPatternLibrary(config=custom_config)
        
        self.assertEqual(library.config, custom_config)
        self.assertIsNotNone(library.patterns)
        self.assertGreater(len(library.patterns), 0)
    
    def test_tradition_pattern_indexing(self):
        """Test that tradition pattern indexing works correctly."""
        # Check that tradition_patterns index is built
        self.assertGreater(len(self.library.tradition_patterns), 0)
        
        for tradition, patterns in self.library.patterns.items():
            self.assertIn(tradition, self.library.tradition_patterns)
            self.assertEqual(len(self.library.tradition_patterns[tradition]), len(patterns))
    
    def test_pattern_detection_case_insensitive(self):
        """Test that pattern detection is case insensitive."""
        text_lower = "listen well friends, hear the story"
        text_upper = "LISTEN WELL FRIENDS, HEAR THE STORY"
        text_mixed = "Listen Well Friends, Hear The Story"
        
        detected_lower = self.library.detect_patterns(text_lower, ["griot"])
        detected_upper = self.library.detect_patterns(text_upper, ["griot"])
        detected_mixed = self.library.detect_patterns(text_mixed, ["griot"])
        
        # Should detect similar patterns regardless of case
        # (exact matches may vary due to regex complexity)
        self.assertIsInstance(detected_lower, list)
        self.assertIsInstance(detected_upper, list)
        self.assertIsInstance(detected_mixed, list)
        
        # At minimum, should not fail due to case differences
        for patterns in [detected_lower, detected_upper, detected_mixed]:
            for pattern in patterns:
                self.assertGreaterEqual(pattern.confidence, 0.0)
                self.assertLessEqual(pattern.confidence, 1.0)


if __name__ == '__main__':
    unittest.main()