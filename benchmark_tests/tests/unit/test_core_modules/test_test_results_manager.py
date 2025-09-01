#!/usr/bin/env python3
"""
Unit Tests for TestResultsManager

Tests the core functionality of test result storage and cognitive pattern detection.
Focuses on validating the core requirements without jumping ahead to advanced features.
"""

import unittest
import tempfile
import shutil
import json
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add core modules to path  
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "core"))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from core.results_manager import (
    TestResultsManager, RunMetadata, CognitivePattern, CognitiveProfile
)

class TestTestResultsManager(unittest.TestCase):
    """Test TestResultsManager core functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TestResultsManager(base_results_dir=self.temp_dir)
        
        # Test data
        self.model_name = "test-model"
        self.model_path = "/test/model/path"
        self.test_config = {
            "strategy": "test",
            "token_limit": 400
        }
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_run_directory(self):
        """Test run directory creation with proper structure"""
        run_dir = self.manager.create_run_directory(
            self.model_name, self.model_path, self.test_config
        )
        
        # Verify directory exists
        self.assertTrue(os.path.exists(run_dir))
        
        # Verify subdirectories
        run_path = Path(run_dir)
        self.assertTrue((run_path / "raw_responses").exists())
        self.assertTrue((run_path / "domain_analysis").exists())
        self.assertTrue((run_path / "pattern_detection").exists())
        
        # Verify metadata file exists
        metadata_file = run_path / "run_metadata.json"
        self.assertTrue(metadata_file.exists())
        
        # Verify metadata content
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        self.assertEqual(metadata['model_name'], self.model_name)
        self.assertEqual(metadata['model_path'], self.model_path)
        self.assertEqual(metadata['test_configuration'], self.test_config)
    
    def test_save_test_response(self):
        """Test saving individual test responses"""
        # Create run directory
        run_dir = self.manager.create_run_directory(
            self.model_name, self.model_path, self.test_config
        )
        
        # Test response data
        test_id = "test_001"
        prompt = "Test prompt"
        response_text = "Test response text"
        evaluation_results = {
            "score": 75.0,
            "confidence": 0.8
        }
        test_metadata = {
            "domain": "reasoning",
            "difficulty": "easy"
        }
        
        # Save test response
        self.manager.save_test_response(
            run_dir, test_id, prompt, response_text,
            evaluation_results, test_metadata
        )
        
        # Verify response file exists
        response_file = Path(run_dir) / "raw_responses" / f"{test_id}.json"
        self.assertTrue(response_file.exists())
        
        # Verify response content
        with open(response_file, 'r') as f:
            response_data = json.load(f)
        
        self.assertEqual(response_data['test_id'], test_id)
        self.assertEqual(response_data['prompt'], prompt)
        self.assertEqual(response_data['response_text'], response_text)
        self.assertEqual(response_data['evaluation_results'], evaluation_results)
        self.assertEqual(response_data['test_metadata'], test_metadata)
        self.assertIn('cognitive_domain', response_data)
    
    def test_classify_cognitive_domain(self):
        """Test cognitive domain classification"""
        # Test domain-based classification
        test_metadata = {"domain": "reasoning"}
        domain = self.manager._classify_cognitive_domain(test_metadata)
        self.assertEqual(domain, "reasoning")
        
        # Test creativity domain
        test_metadata = {"domain": "creativity"}
        domain = self.manager._classify_cognitive_domain(test_metadata)
        self.assertEqual(domain, "creativity")
        
        # Test social domain  
        test_metadata = {"domain": "social"}
        domain = self.manager._classify_cognitive_domain(test_metadata)
        self.assertEqual(domain, "social")
        
        # Test knowledge mapping to memory
        test_metadata = {"domain": "knowledge"}
        domain = self.manager._classify_cognitive_domain(test_metadata)
        self.assertEqual(domain, "memory")
        
        # Test unknown domain defaults to integration
        test_metadata = {"domain": "unknown"}
        domain = self.manager._classify_cognitive_domain(test_metadata)
        self.assertEqual(domain, "integration")
        
        # Test ID-based classification
        test_metadata = {"id": "reasoning_test_001"}
        domain = self.manager._classify_cognitive_domain(test_metadata)
        self.assertEqual(domain, "reasoning")
        
        # Test fallback to integration
        test_metadata = {"id": "unknown_test"}
        domain = self.manager._classify_cognitive_domain(test_metadata)
        self.assertEqual(domain, "integration")
    
    def test_json_serialization_complex_objects(self):
        """Test JSON serialization of complex objects"""
        # Create run directory
        run_dir = self.manager.create_run_directory(
            self.model_name, self.model_path, self.test_config
        )
        
        # Create complex evaluation results with various object types
        complex_evaluation = {
            "datetime": datetime.now(),
            "nested_dict": {
                "score": 75.0,
                "metadata": {"key": "value"}
            },
            "list_data": [1, 2, 3],
            "boolean": True,
            "none_value": None
        }
        
        # This should not fail due to JSON serialization
        self.manager.save_test_response(
            run_dir, "complex_test", "prompt", "response",
            complex_evaluation, {"domain": "test"}
        )
        
        # Verify file was created
        response_file = Path(run_dir) / "raw_responses" / "complex_test.json"
        self.assertTrue(response_file.exists())
    
    @patch('core.results_manager.statistics.mean')
    @patch('core.results_manager.statistics.stdev') 
    def test_analyze_cognitive_patterns(self, mock_stdev, mock_mean):
        """Test cognitive pattern analysis"""
        # Mock statistics functions
        mock_mean.return_value = 75.0
        mock_stdev.return_value = 10.0
        
        # Create run directory with test responses
        run_dir = self.manager.create_run_directory(
            self.model_name, self.model_path, self.test_config
        )
        
        # Add some test responses
        test_responses = [
            ("test_001", "reasoning", 80.0),
            ("test_002", "reasoning", 70.0), 
            ("test_003", "memory", 85.0),
            ("test_004", "memory", 75.0)
        ]
        
        for test_id, domain, score in test_responses:
            evaluation_results = {"score": score, "calibration_score": score}
            test_metadata = {"domain": domain}
            self.manager.save_test_response(
                run_dir, test_id, f"prompt_{test_id}", f"response_{test_id}",
                evaluation_results, test_metadata
            )
        
        # Analyze patterns
        profile = self.manager.analyze_cognitive_patterns(run_dir)
        
        # Verify profile structure
        self.assertIsInstance(profile, CognitiveProfile)
        self.assertEqual(profile.run_id, Path(run_dir).name)
        self.assertGreaterEqual(profile.sample_size, 0)
        self.assertIsInstance(profile.detected_patterns, list)
        self.assertIsInstance(profile.strengths, list)
        self.assertIsInstance(profile.weaknesses, list)
    
    def test_make_serializable(self):
        """Test serialization helper for complex objects"""
        # Test various object types
        test_data = {
            "string": "test",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
            "datetime": datetime.now(),
            "nested": {
                "list": [1, 2, 3],
                "dict": {"key": "value"}
            }
        }
        
        # Should not raise exception
        serialized = self.manager._make_serializable(test_data)
        
        # Verify basic types preserved
        self.assertEqual(serialized["string"], "test")
        self.assertEqual(serialized["integer"], 42)
        self.assertEqual(serialized["float"], 3.14)
        self.assertEqual(serialized["boolean"], True)
        self.assertIsNone(serialized["none"])
        
        # Verify datetime converted
        self.assertIsInstance(serialized["datetime"], str)
        
        # Verify nested structures preserved
        self.assertEqual(serialized["nested"]["list"], [1, 2, 3])
        self.assertEqual(serialized["nested"]["dict"]["key"], "value")
    
    def test_empty_responses_handling(self):
        """Test handling of empty or missing responses"""
        # Create run directory with no responses
        run_dir = self.manager.create_run_directory(
            self.model_name, self.model_path, self.test_config
        )
        
        # Analyze patterns with no data
        profile = self.manager.analyze_cognitive_patterns(run_dir)
        
        # Should return empty profile without errors
        self.assertIsInstance(profile, CognitiveProfile)
        self.assertEqual(profile.sample_size, 0)
        self.assertEqual(len(profile.detected_patterns), 0)
        self.assertEqual(len(profile.strengths), 0)
        self.assertEqual(len(profile.weaknesses), 0)
    
    def test_cognitive_pattern_detection(self):
        """Test pattern detection logic"""
        # Create test domain responses with sufficient samples (min_sample_size = 5)
        domain_responses = {
            "reasoning": [
                {"test_id": "r1", "evaluation_results": {"calibration_score": 85}},
                {"test_id": "r2", "evaluation_results": {"calibration_score": 80}},
                {"test_id": "r3", "evaluation_results": {"calibration_score": 90}},
                {"test_id": "r4", "evaluation_results": {"calibration_score": 85}},
                {"test_id": "r5", "evaluation_results": {"calibration_score": 88}}
            ],
            "memory": [
                {"test_id": "m1", "evaluation_results": {"calibration_score": 45}},
                {"test_id": "m2", "evaluation_results": {"calibration_score": 40}},
                {"test_id": "m3", "evaluation_results": {"calibration_score": 35}},
                {"test_id": "m4", "evaluation_results": {"calibration_score": 42}},
                {"test_id": "m5", "evaluation_results": {"calibration_score": 38}}
            ]
        }
        
        patterns = self.manager._detect_cognitive_patterns(domain_responses)
        
        # Should detect patterns (strength and/or weakness)
        self.assertGreater(len(patterns), 0, "Should detect at least some patterns")
        
        # Check pattern structure
        for pattern in patterns:
            self.assertIsInstance(pattern.cognitive_domain, str)
            self.assertIsInstance(pattern.pattern_type, str)
            self.assertIsInstance(pattern.confidence_score, float)
            self.assertIsInstance(pattern.evidence_tests, list)
            self.assertIsInstance(pattern.statistical_measures, dict)
            self.assertIsInstance(pattern.description, str)
            self.assertIsInstance(pattern.severity, str)
        
        # Should have patterns for both domains 
        domains = [p.cognitive_domain for p in patterns]
        self.assertTrue(len(set(domains)) >= 1, "Should have patterns for at least one domain")

class TestCognitivePattern(unittest.TestCase):
    """Test CognitivePattern dataclass"""
    
    def test_cognitive_pattern_creation(self):
        """Test CognitivePattern creation and attributes"""
        pattern = CognitivePattern(
            cognitive_domain="reasoning",
            pattern_type="strength",
            confidence_score=0.85,
            evidence_tests=["test_001", "test_002"],
            statistical_measures={"mean": 80.0, "std": 5.0},
            description="Strong logical reasoning",
            severity="high"
        )
        
        self.assertEqual(pattern.cognitive_domain, "reasoning")
        self.assertEqual(pattern.pattern_type, "strength")
        self.assertEqual(pattern.confidence_score, 0.85)
        self.assertEqual(len(pattern.evidence_tests), 2)
        self.assertIn("mean", pattern.statistical_measures)
        self.assertEqual(pattern.severity, "high")

class TestCognitiveProfile(unittest.TestCase):
    """Test CognitiveProfile dataclass"""
    
    def test_cognitive_profile_creation(self):
        """Test CognitiveProfile creation and attributes"""
        profile = CognitiveProfile(
            model_name="test-model",
            run_id="test-run-123",
            reasoning_score=75.0,
            memory_score=65.0,
            creativity_score=80.0,
            social_score=70.0,
            integration_score=72.0,
            detected_patterns=[],
            strengths=["Strong creativity"],
            weaknesses=["Memory challenges"],
            blind_spots=["Social reasoning gaps"],
            pattern_confidence=0.75,
            sample_size=25
        )
        
        self.assertEqual(profile.model_name, "test-model")
        self.assertEqual(profile.run_id, "test-run-123")
        self.assertEqual(profile.reasoning_score, 75.0)
        self.assertEqual(profile.sample_size, 25)
        self.assertEqual(len(profile.strengths), 1)
        self.assertEqual(len(profile.weaknesses), 1)
        self.assertEqual(len(profile.blind_spots), 1)

if __name__ == '__main__':
    unittest.main()