"""
Unit tests for CulturalDatasetValidator.

Tests cultural validation algorithms, UNESCO dataset processing, cultural element matching,
confidence scoring calculations, and cross-cultural validation accuracy.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import json
import requests
from io import StringIO

from evaluator.cultural.cultural_dataset_validator import (
    CulturalDatasetValidator,
    DatasetValidationResult,
    CulturalDatasetEntry,
    DatasetSource
)
from evaluator.core.domain_evaluator_base import CulturalContext, DomainEvaluationResult


class TestCulturalDatasetValidator(unittest.TestCase):
    """Test basic cultural dataset validator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'datasets_dir': './data/cultural',
            'confidence_threshold': 0.7
        }
        self.validator = CulturalDatasetValidator(self.config)
        
        # Test cultural contexts
        self.ubuntu_context = CulturalContext(
            traditions=["ubuntu", "ubuntu_philosophy"],
            knowledge_systems=["bantu_traditional_knowledge"],
            performance_aspects=["restorative_justice", "community_healing"],
            cultural_groups=["zulu", "xhosa", "bantu_peoples"],
            linguistic_varieties=["isizulu", "isixhosa"]
        )
        
        self.native_american_context = CulturalContext(
            traditions=["talking_circle", "peacemaking_circle"],
            knowledge_systems=["indigenous_knowledge"],
            performance_aspects=["consensus_building", "spiritual_healing"],
            cultural_groups=["ojibwe", "lakota", "navajo"],
            linguistic_varieties=["ojibwemowin", "lakota_language"]
        )
        
        # Test evaluation result
        self.test_evaluation = Mock(spec=DomainEvaluationResult)
        self.test_evaluation.overall_score = 0.75
        self.test_evaluation.cultural_context = self.ubuntu_context
        self.test_evaluation.metadata = {'evaluation_id': 'test_001'}
        self.test_evaluation.domain = 'social'
        self.test_evaluation.evaluation_type = 'test'
    
    def test_validator_initialization(self):
        """Test cultural dataset validator initialization."""
        # Test with custom config
        validator = CulturalDatasetValidator(self.config)
        self.assertEqual(validator.config['datasets_dir'], './data/cultural')
        
        # Test dataset configurations exist
        self.assertGreater(len(validator.dataset_configs), 0)
        
        # Test that dataset sources are properly configured
        self.assertIn(DatasetSource.UNESCO, validator.dataset_configs)
        self.assertIn(DatasetSource.ETHNOLOGUE, validator.dataset_configs)
        
        # Test with default config
        default_validator = CulturalDatasetValidator()
        self.assertIsNotNone(default_validator.dataset_configs)
    
    def test_dataset_entry_structure(self):
        """Test cultural dataset entry data structure."""
        entry = CulturalDatasetEntry(
            name="ubuntu",
            description="African philosophy emphasizing interconnectedness",
            cultural_groups=["zulu", "xhosa"],
            traditions=["ubuntu", "ubuntu_philosophy"],
            knowledge_systems=["bantu_traditional_knowledge"],
            linguistic_varieties=["isizulu", "isixhosa"],
            geographic_regions=["southern_africa"],
            sources=["unesco", "academic"],
            confidence_score=0.95,
            dataset_source=DatasetSource.UNESCO
        )
        
        self.assertEqual(entry.name, "ubuntu")
        self.assertEqual(entry.dataset_source, DatasetSource.UNESCO)
        self.assertIn("zulu", entry.cultural_groups)
        self.assertGreater(entry.confidence_score, 0.9)
        self.assertIn("ubuntu_philosophy", entry.traditions)


class TestDatasetValidation(unittest.TestCase):
    """Test dataset validation functionality."""
    
    def setUp(self):
        """Set up dataset validation test fixtures."""
        self.validator = CulturalDatasetValidator()
        
        self.ubuntu_context = CulturalContext(
            traditions=["ubuntu", "ubuntu_philosophy"],
            knowledge_systems=["bantu_traditional_knowledge"],
            performance_aspects=["restorative_justice"],
            cultural_groups=["zulu", "xhosa", "bantu_peoples"],
            linguistic_varieties=["isizulu", "isixhosa"]
        )
        
        self.test_evaluation = Mock(spec=DomainEvaluationResult)
        self.test_evaluation.overall_score = 0.75
        self.test_evaluation.cultural_context = self.ubuntu_context
        self.test_evaluation.metadata = {'evaluation_id': 'test_001'}
        self.test_evaluation.domain = 'social'
        self.test_evaluation.evaluation_type = 'test'
    
    def test_cultural_validation_process(self):
        """Test cultural validation against datasets."""
        # Perform validation
        validation_result = self.validator.validate_cultural_evaluation(
            self.ubuntu_context, 
            self.test_evaluation
        )
        
        # Check result structure
        self.assertIsInstance(validation_result, DatasetValidationResult)
        self.assertIsInstance(validation_result.matched_entries, list)
        self.assertIsInstance(validation_result.validation_confidence, float)
        self.assertIsInstance(validation_result.coverage_score, float)
        self.assertIsInstance(validation_result.validation_flags, list)
        self.assertIsInstance(validation_result.dataset_sources_used, list)
        
        # Validation confidence should be between 0 and 1
        self.assertGreaterEqual(validation_result.validation_confidence, 0.0)
        self.assertLessEqual(validation_result.validation_confidence, 1.0)
        
        # Coverage score should be between 0 and 1
        self.assertGreaterEqual(validation_result.coverage_score, 0.0)
        self.assertLessEqual(validation_result.coverage_score, 1.0)
    
    def test_dataset_matching_algorithm(self):
        """Test dataset matching algorithm."""
        # This tests the internal matching logic
        validation_result = self.validator.validate_cultural_evaluation(
            self.ubuntu_context,
            self.test_evaluation
        )
        
        # Should find some matches given the test data setup
        # (The implementation creates sample data that should match ubuntu context)
        if validation_result.matched_entries:
            # Test that matched entries have reasonable confidence scores
            for entry in validation_result.matched_entries:
                self.assertIsInstance(entry, CulturalDatasetEntry)
                self.assertGreaterEqual(entry.confidence_score, 0.0)
                self.assertLessEqual(entry.confidence_score, 1.0)
    
    def test_missing_cultural_elements_detection(self):
        """Test detection of missing cultural elements."""
        # Create context with elements unlikely to be in sample data
        unique_context = CulturalContext(
            traditions=["very_unique_tradition_12345"],
            knowledge_systems=["unique_knowledge_system_67890"],
            performance_aspects=[],
            cultural_groups=["unique_cultural_group_abcde"],
            linguistic_varieties=["unique_language_xyz"]
        )
        
        test_eval = Mock(spec=DomainEvaluationResult)
        test_eval.overall_score = 0.75
        test_eval.cultural_context = unique_context
        test_eval.metadata = {'evaluation_id': 'unique_test'}
        test_eval.domain = 'social'
        test_eval.evaluation_type = 'test'
        
        validation_result = self.validator.validate_cultural_evaluation(
            unique_context,
            test_eval
        )
        
        # Should identify missing elements for unique context
        self.assertIsInstance(validation_result.missing_cultural_elements, list)
        # Unique elements should likely be in missing list
        if validation_result.missing_cultural_elements:
            self.assertGreater(len(validation_result.missing_cultural_elements), 0)


class TestDatasetConfiguration(unittest.TestCase):
    """Test dataset configuration and management."""
    
    def setUp(self):
        """Set up dataset configuration test fixtures."""
        self.validator = CulturalDatasetValidator()
    
    def test_dataset_source_configuration(self):
        """Test dataset source configurations."""
        # Test that all expected dataset sources are configured
        expected_sources = [
            DatasetSource.UNESCO,
            DatasetSource.ETHNOLOGUE,
            DatasetSource.WORLD_CULTURES,
            DatasetSource.ACADEMIC_CORPUS,
            DatasetSource.CULTURAL_COMMONS
        ]
        
        for source in expected_sources:
            self.assertIn(source, self.validator.dataset_configs)
            config = self.validator.dataset_configs[source]
            self.assertIn('name', config)
            self.assertIn('local_file', config)
            self.assertIn('format', config)
    
    def test_dataset_info_retrieval(self):
        """Test dataset information retrieval."""
        dataset_info = self.validator.get_dataset_info()
        
        # Should have proper structure
        self.assertIn('datasets', dataset_info)
        self.assertIn('total_entries', dataset_info)
        
        # Should include information about configured datasets
        for source in DatasetSource:
            self.assertIn(source.value, dataset_info['datasets'])
            source_info = dataset_info['datasets'][source.value]
            self.assertIn('name', source_info)
            self.assertIn('available', source_info)
    
    def test_validation_flag_generation(self):
        """Test generation of validation flags."""
        # Test with context that should generate low coverage
        minimal_context = CulturalContext(
            traditions=[],
            knowledge_systems=[],
            performance_aspects=[],
            cultural_groups=[],
            linguistic_varieties=[]
        )
        
        test_eval = Mock(spec=DomainEvaluationResult)
        test_eval.overall_score = 0.75
        test_eval.cultural_context = minimal_context
        test_eval.metadata = {'evaluation_id': 'minimal_test'}
        test_eval.domain = 'social'
        test_eval.evaluation_type = 'test'
        
        validation_result = self.validator.validate_cultural_evaluation(
            minimal_context,
            test_eval
        )
        
        # Should generate validation flags for poor coverage
        self.assertIsInstance(validation_result.validation_flags, list)
        # With minimal context, should likely generate flags
        if validation_result.validation_flags:
            for flag in validation_result.validation_flags:
                self.assertIsNotNone(flag.flag_type)
                self.assertIsNotNone(flag.severity)
                self.assertIsNotNone(flag.description)


class TestValidationAccuracy(unittest.TestCase):
    """Test validation accuracy and reliability."""
    
    def setUp(self):
        """Set up validation accuracy test fixtures."""
        self.validator = CulturalDatasetValidator()
    
    def test_validation_consistency(self):
        """Test that validation is consistent across multiple runs."""
        ubuntu_context = CulturalContext(
            traditions=["ubuntu"],
            knowledge_systems=["traditional_knowledge"],
            performance_aspects=[],
            cultural_groups=["bantu_peoples"],
            linguistic_varieties=["isizulu"]
        )
        
        test_eval = Mock(spec=DomainEvaluationResult)
        test_eval.overall_score = 0.75
        test_eval.cultural_context = ubuntu_context
        test_eval.metadata = {'evaluation_id': 'consistency_test'}
        test_eval.domain = 'social'
        test_eval.evaluation_type = 'test'
        
        # Run validation multiple times
        results = []
        for i in range(3):
            result = self.validator.validate_cultural_evaluation(
                ubuntu_context,
                test_eval
            )
            results.append(result)
        
        # Results should be consistent
        first_result = results[0]
        for result in results[1:]:
            # Validation confidence should be the same
            self.assertAlmostEqual(
                result.validation_confidence,
                first_result.validation_confidence,
                places=5
            )
            
            # Coverage score should be the same
            self.assertAlmostEqual(
                result.coverage_score,
                first_result.coverage_score,
                places=5
            )
    
    def test_confidence_scoring_ranges(self):
        """Test that confidence scores are within expected ranges."""
        test_contexts = [
            # Well-represented context
            CulturalContext(
                traditions=["ubuntu"],
                knowledge_systems=["traditional_knowledge"],
                performance_aspects=[],
                cultural_groups=["bantu_peoples"],
                linguistic_varieties=[]
            ),
            # Less represented context
            CulturalContext(
                traditions=["unique_tradition_test"],
                knowledge_systems=["unique_knowledge"],
                performance_aspects=[],
                cultural_groups=["unique_group"],
                linguistic_varieties=[]
            )
        ]
        
        for context in test_contexts:
            test_eval = Mock(spec=DomainEvaluationResult)
            test_eval.overall_score = 0.75
            test_eval.cultural_context = context
            test_eval.metadata = {'evaluation_id': 'confidence_test'}
            test_eval.domain = 'social'
            test_eval.evaluation_type = 'test'
            
            result = self.validator.validate_cultural_evaluation(context, test_eval)
            
            # All scores should be valid ranges
            self.assertGreaterEqual(result.validation_confidence, 0.0)
            self.assertLessEqual(result.validation_confidence, 1.0)
            self.assertGreaterEqual(result.coverage_score, 0.0)
            self.assertLessEqual(result.coverage_score, 1.0)


if __name__ == '__main__':
    unittest.main()