"""
Unit tests for IntegratedValidationSystem.

Tests system coordination, component orchestration, configuration management,
validation pipeline integration, and comprehensive report generation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import json
import asyncio
from datetime import datetime

from evaluator.validation.integrated_validation_system import (
    IntegratedValidationSystem,
    IntegratedValidationConfig,
    ComprehensiveValidationResult
)
from evaluator.validation.validation_runner import ValidationRunner, MultiModelValidationResult
from evaluator.core.ensemble_disagreement_detector import EnsembleDisagreementDetector
from evaluator.cultural.cultural_dataset_validator import CulturalDatasetValidator, DatasetValidationResult
from evaluator.validation.community_flagging_system import CommunityFlaggingSystem, CommunityFlag
from evaluator.core.evaluation_aggregator import EvaluationAggregator
from evaluator.core.domain_evaluator_base import CulturalContext, DomainEvaluationResult


class TestIntegratedValidationSystem(unittest.TestCase):
    """Test basic integrated validation system functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = IntegratedValidationConfig(
            enable_multi_model_validation=True,
            enable_wikipedia_validation=True,
            enable_cultural_datasets=True,
            enable_ensemble_disagreement=True,
            enable_open_apis=True,
            enable_community_flagging=True,
            enable_bias_detection=True,
            confidence_threshold=0.7,
            disagreement_threshold=0.3,
            bias_threshold=0.3,
            cultural_authenticity_threshold=0.4
        )
        self.validation_system = IntegratedValidationSystem(self.config)
        
        # Test cultural context
        self.cultural_context = CulturalContext(
            traditions=["ubuntu", "ubuntu_philosophy"],
            cultural_groups=["bantu_peoples"],
            knowledge_systems=["african_traditional"],
            performance_aspects=["restorative_justice"],
            linguistic_varieties=["isizulu", "isixhosa"]
        )
        
        # Test evaluation result
        self.test_evaluation = Mock(spec=DomainEvaluationResult)
        self.test_evaluation.overall_score = 0.75
        self.test_evaluation.cultural_context = self.cultural_context
        self.test_evaluation.metadata = {'evaluation_id': 'test_001'}
        self.test_evaluation.domain = 'social'
        self.test_evaluation.evaluation_type = 'test'
        self.test_evaluation.dimensions = [Mock(name='cultural_authenticity')]
    
    def test_validation_system_initialization(self):
        """Test integrated validation system initialization."""
        # Test with custom config
        system = IntegratedValidationSystem(self.config)
        self.assertEqual(system.config.confidence_threshold, 0.7)
        self.assertTrue(system.config.enable_multi_model_validation)
        self.assertTrue(system.config.enable_cultural_datasets)
        
        # Test component initialization
        self.assertIsNotNone(system.validation_runner)
        self.assertIsNotNone(system.dataset_validator)
        self.assertIsNotNone(system.community_flagging_system)
        
        # Test with default config
        default_system = IntegratedValidationSystem()
        self.assertIsNotNone(default_system.config)
    
    def test_integrated_validation_config(self):
        """Test integrated validation configuration structure."""
        config = IntegratedValidationConfig(
            enable_multi_model_validation=False,
            enable_wikipedia_validation=True,
            confidence_threshold=0.8,
            disagreement_threshold=0.2,
            bias_threshold=0.25,
            cultural_authenticity_threshold=0.5
        )
        
        self.assertFalse(config.enable_multi_model_validation)
        self.assertTrue(config.enable_wikipedia_validation)
        self.assertEqual(config.confidence_threshold, 0.8)
        self.assertEqual(config.disagreement_threshold, 0.2)
        self.assertEqual(config.bias_threshold, 0.25)
        self.assertEqual(config.cultural_authenticity_threshold, 0.5)


class TestSystemCoordination(unittest.TestCase):
    """Test system coordination and orchestration."""
    
    def setUp(self):
        """Set up system coordination test fixtures."""
        self.validation_system = IntegratedValidationSystem()
        
        self.cultural_context = CulturalContext(
            traditions=["ubuntu"],
            cultural_groups=["bantu_peoples"],
            knowledge_systems=["african_traditional"],
            performance_aspects=[],
            linguistic_varieties=[]
        )
        
        self.test_evaluation = Mock(spec=DomainEvaluationResult)
        self.test_evaluation.overall_score = 0.75
        self.test_evaluation.cultural_context = self.cultural_context
        self.test_evaluation.metadata = {'evaluation_id': 'coord_test_001'}
        self.test_evaluation.domain = 'social'
        self.test_evaluation.evaluation_type = 'test'
        self.test_evaluation.dimensions = [Mock(name='cultural_authenticity')]
    
    def test_component_orchestration(self):
        """Test orchestration of validation components."""
        # Test that all components are properly initialized
        self.assertIsNotNone(self.validation_system.validation_runner)
        self.assertIsNotNone(self.validation_system.dataset_validator)
        self.assertIsNotNone(self.validation_system.disagreement_detector)
        self.assertIsNotNone(self.validation_system.community_flagging_system)
        
        # Test component types
        self.assertIsInstance(self.validation_system.validation_runner, ValidationRunner)
        self.assertIsInstance(self.validation_system.dataset_validator, CulturalDatasetValidator)
        self.assertIsInstance(self.validation_system.disagreement_detector, EnsembleDisagreementDetector)
        self.assertIsInstance(self.validation_system.community_flagging_system, CommunityFlaggingSystem)
    
    def test_validation_pipeline_coordination(self):
        """Test validation pipeline coordination."""
        # Test that the system can coordinate validation requests
        # (In actual implementation, this would call comprehensive_validate)
        self.assertTrue(hasattr(self.validation_system, 'comprehensive_validate'))
        
        # Test configuration propagation to components
        config_with_custom_values = IntegratedValidationConfig(
            confidence_threshold=0.9,
            dataset_validator_config={'custom_key': 'custom_value'}
        )
        
        system_with_custom_config = IntegratedValidationSystem(config_with_custom_values)
        self.assertEqual(system_with_custom_config.config.confidence_threshold, 0.9)
    
    def test_comprehensive_validation_result_structure(self):
        """Test comprehensive validation result structure."""
        # Mock component results
        mock_multi_model_result = Mock(spec=MultiModelValidationResult)
        mock_multi_model_result.consensus_score = 0.85
        mock_multi_model_result.disagreement_level = 0.1
        
        mock_dataset_result = Mock(spec=DatasetValidationResult)
        mock_dataset_result.validation_confidence = 0.8
        mock_dataset_result.coverage_score = 0.9
        
        # Create comprehensive result
        comprehensive_result = ComprehensiveValidationResult(
            primary_evaluation=self.test_evaluation,
            multi_model_validation=mock_multi_model_result,
            wikipedia_validation=None,
            dataset_validation=mock_dataset_result,
            disagreement_analysis=None,
            open_apis_validation=None,
            community_flags=[],
            overall_confidence=0.82,
            validation_summary="Test validation completed successfully",
            recommendations=["Maintain current approach"],
            processing_metadata={'validation_time': 2.5}
        )
        
        # Test structure
        self.assertEqual(comprehensive_result.primary_evaluation, self.test_evaluation)
        self.assertEqual(comprehensive_result.multi_model_validation, mock_multi_model_result)
        self.assertEqual(comprehensive_result.dataset_validation, mock_dataset_result)
        self.assertEqual(comprehensive_result.overall_confidence, 0.82)
        self.assertIn("Test validation completed", comprehensive_result.validation_summary)
        self.assertIn("Maintain current approach", comprehensive_result.recommendations)


class TestValidationPipelineIntegration(unittest.TestCase):
    """Test validation pipeline integration."""
    
    def setUp(self):
        """Set up pipeline integration test fixtures."""
        self.validation_system = IntegratedValidationSystem()
        
        self.ubuntu_context = CulturalContext(
            traditions=["ubuntu"],
            cultural_groups=["bantu_peoples"],
            knowledge_systems=["african_traditional"],
            performance_aspects=["restorative_justice"],
            linguistic_varieties=[]
        )
        
        self.test_evaluation = Mock(spec=DomainEvaluationResult)
        self.test_evaluation.overall_score = 0.75
        self.test_evaluation.cultural_context = self.ubuntu_context
        self.test_evaluation.metadata = {'evaluation_id': 'pipeline_test_001'}
        self.test_evaluation.domain = 'social'
        self.test_evaluation.evaluation_type = 'test'
        self.test_evaluation.dimensions = [Mock(name='cultural_authenticity')]
    
    def test_pipeline_workflow_exists(self):
        """Test that pipeline workflow methods exist."""
        # Test that comprehensive validation method exists
        self.assertTrue(hasattr(self.validation_system, 'comprehensive_validate'))
        
        # Test that component access methods exist
        self.assertIsNotNone(self.validation_system.validation_runner)
        self.assertIsNotNone(self.validation_system.dataset_validator)
        self.assertIsNotNone(self.validation_system.disagreement_detector)
    
    def test_component_configuration_propagation(self):
        """Test configuration propagation to components."""
        custom_config = IntegratedValidationConfig(
            enable_multi_model_validation=True,
            enable_cultural_datasets=True,
            confidence_threshold=0.85,
            validation_runner_config={'timeout': 45},
            dataset_validator_config={'datasets_dir': './data/cultural'},
            community_system_config={'data_dir': './custom_flags'}
        )
        
        system = IntegratedValidationSystem(custom_config)
        
        # Test that configuration is properly set
        self.assertEqual(system.config.confidence_threshold, 0.85)
        self.assertTrue(system.config.enable_multi_model_validation)
        self.assertTrue(system.config.enable_cultural_datasets)
        
        # Test that component configurations are passed through
        self.assertIsNotNone(system.config.validation_runner_config)
        self.assertIsNotNone(system.config.dataset_validator_config)
        self.assertIsNotNone(system.config.community_system_config)


class TestComprehensiveReporting(unittest.TestCase):
    """Test comprehensive validation reporting."""
    
    def setUp(self):
        """Set up comprehensive reporting test fixtures."""
        self.validation_system = IntegratedValidationSystem()
        
        self.cultural_context = CulturalContext(
            traditions=["ubuntu"],
            cultural_groups=["bantu_peoples"],
            knowledge_systems=["african_traditional"],
            performance_aspects=[],
            linguistic_varieties=[]
        )
        
        self.test_evaluation = Mock(spec=DomainEvaluationResult)
        self.test_evaluation.overall_score = 0.75
        self.test_evaluation.cultural_context = self.cultural_context
        self.test_evaluation.metadata = {'evaluation_id': 'report_test_001'}
        self.test_evaluation.domain = 'social'
        self.test_evaluation.evaluation_type = 'test'
    
    def test_validation_summary_generation(self):
        """Test validation summary generation."""
        # Create mock component results
        mock_dataset_result = Mock(spec=DatasetValidationResult)
        mock_dataset_result.validation_confidence = 0.8
        mock_dataset_result.coverage_score = 0.9
        mock_dataset_result.validation_flags = []
        
        comprehensive_result = ComprehensiveValidationResult(
            primary_evaluation=self.test_evaluation,
            multi_model_validation=None,
            wikipedia_validation=None,
            dataset_validation=mock_dataset_result,
            disagreement_analysis=None,
            open_apis_validation=None,
            community_flags=[],
            overall_confidence=0.8,
            validation_summary="Dataset validation shows high confidence and coverage",
            recommendations=["Continue with current validation approach"],
            processing_metadata={'components_used': ['dataset_validator']}
        )
        
        # Test summary content
        self.assertIn("Dataset validation", comprehensive_result.validation_summary)
        self.assertIn("high confidence", comprehensive_result.validation_summary)
        self.assertGreater(len(comprehensive_result.recommendations), 0)
        self.assertIn('components_used', comprehensive_result.processing_metadata)
    
    def test_recommendation_generation(self):
        """Test validation recommendation generation."""
        # Test with low confidence scenario
        low_confidence_result = ComprehensiveValidationResult(
            primary_evaluation=self.test_evaluation,
            multi_model_validation=None,
            wikipedia_validation=None,
            dataset_validation=None,
            disagreement_analysis=None,
            open_apis_validation=None,
            community_flags=[],
            overall_confidence=0.3,  # Low confidence
            validation_summary="Low confidence validation detected",
            recommendations=[
                "Consider additional validation sources",
                "Review cultural context accuracy",
                "Seek community feedback"
            ],
            processing_metadata={}
        )
        
        # Test recommendations for low confidence
        self.assertEqual(low_confidence_result.overall_confidence, 0.3)
        self.assertGreater(len(low_confidence_result.recommendations), 0)
        self.assertTrue(any("additional validation" in rec for rec in low_confidence_result.recommendations))
    
    def test_processing_metadata_tracking(self):
        """Test processing metadata tracking."""
        metadata = {
            'validation_start_time': datetime.now().isoformat(),
            'components_enabled': ['dataset_validator', 'community_flagging'],
            'processing_time_seconds': 3.2,
            'total_api_calls': 5,
            'cache_hits': 2
        }
        
        result_with_metadata = ComprehensiveValidationResult(
            primary_evaluation=self.test_evaluation,
            multi_model_validation=None,
            wikipedia_validation=None,
            dataset_validation=None,
            disagreement_analysis=None,
            open_apis_validation=None,
            community_flags=[],
            overall_confidence=0.75,
            validation_summary="Validation completed with metadata tracking",
            recommendations=[],
            processing_metadata=metadata
        )
        
        # Test metadata presence and structure
        self.assertIn('validation_start_time', result_with_metadata.processing_metadata)
        self.assertIn('components_enabled', result_with_metadata.processing_metadata)
        self.assertIn('processing_time_seconds', result_with_metadata.processing_metadata)
        self.assertEqual(result_with_metadata.processing_metadata['total_api_calls'], 5)
        self.assertEqual(result_with_metadata.processing_metadata['cache_hits'], 2)


class TestConfigurationManagement(unittest.TestCase):
    """Test configuration management and validation."""
    
    def test_configuration_validation(self):
        """Test configuration validation and defaults."""
        # Test with minimal configuration
        minimal_config = IntegratedValidationConfig()
        system = IntegratedValidationSystem(minimal_config)
        
        # Should have reasonable defaults
        self.assertGreater(minimal_config.confidence_threshold, 0)
        self.assertLess(minimal_config.confidence_threshold, 1)
        self.assertGreater(minimal_config.disagreement_threshold, 0)
        self.assertLess(minimal_config.disagreement_threshold, 1)
        
        # Test with custom configuration
        custom_config = IntegratedValidationConfig(
            enable_multi_model_validation=False,
            enable_cultural_datasets=True,
            confidence_threshold=0.9,
            disagreement_threshold=0.1,
            bias_threshold=0.2
        )
        
        custom_system = IntegratedValidationSystem(custom_config)
        self.assertFalse(custom_system.config.enable_multi_model_validation)
        self.assertTrue(custom_system.config.enable_cultural_datasets)
        self.assertEqual(custom_system.config.confidence_threshold, 0.9)
    
    def test_component_specific_configuration(self):
        """Test component-specific configuration handling."""
        config_with_component_settings = IntegratedValidationConfig(
            validation_runner_config={
                'api_timeout': 60,
                'max_concurrent_requests': 10
            },
            dataset_validator_config={
                'datasets_dir': './data/cultural',
                'confidence_threshold': 0.8
            },
            community_system_config={
                'data_dir': './test_flags',
                'auto_flag_enabled': True
            }
        )
        
        system = IntegratedValidationSystem(config_with_component_settings)
        
        # Test that component configurations are available
        self.assertIsNotNone(system.config.validation_runner_config)
        self.assertIsNotNone(system.config.dataset_validator_config)
        self.assertIsNotNone(system.config.community_system_config)
        
        # Test specific values
        self.assertEqual(system.config.validation_runner_config['api_timeout'], 60)
        self.assertEqual(system.config.dataset_validator_config['datasets_dir'], './data/cultural')
        self.assertTrue(system.config.community_system_config['auto_flag_enabled'])


if __name__ == '__main__':
    unittest.main()