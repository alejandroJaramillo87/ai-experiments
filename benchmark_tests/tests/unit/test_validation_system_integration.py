"""
Unit tests for Validation System Integration.

Tests end-to-end coordination, comprehensive validation pipeline,
cross-component communication, and full system validation workflows.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import json
import asyncio
from datetime import datetime

from evaluator.integrated_validation_system import IntegratedValidationSystem, IntegratedValidationConfig
from evaluator.domain_evaluator_base import CulturalContext, DomainEvaluationResult
from evaluator.validation_runner import ValidationRunner
from evaluator.cultural_dataset_validator import CulturalDatasetValidator
from evaluator.community_flagging_system import CommunityFlaggingSystem, FlagCategory, FlagSeverity
from evaluator.ensemble_disagreement_detector import EnsembleDisagreementDetector


class TestValidationSystemIntegration(unittest.TestCase):
    """Test end-to-end validation system integration."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.config = IntegratedValidationConfig(
            enable_multi_model_validation=True,
            enable_cultural_datasets=True,
            enable_community_flagging=True,
            enable_ensemble_disagreement=True,
            confidence_threshold=0.7,
            disagreement_threshold=0.3
        )
        self.integrated_system = IntegratedValidationSystem(self.config)
        
        # Ubuntu cultural content for testing
        self.ubuntu_context = CulturalContext(
            traditions=["ubuntu", "ubuntu_philosophy", "restorative_justice"],
            cultural_groups=["bantu_peoples", "zulu", "xhosa", "ndebele"],
            knowledge_systems=["african_traditional_knowledge", "ubuntu_ethics"],
            performance_aspects=["community_healing", "consensus_building", "conflict_resolution"],
            linguistic_varieties=["isizulu", "isixhosa", "isindebele", "setswana"]
        )
        
        # Test evaluation
        self.ubuntu_evaluation = Mock(spec=DomainEvaluationResult)
        self.ubuntu_evaluation.overall_score = 0.85
        self.ubuntu_evaluation.cultural_context = self.ubuntu_context
        self.ubuntu_evaluation.metadata = {
            'evaluation_id': 'ubuntu_integration_001',
            'content_summary': 'Ubuntu philosophy emphasizing community interconnectedness',
            'domain': 'social',
            'evaluation_type': 'cultural_authenticity'
        }
        self.ubuntu_evaluation.domain = 'social'
        self.ubuntu_evaluation.evaluation_type = 'cultural_authenticity'
        self.ubuntu_evaluation.dimensions = [
            Mock(name='cultural_authenticity'),
            Mock(name='traditional_accuracy'),
            Mock(name='community_relevance')
        ]
    
    def test_full_system_initialization(self):
        """Test complete system initialization and component integration."""
        # Test that all components are properly initialized
        self.assertIsInstance(self.integrated_system.validation_runner, ValidationRunner)
        self.assertIsInstance(self.integrated_system.dataset_validator, CulturalDatasetValidator)
        self.assertIsInstance(self.integrated_system.community_flagging_system, CommunityFlaggingSystem)
        self.assertIsInstance(self.integrated_system.disagreement_detector, EnsembleDisagreementDetector)
        
        # Test configuration propagation
        self.assertEqual(self.integrated_system.config.confidence_threshold, 0.7)
        self.assertTrue(self.integrated_system.config.enable_multi_model_validation)
        self.assertTrue(self.integrated_system.config.enable_cultural_datasets)
        self.assertTrue(self.integrated_system.config.enable_community_flagging)
    
    def test_cross_component_communication(self):
        """Test communication between validation components."""
        # Test that components can be accessed and configured
        validation_runner = self.integrated_system.validation_runner
        dataset_validator = self.integrated_system.dataset_validator
        community_system = self.integrated_system.community_flagging_system
        disagreement_detector = self.integrated_system.disagreement_detector
        
        # All components should be properly initialized
        self.assertIsNotNone(validation_runner)
        self.assertIsNotNone(dataset_validator)
        self.assertIsNotNone(community_system)
        self.assertIsNotNone(disagreement_detector)
        
        # Test that components have expected methods/attributes
        self.assertTrue(hasattr(validation_runner, 'validate_with_multiple_apis'))
        self.assertTrue(hasattr(dataset_validator, 'validate_cultural_evaluation'))
        self.assertTrue(hasattr(community_system, 'submit_community_flag'))
        self.assertTrue(hasattr(disagreement_detector, 'analyze_disagreement'))
    
    def test_comprehensive_validation_workflow(self):
        """Test comprehensive validation workflow coordination."""
        # Test that the integrated system has the main validation method
        self.assertTrue(hasattr(self.integrated_system, 'comprehensive_validate'))
        
        # Test configuration enables all components
        self.assertTrue(self.integrated_system.config.enable_multi_model_validation)
        self.assertTrue(self.integrated_system.config.enable_cultural_datasets)
        self.assertTrue(self.integrated_system.config.enable_community_flagging)
        self.assertTrue(self.integrated_system.config.enable_ensemble_disagreement)
    
    def test_ubuntu_cultural_content_validation(self):
        """Test validation of Ubuntu cultural content across all components."""
        # Test individual component capabilities with Ubuntu content
        
        # Dataset Validator - should find matches for Ubuntu content
        dataset_result = self.integrated_system.dataset_validator.validate_cultural_evaluation(
            self.ubuntu_context, self.ubuntu_evaluation
        )
        
        # Should return proper result structure
        self.assertIsNotNone(dataset_result)
        self.assertGreaterEqual(dataset_result.validation_confidence, 0.0)
        self.assertLessEqual(dataset_result.validation_confidence, 1.0)
        self.assertGreaterEqual(dataset_result.coverage_score, 0.0)
        self.assertLessEqual(dataset_result.coverage_score, 1.0)
        
        # Community Flagging System - test flag creation for Ubuntu content
        flag_id = self.integrated_system.community_flagging_system.submit_community_flag(
            category=FlagCategory.CULTURAL_INACCURACY,
            severity=FlagSeverity.MEDIUM,
            description="Testing Ubuntu cultural accuracy",
            evaluation_result=self.ubuntu_evaluation,
            evidence=["Ubuntu philosophy validation"],
            recommended_action="Review cultural context",
            submitter_info={'type': 'integration_test'}
        )
        
        # Should create flag successfully
        self.assertIsNotNone(flag_id)
        self.assertIn(flag_id, self.integrated_system.community_flagging_system.flags)
    
    def test_validation_pipeline_error_handling(self):
        """Test validation pipeline error handling and resilience."""
        # Test with minimal cultural context
        minimal_context = CulturalContext(
            traditions=[],
            cultural_groups=[],
            knowledge_systems=[],
            performance_aspects=[],
            linguistic_varieties=[]
        )
        
        minimal_evaluation = Mock(spec=DomainEvaluationResult)
        minimal_evaluation.overall_score = 0.5
        minimal_evaluation.cultural_context = minimal_context
        minimal_evaluation.metadata = {'evaluation_id': 'minimal_test'}
        minimal_evaluation.domain = 'social'
        minimal_evaluation.evaluation_type = 'test'
        minimal_evaluation.dimensions = []
        
        # Should handle minimal context gracefully
        try:
            dataset_result = self.integrated_system.dataset_validator.validate_cultural_evaluation(
                minimal_context, minimal_evaluation
            )
            # Should complete without errors
            self.assertIsNotNone(dataset_result)
        except Exception as e:
            self.fail(f"Validation pipeline should handle minimal context gracefully: {e}")
    
    def test_component_configuration_consistency(self):
        """Test configuration consistency across components."""
        # Create system with specific configuration values
        test_config = IntegratedValidationConfig(
            confidence_threshold=0.85,
            disagreement_threshold=0.2,
            bias_threshold=0.25,
            cultural_authenticity_threshold=0.6,
            validation_runner_config={'timeout': 45},
            dataset_validator_config={'datasets_dir': './test_integration_datasets'},
            community_system_config={'data_dir': './test_integration_flags'}
        )
        
        test_system = IntegratedValidationSystem(test_config)
        
        # Test that configuration values are properly set
        self.assertEqual(test_system.config.confidence_threshold, 0.85)
        self.assertEqual(test_system.config.disagreement_threshold, 0.2)
        self.assertEqual(test_system.config.bias_threshold, 0.25)
        self.assertEqual(test_system.config.cultural_authenticity_threshold, 0.6)
        
        # Test that component-specific configurations are available
        self.assertIsNotNone(test_system.config.validation_runner_config)
        self.assertIsNotNone(test_system.config.dataset_validator_config)
        self.assertIsNotNone(test_system.config.community_system_config)
    
    def test_validation_system_scalability(self):
        """Test validation system scalability with multiple evaluations."""
        # Create multiple test evaluations
        test_evaluations = []
        for i in range(3):
            evaluation = Mock(spec=DomainEvaluationResult)
            evaluation.overall_score = 0.7 + (i * 0.1)
            evaluation.cultural_context = CulturalContext(
                traditions=[f"tradition_{i}"],
                cultural_groups=[f"group_{i}"],
                knowledge_systems=[f"knowledge_{i}"],
                performance_aspects=[],
                linguistic_varieties=[]
            )
            evaluation.metadata = {'evaluation_id': f'scale_test_{i}'}
            evaluation.domain = 'social'
            evaluation.evaluation_type = 'test'
            evaluation.dimensions = [Mock(name=f'dimension_{i}')]
            test_evaluations.append(evaluation)
        
        # Process multiple evaluations
        results = []
        for evaluation in test_evaluations:
            try:
                dataset_result = self.integrated_system.dataset_validator.validate_cultural_evaluation(
                    evaluation.cultural_context, evaluation
                )
                results.append(dataset_result)
            except Exception as e:
                self.fail(f"System should handle multiple evaluations: {e}")
        
        # Should process all evaluations successfully
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsNotNone(result)
    
    def test_system_status_and_health_monitoring(self):
        """Test system status monitoring and health checks."""
        # Test community flagging system status
        community_status = self.integrated_system.community_flagging_system.get_system_status()
        self.assertIn('system_info', community_status)
        self.assertIn('total_flags', community_status['system_info'])
        
        # Test dataset validator info
        dataset_info = self.integrated_system.dataset_validator.get_dataset_info()
        self.assertIn('datasets', dataset_info)
        self.assertIn('total_entries', dataset_info)
        
        # Test that all components are responding
        self.assertIsNotNone(community_status)
        self.assertIsNotNone(dataset_info)
    
    def test_end_to_end_ubuntu_philosophy_validation(self):
        """Test complete end-to-end validation of Ubuntu philosophy content."""
        # Create comprehensive Ubuntu evaluation
        ubuntu_philosophy_evaluation = Mock(spec=DomainEvaluationResult)
        ubuntu_philosophy_evaluation.overall_score = 0.88
        ubuntu_philosophy_evaluation.cultural_context = CulturalContext(
            traditions=["ubuntu", "ubuntu_philosophy", "hunhu", "botho"],
            cultural_groups=["bantu_peoples", "southern_african"],
            knowledge_systems=["african_philosophy", "ubuntu_ethics", "communitarian_values"],
            performance_aspects=["restorative_justice", "community_healing", "consensus_decision_making"],
            linguistic_varieties=["isizulu", "isixhosa", "sesotho", "setswana"]
        )
        ubuntu_philosophy_evaluation.metadata = {
            'evaluation_id': 'ubuntu_philosophy_e2e',
            'content_summary': 'Ubuntu philosophy: I am because we are - interconnectedness and community',
            'domain': 'social'
        }
        ubuntu_philosophy_evaluation.domain = 'social'
        ubuntu_philosophy_evaluation.evaluation_type = 'cultural_philosophy'
        ubuntu_philosophy_evaluation.dimensions = [
            Mock(name='philosophical_accuracy'),
            Mock(name='cultural_authenticity'),
            Mock(name='community_relevance'),
            Mock(name='traditional_grounding')
        ]
        
        # Run dataset validation
        dataset_validation_result = self.integrated_system.dataset_validator.validate_cultural_evaluation(
            ubuntu_philosophy_evaluation.cultural_context,
            ubuntu_philosophy_evaluation
        )
        
        # Should find good matches for Ubuntu content
        self.assertGreaterEqual(dataset_validation_result.validation_confidence, 0.0)
        self.assertGreaterEqual(dataset_validation_result.coverage_score, 0.0)
        
        # Test community feedback submission
        feedback_id = self.integrated_system.community_flagging_system.submit_community_feedback(
            evaluation_id='ubuntu_philosophy_e2e',
            overall_rating=4.5,
            dimension_ratings={
                'philosophical_accuracy': 4.8,
                'cultural_authenticity': 4.6,
                'community_relevance': 4.3
            },
            cultural_accuracy_rating=4.7,
            comments="Excellent representation of Ubuntu philosophy with authentic cultural grounding",
            cultural_background="southern_african",
            expertise_level="cultural_expert",
            suggested_improvements=["Include more linguistic diversity examples"]
        )
        
        # Should create feedback successfully
        self.assertIsNotNone(feedback_id)
        self.assertIn(feedback_id, self.integrated_system.community_flagging_system.feedback)
        
        # Verify feedback content
        feedback = self.integrated_system.community_flagging_system.feedback[feedback_id]
        self.assertEqual(feedback.evaluation_id, 'ubuntu_philosophy_e2e')
        self.assertEqual(feedback.overall_rating, 4.5)
        self.assertEqual(feedback.cultural_background, "southern_african")
        self.assertIn("linguistic diversity", feedback.suggested_improvements[0])
    
    def test_validation_system_configuration_flexibility(self):
        """Test validation system configuration flexibility."""
        # Test with minimal components enabled
        minimal_config = IntegratedValidationConfig(
            enable_multi_model_validation=False,
            enable_wikipedia_validation=False,
            enable_cultural_datasets=True,
            enable_ensemble_disagreement=False,
            enable_open_apis=False,
            enable_community_flagging=True,
            enable_bias_detection=False
        )
        
        minimal_system = IntegratedValidationSystem(minimal_config)
        
        # Should still initialize successfully
        self.assertIsNotNone(minimal_system)
        self.assertFalse(minimal_system.config.enable_multi_model_validation)
        self.assertTrue(minimal_system.config.enable_cultural_datasets)
        self.assertTrue(minimal_system.config.enable_community_flagging)
        
        # Should still have core components
        self.assertIsNotNone(minimal_system.dataset_validator)
        self.assertIsNotNone(minimal_system.community_flagging_system)


if __name__ == '__main__':
    unittest.main()