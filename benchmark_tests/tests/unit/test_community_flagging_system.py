"""
Unit tests for CommunityFlaggingSystem.

Tests flagging algorithms, auto-flagging rules and thresholds, analytics calculations,
flag review workflows, and CSV/JSON export functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import json
import csv
import io
import statistics
from datetime import datetime, timedelta

from evaluator.community_flagging_system import (
    CommunityFlaggingSystem,
    CommunityFlag,
    FlagCategory,
    FlagSeverity,
    ReviewStatus,
    FlagAnalytics,
    CommunityFeedback
)
from evaluator.domain_evaluator_base import (
    DomainEvaluationResult,
    CulturalContext
)
from evaluator.evaluation_aggregator import ValidationFlag


class TestCommunityFlaggingSystem(unittest.TestCase):
    """Test basic community flagging system functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'bias_threshold': 0.3,
            'low_confidence_threshold': 0.5,
            'high_disagreement_threshold': 0.7,
            'cultural_authenticity_threshold': 0.6
        }
        self.flagging_system = CommunityFlaggingSystem(self.config)
        
        # Test evaluation result - using Mock for actual API
        self.test_evaluation = Mock(spec=DomainEvaluationResult)
        self.test_evaluation.overall_score = 0.75
        self.test_evaluation.metadata = {'evaluation_id': 'test_001', 'evaluation_confidence': 0.65}
        self.test_evaluation.domain = 'social'
        self.test_evaluation.evaluation_type = 'test'
        self.test_evaluation.dimensions = [Mock(name='cultural_authenticity'), Mock(name='traditional_accuracy')]
        
        self.cultural_context = CulturalContext(
            traditions=["ubuntu", "ubuntu_philosophy"],
            cultural_groups=["bantu_peoples"],
            knowledge_systems=["african_traditional"],
            performance_aspects=[],
            linguistic_varieties=[]
        )
        self.test_evaluation.cultural_context = self.cultural_context
        self.test_evaluation.calculate_cultural_competence = Mock(return_value=0.65)
    
    def test_flagging_system_initialization(self):
        """Test flagging system initialization and configuration."""
        # Test with custom config
        system = CommunityFlaggingSystem(self.config)
        self.assertEqual(system.config['bias_threshold'], 0.3)
        
        # Test auto-flagging rules initialization
        self.assertGreater(len(system.auto_flag_rules), 0)
        
        # Test default config
        default_system = CommunityFlaggingSystem()
        self.assertIsNotNone(default_system.auto_flag_rules)
    
    def test_flag_data_structure(self):
        """Test flag data structure and validation."""
        # Test using actual API method
        flag_id = self.flagging_system.submit_community_flag(
            category=FlagCategory.CULTURAL_INACCURACY,
            severity=FlagSeverity.HIGH,
            description="High cultural bias detected in evaluation",
            evaluation_result=self.test_evaluation,
            evidence=["Bias score: 0.85", "Cultural group: underrepresented"],
            recommended_action="Review cultural accuracy",
            submitter_info={'type': 'auto_system'}
        )
        
        # Retrieve the created flag
        created_flag = self.flagging_system.flags[flag_id]
        self.assertEqual(created_flag.category, FlagCategory.CULTURAL_INACCURACY)
        self.assertEqual(created_flag.severity, FlagSeverity.HIGH)
        self.assertEqual(created_flag.review_status, ReviewStatus.PENDING)
        self.assertIn("bantu_peoples", created_flag.cultural_groups_affected)
        self.assertEqual(created_flag.evaluation_id, 'test_001')


class TestAutoFlaggingRulesAndThresholds(unittest.TestCase):
    """Test auto-flagging rules and threshold algorithms."""
    
    def setUp(self):
        """Set up auto-flagging test fixtures."""
        self.config = {
            'bias_threshold': 0.3,
            'low_confidence_threshold': 0.5,
            'high_disagreement_threshold': 0.7,
            'cultural_authenticity_threshold': 0.6
        }
        self.flagging_system = CommunityFlaggingSystem(self.config)
        
        self.cultural_context = CulturalContext(
            traditions=["ubuntu"],
            cultural_groups=["bantu_peoples"],
            knowledge_systems=["african_traditional"],
            performance_aspects=[],
            linguistic_varieties=[]
        )
    
    def test_low_confidence_auto_flagging(self):
        """Test auto-flagging for low confidence evaluations."""
        # Test low confidence evaluation
        low_confidence_evaluation = Mock(spec=DomainEvaluationResult)
        low_confidence_evaluation.overall_score = 0.6
        low_confidence_evaluation.cultural_context = self.cultural_context
        low_confidence_evaluation.metadata = {'evaluation_confidence': 0.2}  # Very low confidence
        low_confidence_evaluation.domain = 'social'
        low_confidence_evaluation.evaluation_type = 'test'
        low_confidence_evaluation.dimensions = [Mock(name='cultural_authenticity')]
        low_confidence_evaluation.calculate_cultural_competence = Mock(return_value=0.3)
        
        flags = self.flagging_system.auto_flag_evaluation(
            low_confidence_evaluation, []
        )
        
        # Should generate auto flags for low confidence
        self.assertGreater(len(flags), 0)
        
        # Verify flag was created with expected attributes
        created_flag = self.flagging_system.flags[flags[0]]
        self.assertIn('confidence', created_flag.description.lower())
        self.assertEqual(created_flag.category, FlagCategory.EVALUATION_ERROR)
    
    def test_high_confidence_no_flagging(self):
        """Test that high confidence evaluations don't get auto-flagged."""
        # Test high confidence evaluation
        high_confidence_evaluation = Mock(spec=DomainEvaluationResult)
        high_confidence_evaluation.overall_score = 0.75
        high_confidence_evaluation.cultural_context = self.cultural_context
        high_confidence_evaluation.metadata = {'evaluation_confidence': 0.8}  # High confidence
        high_confidence_evaluation.domain = 'social'
        high_confidence_evaluation.evaluation_type = 'test'
        high_confidence_evaluation.dimensions = [Mock(name='cultural_authenticity')]
        high_confidence_evaluation.calculate_cultural_competence = Mock(return_value=0.7)
        
        flags = self.flagging_system.auto_flag_evaluation(
            high_confidence_evaluation, []
        )
        
        # Should not generate flags for high confidence evaluation
        self.assertEqual(len(flags), 0)
    
    def test_cultural_authenticity_auto_flagging(self):
        """Test auto-flagging for low cultural authenticity."""
        # Test low cultural authenticity
        low_authenticity_evaluation = Mock(spec=DomainEvaluationResult)
        low_authenticity_evaluation.overall_score = 0.8
        low_authenticity_evaluation.cultural_context = self.cultural_context
        low_authenticity_evaluation.metadata = {'evaluation_confidence': 0.8}
        low_authenticity_evaluation.domain = 'social'
        low_authenticity_evaluation.evaluation_type = 'test'
        low_authenticity_evaluation.dimensions = [Mock(name='cultural_authenticity')]
        low_authenticity_evaluation.calculate_cultural_competence = Mock(return_value=0.1)  # Very low
        
        flags = self.flagging_system.auto_flag_evaluation(
            low_authenticity_evaluation, []
        )
        
        # Should generate cultural authenticity flag
        self.assertGreater(len(flags), 0)
        created_flag = self.flagging_system.flags[flags[0]]
        self.assertEqual(created_flag.category, FlagCategory.CULTURAL_INACCURACY)
        self.assertEqual(created_flag.severity, FlagSeverity.HIGH)


class TestAnalyticsCalculations(unittest.TestCase):
    """Test analytics and reporting calculations."""
    
    def setUp(self):
        """Set up analytics test fixtures."""
        self.flagging_system = CommunityFlaggingSystem()
        
        # Create test flags using the actual API
        for i in range(5):
            evaluation = Mock(spec=DomainEvaluationResult)
            evaluation.overall_score = 0.7
            evaluation.cultural_context = CulturalContext(
                traditions=["test"],
                cultural_groups=[f"group_{i % 3}"],
                knowledge_systems=["test"],
                performance_aspects=[],
                linguistic_varieties=[]
            )
            evaluation.metadata = {'evaluation_id': f'eval_{i:03d}'}
            evaluation.domain = 'social'
            evaluation.evaluation_type = 'test'
            evaluation.dimensions = [Mock(name='test_dimension')]
            
            self.flagging_system.submit_community_flag(
                category=FlagCategory.CULTURAL_INACCURACY if i % 2 == 0 else FlagCategory.BIAS_DETECTED,
                severity=FlagSeverity.HIGH if i % 2 == 0 else FlagSeverity.MEDIUM,
                description=f"Test flag {i}",
                evaluation_result=evaluation,
                evidence=[f"Evidence {i}"],
                recommended_action="Test action",
                submitter_info={'type': 'test'}
            )
    
    def test_flag_frequency_analysis(self):
        """Test flag frequency analysis calculations."""
        analytics = self.flagging_system.generate_flag_analytics()
        
        # Test total counts
        self.assertEqual(analytics.total_flags, 5)
        
        # Test flag category distribution
        self.assertIn('cultural_inaccuracy', analytics.flags_by_category)
        self.assertIn('bias_detected', analytics.flags_by_category)
        
        # Test severity distribution
        self.assertIn('high', analytics.flags_by_severity)
        self.assertIn('medium', analytics.flags_by_severity)
        
        # Test status distribution
        self.assertIn('pending', analytics.flags_by_status)
    
    def test_basic_export_functionality(self):
        """Test basic export functionality."""
        # Test CSV export
        csv_file_path = self.flagging_system.export_flagged_items_csv()
        self.assertTrue(csv_file_path.endswith('.csv'))
        
        # Test JSON export
        json_file_path = self.flagging_system.export_feedback_json()
        self.assertTrue(json_file_path.endswith('.json'))
        
        # Test system status
        status = self.flagging_system.get_system_status()
        self.assertIn('system_info', status)
        self.assertIn('total_flags', status['system_info'])


class TestFlagReviewWorkflow(unittest.TestCase):
    """Test flag review workflows."""
    
    def setUp(self):
        """Set up flag review test fixtures."""
        self.flagging_system = CommunityFlaggingSystem()
        
        # Create test evaluation
        self.test_evaluation = Mock(spec=DomainEvaluationResult)
        self.test_evaluation.overall_score = 0.75
        self.test_evaluation.cultural_context = CulturalContext(
            traditions=["ubuntu"],
            cultural_groups=["bantu_peoples"],
            knowledge_systems=["african_traditional"],
            performance_aspects=[],
            linguistic_varieties=[]
        )
        self.test_evaluation.metadata = {'evaluation_id': 'test_001'}
        self.test_evaluation.domain = 'social'
        self.test_evaluation.evaluation_type = 'test'
        self.test_evaluation.dimensions = [Mock(name='cultural_authenticity')]
    
    def test_flag_review_process(self):
        """Test flag review and status update."""
        # Create flag for review
        flag_id = self.flagging_system.submit_community_flag(
            category=FlagCategory.CULTURAL_INACCURACY,
            severity=FlagSeverity.HIGH,
            description="Bias flag for review",
            evaluation_result=self.test_evaluation,
            evidence=["Bias detected in cultural evaluation"],
            recommended_action="Review evaluation",
            submitter_info={'type': 'auto_system'}
        )
        
        # Test review process
        success = self.flagging_system.review_flag(
            flag_id=flag_id,
            reviewer_notes="Confirmed bias in evaluation",
            new_status=ReviewStatus.RESOLVED,
            resolution_notes="Issue addressed"
        )
        
        self.assertTrue(success)
        
        # Check flag status was updated
        updated_flag = self.flagging_system.flags[flag_id]
        self.assertEqual(updated_flag.review_status, ReviewStatus.RESOLVED)
        self.assertEqual(updated_flag.resolution_notes, "Issue addressed")
        self.assertGreater(len(updated_flag.reviewer_notes), 0)
    
    def test_get_pending_flags(self):
        """Test retrieval of pending flags."""
        # Create some flags
        for i in range(3):
            self.flagging_system.submit_community_flag(
                category=FlagCategory.BIAS_DETECTED,
                severity=FlagSeverity.MEDIUM,
                description=f"Test flag {i}",
                evaluation_result=self.test_evaluation,
                evidence=[f"Evidence {i}"],
                recommended_action="Review",
                submitter_info={'type': 'test'}
            )
        
        # Get pending flags
        pending_flags = self.flagging_system.get_pending_flags()
        
        # Should have all 3 flags pending
        self.assertEqual(len(pending_flags), 3)
        for flag in pending_flags:
            self.assertEqual(flag.review_status, ReviewStatus.PENDING)
    
    def test_community_feedback(self):
        """Test community feedback submission."""
        # Submit feedback
        feedback_id = self.flagging_system.submit_community_feedback(
            evaluation_id="eval_001",
            overall_rating=4.0,
            dimension_ratings={"cultural_authenticity": 3.5, "accuracy": 4.5},
            cultural_accuracy_rating=3.5,
            comments="Good evaluation with minor cultural issues",
            cultural_background="bantu_peoples",
            expertise_level="community_member",
            suggested_improvements=["Include more cultural context"]
        )
        
        # Verify feedback was created
        feedback = self.flagging_system.feedback[feedback_id]
        self.assertEqual(feedback.evaluation_id, "eval_001")
        self.assertEqual(feedback.overall_rating, 4.0)
        self.assertEqual(feedback.cultural_background, "bantu_peoples")
        self.assertIn("Include more cultural context", feedback.suggested_improvements)


if __name__ == '__main__':
    unittest.main()