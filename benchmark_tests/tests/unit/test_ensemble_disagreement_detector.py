"""
Unit tests for EnsembleDisagreementDetector.

Tests algorithmic disagreement analysis, strategy evaluation modifications,
consensus calculations, and reliability scoring across multiple evaluation strategies.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import statistics
import numpy as np
import math

from evaluator.ensemble_disagreement_detector import (
    EnsembleDisagreementDetector,
    EvaluationStrategy,
    EvaluationConfiguration,
    EnsembleEvaluationResult,
    DisagreementAnalysis,
    EnsembleDisagreementResult
)
from evaluator.domain_evaluator_base import (
    DomainEvaluationResult,
    EvaluationDimension,
    CulturalContext,
    MultiDimensionalEvaluator
)
from evaluator.evaluation_aggregator import ValidationFlag


class TestEnsembleDisagreementDetector(unittest.TestCase):
    """Test basic ensemble disagreement detector functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = EnsembleDisagreementDetector()
        
        # Create mock cultural context
        self.cultural_context = CulturalContext(
            traditions=["tradition_a", "tradition_b"],
            knowledge_systems=["system_a"],
            performance_aspects=["aspect_1", "aspect_2"],
            cultural_groups=["group_x", "group_y"],
            linguistic_varieties=["variety_1"]
        )
        
        # Create sample evaluation dimensions
        self.cultural_dim = EvaluationDimension(
            name="cultural_authenticity",
            score=0.8,
            confidence=0.9,
            cultural_relevance=0.85,
            evidence=["Strong cultural markers"],
            cultural_markers=["tradition_a", "cultural_element"]
        )
        
        self.accuracy_dim = EvaluationDimension(
            name="traditional_accuracy",
            score=0.75,
            confidence=0.8,
            cultural_relevance=0.7,
            evidence=["Good traditional accuracy"],
            cultural_markers=["traditional_element"]
        )
    
    def test_detector_initialization(self):
        """Test detector initialization and strategy configuration."""
        # Default initialization
        detector = EnsembleDisagreementDetector()
        self.assertEqual(detector.disagreement_threshold, 0.3)
        self.assertEqual(detector.consensus_threshold, 0.7)
        
        # Verify all strategies are initialized
        expected_strategies = {
            EvaluationStrategy.STANDARD,
            EvaluationStrategy.CONSERVATIVE,
            EvaluationStrategy.AGGRESSIVE,
            EvaluationStrategy.CULTURAL_FOCUSED,
            EvaluationStrategy.DIMENSION_WEIGHTED
        }
        self.assertEqual(set(detector.evaluation_strategies.keys()), expected_strategies)
        
        # Custom config
        config = {
            'disagreement_threshold': 0.4,
            'consensus_threshold': 0.8
        }
        detector = EnsembleDisagreementDetector(config)
        self.assertEqual(detector.disagreement_threshold, 0.4)
        self.assertEqual(detector.consensus_threshold, 0.8)
    
    def test_evaluation_strategy_configurations(self):
        """Test that evaluation strategy configurations are properly set."""
        strategies = self.detector.evaluation_strategies
        
        # Test standard strategy
        standard_config = strategies[EvaluationStrategy.STANDARD]
        self.assertEqual(standard_config.cultural_emphasis, 0.5)
        self.assertEqual(standard_config.confidence_threshold, 0.5)
        self.assertEqual(standard_config.scoring_bias, 0.0)
        
        # Test conservative strategy
        conservative_config = strategies[EvaluationStrategy.CONSERVATIVE]
        self.assertEqual(conservative_config.cultural_emphasis, 0.3)
        self.assertEqual(conservative_config.confidence_threshold, 0.7)
        self.assertEqual(conservative_config.scoring_bias, -0.2)  # More lenient
        
        # Test aggressive strategy
        aggressive_config = strategies[EvaluationStrategy.AGGRESSIVE]
        self.assertEqual(aggressive_config.cultural_emphasis, 0.7)
        self.assertEqual(aggressive_config.confidence_threshold, 0.3)
        self.assertEqual(aggressive_config.scoring_bias, 0.2)  # Stricter
        
        # Test cultural focused strategy
        cultural_config = strategies[EvaluationStrategy.CULTURAL_FOCUSED]
        self.assertEqual(cultural_config.cultural_emphasis, 0.9)
        self.assertEqual(cultural_config.confidence_threshold, 0.6)
        
        # Test dimension weighted strategy
        weighted_config = strategies[EvaluationStrategy.DIMENSION_WEIGHTED]
        self.assertIn('cultural_authenticity', weighted_config.weights)
        self.assertIn('traditional_accuracy', weighted_config.weights)


class TestDisagreementAnalysisAlgorithms(unittest.TestCase):
    """Test disagreement analysis algorithmic functions."""
    
    def setUp(self):
        """Set up algorithmic test fixtures."""
        self.detector = EnsembleDisagreementDetector()
    
    def create_mock_ensemble_results(self, overall_scores: List[float]) -> List[EnsembleEvaluationResult]:
        """Create mock ensemble results with specified overall scores."""
        ensemble_results = []
        strategies = list(EvaluationStrategy)[:len(overall_scores)]
        
        for i, score in enumerate(overall_scores):
            # Create mock evaluation result
            mock_eval_result = Mock()
            mock_eval_result.overall_score = score
            mock_eval_result.dimensions = [
                EvaluationDimension(
                    name="test_dimension",
                    score=score,
                    confidence=0.8,
                    cultural_relevance=0.7,
                    evidence=[f"Evidence for score {score}"],
                    cultural_markers=[f"marker_{i}"]
                )
            ]
            
            # Create ensemble result
            ensemble_result = EnsembleEvaluationResult(
                strategy=strategies[i],
                evaluation_result=mock_eval_result,
                configuration=self.detector.evaluation_strategies[strategies[i]],
                evaluation_time=1.0
            )
            ensemble_results.append(ensemble_result)
        
        return ensemble_results
    
    def test_coefficient_of_variation_calculations(self):
        """Test coefficient of variation calculations with various distributions."""
        # Test with known values
        test_scores = [0.6, 0.8, 0.7, 0.9, 0.75]
        ensemble_results = self.create_mock_ensemble_results(test_scores)
        
        disagreement_analysis = self.detector._analyze_disagreement(ensemble_results)
        
        # Calculate expected coefficient of variation
        mean_score = statistics.mean(test_scores)
        std_dev = statistics.stdev(test_scores)
        expected_coeff_var = std_dev / mean_score
        
        self.assertAlmostEqual(disagreement_analysis.mean_score, mean_score, places=6)
        self.assertAlmostEqual(disagreement_analysis.standard_deviation, std_dev, places=6)
        self.assertAlmostEqual(disagreement_analysis.coefficient_of_variation, expected_coeff_var, places=6)
        
        # Test consensus level calculation
        expected_consensus = max(0.0, 1.0 - expected_coeff_var * 2)
        self.assertAlmostEqual(disagreement_analysis.consensus_level, expected_consensus, places=6)
    
    def test_strategy_outlier_detection_with_z_scores(self):
        """Test strategy outlier detection using z-scores."""
        # Create scores with clear outliers
        test_scores = [0.8, 0.85, 0.82, 0.1, 0.83]  # 0.1 is clear outlier
        ensemble_results = self.create_mock_ensemble_results(test_scores)
        
        disagreement_analysis = self.detector._analyze_disagreement(ensemble_results)
        
        # Calculate z-scores manually
        mean_score = statistics.mean(test_scores)
        std_dev = statistics.stdev(test_scores)
        outlier_threshold = 2.0
        
        expected_outliers = []
        for i, score in enumerate(test_scores):
            z_score = abs(score - mean_score) / std_dev
            if z_score > outlier_threshold:
                expected_outliers.append(list(EvaluationStrategy)[i])
        
        # Should detect the outlier strategy
        self.assertGreater(len(disagreement_analysis.strategy_outliers), 0)
        if expected_outliers:
            self.assertIn(expected_outliers[0], disagreement_analysis.strategy_outliers)
    
    def test_consensus_calculation_with_different_agreement_patterns(self):
        """Test consensus calculation with different agreement patterns."""
        # Test perfect agreement
        perfect_scores = [0.8, 0.8, 0.8, 0.8]
        perfect_results = self.create_mock_ensemble_results(perfect_scores)
        perfect_analysis = self.detector._analyze_disagreement(perfect_results)
        self.assertEqual(perfect_analysis.consensus_level, 1.0)
        
        # Test maximum disagreement
        extreme_scores = [0.0, 0.5, 1.0, 0.25]
        extreme_results = self.create_mock_ensemble_results(extreme_scores)
        extreme_analysis = self.detector._analyze_disagreement(extreme_results)
        self.assertLess(extreme_analysis.consensus_level, 0.3)
        
        # Test moderate disagreement
        moderate_scores = [0.7, 0.8, 0.75, 0.85]
        moderate_results = self.create_mock_ensemble_results(moderate_scores)
        moderate_analysis = self.detector._analyze_disagreement(moderate_results)
        self.assertGreater(moderate_analysis.consensus_level, 0.5)
        self.assertLess(moderate_analysis.consensus_level, 1.0)
    
    def test_dimension_disagreement_calculations(self):
        """Test dimension-level disagreement calculations."""
        # Create ensemble results with varying dimension scores
        ensemble_results = []
        dimension_scores = [
            {"dim_a": 0.8, "dim_b": 0.7},
            {"dim_a": 0.6, "dim_b": 0.75},
            {"dim_a": 0.9, "dim_b": 0.65}
        ]
        
        for i, dim_scores in enumerate(dimension_scores):
            mock_eval_result = Mock()
            mock_eval_result.overall_score = 0.75
            mock_eval_result.dimensions = [
                EvaluationDimension(
                    name=dim_name, score=score, confidence=0.8, cultural_relevance=0.7,
                    evidence=[], cultural_markers=[]
                )
                for dim_name, score in dim_scores.items()
            ]
            
            ensemble_result = EnsembleEvaluationResult(
                strategy=list(EvaluationStrategy)[i],
                evaluation_result=mock_eval_result,
                configuration=Mock(),
                evaluation_time=1.0
            )
            ensemble_results.append(ensemble_result)
        
        dimension_disagreements = self.detector._calculate_dimension_disagreements(ensemble_results)
        
        # Calculate expected disagreement for dim_a: [0.8, 0.6, 0.9]
        dim_a_scores = [0.8, 0.6, 0.9]
        dim_a_mean = statistics.mean(dim_a_scores)
        dim_a_std = statistics.stdev(dim_a_scores)
        expected_dim_a_disagreement = dim_a_std / dim_a_mean if dim_a_mean > 0 else 0.0
        
        self.assertAlmostEqual(dimension_disagreements["dim_a"], expected_dim_a_disagreement, places=6)
    
    def test_reliability_scoring_algorithm(self):
        """Test reliability scoring with different disagreement scenarios."""
        # High consensus scenario
        high_consensus_scores = [0.8, 0.82, 0.78, 0.81]
        high_consensus_results = self.create_mock_ensemble_results(high_consensus_scores)
        high_consensus_analysis = self.detector._analyze_disagreement(high_consensus_results)
        high_reliability = self.detector._calculate_reliability_score(high_consensus_analysis)
        
        # Low consensus scenario
        low_consensus_scores = [0.3, 0.9, 0.1, 0.7]
        low_consensus_results = self.create_mock_ensemble_results(low_consensus_scores)
        low_consensus_analysis = self.detector._analyze_disagreement(low_consensus_results)
        low_reliability = self.detector._calculate_reliability_score(low_consensus_analysis)
        
        # High consensus should have higher reliability
        self.assertGreater(high_reliability, low_reliability)
        
        # Test bonus for high consensus (> 0.8)
        perfect_scores = [0.85, 0.85, 0.85, 0.85]
        perfect_results = self.create_mock_ensemble_results(perfect_scores)
        perfect_analysis = self.detector._analyze_disagreement(perfect_results)
        perfect_reliability = self.detector._calculate_reliability_score(perfect_analysis)
        
        # Should get bonus for perfect consensus
        self.assertGreaterEqual(perfect_reliability, 0.9)


class TestStrategyModificationLogic(unittest.TestCase):
    """Test strategy-based score modification algorithms."""
    
    def setUp(self):
        """Set up strategy modification test fixtures."""
        self.detector = EnsembleDisagreementDetector()
        
        self.test_dimension = EvaluationDimension(
            name="test_dimension",
            score=0.7,
            confidence=0.8,
            cultural_relevance=0.75,
            evidence=["Test evidence"],
            cultural_markers=["test_marker"]
        )
        
        self.test_result = DomainEvaluationResult(
            domain="test_domain",
            evaluation_type="test_type",
            overall_score=0.72,
            dimensions=[self.test_dimension],
            cultural_context=CulturalContext(),
            metadata={},
            processing_notes=["Test processing"]
        )
    
    def test_scoring_bias_application(self):
        """Test scoring bias modification logic."""
        # Test positive bias (stricter)
        positive_bias_config = EvaluationConfiguration(
            strategy=EvaluationStrategy.AGGRESSIVE,
            weights={},
            cultural_emphasis=0.5,
            confidence_threshold=0.5,
            scoring_bias=0.5  # Strong positive bias
        )
        
        modified_result = self.detector._apply_strategy_modifications(
            self.test_result, positive_bias_config
        )
        
        # Score should be adjusted upward (within bounds)
        original_score = self.test_dimension.score
        modified_score = modified_result.dimensions[0].score
        expected_adjustment = positive_bias_config.scoring_bias * 0.1
        expected_score = min(1.0, max(0.0, original_score + expected_adjustment))
        
        self.assertAlmostEqual(modified_score, expected_score, places=6)
        
        # Test negative bias (more lenient)
        negative_bias_config = EvaluationConfiguration(
            strategy=EvaluationStrategy.CONSERVATIVE,
            weights={},
            cultural_emphasis=0.5,
            confidence_threshold=0.5,
            scoring_bias=-0.3  # Negative bias
        )
        
        modified_result = self.detector._apply_strategy_modifications(
            self.test_result, negative_bias_config
        )
        
        modified_score = modified_result.dimensions[0].score
        expected_adjustment = negative_bias_config.scoring_bias * 0.1
        expected_score = min(1.0, max(0.0, original_score + expected_adjustment))
        
        self.assertAlmostEqual(modified_score, expected_score, places=6)
    
    def test_cultural_emphasis_application(self):
        """Test cultural emphasis modification logic."""
        high_cultural_emphasis_config = EvaluationConfiguration(
            strategy=EvaluationStrategy.CULTURAL_FOCUSED,
            weights={},
            cultural_emphasis=0.9,  # High cultural emphasis
            confidence_threshold=0.5,
            scoring_bias=0.0
        )
        
        modified_result = self.detector._apply_strategy_modifications(
            self.test_result, high_cultural_emphasis_config
        )
        
        # Score should be boosted based on cultural relevance
        original_score = self.test_dimension.score
        modified_score = modified_result.dimensions[0].score
        
        cultural_boost = (high_cultural_emphasis_config.cultural_emphasis - 0.5) * \
                        self.test_dimension.cultural_relevance * 0.2
        expected_score = min(1.0, max(0.0, original_score + cultural_boost))
        
        self.assertAlmostEqual(modified_score, expected_score, places=6)
    
    def test_confidence_threshold_application(self):
        """Test confidence threshold modification logic."""
        high_threshold_config = EvaluationConfiguration(
            strategy=EvaluationStrategy.CONSERVATIVE,
            weights={},
            cultural_emphasis=0.5,
            confidence_threshold=0.9,  # High threshold
            scoring_bias=0.0
        )
        
        modified_result = self.detector._apply_strategy_modifications(
            self.test_result, high_threshold_config
        )
        
        # Confidence should be reduced for dimensions below threshold
        original_confidence = self.test_dimension.confidence
        modified_confidence = modified_result.dimensions[0].confidence
        
        if original_confidence < high_threshold_config.confidence_threshold:
            expected_confidence = original_confidence * 0.5
        else:
            expected_confidence = original_confidence
        
        self.assertAlmostEqual(modified_confidence, expected_confidence, places=6)
    
    def test_weighted_scoring_application(self):
        """Test dimension weight application."""
        weighted_config = EvaluationConfiguration(
            strategy=EvaluationStrategy.DIMENSION_WEIGHTED,
            weights={"test_dimension": 1.5},  # Higher weight
            cultural_emphasis=0.5,
            confidence_threshold=0.5,
            scoring_bias=0.0
        )
        
        modified_result = self.detector._apply_strategy_modifications(
            self.test_result, weighted_config
        )
        
        # Confidence should be adjusted based on weight
        original_confidence = self.test_dimension.confidence
        modified_confidence = modified_result.dimensions[0].confidence
        weight = weighted_config.weights["test_dimension"]
        expected_confidence = original_confidence * weight
        
        self.assertLessEqual(modified_confidence, min(expected_confidence, 1.0))
    
    def test_overall_score_recalculation(self):
        """Test overall score recalculation after modifications."""
        test_config = EvaluationConfiguration(
            strategy=EvaluationStrategy.STANDARD,
            weights={},
            cultural_emphasis=0.7,  # Will boost score
            confidence_threshold=0.5,
            scoring_bias=0.1  # Will also boost score
        )
        
        modified_result = self.detector._apply_strategy_modifications(
            self.test_result, test_config
        )
        
        # Calculate expected overall score manually
        modified_dim = modified_result.dimensions[0]
        weight = modified_dim.confidence * modified_dim.cultural_relevance
        expected_overall = modified_dim.score * weight / weight  # Only one dimension
        
        self.assertAlmostEqual(modified_result.overall_score, expected_overall, places=6)
        
        # Verify processing notes are updated
        self.assertIn("Modified by strategy:", str(modified_result.processing_notes))


class TestConsensusResultCreation(unittest.TestCase):
    """Test consensus result creation algorithms."""
    
    def setUp(self):
        """Set up consensus result test fixtures."""
        self.detector = EnsembleDisagreementDetector()
    
    def create_ensemble_results_with_dimensions(self) -> List[EnsembleEvaluationResult]:
        """Create ensemble results with multiple dimensions for consensus testing."""
        results = []
        
        # Strategy 1 results
        dim1_s1 = EvaluationDimension(
            name="cultural_authenticity", score=0.8, confidence=0.9, cultural_relevance=0.85,
            evidence=["Strong cultural"], cultural_markers=["marker1"]
        )
        dim2_s1 = EvaluationDimension(
            name="traditional_accuracy", score=0.75, confidence=0.8, cultural_relevance=0.7,
            evidence=["Good traditional"], cultural_markers=["marker2"]
        )
        
        eval_result_1 = Mock()
        eval_result_1.domain = "test_domain"
        eval_result_1.evaluation_type = "test_type"
        eval_result_1.overall_score = 0.78
        eval_result_1.dimensions = [dim1_s1, dim2_s1]
        eval_result_1.cultural_context = CulturalContext()
        eval_result_1.metadata = {"strategy": "strategy1"}
        eval_result_1.processing_notes = ["Strategy 1 notes"]
        
        results.append(EnsembleEvaluationResult(
            strategy=EvaluationStrategy.STANDARD,
            evaluation_result=eval_result_1,
            configuration=Mock(),
            evaluation_time=1.0
        ))
        
        # Strategy 2 results
        dim1_s2 = EvaluationDimension(
            name="cultural_authenticity", score=0.85, confidence=0.85, cultural_relevance=0.8,
            evidence=["Very strong cultural"], cultural_markers=["marker1", "marker3"]
        )
        dim2_s2 = EvaluationDimension(
            name="traditional_accuracy", score=0.7, confidence=0.75, cultural_relevance=0.75,
            evidence=["Adequate traditional"], cultural_markers=["marker2"]
        )
        
        eval_result_2 = Mock()
        eval_result_2.domain = "test_domain"
        eval_result_2.evaluation_type = "test_type"
        eval_result_2.overall_score = 0.82
        eval_result_2.dimensions = [dim1_s2, dim2_s2]
        eval_result_2.cultural_context = CulturalContext()
        eval_result_2.metadata = {"strategy": "strategy2"}
        eval_result_2.processing_notes = ["Strategy 2 notes"]
        
        results.append(EnsembleEvaluationResult(
            strategy=EvaluationStrategy.CULTURAL_FOCUSED,
            evaluation_result=eval_result_2,
            configuration=Mock(),
            evaluation_time=1.2
        ))
        
        return results
    
    def test_consensus_dimension_creation(self):
        """Test consensus dimension creation algorithms."""
        ensemble_results = self.create_ensemble_results_with_dimensions()
        
        consensus_dimensions = self.detector._create_consensus_dimensions(ensemble_results)
        
        # Should have both dimensions
        dimension_names = [dim.name for dim in consensus_dimensions]
        self.assertIn("cultural_authenticity", dimension_names)
        self.assertIn("traditional_accuracy", dimension_names)
        
        # Test cultural_authenticity consensus values
        cultural_dim = next(dim for dim in consensus_dimensions if dim.name == "cultural_authenticity")
        
        # Calculate expected consensus score: mean of [0.8, 0.85]
        expected_score = statistics.mean([0.8, 0.85])
        self.assertAlmostEqual(cultural_dim.score, expected_score, places=6)
        
        # Calculate expected consensus confidence: mean of [0.9, 0.85]
        expected_confidence = statistics.mean([0.9, 0.85])
        self.assertAlmostEqual(cultural_dim.confidence, expected_confidence, places=6)
        
        # Calculate expected consensus cultural relevance: mean of [0.85, 0.8]
        expected_cultural_relevance = statistics.mean([0.85, 0.8])
        self.assertAlmostEqual(cultural_dim.cultural_relevance, expected_cultural_relevance, places=6)
        
        # Test evidence aggregation (should combine and deduplicate)
        expected_evidence = ["Strong cultural", "Very strong cultural"]
        self.assertEqual(set(cultural_dim.evidence), set(expected_evidence))
        
        # Test cultural markers aggregation (should deduplicate)
        expected_markers = ["marker1", "marker3"]
        self.assertEqual(set(cultural_dim.cultural_markers), set(expected_markers))
    
    def test_consensus_result_metadata_aggregation(self):
        """Test consensus result metadata aggregation."""
        ensemble_results = self.create_ensemble_results_with_dimensions()
        disagreement_analysis = self.detector._analyze_disagreement(ensemble_results)
        
        consensus_result = self.detector._create_consensus_result(
            ensemble_results, disagreement_analysis
        )
        
        # Test metadata aggregation
        self.assertTrue(consensus_result.metadata.get('ensemble_evaluation', False))
        self.assertIn('strategies_used', consensus_result.metadata)
        self.assertIn('consensus_level', consensus_result.metadata)
        self.assertIn('disagreement_analysis', consensus_result.metadata)
        
        strategies_used = consensus_result.metadata['strategies_used']
        expected_strategies = ['standard', 'cultural_focused']
        self.assertEqual(set(strategies_used), set(expected_strategies))
        
        # Test processing notes aggregation
        self.assertIn("Consensus from 2 evaluation strategies", 
                     ' '.join(consensus_result.processing_notes))


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error handling in ensemble disagreement detection."""
    
    def setUp(self):
        """Set up edge case test fixtures."""
        self.detector = EnsembleDisagreementDetector()
    
    def test_single_strategy_result_handling(self):
        """Test handling when only one strategy succeeds."""
        single_scores = [0.75]
        single_results = []
        
        mock_eval_result = Mock()
        mock_eval_result.overall_score = 0.75
        mock_eval_result.dimensions = [
            EvaluationDimension(
                name="test_dim", score=0.75, confidence=0.8, cultural_relevance=0.7,
                evidence=[], cultural_markers=[]
            )
        ]
        mock_eval_result.cultural_context = CulturalContext(cultural_groups=["test_group"])
        
        single_result = EnsembleEvaluationResult(
            strategy=EvaluationStrategy.STANDARD,
            evaluation_result=mock_eval_result,
            configuration=Mock(),
            evaluation_time=1.0
        )
        single_results.append(single_result)
        
        # Test single strategy result creation
        ensemble_result = self.detector._create_single_strategy_result(single_result)
        
        self.assertEqual(len(ensemble_result.ensemble_results), 1)
        self.assertEqual(ensemble_result.disagreement_analysis.consensus_level, 1.0)
        self.assertEqual(ensemble_result.disagreement_analysis.score_variance, 0.0)
        self.assertEqual(ensemble_result.consensus_result.overall_score, 0.75)
        self.assertEqual(ensemble_result.evaluation_reliability_score, 0.7)  # Moderate reliability
        
        # Should have appropriate validation flag
        flag_types = [flag.flag_type for flag in ensemble_result.validation_flags]
        self.assertIn('single_strategy_only', flag_types)
    
    def test_empty_ensemble_result_handling(self):
        """Test handling when no strategies succeed."""
        empty_result = self.detector._create_single_strategy_result(None)
        
        self.assertEqual(len(empty_result.ensemble_results), 0)
        self.assertEqual(empty_result.disagreement_analysis.consensus_level, 0.0)
        self.assertEqual(empty_result.evaluation_reliability_score, 0.0)
        self.assertIsNone(empty_result.consensus_result)
        
        # Should have ensemble failure flag
        flag_types = [flag.flag_type for flag in empty_result.validation_flags]
        self.assertIn('ensemble_failure', flag_types)
    
    def test_extreme_disagreement_handling(self):
        """Test handling of extreme disagreement scenarios."""
        extreme_scores = [0.0, 1.0, 0.25, 0.75]  # Maximum possible disagreement
        ensemble_results = []
        
        for i, score in enumerate(extreme_scores):
            mock_eval_result = Mock()
            mock_eval_result.overall_score = score
            mock_eval_result.dimensions = [
                EvaluationDimension(
                    name="test_dim", score=score, confidence=0.8, cultural_relevance=0.7,
                    evidence=[], cultural_markers=[]
                )
            ]
            
            ensemble_result = EnsembleEvaluationResult(
                strategy=list(EvaluationStrategy)[i],
                evaluation_result=mock_eval_result,
                configuration=Mock(),
                evaluation_time=1.0
            )
            ensemble_results.append(ensemble_result)
        
        disagreement_analysis = self.detector._analyze_disagreement(ensemble_results)
        
        # Should handle extreme disagreement gracefully
        self.assertGreaterEqual(disagreement_analysis.consensus_level, 0.0)
        self.assertLessEqual(disagreement_analysis.consensus_level, 1.0)
        self.assertGreater(disagreement_analysis.coefficient_of_variation, 0.5)  # High variation
        self.assertGreaterEqual(len(disagreement_analysis.strategy_outliers), 0)  # May detect outliers
    
    def test_identical_scores_handling(self):
        """Test handling when all strategies produce identical scores."""
        identical_scores = [0.75, 0.75, 0.75, 0.75]
        ensemble_results = []
        
        for i, score in enumerate(identical_scores):
            mock_eval_result = Mock()
            mock_eval_result.overall_score = score
            mock_eval_result.dimensions = [
                EvaluationDimension(
                    name="test_dim", score=score, confidence=0.8, cultural_relevance=0.7,
                    evidence=[], cultural_markers=[]
                )
            ]
            
            ensemble_result = EnsembleEvaluationResult(
                strategy=list(EvaluationStrategy)[i],
                evaluation_result=mock_eval_result,
                configuration=Mock(),
                evaluation_time=1.0
            )
            ensemble_results.append(ensemble_result)
        
        disagreement_analysis = self.detector._analyze_disagreement(ensemble_results)
        
        # Should indicate perfect consensus
        self.assertEqual(disagreement_analysis.consensus_level, 1.0)
        self.assertEqual(disagreement_analysis.standard_deviation, 0.0)
        self.assertEqual(disagreement_analysis.coefficient_of_variation, 0.0)
        self.assertEqual(len(disagreement_analysis.strategy_outliers), 0)
        self.assertEqual(len(disagreement_analysis.high_disagreement_dimensions), 0)


if __name__ == '__main__':
    unittest.main()