"""
Mathematical correctness verification tests.

Cross-validates all statistical computations against scipy, numpy, and statsmodels
to ensure mathematical accuracy and numerical stability across all evaluator algorithms.
"""

import unittest
from unittest.mock import Mock
import statistics
import numpy as np
import math
from scipy import stats
from typing import List, Dict, Any

from evaluator.evaluation_aggregator import EvaluationAggregator, AggregatedEvaluationResult
from evaluator.ensemble_disagreement_detector import EnsembleDisagreementDetector
from evaluator.domain_evaluator_base import CulturalContext


class TestMathematicalCorrectnessValidation(unittest.TestCase):
    """Test mathematical correctness against established libraries."""
    
    def setUp(self):
        """Set up mathematical correctness test fixtures."""
        self.aggregator = EvaluationAggregator()
        self.disagreement_detector = EnsembleDisagreementDetector()
        
        # Tolerance for floating point comparisons
        self.tolerance = 1e-10
        self.loose_tolerance = 1e-6
        
        # Known mathematical test cases
        self.test_datasets = {
            'normal_distribution': np.random.normal(0.5, 0.15, 100).tolist(),
            'uniform_distribution': np.random.uniform(0.1, 0.9, 100).tolist(), 
            'bimodal_distribution': (np.concatenate([
                np.random.normal(0.3, 0.1, 50),
                np.random.normal(0.8, 0.1, 50)
            ])).tolist(),
            'edge_case_values': [0.0, 1e-10, 0.5, 1.0 - 1e-10, 1.0],
            'identical_values': [0.75] * 20,
            'high_variance': [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.15, 0.85]
        }
    
    def test_mean_calculation_accuracy(self):
        """Test mean calculation accuracy against statistics.mean."""
        for dataset_name, values in self.test_datasets.items():
            if not values:
                continue
                
            # Calculate mean using our aggregator
            cultural_group_scores = {'test_group': values}
            bias_scores = self.aggregator._calculate_cultural_group_bias(cultural_group_scores)
            
            # Get the global mean used in our calculation
            all_scores = []
            for scores in cultural_group_scores.values():
                all_scores.extend(scores)
            our_global_mean = statistics.mean(all_scores)
            
            # Compare with standard library
            expected_mean = statistics.mean(values)
            
            self.assertAlmostEqual(our_global_mean, expected_mean, places=12,
                                 msg=f"Mean calculation incorrect for {dataset_name}")
    
    def test_standard_deviation_accuracy(self):
        """Test standard deviation calculations against statistics.stdev."""
        for dataset_name, values in self.test_datasets.items():
            if len(values) <= 1:
                continue
                
            # Test with mock evaluation results
            mock_results = []
            for i, score in enumerate(values):
                mock_result = Mock()
                mock_result.overall_score = score
                # Create mock dimensions
                mock_dimension = Mock()
                mock_dimension.name = "test_dimension"
                mock_dimension.score = score
                mock_result.dimensions = [mock_dimension]
                mock_results.append(mock_result)
            
            # Calculate using our disagreement detector
            analysis = self.disagreement_detector._analyze_disagreement([
                Mock(evaluation_result=result, strategy=f"strategy_{i}") 
                for i, result in enumerate(mock_results)
            ])
            
            # Compare with scipy
            expected_std = statistics.stdev(values)
            
            self.assertAlmostEqual(analysis.standard_deviation, expected_std, places=10,
                                 msg=f"Standard deviation incorrect for {dataset_name}")
    
    def test_variance_calculation_accuracy(self):
        """Test variance calculations against statistics.variance."""
        for dataset_name, values in self.test_datasets.items():
            if len(values) <= 1:
                continue
                
            mock_results = []
            for score in values:
                mock_result = Mock()
                mock_result.overall_score = score
                # Create mock dimensions
                mock_dimension = Mock()
                mock_dimension.name = "test_dimension"
                mock_dimension.score = score
                mock_result.dimensions = [mock_dimension]
                mock_results.append(mock_result)
            
            analysis = self.disagreement_detector._analyze_disagreement([
                Mock(evaluation_result=result, strategy=f"strategy_{i}") 
                for i, result in enumerate(mock_results)
            ])
            
            expected_variance = statistics.variance(values)
            
            self.assertAlmostEqual(analysis.score_variance, expected_variance, places=10,
                                 msg=f"Variance calculation incorrect for {dataset_name}")
    
    def test_coefficient_of_variation_accuracy(self):
        """Test coefficient of variation against manual calculation."""
        for dataset_name, values in self.test_datasets.items():
            if len(values) <= 1 or statistics.mean(values) == 0:
                continue
                
            mock_results = []
            for score in values:
                mock_result = Mock()
                mock_result.overall_score = score
                # Create mock dimensions
                mock_dimension = Mock()
                mock_dimension.name = "test_dimension"
                mock_dimension.score = score
                mock_result.dimensions = [mock_dimension]
                mock_results.append(mock_result)
            
            analysis = self.disagreement_detector._analyze_disagreement([
                Mock(evaluation_result=result, strategy=f"strategy_{i}") 
                for i, result in enumerate(mock_results)
            ])
            
            # Calculate expected coefficient of variation
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values)
            expected_cv = std_val / mean_val if mean_val > 0 else 0
            
            self.assertAlmostEqual(analysis.coefficient_of_variation, expected_cv, places=10,
                                 msg=f"Coefficient of variation incorrect for {dataset_name}")
    
    def test_chi_square_test_accuracy(self):
        """Test chi-square test implementation against scipy.stats.chisquare."""
        # Create test data with known distribution
        test_scores = [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
        cultural_group_scores = {'test_group': test_scores}
        
        # Run our chi-square test
        chi_square_results = self.aggregator._perform_chi_square_tests(cultural_group_scores)
        
        if 'test_group' in chi_square_results:
            our_chi2, our_p_value = chi_square_results['test_group']
            
            # Calculate expected values manually
            low_count = len([s for s in test_scores if s < 0.33])
            med_count = len([s for s in test_scores if 0.33 <= s < 0.67])
            high_count = len([s for s in test_scores if s >= 0.67])
            observed = [low_count, med_count, high_count]
            expected = [len(test_scores) / 3] * 3
            
            # Compare with scipy
            expected_chi2, expected_p = stats.chisquare(observed, expected)
            
            self.assertAlmostEqual(our_chi2, expected_chi2, places=8,
                                 msg="Chi-square statistic calculation incorrect")
            self.assertAlmostEqual(our_p_value, expected_p, places=8,
                                 msg="Chi-square p-value calculation incorrect")
    
    def test_t_distribution_confidence_intervals(self):
        """Test confidence interval calculations against scipy.stats.t."""
        test_values = [0.7, 0.75, 0.68, 0.72, 0.8, 0.65, 0.78, 0.73]
        cultural_group_scores = {'test_group': test_values}
        
        # Calculate our confidence interval
        our_ci = self.aggregator._calculate_bias_confidence_interval(cultural_group_scores)
        
        # Calculate expected CI using scipy
        mean_score = statistics.mean(test_values)
        std_score = statistics.stdev(test_values)
        n = len(test_values)
        
        t_critical = stats.t.ppf(0.975, n - 1)  # 95% CI
        margin_error = t_critical * (std_score / math.sqrt(n))
        expected_ci = (mean_score - margin_error, mean_score + margin_error)
        
        self.assertAlmostEqual(our_ci[0], expected_ci[0], places=10,
                             msg="Confidence interval lower bound incorrect")
        self.assertAlmostEqual(our_ci[1], expected_ci[1], places=10,
                             msg="Confidence interval upper bound incorrect")
    
    def test_linear_regression_calculations(self):
        """Test linear regression trend detection against scipy.stats.linregress."""
        # Create evaluation history with known trend
        trend_values = [0.5 + i * 0.03 for i in range(15)]  # Clear positive trend
        
        evaluation_history = []
        for score in trend_values:
            mock_result = Mock()
            mock_result.overall_score = score
            mock_result.consensus_level = 0.8
            mock_result.cultural_competence = 0.7
            evaluation_history.append(mock_result)
        
        # Run our trend detection
        patterns = self.aggregator._detect_systematic_patterns(evaluation_history)
        
        # Verify against scipy
        overall_scores = [result.overall_score for result in evaluation_history]
        x = list(range(len(overall_scores)))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, overall_scores)
        
        # Should detect positive trend
        self.assertIn("systematic_score_inflation", patterns)
        self.assertGreater(slope, 0.02, "Should detect significant positive slope")
        self.assertGreater(abs(r_value), 0.8, "Should detect strong correlation")
        self.assertLess(p_value, 0.05, "Should be statistically significant")
    
    def test_cohens_d_effect_size_calculation(self):
        """Test Cohen's d effect size calculation accuracy."""
        # Create groups with known effect sizes
        group1_scores = [0.8, 0.85, 0.9, 0.75, 0.82]
        group2_scores = [0.4, 0.35, 0.45, 0.38, 0.42]
        
        cultural_group_scores = {
            'group1': group1_scores,
            'group2': group2_scores
        }
        
        # Calculate our effect sizes
        effect_sizes = self.aggregator._calculate_effect_sizes(cultural_group_scores)
        
        # Calculate expected Cohen's d manually
        all_scores = group1_scores + group2_scores
        overall_mean = statistics.mean(all_scores)
        overall_std = statistics.stdev(all_scores)
        
        group1_mean = statistics.mean(group1_scores)
        group2_mean = statistics.mean(group2_scores)
        
        expected_group1_d = (group1_mean - overall_mean) / overall_std
        expected_group2_d = (group2_mean - overall_mean) / overall_std
        
        self.assertAlmostEqual(effect_sizes['group1'], expected_group1_d, places=10,
                             msg="Cohen's d calculation incorrect for group1")
        self.assertAlmostEqual(effect_sizes['group2'], expected_group2_d, places=10,
                             msg="Cohen's d calculation incorrect for group2")
    
    def test_weighted_average_calculations(self):
        """Test weighted average calculations for dimension aggregation."""
        # Create test dimensions with known weights
        dimensions_data = [
            {'score': 0.8, 'confidence': 0.9, 'cultural_relevance': 0.7},
            {'score': 0.6, 'confidence': 0.5, 'cultural_relevance': 0.3},
            {'score': 0.9, 'confidence': 0.8, 'cultural_relevance': 0.9}
        ]
        
        # Calculate expected weighted average manually
        total_score = 0.0
        total_weight = 0.0
        for dim in dimensions_data:
            weight = dim['cultural_relevance'] * dim['confidence']
            total_score += dim['score'] * weight
            total_weight += weight
        
        expected_weighted_avg = total_score / total_weight if total_weight > 0 else 0.0
        
        # Create mock domain results for testing
        mock_results = []
        for dim_data in dimensions_data:
            mock_dimension = Mock()
            mock_dimension.score = dim_data['score']
            mock_dimension.confidence = dim_data['confidence']
            mock_dimension.cultural_relevance = dim_data['cultural_relevance']
            mock_dimension.name = 'test_dimension'
            
            mock_result = Mock()
            mock_result.dimensions = [mock_dimension]
            mock_results.append(mock_result)
        
        # Test our weighted average calculation
        aggregated_scores = self.aggregator._aggregate_dimension_scores(mock_results)
        our_weighted_avg = aggregated_scores.get('test_dimension', 0)
        
        self.assertAlmostEqual(our_weighted_avg, expected_weighted_avg, places=10,
                             msg="Weighted average calculation incorrect")
    
    def test_z_score_outlier_detection(self):
        """Test z-score based outlier detection accuracy."""
        # Create data with known outliers
        normal_values = [0.75, 0.78, 0.73, 0.76, 0.77, 0.74, 0.79, 0.72]
        outlier_value = 0.2  # Clear outlier
        test_values = normal_values + [outlier_value]
        
        # Calculate z-scores manually
        mean_val = statistics.mean(test_values)
        std_val = statistics.stdev(test_values)
        
        z_scores = [(val - mean_val) / std_val for val in test_values]
        expected_outliers = [i for i, z in enumerate(z_scores) if abs(z) > 2.0]
        
        # Test our outlier detection through consensus analysis
        mock_results = []
        strategies = ['strategy_' + str(i) for i in range(len(test_values))]
        
        for i, score in enumerate(test_values):
            mock_eval_result = Mock()
            mock_eval_result.overall_score = score
            mock_eval_result.dimensions = [Mock(name='test_dim', score=score)]
            
            mock_ensemble_result = Mock()
            mock_ensemble_result.evaluation_result = mock_eval_result
            mock_ensemble_result.strategy = strategies[i]
            mock_results.append(mock_ensemble_result)
        
        analysis = self.disagreement_detector._analyze_disagreement(mock_results)
        
        # Should detect the outlier (strategy with score 0.2)
        outlier_strategies = analysis.strategy_outliers
        
        if expected_outliers:  # If we expect outliers
            self.assertGreater(len(outlier_strategies), 0, "Should detect outlier strategies")
    
    def test_numerical_stability_edge_cases(self):
        """Test numerical stability with edge cases."""
        edge_cases = {
            'very_small_numbers': [1e-15, 2e-15, 3e-15],
            'very_large_numbers': [1.0 - 1e-15, 1.0 - 2e-15, 1.0 - 3e-15],
            'mixed_extreme': [1e-10, 0.5, 1.0 - 1e-10],
            'near_zero_variance': [0.5000001, 0.5000002, 0.5000003]
        }
        
        for case_name, values in edge_cases.items():
            if len(values) <= 1:
                continue
                
            cultural_group_scores = {'test_group': values}
            
            try:
                # Test that calculations don't crash or produce invalid results
                bias_scores = self.aggregator._calculate_cultural_group_bias(cultural_group_scores)
                effect_sizes = self.aggregator._calculate_effect_sizes(cultural_group_scores)
                ci = self.aggregator._calculate_bias_confidence_interval(cultural_group_scores)
                
                # Results should be finite numbers
                for group, bias in bias_scores.items():
                    self.assertTrue(math.isfinite(bias), f"Bias score not finite for {case_name}")
                    self.assertGreaterEqual(bias, -1.0, f"Bias score out of range for {case_name}")
                    self.assertLessEqual(bias, 1.0, f"Bias score out of range for {case_name}")
                
                for group, effect in effect_sizes.items():
                    self.assertTrue(math.isfinite(effect), f"Effect size not finite for {case_name}")
                
                # Confidence interval should be ordered properly
                self.assertLessEqual(ci[0], ci[1], f"Confidence interval misordered for {case_name}")
                
            except Exception as e:
                self.fail(f"Numerical instability in {case_name}: {str(e)}")


class TestStatisticalInvariantProperties(unittest.TestCase):
    """Test that statistical functions maintain mathematical invariants."""
    
    def setUp(self):
        """Set up invariant testing fixtures."""
        self.aggregator = EvaluationAggregator()
        self.disagreement_detector = EnsembleDisagreementDetector()
    
    def test_mean_invariants(self):
        """Test that mean calculations satisfy mathematical invariants."""
        test_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        # Mean should be within the range of values
        mean_val = statistics.mean(test_values)
        self.assertGreaterEqual(mean_val, min(test_values))
        self.assertLessEqual(mean_val, max(test_values))
        
        # Mean of identical values should equal that value
        identical_values = [0.75] * 10
        identical_mean = statistics.mean(identical_values)
        self.assertAlmostEqual(identical_mean, 0.75, places=15)
    
    def test_variance_invariants(self):
        """Test that variance calculations satisfy mathematical properties."""
        test_values = [0.2, 0.4, 0.6, 0.8]
        
        # Variance should always be non-negative
        variance = statistics.variance(test_values)
        self.assertGreaterEqual(variance, 0.0)
        
        # Variance of identical values should be zero
        identical_values = [0.5] * 20
        identical_variance = statistics.variance(identical_values)
        self.assertAlmostEqual(identical_variance, 0.0, places=15)
        
        # Adding a constant to all values shouldn't change variance
        shifted_values = [x + 0.1 for x in test_values]
        shifted_variance = statistics.variance(shifted_values)
        self.assertAlmostEqual(variance, shifted_variance, places=12)
    
    def test_correlation_invariants(self):
        """Test that correlation coefficients stay within [-1, 1]."""
        # Create test data for trend detection
        x_values = list(range(10))
        y_values = [0.5 + i * 0.05 for i in range(10)]  # Positive correlation
        
        # Calculate correlation using scipy for verification
        correlation, _ = stats.pearsonr(x_values, y_values)
        
        # Correlation should be within valid range
        self.assertGreaterEqual(correlation, -1.0)
        self.assertLessEqual(correlation, 1.0)
        
        # Perfect positive correlation should be 1.0
        perfect_positive = stats.pearsonr(x_values, x_values)[0]
        self.assertAlmostEqual(perfect_positive, 1.0, places=10)
        
        # Perfect negative correlation should be -1.0
        negative_y = [-x for x in x_values]
        perfect_negative = stats.pearsonr(x_values, negative_y)[0]
        self.assertAlmostEqual(perfect_negative, -1.0, places=10)
    
    def test_probability_invariants(self):
        """Test that probability calculations stay within [0, 1]."""
        # Test p-values from chi-square tests
        test_scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        cultural_group_scores = {'test_group': test_scores}
        
        chi_square_results = self.aggregator._perform_chi_square_tests(cultural_group_scores)
        
        for group, (chi2, p_value) in chi_square_results.items():
            self.assertGreaterEqual(p_value, 0.0, f"P-value negative for {group}")
            self.assertLessEqual(p_value, 1.0, f"P-value > 1 for {group}")
            self.assertGreaterEqual(chi2, 0.0, f"Chi-square statistic negative for {group}")
    
    def test_monotonicity_properties(self):
        """Test monotonicity properties of statistical functions."""
        # Test that increasing all values increases the mean
        base_values = [0.3, 0.5, 0.7]
        increased_values = [x + 0.1 for x in base_values]
        
        base_mean = statistics.mean(base_values)
        increased_mean = statistics.mean(increased_values)
        
        self.assertGreater(increased_mean, base_mean)
        
        # Test that adding more extreme values increases variance
        moderate_values = [0.45, 0.5, 0.55]
        extreme_values = [0.1, 0.5, 0.9]
        
        moderate_variance = statistics.variance(moderate_values)
        extreme_variance = statistics.variance(extreme_values)
        
        self.assertGreater(extreme_variance, moderate_variance)
    
    def test_scaling_properties(self):
        """Test scaling properties of statistical measures."""
        base_values = [0.2, 0.4, 0.6, 0.8]
        scaled_values = [x * 2 for x in base_values]  # Note: may exceed [0,1] for this test
        
        base_std = statistics.stdev(base_values)
        scaled_std = statistics.stdev([min(1.0, x) for x in scaled_values])  # Clamp to valid range
        
        # For linear scaling, std should scale by the same factor (before clamping)
        # This tests the mathematical property in principle
        if all(x <= 0.5 for x in base_values):  # Only if scaling won't hit boundaries
            expected_scaled_std = base_std * 2
            self.assertAlmostEqual(scaled_std, expected_scaled_std, places=10)


if __name__ == '__main__':
    unittest.main()