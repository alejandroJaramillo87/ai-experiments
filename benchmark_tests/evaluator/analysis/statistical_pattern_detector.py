"""
Statistical Pattern Detector

Analyzes cognitive patterns across reasoning categories to identify systematic
model strengths and weaknesses using statistical significance testing.
"""

import json
import glob
import statistics
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import logging

try:
    from scipy import stats
    from scipy.stats import f_oneway
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available - using simplified statistical analysis")

logger = logging.getLogger(__name__)


@dataclass
class CategoryPattern:
    """Pattern analysis for a specific reasoning category"""
    category: str
    test_count: int
    overall_score_mean: float
    overall_score_std: float
    cognitive_metrics: Dict[str, float]
    cognitive_std: Dict[str, float]


@dataclass
class PatternAnalysis:
    """Complete statistical pattern analysis results"""
    categories: List[CategoryPattern]
    inter_category_variance: Dict[str, float]
    statistical_significance: Dict[str, float]  # p-values
    effect_sizes: Dict[str, float]  # Cohen's d
    classification_accuracy: float
    pattern_summary: str
    recommendations: List[str]


class StatisticalPatternDetector:
    """
    Analyzes cognitive patterns across reasoning categories for systematic
    model capability identification.
    """
    
    def __init__(self):
        self.cognitive_metrics = [
            'task_understanding',
            'instruction_following', 
            'context_awareness',
            'logical_structure',
            'evidence_integration',
            'inference_quality',
            'relevance_score',
            'depth_score',
            'coherence_score',
            'mathematical_reasoning',
            'cultural_sensitivity',
            'creative_synthesis',
            'analytical_decomposition'
        ]
    
    def analyze_reasoning_patterns(self, results_directory: str) -> PatternAnalysis:
        """
        Perform comprehensive statistical pattern analysis on reasoning test results.
        
        Args:
            results_directory: Directory containing JSON result files
            
        Returns:
            PatternAnalysis with complete statistical findings
        """
        logger.info(f"Starting pattern analysis on {results_directory}")
        
        # Load and categorize results
        category_data = self._load_categorized_results(results_directory)
        
        if not category_data:
            return self._create_empty_analysis("No results found")
        
        # Calculate category patterns
        category_patterns = self._calculate_category_patterns(category_data)
        
        # Statistical significance testing
        statistical_significance = self._calculate_statistical_significance(category_data)
        
        # Effect size calculations
        effect_sizes = self._calculate_effect_sizes(category_data)
        
        # Inter-category variance analysis
        inter_category_variance = self._calculate_inter_category_variance(category_patterns)
        
        # Classification accuracy assessment
        classification_accuracy = self._assess_classification_accuracy(category_data)
        
        # Generate pattern summary and recommendations
        pattern_summary = self._generate_pattern_summary(category_patterns, statistical_significance)
        recommendations = self._generate_recommendations(category_patterns, statistical_significance, effect_sizes)
        
        return PatternAnalysis(
            categories=category_patterns,
            inter_category_variance=inter_category_variance,
            statistical_significance=statistical_significance,
            effect_sizes=effect_sizes,
            classification_accuracy=classification_accuracy,
            pattern_summary=pattern_summary,
            recommendations=recommendations
        )
    
    def _load_categorized_results(self, results_directory: str) -> Dict[str, List[Dict]]:
        """Load and categorize test results by reasoning category"""
        result_files = glob.glob(f"{results_directory}/*_result.json")
        category_data = defaultdict(list)
        
        logger.info(f"Loading {len(result_files)} result files")
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                
                test_id = result['test_id']
                eval_result = result.get('evaluation_result', {})
                metrics = eval_result.get('metrics', {})
                
                # Determine category from test_id
                category = self._determine_category(test_id)
                
                if category == 'unknown':
                    logger.warning(f"Unknown category for test_id: {test_id}")
                    continue
                
                # Extract cognitive metrics
                cognitive_scores = {}
                for metric in self.cognitive_metrics:
                    cognitive_scores[metric] = metrics.get(metric, 0.0)
                
                cognitive_scores['overall_score'] = eval_result.get('overall_score', 0)
                cognitive_scores['test_id'] = test_id
                
                category_data[category].append(cognitive_scores)
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        return dict(category_data)
    
    def _determine_category(self, test_id: str) -> str:
        """Map test_id to reasoning category"""
        category_mapping = {
            'basic_': 'basic_logic_patterns',
            'chain_': 'chain_of_thought', 
            'inference_': 'multi_step_inference',
            'math_': 'elementary_math_science',
            'cultural_': 'cultural_reasoning',
            'verify_': 'self_verification_reflection',
            'logic_': 'logic_systems_comparison'
        }
        
        for prefix, category in category_mapping.items():
            if test_id.startswith(prefix):
                return category
        
        return 'unknown'
    
    def _calculate_category_patterns(self, category_data: Dict[str, List[Dict]]) -> List[CategoryPattern]:
        """Calculate statistical patterns for each category"""
        patterns = []
        
        for category, scores_list in category_data.items():
            if not scores_list:
                continue
            
            # Overall score statistics
            overall_scores = [s['overall_score'] for s in scores_list]
            overall_mean = statistics.mean(overall_scores)
            overall_std = statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0
            
            # Cognitive metric averages and standard deviations
            cognitive_metrics = {}
            cognitive_std = {}
            
            for metric in self.cognitive_metrics:
                values = [s.get(metric, 0.0) for s in scores_list]
                cognitive_metrics[metric] = statistics.mean(values)
                cognitive_std[metric] = statistics.stdev(values) if len(values) > 1 else 0.0
            
            pattern = CategoryPattern(
                category=category,
                test_count=len(scores_list),
                overall_score_mean=overall_mean,
                overall_score_std=overall_std,
                cognitive_metrics=cognitive_metrics,
                cognitive_std=cognitive_std
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def _calculate_statistical_significance(self, category_data: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Calculate p-values for inter-category differences using ANOVA"""
        significance_results = {}
        
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available - skipping ANOVA testing")
            return significance_results
        
        # Test each cognitive metric for significant differences between categories
        for metric in self.cognitive_metrics + ['overall_score']:
            category_values = []
            
            for category, scores_list in category_data.items():
                if len(scores_list) >= 2:  # Need at least 2 samples per group
                    values = [s.get(metric, 0.0) for s in scores_list]
                    category_values.append(values)
            
            if len(category_values) >= 2:  # Need at least 2 categories
                try:
                    f_stat, p_value = f_oneway(*category_values)
                    # Handle NaN p-values (when all values are constant)
                    if np.isnan(p_value) or np.isinf(p_value):
                        significance_results[metric] = 1.0  # No significance for constant values
                    else:
                        significance_results[metric] = p_value
                except Exception as e:
                    logger.warning(f"ANOVA failed for {metric}: {e}")
                    significance_results[metric] = 1.0  # No significance
            else:
                significance_results[metric] = 1.0  # Insufficient data
        
        return significance_results
    
    def _calculate_effect_sizes(self, category_data: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Calculate Cohen's d effect sizes between categories"""
        effect_sizes = {}
        
        categories = list(category_data.keys())
        if len(categories) < 2:
            return effect_sizes
        
        # Calculate effect sizes for key metrics
        key_metrics = ['task_understanding', 'logical_structure', 'overall_score']
        
        for metric in key_metrics:
            max_effect_size = 0.0
            
            # Compare all category pairs
            for i in range(len(categories)):
                for j in range(i + 1, len(categories)):
                    cat1, cat2 = categories[i], categories[j]
                    
                    values1 = [s.get(metric, 0.0) for s in category_data[cat1]]
                    values2 = [s.get(metric, 0.0) for s in category_data[cat2]]
                    
                    if len(values1) >= 2 and len(values2) >= 2:
                        cohens_d = self._cohens_d(values1, values2)
                        max_effect_size = max(max_effect_size, abs(cohens_d))
            
            effect_sizes[metric] = max_effect_size
        
        return effect_sizes
    
    def _cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0
        
        mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
        std1, std2 = statistics.stdev(group1), statistics.stdev(group2)
        
        # Pooled standard deviation
        pooled_std = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        pooled_std = pooled_std**0.5
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def _calculate_inter_category_variance(self, patterns: List[CategoryPattern]) -> Dict[str, float]:
        """Calculate variance between categories for each metric"""
        variance_results = {}
        
        for metric in self.cognitive_metrics + ['overall_score']:
            if metric == 'overall_score':
                values = [p.overall_score_mean for p in patterns]
            else:
                values = [p.cognitive_metrics.get(metric, 0.0) for p in patterns]
            
            if len(values) > 1:
                variance_results[metric] = statistics.stdev(values)
            else:
                variance_results[metric] = 0.0
        
        return variance_results
    
    def _assess_classification_accuracy(self, category_data: Dict[str, List[Dict]]) -> float:
        """Assess how well cognitive metrics can predict reasoning category"""
        # Simplified classification accuracy using nearest neighbor
        correct_predictions = 0
        total_predictions = 0
        
        categories = list(category_data.keys())
        if len(categories) < 2:
            return 0.0
        
        # Calculate category centroids for key metrics
        category_centroids = {}
        key_metrics = ['task_understanding', 'logical_structure', 'evidence_integration']
        
        for category in categories:
            scores_list = category_data[category]
            if scores_list:
                centroid = {}
                for metric in key_metrics:
                    values = [s.get(metric, 0.0) for s in scores_list]
                    centroid[metric] = statistics.mean(values)
                category_centroids[category] = centroid
        
        # Test each data point against centroids
        for true_category, scores_list in category_data.items():
            for scores in scores_list:
                # Find nearest centroid
                min_distance = float('inf')
                predicted_category = None
                
                test_point = {metric: scores.get(metric, 0.0) for metric in key_metrics}
                
                for category, centroid in category_centroids.items():
                    distance = self._euclidean_distance(test_point, centroid)
                    if distance < min_distance:
                        min_distance = distance
                        predicted_category = category
                
                if predicted_category == true_category:
                    correct_predictions += 1
                total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _euclidean_distance(self, point1: Dict[str, float], point2: Dict[str, float]) -> float:
        """Calculate Euclidean distance between two points"""
        distance = 0.0
        for key in point1:
            if key in point2:
                distance += (point1[key] - point2[key]) ** 2
        return distance ** 0.5
    
    def _generate_pattern_summary(self, patterns: List[CategoryPattern], significance: Dict[str, float]) -> str:
        """Generate human-readable pattern summary"""
        summary = ["COGNITIVE PATTERN ANALYSIS SUMMARY", "="*50, ""]
        
        # Overall assessment
        significant_metrics = [m for m, p in significance.items() if p < 0.05]
        if significant_metrics:
            summary.append(f"✅ PATTERNS DETECTED: {len(significant_metrics)} metrics show significant differences")
            summary.append(f"Significant metrics: {', '.join(significant_metrics)}")
        else:
            summary.append("⚠️  LIMITED PATTERNS: No statistically significant differences detected")
        
        summary.append("")
        
        # Category rankings
        summary.append("CATEGORY PERFORMANCE RANKING:")
        sorted_patterns = sorted(patterns, key=lambda p: p.overall_score_mean, reverse=True)
        for i, pattern in enumerate(sorted_patterns, 1):
            summary.append(f"{i}. {pattern.category}: {pattern.overall_score_mean:.1f}/100")
        
        summary.append("")
        
        # Cognitive strengths by category
        summary.append("COGNITIVE STRENGTHS BY CATEGORY:")
        for pattern in sorted_patterns:
            top_metrics = sorted(pattern.cognitive_metrics.items(), key=lambda x: x[1], reverse=True)[:3]
            top_metrics_str = ", ".join([f"{m}: {v:.2f}" for m, v in top_metrics if v > 0.1])
            if top_metrics_str:
                summary.append(f"• {pattern.category}: {top_metrics_str}")
        
        return "\n".join(summary)
    
    def _generate_recommendations(self, patterns: List[CategoryPattern], 
                                significance: Dict[str, float], 
                                effect_sizes: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on pattern analysis"""
        recommendations = []
        
        # Statistical significance recommendations
        significant_metrics = [m for m, p in significance.items() if p < 0.05]
        if significant_metrics:
            recommendations.append(f"✅ Strong cognitive discrimination detected in {len(significant_metrics)} metrics")
            recommendations.append("Continue with full-scale pattern analysis across all domains")
        else:
            recommendations.append("⚠️ Limited statistical significance - consider increasing sample size")
            recommendations.append("Or pivot to qualitative LLM panel evaluation approach")
        
        # Effect size recommendations  
        large_effects = [m for m, d in effect_sizes.items() if d > 0.8]
        if large_effects:
            recommendations.append(f"🎯 Large effect sizes detected in: {', '.join(large_effects)}")
            recommendations.append("These metrics show strong practical significance for model evaluation")
        
        # Category-specific recommendations
        sorted_patterns = sorted(patterns, key=lambda p: p.overall_score_mean, reverse=True)
        lowest_category = sorted_patterns[-1]
        highest_category = sorted_patterns[0]
        
        recommendations.append(f"📊 Strongest performance: {highest_category.category} ({highest_category.overall_score_mean:.1f}/100)")
        recommendations.append(f"📉 Weakest performance: {lowest_category.category} ({lowest_category.overall_score_mean:.1f}/100)")
        
        # Model improvement recommendations
        weak_metrics = []
        for pattern in patterns:
            for metric, value in pattern.cognitive_metrics.items():
                if value < 0.3 and metric in ['task_understanding', 'logical_structure', 'evidence_integration']:
                    weak_metrics.append(metric)
        
        if weak_metrics:
            unique_weak = list(set(weak_metrics))
            recommendations.append(f"🔧 Focus improvement areas: {', '.join(unique_weak)}")
        
        return recommendations
    
    def _create_empty_analysis(self, reason: str) -> PatternAnalysis:
        """Create empty analysis result with error message"""
        return PatternAnalysis(
            categories=[],
            inter_category_variance={},
            statistical_significance={},
            effect_sizes={},
            classification_accuracy=0.0,
            pattern_summary=f"Analysis failed: {reason}",
            recommendations=[f"Error: {reason}"]
        )


def run_pattern_analysis(results_directory: str, output_file: Optional[str] = None) -> PatternAnalysis:
    """
    Convenience function to run complete pattern analysis
    
    Args:
        results_directory: Directory with JSON result files
        output_file: Optional file to save analysis results
        
    Returns:
        PatternAnalysis results
    """
    detector = StatisticalPatternDetector()
    analysis = detector.analyze_reasoning_patterns(results_directory)
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({
                'pattern_summary': analysis.pattern_summary,
                'statistical_significance': analysis.statistical_significance,
                'effect_sizes': analysis.effect_sizes,
                'classification_accuracy': analysis.classification_accuracy,
                'recommendations': analysis.recommendations,
                'categories': [
                    {
                        'category': p.category,
                        'test_count': p.test_count,
                        'overall_score_mean': p.overall_score_mean,
                        'cognitive_metrics': p.cognitive_metrics
                    }
                    for p in analysis.categories
                ]
            }, f, indent=2)
        
        logger.info(f"Analysis saved to {output_file}")
    
    return analysis