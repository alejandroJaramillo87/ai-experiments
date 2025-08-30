from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import statistics
import numpy as np
from enum import Enum
from itertools import combinations

from .domain_evaluator_base import DomainEvaluationResult, EvaluationDimension, CulturalContext, MultiDimensionalEvaluator
from .evaluation_aggregator import ValidationFlag, EvaluationAggregator
from ..subjects.domain_evaluation_router import DomainEvaluationRouter


class EvaluationStrategy(Enum):
    """Different evaluation strategies for ensemble disagreement detection."""
    STANDARD = "standard"
    CONSERVATIVE = "conservative" 
    AGGRESSIVE = "aggressive"
    CULTURAL_FOCUSED = "cultural_focused"
    DIMENSION_WEIGHTED = "dimension_weighted"


@dataclass
class EvaluationConfiguration:
    """Configuration for an evaluation strategy."""
    strategy: EvaluationStrategy
    weights: Dict[str, float]  # dimension -> weight
    cultural_emphasis: float  # 0.0 to 1.0
    confidence_threshold: float
    scoring_bias: float  # -1.0 (lenient) to 1.0 (strict)


@dataclass
class EnsembleEvaluationResult:
    """Result from ensemble evaluation."""
    strategy: EvaluationStrategy
    evaluation_result: DomainEvaluationResult
    configuration: EvaluationConfiguration
    evaluation_time: float


@dataclass
class DisagreementAnalysis:
    """Analysis of disagreement between ensemble evaluations."""
    mean_score: float
    score_variance: float
    score_range: Tuple[float, float]
    standard_deviation: float
    coefficient_of_variation: float
    dimension_disagreements: Dict[str, float]  # dimension -> disagreement level
    strategy_outliers: List[EvaluationStrategy]
    consensus_level: float  # 0.0 to 1.0
    high_disagreement_dimensions: List[str]


@dataclass
class EnsembleDisagreementResult:
    """Result of ensemble disagreement analysis."""
    ensemble_results: List[EnsembleEvaluationResult]
    disagreement_analysis: DisagreementAnalysis
    consensus_result: DomainEvaluationResult
    validation_flags: List[ValidationFlag]
    recommended_strategies: List[EvaluationStrategy]
    cultural_authenticity_consensus: float
    evaluation_reliability_score: float


class EnsembleDisagreementDetector:
    """Detects disagreement patterns by running evaluations with different strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.evaluation_strategies = self._initialize_evaluation_strategies()
        self.disagreement_threshold = self.config.get('disagreement_threshold', 0.3)
        self.consensus_threshold = self.config.get('consensus_threshold', 0.7)
        
    def _initialize_evaluation_strategies(self) -> Dict[EvaluationStrategy, EvaluationConfiguration]:
        """Initialize different evaluation strategies for ensemble analysis."""
        return {
            EvaluationStrategy.STANDARD: EvaluationConfiguration(
                strategy=EvaluationStrategy.STANDARD,
                weights={},  # Equal weights
                cultural_emphasis=0.5,
                confidence_threshold=0.5,
                scoring_bias=0.0
            ),
            EvaluationStrategy.CONSERVATIVE: EvaluationConfiguration(
                strategy=EvaluationStrategy.CONSERVATIVE,
                weights={},  # Equal weights
                cultural_emphasis=0.3,
                confidence_threshold=0.7,
                scoring_bias=-0.2  # More lenient scoring
            ),
            EvaluationStrategy.AGGRESSIVE: EvaluationConfiguration(
                strategy=EvaluationStrategy.AGGRESSIVE,
                weights={},  # Equal weights
                cultural_emphasis=0.7,
                confidence_threshold=0.3,
                scoring_bias=0.2  # Stricter scoring
            ),
            EvaluationStrategy.CULTURAL_FOCUSED: EvaluationConfiguration(
                strategy=EvaluationStrategy.CULTURAL_FOCUSED,
                weights={},  # Will be set dynamically based on cultural relevance
                cultural_emphasis=0.9,
                confidence_threshold=0.6,
                scoring_bias=0.1
            ),
            EvaluationStrategy.DIMENSION_WEIGHTED: EvaluationConfiguration(
                strategy=EvaluationStrategy.DIMENSION_WEIGHTED,
                weights={  # Dynamic weights based on dimension importance
                    'cultural_authenticity': 0.3,
                    'traditional_accuracy': 0.3,
                    'contextual_appropriateness': 0.2,
                    'linguistic_competence': 0.2
                },
                cultural_emphasis=0.6,
                confidence_threshold=0.5,
                scoring_bias=0.0
            )
        }
    
    def detect_evaluation_disagreement(self, 
                                     content: str,
                                     cultural_context: CulturalContext,
                                     domain: str,
                                     test_metadata: Dict[str, Any],
                                     evaluator: MultiDimensionalEvaluator,
                                     strategies: List[EvaluationStrategy] = None) -> EnsembleDisagreementResult:
        """
        Detect disagreement by running evaluation with multiple strategies.
        
        Args:
            content: Content to evaluate
            cultural_context: Cultural context
            domain: Evaluation domain
            test_metadata: Test metadata
            evaluator: Base evaluator to use with different strategies
            strategies: Strategies to use (default: all)
            
        Returns:
            EnsembleDisagreementResult with disagreement analysis
        """
        if strategies is None:
            strategies = list(EvaluationStrategy)
        
        # Run evaluations with different strategies
        ensemble_results = []
        for strategy in strategies:
            try:
                result = self._evaluate_with_strategy(
                    strategy, content, cultural_context, test_metadata, evaluator
                )
                ensemble_results.append(result)
            except Exception as e:
                print(f"Failed to evaluate with strategy {strategy.value}: {str(e)}")
                continue
        
        if len(ensemble_results) < 2:
            return self._create_single_strategy_result(ensemble_results[0] if ensemble_results else None)
        
        # Analyze disagreement
        disagreement_analysis = self._analyze_disagreement(ensemble_results)
        
        # Create consensus result
        consensus_result = self._create_consensus_result(ensemble_results, disagreement_analysis)
        
        # Generate validation flags
        validation_flags = self._generate_disagreement_flags(disagreement_analysis, cultural_context)
        
        # Recommend strategies
        recommended_strategies = self._recommend_strategies(ensemble_results, disagreement_analysis)
        
        # Calculate cultural authenticity consensus
        cultural_consensus = self._calculate_cultural_consensus(ensemble_results)
        
        # Calculate evaluation reliability
        reliability_score = self._calculate_reliability_score(disagreement_analysis)
        
        return EnsembleDisagreementResult(
            ensemble_results=ensemble_results,
            disagreement_analysis=disagreement_analysis,
            consensus_result=consensus_result,
            validation_flags=validation_flags,
            recommended_strategies=recommended_strategies,
            cultural_authenticity_consensus=cultural_consensus,
            evaluation_reliability_score=reliability_score
        )
    
    def _evaluate_with_strategy(self, 
                              strategy: EvaluationStrategy,
                              content: str,
                              cultural_context: CulturalContext,
                              test_metadata: Dict[str, Any],
                              base_evaluator: MultiDimensionalEvaluator) -> EnsembleEvaluationResult:
        """Evaluate content using a specific strategy."""
        import time
        start_time = time.time()
        
        configuration = self.evaluation_strategies[strategy]
        
        # Create modified evaluator based on strategy
        modified_evaluator = self._create_strategy_evaluator(base_evaluator, configuration)
        
        # Run evaluation
        evaluation_result = modified_evaluator.evaluate(content, test_metadata, cultural_context)
        
        # Apply strategy-specific modifications to result
        modified_result = self._apply_strategy_modifications(evaluation_result, configuration)
        
        return EnsembleEvaluationResult(
            strategy=strategy,
            evaluation_result=modified_result,
            configuration=configuration,
            evaluation_time=time.time() - start_time
        )
    
    def _create_strategy_evaluator(self, 
                                 base_evaluator: MultiDimensionalEvaluator,
                                 configuration: EvaluationConfiguration) -> MultiDimensionalEvaluator:
        """Create a modified evaluator based on strategy configuration."""
        # For now, return the base evaluator
        # In a full implementation, this would create a wrapper that modifies behavior
        return base_evaluator
    
    def _apply_strategy_modifications(self, 
                                    result: DomainEvaluationResult,
                                    configuration: EvaluationConfiguration) -> DomainEvaluationResult:
        """Apply strategy-specific modifications to evaluation result."""
        # Modify dimension scores based on strategy
        modified_dimensions = []
        
        for dimension in result.dimensions:
            modified_score = dimension.score
            
            # Apply scoring bias
            if configuration.scoring_bias != 0:
                bias_adjustment = configuration.scoring_bias * 0.1  # 10% max adjustment
                modified_score = max(0.0, min(1.0, modified_score + bias_adjustment))
            
            # Apply cultural emphasis
            if configuration.cultural_emphasis > 0.5:
                cultural_boost = (configuration.cultural_emphasis - 0.5) * dimension.cultural_relevance * 0.2
                modified_score = max(0.0, min(1.0, modified_score + cultural_boost))
            
            # Apply weights if specified
            if configuration.weights and dimension.name in configuration.weights:
                weight = configuration.weights[dimension.name]
                # Adjust confidence based on weight
                adjusted_confidence = dimension.confidence * weight
            else:
                adjusted_confidence = dimension.confidence
            
            # Apply confidence threshold
            if adjusted_confidence < configuration.confidence_threshold:
                adjusted_confidence *= 0.5  # Reduce confidence for low-confidence results
            
            modified_dimensions.append(EvaluationDimension(
                name=dimension.name,
                score=modified_score,
                confidence=adjusted_confidence,
                cultural_relevance=dimension.cultural_relevance,
                evidence=dimension.evidence,
                cultural_markers=dimension.cultural_markers
            ))
        
        # Recalculate overall score
        if modified_dimensions:
            total_score = 0.0
            total_weight = 0.0
            for dim in modified_dimensions:
                weight = dim.confidence * dim.cultural_relevance
                total_score += dim.score * weight
                total_weight += weight
            
            overall_score = total_score / total_weight if total_weight > 0 else 0.0
        else:
            overall_score = result.overall_score
        
        # Create modified result
        return DomainEvaluationResult(
            domain=result.domain,
            evaluation_type=result.evaluation_type,
            overall_score=overall_score,
            dimensions=modified_dimensions,
            cultural_context=result.cultural_context,
            metadata=result.metadata,
            processing_notes=result.processing_notes + [f"Modified by strategy: {configuration.strategy.value}"]
        )
    
    def _analyze_disagreement(self, ensemble_results: List[EnsembleEvaluationResult]) -> DisagreementAnalysis:
        """Analyze disagreement patterns across ensemble results."""
        overall_scores = [result.evaluation_result.overall_score for result in ensemble_results]
        
        # Overall score statistics
        mean_score = statistics.mean(overall_scores)
        score_variance = statistics.variance(overall_scores) if len(overall_scores) > 1 else 0.0
        score_range = (min(overall_scores), max(overall_scores))
        std_dev = statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0.0
        coeff_var = std_dev / mean_score if mean_score > 0 else 0.0
        
        # Dimension-level disagreement
        dimension_disagreements = self._calculate_dimension_disagreements(ensemble_results)
        
        # Identify outlier strategies
        strategy_outliers = self._identify_strategy_outliers(ensemble_results, mean_score, std_dev)
        
        # Calculate consensus level
        consensus_level = max(0.0, 1.0 - coeff_var * 2)  # Higher coefficient of variation = lower consensus
        
        # High disagreement dimensions
        high_disagreement_dimensions = [
            dim for dim, disagreement in dimension_disagreements.items() 
            if disagreement > self.disagreement_threshold
        ]
        
        return DisagreementAnalysis(
            mean_score=mean_score,
            score_variance=score_variance,
            score_range=score_range,
            standard_deviation=std_dev,
            coefficient_of_variation=coeff_var,
            dimension_disagreements=dimension_disagreements,
            strategy_outliers=strategy_outliers,
            consensus_level=consensus_level,
            high_disagreement_dimensions=high_disagreement_dimensions
        )
    
    def _calculate_dimension_disagreements(self, ensemble_results: List[EnsembleEvaluationResult]) -> Dict[str, float]:
        """Calculate disagreement for each dimension across strategies."""
        dimension_scores = {}
        
        # Collect scores by dimension
        for result in ensemble_results:
            for dimension in result.evaluation_result.dimensions:
                if dimension.name not in dimension_scores:
                    dimension_scores[dimension.name] = []
                dimension_scores[dimension.name].append(dimension.score)
        
        # Calculate disagreement (coefficient of variation) for each dimension
        disagreements = {}
        for dimension, scores in dimension_scores.items():
            if len(scores) > 1:
                mean_score = statistics.mean(scores)
                std_dev = statistics.stdev(scores)
                disagreements[dimension] = std_dev / mean_score if mean_score > 0 else 0.0
            else:
                disagreements[dimension] = 0.0
        
        return disagreements
    
    def _identify_strategy_outliers(self, 
                                  ensemble_results: List[EnsembleEvaluationResult],
                                  mean_score: float,
                                  std_dev: float) -> List[EvaluationStrategy]:
        """Identify strategies that produce outlier results."""
        outliers = []
        outlier_threshold = 1.5  # 1.5 standard deviations for better sensitivity
        
        for result in ensemble_results:
            score_diff = abs(result.evaluation_result.overall_score - mean_score)
            if std_dev > 0 and score_diff / std_dev > outlier_threshold:
                outliers.append(result.strategy)
        
        return outliers
    
    def _create_consensus_result(self, 
                               ensemble_results: List[EnsembleEvaluationResult],
                               disagreement_analysis: DisagreementAnalysis) -> DomainEvaluationResult:
        """Create consensus result from ensemble evaluations."""
        if not ensemble_results:
            return None
        
        # Use the first result as template
        template_result = ensemble_results[0].evaluation_result
        
        # Create consensus dimensions
        consensus_dimensions = self._create_consensus_dimensions(ensemble_results)
        
        # Use consensus mean score
        consensus_score = disagreement_analysis.mean_score
        
        # Aggregate metadata
        consensus_metadata = template_result.metadata.copy()
        consensus_metadata.update({
            'ensemble_evaluation': True,
            'strategies_used': [result.strategy.value for result in ensemble_results],
            'consensus_level': disagreement_analysis.consensus_level,
            'disagreement_analysis': {
                'score_variance': disagreement_analysis.score_variance,
                'coefficient_of_variation': disagreement_analysis.coefficient_of_variation,
                'high_disagreement_dimensions': disagreement_analysis.high_disagreement_dimensions
            }
        })
        
        # Aggregate processing notes
        consensus_notes = template_result.processing_notes + [
            f"Consensus from {len(ensemble_results)} evaluation strategies",
            f"Consensus level: {disagreement_analysis.consensus_level:.2f}",
            f"Score variance: {disagreement_analysis.score_variance:.3f}"
        ]
        
        return DomainEvaluationResult(
            domain=template_result.domain,
            evaluation_type=template_result.evaluation_type,
            overall_score=consensus_score,
            dimensions=consensus_dimensions,
            cultural_context=template_result.cultural_context,
            metadata=consensus_metadata,
            processing_notes=consensus_notes
        )
    
    def _create_consensus_dimensions(self, ensemble_results: List[EnsembleEvaluationResult]) -> List[EvaluationDimension]:
        """Create consensus dimensions from ensemble results."""
        dimension_data = {}
        
        # Collect all dimension data
        for result in ensemble_results:
            for dimension in result.evaluation_result.dimensions:
                if dimension.name not in dimension_data:
                    dimension_data[dimension.name] = {
                        'scores': [],
                        'confidences': [],
                        'cultural_relevances': [],
                        'evidence_sets': [],
                        'cultural_markers_sets': []
                    }
                
                dimension_data[dimension.name]['scores'].append(dimension.score)
                dimension_data[dimension.name]['confidences'].append(dimension.confidence)
                dimension_data[dimension.name]['cultural_relevances'].append(dimension.cultural_relevance)
                dimension_data[dimension.name]['evidence_sets'].append(dimension.evidence)
                dimension_data[dimension.name]['cultural_markers_sets'].append(dimension.cultural_markers)
        
        # Create consensus dimensions
        consensus_dimensions = []
        for dimension_name, data in dimension_data.items():
            # Calculate consensus values
            consensus_score = statistics.mean(data['scores'])
            consensus_confidence = statistics.mean(data['confidences'])
            consensus_cultural_relevance = statistics.mean(data['cultural_relevances'])
            
            # Aggregate evidence and markers
            all_evidence = []
            all_markers = []
            for evidence_set in data['evidence_sets']:
                all_evidence.extend(evidence_set)
            for marker_set in data['cultural_markers_sets']:
                all_markers.extend(marker_set)
            
            # Remove duplicates
            unique_evidence = list(set(all_evidence))
            unique_markers = list(set(all_markers))
            
            consensus_dimensions.append(EvaluationDimension(
                name=dimension_name,
                score=consensus_score,
                confidence=consensus_confidence,
                cultural_relevance=consensus_cultural_relevance,
                evidence=unique_evidence,
                cultural_markers=unique_markers
            ))
        
        return consensus_dimensions
    
    def _generate_disagreement_flags(self, 
                                   disagreement_analysis: DisagreementAnalysis,
                                   cultural_context: CulturalContext) -> List[ValidationFlag]:
        """Generate validation flags based on disagreement analysis."""
        flags = []
        
        # High overall disagreement
        if disagreement_analysis.consensus_level < self.consensus_threshold:
            severity = 'high' if disagreement_analysis.consensus_level < 0.3 else 'medium'
            flags.append(ValidationFlag(
                flag_type='high_ensemble_disagreement',
                severity=severity,
                description=f"High disagreement between evaluation strategies: consensus level {disagreement_analysis.consensus_level:.2f}",
                affected_dimensions=disagreement_analysis.high_disagreement_dimensions,
                cultural_groups=cultural_context.cultural_groups,
                recommendation="Consider manual review due to evaluation strategy disagreement"
            ))
        
        # Dimension-specific disagreement
        for dimension in disagreement_analysis.high_disagreement_dimensions:
            disagreement_level = disagreement_analysis.dimension_disagreements.get(dimension, 0)
            flags.append(ValidationFlag(
                flag_type='dimension_disagreement',
                severity='medium',
                description=f"High disagreement on dimension '{dimension}': {disagreement_level:.2f}",
                affected_dimensions=[dimension],
                cultural_groups=cultural_context.cultural_groups,
                recommendation=f"Review evaluation criteria for dimension: {dimension}"
            ))
        
        # Strategy outliers
        if disagreement_analysis.strategy_outliers:
            flags.append(ValidationFlag(
                flag_type='strategy_outliers',
                severity='medium',
                description=f"Outlier strategies detected: {[s.value for s in disagreement_analysis.strategy_outliers]}",
                affected_dimensions=[],
                cultural_groups=cultural_context.cultural_groups,
                recommendation="Investigate why certain strategies produce outlier results"
            ))
        
        return flags
    
    def _recommend_strategies(self, 
                            ensemble_results: List[EnsembleEvaluationResult],
                            disagreement_analysis: DisagreementAnalysis) -> List[EvaluationStrategy]:
        """Recommend evaluation strategies based on results."""
        recommendations = []
        
        # Prefer strategies that are not outliers and have reasonable performance
        for result in ensemble_results:
            if (result.strategy not in disagreement_analysis.strategy_outliers and
                0.3 <= result.evaluation_result.overall_score <= 0.9):  # Reasonable score range
                recommendations.append(result.strategy)
        
        # If no good recommendations, use standard strategy
        if not recommendations:
            recommendations.append(EvaluationStrategy.STANDARD)
        
        # Always include cultural focused for cultural evaluations
        if EvaluationStrategy.CULTURAL_FOCUSED not in recommendations:
            recommendations.append(EvaluationStrategy.CULTURAL_FOCUSED)
        
        return recommendations[:3]  # Return top 3 recommendations
    
    def _calculate_cultural_consensus(self, ensemble_results: List[EnsembleEvaluationResult]) -> float:
        """Calculate consensus on cultural authenticity across strategies."""
        cultural_scores = []
        
        for result in ensemble_results:
            cultural_competence = result.evaluation_result.calculate_cultural_competence()
            cultural_scores.append(cultural_competence)
        
        if not cultural_scores:
            return 0.0
        
        # Calculate consensus as inverse of coefficient of variation
        mean_cultural = statistics.mean(cultural_scores)
        if mean_cultural == 0:
            return 0.0
        
        std_cultural = statistics.stdev(cultural_scores) if len(cultural_scores) > 1 else 0
        coeff_var = std_cultural / mean_cultural
        
        return max(0.0, 1.0 - coeff_var)
    
    def _calculate_reliability_score(self, disagreement_analysis: DisagreementAnalysis) -> float:
        """Calculate overall evaluation reliability score."""
        # Base reliability on consensus level and score variance
        variance_penalty = min(1.0, disagreement_analysis.score_variance * 5)  # Penalize high variance
        reliability = disagreement_analysis.consensus_level * (1.0 - variance_penalty)
        
        # Bonus for having multiple strategies agree
        if disagreement_analysis.consensus_level > 0.8:
            reliability += 0.1
        
        return max(0.0, min(1.0, reliability))
    
    def _create_single_strategy_result(self, single_result: EnsembleEvaluationResult) -> EnsembleDisagreementResult:
        """Create result when only one strategy was successful."""
        if not single_result:
            # Create empty result
            return EnsembleDisagreementResult(
                ensemble_results=[],
                disagreement_analysis=DisagreementAnalysis(
                    mean_score=0.0,
                    score_variance=0.0,
                    score_range=(0.0, 0.0),
                    standard_deviation=0.0,
                    coefficient_of_variation=0.0,
                    dimension_disagreements={},
                    strategy_outliers=[],
                    consensus_level=0.0,
                    high_disagreement_dimensions=[]
                ),
                consensus_result=None,
                validation_flags=[ValidationFlag(
                    flag_type='ensemble_failure',
                    severity='high',
                    description="Ensemble evaluation failed - no successful strategies",
                    affected_dimensions=[],
                    cultural_groups=[],
                    recommendation="Check evaluation configuration and strategy implementation"
                )],
                recommended_strategies=[EvaluationStrategy.STANDARD],
                cultural_authenticity_consensus=0.0,
                evaluation_reliability_score=0.0
            )
        
        return EnsembleDisagreementResult(
            ensemble_results=[single_result],
            disagreement_analysis=DisagreementAnalysis(
                mean_score=single_result.evaluation_result.overall_score,
                score_variance=0.0,
                score_range=(single_result.evaluation_result.overall_score, single_result.evaluation_result.overall_score),
                standard_deviation=0.0,
                coefficient_of_variation=0.0,
                dimension_disagreements={},
                strategy_outliers=[],
                consensus_level=1.0,  # Perfect consensus with one result
                high_disagreement_dimensions=[]
            ),
            consensus_result=single_result.evaluation_result,
            validation_flags=[ValidationFlag(
                flag_type='single_strategy_only',
                severity='medium',
                description="Only one evaluation strategy succeeded",
                affected_dimensions=[],
                cultural_groups=single_result.evaluation_result.cultural_context.cultural_groups,
                recommendation="Consider investigating why other strategies failed"
            )],
            recommended_strategies=[single_result.strategy],
            cultural_authenticity_consensus=single_result.evaluation_result.calculate_cultural_competence(),
            evaluation_reliability_score=0.7  # Moderate reliability with single strategy
        )
    
    # Alias method for test compatibility
    def analyze_disagreement(self, *args, **kwargs):
        """Alias for detect_evaluation_disagreement method for test compatibility."""
        return self.detect_evaluation_disagreement(*args, **kwargs)