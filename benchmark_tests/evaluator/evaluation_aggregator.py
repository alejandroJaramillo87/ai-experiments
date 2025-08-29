from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import statistics
from collections import defaultdict, Counter

from .domain_evaluator_base import DomainEvaluationResult, EvaluationDimension, CulturalContext


@dataclass
class AggregatedEvaluationResult:
    """Aggregated results from multiple domain evaluators."""
    overall_score: float  # 0.0 to 1.0
    domain_scores: Dict[str, float]  # Domain -> score
    dimension_scores: Dict[str, float]  # Dimension -> aggregated score
    cultural_competence: float  # Overall cultural competence
    cultural_markers: List[str]  # All detected cultural markers
    consensus_level: float  # Agreement between evaluators (0.0 to 1.0)
    evaluation_coverage: float  # Percentage of expected evaluations completed
    metadata: Dict[str, Any]
    processing_notes: List[str]
    domain_results: List[DomainEvaluationResult]  # Individual results


@dataclass
class EvaluationConsensus:
    """Analysis of consensus between domain evaluators."""
    dimension: str
    scores: List[float]
    mean_score: float
    std_deviation: float
    consensus_level: float  # 1 - (std_dev / mean) if mean > 0
    outlier_domains: List[str]  # Domains with scores far from mean


class EvaluationAggregator:
    """Aggregates results from multiple domain-specific evaluators."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.consensus_threshold = self.config.get('consensus_threshold', 0.7)
        self.outlier_threshold = self.config.get('outlier_threshold', 2.0)  # std deviations
        
    def aggregate_results(self, 
                         domain_results: List[DomainEvaluationResult],
                         expected_domains: List[str] = None) -> AggregatedEvaluationResult:
        """
        Aggregate results from multiple domain evaluators.
        
        Args:
            domain_results: List of results from domain evaluators
            expected_domains: List of domains that should have been evaluated
            
        Returns:
            AggregatedEvaluationResult with combined analysis
        """
        if not domain_results:
            return self._create_empty_result(expected_domains or [])
        
        # Calculate domain scores
        domain_scores = {result.domain: result.overall_score for result in domain_results}
        
        # Aggregate dimension scores
        dimension_scores = self._aggregate_dimension_scores(domain_results)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(domain_results)
        
        # Calculate cultural competence
        cultural_competence = self._calculate_cultural_competence(domain_results)
        
        # Collect cultural markers
        cultural_markers = self._collect_cultural_markers(domain_results)
        
        # Calculate consensus
        consensus_level = self._calculate_consensus(domain_results)
        
        # Calculate evaluation coverage
        evaluation_coverage = self._calculate_coverage(domain_results, expected_domains)
        
        # Generate metadata and notes
        metadata = self._generate_metadata(domain_results)
        processing_notes = self._generate_processing_notes(domain_results)
        
        return AggregatedEvaluationResult(
            overall_score=overall_score,
            domain_scores=domain_scores,
            dimension_scores=dimension_scores,
            cultural_competence=cultural_competence,
            cultural_markers=cultural_markers,
            consensus_level=consensus_level,
            evaluation_coverage=evaluation_coverage,
            metadata=metadata,
            processing_notes=processing_notes,
            domain_results=domain_results
        )
    
    def _aggregate_dimension_scores(self, 
                                   domain_results: List[DomainEvaluationResult]) -> Dict[str, float]:
        """Aggregate scores for each dimension across domains."""
        dimension_data = defaultdict(list)
        
        # Collect scores for each dimension
        for result in domain_results:
            for dim in result.dimensions:
                dimension_data[dim.name].append(dim.score)
        
        # Calculate aggregated scores
        aggregated_scores = {}
        for dimension, scores in dimension_data.items():
            if scores:
                # Weight by cultural relevance if available
                weighted_scores = []
                weights = []
                
                for result in domain_results:
                    for dim in result.dimensions:
                        if dim.name == dimension:
                            weighted_scores.append(dim.score)
                            weights.append(dim.cultural_relevance * dim.confidence)
                
                if weights and sum(weights) > 0:
                    aggregated_scores[dimension] = sum(s * w for s, w in zip(weighted_scores, weights)) / sum(weights)
                else:
                    aggregated_scores[dimension] = statistics.mean(scores)
        
        return aggregated_scores
    
    def _calculate_overall_score(self, domain_results: List[DomainEvaluationResult]) -> float:
        """Calculate overall score across all domains."""
        if not domain_results:
            return 0.0
        
        # Weight domain scores by their cultural competence
        total_score = 0.0
        total_weight = 0.0
        
        for result in domain_results:
            cultural_comp = result.calculate_cultural_competence()
            weight = max(0.1, cultural_comp)  # Minimum weight to avoid zero division
            total_score += result.overall_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_cultural_competence(self, domain_results: List[DomainEvaluationResult]) -> float:
        """Calculate overall cultural competence score."""
        if not domain_results:
            return 0.0
        
        cultural_scores = [result.calculate_cultural_competence() for result in domain_results]
        return statistics.mean(cultural_scores) if cultural_scores else 0.0
    
    def _collect_cultural_markers(self, domain_results: List[DomainEvaluationResult]) -> List[str]:
        """Collect all unique cultural markers from domain results."""
        all_markers = []
        for result in domain_results:
            all_markers.extend(result.get_cultural_markers())
        
        # Count occurrences and return sorted by frequency
        marker_counts = Counter(all_markers)
        return [marker for marker, count in marker_counts.most_common()]
    
    def _calculate_consensus(self, domain_results: List[DomainEvaluationResult]) -> float:
        """Calculate consensus level between domain evaluators."""
        if len(domain_results) < 2:
            return 1.0
        
        # Get consensus for each dimension
        dimension_consensuses = self._analyze_dimension_consensus(domain_results)
        
        if not dimension_consensuses:
            return 0.0
        
        consensus_scores = [cons.consensus_level for cons in dimension_consensuses.values()]
        return statistics.mean(consensus_scores) if consensus_scores else 0.0
    
    def _analyze_dimension_consensus(self, 
                                   domain_results: List[DomainEvaluationResult]) -> Dict[str, EvaluationConsensus]:
        """Analyze consensus for each dimension across domains."""
        dimension_data = defaultdict(lambda: {'scores': [], 'domains': []})
        
        # Collect dimension scores by domain
        for result in domain_results:
            for dim in result.dimensions:
                dimension_data[dim.name]['scores'].append(dim.score)
                dimension_data[dim.name]['domains'].append(result.domain)
        
        consensus_analysis = {}
        for dimension, data in dimension_data.items():
            scores = data['scores']
            domains = data['domains']
            
            if len(scores) < 2:
                consensus_analysis[dimension] = EvaluationConsensus(
                    dimension=dimension,
                    scores=scores,
                    mean_score=scores[0] if scores else 0.0,
                    std_deviation=0.0,
                    consensus_level=1.0,
                    outlier_domains=[]
                )
                continue
            
            mean_score = statistics.mean(scores)
            std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
            
            # Calculate consensus level (higher is better)
            if mean_score > 0:
                consensus_level = max(0.0, 1.0 - (std_dev / mean_score))
            else:
                consensus_level = 1.0 if std_dev == 0 else 0.0
            
            # Identify outliers
            outlier_domains = []
            if std_dev > 0:
                for score, domain in zip(scores, domains):
                    z_score = abs(score - mean_score) / std_dev
                    if z_score > self.outlier_threshold:
                        outlier_domains.append(domain)
            
            consensus_analysis[dimension] = EvaluationConsensus(
                dimension=dimension,
                scores=scores,
                mean_score=mean_score,
                std_deviation=std_dev,
                consensus_level=consensus_level,
                outlier_domains=outlier_domains
            )
        
        return consensus_analysis
    
    def _calculate_coverage(self, 
                           domain_results: List[DomainEvaluationResult],
                           expected_domains: List[str] = None) -> float:
        """Calculate evaluation coverage percentage."""
        if not expected_domains:
            return 1.0 if domain_results else 0.0
        
        evaluated_domains = {result.domain for result in domain_results}
        expected_set = set(expected_domains)
        
        return len(evaluated_domains.intersection(expected_set)) / len(expected_set)
    
    def _generate_metadata(self, domain_results: List[DomainEvaluationResult]) -> Dict[str, Any]:
        """Generate aggregation metadata."""
        return {
            'total_domains_evaluated': len(domain_results),
            'domains': [result.domain for result in domain_results],
            'evaluation_types': list(set(result.evaluation_type for result in domain_results)),
            'total_dimensions': sum(len(result.dimensions) for result in domain_results),
            'aggregation_method': 'weighted_cultural_competence',
            'consensus_threshold': self.consensus_threshold,
            'outlier_threshold': self.outlier_threshold
        }
    
    def _generate_processing_notes(self, domain_results: List[DomainEvaluationResult]) -> List[str]:
        """Generate processing notes for aggregated results."""
        notes = []
        
        # Domain coverage
        notes.append(f"Aggregated results from {len(domain_results)} domain evaluators")
        
        # Success/failure summary
        successful_results = [r for r in domain_results if r.overall_score > 0]
        failed_results = [r for r in domain_results if r.overall_score == 0]
        
        if successful_results:
            notes.append(f"Successfully evaluated {len(successful_results)} domains: {[r.domain for r in successful_results]}")
        
        if failed_results:
            notes.append(f"Failed evaluation in {len(failed_results)} domains: {[r.domain for r in failed_results]}")
        
        # Consensus analysis
        consensus_analysis = self._analyze_dimension_consensus(domain_results)
        low_consensus_dims = [dim for dim, cons in consensus_analysis.items() 
                             if cons.consensus_level < self.consensus_threshold]
        
        if low_consensus_dims:
            notes.append(f"Low consensus on dimensions: {low_consensus_dims}")
        
        # Cultural marker summary
        all_markers = self._collect_cultural_markers(domain_results)
        if all_markers:
            notes.append(f"Detected {len(set(all_markers))} unique cultural markers")
        
        return notes
    
    def _create_empty_result(self, expected_domains: List[str]) -> AggregatedEvaluationResult:
        """Create empty aggregated result when no domain results available."""
        return AggregatedEvaluationResult(
            overall_score=0.0,
            domain_scores={},
            dimension_scores={},
            cultural_competence=0.0,
            cultural_markers=[],
            consensus_level=0.0,
            evaluation_coverage=0.0,
            metadata={'total_domains_evaluated': 0, 'expected_domains': expected_domains},
            processing_notes=["No domain evaluation results to aggregate"],
            domain_results=[]
        )
    
    def get_consensus_report(self, aggregated_result: AggregatedEvaluationResult) -> Dict[str, Any]:
        """Generate detailed consensus analysis report."""
        consensus_analysis = self._analyze_dimension_consensus(aggregated_result.domain_results)
        
        report = {
            'overall_consensus': aggregated_result.consensus_level,
            'dimension_analysis': {},
            'outlier_summary': defaultdict(list),
            'recommendations': []
        }
        
        for dimension, consensus in consensus_analysis.items():
            report['dimension_analysis'][dimension] = {
                'mean_score': consensus.mean_score,
                'std_deviation': consensus.std_deviation,
                'consensus_level': consensus.consensus_level,
                'outlier_domains': consensus.outlier_domains
            }
            
            for domain in consensus.outlier_domains:
                report['outlier_summary'][domain].append(dimension)
        
        # Generate recommendations
        if aggregated_result.consensus_level < self.consensus_threshold:
            report['recommendations'].append("Consider reviewing evaluation criteria due to low consensus")
        
        if report['outlier_summary']:
            report['recommendations'].append(f"Review outlier domains: {list(report['outlier_summary'].keys())}")
        
        return report