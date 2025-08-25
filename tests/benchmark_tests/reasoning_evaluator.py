"""
Reasoning Evaluator Module

A comprehensive system for evaluating reasoning quality in language model responses.
Provides both Python-based structural metrics and integration framework for LLM evaluation.

Author: Claude Code
Version: 1.0.0
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Enumeration of reasoning types for specialized evaluation"""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    MULTI_STEP = "multi_step"
    VERIFICATION = "verification"
    MATHEMATICAL = "mathematical"
    MULTI_HOP = "multi_hop"
    SCAFFOLDED = "scaffolded"
    BACKWARD = "backward"
    GENERAL = "general"


@dataclass
class EvaluationMetrics:
    """Container for evaluation metric scores"""
    step_clarity: float
    logical_consistency: float
    evidence_integration: float
    analysis_depth: float
    verification_effort: float
    comprehensive_coverage: float
    reasoning_pattern: float
    overall_score: float
    word_count: int
    confidence_score: float


@dataclass
class EvaluationResult:
    """Complete evaluation result with detailed breakdown"""
    metrics: EvaluationMetrics
    reasoning_type: ReasoningType
    detailed_analysis: Dict[str, any]
    recommendations: List[str]
    timestamp: str


class ReasoningEvaluator:
    """
    Main class for evaluating reasoning quality in language model responses.
    
    Provides comprehensive analysis through multiple evaluation approaches:
    - Python-based structural metrics
    - Reasoning-type-specific patterns
    - Domain-specific analysis
    - Optional LLM integration for semantic evaluation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ReasoningEvaluator
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = self._load_config(config_path)
        self._initialize_patterns()
        logger.info("ReasoningEvaluator initialized successfully")
    
    def evaluate_response(self, 
                         response_text: str, 
                         test_name: str, 
                         reasoning_type: Optional[Union[str, ReasoningType]] = None,
                         use_llm_evaluation: bool = False) -> EvaluationResult:
        """
        Main evaluation method that analyzes a response for reasoning quality
        
        Args:
            response_text: The model's response to evaluate
            test_name: Name of the test case
            reasoning_type: Type of reasoning expected (auto-detected if None)
            use_llm_evaluation: Whether to include LLM-based evaluation
            
        Returns:
            EvaluationResult: Comprehensive evaluation results
        """
        if not response_text or len(response_text.strip()) < 50:
            return self._create_minimal_result(response_text, "Response too short for analysis")
        
        # Auto-detect reasoning type if not provided
        if reasoning_type is None:
            reasoning_type = self._detect_reasoning_type(test_name)
        elif isinstance(reasoning_type, str):
            reasoning_type = ReasoningType(reasoning_type.lower())
        
        # Perform core evaluation
        metrics = self._evaluate_core_metrics(response_text, reasoning_type)
        
        # Add reasoning-type-specific analysis
        specialized_analysis = self._evaluate_specialized_patterns(response_text, reasoning_type)
        
        # Combine results
        detailed_analysis = {
            "core_metrics": metrics.__dict__,
            "specialized_analysis": specialized_analysis,
            "text_statistics": self._calculate_text_statistics(response_text),
            "reasoning_indicators": self._extract_reasoning_indicators(response_text, reasoning_type)
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, reasoning_type)
        
        # Optional LLM evaluation
        if use_llm_evaluation:
            llm_results = self._perform_llm_evaluation(response_text, reasoning_type)
            detailed_analysis["llm_evaluation"] = llm_results
        
        return EvaluationResult(
            metrics=metrics,
            reasoning_type=reasoning_type,
            detailed_analysis=detailed_analysis,
            recommendations=recommendations,
            timestamp=self._get_timestamp()
        )
    
    def evaluate_batch(self, 
                      responses: List[Tuple[str, str]], 
                      reasoning_types: Optional[List[Union[str, ReasoningType]]] = None,
                      use_llm_evaluation: bool = False) -> List[EvaluationResult]:
        """
        Evaluate multiple responses in batch
        
        Args:
            responses: List of (response_text, test_name) tuples
            reasoning_types: Optional list of reasoning types
            use_llm_evaluation: Whether to use LLM evaluation
            
        Returns:
            List of EvaluationResult objects
        """
        results = []
        reasoning_types = reasoning_types or [None] * len(responses)
        
        for i, (response_text, test_name) in enumerate(responses):
            reasoning_type = reasoning_types[i] if i < len(reasoning_types) else None
            result = self.evaluate_response(response_text, test_name, reasoning_type, use_llm_evaluation)
            results.append(result)
            
        return results
    
    def generate_summary_report(self, results: List[EvaluationResult]) -> Dict[str, any]:
        """
        Generate a summary report from multiple evaluation results
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary containing summary statistics and insights
        """
        if not results:
            return {"error": "No results to summarize"}
        
        # Calculate aggregate statistics
        overall_scores = [r.metrics.overall_score for r in results]
        reasoning_types = [r.reasoning_type.value for r in results]
        
        summary = {
            "total_evaluations": len(results),
            "average_score": np.mean(overall_scores),
            "median_score": np.median(overall_scores),
            "score_std": np.std(overall_scores),
            "score_range": [np.min(overall_scores), np.max(overall_scores)],
            "reasoning_type_distribution": self._calculate_distribution(reasoning_types),
            "metric_averages": self._calculate_metric_averages(results),
            "top_performing_tests": self._identify_top_performers(results, n=5),
            "areas_for_improvement": self._identify_improvement_areas(results)
        }
        
        return summary
    
    # Private methods for core functionality
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                logger.warning(f"Config file not found: {config_path}. Using defaults.")
        
        # Import default config
        try:
            from evaluation_config import DEFAULT_CONFIG
            return DEFAULT_CONFIG
        except ImportError:
            logger.warning("evaluation_config.py not found. Using minimal defaults.")
            return self._get_minimal_config()
    
    def _get_minimal_config(self) -> Dict:
        """Minimal configuration for basic operation"""
        return {
            "weights": {
                "step_clarity": 0.15,
                "logical_consistency": 0.20,
                "evidence_integration": 0.15,
                "analysis_depth": 0.15,
                "verification_effort": 0.10,
                "comprehensive_coverage": 0.10,
                "reasoning_pattern": 0.15
            },
            "thresholds": {
                "minimum_word_count": 50,
                "excellent_score": 80.0,
                "good_score": 65.0,
                "poor_score": 40.0
            }
        }
    
    def _initialize_patterns(self):
        """Initialize pattern recognition data structures"""
        # These will be populated with sophisticated patterns
        self.step_indicators = ['first', 'second', 'third', 'next', 'then', 'finally', 'step 1', 'step 2', 'step 3']
        self.logic_connectors = ['because', 'therefore', 'consequently', 'thus', 'hence', 'as a result', 'due to', 'since', 'given that']
        self.evidence_indicators = ['according to', 'based on', 'evidence shows', 'data indicates', 'studies show', 'research suggests']
        self.verification_indicators = ['verify', 'validate', 'confirm', 'double-check', 'review', 'examine', 'challenge', 'question']
        
        # Initialize reasoning type detection patterns
        self.reasoning_type_patterns = self._create_reasoning_patterns()
    
    def _create_reasoning_patterns(self) -> Dict[ReasoningType, Dict]:
        """Create patterns for detecting different reasoning types"""
        return {
            ReasoningType.CHAIN_OF_THOUGHT: {
                "keywords": ["step", "first", "second", "then", "next", "finally", "therefore"],
                "patterns": [r"step \d+", r"first.+second.+third", r"then.+therefore"]
            },
            ReasoningType.MULTI_HOP: {
                "keywords": ["document", "source", "according to", "based on", "evidence from"],
                "patterns": [r"document [A-Z]", r"according to.+from.+", r"evidence.+suggests.+because"]
            },
            ReasoningType.VERIFICATION: {
                "keywords": ["verify", "check", "confirm", "validate", "review", "double-check"],
                "patterns": [r"let me.+check", r"verify.+assumption", r"review.+conclusion"]
            },
            ReasoningType.MATHEMATICAL: {
                "keywords": ["calculate", "equation", "formula", "probability", "statistics"],
                "patterns": [r"\d+%", r"probability.+\d", r"equation.+equals"]
            }
        }
    
    def _detect_reasoning_type(self, test_name: str) -> ReasoningType:
        """Auto-detect reasoning type from test name"""
        test_name_lower = test_name.lower()
        
        if "chain-of-thought" in test_name_lower or "chain" in test_name_lower:
            return ReasoningType.CHAIN_OF_THOUGHT
        elif "multi-hop" in test_name_lower or "multi-source" in test_name_lower:
            return ReasoningType.MULTI_HOP
        elif "verification" in test_name_lower or "self-check" in test_name_lower:
            return ReasoningType.VERIFICATION
        elif "mathematical" in test_name_lower or "probability" in test_name_lower:
            return ReasoningType.MATHEMATICAL
        elif "backward" in test_name_lower or "reverse" in test_name_lower:
            return ReasoningType.BACKWARD
        elif "scaffolded" in test_name_lower or "structured" in test_name_lower:
            return ReasoningType.SCAFFOLDED
        elif "multi-step" in test_name_lower or "decomposition" in test_name_lower:
            return ReasoningType.MULTI_STEP
        else:
            return ReasoningType.GENERAL
    
    def _evaluate_core_metrics(self, response_text: str, reasoning_type: ReasoningType) -> EvaluationMetrics:
        """Evaluate core reasoning metrics"""
        text = response_text.lower()
        word_count = len(response_text.split())
        
        # Core metric calculations
        step_clarity = self._calculate_step_clarity(text)
        logical_consistency = self._calculate_logical_consistency(text)
        evidence_integration = self._calculate_evidence_integration(text)
        analysis_depth = self._calculate_analysis_depth(text)
        verification_effort = self._calculate_verification_effort(text)
        comprehensive_coverage = self._calculate_comprehensive_coverage(response_text, word_count)
        reasoning_pattern = self._calculate_reasoning_pattern_score(text, reasoning_type)
        
        # Calculate weighted overall score
        weights = self.config["weights"]
        overall_score = (
            step_clarity * weights["step_clarity"] +
            logical_consistency * weights["logical_consistency"] +
            evidence_integration * weights["evidence_integration"] +
            analysis_depth * weights["analysis_depth"] +
            verification_effort * weights["verification_effort"] +
            comprehensive_coverage * weights["comprehensive_coverage"] +
            reasoning_pattern * weights["reasoning_pattern"]
        )
        
        # Calculate confidence score based on response length and complexity
        confidence_score = self._calculate_confidence_score(word_count, overall_score)
        
        return EvaluationMetrics(
            step_clarity=round(step_clarity, 1),
            logical_consistency=round(logical_consistency, 1),
            evidence_integration=round(evidence_integration, 1),
            analysis_depth=round(analysis_depth, 1),
            verification_effort=round(verification_effort, 1),
            comprehensive_coverage=round(comprehensive_coverage, 1),
            reasoning_pattern=round(reasoning_pattern, 1),
            overall_score=round(overall_score, 1),
            word_count=word_count,
            confidence_score=round(confidence_score, 1)
        )
    
    def _calculate_step_clarity(self, text: str) -> float:
        """Calculate step clarity score based on structured thinking indicators"""
        step_count = sum(1 for indicator in self.step_indicators if indicator in text)
        return min(step_count * 10, 100)
    
    def _calculate_logical_consistency(self, text: str) -> float:
        """Calculate logical consistency score based on logical connectors"""
        logic_count = sum(1 for connector in self.logic_connectors if connector in text)
        return min(logic_count * 8, 100)
    
    def _calculate_evidence_integration(self, text: str) -> float:
        """Calculate evidence integration score"""
        evidence_count = sum(1 for indicator in self.evidence_indicators if indicator in text)
        return min(evidence_count * 12, 100)
    
    def _calculate_analysis_depth(self, text: str) -> float:
        """Calculate analysis depth score"""
        depth_indicators = ['analyze', 'synthesize', 'evaluate', 'interpret', 'assess', 'examine', 'investigate']
        depth_count = sum(1 for indicator in depth_indicators if indicator in text)
        return min(depth_count * 10, 100)
    
    def _calculate_verification_effort(self, text: str) -> float:
        """Calculate verification effort score"""
        verification_count = sum(1 for indicator in self.verification_indicators if indicator in text)
        return min(verification_count * 15, 100)
    
    def _calculate_comprehensive_coverage(self, text: str, word_count: int) -> float:
        """Calculate comprehensive coverage based on length and thoroughness"""
        return min(word_count / 50, 100)
    
    def _calculate_reasoning_pattern_score(self, text: str, reasoning_type: ReasoningType) -> float:
        """Calculate reasoning pattern score specific to the reasoning type"""
        if reasoning_type not in self.reasoning_type_patterns:
            return 50.0  # Default score for unknown types
        
        patterns = self.reasoning_type_patterns[reasoning_type]
        keyword_score = sum(10 for keyword in patterns["keywords"] if keyword in text)
        pattern_score = sum(20 for pattern in patterns["patterns"] if re.search(pattern, text))
        
        return min(keyword_score + pattern_score, 100)
    
    def _calculate_confidence_score(self, word_count: int, overall_score: float) -> float:
        """Calculate confidence in the evaluation based on text characteristics"""
        # More text generally means more reliable evaluation
        length_factor = min(word_count / 500, 1.0)
        # Higher scores with sufficient text get higher confidence
        score_factor = overall_score / 100
        return (length_factor * 0.6 + score_factor * 0.4) * 100
    
    def _evaluate_specialized_patterns(self, response_text: str, reasoning_type: ReasoningType) -> Dict:
        """Evaluate specialized patterns based on reasoning type"""
        # Placeholder for specialized analysis - to be expanded
        return {
            "reasoning_type": reasoning_type.value,
            "specialized_score": 75.0,  # Placeholder
            "pattern_matches": []
        }
    
    def _calculate_text_statistics(self, text: str) -> Dict:
        """Calculate basic text statistics"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_sentence_length": len(words) / max(len(sentences), 1),
            "unique_words": len(set(word.lower().strip('.,!?;:') for word in words)),
            "vocabulary_diversity": len(set(word.lower().strip('.,!?;:') for word in words)) / max(len(words), 1)
        }
    
    def _extract_reasoning_indicators(self, text: str, reasoning_type: ReasoningType) -> Dict:
        """Extract specific reasoning indicators from text"""
        return {
            "step_indicators_found": [ind for ind in self.step_indicators if ind in text.lower()],
            "logic_connectors_found": [conn for conn in self.logic_connectors if conn in text.lower()],
            "evidence_indicators_found": [ind for ind in self.evidence_indicators if ind in text.lower()],
            "verification_indicators_found": [ind for ind in self.verification_indicators if ind in text.lower()]
        }
    
    def _generate_recommendations(self, metrics: EvaluationMetrics, reasoning_type: ReasoningType) -> List[str]:
        """Generate specific recommendations for improvement"""
        recommendations = []
        threshold = 60.0  # Threshold for recommendations
        
        if metrics.step_clarity < threshold:
            recommendations.append("Improve step-by-step clarity by using explicit step indicators (first, second, then, etc.)")
        
        if metrics.logical_consistency < threshold:
            recommendations.append("Strengthen logical flow with more connecting words (therefore, because, consequently)")
        
        if metrics.evidence_integration < threshold:
            recommendations.append("Better integrate evidence with phrases like 'based on', 'according to', 'data shows'")
        
        if metrics.verification_effort < threshold:
            recommendations.append("Add verification steps with self-checking language (verify, confirm, review)")
        
        if not recommendations:
            recommendations.append("Strong reasoning demonstrated across all metrics")
        
        return recommendations
    
    def _perform_llm_evaluation(self, response_text: str, reasoning_type: ReasoningType) -> Dict:
        """Placeholder for LLM evaluation integration"""
        # This will be implemented when LLM integration is added
        return {
            "llm_score": 0.0,
            "semantic_analysis": "Not implemented yet",
            "llm_recommendations": []
        }
    
    def _create_minimal_result(self, response_text: str, error_message: str) -> EvaluationResult:
        """Create minimal result for error cases"""
        minimal_metrics = EvaluationMetrics(
            step_clarity=0.0, logical_consistency=0.0, evidence_integration=0.0,
            analysis_depth=0.0, verification_effort=0.0, comprehensive_coverage=0.0,
            reasoning_pattern=0.0, overall_score=0.0, word_count=len(response_text.split()),
            confidence_score=0.0
        )
        
        return EvaluationResult(
            metrics=minimal_metrics,
            reasoning_type=ReasoningType.GENERAL,
            detailed_analysis={"error": error_message},
            recommendations=[error_message],
            timestamp=self._get_timestamp()
        )
    
    def _calculate_distribution(self, items: List[str]) -> Dict[str, int]:
        """Calculate distribution of items"""
        distribution = defaultdict(int)
        for item in items:
            distribution[item] += 1
        return dict(distribution)
    
    def _calculate_metric_averages(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate average scores for each metric"""
        if not results:
            return {}
        
        metrics_sum = defaultdict(float)
        for result in results:
            metrics_sum["step_clarity"] += result.metrics.step_clarity
            metrics_sum["logical_consistency"] += result.metrics.logical_consistency
            metrics_sum["evidence_integration"] += result.metrics.evidence_integration
            metrics_sum["analysis_depth"] += result.metrics.analysis_depth
            metrics_sum["verification_effort"] += result.metrics.verification_effort
            metrics_sum["comprehensive_coverage"] += result.metrics.comprehensive_coverage
            metrics_sum["reasoning_pattern"] += result.metrics.reasoning_pattern
        
        n = len(results)
        return {metric: round(score / n, 1) for metric, score in metrics_sum.items()}
    
    def _identify_top_performers(self, results: List[EvaluationResult], n: int = 5) -> List[Dict]:
        """Identify top performing evaluations"""
        sorted_results = sorted(results, key=lambda r: r.metrics.overall_score, reverse=True)
        return [
            {
                "reasoning_type": r.reasoning_type.value,
                "overall_score": r.metrics.overall_score,
                "timestamp": r.timestamp
            }
            for r in sorted_results[:n]
        ]
    
    def _identify_improvement_areas(self, results: List[EvaluationResult]) -> List[str]:
        """Identify common areas for improvement"""
        metric_averages = self._calculate_metric_averages(results)
        weak_areas = []
        
        threshold = 65.0
        for metric, avg_score in metric_averages.items():
            if avg_score < threshold:
                weak_areas.append(f"{metric.replace('_', ' ').title()}: {avg_score}")
        
        return weak_areas or ["All metrics performing well"]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


# Convenience function for quick evaluation
def evaluate_reasoning(response_text: str, test_name: str, **kwargs) -> EvaluationResult:
    """
    Convenience function for quick reasoning evaluation
    
    Args:
        response_text: Text to evaluate
        test_name: Name of the test
        **kwargs: Additional arguments for ReasoningEvaluator
    
    Returns:
        EvaluationResult
    """
    evaluator = ReasoningEvaluator()
    return evaluator.evaluate_response(response_text, test_name, **kwargs)