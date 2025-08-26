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
    """Container for universal evaluation metric scores"""
    organization_quality: float        # step_clarity -> organization_quality
    technical_accuracy: float          # logical_consistency -> technical_accuracy  
    completeness: float                # evidence_integration -> completeness
    thoroughness: float                # analysis_depth -> thoroughness
    reliability: float                 # verification_effort -> reliability
    scope_coverage: float              # comprehensive_coverage -> scope_coverage
    domain_appropriateness: float      # reasoning_pattern -> domain_appropriateness
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


class UniversalEvaluator:
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
        logger.info("UniversalEvaluator initialized successfully")
    
    def evaluate_response(self, 
                         response_text: str, 
                         test_name: str, 
                         reasoning_type: Optional[Union[str, ReasoningType]] = None,
                         test_category: Optional[str] = None,
                         use_llm_evaluation: bool = False) -> EvaluationResult:
        """
        Main evaluation method that analyzes a response using universal metrics
        
        Args:
            response_text: The model's response to evaluate
            test_name: Name of the test case
            reasoning_type: Type of reasoning expected (auto-detected if None)
            test_category: Test category for type-specific evaluation logic
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
        
        # Detect test type for category-specific evaluation
        test_type = self._detect_test_type(test_category)
        
        # Perform core evaluation with universal metrics
        metrics = self._evaluate_universal_metrics(response_text, reasoning_type, test_type, test_category)
        
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
    
    def _detect_test_type(self, test_category: Optional[str]) -> str:
        """Detect test type based on category for universal evaluation"""
        if not test_category:
            return "reasoning"
            
        category_lower = test_category.lower()
        
        # Linux system administration tests
        if any(linux_keyword in category_lower for linux_keyword in 
               ["linux", "log_analysis", "containerization", "security", "monitoring", 
                "backup", "service_management", "networking", "process_management", 
                "system_management", "troubleshooting", "database", "deployment"]):
            return "linux"
        
        # Creative and strategic thinking tests
        elif any(creative_keyword in category_lower for creative_keyword in
                 ["creative", "strategic", "ambiguity", "metacognitive", "constraint"]):
            return "creative"
        
        # Default to reasoning for other categories
        else:
            return "reasoning"
    
    def _evaluate_universal_metrics(self, response_text: str, reasoning_type: ReasoningType, 
                                   test_type: str, test_category: Optional[str] = None) -> EvaluationMetrics:
        """Evaluate universal metrics with category-specific logic"""
        text = response_text.lower()
        word_count = len(response_text.split())
        
        # Universal metric calculations with test-type specific logic
        organization_quality = self._calculate_organization_quality(text, test_type)
        technical_accuracy = self._calculate_technical_accuracy(text, test_type)
        completeness = self._calculate_completeness(text, test_type)
        thoroughness = self._calculate_thoroughness(text, test_type)
        reliability = self._calculate_reliability(text, test_type)
        scope_coverage = self._calculate_scope_coverage(response_text, word_count, test_type)
        domain_appropriateness = self._calculate_domain_appropriateness(text, reasoning_type, test_type)
        
        # Calculate weighted overall score using updated weights
        weights = self.config.get("weights", {
            "organization_quality": 0.15,
            "technical_accuracy": 0.20,
            "completeness": 0.15,
            "thoroughness": 0.15,
            "reliability": 0.10,
            "scope_coverage": 0.10,
            "domain_appropriateness": 0.15
        })
        
        overall_score = (
            organization_quality * weights.get("organization_quality", 0.15) +
            technical_accuracy * weights.get("technical_accuracy", 0.20) +
            completeness * weights.get("completeness", 0.15) +
            thoroughness * weights.get("thoroughness", 0.15) +
            reliability * weights.get("reliability", 0.10) +
            scope_coverage * weights.get("scope_coverage", 0.10) +
            domain_appropriateness * weights.get("domain_appropriateness", 0.15)
        )
        
        # Calculate confidence score based on response length and complexity
        confidence_score = self._calculate_confidence_score(word_count, overall_score)
        
        return EvaluationMetrics(
            organization_quality=round(organization_quality, 1),
            technical_accuracy=round(technical_accuracy, 1),
            completeness=round(completeness, 1),
            thoroughness=round(thoroughness, 1),
            reliability=round(reliability, 1),
            scope_coverage=round(scope_coverage, 1),
            domain_appropriateness=round(domain_appropriateness, 1),
            overall_score=round(overall_score, 1),
            word_count=word_count,
            confidence_score=round(confidence_score, 1)
        )
    
    def _calculate_organization_quality(self, text: str, test_type: str) -> float:
        """Calculate organization quality based on test type"""
        text_lower = text.lower()  # Convert to lowercase for case-insensitive matching
        
        if test_type == "linux":
            # For Linux tests: look for clear command structure, proper syntax
            linux_indicators = ["#!/bin/bash", "if", "then", "else", "for", "while", "&&", "||", 
                              "sudo", "systemctl", "grep", "awk", "sed"]
            indicator_count = sum(1 for indicator in linux_indicators if indicator in text_lower)
            return min(indicator_count * 8, 100)
        elif test_type == "creative":
            # For creative tests: look for structure, paragraphs, organization
            creative_indicators = ["\n\n", "first", "second", "finally", "however", "therefore", 
                                 "in conclusion", "on the other hand"]
            indicator_count = sum(1 for indicator in creative_indicators if indicator in text_lower)
            return min(indicator_count * 12, 100)
        else:
            # For reasoning tests: traditional step clarity
            step_indicators = ["step", "first", "next", "then", "finally", "therefore", "thus", "hence"]
            step_count = sum(1 for indicator in step_indicators if indicator in text_lower)
            return min(step_count * 10, 100)
    
    def _calculate_technical_accuracy(self, text: str, test_type: str) -> float:
        """Calculate technical accuracy based on test type"""
        if test_type == "linux":
            # For Linux tests: look for proper command syntax, best practices
            accurate_patterns = ["sudo", "systemctl", "chmod", "chown", "grep -", "awk", "sed", 
                               "ps aux", "netstat", "ss -", "iptables", "firewall", "crontab"]
            accuracy_count = sum(1 for pattern in accurate_patterns if pattern in text)
            # Check for dangerous patterns (reduce score)
            dangerous_patterns = ["rm -rf /", "chmod 777", "* * * * *", "> /dev/null 2>&1"]
            danger_count = sum(1 for danger in dangerous_patterns if danger in text)
            return min(max(accuracy_count * 12 - danger_count * 20, 0), 100)
        elif test_type == "creative":
            # For creative tests: look for coherent ideas and consistency
            coherence_indicators = ["because", "since", "therefore", "however", "although", 
                                  "furthermore", "moreover", "consequently"]
            coherence_count = sum(1 for indicator in coherence_indicators if indicator in text)
            return min(coherence_count * 10, 100)
        else:
            # For reasoning tests: traditional logical consistency
            logic_connectors = ["because", "therefore", "thus", "hence", "since", "given that", 
                              "it follows that", "consequently"]
            logic_count = sum(1 for connector in logic_connectors if connector in text)
            return min(logic_count * 8, 100)
    
    def _calculate_completeness(self, text: str, test_type: str) -> float:
        """Calculate completeness based on test type"""
        if test_type == "linux":
            # For Linux tests: look for complete command solutions
            completeness_indicators = ["#!/bin/bash", "error handling", "logging", "exit", "return", 
                                     "status", "check", "validate", "test"]
            completeness_count = sum(1 for indicator in completeness_indicators if indicator in text)
            return min(completeness_count * 11, 100)
        elif test_type == "creative":
            # For creative tests: look for addressing all constraints
            creative_completeness = ["requirement", "constraint", "criteria", "all", "every", 
                                   "complete", "comprehensive", "thorough"]
            completeness_count = sum(1 for indicator in creative_completeness if indicator in text)
            return min(completeness_count * 13, 100)
        else:
            # For reasoning tests: traditional evidence integration
            evidence_indicators = ["evidence", "data", "according to", "based on", "research shows", 
                                 "studies indicate", "analysis reveals"]
            evidence_count = sum(1 for indicator in evidence_indicators if indicator in text)
            return min(evidence_count * 12, 100)
    
    def _calculate_thoroughness(self, text: str, test_type: str) -> float:
        """Calculate thoroughness based on test type"""
        if test_type == "linux":
            # For Linux tests: comprehensive solutions and explanations
            thorough_indicators = ["explanation", "comment", "documentation", "verbose", "detailed", 
                                 "comprehensive", "step-by-step", "example"]
            thorough_count = sum(1 for indicator in thorough_indicators if indicator in text)
            return min(thorough_count * 12, 100)
        elif test_type == "creative":
            # For creative tests: depth of creative exploration
            creative_depth = ["explore", "consider", "alternative", "perspective", "angle", 
                            "approach", "innovative", "unique", "original"]
            depth_count = sum(1 for indicator in creative_depth if indicator in text)
            return min(depth_count * 11, 100)
        else:
            # For reasoning tests: traditional analysis depth
            depth_indicators = ['analyze', 'synthesize', 'evaluate', 'interpret', 'assess', 'examine', 'investigate']
            depth_count = sum(1 for indicator in depth_indicators if indicator in text)
            return min(depth_count * 10, 100)
    
    def _calculate_reliability(self, text: str, test_type: str) -> float:
        """Calculate reliability based on test type"""
        if test_type == "linux":
            # For Linux tests: best practices, security, error handling
            reliability_indicators = ["backup", "error", "check", "validate", "secure", "permission", 
                                    "log", "monitor", "test", "verify"]
            reliability_count = sum(1 for indicator in reliability_indicators if indicator in text)
            return min(reliability_count * 10, 100)
        elif test_type == "creative":
            # For creative tests: consistency and constraint adherence
            creative_reliability = ["consistent", "coherent", "logical", "reasonable", "appropriate", 
                                  "suitable", "relevant", "applicable"]
            reliability_count = sum(1 for indicator in creative_reliability if indicator in text)
            return min(reliability_count * 12, 100)
        else:
            # For reasoning tests: traditional verification effort
            verification_indicators = ["verify", "check", "confirm", "validate", "double-check", 
                                     "review", "examine", "test"]
            verification_count = sum(1 for indicator in verification_indicators if indicator in text)
            return min(verification_count * 15, 100)
    
    def _calculate_scope_coverage(self, text: str, word_count: int, test_type: str) -> float:
        """Calculate scope coverage based on test type"""
        if test_type == "linux":
            # For Linux tests: coverage of requirements and edge cases
            base_score = min(word_count / 30, 70)  # Shorter responses can be complete for Linux
            scope_indicators = ["requirement", "specification", "edge case", "exception", 
                              "alternative", "option", "parameter", "configuration"]
            scope_bonus = sum(5 for indicator in scope_indicators if indicator in text.lower())
            return min(base_score + scope_bonus, 100)
        elif test_type == "creative":
            # For creative tests: breadth of creative exploration
            base_score = min(word_count / 40, 80)
            creative_scope = ["aspect", "dimension", "perspective", "viewpoint", "angle", 
                            "consideration", "factor", "element"]
            scope_bonus = sum(3 for indicator in creative_scope if indicator in text.lower())
            return min(base_score + scope_bonus, 100)
        else:
            # For reasoning tests: traditional comprehensive coverage
            return min(word_count / 50, 100)
    
    def _calculate_domain_appropriateness(self, text: str, reasoning_type: ReasoningType, test_type: str) -> float:
        """Calculate domain appropriateness based on test type and reasoning type"""
        if test_type == "linux":
            # For Linux tests: appropriate commands, practices, and terminology
            linux_terms = ["command", "script", "bash", "shell", "system", "service", "daemon", 
                         "process", "file", "directory", "permission", "user", "group", "network"]
            linux_score = sum(5 for term in linux_terms if term in text)
            return min(linux_score, 100)
        elif test_type == "creative":
            # For creative tests: appropriate creative approaches and language
            creative_terms = ["creative", "innovative", "original", "unique", "artistic", "imaginative", 
                            "inventive", "novel", "unconventional", "alternative"]
            creative_score = sum(8 for term in creative_terms if term in text)
            return min(creative_score, 100)
        else:
            # For reasoning tests: traditional reasoning pattern scoring
            if hasattr(self, 'reasoning_type_patterns') and reasoning_type in self.reasoning_type_patterns:
                patterns = self.reasoning_type_patterns[reasoning_type]
                keyword_score = sum(10 for keyword in patterns["keywords"] if keyword in text)
                pattern_score = sum(20 for pattern in patterns["patterns"] if re.search(pattern, text))
                return min(keyword_score + pattern_score, 100)
            else:
                # Basic reasoning appropriateness scoring
                reasoning_terms = ["analysis", "conclusion", "logic", "reasoning", "inference", 
                                 "deduction", "argument", "evidence", "premise"]
                reasoning_score = sum(8 for term in reasoning_terms if term in text)
                return min(reasoning_score, 100)
    
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
        
        if metrics.organization_quality < threshold:
            recommendations.append("Improve step-by-step clarity by using explicit step indicators (first, second, then, etc.)")
        
        if metrics.technical_accuracy < threshold:
            recommendations.append("Strengthen logical flow with more connecting words (therefore, because, consequently)")
        
        if metrics.completeness < threshold:
            recommendations.append("Better integrate evidence with phrases like 'based on', 'according to', 'data shows'")
        
        if metrics.reliability < threshold:
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
            organization_quality=0.0, technical_accuracy=0.0, completeness=0.0,
            thoroughness=0.0, reliability=0.0, scope_coverage=0.0,
            domain_appropriateness=0.0, overall_score=0.0, word_count=len(response_text.split()),
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
            metrics_sum["organization_quality"] += result.metrics.organization_quality
            metrics_sum["technical_accuracy"] += result.metrics.technical_accuracy
            metrics_sum["completeness"] += result.metrics.completeness
            metrics_sum["thoroughness"] += result.metrics.thoroughness
            metrics_sum["reliability"] += result.metrics.reliability
            metrics_sum["scope_coverage"] += result.metrics.scope_coverage
            metrics_sum["domain_appropriateness"] += result.metrics.domain_appropriateness
        
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
def evaluate_reasoning(response_text: str, test_name: str, test_category: Optional[str] = None, **kwargs) -> EvaluationResult:
    """
    Convenience function for universal evaluation
    
    Args:
        response_text: Text to evaluate
        test_name: Name of the test
        test_category: Test category for type-specific evaluation
        **kwargs: Additional arguments for UniversalEvaluator
    
    Returns:
        EvaluationResult
    """
    evaluator = UniversalEvaluator()
    return evaluator.evaluate_response(response_text, test_name, test_category=test_category, **kwargs)