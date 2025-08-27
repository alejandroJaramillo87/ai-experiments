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
        
        # Add professional formatting bonus
        formatting_bonus = self._calculate_formatting_bonus(response_text)
        
        # Calculate weighted overall score using updated weights (adjusted for formatting bonus)
        weights = self.config.get("weights", {
            "organization_quality": 0.14,    # Slightly reduced to make room for formatting
            "technical_accuracy": 0.19,     # Still highest weight
            "completeness": 0.14,
            "thoroughness": 0.14,
            "reliability": 0.10,
            "scope_coverage": 0.09,
            "domain_appropriateness": 0.15
        })  # Total: 0.95 + 0.05 formatting = 1.00
        
        overall_score = (
            organization_quality * weights.get("organization_quality", 0.14) +
            technical_accuracy * weights.get("technical_accuracy", 0.19) +
            completeness * weights.get("completeness", 0.14) +
            thoroughness * weights.get("thoroughness", 0.14) +
            reliability * weights.get("reliability", 0.10) +
            scope_coverage * weights.get("scope_coverage", 0.09) +
            domain_appropriateness * weights.get("domain_appropriateness", 0.15) +
            formatting_bonus * 0.05  # Additional 5% weight for professional formatting
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
        """Calculate organization quality based on sophisticated structural patterns"""
        text_lower = text.lower()
        
        # Base score for any reasonable length content
        base_score = min(len(text.split()) / 20, 40)  # Up to 40 points for substantial content
        
        if test_type == "linux":
            # Linux: command structure, scripts, proper syntax
            linux_indicators = ["#!/bin/bash", "if", "then", "else", "for", "while", "&&", "||", 
                              "sudo", "systemctl", "grep", "awk", "sed"]
            indicator_score = sum(8 for indicator in linux_indicators if indicator in text_lower)
            return min(base_score + indicator_score, 100)
            
        elif test_type == "creative":
            # Creative: sophisticated structure patterns
            structure_patterns = [
                ("\n\n", 5),  # Paragraph breaks
                ("first", 8), ("second", 6), ("third", 6), ("finally", 8),
                ("however", 6), ("therefore", 6), ("in conclusion", 10),
                ("on the other hand", 8), ("furthermore", 6), ("moreover", 6)
            ]
            structure_score = sum(points for pattern, points in structure_patterns if pattern in text_lower)
            return min(base_score + structure_score, 100)
            
        else:
            # Reasoning: sophisticated academic/professional structure
            # Professional structure indicators (much higher value)
            professional_structure = [
                ("introduction", 15), ("conclusion", 15), ("analysis", 12),
                ("summary", 10), ("overview", 10), ("methodology", 12),
                ("framework", 15), ("approach", 8), ("findings", 10)
            ]
            
            # Academic organization patterns
            academic_patterns = [
                ("###", 10), ("##", 8), ("**", 5),  # Headers/formatting
                ("| ", 15),  # Tables (strong indicator of organization)
                ("1.", 8), ("2.", 6), ("3.", 6),  # Numbered lists
                ("- ", 5), ("• ", 5)  # Bullet points
            ]
            
            # Traditional step indicators (lower weight since sophisticated responses may not use these)
            basic_steps = [
                ("step", 6), ("first", 6), ("next", 4), ("then", 4), 
                ("finally", 8), ("therefore", 6), ("thus", 6), ("hence", 6)
            ]
            
            professional_score = sum(points for pattern, points in professional_structure if pattern in text_lower)
            academic_score = sum(points for pattern, points in academic_patterns if pattern in text)  # Case-sensitive for formatting
            basic_score = sum(points for pattern, points in basic_steps if pattern in text_lower)
            
            total_structure_score = professional_score + academic_score + basic_score
            return min(base_score + total_structure_score, 100)
    
    def _calculate_technical_accuracy(self, text: str, test_type: str) -> float:
        """Calculate technical accuracy with sophisticated domain recognition"""
        text_lower = text.lower()
        
        # Base score for substantial, coherent content
        base_score = min(len(text.split()) / 30, 35)  # Up to 35 points for length/coherence
        
        if test_type == "linux":
            # Linux: command syntax and best practices
            accurate_patterns = ["sudo", "systemctl", "chmod", "chown", "grep -", "awk", "sed", 
                               "ps aux", "netstat", "ss -", "iptables", "firewall", "crontab"]
            accuracy_score = sum(12 for pattern in accurate_patterns if pattern in text)
            
            # Penalize dangerous patterns
            dangerous_patterns = ["rm -rf /", "chmod 777", "* * * * *", "> /dev/null 2>&1"]
            danger_penalty = sum(20 for danger in dangerous_patterns if danger in text)
            
            return min(max(base_score + accuracy_score - danger_penalty, 0), 100)
            
        elif test_type == "creative":
            # Creative: logical flow and sophisticated reasoning
            coherence_indicators = ["because", "since", "therefore", "however", "although", 
                                  "furthermore", "moreover", "consequently", "nevertheless", "meanwhile"]
            coherence_score = sum(8 for indicator in coherence_indicators if indicator in text_lower)
            return min(base_score + coherence_score, 100)
            
        else:
            # Reasoning: sophisticated domain expertise and logical precision
            
            # High-value logical connectors (sophisticated reasoning)
            sophisticated_logic = [
                ("therefore", 10), ("consequently", 12), ("hence", 10), ("thus", 8),
                ("it follows that", 15), ("given that", 10), ("assuming", 8),
                ("conversely", 10), ("nevertheless", 10), ("furthermore", 8)
            ]
            
            # Domain expertise indicators
            expertise_indicators = [
                ("analysis", 8), ("framework", 12), ("methodology", 15), ("systematic", 10),
                ("comprehensive", 10), ("empirical", 12), ("theoretical", 12), ("paradigm", 15),
                ("hypothesis", 12), ("premise", 10), ("conclusion", 8), ("inference", 10),
                ("deduction", 12), ("induction", 10), ("synthesis", 12), ("evaluation", 8)
            ]
            
            # Professional terminology (varies by domain)
            professional_terms = [
                ("equilibrium", 15), ("optimization", 12), ("correlation", 10), ("statistical", 10),
                ("probability", 12), ("strategy", 8), ("implementation", 8), ("assessment", 8),
                ("protocol", 10), ("specification", 10), ("validation", 12), ("verification", 12)
            ]
            
            # Mathematical/quantitative precision
            quantitative_patterns = [
                (r"\d+%", 8), (r"\d+\.\d+", 5), ("percentage", 6), ("ratio", 8),
                ("coefficient", 12), ("variable", 8), ("parameter", 10), ("metric", 8)
            ]
            
            sophisticated_score = sum(points for pattern, points in sophisticated_logic if pattern in text_lower)
            expertise_score = sum(points for pattern, points in expertise_indicators if pattern in text_lower)
            professional_score = sum(points for pattern, points in professional_terms if pattern in text_lower)
            
            # Check for quantitative patterns using regex
            import re
            quantitative_score = 0
            for pattern, points in quantitative_patterns:
                if pattern.startswith('r"') or pattern.startswith("r'"):
                    # Regex pattern
                    pattern_str = pattern[2:-1]  # Remove r" and "
                    if re.search(pattern_str, text):
                        quantitative_score += points
                else:
                    # Simple string pattern
                    if pattern in text_lower:
                        quantitative_score += points
            
            total_technical_score = sophisticated_score + expertise_score + professional_score + quantitative_score
            return min(base_score + total_technical_score, 100)
    
    def _calculate_completeness(self, text: str, test_type: str) -> float:
        """Calculate completeness through comprehensive coverage analysis"""
        text_lower = text.lower()
        word_count = len(text.split())
        
        # Base score for comprehensive content length
        base_score = min(word_count / 50, 45)  # Up to 45 points for substantial content
        
        if test_type == "linux":
            # Linux: complete solutions with error handling, validation
            completeness_indicators = ["#!/bin/bash", "error handling", "logging", "exit", "return", 
                                     "status", "check", "validate", "test", "backup", "monitoring"]
            completeness_score = sum(10 for indicator in completeness_indicators if indicator in text_lower)
            return min(base_score + completeness_score, 100)
            
        elif test_type == "creative":
            # Creative: addressing multiple aspects and constraints
            comprehensive_coverage = [
                ("requirement", 12), ("constraint", 12), ("criteria", 10), ("aspect", 8),
                ("dimension", 10), ("perspective", 10), ("approach", 8), ("consideration", 8),
                ("alternative", 10), ("option", 8), ("comprehensive", 15), ("thorough", 12),
                ("complete", 10), ("detailed", 8), ("extensive", 10)
            ]
            coverage_score = sum(points for pattern, points in comprehensive_coverage if pattern in text_lower)
            return min(base_score + coverage_score, 100)
            
        else:
            # Reasoning: comprehensive evidence integration and multi-faceted analysis
            
            # Evidence integration patterns
            evidence_patterns = [
                ("evidence", 10), ("data", 8), ("according to", 12), ("based on", 10),
                ("research shows", 15), ("studies indicate", 15), ("analysis reveals", 12),
                ("findings suggest", 12), ("results demonstrate", 15), ("investigation shows", 12)
            ]
            
            # Comprehensive analysis indicators
            comprehensive_analysis = [
                ("multiple", 8), ("various", 8), ("several", 6), ("different", 6),
                ("range", 8), ("spectrum", 10), ("comprehensive", 15), ("extensive", 10),
                ("thorough", 12), ("complete", 8), ("detailed", 8), ("in-depth", 12)
            ]
            
            # Multi-perspective coverage
            perspective_indicators = [
                ("perspective", 10), ("viewpoint", 10), ("angle", 8), ("standpoint", 10),
                ("approach", 8), ("lens", 10), ("framework", 12), ("context", 8),
                ("dimension", 10), ("aspect", 8), ("facet", 10), ("component", 8)
            ]
            
            # Integration and synthesis patterns
            synthesis_patterns = [
                ("synthesis", 15), ("integration", 12), ("combination", 10), ("merge", 8),
                ("consolidation", 12), ("unification", 12), ("convergence", 10), ("connection", 8),
                ("relationship", 8), ("correlation", 10), ("interdependence", 12)
            ]
            
            evidence_score = sum(points for pattern, points in evidence_patterns if pattern in text_lower)
            comprehensive_score = sum(points for pattern, points in comprehensive_analysis if pattern in text_lower)
            perspective_score = sum(points for pattern, points in perspective_indicators if pattern in text_lower)
            synthesis_score = sum(points for pattern, points in synthesis_patterns if pattern in text_lower)
            
            total_completeness = evidence_score + comprehensive_score + perspective_score + synthesis_score
            return min(base_score + total_completeness, 100)
    
    def _calculate_thoroughness(self, text: str, test_type: str) -> float:
        """Calculate thoroughness through depth and detail analysis"""
        text_lower = text.lower()
        word_count = len(text.split())
        
        # Base score rewards substantial detailed content
        base_score = min(word_count / 40, 40)  # Up to 40 points for detailed content
        
        if test_type == "linux":
            # Linux: comprehensive solutions with detailed explanations
            thorough_indicators = [
                ("explanation", 12), ("comment", 8), ("documentation", 15), ("verbose", 10),
                ("detailed", 12), ("comprehensive", 15), ("step-by-step", 12), ("example", 10),
                ("troubleshooting", 12), ("debugging", 12), ("configuration", 10)
            ]
            thorough_score = sum(points for pattern, points in thorough_indicators if pattern in text_lower)
            return min(base_score + thorough_score, 100)
            
        elif test_type == "creative":
            # Creative: depth of exploration and innovative thinking
            creative_depth_patterns = [
                ("explore", 10), ("consider", 8), ("alternative", 10), ("perspective", 10),
                ("angle", 8), ("approach", 8), ("innovative", 15), ("unique", 12), ("original", 12),
                ("creative", 10), ("imagination", 12), ("inventive", 12), ("novel", 10),
                ("unconventional", 15), ("breakthrough", 15), ("pioneering", 12)
            ]
            depth_score = sum(points for pattern, points in creative_depth_patterns if pattern in text_lower)
            return min(base_score + depth_score, 100)
            
        else:
            # Reasoning: sophisticated analytical depth and intellectual rigor
            
            # Deep analytical processes
            analytical_depth = [
                ("analyze", 12), ("synthesize", 15), ("evaluate", 12), ("interpret", 10),
                ("assess", 10), ("examine", 10), ("investigate", 12), ("scrutinize", 15),
                ("dissect", 12), ("deconstruct", 15), ("unpack", 10), ("elaborate", 8)
            ]
            
            # Intellectual rigor indicators
            rigor_indicators = [
                ("rigorous", 15), ("systematic", 12), ("methodical", 12), ("meticulous", 15),
                ("precise", 10), ("accurate", 8), ("careful", 8), ("thorough", 12),
                ("comprehensive", 12), ("exhaustive", 15), ("detailed", 8), ("extensive", 10)
            ]
            
            # Complex reasoning patterns
            complex_reasoning = [
                ("complex", 10), ("sophisticated", 15), ("nuanced", 15), ("multifaceted", 15),
                ("intricate", 12), ("elaborate", 10), ("comprehensive", 12), ("profound", 15),
                ("deep", 8), ("extensive", 10), ("intensive", 12), ("substantial", 10)
            ]
            
            # Evidence of deep engagement
            engagement_patterns = [
                ("implications", 12), ("consequences", 12), ("ramifications", 15), ("significance", 10),
                ("importance", 8), ("relevance", 8), ("application", 10), ("implementation", 10),
                ("practical", 8), ("theoretical", 10), ("empirical", 12), ("conceptual", 10)
            ]
            
            analytical_score = sum(points for pattern, points in analytical_depth if pattern in text_lower)
            rigor_score = sum(points for pattern, points in rigor_indicators if pattern in text_lower)
            complexity_score = sum(points for pattern, points in complex_reasoning if pattern in text_lower)
            engagement_score = sum(points for pattern, points in engagement_patterns if pattern in text_lower)
            
            total_thoroughness = analytical_score + rigor_score + complexity_score + engagement_score
            return min(base_score + total_thoroughness, 100)
    
    def _calculate_reliability(self, text: str, test_type: str) -> float:
        """Calculate reliability through verification and consistency patterns"""
        text_lower = text.lower()
        
        # Base score for consistent, well-structured content
        base_score = min(len(text.split()) / 60, 30)  # Up to 30 points for substantial content
        
        if test_type == "linux":
            # Linux: best practices, security, error handling, testing
            reliability_indicators = [
                ("backup", 12), ("error", 8), ("check", 10), ("validate", 12), ("secure", 10),
                ("permission", 10), ("log", 8), ("monitor", 10), ("test", 10), ("verify", 12),
                ("robust", 15), ("stable", 12), ("reliable", 15), ("safe", 8), ("secure", 10)
            ]
            reliability_score = sum(points for pattern, points in reliability_indicators if pattern in text_lower)
            return min(base_score + reliability_score, 100)
            
        elif test_type == "creative":
            # Creative: consistency, coherence, and constraint adherence
            creative_reliability = [
                ("consistent", 12), ("coherent", 12), ("logical", 10), ("reasonable", 10),
                ("appropriate", 10), ("suitable", 10), ("relevant", 10), ("applicable", 10),
                ("feasible", 12), ("practical", 10), ("viable", 12), ("realistic", 10),
                ("balanced", 10), ("proportionate", 12), ("well-reasoned", 15)
            ]
            reliability_score = sum(points for pattern, points in creative_reliability if pattern in text_lower)
            return min(base_score + reliability_score, 100)
            
        else:
            # Reasoning: sophisticated verification and quality assurance
            
            # Verification and validation patterns
            verification_patterns = [
                ("verify", 12), ("check", 8), ("confirm", 10), ("validate", 12), ("double-check", 15),
                ("review", 8), ("examine", 8), ("test", 8), ("audit", 12), ("inspect", 10),
                ("scrutinize", 15), ("cross-check", 15), ("re-examine", 12)
            ]
            
            # Quality assurance indicators
            quality_assurance = [
                ("accurate", 12), ("precise", 12), ("correct", 10), ("reliable", 15), ("trustworthy", 15),
                ("credible", 12), ("valid", 10), ("sound", 10), ("robust", 12), ("rigorous", 15),
                ("consistent", 12), ("coherent", 10), ("logical", 10), ("systematic", 12)
            ]
            
            # Self-correction and refinement patterns
            self_correction = [
                ("revise", 12), ("refine", 12), ("improve", 8), ("enhance", 8), ("optimize", 10),
                ("adjust", 8), ("modify", 8), ("update", 8), ("correct", 12), ("amend", 10),
                ("clarify", 10), ("specify", 8), ("elaborate", 8)
            ]
            
            # Confidence and certainty indicators
            confidence_patterns = [
                ("confident", 10), ("certain", 10), ("sure", 6), ("definite", 10), ("clear", 8),
                ("obvious", 8), ("evident", 10), ("apparent", 8), ("established", 12), ("proven", 12),
                ("demonstrated", 12), ("confirmed", 10), ("verified", 12)
            ]
            
            verification_score = sum(points for pattern, points in verification_patterns if pattern in text_lower)
            quality_score = sum(points for pattern, points in quality_assurance if pattern in text_lower)
            correction_score = sum(points for pattern, points in self_correction if pattern in text_lower)
            confidence_score = sum(points for pattern, points in confidence_patterns if pattern in text_lower)
            
            total_reliability = verification_score + quality_score + correction_score + confidence_score
            return min(base_score + total_reliability, 100)
    
    def _calculate_scope_coverage(self, text: str, word_count: int, test_type: str) -> float:
        """Calculate scope coverage through breadth and comprehensiveness analysis"""
        text_lower = text.lower()
        
        if test_type == "linux":
            # Linux: comprehensive solution coverage
            base_score = min(word_count / 25, 60)  # Base score for substantial solutions
            scope_indicators = [
                ("requirement", 8), ("specification", 10), ("edge case", 12), ("exception", 10),
                ("alternative", 8), ("option", 8), ("parameter", 8), ("configuration", 10),
                ("scenario", 8), ("use case", 10), ("implementation", 8), ("deployment", 10)
            ]
            scope_score = sum(points for pattern, points in scope_indicators if pattern in text_lower)
            return min(base_score + scope_score, 100)
            
        elif test_type == "creative":
            # Creative: breadth of exploration and consideration
            base_score = min(word_count / 35, 70)  # Reward substantial creative content
            creative_scope = [
                ("aspect", 8), ("dimension", 10), ("perspective", 10), ("viewpoint", 10),
                ("angle", 8), ("consideration", 10), ("factor", 8), ("element", 8),
                ("possibility", 10), ("scenario", 8), ("variation", 10), ("option", 8),
                ("opportunity", 8), ("potential", 8), ("implication", 10)
            ]
            scope_score = sum(points for pattern, points in creative_scope if pattern in text_lower)
            return min(base_score + scope_score, 100)
            
        else:
            # Reasoning: comprehensive coverage of topic breadth
            base_score = min(word_count / 40, 60)  # Reward substantial analytical content
            
            # Breadth indicators
            breadth_patterns = [
                ("comprehensive", 12), ("extensive", 10), ("broad", 8), ("wide", 8),
                ("range", 8), ("spectrum", 10), ("variety", 8), ("diverse", 8),
                ("multiple", 8), ("various", 8), ("different", 6), ("several", 6)
            ]
            
            # Coverage indicators
            coverage_patterns = [
                ("coverage", 10), ("includes", 6), ("encompasses", 10), ("addresses", 8),
                ("covers", 6), ("spans", 8), ("extends", 8), ("incorporates", 8),
                ("considers", 8), ("examines", 8), ("explores", 8), ("discusses", 6)
            ]
            
            # Multi-domain indicators
            multidomain_patterns = [
                ("interdisciplinary", 15), ("cross-disciplinary", 15), ("multi-faceted", 12),
                ("holistic", 12), ("integrated", 10), ("comprehensive", 12), ("multidimensional", 15)
            ]
            
            breadth_score = sum(points for pattern, points in breadth_patterns if pattern in text_lower)
            coverage_score = sum(points for pattern, points in coverage_patterns if pattern in text_lower)
            multidomain_score = sum(points for pattern, points in multidomain_patterns if pattern in text_lower)
            
            total_scope = breadth_score + coverage_score + multidomain_score
            return min(base_score + total_scope, 100)
    
    def _calculate_domain_appropriateness(self, text: str, reasoning_type: ReasoningType, test_type: str) -> float:
        """Calculate domain appropriateness through sophisticated terminology and expertise recognition"""
        text_lower = text.lower()
        
        # Base score for any substantial professional content
        base_score = min(len(text.split()) / 80, 25)  # Up to 25 points for substantial content
        
        if test_type == "linux":
            # Linux: technical expertise and professional practices
            linux_expertise = [
                ("command", 6), ("script", 8), ("bash", 8), ("shell", 6), ("system", 6),
                ("service", 8), ("daemon", 10), ("process", 6), ("file", 4), ("directory", 6),
                ("permission", 8), ("user", 4), ("group", 6), ("network", 6), ("server", 6),
                ("configuration", 8), ("administration", 10), ("management", 6), ("monitoring", 8)
            ]
            expertise_score = sum(points for pattern, points in linux_expertise if pattern in text_lower)
            return min(base_score + expertise_score, 100)
            
        elif test_type == "creative":
            # Creative: sophisticated creative approaches and innovation language
            creative_sophistication = [
                ("creative", 8), ("innovative", 10), ("original", 10), ("unique", 8), ("artistic", 10),
                ("imaginative", 10), ("inventive", 10), ("novel", 10), ("unconventional", 12),
                ("alternative", 8), ("breakthrough", 15), ("pioneering", 12), ("visionary", 15),
                ("groundbreaking", 15), ("revolutionary", 12), ("transformative", 12)
            ]
            sophistication_score = sum(points for pattern, points in creative_sophistication if pattern in text_lower)
            return min(base_score + sophistication_score, 100)
            
        else:
            # Reasoning: sophisticated domain expertise across multiple fields
            
            # Core reasoning sophistication
            reasoning_sophistication = [
                ("analysis", 8), ("conclusion", 8), ("logic", 8), ("reasoning", 10), ("inference", 10),
                ("deduction", 10), ("induction", 10), ("argument", 8), ("evidence", 8), ("premise", 8),
                ("synthesis", 12), ("evaluation", 10), ("interpretation", 10), ("assessment", 8)
            ]
            
            # Advanced academic terminology
            academic_terminology = [
                ("paradigm", 15), ("framework", 12), ("methodology", 15), ("theoretical", 12),
                ("empirical", 12), ("systematic", 10), ("conceptual", 10), ("analytical", 10),
                ("epistemological", 20), ("ontological", 20), ("phenomenological", 20)
            ]
            
            # Professional domain expertise
            professional_domains = [
                # Philosophy & Theory
                ("philosophical", 12), ("metaphysical", 15), ("existential", 12), ("dialectical", 15),
                # Economics & Business
                ("equilibrium", 15), ("optimization", 12), ("strategic", 10), ("tactical", 10),
                # Science & Research
                ("hypothesis", 12), ("empirical", 12), ("statistical", 12), ("quantitative", 12),
                # Legal & Policy
                ("constitutional", 15), ("jurisprudence", 20), ("precedent", 12), ("statutory", 12),
                # Psychology & Behavior
                ("cognitive", 12), ("behavioral", 10), ("psychological", 12), ("phenomenological", 15)
            ]
            
            # Reasoning type specific patterns (enhanced)
            reasoning_type_bonus = 0
            if hasattr(self, 'reasoning_type_patterns') and reasoning_type in self.reasoning_type_patterns:
                patterns = self.reasoning_type_patterns[reasoning_type]
                reasoning_type_bonus = sum(15 for keyword in patterns["keywords"] if keyword in text_lower)
                import re
                reasoning_type_bonus += sum(25 for pattern in patterns["patterns"] if re.search(pattern, text))
            
            reasoning_score = sum(points for pattern, points in reasoning_sophistication if pattern in text_lower)
            academic_score = sum(points for pattern, points in academic_terminology if pattern in text_lower)
            professional_score = sum(points for pattern, points in professional_domains if pattern in text_lower)
            
            total_domain_score = reasoning_score + academic_score + professional_score + reasoning_type_bonus
            return min(base_score + total_domain_score, 100)
    
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
    
    def _calculate_formatting_bonus(self, text: str) -> float:
        """Calculate bonus points for professional formatting and structure"""
        formatting_score = 0
        
        # Professional headers and structure
        if "###" in text or "##" in text or "**" in text:
            formatting_score += 15  # Markdown headers
        if "# " in text or "## " in text:
            formatting_score += 10  # Additional header formats
        
        # Tables and structured data
        if "| " in text and text.count("|") >= 6:
            formatting_score += 20  # Well-formatted tables
        if "---" in text or "===" in text:
            formatting_score += 10  # Section dividers
        
        # Lists and organization
        if "1." in text and "2." in text and "3." in text:
            formatting_score += 15  # Numbered lists
        if text.count("- ") >= 3 or text.count("• ") >= 3:
            formatting_score += 10  # Bullet points
        
        # Professional sectioning
        if "Part I" in text or "Section" in text or "Chapter" in text:
            formatting_score += 15  # Academic structure
        if "Introduction" in text and "Conclusion" in text:
            formatting_score += 15  # Proper academic format
        
        # Code blocks and technical formatting
        if "```" in text or "```python" in text or "```bash" in text:
            formatting_score += 10  # Code blocks
        if "`" in text and text.count("`") >= 4:
            formatting_score += 5  # Inline code
        
        # Advanced formatting patterns
        if "**Example:**" in text or "**Note:**" in text:
            formatting_score += 10  # Professional callouts
        if text.count("\n\n") >= 5:
            formatting_score += 10  # Good paragraph separation
        
        return min(formatting_score, 100)
    
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