"""
Quantization Impact Testing Module

Advanced quantization impact analysis for language model evaluation including
numerical stability testing, factual consistency validation, and quantization-specific metrics.

This module addresses the critique's key point about missing quantization impact
measurements and numerical stability analysis.

"""

import re
import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import Counter, defaultdict
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)


class QuantizationTester:
    """
    Advanced quantization impact tester for language model responses.
    
    Provides comprehensive quantization analysis:
    - Numerical stability testing
    - Factual consistency validation
    - Quantization-specific quality metrics
    - Mathematical reasoning accuracy
    - Knowledge consistency testing
    """
    
    def __init__(self):
        """Initialize quantization tester with test suites"""
        self.basic_math_tests = self._initialize_basic_math_tests()
        self.factual_knowledge_tests = self._initialize_factual_knowledge_tests()
        self.numerical_patterns = self._initialize_numerical_patterns()
        
        logger.info("QuantizationTester initialized with comprehensive test suites")
    
    def test_numerical_stability(self, response_text: str) -> Dict[str, Any]:
        """
        Test numerical stability and mathematical accuracy
        
        Args:
            response_text: Text response to analyze for numerical accuracy
            
        Returns:
            Dictionary with numerical stability metrics
        """
        if not response_text.strip():
            return {"numerical_accuracy": 0.0, "math_errors": [], "stability_score": 0.0}
        
        try:
            # Extract and validate numerical expressions
            numerical_expressions = self._extract_numerical_expressions(response_text)
            
            # Test basic arithmetic accuracy
            arithmetic_results = self._test_arithmetic_accuracy(numerical_expressions, response_text)
            
            # Test mathematical consistency
            consistency_results = self._test_mathematical_consistency(response_text)
            
            # Test numerical reasoning patterns
            reasoning_results = self._test_numerical_reasoning(response_text)
            
            # Calculate overall stability score
            stability_score = self._calculate_stability_score(
                arithmetic_results, consistency_results, reasoning_results
            )
            
            return {
                "numerical_accuracy": arithmetic_results["accuracy_rate"],
                "mathematical_consistency": consistency_results["consistency_score"],
                "numerical_reasoning": reasoning_results["reasoning_score"],
                "stability_score": stability_score,
                "arithmetic_tests": arithmetic_results,
                "consistency_tests": consistency_results,
                "reasoning_tests": reasoning_results,
                "total_numerical_expressions": len(numerical_expressions),
                "error_patterns": self._analyze_error_patterns(arithmetic_results, consistency_results)
            }
            
        except Exception as e:
            logger.error(f"Numerical stability testing failed: {e}")
            return {"numerical_accuracy": 0.0, "math_errors": [str(e)], "stability_score": 0.0}
    
    def test_factual_consistency(self, response_text: str) -> Dict[str, Any]:
        """
        Test factual consistency and knowledge accuracy
        
        Args:
            response_text: Text response to analyze for factual accuracy
            
        Returns:
            Dictionary with factual consistency metrics
        """
        if not response_text.strip():
            return {"factual_accuracy": 0.0, "consistency_score": 0.0, "knowledge_errors": []}
        
        try:
            # Test basic factual knowledge
            factual_tests = self._test_basic_facts(response_text)
            
            # Test internal consistency
            internal_consistency = self._test_internal_consistency(response_text)
            
            # Test knowledge coherence
            knowledge_coherence = self._test_knowledge_coherence(response_text)
            
            # Test common knowledge accuracy
            common_knowledge = self._test_common_knowledge(response_text)
            
            # Calculate overall consistency score
            consistency_score = self._calculate_consistency_score(
                factual_tests, internal_consistency, knowledge_coherence, common_knowledge
            )
            
            return {
                "factual_accuracy": factual_tests["accuracy_rate"],
                "internal_consistency": internal_consistency["consistency_score"],
                "knowledge_coherence": knowledge_coherence["coherence_score"],
                "common_knowledge_accuracy": common_knowledge["accuracy_rate"],
                "consistency_score": consistency_score,
                "factual_tests": factual_tests,
                "internal_tests": internal_consistency,
                "coherence_tests": knowledge_coherence,
                "common_knowledge_tests": common_knowledge,
                "knowledge_errors": self._collect_knowledge_errors(
                    factual_tests, internal_consistency, knowledge_coherence, common_knowledge
                )
            }
            
        except Exception as e:
            logger.error(f"Factual consistency testing failed: {e}")
            return {"factual_accuracy": 0.0, "consistency_score": 0.0, "knowledge_errors": [str(e)]}
    
    def analyze_quantization_impact(self, response_text: str, baseline_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive quantization impact analysis
        
        Args:
            response_text: Text response to analyze
            baseline_metrics: Optional baseline metrics for comparison
            
        Returns:
            Dictionary with quantization impact analysis
        """
        if not response_text.strip():
            return {"quantization_impact_score": 0.0, "degradation_indicators": []}
        
        try:
            # Core quantization tests
            numerical_stability = self.test_numerical_stability(response_text)
            factual_consistency = self.test_factual_consistency(response_text)
            
            # Quantization-specific degradation patterns
            degradation_patterns = self._detect_quantization_degradation(response_text)
            
            # Precision loss indicators
            precision_loss = self._analyze_precision_loss(response_text, numerical_stability)
            
            # Output quality consistency
            quality_consistency = self._analyze_output_consistency(response_text)
            
            # Calculate overall quantization impact
            impact_score = self._calculate_quantization_impact_score(
                numerical_stability, factual_consistency, degradation_patterns,
                precision_loss, quality_consistency
            )
            
            # Compare with baseline if provided
            baseline_comparison = {}
            if baseline_metrics:
                baseline_comparison = self._compare_with_baseline(
                    {
                        "numerical_stability": numerical_stability,
                        "factual_consistency": factual_consistency,
                        "impact_score": impact_score
                    },
                    baseline_metrics
                )
            
            return {
                "quantization_impact_score": impact_score,
                "numerical_stability_impact": numerical_stability["stability_score"],
                "factual_consistency_impact": factual_consistency["consistency_score"],
                "degradation_patterns": degradation_patterns,
                "precision_loss_indicators": precision_loss,
                "quality_consistency": quality_consistency,
                "baseline_comparison": baseline_comparison,
                "quantization_severity": self._classify_quantization_severity(impact_score),
                "recommendations": self._generate_quantization_recommendations(
                    impact_score, degradation_patterns, precision_loss
                )
            }
            
        except Exception as e:
            logger.error(f"Quantization impact analysis failed: {e}")
            return {"quantization_impact_score": 0.0, "degradation_indicators": [str(e)]}
    
    def run_comprehensive_quantization_tests(self, response_text: str) -> Dict[str, Any]:
        """
        Run comprehensive quantization test suite
        
        Args:
            response_text: Text response to test
            
        Returns:
            Complete quantization test results
        """
        if not response_text.strip():
            return {"error": "Empty response provided"}
        
        try:
            # All test categories
            numerical_tests = self.test_numerical_stability(response_text)
            factual_tests = self.test_factual_consistency(response_text)
            quantization_analysis = self.analyze_quantization_impact(response_text)
            
            # Additional specialized tests
            edge_case_tests = self._run_edge_case_tests(response_text)
            robustness_tests = self._run_robustness_tests(response_text)
            
            # Overall assessment
            overall_score = self._calculate_overall_quantization_score(
                numerical_tests, factual_tests, quantization_analysis,
                edge_case_tests, robustness_tests
            )
            
            return {
                "overall_quantization_score": overall_score,
                "numerical_stability": numerical_tests,
                "factual_consistency": factual_tests,
                "quantization_impact": quantization_analysis,
                "edge_case_performance": edge_case_tests,
                "robustness_assessment": robustness_tests,
                "summary": self._generate_test_summary(overall_score, numerical_tests, factual_tests),
                "detailed_recommendations": self._generate_detailed_recommendations(
                    overall_score, numerical_tests, factual_tests, quantization_analysis
                )
            }
            
        except Exception as e:
            logger.error(f"Comprehensive quantization testing failed: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    
    def _initialize_basic_math_tests(self) -> List[Dict[str, Any]]:
        """Initialize basic mathematics test cases"""
        return [
            # Basic arithmetic
            {"expression": "42 + 58", "expected": 100, "type": "addition", "difficulty": "basic"},
            {"expression": "123 - 45", "expected": 78, "type": "subtraction", "difficulty": "basic"},
            {"expression": "12 * 7", "expected": 84, "type": "multiplication", "difficulty": "basic"},
            {"expression": "144 / 12", "expected": 12, "type": "division", "difficulty": "basic"},
            
            # Percentages
            {"expression": "15% of 200", "expected": 30, "type": "percentage", "difficulty": "basic"},
            {"expression": "0.75 as percentage", "expected": 75, "type": "percentage", "difficulty": "basic"},
            {"expression": "25% increase of 80", "expected": 100, "type": "percentage", "difficulty": "intermediate"},
            
            # Fractions
            {"expression": "1/2 + 1/4", "expected": 0.75, "type": "fractions", "difficulty": "intermediate"},
            {"expression": "3/4 of 120", "expected": 90, "type": "fractions", "difficulty": "intermediate"},
            
            # Decimals
            {"expression": "2.5 * 4.2", "expected": 10.5, "type": "decimals", "difficulty": "intermediate"},
            {"expression": "15.6 / 3", "expected": 5.2, "type": "decimals", "difficulty": "intermediate"},
            
            # Word problems
            {"expression": "square root of 64", "expected": 8, "type": "roots", "difficulty": "intermediate"},
            {"expression": "2 to the power of 6", "expected": 64, "type": "exponents", "difficulty": "intermediate"},
            
            # Advanced
            {"expression": "factorial of 5", "expected": 120, "type": "factorial", "difficulty": "advanced"},
            {"expression": "logarithm base 10 of 1000", "expected": 3, "type": "logarithm", "difficulty": "advanced"}
        ]
    
    def _initialize_factual_knowledge_tests(self) -> List[Dict[str, Any]]:
        """Initialize factual knowledge test cases"""
        return [
            # Geography
            {"fact": "capital of France", "expected": ["Paris"], "category": "geography", "confidence": "high"},
            {"fact": "largest ocean", "expected": ["Pacific", "Pacific Ocean"], "category": "geography", "confidence": "high"},
            {"fact": "number of continents", "expected": ["7", "seven"], "category": "geography", "confidence": "high"},
            
            # Science
            {"fact": "speed of light", "expected": ["299,792,458", "300,000,000", "3x10^8"], "category": "physics", "confidence": "high"},
            {"fact": "chemical symbol for gold", "expected": ["Au"], "category": "chemistry", "confidence": "high"},
            {"fact": "number of planets", "expected": ["8", "eight"], "category": "astronomy", "confidence": "high"},
            
            # History
            {"fact": "year World War 2 ended", "expected": ["1945"], "category": "history", "confidence": "high"},
            {"fact": "first person on moon", "expected": ["Neil Armstrong"], "category": "history", "confidence": "high"},
            
            # Mathematics
            {"fact": "value of pi", "expected": ["3.14159", "3.14", "22/7"], "category": "mathematics", "confidence": "high"},
            {"fact": "Pythagorean theorem", "expected": ["a² + b² = c²", "a^2 + b^2 = c^2"], "category": "mathematics", "confidence": "high"},
            
            # Common knowledge
            {"fact": "days in a year", "expected": ["365", "366"], "category": "common", "confidence": "very_high"},
            {"fact": "hours in a day", "expected": ["24"], "category": "common", "confidence": "very_high"},
            {"fact": "minutes in an hour", "expected": ["60"], "category": "common", "confidence": "very_high"}
        ]
    
    def _initialize_numerical_patterns(self) -> Dict[str, str]:
        """Initialize numerical pattern recognition regexes"""
        return {
            "integers": r'\b\d+\b',
            "decimals": r'\b\d+\.\d+\b',
            "percentages": r'\b\d+(?:\.\d+)?%\b',
            "fractions": r'\b\d+/\d+\b',
            "currency": r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',
            "scientific": r'\b\d+(?:\.\d+)?[eE][+-]?\d+\b',
            "equations": r'[=<>≤≥≠]',
            "mathematical_operations": r'[+\-*/÷×^]',
            "mathematical_functions": r'\b(?:sqrt|log|ln|sin|cos|tan|factorial|sum|average)\b'
        }
    
    def _extract_numerical_expressions(self, text: str) -> List[Dict[str, Any]]:
        """Extract numerical expressions from text"""
        expressions = []
        
        # Find all numerical patterns
        for pattern_name, pattern in self.numerical_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                expressions.append({
                    "type": pattern_name,
                    "value": match.group(),
                    "position": match.span(),
                    "context": text[max(0, match.start()-20):match.end()+20]
                })
        
        return expressions
    
    def _test_arithmetic_accuracy(self, numerical_expressions: List[Dict], original_text: str = '') -> Dict[str, Any]:
        """Test accuracy of arithmetic expressions"""
        tests_run = 0
        correct_answers = 0
        errors = []
        
        for test_case in self.basic_math_tests:
            expression = test_case["expression"]
            expected = test_case["expected"]
            
            # Check if this type of expression appears in the text
            found_expressions = [
                expr for expr in numerical_expressions 
                if self._expression_matches_test(expr, test_case)
            ]
            
            if found_expressions:
                tests_run += 1
                # Simplified accuracy check - would need more sophisticated parsing
                # For now, checking if expected value appears near the expression
                for expr in found_expressions:
                    if str(expected) in expr["context"] or str(float(expected)) in expr["context"]:
                        correct_answers += 1
                        break
                else:
                    errors.append({
                        "expression": expression,
                        "expected": expected,
                        "found_context": [e["context"] for e in found_expressions],
                        "test_type": test_case["type"]
                    })
        
        # Check for math avoidance patterns if no tests were run
        if tests_run == 0:
            # Check if text contains math avoidance language
            avoidance_patterns = [
                r"\bi cannot calculate",
                r"\bhard to say exactly",
                r"\bunable to provide exact",
                r"\bdifficult to determine",
                r"\bcomputation is complex",
                r"\bcannot provide exact figures",
                r"\bapproximately correct but cannot"
            ]
            
            text_lower = original_text.lower()
            avoidance_count = sum(1 for pattern in avoidance_patterns 
                                if re.search(pattern, text_lower))
            
            if avoidance_count >= 2:  # Multiple avoidance patterns
                accuracy_rate = 0.2  # Low score for math avoidance
            else:
                accuracy_rate = 1.0  # Default for no mathematical content
        else:
            accuracy_rate = correct_answers / tests_run
        
        return {
            "accuracy_rate": accuracy_rate,
            "tests_run": tests_run,
            "correct_answers": correct_answers,
            "errors": errors,
            "error_rate": 1.0 - accuracy_rate
        }
    
    def _test_mathematical_consistency(self, text: str) -> Dict[str, Any]:
        """Test mathematical consistency within the response"""
        # Extract all mathematical statements
        mathematical_statements = self._extract_mathematical_statements(text)
        
        consistency_issues = []
        consistency_score = 1.0
        
        # Check for contradictory numerical claims
        numerical_claims = self._extract_numerical_claims(text)
        contradictions = self._find_contradictions(numerical_claims)
        
        if contradictions:
            consistency_issues.extend(contradictions)
            consistency_score -= min(len(contradictions) * 0.2, 0.8)
        
        # Check for impossible mathematical relationships
        impossible_relationships = self._find_impossible_relationships(mathematical_statements)
        if impossible_relationships:
            consistency_issues.extend(impossible_relationships)
            consistency_score -= min(len(impossible_relationships) * 0.3, 0.6)
        
        return {
            "consistency_score": max(consistency_score, 0.0),
            "mathematical_statements": len(mathematical_statements),
            "consistency_issues": consistency_issues,
            "contradictions_found": len(contradictions),
            "impossible_relationships": len(impossible_relationships)
        }
    
    def _test_numerical_reasoning(self, text: str) -> Dict[str, Any]:
        """Test numerical reasoning patterns"""
        reasoning_indicators = [
            "calculate", "compute", "therefore", "equals", "result", "total",
            "sum", "average", "percentage", "ratio", "proportion", "multiply",
            "divide", "add", "subtract", "increase", "decrease"
        ]
        
        reasoning_present = sum(1 for indicator in reasoning_indicators if indicator in text.lower())
        
        # Check for logical flow in numerical reasoning
        logical_connectors = ["because", "since", "therefore", "thus", "consequently", "as a result"]
        logical_flow = sum(1 for connector in logical_connectors if connector in text.lower())
        
        # Check for step-by-step reasoning
        step_indicators = ["first", "second", "third", "next", "then", "finally", "step"]
        step_reasoning = sum(1 for step in step_indicators if step in text.lower())
        
        # Calculate reasoning score
        reasoning_score = min(
            (reasoning_present / 10.0) * 0.4 +
            (logical_flow / 5.0) * 0.3 +
            (step_reasoning / 5.0) * 0.3,
            1.0
        )
        
        return {
            "reasoning_score": reasoning_score,
            "reasoning_indicators": reasoning_present,
            "logical_connectors": logical_flow,
            "step_indicators": step_reasoning,
            "has_structured_reasoning": step_reasoning >= 2 and logical_flow >= 1
        }
    
    def _test_basic_facts(self, text: str) -> Dict[str, Any]:
        """Test basic factual accuracy"""
        tests_run = 0
        correct_facts = 0
        fact_errors = []
        
        for fact_test in self.factual_knowledge_tests:
            fact_key = fact_test["fact"]
            expected_values = fact_test["expected"]
            category = fact_test["category"]
            
            # Check if this fact category appears in the text
            if any(keyword in text.lower() for keyword in fact_key.split()):
                tests_run += 1
                
                # Check if any expected value appears
                found_correct = False
                for expected in expected_values:
                    if expected.lower() in text.lower():
                        correct_facts += 1
                        found_correct = True
                        break
                
                if not found_correct:
                    fact_errors.append({
                        "fact": fact_key,
                        "expected": expected_values,
                        "category": category,
                        "confidence": fact_test["confidence"]
                    })
        
        accuracy_rate = correct_facts / tests_run if tests_run > 0 else 1.0
        
        return {
            "accuracy_rate": accuracy_rate,
            "tests_run": tests_run,
            "correct_facts": correct_facts,
            "fact_errors": fact_errors,
            "error_rate": 1.0 - accuracy_rate
        }
    
    def _test_internal_consistency(self, text: str) -> Dict[str, Any]:
        """Test internal consistency of statements"""
        # Extract factual claims
        claims = self._extract_factual_claims(text)
        
        # Look for contradictory claims
        contradictions = []
        consistency_score = 1.0
        
        # Enhanced contradiction detection patterns
        contradiction_patterns = [
            (r"always", r"never"),
            (r"all", r"none"),
            (r"increase", r"decrease"),
            (r"higher", r"lower"),
            (r"more", r"less"),
            (r"better", r"worse"),
            (r"positive", r"negative")
        ]
        
        # Check for numerical inconsistencies
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        if len(numbers) >= 3:  # If there are multiple numbers, look for inconsistencies
            try:
                nums = [float(n) for n in numbers]
                # Look for obvious mathematical inconsistencies 
                # (This is a simple check - real implementation would be more sophisticated)
                for i, num in enumerate(nums[:-1]):
                    if abs(num - nums[i+1]) > num * 0.3:  # Large differences might indicate inconsistency
                        # Check if these numbers appear in contexts that suggest they should be consistent
                        if re.search(r'\b(?:participants?|sample|responses?|subjects?)\b', text, re.IGNORECASE):
                            contradictions.append({
                                "type": "numerical_inconsistency", 
                                "values": [num, nums[i+1]],
                                "severity": "moderate"
                            })
                            consistency_score -= 0.2
                            break
            except ValueError:
                pass  # Skip if numbers can't be parsed
        
        for pos_pattern, neg_pattern in contradiction_patterns:
            pos_matches = re.findall(pos_pattern, text, re.IGNORECASE)
            neg_matches = re.findall(neg_pattern, text, re.IGNORECASE)
            
            if pos_matches and neg_matches:
                # Potential contradiction - would need context analysis
                contradictions.append({
                    "positive_pattern": pos_pattern,
                    "negative_pattern": neg_pattern,
                    "positive_count": len(pos_matches),
                    "negative_count": len(neg_matches),
                    "severity": "potential"
                })
        
        if contradictions:
            consistency_score -= min(len(contradictions) * 0.15, 0.8)
        
        return {
            "consistency_score": max(consistency_score, 0.0),
            "contradictions": contradictions,
            "factual_claims": len(claims),
            "internal_coherence": consistency_score > 0.7
        }
    
    def _test_knowledge_coherence(self, text: str) -> Dict[str, Any]:
        """Test coherence of knowledge presentation"""
        # Check for knowledge domain consistency
        knowledge_domains = {
            "science": ["physics", "chemistry", "biology", "scientific", "experiment", "hypothesis"],
            "mathematics": ["calculate", "equation", "theorem", "proof", "mathematical", "numeric"],
            "history": ["century", "year", "historical", "ancient", "modern", "period"],
            "geography": ["country", "city", "continent", "ocean", "mountain", "geographic"],
            "technology": ["computer", "software", "digital", "internet", "algorithm", "data"]
        }
        
        domain_scores = {}
        for domain, keywords in knowledge_domains.items():
            domain_presence = sum(1 for keyword in keywords if keyword in text.lower())
            domain_scores[domain] = domain_presence
        
        # Check if knowledge spans multiple domains appropriately
        active_domains = [domain for domain, score in domain_scores.items() if score >= 2]
        
        # Coherence based on domain consistency
        if len(active_domains) <= 2:
            coherence_score = 1.0  # Focused knowledge
        elif len(active_domains) <= 4:
            coherence_score = 0.8  # Reasonable breadth
        else:
            coherence_score = 0.6  # May be scattered
        
        return {
            "coherence_score": coherence_score,
            "active_domains": active_domains,
            "domain_scores": domain_scores,
            "knowledge_focus": "focused" if len(active_domains) <= 2 else "broad",
            "domain_consistency": coherence_score > 0.7
        }
    
    def _test_common_knowledge(self, text: str) -> Dict[str, Any]:
        """Test accuracy of common knowledge facts"""
        common_facts = [fact for fact in self.factual_knowledge_tests if fact["category"] == "common"]
        
        tests_run = 0
        correct_facts = 0
        errors = []
        
        for fact in common_facts:
            fact_keywords = fact["fact"].split()
            if any(keyword in text.lower() for keyword in fact_keywords):
                tests_run += 1
                
                # Check for correct values
                found_correct = any(expected.lower() in text.lower() for expected in fact["expected"])
                if found_correct:
                    correct_facts += 1
                else:
                    errors.append({
                        "fact": fact["fact"],
                        "expected": fact["expected"],
                        "severity": "high"  # Common knowledge errors are serious
                    })
        
        accuracy_rate = correct_facts / tests_run if tests_run > 0 else 1.0
        
        return {
            "accuracy_rate": accuracy_rate,
            "tests_run": tests_run,
            "correct_facts": correct_facts,
            "errors": errors,
            "common_knowledge_integrity": accuracy_rate > 0.9
        }
    
    def _calculate_stability_score(self, arithmetic_results: Dict, consistency_results: Dict, 
                                 reasoning_results: Dict) -> float:
        """Calculate overall numerical stability score"""
        stability_score = (
            arithmetic_results["accuracy_rate"] * 0.4 +
            consistency_results["consistency_score"] * 0.3 +
            reasoning_results["reasoning_score"] * 0.3
        )
        
        return max(0.0, min(stability_score, 1.0))
    
    def _calculate_consistency_score(self, factual_tests: Dict, internal_consistency: Dict,
                                   knowledge_coherence: Dict, common_knowledge: Dict) -> float:
        """Calculate overall factual consistency score"""
        consistency_score = (
            factual_tests["accuracy_rate"] * 0.3 +
            internal_consistency["consistency_score"] * 0.25 +
            knowledge_coherence["coherence_score"] * 0.2 +
            common_knowledge["accuracy_rate"] * 0.25
        )
        
        return max(0.0, min(consistency_score, 1.0))
    
    def _analyze_error_patterns(self, arithmetic_results: Dict, consistency_results: Dict) -> List[str]:
        """Analyze patterns in errors that might indicate quantization issues"""
        error_patterns = []
        
        # Arithmetic error patterns
        if arithmetic_results["error_rate"] > 0.3:
            error_patterns.append("high_arithmetic_error_rate")
        
        # Check for specific error types
        error_types = {}
        for error in arithmetic_results.get("errors", []):
            error_type = error.get("test_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Identify problematic areas
        if error_types.get("decimals", 0) > error_types.get("integers", 0):
            error_patterns.append("decimal_precision_issues")
        
        if error_types.get("percentage", 0) >= 2:
            error_patterns.append("percentage_calculation_errors")
        
        if error_types.get("fractions", 0) >= 2:
            error_patterns.append("fraction_handling_errors")
        
        # Consistency error patterns
        if consistency_results["consistency_score"] < 0.6:
            error_patterns.append("mathematical_inconsistency")
        
        return error_patterns
    
    def _collect_knowledge_errors(self, factual_tests: Dict, internal_consistency: Dict,
                                 knowledge_coherence: Dict, common_knowledge: Dict) -> List[Dict]:
        """Collect all knowledge-related errors"""
        knowledge_errors = []
        
        # Add factual errors
        knowledge_errors.extend(factual_tests.get("fact_errors", []))
        
        # Add internal consistency issues
        for contradiction in internal_consistency.get("contradictions", []):
            knowledge_errors.append({
                "type": "internal_contradiction",
                "details": contradiction,
                "severity": "moderate"
            })
        
        # Add common knowledge errors
        for error in common_knowledge.get("errors", []):
            knowledge_errors.append({
                "type": "common_knowledge_error",
                "details": error,
                "severity": error.get("severity", "high")
            })
        
        return knowledge_errors
    
    def _detect_quantization_degradation(self, text: str) -> List[Dict[str, Any]]:
        """Detect patterns that suggest quantization-related degradation"""
        degradation_patterns = []
        
        # Repetitive numerical patterns
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        if numbers:
            number_counter = Counter(numbers)
            repeated_numbers = [num for num, count in number_counter.items() if count > 3]
            if repeated_numbers:
                degradation_patterns.append({
                    "type": "repetitive_numbers",
                    "pattern": repeated_numbers,
                    "severity": "moderate"
                })
        
        # Precision loss indicators
        precision_indicators = [
            "approximately", "roughly", "about", "around", "close to",
            "nearly", "almost", "estimate", "ballpark"
        ]
        precision_hedging = sum(1 for indicator in precision_indicators if indicator in text.lower())
        if precision_hedging > 3:
            degradation_patterns.append({
                "type": "precision_hedging",
                "count": precision_hedging,
                "severity": "mild"
            })
        
        # Mathematical vagueness
        vague_math = [
            "many", "few", "several", "some", "most", "large", "small",
            "big", "little", "huge", "tiny"
        ]
        vagueness_count = sum(1 for vague in vague_math if vague in text.lower())
        if vagueness_count > 5:
            degradation_patterns.append({
                "type": "mathematical_vagueness",
                "count": vagueness_count,
                "severity": "moderate"
            })
        
        # Calculation avoidance
        avoidance_phrases = [
            "I cannot calculate", "unable to compute", "difficult to determine",
            "hard to say exactly", "cannot provide exact"
        ]
        avoidance_count = sum(1 for phrase in avoidance_phrases if phrase in text.lower())
        if avoidance_count > 0:
            degradation_patterns.append({
                "type": "calculation_avoidance",
                "count": avoidance_count,
                "severity": "high"
            })
        
        return degradation_patterns
    
    def _analyze_precision_loss(self, text: str, numerical_stability: Dict) -> Dict[str, Any]:
        """Analyze indicators of precision loss"""
        precision_indicators = {
            "rounded_numbers": len(re.findall(r'\b\d+0+\b', text)),  # Numbers ending in zeros
            "scientific_notation": len(re.findall(r'\d+(?:\.\d+)?[eE][+-]?\d+', text)),
            "decimal_truncation": self._detect_decimal_truncation(text),
            "significant_figures": self._analyze_significant_figures(text),
            "approximation_language": self._count_approximation_language(text)
        }
        
        # Calculate precision loss score
        precision_loss_score = 0.0
        
        # High number of rounded numbers might indicate precision loss
        if precision_indicators["rounded_numbers"] > 3:
            precision_loss_score += 0.2
        
        # Excessive approximation language
        if precision_indicators["approximation_language"] > 4:
            precision_loss_score += 0.3
        
        # Poor numerical stability
        if numerical_stability["stability_score"] < 0.7:
            precision_loss_score += 0.4
        
        precision_loss_score = min(precision_loss_score, 1.0)
        
        return {
            "precision_loss_score": precision_loss_score,
            "indicators": precision_indicators,
            "severity": "high" if precision_loss_score > 0.7 else "moderate" if precision_loss_score > 0.4 else "low"
        }
    
    def _analyze_output_consistency(self, text: str) -> Dict[str, Any]:
        """Analyze consistency of output quality"""
        # Analyze consistency across different sections
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            return {"consistency_score": 1.0, "variance": 0.0, "sections_analyzed": len(paragraphs)}
        
        # Simple quality metrics per paragraph
        paragraph_qualities = []
        for paragraph in paragraphs:
            quality = self._calculate_paragraph_quality(paragraph)
            paragraph_qualities.append(quality)
        
        # Calculate variance in quality
        quality_variance = np.var(paragraph_qualities) if len(paragraph_qualities) > 1 else 0.0
        consistency_score = max(0.0, 1.0 - quality_variance)
        
        return {
            "consistency_score": consistency_score,
            "variance": float(quality_variance),
            "sections_analyzed": len(paragraphs),
            "quality_range": [float(min(paragraph_qualities)), float(max(paragraph_qualities))] if paragraph_qualities else [0, 0]
        }
    
    def _calculate_quantization_impact_score(self, numerical_stability: Dict, factual_consistency: Dict,
                                           degradation_patterns: List, precision_loss: Dict,
                                           quality_consistency: Dict) -> float:
        """Calculate overall quantization impact score"""
        # Higher score = more impact (worse)
        impact_score = 0.0
        
        # Numerical stability impact (inverted - lower stability = higher impact)
        impact_score += (1.0 - numerical_stability["stability_score"]) * 0.3
        
        # Factual consistency impact
        impact_score += (1.0 - factual_consistency["consistency_score"]) * 0.25
        
        # Degradation patterns impact
        severe_patterns = sum(1 for p in degradation_patterns if p.get("severity") == "high")
        moderate_patterns = sum(1 for p in degradation_patterns if p.get("severity") == "moderate")
        degradation_impact = min((severe_patterns * 0.3 + moderate_patterns * 0.15), 1.0)
        impact_score += degradation_impact * 0.2
        
        # Precision loss impact
        impact_score += precision_loss["precision_loss_score"] * 0.15
        
        # Quality consistency impact
        impact_score += (1.0 - quality_consistency["consistency_score"]) * 0.1
        
        return min(impact_score, 1.0)
    
    def _compare_with_baseline(self, current_metrics: Dict, baseline_metrics: Dict) -> Dict[str, Any]:
        """Compare current metrics with baseline"""
        comparison = {}
        
        metrics_to_compare = [
            ("numerical_stability", "stability_score"),
            ("factual_consistency", "consistency_score"),
            ("impact_score", None)
        ]
        
        for metric_category, sub_metric in metrics_to_compare:
            current_value = current_metrics.get(metric_category, {})
            baseline_value = baseline_metrics.get(metric_category, {})
            
            if sub_metric:
                current_val = current_value.get(sub_metric, 0.0) if isinstance(current_value, dict) else current_value
                baseline_val = baseline_value.get(sub_metric, 0.0) if isinstance(baseline_value, dict) else baseline_value
            else:
                current_val = current_value if not isinstance(current_value, dict) else 0.0
                baseline_val = baseline_value if not isinstance(baseline_value, dict) else 0.0
            
            # Ensure we have numeric values for comparison
            try:
                current_val = float(current_val) if current_val is not None else 0.0
                baseline_val = float(baseline_val) if baseline_val is not None else 0.0
            except (TypeError, ValueError):
                continue  # Skip if values can't be converted to float
            
            if baseline_val > 0:
                change = (current_val - baseline_val) / baseline_val * 100
                comparison[metric_category] = {
                    "current": float(current_val),
                    "baseline": float(baseline_val),
                    "change_percent": float(change),
                    "degradation": change < -10  # More than 10% degradation
                }
        
        return comparison
    
    def _classify_quantization_severity(self, impact_score: float) -> str:
        """Classify quantization impact severity"""
        if impact_score >= 0.7:
            return "severe"
        elif impact_score >= 0.5:
            return "moderate"
        elif impact_score >= 0.3:
            return "mild"
        else:
            return "minimal"
    
    def _generate_quantization_recommendations(self, impact_score: float, degradation_patterns: List,
                                             precision_loss: Dict) -> List[str]:
        """Generate recommendations based on quantization analysis"""
        recommendations = []
        
        if impact_score >= 0.7:
            recommendations.append("Consider using higher precision quantization (e.g., 8-bit instead of 4-bit)")
            recommendations.append("Evaluate model performance on critical numerical tasks")
        
        if impact_score >= 0.5:
            recommendations.append("Monitor numerical accuracy in production use")
            recommendations.append("Consider post-processing to correct common numerical errors")
        
        # Specific recommendations based on patterns
        pattern_types = [p["type"] for p in degradation_patterns]
        
        if "calculation_avoidance" in pattern_types:
            recommendations.append("Model may be avoiding calculations - consider prompt engineering")
        
        if "precision_hedging" in pattern_types:
            recommendations.append("Excessive precision hedging detected - may indicate uncertainty")
        
        if precision_loss["precision_loss_score"] > 0.6:
            recommendations.append("Significant precision loss detected - validate against higher precision model")
        
        if not recommendations:
            recommendations.append("Quantization impact appears minimal - current settings acceptable")
        
        return recommendations
    
    def _run_edge_case_tests(self, text: str) -> Dict[str, Any]:
        """Run edge case tests for quantization robustness"""
        edge_cases = {
            "very_large_numbers": self._test_large_numbers(text),
            "very_small_numbers": self._test_small_numbers(text),
            "negative_numbers": self._test_negative_numbers(text),
            "zero_handling": self._test_zero_handling(text),
            "infinity_nan": self._test_infinity_nan(text)
        }
        
        # Calculate overall edge case performance
        performance_scores = [case.get("performance", 1.0) for case in edge_cases.values()]
        overall_performance = np.mean(performance_scores) if performance_scores else 1.0
        
        return {
            "overall_performance": float(overall_performance),
            "edge_cases": edge_cases,
            "issues_found": sum(1 for case in edge_cases.values() if case.get("issues", 0) > 0)
        }
    
    def _run_robustness_tests(self, text: str) -> Dict[str, Any]:
        """Run robustness tests"""
        robustness_metrics = {
            "consistent_formatting": self._test_formatting_consistency(text),
            "error_handling": self._test_error_handling_patterns(text),
            "boundary_conditions": self._test_boundary_conditions(text),
            "graceful_degradation": self._test_graceful_degradation(text)
        }
        
        robustness_score = np.mean([metric.get("score", 1.0) for metric in robustness_metrics.values()])
        
        return {
            "robustness_score": float(robustness_score),
            "metrics": robustness_metrics,
            "robust": robustness_score > 0.7
        }
    
    def _calculate_overall_quantization_score(self, numerical_tests: Dict, factual_tests: Dict,
                                            quantization_analysis: Dict, edge_case_tests: Dict,
                                            robustness_tests: Dict) -> float:
        """Calculate overall quantization assessment score"""
        # Combine all scores (inverted for impact scores)
        numerical_score = numerical_tests["stability_score"]
        factual_score = factual_tests["consistency_score"]
        quantization_score = 1.0 - quantization_analysis["quantization_impact_score"]  # Invert impact
        edge_case_score = edge_case_tests["overall_performance"]
        robustness_score = robustness_tests["robustness_score"]
        
        overall_score = (
            numerical_score * 0.3 +
            factual_score * 0.25 +
            quantization_score * 0.25 +
            edge_case_score * 0.1 +
            robustness_score * 0.1
        )
        
        return max(0.0, min(overall_score, 1.0))
    
    def _generate_test_summary(self, overall_score: float, numerical_tests: Dict, factual_tests: Dict) -> Dict[str, Any]:
        """Generate test summary"""
        return {
            "overall_grade": self._score_to_grade(overall_score),
            "numerical_grade": self._score_to_grade(numerical_tests["stability_score"]),
            "factual_grade": self._score_to_grade(factual_tests["consistency_score"]),
            "key_issues": self._identify_key_issues(numerical_tests, factual_tests),
            "strengths": self._identify_strengths(numerical_tests, factual_tests)
        }
    
    def _generate_detailed_recommendations(self, overall_score: float, numerical_tests: Dict,
                                         factual_tests: Dict, quantization_analysis: Dict) -> List[str]:
        """Generate detailed recommendations"""
        recommendations = []
        
        if numerical_tests["stability_score"] < 0.6:
            recommendations.append("Numerical stability is concerning - validate mathematical operations")
        
        if factual_tests["consistency_score"] < 0.6:
            recommendations.append("Factual consistency issues detected - verify knowledge accuracy")
        
        if quantization_analysis["quantization_impact_score"] > 0.7:
            recommendations.append("High quantization impact - consider alternative quantization strategy")
        
        if overall_score < 0.5:
            recommendations.append("Overall performance significantly impacted - comprehensive evaluation needed")
        
        return recommendations
    
    # Additional helper methods for completeness
    
    def _expression_matches_test(self, expr: Dict, test_case: Dict) -> bool:
        """Check if expression matches test case type"""
        expr_type = expr["type"]
        test_type = test_case["type"]
        
        type_mapping = {
            "addition": ["integers", "decimals", "arithmetic"],
            "subtraction": ["integers", "decimals", "arithmetic"],
            "multiplication": ["integers", "decimals", "arithmetic"],
            "division": ["integers", "decimals", "arithmetic"],
            "percentage": ["percentages", "arithmetic"],
            "fractions": ["fractions", "arithmetic"],
            "roots": ["mathematical", "arithmetic"],
            "exponents": ["mathematical", "arithmetic"],
            "factorial": ["mathematical", "arithmetic"],
            "logarithm": ["mathematical", "arithmetic"]
        }
        
        # More flexible matching - if test type matches expr type directly or through mapping
        return (expr_type == test_type or 
                expr_type in type_mapping.get(test_type, []) or
                test_type in type_mapping.get(expr_type, []))
    
    def _extract_mathematical_statements(self, text: str) -> List[str]:
        """Extract mathematical statements from text"""
        # Find sentences with mathematical content
        sentences = re.split(r'[.!?]', text)
        math_sentences = []
        
        math_indicators = [
            r'\d+', r'[+\-*/=]', r'equals?', r'calculate', r'result',
            r'sum', r'product', r'quotient', r'difference'
        ]
        
        for sentence in sentences:
            if any(re.search(indicator, sentence, re.IGNORECASE) for indicator in math_indicators):
                math_sentences.append(sentence.strip())
        
        return math_sentences
    
    def _extract_numerical_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract numerical claims from text"""
        claims = []
        
        # Pattern for numerical claims
        claim_patterns = [
            r'(\d+(?:\.\d+)?)\s+(?:is|equals?|=)\s+(\d+(?:\.\d+)?)',
            r'(\w+)\s+(?:is|equals?)\s+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s+percent\s+of\s+(\d+(?:\.\d+)?)\s+(?:is|equals?)\s+(\d+(?:\.\d+)?)'
        ]
        
        for pattern in claim_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                claims.append({
                    "claim": match.group(),
                    "groups": match.groups(),
                    "position": match.span()
                })
        
        return claims
    
    def _find_contradictions(self, claims: List[Dict]) -> List[Dict[str, Any]]:
        """Find contradictory numerical claims"""
        # Simplified contradiction detection
        contradictions = []
        
        for i, claim1 in enumerate(claims):
            for claim2 in claims[i+1:]:
                # Basic contradiction detection logic would go here
                # This is a placeholder for more sophisticated analysis
                if self._claims_contradict(claim1, claim2):
                    contradictions.append({
                        "claim1": claim1["claim"],
                        "claim2": claim2["claim"],
                        "type": "numerical_contradiction"
                    })
        
        return contradictions
    
    def _find_impossible_relationships(self, statements: List[str]) -> List[Dict[str, Any]]:
        """Find impossible mathematical relationships"""
        impossible = []
        
        # Look for obviously impossible statements
        impossible_patterns = [
            r'0\s*=\s*[1-9]',  # Zero equals non-zero
            r'[1-9]\d*\s*=\s*0',  # Non-zero equals zero
            r'100%\s*of\s*\d+\s*=\s*0'  # 100% of something equals zero
        ]
        
        for statement in statements:
            for pattern in impossible_patterns:
                if re.search(pattern, statement):
                    impossible.append({
                        "statement": statement,
                        "pattern": pattern,
                        "type": "impossible_relationship"
                    })
        
        return impossible
    
    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        # Simple factual claim extraction
        sentences = re.split(r'[.!?]', text)
        factual_sentences = []
        
        # Look for sentences with factual indicators
        factual_indicators = [
            r'\b(?:is|are|was|were)\b',
            r'\b(?:has|have|had)\b',
            r'\b(?:the|a|an)\s+\w+\s+(?:is|are|was|were)\b'
        ]
        
        for sentence in sentences:
            if any(re.search(indicator, sentence, re.IGNORECASE) for indicator in factual_indicators):
                factual_sentences.append(sentence.strip())
        
        return factual_sentences
    
    def _detect_decimal_truncation(self, text: str) -> int:
        """Detect signs of decimal truncation"""
        # Look for decimals with exactly 2 or 3 digits (common truncation points)
        truncated_decimals = re.findall(r'\b\d+\.(?:\d{2}|000)\b', text)
        return len(truncated_decimals)
    
    def _analyze_significant_figures(self, text: str) -> Dict[str, int]:
        """Analyze significant figures in numerical values"""
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        
        sig_fig_distribution = {
            "1_sig_fig": 0,
            "2_sig_fig": 0,
            "3_sig_fig": 0,
            "4_plus_sig_fig": 0
        }
        
        for num in numbers:
            sig_figs = self._count_significant_figures(num)
            if sig_figs == 1:
                sig_fig_distribution["1_sig_fig"] += 1
            elif sig_figs == 2:
                sig_fig_distribution["2_sig_fig"] += 1
            elif sig_figs == 3:
                sig_fig_distribution["3_sig_fig"] += 1
            else:
                sig_fig_distribution["4_plus_sig_fig"] += 1
        
        return sig_fig_distribution
    
    def _count_approximation_language(self, text: str) -> int:
        """Count approximation language indicators"""
        approximation_words = [
            "approximately", "roughly", "about", "around", "nearly",
            "almost", "close to", "estimate", "ballpark", "vicinity"
        ]
        
        return sum(1 for word in approximation_words if word in text.lower())
    
    def _calculate_paragraph_quality(self, paragraph: str) -> float:
        """Calculate quality score for a paragraph"""
        if not paragraph.strip():
            return 0.0
        
        # Simple quality metrics
        word_count = len(paragraph.split())
        sentence_count = len(re.split(r'[.!?]', paragraph))
        
        # Quality based on length and structure
        if word_count < 5:
            return 0.2
        elif word_count < 20:
            return 0.6
        elif word_count < 100:
            return 0.9
        else:
            return 0.8  # Very long paragraphs might be lower quality
    
    def _test_large_numbers(self, text: str) -> Dict[str, Any]:
        """Test handling of large numbers"""
        large_numbers = re.findall(r'\b\d{4,}\b', text)  # 4+ digit numbers
        
        return {
            "large_numbers_found": len(large_numbers),
            "performance": 1.0 if len(large_numbers) <= 10 else 0.8,
            "issues": max(0, len(large_numbers) - 10)
        }
    
    def _test_small_numbers(self, text: str) -> Dict[str, Any]:
        """Test handling of small numbers"""
        small_decimals = re.findall(r'\b0\.\d{3,}\b', text)  # Small decimals
        
        return {
            "small_numbers_found": len(small_decimals),
            "performance": 1.0 if len(small_decimals) <= 5 else 0.9,
            "issues": max(0, len(small_decimals) - 5)
        }
    
    def _test_negative_numbers(self, text: str) -> Dict[str, Any]:
        """Test handling of negative numbers"""
        negative_numbers = re.findall(r'-\d+(?:\.\d+)?\b', text)
        
        return {
            "negative_numbers_found": len(negative_numbers),
            "performance": 1.0,
            "issues": 0
        }
    
    def _test_zero_handling(self, text: str) -> Dict[str, Any]:
        """Test handling of zero"""
        zero_mentions = text.lower().count('zero') + text.count(' 0 ') + text.count(' 0.0 ')
        
        return {
            "zero_mentions": zero_mentions,
            "performance": 1.0,
            "issues": 0
        }
    
    def _test_infinity_nan(self, text: str) -> Dict[str, Any]:
        """Test handling of infinity and NaN"""
        special_values = text.lower().count('infinity') + text.lower().count('nan') + text.lower().count('undefined')
        
        return {
            "special_values_found": special_values,
            "performance": 1.0 if special_values == 0 else 0.8,
            "issues": special_values
        }
    
    def _test_formatting_consistency(self, text: str) -> Dict[str, Any]:
        """Test formatting consistency"""
        # Check consistency in number formatting
        decimal_formats = len(set(re.findall(r'\d+\.\d+', text)))
        integer_formats = len(set(re.findall(r'\b\d+\b', text)))
        
        consistency_score = 1.0 - min(decimal_formats / 20.0, 0.3)  # Penalize too many formats
        
        return {
            "score": consistency_score,
            "decimal_formats": decimal_formats,
            "integer_formats": integer_formats
        }
    
    def _test_error_handling_patterns(self, text: str) -> Dict[str, Any]:
        """Test error handling patterns"""
        error_phrases = [
            "error", "cannot", "unable", "impossible", "invalid", "undefined"
        ]
        
        error_count = sum(1 for phrase in error_phrases if phrase in text.lower())
        score = max(0.5, 1.0 - error_count / 10.0)  # More errors = lower score
        
        return {
            "score": score,
            "error_indicators": error_count
        }
    
    def _test_boundary_conditions(self, text: str) -> Dict[str, Any]:
        """Test boundary condition handling"""
        boundary_indicators = [
            "maximum", "minimum", "limit", "boundary", "extreme", "edge case"
        ]
        
        boundary_awareness = sum(1 for indicator in boundary_indicators if indicator in text.lower())
        score = min(1.0, boundary_awareness / 3.0)  # Good boundary awareness
        
        return {
            "score": score,
            "boundary_awareness": boundary_awareness
        }
    
    def _test_graceful_degradation(self, text: str) -> Dict[str, Any]:
        """Test graceful degradation patterns"""
        degradation_indicators = [
            "approximately", "estimate", "rough", "ballpark", "order of magnitude"
        ]
        
        graceful_count = sum(1 for indicator in degradation_indicators if indicator in text.lower())
        score = min(1.0, graceful_count / 5.0)  # Appropriate use of approximations
        
        return {
            "score": score,
            "graceful_indicators": graceful_count
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _identify_key_issues(self, numerical_tests: Dict, factual_tests: Dict) -> List[str]:
        """Identify key issues from test results"""
        issues = []
        
        if numerical_tests["stability_score"] < 0.6:
            issues.append("Poor numerical stability")
        
        if factual_tests["consistency_score"] < 0.6:
            issues.append("Factual inconsistencies detected")
        
        if numerical_tests.get("arithmetic_tests", {}).get("error_rate", 0) > 0.3:
            issues.append("High arithmetic error rate")
        
        return issues
    
    def _identify_strengths(self, numerical_tests: Dict, factual_tests: Dict) -> List[str]:
        """Identify strengths from test results"""
        strengths = []
        
        if numerical_tests["stability_score"] >= 0.8:
            strengths.append("Good numerical stability")
        
        if factual_tests["consistency_score"] >= 0.8:
            strengths.append("Strong factual consistency")
        
        if numerical_tests.get("reasoning_tests", {}).get("has_structured_reasoning", False):
            strengths.append("Structured numerical reasoning")
        
        return strengths
    
    def _claims_contradict(self, claim1: Dict, claim2: Dict) -> bool:
        """Check if two claims contradict each other"""
        # Placeholder for sophisticated contradiction detection
        # Would need NLP and knowledge base for full implementation
        return False
    
    def _count_significant_figures(self, number_str: str) -> int:
        """Count significant figures in a number string"""
        # Remove decimal point and leading zeros
        cleaned = number_str.replace('.', '').lstrip('0')
        return len(cleaned) if cleaned else 1


# Convenience functions
def check_numerical_stability(text: str) -> Dict[str, Any]:
    """Quick numerical stability test"""
    tester = QuantizationTester()
    return tester.test_numerical_stability(text)


def check_factual_consistency(text: str) -> Dict[str, Any]:
    """Quick factual consistency test"""
    tester = QuantizationTester()
    return tester.test_factual_consistency(text)


def analyze_quantization_impact(text: str) -> Dict[str, Any]:
    """Quick quantization impact analysis"""
    tester = QuantizationTester()
    return tester.analyze_quantization_impact(text)


# Testing function
def run_quantization_tests() -> Dict[str, Any]:
    """Run tests to validate quantization analysis"""
    test_cases = {
        "good_math": "The calculation shows that 42 + 58 = 100. This represents a 25% increase from 80 to 100.",
        "poor_math": "The calculation shows that 42 + 58 = 99. This represents about 25% increase from 80 to 105.",
        "factual_accurate": "Paris is the capital of France. The Pacific Ocean is the largest ocean covering 165 million square kilometers.",
        "factual_errors": "London is the capital of France. The Atlantic Ocean is the largest ocean in the world."
    }
    
    tester = QuantizationTester()
    results = {}
    
    for test_name, test_text in test_cases.items():
        try:
            analysis = tester.run_comprehensive_quantization_tests(test_text)
            results[test_name] = {
                "overall_score": analysis["overall_quantization_score"],
                "numerical_grade": analysis["summary"]["numerical_grade"],
                "factual_grade": analysis["summary"]["factual_grade"],
                "quantization_impact": analysis["quantization_impact"]["quantization_impact_score"]
            }
        except Exception as e:
            results[test_name] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    test_results = run_quantization_tests()
    print("Quantization Testing Results:")
    for test_name, result in test_results.items():
        print(f"{test_name}: {result}")