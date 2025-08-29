"""
Knowledge Validator Module

Ground-truth validation system for factual accuracy and knowledge consistency testing.
Addresses the critique's insight about missing factual grounding and knowledge validation
capabilities in language model evaluation.

This module implements:
- Factual grounding tests with expected/failure token patterns
- Knowledge consistency validation across different question formats
- Confidence calibration measurement and analysis
- Comprehensive built-in factual test database

Author: Claude Code
Version: 1.0.0
"""

import re
import math
import logging
import random
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np

# Optional imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Advanced semantic validation disabled.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available. TF-IDF validation disabled.")

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class FactualTest:
    """Individual factual test case"""
    question: str
    expected_tokens: Set[str]
    forbidden_tokens: Set[str]
    category: str
    difficulty: str
    confidence_threshold: float = 0.5


@dataclass
class ValidationResult:
    """Results from knowledge validation testing"""
    test_id: str
    question: str
    response: str
    factual_accuracy: float
    expected_tokens_found: List[str]
    forbidden_tokens_found: List[str]
    confidence_score: float
    knowledge_consistency_score: float
    category: str
    passed: bool


@dataclass
class KnowledgeValidationReport:
    """Comprehensive knowledge validation report"""
    overall_accuracy: float
    category_breakdown: Dict[str, float]
    confidence_calibration: Dict[str, Any]
    consistency_analysis: Dict[str, Any]
    failure_analysis: Dict[str, Any]
    detailed_results: List[ValidationResult]


class KnowledgeValidator:
    """
    Ground-truth knowledge validation system for language models.
    
    Evaluates factual accuracy, knowledge consistency across formats,
    and confidence calibration to assess model reliability and factual grounding.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize knowledge validator
        
        Args:
            embedding_model: Sentence transformer model for semantic analysis
        """
        self.embedding_model_name = embedding_model
        self._embedding_model = None
        
        # Initialize factual test database
        self.factual_tests = self._initialize_factual_tests()
        
        # Initialize validation thresholds
        self.validation_thresholds = {
            'high_accuracy': 0.9,
            'good_accuracy': 0.75,
            'moderate_accuracy': 0.6,
            'poor_accuracy': 0.0
        }
        
        # Confidence markers
        self.confidence_markers = {
            'high': ["certainly", "definitely", "clearly", "obviously", "undoubtedly", "absolutely", "sure", "confident"],
            'medium': ["likely", "probably", "generally", "typically", "usually", "commonly", "often"],
            'low': ["perhaps", "maybe", "possibly", "might", "could be", "seems", "appears", "I think", "I believe"],
            'uncertain': ["unsure", "not sure", "don't know", "unclear", "ambiguous", "uncertain", "confused", "unknown"]
        }
        
        logger.info("KnowledgeValidator initialized")
    
    @property
    def embedding_model(self):
        """Lazy load embedding model"""
        if self._embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
                self._embedding_model = None
        return self._embedding_model
    
    def _initialize_factual_tests(self) -> Dict[str, List[FactualTest]]:
        """Initialize comprehensive factual test database"""
        tests = {
            "geography": [
                FactualTest("What is the capital of France?", 
                           {"paris"}, {"london", "berlin", "rome", "madrid"}, 
                           "geography", "easy"),
                FactualTest("Which country has the largest land area?", 
                           {"russia"}, {"china", "usa", "canada", "brazil"}, 
                           "geography", "medium"),
                FactualTest("What is the highest mountain in the world?", 
                           {"everest", "mount everest"}, {"k2", "kilimanjaro", "denali"}, 
                           "geography", "easy"),
                FactualTest("Which river is the longest in the world?", 
                           {"nile"}, {"amazon", "yangtze", "mississippi"}, 
                           "geography", "medium"),
                FactualTest("What is the smallest country in the world?", 
                           {"vatican", "vatican city"}, {"monaco", "nauru", "liechtenstein"}, 
                           "geography", "medium"),
            ],
            
            "science": [
                FactualTest("What is the chemical symbol for gold?", 
                           {"au"}, {"go", "gd", "ag", "fe"}, 
                           "science", "easy"),
                FactualTest("How many planets are in our solar system?", 
                           {"eight", "8"}, {"nine", "9", "seven", "7", "ten", "10"}, 
                           "science", "easy"),
                FactualTest("What gas makes up most of Earth's atmosphere?", 
                           {"nitrogen"}, {"oxygen", "carbon dioxide", "argon"}, 
                           "science", "medium"),
                FactualTest("What is the speed of light in vacuum?", 
                           {"299792458", "3*10^8", "300000000"}, {"150000000", "500000000"}, 
                           "science", "hard"),
                FactualTest("Which blood type is considered the universal donor?", 
                           {"o negative", "o-"}, {"ab positive", "a positive", "b negative"}, 
                           "science", "medium"),
            ],
            
            "mathematics": [
                FactualTest("What is 7 × 8?", 
                           {"56"}, {"54", "58", "48", "64"}, 
                           "mathematics", "easy"),
                FactualTest("What is the value of π (pi) to two decimal places?", 
                           {"3.14"}, {"3.15", "3.13", "3.16", "22/7"}, 
                           "mathematics", "easy"),
                FactualTest("What is the square root of 144?", 
                           {"12"}, {"11", "13", "14", "16"}, 
                           "mathematics", "easy"),
                FactualTest("What is 2^10?", 
                           {"1024"}, {"1000", "512", "2048", "100"}, 
                           "mathematics", "medium"),
                FactualTest("What is the derivative of x^2?", 
                           {"2x", "2*x"}, {"x", "x^2", "2", "x^3"}, 
                           "mathematics", "medium"),
            ],
            
            "history": [
                FactualTest("In which year did World War II end?", 
                           {"1945"}, {"1944", "1946", "1943", "1947"}, 
                           "history", "easy"),
                FactualTest("Who was the first President of the United States?", 
                           {"george washington", "washington"}, {"john adams", "thomas jefferson", "benjamin franklin"}, 
                           "history", "easy"),
                FactualTest("Which ancient wonder of the world was located in Alexandria?", 
                           {"lighthouse", "pharos"}, {"pyramids", "colossus", "hanging gardens"}, 
                           "history", "medium"),
                FactualTest("What year did the Berlin Wall fall?", 
                           {"1989"}, {"1987", "1990", "1991", "1988"}, 
                           "history", "medium"),
                FactualTest("Who painted the ceiling of the Sistine Chapel?", 
                           {"michelangelo"}, {"leonardo da vinci", "raphael", "donatello"}, 
                           "history", "medium"),
            ],
            
            "literature": [
                FactualTest("Who wrote 'Romeo and Juliet'?", 
                           {"shakespeare", "william shakespeare"}, {"marlowe", "jonson", "webster"}, 
                           "literature", "easy"),
                FactualTest("What is the first book in the Harry Potter series?", 
                           {"philosopher's stone", "sorcerer's stone"}, {"chamber of secrets", "prisoner of azkaban"}, 
                           "literature", "easy"),
                FactualTest("Which novel begins with 'Call me Ishmael'?", 
                           {"moby dick", "moby-dick"}, {"white whale", "pequod", "ahab"}, 
                           "literature", "medium"),
                FactualTest("Who wrote '1984'?", 
                           {"george orwell", "orwell"}, {"aldous huxley", "ray bradbury", "kurt vonnegut"}, 
                           "literature", "medium"),
                FactualTest("In Greek mythology, who is the king of the gods?", 
                           {"zeus"}, {"poseidon", "hades", "apollo", "ares"}, 
                           "literature", "easy"),
            ],
            
            "general": [
                FactualTest("How many days are in a leap year?", 
                           {"366"}, {"365", "364", "367", "360"}, 
                           "general", "easy"),
                FactualTest("What is the largest mammal in the world?", 
                           {"blue whale", "whale"}, {"elephant", "giraffe", "hippopotamus"}, 
                           "general", "easy"),
                FactualTest("Which vitamin is produced when skin is exposed to sunlight?", 
                           {"vitamin d", "d"}, {"vitamin c", "vitamin a", "vitamin b"}, 
                           "general", "medium"),
                FactualTest("What is the hardest natural substance on Earth?", 
                           {"diamond"}, {"quartz", "steel", "titanium", "graphite"}, 
                           "general", "medium"),
                FactualTest("How many strings does a standard guitar have?", 
                           {"six", "6"}, {"four", "5", "seven", "8"}, 
                           "general", "easy"),
            ]
        }
        
        return tests
    
    def validate_factual_knowledge(self, model_evaluator_func: callable, 
                                 categories: Optional[List[str]] = None,
                                 num_tests_per_category: int = 5) -> KnowledgeValidationReport:
        """
        Validate factual knowledge using built-in test database
        
        Args:
            model_evaluator_func: Function that takes a question and returns a response
            categories: List of categories to test (default: all)
            num_tests_per_category: Number of tests per category
            
        Returns:
            KnowledgeValidationReport with comprehensive validation results
        """
        if categories is None:
            categories = list(self.factual_tests.keys())
        
        all_results = []
        category_results = defaultdict(list)
        
        for category in categories:
            if category not in self.factual_tests:
                logger.warning(f"Category '{category}' not found in factual tests")
                continue
            
            # Select tests for this category
            available_tests = self.factual_tests[category]
            selected_tests = random.sample(available_tests, 
                                         min(num_tests_per_category, len(available_tests)))
            
            logger.info(f"Testing {len(selected_tests)} factual questions in category: {category}")
            
            for i, test in enumerate(selected_tests):
                try:
                    # Get model response
                    response = model_evaluator_func(test.question)
                    
                    # Validate response
                    result = self._validate_single_response(test, response, f"{category}_{i}")
                    all_results.append(result)
                    category_results[category].append(result)
                    
                except Exception as e:
                    logger.warning(f"Failed to test question '{test.question}': {e}")
                    continue
        
        # Generate comprehensive report
        return self._generate_validation_report(all_results, category_results)
    
    def test_knowledge_consistency(self, model_evaluator_func: callable,
                                 test_variations: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Test knowledge consistency across different question formats
        
        Args:
            model_evaluator_func: Function that takes a question and returns a response
            test_variations: Custom test variations (default: built-in variations)
            
        Returns:
            Dictionary with consistency analysis results
        """
        if test_variations is None:
            test_variations = self._get_consistency_test_variations()
        
        consistency_results = {}
        
        for topic, variations in test_variations.items():
            logger.info(f"Testing consistency for topic: {topic}")
            
            responses = []
            for variation in variations:
                try:
                    response = model_evaluator_func(variation)
                    responses.append((variation, response))
                except Exception as e:
                    logger.warning(f"Failed to get response for variation '{variation}': {e}")
                    continue
            
            if len(responses) >= 2:
                consistency_score = self._calculate_knowledge_consistency(responses)
                consistency_results[topic] = {
                    'consistency_score': consistency_score,
                    'num_variations': len(responses),
                    'responses': responses,
                    'assessment': self._assess_consistency(consistency_score)
                }
            else:
                logger.warning(f"Insufficient responses for consistency testing: {topic}")
        
        return consistency_results
    
    def _validate_single_response(self, test: FactualTest, response: str, test_id: str) -> ValidationResult:
        """Validate a single response against factual test"""
        response_lower = response.lower().strip()
        
        # Check for expected tokens
        expected_found = []
        for token in test.expected_tokens:
            if token.lower() in response_lower:
                expected_found.append(token)
        
        # Check for forbidden tokens
        forbidden_found = []
        for token in test.forbidden_tokens:
            if token.lower() in response_lower:
                forbidden_found.append(token)
        
        # Calculate factual accuracy score
        accuracy_score = self._calculate_factual_accuracy(expected_found, forbidden_found, test)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(response)
        
        # Calculate knowledge consistency (placeholder - would need multiple responses)
        consistency_score = 1.0  # Default for single response
        
        # Determine if test passed
        passed = (accuracy_score >= test.confidence_threshold and 
                 len(forbidden_found) == 0 and 
                 len(expected_found) > 0)
        
        return ValidationResult(
            test_id=test_id,
            question=test.question,
            response=response,
            factual_accuracy=accuracy_score,
            expected_tokens_found=expected_found,
            forbidden_tokens_found=forbidden_found,
            confidence_score=confidence_score,
            knowledge_consistency_score=consistency_score,
            category=test.category,
            passed=passed
        )
    
    def _calculate_factual_accuracy(self, expected_found: List[str], 
                                  forbidden_found: List[str], 
                                  test: FactualTest) -> float:
        """Calculate factual accuracy score"""
        # Base score from expected tokens found
        expected_ratio = len(expected_found) / max(1, len(test.expected_tokens))
        
        # Penalty for forbidden tokens
        forbidden_penalty = len(forbidden_found) * 0.5
        
        # Combined score
        accuracy = max(0.0, expected_ratio - forbidden_penalty)
        
        return float(np.clip(accuracy, 0.0, 1.0))
    
    def _calculate_confidence_score(self, response: str) -> float:
        """Calculate confidence score based on linguistic markers"""
        response_lower = response.lower()
        
        # Count confidence markers
        high_confidence = sum(1 for marker in self.confidence_markers['high'] 
                            if marker in response_lower)
        medium_confidence = sum(1 for marker in self.confidence_markers['medium'] 
                              if marker in response_lower)
        low_confidence = sum(1 for marker in self.confidence_markers['low'] 
                           if marker in response_lower)
        uncertain = sum(1 for marker in self.confidence_markers['uncertain'] 
                       if marker in response_lower)
        
        # Calculate weighted score
        total_markers = high_confidence + medium_confidence + low_confidence + uncertain
        
        if total_markers == 0:
            return 0.5  # Neutral confidence
        
        # Weighted score: high=1.0, medium=0.7, low=0.3, uncertain=0.0
        weighted_score = (high_confidence * 1.0 + 
                         medium_confidence * 0.7 + 
                         low_confidence * 0.3 + 
                         uncertain * 0.0) / total_markers
        
        return float(np.clip(weighted_score, 0.0, 1.0))
    
    def _get_consistency_test_variations(self) -> Dict[str, List[str]]:
        """Get built-in consistency test variations"""
        return {
            "capital_of_france": [
                "What is the capital of France?",
                "Which city is the capital of France?",
                "France's capital is which city?",
                "What city serves as the capital of France?",
                "The capital city of France is what?"
            ],
            "earth_circumference": [
                "What is the circumference of Earth?",
                "How long is Earth's circumference?",
                "What is the distance around the Earth?",
                "How many kilometers is Earth's circumference?",
                "What is the perimeter of the Earth?"
            ],
            "speed_of_light": [
                "What is the speed of light?",
                "How fast does light travel?",
                "What is light's velocity in vacuum?",
                "At what speed does light move?",
                "How quickly does light travel through space?"
            ],
            "shakespeare_birth": [
                "When was William Shakespeare born?",
                "What year was Shakespeare born?",
                "In which year was William Shakespeare born?",
                "What is Shakespeare's birth year?",
                "When did William Shakespeare come into the world?"
            ]
        }
    
    def _calculate_knowledge_consistency(self, responses: List[Tuple[str, str]]) -> float:
        """Calculate knowledge consistency across response variations"""
        if len(responses) < 2:
            return 1.0
        
        # Extract responses
        response_texts = [response[1] for response in responses]
        
        # Use embedding similarity if available
        if self.embedding_model is not None:
            return self._calculate_embedding_consistency(response_texts)
        elif SKLEARN_AVAILABLE:
            return self._calculate_tfidf_consistency(response_texts)
        else:
            return self._calculate_lexical_consistency(response_texts)
    
    def _calculate_embedding_consistency(self, responses: List[str]) -> float:
        """Calculate consistency using embedding similarity"""
        try:
            embeddings = self.embedding_model.encode(responses)
            similarities = []
            
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    similarities.append(float(similarity))
            
            return float(np.mean(similarities)) if similarities else 1.0
            
        except Exception as e:
            logger.warning(f"Embedding consistency calculation failed: {e}")
            return self._calculate_lexical_consistency(responses)
    
    def _calculate_tfidf_consistency(self, responses: List[str]) -> float:
        """Calculate consistency using TF-IDF similarity"""
        try:
            vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(responses)
            
            similarities = []
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    similarity = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
                    similarities.append(float(similarity))
            
            return float(np.mean(similarities)) if similarities else 1.0
            
        except Exception as e:
            logger.warning(f"TF-IDF consistency calculation failed: {e}")
            return self._calculate_lexical_consistency(responses)
    
    def _calculate_lexical_consistency(self, responses: List[str]) -> float:
        """Calculate consistency using lexical overlap"""
        def jaccard_similarity(text1: str, text2: str) -> float:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0.0
        
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarity = jaccard_similarity(responses[i], responses[j])
                similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 1.0
    
    def _assess_consistency(self, score: float) -> str:
        """Provide qualitative assessment of consistency score"""
        if score >= 0.8:
            return "Highly consistent - Model provides very similar responses"
        elif score >= 0.6:
            return "Moderately consistent - Some variation in responses"
        elif score >= 0.4:
            return "Low consistency - Significant variation in responses"
        else:
            return "Inconsistent - Highly variable responses"
    
    def _generate_validation_report(self, all_results: List[ValidationResult],
                                  category_results: Dict[str, List[ValidationResult]]) -> KnowledgeValidationReport:
        """Generate comprehensive validation report"""
        if not all_results:
            return self._create_empty_report()
        
        # Overall accuracy
        total_tests = len(all_results)
        passed_tests = sum(1 for result in all_results if result.passed)
        overall_accuracy = passed_tests / max(1, total_tests)
        
        # Category breakdown
        category_breakdown = {}
        for category, results in category_results.items():
            category_passed = sum(1 for result in results if result.passed)
            category_accuracy = category_passed / max(1, len(results))
            category_breakdown[category] = category_accuracy
        
        # Confidence calibration analysis
        confidence_calibration = self._analyze_confidence_calibration(all_results)
        
        # Consistency analysis (placeholder - would need consistency test results)
        consistency_analysis = {"note": "Run test_knowledge_consistency() for detailed consistency analysis"}
        
        # Failure analysis
        failure_analysis = self._analyze_failures(all_results)
        
        return KnowledgeValidationReport(
            overall_accuracy=overall_accuracy,
            category_breakdown=category_breakdown,
            confidence_calibration=confidence_calibration,
            consistency_analysis=consistency_analysis,
            failure_analysis=failure_analysis,
            detailed_results=all_results
        )
    
    def _analyze_confidence_calibration(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Analyze confidence calibration across results"""
        if not results:
            return {}
        
        # Group results by confidence bins
        bins = {"high": [], "medium": [], "low": []}
        
        for result in results:
            if result.confidence_score >= 0.8:
                bins["high"].append(result)
            elif result.confidence_score >= 0.5:
                bins["medium"].append(result)
            else:
                bins["low"].append(result)
        
        # Calculate accuracy within each confidence bin
        calibration = {}
        for bin_name, bin_results in bins.items():
            if bin_results:
                bin_accuracy = sum(1 for r in bin_results if r.passed) / len(bin_results)
                calibration[bin_name] = {
                    "count": len(bin_results),
                    "accuracy": bin_accuracy,
                    "average_confidence": np.mean([r.confidence_score for r in bin_results])
                }
            else:
                calibration[bin_name] = {"count": 0, "accuracy": 0.0, "average_confidence": 0.0}
        
        # Calculate calibration score (ideally, high confidence should correlate with high accuracy)
        high_conf_acc = calibration["high"]["accuracy"]
        low_conf_acc = calibration["low"]["accuracy"]
        calibration_score = high_conf_acc - low_conf_acc if calibration["high"]["count"] > 0 and calibration["low"]["count"] > 0 else 0.0
        
        return {
            "calibration_by_confidence": calibration,
            "calibration_score": calibration_score,
            "assessment": self._assess_calibration(calibration_score)
        }
    
    def _assess_calibration(self, score: float) -> str:
        """Assess confidence calibration quality"""
        if score >= 0.3:
            return "Well-calibrated - High confidence correlates with high accuracy"
        elif score >= 0.1:
            return "Moderately calibrated - Some correlation between confidence and accuracy"
        elif score >= -0.1:
            return "Poorly calibrated - Little correlation between confidence and accuracy"
        else:
            return "Miscalibrated - High confidence associated with low accuracy"
    
    def _analyze_failures(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Analyze failure patterns"""
        failed_results = [r for r in results if not r.passed]
        
        if not failed_results:
            return {"total_failures": 0, "failure_rate": 0.0}
        
        # Failure by category
        failure_by_category = defaultdict(int)
        for result in failed_results:
            failure_by_category[result.category] += 1
        
        # Common failure patterns
        forbidden_token_failures = sum(1 for r in failed_results if r.forbidden_tokens_found)
        missing_token_failures = sum(1 for r in failed_results if not r.expected_tokens_found)
        low_confidence_failures = sum(1 for r in failed_results if r.confidence_score < 0.3)
        
        return {
            "total_failures": len(failed_results),
            "failure_rate": len(failed_results) / len(results),
            "failure_by_category": dict(failure_by_category),
            "failure_patterns": {
                "forbidden_tokens": forbidden_token_failures,
                "missing_expected_tokens": missing_token_failures,
                "low_confidence": low_confidence_failures
            }
        }
    
    def _create_empty_report(self) -> KnowledgeValidationReport:
        """Create empty report for error cases"""
        return KnowledgeValidationReport(
            overall_accuracy=0.0,
            category_breakdown={},
            confidence_calibration={},
            consistency_analysis={},
            failure_analysis={},
            detailed_results=[]
        )
    
    def generate_knowledge_assessment(self, validation_report: KnowledgeValidationReport,
                                    consistency_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive knowledge assessment"""
        assessment = {
            "factual_knowledge_assessment": {
                "overall_accuracy": validation_report.overall_accuracy,
                "assessment": self._assess_factual_knowledge(validation_report.overall_accuracy),
                "strengths": [],
                "weaknesses": []
            },
            "category_analysis": validation_report.category_breakdown,
            "confidence_reliability": validation_report.confidence_calibration,
            "recommendations": []
        }
        
        # Identify strengths and weaknesses
        for category, accuracy in validation_report.category_breakdown.items():
            if accuracy >= 0.8:
                assessment["factual_knowledge_assessment"]["strengths"].append(category)
            elif accuracy < 0.5:
                assessment["factual_knowledge_assessment"]["weaknesses"].append(category)
        
        # Add consistency analysis if available
        if consistency_results:
            consistency_scores = [result['consistency_score'] for result in consistency_results.values()]
            avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
            assessment["consistency_assessment"] = {
                "average_consistency": avg_consistency,
                "assessment": self._assess_consistency(avg_consistency),
                "detailed_results": consistency_results
            }
        
        # Generate recommendations
        assessment["recommendations"] = self._generate_recommendations(validation_report, consistency_results)
        
        return assessment
    
    def _assess_factual_knowledge(self, accuracy: float) -> str:
        """Assess overall factual knowledge quality"""
        if accuracy >= self.validation_thresholds['high_accuracy']:
            return "Excellent factual knowledge - Highly reliable for factual questions"
        elif accuracy >= self.validation_thresholds['good_accuracy']:
            return "Good factual knowledge - Generally reliable with occasional errors"
        elif accuracy >= self.validation_thresholds['moderate_accuracy']:
            return "Moderate factual knowledge - Some reliability issues, verification recommended"
        else:
            return "Poor factual knowledge - Significant reliability issues, not recommended for factual queries"
    
    def _generate_recommendations(self, validation_report: KnowledgeValidationReport,
                                consistency_results: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        # Accuracy-based recommendations
        if validation_report.overall_accuracy < 0.6:
            recommendations.append("Model shows poor factual accuracy - consider additional fact-checking")
        
        # Category-based recommendations
        weak_categories = [cat for cat, acc in validation_report.category_breakdown.items() if acc < 0.5]
        if weak_categories:
            recommendations.append(f"Model struggles with: {', '.join(weak_categories)} - avoid using for these domains")
        
        # Confidence calibration recommendations
        calibration_score = validation_report.confidence_calibration.get("calibration_score", 0)
        if calibration_score < 0:
            recommendations.append("Model shows poor confidence calibration - high confidence responses may be unreliable")
        
        # Consistency recommendations
        if consistency_results:
            low_consistency_topics = [topic for topic, result in consistency_results.items() 
                                    if result['consistency_score'] < 0.5]
            if low_consistency_topics:
                recommendations.append(f"Model shows inconsistent responses for: {', '.join(low_consistency_topics)}")
        
        return recommendations