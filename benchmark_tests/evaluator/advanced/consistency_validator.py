"""
Consistency Validator Module

Cross-phrasing consistency testing and semantic equivalence validation for language models.
Addresses the critique's key insight about missing consistency testing across different
question phrasings and response reliability validation.

This module implements:
- Semantic question equivalence detection
- Cross-phrasing consistency scoring  
- Response reliability measurement
- Built-in test question variations

"""

import re
import math
import logging
import itertools
from typing import Dict, List, Optional, Tuple, Union, Any
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
    logging.warning("sentence-transformers not available. Advanced semantic analysis disabled.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available. TF-IDF semantic analysis disabled.")

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ConsistencyTestResult:
    """Results from consistency validation testing"""
    question_group: str
    phrasings: List[str]
    responses: List[str]
    consistency_score: float
    semantic_equivalence_score: float
    response_variance: float
    cluster_analysis: Dict[str, Any]
    reliability_assessment: Dict[str, Any]


@dataclass
class CrossPhrasingResult:
    """Results from cross-phrasing consistency analysis"""
    overall_consistency_score: float
    semantic_equivalence_scores: List[float]
    response_clustering: Dict[str, Any]
    consistency_by_question_type: Dict[str, float]
    reliability_metrics: Dict[str, Any]
    failure_patterns: List[Dict[str, Any]]


class ConsistencyValidator:
    """
    Cross-phrasing consistency validator for language model responses.
    
    Evaluates how consistently a model responds to semantically equivalent 
    questions phrased differently, providing insights into response reliability
    and semantic understanding consistency.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize consistency validator
        
        Args:
            embedding_model: Sentence transformer model for semantic analysis
        """
        self.embedding_model_name = embedding_model
        self._embedding_model = None
        
        # Load built-in test question sets
        self.test_question_sets = self._initialize_test_questions()
        
        # Initialize analysis thresholds
        self.consistency_thresholds = {
            'high_consistency': 0.8,
            'moderate_consistency': 0.6,
            'low_consistency': 0.4,
            'inconsistent': 0.0
        }
        
        logger.info("ConsistencyValidator initialized")
    
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
    
    def _initialize_test_questions(self) -> Dict[str, List[str]]:
        """Initialize built-in test question variations"""
        return {
            "basic_math": [
                "What is 25 + 37?",
                "Calculate the sum of twenty-five and thirty-seven",
                "Add 25 to 37",
                "25 plus 37 equals what?",
                "Find the result of 25 + 37"
            ],
            "factual_knowledge": [
                "What is the capital of France?",
                "Which city is the capital of France?",
                "France's capital city is what?",
                "What city serves as France's capital?",
                "The capital of France is which city?"
            ],
            "logical_reasoning": [
                "If all cats are mammals and all mammals are animals, are all cats animals?",
                "Given that cats are mammals and mammals are animals, does it follow that cats are animals?",
                "All cats → mammals, all mammals → animals. Therefore, all cats → ?",
                "Cats are mammals. Mammals are animals. What can we conclude about cats?",
                "Using logic: cats are mammals, mammals are animals, so cats are what?"
            ],
            "definitional": [
                "What is photosynthesis?",
                "Define photosynthesis",
                "Explain what photosynthesis means",
                "How would you describe photosynthesis?",
                "What does the term photosynthesis refer to?"
            ],
            "comparative": [
                "What is the difference between a virus and bacteria?",
                "How do viruses differ from bacteria?",
                "Compare viruses and bacteria",
                "What distinguishes viruses from bacteria?",
                "Contrast viruses with bacteria"
            ],
            "causal": [
                "Why do seasons change?",
                "What causes seasonal changes?",
                "What makes the seasons change?",
                "What is the reason for changing seasons?",
                "How do seasons change and why?"
            ]
        }
    
    def analyze_cross_phrasing_consistency(self, question_response_pairs: List[Tuple[str, str]], 
                                         question_group: Optional[str] = None) -> CrossPhrasingResult:
        """
        Analyze consistency across different phrasings of questions
        
        Args:
            question_response_pairs: List of (question, response) tuples
            question_group: Optional group identifier for questions
            
        Returns:
            CrossPhrasingResult with comprehensive consistency analysis
        """
        if len(question_response_pairs) < 2:
            logger.warning("Need at least 2 question-response pairs for consistency analysis")
            return self._create_empty_result()
        
        questions = [pair[0] for pair in question_response_pairs]
        responses = [pair[1] for pair in question_response_pairs]
        
        # Calculate semantic equivalence of questions
        question_equivalence = self._calculate_semantic_equivalence(questions)
        
        # Calculate response consistency
        response_consistency = self._calculate_response_consistency(responses)
        
        # Perform response clustering analysis
        cluster_analysis = self._analyze_response_clustering(responses)
        
        # Calculate reliability metrics
        reliability_metrics = self._calculate_reliability_metrics(responses)
        
        # Detect failure patterns
        failure_patterns = self._detect_consistency_failures(questions, responses)
        
        # Calculate overall consistency score
        overall_score = self._calculate_overall_consistency_score(
            question_equivalence, response_consistency, cluster_analysis
        )
        
        return CrossPhrasingResult(
            overall_consistency_score=overall_score,
            semantic_equivalence_scores=question_equivalence,
            response_clustering=cluster_analysis,
            consistency_by_question_type={question_group or "unknown": overall_score},
            reliability_metrics=reliability_metrics,
            failure_patterns=failure_patterns
        )
    
    def test_built_in_consistency(self, model_evaluator_func: callable) -> Dict[str, CrossPhrasingResult]:
        """
        Test consistency using built-in question sets
        
        Args:
            model_evaluator_func: Function that takes a question and returns a response
            
        Returns:
            Dictionary mapping question groups to consistency results
        """
        results = {}
        
        for group_name, questions in self.test_question_sets.items():
            logger.info(f"Testing consistency for {group_name} question group")
            
            # Get responses for all question variations
            question_response_pairs = []
            for question in questions:
                try:
                    response = model_evaluator_func(question)
                    question_response_pairs.append((question, response))
                except Exception as e:
                    logger.warning(f"Failed to get response for question '{question}': {e}")
                    continue
            
            if len(question_response_pairs) >= 2:
                result = self.analyze_cross_phrasing_consistency(
                    question_response_pairs, question_group=group_name
                )
                results[group_name] = result
            else:
                logger.warning(f"Insufficient responses for {group_name} group")
        
        return results
    
    def _calculate_semantic_equivalence(self, questions: List[str]) -> List[float]:
        """Calculate semantic equivalence scores between questions"""
        if not questions or len(questions) < 2:
            return []
        
        if self.embedding_model is not None:
            return self._calculate_embedding_similarity(questions)
        elif SKLEARN_AVAILABLE:
            return self._calculate_tfidf_similarity(questions)
        else:
            return self._calculate_lexical_similarity(questions)
    
    def _calculate_embedding_similarity(self, texts: List[str]) -> List[float]:
        """Calculate similarity using sentence embeddings"""
        try:
            embeddings = self.embedding_model.encode(texts)
            similarities = []
            
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    # Clamp to [0, 1] to handle floating-point precision errors
                    clamped_similarity = max(0.0, min(1.0, float(similarity)))
                    similarities.append(clamped_similarity)
            
            return similarities
        except Exception as e:
            logger.warning(f"Embedding similarity calculation failed: {e}")
            return self._calculate_lexical_similarity(texts)
    
    def _calculate_tfidf_similarity(self, texts: List[str]) -> List[float]:
        """Calculate similarity using TF-IDF vectors"""
        try:
            vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            similarities = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    similarity = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
                    # Clamp to [0, 1] to handle floating-point precision errors
                    clamped_similarity = max(0.0, min(1.0, float(similarity)))
                    similarities.append(clamped_similarity)
            
            return similarities
        except Exception as e:
            logger.warning(f"TF-IDF similarity calculation failed: {e}")
            return self._calculate_lexical_similarity(texts)
    
    def _calculate_lexical_similarity(self, texts: List[str]) -> List[float]:
        """Calculate similarity using lexical overlap (fallback method)"""
        def jaccard_similarity(text1: str, text2: str) -> float:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0.0
        
        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = jaccard_similarity(texts[i], texts[j])
                similarities.append(similarity)
        
        return similarities
    
    def _calculate_response_consistency(self, responses: List[str]) -> float:
        """Calculate consistency score for responses"""
        if len(responses) < 2:
            return 1.0
        
        # Use embedding similarity for responses if available
        if self.embedding_model is not None:
            similarities = self._calculate_embedding_similarity(responses)
        elif SKLEARN_AVAILABLE:
            similarities = self._calculate_tfidf_similarity(responses)
        else:
            similarities = self._calculate_lexical_similarity(responses)
        
        # Return average similarity as consistency score
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _analyze_response_clustering(self, responses: List[str]) -> Dict[str, Any]:
        """Analyze response clustering to identify consistency patterns"""
        if len(responses) < 2:
            return {"clusters": 1, "cluster_sizes": [1], "silhouette_score": 1.0}
        
        try:
            if self.embedding_model is not None:
                embeddings = self.embedding_model.encode(responses)
                
                # Perform hierarchical clustering
                clustering = AgglomerativeClustering(
                    n_clusters=min(3, len(responses)), 
                    linkage='ward'
                )
                cluster_labels = clustering.fit_predict(embeddings)
                
                # Calculate cluster statistics
                unique_clusters = len(set(cluster_labels))
                cluster_sizes = [list(cluster_labels).count(i) for i in range(unique_clusters)]
                
                # Calculate silhouette score (approximation)
                silhouette_score = self._calculate_silhouette_approximation(embeddings, cluster_labels)
                
                return {
                    "clusters": unique_clusters,
                    "cluster_sizes": cluster_sizes,
                    "silhouette_score": silhouette_score,
                    "cluster_labels": cluster_labels.tolist()
                }
            else:
                # Fallback: simple lexical clustering
                return self._simple_response_clustering(responses)
                
        except Exception as e:
            logger.warning(f"Response clustering failed: {e}")
            return {"clusters": len(responses), "cluster_sizes": [1] * len(responses), "silhouette_score": 0.0}
    
    def _simple_response_clustering(self, responses: List[str]) -> Dict[str, Any]:
        """Simple response clustering based on lexical similarity"""
        # Group responses by high lexical similarity
        clusters = []
        used = set()
        
        for i, response1 in enumerate(responses):
            if i in used:
                continue
            
            cluster = [i]
            used.add(i)
            
            for j, response2 in enumerate(responses[i+1:], i+1):
                if j in used:
                    continue
                
                similarity = self._calculate_lexical_similarity([response1, response2])[0]
                if similarity > 0.6:  # High similarity threshold
                    cluster.append(j)
                    used.add(j)
            
            clusters.append(cluster)
        
        return {
            "clusters": len(clusters),
            "cluster_sizes": [len(cluster) for cluster in clusters],
            "silhouette_score": 1.0 - (len(clusters) - 1) / max(1, len(responses) - 1)
        }
    
    def _calculate_silhouette_approximation(self, embeddings: np.ndarray, 
                                         cluster_labels: np.ndarray) -> float:
        """Calculate approximation of silhouette score"""
        if len(set(cluster_labels)) <= 1:
            return 1.0
        
        try:
            from sklearn.metrics import silhouette_score
            return float(silhouette_score(embeddings, cluster_labels))
        except ImportError:
            # Fallback approximation
            unique_clusters = len(set(cluster_labels))
            return 1.0 - (unique_clusters - 1) / max(1, len(embeddings) - 1)
    
    def _calculate_reliability_metrics(self, responses: List[str]) -> Dict[str, Any]:
        """Calculate reliability metrics for responses"""
        if not responses:
            return {}
        
        # Length variance
        lengths = [len(response.split()) for response in responses]
        length_variance = float(np.var(lengths)) if len(lengths) > 1 else 0.0
        
        # Response diversity (unique n-grams)
        all_tokens = []
        for response in responses:
            all_tokens.extend(response.lower().split())
        
        total_tokens = len(all_tokens)
        unique_tokens = len(set(all_tokens))
        diversity_score = unique_tokens / max(1, total_tokens)
        
        # Confidence markers analysis
        confidence_analysis = self._analyze_confidence_markers(responses)
        
        # Repetition analysis
        repetition_analysis = self._analyze_repetition_patterns(responses)
        
        return {
            "length_variance": length_variance,
            "average_length": float(np.mean(lengths)),
            "diversity_score": diversity_score,
            "confidence_analysis": confidence_analysis,
            "repetition_analysis": repetition_analysis
        }
    
    def _analyze_confidence_markers(self, responses: List[str]) -> Dict[str, Any]:
        """Analyze confidence markers in responses"""
        high_confidence = ["certainly", "definitely", "clearly", "obviously", "undoubtedly"]
        low_confidence = ["perhaps", "maybe", "possibly", "might", "could be", "I think"]
        uncertainty = ["unsure", "not sure", "don't know", "unclear", "ambiguous"]
        
        confidence_scores = []
        for response in responses:
            response_lower = response.lower()
            
            high_count = sum(1 for marker in high_confidence if marker in response_lower)
            low_count = sum(1 for marker in low_confidence if marker in response_lower)
            uncertain_count = sum(1 for marker in uncertainty if marker in response_lower)
            
            # Calculate confidence score (simple heuristic)
            confidence_score = (high_count - low_count - uncertain_count * 2) / max(1, len(response.split()) / 10)
            confidence_scores.append(confidence_score)
        
        return {
            "confidence_scores": confidence_scores,
            "average_confidence": float(np.mean(confidence_scores)),
            "confidence_variance": float(np.var(confidence_scores)),
            "high_confidence_responses": sum(1 for score in confidence_scores if score > 0.5),
            "uncertain_responses": sum(1 for score in confidence_scores if score < -0.5)
        }
    
    def _analyze_repetition_patterns(self, responses: List[str]) -> Dict[str, Any]:
        """Analyze repetition patterns in responses"""
        repetition_scores = []
        
        for response in responses:
            words = response.lower().split()
            if len(words) <= 1:
                repetition_scores.append(0.0)
                continue
            
            word_counts = Counter(words)
            total_words = len(words)
            repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
            
            repetition_score = repeated_words / max(1, total_words)
            repetition_scores.append(repetition_score)
        
        return {
            "repetition_scores": repetition_scores,
            "average_repetition": float(np.mean(repetition_scores)),
            "max_repetition": float(max(repetition_scores)) if repetition_scores else 0.0,
            "high_repetition_responses": sum(1 for score in repetition_scores if score > 0.3)
        }
    
    def _detect_consistency_failures(self, questions: List[str], 
                                   responses: List[str]) -> List[Dict[str, Any]]:
        """Detect specific consistency failure patterns"""
        failures = []
        
        if len(responses) < 2:
            return failures
        
        # Check for contradictory responses
        for i, response1 in enumerate(responses):
            for j, response2 in enumerate(responses[i+1:], i+1):
                contradiction_score = self._detect_contradiction(response1, response2)
                if contradiction_score > 0.7:
                    failures.append({
                        "type": "contradiction",
                        "question1": questions[i],
                        "question2": questions[j],
                        "response1": response1,
                        "response2": response2,
                        "severity": contradiction_score
                    })
        
        # Check for incomplete responses
        for i, response in enumerate(responses):
            if self._is_incomplete_response(response):
                failures.append({
                    "type": "incomplete_response",
                    "question": questions[i],
                    "response": response,
                    "severity": 0.8
                })
        
        # Check for off-topic responses
        for i, (question, response) in enumerate(zip(questions, responses)):
            if self._is_off_topic(question, response):
                failures.append({
                    "type": "off_topic",
                    "question": question,
                    "response": response,
                    "severity": 0.9
                })
        
        return failures
    
    def _detect_contradiction(self, response1: str, response2: str) -> float:
        """Detect contradiction between two responses"""
        # Simple heuristic: look for negation patterns
        negation_indicators = [
            ("yes", "no"), ("true", "false"), ("correct", "incorrect"),
            ("is", "is not"), ("will", "will not"), ("can", "cannot")
        ]
        
        response1_lower = response1.lower()
        response2_lower = response2.lower()
        
        contradiction_score = 0.0
        
        for pos, neg in negation_indicators:
            if pos in response1_lower and neg in response2_lower:
                contradiction_score += 0.3
            elif neg in response1_lower and pos in response2_lower:
                contradiction_score += 0.3
        
        return min(contradiction_score, 1.0)
    
    def _is_incomplete_response(self, response: str) -> bool:
        """Check if response appears incomplete"""
        response = response.strip()
        
        # Too short
        if len(response.split()) < 3:
            return True
        
        # Ends abruptly
        incomplete_endings = ["...", "and", "or", "but", "because", "so", "then"]
        if any(response.lower().endswith(ending) for ending in incomplete_endings):
            return True
        
        # Contains incomplete markers
        incomplete_markers = ["[incomplete]", "[truncated]", "...", "etc.", "and so on"]
        if any(marker in response.lower() for marker in incomplete_markers):
            return True
        
        return False
    
    def _is_off_topic(self, question: str, response: str) -> bool:
        """Check if response is off-topic for the question"""
        # Simple heuristic: check if response shares any content words with question
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        response_words = set(re.findall(r'\b\w+\b', response.lower()))
        
        # Remove stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
                      "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", 
                      "has", "had", "do", "does", "did", "will", "would", "could", "should"}
        
        question_words -= stop_words
        response_words -= stop_words
        
        if not question_words or not response_words:
            return False
        
        overlap = len(question_words.intersection(response_words))
        overlap_ratio = overlap / len(question_words)
        
        return overlap_ratio < 0.1  # Very low overlap suggests off-topic
    
    def _calculate_overall_consistency_score(self, question_equivalence: List[float], 
                                           response_consistency: float, 
                                           cluster_analysis: Dict[str, Any]) -> float:
        """Calculate overall consistency score"""
        if not question_equivalence:
            return response_consistency
        
        # Weight components
        question_eq_score = float(np.mean(question_equivalence))
        response_cons_score = response_consistency
        cluster_score = cluster_analysis.get("silhouette_score", 0.0)
        
        # Weighted average
        weights = [0.3, 0.5, 0.2]  # question equivalence, response consistency, clustering
        overall_score = (weights[0] * question_eq_score + 
                        weights[1] * response_cons_score + 
                        weights[2] * cluster_score)
        
        return float(np.clip(overall_score, 0.0, 1.0))
    
    def _create_empty_result(self) -> CrossPhrasingResult:
        """Create empty result for error cases"""
        return CrossPhrasingResult(
            overall_consistency_score=0.0,
            semantic_equivalence_scores=[],
            response_clustering={},
            consistency_by_question_type={},
            reliability_metrics={},
            failure_patterns=[]
        )
    
    def generate_consistency_report(self, results: Dict[str, CrossPhrasingResult]) -> Dict[str, Any]:
        """Generate comprehensive consistency analysis report"""
        if not results:
            return {"error": "No consistency results to analyze"}
        
        # Aggregate statistics
        all_scores = [result.overall_consistency_score for result in results.values()]
        avg_consistency = float(np.mean(all_scores))
        consistency_variance = float(np.var(all_scores))
        
        # Categorize question types by consistency
        high_consistency = [name for name, result in results.items() 
                           if result.overall_consistency_score >= self.consistency_thresholds['high_consistency']]
        
        moderate_consistency = [name for name, result in results.items() 
                               if self.consistency_thresholds['moderate_consistency'] <= 
                               result.overall_consistency_score < self.consistency_thresholds['high_consistency']]
        
        low_consistency = [name for name, result in results.items() 
                          if self.consistency_thresholds['low_consistency'] <= 
                          result.overall_consistency_score < self.consistency_thresholds['moderate_consistency']]
        
        inconsistent = [name for name, result in results.items() 
                       if result.overall_consistency_score < self.consistency_thresholds['low_consistency']]
        
        # Aggregate failure patterns
        all_failures = []
        for result in results.values():
            all_failures.extend(result.failure_patterns)
        
        failure_by_type = defaultdict(int)
        for failure in all_failures:
            failure_by_type[failure.get("type", "unknown")] += 1
        
        return {
            "summary": {
                "total_question_groups": len(results),
                "average_consistency_score": avg_consistency,
                "consistency_variance": consistency_variance,
                "overall_assessment": self._assess_overall_consistency(avg_consistency)
            },
            "consistency_breakdown": {
                "high_consistency": high_consistency,
                "moderate_consistency": moderate_consistency,
                "low_consistency": low_consistency,
                "inconsistent": inconsistent
            },
            "failure_analysis": {
                "total_failures": len(all_failures),
                "failure_by_type": dict(failure_by_type),
                "most_common_failure": max(failure_by_type.items(), key=lambda x: x[1])[0] if failure_by_type else None
            },
            "detailed_results": results
        }
    
    def _assess_overall_consistency(self, score: float) -> str:
        """Provide qualitative assessment of consistency score"""
        if score >= self.consistency_thresholds['high_consistency']:
            return "Highly Consistent - Model provides very similar responses to equivalent questions"
        elif score >= self.consistency_thresholds['moderate_consistency']:
            return "Moderately Consistent - Model shows good consistency with some variation"
        elif score >= self.consistency_thresholds['low_consistency']:
            return "Low Consistency - Model responses vary significantly across equivalent questions"
        else:
            return "Inconsistent - Model provides highly variable responses to equivalent questions"