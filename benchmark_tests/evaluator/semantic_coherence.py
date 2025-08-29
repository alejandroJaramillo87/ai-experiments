"""
Semantic Coherence Analysis Module

Advanced semantic coherence analysis for language model evaluation including
prompt-completion coherence, semantic drift detection, and topic consistency measurement.

This module addresses the critique's key point about missing semantic coherence
measurements beyond simple pattern matching.

Author: Claude Code
Version: 1.0.0
"""

import re
import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import Counter, defaultdict
import numpy as np

# Optional imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Using fallback methods.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available. Some semantic features disabled.")

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("nltk not available. Using fallback text processing.")

# Set up logging
logger = logging.getLogger(__name__)


class SemanticCoherenceAnalyzer:
    """
    Advanced semantic coherence analyzer for language model responses.
    
    Provides comprehensive semantic coherence measurements:
    - Prompt-completion semantic bridge analysis
    - Semantic drift detection via sliding windows
    - Topic consistency scoring
    - Semantic flow analysis
    - Cross-sentence coherence measurement
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic coherence analyzer
        
        Args:
            embedding_model: Sentence transformer model for semantic analysis
        """
        self.embedding_model_name = embedding_model
        self._embedding_model = None
        self._stop_words = self._initialize_stop_words()
        
        logger.info(f"SemanticCoherenceAnalyzer initialized with {embedding_model}")
    
    @property
    def embedding_model(self):
        """Lazy load embedding model"""
        if self._embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
                logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model {self.embedding_model_name}: {e}")
        return self._embedding_model
    
    def _initialize_stop_words(self) -> set:
        """Initialize stop words set"""
        if NLTK_AVAILABLE:
            try:
                nltk.download('stopwords', quiet=True)
                return set(stopwords.words('english'))
            except Exception:
                pass
        
        # Fallback stop words
        return {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once'
        }
    
    def calculate_prompt_completion_coherence(self, prompt: str, completion: str) -> Dict[str, float]:
        """
        Calculate semantic coherence between prompt ending and completion beginning
        
        Args:
            prompt: Input prompt text
            completion: Model completion text
            
        Returns:
            Dictionary with coherence metrics
        """
        if not prompt.strip() or not completion.strip():
            return {"coherence_score": 0.0, "semantic_bridge": 0.0, "topic_alignment": 0.0}
        
        try:
            # Extract key segments
            prompt_ending = self._extract_prompt_ending(prompt)
            completion_beginning = self._extract_completion_beginning(completion)
            
            if self.embedding_model:
                return self._calculate_embedding_coherence(prompt_ending, completion_beginning, prompt, completion)
            else:
                return self._calculate_tfidf_coherence(prompt_ending, completion_beginning, prompt, completion)
                
        except Exception as e:
            logger.error(f"Prompt-completion coherence calculation failed: {e}")
            return {"coherence_score": 0.0, "semantic_bridge": 0.0, "topic_alignment": 0.0}
    
    def measure_semantic_drift(self, text: str, window_size: int = 100, step_size: int = 50) -> Dict[str, Any]:
        """
        Measure semantic drift using sliding window analysis
        
        Args:
            text: Text to analyze
            window_size: Size of semantic windows in tokens
            step_size: Step size between windows
            
        Returns:
            Dictionary with semantic drift metrics
        """
        if not text.strip():
            return {"drift_score": 0.0, "drift_points": [], "stability_score": 1.0, "drift_curve": []}
        
        try:
            # Tokenize text into words
            words = self._tokenize_text(text)
            if len(words) < window_size * 2:
                return {"drift_score": 0.0, "drift_points": [], "stability_score": 1.0, "drift_curve": []}
            
            # Create sliding windows
            windows = []
            window_texts = []
            for i in range(0, len(words) - window_size + 1, step_size):
                window = words[i:i + window_size]
                window_text = " ".join(window)
                windows.append(window)
                window_texts.append(window_text)
            
            if len(window_texts) < 2:
                return {"drift_score": 0.0, "drift_points": [], "stability_score": 1.0, "drift_curve": []}
            
            if self.embedding_model:
                return self._measure_embedding_drift(window_texts)
            else:
                return self._measure_tfidf_drift(window_texts)
                
        except Exception as e:
            logger.error(f"Semantic drift measurement failed: {e}")
            return {"drift_score": 0.0, "drift_points": [], "stability_score": 1.0, "drift_curve": []}
    
    def calculate_topic_consistency(self, text: str, num_topics: int = 5) -> Dict[str, Any]:
        """
        Calculate topic consistency using topic modeling
        
        Args:
            text: Text to analyze
            num_topics: Number of topics for LDA modeling
            
        Returns:
            Dictionary with topic consistency metrics
        """
        if not text.strip():
            return {"consistency_score": 0.0, "topic_distribution": [], "dominant_topic_ratio": 0.0}
        
        try:
            # Split text into segments for topic analysis
            segments = self._split_text_into_segments(text)
            if len(segments) < 3:
                return {"consistency_score": 1.0, "topic_distribution": [1.0], "dominant_topic_ratio": 1.0}
            
            if SKLEARN_AVAILABLE:
                return self._calculate_lda_topic_consistency(segments, num_topics)
            else:
                return self._calculate_simple_topic_consistency(segments)
                
        except Exception as e:
            logger.error(f"Topic consistency calculation failed: {e}")
            return {"consistency_score": 0.0, "topic_distribution": [], "dominant_topic_ratio": 0.0}
    
    def analyze_semantic_flow(self, text: str) -> Dict[str, Any]:
        """
        Analyze semantic flow across the entire response
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with semantic flow metrics
        """
        if not text.strip():
            return {"flow_score": 0.0, "transition_quality": 0.0, "narrative_coherence": 0.0}
        
        try:
            sentences = self._split_into_sentences(text)
            if len(sentences) < 2:
                return {"flow_score": 1.0, "transition_quality": 1.0, "narrative_coherence": 1.0}
            
            # Analyze sentence-to-sentence transitions
            transition_scores = self._calculate_transition_scores(sentences)
            
            # Calculate overall flow metrics
            flow_score = np.mean(transition_scores) if transition_scores else 0.0
            transition_quality = self._assess_transition_quality(transition_scores)
            narrative_coherence = self._assess_narrative_coherence(sentences, transition_scores)
            
            return {
                "flow_score": float(flow_score),
                "transition_quality": float(transition_quality),
                "narrative_coherence": float(narrative_coherence),
                "transition_scores": [float(score) for score in transition_scores],
                "sentence_count": len(sentences)
            }
            
        except Exception as e:
            logger.error(f"Semantic flow analysis failed: {e}")
            return {"flow_score": 0.0, "transition_quality": 0.0, "narrative_coherence": 0.0}
    
    def comprehensive_coherence_analysis(self, text: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive semantic coherence analysis
        
        Args:
            text: Text to analyze
            prompt: Optional prompt for prompt-completion analysis
            
        Returns:
            Complete coherence analysis dictionary
        """
        if not text.strip():
            return self._empty_coherence_analysis()
        
        try:
            # Core coherence metrics
            semantic_flow = self.analyze_semantic_flow(text)
            semantic_drift = self.measure_semantic_drift(text)
            topic_consistency = self.calculate_topic_consistency(text)
            
            # Prompt-completion coherence if prompt provided
            prompt_coherence = {}
            if prompt:
                prompt_coherence = self.calculate_prompt_completion_coherence(prompt, text)
            
            # Cross-sentence coherence
            cross_sentence_coherence = self._calculate_cross_sentence_coherence(text)
            
            # Overall coherence score
            overall_coherence = self._calculate_overall_coherence_score(
                semantic_flow, semantic_drift, topic_consistency, cross_sentence_coherence
            )
            
            return {
                "overall_coherence_score": overall_coherence,
                "semantic_flow": semantic_flow,
                "semantic_drift": semantic_drift,
                "topic_consistency": topic_consistency,
                "cross_sentence_coherence": cross_sentence_coherence,
                "prompt_completion_coherence": prompt_coherence,
                "text_length": len(text),
                "sentence_count": len(self._split_into_sentences(text))
            }
            
        except Exception as e:
            logger.error(f"Comprehensive coherence analysis failed: {e}")
            return self._empty_coherence_analysis()
    
    # Private helper methods
    
    def _extract_prompt_ending(self, prompt: str, num_sentences: int = 2) -> str:
        """Extract the ending of the prompt for coherence analysis"""
        sentences = self._split_into_sentences(prompt)
        if len(sentences) <= num_sentences:
            return prompt
        return " ".join(sentences[-num_sentences:])
    
    def _extract_completion_beginning(self, completion: str, num_sentences: int = 2) -> str:
        """Extract the beginning of the completion for coherence analysis"""
        sentences = self._split_into_sentences(completion)
        if len(sentences) <= num_sentences:
            return completion
        return " ".join(sentences[:num_sentences])
    
    def _calculate_embedding_coherence(self, prompt_ending: str, completion_beginning: str, 
                                     full_prompt: str, full_completion: str) -> Dict[str, float]:
        """Calculate coherence using embedding similarity"""
        try:
            # Calculate direct transition coherence
            texts = [prompt_ending, completion_beginning]
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
            
            transition_similarity = cosine_similarity(
                embeddings[0:1].cpu().numpy(), 
                embeddings[1:2].cpu().numpy()
            )[0, 0]
            
            # Calculate overall topic alignment
            full_texts = [full_prompt, full_completion]
            full_embeddings = self.embedding_model.encode(full_texts, convert_to_tensor=True)
            
            topic_alignment = cosine_similarity(
                full_embeddings[0:1].cpu().numpy(),
                full_embeddings[1:2].cpu().numpy()
            )[0, 0]
            
            # Calculate semantic bridge score (combination of transition and alignment)
            semantic_bridge = (transition_similarity * 0.7) + (topic_alignment * 0.3)
            
            return {
                "coherence_score": float(semantic_bridge),
                "semantic_bridge": float(transition_similarity),
                "topic_alignment": float(topic_alignment)
            }
            
        except Exception as e:
            logger.error(f"Embedding coherence calculation failed: {e}")
            return {"coherence_score": 0.0, "semantic_bridge": 0.0, "topic_alignment": 0.0}
    
    def _calculate_tfidf_coherence(self, prompt_ending: str, completion_beginning: str,
                                 full_prompt: str, full_completion: str) -> Dict[str, float]:
        """Fallback coherence calculation using TF-IDF"""
        if not SKLEARN_AVAILABLE:
            return {"coherence_score": 0.0, "semantic_bridge": 0.0, "topic_alignment": 0.0}
        
        try:
            # TF-IDF based transition coherence
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            transition_texts = [prompt_ending, completion_beginning]
            
            if len(" ".join(transition_texts).split()) < 5:
                return {"coherence_score": 0.0, "semantic_bridge": 0.0, "topic_alignment": 0.0}
            
            tfidf_matrix = vectorizer.fit_transform(transition_texts)
            transition_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0, 0]
            
            # Topic alignment using full texts
            full_texts = [full_prompt, full_completion]
            full_vectorizer = TfidfVectorizer(stop_words='english', max_features=200)
            full_tfidf = full_vectorizer.fit_transform(full_texts)
            topic_alignment = cosine_similarity(full_tfidf[0:1], full_tfidf[1:2])[0, 0]
            
            semantic_bridge = (transition_similarity * 0.7) + (topic_alignment * 0.3)
            
            return {
                "coherence_score": float(semantic_bridge),
                "semantic_bridge": float(transition_similarity),
                "topic_alignment": float(topic_alignment)
            }
            
        except Exception as e:
            logger.error(f"TF-IDF coherence calculation failed: {e}")
            return {"coherence_score": 0.0, "semantic_bridge": 0.0, "topic_alignment": 0.0}
    
    def _measure_embedding_drift(self, window_texts: List[str]) -> Dict[str, Any]:
        """Measure semantic drift using embeddings"""
        try:
            embeddings = self.embedding_model.encode(window_texts, convert_to_tensor=True)
            
            # Calculate pairwise similarities between consecutive windows
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity(
                    embeddings[i:i+1].cpu().numpy(),
                    embeddings[i+1:i+2].cpu().numpy()
                )[0, 0]
                similarities.append(sim)
            
            # Calculate drift metrics
            drift_curve = [1.0 - sim for sim in similarities]  # Convert to drift scores
            drift_score = np.mean(drift_curve)
            stability_score = 1.0 - np.std(similarities)
            
            # Identify significant drift points
            drift_threshold = np.mean(drift_curve) + np.std(drift_curve)
            drift_points = [i for i, drift in enumerate(drift_curve) if drift > drift_threshold]
            
            return {
                "drift_score": float(drift_score),
                "drift_points": drift_points,
                "stability_score": max(float(stability_score), 0.0),
                "drift_curve": [float(d) for d in drift_curve],
                "similarities": [float(s) for s in similarities]
            }
            
        except Exception as e:
            logger.error(f"Embedding drift measurement failed: {e}")
            return {"drift_score": 0.0, "drift_points": [], "stability_score": 1.0, "drift_curve": []}
    
    def _measure_tfidf_drift(self, window_texts: List[str]) -> Dict[str, Any]:
        """Fallback drift measurement using TF-IDF"""
        if not SKLEARN_AVAILABLE:
            return {"drift_score": 0.0, "drift_points": [], "stability_score": 1.0, "drift_curve": []}
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
            tfidf_matrix = vectorizer.fit_transform(window_texts)
            
            similarities = []
            for i in range(len(window_texts) - 1):
                sim = cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix[i+1:i+2])[0, 0]
                similarities.append(sim)
            
            drift_curve = [1.0 - sim for sim in similarities]
            drift_score = np.mean(drift_curve)
            stability_score = 1.0 - np.std(similarities)
            
            drift_threshold = np.mean(drift_curve) + np.std(drift_curve)
            drift_points = [i for i, drift in enumerate(drift_curve) if drift > drift_threshold]
            
            return {
                "drift_score": float(drift_score),
                "drift_points": drift_points,
                "stability_score": max(float(stability_score), 0.0),
                "drift_curve": [float(d) for d in drift_curve]
            }
            
        except Exception as e:
            logger.error(f"TF-IDF drift measurement failed: {e}")
            return {"drift_score": 0.0, "drift_points": [], "stability_score": 1.0, "drift_curve": []}
    
    def _calculate_lda_topic_consistency(self, segments: List[str], num_topics: int) -> Dict[str, Any]:
        """Calculate topic consistency using LDA topic modeling"""
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100, min_df=1)
            tfidf_matrix = vectorizer.fit_transform(segments)
            
            # Adjust number of topics based on available segments
            effective_topics = min(num_topics, len(segments) - 1, tfidf_matrix.shape[1])
            if effective_topics < 1:
                return {"consistency_score": 1.0, "topic_distribution": [1.0], "dominant_topic_ratio": 1.0}
            
            lda = LatentDirichletAllocation(n_components=effective_topics, random_state=42)
            lda.fit(tfidf_matrix)
            
            # Get topic distributions for each segment
            topic_distributions = lda.transform(tfidf_matrix)
            
            # Calculate consistency metrics
            dominant_topics = np.argmax(topic_distributions, axis=1)
            topic_counts = Counter(dominant_topics)
            
            # Dominant topic ratio
            most_common_topic_count = max(topic_counts.values())
            dominant_topic_ratio = most_common_topic_count / len(segments)
            
            # Consistency score based on topic distribution entropy
            topic_probs = np.array(list(topic_counts.values())) / len(segments)
            topic_entropy = -np.sum(topic_probs * np.log2(topic_probs + 1e-10))
            max_entropy = math.log2(len(topic_probs))
            consistency_score = 1.0 - (topic_entropy / max_entropy) if max_entropy > 0 else 1.0
            
            return {
                "consistency_score": float(consistency_score),
                "topic_distribution": [float(p) for p in topic_probs],
                "dominant_topic_ratio": float(dominant_topic_ratio),
                "topic_entropy": float(topic_entropy),
                "num_topics_found": len(topic_counts)
            }
            
        except Exception as e:
            logger.error(f"LDA topic consistency calculation failed: {e}")
            return {"consistency_score": 0.0, "topic_distribution": [], "dominant_topic_ratio": 0.0}
    
    def _calculate_simple_topic_consistency(self, segments: List[str]) -> Dict[str, Any]:
        """Simple topic consistency using keyword overlap"""
        try:
            # Extract key words from each segment
            segment_keywords = []
            for segment in segments:
                words = self._tokenize_text(segment)
                keywords = [word for word in words if len(word) > 3 and word not in self._stop_words]
                segment_keywords.append(set(keywords))
            
            if not segment_keywords:
                return {"consistency_score": 1.0, "topic_distribution": [1.0], "dominant_topic_ratio": 1.0}
            
            # Calculate pairwise keyword overlaps
            overlaps = []
            for i in range(len(segment_keywords)):
                for j in range(i + 1, len(segment_keywords)):
                    if len(segment_keywords[i]) == 0 or len(segment_keywords[j]) == 0:
                        overlap = 0.0
                    else:
                        intersection = len(segment_keywords[i] & segment_keywords[j])
                        union = len(segment_keywords[i] | segment_keywords[j])
                        overlap = intersection / union if union > 0 else 0.0
                    overlaps.append(overlap)
            
            consistency_score = np.mean(overlaps) if overlaps else 1.0
            
            return {
                "consistency_score": float(consistency_score),
                "topic_distribution": [consistency_score],
                "dominant_topic_ratio": float(consistency_score),
                "keyword_overlap": float(consistency_score)
            }
            
        except Exception as e:
            logger.error(f"Simple topic consistency calculation failed: {e}")
            return {"consistency_score": 0.0, "topic_distribution": [], "dominant_topic_ratio": 0.0}
    
    def _calculate_transition_scores(self, sentences: List[str]) -> List[float]:
        """Calculate semantic transition scores between consecutive sentences"""
        if len(sentences) < 2:
            return []
        
        transition_scores = []
        
        if self.embedding_model:
            try:
                embeddings = self.embedding_model.encode(sentences, convert_to_tensor=True)
                for i in range(len(embeddings) - 1):
                    sim = cosine_similarity(
                        embeddings[i:i+1].cpu().numpy(),
                        embeddings[i+1:i+2].cpu().numpy()
                    )[0, 0]
                    transition_scores.append(sim)
            except Exception:
                transition_scores = self._calculate_lexical_transitions(sentences)
        else:
            transition_scores = self._calculate_lexical_transitions(sentences)
        
        return transition_scores
    
    def _calculate_lexical_transitions(self, sentences: List[str]) -> List[float]:
        """Fallback lexical transition calculation"""
        transition_scores = []
        
        for i in range(len(sentences) - 1):
            words1 = set(self._tokenize_text(sentences[i]))
            words2 = set(self._tokenize_text(sentences[i + 1]))
            
            if len(words1) == 0 or len(words2) == 0:
                transition_scores.append(0.0)
            else:
                intersection = len(words1 & words2)
                union = len(words1 | words2)
                jaccard = intersection / union if union > 0 else 0.0
                transition_scores.append(jaccard)
        
        return transition_scores
    
    def _assess_transition_quality(self, transition_scores: List[float]) -> float:
        """Assess overall transition quality"""
        if not transition_scores:
            return 1.0
        
        # Good transitions have moderate similarity (not too high, not too low)
        ideal_range = (0.2, 0.7)
        quality_scores = []
        
        for score in transition_scores:
            if ideal_range[0] <= score <= ideal_range[1]:
                quality_scores.append(1.0)
            elif score < ideal_range[0]:
                # Too little connection
                quality_scores.append(score / ideal_range[0])
            else:
                # Too much similarity (repetitive)
                quality_scores.append(max(0.0, 1.0 - (score - ideal_range[1]) / (1.0 - ideal_range[1])))
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _assess_narrative_coherence(self, sentences: List[str], transition_scores: List[float]) -> float:
        """Assess overall narrative coherence"""
        if len(sentences) < 2:
            return 1.0
        
        # Combine transition consistency with overall flow
        transition_consistency = 1.0 - np.std(transition_scores) if transition_scores else 1.0
        average_transition = np.mean(transition_scores) if transition_scores else 0.5
        
        # Penalty for very short sentences (often indicates fragmentation)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        length_factor = min(1.0, avg_sentence_length / 10.0)  # Ideal around 10+ words
        
        narrative_coherence = (transition_consistency * 0.4 + 
                             average_transition * 0.4 + 
                             length_factor * 0.2)
        
        return max(0.0, min(1.0, narrative_coherence))
    
    def _calculate_cross_sentence_coherence(self, text: str) -> Dict[str, float]:
        """Calculate coherence across all sentence pairs"""
        sentences = self._split_into_sentences(text)
        if len(sentences) < 2:
            return {"average_coherence": 1.0, "coherence_variance": 0.0, "min_coherence": 1.0}
        
        all_similarities = []
        
        if self.embedding_model:
            try:
                embeddings = self.embedding_model.encode(sentences, convert_to_tensor=True)
                similarity_matrix = cosine_similarity(embeddings.cpu().numpy())
                
                # Extract upper triangle (avoiding diagonal)
                for i in range(len(sentences)):
                    for j in range(i + 1, len(sentences)):
                        all_similarities.append(similarity_matrix[i, j])
                        
            except Exception:
                all_similarities = self._calculate_all_lexical_similarities(sentences)
        else:
            all_similarities = self._calculate_all_lexical_similarities(sentences)
        
        if not all_similarities:
            return {"average_coherence": 1.0, "coherence_variance": 0.0, "min_coherence": 1.0}
        
        return {
            "average_coherence": float(np.mean(all_similarities)),
            "coherence_variance": float(np.var(all_similarities)),
            "min_coherence": float(np.min(all_similarities)),
            "max_coherence": float(np.max(all_similarities))
        }
    
    def _calculate_all_lexical_similarities(self, sentences: List[str]) -> List[float]:
        """Calculate lexical similarities between all sentence pairs"""
        similarities = []
        
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                words1 = set(self._tokenize_text(sentences[i]))
                words2 = set(self._tokenize_text(sentences[j]))
                
                if len(words1) == 0 or len(words2) == 0:
                    similarities.append(0.0)
                else:
                    intersection = len(words1 & words2)
                    union = len(words1 | words2)
                    jaccard = intersection / union if union > 0 else 0.0
                    similarities.append(jaccard)
        
        return similarities
    
    def _calculate_overall_coherence_score(self, semantic_flow: Dict, semantic_drift: Dict,
                                         topic_consistency: Dict, cross_sentence: Dict) -> float:
        """Calculate overall coherence score from component metrics"""
        try:
            # Weight different aspects of coherence
            flow_score = semantic_flow.get("flow_score", 0.0)
            drift_stability = semantic_drift.get("stability_score", 0.0)
            topic_score = topic_consistency.get("consistency_score", 0.0)
            cross_coherence = cross_sentence.get("average_coherence", 0.0)
            
            # Weighted combination
            overall_score = (
                flow_score * 0.3 +
                drift_stability * 0.3 +
                topic_score * 0.2 +
                cross_coherence * 0.2
            )
            
            return max(0.0, min(1.0, overall_score))
            
        except Exception as e:
            logger.error(f"Overall coherence calculation failed: {e}")
            return 0.0
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if NLTK_AVAILABLE:
            try:
                return [word.lower() for word in word_tokenize(text) if word.isalnum() and len(word) > 1]
            except Exception:
                pass
        
        # Fallback tokenization
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return [word for word in words if len(word) > 1]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if NLTK_AVAILABLE:
            try:
                nltk.download('punkt', quiet=True)
                sentences = sent_tokenize(text)
                return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
            except Exception:
                pass
        
        # Fallback sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def _split_text_into_segments(self, text: str, segment_size: int = 100) -> List[str]:
        """Split text into segments for topic analysis"""
        words = self._tokenize_text(text)
        segments = []
        
        for i in range(0, len(words), segment_size):
            segment = " ".join(words[i:i + segment_size])
            if len(segment.strip()) > 20:  # Only include substantial segments
                segments.append(segment)
        
        return segments if segments else [text]
    
    def _empty_coherence_analysis(self) -> Dict[str, Any]:
        """Return empty coherence analysis for invalid inputs"""
        return {
            "overall_coherence_score": 0.0,
            "semantic_flow": {"flow_score": 0.0, "transition_quality": 0.0, "narrative_coherence": 0.0},
            "semantic_drift": {"drift_score": 0.0, "drift_points": [], "stability_score": 1.0, "drift_curve": []},
            "topic_consistency": {"consistency_score": 0.0, "topic_distribution": [], "dominant_topic_ratio": 0.0},
            "cross_sentence_coherence": {"average_coherence": 0.0, "coherence_variance": 0.0, "min_coherence": 0.0},
            "prompt_completion_coherence": {},
            "text_length": 0,
            "sentence_count": 0
        }


# Convenience functions for easy integration
def analyze_semantic_coherence(text: str, prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick comprehensive semantic coherence analysis
    
    Args:
        text: Text to analyze
        prompt: Optional prompt for prompt-completion analysis
        
    Returns:
        Complete coherence analysis
    """
    analyzer = SemanticCoherenceAnalyzer()
    return analyzer.comprehensive_coherence_analysis(text, prompt)


def measure_prompt_completion_coherence(prompt: str, completion: str) -> Dict[str, float]:
    """
    Quick prompt-completion coherence measurement
    
    Args:
        prompt: Input prompt
        completion: Model completion
        
    Returns:
        Coherence metrics
    """
    analyzer = SemanticCoherenceAnalyzer()
    return analyzer.calculate_prompt_completion_coherence(prompt, completion)


# Testing function
def run_coherence_tests() -> Dict[str, Any]:
    """
    Run tests to validate coherence analysis
    
    Returns:
        Test results dictionary
    """
    test_cases = {
        "coherent": "The economic situation requires careful analysis. Market conditions have been volatile recently. Therefore, investors should consider diversified portfolios. This strategy helps mitigate risk effectively.",
        "incoherent": "The cat sat on the mat. Quantum physics explains everything. My favorite color is blue. Python programming is useful for data science.",
        "repetitive": "The system is working. The system is working well. The system continues working. The working system is good.",
        "drift": "Let's discuss machine learning algorithms. Deep learning uses neural networks. Cooking pasta requires boiling water. The weather today is sunny and warm."
    }
    
    analyzer = SemanticCoherenceAnalyzer()
    results = {}
    
    for test_name, test_text in test_cases.items():
        try:
            analysis = analyzer.comprehensive_coherence_analysis(test_text)
            results[test_name] = {
                "overall_coherence": analysis["overall_coherence_score"],
                "flow_score": analysis["semantic_flow"]["flow_score"],
                "drift_score": analysis["semantic_drift"]["drift_score"],
                "topic_consistency": analysis["topic_consistency"]["consistency_score"]
            }
        except Exception as e:
            results[test_name] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    # Run tests if executed directly
    test_results = run_coherence_tests()
    print("Semantic Coherence Analysis Test Results:")
    for test_name, result in test_results.items():
        print(f"{test_name}: {result}")