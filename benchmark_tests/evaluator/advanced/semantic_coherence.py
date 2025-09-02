"""
Semantic Coherence Analysis Module

Advanced semantic coherence analysis for language model evaluation including
prompt-completion coherence, semantic drift detection, and topic consistency measurement.

This module addresses the critique's key point about missing semantic coherence
measurements beyond simple pattern matching.

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
        """Lazy load embedding model with robust meta tensor handling"""
        if self._embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Use robust initialization to handle meta tensors
                model_instance, status = self._initialize_embedding_model_robust()
                self._embedding_model = model_instance
                
                if status == 'success':
                    logger.info(f"Loaded embedding model: {self.embedding_model_name}")
                elif status == 'fallback':
                    logger.info(f"Using fallback methods for semantic analysis (embedding model intentionally disabled)")
                else:  # status == 'error'
                    logger.error(f"Failed to initialize embedding model: {self.embedding_model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model {self.embedding_model_name}: {e}")
        return self._embedding_model
    
    def _validate_embeddings_tensor(self, embeddings) -> bool:
        """
        Validate embeddings array has actual data and is properly loaded
        
        Args:
            embeddings: Numpy array or PyTorch tensor to validate
            
        Returns:
            True if embeddings are valid and have data, False otherwise
        """
        if embeddings is None:
            return False
        
        # Handle numpy arrays (which we now use to avoid meta tensor issues)
        if isinstance(embeddings, np.ndarray):
            if embeddings.size == 0:
                logger.warning("Empty numpy array detected")
                return False
            return True
            
        # Handle PyTorch tensors (legacy support)
        if hasattr(embeddings, 'is_meta') and embeddings.is_meta:
            logger.warning("Meta tensor detected - model not properly loaded")
            return False
        if hasattr(embeddings, 'numel') and embeddings.numel() == 0:
            logger.warning("Empty tensor detected")
            return False
        return True
    
    def _initialize_embedding_model_robust(self):
        """
        Initialize SentenceTransformer with unified loading strategy
        
        Uses the unified model loader for consistent behavior across modules.
        Currently defaults to fallback mode for stability, but can be configured.
        
        Returns:
            tuple: (model_instance, status) where status is 'success', 'fallback', or 'error'
        """
        try:
            from .model_loader import UnifiedModelLoader, EvaluatorConfig
            
            # Get configuration from environment (allows user override)
            strategy = EvaluatorConfig.semantic_analyzer_strategy()
            force_cpu = EvaluatorConfig.force_cpu_mode()
            
            result = UnifiedModelLoader.load_embedding_model(
                model_name=self.embedding_model_name,
                strategy=strategy,
                force_cpu=force_cpu
            )
            
            return result.model, result.status
            
        except ImportError as e:
            # Fallback to old behavior if model_loader isn't available
            logger.warning(f"Unified model loader not available: {e}")
            logger.info("Using fallback methods for semantic analysis")
            return None, 'fallback'
    
    def _safe_tensor_to_numpy(self, embeddings):
        """
        Safely convert embeddings to numpy with proper handling
        
        Args:
            embeddings: Numpy array or PyTorch tensor to convert
            
        Returns:
            numpy array or None if conversion fails
        """
        try:
            if not self._validate_embeddings_tensor(embeddings):
                return None
            
            # If it's already a numpy array, return it
            if isinstance(embeddings, np.ndarray):
                return embeddings
            
            # Handle PyTorch tensor conversion
            if hasattr(embeddings, 'is_cuda') and embeddings.is_cuda:
                return embeddings.detach().cpu().numpy()
            else:
                return embeddings.detach().numpy() if hasattr(embeddings, 'detach') else embeddings.numpy()
                
        except Exception as e:
            logger.warning(f"Embeddings to numpy conversion failed: {e}")
            return None
    
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
            # For shorter texts, use sentence-based segmentation
            sentences = self._split_into_sentences(text)
            if len(sentences) >= 3:
                segments = sentences  # Use sentences as segments for short texts
            else:
                segments = self._split_text_into_segments(text)
            
            # Handle very short texts with basic consistency check
            if len(segments) < 2:
                return {"consistency_score": 1.0, "topic_distribution": [1.0], "dominant_topic_ratio": 1.0}
            elif len(segments) == 2:
                # For 2-segment texts, calculate simple topical similarity
                return self._calculate_simple_topic_consistency(segments)
            
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
    
    def detect_repetitive_content(self, text: str) -> Dict[str, Any]:
        """
        Advanced repetitive content detection with multiple pattern analysis
        
        Args:
            text: Text to analyze for repetitive patterns
            
        Returns:
            Dictionary with repetitive content metrics
        """
        if not text.strip():
            return {"repetitive_score": 0.0, "repetition_patterns": [], "severity": "none"}
        
        try:
            # Analyze different types of repetitive patterns
            word_repetition = self._detect_word_repetition(text)
            phrase_repetition = self._detect_phrase_repetition(text)
            sentence_repetition = self._detect_sentence_repetition(text)
            structural_repetition = self._detect_structural_repetition(text)
            exact_repetition = self._detect_exact_sentence_repetition(text)  # New addition
            
            # Calculate overall repetitive score (0.0 = not repetitive, 1.0 = highly repetitive)
            repetitive_score = max(
                word_repetition["score"],
                phrase_repetition["score"],
                sentence_repetition["score"],
                structural_repetition["score"],
                exact_repetition["score"]  # Include exact repetition in scoring
            )
            
            # Determine severity level (adjusted for better sensitivity)
            if repetitive_score >= 0.5:  # Reduced from 0.7 for earlier severe detection
                severity = "severe"
            elif repetitive_score >= 0.3:  # Reduced from 0.4 for earlier moderate detection
                severity = "moderate"
            elif repetitive_score >= 0.2:
                severity = "mild"
            else:
                severity = "none"
            
            # Compile all detected patterns
            all_patterns = []
            all_patterns.extend(word_repetition["patterns"])
            all_patterns.extend(phrase_repetition["patterns"])
            all_patterns.extend(sentence_repetition["patterns"])
            all_patterns.extend(structural_repetition["patterns"])
            all_patterns.extend(exact_repetition["patterns"])  # Include exact repetition patterns
            
            return {
                "repetitive_score": float(repetitive_score),
                "severity": severity,
                "repetition_patterns": all_patterns,
                "word_repetition": word_repetition,
                "phrase_repetition": phrase_repetition,
                "sentence_repetition": sentence_repetition,
                "structural_repetition": structural_repetition,
                "exact_repetition": exact_repetition  # Include exact repetition analysis
            }
            
        except Exception as e:
            logger.error(f"Repetitive content detection failed: {e}")
            return {"repetitive_score": 0.0, "repetition_patterns": [], "severity": "none"}
    
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
            
            # ENHANCEMENT: Advanced repetitive content detection
            repetitive_analysis = self.detect_repetitive_content(text)
            
            # Prompt-completion coherence if prompt provided
            prompt_coherence = {}
            if prompt:
                prompt_coherence = self.calculate_prompt_completion_coherence(prompt, text)
            
            # Cross-sentence coherence
            cross_sentence_coherence = self._calculate_cross_sentence_coherence(text)
            
            # Overall coherence score with repetitive content penalty
            overall_coherence = self._calculate_overall_coherence_score(
                semantic_flow, semantic_drift, topic_consistency, cross_sentence_coherence, repetitive_analysis
            )
            
            return {
                "overall_coherence_score": overall_coherence,
                "semantic_flow": semantic_flow,
                "semantic_drift": semantic_drift,
                "topic_consistency": topic_consistency,
                "cross_sentence_coherence": cross_sentence_coherence,
                "repetitive_content": repetitive_analysis,
                "prompt_completion_coherence": prompt_coherence,
                "text_length": len(text),
                "sentence_count": len(self._split_into_sentences(text))
            }
            
        except Exception as e:
            logger.error(f"Comprehensive coherence analysis failed: {e}")
            return self._empty_coherence_analysis()
    
    # Private helper methods for repetitive content detection
    
    def _detect_word_repetition(self, text: str) -> Dict[str, Any]:
        """Detect excessive word repetition"""
        words = self._tokenize_text(text)
        if len(words) < 10:
            return {"score": 0.0, "patterns": []}
        
        word_counts = Counter(words)
        total_words = len(words)
        unique_words = len(word_counts)
        
        # Calculate repetition metrics
        repetition_patterns = []
        excessive_repetitions = 0
        
        for word, count in word_counts.most_common():
            if len(word) > 3 and word not in self._stop_words:  # Skip short words and stop words
                frequency = count / total_words
                if frequency > 0.04:  # More than 4% of text (further reduced for better sensitivity)
                    repetition_patterns.append({
                        "type": "word",
                        "pattern": word,
                        "count": count,
                        "frequency": frequency
                    })
                    excessive_repetitions += count
        
        # Calculate repetition score
        if excessive_repetitions == 0:
            repetition_score = 0.0
        else:
            repetition_ratio = excessive_repetitions / total_words
            uniqueness_penalty = 1.0 - (unique_words / total_words)
            repetition_score = min(1.0, repetition_ratio * 3 + uniqueness_penalty)
        
        return {
            "score": repetition_score,
            "patterns": repetition_patterns,
            "unique_word_ratio": unique_words / total_words
        }
    
    def _detect_phrase_repetition(self, text: str) -> Dict[str, Any]:
        """Detect repetitive phrases and n-grams"""
        words = self._tokenize_text(text)
        if len(words) < 15:
            return {"score": 0.0, "patterns": []}
        
        phrase_patterns = []
        
        # Check 2-grams, 3-grams, and 4-grams
        for n in [2, 3, 4]:
            ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
            ngram_counts = Counter(ngrams)
            
            for ngram, count in ngram_counts.items():
                if count >= 2:  # Appears 2+ times (reduced for better sensitivity)
                    # Skip if all words are stop words
                    ngram_words = ngram.split()
                    if not all(word in self._stop_words for word in ngram_words):
                        phrase_patterns.append({
                            "type": "phrase",
                            "pattern": ngram,
                            "count": count,
                            "n": n
                        })
        
        # Calculate phrase repetition score
        if not phrase_patterns:
            return {"score": 0.0, "patterns": phrase_patterns}
        
        # Weight by phrase length and frequency
        weighted_repetitions = sum(
            pattern["count"] * pattern["n"] for pattern in phrase_patterns
        )
        repetition_score = min(1.0, weighted_repetitions / (len(words) * 0.3))
        
        return {
            "score": repetition_score,
            "patterns": sorted(phrase_patterns, key=lambda x: x["count"], reverse=True)
        }
    
    def _detect_sentence_repetition(self, text: str) -> Dict[str, Any]:
        """Detect repetitive sentences or sentence patterns"""
        sentences = self._split_into_sentences(text)
        if len(sentences) < 3:
            return {"score": 0.0, "patterns": []}
        
        sentence_patterns = []
        
        # Check for exact sentence repetitions
        sentence_counts = Counter(sentences)
        for sentence, count in sentence_counts.items():
            if count >= 2:
                sentence_patterns.append({
                    "type": "sentence_exact",
                    "pattern": sentence[:100] + "..." if len(sentence) > 100 else sentence,
                    "count": count
                })
        
        # Check for similar sentence structures
        sentence_structures = []
        for sentence in sentences:
            # Create structure pattern (first 3 words + last 2 words)
            words = sentence.split()
            if len(words) >= 5:
                structure = " ".join(words[:3] + ["..."] + words[-2:])
                sentence_structures.append(structure)
        
        structure_counts = Counter(sentence_structures)
        for structure, count in structure_counts.items():
            if count >= 2:  # Reduced threshold for better sensitivity
                sentence_patterns.append({
                    "type": "sentence_structure",
                    "pattern": structure,
                    "count": count
                })
        
        # Calculate sentence repetition score
        if not sentence_patterns:
            return {"score": 0.0, "patterns": sentence_patterns}
        
        repetitive_sentences = sum(pattern["count"] for pattern in sentence_patterns)
        repetition_score = min(1.0, repetitive_sentences / len(sentences))
        
        return {
            "score": repetition_score,
            "patterns": sentence_patterns
        }
    
    def _detect_structural_repetition(self, text: str) -> Dict[str, Any]:
        """Detect repetitive structural patterns"""
        sentences = self._split_into_sentences(text)
        if len(sentences) < 4:
            return {"score": 0.0, "patterns": []}
        
        structural_patterns = []
        
        # Check for repetitive sentence beginnings
        beginnings = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) >= 3:
                beginning = " ".join(words[:3])
                beginnings.append(beginning)
        
        beginning_counts = Counter(beginnings)
        for beginning, count in beginning_counts.items():
            if count >= 2:  # Reduced threshold for better sensitivity
                structural_patterns.append({
                    "type": "sentence_beginning",
                    "pattern": beginning,
                    "count": count
                })
        
        # Check for repetitive sentence endings
        endings = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) >= 3:
                ending = " ".join(words[-3:])
                endings.append(ending)
        
        ending_counts = Counter(endings)
        for ending, count in ending_counts.items():
            if count >= 2:  # Reduced threshold for better sensitivity
                structural_patterns.append({
                    "type": "sentence_ending",
                    "pattern": ending,
                    "count": count
                })
        
        # Check for repetitive punctuation patterns
        punctuation_patterns = re.findall(r'[.!?]+', text)
        punct_counts = Counter(punctuation_patterns)
        for punct, count in punct_counts.items():
            if count >= 5 and punct != "." and punct != "!":  # Exclude normal periods and exclamations
                structural_patterns.append({
                    "type": "punctuation",
                    "pattern": punct,
                    "count": count
                })
        
        # Calculate structural repetition score
        if not structural_patterns:
            return {"score": 0.0, "patterns": structural_patterns}
        
        structural_repetitions = sum(pattern["count"] for pattern in structural_patterns)
        repetition_score = min(1.0, structural_repetitions / (len(sentences) * 2))
        
        return {
            "score": repetition_score,
            "patterns": structural_patterns
        }
    
    def _detect_exact_sentence_repetition(self, text: str) -> Dict[str, Any]:
        """Detect exact sentence repetition more aggressively for edge cases"""
        sentences = self._split_into_sentences(text)
        if len(sentences) < 2:
            return {"score": 0.0, "patterns": []}
        
        # Count exact sentence matches (case-insensitive, trimmed)
        exact_matches = {}
        for sentence in sentences:
            cleaned = sentence.strip().lower()
            if len(cleaned) > 10:  # Skip very short sentences
                exact_matches[cleaned] = exact_matches.get(cleaned, 0) + 1
        
        # Find sentences that appear 2+ times
        repetitive_sentences = []
        total_repetitions = 0
        for sentence, count in exact_matches.items():
            if count >= 2:  # Exact sentence appears multiple times
                repetitive_sentences.append({
                    "type": "exact_sentence_repetition",
                    "pattern": sentence[:80] + ("..." if len(sentence) > 80 else ""),
                    "count": count
                })
                total_repetitions += count - 1  # Count extra occurrences
        
        # Calculate repetition score based on proportion of repeated sentences
        if total_repetitions == 0:
            return {"score": 0.0, "patterns": []}
        
        repetition_ratio = total_repetitions / len(sentences)
        repetition_score = min(repetition_ratio * 2.0, 1.0)  # Scale up for high impact
        
        return {
            "score": repetition_score,
            "patterns": repetitive_sentences
        }
    
    # Existing private helper methods
    
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
        """Calculate coherence using embedding similarity with proper device handling"""
        try:
            # Calculate direct transition coherence
            texts = [prompt_ending, completion_beginning]
            
            # Always use CPU tensors to avoid meta tensor issues
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            
            # Validate and safely convert embeddings
            if not self._validate_embeddings_tensor(embeddings):
                logger.warning("Invalid embeddings tensor, falling back to TF-IDF")
                return self._calculate_tfidf_coherence(prompt_ending, completion_beginning, full_prompt, full_completion)
            
            # Safe embeddings conversion
            emb_numpy = self._safe_tensor_to_numpy(embeddings)
            if emb_numpy is None:
                logger.warning("Embeddings conversion failed, falling back to TF-IDF")
                return self._calculate_tfidf_coherence(prompt_ending, completion_beginning, full_prompt, full_completion)
            
            # Ensure we have the right shape for cosine_similarity
            if len(emb_numpy.shape) == 1:
                emb_numpy = emb_numpy.reshape(1, -1)
            elif len(emb_numpy.shape) == 2:
                pass  # Already correct shape
            else:
                logger.warning("Unexpected embeddings shape, falling back to TF-IDF")
                return self._calculate_tfidf_coherence(prompt_ending, completion_beginning, full_prompt, full_completion)
            
            transition_similarity = cosine_similarity(
                emb_numpy[0:1], 
                emb_numpy[1:2]
            )[0, 0]
            
            # Calculate overall topic alignment
            full_texts = [full_prompt, full_completion]
            full_embeddings = self.embedding_model.encode(full_texts, convert_to_tensor=False)
            
            # Validate and safely convert full embeddings
            if not self._validate_embeddings_tensor(full_embeddings):
                logger.warning("Invalid full embeddings, using transition similarity only")
                topic_alignment = transition_similarity * 0.8  # Use transition as proxy
            else:
                full_emb_numpy = self._safe_tensor_to_numpy(full_embeddings)
                if full_emb_numpy is None:
                    topic_alignment = transition_similarity * 0.8  # Use transition as proxy
                else:
                    # Ensure correct shape
                    if len(full_emb_numpy.shape) == 1:
                        full_emb_numpy = full_emb_numpy.reshape(1, -1)
                    
                    if full_emb_numpy.shape[0] >= 2:
                        topic_alignment = cosine_similarity(
                            full_emb_numpy[0:1],
                            full_emb_numpy[1:2]
                        )[0, 0]
                    else:
                        topic_alignment = transition_similarity * 0.8
            
            # Calculate semantic bridge score (combination of transition and alignment)
            semantic_bridge = (transition_similarity * 0.7) + (topic_alignment * 0.3)
            
            return {
                "coherence_score": float(semantic_bridge),
                "semantic_bridge": float(transition_similarity),
                "topic_alignment": float(topic_alignment)
            }
            
        except Exception as e:
            logger.error(f"Embedding coherence calculation failed: {e}, falling back to TF-IDF")
            return self._calculate_tfidf_coherence(prompt_ending, completion_beginning, full_prompt, full_completion)
    
    def _calculate_tfidf_coherence(self, prompt_ending: str, completion_beginning: str,
                                 full_prompt: str, full_completion: str) -> Dict[str, float]:
        """Fallback coherence calculation using TF-IDF"""
        if not SKLEARN_AVAILABLE:
            return {"coherence_score": 0.0, "semantic_bridge": 0.0, "topic_alignment": 0.0}
        
        try:
            # TF-IDF based transition coherence
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100, min_df=1)
            transition_texts = [prompt_ending, completion_beginning]
            
            if len(" ".join(transition_texts).split()) < 5:
                # For short texts, use simple word overlap as fallback
                words1 = set(prompt_ending.lower().split())
                words2 = set(completion_beginning.lower().split())
                if words1 and words2:
                    overlap = len(words1 & words2) / len(words1 | words2)
                    return {"coherence_score": overlap * 0.5, "semantic_bridge": overlap * 0.5, "topic_alignment": overlap * 0.5}
                return {"coherence_score": 0.2, "semantic_bridge": 0.2, "topic_alignment": 0.2}  # Minimal baseline
            
            tfidf_matrix = vectorizer.fit_transform(transition_texts)
            transition_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0, 0]
            
            # Topic alignment using full texts with enhanced scoring
            full_texts = [full_prompt, full_completion]
            full_vectorizer = TfidfVectorizer(stop_words='english', max_features=200, min_df=1)
            full_tfidf = full_vectorizer.fit_transform(full_texts)
            topic_alignment = cosine_similarity(full_tfidf[0:1], full_tfidf[1:2])[0, 0]
            
            # Enhanced coherence scoring for technical content
            semantic_bridge = (transition_similarity * 0.7) + (topic_alignment * 0.3)
            
            # Boost scores for technical coherence (when both texts contain technical terms)
            technical_boost = self._calculate_technical_coherence_boost(full_prompt, full_completion)
            semantic_bridge = min(1.0, semantic_bridge + technical_boost)
            
            return {
                "coherence_score": float(semantic_bridge),
                "semantic_bridge": float(transition_similarity),
                "topic_alignment": float(topic_alignment)
            }
            
        except Exception as e:
            logger.error(f"TF-IDF coherence calculation failed: {e}")
            return {"coherence_score": 0.0, "semantic_bridge": 0.0, "topic_alignment": 0.0}
    
    def _measure_embedding_drift(self, window_texts: List[str]) -> Dict[str, Any]:
        """Measure semantic drift using embeddings with proper device handling"""
        try:
            # Always use CPU tensors to avoid meta tensor issues
            embeddings = self.embedding_model.encode(window_texts, convert_to_tensor=False)
            
            # Validate embeddings
            if not self._validate_embeddings_tensor(embeddings):
                logger.warning("Invalid embeddings tensor in drift measurement, falling back to TF-IDF")
                return self._measure_tfidf_drift(window_texts)
            
            # Safe embeddings conversion
            emb_numpy = self._safe_tensor_to_numpy(embeddings)
            if emb_numpy is None:
                logger.warning("Embeddings conversion failed in drift measurement, falling back to TF-IDF")
                return self._measure_tfidf_drift(window_texts)
            
            # Ensure correct shape
            if len(emb_numpy.shape) == 1:
                emb_numpy = emb_numpy.reshape(1, -1)
            
            # Calculate pairwise similarities between consecutive windows
            similarities = []
            for i in range(emb_numpy.shape[0] - 1):
                sim = cosine_similarity(
                    emb_numpy[i:i+1],
                    emb_numpy[i+1:i+2]
                )[0, 0]
                similarities.append(sim)
            
            # Calculate drift metrics
            drift_curve = [1.0 - sim for sim in similarities]  # Convert to drift scores
            drift_score = np.mean(drift_curve)
            # Fix: Stability should reward high average similarity with low variance
            avg_similarity = np.mean(similarities)
            similarity_variance = np.std(similarities)
            stability_score = avg_similarity * (1.0 - min(similarity_variance, 0.5))
            
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
            # Fix: Stability should reward high average similarity with low variance
            avg_similarity = np.mean(similarities)
            similarity_variance = np.std(similarities)
            stability_score = avg_similarity * (1.0 - min(similarity_variance, 0.5))
            
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
        """Enhanced topic consistency using domain-aware semantic analysis"""
        try:
            # Define domain vocabularies for topic consistency
            domain_vocabularies = {
                'economic': ['economic', 'market', 'financial', 'monetary', 'policy', 'analysis', 'volatility', 
                           'confidence', 'inflation', 'investment', 'growth', 'budget', 'trade', 'economy'],
                'technical': ['system', 'data', 'algorithm', 'network', 'implementation', 'performance', 
                            'software', 'computer', 'technology', 'digital', 'code', 'programming'],
                'academic': ['research', 'study', 'analysis', 'findings', 'methodology', 'evidence', 
                           'theory', 'hypothesis', 'conclusion', 'investigation'],
                'medical': ['health', 'medical', 'patient', 'treatment', 'diagnosis', 'therapy', 
                          'disease', 'clinical', 'medicine', 'healthcare'],
                'scientific': ['experiment', 'research', 'hypothesis', 'theory', 'data', 'results', 
                             'conclusion', 'method', 'observation', 'analysis']
            }
            
            segment_topics = []
            segment_keywords = []
            
            for segment in segments:
                words = self._tokenize_text(segment.lower())
                keywords = [word for word in words if len(word) > 3 and word not in self._stop_words]
                segment_keywords.append(set(keywords))
                
                # Determine primary topic for this segment
                topic_scores = {}
                for domain, vocab in domain_vocabularies.items():
                    score = sum(1 for word in keywords if word in vocab)
                    topic_scores[domain] = score
                
                # Get dominant topic (or 'mixed' if no clear dominant topic)
                max_score = max(topic_scores.values()) if topic_scores.values() else 0
                if max_score >= 2:  # At least 2 domain words needed
                    dominant_topic = max(topic_scores, key=topic_scores.get)
                    segment_topics.append(dominant_topic)
                else:
                    segment_topics.append('general')
            
            # Calculate consistency based on topic alignment
            if not segment_topics:
                return {"consistency_score": 1.0, "topic_distribution": [1.0], "dominant_topic_ratio": 1.0}
            
            # Count topic distribution
            topic_counts = {}
            for topic in segment_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            # Calculate consistency score
            total_segments = len(segments)
            max_topic_count = max(topic_counts.values())
            dominant_topic_ratio = max_topic_count / total_segments
            
            # Penalty for topic jumping (incoherent text)
            unique_topics = len([topic for topic in topic_counts.keys() if topic != 'general'])
            if unique_topics > 2:  # Too many different topics
                topic_jump_penalty = min(0.4, (unique_topics - 2) * 0.15)
            else:
                topic_jump_penalty = 0.0
            
            # Base consistency from dominant topic ratio
            base_consistency = dominant_topic_ratio
            
            # Apply penalty and ensure bounds
            consistency_score = max(0.0, min(1.0, base_consistency - topic_jump_penalty))
            
            return {
                "consistency_score": float(consistency_score),
                "topic_distribution": [dominant_topic_ratio],
                "dominant_topic_ratio": float(dominant_topic_ratio),
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
                # Always use CPU embeddings to avoid meta tensor issues
                embeddings = self.embedding_model.encode(sentences, convert_to_tensor=False)
                
                # Validate embeddings
                if self._validate_embeddings_tensor(embeddings):
                    emb_numpy = self._safe_tensor_to_numpy(embeddings)
                    if emb_numpy is not None:
                        # Ensure correct shape
                        if len(emb_numpy.shape) == 1:
                            emb_numpy = emb_numpy.reshape(1, -1)
                        
                        for i in range(emb_numpy.shape[0] - 1):
                            sim = cosine_similarity(
                                emb_numpy[i:i+1],
                                emb_numpy[i+1:i+2]
                            )[0, 0]
                            transition_scores.append(sim)
                    else:
                        transition_scores = self._calculate_lexical_transitions(sentences)
                else:
                    transition_scores = self._calculate_lexical_transitions(sentences)
            except Exception as e:
                logger.warning(f"Embedding transition calculation failed: {e}, using lexical fallback")
                transition_scores = self._calculate_lexical_transitions(sentences)
        else:
            transition_scores = self._calculate_lexical_transitions(sentences)
        
        return transition_scores
    
    def _calculate_lexical_transitions(self, sentences: List[str]) -> List[float]:
        """Enhanced lexical transition calculation with semantic coherence markers"""
        transition_scores = []
        
        # Define coherence-indicating patterns and markers
        coherence_markers = {
            'continuity': ['first', 'second', 'third', 'then', 'next', 'after', 'following', 'subsequently'],
            'causality': ['therefore', 'thus', 'hence', 'consequently', 'as a result', 'because', 'since', 'due to'],
            'contrast': ['however', 'nevertheless', 'despite', 'although', 'while', 'whereas', 'in contrast'],
            'elaboration': ['furthermore', 'moreover', 'additionally', 'also', 'in fact', 'indeed', 'specifically'],
            'conclusion': ['finally', 'in conclusion', 'to summarize', 'overall', 'ultimately', 'in summary']
        }
        
        for i in range(len(sentences) - 1):
            sentence1 = sentences[i].lower().strip()
            sentence2 = sentences[i + 1].lower().strip()
            
            words1 = set(self._tokenize_text(sentence1))
            words2 = set(self._tokenize_text(sentence2))
            
            if len(words1) == 0 or len(words2) == 0:
                transition_scores.append(0.0)
                continue
            
            # Base lexical similarity (reduced weight)
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            jaccard = intersection / union if union > 0 else 0.0
            
            # Boost for semantic coherence markers
            coherence_boost = 0.0
            
            # Check for transition markers in second sentence
            for category, markers in coherence_markers.items():
                for marker in markers:
                    if marker in sentence2:
                        if category in ['continuity', 'causality']:
                            coherence_boost += 0.4  # High boost for logical flow
                        elif category in ['elaboration', 'contrast']:
                            coherence_boost += 0.3  # Medium boost for development
                        else:
                            coherence_boost += 0.2  # Base boost for structure
                        break
            
            # Boost for topical consistency (shared domain words)
            domain_terms = {
                'economic': ['economic', 'market', 'financial', 'monetary', 'policy', 'analysis', 'volatility', 'confidence'],
                'technical': ['system', 'data', 'algorithm', 'network', 'implementation', 'performance'],
                'academic': ['research', 'study', 'analysis', 'findings', 'methodology', 'evidence']
            }
            
            for domain, terms in domain_terms.items():
                domain_overlap = sum(1 for term in terms if term in sentence1 and term in sentence2)
                if domain_overlap > 0:
                    coherence_boost += min(0.3, domain_overlap * 0.15)  # Cap at 0.3
            
            # Calculate final transition score
            final_score = min(1.0, jaccard * 0.3 + coherence_boost)  # Reduce jaccard weight
            transition_scores.append(final_score)
        
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
                # Always use CPU embeddings to avoid meta tensor issues
                embeddings = self.embedding_model.encode(sentences, convert_to_tensor=False)
                
                # Validate embeddings
                if self._validate_embeddings_tensor(embeddings):
                    emb_numpy = self._safe_tensor_to_numpy(embeddings)
                    if emb_numpy is not None:
                        # Ensure correct shape
                        if len(emb_numpy.shape) == 1:
                            emb_numpy = emb_numpy.reshape(1, -1)
                        
                        similarity_matrix = cosine_similarity(emb_numpy)
                        
                        # Extract upper triangle (avoiding diagonal)
                        for i in range(emb_numpy.shape[0]):
                            for j in range(i + 1, emb_numpy.shape[0]):
                                all_similarities.append(similarity_matrix[i, j])
                    else:
                        all_similarities = self._calculate_all_lexical_similarities(sentences)
                else:
                    all_similarities = self._calculate_all_lexical_similarities(sentences)
                        
            except Exception as e:
                logger.warning(f"Cross-sentence embedding calculation failed: {e}, using lexical fallback")
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
                                         topic_consistency: Dict, cross_sentence: Dict, 
                                         repetitive_analysis: Optional[Dict] = None) -> float:
        """Calculate overall coherence score from component metrics including repetitive content penalty"""
        try:
            # Weight different aspects of coherence
            flow_score = semantic_flow.get("flow_score", 0.0)
            drift_stability = semantic_drift.get("stability_score", 0.0)
            topic_score = topic_consistency.get("consistency_score", 0.0)
            cross_coherence = cross_sentence.get("average_coherence", 0.0)
            
            # ENHANCEMENT: Apply repetitive content penalty
            repetitive_penalty = 0.0
            if repetitive_analysis:
                repetitive_score = repetitive_analysis.get("repetitive_score", 0.0)
                severity = repetitive_analysis.get("severity", "none")
                
                # Scale penalty based on severity
                if severity == "severe":
                    repetitive_penalty = repetitive_score * 0.6  # Up to 60% penalty
                elif severity == "moderate":
                    repetitive_penalty = repetitive_score * 0.4  # Up to 40% penalty
                elif severity == "mild":
                    repetitive_penalty = repetitive_score * 0.2  # Up to 20% penalty
            
            # Weighted combination with repetitive penalty
            base_score = (
                flow_score * 0.3 +
                drift_stability * 0.3 +
                topic_score * 0.2 +
                cross_coherence * 0.2
            )
            
            # Apply repetitive content penalty
            overall_score = max(0.0, base_score - repetitive_penalty)
            
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
            "repetitive_content": {"repetitive_score": 0.0, "repetition_patterns": [], "severity": "none"},
            "prompt_completion_coherence": {},
            "text_length": 0,
            "sentence_count": 0
        }

    def _calculate_technical_coherence_boost(self, prompt: str, completion: str) -> float:
        """Calculate boost for technical coherence between prompt and completion"""
        # Technical terms that should get coherence boost
        technical_terms = {
            'algorithm', 'implementation', 'network', 'distributed', 'hash', 'table',
            'system', 'data', 'structure', 'node', 'routing', 'protocol', 'consistent',
            'hashing', 'fault', 'tolerance', 'mechanism', 'computer', 'science',
            'database', 'key', 'value', 'lookup', 'performance', 'scalable'
        }
        
        prompt_lower = prompt.lower()
        completion_lower = completion.lower()
        
        prompt_terms = sum(1 for term in technical_terms if term in prompt_lower)
        completion_terms = sum(1 for term in technical_terms if term in completion_lower)
        
        # If both have technical terms, provide boost
        if prompt_terms > 0 and completion_terms > 0:
            return min(0.3, (prompt_terms + completion_terms) * 0.05)  # Up to 0.3 boost
        
        return 0.0


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