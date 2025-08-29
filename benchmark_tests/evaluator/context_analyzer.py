"""
Context Window Analysis Module

Advanced context window analysis for language model evaluation including
token-position quality tracking, context saturation detection, and degradation analysis.

This module addresses the critique's key point about missing context window
analysis and quality degradation detection.

Author: Claude Code
Version: 1.0.0
"""

import re
import math
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from collections import Counter, defaultdict
import numpy as np

# Optional imports with fallbacks
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available. Using fallback tokenization.")

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Some analysis disabled.")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    import scipy.stats as stats
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn/scipy not available. Some statistical analysis disabled.")

# Set up logging
logger = logging.getLogger(__name__)


class ContextWindowAnalyzer:
    """
    Advanced context window analyzer for language model responses.
    
    Provides comprehensive context window analysis:
    - Token-position quality tracking
    - Context saturation detection  
    - Quality degradation point identification
    - Context window limit estimation
    - Position-based metric analysis
    """
    
    def __init__(self, tokenizer_model: str = "cl100k_base", window_size: int = 512):
        """
        Initialize context window analyzer
        
        Args:
            tokenizer_model: Tokenizer model for token counting
            window_size: Default analysis window size
        """
        self.tokenizer_model = tokenizer_model
        self.window_size = window_size
        self.tokenizer = self._initialize_tokenizer(tokenizer_model)
        
        logger.info(f"ContextWindowAnalyzer initialized with {tokenizer_model} tokenizer")
    
    def _initialize_tokenizer(self, model_name: str):
        """Initialize appropriate tokenizer based on model mapping"""
        if not TIKTOKEN_AVAILABLE:
            return None
            
        try:
            # Direct model mappings
            model_mappings = {
                # OpenAI models
                "gpt-4": "gpt-4",
                "gpt-3.5-turbo": "gpt-3.5-turbo", 
                "gpt-oss-20b": "gpt2",  # GPT-OSS uses GPT-2 BPE
                
                # Qwen models  
                "qwen3-30b-a3b-base": "gpt2",  # Qwen compatible with GPT-2 BPE
                "qwen": "gpt2",  # Pattern match for other Qwen models
                
                # Common fallbacks
                "llama": "gpt2",
                "mistral": "gpt2", 
                "claude": "gpt2",
            }
            
            model_lower = model_name.lower()
            
            # Exact match first
            if model_lower in model_mappings:
                encoding_name = model_mappings[model_lower]
            else:
                # Pattern matching
                encoding_name = None
                for pattern, encoding in model_mappings.items():
                    if pattern in model_lower:
                        encoding_name = encoding
                        break
                
                if not encoding_name:
                    encoding_name = "gpt2"  # Universal fallback
            
            # Use appropriate tiktoken method
            if encoding_name in ["gpt-4", "gpt-3.5-turbo"]:
                return tiktoken.encoding_for_model(encoding_name)
            else:
                return tiktoken.get_encoding(encoding_name)
                
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for {model_name}: {e}")
            return None
    
    def analyze_quality_by_position(self, text: str, metrics_calculator: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Analyze quality metrics across token positions
        
        Args:
            text: Text to analyze
            metrics_calculator: Optional function to calculate custom metrics
            
        Returns:
            Dictionary with position-based quality analysis
        """
        if not text.strip():
            return {"error": "Empty text provided", "position_metrics": [], "degradation_points": []}
        
        try:
            # Tokenize text for position tracking
            if self.tokenizer:
                tokens = self.tokenizer.encode(text)
                token_positions = self._map_token_positions(text, tokens)
            else:
                words = text.split()
                tokens = list(range(len(words)))  # Fallback to word indices
                token_positions = self._map_word_positions(text, words)
            
            if len(tokens) < self.window_size:
                # Text too short for position analysis
                return self._single_segment_analysis(text, metrics_calculator)
            
            # Create position-based segments
            segments = self._create_position_segments(text, tokens, token_positions)
            
            # Analyze each segment
            position_metrics = []
            for i, segment_data in enumerate(segments):
                segment_analysis = self._analyze_segment(
                    segment_data, i, len(segments), metrics_calculator
                )
                position_metrics.append(segment_analysis)
            
            # Detect degradation patterns
            degradation_analysis = self._detect_degradation_patterns(position_metrics)
            
            # Calculate position trends
            trend_analysis = self._calculate_position_trends(position_metrics)
            
            return {
                "total_tokens": len(tokens),
                "total_segments": len(segments),
                "segment_size": self.window_size,
                "position_metrics": position_metrics,
                "degradation_analysis": degradation_analysis,
                "trend_analysis": trend_analysis,
                "quality_curve": [seg["quality_score"] for seg in position_metrics]
            }
            
        except Exception as e:
            logger.error(f"Position quality analysis failed: {e}")
            return {"error": str(e), "position_metrics": [], "degradation_points": []}
    
    def detect_context_saturation(self, text: str) -> Dict[str, Any]:
        """
        Detect context saturation points and patterns
        
        Args:
            text: Text to analyze for saturation
            
        Returns:
            Dictionary with saturation analysis
        """
        if not text.strip():
            return {"saturation_detected": False, "saturation_point": None, "saturation_score": 0.0}
        
        try:
            # Multiple saturation detection methods
            repetition_saturation = self._detect_repetition_saturation(text)
            entropy_saturation = self._detect_entropy_saturation(text)
            semantic_saturation = self._detect_semantic_saturation(text)
            vocabulary_saturation = self._detect_vocabulary_saturation(text)
            
            # Combine saturation indicators
            saturation_score = self._calculate_combined_saturation_score(
                repetition_saturation, entropy_saturation, 
                semantic_saturation, vocabulary_saturation
            )
            
            # Determine primary saturation point
            saturation_point = self._identify_primary_saturation_point(
                repetition_saturation, entropy_saturation,
                semantic_saturation, vocabulary_saturation
            )
            
            return {
                "saturation_detected": saturation_score > 0.6,  # 60% threshold
                "saturation_point": saturation_point,
                "saturation_score": saturation_score,
                "repetition_saturation": repetition_saturation,
                "entropy_saturation": entropy_saturation,
                "semantic_saturation": semantic_saturation,
                "vocabulary_saturation": vocabulary_saturation,
                "saturation_type": self._classify_saturation_type(
                    repetition_saturation, entropy_saturation,
                    semantic_saturation, vocabulary_saturation
                )
            }
            
        except Exception as e:
            logger.error(f"Context saturation detection failed: {e}")
            return {"saturation_detected": False, "saturation_point": None, "saturation_score": 0.0}
    
    def find_degradation_points(self, quality_curve: List[float], threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Find significant degradation points in quality curve
        
        Args:
            quality_curve: List of quality scores by position
            threshold: Minimum degradation ratio to flag (0.8 = 20% drop)
            
        Returns:
            List of degradation point information
        """
        if len(quality_curve) < 3:
            return []
        
        try:
            degradation_points = []
            
            # Simple change point detection
            for i in range(2, len(quality_curve)):
                recent_avg = np.mean(quality_curve[max(0, i-2):i])
                current_score = quality_curve[i]
                
                if recent_avg > 0 and current_score < recent_avg * threshold:
                    # Significant drop detected
                    degradation_points.append({
                        "position": i,
                        "quality_drop": recent_avg - current_score,
                        "drop_percentage": (recent_avg - current_score) / recent_avg * 100,
                        "previous_avg": recent_avg,
                        "current_score": current_score,
                        "severity": self._classify_degradation_severity(
                            recent_avg, current_score, threshold
                        )
                    })
            
            # Advanced change point detection using statistical methods
            if SKLEARN_AVAILABLE and len(quality_curve) >= 10:
                statistical_points = self._detect_statistical_change_points(quality_curve)
                
                # Merge with simple detection results
                for stat_point in statistical_points:
                    # Check if not already detected
                    if not any(abs(dp["position"] - stat_point["position"]) <= 2 
                             for dp in degradation_points):
                        degradation_points.append(stat_point)
            
            # Sort by position
            degradation_points.sort(key=lambda x: x["position"])
            
            return degradation_points
            
        except Exception as e:
            logger.error(f"Degradation point detection failed: {e}")
            return []
    
    def estimate_context_limit(self, text: str) -> Dict[str, Any]:
        """
        Estimate effective context window limit based on quality degradation
        
        Args:
            text: Text to analyze
            
        Returns:
            Context limit estimation
        """
        if not text.strip():
            return {"estimated_limit": 0, "confidence": 0.0, "evidence": []}
        
        try:
            # Analyze quality by position
            position_analysis = self.analyze_quality_by_position(text)
            quality_curve = position_analysis.get("quality_curve", [])
            
            if len(quality_curve) < 3:
                return {"estimated_limit": len(text.split()), "confidence": 0.5, "evidence": ["insufficient_data"]}
            
            # Multiple estimation methods
            degradation_limit = self._estimate_from_degradation(quality_curve, position_analysis)
            saturation_limit = self._estimate_from_saturation(text)
            entropy_limit = self._estimate_from_entropy_drop(text)
            
            # Combine estimates
            estimates = [degradation_limit, saturation_limit, entropy_limit]
            estimates = [est for est in estimates if est["limit"] > 0]
            
            if not estimates:
                return {"estimated_limit": len(text.split()), "confidence": 0.3, "evidence": ["no_clear_limit"]}
            
            # Weight estimates by confidence
            weighted_limit = sum(est["limit"] * est["confidence"] for est in estimates) / sum(est["confidence"] for est in estimates)
            average_confidence = np.mean([est["confidence"] for est in estimates])
            
            evidence = []
            for est in estimates:
                evidence.extend(est["evidence"])
            
            return {
                "estimated_limit": int(weighted_limit),
                "confidence": float(average_confidence),
                "evidence": evidence,
                "individual_estimates": estimates,
                "token_based_limit": self._convert_to_token_estimate(int(weighted_limit))
            }
            
        except Exception as e:
            logger.error(f"Context limit estimation failed: {e}")
            return {"estimated_limit": 0, "confidence": 0.0, "evidence": ["error"]}
    
    def comprehensive_context_analysis(self, text: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive context window analysis
        
        Args:
            text: Text to analyze
            prompt: Optional prompt for context analysis
            
        Returns:
            Complete context analysis
        """
        if not text.strip():
            return {"error": "Empty text provided"}
        
        try:
            # Core analyses
            position_analysis = self.analyze_quality_by_position(text)
            saturation_analysis = self.detect_context_saturation(text)
            limit_estimation = self.estimate_context_limit(text)
            
            # Quality curve analysis
            quality_curve = position_analysis.get("quality_curve", [])
            degradation_points = self.find_degradation_points(quality_curve)
            
            # Context efficiency metrics
            efficiency_metrics = self._calculate_context_efficiency(text, quality_curve)
            
            # Length-based insights
            length_analysis = self._analyze_length_patterns(text, quality_curve)
            
            return {
                "text_length": len(text),
                "word_count": len(text.split()),
                "estimated_tokens": self._estimate_token_count(text),
                "position_analysis": position_analysis,
                "saturation_analysis": saturation_analysis,
                "limit_estimation": limit_estimation,
                "degradation_points": degradation_points,
                "efficiency_metrics": efficiency_metrics,
                "length_analysis": length_analysis,
                "context_health_score": self._calculate_context_health_score(
                    position_analysis, saturation_analysis, degradation_points
                )
            }
            
        except Exception as e:
            logger.error(f"Comprehensive context analysis failed: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    
    def _map_token_positions(self, text: str, tokens: List[int]) -> List[Tuple[int, int]]:
        """Map tokens to character positions in text"""
        if not self.tokenizer:
            return [(0, len(text))]
        
        try:
            # Reconstruct text from tokens to map positions
            positions = []
            current_pos = 0
            
            for token in tokens:
                token_text = self.tokenizer.decode([token])
                start_pos = text.find(token_text, current_pos)
                if start_pos == -1:
                    # Fallback for complex tokenization
                    start_pos = current_pos
                end_pos = start_pos + len(token_text)
                positions.append((start_pos, end_pos))
                current_pos = end_pos
            
            return positions
        except Exception:
            # Fallback to even distribution
            chars_per_token = len(text) / len(tokens)
            return [(int(i * chars_per_token), int((i + 1) * chars_per_token)) 
                   for i in range(len(tokens))]
    
    def _map_word_positions(self, text: str, words: List[str]) -> List[Tuple[int, int]]:
        """Map words to character positions in text"""
        positions = []
        current_pos = 0
        
        for word in words:
            start_pos = text.find(word, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            end_pos = start_pos + len(word)
            positions.append((start_pos, end_pos))
            current_pos = end_pos
        
        return positions
    
    def _create_position_segments(self, text: str, tokens: List[int], 
                                token_positions: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        """Create position-based segments for analysis"""
        segments = []
        
        for i in range(0, len(tokens), self.window_size // 2):  # 50% overlap
            end_token = min(i + self.window_size, len(tokens))
            
            if i < len(token_positions) and end_token - 1 < len(token_positions):
                start_char = token_positions[i][0]
                end_char = token_positions[end_token - 1][1]
                segment_text = text[start_char:end_char]
                
                segments.append({
                    "start_token": i,
                    "end_token": end_token,
                    "start_char": start_char,
                    "end_char": end_char,
                    "text": segment_text,
                    "token_count": end_token - i,
                    "char_count": end_char - start_char
                })
        
        return segments
    
    def _analyze_segment(self, segment_data: Dict[str, Any], segment_index: int, 
                        total_segments: int, metrics_calculator: Optional[Callable]) -> Dict[str, Any]:
        """Analyze individual segment quality"""
        segment_text = segment_data["text"]
        
        # Basic quality metrics
        word_count = len(segment_text.split())
        char_count = len(segment_text)
        
        # Repetition analysis
        repetition_score = self._calculate_segment_repetition(segment_text)
        
        # Entropy analysis
        entropy_score = self._calculate_segment_entropy(segment_text)
        
        # Coherence analysis
        coherence_score = self._calculate_segment_coherence(segment_text)
        
        # Overall quality score
        quality_score = self._combine_segment_scores(
            repetition_score, entropy_score, coherence_score
        )
        
        # Custom metrics if provided
        custom_metrics = {}
        if metrics_calculator:
            try:
                custom_metrics = metrics_calculator(segment_text)
            except Exception as e:
                logger.warning(f"Custom metrics calculation failed: {e}")
        
        return {
            "segment_index": segment_index,
            "position_ratio": segment_index / max(total_segments - 1, 1),
            "start_token": segment_data["start_token"],
            "end_token": segment_data["end_token"],
            "word_count": word_count,
            "char_count": char_count,
            "quality_score": quality_score,
            "repetition_score": repetition_score,
            "entropy_score": entropy_score,
            "coherence_score": coherence_score,
            "custom_metrics": custom_metrics
        }
    
    def _calculate_segment_repetition(self, text: str) -> float:
        """Calculate repetition score for segment (0-1, higher = less repetitive)"""
        words = text.lower().split()
        if not words:
            return 1.0
        
        word_counts = Counter(words)
        unique_words = len(word_counts)
        total_words = len(words)
        
        # Repetition penalty
        repetition_penalty = 0
        for count in word_counts.values():
            if count > 1:
                repetition_penalty += (count - 1) / total_words
        
        repetition_score = 1.0 - min(repetition_penalty, 0.8)
        return max(repetition_score, 0.0)
    
    def _calculate_segment_entropy(self, text: str) -> float:
        """Calculate normalized entropy score for segment"""
        words = text.lower().split()
        if not words:
            return 0.0
        
        word_counts = Counter(words)
        total_words = len(words)
        
        entropy = 0.0
        for count in word_counts.values():
            prob = count / total_words
            entropy -= prob * math.log2(prob)
        
        # Normalize by maximum possible entropy
        max_entropy = math.log2(len(word_counts)) if len(word_counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy
        
        return min(normalized_entropy, 1.0)
    
    def _calculate_segment_coherence(self, text: str) -> float:
        """Calculate coherence score for segment"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 1.0
        
        # Simple coherence based on sentence length consistency and logical connectors
        lengths = [len(s.split()) for s in sentences]
        length_consistency = 1.0 - (np.std(lengths) / max(np.mean(lengths), 1.0))
        
        # Logical connectors
        connectors = ['therefore', 'however', 'moreover', 'furthermore', 'consequently', 'thus']
        connector_count = sum(1 for conn in connectors if conn in text.lower())
        connector_bonus = min(connector_count / len(sentences), 0.2)
        
        coherence_score = min(length_consistency + connector_bonus, 1.0)
        return max(coherence_score, 0.0)
    
    def _combine_segment_scores(self, repetition: float, entropy: float, coherence: float) -> float:
        """Combine segment scores into overall quality score"""
        # Weighted combination
        quality_score = (
            repetition * 0.4 +      # Penalize repetition heavily
            entropy * 0.4 +         # Reward entropy/diversity
            coherence * 0.2         # Reward coherence
        )
        return max(0.0, min(quality_score, 1.0))
    
    def _detect_degradation_patterns(self, position_metrics: List[Dict]) -> Dict[str, Any]:
        """Detect degradation patterns in position metrics"""
        if len(position_metrics) < 3:
            return {"degradation_detected": False, "degradation_rate": 0.0}
        
        quality_scores = [seg["quality_score"] for seg in position_metrics]
        
        # Linear degradation trend
        positions = list(range(len(quality_scores)))
        if SKLEARN_AVAILABLE:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(positions, quality_scores)
                
                return {
                    "degradation_detected": slope < -0.05,  # Negative slope
                    "degradation_rate": float(-slope) if slope < 0 else 0.0,
                    "correlation": float(r_value),
                    "significance": float(p_value),
                    "trend": "declining" if slope < -0.05 else "stable" if abs(slope) <= 0.05 else "improving"
                }
            except Exception:
                pass
        
        # Fallback: simple comparison
        first_half_avg = np.mean(quality_scores[:len(quality_scores)//2])
        second_half_avg = np.mean(quality_scores[len(quality_scores)//2:])
        
        degradation_rate = max(0.0, first_half_avg - second_half_avg)
        
        return {
            "degradation_detected": degradation_rate > 0.1,
            "degradation_rate": float(degradation_rate),
            "trend": "declining" if degradation_rate > 0.1 else "stable"
        }
    
    def _calculate_position_trends(self, position_metrics: List[Dict]) -> Dict[str, Any]:
        """Calculate trends across different metrics"""
        if not position_metrics:
            return {}
        
        trends = {}
        metrics_to_analyze = ["quality_score", "repetition_score", "entropy_score", "coherence_score"]
        
        for metric in metrics_to_analyze:
            values = [seg.get(metric, 0.0) for seg in position_metrics]
            
            if len(values) >= 3:
                first_third = np.mean(values[:len(values)//3])
                last_third = np.mean(values[-len(values)//3:])
                
                trend_direction = "improving" if last_third > first_third * 1.1 else \
                                "declining" if last_third < first_third * 0.9 else "stable"
                
                trends[metric] = {
                    "direction": trend_direction,
                    "change_magnitude": abs(last_third - first_third),
                    "start_avg": float(first_third),
                    "end_avg": float(last_third)
                }
        
        return trends
    
    def _detect_repetition_saturation(self, text: str) -> Dict[str, Any]:
        """Detect repetition-based saturation"""
        words = text.split()
        if len(words) < 50:
            return {"detected": False, "saturation_point": None, "severity": 0.0}
        
        # Analyze repetition in sliding windows
        window_size = 50
        repetition_scores = []
        
        for i in range(0, len(words) - window_size + 1, window_size // 2):
            window = words[i:i + window_size]
            word_counts = Counter(window)
            
            # Calculate repetition ratio
            repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
            repetition_ratio = repeated_words / len(window)
            repetition_scores.append(repetition_ratio)
        
        # Find saturation point
        saturation_threshold = 0.3  # 30% repetition
        saturation_point = None
        
        for i, score in enumerate(repetition_scores):
            if score > saturation_threshold:
                saturation_point = i * (window_size // 2)
                break
        
        avg_repetition = np.mean(repetition_scores)
        
        return {
            "detected": avg_repetition > saturation_threshold,
            "saturation_point": saturation_point,
            "severity": float(avg_repetition),
            "repetition_curve": [float(score) for score in repetition_scores]
        }
    
    def _detect_entropy_saturation(self, text: str) -> Dict[str, Any]:
        """Detect entropy-based saturation"""
        words = text.split()
        if len(words) < 50:
            return {"detected": False, "saturation_point": None, "severity": 0.0}
        
        window_size = 50
        entropy_scores = []
        
        for i in range(0, len(words) - window_size + 1, window_size // 2):
            window = words[i:i + window_size]
            word_counts = Counter(window)
            
            # Calculate entropy
            entropy = 0.0
            for count in word_counts.values():
                prob = count / len(window)
                entropy -= prob * math.log2(prob)
            
            # Normalize
            max_entropy = math.log2(len(word_counts)) if len(word_counts) > 1 else 1.0
            normalized_entropy = entropy / max_entropy
            entropy_scores.append(normalized_entropy)
        
        # Detect significant entropy drops
        if len(entropy_scores) < 3:
            return {"detected": False, "saturation_point": None, "severity": 0.0}
        
        # Find where entropy drops significantly
        initial_entropy = np.mean(entropy_scores[:3])
        saturation_point = None
        
        for i in range(3, len(entropy_scores)):
            if entropy_scores[i] < initial_entropy * 0.6:  # 40% drop
                saturation_point = i * (window_size // 2)
                break
        
        entropy_drop = max(0.0, initial_entropy - min(entropy_scores))
        
        return {
            "detected": entropy_drop > 0.2,
            "saturation_point": saturation_point,
            "severity": float(entropy_drop),
            "entropy_curve": [float(score) for score in entropy_scores]
        }
    
    def _detect_semantic_saturation(self, text: str) -> Dict[str, Any]:
        """Detect semantic saturation (placeholder for embedding-based analysis)"""
        # This would use embeddings to detect semantic repetition
        # Simplified version using vocabulary analysis
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 5:
            return {"detected": False, "saturation_point": None, "severity": 0.0}
        
        # Simple semantic diversity based on unique concepts
        all_words = set()
        sentence_novelty = []
        
        for sentence in sentences:
            words = set(sentence.lower().split())
            new_words = words - all_words
            novelty = len(new_words) / max(len(words), 1)
            sentence_novelty.append(novelty)
            all_words.update(words)
        
        # Find where novelty drops significantly
        avg_novelty = np.mean(sentence_novelty)
        saturation_point = None
        
        for i, novelty in enumerate(sentence_novelty):
            if novelty < avg_novelty * 0.3:  # Very low novelty
                saturation_point = i
                break
        
        semantic_decline = max(0.0, max(sentence_novelty) - min(sentence_novelty))
        
        return {
            "detected": semantic_decline > 0.5,
            "saturation_point": saturation_point,
            "severity": float(semantic_decline),
            "novelty_curve": [float(score) for score in sentence_novelty]
        }
    
    def _detect_vocabulary_saturation(self, text: str) -> Dict[str, Any]:
        """Detect vocabulary saturation"""
        words = text.split()
        if len(words) < 100:
            return {"detected": False, "saturation_point": None, "severity": 0.0}
        
        # Track vocabulary growth
        unique_words = set()
        vocab_growth = []
        
        window_size = 50
        for i in range(0, len(words), window_size):
            window = words[i:i + window_size]
            unique_words.update(window)
            vocab_growth.append(len(unique_words))
        
        # Calculate growth rate decline
        if len(vocab_growth) < 3:
            return {"detected": False, "saturation_point": None, "severity": 0.0}
        
        # Find where vocabulary growth plateaus
        growth_rates = []
        for i in range(1, len(vocab_growth)):
            rate = vocab_growth[i] - vocab_growth[i-1]
            growth_rates.append(rate)
        
        initial_rate = np.mean(growth_rates[:3]) if len(growth_rates) >= 3 else growth_rates[0]
        saturation_point = None
        
        for i, rate in enumerate(growth_rates):
            if rate < initial_rate * 0.2:  # Growth slowed to 20% of initial
                saturation_point = i * window_size
                break
        
        growth_decline = max(0.0, initial_rate - min(growth_rates)) if growth_rates else 0.0
        
        return {
            "detected": growth_decline > initial_rate * 0.5,
            "saturation_point": saturation_point,
            "severity": float(growth_decline / max(initial_rate, 1.0)),
            "vocab_growth_curve": vocab_growth,
            "growth_rates": [float(rate) for rate in growth_rates]
        }
    
    def _calculate_combined_saturation_score(self, repetition_sat: Dict, entropy_sat: Dict,
                                           semantic_sat: Dict, vocab_sat: Dict) -> float:
        """Combine different saturation indicators"""
        scores = [
            repetition_sat.get("severity", 0.0) * 0.3,
            entropy_sat.get("severity", 0.0) * 0.3,
            semantic_sat.get("severity", 0.0) * 0.2,
            vocab_sat.get("severity", 0.0) * 0.2
        ]
        
        return min(sum(scores), 1.0)
    
    def _identify_primary_saturation_point(self, repetition_sat: Dict, entropy_sat: Dict,
                                         semantic_sat: Dict, vocab_sat: Dict) -> Optional[int]:
        """Identify the earliest and most significant saturation point"""
        points = []
        
        for sat_type, sat_data in [
            ("repetition", repetition_sat),
            ("entropy", entropy_sat),
            ("semantic", semantic_sat),
            ("vocabulary", vocab_sat)
        ]:
            if sat_data.get("detected") and sat_data.get("saturation_point") is not None:
                points.append({
                    "type": sat_type,
                    "point": sat_data["saturation_point"],
                    "severity": sat_data.get("severity", 0.0)
                })
        
        if not points:
            return None
        
        # Return earliest significant point
        points.sort(key=lambda x: x["point"])
        return points[0]["point"]
    
    def _classify_saturation_type(self, repetition_sat: Dict, entropy_sat: Dict,
                                 semantic_sat: Dict, vocab_sat: Dict) -> str:
        """Classify the primary type of saturation"""
        detections = [
            ("repetition", repetition_sat.get("detected", False), repetition_sat.get("severity", 0.0)),
            ("entropy", entropy_sat.get("detected", False), entropy_sat.get("severity", 0.0)),
            ("semantic", semantic_sat.get("detected", False), semantic_sat.get("severity", 0.0)),
            ("vocabulary", vocab_sat.get("detected", False), vocab_sat.get("severity", 0.0))
        ]
        
        # Find strongest detection
        detected = [(name, severity) for name, detected, severity in detections if detected]
        
        if not detected:
            return "none"
        
        detected.sort(key=lambda x: x[1], reverse=True)
        return detected[0][0]
    
    def _detect_statistical_change_points(self, quality_curve: List[float]) -> List[Dict[str, Any]]:
        """Advanced statistical change point detection"""
        if len(quality_curve) < 10:
            return []
        
        try:
            # Simple variance-based change point detection
            change_points = []
            window_size = max(3, len(quality_curve) // 10)
            
            for i in range(window_size, len(quality_curve) - window_size):
                before = quality_curve[i - window_size:i]
                after = quality_curve[i:i + window_size]
                
                before_mean = np.mean(before)
                after_mean = np.mean(after)
                
                if before_mean > 0 and after_mean < before_mean * 0.8:
                    # Significant drop
                    change_points.append({
                        "position": i,
                        "quality_drop": before_mean - after_mean,
                        "drop_percentage": (before_mean - after_mean) / before_mean * 100,
                        "previous_avg": before_mean,
                        "current_score": after_mean,
                        "severity": "statistical",
                        "method": "variance_analysis"
                    })
            
            return change_points
            
        except Exception as e:
            logger.error(f"Statistical change point detection failed: {e}")
            return []
    
    def _classify_degradation_severity(self, previous_avg: float, current_score: float, threshold: float) -> str:
        """Classify degradation severity"""
        drop_ratio = (previous_avg - current_score) / previous_avg if previous_avg > 0 else 0
        
        if drop_ratio >= 0.5:
            return "severe"
        elif drop_ratio >= 0.3:
            return "moderate"
        elif drop_ratio >= threshold:
            return "mild"
        else:
            return "minimal"
    
    def _estimate_from_degradation(self, quality_curve: List[float], position_analysis: Dict) -> Dict[str, Any]:
        """Estimate context limit from degradation analysis"""
        if not quality_curve or len(quality_curve) < 3:
            return {"limit": 0, "confidence": 0.0, "evidence": []}
        
        degradation_points = self.find_degradation_points(quality_curve)
        
        if not degradation_points:
            return {"limit": len(quality_curve) * self.window_size, "confidence": 0.3, "evidence": ["no_degradation"]}
        
        # Use first significant degradation point
        first_degradation = degradation_points[0]
        estimated_limit = first_degradation["position"] * (self.window_size // 2)  # Account for overlap
        
        confidence = min(first_degradation["drop_percentage"] / 30.0, 1.0)  # Higher drops = higher confidence
        
        return {
            "limit": estimated_limit,
            "confidence": confidence,
            "evidence": [f"degradation_at_position_{first_degradation['position']}"]
        }
    
    def _estimate_from_saturation(self, text: str) -> Dict[str, Any]:
        """Estimate context limit from saturation analysis"""
        saturation_analysis = self.detect_context_saturation(text)
        
        if not saturation_analysis.get("saturation_detected"):
            return {"limit": len(text.split()), "confidence": 0.2, "evidence": ["no_saturation"]}
        
        saturation_point = saturation_analysis.get("saturation_point", 0)
        confidence = saturation_analysis.get("saturation_score", 0.0)
        
        return {
            "limit": saturation_point,
            "confidence": confidence,
            "evidence": [f"saturation_{saturation_analysis.get('saturation_type', 'unknown')}"]
        }
    
    def _estimate_from_entropy_drop(self, text: str) -> Dict[str, Any]:
        """Estimate context limit from entropy analysis"""
        # This would integrate with entropy_calculator.py
        # Simplified version for now
        
        words = text.split()
        if len(words) < 100:
            return {"limit": len(words), "confidence": 0.1, "evidence": ["too_short"]}
        
        # Simple entropy analysis
        window_size = 50
        entropy_scores = []
        
        for i in range(0, len(words) - window_size + 1, window_size):
            window = words[i:i + window_size]
            word_counts = Counter(window)
            
            entropy = 0.0
            for count in word_counts.values():
                prob = count / len(window)
                entropy -= prob * math.log2(prob)
            
            entropy_scores.append(entropy)
        
        if len(entropy_scores) < 3:
            return {"limit": len(words), "confidence": 0.2, "evidence": ["insufficient_segments"]}
        
        # Find significant entropy drop
        initial_entropy = np.mean(entropy_scores[:2])
        for i, entropy in enumerate(entropy_scores[2:], 2):
            if entropy < initial_entropy * 0.7:  # 30% drop
                estimated_limit = i * window_size
                confidence = (initial_entropy - entropy) / initial_entropy
                return {
                    "limit": estimated_limit,
                    "confidence": min(confidence, 1.0),
                    "evidence": [f"entropy_drop_at_{i}"]
                }
        
        return {"limit": len(words), "confidence": 0.3, "evidence": ["no_entropy_drop"]}
    
    def _convert_to_token_estimate(self, word_limit: int) -> Dict[str, int]:
        """Convert word-based limit to token estimate"""
        # Rough conversion ratios
        return {
            "estimated_tokens": int(word_limit * 1.3),  # Average ~1.3 tokens per word
            "conservative_tokens": int(word_limit * 1.5),  # Conservative estimate
            "optimistic_tokens": int(word_limit * 1.1)   # Optimistic estimate
        }
    
    def _calculate_context_efficiency(self, text: str, quality_curve: List[float]) -> Dict[str, float]:
        """Calculate context usage efficiency metrics"""
        if not quality_curve:
            return {"efficiency_score": 0.0, "quality_per_token": 0.0, "optimal_length": 0}
        
        # Quality per unit length
        total_quality = sum(quality_curve)
        text_length = len(text.split())
        quality_per_word = total_quality / text_length if text_length > 0 else 0.0
        
        # Find optimal length (where quality per word is maximized)
        cumulative_quality = np.cumsum(quality_curve)
        cumulative_length = [(i + 1) * (self.window_size // 2) for i in range(len(quality_curve))]
        
        efficiency_ratios = [qual / length for qual, length in zip(cumulative_quality, cumulative_length)]
        optimal_index = np.argmax(efficiency_ratios)
        optimal_length = cumulative_length[optimal_index]
        
        # Overall efficiency score
        efficiency_score = max(efficiency_ratios) if efficiency_ratios else 0.0
        
        return {
            "efficiency_score": float(efficiency_score),
            "quality_per_word": float(quality_per_word),
            "optimal_length": optimal_length,
            "current_efficiency": float(quality_per_word)
        }
    
    def _analyze_length_patterns(self, text: str, quality_curve: List[float]) -> Dict[str, Any]:
        """Analyze patterns related to response length"""
        word_count = len(text.split())
        
        # Length categorization
        if word_count < 100:
            length_category = "short"
        elif word_count < 500:
            length_category = "medium"
        elif word_count < 1000:
            length_category = "long"
        else:
            length_category = "very_long"
        
        # Quality stability across length
        if len(quality_curve) >= 3:
            quality_stability = 1.0 - np.std(quality_curve)
            early_quality = np.mean(quality_curve[:len(quality_curve)//3])
            late_quality = np.mean(quality_curve[-len(quality_curve)//3:])
            quality_retention = late_quality / early_quality if early_quality > 0 else 1.0
        else:
            quality_stability = 1.0
            quality_retention = 1.0
        
        return {
            "length_category": length_category,
            "word_count": word_count,
            "quality_stability": float(max(quality_stability, 0.0)),
            "quality_retention": float(min(quality_retention, 2.0)),  # Cap at 2x
            "length_efficiency": self._assess_length_efficiency(length_category, quality_curve)
        }
    
    def _assess_length_efficiency(self, length_category: str, quality_curve: List[float]) -> str:
        """Assess efficiency of response length"""
        if not quality_curve:
            return "unknown"
        
        avg_quality = np.mean(quality_curve)
        
        if length_category == "short" and avg_quality > 0.7:
            return "efficient"
        elif length_category == "medium" and avg_quality > 0.6:
            return "appropriate"
        elif length_category == "long" and avg_quality > 0.5:
            return "adequate"
        elif length_category == "very_long" and avg_quality > 0.4:
            return "verbose_but_acceptable"
        else:
            return "inefficient"
    
    def _calculate_context_health_score(self, position_analysis: Dict, saturation_analysis: Dict, 
                                      degradation_points: List[Dict]) -> float:
        """Calculate overall context health score"""
        try:
            # Base score from position analysis
            quality_curve = position_analysis.get("quality_curve", [])
            avg_quality = np.mean(quality_curve) if quality_curve else 0.5
            
            # Penalties for issues
            saturation_penalty = 0.0
            if saturation_analysis.get("saturation_detected"):
                saturation_penalty = saturation_analysis.get("saturation_score", 0.0) * 0.3
            
            degradation_penalty = 0.0
            if degradation_points:
                severe_count = sum(1 for dp in degradation_points if dp.get("severity") == "severe")
                degradation_penalty = min(severe_count * 0.2, 0.4)
            
            # Calculate final health score
            health_score = avg_quality - saturation_penalty - degradation_penalty
            return max(0.0, min(health_score, 1.0))
            
        except Exception:
            return 0.5  # Neutral score on error
    
    def _single_segment_analysis(self, text: str, metrics_calculator: Optional[Callable]) -> Dict[str, Any]:
        """Handle analysis for text too short for position-based analysis"""
        segment_analysis = self._analyze_segment(
            {
                "start_token": 0,
                "end_token": len(text.split()),
                "start_char": 0,
                "end_char": len(text),
                "text": text,
                "token_count": len(text.split()),
                "char_count": len(text)
            },
            0, 1, metrics_calculator
        )
        
        return {
            "total_tokens": len(text.split()),
            "total_segments": 1,
            "segment_size": len(text.split()),
            "position_metrics": [segment_analysis],
            "degradation_analysis": {"degradation_detected": False, "degradation_rate": 0.0},
            "trend_analysis": {},
            "quality_curve": [segment_analysis["quality_score"]],
            "note": "Text too short for position-based analysis"
        }
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for the text"""
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback estimation
        return int(len(text.split()) * 1.3)


# Convenience functions
def analyze_context_quality(text: str, window_size: int = 512) -> Dict[str, Any]:
    """Quick context quality analysis"""
    analyzer = ContextWindowAnalyzer(window_size=window_size)
    return analyzer.analyze_quality_by_position(text)


def detect_context_saturation(text: str) -> Dict[str, Any]:
    """Quick context saturation detection"""
    analyzer = ContextWindowAnalyzer()
    return analyzer.detect_context_saturation(text)


def estimate_context_limit(text: str) -> Dict[str, Any]:
    """Quick context limit estimation"""
    analyzer = ContextWindowAnalyzer()
    return analyzer.estimate_context_limit(text)


# Testing function
def run_context_tests() -> Dict[str, Any]:
    """Run tests to validate context analysis"""
    # Generate test cases with different patterns
    test_cases = {
        "short": "This is a short response that should not show degradation.",
        "repetitive": "The system works. " * 50,  # Repetitive pattern
        "degrading": "This is a comprehensive analysis of the problem. " + 
                    "The system shows some issues. " * 20 + 
                    "Error error error. " * 30,  # Quality degradation
        "stable_long": "This analysis covers multiple aspects. " + 
                      "Each section provides detailed insights. " * 100  # Long but stable
    }
    
    analyzer = ContextWindowAnalyzer(window_size=100)  # Smaller window for testing
    results = {}
    
    for test_name, test_text in test_cases.items():
        try:
            analysis = analyzer.comprehensive_context_analysis(test_text)
            results[test_name] = {
                "context_health": analysis.get("context_health_score", 0.0),
                "saturation_detected": analysis.get("saturation_analysis", {}).get("saturation_detected", False),
                "degradation_points": len(analysis.get("degradation_points", [])),
                "estimated_limit": analysis.get("limit_estimation", {}).get("estimated_limit", 0)
            }
        except Exception as e:
            results[test_name] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    test_results = run_context_tests()
    print("Context Analysis Test Results:")
    for test_name, result in test_results.items():
        print(f"{test_name}: {result}")