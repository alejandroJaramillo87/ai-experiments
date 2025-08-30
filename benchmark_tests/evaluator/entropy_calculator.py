"""
Entropy Calculator Module

Advanced entropy calculations for language model evaluation including Shannon entropy
and semantic entropy measurements to assess response diversity and predictability.

This module addresses the critique's key point about missing true entropy measurements
beyond simple vocabulary diversity.

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
    logging.warning("sentence-transformers not available. Semantic entropy disabled.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available. Some semantic features disabled.")

# Set up logging
logger = logging.getLogger(__name__)


class EntropyCalculator:
    """
    Advanced entropy calculator for language model response analysis.
    
    Provides multiple entropy measurements:
    - Shannon entropy at token level
    - Semantic entropy via embeddings
    - Vocabulary diversity entropy
    - N-gram entropy analysis
    """
    
    def __init__(self, model_name: str = "gpt-4", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize entropy calculator with specified models
        
        Args:
            model_name: Tokenizer model name (e.g., 'gpt-4', 'cl100k_base')
            embedding_model: Sentence transformer model for semantic entropy
        """
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        
        # Initialize tokenizer
        self.tokenizer = self._initialize_tokenizer(model_name)
        
        # Initialize embedding model (lazy loaded)
        self._embedding_model = None
        
        logger.info(f"EntropyCalculator initialized with {model_name} tokenizer")
    
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
    
    def calculate_shannon_entropy(self, text: str, use_tokens: bool = True) -> float:
        """
        Calculate Shannon entropy of text using proper tokenization
        
        Args:
            text: Input text to analyze
            use_tokens: If True, use tiktoken tokenization; otherwise use words
            
        Returns:
            Shannon entropy value (bits)
        """
        if not text or not text.strip():
            return 0.0
        
        if use_tokens and self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text)
                if not tokens:
                    return 0.0
                
                # Calculate token frequency distribution
                token_counts = Counter(tokens)
                total_tokens = len(tokens)
                
                # Calculate Shannon entropy
                entropy = 0.0
                for count in token_counts.values():
                    probability = count / total_tokens
                    entropy -= probability * math.log2(probability)
                
                return entropy
                
            except Exception as e:
                logger.warning(f"Token-based entropy calculation failed: {e}")
                # Fallback to word-based calculation
        
        # Fallback: word-based entropy calculation
        words = self._tokenize_words(text)
        if not words:
            return 0.0
        
        word_counts = Counter(words)
        total_words = len(words)
        
        entropy = 0.0
        for count in word_counts.values():
            probability = count / total_words
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def calculate_semantic_entropy(self, text: str, window_size: int = 100) -> Dict[str, float]:
        """
        Calculate semantic entropy using embedding-based analysis
        
        Args:
            text: Input text to analyze
            window_size: Size of semantic windows in tokens
            
        Returns:
            Dictionary with semantic entropy metrics
        """
        if not text or not text.strip():
            return {"semantic_entropy": 0.0, "semantic_diversity": 0.0, "embedding_variance": 0.0}
        
        if not self.embedding_model:
            # Fallback to TF-IDF based semantic analysis
            return self._calculate_tfidf_semantic_entropy(text)
        
        try:
            # Split text into semantic chunks
            sentences = self._split_into_sentences(text)
            if len(sentences) < 2:
                return {"semantic_entropy": 0.0, "semantic_diversity": 0.0, "embedding_variance": 0.0}
            
            # Generate embeddings for each sentence
            embeddings = self.embedding_model.encode(sentences, convert_to_tensor=True)
            
            # Calculate pairwise cosine similarities
            similarities = cosine_similarity(embeddings.cpu().numpy())
            
            # Remove self-similarities (diagonal)
            np.fill_diagonal(similarities, 0)
            
            # Calculate semantic diversity (1 - average similarity)
            avg_similarity = np.mean(similarities)
            semantic_diversity = 1.0 - avg_similarity
            
            # Calculate embedding variance
            embedding_variance = np.var(embeddings.cpu().numpy())
            
            # Calculate semantic entropy based on similarity distribution
            similarity_flat = similarities.flatten()
            similarity_flat = similarity_flat[similarity_flat > 0]  # Remove zeros
            
            if len(similarity_flat) == 0:
                semantic_entropy = 0.0
            else:
                # Bin similarities and calculate entropy
                hist, _ = np.histogram(similarity_flat, bins=10, range=(0, 1))
                hist = hist / np.sum(hist)  # Normalize to probabilities
                hist = hist[hist > 0]  # Remove zero probabilities
                
                semantic_entropy = -np.sum(hist * np.log2(hist))
            
            return {
                "semantic_entropy": float(semantic_entropy),
                "semantic_diversity": float(semantic_diversity),
                "embedding_variance": float(embedding_variance),
                "sentence_count": len(sentences),
                "avg_similarity": float(avg_similarity)
            }
            
        except Exception as e:
            logger.error(f"Semantic entropy calculation failed: {e}")
            return {"semantic_entropy": 0.0, "semantic_diversity": 0.0, "embedding_variance": 0.0}
    
    def calculate_ngram_entropy(self, text: str, n: int = 2) -> float:
        """
        Calculate n-gram entropy to measure local predictability patterns
        
        Args:
            text: Input text to analyze
            n: N-gram size (2 for bigrams, 3 for trigrams)
            
        Returns:
            N-gram entropy value
        """
        if not text or not text.strip():
            return 0.0
        
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text)
                elements = tokens
            except Exception:
                elements = self._tokenize_words(text)
        else:
            elements = self._tokenize_words(text)
        
        if len(elements) < n:
            return 0.0
        
        # Generate n-grams
        ngrams = []
        for i in range(len(elements) - n + 1):
            ngram = tuple(elements[i:i + n])
            ngrams.append(ngram)
        
        if not ngrams:
            return 0.0
        
        # Calculate n-gram frequency distribution
        ngram_counts = Counter(ngrams)
        total_ngrams = len(ngrams)
        
        # Calculate entropy
        entropy = 0.0
        for count in ngram_counts.values():
            probability = count / total_ngrams
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def calculate_vocabulary_entropy(self, text: str) -> Dict[str, float]:
        """
        Calculate vocabulary-based entropy metrics
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with vocabulary entropy metrics
        """
        if not text or not text.strip():
            return {"vocab_entropy": 0.0, "vocab_diversity": 0.0, "unique_ratio": 0.0, "unique_words": 0}
        
        words = self._tokenize_words(text)
        if not words:
            return {"vocab_entropy": 0.0, "vocab_diversity": 0.0, "unique_ratio": 0.0, "unique_words": 0}
        
        word_counts = Counter(words)
        total_words = len(words)
        unique_words = len(word_counts)
        
        # Calculate vocabulary entropy
        vocab_entropy = 0.0
        for count in word_counts.values():
            probability = count / total_words
            vocab_entropy -= probability * math.log2(probability)
        
        # Calculate vocabulary diversity (normalized entropy)
        max_entropy = math.log2(unique_words) if unique_words > 1 else 1.0
        vocab_diversity = vocab_entropy / max_entropy
        
        # Calculate unique word ratio
        unique_ratio = unique_words / total_words
        
        return {
            "vocab_entropy": vocab_entropy,
            "vocab_diversity": vocab_diversity,
            "unique_ratio": unique_ratio,
            "unique_words": unique_words,
            "total_words": total_words
        }
    
    def analyze_entropy_profile(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive entropy analysis providing full entropy profile
        
        Args:
            text: Input text to analyze
            
        Returns:
            Complete entropy analysis dictionary
        """
        if not text or not text.strip():
            return self._empty_entropy_profile()
        
        # Calculate all entropy metrics
        shannon_entropy = self.calculate_shannon_entropy(text, use_tokens=True)
        shannon_entropy_words = self.calculate_shannon_entropy(text, use_tokens=False)
        semantic_metrics = self.calculate_semantic_entropy(text)
        bigram_entropy = self.calculate_ngram_entropy(text, n=2)
        trigram_entropy = self.calculate_ngram_entropy(text, n=3)
        vocab_metrics = self.calculate_vocabulary_entropy(text)
        
        # Calculate entropy quality ratio
        word_count = len(text.split())
        entropy_quality_ratio = shannon_entropy / max(math.log2(word_count), 1.0) if word_count > 0 else 0.0
        
        # Detect entropy patterns
        entropy_patterns = self._analyze_entropy_patterns(text)
        
        return {
            "token_entropy": shannon_entropy,
            "word_entropy": shannon_entropy_words,
            "entropy_quality_ratio": entropy_quality_ratio,
            "semantic_entropy": semantic_metrics["semantic_entropy"],
            "semantic_diversity": semantic_metrics["semantic_diversity"],
            "embedding_variance": semantic_metrics["embedding_variance"],
            "bigram_entropy": bigram_entropy,
            "trigram_entropy": trigram_entropy,
            "vocab_entropy": vocab_metrics["vocab_entropy"],
            "vocab_diversity": vocab_metrics["vocab_diversity"],
            "unique_ratio": vocab_metrics["unique_ratio"],
            "entropy_patterns": entropy_patterns,
            "text_length": len(text),
            "word_count": word_count,
            "unique_words": vocab_metrics["unique_words"]
        }
    
    def compare_entropy_profiles(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Compare entropy profiles between two texts
        
        Args:
            text1: First text to compare
            text2: Second text to compare
            
        Returns:
            Dictionary with entropy comparison metrics
        """
        profile1 = self.analyze_entropy_profile(text1)
        profile2 = self.analyze_entropy_profile(text2)
        
        # Calculate differences for key metrics
        comparisons = {}
        key_metrics = [
            "token_entropy", "semantic_entropy", "semantic_diversity",
            "vocab_entropy", "vocab_diversity", "entropy_quality_ratio"
        ]
        
        for metric in key_metrics:
            val1 = profile1.get(metric, 0.0)
            val2 = profile2.get(metric, 0.0)
            comparisons[f"{metric}_diff"] = val1 - val2
            comparisons[f"{metric}_ratio"] = val2 / max(val1, 0.001)
        
        # Overall entropy similarity score
        entropy_similarity = self._calculate_entropy_similarity(profile1, profile2)
        comparisons["entropy_similarity"] = entropy_similarity
        
        return comparisons
    
    # Private helper methods
    
    def _tokenize_words(self, text: str) -> List[str]:
        """Simple word tokenization fallback"""
        # Basic word tokenization with some preprocessing
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        return [word for word in words if len(word) > 1]  # Filter very short words
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for semantic analysis"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return sentences
    
    def _calculate_tfidf_semantic_entropy(self, text: str) -> Dict[str, float]:
        """Fallback semantic entropy using TF-IDF when embeddings unavailable"""
        if not SKLEARN_AVAILABLE:
            return {"semantic_entropy": 0.0, "semantic_diversity": 0.0, "embedding_variance": 0.0}
        
        try:
            sentences = self._split_into_sentences(text)
            if len(sentences) < 2:
                return {"semantic_entropy": 0.0, "semantic_diversity": 0.0, "embedding_variance": 0.0}
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate similarities
            similarities = cosine_similarity(tfidf_matrix)
            np.fill_diagonal(similarities, 0)
            
            avg_similarity = np.mean(similarities)
            semantic_diversity = 1.0 - avg_similarity
            
            # Simple variance calculation
            embedding_variance = np.var(tfidf_matrix.toarray())
            
            # Calculate entropy from similarity distribution
            similarity_flat = similarities.flatten()
            similarity_flat = similarity_flat[similarity_flat > 0]
            
            if len(similarity_flat) > 0:
                hist, _ = np.histogram(similarity_flat, bins=10)
                hist = hist / np.sum(hist)
                hist = hist[hist > 0]
                semantic_entropy = -np.sum(hist * np.log2(hist))
            else:
                semantic_entropy = 0.0
            
            return {
                "semantic_entropy": float(semantic_entropy),
                "semantic_diversity": float(semantic_diversity),
                "embedding_variance": float(embedding_variance)
            }
            
        except Exception as e:
            logger.error(f"TF-IDF semantic entropy calculation failed: {e}")
            return {"semantic_entropy": 0.0, "semantic_diversity": 0.0, "embedding_variance": 0.0}
    
    def _analyze_entropy_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze entropy patterns within the text"""
        patterns = {
            "has_repetitive_patterns": False,
            "entropy_variance": 0.0,
            "local_entropy_drops": 0,
            "entropy_trend": "stable"
        }
        
        try:
            # Split text into chunks and calculate local entropy
            words = text.split()
            if len(words) < 6:  # Too short for any pattern analysis
                return patterns
            
            # For very short texts, use simple repetition detection
            if len(words) < 20:
                # Normalize words (remove punctuation and lowercase)
                import re
                normalized_words = [re.sub(r'[^\w]', '', word.lower()) for word in words]
                
                word_counts = {}
                for word in normalized_words:
                    if word:  # Skip empty strings after punctuation removal
                        word_counts[word] = word_counts.get(word, 0) + 1
                
                # Check if any word repeats more than 25% of the time (lowered threshold)
                if word_counts:
                    max_freq = max(word_counts.values())
                    if max_freq / len(normalized_words) > 0.25:
                        patterns["has_repetitive_patterns"] = True
                        patterns["entropy_variance"] = 0.5  # Mark as having some variance
                        
                # Also check for repeated phrases (simple bigram check)
                bigrams = []
                for i in range(len(normalized_words) - 1):
                    if normalized_words[i] and normalized_words[i + 1]:
                        bigrams.append(f"{normalized_words[i]} {normalized_words[i + 1]}")
                
                if bigrams:
                    bigram_counts = {}
                    for bigram in bigrams:
                        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
                    
                    max_bigram_freq = max(bigram_counts.values()) if bigram_counts else 0
                    if max_bigram_freq >= 2:  # Same bigram appears at least twice
                        patterns["has_repetitive_patterns"] = True
                        patterns["entropy_variance"] = 0.5
                
                return patterns
            
            chunk_size = max(20, len(words) // 10)  # At least 20 words per chunk
            local_entropies = []
            
            for i in range(0, len(words) - chunk_size + 1, chunk_size // 2):
                chunk = " ".join(words[i:i + chunk_size])
                chunk_entropy = self.calculate_shannon_entropy(chunk, use_tokens=False)
                local_entropies.append(chunk_entropy)
            
            if len(local_entropies) >= 3:
                patterns["entropy_variance"] = float(np.var(local_entropies))
                
                # Detect entropy drops (signs of repetition or degradation)
                drops = 0
                for i in range(1, len(local_entropies)):
                    if local_entropies[i] < local_entropies[i-1] * 0.8:  # 20% drop
                        drops += 1
                patterns["local_entropy_drops"] = drops
                
                # Determine overall trend
                if len(local_entropies) >= 3:
                    first_third = np.mean(local_entropies[:len(local_entropies)//3])
                    last_third = np.mean(local_entropies[-len(local_entropies)//3:])
                    
                    if last_third < first_third * 0.9:
                        patterns["entropy_trend"] = "decreasing"
                    elif last_third > first_third * 1.1:
                        patterns["entropy_trend"] = "increasing"
                    else:
                        patterns["entropy_trend"] = "stable"
                
                # Check for repetitive patterns using multiple criteria
                chunk_based_repetition = (
                    patterns["entropy_variance"] < 0.1 and 
                    patterns["local_entropy_drops"] >= 2
                )
                
                # Also check word-level repetition for confirmation
                import re
                normalized_words = [re.sub(r'[^\w]', '', word.lower()) for word in words]
                word_counts = {}
                for word in normalized_words:
                    if word:
                        word_counts[word] = word_counts.get(word, 0) + 1
                
                word_based_repetition = False
                if word_counts:
                    max_freq = max(word_counts.values())
                    word_based_repetition = max_freq / len(normalized_words) > 0.2
                
                # Check for repeated phrases (bigrams)
                bigrams = []
                for i in range(len(normalized_words) - 1):
                    if normalized_words[i] and normalized_words[i + 1]:
                        bigrams.append(f"{normalized_words[i]} {normalized_words[i + 1]}")
                
                phrase_based_repetition = False
                if bigrams:
                    bigram_counts = {}
                    for bigram in bigrams:
                        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
                    max_bigram_freq = max(bigram_counts.values()) if bigram_counts else 0
                    phrase_based_repetition = max_bigram_freq >= 3  # Higher threshold for longer texts
                
                # Debug: explicit boolean evaluation  
                has_repetitive = chunk_based_repetition or word_based_repetition or phrase_based_repetition
                patterns["has_repetitive_patterns"] = has_repetitive
            else:
                # Not enough chunks for entropy-based analysis, use word/phrase analysis
                import re
                normalized_words = [re.sub(r'[^\w]', '', word.lower()) for word in words]
                word_counts = {}
                for word in normalized_words:
                    if word:
                        word_counts[word] = word_counts.get(word, 0) + 1
                
                word_based_repetition = False
                if word_counts:
                    max_freq = max(word_counts.values())
                    word_based_repetition = max_freq / len(normalized_words) > 0.2
                
                # Check for repeated phrases (bigrams)
                bigrams = []
                for i in range(len(normalized_words) - 1):
                    if normalized_words[i] and normalized_words[i + 1]:
                        bigrams.append(f"{normalized_words[i]} {normalized_words[i + 1]}")
                
                phrase_based_repetition = False
                if bigrams:
                    bigram_counts = {}
                    for bigram in bigrams:
                        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
                    max_bigram_freq = max(bigram_counts.values()) if bigram_counts else 0
                    phrase_based_repetition = max_bigram_freq >= 2  # Lower threshold for fewer chunks
                
                patterns["has_repetitive_patterns"] = word_based_repetition or phrase_based_repetition
        
        except Exception as e:
            logger.error(f"Entropy pattern analysis failed: {e}")
        
        return patterns
    
    def _calculate_entropy_similarity(self, profile1: Dict, profile2: Dict) -> float:
        """Calculate similarity between two entropy profiles"""
        key_metrics = ["token_entropy", "semantic_entropy", "vocab_entropy", "semantic_diversity"]
        
        similarities = []
        for metric in key_metrics:
            val1 = profile1.get(metric, 0.0)
            val2 = profile2.get(metric, 0.0)
            
            if val1 == 0.0 and val2 == 0.0:
                similarities.append(1.0)
            elif val1 == 0.0 or val2 == 0.0:
                similarities.append(0.0)
            else:
                # Calculate relative similarity
                ratio = min(val1, val2) / max(val1, val2)
                similarities.append(ratio)
        
        return float(np.mean(similarities))
    
    def _empty_entropy_profile(self) -> Dict[str, Any]:
        """Return empty entropy profile for invalid inputs"""
        return {
            "token_entropy": 0.0,
            "word_entropy": 0.0,
            "entropy_quality_ratio": 0.0,
            "semantic_entropy": 0.0,
            "semantic_diversity": 0.0,
            "embedding_variance": 0.0,
            "bigram_entropy": 0.0,
            "trigram_entropy": 0.0,
            "vocab_entropy": 0.0,
            "vocab_diversity": 0.0,
            "unique_ratio": 0.0,
            "entropy_patterns": {},
            "text_length": 0,
            "word_count": 0,
            "unique_words": 0
        }


# Convenience functions for easy integration
def calculate_shannon_entropy(text: str, model_name: str = "cl100k_base") -> float:
    """
    Quick Shannon entropy calculation
    
    Args:
        text: Text to analyze
        model_name: Tokenizer model name
        
    Returns:
        Shannon entropy value
    """
    calculator = EntropyCalculator(model_name=model_name)
    return calculator.calculate_shannon_entropy(text)


def analyze_text_entropy(text: str, model_name: str = "cl100k_base") -> Dict[str, Any]:
    """
    Quick comprehensive entropy analysis
    
    Args:
        text: Text to analyze
        model_name: Tokenizer model name
        
    Returns:
        Complete entropy profile
    """
    calculator = EntropyCalculator(model_name=model_name)
    return calculator.analyze_entropy_profile(text)


# Testing function for validation
def run_entropy_tests() -> Dict[str, Any]:
    """
    Run basic tests to validate entropy calculations
    
    Returns:
        Test results dictionary
    """
    test_cases = {
        "empty": "",
        "single_word": "Hello",
        "repetitive": "the the the the the the",
        "diverse": "The quick brown fox jumps over the lazy dog with remarkable agility and grace.",
        "technical": "Algorithm optimization requires careful analysis of computational complexity, memory allocation, and performance bottlenecks in distributed systems.",
        "creative": "In the ethereal twilight, shadows danced across the cobblestone streets while mysterious figures whispered secrets beneath the ancient oak trees."
    }
    
    calculator = EntropyCalculator()
    results = {}
    
    for test_name, test_text in test_cases.items():
        try:
            profile = calculator.analyze_entropy_profile(test_text)
            results[test_name] = {
                "token_entropy": profile["token_entropy"],
                "semantic_diversity": profile["semantic_diversity"],
                "vocab_diversity": profile["vocab_diversity"],
                "entropy_quality_ratio": profile["entropy_quality_ratio"]
            }
        except Exception as e:
            results[test_name] = {"error": str(e)}
    
    return results


if __name__ == "__main__":
    # Run tests if executed directly
    test_results = run_entropy_tests()
    print("Entropy Calculator Test Results:")
    for test_name, result in test_results.items():
        print(f"{test_name}: {result}")