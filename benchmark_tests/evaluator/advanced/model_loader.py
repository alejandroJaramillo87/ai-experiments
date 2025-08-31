"""
Unified Model Loading Utility

Provides consistent embedding model loading across all evaluator modules
with proper error handling and fallback strategies.

Author: Claude Code
Version: 1.0.0 - Unified Model Loading Strategy
"""

import os
import logging
from typing import Tuple, Optional, Any
from enum import Enum

# Optional imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class ModelLoadingStrategy(Enum):
    """Available model loading strategies"""
    AUTO = "auto"           # Try embedding model, fallback if fails
    FORCE_EMBEDDING = "force_embedding"  # Always try to load embedding model
    FORCE_FALLBACK = "force_fallback"    # Always use fallback methods
    

class ModelLoadingResult:
    """Result container for model loading attempts"""
    def __init__(self, model: Optional[Any], status: str, message: str):
        self.model = model
        self.status = status  # 'success', 'fallback', 'error'
        self.message = message
        
    def is_success(self) -> bool:
        return self.status == 'success'
        
    def is_fallback(self) -> bool:
        return self.status == 'fallback'
        
    def is_error(self) -> bool:
        return self.status == 'error'


class EvaluatorConfig:
    """Configuration management for evaluator model loading"""
    
    @staticmethod
    def get_embedding_strategy() -> ModelLoadingStrategy:
        """Get embedding model loading strategy from environment"""
        env_strategy = os.getenv('EVALUATOR_EMBEDDING_STRATEGY', 'auto').lower()
        
        strategy_map = {
            'auto': ModelLoadingStrategy.AUTO,
            'force': ModelLoadingStrategy.FORCE_EMBEDDING,
            'fallback': ModelLoadingStrategy.FORCE_FALLBACK,
            'disable': ModelLoadingStrategy.FORCE_FALLBACK,  # alias
        }
        
        return strategy_map.get(env_strategy, ModelLoadingStrategy.AUTO)
    
    @staticmethod
    def force_cpu_mode() -> bool:
        """Check if CPU-only mode is forced via environment"""
        return os.getenv('EVALUATOR_FORCE_CPU', 'false').lower() in ('true', '1', 'yes')
    
    @staticmethod
    def get_default_embedding_model() -> str:
        """Get default embedding model name from environment"""
        return os.getenv('EVALUATOR_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    
    @staticmethod
    def semantic_analyzer_strategy() -> ModelLoadingStrategy:
        """Get specific strategy for semantic analyzer"""
        # Check for semantic analyzer specific override
        env_strategy = os.getenv('SEMANTIC_ANALYZER_EMBEDDING_STRATEGY', '').lower()
        if env_strategy:
            strategy_map = {
                'auto': ModelLoadingStrategy.AUTO,
                'force': ModelLoadingStrategy.FORCE_EMBEDDING, 
                'fallback': ModelLoadingStrategy.FORCE_FALLBACK,
            }
            return strategy_map.get(env_strategy, ModelLoadingStrategy.FORCE_FALLBACK)
        
        # Default to fallback for semantic analyzer (current stable behavior)
        return ModelLoadingStrategy.FORCE_FALLBACK


class UnifiedModelLoader:
    """
    Unified embedding model loader for all evaluator modules.
    
    Provides consistent model loading behavior across SemanticCoherenceAnalyzer,
    EntropyCalculator, and other modules that need embedding models.
    
    Configuration via environment variables:
    - EVALUATOR_EMBEDDING_STRATEGY: 'auto'|'force'|'fallback' (default: 'auto')
    - EVALUATOR_FORCE_CPU: 'true'|'false' (default: 'false') 
    - EVALUATOR_EMBEDDING_MODEL: model name (default: 'all-MiniLM-L6-v2')
    - SEMANTIC_ANALYZER_EMBEDDING_STRATEGY: override for semantic analyzer
    """
    
    @staticmethod
    def load_embedding_model(
        model_name: str = "all-MiniLM-L6-v2", 
        strategy: ModelLoadingStrategy = ModelLoadingStrategy.AUTO,
        force_cpu: bool = False
    ) -> ModelLoadingResult:
        """
        Load embedding model with unified strategy
        
        Args:
            model_name: Name of the embedding model to load
            strategy: Loading strategy (auto, force_embedding, force_fallback)
            force_cpu: Force model to load on CPU (avoid CUDA issues)
            
        Returns:
            ModelLoadingResult with model instance and status
        """
        # Handle force fallback strategy
        if strategy == ModelLoadingStrategy.FORCE_FALLBACK:
            logger.info("Using fallback methods by configuration (embedding model disabled)")
            return ModelLoadingResult(
                model=None, 
                status='fallback', 
                message="Fallback methods configured by user choice"
            )
        
        # Check if sentence transformers is available
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available, using fallback methods")
            return ModelLoadingResult(
                model=None,
                status='fallback', 
                message="sentence-transformers not installed"
            )
        
        # Attempt to load embedding model
        try:
            logger.info(f"Attempting to load embedding model: {model_name}")
            
            # Load model with device configuration
            if force_cpu:
                device = 'cpu'
                logger.info("Forcing embedding model to use CPU")
            else:
                device = None  # Let sentence-transformers choose
                
            model = SentenceTransformer(model_name, device=device)
            
            logger.info(f"Successfully loaded embedding model: {model_name}")
            return ModelLoadingResult(
                model=model,
                status='success',
                message=f"Embedding model {model_name} loaded successfully"
            )
            
        except Exception as e:
            error_msg = f"Failed to load embedding model {model_name}: {e}"
            
            if strategy == ModelLoadingStrategy.FORCE_EMBEDDING:
                # User explicitly wants embedding model, so this is an error
                logger.error(error_msg)
                return ModelLoadingResult(
                    model=None,
                    status='error',
                    message=error_msg
                )
            else:
                # AUTO strategy: fallback to non-embedding methods
                logger.warning(f"{error_msg}, falling back to keyword-based methods")
                return ModelLoadingResult(
                    model=None,
                    status='fallback',
                    message=f"Embedding model failed, using fallback: {str(e)}"
                )
    
    @staticmethod
    def get_strategy_from_config(use_embedding_models: bool = True) -> ModelLoadingStrategy:
        """
        Convert boolean configuration to strategy enum
        
        Args:
            use_embedding_models: Whether to attempt embedding model loading
            
        Returns:
            Appropriate ModelLoadingStrategy
        """
        if use_embedding_models:
            return ModelLoadingStrategy.AUTO
        else:
            return ModelLoadingStrategy.FORCE_FALLBACK


# Convenience functions for backwards compatibility
def load_embedding_model_unified(model_name: str = "all-MiniLM-L6-v2") -> Tuple[Optional[Any], str]:
    """
    Backwards compatible function returning (model, status) tuple
    
    Returns:
        Tuple of (model_instance, status_string)
    """
    result = UnifiedModelLoader.load_embedding_model(model_name)
    return result.model, result.status


def is_embedding_available() -> bool:
    """Check if embedding models are available for loading"""
    return SENTENCE_TRANSFORMERS_AVAILABLE