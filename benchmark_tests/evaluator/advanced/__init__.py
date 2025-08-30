"""
Advanced analysis systems for sophisticated evaluation metrics.

This module contains advanced analytical tools:
- Entropy and information theory calculations
- Semantic coherence analysis
- Context window analysis
- Consistency validation
- Quantization testing
"""

from .entropy_calculator import *
from .semantic_coherence import *
from .context_analyzer import *
from .consistency_validator import *
from .quantization_tester import *

__all__ = [
    # Entropy analysis
    'EntropyCalculator',
    'calculate_entropy',
    
    # Semantic coherence
    'SemanticCoherenceAnalyzer',
    'analyze_semantic_coherence',
    'measure_prompt_completion_coherence',
    
    # Context analysis
    'ContextWindowAnalyzer',
    'analyze_context_quality',
    'detect_context_saturation',
    'estimate_context_limit',
    
    # Consistency validation
    'ConsistencyValidator',
    'CrossPhrasingResult',
    
    # Quantization testing
    'QuantizationTester',
]