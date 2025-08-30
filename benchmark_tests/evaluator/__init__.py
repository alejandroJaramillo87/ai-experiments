"""
Comprehensive Evaluation System for Language Model Benchmarking.

This package provides a complete evaluation framework organized into specialized modules:

- core: Foundation evaluation systems and configuration
- subjects: Domain-specific evaluators (reasoning, creativity, language, etc.)
- cultural: Cultural authenticity and cross-cultural competence evaluation
- advanced: Sophisticated analysis tools (entropy, coherence, context)
- validation: Quality assurance and community oversight systems
- linguistics: Specialized linguistic analysis tools
- data: Data management and API integrations

Example Usage:
    from evaluator.subjects import UniversalEvaluator, ReasoningType
    from evaluator.cultural import CulturalAuthenticityAnalyzer
    from evaluator.advanced import EntropyCalculator
"""

# Core evaluation systems - import specific classes
from .core.domain_evaluator_base import BaseDomainEvaluator, MultiDimensionalEvaluator, EvaluationDimension, DomainEvaluationResult, CulturalContext
from .core.evaluation_aggregator import EvaluationAggregator, AggregatedEvaluationResult

# Subject-specific evaluators - import main classes  
from .subjects.reasoning_evaluator import UniversalEvaluator, evaluate_reasoning, ReasoningType, EvaluationMetrics, EvaluationResult
from .subjects.creativity_evaluator import CreativityEvaluator
from .subjects.language_evaluator import LanguageEvaluator

__version__ = "2.0.0"
__author__ = "AI Workstation Team"

# Convenient top-level exports for common use cases
__all__ = [
    # Core systems
    "BaseDomainEvaluator",
    "MultiDimensionalEvaluator",
    "EvaluationDimension", 
    "DomainEvaluationResult",
    "CulturalContext",
    "EvaluationAggregator",
    "AggregatedEvaluationResult",
    
    # Most commonly used evaluators
    "UniversalEvaluator",
    "CreativityEvaluator",
    "LanguageEvaluator",
    
    # Common data types
    "ReasoningType",
    "EvaluationMetrics", 
    "EvaluationResult",
    "evaluate_reasoning",
]