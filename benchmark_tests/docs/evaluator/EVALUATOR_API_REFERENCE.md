# Evaluator API Reference

**AI Workstation Benchmark Tests - Complete API Documentation**  
**Date:** September 1, 2024  
**Coverage:** Core evaluator APIs with usage examples, parameters, and return values

---

## Table of Contents

1. [Core Evaluation Pipeline](#core-evaluation-pipeline)
2. [Pattern-Based Evaluator](#pattern-based-evaluator)
3. [Enhanced Universal Evaluator](#enhanced-universal-evaluator)
4. [Cultural Authenticity Analyzer](#cultural-authenticity-analyzer)
5. [Advanced Analysis Components](#advanced-analysis-components)
6. [Validation Systems](#validation-systems)
7. [Error Handling & Fallbacks](#error-handling--fallbacks)

---

## Core Evaluation Pipeline

### `CognitiveEvaluationPipeline`

**Primary interface for comprehensive evaluation with automatic evaluator orchestration.**

#### Constructor

```python
class CognitiveEvaluationPipeline:
    def __init__(self):
        """
        Initialize evaluation pipeline with automatic evaluator detection.
        
        Automatically detects and initializes available sophisticated evaluators:
        - PatternBasedEvaluator
        - EnhancedUniversalEvaluator  
        - CulturalAuthenticityAnalyzer
        - BiasAnalysis components
        
        Gracefully handles missing dependencies with fallback mechanisms.
        """
```

#### Core Methods

##### `evaluate_response()`

```python
def evaluate_response(
    self,
    test_data: Dict[str, Any],
    response_text: str,
    model_id: str
) -> CognitiveEvaluationResult:
    """
    Comprehensive evaluation using all available evaluators.
    
    Args:
        test_data: Test metadata and configuration
            Required keys:
            - 'id': Test identifier
            - 'domain': Cognitive domain (reasoning/creativity/social/etc.)
            Optional keys:
            - 'prompt': Original test prompt
            - 'difficulty': Test difficulty level
            - 'expected_patterns': Expected behavioral patterns
            
        response_text: Model response to evaluate
        
        model_id: Identifier for the model being evaluated
    
    Returns:
        CognitiveEvaluationResult:
            - overall_score: float (0-100)
            - cognitive_domain: str  
            - cognitive_subscores: Dict[str, float]
            - behavioral_patterns: Optional[PatternAnalysisResult]
            - cultural_analysis: Optional[Dict[str, Any]]
            - bias_indicators: Optional[Dict[str, Any]]
            - confidence_score: float (0-1)
            - pattern_strength: float (0-1)
            - consistency_measure: float (0-1)
    
    Example:
        >>> pipeline = CognitiveEvaluationPipeline()
        >>> result = pipeline.evaluate_response(
        ...     test_data={
        ...         'id': 'test_001',
        ...         'domain': 'reasoning',
        ...         'prompt': 'Logic problem: If A then B...'
        ...     },
        ...     response_text="Based on the logical premises...",
        ...     model_id="test-model-v1"
        ... )
        >>> print(f"Score: {result.overall_score}")
        >>> print(f"Domain: {result.cognitive_domain}")
    """
```

#### Properties

```python
@property
def available_evaluators(self) -> List[str]:
    """List of successfully initialized evaluators"""

@property
def cognitive_mappings(self) -> Dict[str, Dict[str, Any]]:
    """Cognitive domain mapping configuration"""
```

---

## Pattern-Based Evaluator

### `PatternBasedEvaluator`

**Key insight-based evaluator focusing on behavioral patterns rather than absolute truth.**

#### Constructor

```python
class PatternBasedEvaluator:
    def __init__(self, 
                 min_confidence_threshold: float = 0.6,
                 pattern_library_size: int = 1000):
        """
        Initialize pattern-based evaluator.
        
        Args:
            min_confidence_threshold: Minimum confidence for pattern detection
            pattern_library_size: Size of behavioral pattern library
        """
```

#### Core Methods

##### `evaluate_patterns()`

```python
def evaluate_patterns(
    self,
    response_text: str,
    prompt: str,
    test_metadata: Dict[str, Any],
    model_id: str,
    comparison_responses: Optional[List[str]] = None
) -> PatternAnalysisResult:
    """
    Analyze behavioral patterns in model response.
    
    Args:
        response_text: Model response to analyze
        prompt: Original prompt that generated the response
        test_metadata: Test configuration and metadata
        model_id: Model identifier for pattern tracking
        comparison_responses: Optional list of responses for comparison
    
    Returns:
        PatternAnalysisResult:
            - response_consistency: float (0-1)
            - behavioral_signature: Dict[str, Any] with keys:
                - 'response_style': str (analytical/creative/balanced)
                - 'verbosity_level': str (concise/medium/verbose)  
                - 'repetition_tendency': float (0-3+)
                - 'vocabulary_richness': float (0-1)
            - pattern_adherence: float (0-1)
            - comparative_ranking: Optional[float] (0-1)
            - quality_indicators: Dict[str, float] with keys:
                - 'coherence_score': float (0-1)
                - 'fluency_score': float (0-1)
                - 'engagement_score': float (0-1)
    
    Example:
        >>> evaluator = PatternBasedEvaluator()
        >>> result = evaluator.evaluate_patterns(
        ...     response_text="The logical analysis shows...",
        ...     prompt="Analyze this logical problem",
        ...     test_metadata={'domain': 'reasoning'},
        ...     model_id='gpt-test'
        ... )
        >>> print(f"Consistency: {result.response_consistency}")
        >>> print(f"Style: {result.behavioral_signature['response_style']}")
    """
```

##### `detect_behavioral_signature()`

```python
def detect_behavioral_signature(self, response_text: str) -> Dict[str, Any]:
    """
    Extract distinctive behavioral patterns from response.
    
    Args:
        response_text: Text to analyze for behavioral patterns
    
    Returns:
        Dict with behavioral signature components:
        - response_style: Detected communication style
        - verbosity_level: Response length tendency  
        - structure_preference: Organizational patterns
        - complexity_handling: Approach to complex topics
    """
```

---

## Enhanced Universal Evaluator  

### `EnhancedUniversalEvaluator`

**Multi-tier evaluation with backward compatibility and advanced analytics.**

#### Constructor

```python
class EnhancedUniversalEvaluator(UniversalEvaluator):
    def __init__(self, enable_advanced_features: bool = True):
        """
        Initialize enhanced evaluator with optional advanced features.
        
        Args:
            enable_advanced_features: Enable sophisticated analysis features
        """
```

#### Core Methods

##### `evaluate_enhanced()`

```python
def evaluate_enhanced(
    self,
    prompt: str,
    response: str,
    reasoning_type: ReasoningType = ReasoningType.GENERAL,
    domain_context: Optional[str] = None,
    expected_elements: Optional[List[str]] = None
) -> EnhancedEvaluationResult:
    """
    Enhanced multi-tier evaluation with advanced analytics.
    
    Args:
        prompt: Original test prompt
        response: Model response to evaluate
        reasoning_type: Type of reasoning required (LOGICAL/CREATIVE/SOCIAL/etc.)
        domain_context: Specific domain context for specialized analysis
        expected_elements: Optional list of expected response elements
    
    Returns:
        EnhancedEvaluationResult:
            - base_metrics: EvaluationMetrics (backward compatibility)
            - enhanced_metrics: EnhancedEvaluationMetrics with:
                - exact_match_score: float (0-1)
                - partial_match_score: float (0-1) 
                - semantic_similarity_score: float (0-1)
                - domain_synthesis_score: float (0-1)
                - conceptual_creativity_score: float (0-1)
                - integration_quality: float (0-1)
                - cultural_depth_score: float (0-1)
            - analysis_confidence: float (0-1)
            - feature_weights: Dict[str, float]
    
    Example:
        >>> evaluator = EnhancedUniversalEvaluator()
        >>> result = evaluator.evaluate_enhanced(
        ...     prompt="Complex reasoning task...",
        ...     response="Multi-step analysis...",
        ...     reasoning_type=ReasoningType.LOGICAL,
        ...     domain_context="abstract_reasoning"
        ... )
        >>> print(f"Synthesis: {result.enhanced_metrics.domain_synthesis_score}")
    """
```

---

## Cultural Authenticity Analyzer

### `CulturalAuthenticityAnalyzer`

**Specialized analysis for cultural content and cross-cultural sensitivity.**

#### Constructor

```python
class CulturalAuthenticityAnalyzer:
    def __init__(self, cultural_database_path: Optional[str] = None):
        """
        Initialize cultural analysis with optional custom database.
        
        Args:
            cultural_database_path: Path to cultural pattern database
        """
```

#### Core Methods

##### `analyze_cultural_authenticity()`

```python
def analyze_cultural_authenticity(
    self,
    content: str,
    cultural_context: Dict[str, Any],
    reference_sources: Optional[List[str]] = None
) -> CulturalAnalysisResult:
    """
    Analyze cultural authenticity and sensitivity of content.
    
    Args:
        content: Text content to analyze
        cultural_context: Cultural context information:
            - 'region': Geographic/cultural region
            - 'tradition': Specific cultural tradition
            - 'language_family': Language family context
            - 'historical_period': Relevant time period
        reference_sources: Optional cultural reference sources
    
    Returns:
        CulturalAnalysisResult:
            - authenticity_score: float (0-1)
            - cultural_sensitivity_score: float (0-1)
            - tradition_accuracy_score: float (0-1)
            - bias_indicators: Dict[str, float]
            - cultural_elements_detected: List[str]
            - recommendations: List[str]
    
    Example:
        >>> analyzer = CulturalAuthenticityAnalyzer()
        >>> result = analyzer.analyze_cultural_authenticity(
        ...     content="Discussion of traditional practices...",
        ...     cultural_context={
        ...         'region': 'East Asian',
        ...         'tradition': 'Buddhist philosophy'
        ...     }
        ... )
        >>> print(f"Authenticity: {result.authenticity_score}")
    """
```

---

## Advanced Analysis Components

### `ConsistencyValidator`

**Cross-phrasing consistency analysis for quality assurance.**

```python
class ConsistencyValidator:
    def validate_consistency(
        self,
        original_response: str,
        rephrased_prompts: List[str],
        model_responses: List[str],
        consistency_threshold: float = 0.8
    ) -> ConsistencyResult:
        """
        Validate response consistency across prompt variations.
        
        Args:
            original_response: Baseline response
            rephrased_prompts: List of rephrased versions of original prompt
            model_responses: Corresponding model responses
            consistency_threshold: Minimum consistency score
        
        Returns:
            ConsistencyResult:
                - overall_consistency: float (0-1)
                - pairwise_similarities: Dict[int, float]
                - consistency_violations: List[Dict[str, Any]]
                - consistency_grade: str (EXCELLENT/GOOD/POOR/INCONSISTENT)
        """
```

### `SemanticCoherence`

**Semantic coherence and drift analysis.**

```python
class SemanticCoherence:
    def analyze_coherence(
        self,
        text: str,
        context_window: int = 3,
        coherence_metrics: List[str] = None
    ) -> CoherenceResult:
        """
        Analyze semantic coherence within text.
        
        Args:
            text: Text to analyze for coherence
            context_window: Window size for coherence analysis
            coherence_metrics: Specific metrics to compute
        
        Returns:
            CoherenceResult:
                - overall_coherence: float (0-1)
                - local_coherence: List[float] (sentence-level)
                - global_coherence: float (document-level)  
                - semantic_drift: float (0-1)
                - topic_consistency: float (0-1)
        """
```

---

## Error Handling & Fallbacks

### Exception Types

```python
class EvaluatorException(Exception):
    """Base exception for evaluator errors"""

class EvaluatorUnavailableException(EvaluatorException):
    """Raised when required evaluator is unavailable"""

class InvalidInputException(EvaluatorException):
    """Raised when input data is invalid"""

class EvaluationTimeoutException(EvaluatorException):  
    """Raised when evaluation times out"""
```

### Fallback Patterns

#### Graceful Degradation Example

```python
try:
    # Try sophisticated evaluation
    if self.pattern_evaluator:
        result = self.pattern_evaluator.evaluate_patterns(...)
    else:
        raise EvaluatorUnavailableException("Pattern evaluator not available")
        
except EvaluatorUnavailableException:
    # Fall back to basic evaluation
    result = self._evaluate_basic_patterns(...)
    logger.warning("Using basic pattern evaluation due to unavailable sophisticated evaluator")
    
except Exception as e:
    # Ultimate fallback
    result = self._create_minimal_result(...)
    logger.error(f"Evaluation failed, using minimal result: {e}")
```

#### Availability Checking

```python
def check_evaluator_availability() -> Dict[str, bool]:
    """
    Check availability of all evaluator components.
    
    Returns:
        Dict mapping evaluator names to availability status
    """
    availability = {}
    
    # Check pattern evaluator
    try:
        from evaluator.subjects.pattern_based_evaluator import PatternBasedEvaluator
        PatternBasedEvaluator()
        availability['pattern_based'] = True
    except ImportError:
        availability['pattern_based'] = False
    
    # Check other evaluators...
    
    return availability
```

---

## Performance Guidelines

### Optimization Tips

1. **Batch Processing**: Group similar evaluations for better performance
2. **Caching**: Cache expensive computations (cultural databases, pattern libraries)
3. **Selective Loading**: Only initialize needed evaluators for specific tasks
4. **Timeout Management**: Set appropriate timeouts for different evaluator complexities

### Resource Management

```python
# Example of efficient evaluator usage
class OptimizedEvaluationManager:
    def __init__(self):
        self._evaluator_cache = {}
    
    def get_evaluator(self, evaluator_type: str):
        """Lazy loading with caching"""
        if evaluator_type not in self._evaluator_cache:
            self._evaluator_cache[evaluator_type] = self._create_evaluator(evaluator_type)
        return self._evaluator_cache[evaluator_type]
```

---

## Integration Examples

### Complete Evaluation Workflow

```python
def complete_evaluation_example():
    """Example of complete evaluation workflow"""
    
    # Initialize pipeline
    pipeline = CognitiveEvaluationPipeline()
    
    # Check what's available
    print(f"Available evaluators: {pipeline.available_evaluators}")
    
    # Prepare test data
    test_data = {
        'id': 'comprehensive_test_001',
        'domain': 'reasoning',
        'difficulty': 'medium',
        'prompt': 'Complex logical reasoning task...'
    }
    
    response_text = "Detailed logical analysis with step-by-step reasoning..."
    
    # Run comprehensive evaluation
    result = pipeline.evaluate_response(
        test_data=test_data,
        response_text=response_text,
        model_id="production-model-v2"
    )
    
    # Access different result components
    print(f"Overall Score: {result.overall_score}")
    print(f"Confidence: {result.confidence_score}")
    
    if result.behavioral_patterns:
        print(f"Response Style: {result.behavioral_patterns.behavioral_signature['response_style']}")
        print(f"Consistency: {result.behavioral_patterns.response_consistency}")
    
    if result.cultural_analysis:
        print(f"Cultural Sensitivity: {result.cultural_analysis['sensitivity_score']}")
    
    return result
```

---

**This API reference provides comprehensive coverage of the evaluator framework. For implementation details and advanced usage patterns, refer to the source code and integration tests.**

---

**Document Version:** 1.0  
**Last Updated:** September 1, 2024  
**Next Review:** After API updates or major feature additions