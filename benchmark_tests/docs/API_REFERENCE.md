# API Reference

This guide provides comprehensive documentation of classes, methods, and interfaces in the AI Model Evaluation Framework. Think of this as your **technical reference manual** - detailed information for developers working with the codebase.

## 📚 **Module Overview**

```
evaluator/
├── core/              # Base classes and interfaces
├── subjects/          # Domain-specific evaluators
├── validation/        # Quality assurance systems  
├── advanced/          # Sophisticated analysis tools
├── cultural/          # Cultural context integration
├── linguistics/       # Language-specific analysis
└── data/              # Evaluator-specific data
```

## 🏗️ **Core Classes and Interfaces**

### **BaseEvaluator** (`evaluator.core.domain_evaluator_base`)

The abstract base class that all evaluators must inherit from.

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class BaseEvaluator(ABC):
    """
    Abstract base class for all evaluators.
    
    Provides the contract that all evaluators must implement
    and common functionality shared across evaluators.
    """
    
    def __init__(self):
        self.evaluator_name: str = "base"
        self.version: str = "1.0.0"
        self.settings: Dict[str, Any] = {}
    
    @abstractmethod
    def evaluate(self, text: str, context: Dict[str, Any]) -> 'DomainEvaluationResult':
        """
        Evaluate the given text and return a structured result.
        
        Args:
            text: The model response to evaluate
            context: Test metadata and cultural context information
            
        Returns:
            DomainEvaluationResult with scores and analysis
            
        Raises:
            EvaluationError: If evaluation fails
        """
        pass
    
    @abstractmethod 
    def get_dimensions(self) -> List[str]:
        """
        Return the list of dimensions this evaluator measures.
        
        Returns:
            List of dimension names (e.g., ["accuracy", "completeness"])
        """
        pass
    
    def configure(self, settings: Dict[str, Any]) -> None:
        """
        Configure the evaluator with custom settings.
        
        Args:
            settings: Configuration dictionary
            
        Raises:
            ConfigurationError: If settings are invalid
        """
        self.settings.update(settings)
        self._validate_settings()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get evaluator metadata including name, version, settings."""
        return {
            "name": self.evaluator_name,
            "version": self.version,
            "settings": self.settings.copy()
        }
    
    def _validate_settings(self) -> None:
        """Validate current settings. Override in subclasses."""
        pass
```

---

### **DomainEvaluationResult** (`evaluator.core.domain_evaluator_base`)

The result object returned by all evaluators.

```python
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class DomainEvaluationResult:
    """
    Result from a domain-specific evaluator.
    
    Provides structured access to evaluation scores, cultural context,
    and supporting evidence.
    """
    
    domain: str                              # Domain name (e.g., "reasoning")
    evaluation_type: str                     # Evaluation type (e.g., "comprehensive") 
    overall_score: float                     # Overall score (0.0 to 1.0)
    dimensions: List['EvaluationDimension']  # Individual dimension scores
    cultural_context: 'CulturalContext'      # Cultural context information
    metadata: Dict[str, Any]                 # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "domain": self.domain,
            "evaluation_type": self.evaluation_type,
            "overall_score": self.overall_score,
            "dimensions": [dim.to_dict() for dim in self.dimensions],
            "cultural_context": self.cultural_context.to_dict(),
            "metadata": self.metadata
        }
    
    def get_dimension_score(self, dimension_name: str) -> Optional[float]:
        """Get score for a specific dimension."""
        for dim in self.dimensions:
            if dim.name == dimension_name:
                return dim.score
        return None
    
    def get_strengths(self, threshold: float = 0.8) -> List[str]:
        """Get dimensions where the model performed well."""
        return [dim.name for dim in self.dimensions if dim.score >= threshold]
    
    def get_weaknesses(self, threshold: float = 0.6) -> List[str]:
        """Get dimensions where the model needs improvement."""
        return [dim.name for dim in self.dimensions if dim.score < threshold]
```

---

### **EvaluationDimension** (`evaluator.core.domain_evaluator_base`)

Represents a single evaluation dimension with detailed scoring information.

```python
@dataclass
class EvaluationDimension:
    """
    Represents a single evaluation dimension with score and cultural context.
    
    Each dimension provides detailed scoring with supporting evidence
    and cultural relevance assessment.
    """
    
    name: str                    # Dimension name (e.g., "technical_accuracy")
    score: float                 # Score (0.0 to 1.0)
    confidence: float            # Confidence in score (0.0 to 1.0)
    cultural_relevance: float    # Cultural relevance (0.0 to 1.0)  
    evidence: List[str]          # Supporting evidence/examples
    cultural_markers: List[str]  # Detected cultural patterns
    
    def __post_init__(self):
        """Validate and normalize scores after initialization."""
        self.score = max(0.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.cultural_relevance = max(0.0, min(1.0, self.cultural_relevance))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dimension to dictionary format."""
        return {
            "name": self.name,
            "score": self.score,
            "confidence": self.confidence, 
            "cultural_relevance": self.cultural_relevance,
            "evidence": self.evidence,
            "cultural_markers": self.cultural_markers
        }
    
    def is_reliable(self, min_confidence: float = 0.7) -> bool:
        """Check if this dimension's score is reliable."""
        return self.confidence >= min_confidence
    
    def has_cultural_context(self) -> bool:
        """Check if this dimension includes cultural context."""
        return len(self.cultural_markers) > 0 or self.cultural_relevance > 0.0
```

---

### **CulturalContext** (`evaluator.core.domain_evaluator_base`)

Represents cultural context information for evaluations.

```python
class CulturalContext:
    """
    Represents cultural context information extracted from test metadata.
    
    Provides structured access to cultural elements that may influence
    evaluation and scoring.
    """
    
    def __init__(self, 
                 traditions: List[str] = None,
                 knowledge_systems: List[str] = None,
                 performance_aspects: List[str] = None,
                 cultural_groups: List[str] = None,
                 linguistic_varieties: List[str] = None):
        self.traditions = traditions or []
        self.knowledge_systems = knowledge_systems or []
        self.performance_aspects = performance_aspects or []
        self.cultural_groups = cultural_groups or []
        self.linguistic_varieties = linguistic_varieties or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cultural context to dictionary format."""
        return {
            "traditions": self.traditions,
            "knowledge_systems": self.knowledge_systems,
            "performance_aspects": self.performance_aspects,
            "cultural_groups": self.cultural_groups,
            "linguistic_varieties": self.linguistic_varieties
        }
    
    def has_cultural_elements(self) -> bool:
        """Check if any cultural elements are present."""
        return any([
            self.traditions,
            self.knowledge_systems,
            self.performance_aspects,
            self.cultural_groups,
            self.linguistic_varieties
        ])
    
    def merge(self, other: 'CulturalContext') -> 'CulturalContext':
        """Merge with another cultural context."""
        return CulturalContext(
            traditions=list(set(self.traditions + other.traditions)),
            knowledge_systems=list(set(self.knowledge_systems + other.knowledge_systems)),
            performance_aspects=list(set(self.performance_aspects + other.performance_aspects)),
            cultural_groups=list(set(self.cultural_groups + other.cultural_groups)),
            linguistic_varieties=list(set(self.linguistic_varieties + other.linguistic_varieties))
        )
```

## 🧪 **Test Execution Framework**

### **BenchmarkTestRunner** (`benchmark_runner.py`)

The main test execution engine.

```python
class BenchmarkTestRunner:
    """
    Flexible test execution engine with concurrent processing capabilities.
    
    Handles test loading, execution, result processing, and performance monitoring
    for comprehensive model evaluation.
    """
    
    def __init__(self, 
                 model_endpoint: str,
                 model_name: str,
                 max_workers: int = 4,
                 performance_monitoring: bool = False):
        self.model_endpoint = model_endpoint
        self.model_name = model_name
        self.max_workers = max_workers
        self.performance_monitoring = performance_monitoring
        self._setup_logging()
        self._initialize_evaluators()
    
    def load_test_suite(self, 
                       test_type: str, 
                       category: Optional[str] = None,
                       difficulty: Optional[str] = None,
                       limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load test definitions based on criteria.
        
        Args:
            test_type: Type of tests (base, instruct, custom)
            category: Specific category to load
            difficulty: Filter by difficulty (easy, medium, hard)
            limit: Maximum number of tests to load
            
        Returns:
            List of test definitions
            
        Raises:
            TestLoadError: If tests cannot be loaded
        """
        pass
    
    def execute_single_test(self, 
                           test_id: str,
                           test_type: str = "base") -> 'TestResult':
        """
        Execute a single test and return results.
        
        Args:
            test_id: Unique identifier for the test
            test_type: Type of test to execute
            
        Returns:
            TestResult with execution results and evaluation
            
        Raises:
            TestExecutionError: If test execution fails
        """
        pass
    
    def execute_concurrent_tests(self, 
                                tests: List[Dict[str, Any]],
                                output_dir: str = "test_results") -> 'BatchTestResults':
        """
        Execute multiple tests concurrently.
        
        Args:
            tests: List of test definitions to execute
            output_dir: Directory to save results
            
        Returns:
            BatchTestResults with aggregated results
            
        Raises:
            BatchExecutionError: If batch execution fails
        """
        pass
    
    def get_performance_metrics(self) -> 'PerformanceMetrics':
        """Get current performance metrics."""
        pass
    
    def cleanup(self) -> None:
        """Clean up resources and temporary files."""
        pass
```

---

### **TestResult** (`benchmark_runner.py`)

Individual test execution result.

```python
@dataclass
class TestResult:
    """
    Result from executing a single test.
    
    Contains the model response, evaluation results, and execution metadata.
    """
    
    test_id: str                              # Test identifier
    test_metadata: Dict[str, Any]             # Original test definition
    model_response: str                       # Raw model response
    evaluation_results: List[DomainEvaluationResult]  # Evaluation results
    performance_metrics: PerformanceMetrics  # Execution performance data
    timestamp: datetime                       # Execution timestamp
    success: bool                            # Whether execution succeeded
    error_message: Optional[str] = None      # Error message if failed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format for serialization."""
        return {
            "test_id": self.test_id,
            "test_metadata": self.test_metadata,
            "model_response": self.model_response,
            "evaluation_results": [result.to_dict() for result in self.evaluation_results],
            "performance_metrics": asdict(self.performance_metrics),
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "error_message": self.error_message
        }
    
    def get_overall_score(self) -> float:
        """Calculate overall score across all evaluations."""
        if not self.evaluation_results:
            return 0.0
        return sum(result.overall_score for result in self.evaluation_results) / len(self.evaluation_results)
    
    def get_dimension_scores(self) -> Dict[str, float]:
        """Get scores for all dimensions across evaluations."""
        scores = {}
        for result in self.evaluation_results:
            for dimension in result.dimensions:
                if dimension.name not in scores:
                    scores[dimension.name] = []
                scores[dimension.name].append(dimension.score)
        
        # Average scores for each dimension
        return {name: sum(values) / len(values) for name, values in scores.items()}
```

---

### **BatchTestResults** (`benchmark_runner.py`)

Results from executing multiple tests.

```python
@dataclass  
class BatchTestResults:
    """
    Results from executing multiple tests in a batch.
    
    Provides aggregated statistics and individual test results.
    """
    
    individual_results: List[TestResult]      # Individual test results
    summary_statistics: Dict[str, Any]        # Aggregated statistics
    execution_metadata: Dict[str, Any]        # Batch execution metadata
    
    def get_success_rate(self) -> float:
        """Calculate percentage of successful test executions."""
        if not self.individual_results:
            return 0.0
        successful = sum(1 for result in self.individual_results if result.success)
        return successful / len(self.individual_results)
    
    def get_average_score(self) -> float:
        """Calculate average score across all successful tests."""
        successful_results = [result for result in self.individual_results if result.success]
        if not successful_results:
            return 0.0
        return sum(result.get_overall_score() for result in successful_results) / len(successful_results)
    
    def get_score_distribution(self) -> Dict[str, int]:
        """Get distribution of scores by range."""
        distribution = {
            "90-100": 0, "80-89": 0, "70-79": 0, 
            "60-69": 0, "below-60": 0
        }
        
        for result in self.individual_results:
            if not result.success:
                continue
            score = result.get_overall_score() * 100
            
            if score >= 90:
                distribution["90-100"] += 1
            elif score >= 80:
                distribution["80-89"] += 1
            elif score >= 70:
                distribution["70-79"] += 1
            elif score >= 60:
                distribution["60-69"] += 1
            else:
                distribution["below-60"] += 1
        
        return distribution
    
    def get_dimension_analysis(self) -> Dict[str, Dict[str, float]]:
        """Get statistical analysis for each dimension."""
        dimension_scores = {}
        
        for result in self.individual_results:
            if not result.success:
                continue
            for dim_name, score in result.get_dimension_scores().items():
                if dim_name not in dimension_scores:
                    dimension_scores[dim_name] = []
                dimension_scores[dim_name].append(score)
        
        analysis = {}
        for dim_name, scores in dimension_scores.items():
            analysis[dim_name] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0.0
            }
        
        return analysis
```

## 📊 **Subject-Specific Evaluators**

### **ReasoningEvaluator** (`evaluator.subjects.reasoning_evaluator`)

Evaluates logical reasoning and analytical thinking.

```python
class ReasoningEvaluator(BaseEvaluator):
    """
    Evaluates logical reasoning, analytical thinking, and problem-solving ability.
    
    Measures organization, technical accuracy, completeness, and reliability
    of reasoning-based responses.
    """
    
    def __init__(self):
        super().__init__()
        self.evaluator_name = "reasoning"
        self.reasoning_patterns = self._load_reasoning_patterns()
    
    def evaluate(self, text: str, context: Dict[str, Any]) -> DomainEvaluationResult:
        """
        Evaluate reasoning quality in the response.
        
        Args:
            text: Model response containing reasoning
            context: Test context including difficulty and cultural factors
            
        Returns:
            DomainEvaluationResult with reasoning-specific scoring
        """
        pass
    
    def get_dimensions(self) -> List[str]:
        """Return reasoning evaluation dimensions."""
        return [
            "organization_quality",   # Logical structure and flow
            "technical_accuracy",     # Factual correctness and logic
            "completeness",          # Thoroughness of analysis
            "reliability"            # Consistency and trustworthiness
        ]
    
    def analyze_logical_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the logical structure of the reasoning."""
        pass
    
    def check_evidence_quality(self, text: str) -> Dict[str, Any]:
        """Evaluate quality and relevance of supporting evidence."""
        pass
    
    def assess_conclusion_validity(self, text: str) -> Dict[str, Any]:
        """Assess whether conclusions follow from premises."""
        pass
```

---

### **CreativityEvaluator** (`evaluator.subjects.creativity_evaluator`)

Evaluates creative and artistic content generation.

```python
class CreativityEvaluator(BaseEvaluator):
    """
    Evaluates creativity, originality, and artistic quality in responses.
    
    Focuses on creative expression, style, engagement, and innovation
    while maintaining coherence and cultural appropriateness.
    """
    
    def __init__(self):
        super().__init__()
        self.evaluator_name = "creativity"
        self.style_patterns = self._load_style_patterns()
    
    def evaluate(self, text: str, context: Dict[str, Any]) -> DomainEvaluationResult:
        """
        Evaluate creative quality in the response.
        
        Args:
            text: Model response containing creative content
            context: Test context including genre and cultural factors
            
        Returns:
            DomainEvaluationResult with creativity-specific scoring
        """
        pass
    
    def get_dimensions(self) -> List[str]:
        """Return creativity evaluation dimensions.""" 
        return [
            "originality",           # Uniqueness and innovation
            "engagement",           # Ability to capture and hold interest
            "style_quality",        # Artistic and stylistic merit
            "coherence"            # Internal consistency and flow
        ]
    
    def measure_originality(self, text: str) -> float:
        """Measure originality and uniqueness of content."""
        pass
    
    def assess_engagement(self, text: str) -> float:
        """Assess how engaging and compelling the content is."""
        pass
    
    def evaluate_style(self, text: str, genre: str) -> Dict[str, Any]:
        """Evaluate stylistic quality appropriate to the genre."""
        pass
```

## 🌍 **Cultural Integration System**

### **CulturalValidator** (`evaluator.cultural.cultural_validator`)

Validates cultural appropriateness and sensitivity.

```python
class CulturalValidator:
    """
    Validates cultural appropriateness and sensitivity in model responses.
    
    Checks for cultural bias, stereotypes, misrepresentation, and ensures
    respectful treatment of cultural elements.
    """
    
    def __init__(self):
        self.cultural_patterns = self._load_cultural_patterns()
        self.sensitivity_rules = self._load_sensitivity_rules()
    
    def validate_response(self, 
                         text: str, 
                         cultural_context: CulturalContext) -> 'CulturalValidationResult':
        """
        Validate cultural appropriateness of a response.
        
        Args:
            text: Model response to validate
            cultural_context: Relevant cultural context
            
        Returns:
            CulturalValidationResult with validation outcomes
        """
        pass
    
    def detect_bias(self, text: str) -> List['BiasDetection']:
        """Detect potential cultural bias in text."""
        pass
    
    def check_stereotypes(self, text: str) -> List['StereotypeDetection']:
        """Check for harmful stereotypes."""
        pass
    
    def validate_representation(self, 
                               text: str, 
                               cultural_groups: List[str]) -> Dict[str, Any]:
        """Validate representation of cultural groups."""
        pass
```

## 🔧 **Configuration and Settings**

### **EvaluationConfig** (`evaluator.core.evaluation_config`)

Central configuration management system.

```python
class EvaluationConfig:
    """
    Central configuration management for evaluation settings.
    
    Handles loading, validation, and management of all configuration
    including evaluator weights, cultural settings, and performance parameters.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config_data = self._load_config()
        self.evaluator_settings = self._extract_evaluator_settings()
    
    def get_evaluator_config(self, evaluator_name: str) -> Dict[str, Any]:
        """Get configuration for a specific evaluator."""
        pass
    
    def get_domain_weights(self, domain: str) -> Dict[str, float]:
        """Get scoring weights for a domain."""
        pass
    
    def get_cultural_settings(self) -> Dict[str, Any]:
        """Get cultural evaluation settings."""
        pass
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any errors."""
        pass
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        pass
```

## 🚨 **Error Handling**

### **Custom Exceptions**

The framework defines several custom exceptions for clear error handling:

```python
class EvaluationError(Exception):
    """Base exception for evaluation-related errors."""
    pass

class TestExecutionError(EvaluationError):
    """Error during test execution."""
    pass

class TestLoadError(EvaluationError):
    """Error loading test definitions."""
    pass

class ConfigurationError(EvaluationError):
    """Error in configuration settings."""
    pass

class CulturalValidationError(EvaluationError):
    """Error during cultural validation."""
    pass

class BatchExecutionError(EvaluationError):
    """Error during batch test execution."""
    pass
```

### **Error Handling Patterns**

```python
# Standard error handling in evaluators
class SafeEvaluator(BaseEvaluator):
    def evaluate(self, text: str, context: Dict[str, Any]) -> DomainEvaluationResult:
        try:
            return self._perform_evaluation(text, context)
        except Exception as e:
            logger.error(f"Evaluation failed for {self.evaluator_name}: {str(e)}")
            return self._create_error_result(str(e))
    
    def _create_error_result(self, error_message: str) -> DomainEvaluationResult:
        """Create a result indicating evaluation failure."""
        return DomainEvaluationResult(
            domain=self.evaluator_name,
            evaluation_type="error",
            overall_score=0.0,
            dimensions=[],
            cultural_context=CulturalContext(),
            metadata={"error": error_message, "success": False}
        )
```

## 📈 **Performance Monitoring**

### **PerformanceMetrics** (`benchmark_runner.py`)

Comprehensive performance monitoring for hardware-optimized execution.

```python
@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for hardware monitoring.
    
    Optimized for RTX 5090 + AMD Ryzen 9950X + 128GB RAM configuration.
    """
    
    # Timing metrics
    start_time: float
    end_time: float
    total_duration: float
    
    # CPU metrics (AMD Ryzen 9950X)
    cpu_usage_percent: float = 0.0
    cpu_frequency_mhz: float = 0.0
    cpu_cores_usage: List[float] = None
    
    # Memory metrics (128GB DDR5)
    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0
    memory_usage_percent: float = 0.0
    
    # GPU metrics (RTX 5090)
    gpu_usage_percent: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_temperature_celsius: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return asdict(self)
    
    def get_efficiency_score(self) -> float:
        """Calculate overall system efficiency score."""
        pass
```

---

This API reference provides the foundation for extending and working with the evaluation framework. All classes follow consistent patterns and provide comprehensive error handling and configuration management.