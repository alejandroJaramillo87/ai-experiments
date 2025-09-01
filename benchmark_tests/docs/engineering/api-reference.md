# API Reference

Key classes and interfaces in the AI Model Evaluation Framework.

## Core Classes

### BenchmarkTestRunner

Main orchestration class for test execution.

```python
class BenchmarkTestRunner:
    def __init__(self, config_path=None, api_endpoint=None)
    def load_test_suite(self, suite_path: str) -> bool
    def execute_single_test(self, test_id: str) -> TestResult
    def execute_concurrent_tests(self, workers: int) -> BatchResults
```

**Key Methods:**
- `load_test_suite()`: Load test definitions from JSON file
- `execute_single_test()`: Run individual test and return results
- `execute_concurrent_tests()`: Run multiple tests with thread pool

### TestResult

Standard result format for individual test evaluation.

```python
@dataclass
class TestResult:
    test_id: str
    response: str
    score: float
    execution_time: float
    evaluation_details: Dict[str, Any]
    metadata: Dict[str, Any]
```

### PerformanceMetrics

Hardware and timing metrics for test execution.

```python
@dataclass
class PerformanceMetrics:
    total_duration: float
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_usage_percent: float
    request_duration: float
```

## Evaluation System

### MultiDimensionalEvaluator

Base class for domain-specific evaluators.

```python
class MultiDimensionalEvaluator:
    def evaluate(self, response: str, test_metadata: Dict) -> DomainEvaluationResult
    def _assess_dimension(self, dimension: str, response: str) -> EvaluationDimension
```

### DomainEvaluationResult

Result format from domain evaluators.

```python
@dataclass
class DomainEvaluationResult:
    domain: str
    overall_score: float
    dimensions: List[EvaluationDimension]
    cultural_context: CulturalContext
    metadata: Dict[str, Any]
```

### EvaluationDimension

Individual scoring dimension within domain evaluation.

```python
@dataclass
class EvaluationDimension:
    name: str
    score: float              # 0.0 to 1.0
    confidence: float         # 0.0 to 1.0  
    cultural_relevance: float # 0.0 to 1.0
    evidence: List[str]
    cultural_markers: List[str]
```

## Configuration Classes

### APIConfiguration

API endpoint configuration for model communication.

```python
@dataclass  
class APIConfiguration:
    endpoint: str
    model_name: str
    timeout: int = 30
    max_tokens: int = 500
    temperature: float = 0.7
```

### TestSuite

Container for loaded test definitions.

```python
@dataclass
class TestSuite:
    name: str
    domain: str
    model_type: str           # "base" or "instruct"
    difficulty: str           # "easy", "medium", "hard"
    tests: List[Dict[str, Any]]
    metadata: Dict[str, Any]
```

## Advanced Analytics

### EntropyCalculator

Information theory analysis of model responses.

```python
class EntropyCalculator:
    def calculate(self, text: str) -> float
    def calculate_ngram_entropy(self, text: str, n: int) -> float
```

### SemanticCoherenceAnalyzer

Text coherence and consistency analysis.

```python
class SemanticCoherenceAnalyzer:
    def analyze(self, text: str) -> Dict[str, float]
    def compute_coherence_score(self, text: str) -> float
```

## Utility Functions

### Model Loading

```python
def load_embedding_model_unified(model_name: str = "all-MiniLM-L6-v2") -> Tuple[Optional[Any], str]
```

### Test Management

```python
def load_and_configure_runner(test_definitions_dir: str = "domains") -> BenchmarkTestRunner
```

### Result Processing

```python
def ensure_json_serializable(obj: Any) -> Any
```

## Environment Configuration

Key environment variables that affect API behavior:

```python
EVALUATOR_EMBEDDING_STRATEGY    # "auto", "force", "fallback"
EVALUATOR_FORCE_CPU            # "true", "false"  
EVALUATOR_EMBEDDING_MODEL      # Model name for embeddings
MODEL_ENDPOINT                 # Default API endpoint
MODEL_NAME                     # Default model name
```

## Usage Patterns

### Basic Evaluation
```python
runner = BenchmarkTestRunner(api_endpoint="http://localhost:8004/completion")
runner.load_test_suite("domains/reasoning/base_models/easy.json")
result = runner.execute_single_test("reasoning_easy_01")
```

### Custom Evaluator
```python
class CustomEvaluator(MultiDimensionalEvaluator):
    def evaluate(self, response, test_metadata):
        # Implementation here
        return DomainEvaluationResult(...)
```

### Performance Monitoring
```python
runner = BenchmarkTestRunner()
runner.performance_monitor.start_monitoring()
result = runner.execute_single_test("test_id")
metrics = runner.performance_monitor.get_summary()
```

## References

- [Architecture](./architecture.md)
- [Basic Usage](./basic-usage.md)
- [Configuration](./configuration.md)