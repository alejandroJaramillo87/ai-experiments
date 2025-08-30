# System Architecture

This document explains how the AI Model Evaluation Framework works from a software engineering perspective. No advanced math required - just solid software design principles.

## 🏗️ **High-Level Architecture**

Think of this system like a **web application with a plugin architecture**:

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                       │
│  (CLI, Make commands, Python scripts)                  │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│                 Test Runner                             │
│  (Orchestration, API calls, result management)         │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│               Evaluator System                          │
│  (Pluggable modules for different thinking types)      │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│                Data Layer                               │
│  (Test definitions, cultural data, result storage)     │
└─────────────────────────────────────────────────────────┘
```

## 🔧 **Core Components**

### **1. Test Runner** (`benchmark_runner.py`)
**Role**: Like an Express.js server - handles requests, orchestrates processing, returns results.

**Responsibilities:**
- Load test definitions from JSON files
- Make API calls to your model  
- Route responses to appropriate evaluators
- Aggregate and save results
- Handle errors and timeouts

**Key Interfaces:**
```python
class BenchmarkTestRunner:
    def load_test_suite(self, test_type: str) -> Dict
    def execute_single_test(self, test_id: str) -> TestResult  
    def execute_concurrent_tests(self, workers: int) -> BatchResults
```

### **2. Evaluator System** (`evaluator/`)
**Role**: Like microservices - each handles a specific domain of evaluation.

**Plugin Architecture:**
```
evaluator/
├── core/           # Base classes and shared utilities
├── subjects/       # Domain-specific evaluators  
├── cultural/       # Cultural understanding evaluators
├── advanced/       # Sophisticated analysis tools
├── validation/     # Quality assurance systems
└── linguistics/    # Language-specific analysis
```

**Each evaluator follows the same contract:**
```python
class BaseEvaluator:
    def evaluate(self, text: str, context: Dict) -> EvaluationResult
    def get_dimensions(self) -> List[str] 
    def configure(self, settings: Dict) -> None
```

### **3. Data Management**
**Role**: Like a database layer - stores definitions, references, results.

**Organization:**
- `data/cultural/` - Reference datasets (like lookup tables)
- `domains/` - Test definitions (like API endpoints)  
- `test_results/` - Generated outputs (like logs)

## 🔌 **Plugin System Design**

### **How Evaluators Work**
Each evaluator is like a **specialized function**:

```python
# Input: Model response + context
input_data = {
    "text": "Model's response to the test",
    "test_metadata": {"category": "reasoning", "difficulty": "medium"},
    "cultural_context": {"traditions": ["griot"], "groups": ["west_african"]}
}

# Processing: Domain-specific analysis
evaluator = ReasoningEvaluator()
result = evaluator.evaluate(input_data)

# Output: Structured score report
{
    "overall_score": 85.4,
    "dimensions": {
        "organization_quality": 92,
        "technical_accuracy": 88, 
        "completeness": 81
    },
    "confidence": 0.89,
    "reasoning": ["Well-structured argument", "Minor gap in evidence"]
}
```

### **Adding New Evaluators**
Like adding a new route handler:

1. **Implement the interface** (`BaseEvaluator`)
2. **Define your scoring logic** (no math required - use business rules)
3. **Register in the system** (add to configuration)  
4. **Test your module** (unit tests)

## 📊 **Data Flow Architecture**

### **Test Execution Flow**
```
User Command → Test Runner → Model API → Evaluator → Results
     ↓              ↓            ↓           ↓         ↓
  make test    Load JSON     HTTP POST    Score      Save JSON
```

### **Detailed Request Flow**
1. **User triggers test** (CLI command or API call)
2. **Test Runner loads definition** from `domains/*.json`
3. **API request sent** to model endpoint  
4. **Response routed** to appropriate evaluator(s)
5. **Evaluator processes** using domain-specific logic
6. **Results aggregated** and saved to `test_results/`
7. **Cleanup performed** (remove temporary artifacts)

### **Configuration Flow**
```
Default Settings → File Config → Environment Variables → CLI Arguments
      ↓                ↓               ↓                     ↓
   Built-in      JSON files      Export vars           --flags
```

## 🏛️ **Design Patterns Used**

### **1. Strategy Pattern** (Evaluators)
Different evaluation strategies for different domains:
```python
class EvaluationContext:
    def set_evaluator(self, evaluator: BaseEvaluator):
        self.evaluator = evaluator
    
    def evaluate(self, text: str) -> Result:
        return self.evaluator.evaluate(text)

# Usage
context.set_evaluator(ReasoningEvaluator())  # For logic problems
context.set_evaluator(CreativityEvaluator()) # For creative tasks
```

### **2. Factory Pattern** (Test Creation)
Create tests based on type and category:
```python
class TestFactory:
    def create_test(self, test_type: str, category: str) -> Test:
        if test_type == "reasoning":
            return ReasoningTest(category)
        elif test_type == "creativity":
            return CreativityTest(category)
```

### **3. Observer Pattern** (Progress Tracking)
Components notify observers of progress:
```python
class TestRunner:
    def __init__(self):
        self.observers = []
    
    def notify_progress(self, event: str, data: Dict):
        for observer in self.observers:
            observer.on_progress(event, data)
```

### **4. Command Pattern** (CLI Interface)
Commands encapsulate operations:
```python
class ExecuteTestCommand:
    def __init__(self, test_id: str, options: Dict):
        self.test_id = test_id
        self.options = options
    
    def execute(self) -> Result:
        # Encapsulated test execution logic
```

## 🔄 **Concurrency Architecture**

### **Thread Pool Design**
Like a web server handling multiple requests:

```python
class ConcurrentTestRunner:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.results_queue = Queue()
    
    def execute_tests(self, test_ids: List[str]) -> List[Result]:
        futures = [
            self.executor.submit(self.execute_single_test, test_id) 
            for test_id in test_ids
        ]
        return [future.result() for future in futures]
```

### **Resource Management**
- **Thread safety**: Each test execution is isolated
- **Memory management**: Results streamed to disk, not held in memory
- **API rate limiting**: Built-in delays to respect model API limits
- **Error isolation**: One test failure doesn't crash the batch

## 📁 **Directory Architecture Rationale**

### **Separation of Concerns**
```
benchmark_tests/
├── evaluator/     # Business logic (pure functions)
├── data/          # Reference data (read-only)  
├── domains/       # Test definitions (configuration)
├── tests/         # Framework tests (quality assurance)
└── docs/          # Documentation (knowledge base)
```

### **Why This Structure?**
- **`evaluator/` modular**: Easy to add new evaluation types
- **`data/` centralized**: Single source of truth for reference data
- **`domains/` configurable**: Tests defined as data, not code  
- **`tests/` comprehensive**: Full test coverage for reliability
- **`docs/` accessible**: Progressive disclosure of complexity

## 🔧 **Extension Points**

### **Adding New Domains**
1. Create JSON test definitions in `domains/new_domain/`
2. Add category mappings
3. Test definitions follow standard schema
4. No code changes needed

### **Adding New Evaluators**  
1. Implement `BaseEvaluator` interface
2. Add to `evaluator/subjects/` or appropriate subdirectory
3. Register in configuration
4. Add unit tests

### **Adding New Analysis Tools**
1. Create module in `evaluator/advanced/`  
2. Implement standard interfaces
3. Integrate with main evaluation pipeline
4. Document usage patterns

## 🚦 **Error Handling Strategy**

### **Graceful Degradation**
Like a robust web application:
- **API failures**: Retry with backoff, then graceful failure
- **Evaluator errors**: Log error, return partial results  
- **Invalid configs**: Validate early, fail fast with clear messages
- **Resource exhaustion**: Queue management, resource monitoring

### **Logging and Debugging**  
- **Structured logging**: JSON logs for automated analysis
- **Debug modes**: Verbose output for development
- **Error tracking**: Breadcrumbs for debugging complex failures
- **Performance monitoring**: Built-in profiling and metrics

---

This architecture prioritizes **maintainability**, **extensibility**, and **reliability** - core software engineering principles that make the system practical for real-world use.