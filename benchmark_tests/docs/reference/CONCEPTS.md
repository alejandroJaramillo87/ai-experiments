# Core Concepts Guide

This guide explains the fundamental concepts behind AI model evaluation using familiar software engineering terminology. No advanced math required - just solid engineering principles.

## 🎯 **What is "Evaluation"?**

Think of evaluation like **automated code review for AI responses**:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Input Prompt  │───▶│  AI Model       │───▶│  Structured Score   │
│  (Test Case)    │    │  (System Under  │    │  Report (0-100)     │
│                 │    │   Test)         │    │                     │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

**Just like code review checks for:**
- Code organization and structure
- Technical correctness 
- Completeness of implementation
- Production readiness

**Model evaluation checks for:**
- Response organization and clarity
- Factual accuracy and logical reasoning
- Completeness of task requirements
- Reliability for real-world use

## 🏗️ **Domains: Different Types of Tasks**

Think of domains like **different microservices** - each handles a specific type of work:

### **Reasoning Domain**
**Like:** Backend logic and algorithms
**Tests:** Problem-solving, analysis, deduction
**Example:** "Given these sales data points, what trends do you see?"

### **Creativity Domain** 
**Like:** Frontend design and user experience
**Tests:** Original content, storytelling, artistic tasks
**Example:** "Write a product marketing campaign for eco-friendly shoes"

### **Language Domain**
**Like:** Database schema design and API contracts
**Tests:** Grammar, translation, linguistic precision
**Example:** "Translate this technical documentation to Spanish"

### **Social Domain**
**Like:** User authentication and access control
**Tests:** Cultural understanding, interpersonal situations  
**Example:** "How would you handle a customer complaint about delayed delivery?"

### **Integration Domain**
**Like:** Full-stack feature implementation
**Tests:** Complex multi-domain problems
**Example:** "Create a business plan that includes market analysis, creative branding, and financial projections"

## 📊 **Dimensions: Quality Metrics**

Dimensions are like **code review criteria** - standardized ways to measure quality:

### **Organization Quality** (0-100)
**Like:** Code structure and readability
- Is the response well-structured and easy to follow?
- Clear introduction, body, conclusion?
- Logical flow of ideas?

### **Technical Accuracy** (0-100)  
**Like:** Correctness and edge case handling
- Are facts and logic correct?
- No misinformation or logical errors?
- Appropriate level of technical detail?

### **Completeness** (0-100)
**Like:** Feature completeness and requirements coverage
- Does it address all parts of the request?
- No missing critical information?
- Adequate depth for the complexity level?

### **Reliability** (0-100)
**Like:** Production readiness and maintainability
- Would you trust this response in a production system?
- Consistent quality across similar requests?
- Appropriate confidence level expressed?

## 🧪 **Test Definitions vs Test Execution**

### **Test Definitions** (`domains/*.json`)
**Like:** API endpoint definitions in OpenAPI/Swagger
- Define what the test should do
- Specify input parameters and expected behavior
- Configuration data, not executable code

```json
{
  "test_id": "reasoning_logic_01",
  "prompt": "Analyze the logical consistency of this argument...",
  "category": "reasoning_general", 
  "difficulty": "medium",
  "expected_dimensions": ["organization_quality", "technical_accuracy"]
}
```

### **Test Execution** (`benchmark_runner.py`)
**Like:** Test runner (Jest, pytest, etc.)
- Loads test definitions 
- Executes tests against the model
- Collects and processes results
- Handles concurrency, retries, and error cases

## 🔧 **Evaluators: Plugin Architecture**

Evaluators work like **middleware components** in a web framework:

### **Base Contract**
Every evaluator implements the same interface:
```python
class BaseEvaluator:
    def evaluate(self, text: str, context: Dict) -> EvaluationResult
    def get_dimensions(self) -> List[str] 
    def configure(self, settings: Dict) -> None
```

### **Specialized Implementations**
Like different middleware for different concerns:

```python
# Like authentication middleware
class ReasoningEvaluator(BaseEvaluator):
    def evaluate(self, text, context):
        # Analyze logical structure, evidence quality
        return {"technical_accuracy": 85, "organization_quality": 92}

# Like CORS middleware  
class CreativityEvaluator(BaseEvaluator):
    def evaluate(self, text, context):
        # Assess originality, engagement, style
        return {"completeness": 88, "reliability": 79}
```

### **Plugin Registration**
Like registering middleware in Express.js:
```python
evaluation_pipeline = EvaluationPipeline()
evaluation_pipeline.register(ReasoningEvaluator(), domains=['reasoning'])
evaluation_pipeline.register(CreativityEvaluator(), domains=['creativity'])
```

## 📈 **Scoring System**

### **Like Percentage Grades**
All scores are **0-100** (think of them as percentages):
- **90-100**: A+ grade - Exceptional quality
- **80-89**: A/B+ grade - Very good, production-ready
- **70-79**: B grade - Good, meets most requirements  
- **60-69**: C grade - Adequate but has issues
- **Below 60**: D/F grade - Significant problems

### **Aggregation Rules**
**Like calculating GPA from individual course grades:**

```python
# Weighted average based on importance
overall_score = (
    organization_quality * 0.25 +
    technical_accuracy * 0.35 +  
    completeness * 0.25 +
    reliability * 0.15
)
```

## 🗂️ **Results and Reporting**

### **Result Structure**
**Like structured log entries** or API responses:

```json
{
  "test_id": "reasoning_logic_01",
  "timestamp": "2024-01-15T10:30:00Z",
  "model_info": {
    "name": "my-model-v1", 
    "endpoint": "http://localhost:8004"
  },
  "scores": {
    "overall_score": 85.4,
    "dimensions": {
      "organization_quality": 92,
      "technical_accuracy": 88,
      "completeness": 81,
      "reliability": 87
    }
  },
  "metadata": {
    "execution_time": 3.2,
    "confidence": 0.89,
    "category": "reasoning_general"
  },
  "reasoning": [
    "Well-structured logical argument",
    "Strong evidence supporting main points", 
    "Minor gap in addressing counterarguments"
  ]
}
```

### **Batch Results**
**Like test suite reports:**
- Individual test results aggregated
- Summary statistics (pass/fail rates, averages)
- Performance metrics (execution times, throughput)
- Error logs and debugging information

## ⚙️ **Configuration System**

### **Hierarchical Configuration**
**Like application configuration in modern frameworks:**

```
Default Settings → File Config → Environment Variables → CLI Arguments
     ↓                ↓               ↓                     ↓
  Built-in       config.json      Export vars           --flags
```

### **Domain-Specific Settings**
**Like microservice configuration:**
```json
{
  "reasoning": {
    "weight_logic": 0.4,
    "weight_evidence": 0.3,
    "weight_structure": 0.3,
    "strict_mode": true
  },
  "creativity": {
    "originality_threshold": 0.7,
    "style_importance": 0.6,
    "allow_abstract": true
  }
}
```

## 🎛️ **Test Categories and Difficulty**

### **Categories: Like Test Suites**
Organize tests by functionality:
- `reasoning_general`: Basic logic and analysis
- `reasoning_complex`: Advanced problem-solving
- `creativity_writing`: Content creation tasks
- `language_grammar`: Linguistic precision tests

### **Difficulty Levels: Like Test Complexity**
- **Easy**: Simple, single-concept tests (unit tests)
- **Medium**: Multi-step problems (integration tests)  
- **Hard**: Complex, open-ended challenges (end-to-end tests)

## 🔄 **Execution Modes**

### **Single Test Mode**
**Like running one unit test:**
```bash
pytest tests/test_user_auth.py::test_login
```
```bash
python benchmark_runner.py --mode single --test-id reasoning_logic_01
```

### **Category Mode**
**Like running a test suite:**
```bash  
pytest tests/integration/
```
```bash
python benchmark_runner.py --mode category --category reasoning_general
```

### **Concurrent Mode**
**Like parallel test execution:**
```bash
pytest -n 4 tests/
```
```bash
python benchmark_runner.py --mode concurrent --workers 4
```

## 🧩 **Cultural Context Integration**

### **Context as Configuration**
**Like feature flags or environment-specific settings:**

```python
cultural_context = {
    "traditions": ["griot", "oral_storytelling"],
    "groups": ["west_african"], 
    "languages": ["english", "yoruba"],
    "geographic": ["nigeria", "ghana"]
}

# Used during evaluation like feature flags
if "griot" in cultural_context["traditions"]:
    evaluator.enable_storytelling_analysis()
```

### **Data-Driven Cultural Understanding**
**Like reference data or lookup tables:**
- Cultural knowledge stored in `data/cultural/`
- Evaluators query this data during analysis
- No hardcoded cultural assumptions in business logic

---

## 🎓 **Summary for Software Engineers**

**Think of this system as:**
- **Test framework** for AI responses (like Jest or pytest)
- **Plugin architecture** for different evaluation types (like middleware)
- **Quality assurance pipeline** with standardized metrics (like code review)
- **Configuration-driven** behavior (like modern web frameworks)
- **Scalable and concurrent** execution (like microservices)

**Key principles:**
- **Separation of concerns**: Tests, evaluators, and data are independent
- **Plugin-based extensibility**: Easy to add new evaluation types
- **Configuration over code**: Behavior controlled by data, not hardcoded logic
- **Standardized interfaces**: Consistent contracts between components
- **Graceful error handling**: System continues working when individual components fail

This makes AI model evaluation as familiar and manageable as any other software engineering task you're already comfortable with.