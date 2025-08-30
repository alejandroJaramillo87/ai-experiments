# Getting Started Guide

This guide will walk you through setting up and running your first AI model evaluations. Think of this as your **onboarding documentation** - designed to get you productive quickly.

## 🎯 **What You'll Learn**

By the end of this guide, you'll be able to:
- ✅ Run the framework's built-in tests to verify everything works
- ✅ Execute your first model evaluation  
- ✅ Understand the results and what they mean
- ✅ Run different types of evaluations (reasoning, creativity, etc.)
- ✅ Customize evaluations for your specific needs

**Time Required**: 15-20 minutes

## 🚀 **Quick Verification** (2 minutes)

Before evaluating any models, let's make sure the framework itself works correctly:

### **Option 1: Using Make (Recommended)**
```bash
# Run all framework tests with automatic cleanup
make test
```

### **Option 2: Using Python Directly**
```bash
# Run tests with cleanup wrapper
python run_tests_with_cleanup.py

# Or run specific test categories
python run_tests_with_cleanup.py tests/unit/ -q
python run_tests_with_cleanup.py tests/integration/ -q
```

### **What This Does**
- Validates all evaluator modules work correctly
- Tests configuration loading and result processing
- Runs integration tests to ensure components work together
- Automatically cleans up any temporary files

**Expected Output**: You should see test results with mostly passing tests. Some tests may be skipped if you don't have a model API configured yet.

## 🔧 **Setting Up Model Evaluation**

### **Prerequisites**
- **Python 3.8+** (check with `python --version`)
- **Running AI Model** accessible via HTTP API
- **Basic familiarity** with command-line tools

### **Model API Requirements**
Your model needs to accept HTTP requests like:
```bash
curl -X POST http://your-model:8004/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "prompt": "What is the capital of France?",
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

**Common Model Setups**:
- **Local models**: Ollama, LocalAI, Text Generation WebUI
- **Cloud APIs**: OpenAI API, Anthropic API, other providers
- **Custom deployments**: Your own model server

### **Configuration**
Set environment variables (optional but recommended):
```bash
export MODEL_ENDPOINT="http://localhost:8004/v1/completions"
export MODEL_NAME="your-model-name"
```

Or specify them in each command with `--endpoint` and `--model` flags.

## 🧪 **Your First Model Evaluation** (5 minutes)

### **Step 1: Single Test Run**
Let's start with one simple reasoning test:

```bash
python benchmark_runner.py \
  --test-type base \
  --mode single \
  --test-id easy_reasoning_01 \
  --endpoint http://localhost:8004/v1/completions \
  --model "your-model-name"
```

### **Step 2: Understanding the Command**
- **`--test-type base`**: Use tests designed for base/foundation models
- **`--mode single`**: Run just one specific test
- **`--test-id easy_reasoning_01`**: Which test to run (from `domains/reasoning/`)
- **`--endpoint`**: Your model's API endpoint
- **`--model`**: Your model's name/identifier

### **Step 3: Check the Results**
The command creates files in `test_results/`:
```
test_results/
├── easy_reasoning_01_response.txt    # Raw model response
└── easy_reasoning_01_scores.json     # Detailed evaluation scores
```

**Sample Score File**:
```json
{
  "test_id": "easy_reasoning_01",
  "overall_score": 78.5,
  "dimensions": {
    "organization_quality": 85,    // Well-structured response
    "technical_accuracy": 82,     // Correct facts and logic
    "completeness": 75,           // Addressed most requirements  
    "reliability": 72             // Acceptable for basic use
  },
  "execution_time": 2.8,
  "model_used": "your-model-name",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### **Interpreting Your First Score**
- **78.5 overall**: Good performance, suitable for most tasks
- **High organization (85)**: Model structures responses well
- **Good accuracy (82)**: Facts and reasoning are mostly correct
- **Moderate completeness (75)**: Addressed most but not all requirements
- **Fair reliability (72)**: Adequate but might need review for critical tasks

## 📊 **Running Different Evaluation Types** (5 minutes)

### **Reasoning Evaluation**
Test logical thinking and problem-solving:
```bash
python benchmark_runner.py --test-type base --mode category \
  --category reasoning_general --model "your-model-name"
```

### **Creativity Evaluation**
Test creative writing and artistic tasks:
```bash
python benchmark_runner.py --test-type base --mode category \
  --category creativity_writing --model "your-model-name"
```

### **Language Evaluation**
Test grammar, translation, and linguistic skills:
```bash
python benchmark_runner.py --test-type base --mode category \
  --category language_grammar --model "your-model-name"
```

### **Social Understanding Evaluation**
Test cultural awareness and interpersonal skills:
```bash
python benchmark_runner.py --test-type base --mode category \
  --category social_cultural --model "your-model-name"
```

### **Comprehensive Evaluation**
Run multiple categories with parallel processing:
```bash
python benchmark_runner.py --test-type base --mode concurrent \
  --workers 4 --category all --model "your-model-name"
```

## 🎛️ **Common Command Patterns**

### **Quick Development Testing**
```bash
# Fast test during model development
python benchmark_runner.py --test-type base --mode single \
  --test-id easy_reasoning_01 --model "your-model"

# Test specific capability
python benchmark_runner.py --test-type base --mode category \
  --category reasoning_general --model "your-model" --limit 3
```

### **Comprehensive Model Analysis**
```bash
# Full evaluation with performance monitoring
python benchmark_runner.py --test-type instruct --mode concurrent \
  --workers 4 --performance-monitoring \
  --model "your-model" --output-dir comprehensive_results

# Compare different model configurations
python benchmark_runner.py --test-type base --mode category \
  --category reasoning_general --model "model-v1" --output-dir v1_results

python benchmark_runner.py --test-type base --mode category \
  --category reasoning_general --model "model-v2" --output-dir v2_results
```

### **Custom Output and Filtering**
```bash
# Save results to specific directory
python benchmark_runner.py --test-type base --mode category \
  --category creativity_writing --model "your-model" \
  --output-dir creative_evaluation_2024

# Run only easy or medium difficulty tests
python benchmark_runner.py --test-type base --mode category \
  --category reasoning_general --difficulty easy --model "your-model"

# Verbose output for debugging
python benchmark_runner.py --test-type base --mode single \
  --test-id easy_reasoning_01 --model "your-model" --verbose
```

## 📁 **Understanding Result Files**

### **File Types Created**
Every evaluation creates several files:

```
test_results/
├── {test_id}_response.txt          # Raw model response
├── {test_id}_scores.json           # Detailed evaluation scores  
├── {test_id}_metadata.json         # Test execution metadata
└── batch_summary.json              # Summary when running multiple tests
```

### **Response File** (`*_response.txt`)
```
=== PROMPT ===
Analyze the logical consistency of the following argument...

=== MODEL RESPONSE ===
The argument contains several logical fallacies. First, it commits 
an ad hominem attack by...
[Full model response text]

=== EVALUATION CONTEXT ===
Test ID: reasoning_logic_01
Category: reasoning_general
Difficulty: medium
Execution Time: 3.2 seconds
```

### **Scores File** (`*_scores.json`)
```json
{
  "test_id": "reasoning_logic_01",
  "overall_score": 85.4,
  "dimensions": {
    "organization_quality": 92,
    "technical_accuracy": 88, 
    "completeness": 81,
    "reliability": 87
  },
  "confidence": 0.89,
  "reasoning": [
    "Well-structured logical argument with clear premise-conclusion flow",
    "Strong evidence provided for main claims", 
    "Minor weakness: could have addressed potential counterarguments",
    "High confidence in accuracy assessment"
  ],
  "metadata": {
    "category": "reasoning_general",
    "difficulty": "medium", 
    "cultural_context": null,
    "execution_time": 3.2,
    "evaluator_version": "1.2.0"
  }
}
```

### **Batch Summary** (`batch_summary.json`)
```json
{
  "summary": {
    "total_tests": 5,
    "completed": 5,
    "failed": 0,
    "average_score": 82.3,
    "execution_time": 45.6
  },
  "scores_by_category": {
    "reasoning_general": 84.1,
    "reasoning_complex": 78.9
  },
  "scores_by_dimension": {
    "organization_quality": 87.2,
    "technical_accuracy": 83.1,
    "completeness": 79.8,  
    "reliability": 81.5
  }
}
```

## 🔍 **Reading and Using Results**

### **Quick Assessment Workflow**
1. **Check overall score** first - gives you the headline
2. **Look at dimension breakdown** - identifies specific strengths/weaknesses  
3. **Read the reasoning** - understand why the score was assigned
4. **Check confidence** - higher confidence means more reliable assessment

### **Score Interpretation Guidelines**

| Score Range | Quality Level | Typical Use Cases |
|-------------|---------------|-------------------|
| 90-100 | Exceptional | Production systems, critical tasks |
| 80-89 | Very Good | Most production use cases |
| 70-79 | Good | Basic tasks, with human review |
| 60-69 | Fair | Development/testing, limited production |
| Below 60 | Needs Work | Not suitable for production use |

### **Dimension-Specific Insights**

**Organization Quality**:
- **High (85+)**: Responses are well-structured and easy to follow
- **Low (60-)**: Responses lack clear structure or logical flow

**Technical Accuracy**:
- **High (85+)**: Facts, logic, and domain knowledge are reliable
- **Low (60-)**: Contains errors, misinformation, or logical flaws

**Completeness**:
- **High (85+)**: Thoroughly addresses all aspects of the request
- **Low (60-)**: Missing key information or requirements

**Reliability**:
- **High (85+)**: Consistent quality, suitable for autonomous use
- **Low (60-)**: Inconsistent, requires human oversight

## ⚡ **Troubleshooting Common Issues**

### **"Connection refused" or API errors**
```bash
# Test your model endpoint directly
curl -X POST http://localhost:8004/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "prompt": "Hello", "max_tokens": 10}'

# Check if model server is running
ps aux | grep your-model-server
```

### **"Test file not found" errors**
```bash
# List available test IDs
ls domains/reasoning/base_models/
ls domains/creativity/base_models/

# Verify test exists
cat domains/reasoning/base_models/easy_reasoning_01.json
```

### **Low or inconsistent scores**
1. **Check model configuration**: temperature, max_tokens, prompt format
2. **Try different test types**: base vs instruct models need different test types
3. **Review raw responses**: look at `*_response.txt` files to understand model behavior
4. **Run multiple tests**: single tests can vary, patterns appear over multiple runs

### **Performance issues**
```bash
# Reduce concurrency
python benchmark_runner.py --workers 1 --model "your-model"

# Run smaller test batches  
python benchmark_runner.py --limit 3 --model "your-model"

# Use faster test categories
python benchmark_runner.py --difficulty easy --model "your-model"
```

## 🎯 **Next Steps**

### **Immediate Actions** (after first successful run)
1. **Run multiple categories** to get a comprehensive view of your model
2. **Compare different model configurations** (temperature, etc.)
3. **Review the raw responses** to understand your model's behavior patterns

### **Deeper Exploration**
1. **Read the [Configuration Guide](CONFIGURATION.md)** to customize evaluations
2. **Check [Interpreting Results](INTERPRETING_RESULTS.md)** for detailed score analysis
3. **Explore different domains** in the `domains/` directory

### **Advanced Usage**
1. **Set up automated benchmarking** with the provided automation tools
2. **Create custom evaluations** for your specific use case
3. **Integrate with CI/CD** for continuous model quality monitoring

---

**Congratulations!** 🎉 You've successfully run your first AI model evaluation. The framework is now ready to help you systematically assess and improve your AI models with confidence.

**Questions or Issues?** Check the other documentation files or look at the test examples in the `domains/` directories for more specific use cases.