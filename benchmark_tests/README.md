# AI Model Evaluation Framework

A practical, engineer-friendly framework for evaluating Large Language Model (LLM) performance across diverse reasoning tasks. Built for software engineers who want reliable model benchmarking without needing a math PhD.

## ✨ **What This Does**

Think of this as **automated code review for AI models** - it runs your models through standardized tests and gives you detailed reports on how well they perform across different types of thinking tasks.

**Perfect for:**
- 🔧 **Engineering teams** evaluating models for production use
- 🚀 **AI engineers** comparing model performance objectively  
- 🏗️ **DevOps teams** setting up automated model testing pipelines
- 📊 **Technical leads** making data-driven model selection decisions

## 🚀 **Quick Start** (5 minutes)

### 1. Run Your First Test

```bash
# Test that everything works (no API calls)
make test-quick

# Run a real evaluation (requires running model)
python run_tests_with_cleanup.py tests/integration/test_validation_integration.py
```

### 2. Evaluate a Model

```bash
# Point to your model API and run basic evaluation
python benchmark_runner.py --test-type base --mode single --test-id easy_reasoning_01 \
  --endpoint http://localhost:8004/v1/completions \
  --model "your-model-name"
```

### 3. View Results

Check the generated `test_results/` directory for detailed JSON reports and plain text responses.

## 📋 **Core Concepts** (Software Engineer Perspective)

### **What is "Evaluation"?**
Like unit testing, but for AI reasoning ability:
- **Input**: A reasoning task (prompt)  
- **Process**: Model generates response
- **Output**: Structured score report (0-100 scale)

### **What are "Domains"?**
Different types of thinking tasks, like different microservices:
- **Reasoning**: Logic problems, analysis, deduction
- **Creativity**: Original content, storytelling, artistic tasks  
- **Language**: Grammar, translation, linguistic analysis
- **Social**: Cultural understanding, interpersonal situations
- **Integration**: Complex multi-domain problems

### **What are "Dimensions"?**
Think of them as **quality metrics** - like code review criteria:
- **Organization**: Is the response well-structured?
- **Accuracy**: Are the facts and logic correct? 
- **Completeness**: Does it address all requirements?
- **Reliability**: Would you trust this in production?

**No calculus required** - the system handles all scoring automatically.

## 🏗️ **System Architecture**

```
benchmark_tests/
├── 📁 data/              # Test datasets and cultural references  
├── 📁 domains/           # Test definitions organized by thinking type
├── 📁 evaluator/         # The evaluation engine (modular components)
├── 📁 tests/            # Framework tests (unit, integration, analysis)
├── 🎯 Makefile          # Convenient commands (make test, make clean)
├── 🤖 benchmark_runner.py        # Main test execution engine
└── 📚 docs/             # Detailed documentation
```

### **Key Components**

- **📝 Test Definitions**: JSON files defining what to test (`domains/`)
- **🧠 Evaluators**: Pluggable modules that score different types of thinking (`evaluator/`)  
- **🚀 Test Runner**: Orchestrates execution, handles APIs, manages results (`benchmark_runner.py`)
- **🧹 Automation**: Built-in cleanup, convenient commands (`Makefile`)

## 📊 **Understanding Results**

### **Scores Explained**
All scores are **0-100** (like percentages):
- **90-100**: Exceptional - production ready for complex tasks
- **80-89**: Very good - reliable for most use cases
- **70-79**: Good - adequate with some limitations  
- **60-69**: Fair - needs improvement for critical tasks
- **Below 60**: Poor - significant issues detected

### **Sample Result File**
```json
{
  "test_id": "reasoning_logic_01",
  "overall_score": 85.4,
  "dimensions": {
    "organization_quality": 92,    // Well-structured response
    "technical_accuracy": 88,     // Facts and logic correct
    "completeness": 81,           // Addressed all requirements
    "reliability": 87             // Trustworthy for production
  },
  "execution_time": 3.2,
  "model_used": "your-model-name"
}
```

## 🛠️ **Common Workflows**

### **Development Testing**
```bash
# Quick check during development
make test-quick

# Test specific functionality  
make test-unit
make test-integration
```

### **Model Comparison** 
```bash
# Test Model A
python benchmark_runner.py --test-type base --mode category \
  --category reasoning_general --model "model-a" --output-dir results_a

# Test Model B  
python benchmark_runner.py --test-type base --mode category \
  --category reasoning_general --model "model-b" --output-dir results_b

# Compare results
python analyze_results.py results_a results_b
```

### **Production Pipeline**
```bash
# Full evaluation suite with cleanup
make test

# Custom evaluation with performance monitoring
python benchmark_runner.py --test-type instruct --mode concurrent \
  --workers 4 --performance-monitoring --category reasoning_comprehensive
```

## 📚 **Documentation**

### **For Users**
- **[Getting Started Guide](docs/GETTING_STARTED.md)** - Detailed setup and first steps
- **[Configuration Guide](docs/CONFIGURATION.md)** - All settings explained
- **[Interpreting Results](docs/INTERPRETING_RESULTS.md)** - Understanding scores and reports

### **For Developers**
- **[Architecture Overview](docs/ARCHITECTURE.md)** - How the system works
- **[Extension Guide](docs/EXTENDING.md)** - Adding new evaluators and domains  
- **[API Reference](docs/API_REFERENCE.md)** - Classes and methods
- **[Contributing Guide](docs/CONTRIBUTING.md)** - Development workflow

### **Operations**
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Running in production
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Cleanup Guide](CLEANUP_GUIDE.md)** - Managing test artifacts

## 🎛️ **Available Commands**

```bash
# Testing (automatic cleanup included)
make test                 # Run all tests
make test-unit           # Unit tests only  
make test-integration    # Integration tests only
make test-quick          # Quick development test
make clean               # Clean up artifacts only

# Custom test execution
python run_tests_with_cleanup.py [pytest-args]    # Tests with auto-cleanup
python benchmark_runner.py [options]              # Direct model evaluation
python cleanup_test_artifacts.py                  # Manual cleanup
```

## ⚙️ **Configuration**

### **Basic Setup**
No configuration needed for testing the framework itself.

### **Model Evaluation Setup**  
Point to your model API:
```bash
export MODEL_ENDPOINT="http://localhost:8004/v1/completions"
export MODEL_NAME="your-model-name"
```

### **Advanced Configuration**
See [Configuration Guide](docs/CONFIGURATION.md) for:
- Custom evaluation criteria
- Domain-specific settings  
- Performance tuning
- Cultural evaluation settings

## 🤝 **Contributing**

We welcome contributions! This project follows standard open source practices:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Test** your changes (`make test`)  
4. **Commit** your changes (`git commit -m 'Add amazing feature'`)
5. **Push** to the branch (`git push origin feature/amazing-feature`)
6. **Open** a Pull Request

See [Contributing Guide](docs/CONTRIBUTING.md) for detailed development workflow.

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 **Support**

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)  
- **Documentation**: Check [docs/](docs/) directory
- **Troubleshooting**: See [Troubleshooting Guide](docs/TROUBLESHOOTING.md)

---

**Made for software engineers, by software engineers.** No PhD required. 🚀