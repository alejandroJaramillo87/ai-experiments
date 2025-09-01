# Configuration

System configuration options for the AI Model Evaluation Framework.

## Basic Configuration

The framework requires no configuration for testing itself. Model evaluation requires specifying an API endpoint.

## Environment Variables

### Model Configuration
```bash
export MODEL_ENDPOINT="http://localhost:8004/completion"    # API endpoint
export MODEL_NAME="your-model-name"                        # Model identifier
```

### Evaluator Configuration
```bash
export EVALUATOR_EMBEDDING_STRATEGY="auto"          # auto|force|fallback
export EVALUATOR_FORCE_CPU="false"                  # true|false
export EVALUATOR_EMBEDDING_MODEL="all-MiniLM-L6-v2" # Embedding model name
```

### Advanced Configuration
```bash
export SEMANTIC_ANALYZER_EMBEDDING_STRATEGY="fallback"  # Override for semantic analyzer
export PYTHONPATH="."                                   # Ensure proper imports
```

## Evaluator Strategies

### Embedding Strategy Options

**auto** (default): Try embedding models, fallback if unavailable
```bash
export EVALUATOR_EMBEDDING_STRATEGY="auto"
```

**force**: Always attempt embedding model loading
```bash  
export EVALUATOR_EMBEDDING_STRATEGY="force"
```

**fallback**: Skip embedding models, use keyword-based methods
```bash
export EVALUATOR_EMBEDDING_STRATEGY="fallback" 
```

### CPU/GPU Configuration

Force CPU-only operation:
```bash
export EVALUATOR_FORCE_CPU="true"
```

Allow GPU usage (default):
```bash
export EVALUATOR_FORCE_CPU="false"
```

## API Configuration

### Endpoint Formats

**llama.cpp format**:
```bash
export MODEL_ENDPOINT="http://localhost:8004/completion"
```

**OpenAI-compatible format**:
```bash
export MODEL_ENDPOINT="http://localhost:8004/v1/completions"
```

### Request Parameters

Default parameters are built into the framework. To customize, modify the API configuration in code:

```python
runner = BenchmarkTestRunner(api_endpoint="http://localhost:8004/completion")
```

## Output Configuration

### Result Directory

Results are saved to `test_results/` by default. Override with command-line option:
```bash
python benchmark_runner.py --output-dir custom_results/
```

### Logging Configuration

Standard Python logging is used. Control verbosity:
```bash
export PYTHONVERBOSE=1  # Verbose Python output
```

## Performance Configuration

### Concurrent Execution

Configure worker threads for concurrent evaluation:
```bash
python benchmark_runner.py --workers 4
```

### Monitoring

Enable performance monitoring:
```bash
python benchmark_runner.py --performance-monitoring
```

### Memory Management

For systems with limited memory:
```bash
export EVALUATOR_EMBEDDING_STRATEGY="fallback"  # Reduce memory usage
python benchmark_runner.py --workers 1          # Single-threaded execution
```

## Domain Configuration

### Test Selection

Select specific domains:
```bash
python benchmark_runner.py --test-type base --category reasoning_general
```

Select model type:
```bash
python benchmark_runner.py --test-type instruct  # Instruction-following tests
python benchmark_runner.py --test-type base      # Base model completion tests
```

## Development Configuration

### Testing Framework

Configure test execution:
```bash
make test MODE=quick        # Fast development tests
make test MODE=coverage     # With coverage reporting
make test SUITE=unit        # Specific test suites
```

### Debug Configuration

Enable debugging features:
```bash
make debug-help                    # Show debug options
export PYTHONPATH=.                # Ensure imports work
python -v benchmark_runner.py      # Verbose Python execution
```

## Configuration Validation

Verify configuration:
```bash
make check-prerequisites           # Complete system check
echo $EVALUATOR_EMBEDDING_STRATEGY # Check environment variables
echo $MODEL_ENDPOINT              # Verify API endpoint
```

Test configuration:
```bash
make test-enhanced-evaluator       # Test evaluator configuration
make test-api-suite               # Test API connectivity
```

## Configuration Examples

### Minimal Setup
```bash
export MODEL_ENDPOINT="http://localhost:8004/completion"
make test MODE=quick
python benchmark_runner.py --test-type base --mode single --test-id reasoning_easy_01
```

### Development Setup
```bash
export EVALUATOR_EMBEDDING_STRATEGY="fallback"  # Fast startup
export EVALUATOR_FORCE_CPU="true"              # Consistent environment
make test SUITE=unit                            # Fast testing
```

### Production Setup
```bash
export MODEL_ENDPOINT="http://production-api:8004/completion"
export MODEL_NAME="production-model-v1"
python benchmark_runner.py --performance-monitoring --workers 4
```

## References

- [Basic Usage](./basic-usage.md)
- [API Reference](./api-reference.md)
- [Troubleshooting](./troubleshooting.md)