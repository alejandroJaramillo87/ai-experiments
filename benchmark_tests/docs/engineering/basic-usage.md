# Basic Usage

Common commands and workflows for the AI Model Evaluation Framework.

## Framework Testing

Test the framework itself:
```bash
make test                    # Full test suite
make test MODE=quick         # Fast development tests  
make test SUITE=unit         # Unit tests only
make test MODE=coverage      # With coverage reporting
```

Clean up artifacts:
```bash
make clean                   # Basic cleanup
make clean-all              # Deep cleanup
```

## Model Evaluation

### Prerequisites Check

Verify system readiness before evaluation:
```bash
make check-prerequisites    # Complete system check
make test-api-suite        # Test LLM server connectivity
```

### Basic Evaluation

Evaluate a single test:
```bash
python benchmark_runner.py --test-type base --mode single \
  --test-id reasoning_easy_01 \
  --endpoint http://localhost:8004/completion \
  --model "your-model"
```

Evaluate by category:
```bash
python benchmark_runner.py --test-type base --mode category \
  --category reasoning_general \
  --endpoint http://localhost:8004/completion
```

### Advanced Options

Concurrent evaluation:
```bash
python benchmark_runner.py --test-type instruct --mode concurrent \
  --workers 4 --category creativity_narrative
```

With performance monitoring:
```bash
python benchmark_runner.py --performance-monitoring \
  --output-dir custom_results/
```

## Domain Management

Convert domains to instruct format:
```bash
make convert-core-domains           # All core domains
make convert-creativity-tests       # Creativity domain only
```

Test domain conversions:
```bash
make test-core-domains-conversion   # Test all conversions
make test-creativity-conversion     # Test creativity conversion
```

## Result Analysis

Results are saved to `test_results/` directory by default. Each evaluation produces:

- JSON result files with scores and metadata
- Plain text response files
- Performance metrics (if enabled)

## Common Workflows

### Development Testing
```bash
make test MODE=quick        # Fast iteration
make test SUITE=unit        # Focus on unit tests
```

### Model Comparison
```bash
# Test Model A
python benchmark_runner.py --model "model-a" --output-dir results_a/

# Test Model B  
python benchmark_runner.py --model "model-b" --output-dir results_b/

# Compare results manually or with analysis tools
```

### Production Validation
```bash
make check-prerequisites   # System readiness
make test                 # Full framework validation
# Then run model evaluations
```

## Environment Variables

Optional environment variables:
```bash
export MODEL_ENDPOINT="http://localhost:8004/completion"
export MODEL_NAME="your-model-name"
export EVALUATOR_EMBEDDING_STRATEGY="fallback"  # Disable embedding models
export EVALUATOR_FORCE_CPU="true"               # Force CPU-only mode
```

## References

- [Architecture](./architecture.md)
- [Configuration](./configuration.md)
- [Troubleshooting](./troubleshooting.md)