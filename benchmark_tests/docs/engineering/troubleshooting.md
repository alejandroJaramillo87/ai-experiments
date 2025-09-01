# Troubleshooting

Common issues and solutions for the AI Model Evaluation Framework.

## Installation Issues

### Import Errors

**Problem**: `ModuleNotFoundError` when running tests or evaluations.

**Solution**: Ensure you're running from the `benchmark_tests` directory:
```bash
cd benchmark_tests
python benchmark_runner.py
```

**Problem**: Missing dependencies.

**Solution**: Install required packages:
```bash
pip install -r requirements.txt
```

### Permission Errors

**Problem**: Permission denied during package installation.

**Solution**: Use user installation:
```bash
pip install --user -r requirements.txt
```

## Evaluation Issues

### API Connection Failures

**Problem**: Connection refused to model endpoint.

**Solution**: Verify the model server is running:
```bash
make check-prerequisites
make test-api-suite
```

**Problem**: Timeout errors during model evaluation.

**Solution**: Increase timeout in configuration or check model response speed:
```bash
# Check if model is responding
curl -X POST http://localhost:8004/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Test","n_predict":5}'
```

### Evaluation Errors

**Problem**: `EnhancedUniversalEvaluator not available` warnings.

**Solution**: This is normal. The system uses fallback evaluation methods automatically.

**Problem**: `NVIDIA GPU not found` warnings.

**Solution**: Install NVIDIA drivers and nvidia-ml-py3 for GPU monitoring, or ignore if not needed.

### Memory Issues

**Problem**: Out of memory during concurrent evaluation.

**Solution**: Reduce worker count:
```bash
python benchmark_runner.py --workers 2  # Instead of default 4
```

**Problem**: Large result files consuming disk space.

**Solution**: Regular cleanup:
```bash
make clean-all
```

## Framework Testing Issues

### Test Failures

**Problem**: Some tests fail during `make test`.

**Solution**: Check if failures are in optional components:
```bash
make test MODE=quick  # Run essential tests only
```

**Problem**: Integration tests fail without model server.

**Solution**: Skip integration tests or set up model server:
```bash
make test SUITE=unit  # Unit tests only
```

### Performance Issues

**Problem**: Tests run slowly.

**Solution**: Use quick mode for development:
```bash
make test MODE=quick SUITE=unit
```

## Configuration Issues

### Environment Variables

**Problem**: Configuration not taking effect.

**Solution**: Verify environment variables are set:
```bash
echo $EVALUATOR_EMBEDDING_STRATEGY
echo $MODEL_ENDPOINT
```

**Problem**: Embedding model loading fails.

**Solution**: Force fallback mode:
```bash
export EVALUATOR_EMBEDDING_STRATEGY=fallback
export EVALUATOR_FORCE_CPU=true
```

## Domain Issues

### Missing Test Files

**Problem**: Test definitions not found.

**Solution**: Verify domain files exist:
```bash
ls domains/reasoning/base_models/
ls domains/creativity/instruct_models/
```

**Problem**: JSON parsing errors in domain files.

**Solution**: Validate JSON syntax:
```bash
python -m json.tool domains/reasoning/base_models/easy.json
```

## Output Issues

### Result Files

**Problem**: Results not being saved.

**Solution**: Check output directory permissions:
```bash
mkdir -p test_results
chmod 755 test_results
```

**Problem**: Incomplete result files.

**Solution**: Check for evaluation errors in logs and ensure sufficient disk space.

## Debug Commands

Diagnostic commands for troubleshooting:

```bash
make debug-help                    # Show debug utilities
make test-enhanced-evaluator       # Test evaluator system
make test-domain-loading           # Test domain file loading
make test-semantic-analyzer        # Test specific components
make debug-calibration-framework   # Step-by-step debugging
```

## Getting Help

### Log Analysis

Enable verbose logging:
```bash
export PYTHONPATH=.
python -v benchmark_runner.py
```

### System Information

Check system status:
```bash
make check-prerequisites  # Complete system check
python --version          # Python version
pip list                  # Installed packages
```

### Common Solutions

1. **Clear cache**: `make clean-all`
2. **Reinstall dependencies**: `pip install --upgrade -r requirements.txt`  
3. **Check working directory**: Ensure you're in `benchmark_tests/`
4. **Verify file permissions**: `chmod +x benchmark_runner.py`
5. **Test basic functionality**: `make test MODE=quick SUITE=unit`

## References

- [Installation](./installation.md)
- [Basic Usage](./basic-usage.md)
- [Configuration](./configuration.md)