# Contributing

Development workflow and guidelines for the AI Model Evaluation Framework.

## Development Setup

Clone and setup the development environment:
```bash
git clone <repository-url>
cd benchmark_tests
pip install -r requirements.txt
```

Verify setup:
```bash
make test MODE=quick
```

## Development Workflow

### Standard Contribution Process

1. Fork the repository
2. Create a feature branch from main
3. Make changes with tests
4. Verify all tests pass
5. Submit pull request

### Branch Naming

Use descriptive branch names:
```bash
git checkout -b feature/add-new-evaluator
git checkout -b fix/api-timeout-handling  
git checkout -b docs/update-installation-guide
```

### Commit Guidelines

Write clear commit messages:
```bash
git commit -m "Add entropy calculation to creativity evaluator"
git commit -m "Fix timeout handling in benchmark runner"
git commit -m "Update installation documentation"
```

## Testing Requirements

### Before Submitting

All contributions must pass:
```bash
make test                    # Full test suite
make test MODE=coverage      # Coverage requirements
```

### Test Categories

**Unit Tests**: Test individual components
```bash
make test SUITE=unit
```

**Integration Tests**: Test component interactions
```bash  
make test SUITE=integration
```

**Framework Tests**: Test end-to-end workflows
```bash
make test-enhanced-evaluator
make test-api-suite
```

## Code Standards

### Python Style

Follow PEP 8 guidelines. Key requirements:
- 4-space indentation
- Maximum line length 88 characters
- Descriptive variable names
- Type hints for function signatures

### Documentation

All new modules require:
- Module-level docstring explaining purpose
- Function docstrings for public functions
- Class docstrings for public classes
- Inline comments for complex logic

### Error Handling

Use consistent error handling patterns:
```python
try:
    result = risky_operation()
except SpecificException as e:
    logger.warning(f"Operation failed: {e}")
    return fallback_result()
```

## Adding New Components

### New Evaluators

Create evaluators by inheriting from base classes:
```python
class NewDomainEvaluator(MultiDimensionalEvaluator):
    def evaluate(self, response, test_metadata):
        # Implementation here
        return DomainEvaluationResult(...)
```

### New Domains

Add domain test definitions:
1. Create JSON files in `domains/new_domain/`
2. Follow existing schema patterns
3. Include cultural context metadata
4. Add both base and instruct model versions

### New Tests

Add tests for new functionality:
```python
def test_new_feature():
    # Arrange
    setup_test_data()
    
    # Act  
    result = execute_feature()
    
    # Assert
    assert result.is_valid()
```

## Architecture Guidelines

### Modularity

Keep components loosely coupled:
- Use dependency injection where possible
- Avoid direct imports between domain modules
- Implement clear interfaces

### Error Recovery

Implement graceful degradation:
- Provide fallback implementations
- Log warnings rather than failing completely
- Use configuration to control behavior

### Performance

Consider performance implications:
- Avoid loading large models unnecessarily
- Cache expensive computations
- Use lazy loading for optional components

## Documentation Requirements

### Linux Philosophy Compliance

All documentation must follow established standards:
- Single purpose per document
- Maximum 4 heading levels
- No emojis or excessive formatting
- Clear, direct language

### Required Documentation

New features require:
- Updates to relevant engineering documentation
- Domain documentation if adding new domains
- API reference updates for new interfaces

## Review Process

### Pull Request Requirements

Include in pull request:
- Clear description of changes
- Test results demonstrating functionality
- Documentation updates
- Reference to related issues

### Review Criteria

Reviewers evaluate:
- Code functionality and correctness
- Test coverage and quality
- Documentation completeness
- Architecture consistency
- Performance implications

## Common Development Tasks

### Running Specific Tests
```bash
make test-specific FILE=tests/unit/test_new_feature.py
```

### Adding Debug Output
```bash
python -v benchmark_runner.py  # Verbose execution
export PYTHONPATH=.            # Ensure imports work
```

### Testing Configuration Changes
```bash
export EVALUATOR_EMBEDDING_STRATEGY=fallback
make test-enhanced-evaluator
```

## Getting Help

### Development Resources
- Review existing code patterns in similar modules
- Check [Architecture](./architecture.md) for design principles  
- Use [Troubleshooting](./troubleshooting.md) for common issues

### Debug Tools
```bash
make debug-help                    # Available debug utilities
make debug-calibration-framework   # Framework debugging
```

## References

- [Architecture](./architecture.md)
- [API Reference](./api-reference.md)
- [Basic Usage](./basic-usage.md)