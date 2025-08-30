# Contributing Guide

This guide covers how to contribute to the AI Model Evaluation Framework. Think of this as your **development workflow manual** - everything you need to know about coding standards, testing practices, and contribution processes.

## 🎯 **Quick Start for Contributors**

### **Prerequisites**
- **Python 3.8+** with virtual environment support
- **Git** with basic knowledge of branching workflows
- **Text editor** with Python support (VS Code, PyCharm, etc.)
- **Basic understanding** of testing frameworks (pytest)

### **Initial Setup**
```bash
# Clone and setup
git clone https://github.com/your-org/benchmark-tests.git
cd benchmark_tests

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (when available)
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests to verify setup
make test
```

### **Your First Contribution** (10 minutes)
```bash
# Create feature branch
git checkout -b feature/improve-documentation

# Make your changes (start small!)
echo "# Improvement notes" >> NOTES.md

# Run tests
make test

# Commit and push
git add NOTES.md
git commit -m "Add development notes file"
git push origin feature/improve-documentation

# Create pull request
# Visit GitHub and create PR from your branch
```

## 🏗️ **Project Structure for Contributors**

### **Development Focus Areas**
```
benchmark_tests/
├── evaluator/           # 🎯 Core contribution area
│   ├── subjects/        # Add new domain evaluators here
│   ├── advanced/        # Add analysis tools here
│   └── cultural/        # Add cultural validation here
├── domains/             # 🎯 Add new test definitions here
├── tests/               # 🎯 Add tests for your contributions
├── docs/                # 🎯 Update documentation here
└── scripts/             # Add utility scripts here
```

### **Common Contribution Types**
1. **New Evaluators** → `evaluator/subjects/`
2. **Test Definitions** → `domains/{domain}/`
3. **Analysis Tools** → `evaluator/advanced/`
4. **Bug Fixes** → Any relevant directory
5. **Documentation** → `docs/` or inline docstrings
6. **Utilities** → `scripts/` or `evaluator/core/`

## 📝 **Coding Standards**

### **Python Code Style**
We follow **PEP 8** with some project-specific conventions:

```python
# Good: Clear, descriptive names
class ReasoningEvaluator(BaseEvaluator):
    """Evaluates logical reasoning and analytical thinking."""
    
    def __init__(self):
        super().__init__()
        self.evaluator_name = "reasoning"
        self.confidence_threshold = 0.7
    
    def evaluate(self, text: str, context: Dict[str, Any]) -> DomainEvaluationResult:
        """
        Evaluate reasoning quality in the response.
        
        Args:
            text: Model response to evaluate
            context: Test metadata and cultural context
            
        Returns:
            DomainEvaluationResult with reasoning scores
            
        Raises:
            EvaluationError: If evaluation fails
        """
        try:
            return self._perform_reasoning_analysis(text, context)
        except Exception as e:
            logger.error(f"Reasoning evaluation failed: {str(e)}")
            raise EvaluationError(f"Failed to evaluate reasoning: {str(e)}")

# Bad: Unclear names, no documentation
class RE(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.x = "reasoning"
        self.y = 0.7
    
    def evaluate(self, t, c):
        return self.do_stuff(t, c)
```

### **Naming Conventions**
- **Classes**: `PascalCase` (e.g., `CreativityEvaluator`)
- **Functions/Methods**: `snake_case` (e.g., `evaluate_response`)  
- **Variables**: `snake_case` (e.g., `overall_score`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRY_ATTEMPTS`)
- **Files**: `snake_case.py` (e.g., `reasoning_evaluator.py`)

### **Documentation Standards**
Every public function and class must have docstrings:

```python
def analyze_cultural_context(text: str, 
                           cultural_groups: List[str],
                           strict_mode: bool = False) -> CulturalAnalysis:
    """
    Analyze cultural context and appropriateness in text.
    
    This function examines text for cultural elements, potential bias,
    and appropriateness within specified cultural contexts.
    
    Args:
        text: The text to analyze for cultural content
        cultural_groups: List of relevant cultural group identifiers
        strict_mode: If True, apply stricter validation criteria
        
    Returns:
        CulturalAnalysis object containing:
        - bias_score: Cultural bias assessment (0.0-1.0)
        - appropriateness_score: Cultural appropriateness (0.0-1.0)
        - detected_elements: List of detected cultural elements
        - recommendations: List of improvement suggestions
        
    Raises:
        CulturalValidationError: If cultural data is invalid
        ValueError: If cultural_groups is empty
        
    Example:
        >>> groups = ["west_african", "yoruba"]
        >>> analysis = analyze_cultural_context(
        ...     "Traditional storytelling through griot performance",
        ...     groups,
        ...     strict_mode=True
        ... )
        >>> analysis.appropriateness_score
        0.92
    """
    if not cultural_groups:
        raise ValueError("cultural_groups cannot be empty")
    
    # Implementation...
```

### **Type Hints**
Always use type hints for function parameters and return values:

```python
from typing import Dict, List, Optional, Union, Tuple, Any

# Good: Clear type hints
def process_evaluation_results(
    results: List[DomainEvaluationResult],
    weights: Dict[str, float],
    minimum_confidence: float = 0.7
) -> Tuple[float, Dict[str, Any]]:
    """Process and aggregate evaluation results."""
    pass

# Bad: No type hints
def process_evaluation_results(results, weights, minimum_confidence=0.7):
    pass
```

## 🧪 **Testing Standards**

### **Test Organization**
```
tests/
├── unit/                # Test individual components
│   ├── test_evaluators/     # Evaluator-specific tests
│   ├── test_core/          # Core functionality tests
│   └── test_utils/         # Utility function tests
├── integration/         # Test component interactions
│   ├── test_full_pipeline/  # End-to-end pipeline tests
│   └── test_api_integration/ # API integration tests
└── functional/          # Test user workflows
    ├── test_cli/           # Command-line interface tests
    └── test_scenarios/     # Real-world scenario tests
```

### **Writing Unit Tests**
Every new evaluator or component needs comprehensive unit tests:

```python
import unittest
from unittest.mock import Mock, patch
from evaluator.subjects.reasoning_evaluator import ReasoningEvaluator
from evaluator.core.domain_evaluator_base import CulturalContext

class TestReasoningEvaluator(unittest.TestCase):
    """Test cases for ReasoningEvaluator."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.evaluator = ReasoningEvaluator()
        self.sample_context = {
            "test_id": "reasoning_01",
            "difficulty": "medium",
            "cultural_context": CulturalContext()
        }
    
    def test_high_quality_reasoning(self):
        """Test evaluation of high-quality reasoning response."""
        # Arrange
        reasoning_text = '''
        To solve this problem, I'll analyze it step by step:
        
        1. First, let me identify the key assumptions:
           - The data shows a clear trend over time
           - External factors remain constant
           
        2. Next, I'll examine the evidence:
           - Historical data supports this pattern
           - Statistical significance is high (p < 0.05)
           
        3. Therefore, I can conclude that the hypothesis is supported
           by the available evidence.
        '''
        
        # Act
        result = self.evaluator.evaluate(reasoning_text, self.sample_context)
        
        # Assert
        self.assertGreater(result.overall_score, 0.8)
        self.assertGreater(result.get_dimension_score("organization_quality"), 0.85)
        self.assertGreater(result.get_dimension_score("technical_accuracy"), 0.8)
        self.assertTrue(result.metadata.get("has_logical_structure"))
        self.assertGreater(len(result.dimensions[0].evidence), 0)
    
    def test_poor_quality_reasoning(self):
        """Test evaluation of poor reasoning response."""
        # Arrange
        poor_reasoning = "I think this is true because it seems right to me."
        
        # Act  
        result = self.evaluator.evaluate(poor_reasoning, self.sample_context)
        
        # Assert
        self.assertLess(result.overall_score, 0.6)
        self.assertLess(result.get_dimension_score("technical_accuracy"), 0.5)
        self.assertIn("lacks supporting evidence", 
                     [reason.lower() for reason in result.dimensions[0].evidence])
    
    def test_empty_input_handling(self):
        """Test handling of empty or invalid input."""
        # Test empty string
        result = self.evaluator.evaluate("", self.sample_context)
        self.assertEqual(result.overall_score, 0.0)
        
        # Test whitespace only
        result = self.evaluator.evaluate("   \n\t  ", self.sample_context)
        self.assertEqual(result.overall_score, 0.0)
    
    def test_configuration_validation(self):
        """Test evaluator configuration validation."""
        # Test valid configuration
        valid_config = {
            "weights": {"organization_quality": 0.3, "technical_accuracy": 0.4,
                       "completeness": 0.2, "reliability": 0.1},
            "strict_mode": True
        }
        self.evaluator.configure(valid_config)
        self.assertEqual(self.evaluator.settings["strict_mode"], True)
        
        # Test invalid configuration
        invalid_config = {"weights": {"invalid_dimension": 1.0}}
        with self.assertRaises(ConfigurationError):
            self.evaluator.configure(invalid_config)
    
    @patch('evaluator.subjects.reasoning_evaluator.logger')
    def test_error_handling(self, mock_logger):
        """Test error handling and logging."""
        # Force an error by passing invalid context
        invalid_context = {"invalid": "context"}
        
        result = self.evaluator.evaluate("test text", invalid_context)
        
        # Verify error was logged
        mock_logger.error.assert_called()
        
        # Verify graceful failure
        self.assertEqual(result.overall_score, 0.0)
        self.assertFalse(result.metadata.get("success", True))
    
    def test_cultural_context_integration(self):
        """Test integration with cultural context."""
        # Create cultural context
        cultural_context = CulturalContext(
            traditions=["oral_tradition"],
            cultural_groups=["west_african"]
        )
        context = {**self.sample_context, "cultural_context": cultural_context}
        
        # Test with culturally relevant text
        cultural_text = '''
        In the tradition of griot storytelling, this analysis follows
        a narrative structure that builds understanding through layered
        meaning and communal wisdom.
        '''
        
        result = self.evaluator.evaluate(cultural_text, context)
        
        # Verify cultural elements are recognized
        self.assertTrue(any(dim.cultural_relevance > 0 for dim in result.dimensions))
        self.assertGreater(len(result.cultural_context.traditions), 0)

if __name__ == '__main__':
    unittest.main()
```

### **Integration Test Example**
```python
class TestEvaluationPipeline(unittest.TestCase):
    """Test the complete evaluation pipeline."""
    
    def test_full_evaluation_workflow(self):
        """Test complete workflow from test loading to result generation."""
        # Arrange
        runner = BenchmarkTestRunner(
            model_endpoint="http://mock-model:8000",
            model_name="test-model"
        )
        
        # Act
        with patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = {
                "choices": [{"text": "This is a well-reasoned response..."}]
            }
            
            result = runner.execute_single_test(
                test_id="reasoning_basic_01",
                test_type="base"
            )
        
        # Assert
        self.assertTrue(result.success)
        self.assertGreater(len(result.evaluation_results), 0)
        self.assertIsNotNone(result.performance_metrics)
```

### **Running Tests**
```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration

# Run specific test file
python -m pytest tests/unit/test_reasoning_evaluator.py -v

# Run with coverage
python -m pytest --cov=evaluator tests/unit/ --cov-report=html

# Run tests matching pattern
python -m pytest -k "test_reasoning" -v
```

## 🔄 **Development Workflow**

### **Branch Naming Conventions**
- **Feature**: `feature/add-sentiment-evaluator`
- **Bug Fix**: `bugfix/fix-cultural-validation-error`
- **Documentation**: `docs/update-api-reference`
- **Refactoring**: `refactor/simplify-scoring-logic`
- **Performance**: `perf/optimize-concurrent-execution`

### **Commit Message Standards**
Follow conventional commit format:

```bash
# Good commit messages
feat(evaluator): add sentiment analysis evaluator
fix(cultural): resolve bias detection false positives  
docs(api): update evaluator interface documentation
test(reasoning): add edge case tests for empty responses
refactor(core): simplify result aggregation logic
perf(concurrent): optimize thread pool utilization

# Bad commit messages
"Fix stuff"
"Update files"
"Changes"
```

### **Pull Request Process**

#### **Before Creating PR**
```bash
# Ensure your branch is up to date
git checkout main
git pull origin main
git checkout your-feature-branch
git rebase main

# Run full test suite
make test

# Check code style
flake8 evaluator/ --max-line-length=100
black evaluator/ --check

# Update documentation if needed
# Add/update docstrings, README sections, etc.
```

#### **PR Description Template**
```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Changes Made
- Added new `SentimentEvaluator` class to `evaluator/subjects/`
- Implemented sentiment analysis for text responses
- Added comprehensive unit tests
- Updated API reference documentation

## Testing
- [ ] All existing tests pass
- [ ] Added new tests for new functionality  
- [ ] Tested manually with sample data
- [ ] Updated integration tests if needed

## Documentation
- [ ] Updated docstrings
- [ ] Updated API reference if needed
- [ ] Updated user guides if needed
- [ ] Added example usage

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] My code has comprehensive error handling
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] New and existing unit tests pass locally
```

#### **Code Review Guidelines**

**For Reviewers**:
- Focus on correctness, maintainability, and performance
- Check for proper error handling and edge cases
- Verify tests are comprehensive and meaningful
- Ensure documentation is updated and accurate
- Consider security implications of changes

**For Contributors**:
- Respond to feedback constructively
- Make requested changes promptly
- Ask questions if feedback is unclear
- Update tests when changing implementation
- Keep PR scope focused and small when possible

## 📊 **Performance Guidelines**

### **Evaluation Performance**
Evaluators should be efficient and scalable:

```python
class PerformantEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        # Cache expensive computations
        self._pattern_cache = {}
        self._model_cache = None
    
    def evaluate(self, text: str, context: Dict[str, Any]) -> DomainEvaluationResult:
        # Avoid expensive operations in tight loops
        if text in self._pattern_cache:
            patterns = self._pattern_cache[text]
        else:
            patterns = self._extract_patterns(text)
            self._pattern_cache[text] = patterns
        
        # Use lazy loading for expensive resources
        model = self._get_model()
        
        return self._compute_result(patterns, model, context)
    
    def _get_model(self):
        """Lazy load expensive model only when needed."""
        if self._model_cache is None:
            self._model_cache = self._load_expensive_model()
        return self._model_cache
```

### **Memory Management**
```python
# Good: Process results in batches to avoid memory issues
def process_large_batch(self, texts: List[str]) -> List[DomainEvaluationResult]:
    batch_size = 100
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = [self.evaluate(text, {}) for text in batch]
        results.extend(batch_results)
        
        # Clear intermediate data to free memory
        del batch_results
    
    return results
```

### **Concurrency Considerations**
```python
import threading
from concurrent.futures import ThreadPoolExecutor

class ThreadSafeEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()
        self._shared_resource = {}
    
    def evaluate(self, text: str, context: Dict[str, Any]) -> DomainEvaluationResult:
        # Use locks for shared resource access
        with self._lock:
            if text not in self._shared_resource:
                self._shared_resource[text] = self._compute_expensive_feature(text)
        
        # Thread-local processing
        return self._process_result(text, context)
```

## 🐛 **Debugging and Troubleshooting**

### **Logging Best Practices**
```python
import logging

# Set up logger for your module
logger = logging.getLogger(__name__)

class DebuggableEvaluator(BaseEvaluator):
    def evaluate(self, text: str, context: Dict[str, Any]) -> DomainEvaluationResult:
        logger.debug(f"Evaluating text of length {len(text)}")
        
        try:
            # Log key decision points
            if self._is_complex_text(text):
                logger.info("Processing complex text with enhanced analysis")
                result = self._enhanced_evaluation(text, context)
            else:
                logger.debug("Using standard evaluation approach")
                result = self._standard_evaluation(text, context)
            
            logger.debug(f"Evaluation completed with score {result.overall_score}")
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            raise EvaluationError(f"Failed to evaluate: {str(e)}")
```

### **Common Issues and Solutions**

**Issue: Evaluator returns inconsistent scores**
```python
# Solution: Add debugging output to understand score calculation
def _calculate_dimension_score(self, metrics: Dict[str, float]) -> float:
    logger.debug(f"Input metrics: {metrics}")
    
    weighted_score = sum(
        metrics[key] * self.weights.get(key, 0.0)
        for key in metrics.keys()
    )
    
    logger.debug(f"Weighted score before normalization: {weighted_score}")
    
    normalized_score = max(0.0, min(1.0, weighted_score))
    
    logger.debug(f"Final normalized score: {normalized_score}")
    return normalized_score
```

**Issue: Cultural validation failing unexpectedly**
```python
# Solution: Add detailed cultural context logging
def validate_cultural_context(self, text: str, context: CulturalContext) -> bool:
    logger.debug(f"Validating cultural context: {context.to_dict()}")
    
    detected_elements = self._detect_cultural_elements(text)
    logger.debug(f"Detected cultural elements: {detected_elements}")
    
    conflicts = self._check_cultural_conflicts(detected_elements, context)
    if conflicts:
        logger.warning(f"Cultural conflicts detected: {conflicts}")
        return False
    
    return True
```

---

## 🎉 **Recognition and Community**

### **Contributor Recognition**
- Contributors are acknowledged in release notes
- Significant contributions are recognized in the main README
- Community contributions are highlighted in project documentation

### **Getting Help**
- **Code Questions**: Create GitHub issue with "question" label
- **Bug Reports**: Use bug report template in GitHub issues
- **Feature Requests**: Use feature request template
- **General Discussion**: Use GitHub Discussions

### **Community Guidelines**
- Be respectful and inclusive in all interactions
- Provide constructive feedback in code reviews
- Help newcomers get started with their first contributions
- Share knowledge and best practices

---

**Ready to contribute?** Start with a small improvement and work your way up to larger features. Every contribution makes the project better! 🚀