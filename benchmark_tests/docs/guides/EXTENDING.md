# Extension Guide

This guide shows how to extend the AI model evaluation framework with new evaluators, domains, and analysis tools. Think of this as your **developer customization manual** - everything you need to add new capabilities.

## 🎯 **Extension Overview**

The system is designed for easy extension through:
- **Plugin Architecture**: Add new evaluators without modifying existing code
- **Data-Driven Tests**: Add new test types through JSON configuration
- **Modular Design**: Each component can be extended independently
- **Standard Interfaces**: Consistent contracts make integration simple

## 🧩 **Adding New Evaluators** (Recommended Starting Point)

### **Step 1: Understand the Base Interface**

All evaluators inherit from `BaseEvaluator`:

```python
from evaluator.core.base_evaluator import BaseEvaluator
from evaluator.core.evaluation_result import EvaluationResult

class YourCustomEvaluator(BaseEvaluator):
    """
    Evaluates responses for your specific domain.
    """
    
    def __init__(self):
        super().__init__()
        self.evaluator_name = "custom_domain"
        self.version = "1.0.0"
    
    def evaluate(self, text: str, context: dict) -> EvaluationResult:
        """
        Main evaluation method - analyze the text and return scores.
        
        Args:
            text: The model's response to evaluate
            context: Test metadata and cultural context
            
        Returns:
            EvaluationResult with scores and reasoning
        """
        # Your evaluation logic here
        pass
    
    def get_dimensions(self) -> list:
        """Return the dimensions this evaluator scores."""
        return ["custom_quality", "domain_expertise", "practical_value"]
    
    def configure(self, settings: dict) -> None:
        """Configure evaluator with custom settings."""
        self.settings = settings
```

### **Step 2: Implement Your Evaluation Logic**

**Example: Code Quality Evaluator**

```python
import re
from typing import Dict, List
from evaluator.core.base_evaluator import BaseEvaluator
from evaluator.core.evaluation_result import EvaluationResult

class CodeQualityEvaluator(BaseEvaluator):
    """
    Evaluates code quality in model responses.
    Useful for evaluating coding assistance, technical documentation, etc.
    """
    
    def __init__(self):
        super().__init__()
        self.evaluator_name = "code_quality"
        self.version = "1.0.0"
        
        # Define code quality indicators
        self.good_patterns = [
            r'def\s+\w+\([^)]*\):',  # Function definitions
            r'class\s+\w+',          # Class definitions  
            r'#\s+\w+',              # Comments
            r'""".*?"""',            # Docstrings
        ]
        
        self.bad_patterns = [
            r'\b[A-Z]{3,}\b',        # ALL_CAPS variables (usually bad)
            r'\bprint\(',            # Print statements (debugging artifacts)
            r'TODO|FIXME|HACK',      # TODO comments
        ]
    
    def evaluate(self, text: str, context: dict) -> EvaluationResult:
        """Evaluate code quality in the response."""
        
        # Check if response contains code
        has_code = self._contains_code(text)
        if not has_code:
            return self._create_no_code_result()
        
        # Analyze different quality dimensions
        structure_score = self._analyze_structure(text)
        readability_score = self._analyze_readability(text) 
        best_practices_score = self._analyze_best_practices(text)
        documentation_score = self._analyze_documentation(text)
        
        # Calculate overall score
        overall_score = (
            structure_score * 0.3 +
            readability_score * 0.25 +
            best_practices_score * 0.25 + 
            documentation_score * 0.2
        )
        
        # Generate detailed reasoning
        reasoning = self._generate_reasoning(
            structure_score, readability_score, 
            best_practices_score, documentation_score
        )
        
        return EvaluationResult(
            overall_score=overall_score,
            dimensions={
                "code_structure": structure_score,
                "readability": readability_score,
                "best_practices": best_practices_score,
                "documentation": documentation_score
            },
            confidence=self._calculate_confidence(text),
            reasoning=reasoning,
            metadata={
                "has_code": True,
                "code_lines": len([line for line in text.split('\n') if line.strip()]),
                "evaluator": self.evaluator_name,
                "version": self.version
            }
        )
    
    def _contains_code(self, text: str) -> bool:
        """Check if text contains code snippets."""
        code_indicators = [
            '```',                    # Code blocks
            'def ', 'class ',         # Python keywords
            'function ', 'var ',      # JavaScript keywords
            'import ', 'from ',       # Import statements
            'return ', 'if ', 'for '  # Control flow
        ]
        return any(indicator in text.lower() for indicator in code_indicators)
    
    def _analyze_structure(self, text: str) -> float:
        """Analyze code structure quality (0-100)."""
        score = 70  # Start with baseline
        
        # Bonus for good structure patterns
        for pattern in self.good_patterns:
            if re.search(pattern, text):
                score += 5
                
        # Penalty for poor structure
        lines = text.split('\n')
        very_long_lines = [line for line in lines if len(line) > 120]
        score -= len(very_long_lines) * 2
        
        return max(0, min(100, score))
    
    def _analyze_readability(self, text: str) -> float:
        """Analyze code readability (0-100)."""
        score = 70
        
        # Check for meaningful variable names
        if re.search(r'\b[a-zA-Z][a-zA-Z_]*[a-zA-Z]\b', text):
            score += 10
            
        # Check for comments
        comment_ratio = len(re.findall(r'#.*', text)) / max(1, len(text.split('\n')))
        score += comment_ratio * 30
        
        return max(0, min(100, score))
    
    def _analyze_best_practices(self, text: str) -> float:
        """Analyze adherence to best practices (0-100)."""
        score = 80
        
        # Penalty for bad patterns
        for pattern in self.bad_patterns:
            matches = len(re.findall(pattern, text))
            score -= matches * 5
            
        # Bonus for good practices
        if 'try:' in text and 'except' in text:
            score += 10  # Error handling
            
        return max(0, min(100, score))
    
    def _analyze_documentation(self, text: str) -> float:
        """Analyze code documentation quality (0-100)."""
        score = 60
        
        # Check for docstrings
        if '"""' in text:
            score += 20
            
        # Check for inline comments
        comment_lines = len(re.findall(r'#.*\S', text))
        code_lines = len([line for line in text.split('\n') 
                         if line.strip() and not line.strip().startswith('#')])
        
        if code_lines > 0:
            comment_ratio = comment_lines / code_lines
            score += min(20, comment_ratio * 100)
        
        return max(0, min(100, score))
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence in the evaluation (0.0-1.0)."""
        # More code = higher confidence
        code_lines = len([line for line in text.split('\n') 
                         if line.strip() and not line.strip().startswith('#')])
        
        if code_lines >= 10:
            return 0.9
        elif code_lines >= 5:
            return 0.8
        elif code_lines >= 3:
            return 0.7
        else:
            return 0.6
    
    def _generate_reasoning(self, structure: float, readability: float, 
                          best_practices: float, documentation: float) -> List[str]:
        """Generate human-readable reasoning for the scores."""
        reasoning = []
        
        # Structure assessment
        if structure >= 85:
            reasoning.append("Excellent code structure with clear organization")
        elif structure >= 70:
            reasoning.append("Good code structure with minor issues")
        else:
            reasoning.append("Code structure needs improvement")
        
        # Readability assessment  
        if readability >= 85:
            reasoning.append("Code is highly readable with clear variable names")
        elif readability >= 70:
            reasoning.append("Code readability is acceptable")
        else:
            reasoning.append("Code readability could be improved")
            
        # Best practices assessment
        if best_practices >= 85:
            reasoning.append("Follows coding best practices well")
        elif best_practices < 70:
            reasoning.append("Some coding practices need improvement")
            
        # Documentation assessment
        if documentation >= 80:
            reasoning.append("Well-documented code with good comments")
        elif documentation < 60:
            reasoning.append("Code documentation is insufficient")
        
        return reasoning
    
    def _create_no_code_result(self) -> EvaluationResult:
        """Handle responses that don't contain code."""
        return EvaluationResult(
            overall_score=0,
            dimensions={
                "code_structure": 0,
                "readability": 0, 
                "best_practices": 0,
                "documentation": 0
            },
            confidence=0.9,
            reasoning=["Response does not contain code to evaluate"],
            metadata={"has_code": False, "evaluator": self.evaluator_name}
        )
    
    def get_dimensions(self) -> List[str]:
        """Return dimensions this evaluator measures."""
        return ["code_structure", "readability", "best_practices", "documentation"]
    
    def configure(self, settings: Dict) -> None:
        """Configure evaluator with custom settings."""
        if 'good_patterns' in settings:
            self.good_patterns.extend(settings['good_patterns'])
        if 'bad_patterns' in settings:
            self.bad_patterns.extend(settings['bad_patterns'])
```

### **Step 3: Register Your Evaluator**

Add your evaluator to the system configuration:

**evaluator/__init__.py**:
```python
from .subjects.code_quality_evaluator import CodeQualityEvaluator

# Register evaluator
EVALUATOR_REGISTRY = {
    'reasoning': ReasoningEvaluator,
    'creativity': CreativityEvaluator,  
    'language': LanguageEvaluator,
    'code_quality': CodeQualityEvaluator,  # Your new evaluator
}
```

**domain_config.json**:
```json
{
  "code_review": {
    "evaluators": ["code_quality"],
    "weights": {
      "code_structure": 0.3,
      "readability": 0.25,
      "best_practices": 0.25,
      "documentation": 0.2
    },
    "min_code_lines": 3,
    "require_comments": true
  }
}
```

### **Step 4: Test Your Evaluator**

Create unit tests for your evaluator:

**tests/unit/test_code_quality_evaluator.py**:
```python
import unittest
from evaluator.subjects.code_quality_evaluator import CodeQualityEvaluator

class TestCodeQualityEvaluator(unittest.TestCase):
    
    def setUp(self):
        self.evaluator = CodeQualityEvaluator()
    
    def test_high_quality_code(self):
        """Test evaluation of high-quality code."""
        code_text = '''
        def calculate_fibonacci(n: int) -> int:
            """
            Calculate the nth Fibonacci number.
            
            Args:
                n: The position in the Fibonacci sequence
                
            Returns:
                The nth Fibonacci number
            """
            if n <= 1:
                return n
            return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
        '''
        
        result = self.evaluator.evaluate(code_text, {})
        
        self.assertGreater(result.overall_score, 80)
        self.assertGreater(result.dimensions['documentation'], 80)
        self.assertTrue(result.confidence > 0.8)
    
    def test_poor_quality_code(self):
        """Test evaluation of poor-quality code."""
        code_text = '''
        def f(x):
            print(x)  # TODO: fix this hack
            return x*2
        '''
        
        result = self.evaluator.evaluate(code_text, {})
        
        self.assertLess(result.overall_score, 60)
        self.assertLess(result.dimensions['documentation'], 50)
    
    def test_no_code_content(self):
        """Test handling of non-code content."""
        text = "This is just regular text with no code."
        
        result = self.evaluator.evaluate(text, {})
        
        self.assertEqual(result.overall_score, 0)
        self.assertFalse(result.metadata['has_code'])

if __name__ == '__main__':
    unittest.main()
```

Run your tests:
```bash
python -m pytest tests/unit/test_code_quality_evaluator.py -v
```

## 🎯 **Adding New Test Domains**

### **Step 1: Create Domain Directory Structure**
```bash
mkdir -p domains/code_review/base_models
mkdir -p domains/code_review/instruct_models
```

### **Step 2: Define Test Cases**

**domains/code_review/base_models/basic_python_01.json**:
```json
{
  "test_id": "basic_python_01",
  "category": "code_review_basic",
  "difficulty": "easy",
  "prompt": "Write a Python function that takes a list of numbers and returns the sum of all even numbers in the list. Include proper documentation and error handling.",
  "metadata": {
    "domain": "code_review",
    "programming_language": "python",
    "estimated_time": 60,
    "requires_cultural_context": false,
    "expected_evaluators": ["code_quality"],
    "target_dimensions": ["code_structure", "documentation", "best_practices"]
  },
  "expected_elements": [
    "function definition with clear name",
    "input validation or error handling", 
    "docstring documentation",
    "efficient algorithm implementation",
    "appropriate return value"
  ]
}
```

**domains/code_review/base_models/complex_algorithm_01.json**:
```json
{
  "test_id": "complex_algorithm_01", 
  "category": "code_review_complex",
  "difficulty": "hard",
  "prompt": "Implement a binary search tree class in Python with insert, search, and delete methods. Include comprehensive documentation, error handling, and example usage.",
  "metadata": {
    "domain": "code_review",
    "programming_language": "python",
    "estimated_time": 300,
    "requires_cultural_context": false,
    "expected_evaluators": ["code_quality", "reasoning"],
    "target_dimensions": ["code_structure", "technical_accuracy", "documentation", "completeness"]
  },
  "expected_elements": [
    "class definition with proper structure",
    "all required methods implemented",
    "comprehensive docstrings",
    "error handling for edge cases",
    "example usage demonstration",
    "appropriate data structure design"
  ]
}
```

### **Step 3: Create Category Mapping**

**domains/code_review/categories.json**:
```json
{
  "categories": {
    "code_review_basic": {
      "description": "Basic code quality and structure evaluation",
      "difficulty_range": ["easy", "medium"],
      "evaluators": ["code_quality"],
      "expected_response_time": 120
    },
    "code_review_complex": {
      "description": "Complex algorithm and architecture evaluation", 
      "difficulty_range": ["medium", "hard"],
      "evaluators": ["code_quality", "reasoning"],
      "expected_response_time": 300
    },
    "code_review_optimization": {
      "description": "Performance and optimization assessment",
      "difficulty_range": ["medium", "hard"],
      "evaluators": ["code_quality", "reasoning"],
      "expected_response_time": 240
    }
  }
}
```

### **Step 4: Test Your New Domain**

```bash
# Test single code review task
python benchmark_runner.py --test-type base --mode single \
  --test-id basic_python_01 --model "coding-model"

# Test entire code review category
python benchmark_runner.py --test-type base --mode category \
  --category code_review_basic --model "coding-model"
```

## 🔧 **Adding Advanced Analysis Tools**

### **Example: Code Complexity Analyzer**

**evaluator/advanced/code_complexity_analyzer.py**:
```python
import ast
import re
from typing import Dict, List, Tuple

class CodeComplexityAnalyzer:
    """
    Analyzes code complexity metrics in model responses.
    Integrates with existing evaluators to provide deeper insights.
    """
    
    def __init__(self):
        self.analyzer_name = "code_complexity"
        self.version = "1.0.0"
    
    def analyze_complexity(self, code_text: str) -> Dict:
        """
        Analyze various complexity metrics.
        
        Returns:
            Dict with complexity metrics and analysis
        """
        try:
            # Parse Python code into AST
            tree = ast.parse(code_text)
            
            metrics = {
                'cyclomatic_complexity': self._cyclomatic_complexity(tree),
                'nesting_depth': self._max_nesting_depth(tree),
                'function_count': self._count_functions(tree),
                'line_count': len(code_text.split('\n')),
                'complexity_rating': 'low'  # Will be calculated
            }
            
            # Calculate overall complexity rating
            metrics['complexity_rating'] = self._calculate_complexity_rating(metrics)
            
            return {
                'metrics': metrics,
                'recommendations': self._generate_recommendations(metrics),
                'confidence': self._calculate_confidence(code_text)
            }
            
        except SyntaxError:
            return self._handle_syntax_error(code_text)
    
    def _cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            # Decision points that increase complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _max_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth."""
        def get_depth(node, current_depth=0):
            max_depth = current_depth
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                    child_depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            
            return max_depth
        
        return get_depth(tree)
    
    def _count_functions(self, tree: ast.AST) -> int:
        """Count number of function definitions."""
        return len([node for node in ast.walk(tree) 
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))])
    
    def _calculate_complexity_rating(self, metrics: Dict) -> str:
        """Calculate overall complexity rating."""
        cyclomatic = metrics['cyclomatic_complexity']
        nesting = metrics['nesting_depth']
        
        if cyclomatic > 15 or nesting > 5:
            return 'very_high'
        elif cyclomatic > 10 or nesting > 4:
            return 'high'
        elif cyclomatic > 5 or nesting > 3:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if metrics['cyclomatic_complexity'] > 10:
            recommendations.append("Consider breaking down complex functions into smaller ones")
        
        if metrics['nesting_depth'] > 4:
            recommendations.append("Reduce nesting depth by using guard clauses or extracting methods")
        
        if metrics['line_count'] > 50 and metrics['function_count'] == 1:
            recommendations.append("Consider splitting long functions into multiple smaller functions")
        
        if not recommendations:
            recommendations.append("Code complexity is within acceptable ranges")
        
        return recommendations
    
    def _calculate_confidence(self, code_text: str) -> float:
        """Calculate confidence in the analysis."""
        # Higher confidence for longer, more complete code
        lines = len([line for line in code_text.split('\n') if line.strip()])
        
        if lines >= 20:
            return 0.95
        elif lines >= 10:
            return 0.85
        elif lines >= 5:
            return 0.75
        else:
            return 0.65
    
    def _handle_syntax_error(self, code_text: str) -> Dict:
        """Handle code that can't be parsed."""
        return {
            'metrics': {
                'cyclomatic_complexity': 0,
                'nesting_depth': 0,
                'function_count': 0,
                'line_count': len(code_text.split('\n')),
                'complexity_rating': 'unknown'
            },
            'recommendations': ["Code contains syntax errors that prevent analysis"],
            'confidence': 0.1
        }
```

### **Integration with Existing Evaluators**

Modify your code quality evaluator to use the complexity analyzer:

```python
from evaluator.advanced.code_complexity_analyzer import CodeComplexityAnalyzer

class CodeQualityEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.complexity_analyzer = CodeComplexityAnalyzer()
    
    def evaluate(self, text: str, context: dict) -> EvaluationResult:
        # ... existing code ...
        
        # Add complexity analysis
        complexity_analysis = self.complexity_analyzer.analyze_complexity(text)
        
        # Adjust scores based on complexity
        if complexity_analysis['metrics']['complexity_rating'] == 'very_high':
            best_practices_score *= 0.8  # Penalize high complexity
        
        # Add complexity info to metadata
        result.metadata.update({
            'complexity_analysis': complexity_analysis
        })
        
        return result
```

## 📚 **Creating Custom Configuration Schemas**

### **Evaluator Configuration Schema**

**evaluator/schemas/code_quality_config.json**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Code Quality Evaluator Configuration",
  "type": "object",
  "properties": {
    "weights": {
      "type": "object",
      "properties": {
        "code_structure": {"type": "number", "minimum": 0, "maximum": 1},
        "readability": {"type": "number", "minimum": 0, "maximum": 1},
        "best_practices": {"type": "number", "minimum": 0, "maximum": 1},
        "documentation": {"type": "number", "minimum": 0, "maximum": 1}
      },
      "required": ["code_structure", "readability", "best_practices", "documentation"]
    },
    "complexity_thresholds": {
      "type": "object",
      "properties": {
        "max_cyclomatic": {"type": "integer", "minimum": 1},
        "max_nesting": {"type": "integer", "minimum": 1},
        "max_function_length": {"type": "integer", "minimum": 1}
      }
    },
    "programming_languages": {
      "type": "array",
      "items": {"type": "string"},
      "default": ["python", "javascript", "java", "cpp"]
    }
  },
  "required": ["weights"]
}
```

### **Test Domain Schema**

**domains/schemas/code_review_test.json**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Code Review Test Definition",
  "type": "object",
  "properties": {
    "test_id": {"type": "string"},
    "category": {"type": "string", "pattern": "^code_review_"},
    "difficulty": {"enum": ["easy", "medium", "hard"]},
    "prompt": {"type": "string", "minLength": 50},
    "metadata": {
      "type": "object",
      "properties": {
        "programming_language": {"type": "string"},
        "expected_evaluators": {"type": "array", "items": {"type": "string"}},
        "target_dimensions": {"type": "array", "items": {"type": "string"}}
      },
      "required": ["programming_language", "expected_evaluators"]
    },
    "expected_elements": {
      "type": "array",
      "items": {"type": "string"},
      "minItems": 1
    }
  },
  "required": ["test_id", "category", "difficulty", "prompt", "metadata"]
}
```

## 🚀 **Best Practices for Extensions**

### **Code Quality**
1. **Follow existing patterns**: Study existing evaluators before creating new ones
2. **Comprehensive testing**: Test edge cases, error conditions, and normal operation
3. **Clear documentation**: Document your evaluator's purpose, usage, and limitations
4. **Performance considerations**: Optimize for reasonable execution times

### **Error Handling**
```python
class RobustEvaluator(BaseEvaluator):
    def evaluate(self, text: str, context: dict) -> EvaluationResult:
        try:
            return self._perform_evaluation(text, context)
        except Exception as e:
            # Log the error
            logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            
            # Return graceful failure result
            return EvaluationResult(
                overall_score=0,
                dimensions={dim: 0 for dim in self.get_dimensions()},
                confidence=0.0,
                reasoning=[f"Evaluation failed due to: {str(e)}"],
                metadata={"error": str(e), "evaluator": self.evaluator_name}
            )
```

### **Configuration Management**
```python
class ConfigurableEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.config = self._load_default_config()
    
    def configure(self, settings: dict) -> None:
        """Merge custom settings with defaults."""
        self.config = {**self.config, **settings}
        self._validate_config()
    
    def _load_default_config(self) -> dict:
        """Load default configuration."""
        return {
            "strict_mode": False,
            "timeout": 30,
            "min_confidence": 0.7
        }
    
    def _validate_config(self):
        """Validate configuration values."""
        if self.config.get("min_confidence", 0) > 1.0:
            raise ValueError("min_confidence must be <= 1.0")
```

---

This extension system allows you to add sophisticated evaluation capabilities while maintaining the framework's consistency and reliability. Start with simple evaluators and gradually add complexity as you become more familiar with the system.