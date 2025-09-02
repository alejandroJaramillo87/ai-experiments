# Evaluator Framework Integration Guide

**AI Workstation Benchmark Tests - Developer Integration Guide**  
**Date:** September 1, 2024  
**Audience:** Developers integrating with the evaluator framework

---

## Overview

This guide provides practical instructions for integrating with the AI Workstation evaluator framework, including setup procedures, common usage patterns, troubleshooting, and best practices.

### Quick Start Checklist

- [ ] Install required dependencies
- [ ] Understand evaluator categories and capabilities  
- [ ] Choose appropriate integration approach
- [ ] Implement error handling and fallbacks
- [ ] Test with available evaluators
- [ ] Monitor performance and accuracy

---

## Installation & Setup

### Dependencies

```bash
# Core dependencies (required)
pip install numpy scipy statistics

# Advanced analysis (optional but recommended)
pip install scikit-learn nltk

# Cultural analysis (optional)
pip install cultural-patterns-db  # Custom package

# Performance monitoring (optional)
pip install psutil memory-profiler
```

### Environment Setup

```python
import sys
from pathlib import Path

# Add evaluator framework to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))
```

### Verification Script

```python
def verify_installation():
    """Verify evaluator framework installation"""
    
    print("🔍 Verifying Evaluator Framework Installation")
    print("=" * 50)
    
    # Check core pipeline
    try:
        from core.cognitive_evaluation_pipeline import CognitiveEvaluationPipeline
        pipeline = CognitiveEvaluationPipeline()
        print("✅ Core pipeline: Available")
        print(f"   Available evaluators: {len(pipeline.available_evaluators)}")
    except ImportError as e:
        print(f"❌ Core pipeline: Failed - {e}")
    
    # Check individual evaluators
    evaluator_checks = [
        ("PatternBasedEvaluator", "evaluator.subjects.pattern_based_evaluator", "PatternBasedEvaluator"),
        ("EnhancedUniversalEvaluator", "evaluator.subjects.enhanced_universal_evaluator", "EnhancedUniversalEvaluator"),
        ("CulturalAuthenticityAnalyzer", "evaluator.cultural.cultural_authenticity", "CulturalAuthenticityAnalyzer"),
        ("ConsistencyValidator", "evaluator.advanced.consistency_validator", "ConsistencyValidator"),
    ]
    
    for name, module_path, class_name in evaluator_checks:
        try:
            module = __import__(module_path, fromlist=[class_name])
            evaluator_class = getattr(module, class_name)
            evaluator = evaluator_class()
            print(f"✅ {name}: Available")
        except ImportError:
            print(f"⚠️  {name}: Not available (optional)")
        except Exception as e:
            print(f"❌ {name}: Error - {e}")

if __name__ == "__main__":
    verify_installation()
```

---

## Integration Approaches

### 1. High-Level Pipeline Integration (Recommended)

**Best for:** Most applications requiring comprehensive evaluation

```python
from core.cognitive_evaluation_pipeline import CognitiveEvaluationPipeline

class MyApplication:
    def __init__(self):
        self.evaluator = CognitiveEvaluationPipeline()
    
    def evaluate_model_response(self, test_case, response):
        """Evaluate model response using comprehensive pipeline"""
        
        # Prepare test data
        test_data = {
            'id': test_case.get('id', 'unknown'),
            'domain': test_case.get('domain', 'general'),
            'prompt': test_case.get('prompt', ''),
            'difficulty': test_case.get('difficulty', 'medium')
        }
        
        # Run evaluation
        result = self.evaluator.evaluate_response(
            test_data=test_data,
            response_text=response,
            model_id=self.model_name
        )
        
        # Extract key metrics
        return {
            'score': result.overall_score,
            'confidence': result.confidence_score,
            'domain': result.cognitive_domain,
            'patterns': result.behavioral_patterns.behavioral_signature if result.behavioral_patterns else None,
            'details': result.evaluation_details
        }
```

### 2. Direct Evaluator Integration

**Best for:** Specific use cases requiring particular evaluators

```python
from evaluator.subjects.pattern_based_evaluator import PatternBasedEvaluator
from evaluator.cultural.cultural_authenticity import CulturalAuthenticityAnalyzer

class SpecializedAnalyzer:
    def __init__(self):
        self.pattern_evaluator = PatternBasedEvaluator()
        
        # Optional cultural evaluator
        try:
            self.cultural_evaluator = CulturalAuthenticityAnalyzer()
            self.has_cultural = True
        except ImportError:
            self.has_cultural = False
    
    def analyze_creative_response(self, prompt, response, model_id):
        """Specialized analysis for creative content"""
        
        # Pattern analysis (always available)
        pattern_result = self.pattern_evaluator.evaluate_patterns(
            response_text=response,
            prompt=prompt,
            test_metadata={'domain': 'creativity'},
            model_id=model_id
        )
        
        results = {
            'creativity_score': pattern_result.quality_indicators.get('engagement_score', 0) * 100,
            'consistency': pattern_result.response_consistency,
            'style': pattern_result.behavioral_signature.get('response_style', 'unknown')
        }
        
        # Cultural analysis if available
        if self.has_cultural and self._contains_cultural_content(response):
            cultural_result = self.cultural_evaluator.analyze_cultural_authenticity(
                content=response,
                cultural_context={'region': 'auto_detect'}
            )
            results['cultural_sensitivity'] = cultural_result.cultural_sensitivity_score
        
        return results
```

### 3. Custom Evaluator Development

**Best for:** Domain-specific evaluation needs

```python
from evaluator.core.domain_evaluator_base import DomainEvaluatorBase
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class CustomEvaluationResult:
    """Custom evaluation result structure"""
    custom_score: float
    domain_specific_metrics: Dict[str, float]
    analysis_details: Dict[str, Any]

class CustomDomainEvaluator(DomainEvaluatorBase):
    """Custom evaluator for specific domain needs"""
    
    def __init__(self, domain_config: Dict[str, Any]):
        super().__init__()
        self.domain_config = domain_config
        self.evaluation_criteria = domain_config.get('criteria', {})
    
    def evaluate(self, prompt: str, response: str, metadata: Dict[str, Any] = None) -> CustomEvaluationResult:
        """Custom evaluation logic"""
        
        # Domain-specific analysis
        domain_score = self._analyze_domain_specifics(response)
        
        # Integrate with existing framework if needed
        base_metrics = {}
        if hasattr(self, '_use_base_evaluators') and self._use_base_evaluators:
            base_metrics = self._get_base_evaluation_metrics(prompt, response)
        
        # Custom scoring logic
        custom_metrics = {
            'domain_adherence': domain_score,
            'criteria_satisfaction': self._check_criteria(response),
            'format_compliance': self._check_format(response)
        }
        
        overall_score = self._calculate_weighted_score(custom_metrics)
        
        return CustomEvaluationResult(
            custom_score=overall_score,
            domain_specific_metrics=custom_metrics,
            analysis_details={'base_metrics': base_metrics, 'metadata': metadata}
        )
    
    def _analyze_domain_specifics(self, response: str) -> float:
        """Implement domain-specific analysis logic"""
        # Custom implementation
        return 0.8  # Example
    
    # Additional custom methods...
```

---

## Common Usage Patterns

### Batch Evaluation

```python
def batch_evaluate(test_cases, model_responses):
    """Efficient batch evaluation with progress tracking"""
    
    pipeline = CognitiveEvaluationPipeline()
    results = []
    
    total_tests = len(test_cases)
    for i, (test_case, response) in enumerate(zip(test_cases, model_responses)):
        
        # Progress tracking
        if i % 10 == 0:
            print(f"Progress: {i}/{total_tests} ({i/total_tests*100:.1f}%)")
        
        # Evaluate with error handling
        try:
            result = pipeline.evaluate_response(
                test_data=test_case,
                response_text=response,
                model_id="batch-model"
            )
            results.append({
                'test_id': test_case.get('id'),
                'score': result.overall_score,
                'success': True,
                'error': None
            })
            
        except Exception as e:
            results.append({
                'test_id': test_case.get('id'),
                'score': 0,
                'success': False,
                'error': str(e)
            })
    
    return results
```

### Comparative Model Analysis

```python
def compare_models(test_cases, model_outputs):
    """Compare multiple models using evaluator framework"""
    
    pipeline = CognitiveEvaluationPipeline()
    model_results = {}
    
    for model_name, responses in model_outputs.items():
        model_scores = []
        
        for test_case, response in zip(test_cases, responses):
            result = pipeline.evaluate_response(
                test_data=test_case,
                response_text=response,
                model_id=model_name
            )
            
            model_scores.append({
                'test_id': test_case['id'],
                'score': result.overall_score,
                'domain': result.cognitive_domain,
                'patterns': result.behavioral_patterns
            })
        
        model_results[model_name] = {
            'average_score': sum(s['score'] for s in model_scores) / len(model_scores),
            'detailed_scores': model_scores,
            'behavioral_summary': analyze_behavioral_patterns(model_scores)
        }
    
    return model_results

def analyze_behavioral_patterns(scores):
    """Analyze behavioral patterns across test results"""
    patterns = {}
    
    if scores and scores[0].get('patterns'):
        # Extract common patterns
        styles = [s['patterns'].behavioral_signature.get('response_style') for s in scores if s.get('patterns')]
        patterns['dominant_style'] = max(set(styles), key=styles.count) if styles else 'unknown'
        
        # Other pattern analysis...
    
    return patterns
```

### Real-Time Evaluation

```python
import threading
import queue
from typing import Callable

class RealTimeEvaluator:
    """Real-time evaluation with async processing"""
    
    def __init__(self, result_callback: Callable = None):
        self.pipeline = CognitiveEvaluationPipeline()
        self.evaluation_queue = queue.Queue()
        self.result_callback = result_callback
        self.worker_thread = None
        self.running = False
    
    def start(self):
        """Start real-time evaluation worker"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._evaluation_worker)
        self.worker_thread.start()
    
    def stop(self):
        """Stop real-time evaluation worker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
    
    def submit_evaluation(self, test_data, response_text, model_id):
        """Submit evaluation request"""
        self.evaluation_queue.put({
            'test_data': test_data,
            'response_text': response_text,
            'model_id': model_id,
            'timestamp': time.time()
        })
    
    def _evaluation_worker(self):
        """Worker thread for processing evaluations"""
        while self.running:
            try:
                request = self.evaluation_queue.get(timeout=1)
                
                # Process evaluation
                result = self.pipeline.evaluate_response(
                    test_data=request['test_data'],
                    response_text=request['response_text'], 
                    model_id=request['model_id']
                )
                
                # Callback with result
                if self.result_callback:
                    self.result_callback(request, result)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Evaluation error: {e}")
```

---

## Error Handling & Fallbacks

### Comprehensive Error Handling

```python
from evaluator.core.evaluation_exceptions import EvaluatorException, EvaluatorUnavailableException

def robust_evaluation(test_data, response_text, model_id):
    """Robust evaluation with comprehensive error handling"""
    
    try:
        # Try full evaluation pipeline
        pipeline = CognitiveEvaluationPipeline()
        result = pipeline.evaluate_response(test_data, response_text, model_id)
        
        return {
            'success': True,
            'result': result,
            'evaluation_method': 'full_pipeline',
            'warnings': []
        }
        
    except EvaluatorUnavailableException as e:
        # Sophisticated evaluator unavailable, try basic evaluation
        warnings = [f"Advanced evaluator unavailable: {e}"]
        
        try:
            basic_result = basic_evaluation_fallback(test_data, response_text, model_id)
            return {
                'success': True,
                'result': basic_result,
                'evaluation_method': 'basic_fallback',
                'warnings': warnings
            }
            
        except Exception as fallback_error:
            # Ultimate fallback
            minimal_result = minimal_evaluation_fallback(test_data, response_text)
            return {
                'success': True,
                'result': minimal_result,
                'evaluation_method': 'minimal_fallback',
                'warnings': warnings + [f"Basic evaluation failed: {fallback_error}"]
            }
    
    except Exception as e:
        # Unexpected error
        return {
            'success': False,
            'result': None,
            'evaluation_method': 'failed',
            'error': str(e)
        }

def basic_evaluation_fallback(test_data, response_text, model_id):
    """Basic evaluation fallback when sophisticated evaluators unavailable"""
    
    # Simple heuristic-based evaluation
    score = len(response_text.split()) * 2  # Example heuristic
    score = min(score, 100)  # Cap at 100
    
    return {
        'overall_score': score,
        'cognitive_domain': test_data.get('domain', 'general'),
        'evaluation_method': 'heuristic',
        'confidence_score': 0.3  # Low confidence for heuristic
    }

def minimal_evaluation_fallback(test_data, response_text):
    """Minimal evaluation when all else fails"""
    return {
        'overall_score': 50.0,  # Neutral score
        'cognitive_domain': 'unknown',
        'evaluation_method': 'minimal',
        'confidence_score': 0.1
    }
```

### Dependency Checking

```python
def check_evaluator_dependencies():
    """Check and report evaluator dependency status"""
    
    dependency_status = {}
    
    # Core dependencies
    core_deps = ['numpy', 'scipy', 'statistics']
    for dep in core_deps:
        try:
            __import__(dep)
            dependency_status[dep] = {'status': 'available', 'required': True}
        except ImportError:
            dependency_status[dep] = {'status': 'missing', 'required': True}
    
    # Optional dependencies
    optional_deps = ['sklearn', 'nltk', 'cultural-patterns-db']
    for dep in optional_deps:
        try:
            __import__(dep)
            dependency_status[dep] = {'status': 'available', 'required': False}
        except ImportError:
            dependency_status[dep] = {'status': 'missing', 'required': False}
    
    # Evaluator availability
    evaluators = {
        'PatternBasedEvaluator': ('evaluator.subjects.pattern_based_evaluator', 'PatternBasedEvaluator'),
        'EnhancedUniversalEvaluator': ('evaluator.subjects.enhanced_universal_evaluator', 'EnhancedUniversalEvaluator'),
        'CulturalAuthenticityAnalyzer': ('evaluator.cultural.cultural_authenticity', 'CulturalAuthenticityAnalyzer')
    }
    
    for name, (module_path, class_name) in evaluators.items():
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            dependency_status[name] = {'status': 'available', 'required': False}
        except ImportError:
            dependency_status[name] = {'status': 'missing', 'required': False}
    
    return dependency_status
```

---

## Performance Optimization

### Caching Strategies

```python
import functools
import hashlib

class EvaluatorCache:
    """Caching layer for expensive evaluations"""
    
    def __init__(self, cache_size: int = 1000):
        self.cache = {}
        self.cache_size = cache_size
        self.access_order = []
    
    def get_cache_key(self, test_data, response_text, model_id):
        """Generate cache key for evaluation"""
        content = f"{test_data.get('id', '')}{response_text}{model_id}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, cache_key):
        """Get cached result"""
        if cache_key in self.cache:
            # Update access order
            self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            return self.cache[cache_key]
        return None
    
    def put(self, cache_key, result):
        """Cache evaluation result"""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
        self.access_order.append(cache_key)

class CachedEvaluator:
    """Evaluator with caching support"""
    
    def __init__(self):
        self.pipeline = CognitiveEvaluationPipeline()
        self.cache = EvaluatorCache()
    
    def evaluate_response(self, test_data, response_text, model_id):
        """Evaluate with caching"""
        
        # Check cache first
        cache_key = self.cache.get_cache_key(test_data, response_text, model_id)
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        # Evaluate and cache
        result = self.pipeline.evaluate_response(test_data, response_text, model_id)
        self.cache.put(cache_key, result)
        
        return result
```

### Resource Monitoring

```python
import psutil
import time
from contextlib import contextmanager

@contextmanager
def monitor_resources(operation_name="evaluation"):
    """Monitor resource usage during evaluation"""
    
    # Get initial resource state
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()
    
    try:
        yield
    finally:
        # Calculate resource usage
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        duration = end_time - start_time
        memory_increase = end_memory - start_memory
        
        print(f"📊 {operation_name} Resource Usage:")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Memory increase: {memory_increase:.2f}MB")
        print(f"   Final memory: {end_memory:.2f}MB")

# Usage example
def monitored_evaluation(test_data, response_text, model_id):
    """Evaluation with resource monitoring"""
    
    with monitor_resources("comprehensive_evaluation"):
        pipeline = CognitiveEvaluationPipeline()
        result = pipeline.evaluate_response(test_data, response_text, model_id)
    
    return result
```

---

## Testing & Validation

### Integration Testing

```python
import unittest

class EvaluatorIntegrationTest(unittest.TestCase):
    """Test evaluator framework integration"""
    
    def setUp(self):
        self.pipeline = CognitiveEvaluationPipeline()
        self.test_data = {
            'id': 'integration_test_001',
            'domain': 'reasoning',
            'prompt': 'Test prompt for integration testing'
        }
        self.test_response = "Test response for evaluator integration testing"
    
    def test_basic_evaluation(self):
        """Test basic evaluation functionality"""
        result = self.pipeline.evaluate_response(
            test_data=self.test_data,
            response_text=self.test_response,
            model_id="test-model"
        )
        
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 100)
        self.assertEqual(result.cognitive_domain, 'reasoning')
    
    def test_evaluator_availability(self):
        """Test evaluator availability checking"""
        available_evaluators = self.pipeline.available_evaluators
        self.assertIsInstance(available_evaluators, list)
        self.assertGreater(len(available_evaluators), 0)
    
    def test_error_handling(self):
        """Test error handling with invalid input"""
        with self.assertRaises((ValueError, TypeError)):
            self.pipeline.evaluate_response(
                test_data=None,  # Invalid input
                response_text=self.test_response,
                model_id="test-model"
            )
    
    def test_fallback_behavior(self):
        """Test fallback behavior when evaluators unavailable"""
        # This would require mocking unavailable evaluators
        pass

if __name__ == '__main__':
    unittest.main()
```

---

## Troubleshooting Guide

### Common Issues

#### 1. Import Errors

**Problem**: `ImportError: No module named 'evaluator.subjects.pattern_based_evaluator'`

**Solution**:
```python
# Check Python path
import sys
print("Python path:", sys.path)

# Add correct paths
sys.path.insert(0, '/path/to/benchmark_tests')
sys.path.insert(0, '/path/to/benchmark_tests/core')
```

#### 2. Evaluator Unavailable

**Problem**: Sophisticated evaluators not available

**Solution**:
```python
# Check dependency status
dependencies = check_evaluator_dependencies()
for name, status in dependencies.items():
    if status['status'] == 'missing' and status['required']:
        print(f"Install required dependency: {name}")
    elif status['status'] == 'missing':
        print(f"Optional dependency missing: {name}")

# Use availability checking
pipeline = CognitiveEvaluationPipeline()
print(f"Available evaluators: {pipeline.available_evaluators}")
```

#### 3. Performance Issues

**Problem**: Slow evaluation performance

**Solution**:
```python
# Use caching
cached_evaluator = CachedEvaluator()

# Monitor resource usage
with monitor_resources():
    result = cached_evaluator.evaluate_response(test_data, response, model_id)

# Optimize for batch processing
results = batch_evaluate(test_cases, responses)
```

#### 4. Memory Issues

**Problem**: High memory usage during evaluation

**Solution**:
```python
# Use selective evaluator loading
class MemoryOptimizedEvaluator:
    def __init__(self):
        self.evaluators = {}  # Lazy loading
    
    def get_evaluator(self, evaluator_type):
        if evaluator_type not in self.evaluators:
            # Load only when needed
            self.evaluators[evaluator_type] = create_evaluator(evaluator_type)
        return self.evaluators[evaluator_type]
    
    def clear_cache(self):
        # Clear loaded evaluators to free memory
        self.evaluators.clear()
```

---

## Best Practices

### 1. Always Check Availability

```python
# Good practice
pipeline = CognitiveEvaluationPipeline()
if 'pattern_based' in pipeline.available_evaluators:
    # Use sophisticated evaluation
    pass
else:
    # Use fallback approach
    pass
```

### 2. Implement Graceful Fallbacks

```python
# Always have fallback options
def evaluate_with_fallbacks(test_data, response, model_id):
    try:
        return full_evaluation(test_data, response, model_id)
    except EvaluatorUnavailableException:
        return basic_evaluation(test_data, response, model_id)
    except Exception:
        return minimal_evaluation(test_data, response, model_id)
```

### 3. Monitor Performance

```python
# Regular performance monitoring
with monitor_resources():
    results = batch_evaluate(test_cases, responses)
```

### 4. Cache Expensive Operations

```python
# Cache results for repeated evaluations
evaluator = CachedEvaluator()
```

### 5. Validate Input Data

```python
def validate_test_data(test_data):
    required_fields = ['id', 'domain']
    for field in required_fields:
        if field not in test_data:
            raise ValueError(f"Missing required field: {field}")
```

---

**This integration guide provides comprehensive coverage for developers working with the evaluator framework. For specific use cases not covered here, refer to the API documentation and example implementations.**

---

**Document Version:** 1.0  
**Last Updated:** September 1, 2024  
**Next Review:** After integration feedback and framework updates