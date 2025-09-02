#!/usr/bin/env python3
"""
Test script for Enhanced Universal Evaluator integration with benchmark_runner.py

This script tests the Phase 1 integration of enhanced evaluation capabilities
with the reasoning domain, focusing on multi-tier scoring functionality.

"""

import sys
import json
import logging
from pathlib import Path

# Add the benchmark_tests directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Import our evaluator components
try:
    from evaluator.subjects.enhanced_universal_evaluator import EnhancedUniversalEvaluator
    from evaluator.subjects import ReasoningType
    print("✅ EnhancedUniversalEvaluator imported successfully")
    ENHANCED_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import EnhancedUniversalEvaluator: {e}")
    ENHANCED_AVAILABLE = False

try:
    from evaluator.subjects import UniversalEvaluator, evaluate_reasoning
    print("✅ UniversalEvaluator imported successfully")
    BASIC_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import UniversalEvaluator: {e}")
    BASIC_AVAILABLE = False

def test_enhanced_evaluator_basic():
    """Test basic enhanced evaluator functionality"""
    print("\n" + "="*50)
    print("TEST 1: Basic Enhanced Evaluator Functionality")
    print("="*50)
    
    if not ENHANCED_AVAILABLE:
        print("❌ Enhanced evaluator not available, skipping test")
        return False
    
    evaluator = EnhancedUniversalEvaluator()
    
    # Test response for Japanese haiku pattern
    test_response = """Softly to the ground
    
This completes the traditional Japanese haiku following the 5-7-5 syllable pattern. 
The phrase "softly to the ground" has exactly 5 syllables and connects naturally 
to the cherry blossom and spring breeze imagery, creating a peaceful closure that 
reflects the gentle nature of falling cherry blossoms in spring."""
    
    test_name = "Japanese Haiku Pattern Completion"
    
    try:
        # Test basic enhanced evaluation
        result = evaluator.evaluate_response(
            response_text=test_response,
            test_name=test_name,
            reasoning_type=ReasoningType.GENERAL
        )
        
        print(f"✅ Basic evaluation successful")
        print(f"   Overall Score: {result.metrics.overall_score:.2f}/100")
        print(f"   Reasoning Type: {result.reasoning_type.value}")
        
        # Check if enhanced metrics are available
        if hasattr(result, 'enhanced_metrics') and result.enhanced_metrics:
            print(f"   Enhanced Metrics Available: ✅")
        else:
            print(f"   Enhanced Metrics: Using base evaluation metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic enhanced evaluation failed: {e}")
        return False

def test_enhanced_evaluator_full():
    """Test full enhanced evaluator with test definition"""
    print("\n" + "="*50)
    print("TEST 2: Full Enhanced Evaluator with Test Definition")
    print("="*50)
    
    if not ENHANCED_AVAILABLE:
        print("❌ Enhanced evaluator not available, skipping test")
        return False
    
    evaluator = EnhancedUniversalEvaluator()
    
    # Create a test definition similar to reasoning domain tests
    test_definition = {
        "id": "test_haiku_01",
        "name": "Japanese Haiku Pattern Completion",
        "category": "basic_logic_patterns",
        "reasoning_type": "pattern_recognition",
        "description": "Complete haiku following traditional Japanese 5-7-5 syllable pattern",
        "prompt": "Complete this haiku...",
        "expected_patterns": ["softly", "ground", "spring", "gentle", "falling"],
        "scoring": {
            "exact_match": 1.0,
            "partial_match": 0.6,
            "semantic_similarity": 0.4
        },
        "metadata": {
            "concepts_tested": ["haiku_structure", "cultural_authenticity", "seasonal_imagery"],
            "domains_integrated": ["language", "creativity"],
            "reasoning_steps": 3
        }
    }
    
    test_response = """Softly to the ground

This completes the traditional Japanese haiku following the 5-7-5 syllable pattern. 
The phrase "softly to the ground" has exactly 5 syllables and connects naturally 
to the cherry blossom and spring breeze imagery, creating a peaceful closure that 
reflects the gentle nature of falling cherry blossoms in spring."""

    try:
        # Test full enhanced evaluation with test definition
        result = evaluator.evaluate_response_enhanced(
            response_text=test_response,
            test_definition=test_definition
        )
        
        print(f"✅ Full enhanced evaluation successful")
        print(f"   Overall Score: {result.metrics.overall_score:.2f}/100")
        
        if result.enhanced_metrics:
            print(f"   Exact Match Score: {result.enhanced_metrics.exact_match_score:.2f}")
            print(f"   Partial Match Score: {result.enhanced_metrics.partial_match_score:.2f}")
            print(f"   Semantic Similarity Score: {result.enhanced_metrics.semantic_similarity_score:.2f}")
            print(f"   Domain Synthesis Score: {result.enhanced_metrics.domain_synthesis_score:.2f}")
        
        if result.integration_analysis:
            print(f"   Multi-domain Test: {result.integration_analysis.get('is_multi_domain', False)}")
            print(f"   Domains Integrated: {len(result.integration_analysis.get('domains_integrated', []))}")
        
        if result.scoring_breakdown:
            print(f"   Scoring Breakdown Available: Yes")
        
        return True
        
    except Exception as e:
        print(f"❌ Full enhanced evaluation failed: {e}")
        return False

def test_backward_compatibility():
    """Test backward compatibility between basic and enhanced evaluators"""
    print("\n" + "="*50)
    print("TEST 3: Backward Compatibility Comparison")
    print("="*50)
    
    if not (BASIC_AVAILABLE and ENHANCED_AVAILABLE):
        print("❌ Both evaluators not available, skipping test")
        return False
    
    test_response = """The pattern shows progression from physical creation to spiritual guidance. 
Following Islamic themes, the completion would be:
'Who granted him knowledge, then blessed him with wisdom'
This maintains the parallel structure while progressing from basic understanding 
to deeper spiritual insight."""
    
    test_name = "Arabic Pattern Recognition Test"
    
    try:
        # Test with basic evaluator
        basic_result = evaluate_reasoning(
            response_text=test_response,
            test_name=test_name,
            reasoning_type=ReasoningType.GENERAL
        )
        
        # Test with enhanced evaluator (backward compatible)
        enhanced_evaluator = EnhancedUniversalEvaluator()
        enhanced_result = enhanced_evaluator.evaluate_response(
            response_text=test_response,
            test_name=test_name,
            reasoning_type=ReasoningType.GENERAL
        )
        
        print(f"✅ Backward compatibility test successful")
        print(f"   Basic Evaluator Score: {basic_result.metrics.overall_score:.2f}/100")
        print(f"   Enhanced Evaluator Score: {enhanced_result.metrics.overall_score:.2f}/100")
        print(f"   Score Difference: {abs(basic_result.metrics.overall_score - enhanced_result.metrics.overall_score):.2f}")
        
        # Check that interfaces are compatible
        assert hasattr(basic_result, 'metrics')
        assert hasattr(enhanced_result, 'metrics')
        assert hasattr(basic_result, 'reasoning_type')
        assert hasattr(enhanced_result, 'reasoning_type')
        
        print(f"   Interface Compatibility: ✅ Confirmed")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False

def test_reasoning_domain_integration():
    """Test integration with actual reasoning domain test"""
    print("\n" + "="*50)
    print("TEST 4: Integration with Reasoning Domain Test")
    print("="*50)
    
    if not ENHANCED_AVAILABLE:
        print("❌ Enhanced evaluator not available, skipping test")
        return False
    
    # Load an actual reasoning test
    try:
        reasoning_test_file = Path("domains/reasoning/base_models/easy.json")
        if not reasoning_test_file.exists():
            print(f"❌ Reasoning test file not found: {reasoning_test_file}")
            return False
        
        with open(reasoning_test_file, 'r') as f:
            test_data = json.load(f)
        
        # Get the first test
        first_test = test_data['tests'][0]
        print(f"   Using test: {first_test['name']}")
        
        # Mock response for the haiku test
        test_response = """Softly to the ground

This completes the traditional Japanese haiku following the 5-7-5 syllable pattern. 
The phrase "softly to the ground" has exactly 5 syllables and connects naturally 
to the cherry blossom and spring breeze imagery, creating a peaceful closure that 
reflects the gentle nature of falling cherry blossoms in spring. This follows 
traditional haiku principles of nature imagery, seasonal reference, and creating 
a moment of contemplation or beauty."""
        
        evaluator = EnhancedUniversalEvaluator()
        
        # Test with basic enhanced evaluation
        basic_enhanced = evaluator.evaluate_response(
            response_text=test_response,
            test_name=first_test.get('name', first_test.get('id')),
            reasoning_type=ReasoningType.GENERAL,
            test_category=first_test.get('category')
        )
        
        print(f"✅ Basic enhanced evaluation successful")
        print(f"   Score: {basic_enhanced.metrics.overall_score:.2f}/100")
        
        # Test with full enhanced evaluation (if the test has the right structure)
        if 'expected_patterns' in first_test or 'metadata' in first_test:
            try:
                full_enhanced = evaluator.evaluate_response_enhanced(
                    response_text=test_response,
                    test_definition=first_test
                )
                
                print(f"✅ Full enhanced evaluation successful")
                print(f"   Enhanced Score: {full_enhanced.metrics.overall_score:.2f}/100")
                
                if full_enhanced.enhanced_metrics:
                    print(f"   Multi-tier scoring active: ✅")
                else:
                    print(f"   Multi-tier scoring: ❌ (no enhanced metrics)")
                    
            except Exception as e:
                print(f"⚠️  Full enhanced evaluation failed (expected): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Reasoning domain integration test failed: {e}")
        return False

def main():
    """Run all enhanced evaluation tests"""
    print("Enhanced Universal Evaluator Integration Test Suite")
    print("="*60)
    
    results = []
    
    # Run all tests
    results.append(("Basic Enhanced Evaluator", test_enhanced_evaluator_basic()))
    results.append(("Full Enhanced Evaluator", test_enhanced_evaluator_full()))
    results.append(("Backward Compatibility", test_backward_compatibility()))
    results.append(("Reasoning Domain Integration", test_reasoning_domain_integration()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTests Passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 All tests passed! Enhanced evaluation integration successful.")
        return 0
    else:
        print("⚠️  Some tests failed. Review the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())