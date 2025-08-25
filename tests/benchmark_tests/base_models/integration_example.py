"""
Integration Example: Using ReasoningEvaluator with Existing Test Suite

This example demonstrates how to integrate the ReasoningEvaluator with the existing
test_base_model_reasoning_complicated.py test suite.

Author: Claude Code
Version: 1.0.0
"""

import sys
import os
import json
from typing import Dict, List, Tuple

# Add the current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from reasoning_evaluator import ReasoningEvaluator, ReasoningType, evaluate_reasoning
import requests
import time


def run_test_with_evaluation_example():
    """
    Example showing how to run a single test case with reasoning evaluation
    """
    print("=== Single Test Case with Reasoning Evaluation ===")
    
    # Example test case (simplified version of actual test)
    test_case = {
        "name": "Complex Test 16: Multi-Document Logical Chain Analysis (Chain-of-Thought)",
        "prompt": """Analyze the following legal cases and trace the logical connections between them...""",
        "params": {
            "max_tokens": 4000,
            "temperature": 0.3,
            "top_p": 0.95,
            "stream": False
        }
    }
    
    # Simulate model response (in real usage, this comes from your API call)
    sample_response = """
    To analyze these legal cases systematically, I need to examine each precedent and trace the logical connections.

    First, let me identify the core legal principles in Morrison v. TechCorp Industries. This case established the "Imminent Harm Doctrine" with four key requirements: (1) the threat must be imminent, (2) normal approval processes must be impossible, (3) the action must be proportionate, and (4) the executive must act in good faith.

    Second, examining Rodriguez v. DataFlow Systems reveals an evolution of this doctrine. The court introduced the "Technical Expertise Privilege," which expands executive authority when specialized knowledge is involved. This builds directly on Morrison's framework but adds the crucial element of domain expertise.

    Third, the Chen case further develops this legal framework by establishing the "Professional Duty Override" doctrine. This creates a hierarchy where professional ethics can supersede corporate interests when public safety is at stake.

    Therefore, these three cases create a progressive legal framework where executive emergency authority depends on: (1) immediacy of threat, (2) relevant expertise, and (3) professional obligations. Each case builds systematically on the previous precedent while expanding the scope of legitimate emergency authority.
    """
    
    # Initialize evaluator
    evaluator = ReasoningEvaluator()
    
    # Perform evaluation
    result = evaluator.evaluate_response(
        response_text=sample_response,
        test_name=test_case["name"],
        reasoning_type=ReasoningType.CHAIN_OF_THOUGHT
    )
    
    # Display results
    print(f"Test Name: {test_case['name']}")
    print(f"Reasoning Type: {result.reasoning_type.value}")
    print(f"Overall Score: {result.metrics.overall_score}/100")
    print(f"Confidence: {result.metrics.confidence_score}%")
    print()
    
    print("Detailed Metrics:")
    print(f"  Step Clarity: {result.metrics.step_clarity}/100")
    print(f"  Logical Consistency: {result.metrics.logical_consistency}/100")
    print(f"  Evidence Integration: {result.metrics.evidence_integration}/100")
    print(f"  Analysis Depth: {result.metrics.analysis_depth}/100")
    print(f"  Verification Effort: {result.metrics.verification_effort}/100")
    print()
    
    print("Recommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")
    print()
    
    print("Text Statistics:")
    stats = result.detailed_analysis["text_statistics"]
    print(f"  Word Count: {stats['word_count']}")
    print(f"  Sentences: {stats['sentence_count']}")
    print(f"  Vocabulary Diversity: {stats['vocabulary_diversity']:.2f}")
    print()


def run_batch_evaluation_example():
    """
    Example showing how to evaluate multiple test results in batch
    """
    print("=== Batch Evaluation Example ===")
    
    # Simulate multiple test responses
    sample_responses = [
        ("First, I need to analyze this step by step. The evidence shows that... Therefore, the conclusion is...", "Chain-of-Thought Test"),
        ("According to document A, the data indicates... Document B confirms this with... Based on both sources...", "Multi-Hop Inference Test"),
        ("Let me verify my reasoning. Checking the assumptions... Confirming the logic... The analysis holds.", "Verification Loop Test"),
        ("Calculating the probability: P(A) = 0.3, P(B|A) = 0.7, therefore P(A∩B) = 0.21...", "Mathematical Reasoning Test"),
        ("Working backwards from the conclusion, the evidence that would support this includes...", "Backward Reasoning Test")
    ]
    
    # Initialize evaluator
    evaluator = ReasoningEvaluator()
    
    # Perform batch evaluation
    results = evaluator.evaluate_batch(sample_responses)
    
    # Generate summary report
    summary = evaluator.generate_summary_report(results)
    
    print(f"Total Evaluations: {summary['total_evaluations']}")
    print(f"Average Score: {summary['average_score']:.1f}/100")
    print(f"Score Range: {summary['score_range'][0]:.1f} - {summary['score_range'][1]:.1f}")
    print()
    
    print("Metric Averages:")
    for metric, avg_score in summary['metric_averages'].items():
        print(f"  {metric.replace('_', ' ').title()}: {avg_score}/100")
    print()
    
    print("Reasoning Type Distribution:")
    for reasoning_type, count in summary['reasoning_type_distribution'].items():
        print(f"  {reasoning_type}: {count}")
    print()


def integrate_with_existing_test_suite():
    """
    Example showing how to modify the existing test suite to include evaluation
    """
    print("=== Integration with Existing Test Suite ===")
    
    # This is how you would modify the existing run_performance_test function
    def run_performance_test_with_evaluation(test_case):
        """Enhanced version of run_performance_test with reasoning evaluation"""
        
        # Original API call code (from your existing test suite)
        api_url = "http://127.0.0.1:8001/v1/completions"  # Your existing API
        headers = {"Content-Type": "application/json"}
        
        request_data = {
            "prompt": test_case["prompt"],
            **test_case["params"]
        }
        
        try:
            print(f"Running test: {test_case['name']}")
            
            # Make API call (your existing code)
            response = requests.post(api_url, headers=headers, json=request_data, timeout=120)
            
            if response.status_code == 200:
                response_data = response.json()
                generated_text = response_data.get("choices", [{}])[0].get("text", "")
                
                if generated_text:
                    # NEW: Add reasoning evaluation
                    evaluation_result = evaluate_reasoning(
                        response_text=generated_text,
                        test_name=test_case["name"]
                    )
                    
                    # Log both original and evaluation results
                    print(f"✓ Test completed successfully")
                    print(f"  Response length: {len(generated_text)} characters")
                    print(f"  Reasoning Score: {evaluation_result.metrics.overall_score}/100")
                    print(f"  Reasoning Type: {evaluation_result.reasoning_type.value}")
                    
                    # Save enhanced results
                    enhanced_result = {
                        "test_name": test_case["name"],
                        "response_text": generated_text,
                        "api_response": response_data,
                        "reasoning_evaluation": {
                            "overall_score": evaluation_result.metrics.overall_score,
                            "metrics": evaluation_result.metrics.__dict__,
                            "reasoning_type": evaluation_result.reasoning_type.value,
                            "recommendations": evaluation_result.recommendations
                        },
                        "timestamp": evaluation_result.timestamp
                    }
                    
                    # Save to file with enhanced data
                    filename = f"enhanced_test_results/{test_case['name'].replace(' ', '_').replace(':', '')}.json"
                    os.makedirs("enhanced_test_results", exist_ok=True)
                    
                    with open(filename, 'w') as f:
                        json.dump(enhanced_result, f, indent=2)
                    
                    return True
                else:
                    print(f"✗ Test failed: Empty response")
                    return False
            else:
                print(f"✗ Test failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Test failed: {str(e)}")
            return False
    
    # Example usage with a sample test case
    sample_test = {
        "name": "Integration Example Test",
        "prompt": "Analyze this step by step: What are the key factors in effective reasoning evaluation?",
        "params": {
            "max_tokens": 500,
            "temperature": 0.3,
            "top_p": 0.95,
            "stream": False
        }
    }
    
    print("This is how you would integrate evaluation into your existing test suite:")
    print("1. Import the ReasoningEvaluator")
    print("2. Add evaluation call after successful API response")
    print("3. Save enhanced results with reasoning metrics")
    print("4. Generate summary reports across all tests")
    print()
    print("The modified run_performance_test function would look like the code above.")


def demonstrate_configuration_options():
    """
    Show different configuration options for the evaluator
    """
    print("=== Configuration Options Example ===")
    
    # Using default configuration
    evaluator_default = ReasoningEvaluator()
    print("Default configuration loaded successfully")
    
    # Using custom configuration (if you create a custom config file)
    try:
        evaluator_custom = ReasoningEvaluator(config_path="custom_evaluation_config.json")
        print("Custom configuration loaded successfully")
    except:
        print("Custom configuration file not found - using defaults")
    
    # Show available reasoning types
    print("\nAvailable reasoning types:")
    for reasoning_type in ReasoningType:
        print(f"  - {reasoning_type.value}")
    
    print("\nConfiguration allows customization of:")
    print("  - Scoring weights for different metrics")
    print("  - Thresholds for quality categories")
    print("  - Reasoning-type-specific patterns")
    print("  - Domain-specific vocabulary")
    print("  - LLM integration settings")


if __name__ == "__main__":
    """
    Run all examples to demonstrate the reasoning evaluation system
    """
    print("ReasoningEvaluator Integration Examples")
    print("=" * 50)
    print()
    
    # Run all examples
    run_test_with_evaluation_example()
    print()
    run_batch_evaluation_example()
    print()
    integrate_with_existing_test_suite()
    print()
    demonstrate_configuration_options()
    print()
    
    print("Integration complete! The ReasoningEvaluator is ready to use with your test suite.")
    print()
    print("Next steps:")
    print("1. Modify your existing run_performance_test function to include evaluation")
    print("2. Run your 50-test suite with enhanced evaluation")
    print("3. Generate comprehensive reasoning quality reports")
    print("4. Use insights to improve model performance and test design")