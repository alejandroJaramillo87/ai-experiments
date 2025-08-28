#!/usr/bin/env python3
"""
Run comprehensive edge case robustness tests
"""

import sys
import os

# Add the benchmark_tests directory to Python path
benchmark_tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, benchmark_tests_dir)

from evaluator.reasoning_evaluator import UniversalEvaluator
import json

def main():
    print("Running comprehensive edge case robustness tests...")
    print("=" * 60)
    
    evaluator = UniversalEvaluator()
    
    # Run the built-in edge case tests
    test_results = evaluator.run_edge_case_tests()
    
    print("TEST RESULTS:")
    print("-" * 30)
    
    for test_name, result in test_results["test_results"].items():
        if "error" in result:
            print(f"❌ {test_name}: ERROR - {result['error']}")
        else:
            score = result.get("overall_score", "N/A")
            print(f"📊 {test_name}: {score}/100")
            
            # Show detected issues if any
            if result.get("detected_issues"):
                detected_case = result["detected_issues"].get("detected_case")
                if detected_case:
                    print(f"   🔍 Detected: {detected_case}")
            
            if result.get("coherence_issues") and result["coherence_issues"].get("failure_type"):
                print(f"   🧠 Coherence: {result['coherence_issues']['failure_type']}")
    
    print("\nVALIDATION RESULTS:")
    print("-" * 30)
    
    validations = test_results["validation"]
    for check, passed in validations.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
    
    passed = test_results["passed_validations"]
    total = test_results["total_validations"]
    success_rate = (passed / total) * 100
    
    print(f"\n📈 Success Rate: {passed}/{total} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 Edge case robustness tests PASSED!")
    else:
        print("⚠️  Edge case robustness needs improvement")
    
    # Save detailed results
    with open("edge_case_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: edge_case_test_results.json")

if __name__ == "__main__":
    main()