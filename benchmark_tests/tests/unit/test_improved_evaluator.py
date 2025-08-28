#!/usr/bin/env python3
"""
Test script for the improved evaluation system
"""

from reasoning_evaluator import UniversalEvaluator

def main():
    # Test the improved evaluation system
    evaluator = UniversalEvaluator()

    # Test 1: Good response
    good_response = '''**Analysis Report**

This is a comprehensive analysis that addresses the key requirements:

## Key Findings
1. The technical solution demonstrates proper implementation
2. Evidence supports the conclusion
3. Multiple perspectives have been considered

## Conclusion
Based on the analysis, the recommended approach provides optimal results.
'''

    # Test 2: Broken repetitive response (like GPT-OSS-20B Test 35)
    broken_response = '''The user might want a report or analysis. The user might want a report or analysis. The user might want a report or analysis. The user might want a report or analysis. The user might want a report or analysis. The user might want a report or analysis. The user might want a report or analysis. The user might want a report or analysis. The user might want a report or analysis. The user might want a report or analysis.'''

    # Test 3: Technical response
    technical_response = '''#!/bin/bash
sudo systemctl restart apache2
systemctl status apache2
grep -i error /var/log/apache2/error.log
'''

    # Test 4: Empty response
    empty_response = ""

    print('Testing improved evaluation system...')
    print('=' * 50)

    # Evaluate responses
    try:
        result1 = evaluator.evaluate_response(good_response, 'Test Good Response')
        result2 = evaluator.evaluate_response(broken_response, 'Test Broken Response') 
        result3 = evaluator.evaluate_response(technical_response, 'Test Technical Response', test_category='linux')
        result4 = evaluator.evaluate_response(empty_response, 'Test Empty Response')

        print(f'Good Response Score: {result1.metrics.overall_score}')
        print(f'  - Organization: {result1.metrics.organization_quality}')
        print(f'  - Technical Accuracy: {result1.metrics.technical_accuracy}')
        print(f'  - Completeness: {result1.metrics.completeness}')
        print()
        
        print(f'Broken Response Score: {result2.metrics.overall_score}')
        print(f'  - Coherence Issues: {result2.detailed_analysis.get("coherence_failure", {}).get("failure_type", "None")}')
        print(f'  - Recommendations: {result2.recommendations[0] if result2.recommendations else "None"}')
        print()
        
        print(f'Technical Response Score: {result3.metrics.overall_score}')
        print(f'  - Organization: {result3.metrics.organization_quality}')
        print(f'  - Technical Accuracy: {result3.metrics.technical_accuracy}')
        print(f'  - Completeness: {result3.metrics.completeness}')
        print()
        
        print(f'Empty Response Score: {result4.metrics.overall_score}')
        print(f'  - Edge Case: {result4.detailed_analysis.get("edge_case_detection", {}).get("detected_case", "None")}')
        print()
        
        print('=' * 50)
        
        # Validation checks
        validations = {
            "Good response scored well (>70)": result1.metrics.overall_score > 70,
            "Broken response heavily penalized (<30)": result2.metrics.overall_score < 30,
            "Technical response appropriately scored": result3.metrics.overall_score > 40,
            "Empty response minimally scored (<10)": result4.metrics.overall_score < 10,
            "Score ranges are valid": all(0 <= r.metrics.overall_score <= 105 for r in [result1, result2, result3, result4])
        }
        
        print("Validation Results:")
        for check, passed in validations.items():
            status = "✅" if passed else "❌"
            print(f"{status} {check}")
        
        passed_count = sum(validations.values())
        total_count = len(validations)
        print(f"\nPassed: {passed_count}/{total_count} validations")
        
        if passed_count == total_count:
            print('🎉 All validations passed! Evaluation system successfully improved!')
        else:
            print('⚠️  Some validations failed. Review needed.')
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()