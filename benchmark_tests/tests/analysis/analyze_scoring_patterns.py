#!/usr/bin/env python3
"""
Analyze current scoring patterns to identify systematic issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evaluator.reasoning_evaluator import UniversalEvaluator
import json

def analyze_test_responses():
    """Analyze various quality levels of responses to identify scoring patterns"""
    
    evaluator = UniversalEvaluator()
    
    test_responses = {
        "excellent_academic": {
            "text": """**Comprehensive Analysis: Market Equilibrium Under Uncertainty**

## Executive Summary
This analysis examines market equilibrium dynamics under uncertainty using advanced econometric modeling. The findings demonstrate significant implications for policy formulation and strategic planning.

## Methodology
Our approach utilizes a multi-stage econometric framework incorporating:
- Stochastic differential equations for price dynamics
- Monte Carlo simulations for uncertainty quantification  
- Bayesian inference for parameter estimation
- Sensitivity analysis across multiple scenarios

## Key Findings
1. **Market Volatility**: Evidence indicates 23.7% increase in volatility during uncertainty periods
2. **Price Discovery**: Asymmetric information creates 15.3% inefficiency in price discovery mechanisms
3. **Risk Premium**: Uncertainty premium averages 4.2% above risk-free rate

## Statistical Analysis
The regression analysis (R² = 0.847, p < 0.001) confirms significant relationships:
- β₁ = 0.234 (volatility coefficient)
- β₂ = -0.156 (information asymmetry coefficient)
- Standard errors: 0.023 and 0.019 respectively

## Policy Implications
Based on empirical evidence, we recommend:
1. Enhanced market transparency regulations
2. Improved information disclosure requirements
3. Risk management framework implementation

## Conclusion
The analysis provides robust evidence for market intervention strategies. The statistical significance of our findings (p < 0.001) supports the theoretical framework and validates policy recommendations.""",
            "expected_score": 95
        },
        
        "good_structured": {
            "text": """**Analysis Report**

This is a comprehensive analysis that addresses the key requirements:

## Key Findings
1. The technical solution demonstrates proper implementation
2. Evidence supports the conclusion
3. Multiple perspectives have been considered

## Methodology
The approach involved systematic review of available evidence and comparison of alternative solutions. The analysis considered both quantitative and qualitative factors.

## Results
The evaluation shows positive outcomes across all metrics. The implementation successfully addresses the core requirements while maintaining system reliability.

## Conclusion
Based on the analysis, the recommended approach provides optimal results.""",
            "expected_score": 75
        },
        
        "basic_adequate": {
            "text": """This analysis looks at the main issues. There are several important points to consider.

First, the data shows some interesting patterns. The results indicate that the approach works well in most cases.

Second, there are some challenges that need to be addressed. These include implementation issues and resource constraints.

Overall, the analysis suggests that the proposed solution is viable. However, more research may be needed to fully validate the approach.""",
            "expected_score": 55
        },
        
        "poor_quality": {
            "text": """I think this is about analysis. Maybe we need to look at some stuff. There might be some things to consider.

The answer could be yes or no. It depends on various factors. Some people might disagree.

This is a complex topic. More information would be helpful.""",
            "expected_score": 25
        }
    }
    
    print("SCORING PATTERN ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    for test_name, test_data in test_responses.items():
        result = evaluator.evaluate_response(test_data["text"], f"Analysis Test: {test_name}")
        
        actual_score = result.metrics.overall_score
        expected_score = test_data["expected_score"]
        gap = actual_score - expected_score
        
        results[test_name] = {
            "actual_score": actual_score,
            "expected_score": expected_score,
            "gap": gap,
            "metrics": {
                "organization_quality": result.metrics.organization_quality,
                "technical_accuracy": result.metrics.technical_accuracy,
                "completeness": result.metrics.completeness,
                "thoroughness": result.metrics.thoroughness,
                "reliability": result.metrics.reliability,
                "scope_coverage": result.metrics.scope_coverage,
                "domain_appropriateness": result.metrics.domain_appropriateness
            }
        }
        
        print(f"\n📊 {test_name.upper()}")
        print(f"Expected: {expected_score} | Actual: {actual_score} | Gap: {gap:+.1f}")
        print(f"Metrics Breakdown:")
        for metric, value in results[test_name]["metrics"].items():
            print(f"  - {metric}: {value}")
    
    # Analysis
    print(f"\n🔍 PATTERN ANALYSIS")
    print("-" * 40)
    
    total_gap = sum(r["gap"] for r in results.values())
    avg_gap = total_gap / len(results)
    
    print(f"Average Gap: {avg_gap:+.1f} points")
    
    # Identify problematic metrics
    metric_gaps = {}
    for metric in ["organization_quality", "technical_accuracy", "completeness", "thoroughness", "reliability", "scope_coverage", "domain_appropriateness"]:
        gaps = []
        for test_name, test_data in test_responses.items():
            expected = test_data["expected_score"]
            actual_metric = results[test_name]["metrics"][metric]
            # Rough estimate of expected metric score (assume proportional to overall)
            expected_metric = expected * (actual_metric / results[test_name]["actual_score"]) if results[test_name]["actual_score"] > 0 else expected
            gap = actual_metric - expected_metric
            gaps.append(gap)
        
        metric_gaps[metric] = sum(gaps) / len(gaps)
    
    print(f"\nMetric-Specific Gaps:")
    for metric, gap in sorted(metric_gaps.items(), key=lambda x: x[1]):
        print(f"  - {metric}: {gap:+.1f}")
    
    # Identify systematic issues
    issues = []
    if avg_gap < -10:
        issues.append("Overall scoring too conservative")
    if metric_gaps.get("technical_accuracy", 0) < -5:
        issues.append("Technical accuracy undervalued")
    if metric_gaps.get("organization_quality", 0) > 5:
        issues.append("Organization quality might be overvalued relative to content")
    if all(gap < 0 for gap in [r["gap"] for r in results.values()]):
        issues.append("Systematic undervaluation across all quality levels")
    
    print(f"\n🚨 IDENTIFIED ISSUES:")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
    
    # Save results
    with open("scoring_pattern_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed analysis saved to: scoring_pattern_analysis.json")
    
    return results, issues

if __name__ == "__main__":
    analyze_test_responses()