#!/usr/bin/env python3
"""
Reference Test Cases for Calibration Validation

Curated test cases with established target score ranges for validating
Enhanced Universal Evaluator calibration across domains.

Based on @docker/Dockerfile.llama-gpu model configuration and empirical
calibration testing from Phase 1 development.

Author: Claude Code
Version: 1.0.0 - Sequential Architecture
"""

from typing import List, Dict, Any

# Reference test cases with empirically validated target ranges
REFERENCE_TEST_CASES: List[Dict[str, Any]] = [
    {
        "name": "Haiku Completion - High Cultural Sophistication",
        "description": "Traditional Japanese haiku completion with cultural authenticity",
        "domain_path": "domains/reasoning/base_models/easy.json",
        "test_id": "basic_01",
        "expected_range": (75, 85),
        "target_mean": 80.0,
        "calibration_tolerance": 5.0,
        "rationale": "Perfect technical compliance (5 syllables, thematic connection), sophisticated poetic technique (personification), cultural authenticity",
        "test_category": "cultural_reasoning",
        "difficulty": "easy",
        "concepts_tested": ["haiku_structure", "cultural_authenticity", "seasonal_imagery"],
        "domains_integrated": ["language", "creativity"],
        "empirical_baseline": {
            "original_issue_score": 19.0,
            "phase_1_improvement": 66.8,
            "controlled_test_score": 86.6,
            "calibration_notes": "Major improvement achieved, still requires fine-tuning for consistent 75-85 range"
        }
    },
    {
        "name": "Multi-Step Reasoning - Moderate Complexity", 
        "description": "Logical reasoning with multi-step decomposition",
        "domain_path": "domains/reasoning/base_models/medium.json",
        "test_id": "complex_test_01",
        "expected_range": (60, 75),
        "target_mean": 67.5,
        "calibration_tolerance": 7.5,
        "rationale": "Requires systematic analysis, logical progression, evidence synthesis",
        "test_category": "logical_reasoning",
        "difficulty": "medium", 
        "concepts_tested": ["multi_step_analysis", "logical_progression", "evidence_synthesis"],
        "domains_integrated": ["reasoning"],
        "empirical_baseline": {
            "baseline_expectation": "Standard reasoning evaluation should handle well",
            "calibration_notes": "Baseline test for reasoning evaluator functionality"
        }
    },
    {
        "name": "Cultural Context - Basic Understanding",
        "description": "Basic cultural context recognition and appropriate response",
        "domain_path": "domains/reasoning/base_models/easy.json", 
        "test_id": "basic_02",
        "expected_range": (50, 65),
        "target_mean": 57.5,
        "calibration_tolerance": 7.5,
        "rationale": "Cultural awareness without deep sophistication requirement",
        "test_category": "cultural_awareness",
        "difficulty": "easy",
        "concepts_tested": ["cultural_recognition", "appropriate_response"],
        "domains_integrated": ["reasoning", "social"],
        "empirical_baseline": {
            "baseline_expectation": "Should demonstrate cultural pattern recognition",
            "calibration_notes": "Test for basic cultural evaluation capabilities"
        }
    },
    {
        "name": "Technical Accuracy - Low Complexity",
        "description": "Straightforward technical correctness verification",
        "domain_path": "domains/reasoning/base_models/easy.json",
        "test_id": "basic_03", 
        "expected_range": (70, 85),
        "target_mean": 77.5,
        "calibration_tolerance": 7.5,
        "rationale": "Clear technical correctness should score highly with enhanced evaluator",
        "test_category": "technical_verification",
        "difficulty": "easy",
        "concepts_tested": ["technical_correctness", "factual_accuracy"],
        "domains_integrated": ["reasoning"],
        "empirical_baseline": {
            "baseline_expectation": "Technical accuracy should be reliably detected",
            "calibration_notes": "Validation of core evaluation functionality"
        }
    },
    {
        "name": "Off-Topic Response - Quality Control",
        "description": "Completely irrelevant response to reasoning prompt",
        "domain_path": "domains/reasoning/base_models/easy.json",
        "test_id": "basic_04",
        "expected_range": (5, 20),
        "target_mean": 12.5,
        "calibration_tolerance": 7.5,
        "rationale": "Off-topic responses should score very low",
        "test_category": "failure_detection", 
        "difficulty": "easy",
        "concepts_tested": ["relevance_detection", "failure_mode_handling"],
        "domains_integrated": ["reasoning"],
        "empirical_baseline": {
            "baseline_expectation": "System should detect and penalize irrelevant responses",
            "calibration_notes": "Critical for evaluation system reliability"
        }
    }
]

# Domain expansion test cases (for future phases)
FUTURE_DOMAIN_TEST_CASES: List[Dict[str, Any]] = [
    {
        "name": "Creative Expression - Moderate Originality",
        "description": "Creative writing with moderate originality and technique",
        "domain_path": "domains/creativity/base_models/medium.json",
        "test_id": "creative_medium_01", 
        "expected_range": (55, 70),
        "target_mean": 62.5,
        "calibration_tolerance": 7.5,
        "rationale": "Creative content with reasonable technique and originality",
        "test_category": "creative_evaluation",
        "difficulty": "medium",
        "concepts_tested": ["creative_originality", "artistic_technique", "expression_quality"],
        "domains_integrated": ["creativity", "language"],
        "status": "future_implementation"
    },
    {
        "name": "Social Context - Communication Appropriateness", 
        "description": "Appropriate social communication in given context",
        "domain_path": "domains/social/base_models/easy.json",
        "test_id": "social_easy_01",
        "expected_range": (45, 60),
        "target_mean": 52.5,
        "calibration_tolerance": 7.5,
        "rationale": "Social appropriateness and communication effectiveness",
        "test_category": "social_evaluation",
        "difficulty": "easy",
        "concepts_tested": ["social_appropriateness", "communication_effectiveness"],
        "domains_integrated": ["social", "language"],
        "status": "future_implementation"
    }
]

def get_active_test_cases() -> List[Dict[str, Any]]:
    """
    Get currently active test cases for calibration validation
    
    Returns:
        List of active test cases
    """
    return REFERENCE_TEST_CASES

def get_test_case_by_name(name: str) -> Dict[str, Any]:
    """
    Get specific test case by name
    
    Args:
        name: Test case name
        
    Returns:
        Test case dictionary or empty dict if not found
    """
    for test_case in REFERENCE_TEST_CASES:
        if test_case['name'] == name:
            return test_case
    return {}

def get_test_cases_by_category(category: str) -> List[Dict[str, Any]]:
    """
    Get test cases by category
    
    Args:
        category: Test category (e.g., 'cultural_reasoning', 'logical_reasoning')
        
    Returns:
        List of matching test cases
    """
    return [tc for tc in REFERENCE_TEST_CASES if tc.get('test_category') == category]

def get_test_cases_by_difficulty(difficulty: str) -> List[Dict[str, Any]]:
    """
    Get test cases by difficulty level
    
    Args:
        difficulty: Difficulty level ('easy', 'medium', 'hard')
        
    Returns:
        List of matching test cases
    """
    return [tc for tc in REFERENCE_TEST_CASES if tc.get('difficulty') == difficulty]

def validate_test_case_structure(test_case: Dict[str, Any]) -> bool:
    """
    Validate test case has required fields
    
    Args:
        test_case: Test case dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        'name', 'description', 'domain_path', 'test_id',
        'expected_range', 'target_mean', 'calibration_tolerance'
    ]
    
    for field in required_fields:
        if field not in test_case:
            return False
    
    # Validate expected_range is tuple of two floats
    expected_range = test_case.get('expected_range')
    if not isinstance(expected_range, tuple) or len(expected_range) != 2:
        return False
    
    # Validate numeric fields
    numeric_fields = ['target_mean', 'calibration_tolerance']
    for field in numeric_fields:
        if not isinstance(test_case.get(field), (int, float)):
            return False
    
    return True

def get_calibration_summary() -> Dict[str, Any]:
    """
    Get summary of calibration test case configuration
    
    Returns:
        Summary dictionary
    """
    active_tests = get_active_test_cases()
    
    summary = {
        'total_test_cases': len(active_tests),
        'categories': list(set(tc.get('test_category', 'unknown') for tc in active_tests)),
        'difficulties': list(set(tc.get('difficulty', 'unknown') for tc in active_tests)),
        'target_range_summary': {
            'min_target': min(tc['expected_range'][0] for tc in active_tests),
            'max_target': max(tc['expected_range'][1] for tc in active_tests),
            'mean_target': sum(tc['target_mean'] for tc in active_tests) / len(active_tests)
        }
    }
    
    return summary

# Validate all test cases on import
if __name__ == "__main__":
    print("Validating reference test cases...")
    
    valid_count = 0
    for i, test_case in enumerate(REFERENCE_TEST_CASES):
        if validate_test_case_structure(test_case):
            valid_count += 1
        else:
            print(f"❌ Invalid test case {i+1}: {test_case.get('name', 'Unknown')}")
    
    print(f"✅ Validated {valid_count}/{len(REFERENCE_TEST_CASES)} test cases")
    
    summary = get_calibration_summary()
    print(f"📊 Summary: {summary}")