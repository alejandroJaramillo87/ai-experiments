#!/usr/bin/env python3
"""
Test Case Extraction Script

Extracts test cases from the monolithic test_base_model_reasoning_complicated.py
and converts them to structured JSON format.

Author: Claude Code
"""

import json
import ast
import sys
import os
from typing import Dict, List, Any

def extract_test_cases(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract test cases from the monolithic Python file
    
    Args:
        file_path: Path to the monolithic test file
        
    Returns:
        List of test case dictionaries
    """
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse the Python file to find COMPLEX_TEST_CASES
    tree = ast.parse(content)
    
    test_cases = []
    
    # Find the COMPLEX_TEST_CASES variable assignment
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'COMPLEX_TEST_CASES':
                    # Extract the list of test cases
                    if isinstance(node.value, ast.List):
                        for i, test_node in enumerate(node.value.elts):
                            if isinstance(test_node, ast.Dict):
                                test_case = extract_dict_from_ast(test_node)
                                # Add metadata
                                test_case['id'] = f"complex_test_{i+1:02d}"
                                test_case['category'] = determine_category(i + 1)
                                test_case['reasoning_type'] = determine_reasoning_type(i + 1)
                                test_cases.append(test_case)
    
    return test_cases

def extract_dict_from_ast(dict_node: ast.Dict) -> Dict[str, Any]:
    """Extract dictionary from AST node"""
    result = {}
    
    for key_node, value_node in zip(dict_node.keys, dict_node.values):
        if isinstance(key_node, ast.Str):  # Python < 3.8
            key = key_node.s
        elif isinstance(key_node, ast.Constant):  # Python >= 3.8
            key = key_node.value
        else:
            continue
            
        if isinstance(value_node, ast.Str):  # Python < 3.8
            result[key] = value_node.s
        elif isinstance(value_node, ast.Constant):  # Python >= 3.8
            result[key] = value_node.value
        elif isinstance(value_node, ast.Dict):
            result[key] = extract_dict_from_ast(value_node)
        elif isinstance(value_node, ast.List):
            result[key] = extract_list_from_ast(value_node)
        else:
            # For complex cases, use literal_eval
            try:
                result[key] = ast.literal_eval(value_node)
            except (ValueError, TypeError):
                result[key] = str(value_node)
    
    return result

def extract_list_from_ast(list_node: ast.List) -> List[Any]:
    """Extract list from AST node"""
    result = []
    
    for item_node in list_node.elts:
        if isinstance(item_node, ast.Str):  # Python < 3.8
            result.append(item_node.s)
        elif isinstance(item_node, ast.Constant):  # Python >= 3.8
            result.append(item_node.value)
        else:
            try:
                result.append(ast.literal_eval(item_node))
            except (ValueError, TypeError):
                result.append(str(item_node))
    
    return result

def determine_category(test_num: int) -> str:
    """Determine category based on test number"""
    if 1 <= test_num <= 15:
        return "complex_synthesis"
    elif 16 <= test_num <= 25:
        return "chain_of_thought"
    elif 26 <= test_num <= 30:
        return "verification_loops"
    elif 31 <= test_num <= 35:
        return "mathematical_reasoning"
    elif 36 <= test_num <= 40:
        return "multi_hop_inference"
    elif 41 <= test_num <= 45:
        return "scaffolded_reasoning"
    elif 46 <= test_num <= 50:
        return "backward_reasoning"
    else:
        return "unknown"

def determine_reasoning_type(test_num: int) -> str:
    """Determine reasoning type for ReasoningEvaluator"""
    if 1 <= test_num <= 15:
        return "general"
    elif 16 <= test_num <= 25:
        return "chain_of_thought"
    elif 26 <= test_num <= 30:
        return "verification"
    elif 31 <= test_num <= 35:
        return "mathematical"
    elif 36 <= test_num <= 40:
        return "multi_hop"
    elif 41 <= test_num <= 45:
        return "scaffolded"
    elif 46 <= test_num <= 50:
        return "backward"
    else:
        return "general"

def enhance_test_case(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance test case with additional metadata"""
    enhanced = {
        "id": test_case["id"],
        "name": test_case["name"],
        "category": test_case["category"],
        "reasoning_type": test_case["reasoning_type"],
        "description": f"Test case for {test_case['category'].replace('_', ' ')} reasoning",
        "prompt": test_case["prompt"],
        "parameters": test_case["params"],
        "metadata": {
            "expected_length": determine_expected_length(test_case["params"].get("max_tokens", 1000)),
            "complexity": "high",
            "context_size": "large",
            "domain": determine_domain(test_case["name"]),
            "timeout_seconds": 600,
            "reasoning_patterns": get_reasoning_patterns(test_case["category"])
        },
        "evaluation_config": {
            "reasoning_type": test_case["reasoning_type"],
            "custom_weights": None,
            "expected_indicators": get_expected_indicators(test_case["category"])
        }
    }
    
    return enhanced

def determine_expected_length(max_tokens: int) -> str:
    """Determine expected response length"""
    if max_tokens < 2000:
        return "short"
    elif max_tokens < 5000:
        return "medium"
    else:
        return "long"

def determine_domain(test_name: str) -> str:
    """Determine domain from test name"""
    name_lower = test_name.lower()
    if "scientific" in name_lower or "research" in name_lower:
        return "scientific"
    elif "legal" in name_lower or "law" in name_lower:
        return "legal"
    elif "medical" in name_lower or "diagnosis" in name_lower:
        return "medical"
    elif "financial" in name_lower or "market" in name_lower:
        return "financial"
    elif "technical" in name_lower or "engineering" in name_lower:
        return "technical"
    else:
        return "general"

def get_reasoning_patterns(category: str) -> List[str]:
    """Get expected reasoning patterns for category"""
    patterns = {
        "complex_synthesis": ["synthesis", "comparison", "integration", "analysis"],
        "chain_of_thought": ["step_by_step", "sequential", "logical_progression"],
        "verification_loops": ["self_check", "verification", "error_detection"],
        "mathematical_reasoning": ["calculation", "logical_deduction", "proof"],
        "multi_hop_inference": ["cross_reference", "connection", "evidence_linking"],
        "scaffolded_reasoning": ["systematic", "structured", "methodical"],
        "backward_reasoning": ["reverse_engineering", "causal_tracing", "conclusion_to_evidence"]
    }
    return patterns.get(category, ["general_reasoning"])

def get_expected_indicators(category: str) -> List[str]:
    """Get expected reasoning indicators for category"""
    indicators = {
        "complex_synthesis": ["based on", "according to", "in contrast", "synthesis"],
        "chain_of_thought": ["first", "second", "then", "therefore", "step"],
        "verification_loops": ["verify", "check", "confirm", "review"],
        "mathematical_reasoning": ["calculate", "probability", "equation", "result"],
        "multi_hop_inference": ["document", "source", "evidence", "connection"],
        "scaffolded_reasoning": ["systematic", "framework", "structure", "analysis"],
        "backward_reasoning": ["reverse", "backward", "conclusion", "working backwards"]
    }
    return indicators.get(category, [])

def main():
    """Main extraction function"""
    # Paths
    monolithic_file = "/home/alejandro/workspace/ai-workstation/tests/benchmark_tests/base_models/test_base_model_reasoning_complicated.py"
    output_file = "/home/alejandro/workspace/ai-workstation/tests/benchmark_tests/base_models/test_definitions/reasoning_tests_complete.json"
    
    print("Starting test case extraction...")
    
    # Extract test cases
    try:
        test_cases = extract_test_cases(monolithic_file)
        print(f"Extracted {len(test_cases)} test cases")
        
        # Enhance test cases with metadata
        enhanced_test_cases = [enhance_test_case(test_case) for test_case in test_cases]
        
        # Create the final structure
        complete_suite = {
            "suite_info": {
                "name": "Advanced Long-Context Reasoning Test Suite",
                "version": "1.0.0",
                "total_tests": len(enhanced_test_cases),
                "extracted_from": "test_base_model_reasoning_complicated.py",
                "extraction_date": "2024-01-01"  # Will be updated when run
            },
            "tests": enhanced_test_cases
        }
        
        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(complete_suite, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Successfully saved {len(enhanced_test_cases)} test cases to {output_file}")
        
        # Print summary by category
        categories = {}
        for test in enhanced_test_cases:
            cat = test["category"]
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\nTest distribution by category:")
        for category, count in categories.items():
            print(f"  {category}: {count} tests")
            
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()