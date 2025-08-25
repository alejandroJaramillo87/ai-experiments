#!/usr/bin/env python3
"""
Instruct Test Extraction Script

Extracts test cases from the instruct model test files and converts them 
to structured JSON format compatible with the unified TestRunner.

Author: Claude Code
"""

import json
import ast
import sys
import os
from typing import Dict, List, Any

def extract_instruct_test_cases(file_path: str, suite_name: str, category_prefix: str) -> List[Dict[str, Any]]:
    """
    Extract test cases from instruct model test files
    
    Args:
        file_path: Path to the test file
        suite_name: Name of the test suite
        category_prefix: Prefix for categorizing tests
        
    Returns:
        List of test case dictionaries
    """
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse the Python file to find TEST_CASES
    tree = ast.parse(content)
    
    test_cases = []
    
    # Find the TEST_CASES variable assignment
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'TEST_CASES':
                    # Extract the list of test cases
                    if isinstance(node.value, ast.List):
                        for i, test_node in enumerate(node.value.elts):
                            if isinstance(test_node, ast.Dict):
                                test_case = extract_dict_from_ast(test_node)
                                # Add metadata for instruct tests
                                test_case['id'] = f"{category_prefix}_test_{i+1:02d}"
                                test_case['suite'] = suite_name
                                test_case['category'] = determine_instruct_category(test_case['name'], category_prefix)
                                test_case['reasoning_type'] = determine_instruct_reasoning_type(test_case['name'])
                                test_case['api_type'] = 'chat'  # All instruct tests use chat API
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

def determine_instruct_category(test_name: str, category_prefix: str) -> str:
    """Determine category based on test name and suite type"""
    name_lower = test_name.lower()
    
    if category_prefix == "reasoning":
        if "deductive" in name_lower or "logic" in name_lower:
            return "reasoning_logic"
        elif "temporal" in name_lower or "puzzle" in name_lower:
            return "reasoning_puzzle"
        elif "epistemic" in name_lower or "modal" in name_lower:
            return "reasoning_modal"
        elif "causal" in name_lower:
            return "reasoning_causal"
        elif "probabilistic" in name_lower or "bayesian" in name_lower:
            return "reasoning_probabilistic"
        else:
            return "reasoning_general"
    
    elif category_prefix == "linux":
        if "monitor" in name_lower or "resource" in name_lower:
            return "linux_monitoring"
        elif "container" in name_lower or "docker" in name_lower:
            return "linux_containers"
        elif "firewall" in name_lower or "iptables" in name_lower or "security" in name_lower:
            return "linux_security"
        elif "backup" in name_lower or "database" in name_lower:
            return "linux_database"
        elif "network" in name_lower:
            return "linux_networking"
        elif "automation" in name_lower or "cron" in name_lower:
            return "linux_automation"
        else:
            return "linux_general"
    
    else:
        return f"{category_prefix}_general"

def determine_instruct_reasoning_type(test_name: str) -> str:
    """Determine reasoning type for ReasoningEvaluator"""
    name_lower = test_name.lower()
    
    if any(word in name_lower for word in ["step", "chain", "sequence"]):
        return "chain_of_thought"
    elif "verification" in name_lower or "check" in name_lower:
        return "verification"
    elif any(word in name_lower for word in ["multi", "hop", "source"]):
        return "multi_hop"
    elif any(word in name_lower for word in ["math", "probability", "statistical"]):
        return "mathematical"
    else:
        return "general"

def enhance_instruct_test_case(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance instruct test case with additional metadata"""
    
    # Convert prompt to messages format for chat API
    messages = [{"role": "user", "content": test_case["prompt"]}]
    
    enhanced = {
        "id": test_case["id"],
        "name": test_case["name"],
        "suite": test_case["suite"],
        "category": test_case["category"],
        "reasoning_type": test_case["reasoning_type"],
        "api_type": test_case["api_type"],
        "description": f"Instruct test case for {test_case['category'].replace('_', ' ')}",
        "prompt": test_case["prompt"],  # Keep original for compatibility
        "messages": messages,  # Chat API format
        "parameters": test_case.get("params", {}),
        "metadata": {
            "expected_length": determine_expected_length(test_case.get("params", {}).get("max_tokens", 1000)),
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
    elif max_tokens < 10000:
        return "medium"
    else:
        return "long"

def determine_domain(test_name: str) -> str:
    """Determine domain from test name"""
    name_lower = test_name.lower()
    if any(word in name_lower for word in ["logic", "deductive", "epistemic"]):
        return "logical"
    elif any(word in name_lower for word in ["linux", "system", "server", "bash"]):
        return "technical"
    elif any(word in name_lower for word in ["docker", "container", "firewall"]):
        return "devops"
    elif any(word in name_lower for word in ["database", "backup", "sql"]):
        return "database"
    elif any(word in name_lower for word in ["network", "security", "monitor"]):
        return "security"
    else:
        return "general"

def get_reasoning_patterns(category: str) -> List[str]:
    """Get expected reasoning patterns for category"""
    if "reasoning" in category:
        return ["logical_deduction", "step_by_step", "analysis", "inference"]
    elif "linux" in category:
        return ["technical_analysis", "system_design", "problem_solving"]
    else:
        return ["general_reasoning", "analysis"]

def get_expected_indicators(category: str) -> List[str]:
    """Get expected reasoning indicators for category"""
    if "reasoning" in category:
        return ["therefore", "because", "given that", "it follows", "we can conclude"]
    elif "linux" in category:
        return ["first", "then", "configure", "setup", "script", "command"]
    else:
        return ["step", "analysis", "solution"]

def main():
    """Main extraction function"""
    
    # Get current directory (should be instruct-models)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create test_definitions directory in current instruct-models directory
    instruct_test_definitions_dir = os.path.join(current_dir, "test_definitions")
    os.makedirs(instruct_test_definitions_dir, exist_ok=True)
    
    # Define test suites to extract (using relative paths from current directory)
    test_suites = [
        {
            "file_path": os.path.join(current_dir, "test_reasoning_complicated.py"),
            "suite_name": "Instruct Reasoning Complex Tests", 
            "category_prefix": "reasoning",
            "output_file": "reasoning_tests.json"
        },
        {
            "file_path": os.path.join(current_dir, "test_linux_complicated.py"),
            "suite_name": "Instruct Linux Complex Tests", 
            "category_prefix": "linux",
            "output_file": "linux_tests.json"
        }
    ]
    
    print("Starting instruct test extraction...")
    
    all_tests = []
    
    for suite_config in test_suites:
        try:
            test_cases = extract_instruct_test_cases(
                suite_config["file_path"], 
                suite_config["suite_name"],
                suite_config["category_prefix"]
            )
            
            print(f"Extracted {len(test_cases)} tests from {suite_config['suite_name']}")
            
            # Enhance test cases with metadata
            enhanced_test_cases = [enhance_instruct_test_case(test_case) for test_case in test_cases]
            all_tests.extend(enhanced_test_cases)
            
            # Save individual suite file in instruct-models directory
            suite_data = {
                "suite_info": {
                    "name": suite_config["suite_name"],
                    "version": "1.0.0",
                    "total_tests": len(enhanced_test_cases),
                    "extracted_from": os.path.basename(suite_config["file_path"]),
                    "extraction_date": "2024-01-01",
                    "api_type": "chat"
                },
                "tests": enhanced_test_cases
            }
            
            output_path = os.path.join(instruct_test_definitions_dir, suite_config['output_file'])
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(suite_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved {len(enhanced_test_cases)} tests to {output_path}")
            
        except Exception as e:
            print(f"❌ Error extracting from {suite_config['file_path']}: {e}")
            continue
    
    # Create combined instruct test suite in instruct-models directory
    if all_tests:
        combined_suite = {
            "suite_info": {
                "name": "Combined Instruct Model Test Suite",
                "version": "1.0.0", 
                "total_tests": len(all_tests),
                "extracted_from": "instruct-models/test_*.py",
                "extraction_date": "2024-01-01",
                "api_type": "chat"
            },
            "tests": all_tests
        }
        
        combined_path = os.path.join(instruct_test_definitions_dir, "instruct_tests_complete.json")
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(combined_suite, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created combined instruct test suite with {len(all_tests)} tests: {combined_path}")
        
        # Create instruct test suite metadata
        metadata = {
            "suite_name": "Instruct Model Test Suite",
            "suite_id": "instruct_comprehensive_v1",
            "description": "Comprehensive test suite for instruct models with chat API format",
            "version": "1.0.0",
            "created_by": "Claude Code",
            "api_config": {
                "endpoint": "http://127.0.0.1:8005/v1/chat/completions",
                "model": "your-instruct-model-name",
                "headers": {
                    "Content-Type": "application/json"
                },
                "type": "chat"
            },
            "global_settings": {
                "default_timeout_seconds": 600,
                "default_retry_attempts": 3,
                "delay_between_tests": 1,
                "output_directory": "test_results",
                "save_raw_responses": True,
                "save_evaluation_results": True,
                "include_performance_metrics": True
            },
            "execution_modes": {
                "sequential": {
                    "enabled": True,
                    "description": "Run tests one after another with delays"
                },
                "concurrent": {
                    "enabled": True,
                    "description": "Run multiple tests simultaneously",
                    "default_workers": 3,
                    "max_workers": 10
                }
            },
            "evaluation_integration": {
                "enabled": True,
                "evaluator_module": "reasoning_evaluator",
                "auto_detect_reasoning_type": True,
                "generate_summary_reports": True
            }
        }
        
        metadata_path = os.path.join(instruct_test_definitions_dir, "test_suite_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created instruct test suite metadata: {metadata_path}")
        
        # Create categories file for instruct tests (flat structure like base models)
        categories = {}
        
        for test in all_tests:
            cat = test["category"]
            
            if cat not in categories:
                categories[cat] = {
                    "description": f"{cat.replace('_', ' ').title()} tests",
                    "test_ids": []
                }
            
            categories[cat]["test_ids"].append(test["id"])
        
        categories_data = {
            "categories": categories
        }
        
        categories_path = os.path.join(instruct_test_definitions_dir, "categories.json")
        with open(categories_path, 'w', encoding='utf-8') as f:
            json.dump(categories_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created instruct categories file: {categories_path}")
        
        # Print summary by category
        print(f"\nInstruct test suite created in: {instruct_test_definitions_dir}")
        print("\nTest distribution by category:")
        for category, info in categories.items():
            count = len(info["test_ids"])
            print(f"  {category}: {count} tests")
            
    print("\n✅ Instruct test extraction completed!")

if __name__ == "__main__":
    main()