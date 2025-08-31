#!/usr/bin/env python3
"""
Core Domains Base-to-Instruct Conversion Script

Converts base model tests to instruct model format for the core domains:
- Language (230 tests) - linguistic diversity and multilingual competency
- Integration (200+ tests) - cross-domain synthesis and knowledge integration  
- Knowledge (200+ tests) - factual accuracy and reasoning over information
- Social (200+ tests) - cultural communication and social understanding

Author: Claude Code
Version: 2.0.0
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

def get_system_message(domain: str, category: str, cultural_context: str = "") -> str:
    """Generate appropriate system message based on domain and category"""
    
    domain_system_messages = {
        "language": {
            "historical_linguistics": "You are a historical linguistics expert with deep knowledge of proto-languages, comparative linguistics, and language family relationships across global linguistic diversity.",
            "writing_systems": "You are knowledgeable about diverse writing systems, scripts, and orthographic traditions from around the world, including ancient and modern writing practices.",
            "multilingual_contact": "You understand multilingual communication patterns, code-switching phenomena, and language contact situations across global communities.",
            "register_systems": "You are expert in sociolinguistic variation, formality systems, honorifics, and social language patterns across cultures.",
            "cultural_communication": "You approach Indigenous and traditional communication patterns with deep respect and cultural sensitivity, understanding diverse discourse traditions.",
            "dialectal_variation": "You understand regional language varieties, creoles, pidgins, and contact language formation across global linguistic contexts.",
            "pragmatic_meaning": "You are skilled in context-dependent communication, cultural pragmatics, and indirect speech patterns across diverse cultures.",
            "language_evolution": "You understand language change processes, typological diversity, and evolutionary linguistics across human language systems.",
            "advanced_code_switching": "You understand sophisticated multilingual competency, cultural identity negotiation through language, and complex multicultural communication patterns."
        },
        "integration": {
            "cross_domain": "You excel at synthesizing knowledge across multiple domains and finding meaningful connections between disparate fields of study.",
            "interdisciplinary": "You are skilled at integrating insights from different academic disciplines to provide comprehensive understanding.",
            "systems_thinking": "You understand complex systems and can analyze interactions between multiple components and levels of organization.",
            "synthesis": "You can combine information from multiple sources and perspectives to create coherent, integrated understanding.",
            "holistic_analysis": "You approach problems from multiple angles and can see the bigger picture while attending to important details."
        },
        "knowledge": {
            "factual_accuracy": "You provide accurate, well-sourced information and can distinguish between reliable and unreliable knowledge claims.",
            "reasoning_over_facts": "You can reason logically over factual information and draw valid conclusions from evidence.",
            "knowledge_integration": "You can connect facts from different domains to provide comprehensive understanding.",
            "information_synthesis": "You are skilled at synthesizing information from multiple sources while maintaining accuracy."
        },
        "social": {
            "cultural_understanding": "You approach cultural diversity with respect and sensitivity, understanding social patterns across different communities.",
            "social_dynamics": "You understand group dynamics, social hierarchies, and interpersonal communication patterns.",
            "cross_cultural": "You are skilled at cross-cultural communication and can bridge different social contexts respectfully.",
            "community_patterns": "You understand how communities function and the social structures that support human cooperation."
        }
    }
    
    category_messages = domain_system_messages.get(domain, {})
    base_message = category_messages.get(category, f"You are knowledgeable about {domain} and can provide accurate, culturally-sensitive information.")
    
    # Add cultural sensitivity for sensitive contexts
    cultural_sensitive_terms = [
        "indigenous", "native", "aboriginal", "tribal", "sacred", "ancestral",
        "traditional", "cultural", "spiritual", "heritage", "ceremonial"
    ]
    
    if any(term in cultural_context.lower() for term in cultural_sensitive_terms):
        base_message += " You approach Indigenous and traditional cultures with deep respect and cultural sensitivity."
    
    return base_message

def convert_prompt_to_messages(base_test: Dict[str, Any], domain: str) -> List[Dict[str, str]]:
    """Convert base model prompt to instruct model messages"""
    
    category = base_test.get("category", "")
    description = base_test.get("description", "")
    prompt = base_test.get("prompt", "")
    
    # Generate system message
    system_msg = get_system_message(domain, category, description + " " + prompt)
    
    # Convert prompt to user instruction
    user_content = convert_prompt_content(prompt, base_test, domain)
    
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]

def convert_prompt_content(prompt: str, base_test: Dict[str, Any], domain: str) -> str:
    """Convert base model prompt to instructional format"""
    
    # For most domains, enhance with clear task instruction
    category = base_test.get("category", "")
    
    # Extract task components
    lines = prompt.split('\n')
    main_content = []
    task_indicators = []
    
    for line in lines:
        line = line.strip()
        if line:
            # Look for explicit task indicators
            if any(word in line.lower() for word in ['provide', 'explain', 'analyze', 'describe', 'compare']):
                task_indicators.append(line)
            else:
                main_content.append(line)
    
    # Rebuild with clearer instruction format
    result_parts = []
    
    # Add main content
    if main_content:
        result_parts.append('\n'.join(main_content))
    
    # Add task instruction based on domain
    if domain == "language":
        if category == "historical_linguistics":
            result_parts.append("Provide your linguistic analysis with:")
            result_parts.append("• Reconstructed forms with proper notation")
            result_parts.append("• Sound change explanations")
            result_parts.append("• Comparative evidence analysis")
        elif category == "multilingual_contact" or category == "advanced_code_switching":
            result_parts.append("Provide your multilingual analysis with:")
            result_parts.append("• Code-switching patterns identification")
            result_parts.append("• Cultural context consideration")
            result_parts.append("• Sociolinguistic factors analysis")
        else:
            result_parts.append(f"Provide your {category.replace('_', ' ')} analysis with detailed explanation.")
    
    elif domain == "integration":
        result_parts.append("Provide your integrated analysis with:")
        result_parts.append("• Cross-domain connections")
        result_parts.append("• Synthesis of key concepts")
        result_parts.append("• Holistic perspective")
    
    elif domain == "knowledge":
        result_parts.append("Provide your knowledge-based response with:")
        result_parts.append("• Accurate factual information")
        result_parts.append("• Clear reasoning process")
        result_parts.append("• Well-supported conclusions")
    
    elif domain == "social":
        result_parts.append("Provide your social analysis with:")
        result_parts.append("• Cultural sensitivity and respect")
        result_parts.append("• Social dynamics understanding")
        result_parts.append("• Community context consideration")
    
    # Add existing task indicators
    if task_indicators:
        result_parts.extend(task_indicators)
    
    return '\n\n'.join(result_parts)

def get_evaluation_criteria(base_test: Dict[str, Any], domain: str) -> Dict[str, float]:
    """Generate evaluation criteria based on domain and test content"""
    
    category = base_test.get("category", "")
    
    domain_criteria = {
        "language": {
            "historical_linguistics": {
                "linguistic_accuracy": 0.35,
                "comparative_analysis": 0.25,
                "methodological_rigor": 0.25,
                "scholarly_presentation": 0.15
            },
            "multilingual_contact": {
                "multilingual_competency": 0.35,
                "cultural_sensitivity": 0.25,
                "sociolinguistic_accuracy": 0.25,
                "communication_effectiveness": 0.15
            },
            "cultural_communication": {
                "cultural_respect": 0.4,
                "communication_accuracy": 0.25,
                "traditional_understanding": 0.2,
                "sensitivity": 0.15
            },
            "default": {
                "linguistic_accuracy": 0.3,
                "cultural_sensitivity": 0.25,
                "analytical_depth": 0.25,
                "clarity": 0.2
            }
        },
        "integration": {
            "default": {
                "synthesis_quality": 0.35,
                "cross_domain_connections": 0.25,
                "analytical_depth": 0.25,
                "coherence": 0.15
            }
        },
        "knowledge": {
            "default": {
                "factual_accuracy": 0.4,
                "reasoning_quality": 0.25,
                "completeness": 0.2,
                "clarity": 0.15
            }
        },
        "social": {
            "default": {
                "cultural_sensitivity": 0.35,
                "social_understanding": 0.25,
                "accuracy": 0.25,
                "respectfulness": 0.15
            }
        }
    }
    
    domain_map = domain_criteria.get(domain, {"default": {"quality": 0.4, "accuracy": 0.3, "clarity": 0.3}})
    return domain_map.get(category, domain_map.get("default", {"quality": 0.4, "accuracy": 0.3, "clarity": 0.3}))

def adjust_parameters(base_params: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """Adjust parameters for instruct model format and domain"""
    
    adjusted = base_params.copy()
    
    # Increase max_tokens for instruct format
    if "max_tokens" in adjusted:
        adjusted["max_tokens"] = min(adjusted["max_tokens"] + 50, 400)
    
    # Adjust temperature based on domain
    if "temperature" in adjusted:
        if domain in ["knowledge", "language"]:
            # Lower temperature for accuracy-focused domains
            adjusted["temperature"] = max(adjusted["temperature"] - 0.1, 0.2)
        elif domain == "social":
            # Slightly lower for cultural sensitivity
            adjusted["temperature"] = max(adjusted["temperature"] - 0.05, 0.3)
        # Integration keeps similar temperature for synthesis creativity
    
    return adjusted

def convert_base_test_to_instruct(base_test: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """Convert a single base model test to instruct format"""
    
    # Determine type field based on domain
    type_field_mapping = {
        "language": "language_type",
        "integration": "integration_type", 
        "knowledge": "knowledge_type",
        "social": "social_type"
    }
    
    type_field = type_field_mapping.get(domain, "test_type")
    
    instruct_test = {
        "id": base_test["id"],
        "name": base_test["name"],
        "category": base_test["category"],
        type_field: base_test.get(type_field, "general"),
        "description": base_test["description"],
        "messages": convert_prompt_to_messages(base_test, domain),
        "parameters": adjust_parameters(base_test["parameters"], domain),
        "evaluation_criteria": get_evaluation_criteria(base_test, domain)
    }
    
    return instruct_test

def convert_domain_tests(domain: str, input_file: str, output_file: str) -> bool:
    """Convert entire domain test suite from base to instruct format"""
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found")
        return False
    
    # Convert suite info
    instruct_data = {
        "suite_info": {
            "name": base_data["suite_info"]["name"].replace("Base Model", "Instruct Model"),
            "version": "2.0.0",
            "total_tests": base_data["suite_info"]["total_tests"],
            "description": base_data["suite_info"]["description"],
            "difficulty": base_data["suite_info"]["difficulty"],
            "target_models": "instruct_models",
            "coverage_scope": base_data["suite_info"]["coverage_scope"],
            "cultural_authenticity": base_data["suite_info"].get("cultural_authenticity", "comprehensive_cross_cultural_representation"),
            "converted_from": f"base_models/{Path(input_file).name}",
            "conversion_date": "2025-01-31"
        },
        "tests": []
    }
    
    # Convert each test
    for base_test in base_data["tests"]:
        instruct_test = convert_base_test_to_instruct(base_test, domain)
        instruct_data["tests"].append(instruct_test)
    
    # Write converted data
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(instruct_data, f, indent=2, ensure_ascii=False)
    
    return True

def main():
    """Main conversion function for core domains"""
    
    domains = ["language", "integration", "knowledge", "social"]
    
    print("🌍 Core Domains Base-to-Instruct Conversion")
    print("=" * 50)
    
    results = {}
    total_converted = 0
    
    for domain in domains:
        print(f"\n📚 Converting {domain.upper()} domain...")
        
        input_file = f"domains/{domain}/base_models/easy.json"
        output_file = f"domains/{domain}/instruct_models/easy.json"
        
        if not Path(input_file).exists():
            print(f"  ⚠️  Input file {input_file} not found - skipping")
            results[domain] = {"status": "skipped", "reason": "input_file_not_found"}
            continue
        
        success = convert_domain_tests(domain, input_file, output_file)
        
        if success:
            # Count tests in converted file
            with open(output_file, 'r') as f:
                data = json.load(f)
                test_count = len(data["tests"])
                total_converted += test_count
            
            results[domain] = {"status": "success", "tests": test_count}
            print(f"  ✅ Converted {test_count} {domain} tests")
        else:
            results[domain] = {"status": "failed"}
            print(f"  ❌ Failed to convert {domain} tests")
    
    # Summary
    print(f"\n📊 CONVERSION SUMMARY")
    print("=" * 30)
    successful = sum(1 for r in results.values() if r["status"] == "success")
    total_domains = len(domains)
    
    print(f"✅ Successful conversions: {successful}/{total_domains}")
    print(f"📝 Total tests converted: {total_converted}")
    
    for domain, result in results.items():
        if result["status"] == "success":
            print(f"  • {domain}: {result['tests']} tests ✅")
        elif result["status"] == "skipped":
            print(f"  • {domain}: skipped ({result['reason']}) ⚠️")
        else:
            print(f"  • {domain}: failed ❌")
    
    if successful == total_domains:
        print("\n🎉 All core domain conversions completed successfully!")
        print("🎯 Ready for enhanced evaluator testing")
        return 0
    else:
        print("\n⚠️  Some conversions had issues - check individual results above")
        return 1

if __name__ == "__main__":
    exit(main())