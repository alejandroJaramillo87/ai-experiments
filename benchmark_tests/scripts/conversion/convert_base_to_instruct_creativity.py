#!/usr/bin/env python3
"""
Creativity Domain Base-to-Instruct Conversion Script

Converts creativity domain base model tests to instruct model format
while preserving cultural authenticity and enhancing instructional clarity.

"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

def get_system_message(category: str, cultural_context: str = "") -> str:
    """Generate appropriate system message based on category and cultural context"""
    
    system_messages = {
        "narrative_creation": "You are a respectful cultural storyteller with deep knowledge of traditional narrative forms and oral storytelling traditions from around the world.",
        "performance_writing": "You are skilled in theatrical traditions and understand various performance styles, character development, and stage craft from different cultural contexts.",
        "descriptive_artistry": "You are an artist and writer with appreciation for diverse aesthetic traditions and the ability to create vivid, culturally-informed descriptions.",
        "musical_linguistic": "You understand musical traditions, rhythm, and the relationship between sound and meaning across different cultural contexts.",
        "adaptive_creativity": "You are skilled at creative transformation and cultural fusion while maintaining respect for source traditions.",
        "problem_narrative": "You understand how traditional cultures use storytelling to convey wisdom and solve problems creatively.",
        "cultural_interpretation": "You are skilled at cross-cultural communication and can explain cultural concepts respectfully and accurately.",
        "collaborative_creation": "You understand collaborative creative traditions and call-and-response patterns from various cultures.",
        "authenticity_recognition": "You are knowledgeable about cultural authenticity, appropriation concerns, and respectful creative practices."
    }
    
    base_message = system_messages.get(category, "You are a creative and culturally-aware assistant.")
    
    # Add specific cultural sensitivity if needed
    cultural_sensitive_terms = [
        "indigenous", "native", "aboriginal", "tribal", "sacred", "ancestral",
        "inuit", "african", "polynesian", "andean", "quechua"
    ]
    
    if any(term in cultural_context.lower() for term in cultural_sensitive_terms):
        base_message += " You approach Indigenous and traditional cultures with deep respect and cultural sensitivity."
    
    return base_message

def convert_prompt_to_messages(base_test: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert base model prompt to instruct model messages"""
    
    category = base_test.get("category", "")
    description = base_test.get("description", "")
    prompt = base_test.get("prompt", "")
    
    # Generate system message
    system_msg = get_system_message(category, description + " " + prompt)
    
    # Convert prompt to user instruction
    user_content = convert_prompt_content(prompt, base_test)
    
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content}
    ]

def convert_prompt_content(prompt: str, base_test: Dict[str, Any]) -> str:
    """Convert base model prompt to instructional format"""
    
    # Extract task and guidelines from prompt
    parts = prompt.split('\n\n')
    
    if len(parts) >= 2:
        # Usually: context + guidelines + task
        context = parts[0]
        
        # Find guidelines section (usually contains bullet points)
        guidelines_section = ""
        task_section = ""
        
        for i, part in enumerate(parts):
            if '- ' in part or '• ' in part:  # Guidelines with bullet points
                guidelines_section = part
            elif any(word in part.lower() for word in ['begin', 'create', 'write', 'complete']):
                task_section = part
        
        # Reconstruct as instruction
        instruction_parts = []
        
        if context and not any(word in context.lower() for word in ['begin', 'create', 'write']):
            instruction_parts.append(context)
        
        if guidelines_section:
            # Clean up guidelines
            guidelines = guidelines_section.replace('- ', '• ').replace('\n- ', '\n• ')
            instruction_parts.append(f"Include these elements:\n{guidelines}")
        
        # Add clear task instruction
        category = base_test.get("category", "")
        creativity_type = base_test.get("creativity_type", "")
        
        if category == "narrative_creation":
            task = "Write your story opening (200-250 words):"
        elif category == "performance_writing":
            task = "Write your performance piece with character names, dialogue, and stage directions (200-300 words):"
        elif category == "descriptive_artistry":
            task = "Write your descriptive piece (200-250 words):"
        elif category == "musical_linguistic":
            task = "Create your musical piece with clear structure and rhythm (200-250 words):"
        else:
            task = f"Create your {creativity_type} response (200-250 words):"
        
        instruction_parts.append(task)
        
        return '\n\n'.join(instruction_parts)
    
    else:
        # Simple prompt - just add instruction format
        return f"{prompt}\n\nProvide your creative response (200-250 words):"

def get_evaluation_criteria(base_test: Dict[str, Any]) -> Dict[str, float]:
    """Generate evaluation criteria based on test content"""
    
    category = base_test.get("category", "")
    description = base_test.get("description", "")
    prompt = base_test.get("prompt", "")
    
    # Base criteria by category
    criteria_map = {
        "narrative_creation": {
            "cultural_authenticity": 0.3,
            "narrative_structure": 0.25,
            "creativity": 0.25,
            "engagement": 0.2
        },
        "performance_writing": {
            "character_development": 0.3,
            "theatrical_elements": 0.25,
            "dialogue_quality": 0.25,
            "performance_directions": 0.2
        },
        "descriptive_artistry": {
            "vivid_imagery": 0.3,
            "cultural_accuracy": 0.25,
            "aesthetic_understanding": 0.25,
            "sensory_richness": 0.2
        },
        "musical_linguistic": {
            "rhythmic_structure": 0.3,
            "musical_understanding": 0.25,
            "lyrical_quality": 0.25,
            "cultural_authenticity": 0.2
        },
        "adaptive_creativity": {
            "creative_transformation": 0.3,
            "cultural_respect": 0.25,
            "innovation": 0.25,
            "coherence": 0.2
        },
        "problem_narrative": {
            "problem_solving": 0.3,
            "narrative_integration": 0.25,
            "wisdom_transmission": 0.25,
            "cultural_context": 0.2
        },
        "cultural_interpretation": {
            "cultural_accuracy": 0.35,
            "clarity": 0.25,
            "respect": 0.25,
            "accessibility": 0.15
        },
        "collaborative_creation": {
            "collaborative_structure": 0.3,
            "cultural_authenticity": 0.25,
            "interactive_elements": 0.25,
            "community_themes": 0.2
        },
        "authenticity_recognition": {
            "cultural_sensitivity": 0.4,
            "accuracy": 0.25,
            "appropriateness": 0.2,
            "respectfulness": 0.15
        }
    }
    
    base_criteria = criteria_map.get(category, {
        "creativity": 0.3,
        "cultural_respect": 0.25,
        "quality": 0.25,
        "authenticity": 0.2
    })
    
    # Adjust for cultural sensitivity
    cultural_sensitive_terms = ["indigenous", "native", "aboriginal", "sacred"]
    if any(term in (description + prompt).lower() for term in cultural_sensitive_terms):
        # Increase cultural/respect weighting
        if "cultural_authenticity" in base_criteria:
            base_criteria["cultural_authenticity"] += 0.1
        if "cultural_respect" in base_criteria:
            base_criteria["cultural_respect"] += 0.1
    
    return base_criteria

def adjust_parameters(base_params: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust parameters for instruct model format"""
    
    adjusted = base_params.copy()
    
    # Increase max_tokens slightly for instruct format
    if "max_tokens" in adjusted:
        adjusted["max_tokens"] = min(adjusted["max_tokens"] + 50, 400)
    
    # Slightly lower temperature for more controlled instruction following
    if "temperature" in adjusted:
        adjusted["temperature"] = max(adjusted["temperature"] - 0.1, 0.3)
    
    return adjusted

def convert_base_test_to_instruct(base_test: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single base model test to instruct format"""
    
    instruct_test = {
        "id": base_test["id"],
        "name": base_test["name"],
        "category": base_test["category"],
        "creativity_type": base_test.get("creativity_type", "generative"),
        "description": base_test["description"],
        "messages": convert_prompt_to_messages(base_test),
        "parameters": adjust_parameters(base_test["parameters"]),
        "evaluation_criteria": get_evaluation_criteria(base_test)
    }
    
    return instruct_test

def convert_creativity_tests(input_file: str, output_file: str):
    """Convert entire creativity test suite from base to instruct format"""
    
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
            "version": "1.0.0",
            "total_tests": base_data["suite_info"]["total_tests"],
            "description": base_data["suite_info"]["description"],
            "difficulty": base_data["suite_info"]["difficulty"],
            "target_models": "instruct_models",
            "coverage_scope": base_data["suite_info"]["coverage_scope"],
            "cultural_authenticity": base_data["suite_info"]["cultural_authenticity"],
            "converted_from": f"base_models/{Path(input_file).name}",
            "conversion_date": "2025-01-31"
        },
        "tests": []
    }
    
    # Convert each test
    for base_test in base_data["tests"]:
        instruct_test = convert_base_test_to_instruct(base_test)
        instruct_data["tests"].append(instruct_test)
    
    # Write converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(instruct_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully converted {len(instruct_data['tests'])} tests")
    print(f"Output written to: {output_file}")
    
    return True

def main():
    """Main conversion function"""
    
    input_file = "domains/creativity/base_models/easy.json"
    output_file = "domains/creativity/instruct_models/easy.json"
    
    print("🎭 Creativity Domain Base-to-Instruct Conversion")
    print("=" * 50)
    
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} not found")
        return 1
    
    # Create output directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    success = convert_creativity_tests(input_file, output_file)
    
    if success:
        print("\n✅ Conversion completed successfully!")
        print(f"📁 Output: {output_file}")
        print(f"🎯 Ready for enhanced evaluator testing")
        return 0
    else:
        print("\n❌ Conversion failed!")
        return 1

if __name__ == "__main__":
    exit(main())