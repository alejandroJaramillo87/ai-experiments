#!/usr/bin/env python3
"""
Centralized Token Limit Configuration

This file defines token limits for different test categories and types.
Modify these values to quickly adjust token limits across all tests.
"""

# Token limits by test category - Phase 1B: Increased math/reasoning to prevent quality truncation
CATEGORY_TOKEN_LIMITS = {
    'basic_logic_patterns': 3000,       # Phase 1B: Increased from 2000 to 3000
    'cultural_reasoning': 4000,         # Phase 1B: Increased from 2500 to 4000  
    'elementary_math_science': 3500,    # Phase 1B: Increased from 2000 to 3500 (math_08 fix)
    'advanced_reasoning': 5000,         # Phase 1B: Increased from 2200 to 5000
    'linguistic_patterns': 3000,       # Phase 1B: Increased from 1800 to 3000
    'philosophical_reasoning': 4500,    # Phase 1B: Increased from 2100 to 4500
    'historical_context': 3500,        # Phase 1B: Increased from 1900 to 3500
    'creative_synthesis': 3500,        # Phase 1B: Increased from 1800 to 3500
}

# Token limits by reasoning type (fallback) - Phase 1B: Increased to match category limits
REASONING_TYPE_TOKEN_LIMITS = {
    'pattern_recognition': 3000,       # Phase 1B: Increased from 2000 to 3000
    'logical_progression': 3000,       # Phase 1B: Increased from 1800 to 3000
    'cultural_inference': 4000,        # Phase 1B: Increased from 2500 to 4000
    'mathematical_reasoning': 3500,    # Phase 1B: Increased from 2000 to 3500 (math_08 fix)
    'verification': 3000,              # Phase 1B: Increased from 1600 to 3000
    'comparative_analysis': 3500,      # Phase 1B: Increased from 1900 to 3500
    'synthesis': 3500,                 # Phase 1B: Increased from 2000 to 3500
}

# Global configuration
DEFAULT_TOKEN_LIMIT = 1800             # Was 1000
GLOBAL_TOKEN_MULTIPLIER = 1.0  # Easy way to scale all limits up/down

# Minimum and maximum bounds - Phase 1B: Increased max to accommodate new limits
MIN_TOKEN_LIMIT = 400
MAX_TOKEN_LIMIT = 5000  # Phase 1B: Increased from 2000 to 5000 for complex reasoning

def get_token_limit_for_test(test_data: dict) -> int:
    """
    Determine appropriate token limit for a test based on its category and type.
    
    Args:
        test_data: Test definition dictionary with 'category' and 'reasoning_type'
        
    Returns:
        int: Recommended token limit for this test
    """
    # Try category first (most specific)
    category = test_data.get('category', '')
    if category in CATEGORY_TOKEN_LIMITS:
        base_limit = CATEGORY_TOKEN_LIMITS[category]
    else:
        # Try reasoning type (fallback)
        reasoning_type = test_data.get('reasoning_type', '')
        base_limit = REASONING_TYPE_TOKEN_LIMITS.get(reasoning_type, DEFAULT_TOKEN_LIMIT)
    
    # Apply global multiplier
    adjusted_limit = int(base_limit * GLOBAL_TOKEN_MULTIPLIER)
    
    # Enforce bounds
    final_limit = max(MIN_TOKEN_LIMIT, min(MAX_TOKEN_LIMIT, adjusted_limit))
    
    return final_limit

def override_test_parameters(test_data: dict) -> dict:
    """
    Override test parameters with centrally configured values.
    
    Args:
        test_data: Original test definition
        
    Returns:
        dict: Test data with updated parameters
    """
    # Make a copy to avoid modifying original
    updated_test = test_data.copy()
    
    # Update token limit
    if 'parameters' in updated_test:
        updated_test['parameters'] = updated_test['parameters'].copy()
        updated_test['parameters']['max_tokens'] = get_token_limit_for_test(test_data)
    
    return updated_test

# Quick configuration presets
PRESETS = {
    'conservative': 0.8,    # Reduce all limits by 20%
    'standard': 1.0,        # Use configured limits
    'generous': 1.5,        # Increase all limits by 50%
    'debug': 2.0,           # Double all limits for debugging
}

def apply_preset(preset_name: str):
    """Apply a preset configuration"""
    global GLOBAL_TOKEN_MULTIPLIER
    if preset_name in PRESETS:
        GLOBAL_TOKEN_MULTIPLIER = PRESETS[preset_name]
        print(f"Applied preset '{preset_name}': multiplier = {GLOBAL_TOKEN_MULTIPLIER}")
    else:
        print(f"Unknown preset: {preset_name}")
        print(f"Available presets: {list(PRESETS.keys())}")

# Configuration summary
def print_configuration_summary():
    """Print current token limit configuration"""
    print("📊 Token Limit Configuration Summary")
    print("=" * 50)
    print(f"Global Multiplier: {GLOBAL_TOKEN_MULTIPLIER}")
    print(f"Default Limit: {int(DEFAULT_TOKEN_LIMIT * GLOBAL_TOKEN_MULTIPLIER)}")
    print()
    print("Category Limits:")
    for category, limit in CATEGORY_TOKEN_LIMITS.items():
        adjusted = int(limit * GLOBAL_TOKEN_MULTIPLIER)
        print(f"  {category}: {adjusted}")
    print()
    print("Reasoning Type Limits:")
    for reasoning_type, limit in REASONING_TYPE_TOKEN_LIMITS.items():
        adjusted = int(limit * GLOBAL_TOKEN_MULTIPLIER)
        print(f"  {reasoning_type}: {adjusted}")

if __name__ == "__main__":
    # Print configuration when run directly
    print_configuration_summary()