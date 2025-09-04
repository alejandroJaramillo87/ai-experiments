#!/usr/bin/env python3
"""
Demo script showing how easy it is to adjust token limits across all tests
"""

from domains.token_limits import (
    get_token_limit_for_test, 
    override_test_parameters, 
    print_configuration_summary,
    CATEGORY_TOKEN_LIMITS,
    GLOBAL_TOKEN_MULTIPLIER
)

def demo_before_after():
    """Show before/after token adjustments for problematic tests"""
    
    print("🎯 Before/After Token Adjustments")
    print("=" * 50)
    
    # The problematic tests we identified
    problematic_tests = [
        {
            'id': 'basic_02', 
            'name': 'African Proverb Pattern (repetition loops)',
            'category': 'basic_logic_patterns',
            'original_score': 55.1,
            'parameters': {'max_tokens': 600}
        },
        {
            'id': 'cultural_09',
            'name': 'Indian Family Communication (meta-reasoning)',
            'category': 'cultural_reasoning', 
            'original_score': 70.2,
            'parameters': {'max_tokens': 600}
        },
        {
            'id': 'math_06',
            'name': 'Andean Quipu Mathematics (good but cut off)',
            'category': 'elementary_math_science',
            'original_score': 53.2,
            'parameters': {'max_tokens': 600}
        }
    ]
    
    for test in problematic_tests:
        updated_test = override_test_parameters(test)
        old_tokens = test['parameters']['max_tokens']
        new_tokens = updated_test['parameters']['max_tokens']
        
        print(f"📝 {test['name']}")
        print(f"   Test ID: {test['id']}")
        print(f"   Original Score: {test['original_score']}")
        print(f"   Token Limit: {old_tokens} → {new_tokens} (+{new_tokens-old_tokens})")
        print(f"   Expected Impact: {'More space for complete responses' if new_tokens > old_tokens else 'No change'}")
        print()

def demo_easy_adjustments():
    """Show how easy it is to make global adjustments"""
    
    print("⚡ Easy Global Adjustments")
    print("=" * 50)
    
    print("🔧 Current Configuration:")
    print(f"   Cultural Reasoning: {CATEGORY_TOKEN_LIMITS['cultural_reasoning']} tokens")
    print(f"   Math Problems: {CATEGORY_TOKEN_LIMITS['elementary_math_science']} tokens")
    print(f"   Pattern Recognition: {CATEGORY_TOKEN_LIMITS['basic_logic_patterns']} tokens")
    print()
    
    print("📝 To adjust all limits by 50%:")
    print("   Edit: /benchmark_tests/domains/token_limits.py")
    print("   Change: GLOBAL_TOKEN_MULTIPLIER = 1.5")
    print("   Result: All tests get 50% more tokens")
    print()
    
    print("📝 To increase just cultural reasoning tokens:")
    print("   Edit: CATEGORY_TOKEN_LIMITS['cultural_reasoning'] = 2000")
    print("   Result: Only cultural tests get more tokens")
    print()
    
    print("📝 No need to edit 210+ individual test files! 🎉")

if __name__ == "__main__":
    print("🚀 Centralized Token Limit Configuration Demo")
    print("=" * 60)
    print()
    
    demo_before_after()
    demo_easy_adjustments()
    
    print("✅ Implementation Complete!")
    print()
    print("Next Steps:")
    print("1. Run tests with new token limits")
    print("2. Compare cutoff rates (target: <20% vs current 96.7%)")
    print("3. Observe improved response quality")