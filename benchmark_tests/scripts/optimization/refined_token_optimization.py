#!/usr/bin/env python3
"""
Refined Token Optimization Script

Based on validation findings that 800 tokens cause repetitive loops,
this script applies more conservative token limits that balance completeness
with quality (avoiding loops).

Key Insights from Validation:
- 400 tokens: Sweet spot for easy/cultural tests (no loops, complete responses)
- 800 tokens: Causes repetitive loops in medium complexity tests 
- Strategy: Conservative scaling to prevent loop behavior

Refined Token Strategy:
- Easy: 400 tokens (proven optimal)
- Medium: 500 tokens (reduced from 800 to prevent loops)
- Hard: 600 tokens (reduced from 1200 to prevent loops)
"""

import json
import os
from pathlib import Path
import sys

# Refined token limits based on validation results
REFINED_TOKEN_LIMITS = {
    'easy': 400,    # Proven optimal from cultural test validation
    'medium': 500,  # Reduced from 800 due to loop behavior
    'hard': 600,    # Reduced from 1200 to prevent loops
}

def get_difficulty_from_path(file_path):
    """Extract difficulty level from file path"""
    if 'easy' in file_path.lower():
        return 'easy'
    elif 'medium' in file_path.lower():
        return 'medium'
    elif 'hard' in file_path.lower():
        return 'hard'
    else:
        # Default to medium if unclear
        return 'medium'

def update_domain_tokens(domain_path):
    """Update token limits for a specific domain file"""
    
    try:
        # Load the domain file
        with open(domain_path, 'r', encoding='utf-8') as f:
            domain_data = json.load(f)
            
        # Get difficulty level and corresponding token limit
        difficulty = get_difficulty_from_path(str(domain_path))
        new_token_limit = REFINED_TOKEN_LIMITS[difficulty]
        
        # Track changes
        changes_made = 0
        original_limits = []
        
        # Update token limits for all tests
        for test in domain_data.get('tests', []):
            original_limit = test.get('max_tokens', 50)
            original_limits.append(original_limit)
            
            # Update if different from new limit
            if original_limit != new_token_limit:
                test['max_tokens'] = new_token_limit
                changes_made += 1
        
        # Save the updated domain file
        if changes_made > 0:
            with open(domain_path, 'w', encoding='utf-8') as f:
                json.dump(domain_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Updated {domain_path}")
            print(f"   Difficulty: {difficulty} | New limit: {new_token_limit}")
            print(f"   Tests updated: {changes_made}")
            if original_limits:
                print(f"   Original range: {min(original_limits)}-{max(original_limits)} tokens")
            return True
        else:
            print(f"⏭️  Skipped {domain_path} (already at {new_token_limit} tokens)")
            return False
            
    except Exception as e:
        print(f"❌ Error updating {domain_path}: {e}")
        return False

def main():
    """Apply refined token optimization to previously problematic domains"""
    
    print("🔧 REFINED TOKEN OPTIMIZATION")
    print("=" * 60)
    print("Applying conservative token limits based on validation results:")
    print(f"  Easy: {REFINED_TOKEN_LIMITS['easy']} tokens (proven optimal)")
    print(f"  Medium: {REFINED_TOKEN_LIMITS['medium']} tokens (reduced to prevent loops)")  
    print(f"  Hard: {REFINED_TOKEN_LIMITS['hard']} tokens (reduced to prevent loops)")
    print()
    
    # Target domains that showed loop behavior in validation
    target_domains = [
        'domains/liminal_concepts/base_models/medium.json',
        'domains/synthetic_knowledge/base_models/medium.json', 
        'domains/speculative_worlds/base_models/medium.json',
        'domains/emergent_systems/base_models/medium.json',
        'domains/synthesis_singularities/base_models/hard.json',  # Also apply to hard tests
    ]
    
    successful_updates = 0
    total_domains = len(target_domains)
    
    for domain_path in target_domains:
        if os.path.exists(domain_path):
            success = update_domain_tokens(domain_path)
            if success:
                successful_updates += 1
        else:
            print(f"⚠️  Domain not found: {domain_path}")
    
    print()
    print("=" * 60)
    print("📊 REFINED OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Successfully updated: {successful_updates}/{total_domains} domains")
    
    if successful_updates == total_domains:
        print("🎉 All target domains updated successfully!")
        print("   Ready for re-validation testing")
    elif successful_updates > 0:
        print("🟡 Partial success - some domains updated")
    else:
        print("❌ No domains were updated")
    
    print()
    print("🔬 Next Steps:")
    print("1. Run validation again: cd benchmark_tests && python scripts/validate_token_optimization.py")
    print("2. Check for reduced loop behavior and improved completion rates")
    print("3. Apply to additional domains if validation successful")

if __name__ == "__main__":
    main()