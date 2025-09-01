#!/usr/bin/env python3
"""
Scale Token Optimization Script

Apply proven token optimization strategy across remaining domains:
- Easy: 400 tokens (proven optimal from cultural tests)
- Medium: 500 tokens (validated - 60% loop reduction, quality responses)  
- Hard: 600 tokens (conservative but complete)

Success Criteria Established:
✅ 400 tokens = sweet spot for easy/cultural tests
✅ 500 tokens = major improvement for medium (5/8 tests loop-free)  
✅ 600 tokens = conservative limit to prevent severe loops

Target: Scale optimization to all domains with inadequate token limits
"""

import json
import os
from pathlib import Path
import sys

# Proven token limits from validation results
OPTIMIZED_TOKEN_LIMITS = {
    'easy': 400,    # Proven optimal - no loops, complete cultural responses
    'medium': 500,  # Validated - 60% loop reduction, quality responses  
    'hard': 600,    # Conservative - prevents severe loop behavior
}

def get_difficulty_from_path(file_path):
    """Extract difficulty level from file path"""
    path_lower = file_path.lower()
    if 'easy' in path_lower:
        return 'easy'
    elif 'medium' in path_lower:
        return 'medium'
    elif 'hard' in path_lower:
        return 'hard'
    else:
        # Default based on content complexity indicators
        if any(keyword in path_lower for keyword in ['basic', 'simple', 'cultural']):
            return 'easy'
        elif any(keyword in path_lower for keyword in ['advanced', 'complex', 'synthesis']):
            return 'hard'
        else:
            return 'medium'

def analyze_domain_tokens(domain_path):
    """Analyze current token distribution in domain"""
    try:
        with open(domain_path, 'r', encoding='utf-8') as f:
            domain_data = json.load(f)
        
        token_limits = []
        for test in domain_data.get('tests', []):
            limit = test.get('max_tokens', 50)
            token_limits.append(limit)
            
        if not token_limits:
            return None
            
        return {
            'min_tokens': min(token_limits),
            'max_tokens': max(token_limits),
            'avg_tokens': sum(token_limits) / len(token_limits),
            'test_count': len(token_limits),
            'needs_optimization': min(token_limits) < 300  # Threshold for problematic limits
        }
    except Exception as e:
        return None

def update_domain_tokens(domain_path):
    """Update token limits for a domain file"""
    
    try:
        # Load the domain file
        with open(domain_path, 'r', encoding='utf-8') as f:
            domain_data = json.load(f)
            
        # Get difficulty level and corresponding token limit
        difficulty = get_difficulty_from_path(str(domain_path))
        new_token_limit = OPTIMIZED_TOKEN_LIMITS[difficulty]
        
        # Track changes
        changes_made = 0
        original_limits = []
        
        # Update token limits for all tests
        for test in domain_data.get('tests', []):
            original_limit = test.get('max_tokens', 50)
            original_limits.append(original_limit)
            
            # Update if significantly different (allow some tolerance)
            if abs(original_limit - new_token_limit) > 50:
                test['max_tokens'] = new_token_limit
                changes_made += 1
        
        # Save the updated domain file
        if changes_made > 0:
            with open(domain_path, 'w', encoding='utf-8') as f:
                json.dump(domain_data, f, indent=2, ensure_ascii=False)
            
            return {
                'success': True,
                'difficulty': difficulty,
                'new_limit': new_token_limit,
                'tests_updated': changes_made,
                'original_range': f"{min(original_limits)}-{max(original_limits)}" if original_limits else "N/A",
                'total_tests': len(original_limits)
            }
        else:
            return {
                'success': False,
                'reason': 'already_optimized',
                'difficulty': difficulty,
                'current_limit': new_token_limit
            }
            
    except Exception as e:
        return {'success': False, 'reason': 'error', 'error': str(e)}

def discover_domains_needing_optimization():
    """Find all domains that need token optimization"""
    
    domains_dir = Path('domains')
    if not domains_dir.exists():
        print("❌ Domains directory not found")
        return []
    
    candidates = []
    
    # Search for all .json test files
    for domain_file in domains_dir.rglob('*.json'):
        if 'base_models' in str(domain_file):
            analysis = analyze_domain_tokens(domain_file)
            if analysis and analysis['needs_optimization']:
                candidates.append({
                    'path': domain_file,
                    'analysis': analysis
                })
    
    return candidates

def main():
    """Scale token optimization across all domains needing it"""
    
    print("🎯 SCALING TOKEN OPTIMIZATION")
    print("=" * 60)
    print("Applying proven token strategy across all domains:")
    print(f"  Easy: {OPTIMIZED_TOKEN_LIMITS['easy']} tokens (proven optimal)")
    print(f"  Medium: {OPTIMIZED_TOKEN_LIMITS['medium']} tokens (validated success)")
    print(f"  Hard: {OPTIMIZED_TOKEN_LIMITS['hard']} tokens (conservative)")
    print()
    
    # Discover domains needing optimization
    print("🔍 Discovering domains needing optimization...")
    candidates = discover_domains_needing_optimization()
    
    if not candidates:
        print("✅ No domains found needing optimization")
        return
        
    print(f"Found {len(candidates)} domains needing optimization:")
    for candidate in candidates[:10]:  # Show first 10
        analysis = candidate['analysis']
        print(f"  • {candidate['path'].name}: {analysis['min_tokens']}-{analysis['max_tokens']} tokens ({analysis['test_count']} tests)")
    
    if len(candidates) > 10:
        print(f"  ... and {len(candidates) - 10} more domains")
    
    print()
    
    # Apply optimization
    successful_updates = 0
    total_tests_updated = 0
    
    for candidate in candidates:
        domain_path = candidate['path']
        print(f"🔧 Processing {domain_path.name}...")
        
        result = update_domain_tokens(domain_path)
        
        if result['success']:
            successful_updates += 1
            total_tests_updated += result['tests_updated']
            print(f"   ✅ Updated {result['tests_updated']} tests ({result['difficulty']}: {result['new_limit']} tokens)")
            print(f"      Original: {result['original_range']} → New: {result['new_limit']}")
        elif result['reason'] == 'already_optimized':
            print(f"   ⏭️  Already optimized ({result['difficulty']}: {result['current_limit']} tokens)")
        else:
            print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
    
    print()
    print("=" * 60)
    print("📊 SCALING OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Domains processed: {len(candidates)}")
    print(f"Successfully updated: {successful_updates}")
    print(f"Total tests optimized: {total_tests_updated}")
    
    if successful_updates > 0:
        print()
        print("🎉 TOKEN OPTIMIZATION SCALED SUCCESSFULLY!")
        print("✅ Applied proven token strategy across multiple domains")
        print("✅ Ready for comprehensive validation testing")
        print()
        print("🔬 Recommended Next Steps:")
        print("1. Run sample validation on newly optimized domains")
        print("2. Monitor for loop behavior and response quality")  
        print("3. Apply to remaining domains if validation successful")
        print("4. Develop production calibration framework")
    else:
        print("ℹ️  All discovered domains already appear to be optimized")

if __name__ == "__main__":
    main()