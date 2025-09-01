#!/usr/bin/env python3
"""
Token Optimization Script for Benchmark Tests
Systematically fixes critically low token limits across all domains.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Token allocation by complexity level
TOKEN_LIMITS = {
    'easy': 400,    # Proven optimal from cultural test calibration
    'medium': 800,  # 2x for increased complexity
    'hard': 1200,   # 3x for maximum complexity
}

def get_complexity_from_path(file_path: str) -> str:
    """Extract complexity level from file path."""
    if 'easy.json' in file_path:
        return 'easy'
    elif 'medium.json' in file_path:
        return 'medium'
    elif 'hard.json' in file_path:
        return 'hard'
    else:
        return 'unknown'

def find_severely_limited_files() -> List[str]:
    """Find all JSON files with severely limited tokens (10-99)."""
    domains_dir = Path('/home/alejandro/workspace/ai-workstation/benchmark_tests/domains')
    limited_files = []
    
    for json_file in domains_dir.rglob('*.json'):
        try:
            with open(json_file, 'r') as f:
                content = f.read()
                # Look for max_tokens values between 10-99
                if re.search(r'"max_tokens":\s*[1-9][0-9](?:,|\s)', content):
                    limited_files.append(str(json_file))
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    return limited_files

def update_token_limits(file_path: str) -> Dict:
    """Update token limits in a JSON file."""
    complexity = get_complexity_from_path(file_path)
    target_tokens = TOKEN_LIMITS.get(complexity, 400)  # Default to 400
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        updates_made = 0
        
        # Update tests if they exist
        if 'tests' in data:
            for test in data['tests']:
                if 'max_tokens' in test and test['max_tokens'] < 100:
                    old_tokens = test['max_tokens']
                    test['max_tokens'] = target_tokens
                    updates_made += 1
                    print(f"  {test.get('id', 'unknown')}: {old_tokens} → {target_tokens}")
        
        # Also check for parameters.max_tokens structure
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and 'parameters' in value:
                    params = value['parameters']
                    if 'max_tokens' in params and params['max_tokens'] < 100:
                        old_tokens = params['max_tokens']
                        params['max_tokens'] = target_tokens
                        updates_made += 1
                        print(f"  {key}: {old_tokens} → {target_tokens}")
        
        # Save updated file
        if updates_made > 0:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        return {
            'file': file_path,
            'complexity': complexity,
            'target_tokens': target_tokens,
            'updates_made': updates_made
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {'file': file_path, 'error': str(e)}

def main():
    """Main token optimization process."""
    print("🚀 Starting Token Optimization Across All Domains")
    print("=" * 60)
    
    # Find all severely limited files
    limited_files = find_severely_limited_files()
    print(f"Found {len(limited_files)} files with severely limited tokens (<100)")
    print()
    
    # Process each file
    total_updates = 0
    results = []
    
    for file_path in limited_files:
        print(f"📁 Processing: {Path(file_path).name}")
        result = update_token_limits(file_path)
        results.append(result)
        
        if 'updates_made' in result:
            total_updates += result['updates_made']
        
        print()
    
    # Summary
    print("=" * 60)
    print("🎯 TOKEN OPTIMIZATION SUMMARY")
    print(f"Files processed: {len(limited_files)}")
    print(f"Total tests updated: {total_updates}")
    print()
    
    # Group by complexity
    by_complexity = {}
    for result in results:
        if 'complexity' in result:
            complexity = result['complexity']
            if complexity not in by_complexity:
                by_complexity[complexity] = []
            by_complexity[complexity].append(result)
    
    for complexity, files in by_complexity.items():
        target = TOKEN_LIMITS.get(complexity, 400)
        print(f"📊 {complexity.upper()} tests: {len(files)} files → {target} tokens")
    
    print()
    print("✅ Token optimization complete!")

if __name__ == "__main__":
    main()