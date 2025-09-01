#!/usr/bin/env python3
"""
Token Optimization Validation Script

Validates that our systematic token optimization (50→800-1200 tokens) doesn't 
introduce repetitive loops while maintaining complete responses.

Validates samples from newly optimized domains:
- liminal_concepts (medium): 50→800 tokens
- synthetic_knowledge (medium): 50→800 tokens  
- speculative_worlds (medium): 50→800 tokens
- emergent_systems (medium): 50→800 tokens

Success Criteria:
✅ completion_tokens > 200 (complete responses)
✅ finish_reason = "stop" (not truncated) 
✅ repetitive_loops = 0 (no loop behavior)
✅ coherence_failure = false (maintains quality)
"""

import json
import os
import sys
import requests
import time
import re
from pathlib import Path

# Add the benchmark_tests directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

class TokenOptimizationValidator:
    def __init__(self, endpoint="http://127.0.0.1:8004/v1/completions"):
        self.endpoint = endpoint
        self.results = []
        
    def validate_domain_test(self, domain_path, test_id):
        """Validate a specific test from a domain"""
        print(f"\n🔍 Validating {domain_path} → {test_id}")
        
        # Load test definition
        try:
            with open(domain_path, 'r', encoding='utf-8') as f:
                domain_data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to load {domain_path}: {e}")
            return None
            
        # Find the specific test
        test_case = None
        for test in domain_data.get('tests', []):
            if test['id'] == test_id:
                test_case = test
                break
                
        if not test_case:
            print(f"❌ Test {test_id} not found in {domain_path}")
            return None
            
        print(f"   📝 Prompt: {test_case['prompt'][:100]}...")
        print(f"   🎯 Max tokens: {test_case.get('max_tokens', 'not set')}")
        
        # Make API request
        payload = {
            "model": "/app/models/hf/DeepSeek-R1-0528-Qwen3-8b",
            "prompt": test_case['prompt'],
            "max_tokens": test_case.get('max_tokens', 800),
            "temperature": test_case.get('temperature', 0.0),
            "top_p": test_case.get('top_p', 1.0),
            "stream": False
        }
        
        try:
            response = requests.post(self.endpoint, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Extract response details
            choice = result['choices'][0]
            response_text = choice['text']
            finish_reason = choice['finish_reason']
            completion_tokens = result['usage']['completion_tokens']
            
            # Analyze response quality
            analysis = self.analyze_response(response_text, completion_tokens, finish_reason)
            
            # Store result
            validation_result = {
                'domain': domain_path.split('/')[-3],  # Extract domain name
                'test_id': test_id,
                'max_tokens': test_case.get('max_tokens', 800),
                'completion_tokens': completion_tokens,
                'finish_reason': finish_reason,
                'response_length': len(response_text),
                'analysis': analysis,
                'prompt': test_case['prompt'][:100] + "...",
                'response_preview': response_text[:200] + "..." if len(response_text) > 200 else response_text
            }
            
            self.results.append(validation_result)
            
            # Print validation result
            status = "✅ PASS" if analysis['validation_passed'] else "❌ FAIL"
            print(f"   {status} | Tokens: {completion_tokens}/{test_case.get('max_tokens', 800)} | Reason: {finish_reason}")
            if analysis['repetitive_loops'] > 0:
                print(f"   🔄 WARNING: {analysis['repetitive_loops']} repetitive loops detected")
            if analysis['coherence_issues']:
                print(f"   ⚠️  WARNING: Coherence issues detected")
                
            return validation_result
            
        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def analyze_response(self, response_text, completion_tokens, finish_reason):
        """Analyze response quality and detect issues"""
        
        # Detect repetitive loops (pattern from our previous analysis)
        sentences = response_text.split('.')
        sentence_counts = {}
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Ignore very short fragments
                sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1
        
        # Count how many sentences appear multiple times
        repetitive_loops = sum(1 for count in sentence_counts.values() if count > 2)
        
        # Basic coherence check - look for signs of degradation
        coherence_issues = (
            response_text.count('...') > 5 or  # Too many ellipses
            len(response_text.split()) < 20 or  # Too short for the token allocation
            response_text.count('?') > 10 or   # Too many questions (confusion)
            'error' in response_text.lower()   # Contains error mentions
        )
        
        # Validation criteria
        validation_passed = (
            completion_tokens > 200 and         # Complete response
            finish_reason == "stop" and         # Not truncated
            repetitive_loops == 0 and          # No loops
            not coherence_issues               # Maintains coherence
        )
        
        return {
            'validation_passed': validation_passed,
            'repetitive_loops': repetitive_loops,
            'coherence_issues': coherence_issues,
            'word_count': len(response_text.split()),
            'completion_rate': completion_tokens / 800 if completion_tokens else 0
        }
    
    def run_validation_suite(self):
        """Run validation on key optimized domains"""
        
        test_cases = [
            # Medium difficulty tests that were upgraded from 50→800 tokens
            ("domains/liminal_concepts/base_models/medium.json", "lc_001"),
            ("domains/liminal_concepts/base_models/medium.json", "lc_003"), 
            ("domains/synthetic_knowledge/base_models/medium.json", "sk_001"),
            ("domains/synthetic_knowledge/base_models/medium.json", "sk_004"),
            ("domains/speculative_worlds/base_models/medium.json", "sw_001"),
            ("domains/speculative_worlds/base_models/medium.json", "sw_003"),
            ("domains/emergent_systems/base_models/medium.json", "es_001"),
            ("domains/emergent_systems/base_models/medium.json", "es_005"),
        ]
        
        print("🚀 Token Optimization Validation Suite")
        print("=" * 60)
        print("Testing newly optimized domains (50→800 tokens)")
        print("Checking for: complete responses, no truncation, no loops")
        
        successful_validations = 0
        total_tests = len(test_cases)
        
        for domain_path, test_id in test_cases:
            result = self.validate_domain_test(domain_path, test_id)
            if result and result['analysis']['validation_passed']:
                successful_validations += 1
            time.sleep(2)  # Rate limiting
        
        # Generate summary report
        self.generate_summary_report(successful_validations, total_tests)
    
    def generate_summary_report(self, successful, total):
        """Generate comprehensive validation summary"""
        
        print("\n" + "=" * 60)
        print("📊 TOKEN OPTIMIZATION VALIDATION SUMMARY")
        print("=" * 60)
        
        print(f"✅ Successful validations: {successful}/{total} ({successful/total*100:.1f}%)")
        
        if successful == total:
            print("🎉 ALL VALIDATIONS PASSED! Token optimization is successful.")
            print("   ✅ No repetitive loops detected")
            print("   ✅ Complete responses (>200 tokens)")
            print("   ✅ Proper completion (finish_reason = 'stop')")
            print("   ✅ Ready for production deployment")
        elif successful >= total * 0.75:
            print("🟡 MOSTLY SUCCESSFUL - Minor issues detected")
            print("   ⚠️  Some tests may need fine-tuning")
        else:
            print("❌ VALIDATION FAILED - Major issues detected")
            print("   🚨 Token limits may need adjustment")
        
        # Detailed breakdown
        if self.results:
            print(f"\n📈 DETAILED METRICS:")
            avg_completion = sum(r['completion_tokens'] for r in self.results) / len(self.results)
            print(f"   Average completion tokens: {avg_completion:.0f}")
            
            finish_reasons = {}
            for result in self.results:
                reason = result['finish_reason']
                finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
            print(f"   Finish reasons: {finish_reasons}")
            
            loop_count = sum(r['analysis']['repetitive_loops'] for r in self.results)
            print(f"   Total repetitive loops detected: {loop_count}")
        
        # Save detailed results
        output_file = "test_results/token_optimization_validation.json"
        try:
            os.makedirs("test_results", exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump({
                    'summary': {
                        'successful_validations': successful,
                        'total_tests': total,
                        'success_rate': successful/total,
                        'timestamp': time.time()
                    },
                    'detailed_results': self.results
                }, f, indent=2)
            print(f"\n💾 Detailed results saved to: {output_file}")
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")

def main():
    validator = TokenOptimizationValidator()
    validator.run_validation_suite()

if __name__ == "__main__":
    main()