#!/usr/bin/env python3
"""
Test script to validate converted core domain tests with Enhanced Universal Evaluator

Validates all 4 core domains:
- Language (230 tests) - linguistic diversity and multilingual competency
- Integration (30 tests) - cross-domain synthesis 
- Knowledge (210 tests) - factual accuracy and reasoning
- Social (210 tests) - cultural communication and social understanding

Total: 680 converted instruct model tests
"""

import json
import sys
from pathlib import Path

# Add the parent directory to sys.path to import from evaluator
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluator.subjects.enhanced_universal_evaluator import EnhancedUniversalEvaluator

def test_core_domains_conversion():
    """Test converted core domain tests with the enhanced evaluator"""
    
    print("🌍 Testing Enhanced Universal Evaluator with Converted Core Domain Tests")
    print("=" * 75)
    
    domains = ["language", "integration", "knowledge", "social"]
    all_results = []
    total_tests_loaded = 0
    
    # Initialize enhanced evaluator
    evaluator = EnhancedUniversalEvaluator()
    
    for domain in domains:
        print(f"\n📚 Testing {domain.upper()} domain...")
        
        # Load converted tests
        test_file = Path(__file__).parent.parent.parent / f"domains/{domain}/instruct_models/easy.json"
        
        if not test_file.exists():
            print(f"   ❌ Test file not found: {test_file}")
            continue
        
        with open(test_file, 'r', encoding='utf-8') as f:
            domain_data = json.load(f)
        
        tests = domain_data['tests']
        total_tests_loaded += len(tests)
        print(f"   📊 Loaded {len(tests)} {domain} tests")
        
        # Test samples from different categories
        domain_results = validate_domain_samples(evaluator, tests, domain)
        all_results.extend(domain_results)
        
        # Domain summary
        successful = sum(1 for r in domain_results if r['success'])
        print(f"   ✅ {successful}/{len(domain_results)} sample tests passed")
    
    # Overall summary
    print(f"\n📊 OVERALL SUMMARY")
    print("=" * 40)
    total_successful = sum(1 for r in all_results if r['success'])
    total_tested = len(all_results)
    
    print(f"📚 Total tests loaded: {total_tests_loaded}")
    print(f"🧪 Sample tests evaluated: {total_tested}")
    print(f"✅ Successful evaluations: {total_successful}/{total_tested}")
    
    if total_successful == total_tested:
        print("🎉 All core domain test conversions work perfectly with Enhanced Universal Evaluator!")
        print("🎯 Ready for full-scale evaluation testing")
        return True
    else:
        print("⚠️  Some issues found - check individual test results above")
        
        # Show failed tests
        failed_tests = [r for r in all_results if not r['success']]
        if failed_tests:
            print(f"\n❌ Failed tests ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"   • {test['domain']}: {test['test_id']} - {test.get('error', 'Unknown error')}")
        
        return False

def validate_domain_samples(evaluator, tests, domain):
    """Test sample tests from a domain"""
    
    # Select diverse samples based on domain size
    if len(tests) >= 100:
        # Large domain - test 4 samples across categories
        sample_indices = [0, len(tests)//4, len(tests)//2, len(tests)-1]
    elif len(tests) >= 30:
        # Medium domain - test 3 samples
        sample_indices = [0, len(tests)//2, len(tests)-1]
    else:
        # Small domain - test 2 samples
        sample_indices = [0, len(tests)-1]
    
    results = []
    
    for i, idx in enumerate(sample_indices):
        if idx >= len(tests):
            continue
            
        test_def = tests[idx]
        print(f"     🧪 Sample {i+1}: {test_def['name']}")
        
        # Generate appropriate sample response
        sample_response = generate_sample_response(test_def, domain)
        
        try:
            # Evaluate with enhanced evaluator
            result = evaluator.evaluate_response_enhanced(sample_response, test_def)
            
            results.append({
                'domain': domain,
                'test_id': test_def['id'],
                'category': test_def['category'],
                'score': result.metrics.overall_score,
                'analysis_length': len(str(result.detailed_analysis)) if result.detailed_analysis else 0,
                'success': True
            })
            
            print(f"       ✅ Score: {result.metrics.overall_score:.1f}")
            
        except Exception as e:
            results.append({
                'domain': domain,
                'test_id': test_def['id'],
                'category': test_def['category'],
                'error': str(e),
                'success': False
            })
            print(f"       ❌ Error: {e}")
    
    return results

def generate_sample_response(test_def, domain):
    """Generate domain-appropriate sample responses"""
    
    category = test_def.get('category', '')
    
    if domain == "language":
        if category == "historical_linguistics":
            return """The Proto-Indo-European reconstruction for 'mother' is *méh₂tēr (nominative *méh₂tḗr). 
            
            Sound changes observed:
            • Sanskrit mātṛ́- shows loss of laryngeal *h₂ and vocalization
            • Greek mētēr preserves the long vowel from laryngeal compensation
            • Latin māter shows regular development with loss of final *-r in nominative
            • Old Church Slavonic mati shows Balto-Slavic shortening and palatalization
            • Gothic *modar shows Germanic Verner's Law alternation
            
            This reconstruction demonstrates the regular correspondences in the kinship terminology system across Indo-European languages."""
            
        elif category == "multilingual_contact" or category == "advanced_code_switching":
            return """The code-switching pattern here shows matrix language dominance with embedded language elements for specific cultural concepts. The speaker maintains grammatical structure of the base language while inserting lexical items that carry specific cultural meaning not easily translated. This reflects identity maintenance and in-group solidarity while communicating across linguistic boundaries. The switching points occur at syntactic boundaries, following Myers-Scotton's Matrix Language Frame model."""
            
        else:
            return f"""This linguistic phenomenon demonstrates systematic patterns in {category.replace('_', ' ')}. The analysis shows regular phonological, morphological, and syntactic processes that reflect deeper structural principles in human language systems. These patterns are consistent with cross-linguistic typological universals and provide insight into the cognitive architecture of language."""
    
    elif domain == "integration":
        return """This interdisciplinary analysis reveals fundamental connections between seemingly disparate domains. The synthesis demonstrates how principles from one field illuminate understanding in another, creating emergent insights that transcend individual disciplinary boundaries. The holistic perspective shows that complex phenomena often require multi-dimensional analysis to fully comprehend their significance and implications across different levels of organization."""
    
    elif domain == "knowledge":
        return """Based on the available evidence, the factual analysis indicates clear causal relationships supported by empirical data. The reasoning follows logical principles, with premises leading to valid conclusions. The information synthesis draws from multiple reliable sources, demonstrating consistency across different knowledge bases. This comprehensive approach ensures accuracy while acknowledging limitations and areas requiring further investigation."""
    
    elif domain == "social":
        return """This social phenomenon reflects complex cultural dynamics that must be understood within specific community contexts. The analysis respects cultural diversity while identifying universal patterns of human cooperation and communication. Understanding these social structures requires sensitivity to power dynamics, historical context, and community values. The approach maintains cultural humility while providing meaningful insight into human social organization."""
    
    else:
        return f"""This analysis demonstrates comprehensive understanding of {domain} concepts with appropriate depth and cultural sensitivity. The response integrates multiple perspectives while maintaining accuracy and respect for diverse viewpoints."""

if __name__ == "__main__":
    success = test_core_domains_conversion()
    exit(0 if success else 1)