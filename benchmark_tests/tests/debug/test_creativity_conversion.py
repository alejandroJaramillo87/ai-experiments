#!/usr/bin/env python3
"""
Test script to validate converted creativity tests with Enhanced Universal Evaluator
"""

import json
import sys
from pathlib import Path

# Add the parent directory to sys.path to import from evaluator
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluator.subjects.enhanced_universal_evaluator import EnhancedUniversalEvaluator

def test_creativity_conversion():
    """Test a sample of converted creativity tests with the enhanced evaluator"""
    
    print("🎭 Testing Enhanced Universal Evaluator with Converted Creativity Tests")
    print("=" * 70)
    
    # Load converted creativity tests
    creativity_file = Path(__file__).parent.parent.parent / "domains/creativity/instruct_models/easy.json"
    
    with open(creativity_file, 'r', encoding='utf-8') as f:
        creativity_data = json.load(f)
    
    print(f"📚 Loaded {len(creativity_data['tests'])} converted creativity tests")
    
    # Initialize enhanced evaluator
    evaluator = EnhancedUniversalEvaluator()
    
    # Test sample from different categories
    sample_tests = [
        # Narrative creation
        creativity_data['tests'][0],  # West African Griot
        # Performance writing  
        creativity_data['tests'][25], # First performance test
        # Cultural authenticity
        creativity_data['tests'][200] if len(creativity_data['tests']) > 200 else creativity_data['tests'][-1]
    ]
    
    results = []
    
    for i, test_def in enumerate(sample_tests):
        print(f"\n🧪 Test {i+1}: {test_def['name']}")
        print(f"   Category: {test_def['category']}")
        
        # Simulate a response for testing
        sample_response = generate_sample_response(test_def)
        
        try:
            # Evaluate with enhanced evaluator
            result = evaluator.evaluate_response_enhanced(sample_response, test_def)
            
            results.append({
                'test_id': test_def['id'],
                'category': test_def['category'],
                'score': result.metrics.overall_score,
                'feedback_length': len(str(result.detailed_analysis)) if result.detailed_analysis else 0,
                'success': True
            })
            
            print(f"   ✅ Score: {result.metrics.overall_score:.3f}")
            print(f"   📝 Analysis: {len(str(result.detailed_analysis)) if result.detailed_analysis else 0} characters")
            
        except Exception as e:
            results.append({
                'test_id': test_def['id'],
                'category': test_def['category'], 
                'error': str(e),
                'success': False
            })
            print(f"   ❌ Error: {e}")
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("=" * 30)
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    print(f"✅ Successful evaluations: {successful}/{total}")
    
    if successful == total:
        print("🎉 All creativity test conversions work perfectly with Enhanced Universal Evaluator!")
        return True
    else:
        print("⚠️  Some issues found - check individual test results above")
        return False

def generate_sample_response(test_def):
    """Generate a sample response appropriate for the test category"""
    
    category = test_def.get('category', '')
    
    if category == 'narrative_creation':
        return """Listen well, my people, gather close to hear the tale of Amina, daughter of the blacksmith, granddaughter of wise Fatou who could speak with the wind spirits. In the time when our ancestors walked alongside the baobab trees, when the rivers sang their ancient songs, there lived a young woman whose heart was torn between two paths..."""
        
    elif category == 'performance_writing':
        return """[Stage dim, single spotlight on SPIRIT OF THE FALLEN WARRIOR]

SPIRIT: (moving with slow, deliberate steps) The autumn leaves... they fall as I once fell, upon the battlefield of honor...

MONK: (entering from stage left, humble posture) Troubled spirit, what binds you to this earthly realm?

SPIRIT: My name... forgotten like morning mist..."""
        
    else:
        return """This creative response demonstrates cultural awareness and sensitivity while engaging with the traditional form respectfully. The work incorporates authentic elements while being appropriate for cross-cultural appreciation."""

if __name__ == "__main__":
    success = test_creativity_conversion()
    exit(0 if success else 1)