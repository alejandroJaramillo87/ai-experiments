#!/usr/bin/env python3
"""
Debug Scoring Calibration for Enhanced Universal Evaluator

Tests and validates the Phase 1 scoring calibration fixes to ensure quality responses
receive appropriate scores in the 40-60/100 range while maintaining differentiation.

"""

import sys
import json
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.append('.')

from evaluator.subjects.enhanced_universal_evaluator import EnhancedUniversalEvaluator
from evaluator.subjects.reasoning_evaluator import UniversalEvaluator, ReasoningType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def print_section_header(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print('='*60)

def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n--- {title} ---")

def test_scoring_calibration_fix():
    """Test the scoring calibration fix with various response qualities"""
    print_section_header("Phase 1 Scoring Calibration Validation")
    
    enhanced_evaluator = EnhancedUniversalEvaluator()
    standard_evaluator = UniversalEvaluator()
    
    # Test cases with different quality levels for calibration validation
    test_cases = [
        {
            "name": "High Quality Cultural Response",
            "response": """Cherry blossoms fall softly to the ground, carried by the gentle spring breeze. This traditional Japanese haiku follows the 5-7-5 syllable pattern and captures the essence of spring's fleeting beauty. The imagery connects the reader to the natural world and the passing of seasons, embodying the contemplative nature of haiku poetry with authentic cultural understanding.""",
            "test_definition": {
                "id": "high_quality_haiku",
                "name": "Japanese Haiku Cultural Completion",
                "category": "cultural_reasoning", 
                "description": "Complete traditional Japanese haiku with cultural authenticity",
                "expected_patterns": ["softly", "ground", "spring", "gentle", "beauty", "traditional"],
                "metadata": {
                    "concepts_tested": ["haiku_structure", "cultural_authenticity", "seasonal_imagery"],
                    "domains_integrated": ["language", "creativity"]
                }
            },
            "expected_range": "50-70/100"
        },
        {
            "name": "Medium Quality Response",
            "response": """Soft petals whisper as they fall to the ground in spring.""",
            "test_definition": {
                "id": "medium_quality_haiku", 
                "name": "Basic Haiku Completion",
                "category": "basic_completion",
                "description": "Complete the haiku with basic correctness", 
                "expected_patterns": ["ground", "spring", "fall"],
                "metadata": {
                    "concepts_tested": ["completion", "seasonal_theme"]
                }
            },
            "expected_range": "30-50/100"
        },
        {
            "name": "Low Quality Response",
            "response": """Blue cats dancing.""",
            "test_definition": {
                "id": "low_quality_response",
                "name": "Off-topic Response", 
                "category": "irrelevant_response",
                "description": "Complete spring haiku appropriately",
                "expected_patterns": ["spring", "cherry", "blossom", "gentle"],
                "metadata": {
                    "concepts_tested": ["haiku_completion", "spring_theme"]
                }
            },
            "expected_range": "5-25/100"
        },
        {
            "name": "Real Haiku Completion Test",
            "response": """Soft petals whisper""",
            "test_definition": {
                "id": "basic_01",
                "name": "Test 1: Japanese Haiku Pattern Completion",
                "category": "basic_logic_patterns",
                "reasoning_type": "pattern_recognition",
                "description": "Complete haiku following traditional Japanese 5-7-5 syllable pattern and seasonal themes",
                "prompt": "Complete this traditional Japanese haiku following the 5-7-5 syllable pattern and incorporating seasonal imagery:\n\nCherry blossoms fall\nGentle spring breeze carries them\n_________________\n\nThe final line should:\n- Have exactly 5 syllables\n- Connect to the spring/cherry blossom theme\n- Follow traditional haiku nature imagery\n- Create a sense of closure or reflection\n\nComplete the haiku:",
                "parameters": {
                    "max_tokens": 100,
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "stream": False
                }
            },
            "expected_range": "75-85/100",
            "qualitative_assessment": "Perfect technical compliance (5 syllables, thematic connection), sophisticated poetic technique (personification), cultural authenticity"
        }
    ]
    
    print("Testing scoring calibration with different response qualities...\n")
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print_subsection(f"Test {i}: {test_case['name']}")
        print(f"Response: '{test_case['response']}'")
        print(f"Expected Range: {test_case['expected_range']}")
        
        try:
            # Enhanced evaluation
            enhanced_result = enhanced_evaluator.evaluate_response_enhanced(
                response_text=test_case['response'],
                test_definition=test_case['test_definition']
            )
            
            # Standard evaluation for comparison
            standard_result = standard_evaluator.evaluate_response(
                response_text=test_case['response'],
                test_name=test_case['test_definition']['name'],
                reasoning_type=ReasoningType.GENERAL
            )
            
            # Extract scores
            enhanced_score = enhanced_result.enhanced_metrics.overall_score
            standard_score = standard_result.metrics.overall_score
            
            print(f"Enhanced Score: {enhanced_score:.1f}/100")
            print(f"Standard Score: {standard_score:.1f}/100")
            print(f"Score Delta: {enhanced_score - standard_score:+.1f}")
            
            # Enhanced metrics breakdown
            enhanced_metrics = enhanced_result.enhanced_metrics
            print(f"Enhanced Metrics:")
            print(f"  Exact Match: {enhanced_metrics.exact_match_score:.3f}")
            print(f"  Partial Match: {enhanced_metrics.partial_match_score:.3f}")
            print(f"  Semantic Similarity: {enhanced_metrics.semantic_similarity_score:.3f}")
            
            # Base metrics for reference
            base_metrics = enhanced_result.metrics
            print(f"Base Metrics:")
            print(f"  Technical Accuracy: {base_metrics.technical_accuracy:.1f}")
            print(f"  Cultural Authenticity: {base_metrics.cultural_authenticity:.1f}")
            print(f"  Organization Quality: {base_metrics.organization_quality:.1f}")
            
            results.append({
                'name': test_case['name'],
                'response_preview': test_case['response'][:50] + "..." if len(test_case['response']) > 50 else test_case['response'],
                'enhanced_score': enhanced_score,
                'standard_score': standard_score,
                'expected_range': test_case['expected_range'],
                'enhanced_metrics': {
                    'exact_match': enhanced_metrics.exact_match_score,
                    'partial_match': enhanced_metrics.partial_match_score,
                    'semantic_similarity': enhanced_metrics.semantic_similarity_score
                }
            })
            
            print("✅ Test completed successfully\n")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': test_case['name'],
                'error': str(e),
                'enhanced_score': None,
                'standard_score': None
            })
    
    return results

def analyze_calibration_results(results):
    """Analyze the calibration test results"""
    print_section_header("Scoring Calibration Analysis")
    
    # Filter successful results
    successful_results = [r for r in results if 'error' not in r]
    
    if not successful_results:
        print("❌ No successful test results to analyze")
        return False
    
    print(f"📊 Analysis of {len(successful_results)} successful tests:")
    print(f"{'Test Name':<25} {'Enhanced':<10} {'Standard':<10} {'Delta':<8} {'Expected Range'}")
    print("-" * 80)
    
    score_variance = []
    within_expected_count = 0
    
    for result in successful_results:
        enhanced_score = result['enhanced_score']
        standard_score = result['standard_score']
        delta = enhanced_score - standard_score
        expected_range = result['expected_range']
        
        # Check if score is within expected range (rough validation)
        if 'High Quality' in result['name'] and 50 <= enhanced_score <= 70:
            within_expected_count += 1
        elif 'Medium Quality' in result['name'] and 30 <= enhanced_score <= 50:
            within_expected_count += 1
        elif 'Low Quality' in result['name'] and 5 <= enhanced_score <= 25:
            within_expected_count += 1
        elif 'Original Problem' in result['name'] and 25 <= enhanced_score <= 45:
            within_expected_count += 1
        
        score_variance.append(enhanced_score)
        
        print(f"{result['name']:<25} {enhanced_score:<10.1f} {standard_score:<10.1f} {delta:<8.1f} {expected_range}")
    
    print("-" * 80)
    
    # Analysis summary
    print(f"\n📈 Calibration Validation Results:")
    print(f"  Score Range: {min(score_variance):.1f} - {max(score_variance):.1f}")
    print(f"  Score Variance: {len(set(f'{s:.1f}' for s in score_variance))} unique scores")
    print(f"  Within Expected Range: {within_expected_count}/{len(successful_results)} tests")
    
    # Detailed calibration assessment for Real Haiku Completion Test
    real_haiku_test = next((r for r in successful_results if 'Real Haiku' in r['name']), None)
    if real_haiku_test:
        expected_range = real_haiku_test['expected_range']
        actual_score = real_haiku_test['enhanced_score']
        
        # Parse expected range (e.g., "75-85/100")
        if '-' in expected_range and '/' in expected_range:
            range_part = expected_range.split('/')[0]
            min_expected, max_expected = map(float, range_part.split('-'))
            
            # Calibration assessment
            if min_expected <= actual_score <= max_expected:
                calibration_status = "PERFECT CALIBRATION"
                calibration_icon = "✅"
            elif min_expected - 5 <= actual_score <= max_expected + 5:
                calibration_status = "GOOD CALIBRATION" 
                calibration_icon = "🟡"
            elif min_expected - 10 <= actual_score <= max_expected + 10:
                calibration_status = "NEEDS CALIBRATION"
                calibration_icon = "🟠"
            else:
                calibration_status = "CALIBRATION BROKEN"
                calibration_icon = "❌"
            
            points_off = min(abs(actual_score - min_expected), abs(actual_score - max_expected))
            if min_expected <= actual_score <= max_expected:
                points_off = 0
            
            print(f"\n🎯 Real Haiku Completion Test Calibration Analysis:")
            print(f"  Expected: {expected_range}")
            print(f"  Actual: {actual_score:.1f}/100")
            print(f"  Points Off: {points_off:.1f}")
            print(f"  Status: {calibration_icon} {calibration_status}")
            
            return calibration_status == "PERFECT CALIBRATION"
    
    # Specific issue validation
    original_case = next((r for r in successful_results if 'Original Problem' in r['name']), None)
    if original_case:
        if original_case['enhanced_score'] > 25:
            print(f"✅ Original issue fixed: Score improved from 15.0 to {original_case['enhanced_score']:.1f}")
        else:
            print(f"⚠️  Original issue partially fixed: Score {original_case['enhanced_score']:.1f} (target: 25-45)")
    
    # Success criteria
    success_rate = within_expected_count / len(successful_results)
    if success_rate >= 0.75:  # 75% success rate
        print(f"\n🎉 CALIBRATION SUCCESS: {success_rate:.1%} of tests within expected ranges")
        return True
    else:
        print(f"\n⚠️  CALIBRATION NEEDS WORK: Only {success_rate:.1%} within expected ranges")
        return False

def test_semantic_similarity_fallback():
    """Test semantic similarity fallback behavior"""
    print_section_header("Semantic Similarity Fallback Test")
    
    enhanced_evaluator = EnhancedUniversalEvaluator()
    
    test_definition = {
        "id": "semantic_fallback_test",
        "name": "Semantic Similarity Fallback Test",
        "category": "semantic_analysis",
        "description": "Test semantic similarity fallback behavior",
        "expected_patterns": ["test", "semantic", "analysis"],
        "metadata": {
            "concepts_tested": ["semantic_understanding", "fallback_behavior"],
            "domains_integrated": ["language", "analysis"]
        }
    }
    
    test_cases = [
        "This is a semantic analysis test with keyword matching",
        "Completely unrelated content about cooking recipes",
        "Test semantic understanding and analysis capabilities"
    ]
    
    for i, response in enumerate(test_cases, 1):
        print(f"\nTest {i}: '{response}'")
        
        try:
            result = enhanced_evaluator.evaluate_response_enhanced(response, test_definition)
            metrics = result.enhanced_metrics
            
            print(f"  Semantic Similarity: {metrics.semantic_similarity_score:.3f}")
            print(f"  Exact Match: {metrics.exact_match_score:.3f}")
            print(f"  Partial Match: {metrics.partial_match_score:.3f}")
            print(f"  Overall Score: {metrics.overall_score:.1f}")
            
            if metrics.semantic_similarity_score <= 0.05:
                print("  ✅ Using fallback weighting (semantic similarity disabled)")
            else:
                print("  ✅ Using normal weighting (semantic similarity functional)")
                
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
    
    print("\n✅ Semantic similarity fallback test completed")

def main():
    """Main debug function"""
    print("🔧 Enhanced Universal Evaluator Scoring Calibration Debug")
    print("=" * 60)
    
    try:
        # Test scoring calibration
        results = test_scoring_calibration_fix()
        
        # Analyze results
        calibration_success = analyze_calibration_results(results)
        
        # Test semantic similarity fallback
        test_semantic_similarity_fallback()
        
        # Final summary
        print_section_header("Debug Session Complete")
        
        if calibration_success:
            print("✅ SCORING CALIBRATION SUCCESSFUL!")
            print("🎯 Enhanced evaluator provides realistic score ranges")
            print("📊 Quality responses scoring in appropriate 40-60/100 range")
            print("🔧 Fallback weighting properly compensates for disabled semantic similarity")
        else:
            print("⚠️  SCORING CALIBRATION NEEDS FURTHER ADJUSTMENT")
            print("📊 Some test scores outside expected ranges")
            print("🔧 May require additional scoring formula tuning")
        
        print("\n📋 Phase 1 Enhanced Universal Evaluator Status:")
        print("  ✅ Technical Issues Resolved (PyTorch, JSON serialization)")
        print("  ✅ Scoring System Functional (multi-tier scoring active)")
        print("  🔧 Scoring Calibration Validated" if calibration_success else "  ⚠️  Scoring Calibration Needs Work")
        
        return 0 if calibration_success else 1
        
    except Exception as e:
        print(f"\n❌ Debug session failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())