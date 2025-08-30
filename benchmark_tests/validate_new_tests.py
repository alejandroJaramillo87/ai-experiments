#!/usr/bin/env python3
"""
Quick validation script for new knowledge systems conflict and logic systems comparison tests.
Tests that new tests trigger appropriate cultural evaluation metrics.

Author: Claude Code
Version: 1.0.0
"""

import sys
import os
import json

# Add evaluator to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from evaluator.reasoning_evaluator import UniversalEvaluator

def test_knowledge_systems_conflict():
    """Test a knowledge systems conflict test with the evaluation pipeline"""
    
    evaluator = UniversalEvaluator()
    
    # Sample response that demonstrates good cultural integration for a knowledge systems conflict test
    sample_response = """
    This scenario presents a fascinating opportunity to explore how different knowledge systems can complement each other in understanding depression and healing.

    **Western Biomedical Perspective:**
    The biomedical model identifies depression as a neurochemical imbalance involving serotonin, dopamine, and norepinephrine. Treatment typically includes SSRIs, cognitive behavioral therapy, and lifestyle modifications. This approach emphasizes evidence-based interventions, standardized diagnostic criteria, and measurable outcomes.

    **Traditional Healing Perspective (respectfully acknowledged):**
    Many indigenous healing traditions understand depression within a broader framework of spiritual, community, and ecological relationships. Healing may involve ceremonial practices, plant medicines, community support, and restoration of cultural connections. These approaches often view mental distress as reflecting disconnection from cultural identity, land relationships, or spiritual harmony.

    **Integration Possibilities:**
    Rather than viewing these as conflicting systems, they can inform each other:
    - Traditional emphasis on community support aligns with research on social determinants of mental health
    - Holistic approaches complement biomedical treatment by addressing spiritual and cultural dimensions
    - Indigenous concepts of balance and harmony offer frameworks for understanding wellness beyond symptom reduction
    - Traditional plant medicines have provided foundations for many pharmaceutical interventions

    **Respectful Approach:**
    It's crucial to recognize that traditional healing systems are complete, sophisticated frameworks developed over millennia. Integration requires genuine partnership with indigenous communities, respect for cultural protocols, and acknowledgment that some traditional knowledge may not be appropriate for outside access or academic study.

    Both systems offer valuable insights for supporting human wellbeing, and the most effective approaches often combine multiple ways of knowing while respecting the integrity of each tradition.
    """
    
    print("Testing knowledge systems conflict evaluation...")
    
    result = evaluator.evaluate_response(
        sample_response,
        "conflict_01_depression_treatment",
        reasoning_type=None  # Use default reasoning evaluation
    )
    
    print(f"Overall Score: {result.metrics.overall_score}")
    print(f"Cultural Authenticity: {result.metrics.cultural_authenticity}")
    print(f"Tradition Respect: {result.metrics.tradition_respect}")
    print(f"Cross-Cultural Coherence: {result.metrics.cross_cultural_coherence}")
    print(f"Organization Quality: {result.metrics.organization_quality}")
    print(f"Technical Accuracy: {result.metrics.technical_accuracy}")
    print(f"Completeness: {result.metrics.completeness}")
    
    return result

def test_logic_systems_comparison():
    """Test a logic systems comparison test with the evaluation pipeline"""
    
    evaluator = UniversalEvaluator()
    
    # Sample response demonstrating good logic systems comparison
    sample_response = """
    This community conflict resolution scenario demonstrates how different logical frameworks can offer complementary insights for effective mediation.

    **Western Negotiation Linear Logic Analysis:**
    This approach systematically identifies each party's positions and underlying interests, then generates creative options for mutual gain. It emphasizes legal frameworks, enforceable agreements, and rational problem-solving. The strength is creating clear, binding resolutions that protect individual rights and establish precedent for future conflicts.

    **Indigenous Peacemaking Circular Logic Analysis:**
    This framework recognizes that conflict affects the entire community web and focuses on healing relationships rather than just resolving disputes. It involves ceremonial acknowledgment of harm, storytelling to build understanding, and ongoing commitment to community harmony. The strength is addressing root causes and strengthening community bonds.

    **Eastern Harmonizing Dialectical Logic Analysis:**
    This approach seeks to transform the conflict itself into creative energy by finding deeper unity beneath surface opposition. It emphasizes balance, sustainable solutions, and cultivating harmony between competing forces. The strength is creating solutions that honor both perspectives while transcending the original conflict.

    **Integrated Mediation Framework:**
    Each logical system offers essential elements:
    - Linear logic provides structure and enforceability
    - Circular logic ensures community healing and relationship repair
    - Dialectical logic creates sustainable balance and transformation

    An effective mediation might begin with Indigenous storytelling and acknowledgment, use Western negotiation techniques to identify concrete solutions, and conclude with Eastern principles of ongoing harmony cultivation. This integrated approach addresses immediate needs while building long-term community resilience.

    The key insight is that different logical frameworks excel in different aspects of conflict resolution, and skillful mediation can draw on multiple approaches while respecting the integrity of each tradition.
    """
    
    print("\nTesting logic systems comparison evaluation...")
    
    result = evaluator.evaluate_response(
        sample_response,
        "logic_07_conflict_mediation",
        reasoning_type=None  # Use default reasoning evaluation
    )
    
    print(f"Overall Score: {result.metrics.overall_score}")
    print(f"Cultural Authenticity: {result.metrics.cultural_authenticity}")
    print(f"Tradition Respect: {result.metrics.tradition_respect}")
    print(f"Cross-Cultural Coherence: {result.metrics.cross_cultural_coherence}")
    print(f"Organization Quality: {result.metrics.organization_quality}")
    print(f"Technical Accuracy: {result.metrics.technical_accuracy}")
    print(f"Completeness: {result.metrics.completeness}")
    
    return result

if __name__ == "__main__":
    print("Validating new test categories with cultural evaluation pipeline...")
    print("=" * 70)
    
    try:
        # Test knowledge systems conflict evaluation
        knowledge_result = test_knowledge_systems_conflict()
        
        # Test logic systems comparison evaluation
        logic_result = test_logic_systems_comparison()
        
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        print(f"Knowledge Systems Conflict Test:")
        print(f"  - Cultural metrics active: {knowledge_result.metrics.cultural_authenticity > 0}")
        print(f"  - Tradition respect measured: {knowledge_result.metrics.tradition_respect > 0}")
        print(f"  - Cross-cultural coherence evaluated: {knowledge_result.metrics.cross_cultural_coherence > 0}")
        
        print(f"\nLogic Systems Comparison Test:")
        print(f"  - Cultural metrics active: {logic_result.metrics.cultural_authenticity > 0}")
        print(f"  - Tradition respect measured: {logic_result.metrics.tradition_respect > 0}")
        print(f"  - Cross-cultural coherence evaluated: {logic_result.metrics.cross_cultural_coherence > 0}")
        
        if (knowledge_result.metrics.cultural_authenticity > 0 and 
            logic_result.metrics.cultural_authenticity > 0):
            print(f"\n✅ SUCCESS: New tests successfully trigger cultural evaluation metrics!")
        else:
            print(f"\n⚠️  WARNING: Some cultural metrics may not be triggering properly")
            
    except Exception as e:
        print(f"❌ ERROR during validation: {e}")
        sys.exit(1)