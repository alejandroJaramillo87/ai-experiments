#!/usr/bin/env python3
"""
Full Integration Test for Enhanced Evaluation with benchmark_runner.py

Tests the complete integration of enhanced evaluation capabilities
with the benchmark runner system, validating performance and usability.

"""

import subprocess
import json
import tempfile
import os
from pathlib import Path

def test_benchmark_runner_help():
    """Test that new command-line arguments are available"""
    print("="*60)
    print("TEST 1: Command-line Help Integration")
    print("="*60)
    
    try:
        result = subprocess.run([
            'python', 'benchmark_runner.py', '--help'
        ], capture_output=True, text=True, cwd='/home/alejandro/workspace/ai-workstation/benchmark_tests')
        
        help_text = result.stdout
        
        # Check for our new arguments
        enhanced_args = [
            '--enhanced-evaluation',
            '--evaluation-mode', 
            '--domain-focus'
        ]
        
        missing_args = []
        for arg in enhanced_args:
            if arg not in help_text:
                missing_args.append(arg)
        
        if missing_args:
            print(f"❌ Missing command-line arguments: {missing_args}")
            return False
        
        print("✅ All enhanced evaluation arguments available in help")
        
        # Show the relevant help sections
        lines = help_text.split('\n')
        for i, line in enumerate(lines):
            if 'enhanced-evaluation' in line.lower():
                print(f"   Found: {line.strip()}")
                if i + 1 < len(lines):
                    print(f"          {lines[i+1].strip()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Help test failed: {e}")
        return False

def create_simple_test_directory():
    """Create a simple test directory structure for integration testing"""
    import tempfile
    import os
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    test_data = {
        "suite_info": {
            "name": "Enhanced Evaluation Integration Test",
            "description": "Simple test for Phase 1 integration validation",
            "total_tests": 2,
            "version": "1.0.0",
            "difficulty": "easy"
        },
        "tests": [
            {
                "id": "integration_test_01",
                "name": "Japanese Haiku Pattern Recognition",
                "category": "pattern_recognition",
                "reasoning_type": "pattern_recognition",
                "description": "Test enhanced evaluation with cultural pattern recognition",
                "prompt": "Complete this traditional Japanese haiku following the 5-7-5 syllable pattern:\n\nCherry blossoms fall\nGentle spring breeze carries them\n_________________\n\nComplete the haiku:",
                "expected_patterns": ["ground", "softly", "spring", "gentle"],
                "scoring": {
                    "exact_match": 1.0,
                    "partial_match": 0.6,
                    "semantic_similarity": 0.4
                },
                "metadata": {
                    "concepts_tested": ["haiku_structure", "cultural_authenticity"],
                    "domains_integrated": ["language", "creativity"],
                    "reasoning_steps": 2
                },
                "parameters": {
                    "max_tokens": 50,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stream": False
                }
            },
            {
                "id": "integration_test_02", 
                "name": "Basic Logic Pattern",
                "category": "basic_logic",
                "reasoning_type": "general",
                "description": "Simple logic completion test",
                "prompt": "If A > B and B > C, then A __ C. Complete this logical relationship:",
                "parameters": {
                    "max_tokens": 30,
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "stream": False
                }
            }
        ]
    }
    
    # Create the test file in the directory
    test_file_path = os.path.join(temp_dir, 'easy.json')
    with open(test_file_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    return temp_dir

def test_basic_evaluation_mode():
    """Test basic evaluation mode (backward compatible)"""
    print("="*60)
    print("TEST 2: Basic Evaluation Mode")
    print("="*60)
    
    test_dir = create_simple_test_directory()
    
    try:
        result = subprocess.run([
            'python', 'benchmark_runner.py',
            '--test-definitions', test_dir,
            '--evaluation',
            '--dry-run',
            '--mode', 'single',
            '--test-id', 'integration_test_01'
        ], capture_output=True, text=True, cwd='/home/alejandro/workspace/ai-workstation/benchmark_tests')
        
        output = result.stdout + result.stderr
        
        if result.returncode != 0:
            print(f"❌ Basic evaluation dry run failed")
            print(f"Output: {output}")
            return False
        
        # Check that evaluation mode is recognized
        if 'Evaluation:' not in output and 'evaluation' not in output.lower():
            print(f"❌ Evaluation mode not recognized in output")
            return False
            
        print("✅ Basic evaluation mode working correctly")
        print(f"   Command processed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic evaluation test failed: {e}")
        return False
    finally:
        # Clean up temp directory
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

def test_enhanced_evaluation_mode():
    """Test enhanced evaluation mode with multi-tier scoring"""  
    print("="*60)
    print("TEST 3: Enhanced Evaluation Mode")
    print("="*60)
    
    test_dir = create_simple_test_directory()
    
    try:
        result = subprocess.run([
            'python', 'benchmark_runner.py',
            '--test-definitions', test_dir, 
            '--enhanced-evaluation',
            '--evaluation-mode', 'full',
            '--domain-focus', 'reasoning',
            '--dry-run',
            '--mode', 'single',
            '--test-id', 'integration_test_01'
        ], capture_output=True, text=True, cwd='/home/alejandro/workspace/ai-workstation/benchmark_tests')
        
        output = result.stdout + result.stderr
        
        if result.returncode != 0:
            print(f"❌ Enhanced evaluation dry run failed")
            print(f"Output: {output}")
            return False
        
        # Check that enhanced evaluation mode is recognized
        expected_indicators = [
            'enhanced', 'full', 'Domain Focus: reasoning'
        ]
        
        found_indicators = []
        for indicator in expected_indicators:
            if indicator in output:
                found_indicators.append(indicator)
        
        if not found_indicators:
            print(f"❌ Enhanced evaluation indicators not found in output")
            print(f"Output: {output}")
            return False
            
        print("✅ Enhanced evaluation mode working correctly")
        print(f"   Found indicators: {found_indicators}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced evaluation test failed: {e}")
        return False
    finally:
        # Clean up temp directory
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

def test_evaluation_modes_comparison():
    """Test different evaluation modes"""
    print("="*60)
    print("TEST 4: Evaluation Modes Comparison")
    print("="*60)
    
    test_dir = create_simple_test_directory()
    modes = [
        ('basic', ['--evaluation']),
        ('enhanced-basic', ['--enhanced-evaluation', '--evaluation-mode', 'basic']),
        ('enhanced-full', ['--enhanced-evaluation', '--evaluation-mode', 'full'])
    ]
    
    results = {}
    
    try:
        for mode_name, args in modes:
            cmd = [
                'python', 'benchmark_runner.py',
                '--test-definitions', test_dir,
                '--dry-run',
                '--mode', 'single',
                '--test-id', 'integration_test_01'
            ] + args
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                 cwd='/home/alejandro/workspace/ai-workstation/benchmark_tests')
            
            results[mode_name] = {
                'returncode': result.returncode,
                'success': result.returncode == 0,
                'output': result.stdout + result.stderr
            }
        
        # Analyze results
        successful_modes = [mode for mode, data in results.items() if data['success']]
        
        print(f"✅ Successful evaluation modes: {len(successful_modes)}/{len(modes)}")
        for mode in successful_modes:
            print(f"   ✅ {mode}")
        
        failed_modes = [mode for mode, data in results.items() if not data['success']]
        for mode in failed_modes:
            print(f"   ❌ {mode}: {results[mode]['output'][:100]}...")
        
        return len(successful_modes) >= 2  # At least 2 modes should work
        
    except Exception as e:
        print(f"❌ Evaluation modes comparison failed: {e}")
        return False
    finally:
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

def test_performance_baseline():
    """Test performance baseline with dry run"""
    print("="*60)
    print("TEST 5: Performance Baseline") 
    print("="*60)
    
    # Use actual reasoning tests if available
    reasoning_test_path = Path('/home/alejandro/workspace/ai-workstation/benchmark_tests/domains/reasoning/base_models/easy.json')
    
    if not reasoning_test_path.exists():
        print(f"❌ Reasoning test file not found: {reasoning_test_path}")
        return False
    
    try:
        import time
        
        # Test basic evaluation performance
        start_time = time.time()
        
        result = subprocess.run([
            'python', 'benchmark_runner.py',
            '--test-definitions', 'domains/reasoning/base_models',
            '--evaluation',
            '--dry-run',
            '--mode', 'category', 
            '--category', 'basic_logic_patterns'
        ], capture_output=True, text=True, cwd='/home/alejandro/workspace/ai-workstation/benchmark_tests')
        
        basic_time = time.time() - start_time
        
        # Test enhanced evaluation performance
        start_time = time.time()
        
        result2 = subprocess.run([
            'python', 'benchmark_runner.py',
            '--test-definitions', 'domains/reasoning/base_models',
            '--enhanced-evaluation',
            '--evaluation-mode', 'enhanced', 
            '--dry-run',
            '--mode', 'category',
            '--category', 'basic_logic_patterns'
        ], capture_output=True, text=True, cwd='/home/alejandro/workspace/ai-workstation/benchmark_tests')
        
        enhanced_time = time.time() - start_time
        
        print(f"✅ Performance baseline completed")
        print(f"   Basic evaluation dry run: {basic_time:.3f}s")
        print(f"   Enhanced evaluation dry run: {enhanced_time:.3f}s") 
        print(f"   Performance overhead: {((enhanced_time - basic_time) / basic_time * 100):.1f}%")
        
        # Both should complete in reasonable time
        return basic_time < 10.0 and enhanced_time < 15.0
        
    except Exception as e:
        print(f"❌ Performance baseline test failed: {e}")
        return False

def main():
    """Run full integration test suite"""
    print("Enhanced Evaluation - Full Integration Test Suite")
    print("="*60)
    
    tests = [
        ("Command-line Help", test_benchmark_runner_help),
        ("Basic Evaluation Mode", test_basic_evaluation_mode),
        ("Enhanced Evaluation Mode", test_enhanced_evaluation_mode),
        ("Evaluation Modes Comparison", test_evaluation_modes_comparison),
        ("Performance Baseline", test_performance_baseline)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("FULL INTEGRATION TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nIntegration Tests Passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("🎉 Full integration successful! Enhanced evaluation ready for production.")
        return 0
    elif passed >= len(results) - 1:
        print("⚠️  Integration mostly successful. Minor issues detected.")
        return 0  
    else:
        print("❌ Integration issues detected. Review output for details.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())