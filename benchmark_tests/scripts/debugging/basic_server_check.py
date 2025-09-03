#!/usr/bin/env python3
"""
Basic Server Diagnostic for Timeout Issues

Simple check to verify if server is responding before analyzing concurrent execution.
"""

import subprocess
import time
import requests
import json


def check_server_basic():
    """Basic server connectivity check"""
    print("🔍 Basic Server Diagnostic")
    print("=" * 50)
    
    # Test 1: Port check
    print("\n1. Port connectivity check...")
    try:
        result = subprocess.run(
            ["curl", "-s", "--connect-timeout", "5", "http://localhost:8004/health"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("✅ Server port 8004 responds to health check")
            print(f"   Response: {result.stdout.strip()[:100]}")
        else:
            print(f"❌ Server health check failed (code: {result.returncode})")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ Server health check error: {e}")
        return False
    
    # Test 2: Simple completion API test
    print("\n2. Basic completion API test...")
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:8004/v1/completions",
            json={"prompt": "Hello", "max_tokens": 5},
            timeout=30
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ Completion API works ({elapsed:.1f}s)")
            try:
                result = response.json()
                if "choices" in result:
                    print(f"   Generated: {result['choices'][0].get('text', '')[:50]}")
                else:
                    print(f"   Response structure: {list(result.keys())}")
            except:
                print("   Response received but couldn't parse JSON")
        else:
            print(f"❌ Completion API failed (status: {response.status_code})")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Completion API error: {e}")
        return False
    
    print(f"\n✅ Server appears healthy - basic checks passed")
    return True


def test_single_benchmark():
    """Test a single benchmark execution"""
    print("\n" + "=" * 50)
    print("🧪 Single Benchmark Test")
    print("=" * 50)
    
    print("\n3. Single test execution timing...")
    try:
        start_time = time.time()
        result = subprocess.run([
            "python3", "benchmark_runner.py",
            "--test-type", "base",
            "--test-id", "basic_01", 
            "--endpoint", "http://localhost:8004",
            "--model", "llama3"
        ], capture_output=True, text=True, timeout=60, 
           cwd="/home/alejandro/workspace/ai-workstation/benchmark_tests")
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Single test completed in {elapsed:.1f}s")
            # Look for timing info in output
            if "execution_time" in result.stdout.lower():
                print("   Execution details found in output")
            return True
        else:
            print(f"❌ Single test failed ({elapsed:.1f}s)")
            print(f"   Exit code: {result.returncode}")
            print(f"   stderr: {result.stderr[:300]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Single test timed out after 60s")
        return False
    except Exception as e:
        print(f"❌ Single test error: {e}")
        return False


def test_two_sequential():
    """Test two tests sequentially"""
    print("\n4. Two sequential tests timing...")
    try:
        start_time = time.time()
        result = subprocess.run([
            "python3", "benchmark_runner.py",
            "--test-type", "base",
            "--test-id", "basic_01,basic_02",
            "--mode", "sequential",
            "--endpoint", "http://localhost:8004",
            "--model", "llama3"
        ], capture_output=True, text=True, timeout=120,
           cwd="/home/alejandro/workspace/ai-workstation/benchmark_tests")
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Sequential tests completed in {elapsed:.1f}s")
            return True
        else:
            print(f"❌ Sequential tests failed ({elapsed:.1f}s)")
            print(f"   Exit code: {result.returncode}")
            print(f"   stderr: {result.stderr[:300]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Sequential tests timed out after 120s")
        return False
    except Exception as e:
        print(f"❌ Sequential tests error: {e}")
        return False


def main():
    """Main diagnostic sequence"""
    print("🔧 CONCURRENT TIMEOUT DIAGNOSTIC")
    print("=" * 50)
    print("This will help identify where the timeout issue occurs")
    print()
    
    # Step 1: Basic server check
    if not check_server_basic():
        print("\n🚨 ISSUE: Server is not responding properly")
        print("   Fix server issues before testing concurrent execution")
        return
    
    # Step 2: Single benchmark test
    if not test_single_benchmark():
        print("\n🚨 ISSUE: Single benchmark execution is broken")
        print("   Fix basic benchmark execution before testing concurrency")
        return
    
    # Step 3: Sequential tests
    if not test_two_sequential():
        print("\n🚨 ISSUE: Sequential execution is broken or too slow")
        print("   Concurrent execution will definitely timeout if sequential is slow")
        return
    
    print("\n" + "=" * 50)
    print("✅ PRELIMINARY CHECKS PASSED")
    print("=" * 50)
    print("Server and basic execution work correctly.")
    print("The timeout issue is likely specific to concurrent execution logic.")
    print("\nNext step: Analyze concurrent execution implementation")


if __name__ == "__main__":
    main()