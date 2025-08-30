#!/usr/bin/env python3
"""
Test Runner with Automatic Cleanup
Runs benchmark tests and automatically cleans up artifacts afterwards.

This script provides a clean testing experience by:
1. Running the specified tests 
2. Automatically cleaning up any leftover directories and result files
3. Providing a summary of test results and cleanup actions

Usage:
    python run_tests_with_cleanup.py [pytest_args...]
    
Examples:
    python run_tests_with_cleanup.py                    # Run all tests
    python run_tests_with_cleanup.py tests/unit/       # Run unit tests only
    python run_tests_with_cleanup.py -v                # Run with verbose output
    python run_tests_with_cleanup.py tests/unit/ -k "test_evaluator"  # Specific tests

Author: Claude Code
Version: 1.0.0
"""

import sys
import subprocess
import os
from pathlib import Path

# Import our cleanup function
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
from cleanup_test_artifacts import main as cleanup_main


def run_tests(pytest_args):
    """Run pytest with the provided arguments."""
    print("🧪 RUNNING BENCHMARK TESTS")
    print("=" * 60)
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"] + pytest_args
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run the tests
    try:
        result = subprocess.run(cmd, cwd=script_dir, capture_output=False)
        return result.returncode
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


def main():
    """Main test runner with cleanup."""
    # Get pytest arguments from command line
    pytest_args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # If no args provided, run all non-functional tests by default
    if not pytest_args:
        pytest_args = ["tests/unit/", "tests/integration/", "--tb=short", "-q"]
    
    try:
        # Run the tests
        exit_code = run_tests(pytest_args)
        
        # Always run cleanup, regardless of test results
        print("\n" + "=" * 60)
        print("🧹 POST-TEST CLEANUP")
        print("=" * 60)
        
        cleanup_main()
        
        # Final summary
        print("\n" + "=" * 60)
        print("🏁 TEST SESSION COMPLETE")
        print("=" * 60)
        
        if exit_code == 0:
            print("✅ All tests passed and artifacts cleaned up!")
        else:
            print("⚠️  Some tests failed, but artifacts have been cleaned up.")
            print("   Check the test output above for details.")
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\n\n⚡ Test run interrupted by user")
        print("Running cleanup before exit...")
        cleanup_main()
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Running cleanup before exit...")
        cleanup_main()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)