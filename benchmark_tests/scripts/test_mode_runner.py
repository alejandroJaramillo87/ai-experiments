#!/usr/bin/env python3
"""
Test Mode Runner Script

Executes tests in different modes (production, development, debug) with optimized
configurations based on the test mode configuration system.

Usage:
    python scripts/test_mode_runner.py <mode> [additional_args...]
    
Examples:
    python scripts/test_mode_runner.py development
    python scripts/test_mode_runner.py production --verbose
    python scripts/test_mode_runner.py debug -k test_evaluator
"""

import sys
import os
import subprocess
import logging
from pathlib import Path
from typing import List, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.test_mode_config import (
    TestMode, TestModeManager, initialize_test_mode, get_test_mode_manager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestModeRunner:
    """Runs tests in different modes with appropriate configurations"""
    
    def __init__(self):
        self.manager: Optional[TestModeManager] = None
        
    def parse_mode(self, mode_str: str) -> TestMode:
        """Parse mode string to TestMode enum"""
        mode_mapping = {
            'production': TestMode.PRODUCTION,
            'prod': TestMode.PRODUCTION,
            'production_full': TestMode.PRODUCTION_FULL,
            'prod_full': TestMode.PRODUCTION_FULL,
            'development': TestMode.DEVELOPMENT,
            'dev': TestMode.DEVELOPMENT,
            'debug': TestMode.DEBUG,
            'dbg': TestMode.DEBUG
        }
        
        mode = mode_mapping.get(mode_str.lower())
        if not mode:
            raise ValueError(f"Invalid test mode: {mode_str}. Valid modes: {list(mode_mapping.keys())}")
        
        return mode
    
    def run_tests(self, mode: TestMode, additional_args: List[str] = None) -> int:
        """Run tests in specified mode"""
        additional_args = additional_args or []
        
        try:
            # Initialize test mode
            logger.info(f"Initializing test mode: {mode.value}")
            config = initialize_test_mode(mode)
            self.manager = get_test_mode_manager()
            
            # Display configuration
            print(f"⚙️ Configuration:")
            print(f"   Mode: {config.mode.value}")
            print(f"   Description: {config.description}")
            print(f"   Chunk Size: {config.chunk_size}")
            print(f"   Max Concurrent: {config.max_concurrent}")
            print(f"   Timeout Per Test: {config.timeout_per_test}s")
            print(f"   Memory Limit: {config.memory_limit_mb}MB")
            print(f"   Coverage Enabled: {config.enable_coverage}")
            print(f"   Verbose Output: {config.verbose_output}")
            print(f"   Include Functional: {config.include_functional_tests}")
            print(f"   Include Calibration: {config.include_calibration_tests}")
            print()
            
            # Check server availability for server-dependent modes
            if mode in [TestMode.PRODUCTION_FULL] or (
                config.include_functional_tests or config.include_calibration_tests
            ):
                print("🔍 Checking server availability...")
                server_available = self.manager.check_server_availability()
                
                if not server_available:
                    if mode == TestMode.PRODUCTION_FULL:
                        print("❌ Server not available - Full production mode requires LLM server")
                        print("💡 Please start the LLM server at http://localhost:8004")
                        print("💡 Or use 'make test-production' for server-independent tests")
                        return 1
                    else:
                        print("⚠️ Server not available - skipping server-dependent tests")
                        # Temporarily disable server-dependent tests
                        config.include_functional_tests = False
                        config.include_calibration_tests = False
                else:
                    print("✅ Server is available")
            
            # Get test directories and pytest arguments
            test_dirs = self.manager.get_test_directories()
            pytest_args = self.manager.get_pytest_args()
            
            print(f"🚀 Running {mode.value} tests:")
            print(f"   Test Directories: {', '.join(test_dirs)}")
            print(f"   Pytest Args: {' '.join(pytest_args)}")
            print()
            
            # Build pytest command
            cmd = ['python', '-m', 'pytest'] + pytest_args + test_dirs + additional_args
            
            # Filter out empty args
            cmd = [arg for arg in cmd if arg.strip()]
            
            logger.info(f"Executing command: {' '.join(cmd)}")
            
            # Run pytest
            result = subprocess.run(cmd, cwd=project_root)
            return result.returncode
            
        except Exception as e:
            logger.error(f"Error running tests in {mode.value} mode: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    def display_mode_info(self, mode: TestMode):
        """Display information about a test mode without running tests"""
        config = initialize_test_mode(mode)
        manager = get_test_mode_manager()
        
        print(f"📊 Test Mode Information: {mode.value}")
        print("=" * 50)
        print(f"Description: {config.description}")
        print(f"Chunk Size: {config.chunk_size}")
        print(f"Max Concurrent: {config.max_concurrent}")
        print(f"Timeout Per Test: {config.timeout_per_test}s")
        print(f"Memory Limit: {config.memory_limit_mb}MB")
        print(f"Coverage Enabled: {config.enable_coverage}")
        print(f"Verbose Output: {config.verbose_output}")
        print(f"Include Functional Tests: {config.include_functional_tests}")
        print(f"Include Calibration Tests: {config.include_calibration_tests}")
        print(f"Include Integration Tests: {config.include_integration_tests}")
        print(f"Include Network Failure Tests: {config.include_network_failure_tests}")
        print(f"Success Rate Threshold: {config.min_success_rate}")
        print(f"Calibration Threshold: {config.calibration_threshold}")
        print()
        print(f"Test Directories: {', '.join(manager.get_test_directories())}")
        print(f"Pytest Args: {' '.join(manager.get_pytest_args())}")
    
def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_mode_runner.py <mode> [additional_args...]")
        print("Modes: production, development, debug")
        print("Options:")
        print("  --info    Show mode information without running tests")
        return 1
    
    mode_str = sys.argv[1]
    additional_args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    # Check for info flag
    if '--info' in additional_args:
        additional_args.remove('--info')
        show_info = True
    else:
        show_info = False
    
    runner = TestModeRunner()
    
    try:
        mode = runner.parse_mode(mode_str)
        
        if show_info:
            runner.display_mode_info(mode)
            return 0
        else:
            return runner.run_tests(mode, additional_args)
            
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️ Test execution interrupted by user")
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())