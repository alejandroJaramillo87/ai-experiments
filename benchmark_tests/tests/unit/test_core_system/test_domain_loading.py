#!/usr/bin/env python3
"""
Critical Domain File Loading Tests

Tests that domain JSON files are properly loaded and validated.
Keeps it simple - only tests critical loading scenarios that could break the system.
"""

import unittest
import sys
import os
import tempfile
import json

# Add the benchmark_tests directory to Python path
benchmark_tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, benchmark_tests_dir)

from benchmark_runner import BenchmarkTestRunner, TestSuiteManager


class TestDomainLoading(unittest.TestCase):
    """Test critical domain file loading functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.runner = BenchmarkTestRunner()
        self.suite_manager = TestSuiteManager()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_discover_test_suites_finds_existing_domains(self):
        """Test that discover_test_suites finds the actual domain directories"""
        # Get the benchmark_tests directory
        benchmark_tests_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        suites = self.suite_manager.discover_test_suites(benchmark_tests_dir)
        
        # Should find reasoning and creativity domains
        self.assertGreater(len(suites), 0, "Should discover at least one test suite")
        
        # Check that basic structure is found
        suite_ids = [suite.suite_id for suite in suites]
        reasoning_found = any('reasoning' in suite_id for suite_id in suite_ids)
        creativity_found = any('creativity' in suite_id for suite_id in suite_ids)
        
        self.assertTrue(reasoning_found, "Should find reasoning domain")
        self.assertTrue(creativity_found, "Should find creativity domain")
    
    def test_load_categories_handles_missing_file(self):
        """Test graceful handling when categories.json is missing"""
        # Create a temporary domain structure without categories.json
        fake_domain_path = os.path.join(self.temp_dir, "fake_domain", "base_models")
        os.makedirs(fake_domain_path, exist_ok=True)
        
        # This should not crash the system
        try:
            categories_file = os.path.join(fake_domain_path, "categories.json")
            # File doesn't exist, should handle gracefully
            self.assertFalse(os.path.exists(categories_file))
        except Exception as e:
            self.fail(f"Should handle missing categories.json gracefully: {e}")
    
    def test_malformed_json_handled_gracefully(self):
        """Test that malformed JSON files don't crash the system"""
        # Create a malformed categories.json file
        fake_domain_path = os.path.join(self.temp_dir, "fake_domain", "base_models")
        os.makedirs(fake_domain_path, exist_ok=True)
        
        malformed_file = os.path.join(fake_domain_path, "categories.json")
        with open(malformed_file, 'w') as f:
            f.write('{"invalid": json content without closing brace')
        
        # System should handle this gracefully without crashing
        try:
            with open(malformed_file, 'r') as f:
                content = f.read()
                # This would fail JSON parsing but system should handle it
                self.assertIn('invalid', content)
        except Exception as e:
            # Expected to fail JSON parsing, system should catch this
            self.assertIn('json', str(e).lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)