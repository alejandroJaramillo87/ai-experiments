#!/usr/bin/env python3
"""
Test Scripts Organization

Verify that all reorganized scripts can be imported correctly and that
the directory structure changes don't break existing functionality.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class TestScriptImports(unittest.TestCase):
    """Test that reorganized scripts can be imported"""
    
    def test_calibration_scripts_exist(self):
        """Test calibration scripts exist in new location"""
        calibration_dir = project_root / "scripts" / "calibration"
        
        expected_files = [
            "calibration_success_analysis.py",
            "production_calibration_framework.py"
        ]
        
        for filename in expected_files:
            file_path = calibration_dir / filename
            self.assertTrue(file_path.exists(), f"Missing {filename} in calibration/")
    
    def test_optimization_scripts_exist(self):
        """Test optimization scripts exist in new location"""
        optimization_dir = project_root / "scripts" / "optimization"
        
        expected_files = [
            "token_optimization.py",
            "refined_token_optimization.py", 
            "scale_token_optimization.py"
        ]
        
        for filename in expected_files:
            file_path = optimization_dir / filename
            self.assertTrue(file_path.exists(), f"Missing {filename} in optimization/")
    
    def test_validation_scripts_exist(self):
        """Test validation scripts exist in new location"""
        validation_dir = project_root / "scripts" / "validation"
        
        expected_files = [
            "easy_domain_validation.py",
            "validate_token_optimization.py",
            "validate_new_tests.py"
        ]
        
        for filename in expected_files:
            file_path = validation_dir / filename
            self.assertTrue(file_path.exists(), f"Missing {filename} in validation/")
    
    def test_benchmarking_scripts_exist(self):
        """Test benchmarking scripts exist in new location"""
        benchmarking_dir = project_root / "scripts" / "benchmarking"
        
        expected_files = [
            "multi_model_benchmarking.py",
            "enhanced_cognitive_validation.py",
            "comprehensive_easy_domain_testing.py"
        ]
        
        for filename in expected_files:
            file_path = benchmarking_dir / filename
            self.assertTrue(file_path.exists(), f"Missing {filename} in benchmarking/")
    
    def test_conversion_scripts_exist(self):
        """Test conversion scripts exist in new location"""
        conversion_dir = project_root / "scripts" / "conversion"
        
        expected_files = [
            "convert_base_to_instruct_creativity.py",
            "convert_core_domains_to_instruct.py"
        ]
        
        for filename in expected_files:
            file_path = conversion_dir / filename
            self.assertTrue(file_path.exists(), f"Missing {filename} in conversion/")
    
    def test_script_categories_complete(self):
        """Test that no scripts were left in main scripts directory"""
        scripts_dir = project_root / "scripts"
        
        # Get all python files in main scripts directory (excluding __init__.py)
        python_files = [f for f in scripts_dir.glob("*.py") if f.name != "__init__.py"]
        
        # Should be empty except for __init__.py
        self.assertEqual(len(python_files), 0, 
                        f"Found unorganized scripts: {[f.name for f in python_files]}")
    
    def test_script_directory_structure(self):
        """Test that all expected subdirectories exist"""
        scripts_dir = project_root / "scripts"
        
        expected_subdirs = [
            "calibration",
            "optimization", 
            "validation",
            "benchmarking",
            "conversion"
        ]
        
        for subdir in expected_subdirs:
            subdir_path = scripts_dir / subdir
            self.assertTrue(subdir_path.exists(), f"Missing subdirectory: {subdir}")
            self.assertTrue(subdir_path.is_dir(), f"{subdir} is not a directory")
    
    def test_init_file_exists(self):
        """Test that __init__.py exists in scripts directory"""
        init_file = project_root / "scripts" / "__init__.py"
        self.assertTrue(init_file.exists(), "Missing __init__.py in scripts/")

if __name__ == '__main__':
    unittest.main()