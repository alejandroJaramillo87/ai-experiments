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
    
    def test_core_calibration_modules_exist(self):
        """Test calibration modules moved to core"""
        core_dir = project_root / "core"
        
        expected_files = [
            "calibration_engine.py",  # was systematic_base_calibration.py
            "production_calibration.py"  # was production_calibration_framework.py
        ]
        
        for filename in expected_files:
            file_path = core_dir / filename
            self.assertTrue(file_path.exists(), f"Missing {filename} in core/")
    
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
    
    def test_core_benchmarking_modules_exist(self):
        """Test benchmarking modules moved to core"""
        core_dir = project_root / "core"
        
        expected_files = [
            "benchmarking_engine.py",  # was multi_model_benchmarking.py
            "cognitive_validation.py"  # was enhanced_cognitive_validation.py
        ]
        
        for filename in expected_files:
            file_path = core_dir / filename
            self.assertTrue(file_path.exists(), f"Missing {filename} in core/")
    
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
        
        # Get all python files in main scripts directory (excluding __init__.py and approved utility scripts)
        approved_scripts = {"__init__.py", "test_mode_runner.py"}  # test_mode_runner.py is a utility script
        python_files = [f for f in scripts_dir.glob("*.py") if f.name not in approved_scripts]
        
        # Should be empty except for approved scripts
        self.assertEqual(len(python_files), 0, 
                        f"Found unorganized scripts: {[f.name for f in python_files]}")
    
    def test_script_directory_structure(self):
        """Test that remaining script subdirectories exist"""
        scripts_dir = project_root / "scripts"
        
        # After moving calibration and benchmarking to core, only these remain
        expected_subdirs = [
            "optimization", 
            "validation",
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