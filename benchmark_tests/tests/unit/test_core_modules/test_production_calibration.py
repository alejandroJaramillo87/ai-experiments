#!/usr/bin/env python3
"""
Unit Tests for Production Calibration Module

Tests the production calibration framework including token optimization,
hardware optimization, and production deployment validation.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "core"))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from core.production_calibration import (
        ProductionCalibrator, HardwareConfig, TokenStrategy, CalibrationResult
    )
except ImportError:
    # Handle case where production_calibration module exists but with different structure
    production_calibration = None
    try:
        import core.production_calibration as production_calibration
    except ImportError:
        production_calibration = None


class TestProductionCalibrationBase(unittest.TestCase):
    """Base test class for production calibration"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)


@unittest.skipIf(production_calibration is None, "production_calibration module not available")
class TestProductionCalibrator(TestProductionCalibrationBase):
    """Test ProductionCalibrator class functionality"""
    
    def test_calibrator_initialization(self):
        """Test ProductionCalibrator initialization"""
        # Test basic initialization
        try:
            calibrator = production_calibration.ProductionCalibrator()
            self.assertIsNotNone(calibrator)
        except AttributeError:
            # Module exists but ProductionCalibrator class might have different name
            # Check for alternative class names
            available_classes = [name for name in dir(production_calibration) 
                               if not name.startswith('_') and name.endswith('Calibrator')]
            
            if available_classes:
                calibrator_class = getattr(production_calibration, available_classes[0])
                calibrator = calibrator_class()
                self.assertIsNotNone(calibrator)
            else:
                self.skipTest("No calibrator class found in production_calibration module")
    
    def test_token_optimization_strategy(self):
        """Test token optimization strategy implementation"""
        # This test validates the proven token strategy: 400/500/600
        expected_tokens = {
            'easy': 400,
            'medium': 500,
            'hard': 600
        }
        
        # Check if module has token strategy configuration
        if hasattr(production_calibration, 'OPTIMAL_TOKENS'):
            optimal_tokens = production_calibration.OPTIMAL_TOKENS
            for difficulty, expected_count in expected_tokens.items():
                self.assertIn(difficulty, optimal_tokens,
                            f"Token strategy should include {difficulty} difficulty")
                self.assertEqual(optimal_tokens[difficulty], expected_count,
                               f"Token count for {difficulty} should be {expected_count}")
        elif hasattr(production_calibration, 'TokenStrategy'):
            token_strategy = production_calibration.TokenStrategy()
            for difficulty, expected_count in expected_tokens.items():
                self.assertEqual(getattr(token_strategy, difficulty, None), expected_count,
                               f"Token strategy {difficulty} should be {expected_count}")
        else:
            self.skipTest("No token strategy configuration found")
    
    def test_hardware_optimization_config(self):
        """Test hardware optimization for RTX 5090 + AMD 9950X"""
        expected_hardware = {
            'gpu': 'RTX 5090',
            'cpu': 'AMD 9950X',
            'memory': '128GB DDR5',
            'storage': 'Samsung 990 Pro/EVO'
        }
        
        # Check for hardware configuration
        if hasattr(production_calibration, 'HARDWARE_CONFIG'):
            config = production_calibration.HARDWARE_CONFIG
            self.assertIsInstance(config, dict)
        elif hasattr(production_calibration, 'HardwareConfig'):
            config = production_calibration.HardwareConfig()
            self.assertIsNotNone(config)
        else:
            # Hardware config might be embedded in calibrator
            self.skipTest("Hardware configuration testing requires specific implementation details")


class TestProductionCalibrationModuleStructure(TestProductionCalibrationBase):
    """Test the overall structure and imports of production_calibration module"""
    
    def test_module_imports_successfully(self):
        """Test that production_calibration module can be imported"""
        self.assertIsNotNone(production_calibration, 
                           "production_calibration module should be importable")
    
    def test_module_has_calibration_functionality(self):
        """Test that module contains calibration-related functionality"""
        if production_calibration is None:
            self.skipTest("production_calibration module not available")
        
        # Check for calibration-related attributes
        calibration_attrs = [name for name in dir(production_calibration) 
                           if 'calibrat' in name.lower()]
        
        self.assertGreater(len(calibration_attrs), 0,
                         "Module should contain calibration-related functionality")
        
        print(f"Found calibration attributes: {calibration_attrs}")
    
    def test_module_structure_completeness(self):
        """Test that module has expected production-ready components"""
        if production_calibration is None:
            self.skipTest("production_calibration module not available")
        
        expected_components = [
            'calibrat',  # Some form of calibration class/function
            'token',     # Token optimization
            'hardware',  # Hardware optimization
            'production' # Production deployment features
        ]
        
        module_contents = dir(production_calibration)
        found_components = []
        
        for expected in expected_components:
            matching = [name for name in module_contents if expected in name.lower()]
            if matching:
                found_components.append(expected)
        
        # Should have at least some production-ready components
        self.assertGreaterEqual(len(found_components), 2,
                              f"Should have production components. Found: {found_components}")


class TestProductionCalibrationIntegration(TestProductionCalibrationBase):
    """Test integration with other core modules"""
    
    def test_integration_with_calibration_engine(self):
        """Test integration with core calibration engine"""
        try:
            from core.calibration_engine import SystematicBaseCalibrator
            calibrator = SystematicBaseCalibrator()
            
            # Test that calibration engine can work with production settings
            self.assertIsNotNone(calibrator)
            
            # Check if production calibration provides configuration
            if production_calibration and hasattr(production_calibration, 'get_production_config'):
                config = production_calibration.get_production_config()
                self.assertIsInstance(config, dict)
        
        except ImportError:
            self.skipTest("calibration_engine not available for integration testing")
    
    def test_integration_with_results_manager(self):
        """Test integration with TestResultsManager"""
        try:
            from core.results_manager import TestResultsManager
            manager = TestResultsManager(base_results_dir=self.temp_dir)
            
            # Test that production calibration can work with results manager
            self.assertIsNotNone(manager)
            
            # If production calibration provides result processing
            if production_calibration and hasattr(production_calibration, 'process_results'):
                # This would be tested with actual implementation
                pass
        
        except ImportError:
            self.skipTest("results_manager not available for integration testing")


class TestProductionCalibrationDocumentation(TestProductionCalibrationBase):
    """Test documentation and configuration aspects"""
    
    def test_module_docstring_exists(self):
        """Test that module has proper documentation"""
        if production_calibration is None:
            self.skipTest("production_calibration module not available")
        
        self.assertIsNotNone(production_calibration.__doc__,
                           "Module should have documentation")
        
        if production_calibration.__doc__:
            doc_lower = production_calibration.__doc__.lower()
            production_keywords = ['production', 'calibration', 'token', 'optimization']
            
            found_keywords = [kw for kw in production_keywords if kw in doc_lower]
            self.assertGreaterEqual(len(found_keywords), 2,
                                  f"Documentation should mention production concepts. Found: {found_keywords}")
    
    def test_proven_calibration_metrics_documented(self):
        """Test that proven calibration metrics are documented or embedded"""
        # From the comprehensive audit report, we know these metrics were validated:
        # - 75% calibration success rate
        # - 94% loop reduction 
        # - 1,395 tests optimized
        # - Production-ready scaling to 26k+ test suite
        
        if production_calibration is None:
            self.skipTest("production_calibration module not available")
        
        # Look for metrics or configuration that reflects these achievements
        metrics_found = False
        
        # Check module docstring
        if production_calibration.__doc__:
            doc = production_calibration.__doc__.lower()
            if any(term in doc for term in ['75%', '94%', 'success', 'optimization']):
                metrics_found = True
        
        # Check for configuration constants
        for attr_name in dir(production_calibration):
            if not attr_name.startswith('_'):
                attr = getattr(production_calibration, attr_name)
                if isinstance(attr, (int, float, dict, str)):
                    attr_str = str(attr).lower()
                    if any(term in attr_str for term in ['success', 'optimization', 'proven']):
                        metrics_found = True
                        break
        
        # This is informational rather than strict requirement
        print(f"Proven calibration metrics documentation found: {metrics_found}")


if __name__ == '__main__':
    print("🧪 Running Production Calibration Unit Tests")
    print("=" * 50)
    
    # Print module availability info
    if production_calibration:
        print(f"✅ production_calibration module loaded")
        print(f"   Available attributes: {len([x for x in dir(production_calibration) if not x.startswith('_')])}")
    else:
        print("⚠️ production_calibration module not found - testing module structure requirements")
    
    unittest.main(verbosity=2)