"""
Python Environment Detector Registration
========================================

Registration and configuration for PythonEnvChangeDetector.
"""

from ..registry import ChangeDetectorRegistry
from .python_env_detector import PythonEnvChangeDetector
from ...config import PythonEnvDetectorConfig


def register_python_env_detector(registry: ChangeDetectorRegistry, 
                                config: PythonEnvDetectorConfig = None) -> None:
    """Register PythonEnvChangeDetector with the registry."""
    detector_config = config or PythonEnvDetectorConfig()
    detector = PythonEnvChangeDetector(detector_config)
    
    registry.register_detector("python_env", detector)


# Auto-register when module is imported
def auto_register(registry: ChangeDetectorRegistry) -> None:
    """Auto-register Python environment detector with default configuration."""
    register_python_env_detector(registry)