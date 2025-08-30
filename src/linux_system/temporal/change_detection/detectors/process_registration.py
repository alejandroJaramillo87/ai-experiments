"""
Process Detector Registration
============================

Registration and configuration for ProcessChangeDetector.
"""

from ..registry import ChangeDetectorRegistry
from .process_detector import ProcessChangeDetector
from ...config import ProcessDetectorConfig


def register_process_detector(registry: ChangeDetectorRegistry, 
                            config: ProcessDetectorConfig = None) -> None:
    """Register ProcessChangeDetector with the registry."""
    detector_config = config or ProcessDetectorConfig()
    detector = ProcessChangeDetector(detector_config)
    
    registry.register_detector("processes", detector)


# Auto-register when module is imported
def auto_register(registry: ChangeDetectorRegistry) -> None:
    """Auto-register process detector with default configuration."""
    register_process_detector(registry)