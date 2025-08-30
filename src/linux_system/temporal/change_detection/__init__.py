"""
Change Detection Framework
==========================

Pluggable framework for detecting meaningful changes between system snapshots.
Each category of system data (GPU, processes, Python env, etc.) has its own
specialized detector that understands the domain-specific semantics.

Core Components:
- BaseChangeDetector: Abstract base for all change detectors
- SystemChangeDetector: Main orchestrator managing all detectors
- SignificanceCalculator: Determines importance of detected changes
- ChangeRegistry: Plugin system for registering new detectors
"""

from .base_detector import BaseChangeDetector, ChangeDetectorError
from .system_detector import SystemChangeDetector
from .significance import SignificanceCalculator
from .registry import ChangeDetectorRegistry

__all__ = [
    'BaseChangeDetector',
    'ChangeDetectorError', 
    'SystemChangeDetector',
    'SignificanceCalculator',
    'ChangeDetectorRegistry'
]