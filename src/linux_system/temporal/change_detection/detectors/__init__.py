"""
Specialized Change Detectors
============================

Domain-specific change detectors for different system categories.
Each detector understands the unique characteristics and semantics
of its category for accurate change detection.

Available Detectors:
- GPUChangeDetector: RTX 5090-optimized GPU monitoring
- ProcessChangeDetector: Process lifecycle and resource tracking
- PythonEnvChangeDetector: AI/ML environment monitoring
- MemoryChangeDetector: System memory analysis
- StorageChangeDetector: Storage and I/O monitoring
- SecurityChangeDetector: Security posture monitoring
- NetworkChangeDetector: Network interface and traffic analysis
"""

from .gpu_detector import GPUChangeDetector
from .gpu_registration import register_gpu_detector
from .process_detector import ProcessChangeDetector
from .process_registration import register_process_detector
from .python_env_detector import PythonEnvChangeDetector
from .python_env_registration import register_python_env_detector
from .memory_detector import MemoryChangeDetector
from .storage_detector import StorageChangeDetector
from .security_detector import SecurityChangeDetector
from .network_detector import NetworkChangeDetector

__all__ = [
    'GPUChangeDetector',
    'register_gpu_detector',
    'ProcessChangeDetector',
    'register_process_detector',
    'PythonEnvChangeDetector',
    'register_python_env_detector',
    'MemoryChangeDetector',
    'StorageChangeDetector',
    'SecurityChangeDetector',
    'NetworkChangeDetector'
]