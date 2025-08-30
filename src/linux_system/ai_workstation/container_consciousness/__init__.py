"""
Container Consciousness Module

Provides intelligent monitoring and analysis of AI service containers,
including Docker orchestration awareness, service lifecycle tracking,
and container resource correlation.
"""

from .ai_container_detector import AIContainerOrchestratorDetector
from .service_lifecycle_extractor import AIServiceLifecycleExtractor
from .container_correlator import ContainerResourceCorrelator

__all__ = [
    'AIContainerOrchestratorDetector',
    'AIServiceLifecycleExtractor', 
    'ContainerResourceCorrelator'
]