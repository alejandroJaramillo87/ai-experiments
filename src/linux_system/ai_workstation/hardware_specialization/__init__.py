"""
Hardware Specialization Module

Provides specialized monitoring and intelligence for specific hardware components
in AI workstation environments, including RTX 5090 Blackwell architecture,
AMD Zen 5 processors, and advanced thermal management systems.
"""

from .rtx5090_blackwell_detector import RTX5090BlackwallDetector
from .amd_zen5_detector import AMDZen5WorkloadDetector  
from .thermal_intelligence_detector import ThermalIntelligenceDetector

__all__ = [
    'RTX5090BlackwallDetector',
    'AMDZen5WorkloadDetector',
    'ThermalIntelligenceDetector'
]