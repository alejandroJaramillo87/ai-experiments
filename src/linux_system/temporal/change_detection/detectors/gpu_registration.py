"""
GPU Detector Registration
=========================

Registers the RTX 5090 GPU detector with the global registry
and provides configuration defaults.
"""

from ..registry import register_detector
from ...config import GPUDetectorConfig
from .gpu_detector import GPUChangeDetector


def register_gpu_detector():
    """Register the GPU change detector with default RTX 5090 configuration."""
    
    # Create RTX 5090-optimized configuration
    gpu_config = GPUDetectorConfig(
        # RTX 5090 thermal thresholds
        thermal_warning_threshold=80,
        thermal_critical_threshold=85,
        thermal_throttling_threshold=88,
        rapid_temp_change_threshold=5.0,
        
        # 32GB memory thresholds
        memory_significant_change_mb=512,
        memory_critical_usage_percent=95.0,
        
        # High power RTX 5090 settings
        power_significant_change_w=50,
        power_critical_threshold_w=400,
        
        # Process tracking enabled
        track_gpu_process_spawning=True,
        track_gpu_memory_per_process=True,
        
        # Performance optimizations
        enable_nvidia_ml_monitoring=True,
        enable_parallel_detection=True
    )
    
    # Register with the global registry
    register_detector("nvidia_gpu", GPUChangeDetector, gpu_config)


# Auto-register when module is imported
register_gpu_detector()