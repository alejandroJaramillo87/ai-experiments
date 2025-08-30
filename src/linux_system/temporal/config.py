"""
Configuration Management for Temporal System Intelligence
=========================================================

Centralized configuration for all temporal system components with validation,
defaults, and environment-specific tuning for RTX 5090 + AMD 9950X setup.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class ChangeDetectorConfig:
    """Base configuration for change detectors."""
    # Significance thresholds
    min_significance_threshold: float = 0.1  # Ignore changes below this
    high_significance_threshold: float = 0.7  # Mark as important above this
    
    # Change filtering
    ignore_patterns: List[str] = field(default_factory=lambda: [
        'process:pid:*',  # PIDs change constantly, not meaningful
        'timestamp:*',    # Timestamps always change
    ])
    
    # Performance tuning
    max_changes_per_category: int = 100  # Limit to prevent overwhelming
    enable_parallel_detection: bool = True


@dataclass  
class GPUDetectorConfig(ChangeDetectorConfig):
    """RTX 5090-specific GPU change detection configuration."""
    # Thermal thresholds (°C) - RTX 5090 specific
    thermal_warning_threshold: int = 80
    thermal_critical_threshold: int = 85
    thermal_throttling_threshold: int = 90
    rapid_temp_change_threshold: float = 5.0  # °C per collection interval
    
    # Memory thresholds (MB)
    memory_significant_change_mb: int = 512  # 512MB changes are significant
    memory_critical_usage_percent: float = 95.0  # Above 95% is critical
    
    # Power thresholds (Watts) - RTX 5090 has high power draw
    power_significant_change_w: int = 50  # 50W changes matter
    power_critical_threshold_w: int = 400  # Getting close to limit
    
    # Process detection
    track_gpu_process_spawning: bool = True
    track_gpu_memory_per_process: bool = True
    
    # Performance optimization
    enable_nvidia_ml_monitoring: bool = True  # Use nvidia-ml-py if available


@dataclass
class ProcessDetectorConfig(ChangeDetectorConfig):
    """Process change detection configuration.""" 
    # Resource thresholds
    memory_significant_change_mb: int = 100  # 100MB RSS change is significant
    cpu_significant_change_percent: float = 10.0  # 10% CPU change matters
    
    # Process lifecycle
    track_process_spawning: bool = True
    track_process_termination: bool = True
    track_parent_child_relationships: bool = True
    
    # Resource monitoring
    track_memory_usage: bool = True
    track_cpu_usage: bool = True
    track_io_usage: bool = True
    
    # Filtering
    ignore_system_processes: bool = True  # Filter out kernel threads, etc.
    ignore_short_lived_processes: bool = True  # Processes < 1 second


@dataclass
class PythonEnvDetectorConfig(ChangeDetectorConfig):
    """Python environment change detection configuration."""
    # Package tracking
    track_pip_packages: bool = True
    track_conda_packages: bool = True
    track_poetry_packages: bool = True
    
    # Environment tracking  
    track_virtual_environments: bool = True
    track_python_versions: bool = True
    
    # AI/ML specific
    track_cuda_availability: bool = True
    track_framework_versions: bool = True  # PyTorch, TensorFlow, etc.
    
    # Package filtering
    ignore_development_packages: bool = False  # Track dev dependencies
    significant_version_changes: List[str] = field(default_factory=lambda: [
        'torch', 'tensorflow', 'transformers', 'cuda', 'nvidia'
    ])


@dataclass
class EventExtractorConfig:
    """Configuration for event extraction from changes."""
    # Confidence thresholds
    min_event_confidence: float = 0.6  # Don't create low-confidence events
    high_confidence_threshold: float = 0.8  # Mark as high-confidence above this
    
    # Event creation
    enable_predictive_effects: bool = True  # Predict what events might cause
    enable_context_enrichment: bool = True  # Add rich context to events
    
    # Correlation settings
    max_events_per_correlation: int = 10  # Limit correlation complexity
    correlation_time_window_seconds: int = 300  # 5 minutes for finding correlations


@dataclass
class ThermalEventConfig:
    """Thermal event extraction configuration."""
    # RTX 5090 thermal characteristics
    idle_temperature_max: int = 35  # Above this when idle is concerning
    load_temperature_normal: int = 75  # Normal under load
    throttling_imminent_temp: int = 83  # Throttling likely soon
    
    # Event generation
    generate_trend_events: bool = True  # Create events for temperature trends
    trend_analysis_window_minutes: int = 10  # Analyze trends over 10 minutes


@dataclass
class TemporalStorageConfig:
    """Configuration for temporal storage and memory management."""
    # Memory hierarchy
    recent_buffer_capacity: int = 2880  # 48 hours at 1-minute intervals
    daily_retention_days: int = 90  # Keep daily summaries for 90 days
    pattern_retention_months: int = 12  # Long-term patterns for 1 year
    
    # Storage thresholds
    compression_threshold_mb: int = 100  # Compress when buffer exceeds 100MB
    max_memory_usage_mb: int = 500  # Total memory limit for temporal storage
    
    # Indexing
    enable_search_indexing: bool = True
    index_categories: List[str] = field(default_factory=lambda: [
        'nvidia_gpu', 'processes', 'python_env', 'memory', 'storage'
    ])
    
    # Persistence
    enable_disk_persistence: bool = True
    storage_directory: Optional[Path] = None  # Will default to ~/.ai-workstation/temporal


@dataclass
class CorrelationEngineConfig:
    """Configuration for event correlation and pattern detection."""
    # Temporal analysis
    temporal_window_seconds: int = 600  # 10 minutes for temporal correlations
    min_pattern_occurrences: int = 3  # Need 3+ occurrences to be a pattern
    
    # Causal inference
    enable_causal_inference: bool = True
    causal_confidence_threshold: float = 0.7
    max_causal_chain_length: int = 5  # Limit causal chain complexity
    
    # Pattern detection
    enable_pattern_detection: bool = True
    pattern_similarity_threshold: float = 0.8  # How similar to be a pattern match
    
    # Anomaly detection
    enable_anomaly_detection: bool = True
    anomaly_sensitivity: float = 0.8  # Higher = more sensitive


@dataclass 
class TemporalSystemConfig:
    """Main configuration for the entire temporal system."""
    # Component configurations
    change_detection: ChangeDetectorConfig = field(default_factory=ChangeDetectorConfig)
    gpu_detection: GPUDetectorConfig = field(default_factory=GPUDetectorConfig)
    process_detection: ProcessDetectorConfig = field(default_factory=ProcessDetectorConfig)
    python_env_detection: PythonEnvDetectorConfig = field(default_factory=PythonEnvDetectorConfig)
    event_extraction: EventExtractorConfig = field(default_factory=EventExtractorConfig)
    thermal_events: ThermalEventConfig = field(default_factory=ThermalEventConfig)
    storage: TemporalStorageConfig = field(default_factory=TemporalStorageConfig)
    correlation: CorrelationEngineConfig = field(default_factory=CorrelationEngineConfig)
    
    # Global settings
    collection_interval_seconds: int = 60  # How often to collect
    enable_debug_logging: bool = False
    enable_performance_monitoring: bool = True
    
    # Safety settings (inherited from BaseCollector philosophy)
    max_processing_time_seconds: int = 30  # Timeout for temporal processing
    enable_graceful_degradation: bool = True  # Continue with errors
    max_error_rate_percent: float = 10.0  # Shut down if > 10% operations fail
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate thresholds are in valid ranges
        if not 0.0 <= self.change_detection.min_significance_threshold <= 1.0:
            issues.append("min_significance_threshold must be between 0.0 and 1.0")
            
        if self.gpu_detection.thermal_warning_threshold >= self.gpu_detection.thermal_critical_threshold:
            issues.append("thermal_warning_threshold must be less than thermal_critical_threshold")
            
        if self.storage.recent_buffer_capacity <= 0:
            issues.append("recent_buffer_capacity must be positive")
            
        if self.collection_interval_seconds <= 0:
            issues.append("collection_interval_seconds must be positive")
            
        # Validate storage directory if specified
        if self.storage.storage_directory and not self.storage.storage_directory.parent.exists():
            issues.append(f"Storage directory parent does not exist: {self.storage.storage_directory.parent}")
        
        return issues
    
    @classmethod
    def create_default(cls) -> 'TemporalSystemConfig':
        """Create default configuration optimized for RTX 5090 + AMD 9950X."""
        config = cls()
        
        # Set default storage directory
        config.storage.storage_directory = Path.home() / '.ai-workstation' / 'temporal'
        
        # Optimize for high-end hardware
        config.change_detection.enable_parallel_detection = True
        config.storage.max_memory_usage_mb = 500  # Can afford more memory
        config.correlation.enable_causal_inference = True  # Use full capabilities
        
        return config
    
    @classmethod
    def create_development(cls) -> 'TemporalSystemConfig':
        """Create development configuration with more debugging."""
        config = cls.create_default()
        
        # Development-friendly settings
        config.enable_debug_logging = True
        config.collection_interval_seconds = 30  # More frequent for testing
        config.storage.recent_buffer_capacity = 720  # 12 hours for development
        
        # More sensitive detection for testing
        config.change_detection.min_significance_threshold = 0.05
        config.gpu_detection.rapid_temp_change_threshold = 3.0
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'change_detection': self.change_detection.__dict__,
            'gpu_detection': self.gpu_detection.__dict__,
            'process_detection': self.process_detection.__dict__,
            'python_env_detection': self.python_env_detection.__dict__,
            'event_extraction': self.event_extraction.__dict__,
            'thermal_events': self.thermal_events.__dict__,
            'storage': {
                **self.storage.__dict__,
                'storage_directory': str(self.storage.storage_directory) if self.storage.storage_directory else None
            },
            'correlation': self.correlation.__dict__,
            'collection_interval_seconds': self.collection_interval_seconds,
            'enable_debug_logging': self.enable_debug_logging,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'max_processing_time_seconds': self.max_processing_time_seconds,
            'enable_graceful_degradation': self.enable_graceful_degradation,
            'max_error_rate_percent': self.max_error_rate_percent
        }