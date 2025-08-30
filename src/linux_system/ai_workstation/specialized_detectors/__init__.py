"""
AI Workstation Specialized Detectors - Advanced Intelligence Suite
==================================================================

Comprehensive suite of specialized detectors for production AI workstation monitoring
and optimization. Each detector provides deep intelligence for specific aspects of
the AI workstation ecosystem with advanced analytics, predictive capabilities, and
optimization recommendations.

Specialized Detectors:
- AIContainerOrchestratorDetector: Docker service lifecycle and AI container intelligence
- RTX5090BlackwellDetector: NVIDIA RTX 5090 Blackwell architecture deep monitoring
- AMDZen5WorkloadDetector: AMD 9950X Zen 5 architecture and workload optimization
- AIModelLifecycleDetector: AI model loading, performance, and lifecycle management
- ThermalIntelligenceDetector: Comprehensive thermal monitoring and predictive analysis

Each detector provides:
- Real-time monitoring with historical trend analysis
- Predictive analytics and optimization recommendations
- Change detection for temporal intelligence integration
- Performance correlation analysis across system components
- Specialized alerts and threshold management for AI workstation optimization

This specialized detector suite transforms generic system monitoring into
AI workstation-specific intelligence for optimal performance and reliability.
"""

from .ai_container_orchestrator import (
    AIContainerOrchestratorDetector,
    ContainerMetrics,
    ServiceDependency,
    ModelLoadEvent,
    ContainerResourcePattern
)

from .rtx5090_blackwell_detector import (
    RTX5090BlackwellDetector,
    BlackwellMetrics,
    TensorCoreMetrics,
    CUDAKernelProfile,
    MemoryBandwidthAnalysis,
    ThermalIntelligence
)

from .amd_zen5_workload_detector import (
    AMDZen5WorkloadDetector,
    Zen5CoreMetrics,
    AOCLPerformanceMetrics,
    MemoryBandwidthAnalysis as Zen5MemoryBandwidthAnalysis,
    WorkloadCorrelation,
    CorePinningAnalysis
)

from .ai_model_lifecycle_detector import (
    AIModelLifecycleDetector,
    ModelMetadata,
    ModelInstance,
    ModelPerformanceProfile,
    ModelLoadEvent as ModelLifecycleEvent
)

from .thermal_intelligence_detector import (
    ThermalIntelligenceDetector,
    ThermalZone,
    CoolingSystem,
    ThermalPrediction,
    WorkloadThermalCorrelation
)

# Detector registry for dynamic discovery and integration
SPECIALIZED_DETECTORS = {
    'ai_container_orchestrator': AIContainerOrchestratorDetector,
    'rtx5090_blackwell': RTX5090BlackwellDetector,
    'amd_zen5_workload': AMDZen5WorkloadDetector,
    'ai_model_lifecycle': AIModelLifecycleDetector,
    'thermal_intelligence': ThermalIntelligenceDetector
}

# Detector categories for organizational purposes
DETECTOR_CATEGORIES = {
    'container_intelligence': ['ai_container_orchestrator'],
    'hardware_optimization': ['rtx5090_blackwell', 'amd_zen5_workload'],
    'ai_workload_management': ['ai_model_lifecycle'],
    'thermal_management': ['thermal_intelligence'],
    'all_specialized': list(SPECIALIZED_DETECTORS.keys())
}

# Integration priorities for system consciousness orchestration
DETECTOR_PRIORITIES = {
    'thermal_intelligence': 1,  # Highest priority - safety critical
    'rtx5090_blackwell': 2,     # High priority - expensive hardware
    'amd_zen5_workload': 3,     # High priority - CPU optimization
    'ai_container_orchestrator': 4,  # Medium priority - service management
    'ai_model_lifecycle': 5     # Medium priority - model optimization
}

# Detector dependencies and integration requirements
DETECTOR_DEPENDENCIES = {
    'ai_container_orchestrator': {
        'requires': ['docker', 'container_runtime'],
        'optional': ['nvidia_runtime'],
        'integrates_with': ['rtx5090_blackwell', 'amd_zen5_workload']
    },
    'rtx5090_blackwell': {
        'requires': ['nvidia_smi', 'nvidia_drivers'],
        'optional': ['nvidia_ml_py'],
        'integrates_with': ['thermal_intelligence', 'ai_container_orchestrator']
    },
    'amd_zen5_workload': {
        'requires': ['psutil', 'cpu_info'],
        'optional': ['aocl_libraries', 'numactl'],
        'integrates_with': ['thermal_intelligence', 'ai_container_orchestrator']
    },
    'ai_model_lifecycle': {
        'requires': ['requests', 'psutil'],
        'optional': ['model_apis'],
        'integrates_with': ['ai_container_orchestrator', 'rtx5090_blackwell']
    },
    'thermal_intelligence': {
        'requires': ['psutil'],
        'optional': ['sensors', 'hwmon'],
        'integrates_with': ['rtx5090_blackwell', 'amd_zen5_workload']
    }
}

# Default configuration templates for each detector
DEFAULT_DETECTOR_CONFIGS = {
    'ai_container_orchestrator': {
        'collection_interval': 30,  # seconds
        'service_health_checks': True,
        'resource_pattern_analysis': True,
        'model_event_detection': True
    },
    'rtx5090_blackwell': {
        'collection_interval': 15,  # seconds
        'thermal_prediction': True,
        'tensor_core_analysis': True,
        'memory_bandwidth_analysis': True,
        'performance_profiling': True
    },
    'amd_zen5_workload': {
        'collection_interval': 20,  # seconds
        'core_pinning_analysis': True,
        'aocl_performance_tracking': True,
        'numa_optimization': True,
        'workload_correlation': True
    },
    'ai_model_lifecycle': {
        'collection_interval': 45,  # seconds
        'model_discovery': True,
        'performance_analysis': True,
        'lifecycle_event_tracking': True,
        'cross_service_optimization': True
    },
    'thermal_intelligence': {
        'collection_interval': 10,  # seconds - more frequent for safety
        'predictive_analysis': True,
        'cooling_optimization': True,
        'workload_correlation': True,
        'emergency_response': True
    }
}


def get_detector_class(detector_name: str):
    """Get detector class by name."""
    return SPECIALIZED_DETECTORS.get(detector_name)


def get_detectors_by_category(category: str):
    """Get list of detector names by category."""
    return DETECTOR_CATEGORIES.get(category, [])


def create_detector_instance(detector_name: str, config: dict = None):
    """Create an instance of a specialized detector."""
    detector_class = get_detector_class(detector_name)
    if not detector_class:
        raise ValueError(f"Unknown detector: {detector_name}")
    
    # Merge with default configuration
    final_config = DEFAULT_DETECTOR_CONFIGS.get(detector_name, {}).copy()
    if config:
        final_config.update(config)
    
    return detector_class(final_config)


def get_detector_dependencies(detector_name: str):
    """Get dependencies for a specific detector."""
    return DETECTOR_DEPENDENCIES.get(detector_name, {})


def get_integration_priority(detector_name: str):
    """Get integration priority for a detector (lower number = higher priority)."""
    return DETECTOR_PRIORITIES.get(detector_name, 999)


def validate_detector_requirements(detector_name: str):
    """Validate that detector requirements are met."""
    deps = get_detector_dependencies(detector_name)
    required = deps.get('requires', [])
    
    # Basic validation (could be extended with actual dependency checking)
    missing_requirements = []
    
    for requirement in required:
        if requirement == 'docker':
            try:
                import docker
            except ImportError:
                missing_requirements.append('docker python package')
        elif requirement == 'nvidia_smi':
            import shutil
            if not shutil.which('nvidia-smi'):
                missing_requirements.append('nvidia-smi command')
        elif requirement == 'psutil':
            try:
                import psutil
            except ImportError:
                missing_requirements.append('psutil python package')
        elif requirement == 'requests':
            try:
                import requests
            except ImportError:
                missing_requirements.append('requests python package')
    
    return {
        'valid': len(missing_requirements) == 0,
        'missing_requirements': missing_requirements
    }


class SpecializedDetectorOrchestrator:
    """
    Orchestrator for managing specialized AI workstation detectors.
    
    Provides centralized management, configuration, and coordination
    of all specialized detectors with dependency management and
    integration optimization.
    """
    
    def __init__(self, enabled_detectors: list = None, global_config: dict = None):
        """
        Initialize detector orchestrator.
        
        Args:
            enabled_detectors: List of detector names to enable (default: all)
            global_config: Global configuration applied to all detectors
        """
        self.enabled_detectors = enabled_detectors or list(SPECIALIZED_DETECTORS.keys())
        self.global_config = global_config or {}
        self.detector_instances = {}
        self.last_collection_time = {}
        self.integration_map = {}
        
        self._initialize_detectors()
        self._build_integration_map()
    
    def _initialize_detectors(self):
        """Initialize enabled detector instances."""
        # Sort by priority for initialization order
        sorted_detectors = sorted(
            self.enabled_detectors,
            key=lambda x: get_integration_priority(x)
        )
        
        for detector_name in sorted_detectors:
            try:
                # Validate requirements
                validation = validate_detector_requirements(detector_name)
                if not validation['valid']:
                    print(f"Warning: {detector_name} requirements not met: {validation['missing_requirements']}")
                    continue
                
                # Create detector instance
                detector_config = self.global_config.copy()
                detector_config.update(DEFAULT_DETECTOR_CONFIGS.get(detector_name, {}))
                
                self.detector_instances[detector_name] = create_detector_instance(
                    detector_name, detector_config
                )
                
                print(f"Initialized specialized detector: {detector_name}")
                
            except Exception as e:
                print(f"Failed to initialize detector {detector_name}: {e}")
    
    def _build_integration_map(self):
        """Build integration map between detectors."""
        for detector_name in self.detector_instances.keys():
            deps = get_detector_dependencies(detector_name)
            integrates_with = deps.get('integrates_with', [])
            
            self.integration_map[detector_name] = [
                dep for dep in integrates_with 
                if dep in self.detector_instances
            ]
    
    async def collect_all_specialized_data(self):
        """Collect data from all enabled specialized detectors."""
        specialized_data = {}
        
        for detector_name, detector_instance in self.detector_instances.items():
            try:
                if hasattr(detector_instance, 'collect_container_metrics'):
                    data = await detector_instance.collect_container_metrics()
                elif hasattr(detector_instance, 'collect_blackwall_metrics'):
                    data = await detector_instance.collect_blackwall_metrics()
                elif hasattr(detector_instance, 'collect_zen5_metrics'):
                    data = await detector_instance.collect_zen5_metrics()
                elif hasattr(detector_instance, 'collect_model_lifecycle_metrics'):
                    data = await detector_instance.collect_model_lifecycle_metrics()
                elif hasattr(detector_instance, 'collect_thermal_intelligence'):
                    data = await detector_instance.collect_thermal_intelligence()
                else:
                    # Generic collection method
                    data = await detector_instance.collect()
                
                specialized_data[detector_name] = data
                
            except Exception as e:
                print(f"Error collecting from {detector_name}: {e}")
                specialized_data[detector_name] = {'error': str(e)}
        
        return specialized_data
    
    def get_cross_detector_correlations(self, data: dict):
        """Analyze correlations across different detectors."""
        correlations = {}
        
        # Example: GPU temperature vs CPU workload correlation
        if ('rtx5090_blackwell' in data and 'amd_zen5_workload' in data):
            gpu_data = data['rtx5090_blackwell']
            cpu_data = data['amd_zen5_workload']
            
            if ('basic_metrics' in gpu_data and 
                'cpu_core_metrics' in cpu_data and
                'aggregate_metrics' in cpu_data['cpu_core_metrics']):
                
                gpu_temp = gpu_data['basic_metrics'].get('temperature_c', 0)
                cpu_util = cpu_data['cpu_core_metrics']['aggregate_metrics'].get('total_utilization', 0)
                
                correlations['gpu_cpu_thermal_workload'] = {
                    'gpu_temperature_c': gpu_temp,
                    'cpu_utilization_percent': cpu_util,
                    'correlation_strength': 'analyzing',
                    'optimization_opportunity': gpu_temp > 80 and cpu_util < 50
                }
        
        return correlations
    
    def get_enabled_detectors(self):
        """Get list of enabled detector names."""
        return list(self.detector_instances.keys())
    
    def get_detector_status(self):
        """Get status of all detectors."""
        status = {}
        for detector_name, detector_instance in self.detector_instances.items():
            status[detector_name] = {
                'enabled': True,
                'class': detector_instance.__class__.__name__,
                'integration_priority': get_integration_priority(detector_name),
                'integrates_with': self.integration_map.get(detector_name, [])
            }
        
        # Add disabled detectors
        for detector_name in SPECIALIZED_DETECTORS.keys():
            if detector_name not in self.detector_instances:
                validation = validate_detector_requirements(detector_name)
                status[detector_name] = {
                    'enabled': False,
                    'reason': 'requirements_not_met' if not validation['valid'] else 'not_enabled',
                    'missing_requirements': validation.get('missing_requirements', [])
                }
        
        return status


# Export all classes and functions
__all__ = [
    # Detector classes
    'AIContainerOrchestratorDetector',
    'RTX5090BlackwellDetector', 
    'AMDZen5WorkloadDetector',
    'AIModelLifecycleDetector',
    'ThermalIntelligenceDetector',
    
    # Data classes
    'ContainerMetrics',
    'ServiceDependency',
    'ModelLoadEvent',
    'ContainerResourcePattern',
    'BlackwellMetrics',
    'TensorCoreMetrics',
    'CUDAKernelProfile',
    'MemoryBandwidthAnalysis',
    'ThermalIntelligence',
    'Zen5CoreMetrics',
    'AOCLPerformanceMetrics',
    'WorkloadCorrelation',
    'CorePinningAnalysis',
    'ModelMetadata',
    'ModelInstance',
    'ModelPerformanceProfile',
    'ThermalZone',
    'CoolingSystem',
    'ThermalPrediction',
    'WorkloadThermalCorrelation',
    
    # Registry and configuration
    'SPECIALIZED_DETECTORS',
    'DETECTOR_CATEGORIES',
    'DETECTOR_PRIORITIES',
    'DETECTOR_DEPENDENCIES',
    'DEFAULT_DETECTOR_CONFIGS',
    
    # Utility functions
    'get_detector_class',
    'get_detectors_by_category',
    'create_detector_instance',
    'get_detector_dependencies',
    'get_integration_priority',
    'validate_detector_requirements',
    
    # Orchestrator
    'SpecializedDetectorOrchestrator'
]