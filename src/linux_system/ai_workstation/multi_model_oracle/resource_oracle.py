"""
Multi-Model Resource Oracle

Intelligent resource optimization engine that integrates container consciousness,
hardware specialization, and thermal intelligence to make optimal resource
allocation decisions for concurrent AI model execution. Provides sophisticated
workload routing, resource contention resolution, and performance optimization.

Features:
- Intelligent resource allocation across CPU/GPU/memory domains
- Multi-model placement optimization for concurrent inference
- Dynamic load balancing based on real-time system state
- Resource contention detection and automated resolution
- Thermal-aware workload distribution and throttling prevention
- Performance prediction and proactive optimization recommendations
"""

import logging
import threading
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
import heapq

from ..container_consciousness.ai_container_detector import AIServiceState, ContainerHealthMetrics
from ..container_consciousness.container_correlator import ResourceCorrelationInsight
from ..hardware_specialization.rtx5090_blackwell_detector import BlackwellMetrics
from ..hardware_specialization.amd_zen5_detector import Zen5PerformanceCounters
from ..hardware_specialization.thermal_intelligence_detector import ThermalProfile, ComponentThermalState


logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of system resources."""
    CPU_CORES = "cpu_cores"
    SYSTEM_MEMORY = "system_memory"
    GPU_VRAM = "gpu_vram"
    GPU_COMPUTE = "gpu_compute"
    STORAGE_IO = "storage_io"
    NETWORK_BW = "network_bandwidth"
    THERMAL_CAPACITY = "thermal_capacity"


class OptimizationStrategy(Enum):
    """Resource optimization strategies."""
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_LATENCY = "minimize_latency"
    BALANCE_EFFICIENCY = "balance_efficiency"
    THERMAL_CONSERVATIVE = "thermal_conservative"
    POWER_EFFICIENT = "power_efficient"


class ServicePriority(Enum):
    """Service priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class ResourceCapacity:
    """Resource capacity and availability information."""
    resource_type: ResourceType
    total_capacity: float
    available_capacity: float
    reserved_capacity: float
    utilization_percent: float
    efficiency_score: float  # 0-1
    bottleneck_risk: float   # 0-1
    thermal_impact: float    # Thermal load from this resource


@dataclass
class ServiceResourceRequirement:
    """Resource requirements for a service or workload."""
    service_name: str
    workload_type: str  # inference, training, mixed
    model_size: Optional[str]  # 7B, 13B, 30B, etc.
    
    # Resource requirements
    cpu_cores_required: float
    memory_gb_required: float
    vram_gb_required: float
    gpu_compute_required: float  # 0-1
    storage_iops_required: float
    network_mbps_required: float
    
    # Performance characteristics
    expected_throughput: float  # requests/sec
    latency_sla: float  # max acceptable latency in ms
    priority: ServicePriority
    thermal_generation: float  # Watts of heat generated
    
    # Flexibility parameters
    can_use_cpu_fallback: bool
    can_share_gpu: bool
    can_queue_requests: bool
    max_acceptable_delay: float  # seconds


@dataclass
class ResourceAllocation:
    """Resource allocation decision for a service."""
    service_name: str
    allocated_resources: Dict[ResourceType, float]
    allocation_score: float  # 0-1, quality of allocation
    expected_performance: Dict[str, float]  # throughput, latency, etc.
    thermal_impact: float
    power_consumption: float
    constraints_satisfied: bool
    optimization_notes: List[str]


@dataclass
class OptimizationDecision:
    """Comprehensive optimization decision for the AI workstation."""
    timestamp: datetime
    decision_id: str
    strategy: OptimizationStrategy
    confidence: float  # 0-1
    
    # Resource allocations
    service_allocations: Dict[str, ResourceAllocation]
    
    # System-wide optimization
    load_balancing_changes: Dict[str, Any]
    thermal_management_actions: List[str]
    performance_improvements: List[str]
    
    # Implementation details
    immediate_actions: List[str]
    gradual_optimizations: List[str]
    monitoring_requirements: List[str]
    
    # Risk assessment
    implementation_risk: str  # low, medium, high
    rollback_plan: Optional[str]
    success_metrics: Dict[str, float]


@dataclass
class SystemResourceState:
    """Current state of all system resources."""
    timestamp: datetime
    
    # Resource capacities
    cpu_capacity: ResourceCapacity
    memory_capacity: ResourceCapacity
    vram_capacity: ResourceCapacity
    gpu_compute_capacity: ResourceCapacity
    storage_capacity: ResourceCapacity
    thermal_capacity: ResourceCapacity
    
    # Service states
    active_services: Dict[str, ServiceResourceRequirement]
    service_performance: Dict[str, Dict[str, float]]
    
    # System constraints
    thermal_constraints: Dict[str, float]
    power_constraints: Dict[str, float]
    sla_constraints: Dict[str, float]
    
    # Optimization opportunities
    identified_bottlenecks: List[str]
    optimization_opportunities: List[str]
    resource_waste: Dict[ResourceType, float]


class MultiModelResourceOracle:
    """
    Intelligent resource optimization engine for multi-model AI workstation.
    
    Integrates container consciousness, hardware specialization, and thermal
    intelligence to make optimal resource allocation decisions that maximize
    performance while respecting thermal, power, and SLA constraints.
    """
    
    def __init__(self, optimization_interval: float = 30.0):
        self.optimization_interval = optimization_interval
        
        # AI workstation configuration
        self.workstation_config = {
            'cpu_cores': 16,
            'cpu_threads': 32,
            'system_memory_gb': 128,
            'gpu_vram_gb': 32,
            'gpu_compute_units': 128,  # RTX 5090 SM count estimate
            'thermal_capacity_watts': 845,  # 270W CPU + 575W GPU
            'power_budget_watts': 1200,
            'cooling_effectiveness': 0.85
        }
        
        # Service configurations (from docker-compose)
        self.service_configs = {
            'llama-cpu-0': {
                'assigned_cores': list(range(0, 8)),
                'memory_limit_gb': 32,
                'service_type': 'cpu_inference',
                'priority': ServicePriority.HIGH
            },
            'llama-cpu-1': {
                'assigned_cores': list(range(8, 16)),
                'memory_limit_gb': 32,
                'service_type': 'cpu_inference',
                'priority': ServicePriority.HIGH
            },
            'llama-cpu-2': {
                'assigned_cores': list(range(16, min(24, 16))),  # Adjust for 16-core system
                'memory_limit_gb': 32,
                'service_type': 'cpu_inference',
                'priority': ServicePriority.NORMAL
            },
            'llama-gpu': {
                'gpu_access': True,
                'vram_limit_gb': 20,
                'service_type': 'gpu_inference',
                'priority': ServicePriority.CRITICAL
            },
            'vllm-gpu': {
                'gpu_access': True,
                'vram_limit_gb': 12,
                'service_type': 'vllm_inference',
                'priority': ServicePriority.HIGH
            },
            'open-webui': {
                'cpu_limit': 2,
                'memory_limit_gb': 4,
                'service_type': 'interface',
                'priority': ServicePriority.NORMAL
            }
        }
        
        # Optimization state
        self.current_system_state: Optional[SystemResourceState] = None
        self.optimization_history: deque = deque(maxlen=100)
        self.active_optimizations: Dict[str, OptimizationDecision] = {}
        
        # Resource allocation tracking
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        self.allocation_efficiency: Dict[ResourceType, float] = {}
        self.bottleneck_history: deque = deque(maxlen=50)
        
        # Performance prediction models
        self.performance_models: Dict[str, Any] = {}
        self.optimization_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.optimization_lock = threading.RLock()
        
        # Optimization thresholds and parameters
        self.optimization_thresholds = {
            'cpu_utilization_imbalance': 30.0,  # % difference between services
            'memory_fragmentation': 25.0,       # % fragmentation
            'gpu_contention': 80.0,              # % utilization when contention matters
            'thermal_warning': 75.0,             # % of thermal capacity
            'performance_degradation': 20.0,     # % performance drop
            'sla_violation': 1.5,                # multiplier of target latency
            'efficiency_threshold': 0.7          # minimum efficiency score
        }
        
        logger.info("MultiModelResourceOracle initialized")
        
    def start_optimization_engine(self):
        """Start the continuous resource optimization engine."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._optimization_loop,
            name="ResourceOptimizationEngine",
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Multi-model resource optimization engine started")
        
    def stop_optimization_engine(self):
        """Stop the resource optimization engine."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        logger.info("Multi-model resource optimization engine stopped")
        
    def _optimization_loop(self):
        """Continuous resource optimization loop."""
        while self.monitoring_active:
            try:
                with self.optimization_lock:
                    # Update system resource state
                    self._update_system_resource_state()
                    
                    # Analyze optimization opportunities
                    optimization_opportunities = self._analyze_optimization_opportunities()
                    
                    # Generate optimization decisions
                    if optimization_opportunities:
                        optimization_decision = self._generate_optimization_decision(optimization_opportunities)
                        
                        if optimization_decision and optimization_decision.confidence > 0.7:
                            # Execute high-confidence optimizations
                            self._execute_optimization_decision(optimization_decision)
                            
                    # Update performance models
                    self._update_performance_models()
                    
                time.sleep(self.optimization_interval)
                
            except Exception as e:
                logger.error(f"Error in resource optimization loop: {e}")
                time.sleep(10.0)
                
    def integrate_system_intelligence(
        self,
        container_states: Dict[str, AIServiceState],
        blackwell_metrics: Optional[BlackwellMetrics],
        zen5_metrics: Optional[Zen5PerformanceCounters],
        thermal_profile: Optional[ThermalProfile],
        correlation_insights: List[ResourceCorrelationInsight]
    ) -> OptimizationDecision:
        """
        Integrate intelligence from all system components and generate optimization decisions.
        
        This is the main entry point for the resource oracle, combining all available
        system intelligence to make optimal resource allocation decisions.
        """
        with self.optimization_lock:
            try:
                # Update system state with integrated intelligence
                system_state = self._build_integrated_system_state(
                    container_states, blackwell_metrics, zen5_metrics, 
                    thermal_profile, correlation_insights
                )
                
                self.current_system_state = system_state
                
                # Analyze current resource allocation efficiency
                allocation_analysis = self._analyze_current_allocations(system_state)
                
                # Identify optimization opportunities
                opportunities = self._identify_optimization_opportunities(
                    system_state, allocation_analysis
                )
                
                # Generate comprehensive optimization decision
                optimization_decision = self._create_comprehensive_optimization_decision(
                    system_state, opportunities
                )
                
                # Store for historical analysis
                self.optimization_history.append(optimization_decision)
                
                return optimization_decision
                
            except Exception as e:
                logger.error(f"Error integrating system intelligence: {e}")
                return self._create_fallback_optimization_decision()
                
    def _build_integrated_system_state(
        self,
        container_states: Dict[str, AIServiceState],
        blackwell_metrics: Optional[BlackwellMetrics],
        zen5_metrics: Optional[Zen5PerformanceCounters],
        thermal_profile: Optional[ThermalProfile],
        correlation_insights: List[ResourceCorrelationInsight]
    ) -> SystemResourceState:
        """Build comprehensive system resource state from all intelligence sources."""
        current_time = datetime.now()
        
        # Build resource capacities
        cpu_capacity = self._build_cpu_capacity(container_states, zen5_metrics)
        memory_capacity = self._build_memory_capacity(container_states, zen5_metrics)
        vram_capacity = self._build_vram_capacity(container_states, blackwell_metrics)
        gpu_compute_capacity = self._build_gpu_compute_capacity(blackwell_metrics)
        storage_capacity = self._build_storage_capacity()
        thermal_capacity = self._build_thermal_capacity(thermal_profile)
        
        # Extract active service requirements
        active_services = self._extract_service_requirements(container_states)
        
        # Calculate service performance metrics
        service_performance = self._calculate_service_performance(
            container_states, blackwell_metrics, zen5_metrics
        )
        
        # Determine system constraints
        thermal_constraints = self._extract_thermal_constraints(thermal_profile)
        power_constraints = self._estimate_power_constraints(
            zen5_metrics, blackwell_metrics
        )
        sla_constraints = self._extract_sla_constraints(active_services)
        
        # Identify bottlenecks and opportunities
        bottlenecks = self._identify_system_bottlenecks(
            cpu_capacity, memory_capacity, vram_capacity, gpu_compute_capacity
        )
        
        optimization_opportunities = self._extract_optimization_opportunities(
            correlation_insights, bottlenecks
        )
        
        resource_waste = self._calculate_resource_waste(
            cpu_capacity, memory_capacity, vram_capacity
        )
        
        return SystemResourceState(
            timestamp=current_time,
            cpu_capacity=cpu_capacity,
            memory_capacity=memory_capacity,
            vram_capacity=vram_capacity,
            gpu_compute_capacity=gpu_compute_capacity,
            storage_capacity=storage_capacity,
            thermal_capacity=thermal_capacity,
            active_services=active_services,
            service_performance=service_performance,
            thermal_constraints=thermal_constraints,
            power_constraints=power_constraints,
            sla_constraints=sla_constraints,
            identified_bottlenecks=bottlenecks,
            optimization_opportunities=optimization_opportunities,
            resource_waste=resource_waste
        )
        
    def _build_cpu_capacity(
        self,
        container_states: Dict[str, AIServiceState],
        zen5_metrics: Optional[Zen5PerformanceCounters]
    ) -> ResourceCapacity:
        """Build CPU resource capacity information."""
        total_cores = self.workstation_config['cpu_cores']
        
        # Calculate current CPU utilization
        used_cores = 0.0
        for service_name, state in container_states.items():
            if state.container_metrics and state.container_metrics.status == 'running':
                service_config = self.service_configs.get(service_name, {})
                assigned_cores = len(service_config.get('assigned_cores', []))
                cpu_percent = state.container_metrics.cpu_usage_percent
                used_cores += (assigned_cores * cpu_percent / 100.0)
                
        available_cores = total_cores - used_cores
        utilization_percent = (used_cores / total_cores) * 100
        
        # Calculate efficiency based on Zen 5 metrics
        efficiency_score = 0.8  # Default
        if zen5_metrics:
            # IPC efficiency (Zen 5 target: ~2.5 IPC)
            ipc_efficiency = min(1.0, zen5_metrics.ipc / 2.5)
            
            # Cache efficiency
            cache_efficiency = (
                zen5_metrics.l1_cache_hit_rate * 0.4 +
                zen5_metrics.l2_cache_hit_rate * 0.3 +
                zen5_metrics.l3_cache_hit_rate * 0.3
            ) / 100.0
            
            efficiency_score = (ipc_efficiency + cache_efficiency) / 2.0
            
        # Bottleneck risk assessment
        bottleneck_risk = 0.0
        if utilization_percent > 80:
            bottleneck_risk = (utilization_percent - 80) / 20.0
            
        # Thermal impact
        thermal_impact = used_cores * 10.625  # ~170W TDP / 16 cores
        
        return ResourceCapacity(
            resource_type=ResourceType.CPU_CORES,
            total_capacity=total_cores,
            available_capacity=available_cores,
            reserved_capacity=0.0,  # No reserved cores in current setup
            utilization_percent=utilization_percent,
            efficiency_score=efficiency_score,
            bottleneck_risk=bottleneck_risk,
            thermal_impact=thermal_impact
        )
        
    def _build_memory_capacity(
        self,
        container_states: Dict[str, AIServiceState],
        zen5_metrics: Optional[Zen5PerformanceCounters]
    ) -> ResourceCapacity:
        """Build system memory capacity information."""
        total_memory_gb = self.workstation_config['system_memory_gb']
        
        # Calculate current memory usage
        used_memory_gb = 0.0
        for service_name, state in container_states.items():
            if state.container_metrics and state.container_metrics.status == 'running':
                used_memory_gb += state.container_metrics.memory_usage_mb / 1024.0
                
        available_memory_gb = total_memory_gb - used_memory_gb
        utilization_percent = (used_memory_gb / total_memory_gb) * 100
        
        # Memory efficiency (DDR5-6000 bandwidth utilization)
        efficiency_score = 0.75  # Default
        if zen5_metrics:
            bandwidth_efficiency = zen5_metrics.memory_bandwidth_utilization / 100.0
            numa_efficiency = zen5_metrics.numa_local_access_ratio
            efficiency_score = (bandwidth_efficiency + numa_efficiency) / 2.0
            
        # Bottleneck risk
        bottleneck_risk = max(0.0, (utilization_percent - 70) / 30.0)
        
        # Memory has minimal direct thermal impact
        thermal_impact = used_memory_gb * 0.5  # Estimate 0.5W per GB
        
        return ResourceCapacity(
            resource_type=ResourceType.SYSTEM_MEMORY,
            total_capacity=total_memory_gb,
            available_capacity=available_memory_gb,
            reserved_capacity=8.0,  # Reserve 8GB for system
            utilization_percent=utilization_percent,
            efficiency_score=efficiency_score,
            bottleneck_risk=bottleneck_risk,
            thermal_impact=thermal_impact
        )
        
    def _build_vram_capacity(
        self,
        container_states: Dict[str, AIServiceState],
        blackwell_metrics: Optional[BlackwellMetrics]
    ) -> ResourceCapacity:
        """Build GPU VRAM capacity information."""
        total_vram_gb = self.workstation_config['gpu_vram_gb']
        
        # Get VRAM usage from Blackwall metrics if available
        if blackwall_metrics:
            used_vram_gb = blackwall_metrics.used_memory_gb
            available_vram_gb = blackwall_metrics.free_memory_gb
            utilization_percent = blackwall_metrics.memory_utilization
            fragmentation = blackwall_metrics.memory_fragmentation_percent
        else:
            # Estimate from container states
            used_vram_gb = 0.0
            gpu_services = ['llama-gpu', 'vllm-gpu']
            for service_name in gpu_services:
                if service_name in container_states:
                    state = container_states[service_name]
                    if state.container_metrics and state.container_metrics.status == 'running':
                        # Estimate VRAM usage based on service type
                        if service_name == 'llama-gpu':
                            used_vram_gb += 20.0  # Large model estimate
                        elif service_name == 'vllm-gpu':
                            used_vram_gb += 12.0  # vLLM estimate
                            
            available_vram_gb = total_vram_gb - used_vram_gb
            utilization_percent = (used_vram_gb / total_vram_gb) * 100
            fragmentation = 0.0
            
        # VRAM efficiency (lower fragmentation = higher efficiency)
        efficiency_score = max(0.1, 1.0 - (fragmentation / 100.0))
        
        # VRAM bottleneck risk (critical for AI workloads)
        bottleneck_risk = 0.0
        if utilization_percent > 75:
            bottleneck_risk = (utilization_percent - 75) / 25.0
            
        # VRAM thermal impact (GDDR7 heat generation)
        thermal_impact = used_vram_gb * 2.0  # Estimate 2W per GB
        
        return ResourceCapacity(
            resource_type=ResourceType.GPU_VRAM,
            total_capacity=total_vram_gb,
            available_capacity=available_vram_gb,
            reserved_capacity=2.0,  # Reserve 2GB for system/context
            utilization_percent=utilization_percent,
            efficiency_score=efficiency_score,
            bottleneck_risk=bottleneck_risk,
            thermal_impact=thermal_impact
        )
        
    def _build_gpu_compute_capacity(
        self, blackwall_metrics: Optional[BlackwallMetrics]
    ) -> ResourceCapacity:
        """Build GPU compute capacity information."""
        total_compute_units = self.workstation_config['gpu_compute_units']
        
        if blackwall_metrics:
            utilization_percent = blackwall_metrics.gpu_utilization
            tensor_utilization = blackwall_metrics.tensor_core_utilization
            
            # Compute efficiency based on tensor core usage
            efficiency_score = (utilization_percent / 100.0) * (tensor_utilization / 100.0)
            
            # Thermal impact from GPU compute
            thermal_impact = (utilization_percent / 100.0) * 575.0  # RTX 5090 max TDP
        else:
            utilization_percent = 0.0
            efficiency_score = 0.0
            thermal_impact = 0.0
            
        used_compute = (utilization_percent / 100.0) * total_compute_units
        available_compute = total_compute_units - used_compute
        
        # GPU compute bottleneck risk
        bottleneck_risk = max(0.0, (utilization_percent - 80) / 20.0)
        
        return ResourceCapacity(
            resource_type=ResourceType.GPU_COMPUTE,
            total_capacity=total_compute_units,
            available_capacity=available_compute,
            reserved_capacity=0.0,
            utilization_percent=utilization_percent,
            efficiency_score=efficiency_score,
            bottleneck_risk=bottleneck_risk,
            thermal_impact=thermal_impact
        )
        
    def _build_storage_capacity(self) -> ResourceCapacity:
        """Build storage I/O capacity information."""
        # Storage capacity is less critical for inference workloads
        # Mainly matters for model loading and data streaming
        
        return ResourceCapacity(
            resource_type=ResourceType.STORAGE_IO,
            total_capacity=12400.0,  # Samsung 990 Pro 2TB sequential read MB/s
            available_capacity=10000.0,  # Assume some usage
            reserved_capacity=1000.0,    # Reserve for system
            utilization_percent=20.0,    # Light usage during inference
            efficiency_score=0.9,        # NVMe is efficient
            bottleneck_risk=0.1,         # Low risk for inference
            thermal_impact=5.0           # Minimal thermal impact
        )
        
    def _build_thermal_capacity(
        self, thermal_profile: Optional[ThermalProfile]
    ) -> ResourceCapacity:
        """Build thermal capacity information."""
        total_thermal_capacity = self.workstation_config['thermal_capacity_watts']
        
        if thermal_profile:
            current_heat_generation = thermal_profile.power_heat_generation
            thermal_efficiency = thermal_profile.thermal_efficiency
            utilization_percent = thermal_profile.thermal_capacity_utilization * 100
            
            # Check for thermal throttling
            cpu_throttling = thermal_profile.cpu_thermal.throttling_active
            gpu_throttling = thermal_profile.gpu_thermal.throttling_active
            any_throttling = cpu_throttling or gpu_throttling
            
            # Efficiency based on cooling effectiveness
            efficiency_score = thermal_efficiency
            
            # Bottleneck risk based on utilization and throttling
            bottleneck_risk = utilization_percent / 100.0
            if any_throttling:
                bottleneck_risk = 1.0
        else:
            current_heat_generation = 200.0  # Conservative estimate
            thermal_efficiency = 0.8
            utilization_percent = 25.0
            efficiency_score = 0.8
            bottleneck_risk = 0.2
            
        available_capacity = total_thermal_capacity - current_heat_generation
        
        return ResourceCapacity(
            resource_type=ResourceType.THERMAL_CAPACITY,
            total_capacity=total_thermal_capacity,
            available_capacity=available_capacity,
            reserved_capacity=50.0,  # Safety margin
            utilization_percent=utilization_percent,
            efficiency_score=efficiency_score,
            bottleneck_risk=bottleneck_risk,
            thermal_impact=0.0  # Thermal capacity doesn't generate heat
        )
        
    def _extract_service_requirements(
        self, container_states: Dict[str, AIServiceState]
    ) -> Dict[str, ServiceResourceRequirement]:
        """Extract resource requirements for active services."""
        service_requirements = {}
        
        for service_name, state in container_states.items():
            if state.container_metrics and state.container_metrics.status == 'running':
                config = self.service_configs.get(service_name, {})
                
                # Estimate resource requirements based on service type and current usage
                requirement = ServiceResourceRequirement(
                    service_name=service_name,
                    workload_type=config.get('service_type', 'unknown'),
                    model_size=self._estimate_model_size(state),
                    cpu_cores_required=len(config.get('assigned_cores', [])),
                    memory_gb_required=config.get('memory_limit_gb', 8),
                    vram_gb_required=config.get('vram_limit_gb', 0),
                    gpu_compute_required=1.0 if config.get('gpu_access', False) else 0.0,
                    storage_iops_required=100.0,  # Minimal for inference
                    network_mbps_required=50.0,   # API traffic
                    expected_throughput=self._estimate_throughput(state),
                    latency_sla=self._get_latency_sla(service_name),
                    priority=config.get('priority', ServicePriority.NORMAL),
                    thermal_generation=self._estimate_thermal_generation(state, config),
                    can_use_cpu_fallback=service_name.startswith('llama-gpu'),
                    can_share_gpu=service_name in ['vllm-gpu'],
                    can_queue_requests=True,
                    max_acceptable_delay=5.0
                )
                
                service_requirements[service_name] = requirement
                
        return service_requirements
        
    def _estimate_model_size(self, state: AIServiceState) -> Optional[str]:
        """Estimate model size from service state."""
        if state.model_loaded:
            model_name = state.model_loaded.lower()
            if '30b' in model_name or '33b' in model_name:
                return '30B'
            elif '13b' in model_name:
                return '13B'
            elif '7b' in model_name:
                return '7B'
        return None
        
    def _estimate_throughput(self, state: AIServiceState) -> float:
        """Estimate expected throughput for a service."""
        if state.container_metrics:
            # Rough estimation based on CPU/memory usage
            cpu_usage = state.container_metrics.cpu_usage_percent
            if cpu_usage > 50:
                return 2.0  # requests/sec
            elif cpu_usage > 20:
                return 1.0
            else:
                return 0.5
        return 1.0
        
    def _get_latency_sla(self, service_name: str) -> float:
        """Get latency SLA for a service in milliseconds."""
        sla_map = {
            'llama-gpu': 1000.0,    # 1 second for GPU inference
            'vllm-gpu': 500.0,      # 500ms for vLLM
            'llama-cpu-0': 3000.0,  # 3 seconds for CPU inference
            'llama-cpu-1': 3000.0,
            'llama-cpu-2': 3000.0,
            'open-webui': 100.0     # 100ms for UI responses
        }
        return sla_map.get(service_name, 2000.0)
        
    def _estimate_thermal_generation(
        self, state: AIServiceState, config: Dict[str, Any]
    ) -> float:
        """Estimate thermal generation from a service in Watts."""
        if config.get('gpu_access', False):
            # GPU services generate significant heat
            if state.container_metrics:
                # Estimate based on GPU usage (would be more accurate with actual GPU metrics)
                return 200.0  # Base GPU heat
        else:
            # CPU services
            if state.container_metrics:
                cpu_usage = state.container_metrics.cpu_usage_percent
                assigned_cores = len(config.get('assigned_cores', []))
                return (cpu_usage / 100.0) * assigned_cores * 10.625  # ~170W / 16 cores
                
        return 50.0  # Default estimate
        
    def _calculate_service_performance(
        self,
        container_states: Dict[str, AIServiceState],
        blackwall_metrics: Optional[BlackwellMetrics],
        zen5_metrics: Optional[Zen5PerformanceCounters]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for each service."""
        service_performance = {}
        
        for service_name, state in container_states.items():
            if state.container_metrics and state.container_metrics.status == 'running':
                config = self.service_configs.get(service_name, {})
                
                performance = {
                    'cpu_efficiency': 0.0,
                    'memory_efficiency': 0.0,
                    'throughput_estimate': 0.0,
                    'latency_estimate': 0.0,
                    'resource_efficiency': 0.0
                }
                
                # CPU efficiency
                if not config.get('gpu_access', False) and zen5_metrics:
                    performance['cpu_efficiency'] = zen5_metrics.ipc / 2.5  # Zen 5 target IPC
                    
                # Memory efficiency
                if state.container_metrics.memory_limit_mb > 0:
                    memory_utilization = state.container_metrics.memory_usage_mb / state.container_metrics.memory_limit_mb
                    performance['memory_efficiency'] = min(1.0, memory_utilization * 2.0)  # Sweet spot ~50% usage
                    
                # GPU efficiency (for GPU services)
                if config.get('gpu_access', False) and blackwall_metrics:
                    gpu_efficiency = blackwall_metrics.gpu_utilization / 100.0
                    tensor_efficiency = blackwall_metrics.tensor_core_utilization / 100.0
                    performance['gpu_efficiency'] = (gpu_efficiency + tensor_efficiency) / 2.0
                    
                # Estimate throughput and latency
                performance['throughput_estimate'] = self._estimate_throughput(state)
                performance['latency_estimate'] = self._estimate_latency(state, config)
                
                # Overall resource efficiency
                efficiency_values = [v for v in performance.values() if v > 0]
                if efficiency_values:
                    performance['resource_efficiency'] = sum(efficiency_values) / len(efficiency_values)
                    
                service_performance[service_name] = performance
                
        return service_performance
        
    def _estimate_latency(self, state: AIServiceState, config: Dict[str, Any]) -> float:
        """Estimate current latency for a service in milliseconds."""
        base_latency = self._get_latency_sla(state.service_config.name) * 0.7  # 70% of SLA
        
        # Adjust based on resource usage
        if state.container_metrics:
            cpu_usage = state.container_metrics.cpu_usage_percent
            if cpu_usage > 90:
                base_latency *= 1.5  # High CPU usage increases latency
            elif cpu_usage < 20:
                base_latency *= 0.8  # Low CPU usage reduces latency
                
        return base_latency
        
    def _extract_thermal_constraints(
        self, thermal_profile: Optional[ThermalProfile]
    ) -> Dict[str, float]:
        """Extract thermal constraints that limit performance."""
        constraints = {}
        
        if thermal_profile:
            # Temperature-based constraints
            if thermal_profile.cpu_thermal.throttling_imminent:
                constraints['cpu_thermal_limit'] = 0.8  # Reduce to 80% capacity
            if thermal_profile.gpu_thermal.throttling_imminent:
                constraints['gpu_thermal_limit'] = 0.8
                
            # Overall thermal capacity constraint
            if thermal_profile.thermal_capacity_utilization > 0.8:
                constraints['system_thermal_limit'] = 0.9
                
        return constraints
        
    def _estimate_power_constraints(
        self,
        zen5_metrics: Optional[Zen5PerformanceCounters],
        blackwall_metrics: Optional[BlackwallMetrics]
    ) -> Dict[str, float]:
        """Estimate power consumption constraints."""
        constraints = {}
        
        current_power = 0.0
        
        if zen5_metrics:
            current_power += zen5_metrics.package_power_watts
        else:
            current_power += 100.0  # Conservative CPU estimate
            
        if blackwall_metrics:
            current_power += blackwall_metrics.power_usage
        else:
            current_power += 200.0  # Conservative GPU estimate
            
        # Add system power
        current_power += 50.0
        
        power_budget = self.workstation_config['power_budget_watts']
        power_utilization = current_power / power_budget
        
        if power_utilization > 0.85:
            constraints['power_limit'] = 0.9  # Limit to 90% of budget
            
        return constraints
        
    def _extract_sla_constraints(
        self, active_services: Dict[str, ServiceResourceRequirement]
    ) -> Dict[str, float]:
        """Extract SLA constraints for active services."""
        constraints = {}
        
        for service_name, requirement in active_services.items():
            if requirement.priority in [ServicePriority.CRITICAL, ServicePriority.HIGH]:
                constraints[f'{service_name}_latency_sla'] = requirement.latency_sla
                constraints[f'{service_name}_throughput_sla'] = requirement.expected_throughput
                
        return constraints
        
    def _identify_system_bottlenecks(
        self,
        cpu_capacity: ResourceCapacity,
        memory_capacity: ResourceCapacity,
        vram_capacity: ResourceCapacity,
        gpu_compute_capacity: ResourceCapacity
    ) -> List[str]:
        """Identify current system bottlenecks."""
        bottlenecks = []
        
        # Check each resource for bottleneck conditions
        if cpu_capacity.bottleneck_risk > 0.7:
            bottlenecks.append(f"CPU cores at {cpu_capacity.utilization_percent:.1f}% utilization")
            
        if memory_capacity.bottleneck_risk > 0.7:
            bottlenecks.append(f"System memory at {memory_capacity.utilization_percent:.1f}% utilization")
            
        if vram_capacity.bottleneck_risk > 0.7:
            bottlenecks.append(f"GPU VRAM at {vram_capacity.utilization_percent:.1f}% utilization")
            
        if gpu_compute_capacity.bottleneck_risk > 0.7:
            bottlenecks.append(f"GPU compute at {gpu_compute_capacity.utilization_percent:.1f}% utilization")
            
        # Check for efficiency bottlenecks
        if cpu_capacity.efficiency_score < 0.6:
            bottlenecks.append("CPU efficiency below optimal")
            
        if vram_capacity.efficiency_score < 0.6:
            bottlenecks.append("GPU memory fragmentation")
            
        return bottlenecks
        
    def _extract_optimization_opportunities(
        self,
        correlation_insights: List[ResourceCorrelationInsight],
        bottlenecks: List[str]
    ) -> List[str]:
        """Extract optimization opportunities from correlation analysis."""
        opportunities = []
        
        # Extract opportunities from correlation insights
        for insight in correlation_insights:
            if insight.optimization_opportunity:
                opportunities.append(insight.optimization_opportunity)
                
        # Add bottleneck-derived opportunities
        for bottleneck in bottlenecks:
            if "CPU" in bottleneck:
                opportunities.append("Optimize CPU workload distribution")
            if "VRAM" in bottleneck:
                opportunities.append("Implement model quantization or memory optimization")
            if "GPU compute" in bottleneck:
                opportunities.append("Optimize GPU utilization patterns")
                
        return opportunities
        
    def _calculate_resource_waste(
        self,
        cpu_capacity: ResourceCapacity,
        memory_capacity: ResourceCapacity,
        vram_capacity: ResourceCapacity
    ) -> Dict[ResourceType, float]:
        """Calculate resource waste/underutilization."""
        waste = {}
        
        # CPU waste (cores allocated but underutilized)
        if cpu_capacity.utilization_percent < 30:
            waste[ResourceType.CPU_CORES] = 30 - cpu_capacity.utilization_percent
            
        # Memory waste (memory allocated but not used)
        if memory_capacity.utilization_percent < 40:
            waste[ResourceType.SYSTEM_MEMORY] = 40 - memory_capacity.utilization_percent
            
        # VRAM waste (less critical as models need to be loaded)
        if vram_capacity.utilization_percent < 20:
            waste[ResourceType.GPU_VRAM] = 20 - vram_capacity.utilization_percent
            
        return waste
        
    def _update_system_resource_state(self):
        """Update system resource state (placeholder for integration)."""
        # This would be called by the optimization loop to update state
        # For now, it's a placeholder since state is updated via integrate_system_intelligence
        pass
        
    def _analyze_optimization_opportunities(self) -> List[str]:
        """Analyze current system state for optimization opportunities."""
        opportunities = []
        
        if not self.current_system_state:
            return opportunities
            
        # Check for resource imbalances
        if self.current_system_state.cpu_capacity.utilization_percent < 30:
            opportunities.append("CPU underutilization")
            
        if self.current_system_state.vram_capacity.utilization_percent > 85:
            opportunities.append("VRAM pressure")
            
        # Check for efficiency issues
        if self.current_system_state.cpu_capacity.efficiency_score < 0.7:
            opportunities.append("CPU efficiency optimization")
            
        return opportunities
        
    def _generate_optimization_decision(self, opportunities: List[str]) -> Optional[OptimizationDecision]:
        """Generate optimization decision based on identified opportunities."""
        if not opportunities or not self.current_system_state:
            return None
            
        # This is a simplified implementation
        # In practice, this would use sophisticated algorithms to generate optimal decisions
        
        decision_id = f"opt_{int(datetime.now().timestamp())}"
        
        return OptimizationDecision(
            timestamp=datetime.now(),
            decision_id=decision_id,
            strategy=OptimizationStrategy.BALANCE_EFFICIENCY,
            confidence=0.8,
            service_allocations={},
            load_balancing_changes={},
            thermal_management_actions=[],
            performance_improvements=opportunities,
            immediate_actions=[],
            gradual_optimizations=[],
            monitoring_requirements=[],
            implementation_risk="low",
            rollback_plan=None,
            success_metrics={}
        )
        
    def _execute_optimization_decision(self, decision: OptimizationDecision):
        """Execute an optimization decision (placeholder)."""
        # This would implement the actual optimization changes
        # For now, just log the decision
        logger.info(f"Executing optimization decision {decision.decision_id} with {len(decision.performance_improvements)} improvements")
        
    def _update_performance_models(self):
        """Update performance prediction models (placeholder)."""
        # This would update ML models based on historical performance
        pass
        
    def _analyze_current_allocations(
        self, system_state: SystemResourceState
    ) -> Dict[str, Any]:
        """Analyze efficiency of current resource allocations."""
        analysis = {
            'overall_efficiency': 0.0,
            'resource_utilization': {},
            'bottlenecks': system_state.identified_bottlenecks,
            'waste': system_state.resource_waste,
            'recommendations': []
        }
        
        # Calculate overall efficiency
        resource_efficiencies = [
            system_state.cpu_capacity.efficiency_score,
            system_state.memory_capacity.efficiency_score,
            system_state.vram_capacity.efficiency_score,
            system_state.gpu_compute_capacity.efficiency_score
        ]
        
        analysis['overall_efficiency'] = sum(resource_efficiencies) / len(resource_efficiencies)
        
        # Resource utilization summary
        analysis['resource_utilization'] = {
            'cpu': system_state.cpu_capacity.utilization_percent,
            'memory': system_state.memory_capacity.utilization_percent,
            'vram': system_state.vram_capacity.utilization_percent,
            'gpu_compute': system_state.gpu_compute_capacity.utilization_percent,
            'thermal': system_state.thermal_capacity.utilization_percent
        }
        
        return analysis
        
    def _identify_optimization_opportunities(
        self, system_state: SystemResourceState, allocation_analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify specific optimization opportunities."""
        opportunities = []
        
        # From system state
        opportunities.extend(system_state.optimization_opportunities)
        
        # From allocation analysis
        if allocation_analysis['overall_efficiency'] < 0.7:
            opportunities.append("Improve overall resource efficiency")
            
        # Check for specific resource issues
        util = allocation_analysis['resource_utilization']
        if util['cpu'] < 40 and util['gpu_compute'] > 80:
            opportunities.append("Rebalance workload from GPU to CPU services")
            
        if util['vram'] > 85:
            opportunities.append("Optimize GPU memory usage")
            
        if util['thermal'] > 80:
            opportunities.append("Implement thermal-aware workload management")
            
        return opportunities
        
    def _create_comprehensive_optimization_decision(
        self, system_state: SystemResourceState, opportunities: List[str]
    ) -> OptimizationDecision:
        """Create comprehensive optimization decision with specific actions."""
        decision_id = f"comprehensive_opt_{int(datetime.now().timestamp())}"
        
        # Determine strategy based on system state
        strategy = self._determine_optimal_strategy(system_state)
        
        # Generate service allocations
        service_allocations = self._generate_service_allocations(system_state)
        
        # Generate specific actions
        immediate_actions = self._generate_immediate_actions(system_state, opportunities)
        gradual_optimizations = self._generate_gradual_optimizations(opportunities)
        thermal_actions = self._generate_thermal_management_actions(system_state)
        
        # Calculate confidence based on system state certainty
        confidence = self._calculate_decision_confidence(system_state)
        
        return OptimizationDecision(
            timestamp=datetime.now(),
            decision_id=decision_id,
            strategy=strategy,
            confidence=confidence,
            service_allocations=service_allocations,
            load_balancing_changes=self._generate_load_balancing_changes(system_state),
            thermal_management_actions=thermal_actions,
            performance_improvements=opportunities,
            immediate_actions=immediate_actions,
            gradual_optimizations=gradual_optimizations,
            monitoring_requirements=self._generate_monitoring_requirements(system_state),
            implementation_risk=self._assess_implementation_risk(system_state),
            rollback_plan=self._generate_rollback_plan(),
            success_metrics=self._define_success_metrics(system_state)
        )
        
    def _determine_optimal_strategy(self, system_state: SystemResourceState) -> OptimizationStrategy:
        """Determine optimal optimization strategy based on system state."""
        # Check thermal constraints
        if system_state.thermal_capacity.utilization_percent > 80:
            return OptimizationStrategy.THERMAL_CONSERVATIVE
            
        # Check for bottlenecks
        if system_state.identified_bottlenecks:
            return OptimizationStrategy.MAXIMIZE_THROUGHPUT
            
        # Check power efficiency
        total_power = (
            system_state.cpu_capacity.thermal_impact +
            system_state.gpu_compute_capacity.thermal_impact +
            50  # System power
        )
        if total_power > 800:  # High power usage
            return OptimizationStrategy.POWER_EFFICIENT
            
        # Default to balanced approach
        return OptimizationStrategy.BALANCE_EFFICIENCY
        
    def _generate_service_allocations(
        self, system_state: SystemResourceState
    ) -> Dict[str, ResourceAllocation]:
        """Generate optimal resource allocations for each service."""
        allocations = {}
        
        for service_name, requirement in system_state.active_services.items():
            # Generate allocation based on current capacity and requirements
            allocation = ResourceAllocation(
                service_name=service_name,
                allocated_resources={
                    ResourceType.CPU_CORES: requirement.cpu_cores_required,
                    ResourceType.SYSTEM_MEMORY: requirement.memory_gb_required,
                    ResourceType.GPU_VRAM: requirement.vram_gb_required,
                    ResourceType.GPU_COMPUTE: requirement.gpu_compute_required
                },
                allocation_score=0.8,  # Placeholder
                expected_performance={
                    'throughput': requirement.expected_throughput,
                    'latency': requirement.latency_sla * 0.7
                },
                thermal_impact=requirement.thermal_generation,
                power_consumption=requirement.thermal_generation * 1.2,
                constraints_satisfied=True,
                optimization_notes=[]
            )
            
            allocations[service_name] = allocation
            
        return allocations
        
    def _generate_immediate_actions(
        self, system_state: SystemResourceState, opportunities: List[str]
    ) -> List[str]:
        """Generate immediate actions based on system state."""
        actions = []
        
        # Critical thermal management
        if system_state.thermal_capacity.utilization_percent > 85:
            actions.append("Reduce workload on high-thermal services")
            
        # Critical resource pressure
        if system_state.vram_capacity.utilization_percent > 90:
            actions.append("Implement emergency VRAM optimization")
            
        # Service health issues
        for bottleneck in system_state.identified_bottlenecks:
            if "CPU" in bottleneck:
                actions.append("Rebalance CPU workload distribution")
                
        return actions
        
    def _generate_gradual_optimizations(self, opportunities: List[str]) -> List[str]:
        """Generate gradual optimization actions."""
        optimizations = []
        
        for opportunity in opportunities:
            if "efficiency" in opportunity.lower():
                optimizations.append(f"Implement {opportunity} over next optimization cycle")
            elif "optimization" in opportunity.lower():
                optimizations.append(f"Schedule {opportunity} for next maintenance window")
                
        return optimizations
        
    def _generate_thermal_management_actions(
        self, system_state: SystemResourceState
    ) -> List[str]:
        """Generate thermal management actions."""
        actions = []
        
        thermal_util = system_state.thermal_capacity.utilization_percent
        
        if thermal_util > 75:
            actions.append("Increase cooling fan speeds")
            actions.append("Implement thermal-aware workload scheduling")
            
        if thermal_util > 85:
            actions.append("Reduce concurrent high-thermal workloads")
            
        return actions
        
    def _generate_load_balancing_changes(self, system_state: SystemResourceState) -> Dict[str, Any]:
        """Generate load balancing changes."""
        changes = {}
        
        # Check CPU service load balance
        cpu_services = ['llama-cpu-0', 'llama-cpu-1', 'llama-cpu-2']
        cpu_utilizations = []
        
        for service_name in cpu_services:
            if service_name in system_state.active_services:
                # Would get actual utilization from service performance
                cpu_utilizations.append(50.0)  # Placeholder
                
        if cpu_utilizations:
            max_util = max(cpu_utilizations)
            min_util = min(cpu_utilizations)
            
            if max_util - min_util > 30:  # Significant imbalance
                changes['cpu_rebalancing'] = {
                    'action': 'redistribute_requests',
                    'target_balance': 'even_distribution',
                    'priority': 'medium'
                }
                
        return changes
        
    def _calculate_decision_confidence(self, system_state: SystemResourceState) -> float:
        """Calculate confidence level for the optimization decision."""
        confidence_factors = []
        
        # Data availability
        if system_state.cpu_capacity.efficiency_score > 0:
            confidence_factors.append(0.8)
        if system_state.vram_capacity.efficiency_score > 0:
            confidence_factors.append(0.8)
        if system_state.thermal_capacity.efficiency_score > 0:
            confidence_factors.append(0.9)
            
        # System stability
        if not system_state.identified_bottlenecks:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
            
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
        
    def _generate_monitoring_requirements(self, system_state: SystemResourceState) -> List[str]:
        """Generate monitoring requirements for optimization tracking."""
        requirements = [
            "Monitor CPU utilization per service",
            "Track GPU VRAM usage and fragmentation",
            "Monitor thermal state and cooling effectiveness",
            "Track service-level performance metrics"
        ]
        
        if system_state.thermal_capacity.utilization_percent > 70:
            requirements.append("Enhanced thermal monitoring with 1-minute intervals")
            
        return requirements
        
    def _assess_implementation_risk(self, system_state: SystemResourceState) -> str:
        """Assess risk level for implementing optimization decisions."""
        risk_factors = []
        
        # Thermal risk
        if system_state.thermal_capacity.utilization_percent > 80:
            risk_factors.append("thermal")
            
        # Resource pressure risk
        if system_state.vram_capacity.utilization_percent > 85:
            risk_factors.append("vram_pressure")
            
        # Service stability risk
        if len(system_state.identified_bottlenecks) > 2:
            risk_factors.append("multiple_bottlenecks")
            
        if len(risk_factors) >= 2:
            return "high"
        elif len(risk_factors) == 1:
            return "medium"
        else:
            return "low"
            
    def _generate_rollback_plan(self) -> Optional[str]:
        """Generate rollback plan for optimization changes."""
        return "Restore previous resource allocation configuration and monitor for stability"
        
    def _define_success_metrics(self, system_state: SystemResourceState) -> Dict[str, float]:
        """Define success metrics for optimization decisions."""
        return {
            'target_cpu_efficiency': 0.8,
            'target_memory_efficiency': 0.75,
            'target_thermal_utilization': min(75.0, system_state.thermal_capacity.utilization_percent),
            'target_overall_efficiency': 0.85,
            'max_acceptable_latency_increase': 1.2  # 20% increase max
        }
        
    def _create_fallback_optimization_decision(self) -> OptimizationDecision:
        """Create a safe fallback optimization decision."""
        return OptimizationDecision(
            timestamp=datetime.now(),
            decision_id=f"fallback_{int(datetime.now().timestamp())}",
            strategy=OptimizationStrategy.BALANCE_EFFICIENCY,
            confidence=0.3,
            service_allocations={},
            load_balancing_changes={},
            thermal_management_actions=["Monitor thermal state"],
            performance_improvements=["Maintain current configuration"],
            immediate_actions=["No immediate changes"],
            gradual_optimizations=["Continue monitoring"],
            monitoring_requirements=["Standard monitoring"],
            implementation_risk="low",
            rollback_plan="No changes to rollback",
            success_metrics={}
        )
        
    def get_optimization_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization status summary."""
        if not self.current_system_state:
            return {'status': 'unavailable', 'reason': 'No system state available'}
            
        recent_decision = None
        if self.optimization_history:
            recent_decision = self.optimization_history[-1]
            
        return {
            'status': 'active',
            'optimization_engine': 'Multi-Model Resource Oracle',
            'monitoring_active': self.monitoring_active,
            'current_system_state': {
                'cpu_utilization': self.current_system_state.cpu_capacity.utilization_percent,
                'memory_utilization': self.current_system_state.memory_capacity.utilization_percent,
                'vram_utilization': self.current_system_state.vram_capacity.utilization_percent,
                'thermal_utilization': self.current_system_state.thermal_capacity.utilization_percent,
                'identified_bottlenecks': self.current_system_state.identified_bottlenecks,
                'optimization_opportunities': self.current_system_state.optimization_opportunities
            },
            'resource_efficiency': {
                'cpu_efficiency': self.current_system_state.cpu_capacity.efficiency_score,
                'memory_efficiency': self.current_system_state.memory_capacity.efficiency_score,
                'vram_efficiency': self.current_system_state.vram_capacity.efficiency_score,
                'gpu_efficiency': self.current_system_state.gpu_compute_capacity.efficiency_score,
                'thermal_efficiency': self.current_system_state.thermal_capacity.efficiency_score
            },
            'recent_optimization': {
                'decision_id': recent_decision.decision_id if recent_decision else None,
                'strategy': recent_decision.strategy.value if recent_decision else None,
                'confidence': recent_decision.confidence if recent_decision else 0.0,
                'implementation_risk': recent_decision.implementation_risk if recent_decision else None,
                'performance_improvements': recent_decision.performance_improvements if recent_decision else []
            } if recent_decision else None,
            'active_services': list(self.current_system_state.active_services.keys()),
            'service_performance': self.current_system_state.service_performance,
            'constraints': {
                'thermal': self.current_system_state.thermal_constraints,
                'power': self.current_system_state.power_constraints,
                'sla': self.current_system_state.sla_constraints
            },
            'optimization_stats': {
                'total_optimizations': len(self.optimization_history),
                'average_confidence': sum(opt.confidence for opt in self.optimization_history) / len(self.optimization_history) if self.optimization_history else 0.0,
                'last_optimization': recent_decision.timestamp.isoformat() if recent_decision else None
            }
        }
        
    def __del__(self):
        """Cleanup on object destruction."""
        try:
            self.stop_optimization_engine()
        except Exception:
            pass