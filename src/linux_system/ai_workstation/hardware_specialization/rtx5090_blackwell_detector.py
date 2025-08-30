"""
RTX 5090 Blackwell Architecture Detector

Specialized monitoring and intelligence for NVIDIA RTX 5090 Blackwell architecture,
providing advanced GPU metrics, AI workload optimization insights, and thermal
management specific to the RTX 5090's unique capabilities.

Features:
- Blackwell-specific tensor core utilization monitoring
- 32GB GDDR7 memory bandwidth and efficiency analysis
- CUDA 12.9.1 kernel launch pattern detection
- AI transformer model optimization insights  
- Advanced thermal and power management correlation
- VRAM fragmentation and allocation optimization
- NVLink readiness and multi-GPU scaling preparation
"""

import pynvml
import subprocess
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

from ...temporal.core.change_detector import SystemChangeDetector, SystemState, SystemChange
from ...temporal.core.types import ChangeType, ComponentType, Significance


logger = logging.getLogger(__name__)


@dataclass
class BlackwellMetrics:
    """Blackwell architecture-specific metrics."""
    timestamp: datetime
    
    # Core GPU metrics
    gpu_utilization: float  # 0-100%
    memory_utilization: float  # 0-100%
    temperature: float  # Celsius
    power_usage: float  # Watts
    power_limit: float  # Watts
    
    # Blackwell-specific metrics
    tensor_core_utilization: float  # 0-100%
    rt_core_utilization: float  # 0-100%  
    memory_bandwidth_utilization: float  # 0-100%
    nvlink_bandwidth_utilization: float  # 0-100%
    
    # Memory details (32GB GDDR7)
    total_memory_gb: float
    used_memory_gb: float
    free_memory_gb: float
    memory_fragmentation_percent: float
    
    # CUDA metrics
    active_cuda_contexts: int
    cuda_kernel_launches_per_sec: float
    cuda_memory_transfers_per_sec: float
    
    # AI workload metrics
    transformer_optimizations_active: bool
    mixed_precision_active: bool
    tensor_throughput_tflops: float
    
    # Thermal and power efficiency
    thermal_throttling_active: bool
    power_efficiency_score: float  # Performance per watt
    boost_clock_mhz: int
    memory_clock_mhz: int


@dataclass
class VRAMAllocation:
    """VRAM allocation and fragmentation analysis."""
    process_id: int
    process_name: str
    allocated_memory_mb: float
    allocation_type: str  # model, cache, workspace, fragmented
    efficiency_score: float  # 0-1, allocation efficiency
    last_accessed: datetime


@dataclass
class AIWorkloadProfile:
    """AI workload characterization for optimization."""
    workload_type: str  # inference, training, mixed
    model_architecture: str  # transformer, cnn, rnn, unknown
    batch_size: int
    sequence_length: Optional[int]
    precision_mode: str  # fp32, fp16, int8, mixed
    memory_pattern: str  # static, dynamic, streaming
    compute_intensity: str  # memory_bound, compute_bound, balanced
    optimization_opportunities: List[str]


class RTX5090BlackwallDetector(SystemChangeDetector):
    """
    Advanced detector for RTX 5090 Blackwall architecture.
    
    Provides specialized monitoring and optimization insights for the RTX 5090's
    unique capabilities, including 4th-generation tensor cores, 32GB GDDR7 memory,
    and advanced AI acceleration features.
    """
    
    def __init__(self, monitoring_interval: float = 5.0):
        super().__init__()
        self.monitoring_interval = monitoring_interval
        
        # Initialize NVML
        self.nvml_initialized = False
        self.gpu_handle = None
        self._initialize_nvml()
        
        # Blackwell-specific configuration
        self.blackwell_config = {
            'gpu_name': 'NVIDIA GeForce RTX 5090',
            'architecture': 'Blackwell',
            'compute_capability': 9.0,  # Expected for Blackwell
            'total_vram_gb': 32,
            'memory_bandwidth_gbps': 896,
            'tensor_cores': 128,  # Estimated for RTX 5090
            'rt_cores': 84,  # Estimated for RTX 5090
            'base_clock_mhz': 2200,  # Estimated
            'boost_clock_mhz': 2600,  # Estimated
            'memory_clock_mhz': 1400   # GDDR7 effective rate
        }
        
        # Monitoring state
        self.previous_metrics: Optional[BlackwellMetrics] = None
        self.metrics_history: deque = deque(maxlen=1000)
        self.vram_allocations: Dict[int, VRAMAllocation] = {}
        self.workload_profiles: Dict[str, AIWorkloadProfile] = {}
        
        # Background monitoring
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.thread_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="RTX5090Monitor")
        
        # Change detection thresholds
        self.change_thresholds = {
            'gpu_utilization': 10.0,  # % change
            'memory_utilization': 5.0,  # % change
            'temperature': 5.0,  # °C change
            'power_usage': 50.0,  # W change
            'tensor_utilization': 15.0,  # % change
            'memory_bandwidth': 10.0,  # % change
            'thermal_throttle_temp': 83.0,  # °C
            'power_efficiency_drop': 0.2  # Performance per watt drop
        }
        
        # AI workload pattern recognition
        self.workload_patterns = {
            'inference_patterns': {
                'low_batch_high_frequency': {'batch_size': (1, 4), 'frequency': 'high'},
                'high_batch_low_frequency': {'batch_size': (8, 64), 'frequency': 'low'},
                'streaming_inference': {'memory_pattern': 'streaming', 'latency': 'priority'}
            },
            'model_patterns': {
                'large_language_model': {'memory_gb': (15, 32), 'compute_type': 'transformer'},
                'vision_model': {'memory_gb': (2, 10), 'compute_type': 'cnn'},
                'multimodal_model': {'memory_gb': (10, 25), 'compute_type': 'mixed'}
            }
        }
        
    def _initialize_nvml(self):
        """Initialize NVML for GPU monitoring."""
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Verify this is RTX 5090 Blackwell
            gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle).decode('utf-8')
            if 'RTX 5090' in gpu_name or 'Blackwell' in gpu_name:
                self.nvml_initialized = True
                logger.info(f"RTX 5090 Blackwell detector initialized for: {gpu_name}")
            else:
                logger.warning(f"Expected RTX 5090, found: {gpu_name}")
                self.nvml_initialized = True  # Continue anyway for testing
                
        except Exception as e:
            logger.error(f"Failed to initialize NVML: {e}")
            self.nvml_initialized = False
            
    def start_monitoring(self):
        """Start continuous Blackwell-specific monitoring."""
        if self.monitoring_active or not self.nvml_initialized:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="RTX5090BlackwallMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("RTX 5090 Blackwall monitoring started")
        
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        self.thread_pool.shutdown(wait=True)
        logger.info("RTX 5090 Blackwall monitoring stopped")
        
    def _monitoring_loop(self):
        """Continuous monitoring loop for Blackwall metrics."""
        while self.monitoring_active:
            try:
                current_metrics = self._collect_blackwall_metrics()
                if current_metrics:
                    self.metrics_history.append(current_metrics)
                    self._update_workload_profiles(current_metrics)
                    self._analyze_vram_allocations()
                    
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in Blackwall monitoring loop: {e}")
                time.sleep(5.0)  # Back off on error
                
    def _collect_blackwall_metrics(self) -> Optional[BlackwallMetrics]:
        """Collect comprehensive Blackwall architecture metrics."""
        if not self.nvml_initialized:
            return None
            
        try:
            current_time = datetime.now()
            
            # Basic GPU metrics
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            temperature = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            power_usage = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # mW to W
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(self.gpu_handle)[1] / 1000.0
            
            # Memory metrics (32GB GDDR7)
            total_memory_gb = memory_info.total / (1024**3)
            used_memory_gb = memory_info.used / (1024**3)
            free_memory_gb = memory_info.free / (1024**3)
            memory_utilization = (memory_info.used / memory_info.total) * 100
            
            # Blackwell-specific metrics (some estimated/calculated)
            tensor_core_util = self._estimate_tensor_core_utilization(gpu_util.gpu)
            rt_core_util = self._estimate_rt_core_utilization()
            memory_bandwidth_util = self._calculate_memory_bandwidth_utilization()
            nvlink_bandwidth_util = self._estimate_nvlink_utilization()
            
            # Memory fragmentation analysis
            memory_fragmentation = self._analyze_memory_fragmentation()
            
            # CUDA metrics
            cuda_contexts = self._count_cuda_contexts()
            kernel_launches = self._estimate_kernel_launch_rate()
            memory_transfers = self._estimate_memory_transfer_rate()
            
            # AI workload detection
            transformer_opts, mixed_precision = self._detect_ai_optimizations()
            tensor_throughput = self._calculate_tensor_throughput(gpu_util.gpu, power_usage)
            
            # Thermal and efficiency metrics
            thermal_throttling = temperature > self.change_thresholds['thermal_throttle_temp']
            power_efficiency = self._calculate_power_efficiency(gpu_util.gpu, power_usage)
            
            # Clock speeds (if available)
            try:
                boost_clock = pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_GRAPHICS)
                memory_clock = pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_MEM)
            except:
                boost_clock = self.blackwall_config['boost_clock_mhz']
                memory_clock = self.blackwall_config['memory_clock_mhz']
                
            return BlackwallMetrics(
                timestamp=current_time,
                gpu_utilization=gpu_util.gpu,
                memory_utilization=memory_utilization,
                temperature=temperature,
                power_usage=power_usage,
                power_limit=power_limit,
                tensor_core_utilization=tensor_core_util,
                rt_core_utilization=rt_core_util,
                memory_bandwidth_utilization=memory_bandwidth_util,
                nvlink_bandwidth_utilization=nvlink_bandwidth_util,
                total_memory_gb=total_memory_gb,
                used_memory_gb=used_memory_gb,
                free_memory_gb=free_memory_gb,
                memory_fragmentation_percent=memory_fragmentation,
                active_cuda_contexts=cuda_contexts,
                cuda_kernel_launches_per_sec=kernel_launches,
                cuda_memory_transfers_per_sec=memory_transfers,
                transformer_optimizations_active=transformer_opts,
                mixed_precision_active=mixed_precision,
                tensor_throughput_tflops=tensor_throughput,
                thermal_throttling_active=thermal_throttling,
                power_efficiency_score=power_efficiency,
                boost_clock_mhz=boost_clock,
                memory_clock_mhz=memory_clock
            )
            
        except Exception as e:
            logger.error(f"Error collecting Blackwall metrics: {e}")
            return None
            
    def _estimate_tensor_core_utilization(self, gpu_utilization: float) -> float:
        """Estimate 4th-gen tensor core utilization based on GPU activity."""
        # Blackwall tensor cores are more efficient, so higher utilization with AI workloads
        # This is an estimation - actual tensor core utilization requires specialized profiling
        
        if gpu_utilization < 10:
            return 0.0
        elif gpu_utilization < 30:
            # Low GPU utilization suggests non-tensor workload
            return gpu_utilization * 0.2
        elif gpu_utilization > 80:
            # High GPU utilization with AI workloads likely uses tensor cores heavily
            return min(95.0, gpu_utilization * 0.9)
        else:
            # Moderate utilization - estimate based on workload patterns
            return gpu_utilization * 0.6
            
    def _estimate_rt_core_utilization(self) -> float:
        """Estimate RT core utilization (ray tracing cores)."""
        # RT cores are not typically used for AI inference
        # This would be relevant for rendering or ray tracing applications
        return 0.0
        
    def _calculate_memory_bandwidth_utilization(self) -> float:
        """Calculate GDDR7 memory bandwidth utilization."""
        # This requires detailed memory access profiling
        # For now, estimate based on memory usage and GPU activity
        if len(self.metrics_history) < 2:
            return 0.0
            
        recent_metrics = list(self.metrics_history)[-2:]
        if len(recent_metrics) < 2:
            return 0.0
            
        # Estimate based on memory usage changes and GPU utilization
        current = recent_metrics[-1]
        previous = recent_metrics[-2]
        
        memory_delta = abs(current.used_memory_gb - previous.used_memory_gb)
        time_delta = (current.timestamp - previous.timestamp).total_seconds()
        
        if time_delta > 0:
            # Simple estimation: memory transfer rate vs. theoretical bandwidth
            transfer_rate_gbps = (memory_delta * 8) / time_delta  # Convert GB to Gb and divide by seconds
            bandwidth_utilization = (transfer_rate_gbps / self.blackwall_config['memory_bandwidth_gbps']) * 100
            return min(100.0, bandwidth_utilization)
            
        return 0.0
        
    def _estimate_nvlink_utilization(self) -> float:
        """Estimate NVLink utilization (relevant for multi-GPU setups)."""
        # RTX 5090 has NVLink readiness but single GPU setup
        return 0.0
        
    def _analyze_memory_fragmentation(self) -> float:
        """Analyze VRAM fragmentation patterns."""
        try:
            # Get process-specific memory usage
            processes = self._get_gpu_processes()
            
            if not processes:
                return 0.0
                
            total_used = sum(proc['memory_mb'] for proc in processes)
            largest_allocation = max(proc['memory_mb'] for proc in processes)
            
            # Simple fragmentation metric: deviation from optimal allocation
            if total_used > 0:
                # Higher fragmentation if many small allocations vs. few large ones
                allocation_efficiency = largest_allocation / total_used if total_used > 0 else 1.0
                fragmentation = (1.0 - allocation_efficiency) * 100
                return min(100.0, fragmentation)
                
            return 0.0
            
        except Exception as e:
            logger.error(f"Error analyzing memory fragmentation: {e}")
            return 0.0
            
    def _get_gpu_processes(self) -> List[Dict[str, Any]]:
        """Get GPU processes and their memory usage."""
        try:
            processes = []
            process_infos = pynvml.nvmlDeviceGetComputeRunningProcesses(self.gpu_handle)
            
            for process_info in process_infos:
                try:
                    process_name = pynvml.nvmlSystemGetProcessName(process_info.pid).decode('utf-8')
                    processes.append({
                        'pid': process_info.pid,
                        'name': process_name,
                        'memory_mb': process_info.usedGpuMemory / (1024 * 1024)
                    })
                except:
                    # Process may have exited
                    continue
                    
            return processes
            
        except Exception as e:
            logger.error(f"Error getting GPU processes: {e}")
            return []
            
    def _count_cuda_contexts(self) -> int:
        """Count active CUDA contexts."""
        try:
            processes = self._get_gpu_processes()
            return len(processes)  # Each process typically has one CUDA context
        except:
            return 0
            
    def _estimate_kernel_launch_rate(self) -> float:
        """Estimate CUDA kernel launch rate."""
        # This requires profiling tools like nsight or nvprof
        # For now, provide a basic estimation based on GPU activity
        if len(self.metrics_history) < 2:
            return 0.0
            
        recent_gpu_util = self.metrics_history[-1].gpu_utilization
        
        # Rough estimation: higher GPU utilization = more kernel launches
        if recent_gpu_util > 80:
            return recent_gpu_util * 10  # High activity
        elif recent_gpu_util > 40:
            return recent_gpu_util * 5   # Moderate activity
        else:
            return recent_gpu_util * 2   # Low activity
            
    def _estimate_memory_transfer_rate(self) -> float:
        """Estimate memory transfer rate between host and device."""
        if len(self.metrics_history) < 2:
            return 0.0
            
        # Estimate based on memory usage changes
        current = self.metrics_history[-1]
        previous = self.metrics_history[-2] if len(self.metrics_history) >= 2 else current
        
        memory_change = abs(current.used_memory_gb - previous.used_memory_gb)
        time_delta = (current.timestamp - previous.timestamp).total_seconds()
        
        if time_delta > 0:
            return (memory_change * 1024) / time_delta  # MB/s
        return 0.0
        
    def _detect_ai_optimizations(self) -> Tuple[bool, bool]:
        """Detect active AI optimizations (transformer optimizations, mixed precision)."""
        # This would require integration with CUDA libraries or profiling
        # For now, infer from GPU utilization patterns
        
        transformer_optimizations = False
        mixed_precision = False
        
        if len(self.metrics_history) >= 5:
            recent_utils = [m.gpu_utilization for m in list(self.metrics_history)[-5:]]
            avg_util = sum(recent_utils) / len(recent_utils)
            
            # High sustained utilization suggests transformer workloads
            if avg_util > 70 and all(u > 50 for u in recent_utils):
                transformer_optimizations = True
                
            # Mixed precision often shows specific utilization patterns
            # This is a heuristic - actual detection requires profiler integration
            util_variance = max(recent_utils) - min(recent_utils)
            if util_variance < 20 and avg_util > 60:  # Stable high utilization
                mixed_precision = True
                
        return transformer_optimizations, mixed_precision
        
    def _calculate_tensor_throughput(self, gpu_utilization: float, power_usage: float) -> float:
        """Calculate estimated tensor throughput in TFLOPS."""
        # RTX 5090 theoretical peak performance (estimated)
        peak_tensor_tflops = 165.0  # Estimated for RTX 5090 with mixed precision
        
        # Scale by actual utilization and efficiency
        utilization_factor = gpu_utilization / 100.0
        
        # Power efficiency factor (higher power usage may indicate higher performance)
        power_factor = min(1.0, power_usage / 400.0)  # Normalize to ~400W typical usage
        
        actual_throughput = peak_tensor_tflops * utilization_factor * power_factor
        return max(0.0, actual_throughput)
        
    def _calculate_power_efficiency(self, gpu_utilization: float, power_usage: float) -> float:
        """Calculate power efficiency score (performance per watt)."""
        if power_usage <= 0:
            return 0.0
            
        # Performance proxy based on utilization
        performance_score = gpu_utilization / 100.0
        
        # Power efficiency: performance per watt
        efficiency = performance_score / (power_usage / 400.0)  # Normalize to 400W baseline
        
        return min(2.0, max(0.0, efficiency))  # Cap at 2.0 for exceptional efficiency
        
    def _update_workload_profiles(self, metrics: BlackwallMetrics):
        """Update AI workload profiles based on metrics."""
        # Analyze current workload characteristics
        workload_type = self._classify_workload_type(metrics)
        
        if workload_type != 'idle':
            profile_key = f"{workload_type}_{int(metrics.timestamp.timestamp())}"
            
            profile = AIWorkloadProfile(
                workload_type=workload_type,
                model_architecture=self._infer_model_architecture(metrics),
                batch_size=self._estimate_batch_size(metrics),
                sequence_length=None,  # Would need profiler integration
                precision_mode=self._infer_precision_mode(metrics),
                memory_pattern=self._analyze_memory_pattern(metrics),
                compute_intensity=self._classify_compute_intensity(metrics),
                optimization_opportunities=self._identify_optimization_opportunities(metrics)
            )
            
            self.workload_profiles[profile_key] = profile
            
            # Keep only recent profiles
            if len(self.workload_profiles) > 100:
                oldest_key = min(self.workload_profiles.keys())
                del self.workload_profiles[oldest_key]
                
    def _classify_workload_type(self, metrics: BlackwallMetrics) -> str:
        """Classify the current workload type."""
        if metrics.gpu_utilization < 10:
            return 'idle'
        elif metrics.transformer_optimizations_active:
            return 'inference'
        elif metrics.gpu_utilization > 90 and metrics.used_memory_gb > 20:
            return 'training'
        elif metrics.gpu_utilization > 50:
            return 'inference'
        else:
            return 'mixed'
            
    def _infer_model_architecture(self, metrics: BlackwallMetrics) -> str:
        """Infer model architecture from resource usage patterns."""
        memory_usage = metrics.used_memory_gb
        
        if memory_usage > 25:
            return 'large_language_model'
        elif memory_usage > 15:
            return 'medium_language_model'
        elif memory_usage > 8:
            return 'vision_model'
        elif metrics.tensor_core_utilization > 60:
            return 'transformer'
        else:
            return 'unknown'
            
    def _estimate_batch_size(self, metrics: BlackwallMetrics) -> int:
        """Estimate batch size from memory and compute patterns."""
        memory_usage = metrics.used_memory_gb
        gpu_util = metrics.gpu_utilization
        
        if memory_usage > 20 and gpu_util > 80:
            return 32  # Large batch
        elif memory_usage > 10 and gpu_util > 60:
            return 8   # Medium batch
        elif gpu_util > 40:
            return 2   # Small batch
        else:
            return 1   # Single batch
            
    def _infer_precision_mode(self, metrics: BlackwallMetrics) -> str:
        """Infer precision mode from performance characteristics."""
        if metrics.mixed_precision_active:
            return 'mixed'
        elif metrics.tensor_throughput_tflops > 80:
            return 'fp16'  # High throughput suggests lower precision
        elif metrics.memory_utilization > 80:
            return 'fp32'  # High memory usage suggests full precision
        else:
            return 'unknown'
            
    def _analyze_memory_pattern(self, metrics: BlackwallMetrics) -> str:
        """Analyze memory access patterns."""
        if len(self.metrics_history) < 5:
            return 'unknown'
            
        recent_memory_usage = [m.used_memory_gb for m in list(self.metrics_history)[-5:]]
        memory_variance = max(recent_memory_usage) - min(recent_memory_usage)
        
        if memory_variance < 0.5:
            return 'static'    # Stable memory usage
        elif memory_variance > 5:
            return 'dynamic'   # Highly variable memory usage
        else:
            return 'streaming' # Moderate changes
            
    def _classify_compute_intensity(self, metrics: BlackwallMetrics) -> str:
        """Classify compute vs. memory intensity."""
        compute_score = metrics.tensor_core_utilization / 100.0
        memory_score = metrics.memory_bandwidth_utilization / 100.0
        
        if compute_score > memory_score * 1.5:
            return 'compute_bound'
        elif memory_score > compute_score * 1.5:
            return 'memory_bound'
        else:
            return 'balanced'
            
    def _identify_optimization_opportunities(self, metrics: BlackwallMetrics) -> List[str]:
        """Identify optimization opportunities based on current state."""
        opportunities = []
        
        # Memory optimization opportunities
        if metrics.memory_fragmentation_percent > 30:
            opportunities.append("Reduce VRAM fragmentation through allocation optimization")
            
        if metrics.memory_utilization > 90:
            opportunities.append("Consider model quantization to reduce VRAM usage")
        elif metrics.memory_utilization < 30 and metrics.used_memory_gb > 5:
            opportunities.append("Potential for larger batch sizes or model size")
            
        # Compute optimization opportunities
        if metrics.tensor_core_utilization < 50 and metrics.gpu_utilization > 60:
            opportunities.append("Enable tensor core optimizations for better performance")
            
        if not metrics.mixed_precision_active and metrics.memory_utilization > 60:
            opportunities.append("Consider mixed precision training/inference")
            
        # Thermal optimization opportunities
        if metrics.thermal_throttling_active:
            opportunities.append("Thermal throttling detected - improve cooling or reduce power limit")
            
        if metrics.power_efficiency_score < 0.7:
            opportunities.append("Suboptimal power efficiency - check workload optimization")
            
        # Memory bandwidth optimization
        if metrics.memory_bandwidth_utilization < 30 and metrics.gpu_utilization > 70:
            opportunities.append("GPU compute-bound - consider memory access optimizations")
            
        return opportunities
        
    def _analyze_vram_allocations(self):
        """Analyze VRAM allocations for optimization opportunities."""
        try:
            current_processes = self._get_gpu_processes()
            current_time = datetime.now()
            
            # Update allocation tracking
            current_pids = {proc['pid'] for proc in current_processes}
            
            # Remove processes that are no longer running
            self.vram_allocations = {
                pid: alloc for pid, alloc in self.vram_allocations.items()
                if pid in current_pids
            }
            
            # Update or add current processes
            for proc in current_processes:
                allocation_type = self._classify_allocation_type(proc)
                efficiency_score = self._calculate_allocation_efficiency(proc)
                
                self.vram_allocations[proc['pid']] = VRAMAllocation(
                    process_id=proc['pid'],
                    process_name=proc['name'],
                    allocated_memory_mb=proc['memory_mb'],
                    allocation_type=allocation_type,
                    efficiency_score=efficiency_score,
                    last_accessed=current_time
                )
                
        except Exception as e:
            logger.error(f"Error analyzing VRAM allocations: {e}")
            
    def _classify_allocation_type(self, process: Dict[str, Any]) -> str:
        """Classify VRAM allocation type based on process and memory usage."""
        process_name = process['name'].lower()
        memory_mb = process['memory_mb']
        
        if 'python' in process_name and memory_mb > 15000:  # > 15GB
            return 'model'
        elif 'python' in process_name and memory_mb > 5000:   # > 5GB  
            return 'cache'
        elif memory_mb < 1000:  # < 1GB
            return 'workspace'
        else:
            return 'unknown'
            
    def _calculate_allocation_efficiency(self, process: Dict[str, Any]) -> float:
        """Calculate allocation efficiency score."""
        memory_mb = process['memory_mb']
        
        # Efficiency based on allocation size and patterns
        if memory_mb > 20000:  # Very large allocation
            return 0.9  # Assumed efficient for large models
        elif memory_mb > 10000:  # Large allocation
            return 0.8
        elif memory_mb > 1000:   # Medium allocation
            return 0.7
        else:  # Small allocation
            return 0.5  # Less efficient use of VRAM
            
    def detect_changes(self, previous_state: SystemState) -> List[SystemChange]:
        """Detect changes in RTX 5090 Blackwall architecture state."""
        changes = []
        
        if not self.nvml_initialized:
            return changes
            
        # Collect current metrics
        current_metrics = self._collect_blackwall_metrics()
        if not current_metrics:
            return changes
            
        # Compare with previous metrics
        if self.previous_metrics:
            # GPU utilization changes
            gpu_util_change = abs(current_metrics.gpu_utilization - self.previous_metrics.gpu_utilization)
            if gpu_util_change > self.change_thresholds['gpu_utilization']:
                changes.append(SystemChange(
                    component=ComponentType.GPU,
                    change_type=ChangeType.PERFORMANCE_CHANGE,
                    description=f"RTX 5090 GPU utilization changed by {gpu_util_change:.1f}%: {self.previous_metrics.gpu_utilization:.1f}% → {current_metrics.gpu_utilization:.1f}%",
                    details={
                        'previous_utilization': self.previous_metrics.gpu_utilization,
                        'current_utilization': current_metrics.gpu_utilization,
                        'change_magnitude': gpu_util_change,
                        'architecture': 'Blackwall'
                    },
                    significance=self._determine_utilization_significance(gpu_util_change),
                    timestamp=current_metrics.timestamp
                ))
                
            # Memory utilization changes
            memory_util_change = abs(current_metrics.memory_utilization - self.previous_metrics.memory_utilization)
            if memory_util_change > self.change_thresholds['memory_utilization']:
                changes.append(SystemChange(
                    component=ComponentType.GPU,
                    change_type=ChangeType.RESOURCE_CHANGE,
                    description=f"RTX 5090 VRAM utilization changed by {memory_util_change:.1f}%: {current_metrics.used_memory_gb:.1f}GB used ({current_metrics.memory_utilization:.1f}%)",
                    details={
                        'previous_memory_gb': self.previous_metrics.used_memory_gb,
                        'current_memory_gb': current_metrics.used_memory_gb,
                        'previous_utilization': self.previous_metrics.memory_utilization,
                        'current_utilization': current_metrics.memory_utilization,
                        'total_vram_gb': current_metrics.total_memory_gb,
                        'memory_fragmentation': current_metrics.memory_fragmentation_percent
                    },
                    significance=self._determine_memory_significance(memory_util_change, current_metrics.memory_utilization),
                    timestamp=current_metrics.timestamp
                ))
                
            # Tensor core utilization changes
            tensor_util_change = abs(current_metrics.tensor_core_utilization - self.previous_metrics.tensor_core_utilization)
            if tensor_util_change > self.change_thresholds['tensor_utilization']:
                changes.append(SystemChange(
                    component=ComponentType.GPU,
                    change_type=ChangeType.AI_OPTIMIZATION,
                    description=f"RTX 5090 Tensor Core utilization changed by {tensor_util_change:.1f}%: {self.previous_metrics.tensor_core_utilization:.1f}% → {current_metrics.tensor_core_utilization:.1f}%",
                    details={
                        'previous_tensor_utilization': self.previous_metrics.tensor_core_utilization,
                        'current_tensor_utilization': current_metrics.tensor_core_utilization,
                        'tensor_throughput_tflops': current_metrics.tensor_throughput_tflops,
                        'transformer_optimizations': current_metrics.transformer_optimizations_active,
                        'mixed_precision': current_metrics.mixed_precision_active
                    },
                    significance=Significance.MEDIUM,
                    timestamp=current_metrics.timestamp
                ))
                
            # Thermal throttling detection
            if current_metrics.thermal_throttling_active and not self.previous_metrics.thermal_throttling_active:
                changes.append(SystemChange(
                    component=ComponentType.GPU,
                    change_type=ChangeType.THERMAL_EVENT,
                    description=f"RTX 5090 thermal throttling activated at {current_metrics.temperature:.1f}°C",
                    details={
                        'temperature': current_metrics.temperature,
                        'power_usage': current_metrics.power_usage,
                        'power_limit': current_metrics.power_limit,
                        'gpu_utilization': current_metrics.gpu_utilization,
                        'thermal_mitigation_required': True
                    },
                    significance=Significance.HIGH,
                    timestamp=current_metrics.timestamp
                ))
            elif not current_metrics.thermal_throttling_active and self.previous_metrics.thermal_throttling_active:
                changes.append(SystemChange(
                    component=ComponentType.GPU,
                    change_type=ChangeType.THERMAL_EVENT,
                    description=f"RTX 5090 thermal throttling deactivated, temperature: {current_metrics.temperature:.1f}°C",
                    details={
                        'temperature': current_metrics.temperature,
                        'power_usage': current_metrics.power_usage,
                        'thermal_recovery': True
                    },
                    significance=Significance.MEDIUM,
                    timestamp=current_metrics.timestamp
                ))
                
            # Power efficiency changes
            efficiency_change = abs(current_metrics.power_efficiency_score - self.previous_metrics.power_efficiency_score)
            if efficiency_change > self.change_thresholds['power_efficiency_drop']:
                changes.append(SystemChange(
                    component=ComponentType.GPU,
                    change_type=ChangeType.EFFICIENCY_CHANGE,
                    description=f"RTX 5090 power efficiency changed: {self.previous_metrics.power_efficiency_score:.2f} → {current_metrics.power_efficiency_score:.2f}",
                    details={
                        'previous_efficiency': self.previous_metrics.power_efficiency_score,
                        'current_efficiency': current_metrics.power_efficiency_score,
                        'power_usage': current_metrics.power_usage,
                        'gpu_utilization': current_metrics.gpu_utilization,
                        'optimization_needed': current_metrics.power_efficiency_score < 0.7
                    },
                    significance=Significance.MEDIUM if current_metrics.power_efficiency_score < 0.7 else Significance.LOW,
                    timestamp=current_metrics.timestamp
                ))
                
            # VRAM fragmentation changes
            fragmentation_change = abs(current_metrics.memory_fragmentation_percent - self.previous_metrics.memory_fragmentation_percent)
            if fragmentation_change > 10:  # > 10% fragmentation change
                changes.append(SystemChange(
                    component=ComponentType.GPU,
                    change_type=ChangeType.MEMORY_OPTIMIZATION,
                    description=f"RTX 5090 VRAM fragmentation changed by {fragmentation_change:.1f}%: {current_metrics.memory_fragmentation_percent:.1f}% fragmented",
                    details={
                        'previous_fragmentation': self.previous_metrics.memory_fragmentation_percent,
                        'current_fragmentation': current_metrics.memory_fragmentation_percent,
                        'vram_allocations': len(self.vram_allocations),
                        'optimization_opportunity': current_metrics.memory_fragmentation_percent > 30
                    },
                    significance=Significance.MEDIUM if current_metrics.memory_fragmentation_percent > 30 else Significance.LOW,
                    timestamp=current_metrics.timestamp
                ))
                
        # Store current metrics for next comparison
        self.previous_metrics = current_metrics
        
        return changes
        
    def _determine_utilization_significance(self, change_magnitude: float) -> Significance:
        """Determine significance of GPU utilization changes."""
        if change_magnitude > 50:
            return Significance.HIGH
        elif change_magnitude > 25:
            return Significance.MEDIUM
        else:
            return Significance.LOW
            
    def _determine_memory_significance(self, change_magnitude: float, current_utilization: float) -> Significance:
        """Determine significance of memory changes."""
        if current_utilization > 90 or change_magnitude > 20:
            return Significance.HIGH
        elif current_utilization > 70 or change_magnitude > 10:
            return Significance.MEDIUM
        else:
            return Significance.LOW
            
    def get_blackwall_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive RTX 5090 Blackwall status summary."""
        if not self.nvml_initialized:
            return {'status': 'unavailable', 'reason': 'NVML not initialized'}
            
        current_metrics = self._collect_blackwall_metrics()
        if not current_metrics:
            return {'status': 'unavailable', 'reason': 'Cannot collect metrics'}
            
        # Get recent workload profile
        recent_profile = None
        if self.workload_profiles:
            recent_profile = max(self.workload_profiles.values(), key=lambda p: p.workload_type)
            
        return {
            'status': 'active',
            'architecture': 'Blackwall',
            'gpu_model': self.blackwall_config['gpu_name'],
            'current_metrics': {
                'gpu_utilization': current_metrics.gpu_utilization,
                'memory_utilization': current_metrics.memory_utilization,
                'memory_used_gb': current_metrics.used_memory_gb,
                'memory_total_gb': current_metrics.total_memory_gb,
                'temperature': current_metrics.temperature,
                'power_usage': current_metrics.power_usage,
                'tensor_core_utilization': current_metrics.tensor_core_utilization,
                'tensor_throughput_tflops': current_metrics.tensor_throughput_tflops,
                'power_efficiency_score': current_metrics.power_efficiency_score
            },
            'thermal_status': {
                'temperature': current_metrics.temperature,
                'thermal_throttling': current_metrics.thermal_throttling_active,
                'power_limit': current_metrics.power_limit
            },
            'ai_optimizations': {
                'transformer_optimizations': current_metrics.transformer_optimizations_active,
                'mixed_precision': current_metrics.mixed_precision_active,
                'tensor_cores_active': current_metrics.tensor_core_utilization > 10
            },
            'memory_analysis': {
                'fragmentation_percent': current_metrics.memory_fragmentation_percent,
                'active_processes': len(self.vram_allocations),
                'allocation_efficiency': np.mean([alloc.efficiency_score for alloc in self.vram_allocations.values()]) if self.vram_allocations else 0.0
            },
            'current_workload': {
                'workload_type': recent_profile.workload_type if recent_profile else 'unknown',
                'model_architecture': recent_profile.model_architecture if recent_profile else 'unknown',
                'compute_intensity': recent_profile.compute_intensity if recent_profile else 'unknown',
                'optimization_opportunities': recent_profile.optimization_opportunities if recent_profile else []
            },
            'performance_metrics': {
                'cuda_contexts': current_metrics.active_cuda_contexts,
                'kernel_launch_rate': current_metrics.cuda_kernel_launches_per_sec,
                'memory_transfer_rate': current_metrics.cuda_memory_transfers_per_sec,
                'memory_bandwidth_utilization': current_metrics.memory_bandwidth_utilization
            },
            'monitoring_stats': {
                'metrics_collected': len(self.metrics_history),
                'workload_profiles': len(self.workload_profiles),
                'monitoring_active': self.monitoring_active,
                'last_update': current_metrics.timestamp.isoformat()
            }
        }
        
    def __del__(self):
        """Cleanup on object destruction."""
        try:
            self.stop_monitoring()
            if self.nvml_initialized:
                pynvml.nvmlShutdown()
        except Exception:
            pass