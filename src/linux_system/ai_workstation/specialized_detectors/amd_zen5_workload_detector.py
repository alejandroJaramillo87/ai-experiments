"""
AMD Zen 5 Workload Intelligence Detector - Advanced CPU Architecture Monitoring
===============================================================================

Specialized detector for AMD Zen 5 (9950X) architecture with deep understanding
of the 16-core, 32-thread configuration optimized for AI workstation workloads.
Provides comprehensive CPU intelligence including core pinning analysis, AOCL
mathematical library utilization, memory bandwidth optimization, and workload
correlation with GPU inference tasks.

Key Capabilities:
- Zen 5 architecture-specific telemetry and performance analysis
- Core pinning effectiveness for containerized AI services
- AOCL (AMD Optimizing C/C++ Compiler Libraries) performance tracking
- 128GB DDR5-6000 memory bandwidth utilization analysis
- CPU vs GPU inference efficiency comparison and optimization
- Multi-threading analysis for concurrent AI workload execution
- Thermal and power management for sustained high-performance computing
- NUMA topology optimization for large model processing
"""

import asyncio
import json
import logging
import subprocess
import psutil
import os
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
from pathlib import Path

from ..base_collector import BaseCollector
from ...temporal.types import SystemChange, ChangeType

logger = logging.getLogger(__name__)


@dataclass
class Zen5CoreMetrics:
    """Individual CPU core metrics for Zen 5 architecture."""
    core_id: int
    physical_id: int
    utilization: float
    frequency_mhz: int
    temperature_c: float
    power_draw_w: Optional[float]
    cache_misses: Optional[int]
    instructions_per_cycle: Optional[float]
    branch_misses: Optional[int]
    pinned_processes: List[str]
    workload_type: str  # 'idle', 'ai_inference', 'system', 'mixed'


@dataclass
class AOCLPerformanceMetrics:
    """AMD Optimizing C/C++ Compiler Libraries performance metrics."""
    blas_operations: int
    lapack_operations: int
    fftw_operations: int
    vector_operations: int
    mathematical_throughput_gflops: float
    cache_efficiency: float
    vectorization_effectiveness: float
    optimization_level: str
    library_version: str


@dataclass
class MemoryBandwidthAnalysis:
    """DDR5-6000 memory bandwidth analysis for large AI models."""
    total_bandwidth_gb_s: float
    achieved_bandwidth_gb_s: float
    bandwidth_utilization: float
    memory_channels_active: int
    numa_efficiency: float
    large_page_utilization: float
    memory_latency_ns: float
    cache_hierarchy_efficiency: Dict[str, float]


@dataclass
class WorkloadCorrelation:
    """CPU-GPU workload correlation analysis."""
    cpu_inference_efficiency: float
    gpu_inference_efficiency: float
    load_balance_score: float
    resource_contention_detected: bool
    optimal_workload_distribution: Dict[str, float]
    bottleneck_analysis: List[str]


@dataclass
class CorePinningAnalysis:
    """Container core pinning effectiveness analysis."""
    container_name: str
    assigned_cores: List[int]
    actual_core_usage: List[int]
    pinning_effectiveness: float
    core_migration_events: int
    numa_compliance: bool
    performance_impact: str


class AMDZen5WorkloadDetector:
    """
    Specialized detector for AMD Zen 5 workload intelligence.
    
    Monitors CPU architecture optimization for AI workstation workloads,
    including core pinning, mathematical library utilization, and
    memory bandwidth optimization for large model processing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize AMD Zen 5 workload detector."""
        self.config = config or {}
        
        # Zen 5 (9950X) architecture specifications
        self.arch_specs = {
            'architecture': 'Zen 5',
            'model': '9950X',
            'cores': 16,
            'threads': 32,
            'base_frequency_ghz': 4.3,
            'boost_frequency_ghz': 5.7,
            'l1_cache_kb': 64,  # per core (32KB I + 32KB D)
            'l2_cache_kb': 1024,  # per core
            'l3_cache_mb': 64,  # total
            'memory_channels': 2,
            'memory_speed_mhz': 6000,  # DDR5-6000
            'memory_capacity_gb': 128,
            'tdp_w': 170,
            'max_temp_c': 95,
            'numa_nodes': 2,  # Typical dual-NUMA configuration
            'pcie_lanes': 24
        }
        
        # AI workstation service core assignments (from docker-compose)
        self.ai_service_pinning = {
            'llama-cpu-1': [0, 1, 2, 3, 4, 5, 6, 7],
            'llama-cpu-2': [8, 9, 10, 11, 12, 13, 14, 15],
            'llama-cpu-3': [16, 17, 18, 19, 20, 21, 22, 23],  # If hyperthreading
            'system_reserved': [24, 25, 26, 27, 28, 29, 30, 31]
        }
        
        # Performance thresholds
        self.thresholds = {
            'cpu_high': 80.0,
            'cpu_critical': 95.0,
            'temperature_warning': 80,
            'temperature_critical': 90,
            'memory_bandwidth_low': 60.0,
            'core_pinning_ineffective': 70.0,
            'numa_efficiency_low': 80.0,
            'cache_miss_high': 10.0,  # percentage
            'thermal_throttle_temp': 93
        }
        
        # Historical data for analysis
        self.core_history = deque(maxlen=60)  # 1 hour at 1-minute intervals
        self.memory_history = deque(maxlen=30)
        self.workload_history = deque(maxlen=20)
        
        # AOCL detection and monitoring
        self.aocl_detected = False
        self.aocl_version = None
        
        logger.info("AMDZen5WorkloadDetector initialized for 9950X AI workstation")
    
    async def collect_zen5_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive AMD Zen 5 workload metrics."""
        try:
            # Core CPU metrics collection
            cpu_metrics = await self._collect_cpu_core_metrics()
            
            # Memory bandwidth analysis
            memory_analysis = await self._analyze_memory_bandwidth()
            
            # AOCL library performance
            aocl_metrics = await self._collect_aocl_metrics()
            
            # Core pinning effectiveness
            pinning_analysis = await self._analyze_core_pinning()
            
            # Workload correlation analysis
            workload_correlation = await self._analyze_workload_correlation()
            
            # NUMA topology optimization
            numa_analysis = await self._analyze_numa_optimization()
            
            # Thermal and power analysis
            thermal_power = await self._analyze_thermal_power()
            
            # AI workload optimization recommendations
            optimization_recommendations = await self._generate_optimization_recommendations(
                cpu_metrics, memory_analysis, pinning_analysis
            )
            
            return {
                'cpu_core_metrics': cpu_metrics,
                'memory_bandwidth_analysis': memory_analysis,
                'aocl_performance': aocl_metrics,
                'core_pinning_analysis': pinning_analysis,
                'workload_correlation': workload_correlation,
                'numa_optimization': numa_analysis,
                'thermal_power_analysis': thermal_power,
                'optimization_recommendations': optimization_recommendations,
                'architecture_specifications': self.arch_specs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error collecting Zen5 metrics: {e}")
            return {'error': str(e)}
    
    async def _collect_cpu_core_metrics(self) -> Dict[str, Any]:
        """Collect detailed per-core metrics for Zen 5 architecture."""
        try:
            core_metrics = {}
            
            # Get per-CPU utilization
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            cpu_freq = psutil.cpu_freq(percpu=True)
            cpu_times = psutil.cpu_times(percpu=True)
            
            # Get thermal data from system sensors
            thermal_data = await self._get_cpu_thermal_data()
            
            cores_data = []
            for i, (util, freq_info, times) in enumerate(zip(cpu_percent, cpu_freq, cpu_times)):
                core_temp = thermal_data.get(f'core_{i}', 0.0) if thermal_data else 0.0
                
                # Determine workload type based on utilization and process analysis
                workload_type = self._classify_core_workload(i, util)
                
                # Get pinned processes for this core
                pinned_processes = self._get_pinned_processes_for_core(i)
                
                core_data = Zen5CoreMetrics(
                    core_id=i,
                    physical_id=i // 2,  # Zen 5 has 2 threads per core
                    utilization=util,
                    frequency_mhz=int(freq_info.current) if freq_info else 0,
                    temperature_c=core_temp,
                    power_draw_w=None,  # Would need specialized monitoring
                    cache_misses=None,  # Would need perf counters
                    instructions_per_cycle=None,
                    branch_misses=None,
                    pinned_processes=pinned_processes,
                    workload_type=workload_type
                )
                
                cores_data.append(core_data)
            
            # Calculate aggregate metrics
            total_utilization = sum(c.utilization for c in cores_data) / len(cores_data)
            max_frequency = max(c.frequency_mhz for c in cores_data)
            avg_temperature = sum(c.temperature_c for c in cores_data) / len(cores_data)
            
            # AI service core utilization analysis
            ai_service_utilization = {}
            for service, cores in self.ai_service_pinning.items():
                if service != 'system_reserved':
                    service_cores = [cores_data[i] for i in cores if i < len(cores_data)]
                    if service_cores:
                        avg_util = sum(c.utilization for c in service_cores) / len(service_cores)
                        ai_service_utilization[service] = {
                            'average_utilization': avg_util,
                            'assigned_cores': cores,
                            'active_cores': len([c for c in service_cores if c.utilization > 5]),
                            'workload_types': [c.workload_type for c in service_cores]
                        }
            
            core_metrics = {
                'individual_cores': [self._core_metrics_to_dict(c) for c in cores_data],
                'aggregate_metrics': {
                    'total_utilization': round(total_utilization, 2),
                    'max_frequency_mhz': max_frequency,
                    'average_temperature_c': round(avg_temperature, 2),
                    'active_cores': len([c for c in cores_data if c.utilization > 5])
                },
                'ai_service_utilization': ai_service_utilization,
                'architecture_utilization': {
                    'physical_cores_active': len(set(c.physical_id for c in cores_data if c.utilization > 5)),
                    'hyperthreading_effectiveness': self._calculate_hyperthreading_effectiveness(cores_data),
                    'load_distribution_variance': self._calculate_load_variance(cores_data)
                }
            }
            
            # Store for historical analysis
            self.core_history.append({
                'timestamp': datetime.now(),
                'total_utilization': total_utilization,
                'max_frequency': max_frequency,
                'avg_temperature': avg_temperature
            })
            
            return core_metrics
            
        except Exception as e:
            logger.error(f"Error collecting CPU core metrics: {e}")
            return {'error': str(e)}
    
    async def _get_cpu_thermal_data(self) -> Dict[str, float]:
        """Get CPU thermal data from system sensors."""
        thermal_data = {}
        
        try:
            # Try to get thermal data from psutil
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                
                # Look for CPU temperature sensors
                for name, entries in temps.items():
                    if any(keyword in name.lower() for keyword in ['cpu', 'core', 'package']):
                        for i, entry in enumerate(entries):
                            if 'core' in entry.label.lower():
                                core_num = re.search(r'(\d+)', entry.label)
                                if core_num:
                                    thermal_data[f'core_{core_num.group(1)}'] = entry.current
                            elif 'package' in entry.label.lower() or 'cpu' in entry.label.lower():
                                thermal_data['package'] = entry.current
            
        except Exception as e:
            logger.warning(f"Could not get thermal data: {e}")
        
        return thermal_data
    
    def _classify_core_workload(self, core_id: int, utilization: float) -> str:
        """Classify workload type for a CPU core."""
        if utilization < 5:
            return 'idle'
        elif utilization > 80:
            # Check if this core is assigned to AI services
            for service, cores in self.ai_service_pinning.items():
                if core_id in cores and service.startswith('llama-cpu'):
                    return 'ai_inference'
            return 'high_load'
        elif utilization > 30:
            return 'active'
        else:
            return 'light_load'
    
    def _get_pinned_processes_for_core(self, core_id: int) -> List[str]:
        """Get processes pinned to a specific core."""
        pinned_processes = []
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_affinity']):
                try:
                    affinity = proc.info['cpu_affinity']
                    if affinity and core_id in affinity:
                        pinned_processes.append(proc.info['name'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.warning(f"Could not get process affinity: {e}")
        
        return pinned_processes
    
    def _calculate_hyperthreading_effectiveness(self, cores_data: List[Zen5CoreMetrics]) -> float:
        """Calculate hyperthreading effectiveness."""
        if len(cores_data) < self.arch_specs['cores'] * 2:
            return 0.0
        
        physical_core_utils = {}
        for core in cores_data:
            phys_id = core.physical_id
            if phys_id not in physical_core_utils:
                physical_core_utils[phys_id] = []
            physical_core_utils[phys_id].append(core.utilization)
        
        effectiveness_scores = []
        for phys_id, utils in physical_core_utils.items():
            if len(utils) == 2:  # Should have 2 logical cores per physical
                # Effectiveness is higher when both threads are balanced
                balance = 1 - abs(utils[0] - utils[1]) / max(utils[0] + utils[1], 1)
                effectiveness_scores.append(balance)
        
        return sum(effectiveness_scores) / len(effectiveness_scores) * 100 if effectiveness_scores else 0.0
    
    def _calculate_load_variance(self, cores_data: List[Zen5CoreMetrics]) -> float:
        """Calculate load distribution variance across cores."""
        utilizations = [c.utilization for c in cores_data]
        mean_util = sum(utilizations) / len(utilizations)
        variance = sum((u - mean_util) ** 2 for u in utilizations) / len(utilizations)
        return variance
    
    async def _analyze_memory_bandwidth(self) -> Dict[str, Any]:
        """Analyze DDR5-6000 memory bandwidth utilization."""
        try:
            memory_analysis = {
                'memory_specifications': {
                    'type': 'DDR5-6000',
                    'channels': self.arch_specs['memory_channels'],
                    'capacity_gb': self.arch_specs['memory_capacity_gb'],
                    'theoretical_bandwidth_gb_s': 96.0  # DDR5-6000 dual channel
                }
            }
            
            # Get memory statistics
            mem_stats = psutil.virtual_memory()
            swap_stats = psutil.swap_memory()
            
            # Calculate memory bandwidth utilization (approximation)
            memory_utilization = mem_stats.percent
            available_bandwidth = memory_analysis['memory_specifications']['theoretical_bandwidth_gb_s']
            estimated_bandwidth_usage = available_bandwidth * (memory_utilization / 100.0)
            
            memory_analysis.update({
                'current_utilization': {
                    'used_gb': mem_stats.used / (1024**3),
                    'available_gb': mem_stats.available / (1024**3),
                    'utilization_percent': memory_utilization,
                    'swap_used_gb': swap_stats.used / (1024**3),
                    'swap_utilization_percent': swap_stats.percent
                },
                'bandwidth_analysis': {
                    'estimated_bandwidth_usage_gb_s': round(estimated_bandwidth_usage, 2),
                    'bandwidth_efficiency': round((estimated_bandwidth_usage / available_bandwidth) * 100, 2),
                    'bandwidth_headroom_gb_s': round(available_bandwidth - estimated_bandwidth_usage, 2)
                }
            })
            
            # Large page analysis (important for AI workloads)
            try:
                # Check hugepage usage
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                
                hugepage_size = re.search(r'Hugepagesize:\s*(\d+)\s*kB', meminfo)
                hugepage_total = re.search(r'HugePages_Total:\s*(\d+)', meminfo)
                hugepage_free = re.search(r'HugePages_Free:\s*(\d+)', meminfo)
                
                if hugepage_size and hugepage_total and hugepage_free:
                    hugepage_size_kb = int(hugepage_size.group(1))
                    hugepage_total_count = int(hugepage_total.group(1))
                    hugepage_free_count = int(hugepage_free.group(1))
                    hugepage_used_count = hugepage_total_count - hugepage_free_count
                    
                    memory_analysis['large_page_analysis'] = {
                        'hugepage_size_mb': hugepage_size_kb / 1024,
                        'hugepage_total': hugepage_total_count,
                        'hugepage_used': hugepage_used_count,
                        'hugepage_utilization_percent': (hugepage_used_count / hugepage_total_count * 100) if hugepage_total_count > 0 else 0,
                        'total_hugepage_memory_gb': (hugepage_total_count * hugepage_size_kb) / (1024 * 1024)
                    }
                
            except Exception as e:
                logger.warning(f"Could not analyze large pages: {e}")
            
            # NUMA analysis
            numa_analysis = await self._analyze_numa_memory_distribution()
            if numa_analysis:
                memory_analysis['numa_memory_distribution'] = numa_analysis
            
            # Store for historical analysis
            self.memory_history.append({
                'timestamp': datetime.now(),
                'utilization_percent': memory_utilization,
                'estimated_bandwidth_usage': estimated_bandwidth_usage
            })
            
            return memory_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing memory bandwidth: {e}")
            return {'error': str(e)}
    
    async def _analyze_numa_memory_distribution(self) -> Optional[Dict[str, Any]]:
        """Analyze NUMA memory distribution."""
        try:
            numa_info = {}
            
            # Check if numactl is available
            result = subprocess.run(['numactl', '--hardware'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                current_node = None
                
                for line in lines:
                    if line.startswith('node'):
                        if 'size:' in line:
                            # Parse node memory size
                            parts = line.split()
                            node_id = parts[1]
                            size_info = ' '.join(parts[3:])  # size info
                            numa_info[f'node_{node_id}'] = {
                                'size': size_info
                            }
                        elif 'cpus:' in line:
                            # Parse node CPU assignments
                            parts = line.split(':')
                            if len(parts) >= 2:
                                node_id = parts[0].split()[1]
                                cpus = parts[1].strip()
                                if f'node_{node_id}' in numa_info:
                                    numa_info[f'node_{node_id}']['cpus'] = cpus
                
                return numa_info
            
        except Exception as e:
            logger.warning(f"Could not analyze NUMA: {e}")
        
        return None
    
    async def _collect_aocl_metrics(self) -> Dict[str, Any]:
        """Collect AMD Optimizing C/C++ Compiler Libraries performance metrics."""
        try:
            aocl_metrics = {
                'aocl_detected': False,
                'version': 'unknown',
                'performance_analysis': {}
            }
            
            # Check for AOCL installation and usage
            aocl_paths = [
                '/opt/AMD/aocl',
                '/usr/local/aocl',
                '/opt/aocl'
            ]
            
            for path in aocl_paths:
                if os.path.exists(path):
                    aocl_metrics['aocl_detected'] = True
                    aocl_metrics['installation_path'] = path
                    break
            
            # Check environment variables for AOCL
            aocl_env_vars = ['AOCL_ROOT', 'BLIS_ROOT', 'LIBFLAME_ROOT']
            aocl_env = {}
            for var in aocl_env_vars:
                if var in os.environ:
                    aocl_env[var] = os.environ[var]
                    aocl_metrics['aocl_detected'] = True
            
            if aocl_env:
                aocl_metrics['environment_variables'] = aocl_env
            
            # Look for AOCL library usage in running processes
            aocl_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
                try:
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    
                    # Check for AOCL library references
                    aocl_indicators = ['libblis', 'libflame', 'aocl', 'blas', 'lapack']
                    if any(indicator in cmdline.lower() for indicator in aocl_indicators):
                        aocl_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_percent': proc.info['memory_percent'],
                            'cmdline_snippet': cmdline[:200]  # First 200 chars
                        })
                        aocl_metrics['aocl_detected'] = True
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if aocl_processes:
                aocl_metrics['active_aocl_processes'] = aocl_processes
                
                # Estimate mathematical throughput based on active processes
                total_cpu_usage = sum(p['cpu_percent'] for p in aocl_processes)
                estimated_gflops = self._estimate_mathematical_throughput(total_cpu_usage)
                
                aocl_metrics['performance_analysis'] = {
                    'active_processes_count': len(aocl_processes),
                    'total_cpu_usage_percent': round(total_cpu_usage, 2),
                    'estimated_mathematical_throughput_gflops': estimated_gflops,
                    'optimization_effectiveness': 'high' if total_cpu_usage > 50 else 'medium' if total_cpu_usage > 20 else 'low'
                }
            
            # Check for mathematical library optimization flags in compiler usage
            optimization_hints = await self._detect_optimization_flags()
            if optimization_hints:
                aocl_metrics['compiler_optimizations'] = optimization_hints
            
            return aocl_metrics
            
        except Exception as e:
            logger.error(f"Error collecting AOCL metrics: {e}")
            return {'error': str(e)}
    
    def _estimate_mathematical_throughput(self, cpu_usage: float) -> float:
        """Estimate mathematical throughput based on CPU usage and Zen 5 capabilities."""
        # Zen 5 estimated peak GFLOPS for mathematical operations
        # This is a rough approximation based on architecture specs
        base_frequency_ghz = self.arch_specs['base_frequency_ghz']
        cores = self.arch_specs['cores']
        
        # Estimated GFLOPS per core for mathematical operations
        gflops_per_core = base_frequency_ghz * 8  # Approximate for vectorized operations
        peak_gflops = cores * gflops_per_core
        
        # Scale by CPU usage
        estimated_gflops = peak_gflops * (cpu_usage / 100.0)
        
        return round(estimated_gflops, 2)
    
    async def _detect_optimization_flags(self) -> Optional[Dict[str, Any]]:
        """Detect compiler optimization flags in use."""
        try:
            optimization_info = {}
            
            # Check for common optimization flags in environment
            optimization_vars = ['CFLAGS', 'CXXFLAGS', 'FFLAGS', 'LDFLAGS']
            for var in optimization_vars:
                if var in os.environ:
                    value = os.environ[var]
                    if any(flag in value for flag in ['-march=', '-mtune=', '-O2', '-O3', '-Ofast']):
                        optimization_info[var] = value
            
            # Look for Zen 5 specific optimizations
            zen5_optimizations = ['-march=znver5', '-mtune=znver5', '-mavx512f']
            
            detected_optimizations = []
            for var, value in optimization_info.items():
                for opt in zen5_optimizations:
                    if opt in value:
                        detected_optimizations.append(opt)
            
            if detected_optimizations:
                optimization_info['zen5_optimizations_detected'] = detected_optimizations
            
            return optimization_info if optimization_info else None
            
        except Exception as e:
            logger.warning(f"Could not detect optimization flags: {e}")
            return None
    
    async def _analyze_core_pinning(self) -> Dict[str, Any]:
        """Analyze core pinning effectiveness for AI containers."""
        try:
            pinning_analysis = {
                'ai_service_pinning': {},
                'pinning_effectiveness_score': 0.0,
                'violations_detected': [],
                'optimization_recommendations': []
            }
            
            # Analyze each AI service's core pinning
            for service_name, assigned_cores in self.ai_service_pinning.items():
                if service_name == 'system_reserved':
                    continue
                
                service_analysis = {
                    'assigned_cores': assigned_cores,
                    'compliance': 'unknown',
                    'migration_events': 0,
                    'effectiveness_score': 0.0
                }
                
                # Look for container processes matching the service
                service_processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_affinity', 'cpu_percent']):
                    try:
                        cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                        
                        # Check if process belongs to this service
                        if (service_name.replace('-', '_') in cmdline or 
                            service_name in proc.info['name'] or
                            any(service_name.split('-')[0] in part for part in cmdline.split())):
                            
                            affinity = proc.info['cpu_affinity']
                            if affinity:
                                service_processes.append({
                                    'pid': proc.info['pid'],
                                    'name': proc.info['name'],
                                    'cpu_affinity': affinity,
                                    'cpu_percent': proc.info['cpu_percent']
                                })
                    
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                if service_processes:
                    # Analyze core affinity compliance
                    total_compliance = 0
                    for proc in service_processes:
                        affinity_set = set(proc['cpu_affinity'])
                        assigned_set = set(assigned_cores)
                        
                        if affinity_set.issubset(assigned_set):
                            compliance = 100.0
                        else:
                            overlap = len(affinity_set.intersection(assigned_set))
                            compliance = (overlap / len(assigned_set)) * 100
                        
                        total_compliance += compliance
                    
                    service_analysis['effectiveness_score'] = total_compliance / len(service_processes)
                    service_analysis['compliance'] = 'good' if service_analysis['effectiveness_score'] > 80 else 'poor'
                    service_analysis['active_processes'] = len(service_processes)
                    
                    # Check for violations
                    if service_analysis['effectiveness_score'] < self.thresholds['core_pinning_ineffective']:
                        pinning_analysis['violations_detected'].append({
                            'service': service_name,
                            'issue': 'ineffective_pinning',
                            'score': service_analysis['effectiveness_score']
                        })
                
                pinning_analysis['ai_service_pinning'][service_name] = service_analysis
            
            # Calculate overall pinning effectiveness
            service_scores = [s['effectiveness_score'] for s in pinning_analysis['ai_service_pinning'].values() 
                            if s['effectiveness_score'] > 0]
            if service_scores:
                pinning_analysis['pinning_effectiveness_score'] = sum(service_scores) / len(service_scores)
            
            # Generate optimization recommendations
            if pinning_analysis['pinning_effectiveness_score'] < self.thresholds['core_pinning_ineffective']:
                pinning_analysis['optimization_recommendations'].extend([
                    'Review container CPU affinity settings',
                    'Ensure proper core isolation in kernel parameters',
                    'Consider adjusting container resource limits'
                ])
            
            return pinning_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing core pinning: {e}")
            return {'error': str(e)}
    
    async def _analyze_workload_correlation(self) -> Dict[str, Any]:
        """Analyze CPU-GPU workload correlation."""
        try:
            correlation_analysis = {
                'cpu_gpu_balance': 'analyzing',
                'resource_distribution': {},
                'bottleneck_analysis': [],
                'optimization_opportunities': []
            }
            
            # Get current CPU utilization
            cpu_util = psutil.cpu_percent(interval=1)
            memory_util = psutil.virtual_memory().percent
            
            # Estimate GPU utilization (would need integration with RTX5090 detector)
            # For now, use placeholder logic
            correlation_analysis['current_metrics'] = {
                'cpu_utilization': cpu_util,
                'memory_utilization': memory_util,
                'estimated_gpu_utilization': 'unknown'  # Would integrate with GPU detector
            }
            
            # Analyze resource distribution across AI services
            ai_service_cpu = {}
            total_ai_cpu = 0
            
            for service_name, assigned_cores in self.ai_service_pinning.items():
                if service_name == 'system_reserved':
                    continue
                
                # Get CPU usage for assigned cores
                cpu_percents = psutil.cpu_percent(interval=0.1, percpu=True)
                service_cpu = sum(cpu_percents[i] for i in assigned_cores if i < len(cpu_percents))
                service_avg = service_cpu / len(assigned_cores)
                
                ai_service_cpu[service_name] = service_avg
                total_ai_cpu += service_avg
            
            correlation_analysis['resource_distribution'] = {
                'ai_service_cpu_utilization': ai_service_cpu,
                'total_ai_cpu_utilization': total_ai_cpu,
                'cpu_services_active': len([s for s, util in ai_service_cpu.items() if util > 10])
            }
            
            # Detect bottlenecks
            if cpu_util > 90:
                correlation_analysis['bottleneck_analysis'].append('CPU bottleneck detected')
            
            if memory_util > 90:
                correlation_analysis['bottleneck_analysis'].append('Memory bottleneck detected')
            
            # Load balance analysis
            if ai_service_cpu:
                cpu_loads = list(ai_service_cpu.values())
                load_variance = max(cpu_loads) - min(cpu_loads)
                
                correlation_analysis['load_balance'] = {
                    'load_variance': load_variance,
                    'balance_quality': 'good' if load_variance < 30 else 'poor'
                }
                
                if load_variance > 40:
                    correlation_analysis['optimization_opportunities'].append({
                        'type': 'load_balancing',
                        'description': 'Uneven load distribution across CPU services',
                        'recommendation': 'Consider request routing optimization'
                    })
            
            return correlation_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing workload correlation: {e}")
            return {'error': str(e)}
    
    async def _analyze_numa_optimization(self) -> Dict[str, Any]:
        """Analyze NUMA topology optimization."""
        try:
            numa_analysis = {
                'numa_topology': 'analyzing',
                'optimization_status': 'unknown',
                'recommendations': []
            }
            
            # Check NUMA topology
            try:
                result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'NUMA node(s):' in line:
                            numa_nodes = line.split(':')[1].strip()
                            numa_analysis['numa_nodes_count'] = int(numa_nodes)
                        elif 'NUMA node0 CPU(s):' in line:
                            node0_cpus = line.split(':')[1].strip()
                            numa_analysis['node0_cpus'] = node0_cpus
                        elif 'NUMA node1 CPU(s):' in line:
                            node1_cpus = line.split(':')[1].strip()
                            numa_analysis['node1_cpus'] = node1_cpus
            
            except Exception as e:
                logger.warning(f"Could not get NUMA topology: {e}")
            
            # Analyze NUMA efficiency for AI services
            if 'numa_nodes_count' in numa_analysis and numa_analysis['numa_nodes_count'] > 1:
                # Check if AI services are NUMA-aware
                numa_violations = []
                
                for service_name, assigned_cores in self.ai_service_pinning.items():
                    if service_name == 'system_reserved':
                        continue
                    
                    # Check if cores span multiple NUMA nodes (assuming 16 cores per node for 9950X)
                    if len(assigned_cores) > 1:
                        min_core = min(assigned_cores)
                        max_core = max(assigned_cores)
                        
                        # Simple heuristic: if cores span more than 16 positions, likely crosses NUMA
                        if max_core - min_core > 15:
                            numa_violations.append(service_name)
                
                if numa_violations:
                    numa_analysis['numa_violations'] = numa_violations
                    numa_analysis['recommendations'].extend([
                        'Review core assignments to minimize NUMA boundary crossings',
                        'Consider binding services to single NUMA nodes',
                        'Optimize memory allocation for NUMA locality'
                    ])
                    numa_analysis['optimization_status'] = 'needs_improvement'
                else:
                    numa_analysis['optimization_status'] = 'good'
            
            return numa_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing NUMA optimization: {e}")
            return {'error': str(e)}
    
    async def _analyze_thermal_power(self) -> Dict[str, Any]:
        """Analyze thermal and power behavior."""
        try:
            thermal_power = {
                'thermal_status': 'monitoring',
                'power_efficiency': 'analyzing',
                'sustainability_analysis': {}
            }
            
            # Get thermal data
            thermal_data = await self._get_cpu_thermal_data()
            if thermal_data:
                max_temp = max(thermal_data.values()) if thermal_data.values() else 0
                avg_temp = sum(thermal_data.values()) / len(thermal_data) if thermal_data.values() else 0
                
                thermal_power.update({
                    'current_temperatures': thermal_data,
                    'max_temperature_c': round(max_temp, 2),
                    'average_temperature_c': round(avg_temp, 2)
                })
                
                # Thermal status assessment
                if max_temp >= self.thresholds['thermal_throttle_temp']:
                    thermal_status = 'critical'
                    thermal_power['immediate_action_required'] = True
                elif max_temp >= self.thresholds['temperature_critical']:
                    thermal_status = 'warning'
                elif max_temp >= self.thresholds['temperature_warning']:
                    thermal_status = 'elevated'
                else:
                    thermal_status = 'normal'
                
                thermal_power['thermal_status'] = thermal_status
            
            # Power efficiency analysis
            cpu_util = psutil.cpu_percent(interval=1)
            if cpu_util > 0:
                # Estimate power efficiency (performance per watt)
                # This is approximate since we don't have direct power measurements
                theoretical_max_power = self.arch_specs['tdp_w']
                estimated_power = theoretical_max_power * (cpu_util / 100.0)
                efficiency_score = cpu_util / (estimated_power / theoretical_max_power) if estimated_power > 0 else 0
                
                thermal_power['power_efficiency'] = {
                    'estimated_power_draw_w': round(estimated_power, 2),
                    'efficiency_score': round(efficiency_score, 2),
                    'power_per_performance_ratio': round(estimated_power / max(cpu_util, 1), 2)
                }
            
            # Sustainability analysis for continuous AI workloads
            thermal_power['sustainability_analysis'] = {
                'continuous_workload_suitable': max_temp < self.thresholds['temperature_warning'] if 'max_temp' in locals() else True,
                'thermal_headroom_c': self.arch_specs['max_temp_c'] - max_temp if 'max_temp' in locals() else 0,
                'recommended_max_utilization_percent': min(100, (self.thresholds['temperature_warning'] / max(max_temp, 1)) * cpu_util) if 'max_temp' in locals() else 100
            }
            
            return thermal_power
            
        except Exception as e:
            logger.error(f"Error analyzing thermal/power: {e}")
            return {'error': str(e)}
    
    async def _generate_optimization_recommendations(self, 
                                                   cpu_metrics: Dict[str, Any],
                                                   memory_analysis: Dict[str, Any],
                                                   pinning_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive optimization recommendations."""
        recommendations = []
        
        # CPU utilization optimization
        if 'aggregate_metrics' in cpu_metrics:
            total_util = cpu_metrics['aggregate_metrics']['total_utilization']
            
            if total_util > 90:
                recommendations.append({
                    'category': 'performance',
                    'priority': 'high',
                    'title': 'CPU Utilization Critical',
                    'description': f'CPU utilization at {total_util}%',
                    'actions': [
                        'Consider scaling out AI workloads',
                        'Review container resource limits',
                        'Implement workload scheduling'
                    ]
                })
            elif total_util < 30:
                recommendations.append({
                    'category': 'efficiency',
                    'priority': 'medium',
                    'title': 'CPU Underutilization',
                    'description': f'CPU utilization only {total_util}%',
                    'actions': [
                        'Consider increasing AI workload intensity',
                        'Optimize power management for efficiency',
                        'Scale down unused services'
                    ]
                })
        
        # Memory optimization
        if 'current_utilization' in memory_analysis:
            mem_util = memory_analysis['current_utilization']['utilization_percent']
            
            if mem_util > 90:
                recommendations.append({
                    'category': 'memory',
                    'priority': 'critical',
                    'title': 'Memory Pressure Critical',
                    'description': f'Memory utilization at {mem_util}%',
                    'actions': [
                        'Review memory allocation for AI models',
                        'Consider memory optimization techniques',
                        'Monitor for memory leaks in AI services'
                    ]
                })
        
        # Core pinning optimization
        if pinning_analysis.get('pinning_effectiveness_score', 0) < 70:
            recommendations.append({
                'category': 'architecture',
                'priority': 'medium',
                'title': 'Core Pinning Ineffective',
                'description': 'CPU core pinning not working optimally',
                'actions': [
                    'Review Docker container CPU affinity settings',
                    'Ensure proper CPU isolation',
                    'Consider updating container orchestration'
                ]
            })
        
        # AOCL optimization
        recommendations.append({
            'category': 'mathematical_optimization',
            'priority': 'low',
            'title': 'Mathematical Library Optimization',
            'description': 'Leverage AMD AOCL for better AI performance',
            'actions': [
                'Ensure AOCL libraries are properly linked',
                'Use Zen 5 optimized compilation flags (-march=znver5)',
                'Consider BLIS and FLAME optimizations for linear algebra'
            ]
        })
        
        return recommendations
    
    def _core_metrics_to_dict(self, metrics: Zen5CoreMetrics) -> Dict[str, Any]:
        """Convert Zen5CoreMetrics to dictionary."""
        return {
            'core_id': metrics.core_id,
            'physical_id': metrics.physical_id,
            'utilization': round(metrics.utilization, 2),
            'frequency_mhz': metrics.frequency_mhz,
            'temperature_c': round(metrics.temperature_c, 2),
            'power_draw_w': metrics.power_draw_w,
            'workload_type': metrics.workload_type,
            'pinned_processes': metrics.pinned_processes
        }
    
    async def detect_changes(self, old_data: Dict[str, Any], new_data: Dict[str, Any]) -> List[SystemChange]:
        """Detect changes in AMD Zen 5 workload state."""
        changes = []
        
        if 'cpu_core_metrics' not in old_data or 'cpu_core_metrics' not in new_data:
            return changes
        
        old_metrics = old_data['cpu_core_metrics']
        new_metrics = new_data['cpu_core_metrics']
        
        # CPU utilization threshold crossings
        if ('aggregate_metrics' in old_metrics and 'aggregate_metrics' in new_metrics):
            old_util = old_metrics['aggregate_metrics']['total_utilization']
            new_util = new_metrics['aggregate_metrics']['total_utilization']
            
            # High CPU utilization threshold
            if (old_util < self.thresholds['cpu_critical'] <= new_util):
                changes.append(SystemChange(
                    category='amd_zen5',
                    change_type=ChangeType.THRESHOLD_CROSSED,
                    entity_id='cpu_utilization_critical',
                    old_value=old_util,
                    new_value=new_util,
                    significance=0.9,
                    metadata={
                        'threshold_type': 'critical_cpu_utilization',
                        'threshold_value': self.thresholds['cpu_critical'],
                        'cores_count': self.arch_specs['cores']
                    },
                    timestamp=datetime.now()
                ))
            
            # Significant utilization change
            util_delta = abs(new_util - old_util)
            if util_delta > 25:  # 25% CPU utilization change
                changes.append(SystemChange(
                    category='amd_zen5',
                    change_type=ChangeType.MODIFIED,
                    entity_id='cpu_utilization_change',
                    old_value=old_util,
                    new_value=new_util,
                    significance=0.7,
                    metadata={
                        'change_type': 'cpu_utilization_shift',
                        'delta_percent': new_util - old_util,
                        'architecture': 'zen5'
                    },
                    timestamp=datetime.now()
                ))
        
        # Temperature threshold crossings
        if ('aggregate_metrics' in old_metrics and 'aggregate_metrics' in new_metrics):
            old_temp = old_metrics['aggregate_metrics'].get('average_temperature_c', 0)
            new_temp = new_metrics['aggregate_metrics'].get('average_temperature_c', 0)
            
            if (old_temp < self.thresholds['temperature_critical'] <= new_temp):
                changes.append(SystemChange(
                    category='amd_zen5',
                    change_type=ChangeType.THRESHOLD_CROSSED,
                    entity_id='cpu_temperature_critical',
                    old_value=old_temp,
                    new_value=new_temp,
                    significance=0.9,
                    metadata={
                        'threshold_type': 'critical_temperature',
                        'threshold_value': self.thresholds['temperature_critical'],
                        'thermal_throttle_risk': True
                    },
                    timestamp=datetime.now()
                ))
        
        # AI service utilization changes
        if ('ai_service_utilization' in old_metrics and 'ai_service_utilization' in new_metrics):
            for service_name in new_metrics['ai_service_utilization']:
                if service_name in old_metrics['ai_service_utilization']:
                    old_service_util = old_metrics['ai_service_utilization'][service_name]['average_utilization']
                    new_service_util = new_metrics['ai_service_utilization'][service_name]['average_utilization']
                    
                    service_delta = abs(new_service_util - old_service_util)
                    if service_delta > 30:  # 30% service utilization change
                        changes.append(SystemChange(
                            category='amd_zen5',
                            change_type=ChangeType.MODIFIED,
                            entity_id=f'ai_service_utilization:{service_name}',
                            old_value=old_service_util,
                            new_value=new_service_util,
                            significance=0.8,
                            metadata={
                                'change_type': 'ai_service_utilization_change',
                                'service_name': service_name,
                                'assigned_cores': self.ai_service_pinning.get(service_name, []),
                                'delta': new_service_util - old_service_util
                            },
                            timestamp=datetime.now()
                        ))
        
        # Core pinning effectiveness changes
        if ('core_pinning_analysis' in old_data and 'core_pinning_analysis' in new_data):
            old_pinning = old_data['core_pinning_analysis']
            new_pinning = new_data['core_pinning_analysis']
            
            old_effectiveness = old_pinning.get('pinning_effectiveness_score', 0)
            new_effectiveness = new_pinning.get('pinning_effectiveness_score', 0)
            
            effectiveness_delta = abs(new_effectiveness - old_effectiveness)
            if effectiveness_delta > 20:  # 20% effectiveness change
                changes.append(SystemChange(
                    category='amd_zen5',
                    change_type=ChangeType.MODIFIED,
                    entity_id='core_pinning_effectiveness',
                    old_value=old_effectiveness,
                    new_value=new_effectiveness,
                    significance=0.6,
                    metadata={
                        'change_type': 'core_pinning_effectiveness_change',
                        'effectiveness_delta': new_effectiveness - old_effectiveness,
                        'ai_services_affected': len(self.ai_service_pinning) - 1
                    },
                    timestamp=datetime.now()
                ))
        
        return changes