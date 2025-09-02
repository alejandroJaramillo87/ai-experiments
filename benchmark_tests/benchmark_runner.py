#!/usr/bin/env python3
"""
BenchmarkTestRunner - Flexible Test Execution Engine

A modular test execution system that separates test definitions from execution logic,
supporting both sequential and concurrent test execution against local LLM APIs.

"""

import json
import time
import logging
import requests
import os
import psutil
import subprocess
import threading
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
# Smart concurrency detection based on backend type
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import UniversalEvaluator for automatic evaluation
try:
    from evaluator.subjects import UniversalEvaluator, ReasoningType, evaluate_reasoning
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    logger.warning("UniversalEvaluator not available - evaluation features disabled")

# Import EnhancedUniversalEvaluator for Phase 1 enhanced evaluation
try:
    from evaluator.subjects.enhanced_universal_evaluator import EnhancedUniversalEvaluator
    ENHANCED_EVALUATION_AVAILABLE = True
except ImportError:
    ENHANCED_EVALUATION_AVAILABLE = False
    logger.warning("EnhancedUniversalEvaluator not available - enhanced evaluation features disabled")

# GPU monitoring for RTX 5090
try:
    import pynvml
    NVIDIA_GPU_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    NVIDIA_GPU_AVAILABLE = False
    logger.warning("pynvml not available - GPU metrics disabled")


def detect_backend_type() -> str:
    """
    Detect LLM backend type from environment or docker logs
    
    Returns:
        'concurrent' for vLLM backends that support concurrency
        'sequential' for llama.cpp backends that require sequential processing
    """
    
    # Check environment variable first (set by Makefile)
    concurrency_mode = os.environ.get('CONCURRENCY_MODE', '').lower()
    if concurrency_mode in ['concurrent', 'sequential']:
        logger.info(f"Backend mode detected from environment: {concurrency_mode}")
        return concurrency_mode
    
    # Try to detect from docker logs
    try:
        result = subprocess.run(
            ['docker', 'compose', 'logs', 'llama-gpu'], 
            capture_output=True, text=True, timeout=5
        )
        
        if result.returncode == 0:
            logs = result.stdout.lower()
            
            # Check for llama.cpp indicators
            if any(term in logs for term in ['llama.cpp', 'llama-server', 'gguf', 'llamacpp']):
                logger.info("Detected llama.cpp backend - using sequential mode")
                return 'sequential'
            
            # Check for vLLM indicators  
            elif any(term in logs for term in ['vllm', 'ray', 'asyncio']):
                logger.info("Detected vLLM backend - enabling concurrent mode")
                return 'concurrent'
        
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        logger.debug("Could not detect backend from docker logs")
    
    # Default to sequential for safety
    logger.info("Unable to detect backend type - defaulting to sequential mode for safety")
    return 'sequential'


@dataclass 
class PerformanceMetrics:
    """Comprehensive performance metrics for hardware monitoring"""
    
    # Timing metrics
    start_time: float
    end_time: float
    total_duration: float
    
    # Request timing breakdown
    request_start_time: float = 0.0
    request_end_time: float = 0.0
    request_duration: float = 0.0
    network_latency: float = 0.0
    
    # CPU metrics (AMD Ryzen 9950X)
    cpu_usage_percent: float = 0.0
    cpu_frequency_mhz: float = 0.0
    cpu_temp_celsius: float = 0.0
    cpu_cores_usage: List[float] = None
    
    # Memory metrics (128GB DDR5)
    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0
    memory_usage_percent: float = 0.0
    memory_swap_used_gb: float = 0.0
    
    # GPU metrics (RTX 5090)
    gpu_usage_percent: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_usage_percent: float = 0.0
    gpu_temp_celsius: float = 0.0
    gpu_power_usage_watts: float = 0.0
    gpu_clock_speed_mhz: float = 0.0
    gpu_memory_clock_mhz: float = 0.0
    
    # Storage I/O metrics (Samsung 990 Pro/EVO)
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0
    disk_read_iops: float = 0.0
    disk_write_iops: float = 0.0
    disk_usage_percent: float = 0.0
    disk_temp_celsius: float = 0.0
    
    # Network I/O metrics
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    network_packets_sent: int = 0
    network_packets_recv: int = 0
    network_rx_mb: float = 0.0
    network_tx_mb: float = 0.0
    
    # Throughput analysis
    tokens_per_second: float = 0.0
    requests_per_second: float = 0.0
    avg_response_size_kb: float = 0.0
    
    # System load
    system_load_1min: float = 0.0
    system_load_5min: float = 0.0
    system_load_15min: float = 0.0
    
    def __post_init__(self):
        if self.cpu_cores_usage is None:
            self.cpu_cores_usage = []


def ensure_json_serializable(obj: Any) -> Any:
    """
    Convert numpy types to native Python types for JSON serialization
    
    This fixes the "Object of type bool_ is not JSON serializable" error
    in benchmark_runner.py result saving.
    
    Args:
        obj: Object to convert (dict, list, or primitive type)
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(ensure_json_serializable(v) for v in obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # Other numpy scalar types
        return obj.item()
    else:
        return obj


@dataclass
class ResourceUtilizationReport:
    """Summary report of resource utilization during test execution"""
    
    # Test execution info
    test_id: str
    test_duration: float
    tokens_generated: int
    
    # Performance summary
    avg_performance: PerformanceMetrics
    peak_performance: PerformanceMetrics
    
    # Bottleneck analysis
    bottlenecks_detected: List[str]
    performance_warnings: List[str]
    
    # Hardware efficiency
    gpu_utilization_efficiency: float = 0.0  # How well GPU was utilized
    memory_pressure_score: float = 0.0       # Memory bottleneck indicator
    cpu_efficiency_score: float = 0.0        # CPU usage optimization
    io_throughput_score: float = 0.0         # Storage performance score
    throughput_tokens_per_second: float = 0.0


@dataclass
class TestResult:
    """Container for single test execution result"""
    test_id: str
    test_name: str
    success: bool
    response_text: str
    execution_time: float
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float
    error_message: Optional[str]
    timestamp: str
    api_response: Dict[str, Any]
    # Evaluation results (optional)
    evaluation_result: Optional[Dict[str, Any]] = None
    reasoning_score: Optional[float] = None
    reasoning_type: Optional[str] = None
    # Performance metrics (optional)
    performance_metrics: Optional[PerformanceMetrics] = None
    utilization_report: Optional[ResourceUtilizationReport] = None


@dataclass
class ExecutionProgress:
    """Container for execution progress information"""
    total_tests: int
    completed_tests: int
    successful_tests: int
    failed_tests: int
    current_test: Optional[str]
    estimated_remaining_time: float
    average_execution_time: float
    start_time: float
    elapsed_time: float
    current_category: Optional[str] = None
    tests_per_second: float = 0.0
    total_tokens_generated: int = 0
    average_tokens_per_second: float = 0.0
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage"""
        if self.total_tests == 0:
            return 0.0
        return (self.completed_tests / self.total_tests) * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of completed tests"""
        if self.completed_tests == 0:
            return 0.0
        return (self.successful_tests / self.completed_tests) * 100
    
    def update_estimates(self) -> None:
        """Update time estimates based on current progress"""
        if self.completed_tests > 0:
            self.average_execution_time = self.elapsed_time / self.completed_tests
            remaining_tests = self.total_tests - self.completed_tests
            self.estimated_remaining_time = remaining_tests * self.average_execution_time
            self.tests_per_second = self.completed_tests / self.elapsed_time if self.elapsed_time > 0 else 0


@dataclass
class APIConfiguration:
    """Container for API configuration"""
    endpoint: str
    model: str
    headers: Dict[str, str]
    timeout: int
    retry_attempts: int
    retry_delay: float
    api_type: str  # "completions" or "chat"


@dataclass
class TestSuite:
    """Container for test suite information"""
    suite_id: str
    name: str
    description: str
    version: str
    total_tests: int
    categories: Dict[str, int]
    test_type: str  # "base" or "instruct"
    created_date: str
    last_modified: str


@dataclass
class CategoryInfo:
    """Container for category information"""
    category_id: str
    name: str
    description: str
    reasoning_focus: str
    temperature_range: Tuple[float, float]
    test_range: Tuple[int, int]
    test_count: int
    test_ids: List[str]
    difficulty_level: str = "medium"
    estimated_time_per_test: float = 30.0


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for RTX 5090, AMD Ryzen 9950X, and 128GB DDR5
    
    Monitors:
    - GPU utilization, memory, temperature, power (RTX 5090)
    - CPU usage, frequency, temperature (AMD Ryzen 9950X)  
    - Memory usage, swap, pressure (128GB DDR5)
    - Storage I/O, temperature (Samsung 990 Pro/EVO)
    - Network I/O and latency
    - System load and bottleneck detection
    """
    
    def __init__(self):
        self.monitoring_active = False
        self.metrics_history: List[PerformanceMetrics] = []
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        self._metrics_lock = threading.Lock()  # Thread safety for metrics_history
        
        # Initialize GPU monitoring
        if NVIDIA_GPU_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # RTX 5090
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")
                self.gpu_handle = None
        else:
            self.gpu_handle = None
    
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start continuous performance monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self._stop_monitoring.clear()
        
        # Thread-safe clear of metrics history
        with self._metrics_lock:
            self.metrics_history.clear()
        
        # Capture baseline
        self.baseline_metrics = self._capture_metrics()
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info(f"Performance monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self) -> List[PerformanceMetrics]:
        """Stop monitoring and return collected metrics"""
        if not self.monitoring_active:
            return []
            
        self.monitoring_active = False
        self._stop_monitoring.set()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        
        # Thread-safe access to metrics history
        with self._metrics_lock:
            collected_count = len(self.metrics_history)
            result = self.metrics_history.copy()
            
        logger.info(f"Performance monitoring stopped. Collected {collected_count} samples")
        return result
    
    def _monitoring_loop(self, interval_seconds: float):
        """Continuous monitoring loop"""
        while not self._stop_monitoring.wait(interval_seconds):
            try:
                metrics = self._capture_metrics()
                # Thread-safe append to metrics history
                with self._metrics_lock:
                    self.metrics_history.append(metrics)
            except Exception as e:
                logger.warning(f"Error capturing metrics: {e}")
    
    def _capture_metrics(self) -> PerformanceMetrics:
        """Capture comprehensive system metrics"""
        current_time = time.time()
        
        # CPU metrics (AMD Ryzen 9950X)
        cpu_usage = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        cpu_cores = psutil.cpu_percent(percpu=True, interval=0.1)
        cpu_temp = self._get_cpu_temperature()
        
        # Memory metrics (128GB DDR5)
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # GPU metrics (RTX 5090)
        gpu_usage, gpu_memory, gpu_temp, gpu_power, gpu_clocks = self._get_gpu_metrics()
        
        # Storage I/O metrics (Samsung NVMe)
        disk_io = psutil.disk_io_counters()
        disk_usage = psutil.disk_usage('/')
        disk_temp = self._get_disk_temperature()
        
        # Network I/O metrics
        network_io = psutil.net_io_counters()
        
        # System load
        load_avg = os.getloadavg()
        
        return PerformanceMetrics(
            # Timing
            start_time=current_time,
            end_time=current_time,
            total_duration=0.0,
            
            # CPU (AMD Ryzen 9950X)
            cpu_usage_percent=cpu_usage,
            cpu_frequency_mhz=cpu_freq.current if cpu_freq else 0.0,
            cpu_temp_celsius=cpu_temp,
            cpu_cores_usage=cpu_cores,
            
            # Memory (128GB DDR5)
            memory_total_gb=memory.total / (1024**3),
            memory_used_gb=memory.used / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            memory_usage_percent=memory.percent,
            memory_swap_used_gb=swap.used / (1024**3),
            
            # GPU (RTX 5090)
            gpu_usage_percent=gpu_usage,
            gpu_memory_total_gb=gpu_memory[0],
            gpu_memory_used_gb=gpu_memory[1],
            gpu_memory_usage_percent=gpu_memory[2],
            gpu_temp_celsius=gpu_temp,
            gpu_power_usage_watts=gpu_power,
            gpu_clock_speed_mhz=gpu_clocks[0],
            gpu_memory_clock_mhz=gpu_clocks[1],
            
            # Storage (Samsung 990 Pro/EVO)
            disk_read_mb=(disk_io.read_bytes / (1024**2)) if disk_io else 0.0,
            disk_write_mb=(disk_io.write_bytes / (1024**2)) if disk_io else 0.0,
            disk_read_iops=disk_io.read_count if disk_io else 0.0,
            disk_write_iops=disk_io.write_count if disk_io else 0.0,
            disk_usage_percent=(disk_usage.used / disk_usage.total) * 100,
            disk_temp_celsius=disk_temp,
            
            # Network
            network_bytes_sent=network_io.bytes_sent if network_io else 0,
            network_bytes_recv=network_io.bytes_recv if network_io else 0,
            network_packets_sent=network_io.packets_sent if network_io else 0,
            network_packets_recv=network_io.packets_recv if network_io else 0,
            
            # System load
            system_load_1min=load_avg[0],
            system_load_5min=load_avg[1],
            system_load_15min=load_avg[2]
        )
    
    def _get_gpu_metrics(self) -> Tuple[float, Tuple[float, float, float], float, float, Tuple[float, float]]:
        """Get RTX 5090 GPU metrics"""
        if not self.gpu_handle or not NVIDIA_GPU_AVAILABLE:
            return 0.0, (0.0, 0.0, 0.0), 0.0, 0.0, (0.0, 0.0)
        
        try:
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            gpu_usage = util.gpu
            
            # GPU memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            gpu_mem_total = mem_info.total / (1024**3)  # GB
            gpu_mem_used = mem_info.used / (1024**3)    # GB
            gpu_mem_percent = (mem_info.used / mem_info.total) * 100
            
            # GPU temperature
            gpu_temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # GPU power usage
            try:
                gpu_power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Watts
            except:
                gpu_power = 0.0
            
            # GPU clock speeds
            try:
                gpu_clock = pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_GRAPHICS)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_MEM)
            except:
                gpu_clock, mem_clock = 0.0, 0.0
            
            return gpu_usage, (gpu_mem_total, gpu_mem_used, gpu_mem_percent), gpu_temp, gpu_power, (gpu_clock, mem_clock)
            
        except Exception as e:
            logger.debug(f"Error getting GPU metrics: {e}")
            return 0.0, (0.0, 0.0, 0.0), 0.0, 0.0, (0.0, 0.0)
    
    def _get_cpu_temperature(self) -> float:
        """Get AMD Ryzen CPU temperature"""
        try:
            # Try different methods for AMD CPU temperature
            sensors = psutil.sensors_temperatures()
            
            # AMD CPU temperature locations
            for sensor_name in ['k10temp', 'coretemp', 'cpu_thermal']:
                if sensor_name in sensors:
                    temps = sensors[sensor_name]
                    if temps:
                        return temps[0].current
            
            # Fallback: try reading from sysfs
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp_millicelsius = int(f.read().strip())
                    return temp_millicelsius / 1000.0
            except:
                pass
                
            return 0.0
        except Exception as e:
            logger.debug(f"Error getting CPU temperature: {e}")
            return 0.0
    
    def _get_disk_temperature(self) -> float:
        """Get NVMe SSD temperature (Samsung 990 Pro/EVO)"""
        try:
            # Try to get NVMe temperature using smartctl
            result = subprocess.run(
                ['smartctl', '-A', '/dev/nvme0n1'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Temperature' in line or 'temperature' in line:
                        parts = line.split()
                        for part in parts:
                            if part.isdigit():
                                return float(part)
            
            return 0.0
        except Exception as e:
            logger.debug(f"Error getting disk temperature: {e}")
            return 0.0
    
    def generate_utilization_report(self, test_id: str, tokens_generated: int) -> ResourceUtilizationReport:
        """Generate comprehensive resource utilization report"""
        if not self.metrics_history:
            return ResourceUtilizationReport(
                test_id=test_id,
                test_duration=0.0,
                tokens_generated=tokens_generated,
                avg_performance=PerformanceMetrics(
                    start_time=0.0,
                    end_time=0.0,
                    total_duration=0.0
                ),
                peak_performance=PerformanceMetrics(
                    start_time=0.0,
                    end_time=0.0,
                    total_duration=0.0
                ),
                bottlenecks_detected=[],
                performance_warnings=[]
            )
        
        # Calculate averages and peaks
        avg_metrics = self._calculate_average_metrics()
        peak_metrics = self._calculate_peak_metrics()
        
        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(avg_metrics, peak_metrics)
        warnings = self._generate_performance_warnings(avg_metrics, peak_metrics)
        
        # Calculate efficiency scores
        gpu_efficiency = self._calculate_gpu_efficiency(avg_metrics)
        memory_pressure = self._calculate_memory_pressure(avg_metrics)
        cpu_efficiency = self._calculate_cpu_efficiency(avg_metrics)
        io_throughput = self._calculate_io_throughput(avg_metrics)
        
        test_duration = (self.metrics_history[-1].start_time - self.metrics_history[0].start_time)
        
        # Calculate throughput
        throughput = tokens_generated / test_duration if test_duration > 0 else 0
        
        return ResourceUtilizationReport(
            test_id=test_id,
            test_duration=test_duration,
            tokens_generated=tokens_generated,
            avg_performance=avg_metrics,
            peak_performance=peak_metrics,
            bottlenecks_detected=bottlenecks,
            performance_warnings=warnings,
            gpu_utilization_efficiency=gpu_efficiency,
            memory_pressure_score=memory_pressure,
            cpu_efficiency_score=cpu_efficiency,
            io_throughput_score=io_throughput,
            throughput_tokens_per_second=throughput
        )
    
    def _calculate_average_metrics(self) -> PerformanceMetrics:
        """Calculate average metrics across monitoring period"""
        if not self.metrics_history:
            return PerformanceMetrics(
                start_time=0.0,
                end_time=0.0,
                total_duration=0.0
            )
        
        # Sum all metrics
        total_count = len(self.metrics_history)
        
        avg_cpu = sum(m.cpu_usage_percent for m in self.metrics_history) / total_count
        avg_memory = sum(m.memory_usage_percent for m in self.metrics_history) / total_count
        avg_gpu = sum(m.gpu_usage_percent for m in self.metrics_history) / total_count
        avg_gpu_temp = sum(m.gpu_temp_celsius for m in self.metrics_history) / total_count
        avg_cpu_temp = sum(m.cpu_temp_celsius for m in self.metrics_history) / total_count
        
        return PerformanceMetrics(
            start_time=self.metrics_history[0].start_time,
            end_time=self.metrics_history[-1].start_time,
            total_duration=self.metrics_history[-1].start_time - self.metrics_history[0].start_time,
            cpu_usage_percent=avg_cpu,
            cpu_temp_celsius=avg_cpu_temp,
            memory_usage_percent=avg_memory,
            gpu_usage_percent=avg_gpu,
            gpu_temp_celsius=avg_gpu_temp
        )
    
    def _calculate_peak_metrics(self) -> PerformanceMetrics:
        """Calculate peak metrics across monitoring period"""
        if not self.metrics_history:
            return PerformanceMetrics(
                start_time=0.0,
                end_time=0.0,
                total_duration=0.0
            )
        
        peak_cpu = max(m.cpu_usage_percent for m in self.metrics_history)
        peak_memory = max(m.memory_usage_percent for m in self.metrics_history)
        peak_gpu = max(m.gpu_usage_percent for m in self.metrics_history)
        peak_gpu_temp = max(m.gpu_temp_celsius for m in self.metrics_history)
        peak_cpu_temp = max(m.cpu_temp_celsius for m in self.metrics_history)
        
        return PerformanceMetrics(
            start_time=self.metrics_history[0].start_time,
            end_time=self.metrics_history[-1].start_time,
            total_duration=self.metrics_history[-1].start_time - self.metrics_history[0].start_time,
            cpu_usage_percent=peak_cpu,
            cpu_temp_celsius=peak_cpu_temp,
            memory_usage_percent=peak_memory,
            gpu_usage_percent=peak_gpu,
            gpu_temp_celsius=peak_gpu_temp
        )
    
    def _detect_bottlenecks(self, avg_metrics: PerformanceMetrics, peak_metrics: PerformanceMetrics) -> List[str]:
        """Detect system bottlenecks based on usage patterns"""
        bottlenecks = []
        
        # GPU bottlenecks (GPU usage >90% is expected and desired)
        if avg_metrics.gpu_memory_usage_percent > 85:
            bottlenecks.append("GPU memory usage high (>85%)")
        if avg_metrics.gpu_temp_celsius > 80:
            bottlenecks.append("GPU temperature high (>80°C)")
        
        # CPU bottlenecks
        if avg_metrics.cpu_usage_percent > 80:
            bottlenecks.append("CPU utilization high (>80%)")
        if avg_metrics.cpu_temp_celsius > 75:
            bottlenecks.append("CPU temperature high (>75°C)")
        
        # Memory bottlenecks
        if avg_metrics.memory_usage_percent > 90:
            bottlenecks.append("Memory usage critical (>90%)")
        if avg_metrics.memory_swap_used_gb > 1.0:
            bottlenecks.append("Swap usage detected (memory pressure)")
        
        return bottlenecks
    
    def _generate_performance_warnings(self, avg_metrics: PerformanceMetrics, peak_metrics: PerformanceMetrics) -> List[str]:
        """Generate performance warnings"""
        warnings = []
        
        # Performance warnings
        if peak_metrics.gpu_usage_percent < 50:
            warnings.append("GPU underutilized (peak <50%) - consider increasing batch size")
        if avg_metrics.memory_usage_percent < 20:
            warnings.append("Memory underutilized (<20%) - system has excess capacity")
        if peak_metrics.cpu_usage_percent > peak_metrics.gpu_usage_percent + 20:
            warnings.append("CPU usage significantly higher than GPU - possible CPU bottleneck")
        
        return warnings
    
    def _calculate_gpu_efficiency(self, metrics: PerformanceMetrics) -> float:
        """Calculate GPU utilization efficiency score (0-100)"""
        # Consider both usage and temperature efficiency
        usage_score = min(metrics.gpu_usage_percent, 100)
        temp_efficiency = max(0, 100 - (metrics.gpu_temp_celsius - 40) * 2) if metrics.gpu_temp_celsius > 40 else 100
        return (usage_score + temp_efficiency) / 2
    
    def _calculate_memory_pressure(self, metrics: PerformanceMetrics) -> float:
        """Calculate memory pressure score (0-100, higher = more pressure)"""
        pressure = metrics.memory_usage_percent
        if metrics.memory_swap_used_gb > 0:
            pressure += 20  # Penalty for swap usage
        return min(pressure, 100)
    
    def _calculate_cpu_efficiency(self, metrics: PerformanceMetrics) -> float:
        """Calculate CPU efficiency score (0-100)"""
        # Balance usage and temperature
        usage_score = min(metrics.cpu_usage_percent, 100)
        temp_efficiency = max(0, 100 - (metrics.cpu_temp_celsius - 50) * 2) if metrics.cpu_temp_celsius > 50 else 100
        return (usage_score + temp_efficiency) / 2
    
    def _calculate_io_throughput(self, metrics: PerformanceMetrics) -> float:
        """Calculate I/O throughput efficiency score (0-100)"""
        # Simple throughput score based on disk usage
        # For NVMe SSDs, anything above 50% usage is considered good
        return min(metrics.disk_usage_percent * 2, 100)


class TestSuiteManager:
    """
    Advanced test suite management system for organizing and filtering test categories
    
    Provides capabilities for:
    - Suite discovery and loading
    - Category filtering and selection
    - Test organization and metadata management
    - Advanced search and filtering
    - Suite statistics and analysis
    """
    
    def __init__(self):
        self.available_suites: Dict[str, TestSuite] = {}
        self.category_registry: Dict[str, CategoryInfo] = {}
        self.test_registry: Dict[str, Dict] = {}
        
    def discover_test_suites(self, base_directory: str) -> List[TestSuite]:
        """
        Discover all available test suites in the new domain-based directory structure
        
        Args:
            base_directory: Base directory to search for test suites
            
        Returns:
            List of discovered TestSuite objects
        """
        discovered_suites = []
        domains_dir = os.path.join(base_directory, "domains")
        
        if not os.path.exists(domains_dir):
            return discovered_suites
        
        # Discover domains (reasoning, linux, etc.)
        for domain_name in os.listdir(domains_dir):
            domain_path = os.path.join(domains_dir, domain_name)
            if not os.path.isdir(domain_path):
                continue
                
            # Discover model types within each domain (base_models, instruct_models)
            for model_type in ["base_models", "instruct_models"]:
                model_type_dir = os.path.join(domain_path, model_type)
                if not os.path.exists(model_type_dir):
                    continue
                    
                categories_path = os.path.join(model_type_dir, "categories.json")
                
                if os.path.exists(categories_path):
                    suite = self._load_domain_suite_metadata(categories_path, domain_name, model_type, model_type_dir)
                    if suite:
                        discovered_suites.append(suite)
                        self.available_suites[suite.suite_id] = suite
        
        return discovered_suites
    
    def _load_suite_metadata(self, metadata_path: str, categories_path: str, suite_type: str) -> Optional[TestSuite]:
        """Load suite metadata from files"""
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            with open(categories_path, 'r') as f:
                categories_data = json.load(f)
            
            # Count tests per category
            category_counts = {}
            total_tests = 0
            
            if 'categories' in categories_data:
                for cat_name, cat_info in categories_data['categories'].items():
                    count = len(cat_info.get('test_ids', []))
                    category_counts[cat_name] = count
                    total_tests += count
                    
                    # Convert suite_type to standardized format
                    suite_type_key = "base" if "base" in suite_type else "instruct"
                    
                    # Register category info
                    self.category_registry[f"{suite_type_key}_{cat_name}"] = CategoryInfo(
                        category_id=cat_name,
                        name=cat_name.replace('_', ' ').title(),
                        description=cat_info.get('description', ''),
                        reasoning_focus=cat_info.get('reasoning_focus', ''),
                        temperature_range=tuple(cat_info.get('temperature_range', [0.1, 0.5])),
                        test_range=tuple(cat_info.get('test_range', [1, count])),
                        test_count=count,
                        test_ids=cat_info.get('test_ids', []),
                        difficulty_level=cat_info.get('difficulty', 'medium')
                    )
            
            # Create TestSuite object
            suite_type_clean = "base" if "base" in suite_type else "instruct"
            return TestSuite(
                suite_id=metadata.get('suite_id', f'{suite_type_clean}_suite'),
                name=metadata.get('suite_name', f'{suite_type_clean.title()} Model Test Suite'),
                description=metadata.get('description', 'No description available'),
                version=metadata.get('version', '1.0.0'),
                total_tests=total_tests,
                categories=category_counts,
                test_type=suite_type_clean,
                created_date=metadata.get('created_date', ''),
                last_modified=metadata.get('last_modified', '')
            )
            
        except Exception as e:
            logger.error(f"Error loading suite metadata from {metadata_path}: {e}")
            return None
    
    def _load_domain_suite_metadata(self, categories_path: str, domain_name: str, model_type: str, model_type_dir: str) -> Optional[TestSuite]:
        """Load suite metadata from domain-based structure"""
        try:
            with open(categories_path, 'r') as f:
                categories_data = json.load(f)
            
            # Count tests per category and test files
            category_counts = {}
            total_tests = 0
            test_files = []
            
            # Count JSON test files in the model_type_dir
            if os.path.exists(model_type_dir):
                for file in os.listdir(model_type_dir):
                    if file.endswith('.json') and file != 'categories.json':
                        test_files.append(file)
            
            if 'categories' in categories_data:
                for cat_name, cat_info in categories_data['categories'].items():
                    count = len(cat_info.get('test_ids', []))
                    category_counts[cat_name] = count
                    total_tests += count
                    
                    # Convert model_type to standardized format
                    suite_type_key = "base" if "base" in model_type else "instruct"
                    
                    # Register category info with domain prefix
                    registry_key = f"{domain_name}_{suite_type_key}_{cat_name}"
                    self.category_registry[registry_key] = CategoryInfo(
                        category_id=cat_name,
                        name=cat_name.replace('_', ' ').title(),
                        description=cat_info.get('description', ''),
                        reasoning_focus=cat_info.get('reasoning_focus', ''),
                        temperature_range=tuple(cat_info.get('temperature_range', [0.1, 0.5])),
                        test_range=tuple(cat_info.get('test_range', [1, count])),
                        test_count=count,
                        test_ids=cat_info.get('test_ids', []),
                        difficulty_level=cat_info.get('difficulty', 'medium')
                    )
            
            # Create TestSuite object
            suite_type_clean = "base" if "base" in model_type else "instruct"
            suite_id = f"{domain_name}_{suite_type_clean}"
            
            return TestSuite(
                suite_id=suite_id,
                name=f"{domain_name.title()} {suite_type_clean.title()} Test Suite",
                description=f"{domain_name.title()} domain tests for {suite_type_clean} models",
                version="1.0.0",
                total_tests=total_tests,
                categories=category_counts,  # Dict[str, int] as expected
                test_type=suite_type_clean,
                created_date="2025-01-26",
                last_modified="2025-01-26"
            )
            
        except Exception as e:
            logger.error(f"Error loading domain suite metadata from {categories_path}: {e}")
            return None
    
    def get_category_info(self, category_id: str, suite_type: str = None) -> Optional[CategoryInfo]:
        """Get detailed information about a category"""
        if suite_type:
            full_category_id = f"{suite_type}_{category_id}"
            return self.category_registry.get(full_category_id)
        
        # Search across all suite types
        for key, cat_info in self.category_registry.items():
            if key.endswith(f"_{category_id}"):
                return cat_info
        return None
    
    def filter_tests_by_criteria(self, suite_type: str, **criteria) -> List[str]:
        """
        Filter tests based on various criteria
        
        Args:
            suite_type: "base" or "instruct"
            **criteria: Filtering criteria such as:
                - category: Category name
                - difficulty: "easy", "medium", "hard"
                - max_time: Maximum estimated time per test
                - temperature_range: Tuple of (min, max) temperature
                - reasoning_focus: Keywords to match in reasoning focus
                
        Returns:
            List of test IDs matching criteria
        """
        matching_tests = []
        
        # Get all categories for the suite type (normalize suite_type)
        suite_type_key = "base" if suite_type in ["base", "base_models"] else "instruct"
        suite_categories = {k: v for k, v in self.category_registry.items() 
                          if k.startswith(f"{suite_type_key}_")}
        
        for cat_key, cat_info in suite_categories.items():
            include_category = True
            
            # Apply filters
            if 'category' in criteria and cat_info.category_id != criteria['category']:
                include_category = False
            
            if 'difficulty' in criteria and cat_info.difficulty_level != criteria['difficulty']:
                include_category = False
            
            if 'max_time' in criteria and cat_info.estimated_time_per_test > criteria['max_time']:
                include_category = False
            
            if 'temperature_range' in criteria:
                min_temp, max_temp = criteria['temperature_range']
                cat_min, cat_max = cat_info.temperature_range
                if cat_max < min_temp or cat_min > max_temp:
                    include_category = False
            
            if 'reasoning_focus' in criteria:
                focus_keywords = criteria['reasoning_focus'].lower().split(',')
                if not any(keyword.strip() in cat_info.reasoning_focus.lower() 
                          for keyword in focus_keywords):
                    include_category = False
            
            if include_category:
                matching_tests.extend(cat_info.test_ids)
        
        return matching_tests
    
    def get_suite_statistics(self, suite_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a test suite"""
        suite = self.available_suites.get(suite_id)
        if not suite:
            return {}
        
        # Get categories for this suite
        suite_categories = {k: v for k, v in self.category_registry.items() 
                          if k.startswith(f"{suite.test_type}_")}
        
        stats = {
            'suite_info': asdict(suite),
            'category_breakdown': {},
            'difficulty_distribution': {},
            'temperature_distribution': {},
            'total_estimated_time': 0,
            'reasoning_focus_analysis': {}
        }
        
        reasoning_focuses = {}
        difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0}
        temp_ranges = []
        
        for cat_info in suite_categories.values():
            stats['category_breakdown'][cat_info.category_id] = {
                'test_count': cat_info.test_count,
                'description': cat_info.description,
                'estimated_time': cat_info.test_count * cat_info.estimated_time_per_test
            }
            
            stats['total_estimated_time'] += cat_info.test_count * cat_info.estimated_time_per_test
            difficulty_counts[cat_info.difficulty_level] += cat_info.test_count
            temp_ranges.append(cat_info.temperature_range)
            
            # Analyze reasoning focuses
            focuses = cat_info.reasoning_focus.split(',')
            for focus in focuses:
                focus = focus.strip()
                reasoning_focuses[focus] = reasoning_focuses.get(focus, 0) + cat_info.test_count
        
        stats['difficulty_distribution'] = difficulty_counts
        stats['reasoning_focus_analysis'] = reasoning_focuses
        
        if temp_ranges:
            stats['temperature_distribution'] = {
                'min_temp': min(r[0] for r in temp_ranges),
                'max_temp': max(r[1] for r in temp_ranges),
                'avg_min_temp': sum(r[0] for r in temp_ranges) / len(temp_ranges),
                'avg_max_temp': sum(r[1] for r in temp_ranges) / len(temp_ranges)
            }
        
        return stats


class BenchmarkTestRunner:
    """
    Flexible test execution engine for LLM benchmarking
    
    Handles loading test definitions from JSON, executing tests against APIs,
    and collecting results with comprehensive error handling and progress tracking.
    """
    
    def __init__(self, config_path: Optional[str] = None, api_endpoint: Optional[str] = None):
        """
        Initialize BenchmarkTestRunner with optional configuration
        
        Args:
            config_path: Path to configuration JSON file
            api_endpoint: API endpoint URL (overrides config/metadata)
        """
        self.tests = {}
        self.test_metadata = {}
        self.categories = {}
        self.api_config = None
        self.config = self._load_config(config_path)
        self.execution_progress = None
        self._start_times = []
        self._api_endpoint_override = api_endpoint
        self._progress_callback = None
        self._verbose_logging = True
        self._last_progress_update = 0
        
        # Initialize test suite manager
        self.suite_manager = TestSuiteManager()
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor()
        self._enable_performance_monitoring = False
        
        logger.info("BenchmarkTestRunner initialized")
    
    def load_test_suite(self, suite_path: str) -> bool:
        """
        Load test suite from JSON file
        
        Args:
            suite_path: Path to test suite JSON file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            with open(suite_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate structure
            if 'tests' not in data:
                logger.error(f"Invalid test suite format: missing 'tests' key")
                return False
            
            # Load tests into registry
            for test in data['tests']:
                if 'id' not in test:
                    logger.warning(f"Test missing 'id' field, skipping")
                    continue
                self.tests[test['id']] = test
            
            logger.info(f"Loaded {len(self.tests)} tests from {suite_path}")
            
            # Initialize progress tracking with loaded tests
            current_time = time.time()
            self.execution_progress = ExecutionProgress(
                total_tests=len(self.tests),
                completed_tests=0,
                successful_tests=0,
                failed_tests=0,
                current_test=None,
                estimated_remaining_time=0.0,
                average_execution_time=0.0,
                start_time=current_time,
                elapsed_time=0.0,
                total_tokens_generated=0
            )
            
            return True
            
        except FileNotFoundError:
            logger.error(f"Test suite file not found: {suite_path}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in test suite file: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading test suite: {e}")
            return False
    
    def load_test_metadata(self, metadata_path: str) -> bool:
        """
        Load suite metadata and API configuration
        
        Args:
            metadata_path: Path to metadata JSON file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.test_metadata = json.load(f)
            
            # Extract API configuration (can be overridden by constructor arg)
            if 'api_config' in self.test_metadata:
                api_data = self.test_metadata['api_config']
                endpoint = self._api_endpoint_override or api_data.get('endpoint', 'http://127.0.0.1:8004/v1/completions')
                
                self.api_config = APIConfiguration(
                    endpoint=endpoint,
                    model=api_data.get('model', 'default-model'),
                    headers=api_data.get('headers', {'Content-Type': 'application/json'}),
                    timeout=api_data.get('timeout', 600),
                    retry_attempts=3,
                    retry_delay=1.0,
                    api_type=self._detect_api_type(endpoint)
                )
            
            logger.info(f"Loaded test metadata from {metadata_path}")
            return True
            
        except FileNotFoundError:
            logger.error(f"Metadata file not found: {metadata_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return False
    
    def load_categories(self, categories_path: str) -> bool:
        """
        Load category definitions
        
        Args:
            categories_path: Path to categories JSON file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            with open(categories_path, 'r', encoding='utf-8') as f:
                self.categories = json.load(f)
            
            logger.info(f"Loaded categories from {categories_path}")
            return True
            
        except FileNotFoundError:
            logger.error(f"Categories file not found: {categories_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading categories: {e}")
            return False
    
    def configure_api(self, endpoint: str, model: str, headers: Dict = None, timeout: int = 600) -> None:
        """
        Configure API connection settings
        
        Args:
            endpoint: Full API endpoint URL (e.g., 'http://127.0.0.1:8004/v1/completions')
            model: Model name
            headers: HTTP headers dictionary
            timeout: Request timeout in seconds
        """
        self.api_config = APIConfiguration(
            endpoint=endpoint,
            model=model,
            headers=headers or {'Content-Type': 'application/json'},
            timeout=timeout,
            retry_attempts=3,
            retry_delay=1.0,
            api_type=self._detect_api_type(endpoint)
        )
        logger.info(f"API configured: {endpoint} ({self.api_config.api_type})")
    
    def execute_single_test(self, test_id: str, enable_performance_monitoring: bool = False) -> TestResult:
        """
        Execute a single test case
        
        Args:
            test_id: ID of test to execute
            enable_performance_monitoring: Enable hardware performance monitoring
            
        Returns:
            TestResult: Result of test execution
        """
        if test_id not in self.tests:
            error_msg = f"Test {test_id} not found"
            logger.error(error_msg)
            return self._create_error_result(test_id, error_msg)
        
        if not self.api_config:
            error_msg = "API not configured"
            logger.error(error_msg)
            return self._create_error_result(test_id, error_msg)
        
        test_case = self.tests[test_id]
        logger.info(f"Executing test: {test_case.get('name', test_id)}")
        
        # Start performance monitoring if enabled
        if enable_performance_monitoring:
            self.performance_monitor.start_monitoring(1.0)  # 1 second interval
        
        performance_metrics = None
        utilization_report = None
        
        try:
            # Build API request
            success, api_response, error_msg = self._make_api_request(test_case)
            
            if success:
                result = self._create_success_result(test_id, test_case, api_response)
            else:
                result = self._create_error_result(test_id, error_msg, api_response)
                
        except Exception as e:
            error_msg = f"Test execution failed: {str(e)}"
            logger.error(error_msg)
            result = self._create_error_result(test_id, error_msg)
        finally:
            # Always ensure monitoring is stopped and metrics are collected
            if enable_performance_monitoring:
                self.performance_monitor.stop_monitoring()
                tokens_generated = 0
                if 'api_response' in locals() and api_response:
                    tokens_generated = api_response.get('usage', {}).get('completion_tokens', 0)
                utilization_report = self.performance_monitor.generate_utilization_report(test_id, tokens_generated)
                
                # Get final metrics snapshot
                if self.performance_monitor.metrics_history:
                    performance_metrics = self.performance_monitor.metrics_history[-1]
                
                # Add performance metrics to result
                if hasattr(locals().get('result'), 'performance_metrics'):
                    result.performance_metrics = performance_metrics
                    result.utilization_report = utilization_report
        
        return result
    
    def get_test_ids_by_category(self, category: str) -> List[str]:
        """
        Get list of test IDs for a specific category
        
        Args:
            category: Category name
            
        Returns:
            List of test IDs in the category
        """
        if not self.categories or 'categories' not in self.categories:
            # Fallback: filter by category field in test data
            return [test_id for test_id, test in self.tests.items() 
                   if test.get('category') == category]
        
        category_data = self.categories['categories'].get(category, {})
        return category_data.get('test_ids', [])
    
    def save_results(self, results: List[TestResult], output_dir: str = None) -> bool:
        """
        Save test results to files (compatible with existing format)
        
        Args:
            results: List of TestResult objects
            output_dir: Output directory (defaults to test_results)
            
        Returns:
            bool: True if saved successfully
        """
        if output_dir is None:
            output_dir = "test_results"
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            for result in results:
                # Create filename compatible with existing format
                filename_safe_name = result.test_name.lower().replace(":", "").replace(" ", "_")
                output_path = os.path.join(output_dir, f"{filename_safe_name}_completion.txt")
                
                # Save in same format as monolithic version
                with open(output_path, "w", encoding="utf-8") as f:
                    test_case = self.tests.get(result.test_id, {})
                    prompt = test_case.get('prompt', 'N/A')
                    
                    f.write(f"PROMPT:\n{'-'*20}\n{prompt}\n\n")
                    f.write(f"COMPLETION:\n{'-'*20}\n{result.response_text}\n\n")
                    f.write(f"METRICS:\n{'-'*20}\n")
                    f.write(f"Duration: {result.execution_time:.2f}s\n")
                    f.write(f"Prompt Tokens: {result.prompt_tokens}\n")
                    f.write(f"Completion Tokens: {result.completion_tokens}\n")
                    f.write(f"Tokens per Second: {result.tokens_per_second:.2f} T/s\n")
                    
                    # Add performance metrics if available
                    if result.performance_metrics:
                        f.write(f"\nPERFORMANCE METRICS:\n{'-'*20}\n")
                        f.write(f"RTX 5090 GPU Usage: {result.performance_metrics.gpu_usage_percent:.1f}%\n")
                        f.write(f"GPU Memory Usage: {result.performance_metrics.gpu_memory_used_gb:.1f}GB / {result.performance_metrics.gpu_memory_total_gb:.1f}GB\n")
                        f.write(f"GPU Temperature: {result.performance_metrics.gpu_temp_celsius:.1f}°C\n")
                        f.write(f"GPU Power Usage: {result.performance_metrics.gpu_power_usage_watts:.1f}W\n")
                        f.write(f"AMD Ryzen CPU Usage: {result.performance_metrics.cpu_usage_percent:.1f}%\n")
                        f.write(f"CPU Temperature: {result.performance_metrics.cpu_temp_celsius:.1f}°C\n")
                        f.write(f"DDR5 Memory Usage: {result.performance_metrics.memory_usage_percent:.1f}%\n")
                        f.write(f"Swap Usage: {result.performance_metrics.memory_swap_used_gb:.1f}GB\n")
                        f.write(f"NVMe Read: {result.performance_metrics.disk_read_mb:.1f}MB\n")
                        f.write(f"NVMe Write: {result.performance_metrics.disk_write_mb:.1f}MB\n")
                        f.write(f"Storage Temperature: {result.performance_metrics.disk_temp_celsius:.1f}°C\n")
                        f.write(f"Network RX: {result.performance_metrics.network_rx_mb:.1f}MB\n")
                        f.write(f"Network TX: {result.performance_metrics.network_tx_mb:.1f}MB\n")
                        
                        # Add utilization report summary if available
                        if result.utilization_report:
                            f.write(f"\nUTILIZATION SUMMARY:\n{'-'*20}\n")
                            f.write(f"Test Duration: {result.utilization_report.test_duration:.2f}s\n")
                            f.write(f"Tokens Generated: {result.utilization_report.tokens_generated}\n")
                            f.write(f"Throughput: {result.utilization_report.throughput_tokens_per_second:.2f} tokens/s\n")
                            
                            if result.utilization_report.bottlenecks_detected:
                                f.write(f"Bottlenecks Detected: {', '.join(result.utilization_report.bottlenecks_detected)}\n")
                            
                            if result.utilization_report.performance_warnings:
                                f.write(f"Performance Warnings:\n")
                                for warning in result.utilization_report.performance_warnings:
                                    f.write(f"  • {warning}\n")
                    
                    # Add evaluation results if available
                    if result.evaluation_result:
                        f.write(f"\nREASONING EVALUATION:\n{'-'*20}\n")
                        f.write(f"Overall Score: {result.reasoning_score}/100\n")
                        f.write(f"Reasoning Type: {result.reasoning_type}\n")
                        
                        # Add detailed metrics
                        metrics = result.evaluation_result.get('metrics', {})
                        for metric_name, metric_value in metrics.items():
                            if isinstance(metric_value, (int, float)) and metric_name != 'overall_score':
                                display_name = metric_name.replace('_', ' ').title()
                                f.write(f"{display_name}: {metric_value:.1f}\n")
                        
                        # Add recommendations if any
                        recommendations = result.evaluation_result.get('recommendations', [])
                        if recommendations:
                            f.write(f"\nRecommendations:\n")
                            for i, rec in enumerate(recommendations, 1):
                                f.write(f"  {i}. {rec}\n")
                    
                    if not result.success and result.error_message:
                        f.write(f"Error: {result.error_message}\n")
                
                # Also save JSON version for programmatic access
                json_path = os.path.join(output_dir, f"{filename_safe_name}_result.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    # Apply JSON serialization fix for numpy types
                    serializable_result = ensure_json_serializable(asdict(result))
                    json.dump(serializable_result, f, indent=2, ensure_ascii=False)
            
            # Create batch results summary
            if len(results) > 1:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                batch_path = os.path.join(output_dir, f"batch_results_{timestamp}.json")
                
                batch_summary = {
                    "execution_summary": {
                        "total_tests": len(results),
                        "successful_tests": len([r for r in results if r.success]),
                        "failed_tests": len([r for r in results if not r.success]),
                        "total_execution_time": sum(r.execution_time for r in results),
                        "average_test_time": sum(r.execution_time for r in results) / len(results) if results else 0
                    },
                    "individual_results": [asdict(result) for result in results]
                }
                
                with open(batch_path, "w", encoding="utf-8") as f:
                    # Apply JSON serialization fix for numpy types in batch results
                    serializable_batch = ensure_json_serializable(batch_summary)
                    json.dump(serializable_batch, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(results)} results to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def execute_sequential(self, test_ids: List[str] = None, delay: float = None, enable_performance_monitoring: bool = False) -> List[TestResult]:
        """
        Execute tests sequentially with delays between tests
        
        Args:
            test_ids: List of test IDs to execute (defaults to all tests)
            delay: Delay in seconds between tests (defaults to config setting)
            enable_performance_monitoring: Enable hardware performance monitoring
            
        Returns:
            List of TestResult objects
        """
        if test_ids is None:
            test_ids = list(self.tests.keys())
        
        if delay is None:
            delay = self.config['execution_defaults'].get('sequential_delay', 1.0)
        
        results = []
        total_tests = len(test_ids)
        start_time = time.time()
        
        # Initialize progress tracking
        self._initialize_progress(total_tests, start_time)
        
        logger.info(f"Starting sequential execution of {total_tests} tests (delay: {delay}s)")
        if enable_performance_monitoring:
            logger.info("Performance monitoring enabled for RTX 5090, AMD Ryzen 9950X, 128GB DDR5 RAM")
        
        for i, test_id in enumerate(test_ids):
            # Update current test
            self._update_progress(current_test_name=test_id)
            
            # Execute test with optional performance monitoring
            result = self.execute_single_test(test_id, enable_performance_monitoring)
            results.append(result)
            
            # Update progress with result
            self._update_progress(test_result=result)
            
            # Log performance metrics if available
            if enable_performance_monitoring and result.utilization_report and result.performance_metrics:
                bottlenecks = ", ".join(result.utilization_report.bottlenecks_detected) if result.utilization_report.bottlenecks_detected else "None"
                logger.info(f"Test {test_id} performance - GPU: {result.performance_metrics.gpu_usage_percent:.1f}%, CPU: {result.performance_metrics.cpu_usage_percent:.1f}%, Memory: {result.performance_metrics.memory_usage_percent:.1f}%, Bottlenecks: {bottlenecks}")
            
            # Add delay between tests (except after the last test)
            if i < len(test_ids) - 1 and delay > 0:
                time.sleep(delay)
        
        # Final progress logging
        self._log_final_progress()
        
        return results
    
    def execute_concurrent(self, test_ids: List[str] = None, workers: int = 3, enable_performance_monitoring: bool = False) -> List[TestResult]:
        """
        Execute tests concurrently using thread pool
        
        Args:
            test_ids: List of test IDs to execute (defaults to all tests)
            workers: Number of concurrent worker threads
            enable_performance_monitoring: Enable hardware performance monitoring
            
        Returns:
            List of TestResult objects
        """
        if test_ids is None:
            test_ids = list(self.tests.keys())
        
        results = []
        total_tests = len(test_ids)
        start_time = time.time()
        
        # Thread-safe progress tracking
        progress_lock = threading.Lock()
        
        # Initialize progress tracking
        self._initialize_progress(total_tests, start_time)
        
        logger.info(f"Starting concurrent execution of {total_tests} tests with {workers} workers")
        if enable_performance_monitoring:
            logger.info("Performance monitoring enabled for RTX 5090, AMD Ryzen 9950X, 128GB DDR5 RAM")
            logger.warning("Note: Performance monitoring in concurrent mode may show mixed metrics across tests")
        
        def update_progress_concurrent(result: TestResult):
            """Thread-safe progress update"""
            with progress_lock:
                self._update_progress(test_result=result)
                
                # Log performance metrics if available (thread-safe)
                if enable_performance_monitoring and result.utilization_report and result.performance_metrics:
                    bottlenecks = ", ".join(result.utilization_report.bottlenecks_detected) if result.utilization_report.bottlenecks_detected else "None"
                    logger.info(f"Test {result.test_id} performance - GPU: {result.performance_metrics.gpu_usage_percent:.1f}%, CPU: {result.performance_metrics.cpu_usage_percent:.1f}%, Memory: {result.performance_metrics.memory_usage_percent:.1f}%, Bottlenecks: {bottlenecks}")
        
        # Execute tests concurrently
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks with performance monitoring parameter
            future_to_test = {
                executor.submit(self.execute_single_test, test_id, enable_performance_monitoring): test_id 
                for test_id in test_ids
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_test):
                test_id = future_to_test[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    update_progress_concurrent(result)
                    
                except Exception as e:
                    # Create error result if thread execution fails
                    error_result = self._create_error_result(test_id, f"Thread execution error: {e}")
                    results.append(error_result)
                    update_progress_concurrent(error_result)
        
        # Final progress logging
        self._log_final_progress()
        
        return results
    
    def execute_category(self, category: str, sequential: bool = True, **kwargs) -> List[TestResult]:
        """
        Execute all tests in a specific category
        
        Args:
            category: Category name
            sequential: Whether to run sequentially (True) or concurrently (False)
            **kwargs: Additional arguments for execution method (including enable_performance_monitoring)
            
        Returns:
            List of TestResult objects
        """
        test_ids = self.get_test_ids_by_category(category)
        
        if not test_ids:
            logger.warning(f"No tests found for category: {category}")
            return []
        
        logger.info(f"Executing {len(test_ids)} tests from category: {category}")
        
        if sequential:
            # Extract only the parameters that execute_sequential accepts
            delay = kwargs.get('delay')
            enable_perf = kwargs.get('enable_performance_monitoring', False)
            return self.execute_sequential(test_ids, delay=delay, enable_performance_monitoring=enable_perf)
        else:
            # Use concurrent execution
            workers = kwargs.get('workers', 3)
            enable_perf = kwargs.get('enable_performance_monitoring', False)
            return self.execute_concurrent(test_ids, workers, enable_perf)
    
    def get_progress(self) -> ExecutionProgress:
        """Get current execution progress information"""
        import time
        current_time = time.time()
        return self.execution_progress or ExecutionProgress(
            total_tests=0,
            completed_tests=0, 
            successful_tests=0,
            failed_tests=0,
            current_test=None,
            estimated_remaining_time=0.0,
            average_execution_time=0.0,
            start_time=current_time,
            elapsed_time=0.0
        )
    
    # Private helper methods
    
    def _detect_api_type(self, endpoint: str) -> str:
        """
        Detect API type from endpoint URL
        
        Args:
            endpoint: API endpoint URL
            
        Returns:
            'chat' or 'completions'
        """
        if "/chat/completions" in endpoint:
            return "chat"
        elif "/completions" in endpoint:
            return "completions"
        else:
            # Default based on common patterns
            return "completions"
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "api_defaults": {
                "timeout": 600,
                "retry_attempts": 3,
                "retry_delay": 1.0
            },
            "execution_defaults": {
                "sequential_delay": 1.0,
                "progress_reporting": True,
                "save_raw_responses": True
            },
            "output_settings": {
                "results_directory": "test_results",
                "include_metadata": True,
                "pretty_print": True
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Error loading config file: {e}. Using defaults.")
        
        return default_config
    
    def _make_api_request(self, test_case: Dict) -> Tuple[bool, Dict, str]:
        """
        Make API request with retry logic
        
        Args:
            test_case: Test case dictionary
            
        Returns:
            Tuple of (success, response_data, error_message)
        """
        # Build request payload based on API type
        if self.api_config.api_type == "chat":
            payload = self._build_chat_payload(test_case)
        else:
            payload = self._build_completions_payload(test_case)
        
        # Attempt request with retries
        for attempt in range(self.api_config.retry_attempts):
            try:
                start_time = time.time()
                response = requests.post(
                    self.api_config.endpoint,
                    headers=self.api_config.headers,
                    json=payload,
                    timeout=self.api_config.timeout
                )
                end_time = time.time()
                
                response.raise_for_status()
                response_data = response.json()
                
                # Add timing information
                response_data['_execution_time'] = end_time - start_time
                
                return True, response_data, ""
                
            except requests.exceptions.Timeout:
                error_msg = f"Request timeout (attempt {attempt + 1})"
                logger.warning(error_msg)
                if attempt < self.api_config.retry_attempts - 1:
                    time.sleep(self.api_config.retry_delay * (attempt + 1))
                    continue
                return False, {}, error_msg
                
            except requests.exceptions.RequestException as e:
                error_msg = f"Request failed: {e}"
                logger.warning(error_msg)
                if attempt < self.api_config.retry_attempts - 1:
                    time.sleep(self.api_config.retry_delay * (attempt + 1))
                    continue
                return False, {}, error_msg
            
            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                logger.error(error_msg)
                return False, {}, error_msg
        
        return False, {}, "Max retries exceeded"
    
    def _build_completions_payload(self, test_case: Dict) -> Dict:
        """Build payload for completions API"""
        return {
            "model": self.api_config.model,
            "prompt": test_case.get('prompt', ''),
            **test_case.get('parameters', {})
        }
    
    def _build_chat_payload(self, test_case: Dict) -> Dict:
        """Build payload for chat API"""
        # Convert test case to chat format
        messages = []
        
        # Check if test case already has messages format (from instruct tests)
        if 'messages' in test_case:
            messages = test_case['messages']
        else:
            # Convert prompt-based test to chat format (from base model tests)
            messages = [{"role": "user", "content": test_case.get('prompt', '')}]
        
        # Convert parameters for chat API
        params = self._convert_params_for_chat(test_case.get('parameters', {}))
        
        return {
            "model": self.api_config.model,
            "messages": messages,
            **params
        }
    
    def _convert_params_for_chat(self, params: Dict) -> Dict:
        """Convert completion parameters to chat parameters"""
        chat_params = params.copy()
        
        # Remove parameters that don't exist in chat API
        chat_params.pop('exclude', None)  # instruct-specific parameter
        chat_params.pop('effort', None)   # instruct-specific parameter
        
        return chat_params
    
    def _create_success_result(self, test_id: str, test_case: Dict, api_response: Dict) -> TestResult:
        """Create TestResult for successful execution"""
        execution_time = api_response.get('_execution_time', 0.0)
        
        # Extract response text based on API type with proper null checks
        completion_text = ""
        choices = api_response.get("choices", [])
        if choices and len(choices) > 0:
            if self.api_config.api_type == "chat":
                # Chat API: response in choices[0].message.content
                message = choices[0].get("message", {})
                completion_text = message.get("content", "") if isinstance(message, dict) else ""
            else:
                # Completions API: response in choices[0].text
                completion_text = choices[0].get("text", "") if isinstance(choices[0], dict) else ""
        elif "content" in api_response:
            # llama.cpp format: response directly in content field
            completion_text = api_response.get("content", "")
        else:
            logger.warning(f"API response missing choices array: {api_response}")
        
        # Extract token usage
        usage = api_response.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        
        # Handle llama.cpp token counts if OpenAI format not available
        if completion_tokens == 0 and prompt_tokens == 0:
            completion_tokens = api_response.get("tokens_predicted", 0)
            prompt_tokens = api_response.get("tokens_evaluated", 0)
        
        tokens_per_second = 0.0
        if completion_tokens > 0 and execution_time > 0:
            tokens_per_second = completion_tokens / execution_time
        
        # Perform automatic reasoning evaluation if available
        evaluation_result = None
        reasoning_score = None
        reasoning_type = None
        
        # Perform automatic reasoning evaluation if available
        if (EVALUATION_AVAILABLE or ENHANCED_EVALUATION_AVAILABLE) and completion_text.strip():
            try:
                # Determine reasoning type from test metadata
                test_reasoning_type = self._get_reasoning_type_for_test(test_case)
                
                # Choose evaluation approach based on configuration
                eval_result = self._perform_evaluation(
                    completion_text, test_case, test_id, test_reasoning_type
                )
                
                if eval_result:
                    # Extract results (supports both basic and enhanced evaluation results)
                    evaluation_result = self._extract_evaluation_result(eval_result)
                    reasoning_score = evaluation_result.get('overall_score', 0)
                    reasoning_type = evaluation_result.get('reasoning_type', 'unknown')
                    
                    # Enhanced logging for enhanced evaluation
                    if hasattr(eval_result, 'enhanced_metrics') and eval_result.enhanced_metrics:
                        logger.info(f"Enhanced evaluation completed for {test_id}: {reasoning_score}/100 "
                                  f"(exact: {eval_result.enhanced_metrics.exact_match_score:.2f}, "
                                  f"partial: {eval_result.enhanced_metrics.partial_match_score:.2f}, "
                                  f"semantic: {eval_result.enhanced_metrics.semantic_similarity_score:.2f})")
                    else:
                        logger.info(f"Evaluation completed for {test_id}: {reasoning_score}/100")
                
            except Exception as e:
                logger.warning(f"Evaluation failed for {test_id}: {e}")
        
        return TestResult(
            test_id=test_id,
            test_name=test_case.get('name', test_id),
            success=True,
            response_text=completion_text,
            execution_time=execution_time,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=tokens_per_second,
            error_message=None,
            timestamp=datetime.now().isoformat(),
            api_response=api_response,
            evaluation_result=evaluation_result,
            reasoning_score=reasoning_score,
            reasoning_type=reasoning_type
        )
    
    def _create_error_result(self, test_id: str, error_message: str, api_response: Dict = None) -> TestResult:
        """Create TestResult for failed execution"""
        test_case = self.tests.get(test_id, {})
        
        return TestResult(
            test_id=test_id,
            test_name=test_case.get('name', test_id),
            success=False,
            response_text="",
            execution_time=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            tokens_per_second=0.0,
            error_message=error_message,
            timestamp=datetime.now().isoformat(),
            api_response=api_response or {}
        )
    
    def _get_reasoning_type_for_test(self, test_case: Dict) -> ReasoningType:
        """
        Determine the appropriate ReasoningType for a test case
        
        Args:
            test_case: Test case dictionary
            
        Returns:
            ReasoningType: The appropriate reasoning type for evaluation
        """
        if not EVALUATION_AVAILABLE:
            return None
            
        # Check if test has explicit reasoning_type
        if 'reasoning_type' in test_case:
            reasoning_type_str = test_case['reasoning_type'].lower()
            try:
                return ReasoningType(reasoning_type_str)
            except ValueError:
                pass
        
        # Infer from category
        category = test_case.get('category', '').lower()
        category_to_reasoning_type = {
            'chain_of_thought': ReasoningType.CHAIN_OF_THOUGHT,
            'mathematical_reasoning': ReasoningType.MATHEMATICAL,
            'multi_hop_inference': ReasoningType.MULTI_HOP,
            'verification_loops': ReasoningType.VERIFICATION,
            'backward_reasoning': ReasoningType.BACKWARD,
            'scaffolded_reasoning': ReasoningType.SCAFFOLDED,
            'complex_synthesis': ReasoningType.GENERAL
        }
        
        return category_to_reasoning_type.get(category, ReasoningType.GENERAL)

    def _perform_evaluation(self, response_text: str, test_case: Dict, test_id: str, reasoning_type) -> Optional[Any]:
        """
        Perform evaluation using appropriate evaluator based on configuration
        
        Args:
            response_text: The model's response
            test_case: Test case dictionary
            test_id: Test identifier
            reasoning_type: Determined reasoning type
            
        Returns:
            Evaluation result (basic or enhanced)
        """
        # Check if enhanced evaluation is requested and available
        if (hasattr(self, 'enhanced_evaluation') and self.enhanced_evaluation and 
            ENHANCED_EVALUATION_AVAILABLE):
            return self._perform_enhanced_evaluation(response_text, test_case, test_id, reasoning_type)
        
        # Check evaluation mode configuration
        elif (hasattr(self, 'evaluation_mode') and self.evaluation_mode in ['enhanced', 'full'] and 
              ENHANCED_EVALUATION_AVAILABLE):
            return self._perform_enhanced_evaluation(response_text, test_case, test_id, reasoning_type)
        
        # Fallback to basic evaluation
        elif EVALUATION_AVAILABLE:
            return self._perform_basic_evaluation(response_text, test_case, test_id, reasoning_type)
        
        return None
    
    def _perform_basic_evaluation(self, response_text: str, test_case: Dict, test_id: str, reasoning_type) -> Any:
        """Perform basic evaluation using standard UniversalEvaluator"""
        return evaluate_reasoning(
            response_text=response_text,
            test_name=test_case.get('name', test_id),
            test_category=test_case.get('category'),
            reasoning_type=reasoning_type
        )
    
    def _perform_enhanced_evaluation(self, response_text: str, test_case: Dict, test_id: str, reasoning_type) -> Any:
        """Perform enhanced evaluation with multi-tier scoring"""
        enhanced_evaluator = EnhancedUniversalEvaluator()
        
        # Check if we should use full enhanced evaluation with test definitions
        if (hasattr(self, 'evaluation_mode') and self.evaluation_mode == 'full'):
            # Use the full enhanced evaluation with test definitions
            return enhanced_evaluator.evaluate_response_enhanced(
                response_text=response_text,
                test_definition=test_case,
                test_name=test_case.get('name', test_id),
                reasoning_type=reasoning_type
            )
        else:
            # Use basic enhanced evaluation (backward compatible)
            return enhanced_evaluator.evaluate_response(
                response_text=response_text,
                test_name=test_case.get('name', test_id),
                reasoning_type=reasoning_type,
                test_category=test_case.get('category')
            )
    
    def _extract_evaluation_result(self, eval_result) -> Dict[str, Any]:
        """Extract evaluation result dictionary supporting both basic and enhanced results"""
        
        # For enhanced evaluations, use the enhanced_metrics overall_score 
        if hasattr(eval_result, 'enhanced_metrics') and eval_result.enhanced_metrics:
            overall_score = eval_result.enhanced_metrics.overall_score
            logger.info(f"BENCHMARK_RUNNER: Using enhanced overall_score: {overall_score:.1f}")
        else:
            overall_score = eval_result.metrics.overall_score
            logger.info(f"BENCHMARK_RUNNER: Using base overall_score: {overall_score:.1f}")
        
        base_result = {
            'overall_score': overall_score,
            'metrics': asdict(eval_result.metrics),
            'reasoning_type': eval_result.reasoning_type.value,
            'recommendations': eval_result.recommendations,
            'detailed_analysis': eval_result.detailed_analysis
        }
        
        # Add enhanced metrics if available
        if hasattr(eval_result, 'enhanced_metrics') and eval_result.enhanced_metrics:
            base_result['enhanced_metrics'] = asdict(eval_result.enhanced_metrics)
        
        if hasattr(eval_result, 'scoring_breakdown') and eval_result.scoring_breakdown:
            base_result['scoring_breakdown'] = eval_result.scoring_breakdown
            
        if hasattr(eval_result, 'integration_analysis') and eval_result.integration_analysis:
            base_result['integration_analysis'] = eval_result.integration_analysis
            
        return base_result
    
    def set_progress_callback(self, callback):
        """Set callback function for progress updates"""
        self._progress_callback = callback
    
    def set_verbose_logging(self, verbose: bool):
        """Enable or disable verbose logging"""
        self._verbose_logging = verbose
    
    def enable_performance_monitoring(self, enable: bool = True):
        """Enable or disable comprehensive performance monitoring"""
        self._enable_performance_monitoring = enable
        if enable:
            logger.info("Performance monitoring enabled - will collect GPU, CPU, memory, and I/O metrics")
        else:
            logger.info("Performance monitoring disabled")
    
    def _initialize_progress(self, total_tests: int, start_time: float):
        """Initialize progress tracking"""
        self.execution_progress = ExecutionProgress(
            total_tests=total_tests,
            completed_tests=0,
            successful_tests=0,
            failed_tests=0,
            current_test=None,
            estimated_remaining_time=0.0,
            average_execution_time=0.0,
            start_time=start_time,
            elapsed_time=0.0,
            total_tokens_generated=0
        )
        logger.info(f"Initialized progress tracking for {total_tests} tests")
    
    def _update_progress(self, test_result: TestResult = None, current_test_name: str = None):
        """Update progress tracking"""
        if not self.execution_progress:
            return
        
        current_time = time.time()
        self.execution_progress.elapsed_time = current_time - self.execution_progress.start_time
        
        if test_result:
            self.execution_progress.completed_tests += 1
            if test_result.success:
                self.execution_progress.successful_tests += 1
                self.execution_progress.total_tokens_generated += test_result.completion_tokens
            else:
                self.execution_progress.failed_tests += 1
        
        if current_test_name:
            self.execution_progress.current_test = current_test_name
            # Extract category from test name if available
            test_data = self.tests.get(current_test_name.split(':')[0], {})
            self.execution_progress.current_category = test_data.get('category', 'unknown')
        
        # Update estimates
        self.execution_progress.update_estimates()
        
        # Update average tokens per second
        if self.execution_progress.total_tokens_generated > 0 and self.execution_progress.elapsed_time > 0:
            self.execution_progress.average_tokens_per_second = (
                self.execution_progress.total_tokens_generated / self.execution_progress.elapsed_time
            )
        
        # Call progress callback if set
        if self._progress_callback:
            self._progress_callback(self.execution_progress)
        
        # Log progress periodically
        if current_time - self._last_progress_update >= 5.0:  # Every 5 seconds
            self._log_progress()
            self._last_progress_update = current_time
    
    def _log_progress(self):
        """Log current progress"""
        if not self.execution_progress or not self._verbose_logging:
            return
        
        progress = self.execution_progress
        
        # Create progress bar
        bar_length = 40
        filled_length = int(bar_length * progress.completion_percentage // 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # Format time remaining
        if progress.estimated_remaining_time > 3600:
            time_remaining = f"{progress.estimated_remaining_time/3600:.1f}h"
        elif progress.estimated_remaining_time > 60:
            time_remaining = f"{progress.estimated_remaining_time/60:.1f}m"
        else:
            time_remaining = f"{progress.estimated_remaining_time:.1f}s"
        
        logger.info(
            f"Progress: |{bar}| {progress.completed_tests}/{progress.total_tests} "
            f"({progress.completion_percentage:.1f}%) "
            f"Success: {progress.success_rate:.1f}% "
            f"ETA: {time_remaining} "
            f"Speed: {progress.tests_per_second:.2f} tests/s "
            f"Tokens: {progress.average_tokens_per_second:.1f} T/s"
        )
        
        if progress.current_test:
            logger.info(f"Current: {progress.current_test}")
    
    def _log_final_progress(self):
        """Log final progress summary"""
        if not self.execution_progress:
            return
        
        progress = self.execution_progress
        logger.info("=" * 60)
        logger.info("EXECUTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {progress.total_tests}")
        logger.info(f"Completed: {progress.completed_tests}")
        logger.info(f"Successful: {progress.successful_tests} ({progress.success_rate:.1f}%)")
        logger.info(f"Failed: {progress.failed_tests}")
        logger.info(f"Total Time: {progress.elapsed_time:.1f}s")
        logger.info(f"Average Time per Test: {progress.average_execution_time:.1f}s")
        logger.info(f"Tests per Second: {progress.tests_per_second:.2f}")
        logger.info(f"Total Tokens Generated: {progress.total_tokens_generated:,}")
        logger.info(f"Average Tokens per Second: {progress.average_tokens_per_second:.1f}")
        logger.info("=" * 60)
    
    def generate_evaluation_summary(self, results: List[TestResult]) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation summary from test results
        
        Args:
            results: List of TestResult objects with evaluation data
            
        Returns:
            Dictionary containing evaluation summary statistics
        """
        if not EVALUATION_AVAILABLE:
            return {"error": "Evaluation not available"}
        
        # Filter results with evaluation data
        evaluated_results = [r for r in results if r.evaluation_result is not None]
        
        if not evaluated_results:
            return {"error": "No evaluation results found"}
        
        # Collect scores and metrics
        scores = [r.reasoning_score for r in evaluated_results]
        reasoning_types = [r.reasoning_type for r in evaluated_results]
        
        # Calculate statistics
        summary = {
            "total_tests": len(results),
            "evaluated_tests": len(evaluated_results),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "reasoning_type_distribution": {},
            "category_performance": {},
            "metric_averages": {}
        }
        
        # Reasoning type distribution
        for rt in reasoning_types:
            summary["reasoning_type_distribution"][rt] = summary["reasoning_type_distribution"].get(rt, 0) + 1
        
        # Category performance
        for result in evaluated_results:
            test_case = self.tests.get(result.test_id, {})
            category = test_case.get('category', 'unknown')
            
            if category not in summary["category_performance"]:
                summary["category_performance"][category] = {"scores": [], "count": 0}
            
            summary["category_performance"][category]["scores"].append(result.reasoning_score)
            summary["category_performance"][category]["count"] += 1
        
        # Calculate category averages
        for category, data in summary["category_performance"].items():
            if data["scores"]:
                data["average_score"] = sum(data["scores"]) / len(data["scores"])
        
        # Metric averages across all evaluations
        if evaluated_results:
            all_metrics = {}
            for result in evaluated_results:
                metrics = result.evaluation_result.get('metrics', {})
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(metric_value)
            
            for metric_name, values in all_metrics.items():
                summary["metric_averages"][metric_name] = sum(values) / len(values)
        
        return summary
    
    def run_single_test(self, domain_path: str, test_id: str, enhanced_evaluation: bool = False) -> Optional[TestResult]:
        """
        Interface method for calibration validation framework
        
        Simplified interface for running single tests with evaluation.
        Used by tests/functional/calibration_validator.py
        
        Args:
            domain_path: Path to domain test file (e.g., "domains/reasoning/base_models/easy.json")
            test_id: Test identifier within the domain file
            enhanced_evaluation: Use enhanced evaluation if available
            
        Returns:
            TestResult with enhanced evaluation if successful, None if failed
        """
        try:
            # Load test from domain path if not already loaded
            if not self.tests or test_id not in self.tests:
                success = self.load_test_suite(domain_path)
                if not success:
                    logger.error(f"Failed to load test suite from {domain_path}")
                    return None
            
            # Configure API if not configured (use override endpoint or defaults)
            if not self.api_config:
                endpoint = self._api_endpoint_override or "http://localhost:8004"
                self.configure_api(endpoint, "llama-cpp-server")
            
            # Execute test
            result = self.execute_single_test(test_id, enable_performance_monitoring=False)
            
            if not result or not result.success:
                logger.error(f"Test execution failed for {test_id}")
                return None
            
            # Apply enhanced evaluation if requested and available
            if enhanced_evaluation and ENHANCED_EVALUATION_AVAILABLE:
                try:
                    enhancer = EnhancedUniversalEvaluator()
                    
                    # Get test definition for enhanced evaluation
                    test_definition = self.tests.get(test_id, {})
                    
                    enhanced_result = enhancer.evaluate_response_enhanced(
                        response_text=result.response_text,
                        test_definition=test_definition
                    )
                    
                    # Attach enhanced score to result
                    if hasattr(enhanced_result, 'enhanced_metrics'):
                        result.enhanced_score = enhanced_result.enhanced_metrics.overall_score
                        logger.info(f"Enhanced evaluation score: {result.enhanced_score:.1f}")
                    
                except Exception as e:
                    logger.warning(f"Enhanced evaluation failed: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in run_single_test: {e}")
            return None


# Convenience functions for quick usage

def load_and_configure_runner(test_definitions_dir: str = "test_definitions", 
                             api_endpoint: str = None,
                             test_type: str = "base",
                             domain: str = None) -> BenchmarkTestRunner:
    """
    Load and configure a BenchmarkTestRunner with domain-based structure
    
    Args:
        test_definitions_dir: Legacy parameter (now ignored, kept for compatibility)
        api_endpoint: Optional API endpoint URL to override defaults
        test_type: Type of tests to load ("base" or "instruct")
        domain: Specific domain to load ("reasoning", "linux", etc.). If None, loads all domains.
        
    Returns:
        Configured BenchmarkTestRunner instance
    """
    runner = BenchmarkTestRunner(api_endpoint=api_endpoint)
    
    # Get the directory where benchmark_runner.py is located (benchmark_tests/)
    runner_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use TestSuiteManager to discover available suites
    suite_manager = TestSuiteManager()
    suite_manager.discover_test_suites(runner_dir)
    
    # Determine model type suffix for suite IDs
    model_type_suffix = "base" if test_type == "base" else "instruct"
    
    # Find all suites matching the test type
    matching_suites = []
    for suite_id, suite in suite_manager.available_suites.items():
        if suite_id.endswith(f"_{model_type_suffix}"):
            if domain is None or suite_id.startswith(f"{domain}_"):
                matching_suites.append(suite)
    
    if not matching_suites:
        available_domains = set()
        for suite_id in suite_manager.available_suites.keys():
            if suite_id.endswith(f"_{model_type_suffix}"):
                domain_name = suite_id.replace(f"_{model_type_suffix}", "")
                available_domains.add(domain_name)
        
        domain_list = ", ".join(sorted(available_domains))
        domain_msg = f" for domain '{domain}'" if domain else ""
        raise ValueError(f"No test suites found for test type '{test_type}'{domain_msg}. Available domains: {domain_list}")
    
    # Load data from all matching suites
    all_tests = {}
    all_categories = {"categories": {}}
    metadata_info = {"name": "Combined Test Suite", "version": "1.0", "description": "Multiple domains combined"}
    
    for suite in matching_suites:
        # Load test files from each domain
        domain_name = suite.suite_id.replace(f"_{model_type_suffix}", "")
        
        # Determine correct model type directory name
        if test_type == "instruct":
            model_dir = "instruct_models"
        else:
            model_dir = "base_models"
        
        domain_base_dir = os.path.join(runner_dir, "domains", domain_name, model_dir)
        
        # Load categories from domain
        categories_path = os.path.join(domain_base_dir, "categories.json")
        if os.path.exists(categories_path):
            with open(categories_path, 'r', encoding='utf-8') as f:
                domain_categories = json.load(f)
                if "categories" in domain_categories:
                    all_categories["categories"].update(domain_categories["categories"])
        
        # Load test files from domain
        test_files_pattern = os.path.join(domain_base_dir, "*.json")
        import glob
        test_files = [f for f in glob.glob(test_files_pattern) if not f.endswith("categories.json") and not f.endswith("test_suite_metadata.json")]
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    if "tests" in file_data:
                        # Handle both array and object formats
                        tests_data = file_data["tests"]
                        if isinstance(tests_data, list):
                            # Convert array of tests to dictionary keyed by test ID
                            for test in tests_data:
                                if "id" in test:
                                    all_tests[test["id"]] = test
                        elif isinstance(tests_data, dict):
                            all_tests.update(tests_data)
            except Exception as e:
                print(f"Warning: Failed to load test file {test_file}: {e}")
    
    # Load the combined data into runner
    runner.tests = all_tests
    runner.categories = all_categories
    
    # Set basic metadata
    runner.test_metadata = metadata_info
    
    return runner


if __name__ == "__main__":
    """Command-line interface for BenchmarkTestRunner"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="BenchmarkTestRunner - Flexible LLM Test Execution Engine")
    parser.add_argument("--endpoint", "-e", 
                       default="http://127.0.0.1:8004/v1/completions",
                       help="API endpoint URL (default: %(default)s)")
    parser.add_argument("--model", "-m", 
                       default="/app/models/hf/DeepSeek-R1-0528-Qwen3-8b",
                       help="Model name (default: %(default)s)")
    parser.add_argument("--mode", 
                       choices=["single", "sequential", "concurrent", "category"],
                       default="single",
                       help="Execution mode (default: %(default)s)")
    parser.add_argument("--test-id", "-t",
                       help="Specific test ID to run (for single mode)")
    parser.add_argument("--category", "-c",
                       help="Category to run (for category mode)")
    parser.add_argument("--workers", "-w", 
                       type=int, default=3,
                       help="Number of concurrent workers (default: %(default)s)")
    parser.add_argument("--delay", "-d", 
                       type=float, default=1.0,
                       help="Delay between sequential tests in seconds (default: %(default)s)")
    parser.add_argument("--output-dir", "-o",
                       default="test_results",
                       help="Output directory for results (default: %(default)s)")
    parser.add_argument("--test-definitions", 
                       default="test_definitions",
                       help="Path to test definitions directory (default: %(default)s)")
    parser.add_argument("--test-type",
                       choices=["base", "instruct"],
                       default="base", 
                       help="Type of tests to run (default: %(default)s)")
    parser.add_argument("--list-categories", action="store_true",
                       help="List available test categories and exit")
    parser.add_argument("--list-tests", action="store_true", 
                       help="List available tests and exit")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be executed without running tests")
    parser.add_argument("--evaluation", action="store_true",
                       help="Enable automatic universal evaluation (requires UniversalEvaluator)")
    parser.add_argument("--enhanced-evaluation", action="store_true",
                       help="Enable enhanced evaluation with multi-tier scoring (Phase 1)")
    parser.add_argument("--evaluation-mode", 
                       choices=["basic", "enhanced", "full"],
                       default="basic",
                       help="Evaluation mode: basic (standard), enhanced (multi-tier), full (with test definitions)")
    parser.add_argument("--domain-focus",
                       choices=["reasoning", "creativity", "language", "social", "integration", "knowledge", "auto"],
                       default="auto", 
                       help="Focus evaluation on specific domain (for Phase 1 testing)")
    parser.add_argument("--eval-summary", action="store_true",
                       help="Generate detailed evaluation summary report")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging with progress reporting")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress progress logging (only show final results)")
    parser.add_argument("--discover-suites", action="store_true",
                       help="Discover and list all available test suites")
    parser.add_argument("--suite-stats", 
                       help="Show detailed statistics for a specific suite ID")
    parser.add_argument("--filter-by", 
                       help="Filter tests by criteria (format: key=value,key=value)")
    parser.add_argument("--category-info", 
                       help="Show detailed information about a specific category")
    parser.add_argument("--reasoning-focus", 
                       help="Filter tests by reasoning focus keywords (comma-separated)")
    parser.add_argument("--performance-monitoring", "--perf", action="store_true",
                       help="Enable comprehensive hardware performance monitoring (RTX 5090, AMD Ryzen 9950X, 128GB DDR5 RAM)")
    
    args = parser.parse_args()
    
    print("BenchmarkTestRunner - Test Execution Engine")
    print("=" * 50)
    
    # Handle suite management commands first (independent of test loading)
    if args.discover_suites or args.suite_stats or args.category_info or args.filter_by or args.reasoning_focus:
        # Initialize standalone suite manager
        suite_manager = TestSuiteManager()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        suite_manager.discover_test_suites(current_dir)
        
        if args.discover_suites:
            print("\nDiscovered Test Suites:")
            print("=" * 50)
            for suite in suite_manager.available_suites.values():
                print(f"Suite ID: {suite.suite_id}")
                print(f"Name: {suite.name}")
                print(f"Type: {suite.test_type}")
                print(f"Version: {suite.version}")
                print(f"Total Tests: {suite.total_tests}")
                print(f"Categories: {len(suite.categories)}")
                print(f"Description: {suite.description}")
                print("-" * 30)
            sys.exit(0)
        
        if args.suite_stats:
            print(f"\nDetailed Statistics for Suite: {args.suite_stats}")
            print("=" * 60)
            stats = suite_manager.get_suite_statistics(args.suite_stats)
            if stats:
                suite_info = stats['suite_info']
                print(f"Name: {suite_info['name']}")
                print(f"Type: {suite_info['test_type']}")
                print(f"Version: {suite_info['version']}")
                print(f"Total Tests: {suite_info['total_tests']}")
                print(f"Estimated Total Time: {stats['total_estimated_time']:.1f} seconds")
                
                print(f"\nCategory Breakdown:")
                for cat, info in stats['category_breakdown'].items():
                    print(f"  {cat}: {info['test_count']} tests ({info['estimated_time']:.1f}s)")
                    print(f"    {info['description']}")
                
                print(f"\nReasoning Focus Analysis:")
                for focus, count in stats['reasoning_focus_analysis'].items():
                    print(f"  {focus}: {count} tests")
                
                if stats['temperature_distribution']:
                    temp_dist = stats['temperature_distribution']
                    print(f"\nTemperature Range Distribution:")
                    print(f"  Min: {temp_dist['min_temp']:.3f}")
                    print(f"  Max: {temp_dist['max_temp']:.3f}")
                    print(f"  Avg Range: {temp_dist['avg_min_temp']:.3f} - {temp_dist['avg_max_temp']:.3f}")
            else:
                print(f"❌ Suite '{args.suite_stats}' not found")
            sys.exit(0)
        
        if args.category_info:
            print(f"\nCategory Information: {args.category_info}")
            print("=" * 50)
            cat_info = suite_manager.get_category_info(args.category_info, args.test_type)
            if cat_info:
                print(f"Name: {cat_info.name}")
                print(f"Description: {cat_info.description}")
                print(f"Test Count: {cat_info.test_count}")
                print(f"Reasoning Focus: {cat_info.reasoning_focus}")
                print(f"Temperature Range: {cat_info.temperature_range[0]:.3f} - {cat_info.temperature_range[1]:.3f}")
                print(f"Test Range: {cat_info.test_range[0]} - {cat_info.test_range[1]}")
                print(f"Difficulty Level: {cat_info.difficulty_level}")
                print(f"Estimated Time per Test: {cat_info.estimated_time_per_test:.1f}s")
                print(f"Test IDs: {', '.join(cat_info.test_ids[:5])}{'...' if len(cat_info.test_ids) > 5 else ''}")
            else:
                print(f"❌ Category '{args.category_info}' not found for test type '{args.test_type}'")
            sys.exit(0)
        
        if args.filter_by or args.reasoning_focus:
            criteria = {}
            
            if args.filter_by:
                print(f"\nFiltering Tests by Criteria: {args.filter_by}")
                # Parse filter criteria
                for criterion in args.filter_by.split(','):
                    if '=' in criterion:
                        key, value = criterion.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Convert some values to appropriate types
                        if key == 'temperature_range' and ',' in value:
                            min_temp, max_temp = map(float, value.split(','))
                            criteria[key] = (min_temp, max_temp)
                        elif key == 'max_time':
                            criteria[key] = float(value)
                        else:
                            criteria[key] = value
            
            if args.reasoning_focus:
                print(f"\nFiltering Tests by Reasoning Focus: {args.reasoning_focus}")
                criteria['reasoning_focus'] = args.reasoning_focus
            
            print("=" * 50)
            matching_tests = suite_manager.filter_tests_by_criteria(args.test_type, **criteria)
            
            print(f"Found {len(matching_tests)} tests matching criteria:")
            for test_id in matching_tests[:20]:  # Show first 20
                print(f"  {test_id}")
            
            if len(matching_tests) > 20:
                print(f"  ... and {len(matching_tests) - 20} more")
            
            sys.exit(0)
    
    print(f"API Endpoint: {args.endpoint}")
    print(f"Model: {args.model}")
    print(f"Test Type: {args.test_type}")
    print(f"Mode: {args.mode}")
    
    # Display evaluation configuration
    if args.evaluation or args.enhanced_evaluation:
        eval_info = []
        if args.evaluation:
            eval_info.append("basic")
        if args.enhanced_evaluation and ENHANCED_EVALUATION_AVAILABLE:
            eval_info.append(f"enhanced ({args.evaluation_mode})")
        elif args.enhanced_evaluation and not ENHANCED_EVALUATION_AVAILABLE:
            eval_info.append("enhanced (unavailable)")
        
        print(f"Evaluation: {', '.join(eval_info)}")
        
        if args.domain_focus != "auto":
            print(f"Domain Focus: {args.domain_focus}")
    
    if args.enhanced_evaluation and not ENHANCED_EVALUATION_AVAILABLE:
        print("⚠️  Enhanced evaluation requested but EnhancedUniversalEvaluator not available")
    
    # Load and configure runner
    try:
        runner = load_and_configure_runner(test_definitions_dir=args.test_definitions, 
                                         api_endpoint=args.endpoint,
                                         test_type=args.test_type)
        runner.configure_api(args.endpoint, args.model)
        
        # Configure logging verbosity
        if args.quiet:
            runner.set_verbose_logging(False)
            logging.getLogger().setLevel(logging.WARNING)
        elif args.verbose:
            runner.set_verbose_logging(True)
            logging.getLogger().setLevel(logging.INFO)
        
        # Configure enhanced evaluation options
        if hasattr(args, 'enhanced_evaluation'):
            runner.enhanced_evaluation = args.enhanced_evaluation
        if hasattr(args, 'evaluation_mode'):
            runner.evaluation_mode = args.evaluation_mode
        if hasattr(args, 'domain_focus'):
            runner.domain_focus = args.domain_focus
        
        print(f"Loaded {len(runner.tests)} tests from {args.test_definitions}")
        
    except Exception as e:
        print(f"❌ Error loading test suite: {e}")
        sys.exit(1)
    
    # Handle regular list commands (these need the runner loaded)
    if args.list_categories:
        print("\nAvailable Categories:")
        if runner.categories and 'categories' in runner.categories:
            for category, info in runner.categories['categories'].items():
                test_count = len(info.get('test_ids', []))
                print(f"  {category}: {test_count} tests - {info.get('description', 'No description')}")
        else:
            # Fallback: group by category field in tests
            categories = {}
            for test in runner.tests.values():
                cat = test.get('category', 'unknown')
                categories[cat] = categories.get(cat, 0) + 1
            for category, count in categories.items():
                print(f"  {category}: {count} tests")
        sys.exit(0)
    
    if args.list_tests:
        print("\nAvailable Tests:")
        tests_to_show = runner.tests.items()
        
        # If category is specified, filter tests
        if args.category:
            filtered_tests = [(test_id, test) for test_id, test in tests_to_show 
                             if test.get('category', 'unknown') == args.category]
            if not filtered_tests:
                print(f"  No tests found for category: {args.category}")
            else:
                print(f"  Filtered by category '{args.category}':")
                tests_to_show = filtered_tests
        
        for test_id, test in tests_to_show:
            category = test.get('category', 'unknown')
            print(f"  {test_id}: {test.get('name', 'No name')} [{category}]")
        sys.exit(0)
    
    # Determine what to execute
    test_ids_to_run = []
    execution_description = ""
    
    if args.mode == "single":
        if args.test_id:
            if args.test_id in runner.tests:
                test_ids_to_run = [args.test_id]
                execution_description = f"single test: {args.test_id}"
            else:
                print(f"❌ Test ID '{args.test_id}' not found")
                sys.exit(1)
        else:
            # Default to first test
            test_ids_to_run = [list(runner.tests.keys())[0]]
            execution_description = f"single test (first): {test_ids_to_run[0]}"
    
    elif args.mode == "category":
        if not args.category:
            print("❌ Category mode requires --category parameter")
            sys.exit(1)
        test_ids_to_run = runner.get_test_ids_by_category(args.category)
        if not test_ids_to_run:
            print(f"❌ No tests found for category: {args.category}")
            sys.exit(1)
        execution_description = f"category '{args.category}': {len(test_ids_to_run)} tests"
    
    elif args.mode in ["sequential", "concurrent"]:
        test_ids_to_run = list(runner.tests.keys())
        execution_description = f"all {len(test_ids_to_run)} tests ({args.mode})"
    
    print(f"\nPlanned execution: {execution_description}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        for i, test_id in enumerate(test_ids_to_run[:5]):  # Show first 5
            test_case = runner.tests.get(test_id, {})
            test_name = test_case.get('name', f'Unknown test: {test_id}')
            print(f"  {i+1}. {test_id}: {test_name}")
        if len(test_ids_to_run) > 5:
            print(f"  ... and {len(test_ids_to_run) - 5} more tests")
        print(f"\nExecution parameters:")
        print(f"  Workers (concurrent): {args.workers}")
        print(f"  Delay (sequential): {args.delay}s")
        print(f"  Output directory: {args.output_dir}")
        sys.exit(0)
    
    # Execute tests
    print(f"\n🚀 Starting execution...")
    if args.performance_monitoring:
        print("🔍 Hardware performance monitoring enabled (RTX 5090, AMD Ryzen 9950X, 128GB DDR5 RAM)")
    start_time = time.time()
    
    try:
        if args.mode == "single":
            results = [runner.execute_single_test(test_ids_to_run[0], enable_performance_monitoring=args.performance_monitoring)]
        elif args.mode == "sequential":
            results = runner.execute_sequential(test_ids_to_run, delay=args.delay, enable_performance_monitoring=args.performance_monitoring)
        elif args.mode == "concurrent":
            # Smart backend detection for concurrency support
            backend_mode = detect_backend_type()
            if backend_mode == 'concurrent':
                print("✅ vLLM backend detected - using concurrent execution")
                results = runner.execute_concurrent(test_ids_to_run, workers=args.workers, enable_performance_monitoring=args.performance_monitoring)
            else:
                print("⚠️ llama.cpp backend detected - using sequential execution for compatibility")
                results = runner.execute_sequential(test_ids_to_run, delay=args.delay, enable_performance_monitoring=args.performance_monitoring)
        elif args.mode == "category":
            # Smart backend detection for category execution
            backend_mode = detect_backend_type()
            if backend_mode == 'concurrent':
                print("✅ vLLM backend detected - using concurrent execution for category")
                results = runner.execute_category(args.category, sequential=False,
                                                workers=args.workers, enable_performance_monitoring=args.performance_monitoring)
            else:
                print("⚠️ llama.cpp backend detected - using sequential execution for category")
                results = runner.execute_category(args.category, sequential=True,
                                                workers=args.workers, delay=args.delay, enable_performance_monitoring=args.performance_monitoring)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Save results
        success_count = sum(1 for r in results if r.success)
        print(f"Attempting to save {len(results)} results to {args.output_dir}...")
        save_success = runner.save_results(results, args.output_dir)
        if save_success:
            print(f"✓ Successfully saved {len(results)} results to {args.output_dir}")
        else:
            print(f"✗ Failed to save results to {args.output_dir}")
        
        # Summary
        print(f"\n{'='*50}")
        print(f"EXECUTION SUMMARY")
        print(f"{'='*50}")
        print(f"Total tests: {len(results)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {len(results) - success_count}")
        success_rate = (success_count/len(results)*100) if len(results) > 0 else 0
        avg_time_per_test = total_time/len(results) if len(results) > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average time per test: {avg_time_per_test:.1f}s")
        
        if results and success_count > 0:
            avg_tokens_per_sec = sum(r.tokens_per_second for r in results if r.success) / success_count
            print(f"Average tokens/sec: {avg_tokens_per_sec:.1f} T/s")
        
        # Performance monitoring summary
        if args.performance_monitoring and any(r.performance_metrics for r in results):
            perf_results = [r for r in results if r.performance_metrics]
            print(f"\n{'='*50}")
            print(f"HARDWARE PERFORMANCE SUMMARY")
            print(f"{'='*50}")
            
            # Calculate averages
            avg_gpu_usage = sum(r.performance_metrics.gpu_usage_percent for r in perf_results) / len(perf_results)
            avg_cpu_usage = sum(r.performance_metrics.cpu_usage_percent for r in perf_results) / len(perf_results)
            avg_memory_usage = sum(r.performance_metrics.memory_usage_percent for r in perf_results) / len(perf_results)
            avg_gpu_temp = sum(r.performance_metrics.gpu_temp_celsius for r in perf_results) / len(perf_results)
            avg_cpu_temp = sum(r.performance_metrics.cpu_temp_celsius for r in perf_results) / len(perf_results)
            
            print(f"RTX 5090 GPU - Average Usage: {avg_gpu_usage:.1f}%, Temperature: {avg_gpu_temp:.1f}°C")
            print(f"AMD Ryzen CPU - Average Usage: {avg_cpu_usage:.1f}%, Temperature: {avg_cpu_temp:.1f}°C")  
            print(f"128GB DDR5 RAM - Average Usage: {avg_memory_usage:.1f}%")
            
            # Count bottlenecks
            all_bottlenecks = []
            for r in perf_results:
                if r.utilization_report and r.utilization_report.bottlenecks_detected:
                    all_bottlenecks.extend(r.utilization_report.bottlenecks_detected)
            
            if all_bottlenecks:
                bottleneck_counts = {}
                for bottleneck in all_bottlenecks:
                    bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1
                
                print(f"Bottlenecks detected:")
                for bottleneck, count in sorted(bottleneck_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {bottleneck}: {count} occurrences")
        
        # Evaluation summary
        if EVALUATION_AVAILABLE and any(r.evaluation_result for r in results):
            evaluated_results = [r for r in results if r.evaluation_result]
            avg_reasoning_score = sum(r.reasoning_score for r in evaluated_results) / len(evaluated_results)
            print(f"\n{'='*50}")
            print(f"REASONING EVALUATION SUMMARY")
            print(f"{'='*50}")
            print(f"Evaluated tests: {len(evaluated_results)}")
            print(f"Average reasoning score: {avg_reasoning_score:.1f}/100")
            
            # Show per-category scores if available
            category_scores = {}
            for result in evaluated_results:
                test_case = runner.tests.get(result.test_id, {})
                category = test_case.get('category', 'unknown')
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(result.reasoning_score)
            
            if len(category_scores) > 1:
                print(f"\nCategory Performance:")
                for category, scores in category_scores.items():
                    avg_score = sum(scores) / len(scores)
                    print(f"  {category}: {avg_score:.1f}/100 ({len(scores)} tests)")
            
            # Generate detailed summary if requested
            if args.eval_summary:
                print(f"\n{'='*50}")
                print(f"DETAILED EVALUATION ANALYSIS")
                print(f"{'='*50}")
                eval_summary = runner.generate_evaluation_summary(results)
                
                print(f"Score Distribution:")
                print(f"  Min: {eval_summary['min_score']:.1f}")
                print(f"  Max: {eval_summary['max_score']:.1f}")
                print(f"  Average: {eval_summary['average_score']:.1f}")
                
                print(f"\nReasoning Type Distribution:")
                for rt, count in eval_summary['reasoning_type_distribution'].items():
                    print(f"  {rt}: {count} tests")
                
                print(f"\nMetric Averages:")
                for metric, avg in eval_summary['metric_averages'].items():
                    if metric != 'overall_score':  # Already shown above
                        print(f"  {metric.replace('_', ' ').title()}: {avg:.1f}")
        elif args.evaluation and EVALUATION_AVAILABLE:
            print(f"\n⚠️  Evaluation was enabled but no results contain evaluation data")
        elif args.evaluation and not EVALUATION_AVAILABLE:
            print(f"\n⚠️  Evaluation requested but UniversalEvaluator not available")
        
        print(f"\nResults saved to: {args.output_dir}/")
        
    except KeyboardInterrupt:
        print("\n❌ Execution cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        sys.exit(1)