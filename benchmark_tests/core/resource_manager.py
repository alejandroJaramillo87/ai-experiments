#!/usr/bin/env python3
"""
Advanced Resource Management System

Handles memory usage optimization, garbage collection, and resource cleanup 
for large-scale test suite execution (26k+ tests) while maintaining performance.

Features:
- Intelligent memory monitoring and cleanup
- Automatic garbage collection triggers  
- Resource usage analytics and reporting
- Memory leak detection and prevention
- Performance optimization strategies

Designed to keep memory usage under 16GB as specified in requirements.
"""

import gc
import psutil
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import weakref
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class ResourceMetrics:
    """Resource usage metrics snapshot"""
    timestamp: float
    memory_mb: float
    cpu_percent: float
    peak_memory_mb: float
    gc_collections: int
    active_threads: int
    open_files: int

@dataclass 
class ResourceLimits:
    """Resource usage limits and thresholds"""
    max_memory_mb: int = 14000      # 14GB limit (2GB buffer from 16GB requirement)
    cleanup_threshold_mb: int = 10000  # Start cleanup at 10GB
    critical_threshold_mb: int = 12000  # Critical cleanup at 12GB
    max_open_files: int = 1000      # File handle limit
    max_threads: int = 50           # Thread limit
    gc_frequency: int = 100         # Cleanup every N operations

class ResourceTracker:
    """Track resource usage over time"""
    
    def __init__(self, history_size: int = 1000):
        self.process = psutil.Process()
        self.history: List[ResourceMetrics] = []
        self.history_size = history_size
        self.peak_memory = 0.0
        self.operation_count = 0
        self._lock = threading.Lock()
    
    def capture_snapshot(self) -> ResourceMetrics:
        """Capture current resource usage snapshot"""
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # Update peak memory
            self.peak_memory = max(self.peak_memory, memory_mb)
            
            # Get GC statistics
            gc_stats = gc.get_stats()
            total_collections = sum(stat['collections'] for stat in gc_stats)
            
            snapshot = ResourceMetrics(
                timestamp=time.time(),
                memory_mb=memory_mb,
                cpu_percent=self.process.cpu_percent(),
                peak_memory_mb=self.peak_memory,
                gc_collections=total_collections,
                active_threads=threading.active_count(),
                open_files=len(self.process.open_files())
            )
            
            # Store in history (thread-safe)
            with self._lock:
                self.history.append(snapshot)
                if len(self.history) > self.history_size:
                    self.history.pop(0)
            
            return snapshot
            
        except Exception as e:
            logger.warning(f"Failed to capture resource snapshot: {e}")
            return ResourceMetrics(
                timestamp=time.time(), memory_mb=0, cpu_percent=0,
                peak_memory_mb=0, gc_collections=0, active_threads=0, open_files=0
            )
    
    def get_recent_metrics(self, seconds: int = 60) -> List[ResourceMetrics]:
        """Get metrics from recent time period"""
        cutoff_time = time.time() - seconds
        with self._lock:
            return [m for m in self.history if m.timestamp > cutoff_time]
    
    def analyze_memory_trend(self, window_seconds: int = 300) -> Dict[str, float]:
        """Analyze memory usage trends"""
        recent = self.get_recent_metrics(window_seconds)
        if len(recent) < 2:
            return {'trend': 0.0, 'rate_mb_per_min': 0.0, 'stability': 1.0}
        
        # Calculate memory trend
        first_memory = recent[0].memory_mb
        last_memory = recent[-1].memory_mb
        time_diff = recent[-1].timestamp - recent[0].timestamp
        
        if time_diff > 0:
            rate_mb_per_min = (last_memory - first_memory) / (time_diff / 60)
            trend = 1.0 if rate_mb_per_min > 10 else (-1.0 if rate_mb_per_min < -10 else 0.0)
        else:
            rate_mb_per_min = 0.0
            trend = 0.0
        
        # Calculate stability (lower is more stable)
        memories = [m.memory_mb for m in recent]
        avg_memory = sum(memories) / len(memories)
        variance = sum((m - avg_memory) ** 2 for m in memories) / len(memories)
        stability = 1.0 / (1.0 + variance / 100)  # Normalize
        
        return {
            'trend': trend,
            'rate_mb_per_min': rate_mb_per_min,
            'stability': stability,
            'current_mb': last_memory,
            'peak_mb': self.peak_memory
        }

class MemoryOptimizer:
    """Intelligent memory optimization and cleanup"""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.cleanup_callbacks: List[Callable[[], int]] = []
        self.last_cleanup = 0
        self.cleanup_count = 0
        
    def register_cleanup_callback(self, callback: Callable[[], int]):
        """Register a cleanup callback that returns bytes freed"""
        self.cleanup_callbacks.append(callback)
        
    def force_garbage_collection(self) -> int:
        """Force comprehensive garbage collection"""
        logger.debug("Forcing garbage collection")
        
        # Collect all generations
        collected = 0
        for generation in range(3):
            collected += gc.collect(generation)
        
        # Additional cleanup strategies
        gc.collect()  # Final collection
        
        logger.debug(f"Garbage collection freed {collected} objects")
        return collected
        
    def cleanup_large_objects(self) -> int:
        """Clean up large objects and caches"""
        freed_bytes = 0
        
        # Run registered cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                freed_bytes += callback()
            except Exception as e:
                logger.warning(f"Cleanup callback failed: {e}")
        
        return freed_bytes
        
    def optimize_memory_usage(self, current_mb: float, force: bool = False) -> Dict[str, Any]:
        """Optimize memory usage based on current usage and thresholds"""
        optimization_actions = []
        bytes_freed = 0
        
        # Skip if recently cleaned and not forced
        if not force and (time.time() - self.last_cleanup) < 30:
            return {'actions': [], 'bytes_freed': 0, 'memory_mb': current_mb}
        
        # Determine cleanup level
        if current_mb >= self.limits.critical_threshold_mb or force:
            level = 'critical'
        elif current_mb >= self.limits.cleanup_threshold_mb:
            level = 'moderate'
        else:
            level = 'none'
        
        if level != 'none':
            logger.info(f"Starting {level} memory optimization (current: {current_mb:.1f}MB)")
            
            # Garbage collection
            collected_objects = self.force_garbage_collection()
            optimization_actions.append(f"gc_collected_{collected_objects}_objects")
            
            # Cleanup large objects and caches
            if level == 'critical':
                cleanup_bytes = self.cleanup_large_objects()
                bytes_freed += cleanup_bytes
                optimization_actions.append(f"cache_cleanup_{cleanup_bytes}_bytes")
            
            # Update tracking
            self.last_cleanup = time.time()
            self.cleanup_count += 1
            
            logger.info(f"Memory optimization complete: {len(optimization_actions)} actions, "
                       f"{bytes_freed} bytes freed")
        
        return {
            'level': level,
            'actions': optimization_actions,
            'bytes_freed': bytes_freed,
            'memory_mb': current_mb
        }

class ResourceManager:
    """Main resource management controller"""
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        self.limits = limits or ResourceLimits()
        self.tracker = ResourceTracker()
        self.optimizer = MemoryOptimizer(limits)
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.operation_count = 0
        
        logger.info("ResourceManager initialized")
        logger.info(f"Memory limits: {self.limits.max_memory_mb}MB max, "
                   f"{self.limits.cleanup_threshold_mb}MB cleanup threshold")
    
    def start_monitoring(self, interval_seconds: int = 5):
        """Start background resource monitoring"""
        if self.monitoring:
            logger.warning("Resource monitoring already active")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval_seconds,), 
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Resource monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop background resource monitoring"""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5.0)
            logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self, interval_seconds: int):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                snapshot = self.tracker.capture_snapshot()
                
                # Check if optimization needed
                if snapshot.memory_mb >= self.limits.cleanup_threshold_mb:
                    self.optimizer.optimize_memory_usage(snapshot.memory_mb)
                
                # Check resource limits
                self._check_resource_limits(snapshot)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(interval_seconds * 2)  # Back off on error
    
    def _check_resource_limits(self, snapshot: ResourceMetrics):
        """Check if resource limits are exceeded"""
        warnings = []
        
        if snapshot.memory_mb >= self.limits.max_memory_mb:
            warnings.append(f"Memory usage ({snapshot.memory_mb:.1f}MB) exceeds limit ({self.limits.max_memory_mb}MB)")
        
        if snapshot.open_files >= self.limits.max_open_files:
            warnings.append(f"Open files ({snapshot.open_files}) exceeds limit ({self.limits.max_open_files})")
        
        if snapshot.active_threads >= self.limits.max_threads:
            warnings.append(f"Active threads ({snapshot.active_threads}) exceeds limit ({self.limits.max_threads})")
        
        for warning in warnings:
            logger.warning(f"Resource limit exceeded: {warning}")
    
    def checkpoint_operation(self, operation_name: str = "operation") -> ResourceMetrics:
        """Checkpoint an operation for resource tracking"""
        self.operation_count += 1
        snapshot = self.tracker.capture_snapshot()
        
        # Periodic cleanup based on operation count
        if self.operation_count % self.limits.gc_frequency == 0:
            self.optimizer.optimize_memory_usage(snapshot.memory_mb)
        
        logger.debug(f"Operation checkpoint '{operation_name}': {snapshot.memory_mb:.1f}MB")
        return snapshot
    
    @contextmanager
    def managed_execution(self, operation_name: str = "managed_operation"):
        """Context manager for resource-managed execution"""
        start_snapshot = self.checkpoint_operation(f"{operation_name}_start")
        
        try:
            yield self
        finally:
            end_snapshot = self.checkpoint_operation(f"{operation_name}_end")
            
            memory_delta = end_snapshot.memory_mb - start_snapshot.memory_mb
            if memory_delta > 100:  # More than 100MB increase
                logger.info(f"Operation '{operation_name}' used {memory_delta:.1f}MB "
                           f"(start: {start_snapshot.memory_mb:.1f}MB, end: {end_snapshot.memory_mb:.1f}MB)")
                
                # Force cleanup if significant memory increase
                self.optimizer.optimize_memory_usage(end_snapshot.memory_mb, force=True)
    
    def get_resource_report(self) -> Dict[str, Any]:
        """Generate comprehensive resource usage report"""
        current_snapshot = self.tracker.capture_snapshot()
        trend_analysis = self.tracker.analyze_memory_trend()
        
        return {
            'current': {
                'memory_mb': current_snapshot.memory_mb,
                'peak_memory_mb': self.tracker.peak_memory,
                'cpu_percent': current_snapshot.cpu_percent,
                'active_threads': current_snapshot.active_threads,
                'open_files': current_snapshot.open_files
            },
            'limits': {
                'max_memory_mb': self.limits.max_memory_mb,
                'cleanup_threshold_mb': self.limits.cleanup_threshold_mb,
                'critical_threshold_mb': self.limits.critical_threshold_mb
            },
            'trends': trend_analysis,
            'operations': {
                'total_count': self.operation_count,
                'cleanup_count': self.optimizer.cleanup_count
            },
            'status': {
                'monitoring_active': self.monitoring,
                'memory_status': 'critical' if current_snapshot.memory_mb >= self.limits.critical_threshold_mb
                                else 'warning' if current_snapshot.memory_mb >= self.limits.cleanup_threshold_mb  
                                else 'normal'
            }
        }
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get current resource usage metrics snapshot"""
        return self.tracker.capture_snapshot()
    
    def analyze_resource_usage(self) -> Dict[str, Any]:
        """Analyze resource usage trends and provide recommendations"""
        memory_trend = self.tracker.analyze_memory_trend(300)  # 5-minute window
        
        # Determine status based on current memory usage
        current_mb = memory_trend.get('current_mb', 0.0)
        if current_mb >= self.limits.critical_threshold_mb:
            status = 'critical'
        elif current_mb >= self.limits.cleanup_threshold_mb:
            status = 'warning'
        else:
            status = 'normal'
        
        # Generate recommendations based on trend analysis
        recommendations = []
        
        if memory_trend.get('trend', 0.0) > 0 and memory_trend.get('rate_mb_per_min', 0.0) > 10:
            recommendations.append("Memory usage is increasing rapidly - consider immediate cleanup")
        
        if memory_trend.get('stability', 1.0) < 0.5:
            recommendations.append("Memory usage is unstable - investigate potential memory leaks")
        
        if current_mb >= self.limits.cleanup_threshold_mb:
            recommendations.append("Memory usage above cleanup threshold - optimization recommended")
        
        if current_mb >= self.limits.critical_threshold_mb:
            recommendations.append("Critical memory usage - immediate optimization required")
        
        if not recommendations:
            recommendations.append("Memory usage is within normal parameters")
        
        return {
            'memory_trend': memory_trend,
            'status': status,
            'recommendations': recommendations
        }
    
    def force_optimization(self) -> Dict[str, Any]:
        """Force immediate memory optimization"""
        current_snapshot = self.tracker.capture_snapshot()
        return self.optimizer.optimize_memory_usage(current_snapshot.memory_mb, force=True)

# Global resource manager instance
_global_resource_manager: Optional[ResourceManager] = None

def get_resource_manager(limits: Optional[ResourceLimits] = None) -> ResourceManager:
    """Get or create global resource manager instance"""
    global _global_resource_manager
    
    if _global_resource_manager is None:
        _global_resource_manager = ResourceManager(limits)
    
    return _global_resource_manager

def cleanup_resources():
    """Cleanup global resources"""
    global _global_resource_manager
    
    if _global_resource_manager:
        _global_resource_manager.stop_monitoring()
        _global_resource_manager = None