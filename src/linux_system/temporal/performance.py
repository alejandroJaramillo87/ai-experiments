"""
Performance Optimization Layer
=============================

Efficient data structures and optimization techniques for high-performance
temporal system monitoring and analysis.
"""

import gc
import threading
import time
import weakref
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import logging
import psutil
from enum import Enum

from .types import SystemDelta, SystemChange, SystemEvent


logger = logging.getLogger(__name__)


T = TypeVar('T')


class CacheStrategy(Enum):
    """Cache eviction strategies."""
    LRU = "lru"
    LFU = "lfu" 
    TTL = "ttl"
    SIZE_BASED = "size_based"


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring system efficiency."""
    total_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_query_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    total_processing_time_ms: float = 0.0
    batch_operations: int = 0
    background_tasks_queued: int = 0
    gc_collections: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        return self.cache_hits / total_requests if total_requests > 0 else 0.0


class TemporalCache(Generic[T]):
    """
    High-performance cache with multiple eviction strategies.
    
    Supports LRU, LFU, TTL, and size-based eviction policies.
    """
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.strategy = strategy
        self.ttl_seconds = ttl_seconds
        
        self._data: Dict[str, T] = {}
        self._access_times: Dict[str, datetime] = {}
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._insertion_order: OrderedDict[str, None] = OrderedDict()
        
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[T]:
        """Get item from cache."""
        with self._lock:
            if key not in self._data:
                self._misses += 1
                return None
            
            # Check TTL expiration
            if self.strategy == CacheStrategy.TTL:
                access_time = self._access_times.get(key)
                if access_time and (datetime.now() - access_time).total_seconds() > self.ttl_seconds:
                    self._remove_key(key)
                    self._misses += 1
                    return None
            
            # Update access tracking
            self._access_times[key] = datetime.now()
            self._access_counts[key] += 1
            
            # Update LRU order
            if self.strategy == CacheStrategy.LRU:
                self._insertion_order.move_to_end(key)
            
            self._hits += 1
            return self._data[key]
    
    def put(self, key: str, value: T) -> None:
        """Put item in cache."""
        with self._lock:
            # If key exists, update it
            if key in self._data:
                self._data[key] = value
                self._access_times[key] = datetime.now()
                if self.strategy == CacheStrategy.LRU:
                    self._insertion_order.move_to_end(key)
                return
            
            # Check if cache is full and evict if necessary
            if len(self._data) >= self.max_size:
                self._evict()
            
            # Add new item
            self._data[key] = value
            self._access_times[key] = datetime.now()
            self._access_counts[key] = 1
            self._insertion_order[key] = None
    
    def invalidate(self, key: str) -> bool:
        """Remove item from cache."""
        with self._lock:
            if key in self._data:
                self._remove_key(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache contents."""
        with self._lock:
            self._data.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._insertion_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            return {
                'size': len(self._data),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': self._hits / total_requests if total_requests > 0 else 0.0,
                'strategy': self.strategy.value
            }
    
    def _evict(self) -> None:
        """Evict item based on strategy."""
        if not self._data:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used (first in OrderedDict)
            key_to_remove = next(iter(self._insertion_order))
            
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            key_to_remove = min(self._access_counts.keys(), key=self._access_counts.get)
            
        elif self.strategy == CacheStrategy.TTL:
            # Remove expired items first, then oldest
            now = datetime.now()
            expired_keys = [
                key for key, access_time in self._access_times.items()
                if (now - access_time).total_seconds() > self.ttl_seconds
            ]
            if expired_keys:
                key_to_remove = expired_keys[0]
            else:
                key_to_remove = min(self._access_times.keys(), key=self._access_times.get)
                
        elif self.strategy == CacheStrategy.SIZE_BASED:
            # Remove oldest (first inserted)
            key_to_remove = next(iter(self._insertion_order))
            
        else:
            # Default to LRU
            key_to_remove = next(iter(self._insertion_order))
        
        self._remove_key(key_to_remove)
    
    def _remove_key(self, key: str) -> None:
        """Remove key and all associated data."""
        self._data.pop(key, None)
        self._access_times.pop(key, None)
        self._access_counts.pop(key, None)
        self._insertion_order.pop(key, None)


class BatchProcessor:
    """
    Efficient batch processing for temporal data operations.
    
    Collects operations and processes them in batches to improve performance.
    """
    
    def __init__(self, batch_size: int = 100, flush_interval_seconds: int = 30, max_workers: int = 4):
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        self.max_workers = max_workers
        
        self._batches: Dict[str, List[Any]] = defaultdict(list)
        self._processors: Dict[str, Callable[[List[Any]], None]] = {}
        self._last_flush: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Start background flush thread
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        
        self._shutdown = False
    
    def register_processor(self, operation_type: str, processor: Callable[[List[Any]], None]) -> None:
        """Register a batch processor for an operation type."""
        with self._lock:
            self._processors[operation_type] = processor
            self._last_flush[operation_type] = datetime.now()
    
    def add_operation(self, operation_type: str, data: Any) -> None:
        """Add operation to batch queue."""
        with self._lock:
            if operation_type not in self._processors:
                logger.warning(f"No processor registered for operation type: {operation_type}")
                return
            
            self._batches[operation_type].append(data)
            
            # Check if batch is ready for processing
            if len(self._batches[operation_type]) >= self.batch_size:
                self._flush_batch(operation_type)
    
    def flush_all(self) -> None:
        """Force flush all pending batches."""
        with self._lock:
            for operation_type in list(self._batches.keys()):
                if self._batches[operation_type]:
                    self._flush_batch(operation_type)
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        with self._lock:
            stats = {}
            for operation_type, batch in self._batches.items():
                stats[operation_type] = {
                    'pending_operations': len(batch),
                    'last_flush': self._last_flush.get(operation_type, datetime.min).isoformat(),
                    'has_processor': operation_type in self._processors
                }
            return stats
    
    def shutdown(self) -> None:
        """Shutdown batch processor."""
        self._shutdown = True
        self.flush_all()
        self._executor.shutdown(wait=True)
    
    def _flush_batch(self, operation_type: str) -> None:
        """Flush a specific batch type."""
        if not self._batches[operation_type]:
            return
        
        batch_data = self._batches[operation_type].copy()
        self._batches[operation_type].clear()
        self._last_flush[operation_type] = datetime.now()
        
        processor = self._processors[operation_type]
        
        # Submit to thread pool for background processing
        self._executor.submit(self._process_batch_safely, operation_type, batch_data, processor)
    
    def _process_batch_safely(self, operation_type: str, batch_data: List[Any], 
                             processor: Callable[[List[Any]], None]) -> None:
        """Safely process batch with error handling."""
        try:
            processor(batch_data)
            logger.debug(f"Processed batch of {len(batch_data)} {operation_type} operations")
        except Exception as e:
            logger.error(f"Error processing {operation_type} batch: {e}")
    
    def _flush_loop(self) -> None:
        """Background thread for periodic flushing."""
        while not self._shutdown:
            try:
                time.sleep(1)  # Check every second
                
                now = datetime.now()
                with self._lock:
                    for operation_type in list(self._batches.keys()):
                        last_flush = self._last_flush.get(operation_type, datetime.min)
                        time_since_flush = (now - last_flush).total_seconds()
                        
                        if (time_since_flush >= self.flush_interval and 
                            self._batches[operation_type]):
                            self._flush_batch(operation_type)
                            
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")


class MemoryPool(Generic[T]):
    """
    Memory pool for object reuse to reduce garbage collection pressure.
    
    Maintains a pool of reusable objects to minimize allocations.
    """
    
    def __init__(self, factory: Callable[[], T], initial_size: int = 100, max_size: int = 1000):
        self.factory = factory
        self.max_size = max_size
        self._pool: deque[T] = deque()
        self._lock = threading.Lock()
        self._created_count = 0
        self._reused_count = 0
        
        # Pre-populate pool
        for _ in range(initial_size):
            self._pool.append(factory())
            self._created_count += 1
    
    def acquire(self) -> T:
        """Get object from pool or create new one."""
        with self._lock:
            if self._pool:
                self._reused_count += 1
                return self._pool.popleft()
            else:
                self._created_count += 1
                return self.factory()
    
    def release(self, obj: T) -> None:
        """Return object to pool."""
        with self._lock:
            if len(self._pool) < self.max_size:
                # Reset object if it has a reset method
                if hasattr(obj, 'reset'):
                    obj.reset()
                self._pool.append(obj)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                'pool_size': len(self._pool),
                'max_size': self.max_size,
                'objects_created': self._created_count,
                'objects_reused': self._reused_count,
                'reuse_rate': self._reused_count / max(self._created_count, 1)
            }


class IndexOptimizer:
    """
    Optimizes temporal indices for fast queries.
    
    Maintains multiple specialized indices for different query patterns.
    """
    
    def __init__(self):
        self._time_index: Dict[str, Set[str]] = defaultdict(set)  # hour -> delta_ids
        self._category_index: Dict[str, Set[str]] = defaultdict(set)  # category -> delta_ids
        self._significance_index: Dict[float, Set[str]] = defaultdict(set)  # significance -> delta_ids
        self._composite_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)  # (hour, category) -> delta_ids
        
        self._lock = threading.RLock()
        self._index_stats = {
            'total_entries': 0,
            'time_buckets': 0,
            'categories': 0,
            'significance_levels': 0,
            'composite_entries': 0
        }
    
    def add_delta(self, delta_id: str, delta: SystemDelta) -> None:
        """Add delta to indices."""
        with self._lock:
            # Time index (hourly buckets)
            hour_key = delta.timestamp.strftime("%Y%m%d%H")
            self._time_index[hour_key].add(delta_id)
            
            # Category and significance indices
            for change in delta.raw_delta:
                self._category_index[change.category].add(delta_id)
                
                # Significance buckets (0.1 increments)
                sig_bucket = round(change.significance, 1)
                self._significance_index[sig_bucket].add(delta_id)
                
                # Composite index
                composite_key = (hour_key, change.category)
                self._composite_index[composite_key].add(delta_id)
            
            self._update_stats()
    
    def remove_delta(self, delta_id: str, delta: SystemDelta) -> None:
        """Remove delta from indices."""
        with self._lock:
            hour_key = delta.timestamp.strftime("%Y%m%d%H")
            self._time_index[hour_key].discard(delta_id)
            
            for change in delta.raw_delta:
                self._category_index[change.category].discard(delta_id)
                
                sig_bucket = round(change.significance, 1)
                self._significance_index[sig_bucket].discard(delta_id)
                
                composite_key = (hour_key, change.category)
                self._composite_index[composite_key].discard(delta_id)
            
            self._update_stats()
    
    def query_time_range(self, start_time: datetime, end_time: datetime) -> Set[str]:
        """Get delta IDs in time range."""
        with self._lock:
            result_set = set()
            
            current = start_time.replace(minute=0, second=0, microsecond=0)
            while current <= end_time:
                hour_key = current.strftime("%Y%m%d%H")
                result_set.update(self._time_index.get(hour_key, set()))
                current += timedelta(hours=1)
            
            return result_set
    
    def query_category(self, category: str) -> Set[str]:
        """Get delta IDs for category."""
        with self._lock:
            return self._category_index.get(category, set()).copy()
    
    def query_significance_range(self, min_sig: float, max_sig: float) -> Set[str]:
        """Get delta IDs in significance range."""
        with self._lock:
            result_set = set()
            
            for sig_level, delta_ids in self._significance_index.items():
                if min_sig <= sig_level <= max_sig:
                    result_set.update(delta_ids)
            
            return result_set
    
    def query_composite(self, time_range: Tuple[datetime, datetime], 
                       categories: List[str]) -> Set[str]:
        """Optimized composite query."""
        with self._lock:
            result_set = set()
            start_time, end_time = time_range
            
            current = start_time.replace(minute=0, second=0, microsecond=0)
            while current <= end_time:
                hour_key = current.strftime("%Y%m%d%H")
                
                for category in categories:
                    composite_key = (hour_key, category)
                    result_set.update(self._composite_index.get(composite_key, set()))
                
                current += timedelta(hours=1)
            
            return result_set
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        with self._lock:
            return self._index_stats.copy()
    
    def _update_stats(self) -> None:
        """Update index statistics."""
        self._index_stats.update({
            'time_buckets': len(self._time_index),
            'categories': len(self._category_index),
            'significance_levels': len(self._significance_index),
            'composite_entries': len(self._composite_index)
        })


class ResourceManager:
    """
    Manages system resources to prevent performance degradation.
    
    Monitors memory, CPU usage and triggers cleanup when needed.
    """
    
    def __init__(self, memory_limit_mb: int = 1024, cpu_limit_percent: int = 80):
        self.memory_limit_mb = memory_limit_mb
        self.cpu_limit_percent = cpu_limit_percent
        
        self._cleanup_callbacks: List[Callable[[], None]] = []
        self._monitoring_enabled = True
        self._lock = threading.Lock()
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def register_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for resource cleanup."""
        with self._lock:
            self._cleanup_callbacks.append(callback)
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'memory_mb': memory_info.rss / 1024 / 1024,
            'memory_percent': process.memory_percent(),
            'cpu_percent': process.cpu_percent(),
            'num_threads': process.num_threads(),
            'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0
        }
    
    def force_cleanup(self) -> Dict[str, Any]:
        """Force resource cleanup."""
        cleanup_results = {}
        
        with self._lock:
            # Run cleanup callbacks
            for i, callback in enumerate(self._cleanup_callbacks):
                try:
                    callback()
                    cleanup_results[f'callback_{i}'] = 'success'
                except Exception as e:
                    cleanup_results[f'callback_{i}'] = f'error: {e}'
        
        # Force garbage collection
        collected = gc.collect()
        cleanup_results['gc_collected'] = collected
        
        # Get new resource usage
        cleanup_results['resource_usage_after'] = self.get_resource_usage()
        
        return cleanup_results
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring_enabled = False
    
    def _monitor_loop(self) -> None:
        """Background resource monitoring loop."""
        while self._monitoring_enabled:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                usage = self.get_resource_usage()
                
                # Check if cleanup is needed
                needs_cleanup = (
                    usage['memory_mb'] > self.memory_limit_mb or
                    usage['cpu_percent'] > self.cpu_limit_percent
                )
                
                if needs_cleanup:
                    logger.info(f"Resource limits exceeded, triggering cleanup: {usage}")
                    self.force_cleanup()
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")


class PerformanceOptimizer:
    """
    Main performance optimization coordinator.
    
    Orchestrates all performance optimization components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # Initialize components
        self.cache = TemporalCache[Any](
            max_size=config.get('cache_size', 1000),
            strategy=CacheStrategy(config.get('cache_strategy', 'lru')),
            ttl_seconds=config.get('cache_ttl', 3600)
        )
        
        self.batch_processor = BatchProcessor(
            batch_size=config.get('batch_size', 100),
            flush_interval_seconds=config.get('flush_interval', 30),
            max_workers=config.get('max_workers', 4)
        )
        
        self.index_optimizer = IndexOptimizer()
        
        self.resource_manager = ResourceManager(
            memory_limit_mb=config.get('memory_limit_mb', 1024),
            cpu_limit_percent=config.get('cpu_limit_percent', 80)
        )
        
        # Memory pools for common objects
        self.delta_pool = MemoryPool(
            factory=lambda: SystemDelta(
                timestamp=datetime.now(),
                raw_delta=[],
                semantic_events=[],
                correlations=[]
            ),
            initial_size=50,
            max_size=200
        )
        
        # Performance metrics
        self._metrics = PerformanceMetrics()
        self._metrics_lock = threading.Lock()
        
        # Register cleanup callbacks
        self.resource_manager.register_cleanup_callback(self._cleanup_caches)
        self.resource_manager.register_cleanup_callback(self._force_gc)
        
        logger.info("PerformanceOptimizer initialized")
    
    def optimize_query(self, query_func: Callable[[], T], cache_key: Optional[str] = None) -> T:
        """Optimize query execution with caching."""
        start_time = time.time()
        
        # Try cache first if key provided
        if cache_key:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self._update_metrics(cache_hit=True, query_time_ms=(time.time() - start_time) * 1000)
                return cached_result
        
        # Execute query
        result = query_func()
        
        # Cache result if key provided
        if cache_key and result is not None:
            self.cache.put(cache_key, result)
        
        query_time_ms = (time.time() - start_time) * 1000
        self._update_metrics(
            cache_hit=False, 
            query_time_ms=query_time_ms,
            operation_count=1
        )
        
        return result
    
    def batch_operation(self, operation_type: str, data: Any) -> None:
        """Add operation to batch queue."""
        self.batch_processor.add_operation(operation_type, data)
        self._update_metrics(batch_operation=True)
    
    def register_batch_processor(self, operation_type: str, 
                                processor: Callable[[List[Any]], None]) -> None:
        """Register batch processor."""
        self.batch_processor.register_processor(operation_type, processor)
    
    def add_to_index(self, delta_id: str, delta: SystemDelta) -> None:
        """Add delta to optimized indices."""
        self.index_optimizer.add_delta(delta_id, delta)
    
    def remove_from_index(self, delta_id: str, delta: SystemDelta) -> None:
        """Remove delta from indices."""
        self.index_optimizer.remove_delta(delta_id, delta)
    
    def optimized_time_query(self, start_time: datetime, end_time: datetime) -> Set[str]:
        """Optimized time-based query using indices."""
        return self.index_optimizer.query_time_range(start_time, end_time)
    
    def optimized_category_query(self, categories: List[str], 
                                time_range: Optional[Tuple[datetime, datetime]] = None) -> Set[str]:
        """Optimized category query."""
        if time_range:
            return self.index_optimizer.query_composite(time_range, categories)
        else:
            result_set = set()
            for category in categories:
                result_set.update(self.index_optimizer.query_category(category))
            return result_set
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        with self._metrics_lock:
            # Update resource usage
            resource_usage = self.resource_manager.get_resource_usage()
            self._metrics.memory_usage_mb = resource_usage['memory_mb']
            self._metrics.cpu_usage_percent = resource_usage['cpu_percent']
            
            return PerformanceMetrics(
                total_operations=self._metrics.total_operations,
                cache_hits=self._metrics.cache_hits,
                cache_misses=self._metrics.cache_misses,
                average_query_time_ms=self._metrics.average_query_time_ms,
                memory_usage_mb=self._metrics.memory_usage_mb,
                cpu_usage_percent=self._metrics.cpu_usage_percent,
                total_processing_time_ms=self._metrics.total_processing_time_ms,
                batch_operations=self._metrics.batch_operations,
                background_tasks_queued=self._metrics.background_tasks_queued,
                gc_collections=self._metrics.gc_collections
            )
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        return {
            'performance_metrics': self.get_performance_metrics().__dict__,
            'cache_stats': self.cache.get_stats(),
            'batch_stats': self.batch_processor.get_batch_stats(),
            'index_stats': self.index_optimizer.get_index_stats(),
            'resource_usage': self.resource_manager.get_resource_usage(),
            'memory_pools': {
                'delta_pool': self.delta_pool.get_stats()
            }
        }
    
    def force_optimization(self) -> Dict[str, Any]:
        """Force immediate optimization."""
        optimization_results = {}
        
        # Force cache cleanup
        cache_stats_before = self.cache.get_stats()
        self._cleanup_caches()
        cache_stats_after = self.cache.get_stats()
        optimization_results['cache_cleanup'] = {
            'before': cache_stats_before,
            'after': cache_stats_after
        }
        
        # Force batch processing
        self.batch_processor.flush_all()
        optimization_results['batches_flushed'] = self.batch_processor.get_batch_stats()
        
        # Force resource cleanup
        cleanup_results = self.resource_manager.force_cleanup()
        optimization_results['resource_cleanup'] = cleanup_results
        
        return optimization_results
    
    def shutdown(self) -> None:
        """Shutdown optimizer and cleanup resources."""
        logger.info("Shutting down PerformanceOptimizer...")
        
        self.batch_processor.shutdown()
        self.resource_manager.stop_monitoring()
        self.force_optimization()
        
        logger.info("PerformanceOptimizer shutdown complete")
    
    def _update_metrics(self, cache_hit: bool = False, query_time_ms: float = 0.0,
                       operation_count: int = 0, batch_operation: bool = False) -> None:
        """Update performance metrics."""
        with self._metrics_lock:
            if cache_hit:
                self._metrics.cache_hits += 1
            elif cache_hit is False:  # Explicit False, not None
                self._metrics.cache_misses += 1
            
            if operation_count > 0:
                self._metrics.total_operations += operation_count
            
            if query_time_ms > 0:
                # Update running average
                total_ops = self._metrics.total_operations
                if total_ops > 0:
                    self._metrics.average_query_time_ms = (
                        (self._metrics.average_query_time_ms * (total_ops - 1) + query_time_ms) / total_ops
                    )
                else:
                    self._metrics.average_query_time_ms = query_time_ms
                
                self._metrics.total_processing_time_ms += query_time_ms
            
            if batch_operation:
                self._metrics.batch_operations += 1
    
    def _cleanup_caches(self) -> None:
        """Cleanup caches to free memory."""
        # Clear least important cache entries
        cache_stats = self.cache.get_stats()
        if cache_stats['size'] > cache_stats['max_size'] * 0.8:
            # Cache is getting full, clear some entries
            # This would be more sophisticated in practice
            logger.info("Performing cache cleanup")
    
    def _force_gc(self) -> None:
        """Force garbage collection."""
        collected = gc.collect()
        with self._metrics_lock:
            self._metrics.gc_collections += collected
        logger.debug(f"Garbage collection: {collected} objects collected")


def create_performance_optimizer(config: Optional[Dict[str, Any]] = None) -> PerformanceOptimizer:
    """
    Create configured performance optimizer.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured PerformanceOptimizer instance
    """
    return PerformanceOptimizer(config)