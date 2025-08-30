"""
Temporal Storage Main Interface
==============================

Primary interface for storing and retrieving temporal system data.
Orchestrates hierarchical storage layers and provides unified access.
"""

import os
import asyncio
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, AsyncIterator, Union
from dataclasses import dataclass, field
from collections import defaultdict

from ..types import SystemDelta, SystemChange, SystemEvent, EventCorrelation
from ..config import TemporalStorageConfig
from .recent_buffer import RecentBuffer
from .daily_aggregator import DailyAggregator
from .pattern_store import PatternStore
from .search_index import TemporalSearchIndex
from .query_engine import TemporalQueryEngine, TemporalQuery
from .compression import DataCompressor


@dataclass
class StorageStats:
    """Statistics about temporal storage usage."""
    recent_buffer_size: int
    daily_summaries_count: int
    pattern_count: int
    total_events_stored: int
    memory_usage_mb: float
    disk_usage_mb: float
    compression_ratio: float
    oldest_data_timestamp: Optional[datetime]
    newest_data_timestamp: Optional[datetime]


@dataclass
class StorageHealth:
    """Health metrics for temporal storage."""
    is_healthy: bool
    memory_pressure: float  # 0.0 to 1.0
    disk_pressure: float    # 0.0 to 1.0
    error_rate: float       # Recent error rate
    performance_ms: float   # Average operation time
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class TemporalStorage:
    """
    Hierarchical temporal storage system with efficient querying.
    
    Provides system memory that preserves recent detail while 
    compressing historical data and extracting long-term patterns.
    """
    
    def __init__(self, config: TemporalStorageConfig):
        self.config = config
        self._initialize_storage_directory()
        
        # Storage layer components
        self.recent_buffer = RecentBuffer(config.recent_buffer_capacity)
        self.daily_aggregator = DailyAggregator(
            config.daily_retention_days,
            self._get_daily_storage_path()
        )
        self.pattern_store = PatternStore(
            config.pattern_retention_months,
            self._get_pattern_storage_path()
        )
        self.search_index = TemporalSearchIndex(config.index_categories)
        self.query_engine = TemporalQueryEngine(
            self.recent_buffer,
            self.daily_aggregator, 
            self.pattern_store,
            self.search_index
        )
        self.compressor = DataCompressor()
        
        # Threading and async support
        self._storage_lock = threading.RLock()
        self._background_tasks = []
        self._shutdown_event = threading.Event()
        
        # Performance tracking
        self._operation_times = defaultdict(list)
        self._error_count = 0
        self._total_operations = 0
        
        # Start background maintenance
        self._start_background_maintenance()
    
    def _initialize_storage_directory(self):
        """Initialize storage directory structure."""
        if self.config.storage_directory:
            self.storage_path = Path(self.config.storage_directory)
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.storage_path / 'daily').mkdir(exist_ok=True)
            (self.storage_path / 'patterns').mkdir(exist_ok=True)
            (self.storage_path / 'index').mkdir(exist_ok=True)
            (self.storage_path / 'backups').mkdir(exist_ok=True)
        else:
            self.storage_path = None
    
    def _get_daily_storage_path(self) -> Optional[Path]:
        """Get path for daily storage."""
        return self.storage_path / 'daily' if self.storage_path else None
    
    def _get_pattern_storage_path(self) -> Optional[Path]:
        """Get path for pattern storage."""
        return self.storage_path / 'patterns' if self.storage_path else None
    
    def store(self, system_delta: SystemDelta) -> None:
        """
        Store system delta across all storage layers.
        
        Args:
            system_delta: The system delta to store
        """
        start_time = datetime.now()
        
        try:
            with self._storage_lock:
                # Store in recent buffer (always in memory)
                self.recent_buffer.append(system_delta)
                
                # Update daily aggregation
                self.daily_aggregator.update(system_delta)
                
                # Learn patterns from this delta
                self.pattern_store.learn_from(system_delta)
                
                # Update search index
                self.search_index.update(system_delta)
                
                # Check for compression needs
                if self._should_compress():
                    self._schedule_compression()
            
            self._record_operation_time('store', start_time)
            self._total_operations += 1
            
        except Exception as e:
            self._error_count += 1
            self._log_error(f"Error storing system delta: {e}")
            raise
    
    def query(self, query: TemporalQuery) -> List[SystemDelta]:
        """
        Query temporal data across all storage layers.
        
        Args:
            query: Temporal query specification
            
        Returns:
            List of matching system deltas
        """
        start_time = datetime.now()
        
        try:
            with self._storage_lock:
                results = self.query_engine.execute_query(query)
            
            self._record_operation_time('query', start_time)
            return results
            
        except Exception as e:
            self._error_count += 1
            self._log_error(f"Error executing query: {e}")
            raise
    
    async def query_async(self, query: TemporalQuery) -> AsyncIterator[SystemDelta]:
        """
        Asynchronously query temporal data.
        
        Args:
            query: Temporal query specification
            
        Yields:
            Matching system deltas
        """
        # Run query in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def _execute_query():
            return self.query(query)
        
        results = await loop.run_in_executor(None, _execute_query)
        
        for result in results:
            yield result
    
    def get_recent_events(self, 
                         hours: int = 24,
                         categories: Optional[List[str]] = None,
                         min_significance: Optional[float] = None) -> List[SystemEvent]:
        """
        Get recent events with optional filtering.
        
        Args:
            hours: Hours of history to retrieve
            categories: Optional category filter
            min_significance: Optional significance threshold
            
        Returns:
            List of recent events
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        query = TemporalQuery(
            start_time=cutoff_time,
            end_time=datetime.now(),
            categories=categories,
            min_significance=min_significance,
            result_type='events'
        )
        
        deltas = self.query(query)
        events = []
        
        for delta in deltas:
            events.extend(delta.semantic_events)
        
        return events
    
    def get_system_timeline(self, 
                          start_time: datetime,
                          end_time: datetime,
                          granularity: str = 'hour') -> Dict[datetime, Dict[str, Any]]:
        """
        Get system timeline with specified granularity.
        
        Args:
            start_time: Timeline start
            end_time: Timeline end  
            granularity: Time granularity ('minute', 'hour', 'day')
            
        Returns:
            Timeline data grouped by time buckets
        """
        query = TemporalQuery(
            start_time=start_time,
            end_time=end_time,
            result_type='timeline',
            granularity=granularity
        )
        
        deltas = self.query(query)
        return self._build_timeline(deltas, granularity)
    
    def get_pattern_analysis(self, 
                           pattern_type: str,
                           time_range: timedelta = timedelta(days=30)) -> Dict[str, Any]:
        """
        Get pattern analysis for specified type and time range.
        
        Args:
            pattern_type: Type of pattern to analyze
            time_range: Time range for analysis
            
        Returns:
            Pattern analysis results
        """
        end_time = datetime.now()
        start_time = end_time - time_range
        
        return self.pattern_store.analyze_pattern(
            pattern_type=pattern_type,
            start_time=start_time,
            end_time=end_time
        )
    
    def get_storage_stats(self) -> StorageStats:
        """Get comprehensive storage statistics."""
        recent_deltas = list(self.recent_buffer.get_all())
        daily_summaries = self.daily_aggregator.get_summary_count()
        patterns = self.pattern_store.get_pattern_count()
        
        # Calculate memory usage
        memory_usage = self._calculate_memory_usage()
        
        # Calculate disk usage
        disk_usage = self._calculate_disk_usage()
        
        # Calculate total events
        total_events = sum(len(delta.semantic_events) for delta in recent_deltas)
        total_events += self.daily_aggregator.get_total_events()
        
        # Calculate compression ratio
        compression_ratio = self.compressor.get_average_compression_ratio()
        
        # Get time range
        oldest_time = None
        newest_time = None
        
        if recent_deltas:
            timestamps = [delta.timestamp for delta in recent_deltas]
            oldest_time = min(timestamps)
            newest_time = max(timestamps)
        
        # Include daily aggregator time range
        daily_time_range = self.daily_aggregator.get_time_range()
        if daily_time_range[0]:
            oldest_time = min(oldest_time or daily_time_range[0], daily_time_range[0])
        if daily_time_range[1]:
            newest_time = max(newest_time or daily_time_range[1], daily_time_range[1])
        
        return StorageStats(
            recent_buffer_size=len(recent_deltas),
            daily_summaries_count=daily_summaries,
            pattern_count=patterns,
            total_events_stored=total_events,
            memory_usage_mb=memory_usage,
            disk_usage_mb=disk_usage,
            compression_ratio=compression_ratio,
            oldest_data_timestamp=oldest_time,
            newest_data_timestamp=newest_time
        )
    
    def get_health_status(self) -> StorageHealth:
        """Get storage system health status."""
        stats = self.get_storage_stats()
        
        # Calculate memory pressure
        memory_pressure = min(stats.memory_usage_mb / self.config.max_memory_usage_mb, 1.0)
        
        # Calculate disk pressure (if disk storage enabled)
        disk_pressure = 0.0
        if self.config.enable_disk_persistence and self.storage_path:
            # Estimate disk pressure based on available space
            disk_free = self._get_available_disk_space()
            if disk_free > 0:
                disk_pressure = min(stats.disk_usage_mb / disk_free, 1.0)
        
        # Calculate error rate
        error_rate = self._error_count / max(self._total_operations, 1)
        
        # Calculate average performance
        all_times = []
        for times in self._operation_times.values():
            all_times.extend(times[-100:])  # Last 100 operations
        
        avg_performance = sum(all_times) / len(all_times) if all_times else 0.0
        
        # Determine overall health
        is_healthy = (
            memory_pressure < 0.8 and
            disk_pressure < 0.9 and
            error_rate < 0.05 and
            avg_performance < 1000  # 1 second
        )
        
        # Generate warnings
        warnings = []
        if memory_pressure > 0.7:
            warnings.append(f"High memory usage: {memory_pressure:.1%}")
        if disk_pressure > 0.8:
            warnings.append(f"High disk usage: {disk_pressure:.1%}")
        if error_rate > 0.02:
            warnings.append(f"Elevated error rate: {error_rate:.2%}")
        
        # Generate errors
        errors = []
        if memory_pressure >= 1.0:
            errors.append("Memory limit exceeded")
        if disk_pressure >= 1.0:
            errors.append("Disk space exhausted")
        if error_rate > 0.1:
            errors.append("High error rate indicates system issues")
        
        return StorageHealth(
            is_healthy=is_healthy,
            memory_pressure=memory_pressure,
            disk_pressure=disk_pressure,
            error_rate=error_rate,
            performance_ms=avg_performance,
            warnings=warnings,
            errors=errors
        )
    
    def compact_storage(self) -> Dict[str, Any]:
        """
        Perform storage compaction and optimization.
        
        Returns:
            Compaction results and statistics
        """
        with self._storage_lock:
            results = {
                'start_time': datetime.now(),
                'operations': []
            }
            
            try:
                # Compact recent buffer
                buffer_results = self.recent_buffer.compact()
                results['operations'].append({
                    'operation': 'buffer_compaction',
                    'results': buffer_results
                })
                
                # Aggregate old data to daily summaries
                aggregation_results = self.daily_aggregator.aggregate_old_data()
                results['operations'].append({
                    'operation': 'daily_aggregation', 
                    'results': aggregation_results
                })
                
                # Update pattern extraction
                pattern_results = self.pattern_store.extract_new_patterns()
                results['operations'].append({
                    'operation': 'pattern_extraction',
                    'results': pattern_results
                })
                
                # Rebuild search index
                index_results = self.search_index.rebuild()
                results['operations'].append({
                    'operation': 'index_rebuild',
                    'results': index_results
                })
                
                results['end_time'] = datetime.now()
                results['duration'] = (results['end_time'] - results['start_time']).total_seconds()
                results['success'] = True
                
                return results
                
            except Exception as e:
                results['error'] = str(e)
                results['success'] = False
                self._log_error(f"Storage compaction failed: {e}")
                raise
    
    def backup_storage(self, backup_path: Optional[Path] = None) -> Path:
        """
        Create backup of temporal storage.
        
        Args:
            backup_path: Optional custom backup path
            
        Returns:
            Path to created backup
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.storage_path / 'backups' / f'temporal_backup_{timestamp}'
        
        backup_path.mkdir(parents=True, exist_ok=True)
        
        with self._storage_lock:
            # Backup recent buffer
            self.recent_buffer.save_to_file(backup_path / 'recent_buffer.json')
            
            # Backup daily aggregations
            self.daily_aggregator.backup_to(backup_path / 'daily_summaries')
            
            # Backup patterns
            self.pattern_store.backup_to(backup_path / 'patterns')
            
            # Backup search index
            self.search_index.backup_to(backup_path / 'search_index')
        
        return backup_path
    
    def restore_from_backup(self, backup_path: Path) -> None:
        """
        Restore temporal storage from backup.
        
        Args:
            backup_path: Path to backup directory
        """
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup path not found: {backup_path}")
        
        with self._storage_lock:
            # Clear current data
            self.recent_buffer.clear()
            self.daily_aggregator.clear()
            self.pattern_store.clear()
            self.search_index.clear()
            
            # Restore from backup
            if (backup_path / 'recent_buffer.json').exists():
                self.recent_buffer.load_from_file(backup_path / 'recent_buffer.json')
            
            if (backup_path / 'daily_summaries').exists():
                self.daily_aggregator.restore_from(backup_path / 'daily_summaries')
            
            if (backup_path / 'patterns').exists():
                self.pattern_store.restore_from(backup_path / 'patterns')
            
            if (backup_path / 'search_index').exists():
                self.search_index.restore_from(backup_path / 'search_index')
    
    def shutdown(self) -> None:
        """Gracefully shutdown temporal storage."""
        self._shutdown_event.set()
        
        # Wait for background tasks to complete
        for task in self._background_tasks:
            task.join(timeout=5.0)
        
        # Final persistence if enabled
        if self.config.enable_disk_persistence:
            try:
                self._persist_all_data()
            except Exception as e:
                self._log_error(f"Error during final persistence: {e}")
    
    def _should_compress(self) -> bool:
        """Check if compression is needed."""
        stats = self.get_storage_stats()
        return stats.memory_usage_mb > self.config.compression_threshold_mb
    
    def _schedule_compression(self) -> None:
        """Schedule background compression task."""
        def compress_task():
            try:
                self.compact_storage()
            except Exception as e:
                self._log_error(f"Background compression failed: {e}")
        
        task = threading.Thread(target=compress_task, daemon=True)
        task.start()
        self._background_tasks.append(task)
    
    def _build_timeline(self, deltas: List[SystemDelta], granularity: str) -> Dict[datetime, Dict[str, Any]]:
        """Build timeline from deltas."""
        timeline = defaultdict(lambda: {
            'event_count': 0,
            'change_count': 0,
            'categories': set(),
            'significant_events': []
        })
        
        for delta in deltas:
            # Determine time bucket based on granularity
            if granularity == 'minute':
                bucket = delta.timestamp.replace(second=0, microsecond=0)
            elif granularity == 'hour':
                bucket = delta.timestamp.replace(minute=0, second=0, microsecond=0)
            elif granularity == 'day':
                bucket = delta.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                bucket = delta.timestamp
            
            # Update timeline data
            timeline[bucket]['event_count'] += len(delta.semantic_events)
            timeline[bucket]['change_count'] += len(delta.raw_delta)
            
            # Collect categories
            for change in delta.raw_delta:
                timeline[bucket]['categories'].add(change.category)
            
            # Collect significant events
            significant_events = [e for e in delta.semantic_events if e.confidence > 0.8]
            timeline[bucket]['significant_events'].extend(significant_events)
        
        # Convert sets to lists for JSON serialization
        for bucket_data in timeline.values():
            bucket_data['categories'] = list(bucket_data['categories'])
        
        return dict(timeline)
    
    def _calculate_memory_usage(self) -> float:
        """Calculate current memory usage in MB."""
        import sys
        
        # Estimate memory usage of storage components
        recent_size = sys.getsizeof(self.recent_buffer) / 1024 / 1024
        daily_size = sys.getsizeof(self.daily_aggregator) / 1024 / 1024
        pattern_size = sys.getsizeof(self.pattern_store) / 1024 / 1024
        index_size = sys.getsizeof(self.search_index) / 1024 / 1024
        
        return recent_size + daily_size + pattern_size + index_size
    
    def _calculate_disk_usage(self) -> float:
        """Calculate current disk usage in MB."""
        if not self.storage_path or not self.storage_path.exists():
            return 0.0
        
        total_size = 0
        for path in self.storage_path.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
        
        return total_size / 1024 / 1024
    
    def _get_available_disk_space(self) -> float:
        """Get available disk space in MB."""
        if not self.storage_path:
            return 0.0
        
        try:
            statvfs = os.statvfs(self.storage_path)
            available_bytes = statvfs.f_bavail * statvfs.f_frsize
            return available_bytes / 1024 / 1024
        except (OSError, AttributeError):
            return 0.0
    
    def _record_operation_time(self, operation: str, start_time: datetime) -> None:
        """Record operation timing."""
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000
        self._operation_times[operation].append(duration_ms)
        
        # Keep only recent measurements
        if len(self._operation_times[operation]) > 1000:
            self._operation_times[operation] = self._operation_times[operation][-1000:]
    
    def _persist_all_data(self) -> None:
        """Persist all data to disk."""
        if not self.config.enable_disk_persistence or not self.storage_path:
            return
        
        # Persist each component
        self.daily_aggregator.persist()
        self.pattern_store.persist()
        self.search_index.persist()
    
    def _start_background_maintenance(self) -> None:
        """Start background maintenance tasks."""
        def maintenance_loop():
            while not self._shutdown_event.is_set():
                try:
                    # Check if maintenance is needed
                    health = self.get_health_status()
                    
                    if health.memory_pressure > 0.8:
                        self._schedule_compression()
                    
                    # Periodic persistence
                    if self.config.enable_disk_persistence:
                        self._persist_all_data()
                    
                    # Sleep until next maintenance check
                    self._shutdown_event.wait(timeout=300)  # 5 minutes
                    
                except Exception as e:
                    self._log_error(f"Background maintenance error: {e}")
                    self._shutdown_event.wait(timeout=60)  # Retry in 1 minute
        
        maintenance_task = threading.Thread(target=maintenance_loop, daemon=True)
        maintenance_task.start()
        self._background_tasks.append(maintenance_task)
    
    def _log_error(self, message: str) -> None:
        """Log error message."""
        # Simple logging - in production would use proper logger
        print(f"[TemporalStorage Error] {datetime.now()}: {message}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()