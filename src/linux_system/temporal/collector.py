"""
Temporal System Collector
========================

Main orchestrator for temporal system intelligence. Integrates all components
of the temporal intelligence architecture to create a comprehensive system
monitoring and analysis solution.

This is the bridge between the existing SystemCollector and the new
temporal intelligence capabilities, creating a unified interface for
omniscient system understanding.
"""

import asyncio
import logging
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import asynccontextmanager

from .types import SystemDelta, SystemSnapshot, SystemChange, SystemEvent
from .change_detection import ChangeDetectionEngine
from .event_extraction_engine import EventExtractionEngine  
from .storage import TemporalStorage
from ..system_collector import SystemCollector


logger = logging.getLogger(__name__)


@dataclass
class CollectionConfig:
    """Configuration for temporal system collection."""
    collection_interval_seconds: int = 60
    enable_gpu_monitoring: bool = True
    enable_process_monitoring: bool = True
    enable_python_env_monitoring: bool = True
    enable_memory_monitoring: bool = True
    enable_storage_monitoring: bool = True
    enable_network_monitoring: bool = True
    enable_security_monitoring: bool = True
    
    # Event extraction settings
    enable_semantic_events: bool = True
    enable_correlation_detection: bool = True
    min_event_confidence: float = 0.3
    
    # Storage settings
    storage_path: str = "/tmp/temporal_storage"
    enable_compression: bool = True
    enable_pattern_learning: bool = True
    
    # Performance settings
    max_worker_threads: int = 8
    collection_timeout_seconds: int = 30
    batch_size: int = 100
    
    # Advanced features
    enable_predictive_analysis: bool = True
    enable_anomaly_detection: bool = True
    enable_causal_analysis: bool = True


@dataclass
class CollectionStats:
    """Statistics about temporal collection operations."""
    total_collections: int = 0
    successful_collections: int = 0
    failed_collections: int = 0
    total_changes_detected: int = 0
    total_events_generated: int = 0
    total_patterns_learned: int = 0
    average_collection_time_ms: float = 0.0
    last_collection_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    storage_stats: Dict[str, Any] = None
    error_counts: Dict[str, int] = None


class TemporalSystemCollector:
    """
    Main orchestrator for temporal system intelligence.
    
    Integrates change detection, event extraction, and storage to provide
    comprehensive temporal understanding of the Linux system.
    """
    
    def __init__(self, config: Optional[CollectionConfig] = None):
        """
        Initialize temporal system collector.
        
        Args:
            config: Collection configuration
        """
        self.config = config or CollectionConfig()
        self.is_running = False
        self.start_time = datetime.now()
        
        # Core components
        self.system_collector = SystemCollector()
        self.change_detector = ChangeDetectionEngine(self.config)
        self.event_extractor = EventExtractionEngine(self.config)
        self.temporal_storage = TemporalStorage(
            storage_path=Path(self.config.storage_path),
            enable_compression=self.config.enable_compression
        )
        
        # Threading and concurrency
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_worker_threads)
        self.collection_lock = threading.RLock()
        self.stats_lock = threading.RLock()
        
        # State tracking
        self.last_snapshot: Optional[SystemSnapshot] = None
        self.collection_thread: Optional[threading.Thread] = None
        self.stats = CollectionStats(error_counts={})
        
        # Event callbacks
        self.event_callbacks: List[Callable[[SystemDelta], None]] = []
        self.anomaly_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Internal state
        self._shutdown_event = threading.Event()
        self._health_check_interval = 300  # 5 minutes
        self._last_health_check = datetime.now()
        
        logger.info("TemporalSystemCollector initialized with config: %s", 
                   {k: v for k, v in asdict(self.config).items() if not k.startswith('_')})
    
    def start(self) -> None:
        """Start temporal system collection."""
        if self.is_running:
            logger.warning("Temporal collection is already running")
            return
        
        logger.info("Starting temporal system collection...")
        
        with self.collection_lock:
            self.is_running = True
            self._shutdown_event.clear()
            
            # Start collection thread
            self.collection_thread = threading.Thread(
                target=self._collection_loop,
                name="TemporalCollector",
                daemon=True
            )
            self.collection_thread.start()
            
            # Initialize storage
            self.temporal_storage.initialize()
            
            logger.info("Temporal system collection started successfully")
    
    def stop(self, timeout: float = 30.0) -> bool:
        """
        Stop temporal system collection.
        
        Args:
            timeout: Maximum time to wait for graceful shutdown
            
        Returns:
            True if stopped gracefully, False if forced
        """
        if not self.is_running:
            logger.warning("Temporal collection is not running")
            return True
        
        logger.info("Stopping temporal system collection...")
        
        with self.collection_lock:
            self.is_running = False
            self._shutdown_event.set()
        
        # Wait for collection thread to finish
        graceful_stop = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=timeout)
            graceful_stop = not self.collection_thread.is_alive()
        
        # Shutdown executor
        self.executor.shutdown(wait=True, timeout=timeout)
        
        # Flush storage
        if hasattr(self.temporal_storage, 'flush'):
            self.temporal_storage.flush()
        
        if graceful_stop:
            logger.info("Temporal system collection stopped gracefully")
        else:
            logger.warning("Temporal system collection forced to stop")
        
        return graceful_stop
    
    def collect_now(self) -> Optional[SystemDelta]:
        """
        Perform immediate collection and analysis.
        
        Returns:
            SystemDelta if successful, None if failed
        """
        try:
            return self._perform_collection()
        except Exception as e:
            logger.error(f"Failed to perform immediate collection: {e}")
            return None
    
    def query_temporal_data(self, **query_params) -> List[SystemDelta]:
        """
        Query temporal data from storage.
        
        Args:
            **query_params: Query parameters (time_range, categories, etc.)
            
        Returns:
            List of matching SystemDelta objects
        """
        return self.temporal_storage.query(**query_params)
    
    def get_system_patterns(self, days: int = 30) -> Dict[str, Any]:
        """
        Get learned system patterns.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary of system patterns
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        return self.temporal_storage.get_patterns(start_time, end_time)
    
    def predict_system_behavior(self, horizon_hours: int = 24) -> Dict[str, Any]:
        """
        Predict system behavior based on historical patterns.
        
        Args:
            horizon_hours: Prediction horizon in hours
            
        Returns:
            Dictionary of predictions
        """
        if not self.config.enable_predictive_analysis:
            return {"error": "Predictive analysis disabled"}
        
        patterns = self.get_system_patterns()
        current_time = datetime.now()
        
        # Basic prediction based on patterns
        predictions = {
            "prediction_time": current_time.isoformat(),
            "horizon_hours": horizon_hours,
            "confidence": 0.0,
            "predictions": {}
        }
        
        # Thermal predictions
        if "thermal_patterns" in patterns:
            thermal_pattern = patterns["thermal_patterns"]
            predictions["predictions"]["thermal"] = {
                "expected_peak_temp": thermal_pattern.get("average_peak", 70),
                "expected_peak_time": self._predict_peak_time(current_time, thermal_pattern),
                "thermal_stress_likelihood": thermal_pattern.get("stress_frequency", 0.1)
            }
        
        # Process activity predictions  
        if "process_patterns" in patterns:
            process_pattern = patterns["process_patterns"]
            predictions["predictions"]["processes"] = {
                "expected_heavy_load_periods": self._predict_load_periods(current_time, process_pattern),
                "likely_ml_workloads": process_pattern.get("ml_workload_frequency", 0.0),
                "process_churn_expected": process_pattern.get("churn_rate", 0.0)
            }
        
        # Calculate overall confidence
        pattern_count = len([p for p in patterns.values() if isinstance(p, dict) and "confidence" in p])
        if pattern_count > 0:
            total_confidence = sum(p.get("confidence", 0.0) for p in patterns.values() 
                                 if isinstance(p, dict) and "confidence" in p)
            predictions["confidence"] = total_confidence / pattern_count
        
        return predictions
    
    def detect_anomalies(self, lookback_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Detect system anomalies.
        
        Args:
            lookback_hours: Hours of history to analyze
            
        Returns:
            List of detected anomalies
        """
        if not self.config.enable_anomaly_detection:
            return []
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)
        
        # Get recent data
        recent_deltas = self.temporal_storage.query(
            start_time=start_time,
            end_time=end_time
        )
        
        anomalies = []
        
        # Detect various types of anomalies
        anomalies.extend(self._detect_thermal_anomalies(recent_deltas))
        anomalies.extend(self._detect_process_anomalies(recent_deltas))
        anomalies.extend(self._detect_resource_anomalies(recent_deltas))
        
        # Sort by severity and confidence
        anomalies.sort(key=lambda x: (x.get("severity", 0), x.get("confidence", 0)), reverse=True)
        
        # Notify callbacks
        for anomaly in anomalies:
            for callback in self.anomaly_callbacks:
                try:
                    callback(anomaly)
                except Exception as e:
                    logger.error(f"Error in anomaly callback: {e}")
        
        return anomalies
    
    def add_event_callback(self, callback: Callable[[SystemDelta], None]) -> None:
        """Add callback for new system deltas."""
        self.event_callbacks.append(callback)
    
    def add_anomaly_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback for detected anomalies."""
        self.anomaly_callbacks.append(callback)
    
    def get_collection_stats(self) -> CollectionStats:
        """Get current collection statistics."""
        with self.stats_lock:
            # Update uptime and storage stats
            self.stats.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            self.stats.storage_stats = self.temporal_storage.get_storage_stats()
            
            return CollectionStats(
                total_collections=self.stats.total_collections,
                successful_collections=self.stats.successful_collections,
                failed_collections=self.stats.failed_collections,
                total_changes_detected=self.stats.total_changes_detected,
                total_events_generated=self.stats.total_events_generated,
                total_patterns_learned=self.stats.total_patterns_learned,
                average_collection_time_ms=self.stats.average_collection_time_ms,
                last_collection_time=self.stats.last_collection_time,
                uptime_seconds=self.stats.uptime_seconds,
                storage_stats=self.stats.storage_stats.copy() if self.stats.storage_stats else None,
                error_counts=self.stats.error_counts.copy()
            )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        stats = self.get_collection_stats()
        
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": stats.uptime_seconds / 3600,
            "is_collecting": self.is_running,
            "components": {
                "system_collector": self._check_system_collector_health(),
                "change_detector": self._check_change_detector_health(),
                "event_extractor": self._check_event_extractor_health(),
                "temporal_storage": self._check_storage_health()
            },
            "performance": {
                "collections_per_hour": stats.successful_collections / max(stats.uptime_seconds / 3600, 1),
                "average_collection_time_ms": stats.average_collection_time_ms,
                "success_rate": stats.successful_collections / max(stats.total_collections, 1),
                "changes_per_collection": stats.total_changes_detected / max(stats.successful_collections, 1),
                "events_per_collection": stats.total_events_generated / max(stats.successful_collections, 1)
            },
            "storage": stats.storage_stats or {},
            "errors": stats.error_counts
        }
        
        # Determine overall health status
        component_issues = [name for name, status in health["components"].items() 
                          if not status.get("healthy", True)]
        
        if component_issues:
            health["status"] = "degraded"
            health["issues"] = component_issues
        
        if stats.failed_collections > stats.successful_collections / 2:
            health["status"] = "unhealthy"
            health["critical_issues"] = ["High failure rate in collections"]
        
        return health
    
    def _collection_loop(self) -> None:
        """Main collection loop running in separate thread."""
        logger.info("Starting temporal collection loop...")
        
        while self.is_running and not self._shutdown_event.is_set():
            try:
                start_time = time.time()
                
                # Perform collection
                delta = self._perform_collection()
                
                if delta:
                    # Notify event callbacks
                    for callback in self.event_callbacks:
                        try:
                            callback(delta)
                        except Exception as e:
                            logger.error(f"Error in event callback: {e}")
                
                # Update stats
                collection_time_ms = (time.time() - start_time) * 1000
                self._update_collection_stats(True, collection_time_ms, delta)
                
                # Periodic health checks
                if (datetime.now() - self._last_health_check).total_seconds() > self._health_check_interval:
                    self._perform_health_check()
                    self._last_health_check = datetime.now()
                
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                self._update_collection_stats(False, 0, None)
                self._record_error("collection_loop", str(e))
            
            # Wait for next collection or shutdown signal
            if self._shutdown_event.wait(self.config.collection_interval_seconds):
                break
        
        logger.info("Temporal collection loop stopped")
    
    def _perform_collection(self) -> Optional[SystemDelta]:
        """Perform a single collection cycle."""
        try:
            # Collect current system snapshot
            current_snapshot = self.system_collector.collect()
            collection_time = datetime.now()
            
            if self.last_snapshot is None:
                # First collection - store as baseline
                self.last_snapshot = current_snapshot
                logger.info("Stored initial system baseline")
                return None
            
            # Detect changes
            changes = self.change_detector.detect_changes(
                old_snapshot=self.last_snapshot,
                new_snapshot=current_snapshot
            )
            
            if not changes:
                # No changes detected
                self.last_snapshot = current_snapshot
                return None
            
            # Extract semantic events
            semantic_events = []
            correlations = []
            
            if self.config.enable_semantic_events:
                semantic_events = self.event_extractor.extract_events(
                    changes=changes,
                    old_snapshot=self.last_snapshot,
                    new_snapshot=current_snapshot
                )
            
            if self.config.enable_correlation_detection:
                correlations = self.event_extractor.detect_correlations(
                    events=semantic_events,
                    changes=changes
                )
            
            # Create system delta
            delta = SystemDelta(
                timestamp=collection_time,
                raw_delta=changes,
                semantic_events=semantic_events,
                correlations=correlations,
                snapshot_metadata={
                    'collection_duration_ms': (datetime.now() - collection_time).total_seconds() * 1000,
                    'change_count': len(changes),
                    'event_count': len(semantic_events),
                    'correlation_count': len(correlations)
                }
            )
            
            # Store delta
            self.temporal_storage.store(delta)
            
            # Update last snapshot
            self.last_snapshot = current_snapshot
            
            logger.debug(f"Collected delta with {len(changes)} changes, {len(semantic_events)} events")
            
            return delta
            
        except Exception as e:
            logger.error(f"Failed to perform collection: {e}")
            self._record_error("collection", str(e))
            return None
    
    def _update_collection_stats(self, success: bool, collection_time_ms: float, 
                               delta: Optional[SystemDelta]) -> None:
        """Update collection statistics."""
        with self.stats_lock:
            self.stats.total_collections += 1
            self.stats.last_collection_time = datetime.now()
            
            if success:
                self.stats.successful_collections += 1
                
                # Update timing stats
                if self.stats.average_collection_time_ms == 0:
                    self.stats.average_collection_time_ms = collection_time_ms
                else:
                    # Running average
                    self.stats.average_collection_time_ms = (
                        (self.stats.average_collection_time_ms * (self.stats.successful_collections - 1) + 
                         collection_time_ms) / self.stats.successful_collections
                    )
                
                # Update content stats
                if delta:
                    self.stats.total_changes_detected += len(delta.raw_delta)
                    self.stats.total_events_generated += len(delta.semantic_events)
            else:
                self.stats.failed_collections += 1
    
    def _record_error(self, component: str, error: str) -> None:
        """Record error for statistics."""
        with self.stats_lock:
            if component not in self.stats.error_counts:
                self.stats.error_counts[component] = 0
            self.stats.error_counts[component] += 1
    
    def _perform_health_check(self) -> None:
        """Perform periodic health check."""
        try:
            # Check storage health and perform maintenance
            storage_health = self.temporal_storage.get_health_status()
            
            if not storage_health.get("healthy", False):
                logger.warning(f"Storage health issues detected: {storage_health}")
            
            # Perform storage maintenance if needed
            if storage_health.get("needs_compaction", False):
                logger.info("Performing storage compaction...")
                self.temporal_storage.compact_storage()
            
            # Check for anomalies
            if self.config.enable_anomaly_detection:
                anomalies = self.detect_anomalies(lookback_hours=1)
                if anomalies:
                    logger.info(f"Detected {len(anomalies)} anomalies in health check")
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            self._record_error("health_check", str(e))
    
    def _check_system_collector_health(self) -> Dict[str, Any]:
        """Check SystemCollector component health."""
        try:
            # Test collection
            test_snapshot = self.system_collector.collect()
            return {
                "healthy": test_snapshot is not None,
                "last_check": datetime.now().isoformat(),
                "snapshot_size": len(str(test_snapshot)) if test_snapshot else 0
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    def _check_change_detector_health(self) -> Dict[str, Any]:
        """Check ChangeDetectionEngine health."""
        try:
            detector_status = self.change_detector.get_detector_status()
            return {
                "healthy": True,
                "active_detectors": len([d for d in detector_status.values() if d.get("enabled", False)]),
                "last_check": datetime.now().isoformat(),
                "detector_status": detector_status
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    def _check_event_extractor_health(self) -> Dict[str, Any]:
        """Check EventExtractionEngine health."""
        try:
            extractor_status = self.event_extractor.get_extractor_status()
            return {
                "healthy": True,
                "active_extractors": len([e for e in extractor_status.values() if e.get("enabled", False)]),
                "last_check": datetime.now().isoformat(),
                "extractor_status": extractor_status
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    def _check_storage_health(self) -> Dict[str, Any]:
        """Check TemporalStorage health."""
        try:
            return self.temporal_storage.get_health_status()
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    def _predict_peak_time(self, current_time: datetime, thermal_pattern: Dict[str, Any]) -> str:
        """Predict thermal peak time based on patterns."""
        # Simplified prediction - in practice would use more sophisticated ML
        typical_peak_hour = thermal_pattern.get("typical_peak_hour", 15)  # 3 PM
        
        # Calculate next peak time
        today_peak = current_time.replace(hour=typical_peak_hour, minute=0, second=0, microsecond=0)
        
        if today_peak <= current_time:
            # Peak already passed today, predict tomorrow
            tomorrow_peak = today_peak + timedelta(days=1)
            return tomorrow_peak.isoformat()
        else:
            return today_peak.isoformat()
    
    def _predict_load_periods(self, current_time: datetime, process_pattern: Dict[str, Any]) -> List[str]:
        """Predict heavy load periods."""
        # Simplified prediction
        heavy_load_hours = process_pattern.get("heavy_load_hours", [9, 13, 17])  # 9 AM, 1 PM, 5 PM
        
        periods = []
        for hour in heavy_load_hours:
            period_time = current_time.replace(hour=hour, minute=0, second=0, microsecond=0)
            if period_time <= current_time:
                period_time += timedelta(days=1)
            periods.append(period_time.isoformat())
        
        return periods
    
    def _detect_thermal_anomalies(self, deltas: List[SystemDelta]) -> List[Dict[str, Any]]:
        """Detect thermal anomalies in recent data."""
        anomalies = []
        
        for delta in deltas:
            for change in delta.raw_delta:
                if change.category == "thermal" and change.significance > 0.8:
                    if isinstance(change.new_value, (int, float)) and change.new_value > 85:
                        anomalies.append({
                            "type": "thermal_anomaly",
                            "severity": "high",
                            "confidence": 0.9,
                            "timestamp": change.timestamp.isoformat(),
                            "description": f"Extreme temperature detected: {change.new_value}°C",
                            "affected_component": change.entity_id,
                            "temperature": change.new_value
                        })
        
        return anomalies
    
    def _detect_process_anomalies(self, deltas: List[SystemDelta]) -> List[Dict[str, Any]]:
        """Detect process-related anomalies."""
        anomalies = []
        
        # Track process starts/stops
        process_events = []
        for delta in deltas:
            for event in delta.semantic_events:
                if event.event_type in ["process_started", "process_crashed", "process_high_resource"]:
                    process_events.append(event)
        
        # Detect rapid process churn
        if len(process_events) > 20:  # More than 20 process events
            anomalies.append({
                "type": "process_churn_anomaly", 
                "severity": "medium",
                "confidence": 0.7,
                "timestamp": datetime.now().isoformat(),
                "description": f"High process activity detected: {len(process_events)} events",
                "event_count": len(process_events)
            })
        
        return anomalies
    
    def _detect_resource_anomalies(self, deltas: List[SystemDelta]) -> List[Dict[str, Any]]:
        """Detect resource usage anomalies."""
        anomalies = []
        
        # Look for sudden resource spikes
        for delta in deltas:
            for change in delta.raw_delta:
                if change.category in ["memory", "cpu"] and change.significance > 0.9:
                    if isinstance(change.new_value, (int, float)) and change.new_value > 90:
                        anomalies.append({
                            "type": "resource_spike_anomaly",
                            "severity": "high",
                            "confidence": 0.8,
                            "timestamp": change.timestamp.isoformat(),
                            "description": f"Resource spike in {change.category}: {change.new_value}%",
                            "resource_type": change.category,
                            "usage_percent": change.new_value
                        })
        
        return anomalies
    
    @asynccontextmanager
    async def async_context(self):
        """Async context manager for temporal collection."""
        self.start()
        try:
            yield self
        finally:
            self.stop()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.is_running:
            self.stop(timeout=5.0)


def create_temporal_collector(config: Optional[CollectionConfig] = None) -> TemporalSystemCollector:
    """
    Create a configured temporal system collector.
    
    Args:
        config: Collection configuration
        
    Returns:
        Configured TemporalSystemCollector instance
    """
    return TemporalSystemCollector(config)