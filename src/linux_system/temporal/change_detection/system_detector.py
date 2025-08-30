"""
System Change Detector Orchestrator
===================================

Main orchestrator that manages all change detectors and coordinates
the detection process across all system categories.

This is the primary entry point for change detection, managing the 
detector registry and providing a unified interface for detecting
changes across all monitored system aspects.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..types import SystemChange
from ..config import ChangeDetectorConfig, TemporalSystemConfig
from .base_detector import BaseChangeDetector, ChangeDetectorError
from .registry import ChangeDetectorRegistry, get_registry
from .significance import SignificanceCalculator


class SystemChangeDetector:
    """
    Orchestrates change detection across all system categories.
    
    Features:
    - Manages detector registry and lifecycle
    - Coordinates parallel detection across categories
    - Provides unified change detection interface
    - Handles errors and graceful degradation
    - Tracks performance and health metrics
    """
    
    def __init__(self, config: TemporalSystemConfig):
        """
        Initialize system change detector.
        
        Args:
            config: Complete temporal system configuration
        """
        self.config = config
        self.logger = logging.getLogger("temporal.change_detection.system")
        
        # Use global registry for detector management
        self.registry = get_registry()
        
        # Significance calculator for scoring changes
        self.significance_calculator = SignificanceCalculator()
        
        # Thread pool for parallel detection (if enabled)
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        if config.change_detection.enable_parallel_detection:
            self.thread_pool = ThreadPoolExecutor(
                max_workers=min(8, len(self.registry.get_registered_categories())),
                thread_name_prefix="change_detector"
            )
        
        # Performance tracking
        self.detection_count = 0
        self.total_detection_time = 0.0
        self.total_changes_detected = 0
        self.last_detection_time: Optional[datetime] = None
        
        # Error tracking for system health monitoring
        self.consecutive_detection_errors = 0
        self.max_consecutive_errors = 5
        
        self.logger.info("SystemChangeDetector initialized")
    
    def compute_delta(self, 
                     old_snapshot: Dict[str, Any], 
                     new_snapshot: Dict[str, Any]) -> List[SystemChange]:
        """
        Compute changes between two system snapshots.
        
        This is the main entry point for change detection. It orchestrates
        detection across all registered categories and returns a unified
        list of detected changes.
        
        Args:
            old_snapshot: Previous system snapshot
            new_snapshot: Current system snapshot
            
        Returns:
            List of detected SystemChange objects, sorted by significance
        """
        start_time = time.time()
        
        try:
            # Get available data categories from snapshots
            available_categories = self._get_available_categories(old_snapshot, new_snapshot)
            
            if not available_categories:
                self.logger.warning("No common categories found between snapshots")
                return []
            
            # Detect changes across all categories
            all_changes = self._detect_changes_parallel(
                old_snapshot, new_snapshot, available_categories
            ) if self.thread_pool else self._detect_changes_sequential(
                old_snapshot, new_snapshot, available_categories
            )
            
            # Post-process changes
            processed_changes = self._post_process_changes(all_changes)
            
            # Update performance statistics
            self._update_detection_statistics(processed_changes, start_time)
            
            # Reset error counter on successful detection
            self.consecutive_detection_errors = 0
            
            self.logger.debug(
                f"Detected {len(processed_changes)} changes across "
                f"{len(available_categories)} categories in "
                f"{(time.time() - start_time):.3f}s"
            )
            
            return processed_changes
            
        except Exception as e:
            self._handle_detection_error(e)
            return []
    
    def _get_available_categories(self, 
                                 old_snapshot: Dict[str, Any], 
                                 new_snapshot: Dict[str, Any]) -> List[str]:
        """Get categories that have data in both snapshots."""
        if 'data' not in old_snapshot or 'data' not in new_snapshot:
            return []
        
        old_categories = set(old_snapshot['data'].keys())
        new_categories = set(new_snapshot['data'].keys())
        common_categories = old_categories.intersection(new_categories)
        
        # Filter to only registered and enabled categories
        available_categories = []
        for category in common_categories:
            if (self.registry.is_registered(category) and 
                not self.registry.is_disabled(category)):
                available_categories.append(category)
        
        return available_categories
    
    def _detect_changes_parallel(self, 
                               old_snapshot: Dict[str, Any],
                               new_snapshot: Dict[str, Any], 
                               categories: List[str]) -> List[SystemChange]:
        """Detect changes in parallel across categories."""
        all_changes = []
        
        # Submit detection tasks
        future_to_category = {}
        for category in categories:
            future = self.thread_pool.submit(
                self._detect_changes_for_category,
                category, old_snapshot, new_snapshot
            )
            future_to_category[future] = category
        
        # Collect results as they complete
        for future in as_completed(future_to_category):
            category = future_to_category[future]
            try:
                changes = future.result(timeout=self.config.max_processing_time_seconds)
                all_changes.extend(changes)
                
            except Exception as e:
                self.logger.error(f"Change detection failed for {category}: {e}")
                # Continue with other categories for graceful degradation
        
        return all_changes
    
    def _detect_changes_sequential(self, 
                                  old_snapshot: Dict[str, Any],
                                  new_snapshot: Dict[str, Any], 
                                  categories: List[str]) -> List[SystemChange]:
        """Detect changes sequentially across categories."""
        all_changes = []
        
        for category in categories:
            try:
                changes = self._detect_changes_for_category(
                    category, old_snapshot, new_snapshot
                )
                all_changes.extend(changes)
                
            except Exception as e:
                self.logger.error(f"Change detection failed for {category}: {e}")
                # Continue with other categories for graceful degradation
        
        return all_changes
    
    def _detect_changes_for_category(self, 
                                   category: str,
                                   old_snapshot: Dict[str, Any], 
                                   new_snapshot: Dict[str, Any]) -> List[SystemChange]:
        """Detect changes for a specific category."""
        try:
            # Get appropriate detector configuration
            detector_config = self._get_detector_config(category)
            
            # Get detector instance
            detector = self.registry.get_detector(category, detector_config)
            
            # Extract category data
            old_data = old_snapshot['data'][category]
            new_data = new_snapshot['data'][category]
            
            # Detect changes
            changes = detector.process_changes(old_data, new_data)
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Error detecting changes for {category}: {e}")
            raise ChangeDetectorError(f"Failed to detect changes for {category}: {e}")
    
    def _get_detector_config(self, category: str) -> ChangeDetectorConfig:
        """Get configuration for a specific detector category."""
        # Map categories to specific configuration objects
        config_mapping = {
            'nvidia_gpu': self.config.gpu_detection,
            'processes': self.config.process_detection,
            'python_env': self.config.python_env_detection,
            # Add more mappings as needed
        }
        
        # Return specific config or default change detection config
        return config_mapping.get(category, self.config.change_detection)
    
    def _post_process_changes(self, changes: List[SystemChange]) -> List[SystemChange]:
        """Post-process detected changes for final output."""
        if not changes:
            return []
        
        # Recalculate significance scores with historical context
        for change in changes:
            if change.significance == 0.0:  # Not yet calculated
                change.significance = self.significance_calculator.calculate(
                    change, self._get_recent_changes()
                )
        
        # Sort by significance (most significant first)
        changes.sort(key=lambda c: c.significance, reverse=True)
        
        # Apply global limits
        max_total_changes = self.config.change_detection.max_changes_per_category * 10
        if len(changes) > max_total_changes:
            self.logger.warning(
                f"Truncating {len(changes)} changes to {max_total_changes} "
                f"to prevent overwhelming"
            )
            changes = changes[:max_total_changes]
        
        return changes
    
    def _get_recent_changes(self) -> List[SystemChange]:
        """Get recent changes for historical context (placeholder)."""
        # This would pull from temporal storage in a complete implementation
        # For now, return empty list
        return []
    
    def _update_detection_statistics(self, changes: List[SystemChange], start_time: float):
        """Update detection performance statistics."""
        self.detection_count += 1
        detection_time = time.time() - start_time
        self.total_detection_time += detection_time
        self.total_changes_detected += len(changes)
        self.last_detection_time = datetime.now()
        
        # Log slow detections
        if detection_time > 5.0:
            self.logger.warning(f"Slow change detection: {detection_time:.2f}s")
    
    def _handle_detection_error(self, error: Exception):
        """Handle errors during change detection."""
        self.consecutive_detection_errors += 1
        
        self.logger.error(f"System change detection error: {error}", exc_info=True)
        
        if self.consecutive_detection_errors >= self.max_consecutive_errors:
            self.logger.critical(
                f"Change detection has failed {self.consecutive_detection_errors} "
                f"consecutive times. System may be unhealthy."
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics."""
        # Basic statistics
        stats = {
            'detection_count': self.detection_count,
            'total_changes_detected': self.total_changes_detected,
            'total_detection_time': self.total_detection_time,
            'consecutive_errors': self.consecutive_detection_errors,
            'last_detection_time': (
                self.last_detection_time.isoformat() 
                if self.last_detection_time else None
            ),
            'avg_detection_time': (
                self.total_detection_time / self.detection_count 
                if self.detection_count > 0 else 0.0
            ),
            'avg_changes_per_detection': (
                self.total_changes_detected / self.detection_count
                if self.detection_count > 0 else 0.0
            )
        }
        
        # Registry status
        stats['registry'] = self.registry.get_registry_status()
        
        # Individual detector statistics
        stats['detectors'] = self.registry.get_detector_statistics()
        
        # Significance calculator statistics
        stats['significance_calculator'] = self.significance_calculator.get_statistics()
        
        return stats
    
    def is_healthy(self) -> bool:
        """Check if the system change detector is healthy."""
        return (
            self.consecutive_detection_errors < self.max_consecutive_errors and
            len(self.registry.get_healthy_detectors()) > 0
        )
    
    def reset_error_counters(self):
        """Reset all error counters."""
        self.consecutive_detection_errors = 0
        self.registry.reset_detector_errors()
        self.logger.info("Reset all error counters")
    
    def __del__(self):
        """Cleanup resources on destruction."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)