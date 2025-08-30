"""
Base Change Detector Abstract Framework
=======================================

Abstract base class for all change detectors with common functionality for
significance calculation, change filtering, and error handling.

All specialized detectors (GPU, Process, Python Environment, etc.) inherit
from this base to ensure consistent behavior and extensibility.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

from ..types import SystemChange, ChangeType
from ..config import ChangeDetectorConfig


class ChangeDetectorError(Exception):
    """Base exception for change detection errors."""
    pass


class BaseChangeDetector(ABC):
    """
    Abstract base class for all system change detectors.
    
    Provides common functionality for:
    - Significance calculation
    - Change filtering and noise reduction  
    - Error handling and logging
    - Configuration management
    
    Subclasses must implement detect_changes() for domain-specific logic.
    """
    
    def __init__(self, config: ChangeDetectorConfig, category: str):
        """
        Initialize base change detector.
        
        Args:
            config: Configuration for change detection behavior
            category: Category name this detector handles (e.g., 'nvidia_gpu')
        """
        self.config = config
        self.category = category
        self.logger = logging.getLogger(f"temporal.change_detection.{category}")
        
        # Statistics for performance monitoring
        self.detection_count = 0
        self.total_changes_detected = 0
        self.significant_changes_detected = 0
        self.last_detection_time: Optional[datetime] = None
        
        # Error tracking for graceful degradation
        self.error_count = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
    
    @abstractmethod
    def detect_changes(self, old_data: Dict[str, Any], new_data: Dict[str, Any]) -> List[SystemChange]:
        """
        Detect changes between old and new data snapshots.
        
        This is the main method that subclasses must implement with their
        domain-specific change detection logic.
        
        Args:
            old_data: Previous snapshot data for this category
            new_data: Current snapshot data for this category
            
        Returns:
            List of detected SystemChange objects
            
        Raises:
            ChangeDetectorError: If detection fails critically
        """
        pass
    
    def process_changes(self, old_data: Dict[str, Any], new_data: Dict[str, Any]) -> List[SystemChange]:
        """
        Main entry point for change detection with error handling and filtering.
        
        This method wraps detect_changes() with common processing:
        - Error handling and retry logic
        - Change filtering and noise reduction  
        - Significance calculation
        - Statistics tracking
        
        Args:
            old_data: Previous snapshot data
            new_data: Current snapshot data
            
        Returns:
            Filtered list of significant changes
        """
        start_time = datetime.now()
        
        try:
            # Detect raw changes using domain-specific logic
            raw_changes = self.detect_changes(old_data, new_data)
            
            # Filter and process changes
            processed_changes = self._process_detected_changes(raw_changes)
            
            # Update statistics
            self._update_statistics(processed_changes, start_time)
            
            # Reset error tracking on success
            self.consecutive_errors = 0
            
            return processed_changes
            
        except Exception as e:
            self._handle_detection_error(e)
            return []  # Return empty list on error for graceful degradation
    
    def _process_detected_changes(self, changes: List[SystemChange]) -> List[SystemChange]:
        """Process and filter detected changes."""
        filtered_changes = []
        
        for change in changes:
            # Skip changes that should be ignored
            if self._should_ignore_change(change):
                continue
            
            # Calculate significance if not already set
            if change.significance == 0.0:
                change.significance = self._calculate_significance(change)
            
            # Only include changes above minimum threshold
            if change.significance >= self.config.min_significance_threshold:
                filtered_changes.append(change)
        
        # Limit number of changes to prevent overwhelming
        if len(filtered_changes) > self.config.max_changes_per_category:
            # Sort by significance and take top N
            filtered_changes.sort(key=lambda c: c.significance, reverse=True)
            filtered_changes = filtered_changes[:self.config.max_changes_per_category]
            
            self.logger.warning(
                f"Truncated {len(changes)} changes to {self.config.max_changes_per_category} "
                f"for category {self.category}"
            )
        
        return filtered_changes
    
    def _calculate_significance(self, change: SystemChange) -> float:
        """
        Calculate significance score for a change.
        
        Base implementation provides common scoring logic.
        Subclasses can override for domain-specific scoring.
        
        Args:
            change: The change to score
            
        Returns:
            Significance score between 0.0 and 1.0
        """
        # Base scoring factors
        score = 0.5  # Default significance
        
        # Adjust based on change type
        type_weights = {
            ChangeType.ADDED: 0.7,
            ChangeType.REMOVED: 0.8,
            ChangeType.THRESHOLD_CROSSED: 0.9,
            ChangeType.ANOMALY_DETECTED: 0.9,
            ChangeType.STATE_TRANSITION: 0.8,
            ChangeType.MODIFIED: 0.5
        }
        
        if change.change_type in type_weights:
            score = type_weights[change.change_type]
        
        # Adjust based on metadata hints
        if change.metadata:
            # Critical systems or components are more significant
            if any(keyword in str(change.metadata).lower() 
                   for keyword in ['critical', 'error', 'fail', 'gpu', 'cuda']):
                score += 0.2
            
            # Performance impacts are significant
            if any(keyword in str(change.metadata).lower()
                   for keyword in ['performance', 'memory', 'cpu', 'thermal']):
                score += 0.1
        
        # Ensure score stays in valid range
        return max(0.0, min(1.0, score))
    
    def _should_ignore_change(self, change: SystemChange) -> bool:
        """
        Determine if a change should be ignored based on configured patterns.
        
        Args:
            change: The change to evaluate
            
        Returns:
            True if change should be ignored
        """
        for pattern in self.config.ignore_patterns:
            if self._matches_ignore_pattern(change, pattern):
                return True
        return False
    
    def _matches_ignore_pattern(self, change: SystemChange, pattern: str) -> bool:
        """Check if change matches an ignore pattern."""
        # Simple pattern matching - can be enhanced with regex if needed
        if pattern.endswith('*'):
            prefix = pattern[:-1]
            return change.entity_id.startswith(prefix)
        return change.entity_id == pattern
    
    def _handle_detection_error(self, error: Exception):
        """Handle errors during change detection."""
        self.error_count += 1
        self.consecutive_errors += 1
        
        self.logger.error(
            f"Change detection error in {self.category}: {error}",
            exc_info=True
        )
        
        # Check if we should disable this detector temporarily
        if self.consecutive_errors >= self.max_consecutive_errors:
            self.logger.critical(
                f"Change detector for {self.category} has failed {self.consecutive_errors} "
                f"consecutive times. Consider investigating."
            )
    
    def _update_statistics(self, changes: List[SystemChange], start_time: datetime):
        """Update detection statistics."""
        self.detection_count += 1
        self.total_changes_detected += len(changes)
        self.significant_changes_detected += len([c for c in changes if c.is_significant])
        self.last_detection_time = datetime.now()
        
        detection_duration = (datetime.now() - start_time).total_seconds()
        
        if detection_duration > 1.0:  # Log slow detections
            self.logger.warning(
                f"Slow change detection for {self.category}: {detection_duration:.2f}s"
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics for monitoring."""
        return {
            'category': self.category,
            'detection_count': self.detection_count,
            'total_changes_detected': self.total_changes_detected,
            'significant_changes_detected': self.significant_changes_detected,
            'error_count': self.error_count,
            'consecutive_errors': self.consecutive_errors,
            'last_detection_time': self.last_detection_time.isoformat() if self.last_detection_time else None,
            'avg_changes_per_detection': (
                self.total_changes_detected / self.detection_count 
                if self.detection_count > 0 else 0
            ),
            'significant_change_rate': (
                self.significant_changes_detected / self.total_changes_detected
                if self.total_changes_detected > 0 else 0
            )
        }
    
    def is_healthy(self) -> bool:
        """Check if detector is healthy and should continue operating."""
        return self.consecutive_errors < self.max_consecutive_errors
    
    def reset_error_count(self):
        """Reset error counters (useful for recovery)."""
        self.consecutive_errors = 0
    
    # Utility methods for subclasses
    
    def _extract_numeric_value(self, data: Any, key_path: str, default: float = 0.0) -> float:
        """
        Extract numeric value from nested data structure.
        
        Args:
            data: Data structure to search
            key_path: Dot-separated path to value (e.g., 'gpu.temperature.value')
            default: Default value if path not found
            
        Returns:
            Numeric value or default
        """
        try:
            current = data
            for key in key_path.split('.'):
                if isinstance(current, dict):
                    current = current.get(key)
                elif isinstance(current, (list, tuple)) and key.isdigit():
                    current = current[int(key)]
                else:
                    return default
                    
                if current is None:
                    return default
            
            # Convert to float if possible
            return float(current) if current is not None else default
            
        except (KeyError, IndexError, ValueError, TypeError):
            return default
    
    def _extract_string_value(self, data: Any, key_path: str, default: str = "") -> str:
        """Extract string value from nested data structure."""
        try:
            current = data
            for key in key_path.split('.'):
                if isinstance(current, dict):
                    current = current.get(key)
                elif isinstance(current, (list, tuple)) and key.isdigit():
                    current = current[int(key)]
                else:
                    return default
                    
                if current is None:
                    return default
            
            return str(current) if current is not None else default
            
        except (KeyError, IndexError, ValueError, TypeError):
            return default
    
    def _create_change(self, 
                      change_type: ChangeType,
                      entity_id: str,
                      old_value: Any,
                      new_value: Any,
                      metadata: Optional[Dict[str, Any]] = None,
                      significance: Optional[float] = None) -> SystemChange:
        """
        Create a SystemChange with consistent formatting.
        
        Args:
            change_type: Type of change detected
            entity_id: Identifier for the entity that changed
            old_value: Previous value
            new_value: Current value  
            metadata: Additional context about the change
            significance: Pre-calculated significance (will calculate if None)
            
        Returns:
            Properly formatted SystemChange
        """
        change = SystemChange(
            category=self.category,
            change_type=change_type,
            entity_id=entity_id,
            old_value=old_value,
            new_value=new_value,
            significance=significance or 0.0,  # Will be calculated later if 0.0
            metadata=metadata or {},
            timestamp=datetime.now()
        )
        
        return change