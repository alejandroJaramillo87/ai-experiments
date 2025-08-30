"""
Recent Buffer
=============

High-performance circular buffer for storing recent temporal data.
Provides fast access to detailed system history for the last 24-48 hours.
"""

import json
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Iterator, Dict, Any
from collections import deque

from ..types import SystemDelta


class RecentBuffer:
    """
    Circular buffer for recent SystemDelta objects.
    
    Provides thread-safe storage with automatic eviction of old data.
    Optimized for fast append and recent data retrieval.
    """
    
    def __init__(self, capacity: int = 2880):  # 48 hours at 1-minute intervals
        self.capacity = capacity
        self._buffer: deque[SystemDelta] = deque(maxlen=capacity)
        self._lock = threading.RLock()
        self._total_appends = 0
        self._evicted_count = 0
    
    def append(self, delta: SystemDelta) -> None:
        """
        Append new system delta to buffer.
        
        Args:
            delta: SystemDelta to append
        """
        with self._lock:
            # Track eviction if buffer is full
            if len(self._buffer) >= self.capacity:
                self._evicted_count += 1
            
            self._buffer.append(delta)
            self._total_appends += 1
    
    def get_all(self) -> Iterator[SystemDelta]:
        """
        Get all deltas in chronological order.
        
        Yields:
            SystemDelta objects in chronological order
        """
        with self._lock:
            # Convert to list to avoid modification during iteration
            deltas = list(self._buffer)
        
        # Sort by timestamp to ensure chronological order
        deltas.sort(key=lambda d: d.timestamp)
        yield from deltas
    
    def get_recent(self, hours: int = 24) -> Iterator[SystemDelta]:
        """
        Get deltas from the last N hours.
        
        Args:
            hours: Number of hours of history
            
        Yields:
            Recent SystemDelta objects
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            deltas = [d for d in self._buffer if d.timestamp >= cutoff_time]
        
        deltas.sort(key=lambda d: d.timestamp)
        yield from deltas
    
    def get_range(self, start_time: datetime, end_time: datetime) -> Iterator[SystemDelta]:
        """
        Get deltas within time range.
        
        Args:
            start_time: Range start
            end_time: Range end
            
        Yields:
            SystemDelta objects in time range
        """
        with self._lock:
            deltas = [
                d for d in self._buffer 
                if start_time <= d.timestamp <= end_time
            ]
        
        deltas.sort(key=lambda d: d.timestamp)
        yield from deltas
    
    def get_latest(self, count: int = 10) -> List[SystemDelta]:
        """
        Get most recent N deltas.
        
        Args:
            count: Number of deltas to retrieve
            
        Returns:
            List of most recent deltas
        """
        with self._lock:
            deltas = list(self._buffer)
        
        # Sort by timestamp and take the most recent
        deltas.sort(key=lambda d: d.timestamp, reverse=True)
        return deltas[:count]
    
    def find_by_categories(self, categories: List[str]) -> Iterator[SystemDelta]:
        """
        Find deltas containing changes in specified categories.
        
        Args:
            categories: List of categories to match
            
        Yields:
            Matching SystemDelta objects
        """
        category_set = set(categories)
        
        with self._lock:
            for delta in self._buffer:
                delta_categories = {change.category for change in delta.raw_delta}
                if delta_categories & category_set:  # Intersection
                    yield delta
    
    def find_significant(self, min_significance: float = 0.7) -> Iterator[SystemDelta]:
        """
        Find deltas with significant changes.
        
        Args:
            min_significance: Minimum significance threshold
            
        Yields:
            SystemDelta objects with significant changes
        """
        with self._lock:
            for delta in self._buffer:
                # Check if any change exceeds significance threshold
                if any(change.significance >= min_significance for change in delta.raw_delta):
                    yield delta
    
    def get_time_range(self) -> tuple[Optional[datetime], Optional[datetime]]:
        """
        Get time range of data in buffer.
        
        Returns:
            Tuple of (oldest_timestamp, newest_timestamp)
        """
        with self._lock:
            if not self._buffer:
                return None, None
            
            timestamps = [d.timestamp for d in self._buffer]
            return min(timestamps), max(timestamps)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get buffer statistics.
        
        Returns:
            Dictionary with buffer statistics
        """
        with self._lock:
            current_size = len(self._buffer)
            
            if not self._buffer:
                return {
                    'current_size': 0,
                    'capacity': self.capacity,
                    'utilization': 0.0,
                    'total_appends': self._total_appends,
                    'evicted_count': self._evicted_count,
                    'time_range': None,
                    'total_events': 0,
                    'total_changes': 0
                }
            
            # Calculate time statistics
            timestamps = [d.timestamp for d in self._buffer]
            time_range = (min(timestamps), max(timestamps))
            time_span = (max(timestamps) - min(timestamps)).total_seconds()
            
            # Calculate content statistics
            total_events = sum(len(d.semantic_events) for d in self._buffer)
            total_changes = sum(len(d.raw_delta) for d in self._buffer)
            
            # Calculate average rates
            avg_events_per_delta = total_events / current_size
            avg_changes_per_delta = total_changes / current_size
            
            return {
                'current_size': current_size,
                'capacity': self.capacity,
                'utilization': current_size / self.capacity,
                'total_appends': self._total_appends,
                'evicted_count': self._evicted_count,
                'time_range': time_range,
                'time_span_hours': time_span / 3600,
                'total_events': total_events,
                'total_changes': total_changes,
                'avg_events_per_delta': avg_events_per_delta,
                'avg_changes_per_delta': avg_changes_per_delta
            }
    
    def compact(self) -> Dict[str, Any]:
        """
        Perform buffer compaction by removing old or low-significance data.
        
        Returns:
            Compaction results
        """
        with self._lock:
            original_size = len(self._buffer)
            
            if original_size == 0:
                return {'removed_count': 0, 'retained_count': 0}
            
            # Keep significant deltas and recent deltas
            cutoff_time = datetime.now() - timedelta(hours=12)  # Always keep last 12 hours
            min_significance = 0.3  # Remove low-significance old data
            
            retained_deltas = []
            
            for delta in self._buffer:
                # Always keep recent data
                if delta.timestamp >= cutoff_time:
                    retained_deltas.append(delta)
                # Keep older data only if significant
                elif any(change.significance >= min_significance for change in delta.raw_delta):
                    retained_deltas.append(delta)
                elif any(event.confidence >= 0.8 for event in delta.semantic_events):
                    retained_deltas.append(delta)
            
            # Replace buffer contents
            self._buffer.clear()
            self._buffer.extend(retained_deltas)
            
            removed_count = original_size - len(retained_deltas)
            
            return {
                'removed_count': removed_count,
                'retained_count': len(retained_deltas),
                'original_size': original_size,
                'compression_ratio': removed_count / original_size if original_size > 0 else 0.0
            }
    
    def clear(self) -> None:
        """Clear all data from buffer."""
        with self._lock:
            self._buffer.clear()
            self._total_appends = 0
            self._evicted_count = 0
    
    def save_to_file(self, file_path: Path) -> None:
        """
        Save buffer contents to file.
        
        Args:
            file_path: Path to save file
        """
        with self._lock:
            deltas_data = []
            
            for delta in self._buffer:
                # Convert SystemDelta to serializable format
                delta_dict = {
                    'timestamp': delta.timestamp.isoformat(),
                    'collection_duration_ms': getattr(delta, 'collection_duration_ms', 0),
                    'raw_delta': [self._serialize_change(change) for change in delta.raw_delta],
                    'semantic_events': [self._serialize_event(event) for event in delta.semantic_events],
                    'correlations': [self._serialize_correlation(corr) for corr in delta.correlations],
                    'snapshot_metadata': getattr(delta, 'snapshot_metadata', {})
                }
                deltas_data.append(delta_dict)
            
            # Save to JSON file
            with open(file_path, 'w') as f:
                json.dump({
                    'deltas': deltas_data,
                    'metadata': {
                        'capacity': self.capacity,
                        'total_appends': self._total_appends,
                        'evicted_count': self._evicted_count,
                        'saved_at': datetime.now().isoformat()
                    }
                }, f, indent=2)
    
    def load_from_file(self, file_path: Path) -> None:
        """
        Load buffer contents from file.
        
        Args:
            file_path: Path to load file
        """
        if not file_path.exists():
            return
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        with self._lock:
            self.clear()
            
            # Restore metadata
            metadata = data.get('metadata', {})
            self._total_appends = metadata.get('total_appends', 0)
            self._evicted_count = metadata.get('evicted_count', 0)
            
            # Restore deltas
            for delta_dict in data.get('deltas', []):
                try:
                    delta = self._deserialize_delta(delta_dict)
                    self._buffer.append(delta)
                except Exception as e:
                    # Log error but continue with remaining data
                    print(f"Error deserializing delta: {e}")
    
    def _serialize_change(self, change) -> Dict[str, Any]:
        """Serialize SystemChange for JSON storage."""
        from ..types import ChangeType
        
        return {
            'category': change.category,
            'change_type': change.change_type.value,
            'entity_id': change.entity_id,
            'old_value': self._serialize_value(change.old_value),
            'new_value': self._serialize_value(change.new_value),
            'significance': change.significance,
            'metadata': change.metadata,
            'timestamp': change.timestamp.isoformat()
        }
    
    def _serialize_event(self, event) -> Dict[str, Any]:
        """Serialize SystemEvent for JSON storage."""
        from ..types import EventSeverity
        
        return {
            'event_type': event.event_type,
            'entity': event.entity,
            'description': event.description,
            'severity': event.severity.value,
            'timestamp': event.timestamp.isoformat(),
            'causes': [self._serialize_change(cause) for cause in event.causes],
            'predicted_effects': event.predicted_effects,
            'context': event.context,
            'confidence': event.confidence
        }
    
    def _serialize_correlation(self, correlation) -> Dict[str, Any]:
        """Serialize EventCorrelation for JSON storage."""
        return {
            'correlation_type': getattr(correlation, 'correlation_type', ''),
            'events': [self._serialize_event(event) for event in getattr(correlation, 'events', [])],
            'confidence': getattr(correlation, 'confidence', 0.0),
            'description': getattr(correlation, 'description', ''),
            'pattern_signature': getattr(correlation, 'pattern_signature', None)
        }
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize value for JSON storage."""
        if isinstance(value, datetime):
            return value.isoformat()
        elif hasattr(value, '__dict__'):
            # Try to convert objects to dict
            try:
                return vars(value)
            except:
                return str(value)
        else:
            return value
    
    def _deserialize_delta(self, delta_dict: Dict[str, Any]):
        """Deserialize SystemDelta from JSON data."""
        from ..types import SystemDelta, SystemChange, SystemEvent, ChangeType, EventSeverity
        
        # Deserialize changes
        changes = []
        for change_dict in delta_dict.get('raw_delta', []):
            change = SystemChange(
                category=change_dict['category'],
                change_type=ChangeType(change_dict['change_type']),
                entity_id=change_dict['entity_id'],
                old_value=change_dict['old_value'],
                new_value=change_dict['new_value'],
                significance=change_dict['significance'],
                metadata=change_dict.get('metadata', {}),
                timestamp=datetime.fromisoformat(change_dict['timestamp'])
            )
            changes.append(change)
        
        # Deserialize events
        events = []
        for event_dict in delta_dict.get('semantic_events', []):
            # Deserialize cause changes for this event
            event_causes = []
            for cause_dict in event_dict.get('causes', []):
                cause = SystemChange(
                    category=cause_dict['category'],
                    change_type=ChangeType(cause_dict['change_type']),
                    entity_id=cause_dict['entity_id'],
                    old_value=cause_dict['old_value'],
                    new_value=cause_dict['new_value'],
                    significance=cause_dict['significance'],
                    metadata=cause_dict.get('metadata', {}),
                    timestamp=datetime.fromisoformat(cause_dict['timestamp'])
                )
                event_causes.append(cause)
            
            event = SystemEvent(
                event_type=event_dict['event_type'],
                entity=event_dict['entity'],
                description=event_dict['description'],
                severity=EventSeverity(event_dict['severity']),
                timestamp=datetime.fromisoformat(event_dict['timestamp']),
                causes=event_causes,
                predicted_effects=event_dict.get('predicted_effects', []),
                context=event_dict.get('context', {}),
                confidence=event_dict.get('confidence', 0.0)
            )
            events.append(event)
        
        # Create SystemDelta
        delta = SystemDelta(
            timestamp=datetime.fromisoformat(delta_dict['timestamp']),
            raw_delta=changes,
            semantic_events=events,
            correlations=[],  # Skip correlations for now - complex to deserialize
            snapshot_metadata=delta_dict.get('snapshot_metadata', {})
        )
        
        # Add collection duration if present
        if 'collection_duration_ms' in delta_dict:
            delta.collection_duration_ms = delta_dict['collection_duration_ms']
        
        return delta
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)
    
    def __bool__(self) -> bool:
        """Check if buffer is not empty."""
        return len(self._buffer) > 0