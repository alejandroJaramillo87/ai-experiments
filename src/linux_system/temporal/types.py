"""
Core Data Structures for Temporal System Intelligence
=====================================================

Defines the fundamental data types for representing system changes, events,
and correlations over time. These structures form the foundation for building
temporal awareness and system memory.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ChangeType(Enum):
    """Types of system changes that can be detected."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    THRESHOLD_CROSSED = "threshold_crossed"
    ANOMALY_DETECTED = "anomaly_detected"
    STATE_TRANSITION = "state_transition"


class EventSeverity(Enum):
    """Severity levels for system events."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


class CorrelationType(Enum):
    """Types of correlations between events."""
    TEMPORAL_SEQUENCE = "temporal_sequence"
    CAUSAL_CHAIN = "causal_chain"
    RECURRING_PATTERN = "recurring_pattern"
    ANOMALY_CLUSTER = "anomaly_cluster"


@dataclass
class SystemChange:
    """
    Represents a single detected change in system state.
    
    This is the atomic unit of change detection - every difference between
    system snapshots is represented as a SystemChange instance.
    """
    category: str  # 'nvidia_gpu', 'processes', 'python_env', etc.
    change_type: ChangeType
    entity_id: str  # 'gpu:temperature', 'process:1234', 'package:tensorflow'
    old_value: Any
    new_value: Any
    significance: float  # 0.0-1.0, importance of this change
    metadata: Dict[str, Any]
    timestamp: datetime
    
    def __post_init__(self):
        """Validate significance score and ensure metadata is not None."""
        if not 0.0 <= self.significance <= 1.0:
            raise ValueError(f"Significance must be between 0.0 and 1.0, got {self.significance}")
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def change_id(self) -> str:
        """Generate a unique identifier for this change."""
        change_data = f"{self.category}:{self.entity_id}:{self.timestamp.isoformat()}"
        return hashlib.md5(change_data.encode()).hexdigest()[:12]
    
    @property
    def is_significant(self) -> bool:
        """Check if this change is considered significant (>0.5 significance)."""
        return self.significance > 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'category': self.category,
            'change_type': self.change_type.value,
            'entity_id': self.entity_id,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'significance': self.significance,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'change_id': self.change_id
        }


@dataclass
class SystemEvent:
    """
    Represents a semantic event extracted from system changes.
    
    Events are higher-level interpretations of raw changes, providing
    context and meaning to what happened in the system.
    """
    event_type: str  # 'gpu_thermal_event', 'service_started', etc.
    entity: str  # What entity this event relates to
    description: str  # Human-readable description
    severity: EventSeverity
    timestamp: datetime
    causes: List[SystemChange]  # The raw changes that led to this event
    predicted_effects: List[str]  # What we expect this might cause
    context: Dict[str, Any]
    confidence: float = 1.0  # 0.0-1.0, confidence in event interpretation
    
    def __post_init__(self):
        """Validate confidence score and ensure lists/dicts are not None."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.causes is None:
            self.causes = []
        if self.predicted_effects is None:
            self.predicted_effects = []
        if self.context is None:
            self.context = {}
    
    @property
    def event_id(self) -> str:
        """Generate a unique identifier for this event."""
        event_data = f"{self.event_type}:{self.entity}:{self.timestamp.isoformat()}"
        return hashlib.md5(event_data.encode()).hexdigest()[:12]
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence event (>0.8 confidence)."""
        return self.confidence > 0.8
    
    @property
    def primary_cause(self) -> Optional[SystemChange]:
        """Get the most significant cause of this event."""
        if not self.causes:
            return None
        return max(self.causes, key=lambda c: c.significance)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'event_type': self.event_type,
            'entity': self.entity,
            'description': self.description,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'causes': [cause.to_dict() for cause in self.causes],
            'predicted_effects': self.predicted_effects,
            'context': self.context,
            'confidence': self.confidence,
            'event_id': self.event_id
        }


@dataclass
class EventCorrelation:
    """
    Represents a discovered correlation between multiple events.
    
    Correlations capture patterns, causality, and relationships that
    emerge from analyzing multiple events together.
    """
    correlation_type: CorrelationType
    events: List[SystemEvent]
    confidence: float  # 0.0-1.0
    description: str
    pattern_signature: Optional[str] = None  # For pattern matching
    metadata: Dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate confidence score and ensure required fields."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if len(self.events) < 2:
            raise ValueError("Correlation must involve at least 2 events")
    
    @property
    def correlation_id(self) -> str:
        """Generate a unique identifier for this correlation."""
        event_ids = sorted([event.event_id for event in self.events])
        correlation_data = f"{self.correlation_type.value}:{':'.join(event_ids)}"
        return hashlib.md5(correlation_data.encode()).hexdigest()[:12]
    
    @property
    def time_span(self) -> float:
        """Get the time span covered by this correlation in seconds."""
        if len(self.events) < 2:
            return 0.0
        
        timestamps = [event.timestamp for event in self.events]
        return (max(timestamps) - min(timestamps)).total_seconds()
    
    @property
    def involves_critical_events(self) -> bool:
        """Check if any events in this correlation are critical."""
        return any(event.severity == EventSeverity.CRITICAL for event in self.events)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'correlation_type': self.correlation_type.value,
            'events': [event.to_dict() for event in self.events],
            'confidence': self.confidence,
            'description': self.description,
            'pattern_signature': self.pattern_signature,
            'metadata': self.metadata,
            'discovered_at': self.discovered_at.isoformat(),
            'correlation_id': self.correlation_id,
            'time_span_seconds': self.time_span
        }


@dataclass
class SystemDelta:
    """
    Represents the complete set of changes detected between two system snapshots.
    
    This is the primary data structure returned by TemporalSystemCollector,
    containing all detected changes, extracted events, and discovered correlations.
    """
    timestamp: datetime
    collection_duration_ms: int
    raw_delta: List[SystemChange]
    semantic_events: List[SystemEvent]
    correlations: List[EventCorrelation]
    snapshot_metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Ensure all lists are initialized."""
        if self.raw_delta is None:
            self.raw_delta = []
        if self.semantic_events is None:
            self.semantic_events = []
        if self.correlations is None:
            self.correlations = []
        if self.snapshot_metadata is None:
            self.snapshot_metadata = {}
    
    @property
    def delta_id(self) -> str:
        """Generate a unique identifier for this delta."""
        delta_data = f"{self.timestamp.isoformat()}:{len(self.raw_delta)}:{len(self.semantic_events)}"
        return hashlib.md5(delta_data.encode()).hexdigest()[:12]
    
    @property
    def has_significant_changes(self) -> bool:
        """Check if this delta contains any significant changes."""
        return any(change.is_significant for change in self.raw_delta)
    
    @property
    def has_critical_events(self) -> bool:
        """Check if this delta contains any critical events."""
        return any(event.severity == EventSeverity.CRITICAL for event in self.semantic_events)
    
    @property
    def change_categories(self) -> List[str]:
        """Get list of unique categories that had changes."""
        return list(set(change.category for change in self.raw_delta))
    
    @property
    def event_types(self) -> List[str]:
        """Get list of unique event types that occurred."""
        return list(set(event.event_type for event in self.semantic_events))
    
    def get_changes_by_category(self, category: str) -> List[SystemChange]:
        """Get all changes for a specific category."""
        return [change for change in self.raw_delta if change.category == category]
    
    def get_events_by_severity(self, severity: EventSeverity) -> List[SystemEvent]:
        """Get all events of a specific severity level."""
        return [event for event in self.semantic_events if event.severity == severity]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'collection_duration_ms': self.collection_duration_ms,
            'raw_delta': [change.to_dict() for change in self.raw_delta],
            'semantic_events': [event.to_dict() for event in self.semantic_events],
            'correlations': [corr.to_dict() for corr in self.correlations],
            'snapshot_metadata': self.snapshot_metadata,
            'delta_id': self.delta_id,
            'summary': {
                'total_changes': len(self.raw_delta),
                'significant_changes': len([c for c in self.raw_delta if c.is_significant]),
                'total_events': len(self.semantic_events),
                'critical_events': len([e for e in self.semantic_events if e.severity == EventSeverity.CRITICAL]),
                'correlations_found': len(self.correlations),
                'categories_affected': self.change_categories,
                'event_types_detected': self.event_types
            }
        }
    
    @classmethod
    def initial(cls, snapshot: Dict[str, Any]) -> 'SystemDelta':
        """Create an initial delta for the first collection (no changes)."""
        return cls(
            timestamp=datetime.fromisoformat(snapshot['metadata']['timestamp']),
            collection_duration_ms=snapshot['metadata'].get('collection_duration_ms', 0),
            raw_delta=[],
            semantic_events=[],
            correlations=[],
            snapshot_metadata=snapshot['metadata']
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemDelta':
        """Create SystemDelta from dictionary (for deserialization)."""
        # This would be used for loading from storage - implementation would
        # need to handle the reverse conversion of the nested objects
        # For now, just the signature to define the interface
        raise NotImplementedError("Deserialization not yet implemented")


# Type aliases for convenience
ChangeFilter = Union[str, List[str], None]  # For filtering changes by category
EventFilter = Union[EventSeverity, List[EventSeverity], None]  # For filtering events by severity
TimeRange = tuple[datetime, datetime]  # For temporal queries