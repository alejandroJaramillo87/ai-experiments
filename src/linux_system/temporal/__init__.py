"""
Linux System Temporal Intelligence
==================================

Temporal data collection and analysis for creating an omniscient system host.
Transforms snapshot-based monitoring into temporal consciousness with change detection,
event correlation, and predictive intelligence.

Core Components:
- TemporalSystemCollector: Main orchestrator for temporal data collection
- ChangeDetectionEngine: Orchestrates all system change detectors
- EventExtractionEngine: Converts raw changes into semantic events
- TemporalStorage: Hierarchical memory system for temporal data

This system builds on the existing SystemCollector to create a "living memory"
of your Linux machine that evolves alongside the physical system.
"""

from .types import (
    SystemDelta,
    SystemChange, 
    SystemEvent,
    EventCorrelation,
    ChangeType,
    EventSeverity,
    CorrelationType
)

from .collector import TemporalSystemCollector, CollectionConfig, CollectionStats, create_temporal_collector
from .change_detection import ChangeDetectionEngine
from .event_extraction_engine import EventExtractionEngine
from .storage import TemporalStorage

__all__ = [
    # Core types
    'SystemDelta',
    'SystemChange', 
    'SystemEvent',
    'EventCorrelation',
    'ChangeType',
    'EventSeverity', 
    'CorrelationType',
    
    # Main components
    'TemporalSystemCollector',
    'ChangeDetectionEngine',
    'EventExtractionEngine', 
    'TemporalStorage',
    
    # Configuration and utilities
    'CollectionConfig',
    'CollectionStats',
    'create_temporal_collector'
]