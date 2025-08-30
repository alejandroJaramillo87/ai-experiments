"""
Event Extraction Engine
=======================

Orchestrates all system event extractors to provide comprehensive
semantic event generation from system changes.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .types import SystemChange, SystemEvent, SystemSnapshot, EventCorrelation
from .extractors import (
    ThermalEventExtractor,
    ServiceEventExtractor, 
    PackageEventExtractor,
    PerformanceEventExtractor
)

logger = logging.getLogger(__name__)


class EventExtractionEngine:
    """
    Engine that orchestrates all event extraction components.
    
    Manages multiple specialized extractors and provides unified
    interface for generating semantic events from system changes.
    """
    
    def __init__(self, config: Any):
        """
        Initialize event extraction engine.
        
        Args:
            config: Configuration object with extractor settings
        """
        self.config = config
        self.extractors = {}
        self.min_confidence = getattr(config, 'min_event_confidence', 0.3)
        
        # Initialize extractors
        self.extractors['thermal'] = ThermalEventExtractor()
        self.extractors['service'] = ServiceEventExtractor()
        self.extractors['package'] = PackageEventExtractor()
        self.extractors['performance'] = PerformanceEventExtractor()
        
        logger.info(f"Initialized event extraction engine with {len(self.extractors)} extractors: {list(self.extractors.keys())}")
    
    def extract_events(self, changes: List[SystemChange], 
                      old_snapshot: SystemSnapshot,
                      new_snapshot: SystemSnapshot) -> List[SystemEvent]:
        """
        Extract semantic events from system changes.
        
        Args:
            changes: List of detected changes
            old_snapshot: Previous system state
            new_snapshot: Current system state
            
        Returns:
            List of extracted events
        """
        all_events = []
        
        for extractor_name, extractor in self.extractors.items():
            try:
                events = extractor.extract_events(changes, old_snapshot, new_snapshot)
                
                # Filter by confidence threshold
                filtered_events = [e for e in events if e.confidence >= self.min_confidence]
                all_events.extend(filtered_events)
                
                logger.debug(f"Extractor {extractor_name} generated {len(events)} events, {len(filtered_events)} above threshold")
                
            except Exception as e:
                logger.error(f"Error in {extractor_name} extractor: {e}")
        
        logger.debug(f"Total events extracted: {len(all_events)}")
        return all_events
    
    def detect_correlations(self, events: List[SystemEvent],
                          changes: List[SystemChange]) -> List[EventCorrelation]:
        """
        Detect correlations between events.
        
        Args:
            events: List of system events
            changes: List of system changes
            
        Returns:
            List of detected correlations
        """
        correlations = []
        
        if not self.config.enable_correlation_detection:
            return correlations
        
        # Group events by time windows
        time_windows = self._group_events_by_time(events)
        
        for window_events in time_windows:
            if len(window_events) < 2:
                continue
                
            # Look for causal relationships
            causal_correlations = self._detect_causal_correlations(window_events, changes)
            correlations.extend(causal_correlations)
            
            # Look for temporal patterns
            temporal_correlations = self._detect_temporal_correlations(window_events)
            correlations.extend(temporal_correlations)
        
        logger.debug(f"Detected {len(correlations)} event correlations")
        return correlations
    
    def get_extractor_status(self) -> Dict[str, Any]:
        """Get status of all extractors."""
        status = {}
        
        for name, extractor in self.extractors.items():
            try:
                if hasattr(extractor, 'get_status'):
                    extractor_status = extractor.get_status()
                else:
                    extractor_status = {
                        "enabled": True,
                        "healthy": True,
                        "last_check": datetime.now().isoformat()
                    }
                
                status[name] = extractor_status
                
            except Exception as e:
                status[name] = {
                    "enabled": False,
                    "healthy": False,
                    "error": str(e),
                    "last_check": datetime.now().isoformat()
                }
        
        return status
    
    def add_extractor(self, name: str, extractor: Any) -> None:
        """Add a new extractor to the engine."""
        self.extractors[name] = extractor
        logger.info(f"Added extractor: {name}")
    
    def remove_extractor(self, name: str) -> bool:
        """Remove an extractor from the engine."""
        if name in self.extractors:
            del self.extractors[name]
            logger.info(f"Removed extractor: {name}")
            return True
        return False
    
    def get_available_extractors(self) -> List[str]:
        """Get list of available extractor names."""
        return list(self.extractors.keys())
    
    def _group_events_by_time(self, events: List[SystemEvent], 
                             window_seconds: int = 300) -> List[List[SystemEvent]]:
        """Group events into time windows for correlation analysis."""
        if not events:
            return []
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        windows = []
        current_window = [sorted_events[0]]
        window_start = sorted_events[0].timestamp
        
        for event in sorted_events[1:]:
            time_diff = (event.timestamp - window_start).total_seconds()
            
            if time_diff <= window_seconds:
                current_window.append(event)
            else:
                # Start new window
                if len(current_window) > 1:
                    windows.append(current_window)
                current_window = [event]
                window_start = event.timestamp
        
        # Add final window
        if len(current_window) > 1:
            windows.append(current_window)
        
        return windows
    
    def _detect_causal_correlations(self, events: List[SystemEvent], 
                                  changes: List[SystemChange]) -> List[EventCorrelation]:
        """Detect causal relationships between events."""
        correlations = []
        
        # Look for thermal -> performance correlations
        thermal_events = [e for e in events if 'thermal' in e.event_type.lower()]
        performance_events = [e for e in events if 'performance' in e.event_type.lower() or 'slow' in e.event_type.lower()]
        
        for thermal_event in thermal_events:
            for perf_event in performance_events:
                if abs((perf_event.timestamp - thermal_event.timestamp).total_seconds()) < 60:
                    correlation = EventCorrelation(
                        correlation_type="causal",
                        events=[thermal_event, perf_event],
                        confidence=0.7,
                        description=f"Thermal event may have caused performance impact",
                        pattern_signature=f"thermal_to_performance_{thermal_event.severity.value}"
                    )
                    correlations.append(correlation)
        
        # Look for process -> resource correlations
        process_events = [e for e in events if 'process' in e.event_type.lower()]
        resource_events = [e for e in events if any(cat in e.event_type.lower() for cat in ['memory', 'cpu', 'gpu'])]
        
        for process_event in process_events:
            for resource_event in resource_events:
                if abs((resource_event.timestamp - process_event.timestamp).total_seconds()) < 30:
                    correlation = EventCorrelation(
                        correlation_type="causal",
                        events=[process_event, resource_event],
                        confidence=0.6,
                        description=f"Process event related to resource change",
                        pattern_signature=f"process_to_resource_{process_event.event_type}"
                    )
                    correlations.append(correlation)
        
        return correlations
    
    def _detect_temporal_correlations(self, events: List[SystemEvent]) -> List[EventCorrelation]:
        """Detect temporal patterns in events."""
        correlations = []
        
        # Look for repeating patterns
        event_types = [e.event_type for e in events]
        
        # Simple pattern detection - look for repeated sequences
        if len(set(event_types)) < len(event_types):
            # There are repeated event types
            correlation = EventCorrelation(
                correlation_type="temporal_pattern",
                events=events,
                confidence=0.5,
                description=f"Repeated event pattern detected",
                pattern_signature=f"pattern_{'_'.join(event_types[:3])}"
            )
            correlations.append(correlation)
        
        return correlations