"""
Temporal Assertions
===================

Custom assertion utilities for testing temporal system components.
Provides high-level assertions that understand the semantics of
temporal data structures and relationships.
"""

import re
from typing import List, Any, Optional, Pattern
from datetime import datetime, timedelta

from src.linux_system.temporal.types import (
    SystemChange, SystemEvent, EventCorrelation, SystemDelta,
    ChangeType, EventSeverity, CorrelationType
)


class TemporalAssertions:
    """High-level assertion utilities for temporal data testing."""
    
    @staticmethod
    def assert_change_detected(
        changes: List[SystemChange],
        category: str,
        change_type: ChangeType,
        entity_pattern: str,
        min_significance: float = 0.0,
        message: Optional[str] = None
    ):
        """
        Assert that a specific change was detected.
        
        Args:
            changes: List of changes to search
            category: Expected category
            change_type: Expected change type
            entity_pattern: Regex pattern for entity ID
            min_significance: Minimum significance threshold
            message: Custom failure message
        """
        matching_changes = [
            c for c in changes
            if (c.category == category and
                c.change_type == change_type and
                re.search(entity_pattern, c.entity_id) and
                c.significance >= min_significance)
        ]
        
        if not matching_changes:
            default_message = (
                f"Expected change not found: category='{category}', "
                f"type='{change_type.value}', entity_pattern='{entity_pattern}', "
                f"min_significance={min_significance}. "
                f"Found {len(changes)} total changes: "
                f"{[f'{c.category}:{c.change_type.value}:{c.entity_id}' for c in changes[:5]]}"
            )
            raise AssertionError(message or default_message)
    
    @staticmethod
    def assert_no_change_detected(
        changes: List[SystemChange],
        category: str,
        entity_pattern: str,
        message: Optional[str] = None
    ):
        """Assert that no change was detected for specific criteria."""
        matching_changes = [
            c for c in changes
            if (c.category == category and
                re.search(entity_pattern, c.entity_id))
        ]
        
        if matching_changes:
            default_message = (
                f"Unexpected changes found: category='{category}', "
                f"entity_pattern='{entity_pattern}'. "
                f"Found changes: {[c.entity_id for c in matching_changes]}"
            )
            raise AssertionError(message or default_message)
    
    @staticmethod
    def assert_event_generated(
        events: List[SystemEvent],
        event_type: str,
        severity: Optional[EventSeverity] = None,
        min_confidence: float = 0.0,
        entity_pattern: Optional[str] = None,
        message: Optional[str] = None
    ):
        """
        Assert that a specific event was generated.
        
        Args:
            events: List of events to search
            event_type: Expected event type
            severity: Expected severity (optional)
            min_confidence: Minimum confidence threshold
            entity_pattern: Regex pattern for entity (optional)
            message: Custom failure message
        """
        matching_events = []
        
        for event in events:
            if (event.event_type == event_type and
                event.confidence >= min_confidence and
                (severity is None or event.severity == severity) and
                (entity_pattern is None or re.search(entity_pattern, event.entity))):
                matching_events.append(event)
        
        if not matching_events:
            default_message = (
                f"Expected event not found: type='{event_type}', "
                f"severity={severity.value if severity else 'any'}, "
                f"min_confidence={min_confidence}, "
                f"entity_pattern={entity_pattern or 'any'}. "
                f"Found {len(events)} total events: "
                f"{[f'{e.event_type}:{e.severity.value}:{e.entity}' for e in events[:5]]}"
            )
            raise AssertionError(message or default_message)
    
    @staticmethod
    def assert_events_in_sequence(
        events: List[SystemEvent],
        expected_sequence: List[str],
        max_time_gap: timedelta = timedelta(minutes=10),
        message: Optional[str] = None
    ):
        """
        Assert that events occur in a specific sequence within time constraints.
        
        Args:
            events: List of events (assumed to be chronologically sorted)
            expected_sequence: List of expected event types in order
            max_time_gap: Maximum allowed time between events
            message: Custom failure message
        """
        if len(events) < len(expected_sequence):
            default_message = (
                f"Not enough events for sequence. Expected {len(expected_sequence)} "
                f"events: {expected_sequence}, but got {len(events)} events"
            )
            raise AssertionError(message or default_message)
        
        # Find matching subsequence
        sequence_found = False
        for i in range(len(events) - len(expected_sequence) + 1):
            subsequence = events[i:i + len(expected_sequence)]
            
            # Check if event types match
            if [e.event_type for e in subsequence] == expected_sequence:
                # Check timing constraints
                valid_timing = True
                for j in range(len(subsequence) - 1):
                    time_gap = subsequence[j + 1].timestamp - subsequence[j].timestamp
                    if time_gap > max_time_gap:
                        valid_timing = False
                        break
                
                if valid_timing:
                    sequence_found = True
                    break
        
        if not sequence_found:
            default_message = (
                f"Expected event sequence not found: {expected_sequence}. "
                f"Got event types: {[e.event_type for e in events]}. "
                f"Max time gap allowed: {max_time_gap}"
            )
            raise AssertionError(message or default_message)
    
    @staticmethod
    def assert_correlation_found(
        correlations: List[EventCorrelation],
        correlation_type: CorrelationType,
        min_confidence: float = 0.0,
        expected_event_count: Optional[int] = None,
        message: Optional[str] = None
    ):
        """
        Assert that a specific correlation was found.
        
        Args:
            correlations: List of correlations to search
            correlation_type: Expected correlation type
            min_confidence: Minimum confidence threshold
            expected_event_count: Expected number of events in correlation
            message: Custom failure message
        """
        matching_correlations = [
            c for c in correlations
            if (c.correlation_type == correlation_type and
                c.confidence >= min_confidence and
                (expected_event_count is None or len(c.events) == expected_event_count))
        ]
        
        if not matching_correlations:
            default_message = (
                f"Expected correlation not found: type='{correlation_type.value}', "
                f"min_confidence={min_confidence}, "
                f"expected_event_count={expected_event_count}. "
                f"Found {len(correlations)} correlations: "
                f"{[(c.correlation_type.value, len(c.events), c.confidence) for c in correlations]}"
            )
            raise AssertionError(message or default_message)
    
    @staticmethod
    def assert_delta_summary_valid(
        delta: SystemDelta,
        min_changes: int = 0,
        min_events: int = 0,
        expected_categories: Optional[List[str]] = None,
        message: Optional[str] = None
    ):
        """
        Assert that a SystemDelta has valid summary characteristics.
        
        Args:
            delta: SystemDelta to validate
            min_changes: Minimum expected changes
            min_events: Minimum expected events
            expected_categories: List of expected categories with changes
            message: Custom failure message
        """
        if len(delta.raw_delta) < min_changes:
            default_message = (
                f"Expected at least {min_changes} changes, got {len(delta.raw_delta)}"
            )
            raise AssertionError(message or default_message)
        
        if len(delta.semantic_events) < min_events:
            default_message = (
                f"Expected at least {min_events} events, got {len(delta.semantic_events)}"
            )
            raise AssertionError(message or default_message)
        
        if expected_categories:
            actual_categories = set(delta.change_categories)
            expected_categories_set = set(expected_categories)
            
            if not expected_categories_set.issubset(actual_categories):
                missing_categories = expected_categories_set - actual_categories
                default_message = (
                    f"Expected categories missing: {list(missing_categories)}. "
                    f"Got categories: {list(actual_categories)}"
                )
                raise AssertionError(message or default_message)
    
    @staticmethod
    def assert_thermal_threshold_crossing(
        change: SystemChange,
        expected_old_threshold: str,
        expected_new_threshold: str,
        message: Optional[str] = None
    ):
        """Assert that a thermal threshold crossing change has expected metadata."""
        if change.category != "nvidia_gpu" or change.change_type != ChangeType.THRESHOLD_CROSSED:
            default_message = f"Expected GPU thermal threshold crossing, got {change.category}:{change.change_type.value}"
            raise AssertionError(message or default_message)
        
        metadata = change.metadata
        if (metadata.get('old_threshold') != expected_old_threshold or
            metadata.get('new_threshold') != expected_new_threshold):
            default_message = (
                f"Expected threshold transition {expected_old_threshold} → {expected_new_threshold}, "
                f"got {metadata.get('old_threshold')} → {metadata.get('new_threshold')}"
            )
            raise AssertionError(message or default_message)
    
    @staticmethod
    def assert_significance_in_range(
        changes: List[SystemChange],
        min_significance: float,
        max_significance: float,
        message: Optional[str] = None
    ):
        """Assert that all changes have significance scores in expected range."""
        out_of_range_changes = [
            c for c in changes
            if not (min_significance <= c.significance <= max_significance)
        ]
        
        if out_of_range_changes:
            default_message = (
                f"Changes with significance outside range [{min_significance}, {max_significance}]: "
                f"{[(c.entity_id, c.significance) for c in out_of_range_changes]}"
            )
            raise AssertionError(message or default_message)
    
    @staticmethod
    def assert_timestamps_chronological(
        events: List[SystemEvent],
        message: Optional[str] = None
    ):
        """Assert that events are in chronological order."""
        for i in range(len(events) - 1):
            if events[i].timestamp > events[i + 1].timestamp:
                default_message = (
                    f"Events not in chronological order at index {i}: "
                    f"{events[i].timestamp} > {events[i + 1].timestamp}"
                )
                raise AssertionError(message or default_message)
    
    @staticmethod
    def assert_event_has_causes(
        event: SystemEvent,
        expected_cause_count: Optional[int] = None,
        expected_categories: Optional[List[str]] = None,
        message: Optional[str] = None
    ):
        """Assert that an event has appropriate causal changes."""
        if expected_cause_count is not None and len(event.causes) != expected_cause_count:
            default_message = (
                f"Expected {expected_cause_count} causes, got {len(event.causes)}"
            )
            raise AssertionError(message or default_message)
        
        if expected_categories:
            actual_categories = [c.category for c in event.causes]
            for expected_category in expected_categories:
                if expected_category not in actual_categories:
                    default_message = (
                        f"Expected cause category '{expected_category}' not found. "
                        f"Got categories: {actual_categories}"
                    )
                    raise AssertionError(message or default_message)
    
    @staticmethod
    def assert_predicted_effects_present(
        event: SystemEvent,
        expected_effects: List[str],
        message: Optional[str] = None
    ):
        """Assert that an event predicts expected effects."""
        missing_effects = [
            effect for effect in expected_effects
            if effect not in event.predicted_effects
        ]
        
        if missing_effects:
            default_message = (
                f"Expected predicted effects missing: {missing_effects}. "
                f"Got effects: {event.predicted_effects}"
            )
            raise AssertionError(message or default_message)