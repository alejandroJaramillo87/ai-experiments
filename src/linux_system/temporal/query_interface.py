"""
Temporal Query Interface
=======================

High-level interface for searching and analyzing historical system events.
Provides intuitive query capabilities for exploring temporal system data
with natural language-like syntax and powerful filtering.
"""

import re
import logging
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from .types import SystemDelta, SystemChange, SystemEvent, EventSeverity, ChangeType
from .storage import TemporalStorage, TemporalQueryEngine, QueryBuilder


logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Type of temporal query."""
    EVENTS = "events"
    CHANGES = "changes"
    PATTERNS = "patterns"
    ANOMALIES = "anomalies"
    TIMELINE = "timeline"
    SUMMARY = "summary"


class TimeRange(Enum):
    """Predefined time ranges for queries."""
    LAST_HOUR = "last_hour"
    LAST_DAY = "last_day"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    TODAY = "today"
    YESTERDAY = "yesterday"
    THIS_WEEK = "this_week"
    THIS_MONTH = "this_month"


@dataclass
class QueryFilter:
    """Filter criteria for temporal queries."""
    categories: Optional[List[str]] = None
    event_types: Optional[List[str]] = None
    severity: Optional[EventSeverity] = None
    min_significance: Optional[float] = None
    max_significance: Optional[float] = None
    min_confidence: Optional[float] = None
    entities: Optional[List[str]] = None
    text_contains: Optional[str] = None
    change_types: Optional[List[ChangeType]] = None
    exclude_categories: Optional[List[str]] = None
    custom_filters: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class QueryResult:
    """Result from temporal query."""
    query_type: QueryType
    total_results: int
    results: List[Union[SystemDelta, SystemChange, SystemEvent, Dict[str, Any]]]
    execution_time_ms: float
    filters_applied: QueryFilter
    time_range: Tuple[datetime, datetime]
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)


class TemporalQueryInterface:
    """
    High-level interface for querying temporal system data.
    
    Provides intuitive methods for searching historical events, changes,
    and patterns with natural language-like syntax.
    """
    
    def __init__(self, storage: TemporalStorage):
        """
        Initialize query interface.
        
        Args:
            storage: TemporalStorage instance to query
        """
        self.storage = storage
        self.query_engine = TemporalQueryEngine(storage)
        self._query_history: List[QueryResult] = []
    
    # High-level query methods with natural syntax
    
    def when_did(self, description: str, time_range: Union[TimeRange, Tuple[datetime, datetime]] = TimeRange.LAST_DAY) -> QueryResult:
        """
        Find when specific events occurred.
        
        Examples:
        - when_did("GPU temperature spike")
        - when_did("Python packages get updated") 
        - when_did("processes crash")
        
        Args:
            description: Natural description of what to search for
            time_range: Time range to search in
            
        Returns:
            QueryResult with matching events
        """
        # Parse natural language description
        filters = self._parse_description(description)
        start_time, end_time = self._resolve_time_range(time_range)
        
        return self._execute_query(
            query_type=QueryType.EVENTS,
            start_time=start_time,
            end_time=end_time,
            filters=filters
        )
    
    def what_happened(self, time_range: Union[TimeRange, Tuple[datetime, datetime]] = TimeRange.LAST_HOUR) -> QueryResult:
        """
        Get summary of what happened during a time period.
        
        Args:
            time_range: Time range to summarize
            
        Returns:
            QueryResult with timeline summary
        """
        start_time, end_time = self._resolve_time_range(time_range)
        
        return self._execute_query(
            query_type=QueryType.TIMELINE,
            start_time=start_time,
            end_time=end_time,
            filters=QueryFilter()
        )
    
    def find_anomalies(self, time_range: Union[TimeRange, Tuple[datetime, datetime]] = TimeRange.LAST_DAY) -> QueryResult:
        """
        Find anomalous system behavior.
        
        Args:
            time_range: Time range to search for anomalies
            
        Returns:
            QueryResult with detected anomalies
        """
        start_time, end_time = self._resolve_time_range(time_range)
        
        return self._execute_query(
            query_type=QueryType.ANOMALIES,
            start_time=start_time,
            end_time=end_time,
            filters=QueryFilter(min_significance=0.7)
        )
    
    def search_events(self, **kwargs) -> QueryResult:
        """
        Search for events with specific criteria.
        
        Args:
            **kwargs: Search criteria (category, severity, confidence, etc.)
            
        Returns:
            QueryResult with matching events
        """
        # Extract time range
        time_range = kwargs.pop('time_range', TimeRange.LAST_DAY)
        start_time, end_time = self._resolve_time_range(time_range)
        
        # Build filters from kwargs
        filters = QueryFilter(**kwargs)
        
        return self._execute_query(
            query_type=QueryType.EVENTS,
            start_time=start_time,
            end_time=end_time,
            filters=filters
        )
    
    def search_changes(self, **kwargs) -> QueryResult:
        """
        Search for system changes with specific criteria.
        
        Args:
            **kwargs: Search criteria (category, change_type, significance, etc.)
            
        Returns:
            QueryResult with matching changes
        """
        time_range = kwargs.pop('time_range', TimeRange.LAST_DAY)
        start_time, end_time = self._resolve_time_range(time_range)
        
        filters = QueryFilter(**kwargs)
        
        return self._execute_query(
            query_type=QueryType.CHANGES,
            start_time=start_time,
            end_time=end_time,
            filters=filters
        )
    
    def find_patterns(self, time_range: Union[TimeRange, Tuple[datetime, datetime]] = TimeRange.LAST_WEEK) -> QueryResult:
        """
        Find recurring patterns in system behavior.
        
        Args:
            time_range: Time range to analyze for patterns
            
        Returns:
            QueryResult with detected patterns
        """
        start_time, end_time = self._resolve_time_range(time_range)
        
        return self._execute_query(
            query_type=QueryType.PATTERNS,
            start_time=start_time,
            end_time=end_time,
            filters=QueryFilter()
        )
    
    def thermal_analysis(self, time_range: Union[TimeRange, Tuple[datetime, datetime]] = TimeRange.LAST_DAY) -> QueryResult:
        """
        Analyze thermal behavior over time.
        
        Args:
            time_range: Time range for thermal analysis
            
        Returns:
            QueryResult with thermal analysis
        """
        start_time, end_time = self._resolve_time_range(time_range)
        
        return self._execute_query(
            query_type=QueryType.SUMMARY,
            start_time=start_time,
            end_time=end_time,
            filters=QueryFilter(categories=["thermal", "gpu_thermal", "cpu_thermal"])
        )
    
    def process_activity(self, time_range: Union[TimeRange, Tuple[datetime, datetime]] = TimeRange.LAST_DAY) -> QueryResult:
        """
        Analyze process activity patterns.
        
        Args:
            time_range: Time range for process analysis
            
        Returns:
            QueryResult with process activity analysis
        """
        start_time, end_time = self._resolve_time_range(time_range)
        
        return self._execute_query(
            query_type=QueryType.SUMMARY,
            start_time=start_time,
            end_time=end_time,
            filters=QueryFilter(categories=["process", "service"])
        )
    
    def performance_trends(self, time_range: Union[TimeRange, Tuple[datetime, datetime]] = TimeRange.LAST_WEEK) -> QueryResult:
        """
        Analyze performance trends over time.
        
        Args:
            time_range: Time range for performance analysis
            
        Returns:
            QueryResult with performance trend analysis
        """
        start_time, end_time = self._resolve_time_range(time_range)
        
        return self._execute_query(
            query_type=QueryType.SUMMARY,
            start_time=start_time,
            end_time=end_time,
            filters=QueryFilter(categories=["performance", "cpu", "memory", "gpu"])
        )
    
    # Advanced query builder methods
    
    def build_query(self) -> 'QueryBuilder':
        """Get a query builder for complex queries."""
        return AdvancedQueryBuilder(self)
    
    def execute_raw_query(self, query_dict: Dict[str, Any]) -> QueryResult:
        """Execute raw query dictionary."""
        start_time = query_dict.get('start_time', datetime.now() - timedelta(days=1))
        end_time = query_dict.get('end_time', datetime.now())
        query_type = QueryType(query_dict.get('type', 'events'))
        
        # Convert query_dict to QueryFilter
        filter_dict = {k: v for k, v in query_dict.items() 
                      if k not in ['start_time', 'end_time', 'type']}
        filters = QueryFilter(**filter_dict)
        
        return self._execute_query(query_type, start_time, end_time, filters)
    
    # Utility methods
    
    def get_query_history(self) -> List[QueryResult]:
        """Get history of executed queries."""
        return self._query_history.copy()
    
    def get_available_categories(self, time_range: Union[TimeRange, Tuple[datetime, datetime]] = TimeRange.LAST_WEEK) -> List[str]:
        """Get list of available categories for filtering."""
        start_time, end_time = self._resolve_time_range(time_range)
        
        # Query all data in time range to get categories
        deltas = self.query_engine.execute_query_unified(
            start_time=start_time,
            end_time=end_time,
            limit=1000  # Sample to get categories
        )
        
        categories = set()
        for delta in deltas:
            for change in delta.raw_delta:
                categories.add(change.category)
        
        return sorted(list(categories))
    
    def get_available_event_types(self, time_range: Union[TimeRange, Tuple[datetime, datetime]] = TimeRange.LAST_WEEK) -> List[str]:
        """Get list of available event types."""
        start_time, end_time = self._resolve_time_range(time_range)
        
        deltas = self.query_engine.execute_query_unified(
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )
        
        event_types = set()
        for delta in deltas:
            for event in delta.semantic_events:
                event_types.add(event.event_type)
        
        return sorted(list(event_types))
    
    def _execute_query(self, query_type: QueryType, start_time: datetime, 
                      end_time: datetime, filters: QueryFilter) -> QueryResult:
        """Execute temporal query and return formatted results."""
        query_start = datetime.now()
        
        try:
            if query_type == QueryType.EVENTS:
                results = self._query_events(start_time, end_time, filters)
                
            elif query_type == QueryType.CHANGES:
                results = self._query_changes(start_time, end_time, filters)
                
            elif query_type == QueryType.PATTERNS:
                results = self._query_patterns(start_time, end_time, filters)
                
            elif query_type == QueryType.ANOMALIES:
                results = self._query_anomalies(start_time, end_time, filters)
                
            elif query_type == QueryType.TIMELINE:
                results = self._query_timeline(start_time, end_time, filters)
                
            elif query_type == QueryType.SUMMARY:
                results = self._query_summary(start_time, end_time, filters)
                
            else:
                results = []
            
            execution_time = (datetime.now() - query_start).total_seconds() * 1000
            
            # Generate insights
            insights = self._generate_insights(results, query_type, filters)
            
            # Calculate summary stats
            summary_stats = self._calculate_summary_stats(results, query_type)
            
            query_result = QueryResult(
                query_type=query_type,
                total_results=len(results),
                results=results,
                execution_time_ms=execution_time,
                filters_applied=filters,
                time_range=(start_time, end_time),
                summary_stats=summary_stats,
                insights=insights
            )
            
            # Add to query history
            self._query_history.append(query_result)
            
            # Keep only last 100 queries
            if len(self._query_history) > 100:
                self._query_history = self._query_history[-100:]
            
            return query_result
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return QueryResult(
                query_type=query_type,
                total_results=0,
                results=[],
                execution_time_ms=0.0,
                filters_applied=filters,
                time_range=(start_time, end_time),
                insights=[f"Query failed: {str(e)}"]
            )
    
    def _query_events(self, start_time: datetime, end_time: datetime, 
                     filters: QueryFilter) -> List[SystemEvent]:
        """Query system events with filters."""
        deltas = self.query_engine.execute_query_unified(
            start_time=start_time,
            end_time=end_time
        )
        
        events = []
        for delta in deltas:
            for event in delta.semantic_events:
                if self._matches_event_filter(event, filters):
                    events.append(event)
        
        return events
    
    def _query_changes(self, start_time: datetime, end_time: datetime,
                      filters: QueryFilter) -> List[SystemChange]:
        """Query system changes with filters."""
        deltas = self.query_engine.execute_query_unified(
            start_time=start_time,
            end_time=end_time
        )
        
        changes = []
        for delta in deltas:
            for change in delta.raw_delta:
                if self._matches_change_filter(change, filters):
                    changes.append(change)
        
        return changes
    
    def _query_patterns(self, start_time: datetime, end_time: datetime,
                       filters: QueryFilter) -> List[Dict[str, Any]]:
        """Query for patterns in system behavior."""
        # Use pattern store to find patterns
        if hasattr(self.storage, 'pattern_store'):
            patterns = self.storage.pattern_store.get_patterns_in_range(start_time, end_time)
            return [pattern.__dict__ if hasattr(pattern, '__dict__') else pattern for pattern in patterns]
        
        # Fallback: basic pattern detection on events
        events = self._query_events(start_time, end_time, filters)
        return self._detect_simple_patterns(events)
    
    def _query_anomalies(self, start_time: datetime, end_time: datetime,
                        filters: QueryFilter) -> List[Dict[str, Any]]:
        """Query for anomalous system behavior."""
        # Get high-significance changes and events
        filters.min_significance = filters.min_significance or 0.8
        
        changes = self._query_changes(start_time, end_time, filters)
        events = self._query_events(start_time, end_time, filters)
        
        anomalies = []
        
        # Convert significant changes to anomaly format
        for change in changes:
            if change.significance >= 0.8:
                anomalies.append({
                    'type': 'significant_change',
                    'timestamp': change.timestamp.isoformat(),
                    'category': change.category,
                    'description': f"Significant {change.category} change in {change.entity_id}",
                    'significance': change.significance,
                    'details': {
                        'old_value': change.old_value,
                        'new_value': change.new_value,
                        'change_type': change.change_type.value
                    }
                })
        
        # Convert high-confidence critical events
        for event in events:
            if event.severity.value == 'critical' or event.confidence >= 0.9:
                anomalies.append({
                    'type': 'critical_event',
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type,
                    'description': event.description,
                    'confidence': event.confidence,
                    'severity': event.severity.value,
                    'entity': event.entity
                })
        
        return anomalies
    
    def _query_timeline(self, start_time: datetime, end_time: datetime,
                       filters: QueryFilter) -> List[Dict[str, Any]]:
        """Create timeline of system activity."""
        deltas = self.query_engine.execute_query_unified(
            start_time=start_time,
            end_time=end_time
        )
        
        timeline = []
        for delta in deltas:
            if delta.semantic_events or delta.raw_delta:
                timeline.append({
                    'timestamp': delta.timestamp.isoformat(),
                    'events': [
                        {
                            'type': event.event_type,
                            'description': event.description,
                            'severity': event.severity.value,
                            'confidence': event.confidence
                        }
                        for event in delta.semantic_events
                        if self._matches_event_filter(event, filters)
                    ],
                    'changes': [
                        {
                            'category': change.category,
                            'entity': change.entity_id,
                            'significance': change.significance,
                            'change_type': change.change_type.value
                        }
                        for change in delta.raw_delta
                        if self._matches_change_filter(change, filters)
                    ]
                })
        
        return timeline
    
    def _query_summary(self, start_time: datetime, end_time: datetime,
                      filters: QueryFilter) -> List[Dict[str, Any]]:
        """Generate summary analysis of system activity."""
        events = self._query_events(start_time, end_time, filters)
        changes = self._query_changes(start_time, end_time, filters)
        
        # Category analysis
        category_stats = {}
        for change in changes:
            cat = change.category
            if cat not in category_stats:
                category_stats[cat] = {'count': 0, 'avg_significance': 0.0, 'changes': []}
            category_stats[cat]['count'] += 1
            category_stats[cat]['changes'].append(change.significance)
        
        # Calculate averages
        for cat, stats in category_stats.items():
            if stats['changes']:
                stats['avg_significance'] = sum(stats['changes']) / len(stats['changes'])
                stats['max_significance'] = max(stats['changes'])
                del stats['changes']  # Remove raw data
        
        # Event analysis
        event_stats = {}
        for event in events:
            event_type = event.event_type
            if event_type not in event_stats:
                event_stats[event_type] = {'count': 0, 'avg_confidence': 0.0, 'severities': []}
            event_stats[event_type]['count'] += 1
            event_stats[event_type]['severities'].append(event.severity.value)
        
        summary = [
            {
                'analysis_type': 'category_summary',
                'time_range': f"{start_time.isoformat()} to {end_time.isoformat()}",
                'total_changes': len(changes),
                'total_events': len(events),
                'category_breakdown': category_stats,
                'event_breakdown': event_stats
            }
        ]
        
        return summary
    
    def _matches_event_filter(self, event: SystemEvent, filters: QueryFilter) -> bool:
        """Check if event matches filter criteria."""
        if filters.event_types and event.event_type not in filters.event_types:
            return False
        
        if filters.severity and event.severity != filters.severity:
            return False
        
        if filters.min_confidence and event.confidence < filters.min_confidence:
            return False
        
        if filters.entities and event.entity not in filters.entities:
            return False
        
        if filters.text_contains and filters.text_contains.lower() not in event.description.lower():
            return False
        
        return True
    
    def _matches_change_filter(self, change: SystemChange, filters: QueryFilter) -> bool:
        """Check if change matches filter criteria."""
        if filters.categories and change.category not in filters.categories:
            return False
        
        if filters.exclude_categories and change.category in filters.exclude_categories:
            return False
        
        if filters.change_types and change.change_type not in filters.change_types:
            return False
        
        if filters.min_significance and change.significance < filters.min_significance:
            return False
        
        if filters.max_significance and change.significance > filters.max_significance:
            return False
        
        if filters.entities and change.entity_id not in filters.entities:
            return False
        
        return True
    
    def _resolve_time_range(self, time_range: Union[TimeRange, Tuple[datetime, datetime]]) -> Tuple[datetime, datetime]:
        """Resolve time range to start/end timestamps."""
        if isinstance(time_range, tuple):
            return time_range
        
        now = datetime.now()
        
        if time_range == TimeRange.LAST_HOUR:
            return (now - timedelta(hours=1), now)
        elif time_range == TimeRange.LAST_DAY:
            return (now - timedelta(days=1), now)
        elif time_range == TimeRange.LAST_WEEK:
            return (now - timedelta(weeks=1), now)
        elif time_range == TimeRange.LAST_MONTH:
            return (now - timedelta(days=30), now)
        elif time_range == TimeRange.TODAY:
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return (start_of_day, now)
        elif time_range == TimeRange.YESTERDAY:
            yesterday = now - timedelta(days=1)
            start_of_yesterday = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_yesterday = start_of_yesterday + timedelta(days=1)
            return (start_of_yesterday, end_of_yesterday)
        elif time_range == TimeRange.THIS_WEEK:
            days_since_monday = now.weekday()
            start_of_week = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
            return (start_of_week, now)
        elif time_range == TimeRange.THIS_MONTH:
            start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return (start_of_month, now)
        else:
            # Default to last day
            return (now - timedelta(days=1), now)
    
    def _parse_description(self, description: str) -> QueryFilter:
        """Parse natural language description into filter criteria."""
        description_lower = description.lower()
        filters = QueryFilter()
        
        # Extract categories from description
        category_keywords = {
            'gpu': ['gpu', 'graphics', 'cuda', 'nvidia'],
            'thermal': ['temperature', 'thermal', 'hot', 'cool', 'heat'],
            'process': ['process', 'application', 'program'],
            'python': ['python', 'pip', 'package'],
            'memory': ['memory', 'ram', 'oom'],
            'cpu': ['cpu', 'processor'],
            'network': ['network', 'connection', 'internet'],
            'service': ['service', 'daemon', 'systemd']
        }
        
        detected_categories = []
        for category, keywords in category_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                detected_categories.append(category)
        
        if detected_categories:
            filters.categories = detected_categories
        
        # Extract severity indicators
        if any(word in description_lower for word in ['critical', 'severe', 'emergency']):
            filters.severity = EventSeverity.CRITICAL
        elif any(word in description_lower for word in ['high', 'important']):
            filters.severity = EventSeverity.HIGH
        
        # Extract significance indicators
        if any(word in description_lower for word in ['major', 'significant', 'big']):
            filters.min_significance = 0.7
        elif any(word in description_lower for word in ['minor', 'small', 'little']):
            filters.max_significance = 0.3
        
        # Extract specific event types from description
        event_type_patterns = {
            'crash': r'\b(crash|crashed|crashing)\b',
            'start': r'\b(start|started|starting|launch|launched)\b',
            'stop': r'\b(stop|stopped|stopping|end|ended)\b',
            'spike': r'\b(spike|spiked|peak|peaked)\b',
            'update': r'\b(update|updated|upgrade|upgraded)\b'
        }
        
        detected_event_types = []
        for event_type, pattern in event_type_patterns.items():
            if re.search(pattern, description_lower):
                detected_event_types.append(event_type)
        
        if detected_event_types:
            filters.event_types = detected_event_types
        
        # Extract text search terms
        filters.text_contains = description
        
        return filters
    
    def _generate_insights(self, results: List[Any], query_type: QueryType, 
                          filters: QueryFilter) -> List[str]:
        """Generate insights from query results."""
        insights = []
        
        if not results:
            insights.append("No results found for the specified criteria")
            return insights
        
        if query_type == QueryType.EVENTS:
            events = results
            if len(events) > 10:
                insights.append(f"High activity detected: {len(events)} events found")
            
            # Severity distribution
            severities = [e.severity.value for e in events if hasattr(e, 'severity')]
            if severities:
                critical_count = severities.count('critical')
                if critical_count > 0:
                    insights.append(f"Found {critical_count} critical events - requires attention")
        
        elif query_type == QueryType.CHANGES:
            changes = results
            avg_significance = sum(c.significance for c in changes) / len(changes)
            if avg_significance > 0.8:
                insights.append(f"High-impact changes detected (avg significance: {avg_significance:.2f})")
            
            # Category analysis
            categories = [c.category for c in changes]
            most_common = max(set(categories), key=categories.count) if categories else None
            if most_common:
                insights.append(f"Most activity in: {most_common}")
        
        elif query_type == QueryType.ANOMALIES:
            if len(results) > 5:
                insights.append(f"Multiple anomalies detected ({len(results)}) - system may be unstable")
            
        return insights
    
    def _calculate_summary_stats(self, results: List[Any], query_type: QueryType) -> Dict[str, Any]:
        """Calculate summary statistics for results."""
        stats = {
            'result_count': len(results),
            'query_type': query_type.value
        }
        
        if query_type == QueryType.EVENTS and results:
            events = results
            confidences = [e.confidence for e in events if hasattr(e, 'confidence')]
            if confidences:
                stats['avg_confidence'] = sum(confidences) / len(confidences)
                stats['min_confidence'] = min(confidences)
                stats['max_confidence'] = max(confidences)
        
        elif query_type == QueryType.CHANGES and results:
            changes = results
            significances = [c.significance for c in changes if hasattr(c, 'significance')]
            if significances:
                stats['avg_significance'] = sum(significances) / len(significances)
                stats['min_significance'] = min(significances)
                stats['max_significance'] = max(significances)
        
        return stats
    
    def _detect_simple_patterns(self, events: List[SystemEvent]) -> List[Dict[str, Any]]:
        """Detect simple patterns in events."""
        patterns = []
        
        if len(events) < 3:
            return patterns
        
        # Group events by type
        event_type_groups = {}
        for event in events:
            if event.event_type not in event_type_groups:
                event_type_groups[event.event_type] = []
            event_type_groups[event.event_type].append(event)
        
        # Look for repeating patterns
        for event_type, type_events in event_type_groups.items():
            if len(type_events) >= 3:
                # Calculate time intervals
                timestamps = sorted([e.timestamp for e in type_events])
                intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                           for i in range(len(timestamps)-1)]
                
                # Check if intervals are roughly regular
                if len(set(int(interval/60) for interval in intervals)) <= 2:  # Within 1 minute buckets
                    avg_interval = sum(intervals) / len(intervals)
                    patterns.append({
                        'pattern_type': 'recurring_event',
                        'event_type': event_type,
                        'occurrences': len(type_events),
                        'avg_interval_minutes': avg_interval / 60,
                        'confidence': 0.6,
                        'description': f"Recurring {event_type} events every ~{avg_interval/60:.1f} minutes"
                    })
        
        return patterns


class AdvancedQueryBuilder:
    """Advanced query builder for complex temporal queries."""
    
    def __init__(self, interface: TemporalQueryInterface):
        self.interface = interface
        self.query_dict = {}
    
    def time_range(self, start: datetime, end: datetime) -> 'AdvancedQueryBuilder':
        """Set time range."""
        self.query_dict['start_time'] = start
        self.query_dict['end_time'] = end
        return self
    
    def categories(self, *categories: str) -> 'AdvancedQueryBuilder':
        """Filter by categories."""
        self.query_dict['categories'] = list(categories)
        return self
    
    def significance(self, min_sig: float = None, max_sig: float = None) -> 'AdvancedQueryBuilder':
        """Filter by significance range."""
        if min_sig is not None:
            self.query_dict['min_significance'] = min_sig
        if max_sig is not None:
            self.query_dict['max_significance'] = max_sig
        return self
    
    def events(self) -> 'AdvancedQueryBuilder':
        """Query for events."""
        self.query_dict['type'] = 'events'
        return self
    
    def changes(self) -> 'AdvancedQueryBuilder':
        """Query for changes."""
        self.query_dict['type'] = 'changes'
        return self
    
    def execute(self) -> QueryResult:
        """Execute the built query."""
        return self.interface.execute_raw_query(self.query_dict)


def create_query_interface(storage: TemporalStorage) -> TemporalQueryInterface:
    """
    Create a temporal query interface.
    
    Args:
        storage: TemporalStorage instance
        
    Returns:
        Configured TemporalQueryInterface
    """
    return TemporalQueryInterface(storage)