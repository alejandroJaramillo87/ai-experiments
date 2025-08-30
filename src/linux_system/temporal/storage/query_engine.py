"""
Temporal Query Engine
====================

Unified query interface for temporal data across all storage layers.
Orchestrates searches across recent buffer, daily summaries, and pattern store.
"""

from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum

from ..types import SystemDelta, SystemChange, SystemEvent
from .recent_buffer import RecentBuffer
from .daily_aggregator import DailyAggregator, DailySummary
from .pattern_store import PatternStore, SystemPattern
from .search_index import TemporalSearchIndex, SearchResult


class QueryType(Enum):
    """Types of temporal queries."""
    DELTAS = "deltas"           # Return full system deltas
    EVENTS = "events"           # Return semantic events only
    CHANGES = "changes"         # Return raw changes only
    SUMMARIES = "summaries"     # Return daily summaries
    PATTERNS = "patterns"       # Return learned patterns
    TIMELINE = "timeline"       # Return timeline view
    STATISTICS = "statistics"   # Return aggregated statistics


class TimeRange(Enum):
    """Common time range presets."""
    LAST_HOUR = "last_hour"
    LAST_DAY = "last_day"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    LAST_YEAR = "last_year"
    ALL_TIME = "all_time"


@dataclass
class TemporalQuery:
    """Comprehensive temporal query specification."""
    # Time range
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    time_range: Optional[TimeRange] = None
    
    # Content filters
    categories: Optional[List[str]] = None
    event_types: Optional[List[str]] = None
    change_types: Optional[List[str]] = None
    entities: Optional[List[str]] = None  # Entity ID patterns
    
    # Significance and confidence filters
    min_significance: Optional[float] = None
    max_significance: Optional[float] = None
    min_confidence: Optional[float] = None
    
    # Content search
    search_terms: Optional[List[str]] = None
    description_contains: Optional[str] = None
    
    # Result configuration
    result_type: QueryType = QueryType.DELTAS
    limit: int = 100
    offset: int = 0
    sort_by: str = "timestamp"  # "timestamp", "significance", "confidence"
    sort_order: str = "desc"    # "asc", "desc"
    
    # Advanced options
    include_correlations: bool = True
    include_predictions: bool = False
    group_by_category: bool = False
    aggregate_by: Optional[str] = None  # "hour", "day", "week"
    
    # Pattern-specific options
    pattern_types: Optional[List[str]] = None
    min_pattern_confidence: Optional[float] = None
    
    def resolve_time_range(self) -> tuple[Optional[datetime], Optional[datetime]]:
        """Resolve time range preset to actual datetime range."""
        if self.start_time and self.end_time:
            return self.start_time, self.end_time
        
        if not self.time_range:
            return self.start_time, self.end_time
        
        now = datetime.now()
        
        if self.time_range == TimeRange.LAST_HOUR:
            return now - timedelta(hours=1), now
        elif self.time_range == TimeRange.LAST_DAY:
            return now - timedelta(days=1), now
        elif self.time_range == TimeRange.LAST_WEEK:
            return now - timedelta(weeks=1), now
        elif self.time_range == TimeRange.LAST_MONTH:
            return now - timedelta(days=30), now
        elif self.time_range == TimeRange.LAST_YEAR:
            return now - timedelta(days=365), now
        elif self.time_range == TimeRange.ALL_TIME:
            return None, None
        
        return self.start_time, self.end_time


@dataclass
class QueryResult:
    """Unified query result."""
    deltas: List[SystemDelta] = field(default_factory=list)
    events: List[SystemEvent] = field(default_factory=list)
    changes: List[SystemChange] = field(default_factory=list)
    summaries: List[DailySummary] = field(default_factory=list)
    patterns: List[SystemPattern] = field(default_factory=list)
    timeline: Dict[datetime, Dict[str, Any]] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Result metadata
    total_results: int = 0
    query_time_ms: float = 0.0
    sources_used: List[str] = field(default_factory=list)
    
    def get_primary_results(self, query_type: QueryType) -> Any:
        """Get primary results based on query type."""
        if query_type == QueryType.DELTAS:
            return self.deltas
        elif query_type == QueryType.EVENTS:
            return self.events
        elif query_type == QueryType.CHANGES:
            return self.changes
        elif query_type == QueryType.SUMMARIES:
            return self.summaries
        elif query_type == QueryType.PATTERNS:
            return self.patterns
        elif query_type == QueryType.TIMELINE:
            return self.timeline
        elif query_type == QueryType.STATISTICS:
            return self.statistics
        
        return []


class TemporalQueryEngine:
    """
    Unified query engine for temporal data.
    
    Orchestrates searches across all storage layers and provides
    a unified interface for temporal data access.
    """
    
    def __init__(self, recent_buffer: RecentBuffer, daily_aggregator: DailyAggregator,
                 pattern_store: PatternStore, search_index: TemporalSearchIndex):
        self.recent_buffer = recent_buffer
        self.daily_aggregator = daily_aggregator
        self.pattern_store = pattern_store
        self.search_index = search_index
    
    def execute_query(self, query: TemporalQuery) -> List[SystemDelta]:
        """
        Execute temporal query and return results.
        
        Args:
            query: Temporal query specification
            
        Returns:
            List of matching system deltas (legacy interface)
        """
        result = self.execute_query_unified(query)
        return result.get_primary_results(query.result_type)
    
    def execute_query_unified(self, query: TemporalQuery) -> QueryResult:
        """
        Execute temporal query with unified result format.
        
        Args:
            query: Temporal query specification
            
        Returns:
            Unified query result with metadata
        """
        start_time = datetime.now()
        result = QueryResult()
        
        # Resolve time range
        start_time_resolved, end_time_resolved = query.resolve_time_range()
        
        # Route query based on type and time range
        if query.result_type == QueryType.PATTERNS:
            result.patterns = self._query_patterns(query)
            result.sources_used.append("pattern_store")
            
        elif query.result_type == QueryType.SUMMARIES:
            result.summaries = self._query_summaries(query, start_time_resolved, end_time_resolved)
            result.sources_used.append("daily_aggregator")
            
        elif query.result_type == QueryType.STATISTICS:
            result.statistics = self._query_statistics(query, start_time_resolved, end_time_resolved)
            result.sources_used.extend(["recent_buffer", "daily_aggregator", "pattern_store"])
            
        elif query.result_type == QueryType.TIMELINE:
            result.timeline = self._query_timeline(query, start_time_resolved, end_time_resolved)
            result.sources_used.extend(["recent_buffer", "daily_aggregator"])
            
        else:
            # Query for deltas, events, or changes
            result.deltas = self._query_deltas(query, start_time_resolved, end_time_resolved)
            result.sources_used.extend(self._get_sources_used(start_time_resolved, end_time_resolved))
            
            # Extract specific data types if requested
            if query.result_type == QueryType.EVENTS:
                result.events = self._extract_events_from_deltas(result.deltas)
            elif query.result_type == QueryType.CHANGES:
                result.changes = self._extract_changes_from_deltas(result.deltas)
        
        # Set result metadata
        result.total_results = self._count_results(result, query.result_type)
        result.query_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return result
    
    def _query_deltas(self, query: TemporalQuery, start_time: Optional[datetime], 
                     end_time: Optional[datetime]) -> List[SystemDelta]:
        """Query for system deltas across storage layers."""
        deltas = []
        
        # Determine which storage layers to query based on time range
        use_recent = self._should_use_recent_buffer(start_time, end_time)
        use_daily = self._should_use_daily_aggregator(start_time, end_time)
        
        if use_recent:
            recent_deltas = self._query_recent_buffer(query, start_time, end_time)
            deltas.extend(recent_deltas)
        
        if use_daily:
            # For daily aggregator, we need to reconstruct deltas from summaries
            # This is a simplified approach - in practice might store more detail
            daily_deltas = self._query_daily_as_deltas(query, start_time, end_time)
            deltas.extend(daily_deltas)
        
        # Apply additional filters
        deltas = self._filter_deltas(deltas, query)
        
        # Sort results
        deltas = self._sort_deltas(deltas, query.sort_by, query.sort_order)
        
        # Apply limit and offset
        if query.offset > 0:
            deltas = deltas[query.offset:]
        if query.limit > 0:
            deltas = deltas[:query.limit]
        
        return deltas
    
    def _query_recent_buffer(self, query: TemporalQuery, start_time: Optional[datetime],
                           end_time: Optional[datetime]) -> List[SystemDelta]:
        """Query recent buffer for deltas."""
        if start_time and end_time:
            return list(self.recent_buffer.get_range(start_time, end_time))
        elif start_time:
            # Get from start_time to now
            return list(self.recent_buffer.get_range(start_time, datetime.now()))
        elif end_time:
            # Get recent data up to end_time
            all_deltas = list(self.recent_buffer.get_all())
            return [d for d in all_deltas if d.timestamp <= end_time]
        else:
            # Get all recent data
            return list(self.recent_buffer.get_all())
    
    def _query_daily_as_deltas(self, query: TemporalQuery, start_time: Optional[datetime],
                             end_time: Optional[datetime]) -> List[SystemDelta]:
        """Query daily aggregator and convert summaries to delta-like objects."""
        # This is a simplified conversion - daily summaries don't contain full delta info
        # In a production system, might store more detailed daily data
        
        start_date = start_time.date() if start_time else None
        end_date = end_time.date() if end_time else None
        
        if start_date and end_date:
            summaries = self.daily_aggregator.get_summaries_range(start_date, end_date)
        else:
            summaries = self.daily_aggregator.get_recent_summaries(30)  # Last 30 days
        
        # Convert summaries to pseudo-deltas for unified interface
        deltas = []
        for summary in summaries:
            # Create a synthetic delta from summary data
            pseudo_delta = self._create_pseudo_delta_from_summary(summary)
            if pseudo_delta:
                deltas.append(pseudo_delta)
        
        return deltas
    
    def _create_pseudo_delta_from_summary(self, summary: DailySummary) -> Optional[SystemDelta]:
        """Create pseudo-delta from daily summary (simplified)."""
        # This is a placeholder - in practice would need more sophisticated conversion
        from ..types import SystemDelta
        
        # Create timestamp for middle of the day
        timestamp = datetime.combine(summary.date, datetime.min.time().replace(hour=12))
        
        # Create pseudo-delta with summary information
        pseudo_delta = SystemDelta(
            timestamp=timestamp,
            raw_delta=[],  # Summary doesn't contain individual changes
            semantic_events=[],  # Could reconstruct from high_confidence_events
            correlations=[],
            snapshot_metadata={
                'source': 'daily_summary',
                'original_delta_count': summary.total_deltas,
                'summary_data': summary.to_dict()
            }
        )
        
        return pseudo_delta
    
    def _query_patterns(self, query: TemporalQuery) -> List[SystemPattern]:
        """Query pattern store for matching patterns."""
        patterns = []
        
        if query.pattern_types:
            for pattern_type in query.pattern_types:
                type_patterns = self.pattern_store.get_patterns_by_type(pattern_type)
                patterns.extend(type_patterns)
        else:
            # Get all confident patterns
            min_confidence = query.min_pattern_confidence or 0.7
            patterns = self.pattern_store.get_confident_patterns(min_confidence)
        
        # Apply additional filtering
        if query.search_terms:
            patterns = [p for p in patterns if self._pattern_matches_search(p, query.search_terms)]
        
        # Sort by confidence and occurrence count
        patterns.sort(key=lambda p: (p.confidence, p.occurrence_count), reverse=True)
        
        return patterns[:query.limit]
    
    def _query_summaries(self, query: TemporalQuery, start_time: Optional[datetime],
                        end_time: Optional[datetime]) -> List[DailySummary]:
        """Query daily aggregator for summaries."""
        if start_time and end_time:
            start_date = start_time.date()
            end_date = end_time.date()
            return self.daily_aggregator.get_summaries_range(start_date, end_date)
        else:
            # Default to recent summaries
            days = 30  # Default to last 30 days
            if query.limit and query.limit < 100:
                days = min(query.limit, 30)
            return self.daily_aggregator.get_recent_summaries(days)
    
    def _query_statistics(self, query: TemporalQuery, start_time: Optional[datetime],
                         end_time: Optional[datetime]) -> Dict[str, Any]:
        """Query for aggregated statistics."""
        stats = {
            'query_time_range': {
                'start': start_time.isoformat() if start_time else None,
                'end': end_time.isoformat() if end_time else None
            }
        }
        
        # Recent buffer statistics
        if self._should_use_recent_buffer(start_time, end_time):
            recent_stats = self.recent_buffer.get_stats()
            stats['recent_buffer'] = recent_stats
        
        # Daily aggregator statistics
        if self._should_use_daily_aggregator(start_time, end_time):
            daily_stats = {
                'summary_count': self.daily_aggregator.get_summary_count(),
                'total_events': self.daily_aggregator.get_total_events(),
                'time_range': self.daily_aggregator.get_time_range()
            }
            stats['daily_summaries'] = daily_stats
        
        # Pattern store statistics
        pattern_stats = {
            'total_patterns': self.pattern_store.get_pattern_count(),
            'confident_patterns': len(self.pattern_store.get_confident_patterns()),
            'pattern_types': {}
        }
        
        # Get pattern type breakdown
        for pattern_type in ['temporal', 'causal', 'thermal', 'ml_workload']:
            type_patterns = self.pattern_store.get_patterns_by_type(pattern_type)
            pattern_stats['pattern_types'][pattern_type] = len(type_patterns)
        
        stats['patterns'] = pattern_stats
        
        # Search index statistics
        index_stats = self.search_index.get_statistics()
        stats['search_index'] = index_stats
        
        return stats
    
    def _query_timeline(self, query: TemporalQuery, start_time: Optional[datetime],
                       end_time: Optional[datetime]) -> Dict[datetime, Dict[str, Any]]:
        """Query for timeline view of data."""
        # Get deltas for timeline
        deltas = self._query_deltas(query, start_time, end_time)
        
        # Build timeline with specified granularity
        granularity = query.aggregate_by or "hour"
        timeline = {}
        
        for delta in deltas:
            # Determine time bucket
            if granularity == "hour":
                bucket = delta.timestamp.replace(minute=0, second=0, microsecond=0)
            elif granularity == "day":
                bucket = delta.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            elif granularity == "week":
                # Start of week (Monday)
                days_since_monday = delta.timestamp.weekday()
                week_start = delta.timestamp - timedelta(days=days_since_monday)
                bucket = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                bucket = delta.timestamp
            
            # Initialize bucket if not exists
            if bucket not in timeline:
                timeline[bucket] = {
                    'event_count': 0,
                    'change_count': 0,
                    'max_significance': 0.0,
                    'categories': set(),
                    'event_types': set()
                }
            
            # Update bucket data
            timeline[bucket]['event_count'] += len(delta.semantic_events)
            timeline[bucket]['change_count'] += len(delta.raw_delta)
            
            if delta.raw_delta:
                max_sig = max(c.significance for c in delta.raw_delta)
                timeline[bucket]['max_significance'] = max(timeline[bucket]['max_significance'], max_sig)
            
            timeline[bucket]['categories'].update(c.category for c in delta.raw_delta)
            timeline[bucket]['event_types'].update(e.event_type for e in delta.semantic_events)
        
        # Convert sets to lists for JSON serialization
        for bucket_data in timeline.values():
            bucket_data['categories'] = list(bucket_data['categories'])
            bucket_data['event_types'] = list(bucket_data['event_types'])
        
        return timeline
    
    def _filter_deltas(self, deltas: List[SystemDelta], query: TemporalQuery) -> List[SystemDelta]:
        """Apply query filters to deltas."""
        filtered = deltas
        
        # Category filter
        if query.categories:
            category_set = set(query.categories)
            filtered = [
                d for d in filtered
                if any(c.category in category_set for c in d.raw_delta)
            ]
        
        # Event type filter
        if query.event_types:
            event_type_set = set(query.event_types)
            filtered = [
                d for d in filtered
                if any(e.event_type in event_type_set for e in d.semantic_events)
            ]
        
        # Significance filters
        if query.min_significance is not None:
            filtered = [
                d for d in filtered
                if any(c.significance >= query.min_significance for c in d.raw_delta)
            ]
        
        if query.max_significance is not None:
            filtered = [
                d for d in filtered
                if any(c.significance <= query.max_significance for c in d.raw_delta)
            ]
        
        # Confidence filter
        if query.min_confidence is not None:
            filtered = [
                d for d in filtered
                if any(e.confidence >= query.min_confidence for e in d.semantic_events)
            ]
        
        # Search terms filter
        if query.search_terms:
            filtered = [d for d in filtered if self._delta_matches_search(d, query.search_terms)]
        
        # Description contains filter
        if query.description_contains:
            search_term = query.description_contains.lower()
            filtered = [
                d for d in filtered
                if any(search_term in e.description.lower() for e in d.semantic_events)
            ]
        
        return filtered
    
    def _sort_deltas(self, deltas: List[SystemDelta], sort_by: str, sort_order: str) -> List[SystemDelta]:
        """Sort deltas by specified criteria."""
        reverse = sort_order.lower() == "desc"
        
        if sort_by == "timestamp":
            return sorted(deltas, key=lambda d: d.timestamp, reverse=reverse)
        elif sort_by == "significance":
            def get_max_significance(delta):
                if not delta.raw_delta:
                    return 0.0
                return max(c.significance for c in delta.raw_delta)
            return sorted(deltas, key=get_max_significance, reverse=reverse)
        elif sort_by == "confidence":
            def get_max_confidence(delta):
                if not delta.semantic_events:
                    return 0.0
                return max(e.confidence for e in delta.semantic_events)
            return sorted(deltas, key=get_max_confidence, reverse=reverse)
        else:
            # Default to timestamp
            return sorted(deltas, key=lambda d: d.timestamp, reverse=reverse)
    
    def _extract_events_from_deltas(self, deltas: List[SystemDelta]) -> List[SystemEvent]:
        """Extract all events from deltas."""
        events = []
        for delta in deltas:
            events.extend(delta.semantic_events)
        return events
    
    def _extract_changes_from_deltas(self, deltas: List[SystemDelta]) -> List[SystemChange]:
        """Extract all changes from deltas."""
        changes = []
        for delta in deltas:
            changes.extend(delta.raw_delta)
        return changes
    
    def _should_use_recent_buffer(self, start_time: Optional[datetime], 
                                end_time: Optional[datetime]) -> bool:
        """Determine if recent buffer should be used for query."""
        if not start_time and not end_time:
            return True  # Use for all-time queries
        
        now = datetime.now()
        recent_cutoff = now - timedelta(hours=48)  # Recent buffer covers last 48 hours
        
        if end_time and end_time < recent_cutoff:
            return False  # Query is entirely in the past
        
        return True  # Query includes recent data
    
    def _should_use_daily_aggregator(self, start_time: Optional[datetime],
                                   end_time: Optional[datetime]) -> bool:
        """Determine if daily aggregator should be used for query."""
        if not start_time:
            return True  # Use for open-ended queries
        
        now = datetime.now()
        recent_cutoff = now - timedelta(hours=48)  # Beyond recent buffer range
        
        return start_time < recent_cutoff  # Query includes historical data
    
    def _get_sources_used(self, start_time: Optional[datetime],
                         end_time: Optional[datetime]) -> List[str]:
        """Determine which sources were used for query."""
        sources = []
        
        if self._should_use_recent_buffer(start_time, end_time):
            sources.append("recent_buffer")
        
        if self._should_use_daily_aggregator(start_time, end_time):
            sources.append("daily_aggregator")
        
        sources.append("search_index")  # Always used for indexing
        
        return sources
    
    def _count_results(self, result: QueryResult, query_type: QueryType) -> int:
        """Count total results based on query type."""
        if query_type == QueryType.DELTAS:
            return len(result.deltas)
        elif query_type == QueryType.EVENTS:
            return len(result.events)
        elif query_type == QueryType.CHANGES:
            return len(result.changes)
        elif query_type == QueryType.SUMMARIES:
            return len(result.summaries)
        elif query_type == QueryType.PATTERNS:
            return len(result.patterns)
        elif query_type == QueryType.TIMELINE:
            return len(result.timeline)
        elif query_type == QueryType.STATISTICS:
            return 1  # Statistics is a single result
        
        return 0
    
    def _delta_matches_search(self, delta: SystemDelta, search_terms: List[str]) -> bool:
        """Check if delta matches search terms."""
        search_content = []
        
        # Collect searchable content
        for change in delta.raw_delta:
            search_content.extend([
                change.category,
                change.entity_id,
                str(change.change_type.value)
            ])
        
        for event in delta.semantic_events:
            search_content.extend([
                event.event_type,
                event.entity,
                event.description
            ])
        
        # Check if any search term matches content
        content_str = " ".join(search_content).lower()
        return any(term.lower() in content_str for term in search_terms)
    
    def _pattern_matches_search(self, pattern: SystemPattern, search_terms: List[str]) -> bool:
        """Check if pattern matches search terms."""
        search_content = [
            pattern.name,
            pattern.description,
            pattern.pattern_type
        ]
        
        content_str = " ".join(search_content).lower()
        return any(term.lower() in content_str for term in search_terms)