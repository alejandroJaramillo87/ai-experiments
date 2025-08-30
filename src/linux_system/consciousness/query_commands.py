"""
Query Commands - Command Pattern for System Intelligence
=======================================================

Command pattern implementation for unified query processing across all
system consciousness capabilities. Provides consistent interface for:

- Current state queries
- Temporal/historical queries  
- Conversational queries
- Pattern analysis queries

Key principles:
- Command pattern for consistent query execution
- Async support for scalable processing
- Validation and optimization built-in
- Extensible for new query types
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .data_access_layer import DataAccessLayer, DataAccessResult, QueryFilters, QueryType
from ..temporal.types import TimeRange

logger = logging.getLogger(__name__)


class QueryPriority(Enum):
    """Priority levels for query execution."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class QueryContext:
    """Context information for query execution."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    priority: QueryPriority = QueryPriority.NORMAL
    timeout_seconds: int = 30
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QueryResult:
    """Result from query command execution."""
    data: Any
    success: bool
    query_type: str
    execution_time_ms: float
    timestamp: datetime
    context: QueryContext
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class QueryValidationResult:
    """Result from query validation."""
    def __init__(self, valid: bool, error: Optional[str] = None, warnings: List[str] = None):
        self.valid = valid
        self.error = error
        self.warnings = warnings or []


class QueryCommand(ABC):
    """
    Abstract base class for all query commands.
    
    Implements command pattern for consistent query processing
    across all system consciousness capabilities.
    """
    
    def __init__(self, context: QueryContext):
        """Initialize query command with context."""
        self.context = context
        self.start_time: Optional[datetime] = None
        
    @abstractmethod
    async def execute(self, data_access: DataAccessLayer) -> QueryResult:
        """Execute the query command."""
        pass
    
    @abstractmethod
    def validate(self) -> QueryValidationResult:
        """Validate query parameters."""
        pass
    
    def get_query_type(self) -> str:
        """Get query type identifier."""
        return self.__class__.__name__.replace('Query', '').lower()
    
    def _start_timing(self):
        """Start execution timing."""
        self.start_time = datetime.now()
    
    def _get_execution_time_ms(self) -> float:
        """Get execution time in milliseconds."""
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds() * 1000
        return 0.0
    
    def _create_result(self, data: Any, success: bool = True, error: Optional[str] = None) -> QueryResult:
        """Create standardized query result."""
        return QueryResult(
            data=data,
            success=success,
            query_type=self.get_query_type(),
            execution_time_ms=self._get_execution_time_ms(),
            timestamp=datetime.now(),
            context=self.context,
            error=error,
            metadata={}
        )


class CurrentStateQuery(QueryCommand):
    """
    Command for querying current system state.
    
    Retrieves real-time metrics from active system monitoring.
    """
    
    def __init__(self, 
                 metric_types: List[str],
                 filters: Optional[Dict[str, Any]] = None,
                 context: Optional[QueryContext] = None):
        """Initialize current state query."""
        super().__init__(context or QueryContext())
        self.metric_types = metric_types
        self.filters = filters or {}
        
    async def execute(self, data_access: DataAccessLayer) -> QueryResult:
        """Execute current state query."""
        self._start_timing()
        
        try:
            # Create query filters
            query_filters = QueryFilters(
                metric_types=self.metric_types,
                include_metadata=self.filters.get('include_metadata', True),
                include_predictions=self.filters.get('include_predictions', False),
                max_results=self.filters.get('max_results')
            )
            
            # Execute query through data access layer
            result = await data_access.get_current_metrics(
                metric_types=self.metric_types,
                filters=query_filters
            )
            
            if result.status == "success":
                return self._create_result(result.data, success=True)
            else:
                return self._create_result({}, success=False, error=result.error)
                
        except Exception as e:
            logger.error(f"Error executing current state query: {e}")
            return self._create_result({}, success=False, error=str(e))
    
    def validate(self) -> QueryValidationResult:
        """Validate current state query parameters."""
        if not self.metric_types:
            return QueryValidationResult(False, "metric_types cannot be empty")
        
        # Validate metric types
        valid_metrics = {
            "system", "gpu", "containers", "processes", "memory", 
            "storage", "network", "thermal", "all"
        }
        
        invalid_metrics = set(self.metric_types) - valid_metrics
        if invalid_metrics:
            return QueryValidationResult(
                False, 
                f"Invalid metric types: {list(invalid_metrics)}"
            )
        
        return QueryValidationResult(True)


class TemporalQuery(QueryCommand):
    """
    Command for querying temporal/historical data.
    
    Retrieves historical system evolution data with time range support.
    """
    
    def __init__(self,
                 time_range: Union[TimeRange, str],
                 metric_types: Optional[List[str]] = None,
                 aggregation: Optional[str] = None,
                 filters: Optional[Dict[str, Any]] = None,
                 context: Optional[QueryContext] = None):
        """Initialize temporal query."""
        super().__init__(context or QueryContext())
        self.time_range = self._parse_time_range(time_range)
        self.metric_types = metric_types
        self.aggregation = aggregation
        self.filters = filters or {}
    
    def _parse_time_range(self, time_range: Union[TimeRange, str]) -> TimeRange:
        """Parse time range from string or TimeRange object."""
        if isinstance(time_range, TimeRange):
            return time_range
        
        # Parse string time ranges like "4h", "1d", "1w"
        now = datetime.now()
        
        if time_range == "1h":
            start_time = now - timedelta(hours=1)
        elif time_range == "4h":
            start_time = now - timedelta(hours=4)
        elif time_range == "1d":
            start_time = now - timedelta(days=1)
        elif time_range == "1w":
            start_time = now - timedelta(weeks=1)
        elif time_range == "1m":
            start_time = now - timedelta(days=30)
        else:
            # Default to 4 hours
            start_time = now - timedelta(hours=4)
        
        return TimeRange(start_time=start_time, end_time=now)
    
    async def execute(self, data_access: DataAccessLayer) -> QueryResult:
        """Execute temporal query."""
        self._start_timing()
        
        try:
            # Create query filters
            query_filters = QueryFilters(
                metric_types=self.metric_types,
                include_metadata=self.filters.get('include_metadata', True),
                max_results=self.filters.get('max_results')
            )
            
            # Execute temporal query
            result = await data_access.get_temporal_data(
                time_range=self.time_range,
                metric_types=self.metric_types,
                filters=query_filters
            )
            
            if result.status == "success":
                return self._create_result(result.data, success=True)
            else:
                return self._create_result([], success=False, error=result.error)
                
        except Exception as e:
            logger.error(f"Error executing temporal query: {e}")
            return self._create_result([], success=False, error=str(e))
    
    def validate(self) -> QueryValidationResult:
        """Validate temporal query parameters."""
        if not self.time_range:
            return QueryValidationResult(False, "time_range is required")
        
        # Check time range validity
        if self.time_range.start_time >= self.time_range.end_time:
            return QueryValidationResult(False, "start_time must be before end_time")
        
        # Check reasonable time range (not too far in past/future)
        max_range = timedelta(days=365)  # 1 year max
        if self.time_range.end_time - self.time_range.start_time > max_range:
            return QueryValidationResult(
                False, 
                f"Time range too large (max: {max_range.days} days)"
            )
        
        return QueryValidationResult(True)


class EventQuery(QueryCommand):
    """
    Command for querying temporal events.
    
    Retrieves system events with filtering and significance thresholds.
    """
    
    def __init__(self,
                 event_types: Optional[List[str]] = None,
                 significance_threshold: Optional[float] = None,
                 time_range: Optional[Union[TimeRange, str]] = None,
                 filters: Optional[Dict[str, Any]] = None,
                 context: Optional[QueryContext] = None):
        """Initialize event query."""
        super().__init__(context or QueryContext())
        self.event_types = event_types
        self.significance_threshold = significance_threshold
        self.time_range = self._parse_time_range(time_range) if time_range else None
        self.filters = filters or {}
    
    def _parse_time_range(self, time_range: Union[TimeRange, str]) -> TimeRange:
        """Parse time range (same logic as TemporalQuery)."""
        if isinstance(time_range, TimeRange):
            return time_range
        
        now = datetime.now()
        
        if time_range == "1h":
            start_time = now - timedelta(hours=1)
        elif time_range == "4h":
            start_time = now - timedelta(hours=4)
        elif time_range == "1d":
            start_time = now - timedelta(days=1)
        elif time_range == "1w":
            start_time = now - timedelta(weeks=1)
        else:
            start_time = now - timedelta(hours=4)  # Default
        
        return TimeRange(start_time=start_time, end_time=now)
    
    async def execute(self, data_access: DataAccessLayer) -> QueryResult:
        """Execute event query."""
        self._start_timing()
        
        try:
            # Execute event query
            result = await data_access.get_temporal_events(
                event_types=self.event_types,
                significance_threshold=self.significance_threshold
            )
            
            # Filter by time range if specified
            if self.time_range and result.status == "success":
                filtered_events = self._filter_events_by_time(result.data)
                result.data = filtered_events
            
            if result.status == "success":
                return self._create_result(result.data, success=True)
            else:
                return self._create_result([], success=False, error=result.error)
                
        except Exception as e:
            logger.error(f"Error executing event query: {e}")
            return self._create_result([], success=False, error=str(e))
    
    def _filter_events_by_time(self, events: List[Any]) -> List[Any]:
        """Filter events by time range."""
        if not self.time_range or not events:
            return events
        
        filtered = []
        for event in events:
            try:
                # Assume event has timestamp attribute
                event_time = getattr(event, 'timestamp', None)
                if event_time:
                    if isinstance(event_time, str):
                        event_time = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
                    
                    if self.time_range.start_time <= event_time <= self.time_range.end_time:
                        filtered.append(event)
            except Exception as e:
                logger.warning(f"Error filtering event by time: {e}")
                # Include event if we can't parse timestamp
                filtered.append(event)
        
        return filtered
    
    def validate(self) -> QueryValidationResult:
        """Validate event query parameters."""
        if self.significance_threshold is not None:
            if not (0.0 <= self.significance_threshold <= 1.0):
                return QueryValidationResult(
                    False, 
                    "significance_threshold must be between 0.0 and 1.0"
                )
        
        if self.time_range:
            # Validate time range
            if self.time_range.start_time >= self.time_range.end_time:
                return QueryValidationResult(False, "start_time must be before end_time")
        
        return QueryValidationResult(True)


class ConversationalQuery(QueryCommand):
    """
    Command for natural language queries.
    
    Processes natural language input and routes to appropriate
    system intelligence capabilities.
    """
    
    def __init__(self,
                 natural_language: str,
                 session_id: Optional[str] = None,
                 conversation_context: Optional[Dict[str, Any]] = None,
                 context: Optional[QueryContext] = None):
        """Initialize conversational query."""
        super().__init__(context or QueryContext())
        self.natural_language = natural_language
        self.session_id = session_id
        self.conversation_context = conversation_context or {}
    
    async def execute(self, data_access: DataAccessLayer) -> QueryResult:
        """Execute conversational query."""
        self._start_timing()
        
        try:
            # For now, return placeholder - will be implemented with ConversationalAI
            result_data = {
                "message": "Conversational processing not yet implemented",
                "query": self.natural_language,
                "session_id": self.session_id,
                "status": "placeholder"
            }
            
            return self._create_result(result_data, success=True)
            
        except Exception as e:
            logger.error(f"Error executing conversational query: {e}")
            return self._create_result({}, success=False, error=str(e))
    
    def validate(self) -> QueryValidationResult:
        """Validate conversational query parameters."""
        if not self.natural_language or not self.natural_language.strip():
            return QueryValidationResult(False, "natural_language query cannot be empty")
        
        if len(self.natural_language) > 5000:  # Reasonable limit
            return QueryValidationResult(False, "Query too long (max 5000 characters)")
        
        return QueryValidationResult(True)


class PatternQuery(QueryCommand):
    """
    Command for querying detected patterns.
    
    Retrieves learned behavioral patterns and correlations.
    """
    
    def __init__(self,
                 pattern_types: Optional[List[str]] = None,
                 confidence_threshold: Optional[float] = None,
                 filters: Optional[Dict[str, Any]] = None,
                 context: Optional[QueryContext] = None):
        """Initialize pattern query."""
        super().__init__(context or QueryContext())
        self.pattern_types = pattern_types
        self.confidence_threshold = confidence_threshold
        self.filters = filters or {}
    
    async def execute(self, data_access: DataAccessLayer) -> QueryResult:
        """Execute pattern query."""
        self._start_timing()
        
        try:
            # Execute pattern query
            result = await data_access.get_temporal_patterns(
                pattern_types=self.pattern_types
            )
            
            # Apply confidence filtering if specified
            if self.confidence_threshold and result.status == "success":
                filtered_patterns = self._filter_by_confidence(result.data)
                result.data = filtered_patterns
            
            if result.status == "success":
                return self._create_result(result.data, success=True)
            else:
                return self._create_result([], success=False, error=result.error)
                
        except Exception as e:
            logger.error(f"Error executing pattern query: {e}")
            return self._create_result([], success=False, error=str(e))
    
    def _filter_by_confidence(self, patterns: List[Any]) -> List[Any]:
        """Filter patterns by confidence threshold."""
        if not self.confidence_threshold or not patterns:
            return patterns
        
        filtered = []
        for pattern in patterns:
            try:
                confidence = getattr(pattern, 'confidence', 1.0)
                if confidence >= self.confidence_threshold:
                    filtered.append(pattern)
            except:
                # Include pattern if we can't parse confidence
                filtered.append(pattern)
        
        return filtered
    
    def validate(self) -> QueryValidationResult:
        """Validate pattern query parameters."""
        if self.confidence_threshold is not None:
            if not (0.0 <= self.confidence_threshold <= 1.0):
                return QueryValidationResult(
                    False,
                    "confidence_threshold must be between 0.0 and 1.0"
                )
        
        return QueryValidationResult(True)


# Query factory for easy creation
class QueryFactory:
    """Factory for creating query commands."""
    
    @staticmethod
    def create_current_state_query(metric_types: List[str], 
                                 filters: Optional[Dict] = None,
                                 context: Optional[QueryContext] = None) -> CurrentStateQuery:
        """Create current state query."""
        return CurrentStateQuery(metric_types, filters, context)
    
    @staticmethod
    def create_temporal_query(time_range: Union[TimeRange, str],
                            metric_types: Optional[List[str]] = None,
                            aggregation: Optional[str] = None,
                            filters: Optional[Dict] = None,
                            context: Optional[QueryContext] = None) -> TemporalQuery:
        """Create temporal query."""
        return TemporalQuery(time_range, metric_types, aggregation, filters, context)
    
    @staticmethod
    def create_event_query(event_types: Optional[List[str]] = None,
                         significance_threshold: Optional[float] = None,
                         time_range: Optional[Union[TimeRange, str]] = None,
                         filters: Optional[Dict] = None,
                         context: Optional[QueryContext] = None) -> EventQuery:
        """Create event query."""
        return EventQuery(event_types, significance_threshold, time_range, filters, context)
    
    @staticmethod
    def create_conversational_query(natural_language: str,
                                  session_id: Optional[str] = None,
                                  conversation_context: Optional[Dict] = None,
                                  context: Optional[QueryContext] = None) -> ConversationalQuery:
        """Create conversational query."""
        return ConversationalQuery(natural_language, session_id, conversation_context, context)
    
    @staticmethod
    def create_pattern_query(pattern_types: Optional[List[str]] = None,
                           confidence_threshold: Optional[float] = None,
                           filters: Optional[Dict] = None,
                           context: Optional[QueryContext] = None) -> PatternQuery:
        """Create pattern query."""
        return PatternQuery(pattern_types, confidence_threshold, filters, context)