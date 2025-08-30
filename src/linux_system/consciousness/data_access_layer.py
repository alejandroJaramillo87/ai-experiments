"""
Data Access Layer - Repository Pattern for System Intelligence
==============================================================

Repository pattern implementation that abstracts data access between:
- Current state data (real-time system monitoring)
- Temporal data (historical system evolution)  
- Conversational data (chat history and context)

Key principles:
- Clean abstraction between data sources
- Unified interface regardless of data location
- Support for both real-time and historical queries
- Optimized access patterns for different use cases
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Import temporal intelligence components
from ..temporal.types import SystemDelta, SystemEvent, SystemChange, TimeRange
from ..temporal.storage.temporal_storage import TemporalStorage

# Import AI workstation components (will be integrated when ready)
# from ..ai_workstation import AIWorkstationController

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of data queries supported."""
    CURRENT_STATE = "current_state"
    TEMPORAL_RANGE = "temporal_range"
    TEMPORAL_EVENTS = "temporal_events"
    TEMPORAL_PATTERNS = "temporal_patterns"
    CONVERSATIONAL_CONTEXT = "conversational_context"


@dataclass
class QueryFilters:
    """Filters for data queries."""
    metric_types: Optional[List[str]] = None
    significance_threshold: Optional[float] = None
    event_types: Optional[List[str]] = None
    include_metadata: bool = True
    include_predictions: bool = False
    max_results: Optional[int] = None


@dataclass
class DataAccessResult:
    """Result from data access operations."""
    data: Any
    query_type: QueryType
    timestamp: datetime
    metadata: Dict[str, Any]
    status: str = "success"
    error: Optional[str] = None


class BaseRepository(ABC):
    """Abstract base repository for data access."""
    
    @abstractmethod
    async def query(self, filters: QueryFilters, **kwargs) -> DataAccessResult:
        """Execute query with filters."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check repository health."""
        pass


class CurrentStateRepository(BaseRepository):
    """
    Repository for current state data from AI workstation systems.
    
    Provides access to real-time system metrics, container states,
    GPU utilization, and all current system intelligence.
    """
    
    def __init__(self, ai_workstation_controller=None, system_collector=None):
        """Initialize with AI workstation dependencies."""
        self.ai_workstation_controller = ai_workstation_controller
        self.system_collector = system_collector
        
    async def query(self, filters: QueryFilters, **kwargs) -> DataAccessResult:
        """Query current state data."""
        try:
            metric_types = filters.metric_types or ["all"]
            current_data = {}
            
            # Collect current state data from different sources
            for metric_type in metric_types:
                data = await self._get_current_metric_data(metric_type)
                if data:
                    current_data[metric_type] = data
            
            return DataAccessResult(
                data=current_data,
                query_type=QueryType.CURRENT_STATE,
                timestamp=datetime.now(),
                metadata={
                    "metric_types": metric_types,
                    "collection_method": "real_time",
                    "data_sources": list(current_data.keys())
                }
            )
            
        except Exception as e:
            logger.error(f"Error querying current state: {e}")
            return DataAccessResult(
                data={},
                query_type=QueryType.CURRENT_STATE,
                timestamp=datetime.now(),
                metadata={},
                status="error",
                error=str(e)
            )
    
    async def _get_current_metric_data(self, metric_type: str) -> Optional[Dict[str, Any]]:
        """Get current data for specific metric type."""
        try:
            if metric_type == "all" or metric_type == "system":
                return await self._get_system_collector_data()
            elif metric_type == "gpu" or metric_type == "nvidia_gpu":
                return await self._get_gpu_data()
            elif metric_type == "containers" or metric_type == "ai_containers":
                return await self._get_container_data()
            elif metric_type == "processes":
                return await self._get_process_data()
            elif metric_type == "memory":
                return await self._get_memory_data()
            elif metric_type == "storage":
                return await self._get_storage_data()
            elif metric_type == "network":
                return await self._get_network_data()
            elif metric_type == "thermal":
                return await self._get_thermal_data()
            else:
                logger.warning(f"Unknown metric type: {metric_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting current data for {metric_type}: {e}")
            return None
    
    async def _get_system_collector_data(self) -> Dict[str, Any]:
        """Get data from SystemCollector."""
        if not self.system_collector:
            return {"status": "system_collector_not_available"}
        
        try:
            # Placeholder - would integrate with actual SystemCollector
            return {
                "cpu": {"usage": 45.2, "cores": 16, "temperature": 65.0},
                "memory": {"total": 137438953472, "used": 54975581388},
                "timestamp": datetime.now().isoformat(),
                "collector": "system"
            }
        except Exception as e:
            logger.error(f"Error collecting system data: {e}")
            return {"error": str(e)}
    
    async def _get_gpu_data(self) -> Dict[str, Any]:
        """Get current GPU data."""
        try:
            # Placeholder - would integrate with RTX 5090 Blackwell detector
            return {
                "gpu_id": 0,
                "name": "RTX 5090",
                "temperature": 75.0,
                "utilization": {"gpu": 85.0, "memory": 70.0},
                "memory": {"total": 32768, "used": 24576, "free": 8192},
                "power_usage": 450.0,
                "thermal_throttling": False,
                "processes": [],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error collecting GPU data: {e}")
            return {"error": str(e)}
    
    async def _get_container_data(self) -> Dict[str, Any]:
        """Get current container data."""
        try:
            # Placeholder - would integrate with container consciousness
            return {
                "containers": [
                    {
                        "id": "llama-gpu",
                        "status": "running",
                        "cpu_usage": 25.5,
                        "memory_usage": 8589934592,
                        "gpu_access": True,
                        "health_status": "healthy"
                    },
                    {
                        "id": "vllm-gpu", 
                        "status": "running",
                        "cpu_usage": 15.2,
                        "memory_usage": 6442450944,
                        "gpu_access": True,
                        "health_status": "healthy"
                    }
                ],
                "total_containers": 5,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error collecting container data: {e}")
            return {"error": str(e)}
    
    async def _get_process_data(self) -> Dict[str, Any]:
        """Get current process data."""
        try:
            return {
                "active_processes": 150,
                "system_load": {"1min": 2.5, "5min": 2.1, "15min": 1.8},
                "top_cpu_processes": [],
                "top_memory_processes": [],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error collecting process data: {e}")
            return {"error": str(e)}
    
    async def _get_memory_data(self) -> Dict[str, Any]:
        """Get current memory data."""
        try:
            return {
                "total": 137438953472,  # 128GB
                "used": 54975581388,
                "available": 82463372084,
                "swap": {"total": 0, "used": 0},
                "pressure": "none",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error collecting memory data: {e}")
            return {"error": str(e)}
    
    async def _get_storage_data(self) -> Dict[str, Any]:
        """Get current storage data."""
        try:
            return {
                "devices": [
                    {"name": "/dev/nvme0n1", "usage": 45.2, "health": "good"},
                    {"name": "/dev/nvme1n1", "usage": 23.1, "health": "good"}
                ],
                "total_usage": 34.1,
                "io_stats": {"read_iops": 1200, "write_iops": 800},
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error collecting storage data: {e}")
            return {"error": str(e)}
    
    async def _get_network_data(self) -> Dict[str, Any]:
        """Get current network data."""
        try:
            return {
                "interfaces": [
                    {"name": "eth0", "status": "up", "speed": 1000},
                    {"name": "docker0", "status": "up", "speed": 1000}
                ],
                "connections": {"active": 45, "listening": 12},
                "traffic": {"rx_bytes": 1024000, "tx_bytes": 2048000},
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error collecting network data: {e}")
            return {"error": str(e)}
    
    async def _get_thermal_data(self) -> Dict[str, Any]:
        """Get current thermal data."""
        try:
            return {
                "cpu_temperature": 65.0,
                "gpu_temperature": 75.0,
                "ambient_temperature": 24.0,
                "fan_speeds": [1200, 1150, 1300, 1250],
                "thermal_throttling": False,
                "cooling_efficiency": 0.85,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error collecting thermal data: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> bool:
        """Check current state repository health."""
        try:
            # Test basic data collection
            test_result = await self._get_system_collector_data()
            return test_result is not None and "error" not in test_result
        except:
            return False


class TemporalRepository(BaseRepository):
    """
    Repository for temporal/historical data from TemporalStorage.
    
    Provides access to historical system evolution, detected patterns,
    events, and temporal correlations.
    """
    
    def __init__(self, temporal_storage: TemporalStorage):
        """Initialize with temporal storage dependency."""
        self.temporal_storage = temporal_storage
        
    async def query(self, filters: QueryFilters, query_type: QueryType = QueryType.TEMPORAL_RANGE, **kwargs) -> DataAccessResult:
        """Query temporal data based on type and filters."""
        try:
            if query_type == QueryType.TEMPORAL_RANGE:
                return await self._query_temporal_range(filters, **kwargs)
            elif query_type == QueryType.TEMPORAL_EVENTS:
                return await self._query_temporal_events(filters, **kwargs)
            elif query_type == QueryType.TEMPORAL_PATTERNS:
                return await self._query_temporal_patterns(filters, **kwargs)
            else:
                raise ValueError(f"Unsupported temporal query type: {query_type}")
                
        except Exception as e:
            logger.error(f"Error querying temporal data: {e}")
            return DataAccessResult(
                data=[],
                query_type=query_type,
                timestamp=datetime.now(),
                metadata={},
                status="error",
                error=str(e)
            )
    
    async def _query_temporal_range(self, filters: QueryFilters, time_range: TimeRange = None, **kwargs) -> DataAccessResult:
        """Query temporal data for a specific time range."""
        try:
            # Default to last 4 hours if no time range specified
            if not time_range:
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=4)
                time_range = TimeRange(start_time=start_time, end_time=end_time)
            
            # Query temporal storage for deltas in range
            deltas = await self.temporal_storage.query_range(
                start_time=time_range.start_time,
                end_time=time_range.end_time,
                metric_types=filters.metric_types,
                max_results=filters.max_results
            )
            
            return DataAccessResult(
                data=deltas,
                query_type=QueryType.TEMPORAL_RANGE,
                timestamp=datetime.now(),
                metadata={
                    "time_range": {
                        "start": time_range.start_time.isoformat(),
                        "end": time_range.end_time.isoformat()
                    },
                    "metric_types": filters.metric_types,
                    "result_count": len(deltas) if deltas else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Error querying temporal range: {e}")
            raise
    
    async def _query_temporal_events(self, filters: QueryFilters, **kwargs) -> DataAccessResult:
        """Query temporal events with filters."""
        try:
            events = await self.temporal_storage.query_events(
                event_types=filters.event_types,
                significance_threshold=filters.significance_threshold,
                max_results=filters.max_results
            )
            
            return DataAccessResult(
                data=events,
                query_type=QueryType.TEMPORAL_EVENTS,
                timestamp=datetime.now(),
                metadata={
                    "event_types": filters.event_types,
                    "significance_threshold": filters.significance_threshold,
                    "result_count": len(events) if events else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Error querying temporal events: {e}")
            raise
    
    async def _query_temporal_patterns(self, filters: QueryFilters, **kwargs) -> DataAccessResult:
        """Query detected temporal patterns."""
        try:
            patterns = await self.temporal_storage.query_patterns(
                pattern_types=filters.metric_types,
                max_results=filters.max_results
            )
            
            return DataAccessResult(
                data=patterns,
                query_type=QueryType.TEMPORAL_PATTERNS,
                timestamp=datetime.now(),
                metadata={
                    "pattern_types": filters.metric_types,
                    "result_count": len(patterns) if patterns else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Error querying temporal patterns: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check temporal repository health."""
        try:
            return await self.temporal_storage.health_check()
        except:
            return False


class ConversationRepository(BaseRepository):
    """
    Repository for conversational data and context.
    
    Manages chat history, conversation context, and user interaction patterns.
    """
    
    def __init__(self):
        """Initialize conversation repository."""
        self.conversation_history: Dict[str, List[Dict]] = {}
        self.session_contexts: Dict[str, Dict] = {}
        
    async def query(self, filters: QueryFilters, session_id: str = None, **kwargs) -> DataAccessResult:
        """Query conversation data."""
        try:
            if session_id:
                # Get specific session data
                history = self.conversation_history.get(session_id, [])
                context = self.session_contexts.get(session_id, {})
                
                data = {
                    "history": history,
                    "context": context,
                    "session_id": session_id
                }
            else:
                # Get all conversation data
                data = {
                    "all_sessions": list(self.conversation_history.keys()),
                    "total_conversations": sum(len(hist) for hist in self.conversation_history.values())
                }
            
            return DataAccessResult(
                data=data,
                query_type=QueryType.CONVERSATIONAL_CONTEXT,
                timestamp=datetime.now(),
                metadata={
                    "session_id": session_id,
                    "has_history": bool(session_id and session_id in self.conversation_history)
                }
            )
            
        except Exception as e:
            logger.error(f"Error querying conversation data: {e}")
            return DataAccessResult(
                data={},
                query_type=QueryType.CONVERSATIONAL_CONTEXT,
                timestamp=datetime.now(),
                metadata={},
                status="error",
                error=str(e)
            )
    
    async def store_conversation(self, session_id: str, query: str, response: str, metadata: Dict = None):
        """Store conversation exchange."""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response,
            "metadata": metadata or {}
        }
        
        self.conversation_history[session_id].append(conversation_entry)
        
        # Limit history size per session
        max_history = 100
        if len(self.conversation_history[session_id]) > max_history:
            self.conversation_history[session_id] = self.conversation_history[session_id][-max_history:]
    
    async def update_session_context(self, session_id: str, context: Dict):
        """Update session context."""
        self.session_contexts[session_id] = context
    
    async def health_check(self) -> bool:
        """Check conversation repository health."""
        return True  # Simple in-memory implementation


class DataAccessLayer:
    """
    Unified data access layer using repository pattern.
    
    Provides clean abstraction over current state, temporal, and conversational data
    with unified interface for all system consciousness queries.
    """
    
    def __init__(self, 
                 current_repo: Optional[CurrentStateRepository] = None,
                 temporal_repo: Optional[TemporalRepository] = None,
                 conversation_repo: Optional[ConversationRepository] = None):
        """Initialize with repository dependencies."""
        self.current = current_repo or CurrentStateRepository()
        self.temporal = temporal_repo
        self.conversation = conversation_repo or ConversationRepository()
        
    async def get_current_metrics(self, metric_types: List[str], filters: QueryFilters = None) -> DataAccessResult:
        """Get current system metrics."""
        filters = filters or QueryFilters(metric_types=metric_types)
        return await self.current.query(filters)
    
    async def get_temporal_data(self, time_range: TimeRange, metric_types: List[str] = None, filters: QueryFilters = None) -> DataAccessResult:
        """Get temporal data for time range."""
        if not self.temporal:
            raise RuntimeError("Temporal repository not configured")
        
        filters = filters or QueryFilters(metric_types=metric_types)
        return await self.temporal.query(filters, QueryType.TEMPORAL_RANGE, time_range=time_range)
    
    async def get_temporal_events(self, event_types: List[str] = None, significance_threshold: float = None) -> DataAccessResult:
        """Get temporal events."""
        if not self.temporal:
            raise RuntimeError("Temporal repository not configured")
        
        filters = QueryFilters(
            event_types=event_types,
            significance_threshold=significance_threshold
        )
        return await self.temporal.query(filters, QueryType.TEMPORAL_EVENTS)
    
    async def get_temporal_patterns(self, pattern_types: List[str] = None) -> DataAccessResult:
        """Get detected temporal patterns."""
        if not self.temporal:
            raise RuntimeError("Temporal repository not configured")
        
        filters = QueryFilters(metric_types=pattern_types)
        return await self.temporal.query(filters, QueryType.TEMPORAL_PATTERNS)
    
    async def get_conversation_context(self, session_id: str = None) -> DataAccessResult:
        """Get conversation context."""
        return await self.conversation.query(QueryFilters(), session_id=session_id)
    
    async def store_conversation(self, session_id: str, query: str, response: str, metadata: Dict = None):
        """Store conversation exchange."""
        await self.conversation.store_conversation(session_id, query, response, metadata)
    
    async def update_session_context(self, session_id: str, context: Dict):
        """Update session context."""
        await self.conversation.update_session_context(session_id, context)
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all repositories."""
        return {
            "current_state": await self.current.health_check(),
            "temporal": await self.temporal.health_check() if self.temporal else False,
            "conversation": await self.conversation.health_check()
        }
    
    def inject_temporal_repository(self, temporal_repo: TemporalRepository):
        """Inject temporal repository dependency."""
        self.temporal = temporal_repo
        logger.info("Temporal repository injected into DataAccessLayer")
    
    def inject_current_state_components(self, ai_workstation_controller=None, system_collector=None):
        """Inject current state dependencies."""
        self.current.ai_workstation_controller = ai_workstation_controller
        self.current.system_collector = system_collector
        logger.info("Current state components injected into DataAccessLayer")