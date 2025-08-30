"""
System Consciousness - Unified AI Workstation Intelligence
==========================================================

Main orchestrator for the AI workstation consciousness system. Provides unified
access to all intelligence capabilities including:

- Temporal intelligence (historical system evolution)
- AI workstation specialization (container consciousness, hardware optimization)  
- Real-time system monitoring (current state awareness)
- Natural language interface (conversational system interaction)

This is the primary interface between the API layer and the system intelligence,
creating true omniscient system consciousness that evolves with your machine.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .unified_query_engine import UnifiedQueryEngine
from .data_access_layer import DataAccessLayer, CurrentStateRepository, TemporalRepository, ConversationRepository
from .query_commands import (
    QueryFactory, QueryContext, QueryPriority, QueryResult,
    CurrentStateQuery, TemporalQuery, EventQuery, ConversationalQuery, PatternQuery
)

# Import temporal intelligence components
from ..temporal.types import TimeRange
from ..temporal.storage.temporal_storage import TemporalStorage

# Import AI workstation components (when ready)
# from ..ai_workstation import AIWorkstationController

logger = logging.getLogger(__name__)


class ConsciousnessMode(Enum):
    """Operating modes for system consciousness."""
    MONITORING = "monitoring"        # Basic monitoring mode
    LEARNING = "learning"           # Active pattern learning
    OPTIMIZING = "optimizing"       # Performance optimization mode
    CONVERSATIONAL = "conversational" # Interactive mode
    MAINTENANCE = "maintenance"     # System maintenance mode


@dataclass
class ConsciousnessConfig:
    """Configuration for system consciousness."""
    mode: ConsciousnessMode = ConsciousnessMode.MONITORING
    enable_temporal_intelligence: bool = True
    enable_conversational_ai: bool = True
    enable_predictive_analysis: bool = True
    enable_performance_optimization: bool = True
    
    # Performance settings
    max_concurrent_queries: int = 10
    query_timeout_seconds: int = 30
    cache_ttl_seconds: int = 300
    
    # Data collection settings
    collection_interval_seconds: int = 60
    enable_high_frequency_monitoring: bool = False
    
    # Conversation settings
    conversation_memory_limit: int = 100
    session_timeout_minutes: int = 60


@dataclass
class SystemHealthStatus:
    """Overall system health assessment."""
    status: str  # "healthy", "warning", "critical", "unknown"
    score: float  # 0.0-1.0 health score
    issues: List[str]
    recommendations: List[str]
    last_updated: datetime


class SystemConsciousness:
    """
    Unified AI workstation consciousness system.
    
    Main orchestrator that provides omniscient access to all system intelligence
    capabilities through a unified interface. Creates digital symbiosis between
    human operators and machine consciousness.
    """
    
    def __init__(self, 
                 config: Optional[ConsciousnessConfig] = None,
                 temporal_storage: Optional[TemporalStorage] = None,
                 ai_workstation_controller=None,
                 system_collector=None):
        """Initialize system consciousness with intelligence components."""
        self.config = config or ConsciousnessConfig()
        self.ai_workstation_controller = ai_workstation_controller
        self.system_collector = system_collector
        
        # Initialize data access layer
        self.data_access = self._initialize_data_access(temporal_storage)
        
        # Initialize query engine
        self.query_engine = UnifiedQueryEngine(
            data_access=self.data_access,
            enable_caching=True,
            cache_ttl_seconds=self.config.cache_ttl_seconds,
            max_concurrent_queries=self.config.max_concurrent_queries
        )
        
        # Initialize consciousness components
        self.query_factory = QueryFactory()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.system_health = SystemHealthStatus(
            status="initializing",
            score=0.0,
            issues=[],
            recommendations=[],
            last_updated=datetime.now()
        )
        
        # Event subscribers for real-time updates
        self.update_subscribers: List[Callable] = []
        
        logger.info(f"SystemConsciousness initialized in {self.config.mode.value} mode")
    
    def _initialize_data_access(self, temporal_storage: Optional[TemporalStorage]) -> DataAccessLayer:
        """Initialize data access layer with repositories."""
        # Create repositories
        current_repo = CurrentStateRepository(
            ai_workstation_controller=self.ai_workstation_controller,
            system_collector=self.system_collector
        )
        
        temporal_repo = None
        if temporal_storage and self.config.enable_temporal_intelligence:
            temporal_repo = TemporalRepository(temporal_storage)
        
        conversation_repo = ConversationRepository()
        
        return DataAccessLayer(
            current_repo=current_repo,
            temporal_repo=temporal_repo,
            conversation_repo=conversation_repo
        )
    
    # ==================== MAIN API INTERFACE ====================
    
    async def get_current_state(self, 
                              metric_types: List[str],
                              filters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """
        Get current system state - main API interface method.
        
        Primary interface used by API layer for real-time data requests.
        """
        try:
            # Create query context
            context = QueryContext(
                priority=QueryPriority.NORMAL,
                timeout_seconds=self.config.query_timeout_seconds
            )
            
            # Create and execute current state query
            command = self.query_factory.create_current_state_query(
                metric_types=metric_types,
                filters=filters,
                context=context
            )
            
            result = await self.query_engine.execute_query(command)
            
            # Update system health based on results
            await self._update_system_health_from_query(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting current state: {e}")
            return QueryResult(
                data={},
                success=False,
                query_type="current_state",
                execution_time_ms=0.0,
                timestamp=datetime.now(),
                context=QueryContext(),
                error=str(e)
            )
    
    async def get_historical_data(self,
                                data_types: List[str],
                                time_range: Union[str, TimeRange],
                                aggregation: Optional[str] = None,
                                filters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """
        Get historical system data - main API interface method.
        
        Primary interface used by API layer for temporal/historical data requests.
        """
        try:
            if not self.config.enable_temporal_intelligence:
                return QueryResult(
                    data=[],
                    success=False,
                    query_type="temporal",
                    execution_time_ms=0.0,
                    timestamp=datetime.now(),
                    context=QueryContext(),
                    error="Temporal intelligence not enabled"
                )
            
            # Create query context
            context = QueryContext(
                priority=QueryPriority.NORMAL,
                timeout_seconds=self.config.query_timeout_seconds * 2  # Temporal queries may take longer
            )
            
            # Create and execute temporal query
            command = self.query_factory.create_temporal_query(
                time_range=time_range,
                metric_types=data_types,
                aggregation=aggregation,
                filters=filters,
                context=context
            )
            
            result = await self.query_engine.execute_query(command)
            return result
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return QueryResult(
                data=[],
                success=False,
                query_type="temporal",
                execution_time_ms=0.0,
                timestamp=datetime.now(),
                context=QueryContext(),
                error=str(e)
            )
    
    async def process_conversation(self,
                                 message: str,
                                 session_id: Optional[str] = None,
                                 context: Optional[Dict[str, Any]] = None) -> QueryResult:
        """
        Process natural language conversation - main API interface method.
        
        Primary interface used by API layer for chat/conversational requests.
        """
        try:
            if not self.config.enable_conversational_ai:
                return QueryResult(
                    data={"message": "Conversational AI not enabled"},
                    success=False,
                    query_type="conversational",
                    execution_time_ms=0.0,
                    timestamp=datetime.now(),
                    context=QueryContext(),
                    error="Conversational AI not enabled"
                )
            
            # Ensure session exists
            if session_id:
                await self._ensure_session_exists(session_id)
            
            # Create query context
            query_context = QueryContext(
                session_id=session_id,
                priority=QueryPriority.NORMAL,
                timeout_seconds=self.config.query_timeout_seconds
            )
            
            # Create and execute conversational query
            command = self.query_factory.create_conversational_query(
                natural_language=message,
                session_id=session_id,
                conversation_context=context,
                context=query_context
            )
            
            result = await self.query_engine.execute_query(command)
            
            # Store conversation
            if session_id and result.success:
                await self.data_access.store_conversation(
                    session_id=session_id,
                    query=message,
                    response=result.data.get('message', ''),
                    metadata={"timestamp": datetime.now().isoformat()}
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
            return QueryResult(
                data={"message": "Error processing conversation"},
                success=False,
                query_type="conversational",
                execution_time_ms=0.0,
                timestamp=datetime.now(),
                context=QueryContext(),
                error=str(e)
            )
    
    # ==================== ADVANCED INTELLIGENCE METHODS ====================
    
    async def get_system_events(self,
                              event_types: Optional[List[str]] = None,
                              significance_threshold: Optional[float] = None,
                              time_range: Optional[Union[str, TimeRange]] = None) -> QueryResult:
        """Get system events with filtering."""
        try:
            context = QueryContext(timeout_seconds=self.config.query_timeout_seconds)
            
            command = self.query_factory.create_event_query(
                event_types=event_types,
                significance_threshold=significance_threshold,
                time_range=time_range,
                context=context
            )
            
            return await self.query_engine.execute_query(command)
            
        except Exception as e:
            logger.error(f"Error getting system events: {e}")
            return QueryResult(
                data=[],
                success=False,
                query_type="event",
                execution_time_ms=0.0,
                timestamp=datetime.now(),
                context=QueryContext(),
                error=str(e)
            )
    
    async def get_system_patterns(self,
                                pattern_types: Optional[List[str]] = None,
                                confidence_threshold: Optional[float] = None) -> QueryResult:
        """Get detected system patterns."""
        try:
            context = QueryContext(timeout_seconds=self.config.query_timeout_seconds)
            
            command = self.query_factory.create_pattern_query(
                pattern_types=pattern_types,
                confidence_threshold=confidence_threshold,
                context=context
            )
            
            return await self.query_engine.execute_query(command)
            
        except Exception as e:
            logger.error(f"Error getting system patterns: {e}")
            return QueryResult(
                data=[],
                success=False,
                query_type="pattern",
                execution_time_ms=0.0,
                timestamp=datetime.now(),
                context=QueryContext(),
                error=str(e)
            )
    
    async def analyze_system_health(self) -> SystemHealthStatus:
        """Comprehensive system health analysis."""
        try:
            # Get current system metrics
            current_state_result = await self.get_current_state(["all"])
            
            if not current_state_result.success:
                return SystemHealthStatus(
                    status="unknown",
                    score=0.0,
                    issues=["Unable to retrieve current system state"],
                    recommendations=["Check system connectivity"],
                    last_updated=datetime.now()
                )
            
            # Analyze system data for health indicators
            issues = []
            recommendations = []
            health_score = 1.0
            
            current_data = current_state_result.data
            
            # Analyze GPU health
            if "gpu" in current_data:
                gpu_health = self._analyze_gpu_health(current_data["gpu"])
                health_score *= gpu_health["score"]
                issues.extend(gpu_health["issues"])
                recommendations.extend(gpu_health["recommendations"])
            
            # Analyze container health
            if "containers" in current_data:
                container_health = self._analyze_container_health(current_data["containers"])
                health_score *= container_health["score"]
                issues.extend(container_health["issues"])
                recommendations.extend(container_health["recommendations"])
            
            # Analyze memory health
            if "memory" in current_data:
                memory_health = self._analyze_memory_health(current_data["memory"])
                health_score *= memory_health["score"]
                issues.extend(memory_health["issues"])
                recommendations.extend(memory_health["recommendations"])
            
            # Determine overall status
            if health_score >= 0.9:
                status = "healthy"
            elif health_score >= 0.7:
                status = "warning"
            elif health_score >= 0.5:
                status = "critical"
            else:
                status = "critical"
            
            self.system_health = SystemHealthStatus(
                status=status,
                score=health_score,
                issues=issues,
                recommendations=recommendations,
                last_updated=datetime.now()
            )
            
            return self.system_health
            
        except Exception as e:
            logger.error(f"Error analyzing system health: {e}")
            return SystemHealthStatus(
                status="unknown",
                score=0.0,
                issues=[f"Health analysis error: {str(e)}"],
                recommendations=["Check system monitoring components"],
                last_updated=datetime.now()
            )
    
    def _analyze_gpu_health(self, gpu_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze GPU health indicators."""
        issues = []
        recommendations = []
        score = 1.0
        
        try:
            # Check temperature
            temperature = gpu_data.get("temperature", 0)
            if temperature > 85:
                issues.append(f"GPU temperature critical: {temperature}°C")
                recommendations.append("Check GPU cooling and thermal throttling")
                score *= 0.5
            elif temperature > 80:
                issues.append(f"GPU temperature high: {temperature}°C")
                recommendations.append("Monitor GPU thermal performance")
                score *= 0.8
            
            # Check utilization
            utilization = gpu_data.get("utilization", {})
            gpu_util = utilization.get("gpu", 0)
            if gpu_util > 95:
                recommendations.append("GPU at high utilization - consider workload optimization")
                score *= 0.9
            
            # Check memory usage
            memory = gpu_data.get("memory", {})
            memory_used = memory.get("used", 0)
            memory_total = memory.get("total", 1)
            memory_usage_percent = (memory_used / memory_total) * 100
            
            if memory_usage_percent > 95:
                issues.append(f"GPU memory usage critical: {memory_usage_percent:.1f}%")
                recommendations.append("Consider reducing model size or batch size")
                score *= 0.6
            elif memory_usage_percent > 85:
                issues.append(f"GPU memory usage high: {memory_usage_percent:.1f}%")
                score *= 0.8
            
        except Exception as e:
            logger.error(f"Error analyzing GPU health: {e}")
            issues.append("GPU health analysis incomplete")
            score *= 0.9
        
        return {"score": score, "issues": issues, "recommendations": recommendations}
    
    def _analyze_container_health(self, container_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze container health indicators."""
        issues = []
        recommendations = []
        score = 1.0
        
        try:
            containers = container_data.get("containers", [])
            
            for container in containers:
                container_name = container.get("id", "unknown")
                
                # Check health status
                health_status = container.get("health_status", "unknown")
                if health_status != "healthy":
                    issues.append(f"Container {container_name} unhealthy: {health_status}")
                    recommendations.append(f"Check container logs for {container_name}")
                    score *= 0.8
                
                # Check CPU usage
                cpu_usage = container.get("cpu_usage", 0)
                if cpu_usage > 90:
                    issues.append(f"Container {container_name} high CPU usage: {cpu_usage}%")
                    recommendations.append(f"Check workload in {container_name}")
                    score *= 0.9
                
        except Exception as e:
            logger.error(f"Error analyzing container health: {e}")
            issues.append("Container health analysis incomplete")
            score *= 0.9
        
        return {"score": score, "issues": issues, "recommendations": recommendations}
    
    def _analyze_memory_health(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory health indicators."""
        issues = []
        recommendations = []
        score = 1.0
        
        try:
            total = memory_data.get("total", 1)
            used = memory_data.get("used", 0)
            usage_percent = (used / total) * 100
            
            if usage_percent > 95:
                issues.append(f"Memory usage critical: {usage_percent:.1f}%")
                recommendations.append("Consider closing unnecessary applications")
                score *= 0.5
            elif usage_percent > 85:
                issues.append(f"Memory usage high: {usage_percent:.1f}%")
                recommendations.append("Monitor memory usage patterns")
                score *= 0.8
            
            # Check swap usage
            swap_data = memory_data.get("swap", {})
            swap_used = swap_data.get("used", 0)
            if swap_used > 0:
                issues.append(f"Swap in use: {swap_used / (1024**3):.1f}GB")
                recommendations.append("Memory pressure detected - consider adding RAM")
                score *= 0.7
            
        except Exception as e:
            logger.error(f"Error analyzing memory health: {e}")
            issues.append("Memory health analysis incomplete")
            score *= 0.9
        
        return {"score": score, "issues": issues, "recommendations": recommendations}
    
    # ==================== SESSION & SUBSCRIPTION MANAGEMENT ====================
    
    async def _ensure_session_exists(self, session_id: str):
        """Ensure conversation session exists."""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "created": datetime.now(),
                "last_activity": datetime.now(),
                "conversation_count": 0
            }
        else:
            self.active_sessions[session_id]["last_activity"] = datetime.now()
            self.active_sessions[session_id]["conversation_count"] += 1
    
    def subscribe_to_updates(self, callback: Callable, metric_types: List[str]):
        """Subscribe to real-time system updates."""
        # Store callback with subscription info
        subscription_info = {
            "callback": callback,
            "metric_types": metric_types,
            "subscribed_at": datetime.now()
        }
        self.update_subscribers.append(subscription_info)
        logger.info(f"Added update subscriber for {metric_types}")
    
    async def _broadcast_updates(self, update_data: Dict[str, Any]):
        """Broadcast updates to all subscribers."""
        for subscription in self.update_subscribers:
            try:
                await subscription["callback"](update_data)
            except Exception as e:
                logger.error(f"Error broadcasting to subscriber: {e}")
    
    async def _update_system_health_from_query(self, query_result: QueryResult):
        """Update system health based on query results."""
        if query_result.success and query_result.query_type == "current_state":
            # Trigger background health analysis
            asyncio.create_task(self.analyze_system_health())
    
    # ==================== SYSTEM MANAGEMENT ====================
    
    async def start_consciousness_monitoring(self):
        """Start continuous system consciousness monitoring."""
        if self.config.mode == ConsciousnessMode.MONITORING:
            # Start background monitoring task
            asyncio.create_task(self._continuous_monitoring_loop())
            logger.info("Started consciousness monitoring loop")
    
    async def _continuous_monitoring_loop(self):
        """Continuous monitoring loop for system awareness."""
        while True:
            try:
                # Perform periodic health check
                await self.analyze_system_health()
                
                # Sleep for configured interval
                await asyncio.sleep(self.config.collection_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.collection_interval_seconds)
    
    def inject_temporal_storage(self, temporal_storage: TemporalStorage):
        """Inject temporal storage dependency."""
        if self.config.enable_temporal_intelligence:
            temporal_repo = TemporalRepository(temporal_storage)
            self.data_access.inject_temporal_repository(temporal_repo)
            logger.info("Temporal storage injected into SystemConsciousness")
    
    def inject_ai_workstation_controller(self, controller):
        """Inject AI workstation controller dependency."""
        self.ai_workstation_controller = controller
        self.data_access.inject_current_state_components(
            ai_workstation_controller=controller,
            system_collector=self.system_collector
        )
        logger.info("AI workstation controller injected into SystemConsciousness")
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive consciousness system health check."""
        try:
            # Check all subsystems
            data_access_health = await self.data_access.health_check()
            query_engine_health = await self.query_engine.health_check()
            system_health = await self.analyze_system_health()
            
            return {
                "status": "healthy",
                "mode": self.config.mode.value,
                "data_access": data_access_health,
                "query_engine": query_engine_health,
                "system_health": {
                    "status": system_health.status,
                    "score": system_health.score,
                    "issues_count": len(system_health.issues)
                },
                "active_sessions": len(self.active_sessions),
                "update_subscribers": len(self.update_subscribers),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Consciousness health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }
    
    def get_consciousness_stats(self) -> Dict[str, Any]:
        """Get comprehensive consciousness statistics."""
        return {
            "config": {
                "mode": self.config.mode.value,
                "temporal_intelligence": self.config.enable_temporal_intelligence,
                "conversational_ai": self.config.enable_conversational_ai,
                "predictive_analysis": self.config.enable_predictive_analysis
            },
            "query_engine_stats": self.query_engine.get_engine_stats(),
            "active_sessions": len(self.active_sessions),
            "update_subscribers": len(self.update_subscribers),
            "system_health": {
                "status": self.system_health.status,
                "score": self.system_health.score,
                "last_updated": self.system_health.last_updated.isoformat()
            }
        }