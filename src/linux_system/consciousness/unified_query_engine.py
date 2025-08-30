"""
Unified Query Engine - Command Pattern Orchestration
====================================================

Central query processing engine using command pattern to provide unified
access to all system consciousness capabilities. Handles:

- Query validation and optimization
- Command dispatch and execution
- Result aggregation and formatting  
- Performance monitoring and caching
- Error handling and recovery

Key principles:
- Command pattern for consistent query processing
- Strategy pattern for execution optimization
- Async processing for scalability
- Comprehensive error handling and monitoring
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
import time

from .query_commands import (
    QueryCommand, QueryResult, QueryContext, QueryPriority, QueryValidationResult,
    CurrentStateQuery, TemporalQuery, EventQuery, ConversationalQuery, PatternQuery
)
from .data_access_layer import DataAccessLayer

logger = logging.getLogger(__name__)


@dataclass
class QueryStats:
    """Statistics for query execution monitoring."""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_execution_time_ms: float = 0.0
    queries_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_execution(self, query_result: QueryResult):
        """Add query execution to statistics."""
        self.total_queries += 1
        
        if query_result.success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1
            self.recent_errors.append({
                "timestamp": query_result.timestamp.isoformat(),
                "query_type": query_result.query_type,
                "error": query_result.error
            })
        
        # Update average execution time
        if self.total_queries > 1:
            self.average_execution_time_ms = (
                (self.average_execution_time_ms * (self.total_queries - 1) + query_result.execution_time_ms) 
                / self.total_queries
            )
        else:
            self.average_execution_time_ms = query_result.execution_time_ms
        
        # Update query type statistics
        self.queries_by_type[query_result.query_type] += 1
    
    def get_success_rate(self) -> float:
        """Get query success rate."""
        if self.total_queries == 0:
            return 0.0
        return self.successful_queries / self.total_queries
    
    def get_stats_dict(self) -> Dict[str, Any]:
        """Get statistics as dictionary."""
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": self.get_success_rate(),
            "average_execution_time_ms": self.average_execution_time_ms,
            "queries_by_type": dict(self.queries_by_type),
            "recent_errors": list(self.recent_errors)
        }


@dataclass
class QueryPlan:
    """Execution plan for query optimization."""
    command: QueryCommand
    estimated_cost: float  # Execution cost estimate (0.0-1.0)
    cache_key: Optional[str] = None
    optimization_notes: List[str] = field(default_factory=list)
    parallel_execution: bool = False
    timeout_seconds: Optional[int] = None


class QueryCache:
    """Simple query result caching system."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """Initialize query cache."""
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
    
    def get(self, cache_key: str) -> Optional[QueryResult]:
        """Get cached result if valid."""
        if cache_key not in self.cache:
            return None
        
        # Check TTL
        if cache_key in self.access_times:
            age = datetime.now() - self.access_times[cache_key]
            if age.total_seconds() > self.ttl_seconds:
                self._remove(cache_key)
                return None
        
        # Update access time
        self.access_times[cache_key] = datetime.now()
        
        try:
            cached_data = self.cache[cache_key]
            return QueryResult(**cached_data)
        except Exception as e:
            logger.error(f"Error deserializing cached result: {e}")
            self._remove(cache_key)
            return None
    
    def put(self, cache_key: str, result: QueryResult):
        """Cache query result."""
        if not result.success:
            return  # Don't cache failed results
        
        # Manage cache size
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        try:
            # Store serializable data
            self.cache[cache_key] = {
                "data": result.data,
                "success": result.success,
                "query_type": result.query_type,
                "execution_time_ms": result.execution_time_ms,
                "timestamp": result.timestamp,
                "context": result.context,
                "error": result.error,
                "metadata": result.metadata
            }
            self.access_times[cache_key] = datetime.now()
        except Exception as e:
            logger.error(f"Error caching result: {e}")
    
    def _remove(self, cache_key: str):
        """Remove cache entry."""
        self.cache.pop(cache_key, None)
        self.access_times.pop(cache_key, None)
    
    def _evict_oldest(self):
        """Evict oldest cache entry."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove(oldest_key)
    
    def clear(self):
        """Clear all cached results."""
        self.cache.clear()
        self.access_times.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "cache_keys": list(self.cache.keys())
        }


class UnifiedQueryEngine:
    """
    Central query processing engine using command pattern.
    
    Provides unified interface for all system consciousness queries with:
    - Query validation and optimization
    - Command dispatch and execution
    - Caching and performance monitoring
    - Error handling and recovery
    """
    
    def __init__(self, 
                 data_access: DataAccessLayer,
                 enable_caching: bool = True,
                 cache_ttl_seconds: int = 300,
                 max_concurrent_queries: int = 10):
        """Initialize unified query engine."""
        self.data_access = data_access
        self.enable_caching = enable_caching
        self.max_concurrent_queries = max_concurrent_queries
        
        # Initialize components
        self.query_cache = QueryCache(ttl_seconds=cache_ttl_seconds) if enable_caching else None
        self.query_stats = QueryStats()
        self.active_queries: Dict[str, QueryCommand] = {}
        self.query_semaphore = asyncio.Semaphore(max_concurrent_queries)
        
        # Query optimization callbacks
        self.optimization_callbacks: Dict[str, Callable] = {}
        self.validation_callbacks: Dict[str, Callable] = {}
        
        logger.info(f"UnifiedQueryEngine initialized with caching={'enabled' if enable_caching else 'disabled'}")
    
    async def execute_query(self, command: QueryCommand) -> QueryResult:
        """
        Execute query command with full processing pipeline.
        
        Main entry point for all query processing.
        """
        query_id = self._generate_query_id(command)
        start_time = time.time()
        
        try:
            # Add to active queries
            self.active_queries[query_id] = command
            
            # Acquire semaphore for concurrency control
            async with self.query_semaphore:
                # Execute query processing pipeline
                result = await self._execute_query_pipeline(command, query_id)
            
            # Update statistics
            self.query_stats.add_execution(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error in query execution: {e}")
            
            # Create error result
            error_result = QueryResult(
                data={},
                success=False,
                query_type=command.get_query_type(),
                execution_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                context=command.context,
                error=str(e)
            )
            
            self.query_stats.add_execution(error_result)
            return error_result
            
        finally:
            # Remove from active queries
            self.active_queries.pop(query_id, None)
    
    async def _execute_query_pipeline(self, command: QueryCommand, query_id: str) -> QueryResult:
        """Execute complete query processing pipeline."""
        
        # Step 1: Validate query
        validation_result = await self._validate_query(command)
        if not validation_result.valid:
            return QueryResult(
                data={},
                success=False,
                query_type=command.get_query_type(),
                execution_time_ms=0.0,
                timestamp=datetime.now(),
                context=command.context,
                error=f"Validation failed: {validation_result.error}"
            )
        
        # Step 2: Create query plan
        query_plan = await self._create_query_plan(command)
        
        # Step 3: Check cache
        if self.enable_caching and query_plan.cache_key:
            cached_result = self.query_cache.get(query_plan.cache_key)
            if cached_result:
                logger.debug(f"Cache hit for query {query_id}")
                return cached_result
        
        # Step 4: Execute query
        result = await self._execute_query_command(command, query_plan)
        
        # Step 5: Cache result
        if self.enable_caching and query_plan.cache_key and result.success:
            self.query_cache.put(query_plan.cache_key, result)
        
        return result
    
    async def _validate_query(self, command: QueryCommand) -> QueryValidationResult:
        """Validate query command."""
        try:
            # Basic validation from command
            validation_result = command.validate()
            
            if not validation_result.valid:
                return validation_result
            
            # Check timeout
            if command.context.timeout_seconds <= 0:
                return QueryValidationResult(False, "timeout_seconds must be positive")
            
            # Additional validation based on query type
            query_type = command.get_query_type()
            if query_type in self.validation_callbacks:
                custom_validation = await self.validation_callbacks[query_type](command)
                if not custom_validation.valid:
                    return custom_validation
            
            return QueryValidationResult(True)
            
        except Exception as e:
            logger.error(f"Error validating query: {e}")
            return QueryValidationResult(False, f"Validation error: {str(e)}")
    
    async def _create_query_plan(self, command: QueryCommand) -> QueryPlan:
        """Create optimized execution plan for query."""
        try:
            query_type = command.get_query_type()
            
            # Estimate execution cost
            cost = self._estimate_query_cost(command)
            
            # Generate cache key for cacheable queries
            cache_key = None
            if self._is_cacheable_query(command):
                cache_key = self._generate_cache_key(command)
            
            # Create basic plan
            plan = QueryPlan(
                command=command,
                estimated_cost=cost,
                cache_key=cache_key,
                timeout_seconds=command.context.timeout_seconds
            )
            
            # Apply query-specific optimizations
            if query_type in self.optimization_callbacks:
                plan = await self.optimization_callbacks[query_type](plan)
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating query plan: {e}")
            return QueryPlan(command=command, estimated_cost=1.0)
    
    def _estimate_query_cost(self, command: QueryCommand) -> float:
        """Estimate query execution cost (0.0-1.0)."""
        query_type = command.get_query_type()
        
        # Base costs by query type
        base_costs = {
            "currentstate": 0.2,  # Fast - current data only
            "temporal": 0.7,      # Medium - historical data processing  
            "event": 0.5,         # Medium - event filtering
            "conversational": 0.8, # High - NLP processing
            "pattern": 0.9        # High - pattern analysis
        }
        
        base_cost = base_costs.get(query_type, 0.5)
        
        # Adjust based on query specifics
        if isinstance(command, TemporalQuery):
            # Larger time ranges cost more
            time_span = command.time_range.end_time - command.time_range.start_time
            if time_span > timedelta(days=7):
                base_cost += 0.2
            elif time_span > timedelta(days=1):
                base_cost += 0.1
        
        if hasattr(command, 'metric_types') and command.metric_types:
            # More metrics cost more
            if len(command.metric_types) > 5:
                base_cost += 0.1
        
        return min(base_cost, 1.0)
    
    def _is_cacheable_query(self, command: QueryCommand) -> bool:
        """Determine if query results can be cached."""
        # Current state queries are not cacheable (too dynamic)
        if isinstance(command, CurrentStateQuery):
            return False
        
        # Conversational queries are not cacheable (context-dependent)
        if isinstance(command, ConversationalQuery):
            return False
        
        # Historical queries are cacheable
        return True
    
    def _generate_cache_key(self, command: QueryCommand) -> str:
        """Generate cache key for query."""
        try:
            query_type = command.get_query_type()
            
            if isinstance(command, TemporalQuery):
                return f"temporal:{query_type}:{command.time_range.start_time.isoformat()}:{command.time_range.end_time.isoformat()}:{hash(str(command.metric_types))}"
            elif isinstance(command, EventQuery):
                return f"event:{hash(str(command.event_types))}:{command.significance_threshold}"
            elif isinstance(command, PatternQuery):
                return f"pattern:{hash(str(command.pattern_types))}:{command.confidence_threshold}"
            else:
                return f"{query_type}:{hash(str(command.__dict__))}"
                
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return f"{command.get_query_type()}:{datetime.now().isoformat()}"
    
    def _generate_query_id(self, command: QueryCommand) -> str:
        """Generate unique query ID."""
        return f"{command.get_query_type()}_{datetime.now().timestamp()}_{id(command)}"
    
    async def _execute_query_command(self, command: QueryCommand, plan: QueryPlan) -> QueryResult:
        """Execute the actual query command."""
        try:
            # Apply timeout
            timeout = plan.timeout_seconds or command.context.timeout_seconds
            
            result = await asyncio.wait_for(
                command.execute(self.data_access),
                timeout=timeout
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Query timeout after {timeout} seconds")
            return QueryResult(
                data={},
                success=False,
                query_type=command.get_query_type(),
                execution_time_ms=timeout * 1000,
                timestamp=datetime.now(),
                context=command.context,
                error=f"Query timeout after {timeout} seconds"
            )
            
        except Exception as e:
            logger.error(f"Error executing query command: {e}")
            return QueryResult(
                data={},
                success=False,
                query_type=command.get_query_type(),
                execution_time_ms=0.0,
                timestamp=datetime.now(),
                context=command.context,
                error=str(e)
            )
    
    async def execute_batch_queries(self, commands: List[QueryCommand]) -> List[QueryResult]:
        """Execute multiple queries concurrently."""
        try:
            # Execute queries concurrently with semaphore control
            tasks = [self.execute_query(command) for command in commands]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Batch query {i} failed: {result}")
                    error_result = QueryResult(
                        data={},
                        success=False,
                        query_type=commands[i].get_query_type(),
                        execution_time_ms=0.0,
                        timestamp=datetime.now(),
                        context=commands[i].context,
                        error=str(result)
                    )
                    processed_results.append(error_result)
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error executing batch queries: {e}")
            # Return error results for all queries
            return [
                QueryResult(
                    data={},
                    success=False,
                    query_type=cmd.get_query_type(),
                    execution_time_ms=0.0,
                    timestamp=datetime.now(),
                    context=cmd.context,
                    error=f"Batch execution error: {str(e)}"
                )
                for cmd in commands
            ]
    
    def register_optimization_callback(self, query_type: str, callback: Callable):
        """Register query optimization callback."""
        self.optimization_callbacks[query_type] = callback
        logger.info(f"Registered optimization callback for {query_type}")
    
    def register_validation_callback(self, query_type: str, callback: Callable):
        """Register query validation callback."""
        self.validation_callbacks[query_type] = callback
        logger.info(f"Registered validation callback for {query_type}")
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get query engine statistics."""
        return {
            "query_stats": self.query_stats.get_stats_dict(),
            "cache_stats": self.query_cache.get_cache_stats() if self.query_cache else None,
            "active_queries": len(self.active_queries),
            "max_concurrent_queries": self.max_concurrent_queries,
            "optimization_callbacks": list(self.optimization_callbacks.keys()),
            "validation_callbacks": list(self.validation_callbacks.keys())
        }
    
    def clear_cache(self):
        """Clear query cache."""
        if self.query_cache:
            self.query_cache.clear()
            logger.info("Query cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check query engine health."""
        try:
            # Check data access layer health
            data_access_health = await self.data_access.health_check()
            
            return {
                "status": "healthy",
                "data_access_health": data_access_health,
                "active_queries": len(self.active_queries),
                "total_processed": self.query_stats.total_queries,
                "success_rate": self.query_stats.get_success_rate()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }