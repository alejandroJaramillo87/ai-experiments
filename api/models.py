"""
API Models and Schemas - Auto-Generated + Manual Models

Combines auto-generated Pydantic models from src/ collectors with manually
defined models for request/response serialization. This ensures:
- Data-driven API schema from actual system capabilities
- Consistent interface for chat, streaming, and data endpoints
- Automatic updates when collectors change
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field

# Import auto-generated models from ModelGenerator
from .core.model_generator import create_model_generator

# Auto-generate models from src/ collectors
_model_generator = create_model_generator()
_generated_models = _model_generator.generate_all_contracts()

# Make auto-generated models available for import
for model_name, model_class in _generated_models.items():
    globals()[model_name] = model_class

# ==================== CHAT INTERFACE MODELS ====================

class ChatRequest(BaseModel):
    """Request model for natural language questions"""
    question: str = Field(..., description="Natural language question about the system")
    session_id: Optional[str] = Field(None, description="Optional session ID for context")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the query")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "Why did my GPU throttle yesterday at 2pm?",
                "session_id": "user_session_123",
                "context": {"time_focus": "recent"}
            }
        }


class ChatResponse(BaseModel):
    """Response model for natural language answers"""
    response: str = Field(..., description="Natural language response to the question") 
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score for the response")
    data_references: List[Dict[str, Any]] = Field(default_factory=list, description="Referenced data supporting the answer")
    suggested_actions: List[str] = Field(default_factory=list, description="Suggested actions based on analysis")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response generation timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "response": "At 2:47pm yesterday, your GPU throttled because both llama-gpu and vllm-gpu were competing for VRAM, pushing temperature from 78°C to 84°C.",
                "session_id": "user_session_123", 
                "confidence": 0.92,
                "data_references": [{"metric": "gpu_temperature", "timestamp": "2023-11-15T14:47:00Z", "value": 84.0}],
                "suggested_actions": ["Monitor container resource allocation", "Consider staggered model loading"],
                "follow_up_suggestions": ["Show me the thermal pattern", "What about CPU usage during that time?"],
                "processing_time_ms": 245.3
            }
        }


# ==================== API REQUEST/RESPONSE MODELS ====================

class CurrentStateRequest(BaseModel):
    """Request model for bulk current state data"""
    metric_types: List[str] = Field(..., description="List of metric types to retrieve")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Optional filters for data")
    
    class Config:
        schema_extra = {
            "example": {
                "metric_types": ["gpu", "containers", "processes"],
                "filters": {"include_inactive": False}
            }
        }


class CurrentStateResponse(BaseModel):
    """Response model for bulk current state data"""
    data: Dict[str, Any] = Field(..., description="Current state data by metric type")
    timestamp: datetime = Field(default_factory=datetime.now, description="Data collection timestamp")
    metric_types: List[str] = Field(..., description="Requested metric types")
    status: str = Field(default="success", description="Response status")


class HistoricalQueryRequest(BaseModel):
    """Request model for historical data queries"""
    data_types: List[str] = Field(..., description="Types of data to retrieve")
    time_range: str = Field(..., description="Time range (e.g., '4h', '1d', '1w')")
    aggregation: Optional[str] = Field(None, description="Aggregation type (avg, max, min, sum)")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Optional filters")
    granularity: Optional[str] = Field("minute", description="Data granularity for visualization")
    
    class Config:
        schema_extra = {
            "example": {
                "data_types": ["gpu_temperature", "container_resources"],
                "time_range": "4h",
                "aggregation": "avg",
                "granularity": "minute",
                "filters": {"containers": ["llama-gpu", "vllm-gpu"]}
            }
        }


class HistoricalQueryResponse(BaseModel):
    """Response model for historical data queries"""
    data: Dict[str, Any] = Field(..., description="Historical data formatted for visualization")
    timestamp: datetime = Field(default_factory=datetime.now, description="Query response timestamp")
    time_range: str = Field(..., description="Requested time range")
    data_types: List[str] = Field(..., description="Requested data types")
    count: int = Field(default=0, description="Number of data points returned")
    status: str = Field(default="success", description="Query status")


class MetricSubscriptionRequest(BaseModel):
    """Request model for WebSocket metric subscriptions"""
    metric_types: List[str] = Field(..., description="Metric types to subscribe to")
    update_interval: int = Field(default=1000, description="Update interval in milliseconds")
    include_events: bool = Field(default=True, description="Include system events")
    include_changes: bool = Field(default=True, description="Include system changes")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Subscription filters")


# ==================== SYSTEM STATUS MODELS ====================

class SystemStatus(BaseModel):
    """Comprehensive system status information"""
    timestamp: datetime = Field(..., description="Status collection timestamp")
    mode: str = Field(..., description="Current AI workstation operating mode")
    health_status: str = Field(..., description="Overall system health assessment")
    overall_performance_score: float = Field(..., ge=0.0, le=1.0, description="Overall performance score")
    optimization_effectiveness: float = Field(..., ge=0.0, le=1.0, description="Optimization effectiveness score")
    thermal_efficiency: float = Field(..., ge=0.0, le=1.0, description="Thermal management efficiency")
    resource_utilization: Dict[str, float] = Field(..., description="Resource utilization percentages")
    active_workloads: List[str] = Field(..., description="Currently active AI workloads")
    critical_alerts: List[str] = Field(..., description="Critical system alerts")
    recommendations: List[str] = Field(..., description="System optimization recommendations")
    component_status: Dict[str, bool] = Field(..., description="Status of major system components")
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:30:00Z",
                "mode": "optimizing",
                "health_status": "excellent",
                "overall_performance_score": 0.89,
                "resource_utilization": {"gpu": 0.75, "cpu": 0.45, "memory": 0.62},
                "active_workloads": ["llama-gpu:Qwen2.5-32B", "llama-cpu-0:Llama-3.1-8B"],
                "critical_alerts": [],
                "recommendations": ["Consider load balancing across CPU containers"]
            }
        }


class PerformanceInsights(BaseModel):
    """Detailed performance insights and analytics"""
    timestamp: datetime = Field(..., description="Insights collection timestamp")
    uptime_hours: float = Field(..., description="System uptime in hours")
    current_mode: str = Field(..., description="Current operating mode")
    container_intelligence: Dict[str, Any] = Field(..., description="Container orchestration insights")
    hardware_specialization: Dict[str, Any] = Field(..., description="Hardware-specific performance data")
    multi_model_oracle: Dict[str, Any] = Field(..., description="ML prediction and optimization insights")
    performance_trends: Dict[str, Any] = Field(..., description="Performance trend analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "uptime_hours": 48.5,
                "current_mode": "optimizing",
                "container_intelligence": {
                    "service_efficiency": 0.87,
                    "load_balancing_quality": 0.91
                },
                "hardware_specialization": {
                    "rtx5090_blackwall": {"utilization": 0.82, "thermal_state": "optimal"},
                    "amd_zen5": {"core_efficiency": 0.76, "aocl_utilization": 0.43}
                }
            }
        }


# ==================== OPTIMIZATION MODELS ====================

class OptimizationRequest(BaseModel):
    """Request for system optimization execution"""
    targets: Dict[str, float] = Field(..., description="Optimization targets (e.g., {'gpu_performance': 0.15})")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Constraints for optimization")
    timeout_seconds: Optional[int] = Field(300, description="Maximum optimization execution time")
    dry_run: bool = Field(False, description="Whether to perform a dry run without actual changes")
    
    class Config:
        schema_extra = {
            "example": {
                "targets": {
                    "gpu_performance": 0.15,
                    "thermal_efficiency": 0.10,
                    "inference_throughput": 0.20
                },
                "constraints": {
                    "max_temperature": 80,
                    "preserve_model_quality": True
                },
                "timeout_seconds": 300,
                "dry_run": False
            }
        }


class OptimizationResponse(BaseModel):
    """Response from optimization execution"""
    success: bool = Field(..., description="Whether optimization was successful")
    optimization_id: str = Field(..., description="Unique identifier for this optimization")
    actions_executed: int = Field(..., description="Number of optimization actions executed")
    performance_gains: Dict[str, float] = Field(..., description="Actual performance improvements achieved")
    message: str = Field(..., description="Human-readable optimization summary")
    timestamp: datetime = Field(..., description="Optimization completion timestamp")
    execution_time_seconds: Optional[float] = Field(None, description="Total execution time")
    details: Optional[Dict[str, Any]] = Field(None, description="Detailed optimization results")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "optimization_id": "opt_20240115_103045",
                "actions_executed": 5,
                "performance_gains": {"gpu_performance": 0.12, "thermal_efficiency": 0.08},
                "message": "Successfully optimized GPU allocation and thermal management",
                "execution_time_seconds": 45.3
            }
        }


# ==================== REAL-TIME STREAMING MODELS ====================

class SystemMetricsSnapshot(BaseModel):
    """Real-time system metrics for WebSocket streaming"""
    timestamp: datetime = Field(..., description="Metrics collection timestamp")
    
    # GPU metrics (RTX 5090 Blackwall)
    gpu: Dict[str, Any] = Field(..., description="GPU performance and thermal metrics")
    
    # CPU metrics (AMD 9950X Zen 5) 
    cpu: Dict[str, Any] = Field(..., description="CPU utilization and performance metrics")
    
    # Thermal management (15-fan cooling system)
    thermal: Dict[str, Any] = Field(..., description="Thermal management and cooling data")
    
    # Container orchestration (5 AI services)
    containers: List[Dict[str, Any]] = Field(..., description="Container status and resource usage")
    
    # Causal events and correlations
    causal_events: List[Dict[str, Any]] = Field(..., description="Recent causal relationships detected")
    
    # ML predictions and insights
    predictions: Dict[str, Any] = Field(..., description="Performance predictions and forecasts")
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:30:15.123Z",
                "gpu": {
                    "temperature": 78.5,
                    "vram_usage_gb": 24.2,
                    "utilization_percent": 85.0,
                    "tensor_core_utilization": 0.92
                },
                "cpu": {
                    "core_utilization": [45, 67, 23, 89, 34, 56, 78, 91, 12, 45, 67, 34, 56, 78, 23, 45],
                    "temperature": 65.2,
                    "aocl_library_usage": 0.34
                },
                "thermal": {
                    "fan_speeds": {"intake": [1200, 1180, 1220], "exhaust": [1100, 1090, 1110, 1105]},
                    "cooling_efficiency": 0.87
                },
                "containers": [
                    {"name": "llama-gpu", "cpu_percent": 15.2, "memory_mb": 8192, "status": "running"},
                    {"name": "llama-cpu-0", "cpu_percent": 45.6, "memory_mb": 4096, "status": "running"}
                ]
            }
        }


# ==================== DASHBOARD SPECIFIC MODELS ====================

class GPUDashboardData(BaseModel):
    """RTX 5090 dashboard visualization data"""
    timestamp: datetime = Field(..., description="Data collection timestamp")
    temperature: float = Field(..., description="GPU temperature in Celsius")
    vram_usage_gb: float = Field(..., description="VRAM usage in GB")
    vram_total_gb: float = Field(32, description="Total VRAM capacity")
    utilization_percent: float = Field(..., description="GPU utilization percentage")
    tensor_core_utilization: float = Field(..., description="Tensor core utilization (0-1)")
    memory_bandwidth_percent: float = Field(..., description="Memory bandwidth utilization")
    power_draw_watts: float = Field(..., description="Current power draw in watts")
    clock_speeds: Dict[str, int] = Field(..., description="GPU and memory clock speeds")
    thermal_throttling_active: bool = Field(..., description="Whether thermal throttling is active")


class CPUDashboardData(BaseModel):
    """AMD 9950X dashboard visualization data"""
    timestamp: datetime = Field(..., description="Data collection timestamp")
    core_utilization: List[float] = Field(..., description="Per-core utilization percentages")
    temperature: float = Field(..., description="CPU temperature in Celsius")
    aocl_library_usage: float = Field(..., description="AOCL library utilization (0-1)")
    memory_bandwidth_percent: float = Field(..., description="Memory bandwidth utilization")
    instructions_per_second: float = Field(..., description="Instructions executed per second")
    cache_hit_rates: Dict[str, float] = Field(..., description="L1/L2/L3 cache hit rates")
    numa_efficiency: float = Field(..., description="NUMA locality efficiency (0-1)")
    power_draw_watts: float = Field(..., description="CPU power draw in watts")


class ThermalDashboardData(BaseModel):
    """15-fan cooling system dashboard data"""
    timestamp: datetime = Field(..., description="Data collection timestamp")
    component_temperatures: Dict[str, float] = Field(..., description="Temperature of each component")
    fan_speeds: Dict[str, List[int]] = Field(..., description="RPM of each fan group")
    cooling_efficiency: float = Field(..., description="Overall cooling efficiency (0-1)")
    thermal_throttling_risk: float = Field(..., description="Risk of thermal throttling (0-1)")
    predicted_temperature_trend: List[float] = Field(..., description="Predicted temperature trend")
    airflow_optimization_score: float = Field(..., description="Airflow optimization quality (0-1)")


class ContainerDashboardData(BaseModel):
    """Container orchestration dashboard data"""
    timestamp: datetime = Field(..., description="Data collection timestamp")
    services: List[Dict[str, Any]] = Field(..., description="Status of all AI services")
    resource_flows: List[Dict[str, Any]] = Field(..., description="Resource flow data for visualization")
    service_interactions: List[Dict[str, Any]] = Field(..., description="Inter-service communication patterns")
    load_balancing_efficiency: float = Field(..., description="Load balancing quality (0-1)")
    resource_contention: List[Dict[str, Any]] = Field(..., description="Resource contention events")
    inference_patterns: Dict[str, Any] = Field(..., description="Inference request patterns")


# ==================== ERROR MODELS ====================

class APIError(BaseModel):
    """Standard API error response"""
    error: str = Field(..., description="Error type or category")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error occurrence timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier for debugging")


# ==================== CONFIGURATION MODELS ====================

class APIConfig(BaseModel):
    """API configuration settings"""
    version: str = Field("1.0.0", description="API version")
    environment: str = Field("development", description="Environment (development/production)")
    debug_mode: bool = Field(True, description="Whether debug mode is enabled")
    max_chat_history: int = Field(1000, description="Maximum chat history to maintain")
    streaming_update_interval_ms: int = Field(1000, description="WebSocket update interval")
    optimization_timeout_seconds: int = Field(300, description="Default optimization timeout")
    cors_origins: List[str] = Field(["http://localhost:5173"], description="Allowed CORS origins")


# ==================== UTILITY TYPES ====================

# Type aliases for common data structures
MetricsDict = Dict[str, Union[float, int, str, bool]]
TimeSeriesData = List[Dict[str, Union[str, float, int]]]
VisualizationData = Dict[str, Any]
SystemComponentData = Dict[str, Any]