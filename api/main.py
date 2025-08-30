"""
AI Workstation Intelligence Platform - API Gateway

FastAPI application that provides REST and WebSocket APIs for the Svelte frontend,
wrapping the existing AIWorkstationController and ML digital twin system.

Provides:
- Natural language chat interface
- Real-time system data streaming  
- Dashboard data APIs
- System control and optimization endpoints
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import your existing AI workstation system
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from linux_system.ai_workstation import (
    AIWorkstationController, 
    AIWorkstationMode, 
    SystemHealthStatus
)

# API models and schemas
from .models import (
    ChatQuery,
    ChatResponse, 
    SystemStatus,
    PerformanceInsights,
    OptimizationRequest,
    OptimizationResponse
)

# Natural language processing (will implement in Phase 1B)
from .natural_language import NaturalLanguageProcessor

# Real-time streaming manager
from .streaming import RealtimeStreamingManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
workstation_controller: Optional[AIWorkstationController] = None
nl_processor: Optional[NaturalLanguageProcessor] = None
streaming_manager: Optional[RealtimeStreamingManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown"""
    global workstation_controller, nl_processor, streaming_manager
    
    try:
        logger.info("Starting AI Workstation Intelligence Platform...")
        
        # Initialize the AI workstation consciousness system
        workstation_controller = AIWorkstationController()
        
        # Start the workstation in optimizing mode
        result = await workstation_controller.start_ai_workstation(
            AIWorkstationMode.OPTIMIZING
        )
        
        if not result.get("success"):
            raise RuntimeError(f"Failed to start AI workstation: {result.get('error')}")
        
        logger.info(f"AI Workstation started successfully: {result}")
        
        # Initialize natural language processor
        nl_processor = NaturalLanguageProcessor(workstation_controller)
        
        # Initialize real-time streaming manager
        streaming_manager = RealtimeStreamingManager(workstation_controller)
        
        # Start background streaming task
        asyncio.create_task(streaming_manager.start_streaming())
        
        logger.info("All systems initialized successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start AI Workstation system: {e}")
        raise
    finally:
        # Shutdown cleanup
        logger.info("Shutting down AI Workstation Intelligence Platform...")
        
        if streaming_manager:
            await streaming_manager.stop_streaming()
            
        if workstation_controller:
            await workstation_controller.stop_ai_workstation()
        
        logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="AI Workstation Intelligence Platform",
    description="Professional web interface for RTX 5090 + AMD 9950X AI workstation consciousness",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for Svelte frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Svelte dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== HEALTH CHECK ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not workstation_controller:
        raise HTTPException(status_code=503, detail="AI Workstation system not initialized")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_running": workstation_controller.running,
        "components": {
            "ai_workstation": True,
            "natural_language": nl_processor is not None,
            "streaming": streaming_manager is not None
        }
    }


# ==================== CHAT INTERFACE ====================

@app.post("/api/chat/query", response_model=ChatResponse)
async def chat_query(query: ChatQuery) -> ChatResponse:
    """
    Process natural language questions about the AI workstation system
    
    Examples:
    - "Why did my GPU throttle yesterday at 2pm?"
    - "How is my thermal management performing?"
    - "What patterns do you see in my container usage?"
    """
    if not nl_processor:
        raise HTTPException(status_code=503, detail="Natural language processor not available")
    
    try:
        logger.info(f"Processing chat query: {query.question[:100]}...")
        
        # Process the question through natural language pipeline
        response = await nl_processor.process_question(query.question, query.session_id)
        
        logger.info(f"Generated response with confidence: {response.confidence}")
        
        return response
        
    except Exception as e:
        logger.error(f"Chat query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/api/chat/suggestions")
async def get_chat_suggestions() -> List[str]:
    """Get suggested questions for the chat interface"""
    return [
        "How is my GPU performing right now?",
        "Why did thermal throttling happen earlier?",
        "What patterns do you see in my container usage?", 
        "Should I start training a model now?",
        "How efficient is my 15-fan cooling system?",
        "What's the status of my AI services?",
        "Show me my AMD Zen 5 core utilization patterns",
        "What optimization opportunities exist?",
        "How does my weekend usage compare to weekdays?",
        "What caused the recent memory spike?"
    ]


# ==================== SYSTEM STATUS ====================

@app.get("/api/status/overview", response_model=SystemStatus)
async def get_system_status() -> SystemStatus:
    """Get comprehensive system status and health information"""
    if not workstation_controller:
        raise HTTPException(status_code=503, detail="AI Workstation system not available")
    
    try:
        # Get current status from the workstation controller
        status = await workstation_controller.get_workstation_status()
        
        return SystemStatus(
            timestamp=status.timestamp,
            mode=status.mode.value,
            health_status=status.health_status.value,
            overall_performance_score=status.overall_performance_score,
            optimization_effectiveness=status.optimization_effectiveness,
            thermal_efficiency=status.thermal_efficiency,
            resource_utilization=status.resource_utilization,
            active_workloads=status.active_workloads,
            critical_alerts=status.critical_alerts,
            recommendations=status.recommendations,
            component_status={
                "container_consciousness": status.container_consciousness_active,
                "hardware_specialization": status.hardware_specialization_active,
                "multi_model_oracle": status.multi_model_oracle_active,
                "temporal_intelligence": status.temporal_intelligence_active
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


@app.get("/api/insights/performance", response_model=PerformanceInsights)  
async def get_performance_insights() -> PerformanceInsights:
    """Get detailed performance insights and analytics"""
    if not workstation_controller:
        raise HTTPException(status_code=503, detail="AI Workstation system not available")
    
    try:
        insights = await workstation_controller.get_performance_insights()
        
        return PerformanceInsights(
            timestamp=datetime.now(),
            uptime_hours=insights.get("uptime_hours", 0),
            current_mode=insights.get("current_mode", "unknown"),
            container_intelligence=insights.get("container_intelligence", {}),
            hardware_specialization=insights.get("hardware_specialization", {}),
            multi_model_oracle=insights.get("multi_model_oracle", {}),
            performance_trends=insights.get("performance_trends", {})
        )
        
    except Exception as e:
        logger.error(f"Failed to get performance insights: {e}")
        raise HTTPException(status_code=500, detail=f"Insights retrieval failed: {str(e)}")


# ==================== SYSTEM CONTROL ====================

@app.post("/api/control/optimize", response_model=OptimizationResponse)
async def execute_optimization(request: OptimizationRequest) -> OptimizationResponse:
    """Execute system optimization with specific targets"""
    if not workstation_controller:
        raise HTTPException(status_code=503, detail="AI Workstation system not available")
    
    try:
        logger.info(f"Executing optimization with targets: {request.targets}")
        
        # Execute optimization through the workstation controller
        result = await workstation_controller.execute_optimization(request.targets)
        
        return OptimizationResponse(
            success=result.get("success", False),
            optimization_id=result.get("plan_id", "unknown"),
            actions_executed=result.get("actions_executed", 0),
            performance_gains=result.get("performance_gains", {}),
            message=f"Optimization completed with {result.get('actions_successful', 0)} successful actions",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Optimization execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.post("/api/control/mode")
async def set_workstation_mode(mode: str) -> JSONResponse:
    """Change the AI workstation operating mode"""
    if not workstation_controller:
        raise HTTPException(status_code=503, detail="AI Workstation system not available")
    
    try:
        # Convert string to AIWorkstationMode enum
        mode_enum = AIWorkstationMode(mode.lower())
        
        result = await workstation_controller.set_workstation_mode(mode_enum)
        
        return JSONResponse(content={
            "success": result.get("success", False),
            "previous_mode": result.get("previous_mode"),
            "new_mode": result.get("new_mode"),
            "timestamp": datetime.now().isoformat()
        })
        
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
    except Exception as e:
        logger.error(f"Mode change failed: {e}")
        raise HTTPException(status_code=500, detail=f"Mode change failed: {str(e)}")


# ==================== REAL-TIME STREAMING ====================

@app.websocket("/api/stream/system-metrics")
async def stream_system_metrics(websocket: WebSocket):
    """WebSocket endpoint for real-time system metrics streaming"""
    if not streaming_manager:
        await websocket.close(code=1011, reason="Streaming manager not available")
        return
    
    await streaming_manager.handle_websocket_connection(websocket, "system-metrics")


@app.websocket("/api/stream/thermal-data")
async def stream_thermal_data(websocket: WebSocket):
    """WebSocket endpoint for real-time thermal monitoring data"""
    if not streaming_manager:
        await websocket.close(code=1011, reason="Streaming manager not available")
        return
    
    await streaming_manager.handle_websocket_connection(websocket, "thermal-data")


@app.websocket("/api/stream/container-status")
async def stream_container_status(websocket: WebSocket):
    """WebSocket endpoint for real-time container orchestration data"""
    if not streaming_manager:
        await websocket.close(code=1011, reason="Streaming manager not available")
        return
    
    await streaming_manager.handle_websocket_connection(websocket, "container-status")


@app.websocket("/api/stream/performance-metrics")
async def stream_performance_metrics(websocket: WebSocket):
    """WebSocket endpoint for real-time performance analytics"""
    if not streaming_manager:
        await websocket.close(code=1011, reason="Streaming manager not available")
        return
    
    await streaming_manager.handle_websocket_connection(websocket, "performance-metrics")


# ==================== DASHBOARD DATA APIs ====================

@app.get("/api/dashboard/gpu")
async def get_gpu_dashboard_data():
    """Get RTX 5090 dashboard data for d3.js visualizations"""
    if not workstation_controller:
        raise HTTPException(status_code=503, detail="AI Workstation system not available")
    
    try:
        insights = await workstation_controller.get_performance_insights()
        gpu_data = insights.get("hardware_specialization", {}).get("rtx5090_blackwall", {})
        
        return {
            "timestamp": datetime.now().isoformat(),
            "temperature": gpu_data.get("temperature", 0),
            "vram_usage_gb": gpu_data.get("vram_usage_gb", 0),
            "vram_total_gb": 32,
            "utilization_percent": gpu_data.get("utilization_percent", 0),
            "tensor_core_utilization": gpu_data.get("tensor_core_utilization", 0),
            "memory_bandwidth_percent": gpu_data.get("memory_bandwidth_percent", 0),
            "power_draw_watts": gpu_data.get("power_draw_watts", 0),
            "clock_speeds": gpu_data.get("clock_speeds", {}),
            "thermal_throttling_active": gpu_data.get("thermal_throttling_active", False)
        }
        
    except Exception as e:
        logger.error(f"Failed to get GPU dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"GPU data retrieval failed: {str(e)}")


@app.get("/api/dashboard/cpu")
async def get_cpu_dashboard_data():
    """Get AMD 9950X dashboard data for d3.js visualizations"""
    if not workstation_controller:
        raise HTTPException(status_code=503, detail="AI Workstation system not available")
    
    try:
        insights = await workstation_controller.get_performance_insights()
        cpu_data = insights.get("hardware_specialization", {}).get("amd_zen5", {})
        
        return {
            "timestamp": datetime.now().isoformat(),
            "core_utilization": cpu_data.get("core_utilization", [0] * 16),
            "temperature": cpu_data.get("temperature", 0),
            "aocl_library_usage": cpu_data.get("aocl_library_usage", 0),
            "memory_bandwidth_percent": cpu_data.get("memory_bandwidth_percent", 0),
            "instructions_per_second": cpu_data.get("instructions_per_second", 0),
            "cache_hit_rates": cpu_data.get("cache_hit_rates", {}),
            "numa_efficiency": cpu_data.get("numa_efficiency", 0),
            "power_draw_watts": cpu_data.get("power_draw_watts", 0)
        }
        
    except Exception as e:
        logger.error(f"Failed to get CPU dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"CPU data retrieval failed: {str(e)}")


@app.get("/api/dashboard/thermal")
async def get_thermal_dashboard_data():
    """Get thermal management data for 15-fan cooling visualization"""
    if not workstation_controller:
        raise HTTPException(status_code=503, detail="AI Workstation system not available")
    
    try:
        insights = await workstation_controller.get_performance_insights()
        thermal_data = insights.get("hardware_specialization", {}).get("thermal_intelligence", {})
        
        return {
            "timestamp": datetime.now().isoformat(),
            "component_temperatures": thermal_data.get("component_temperatures", {}),
            "fan_speeds": thermal_data.get("fan_speeds", {}),
            "cooling_efficiency": thermal_data.get("cooling_efficiency", 0),
            "thermal_throttling_risk": thermal_data.get("thermal_throttling_risk", 0),
            "predicted_temperature_trend": thermal_data.get("predicted_temperature_trend", []),
            "airflow_optimization_score": thermal_data.get("airflow_optimization_score", 0)
        }
        
    except Exception as e:
        logger.error(f"Failed to get thermal dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Thermal data retrieval failed: {str(e)}")


@app.get("/api/dashboard/containers")
async def get_container_dashboard_data():
    """Get container orchestration data for service flow visualization"""
    if not workstation_controller:
        raise HTTPException(status_code=503, detail="AI Workstation system not available")
    
    try:
        insights = await workstation_controller.get_performance_insights()
        container_data = insights.get("container_intelligence", {})
        
        return {
            "timestamp": datetime.now().isoformat(),
            "services": container_data.get("services", []),
            "resource_flows": container_data.get("resource_flows", []),
            "service_interactions": container_data.get("service_interactions", []),
            "load_balancing_efficiency": container_data.get("load_balancing_efficiency", 0),
            "resource_contention": container_data.get("resource_contention", []),
            "inference_patterns": container_data.get("inference_patterns", {})
        }
        
    except Exception as e:
        logger.error(f"Failed to get container dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Container data retrieval failed: {str(e)}")


# ==================== MAIN APPLICATION ====================

if __name__ == "__main__":
    # Run the server with WebSocket support
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ws_max_size=16777216,  # 16MB for large WebSocket messages
        log_level="info"
    )