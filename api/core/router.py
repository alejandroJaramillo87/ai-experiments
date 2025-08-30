"""
APIRouter - Pure HTTP Routing Layer
===================================

Pure FastAPI routing with zero business logic. This router simply translates
HTTP requests to the SystemConsciousness interface and delegates all processing
to the src/ intelligence layer.

Key principles:
- Zero business logic - pure HTTP translation
- All intelligence delegated to SystemConsciousness
- Clean separation between HTTP concerns and system intelligence
- Data-driven API contracts generated from src/ capabilities
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from fastapi import FastAPI, HTTPException, Depends, Query, Path
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import types for HTTP contracts (will be auto-generated)
from ..models import (
    ChatRequest, ChatResponse,
    CurrentStateRequest, CurrentStateResponse, 
    HistoricalQueryRequest, HistoricalQueryResponse,
    MetricSubscriptionRequest, HealthCheckResponse
)

# Import core API components
from .response_transformer import ResponseTransformer
from .streaming_manager import StreamingManager

# Import consciousness interface (placeholder - will implement next)
# from src.linux_system.consciousness import SystemConsciousness


logger = logging.getLogger(__name__)


class APIRouter:
    """
    Pure HTTP routing layer with zero business logic.
    
    Translates HTTP requests to SystemConsciousness calls and formats responses.
    All intelligence, processing, and system knowledge lives in src/.
    """
    
    def __init__(self, system_consciousness=None):
        """Initialize router with consciousness dependency injection."""
        self.consciousness = system_consciousness  # Will be injected when consciousness is ready
        self.response_transformer = ResponseTransformer()
        self.streaming_manager = StreamingManager(system_consciousness)
        self.app = self._create_fastapi_app()
        
    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application with pure routing setup."""
        app = FastAPI(
            title="AI Workstation Intelligence API",
            description="Pure HTTP interface to AI workstation consciousness system",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # CORS for local development
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Svelte dev servers
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup pure routing with zero logic
        self._setup_routes(app)
        
        return app
    
    def _setup_routes(self, app: FastAPI):
        """Setup pure HTTP routes - no business logic, just delegation."""
        
        # Health check endpoint
        @app.get("/health", response_model=HealthCheckResponse)
        async def health_check():
            """Simple health check with no business logic."""
            return HealthCheckResponse(
                status="healthy",
                timestamp=datetime.now(),
                consciousness_available=self.consciousness is not None
            )
        
        # Current state data endpoints - delegate to consciousness
        @app.get("/api/current/{data_type}")
        async def get_current_data(
            data_type: str = Path(..., description="Type of data to retrieve"),
            filters: Optional[Dict[str, Any]] = Query(None, description="Optional filters")
        ) -> JSONResponse:
            """Pure routing to current state queries - zero business logic."""
            return await self._handle_current_data(data_type, filters or {})
        
        # Historical data endpoints - delegate to consciousness  
        @app.get("/api/historical/{data_type}/{time_range}")
        async def get_historical_data(
            data_type: str = Path(..., description="Type of data to retrieve"),
            time_range: str = Path(..., description="Time range (e.g., '4h', '1d', '1w')"),
            aggregation: Optional[str] = Query(None, description="Aggregation type"),
            filters: Optional[Dict[str, Any]] = Query(None, description="Optional filters")
        ) -> JSONResponse:
            """Pure routing to historical queries - zero business logic."""
            return await self._handle_historical_data(data_type, time_range, aggregation, filters or {})
        
        # Chat endpoints - delegate to consciousness
        @app.post("/api/chat", response_model=ChatResponse)
        async def chat_query(request: ChatRequest) -> ChatResponse:
            """Pure routing to conversational AI - zero business logic."""
            return await self._handle_chat_query(request)
        
        # Bulk current state - for dashboard efficiency
        @app.post("/api/current/bulk")
        async def get_bulk_current_data(request: CurrentStateRequest) -> CurrentStateResponse:
            """Pure routing to bulk current state queries - zero business logic."""
            return await self._handle_bulk_current_data(request)
        
        # WebSocket endpoint for real-time streaming
        @app.websocket("/ws/stream")
        async def websocket_stream(websocket):
            """Pure WebSocket routing - delegate to streaming manager."""
            await self.streaming_manager.handle_websocket_connection(websocket)
    
    async def _handle_current_data(self, data_type: str, filters: Dict[str, Any]) -> JSONResponse:
        """Pure delegation to consciousness - no business logic."""
        if not self.consciousness:
            raise HTTPException(status_code=503, detail="System consciousness not available")
        
        try:
            # Pure delegation - consciousness handles all logic
            result = await self.consciousness.get_current_state(
                metric_types=[data_type],
                filters=filters
            )
            
            # Pure transformation - no business logic
            response_data = self.response_transformer.transform_current_state(result)
            return JSONResponse(content=response_data)
            
        except Exception as e:
            logger.error(f"Error handling current data request for {data_type}: {e}")
            raise HTTPException(status_code=500, detail=f"Error retrieving {data_type} data")
    
    async def _handle_historical_data(self, data_type: str, time_range: str, aggregation: Optional[str], filters: Dict[str, Any]) -> JSONResponse:
        """Pure delegation to consciousness - no business logic."""
        if not self.consciousness:
            raise HTTPException(status_code=503, detail="System consciousness not available")
        
        try:
            # Pure delegation - consciousness handles all temporal logic
            result = await self.consciousness.get_historical_data(
                data_types=[data_type],
                time_range=time_range,
                aggregation=aggregation,
                filters=filters
            )
            
            # Pure transformation - no business logic
            response_data = self.response_transformer.transform_temporal_data(result)
            return JSONResponse(content=response_data)
            
        except Exception as e:
            logger.error(f"Error handling historical data request for {data_type}: {e}")
            raise HTTPException(status_code=500, detail=f"Error retrieving historical {data_type} data")
    
    async def _handle_chat_query(self, request: ChatRequest) -> ChatResponse:
        """Pure delegation to consciousness - no NLP business logic."""
        if not self.consciousness:
            raise HTTPException(status_code=503, detail="System consciousness not available")
        
        try:
            # Pure delegation - consciousness handles all NLP and conversation logic
            result = await self.consciousness.process_conversation(
                message=request.question,
                session_id=request.session_id,
                context=request.context
            )
            
            # Pure transformation - no business logic
            return ChatResponse(
                response=result.message,
                session_id=result.session_id,
                confidence=result.confidence,
                data_references=result.data_references,
                suggested_actions=result.suggested_actions,
                timestamp=result.timestamp
            )
            
        except Exception as e:
            logger.error(f"Error handling chat query: {e}")
            raise HTTPException(status_code=500, detail="Error processing chat query")
    
    async def _handle_bulk_current_data(self, request: CurrentStateRequest) -> CurrentStateResponse:
        """Pure delegation for bulk current state queries."""
        if not self.consciousness:
            raise HTTPException(status_code=503, detail="System consciousness not available")
        
        try:
            # Pure delegation - consciousness handles all logic
            result = await self.consciousness.get_current_state(
                metric_types=request.metric_types,
                filters=request.filters
            )
            
            # Pure transformation
            response_data = self.response_transformer.transform_current_state(result)
            return CurrentStateResponse(
                data=response_data,
                timestamp=datetime.now(),
                metric_types=request.metric_types
            )
            
        except Exception as e:
            logger.error(f"Error handling bulk current data request: {e}")
            raise HTTPException(status_code=500, detail="Error retrieving bulk current data")
    
    def inject_consciousness(self, consciousness):
        """Dependency injection for system consciousness."""
        self.consciousness = consciousness
        self.streaming_manager.consciousness = consciousness
        logger.info("System consciousness injected into API router")
    
    def get_app(self) -> FastAPI:
        """Get configured FastAPI application."""
        return self.app


def create_api_router(consciousness=None) -> APIRouter:
    """Factory function to create configured API router."""
    router = APIRouter(consciousness)
    logger.info("APIRouter created with pure HTTP routing")
    return router