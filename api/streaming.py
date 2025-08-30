"""
Real-time WebSocket Streaming - Thin Delegation Layer

Simple delegation layer for real-time WebSocket streaming that forwards requests
to the sophisticated AI workstation consciousness system. All complex data collection,
temporal intelligence, and consciousness coordination happens in src/.

The actual streaming intelligence lives in the AIWorkstationController's
get_real_time_intelligence_stream() method which provides structured data
from all consciousness systems.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Callable

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketConnectionManager:
    """Simple WebSocket connection management for real-time streaming"""
    
    def __init__(self):
        # Connection pools by stream type
        self.connections: Dict[str, Set[WebSocket]] = {
            "system-metrics": set(),
            "thermal-data": set(), 
            "container-status": set(),
            "performance-metrics": set()
        }
        
        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocket, stream_type: str) -> bool:
        """Accept new WebSocket connection and add to appropriate pool"""
        await websocket.accept()
        
        if stream_type not in self.connections:
            await websocket.close(code=1008, reason=f"Unknown stream type: {stream_type}")
            return False
            
        self.connections[stream_type].add(websocket)
        self.connection_metadata[websocket] = {
            "stream_type": stream_type,
            "connected_at": datetime.now(),
            "message_count": 0
        }
        
        logger.info(f"WebSocket connected to {stream_type} stream. Total connections: {len(self.connections[stream_type])}")
        return True
        
    def disconnect(self, websocket: WebSocket, stream_type: str):
        """Remove WebSocket connection from pool"""
        if websocket in self.connections[stream_type]:
            self.connections[stream_type].remove(websocket)
            
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
            
        logger.info(f"WebSocket disconnected from {stream_type} stream. Remaining connections: {len(self.connections[stream_type])}")
        
    async def broadcast_to_stream(self, stream_type: str, data: Dict[str, Any]):
        """Broadcast data to all connections in a stream"""
        if stream_type not in self.connections:
            logger.warning(f"Unknown stream type: {stream_type}")
            return
            
        if not self.connections[stream_type]:
            return  # No connections to broadcast to
            
        # Serialize data once
        try:
            message = json.dumps(data, default=str)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize data for {stream_type}: {e}")
            return
            
        # Broadcast to all connections in parallel
        disconnected_connections = set()
        
        async def send_to_connection(websocket: WebSocket):
            try:
                await websocket.send_text(message)
                # Update connection metadata
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]["message_count"] += 1
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket connection: {e}")
                disconnected_connections.add(websocket)
                
        # Send to all connections concurrently
        if self.connections[stream_type]:
            await asyncio.gather(
                *[send_to_connection(ws) for ws in self.connections[stream_type].copy()],
                return_exceptions=True
            )
            
        # Clean up disconnected connections
        for ws in disconnected_connections:
            self.disconnect(ws, stream_type)
            
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about current connections"""
        return {
            "total_connections": sum(len(connections) for connections in self.connections.values()),
            "connections_by_stream": {
                stream_type: len(connections) 
                for stream_type, connections in self.connections.items()
            },
            "oldest_connection": min(
                (metadata["connected_at"] for metadata in self.connection_metadata.values()),
                default=None
            ),
            "total_messages_sent": sum(
                metadata["message_count"] for metadata in self.connection_metadata.values()
            )
        }


class RealtimeStreamingManager:
    """
    Thin real-time streaming manager that delegates to AI workstation consciousness.
    
    All complex data collection, temporal intelligence processing, and consciousness
    coordination happens in the AIWorkstationController. This layer handles only
    WebSocket management and HTTP concerns.
    """
    
    def __init__(self, workstation_controller):
        """
        Initialize with reference to the AI workstation controller.
        
        Args:
            workstation_controller: AIWorkstationController with consciousness systems
        """
        self.workstation_controller = workstation_controller
        self.connection_manager = WebSocketConnectionManager()
        
        # Streaming configuration
        self.streaming_active = False
        self.update_interval = 1.0  # 1 second updates
        self.background_tasks: List[asyncio.Task] = []
        
        # Stream type mapping to consciousness data types
        self.stream_type_mapping = {
            "system-metrics": "comprehensive",
            "thermal-data": "thermal",
            "container-status": "containers", 
            "performance-metrics": "predictions"
        }
        
        logger.info("Realtime streaming manager initialized (thin delegation layer)")
        
    async def start_streaming(self):
        """Start background streaming tasks"""
        if self.streaming_active:
            logger.warning("Streaming already active")
            return
            
        self.streaming_active = True
        logger.info("Starting real-time streaming system...")
        
        # Start background streaming task for each data type
        for stream_type in self.stream_type_mapping.keys():
            task = asyncio.create_task(self._streaming_loop(stream_type))
            self.background_tasks.append(task)
            
        logger.info(f"Started {len(self.background_tasks)} streaming tasks")
        
    async def stop_streaming(self):
        """Stop all streaming tasks"""
        logger.info("Stopping real-time streaming system...")
        self.streaming_active = False
        
        # Cancel all background tasks
        for task in self.background_tasks:
            task.cancel()
            
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        logger.info("Real-time streaming stopped")
        
    async def handle_websocket_connection(self, websocket: WebSocket, stream_type: str):
        """Handle individual WebSocket connection lifecycle"""
        if not await self.connection_manager.connect(websocket, stream_type):
            return
            
        try:
            # Send initial data immediately upon connection
            initial_data = await self._get_stream_data(stream_type)
            await websocket.send_text(json.dumps(initial_data, default=str))
            
            # Keep connection alive and handle client messages
            while True:
                try:
                    # Wait for client messages (heartbeat, configuration, etc.)
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    
                    # Handle client configuration messages
                    try:
                        client_message = json.loads(data)
                        await self._handle_client_message(websocket, stream_type, client_message)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from client: {data}")
                        
                except asyncio.TimeoutError:
                    # Send ping to check if connection is still alive
                    await websocket.ping()
                    
        except WebSocketDisconnect:
            logger.info(f"Client disconnected from {stream_type} stream")
        except Exception as e:
            logger.error(f"WebSocket error for {stream_type}: {e}")
        finally:
            self.connection_manager.disconnect(websocket, stream_type)
            
    async def _streaming_loop(self, stream_type: str):
        """Background streaming loop for a specific data type"""
        logger.info(f"Starting streaming loop for {stream_type}")
        
        while self.streaming_active:
            try:
                # Skip if no connections for this stream type
                if not self.connection_manager.connections[stream_type]:
                    await asyncio.sleep(self.update_interval)
                    continue
                    
                # Get data by delegating to consciousness system
                data = await self._get_stream_data(stream_type)
                
                # Add metadata
                data.update({
                    "stream_type": stream_type,
                    "timestamp": datetime.now().isoformat(),
                    "server_time": datetime.now().timestamp()
                })
                
                # Broadcast to all connections
                await self.connection_manager.broadcast_to_stream(stream_type, data)
                
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in streaming loop for {stream_type}: {e}")
                await asyncio.sleep(self.update_interval * 2)  # Back off on error
                
        logger.info(f"Streaming loop stopped for {stream_type}")
        
    async def _get_stream_data(self, stream_type: str) -> Dict[str, Any]:
        """Get stream data by delegating to consciousness system"""
        if not self.workstation_controller:
            return {"error": "AI workstation controller not available"}
        
        # Map API stream type to consciousness data type
        consciousness_stream_type = self.stream_type_mapping.get(stream_type, "comprehensive")
        
        try:
            # Delegate to the sophisticated consciousness system
            return await self.workstation_controller.get_real_time_intelligence_stream(consciousness_stream_type)
            
        except Exception as e:
            logger.error(f"Failed to get stream data from consciousness system: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "stream_type": stream_type,
                "delegation_failed": True
            }
        
    async def _handle_client_message(self, websocket: WebSocket, stream_type: str, message: Dict[str, Any]):
        """Handle messages from WebSocket clients"""
        message_type = message.get("type")
        
        if message_type == "configure":
            # Store configuration for this connection
            config = message.get("config", {})
            if websocket in self.connection_manager.connection_metadata:
                self.connection_manager.connection_metadata[websocket]["config"] = config
                
        elif message_type == "ping":
            # Respond to client ping
            await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}))
            
        elif message_type == "get_stats":
            # Send connection statistics
            stats = self.connection_manager.get_connection_stats()
            await websocket.send_text(json.dumps({"type": "stats", "data": stats}))
            
    def get_streaming_status(self) -> Dict[str, Any]:
        """Get current streaming system status"""
        return {
            "streaming_active": self.streaming_active,
            "active_tasks": len(self.background_tasks),
            "update_interval": self.update_interval,
            "connection_stats": self.connection_manager.get_connection_stats(),
            "data_collectors": list(self.stream_type_mapping.keys()),
            "delegation_layer": "thin",
            "controller_available": self.workstation_controller is not None
        }