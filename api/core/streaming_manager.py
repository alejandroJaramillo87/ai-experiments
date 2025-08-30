"""
Streaming Manager - Real-time WebSocket Streaming
=================================================

Pure WebSocket connection management with observer pattern for real-time
system consciousness updates. Handles subscription management and broadcasting
with zero business logic.

Key principles:
- Observer pattern for real-time updates
- Pure WebSocket management - no business logic
- Subscription-based filtering for efficient streaming
- Graceful connection handling and cleanup
"""

import json
import logging
import asyncio
from typing import Dict, List, Set, Optional, Callable, Any
from datetime import datetime
from collections import defaultdict

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SubscriptionConfig(BaseModel):
    """Configuration for WebSocket metric subscriptions."""
    metric_types: List[str] = Field(default_factory=list, description="Types of metrics to stream")
    update_interval: int = Field(default=1000, description="Update interval in milliseconds")
    include_events: bool = Field(default=True, description="Include system events in stream")
    include_changes: bool = Field(default=True, description="Include system changes in stream")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Optional filters for streaming data")


class MetricUpdate(BaseModel):
    """Structure for streaming metric updates."""
    metric_type: str
    data: Any
    timestamp: datetime
    update_type: str  # "current_state", "change", "event"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StreamingManager:
    """
    Pure WebSocket streaming manager with observer pattern.
    
    Manages WebSocket connections and broadcasts real-time updates from
    system consciousness with zero business logic.
    """
    
    def __init__(self, consciousness=None):
        """Initialize streaming manager with consciousness dependency."""
        self.consciousness = consciousness
        self.active_connections: Dict[WebSocket, SubscriptionConfig] = {}
        self.metric_subscribers: Dict[str, Set[WebSocket]] = defaultdict(set)
        self.update_tasks: Dict[WebSocket, asyncio.Task] = {}
        
    async def handle_websocket_connection(self, websocket: WebSocket):
        """
        Handle new WebSocket connection with pure connection management.
        
        Pure WebSocket handling - no business logic.
        """
        try:
            await websocket.accept()
            logger.info(f"WebSocket connection accepted: {websocket.client}")
            
            # Wait for subscription configuration
            subscription_config = await self._receive_subscription_config(websocket)
            
            if subscription_config:
                await self._add_subscriber(websocket, subscription_config)
                
                # Start streaming updates
                await self._start_streaming(websocket, subscription_config)
            
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {websocket.client}")
        except Exception as e:
            logger.error(f"Error handling WebSocket connection: {e}")
        finally:
            await self._cleanup_connection(websocket)
    
    async def _receive_subscription_config(self, websocket: WebSocket) -> Optional[SubscriptionConfig]:
        """Receive subscription configuration from client."""
        try:
            # Wait for subscription message
            data = await websocket.receive_text()
            config_data = json.loads(data)
            
            if config_data.get("type") == "subscribe":
                return SubscriptionConfig(**config_data.get("config", {}))
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Expected subscription configuration"
                }))
                return None
                
        except json.JSONDecodeError:
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": "Invalid JSON configuration"
            }))
            return None
        except Exception as e:
            logger.error(f"Error receiving subscription config: {e}")
            return None
    
    async def _add_subscriber(self, websocket: WebSocket, config: SubscriptionConfig):
        """Add WebSocket to subscription management."""
        self.active_connections[websocket] = config
        
        # Add to metric-specific subscribers
        for metric_type in config.metric_types:
            self.metric_subscribers[metric_type].add(websocket)
        
        # Subscribe to consciousness updates if available
        if self.consciousness and hasattr(self.consciousness, 'subscribe_to_updates'):
            await self._subscribe_to_consciousness_updates(websocket, config)
        
        logger.info(f"Added WebSocket subscriber for metrics: {config.metric_types}")
    
    async def _subscribe_to_consciousness_updates(self, websocket: WebSocket, config: SubscriptionConfig):
        """Subscribe to consciousness updates for this WebSocket."""
        try:
            # Create update callback for this connection
            async def update_callback(metric_update: Any):
                await self._broadcast_to_subscriber(websocket, metric_update, config)
            
            # Subscribe to consciousness updates (pure delegation)
            self.consciousness.subscribe_to_updates(
                callback=update_callback,
                metric_types=config.metric_types
            )
            
        except Exception as e:
            logger.error(f"Error subscribing to consciousness updates: {e}")
    
    async def _start_streaming(self, websocket: WebSocket, config: SubscriptionConfig):
        """Start streaming updates to WebSocket connection."""
        try:
            # Send initial current state if consciousness available
            if self.consciousness:
                current_state = await self._get_current_state_for_subscription(config)
                if current_state:
                    await self._send_update(websocket, {
                        "type": "current_state",
                        "data": current_state,
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Send confirmation
            await websocket.send_text(json.dumps({
                "type": "subscription_confirmed",
                "config": config.dict(),
                "timestamp": datetime.now().isoformat()
            }))
            
            # Keep connection alive and handle messages
            while True:
                try:
                    # Wait for client messages (subscription changes, ping, etc.)
                    message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    await self._handle_client_message(websocket, message)
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await websocket.send_text(json.dumps({
                        "type": "ping",
                        "timestamp": datetime.now().isoformat()
                    }))
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket streaming ended: {websocket.client}")
        except Exception as e:
            logger.error(f"Error in streaming loop: {e}")
    
    async def _get_current_state_for_subscription(self, config: SubscriptionConfig) -> Optional[Dict]:
        """Get current state data for subscription (pure delegation)."""
        try:
            if not self.consciousness:
                return None
            
            # Pure delegation to consciousness - no business logic
            result = await self.consciousness.get_current_state(
                metric_types=config.metric_types,
                filters=config.filters
            )
            
            # Use response transformer for consistent formatting
            from .response_transformer import ResponseTransformer
            return ResponseTransformer.transform_current_state(result)
            
        except Exception as e:
            logger.error(f"Error getting current state for subscription: {e}")
            return None
    
    async def _handle_client_message(self, websocket: WebSocket, message: str):
        """Handle messages from WebSocket client."""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "ping":
                # Respond to ping
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
                
            elif message_type == "update_subscription":
                # Handle subscription changes
                new_config = SubscriptionConfig(**data.get("config", {}))
                await self._update_subscription(websocket, new_config)
                
            elif message_type == "get_current":
                # Handle one-time current state request
                if self.consciousness:
                    metric_types = data.get("metric_types", [])
                    current_state = await self._get_current_state_for_metrics(metric_types)
                    await self._send_update(websocket, {
                        "type": "current_state_response",
                        "data": current_state,
                        "timestamp": datetime.now().isoformat()
                    })
                    
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from client: {message}")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    async def _update_subscription(self, websocket: WebSocket, new_config: SubscriptionConfig):
        """Update subscription configuration for WebSocket."""
        old_config = self.active_connections.get(websocket)
        
        if old_config:
            # Remove from old metric subscriptions
            for metric_type in old_config.metric_types:
                self.metric_subscribers[metric_type].discard(websocket)
        
        # Add to new subscription
        await self._add_subscriber(websocket, new_config)
        
        await websocket.send_text(json.dumps({
            "type": "subscription_updated",
            "config": new_config.dict(),
            "timestamp": datetime.now().isoformat()
        }))
    
    async def _get_current_state_for_metrics(self, metric_types: List[str]) -> Optional[Dict]:
        """Get current state for specific metrics."""
        try:
            if not self.consciousness or not metric_types:
                return None
            
            result = await self.consciousness.get_current_state(
                metric_types=metric_types
            )
            
            from .response_transformer import ResponseTransformer
            return ResponseTransformer.transform_current_state(result)
            
        except Exception as e:
            logger.error(f"Error getting current state for metrics {metric_types}: {e}")
            return None
    
    async def broadcast_metric_update(self, metric_type: str, update_data: Any):
        """
        Broadcast update to all subscribers of a metric type.
        
        Observer pattern implementation - pure broadcasting.
        """
        subscribers = self.metric_subscribers.get(metric_type, set())
        
        if not subscribers:
            return
        
        update_message = {
            "type": "metric_update",
            "metric_type": metric_type,
            "data": update_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Broadcast to all subscribers concurrently
        broadcast_tasks = []
        for websocket in subscribers.copy():  # Copy to avoid modification during iteration
            task = asyncio.create_task(self._send_update(websocket, update_message))
            broadcast_tasks.append(task)
        
        if broadcast_tasks:
            await asyncio.gather(*broadcast_tasks, return_exceptions=True)
    
    async def _broadcast_to_subscriber(self, websocket: WebSocket, metric_update: Any, config: SubscriptionConfig):
        """Broadcast specific update to single subscriber."""
        try:
            # Filter update based on subscription config
            if self._should_send_update(metric_update, config):
                await self._send_update(websocket, {
                    "type": "realtime_update",
                    "data": metric_update,
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            logger.error(f"Error broadcasting to subscriber: {e}")
    
    def _should_send_update(self, metric_update: Any, config: SubscriptionConfig) -> bool:
        """Check if update should be sent based on subscription config."""
        # Pure filtering logic based on subscription config
        try:
            if hasattr(metric_update, 'metric_type'):
                return metric_update.metric_type in config.metric_types
            elif hasattr(metric_update, 'category'):
                return metric_update.category in config.metric_types
            else:
                return True  # Send if we can't determine type
        except:
            return True  # Default to sending
    
    async def _send_update(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send update message to WebSocket connection."""
        try:
            await websocket.send_text(json.dumps(message))
        except WebSocketDisconnect:
            # Connection closed, will be cleaned up
            pass
        except Exception as e:
            logger.error(f"Error sending WebSocket update: {e}")
            # Remove problematic connection
            await self._cleanup_connection(websocket)
    
    async def _cleanup_connection(self, websocket: WebSocket):
        """Clean up WebSocket connection and subscriptions."""
        try:
            # Remove from active connections
            config = self.active_connections.pop(websocket, None)
            
            if config:
                # Remove from metric subscribers
                for metric_type in config.metric_types:
                    self.metric_subscribers[metric_type].discard(websocket)
            
            # Cancel any running tasks
            task = self.update_tasks.pop(websocket, None)
            if task and not task.done():
                task.cancel()
            
            logger.info(f"Cleaned up WebSocket connection: {websocket.client}")
            
        except Exception as e:
            logger.error(f"Error cleaning up WebSocket connection: {e}")
    
    def get_connection_count(self) -> int:
        """Get number of active WebSocket connections."""
        return len(self.active_connections)
    
    def get_subscriber_count_by_metric(self) -> Dict[str, int]:
        """Get subscriber count by metric type."""
        return {metric: len(subscribers) for metric, subscribers in self.metric_subscribers.items()}