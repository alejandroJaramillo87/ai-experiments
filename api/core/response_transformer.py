"""
Response Transformer - Pure HTTP Response Formatting
====================================================

Pure transformation layer that converts src/ native objects to HTTP JSON responses.
Contains zero business logic - only data format transformation for HTTP transport.

Key principles:
- Pure data transformation only
- No business logic or intelligence
- Convert consciousness responses to HTTP-friendly JSON
- Format data for frontend consumption (D3.js, chat interface)
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class ResponseTransformer:
    """
    Pure response transformation with zero business logic.
    
    Transforms src/ consciousness responses into HTTP JSON format
    suitable for frontend consumption.
    """
    
    @staticmethod
    def transform_current_state(consciousness_result) -> Dict[str, Any]:
        """
        Transform current state result from consciousness to HTTP JSON.
        
        Pure transformation - no business logic.
        """
        try:
            if not consciousness_result:
                return {"data": {}, "timestamp": datetime.now().isoformat(), "status": "no_data"}
            
            # Handle different result types from consciousness
            if hasattr(consciousness_result, 'to_dict'):
                data = consciousness_result.to_dict()
            elif hasattr(consciousness_result, '__dict__'):
                data = consciousness_result.__dict__
            elif isinstance(consciousness_result, dict):
                data = consciousness_result
            else:
                data = {"raw": str(consciousness_result)}
            
            return {
                "data": ResponseTransformer._serialize_data(data),
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "data_type": "current_state"
            }
            
        except Exception as e:
            logger.error(f"Error transforming current state response: {e}")
            return {
                "data": {},
                "timestamp": datetime.now().isoformat(),
                "status": "transformation_error",
                "error": str(e)
            }
    
    @staticmethod
    def transform_temporal_data(temporal_result) -> Dict[str, Any]:
        """
        Transform temporal query result to HTTP JSON optimized for D3.js.
        
        Pure transformation - formats temporal data for visualization.
        """
        try:
            if not temporal_result:
                return {"data": [], "timestamp": datetime.now().isoformat(), "status": "no_data"}
            
            # Handle temporal data collection
            if hasattr(temporal_result, 'deltas'):
                # Multiple SystemDelta objects
                data = [ResponseTransformer._transform_system_delta(delta) for delta in temporal_result.deltas]
            elif hasattr(temporal_result, '__iter__') and not isinstance(temporal_result, (str, dict)):
                # Iterable of temporal objects
                data = [ResponseTransformer._serialize_data(item) for item in temporal_result]
            else:
                # Single temporal object
                data = [ResponseTransformer._serialize_data(temporal_result)]
            
            # Format for D3.js time-series visualization
            formatted_data = ResponseTransformer._format_for_d3js(data)
            
            return {
                "data": formatted_data,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "data_type": "temporal_series",
                "count": len(data) if isinstance(data, list) else 1
            }
            
        except Exception as e:
            logger.error(f"Error transforming temporal data response: {e}")
            return {
                "data": [],
                "timestamp": datetime.now().isoformat(),
                "status": "transformation_error",
                "error": str(e)
            }
    
    @staticmethod
    def transform_conversation_response(conversation_result) -> Dict[str, Any]:
        """
        Transform conversational AI result to chat response format.
        
        Pure transformation for chat interface consumption.
        """
        try:
            if not conversation_result:
                return {
                    "message": "No response available",
                    "confidence": 0.0,
                    "timestamp": datetime.now().isoformat(),
                    "status": "no_response"
                }
            
            # Extract conversation components
            message = getattr(conversation_result, 'message', str(conversation_result))
            confidence = getattr(conversation_result, 'confidence', 0.5)
            data_references = getattr(conversation_result, 'data_references', [])
            suggested_actions = getattr(conversation_result, 'suggested_actions', [])
            session_id = getattr(conversation_result, 'session_id', None)
            
            return {
                "message": message,
                "confidence": confidence,
                "session_id": session_id,
                "data_references": ResponseTransformer._serialize_data(data_references),
                "suggested_actions": suggested_actions,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error transforming conversation response: {e}")
            return {
                "message": "Error processing conversation",
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat(),
                "status": "transformation_error",
                "error": str(e)
            }
    
    @staticmethod
    def _transform_system_delta(delta) -> Dict[str, Any]:
        """Transform SystemDelta object to JSON-serializable format."""
        try:
            return {
                "timestamp": delta.timestamp.isoformat() if hasattr(delta, 'timestamp') else datetime.now().isoformat(),
                "changes": ResponseTransformer._serialize_data(getattr(delta, 'raw_delta', [])),
                "events": ResponseTransformer._serialize_data(getattr(delta, 'semantic_events', [])),
                "correlations": ResponseTransformer._serialize_data(getattr(delta, 'correlations', [])),
                "metadata": ResponseTransformer._serialize_data(getattr(delta, 'metadata', {}))
            }
        except Exception as e:
            logger.error(f"Error transforming SystemDelta: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    @staticmethod
    def _format_for_d3js(data: List[Dict]) -> Dict[str, Any]:
        """
        Format temporal data specifically for D3.js consumption.
        
        Creates time-series structure optimized for visualization.
        """
        try:
            if not data:
                return {"timeseries": [], "metadata": {}}
            
            # Extract time series data
            timeseries = []
            metrics = set()
            
            for item in data:
                timestamp = item.get('timestamp', datetime.now().isoformat())
                
                # Extract numeric metrics for visualization
                if 'changes' in item:
                    for change in item.get('changes', []):
                        if isinstance(change, dict):
                            change_data = {
                                "timestamp": timestamp,
                                "metric": change.get('category', 'unknown'),
                                "value": ResponseTransformer._extract_numeric_value(change.get('new_value')),
                                "significance": change.get('significance', 0.0),
                                "change_type": change.get('change_type', 'unknown')
                            }
                            timeseries.append(change_data)
                            metrics.add(change_data['metric'])
                
                # Extract events for annotation
                if 'events' in item:
                    for event in item.get('events', []):
                        if isinstance(event, dict):
                            event_data = {
                                "timestamp": timestamp,
                                "metric": "events",
                                "value": event.get('severity', 'info'),
                                "event_type": event.get('event_type', 'unknown'),
                                "description": event.get('description', '')
                            }
                            timeseries.append(event_data)
            
            return {
                "timeseries": timeseries,
                "metadata": {
                    "metrics": list(metrics),
                    "count": len(timeseries),
                    "time_range": {
                        "start": min(item['timestamp'] for item in timeseries) if timeseries else None,
                        "end": max(item['timestamp'] for item in timeseries) if timeseries else None
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error formatting data for D3.js: {e}")
            return {"timeseries": [], "metadata": {"error": str(e)}}
    
    @staticmethod
    def _serialize_data(obj) -> Any:
        """
        Serialize complex objects to JSON-compatible format.
        
        Pure serialization with no business logic.
        """
        try:
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, (list, tuple)):
                return [ResponseTransformer._serialize_data(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: ResponseTransformer._serialize_data(value) for key, value in obj.items()}
            elif hasattr(obj, 'to_dict'):
                return ResponseTransformer._serialize_data(obj.to_dict())
            elif hasattr(obj, '__dict__'):
                return ResponseTransformer._serialize_data(obj.__dict__)
            elif hasattr(obj, '__iter__') and not isinstance(obj, str):
                return [ResponseTransformer._serialize_data(item) for item in obj]
            else:
                return str(obj)
        except Exception as e:
            logger.error(f"Error serializing object {type(obj)}: {e}")
            return str(obj)
    
    @staticmethod
    def _extract_numeric_value(value) -> Optional[float]:
        """Extract numeric value from various data types for visualization."""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                # Try to extract numbers from strings like "75°C" or "8GB"
                import re
                numbers = re.findall(r'[-+]?\d*\.?\d+', value)
                return float(numbers[0]) if numbers else None
            else:
                return None
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def transform_error_response(error: Exception, context: str = "unknown") -> Dict[str, Any]:
        """Transform error to standardized HTTP error response."""
        return {
            "status": "error",
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "context": context
            },
            "timestamp": datetime.now().isoformat()
        }