"""
API Core Components
==================

Core API infrastructure components for pure HTTP translation layer:

- router.py: Pure HTTP routing with zero business logic
- response_transformer.py: Transform src/ responses to HTTP JSON
- streaming_manager.py: WebSocket real-time streaming management  
- model_generator.py: Auto-generate Pydantic models from src/ collectors
"""

from .router import APIRouter, create_api_router

__all__ = [
    'APIRouter',
    'create_api_router'
]