"""
Consciousness System - AI Workstation Intelligence Core
======================================================

Unified consciousness system that provides omniscient access to all AI workstation
intelligence capabilities. This package contains:

- SystemConsciousness: Main orchestrator for unified access
- UnifiedQueryEngine: Command pattern query processing  
- DataAccessLayer: Repository pattern for current vs temporal data
- ConversationalAI: Natural language interface to system consciousness

The consciousness system bridges the gap between raw system data and intelligent
understanding, creating a unified interface for the API layer.
"""

from .data_access_layer import DataAccessLayer
# Additional imports will be added as components are implemented

__all__ = [
    'DataAccessLayer'
]