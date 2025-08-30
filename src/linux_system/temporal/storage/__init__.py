"""
Temporal Storage System
======================

Hierarchical memory architecture for storing and querying temporal system data.
Provides efficient storage, retrieval, and analysis of system evolution over time.

Storage Layers:
- Recent Buffer: Full detail for last 48 hours (fast access)
- Daily Summaries: Key events for last 90 days (compressed)  
- Pattern Memory: Long-term behavioral patterns (highly compressed)
- Search Index: Fast temporal queries across all layers

This creates a system memory that works like human memory - recent events
in full detail, older events summarized, and long-term patterns remembered.
"""

from .temporal_storage import TemporalStorage
from .recent_buffer import RecentBuffer
from .daily_aggregator import DailyAggregator
from .pattern_store import PatternStore
from .search_index import TemporalSearchIndex
from .query_engine import TemporalQueryEngine
from .compression import (
    TemporalCompressor, 
    TemporalArchiver,
    CompressionLevel,
    CompressionType,
    CompressionResult,
    CompressionStats,
    create_compression_manager,
    create_archiver
)

__all__ = [
    'TemporalStorage',
    'RecentBuffer', 
    'DailyAggregator',
    'PatternStore',
    'TemporalSearchIndex',
    'TemporalQueryEngine',
    'TemporalCompressor',
    'TemporalArchiver',
    'CompressionLevel',
    'CompressionType',
    'CompressionResult',
    'CompressionStats',
    'create_compression_manager',
    'create_archiver'
]