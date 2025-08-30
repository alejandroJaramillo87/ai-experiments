"""
Event Extraction Framework
===========================

Transforms raw system changes into semantic events with causal understanding,
contextual enrichment, and predictive effects. This module bridges the gap
between raw change detection and higher-level system intelligence.

The event extraction framework operates on the principle that individual
changes often combine to form meaningful events that have semantic significance
beyond their constituent parts. For example:

- GPU temperature + process spawn + memory increase = ML training event
- Package installation + environment creation = development setup event  
- Multiple service restarts + error logs = system stability issue

Key Components:
- SystemEventExtractor: Base framework for event extraction
- EventContext: Rich contextual information for events
- CausalAnalyzer: Identifies cause-effect relationships
- EffectPredictor: Predicts likely outcomes of events
- Specialized extractors for different event types
"""

from .base_extractor import SystemEventExtractor, EventContext
from .causal_analyzer import CausalAnalyzer
from .effect_predictor import EffectPredictor

__all__ = [
    'SystemEventExtractor',
    'EventContext', 
    'CausalAnalyzer',
    'EffectPredictor'
]