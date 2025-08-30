"""
Phase 4: Natural Language Intelligence Interface

Advanced natural language interface for the AI Workstation Consciousness System.
Creates digital symbiosis by mapping natural language queries to sophisticated
machine consciousness systems.

Components:
- QueryIntelligenceRouter: Maps queries to appropriate consciousness systems
- ContextualQueryProcessor: Leverages temporal intelligence for deep context
- ResponseSynthesizer: Converts system insights to natural language
- IntentUnderstandingEngine: ML-based intent classification and understanding

This system enables intuitive human-machine interaction with the sophisticated
consciousness infrastructure, providing contextual, predictive, and causal
responses based on real system intelligence.

Author: AI Workstation Intelligence System
"""

from .query_intelligence_router import QueryIntelligenceRouter
from .contextual_query_processor import ContextualQueryProcessor
from .response_synthesizer import ResponseSynthesizer
from .intent_understanding_engine import IntentUnderstandingEngine, QueryIntent, SystemComponent
from .natural_language_orchestrator import NaturalLanguageOrchestrator

__all__ = [
    'QueryIntelligenceRouter',
    'ContextualQueryProcessor', 
    'ResponseSynthesizer',
    'IntentUnderstandingEngine',
    'QueryIntent',
    'SystemComponent',
    'NaturalLanguageOrchestrator'
]

__version__ = '1.0.0'