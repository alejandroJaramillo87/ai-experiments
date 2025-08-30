"""
Natural Language Processing Integration
======================================

NLP components for conversational AI integration with system consciousness.
All natural language processing logic lives in src/ with full access to
system intelligence capabilities.

Components:
- IntentClassifier: Natural language understanding for system queries
- ContextManager: Conversation state and context management
- ResponseGenerator: Human-readable response generation
- ConversationalAI: Main NLP orchestrator with system consciousness integration
"""

from .intent_classifier import IntentClassifier
from .context_manager import ContextManager  
from .response_generator import ResponseGenerator
from .conversational_ai import ConversationalAI

__all__ = [
    'IntentClassifier',
    'ContextManager',
    'ResponseGenerator', 
    'ConversationalAI'
]