"""
Context Manager - Conversational State and Context Management
============================================================

Manages conversation context, session state, and user interaction patterns
for natural language interface to system consciousness.

Key capabilities:
- Session-based conversation context management
- Context carryover between related queries
- User preference learning and personalization
- Query history analysis and pattern recognition
- Dynamic context enrichment from system state
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import json

from .intent_classifier import ExtractedQuery, QueryIntent, SystemEntity

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    timestamp: datetime
    user_query: str
    extracted_query: ExtractedQuery
    response: str
    response_confidence: float
    system_data_referenced: List[str] = field(default_factory=list)
    follow_up_suggestions: List[str] = field(default_factory=list)
    user_satisfaction: Optional[float] = None  # Could be inferred or explicitly provided


@dataclass
class UserPreferences:
    """User preferences learned from interaction patterns."""
    preferred_entities: Dict[SystemEntity, float] = field(default_factory=dict)  # Entity frequency scores
    preferred_time_ranges: Dict[str, float] = field(default_factory=dict)       # Time range preferences  
    preferred_granularity: Dict[str, float] = field(default_factory=dict)       # Data granularity preferences
    query_complexity_preference: float = 0.5  # 0.0 = simple, 1.0 = complex
    conversation_style: str = "balanced"  # "detailed", "concise", "balanced"
    technical_level: float = 0.5  # 0.0 = basic, 1.0 = expert
    
    def update_preferences(self, conversation_turn: ConversationTurn):
        """Update preferences based on conversation turn."""
        extracted_query = conversation_turn.extracted_query
        
        # Update entity preferences
        for entity in extracted_query.entities:
            current_score = self.preferred_entities.get(entity, 0.0)
            self.preferred_entities[entity] = min(current_score + 0.1, 1.0)
        
        # Update time range preferences
        if extracted_query.time_context and extracted_query.time_context.relative_time:
            time_range = extracted_query.time_context.relative_time
            current_score = self.preferred_time_ranges.get(time_range, 0.0)
            self.preferred_time_ranges[time_range] = min(current_score + 0.1, 1.0)
        
        # Update granularity preferences
        if extracted_query.time_context and extracted_query.time_context.granularity:
            granularity = extracted_query.time_context.granularity
            current_score = self.preferred_granularity.get(granularity, 0.0)
            self.preferred_granularity[granularity] = min(current_score + 0.1, 1.0)
        
        # Adjust technical level based on query complexity
        query_complexity = self._assess_query_complexity(extracted_query)
        self.technical_level = (self.technical_level * 0.9) + (query_complexity * 0.1)
        
        # Adjust complexity preference
        if conversation_turn.response_confidence > 0.8 and conversation_turn.user_satisfaction and conversation_turn.user_satisfaction > 0.7:
            # User was satisfied with this complexity level
            self.query_complexity_preference = (self.query_complexity_preference * 0.9) + (query_complexity * 0.1)
    
    def _assess_query_complexity(self, extracted_query: ExtractedQuery) -> float:
        """Assess complexity of query (0.0 = simple, 1.0 = complex)."""
        complexity = 0.0
        
        # More entities = more complex
        complexity += len(extracted_query.entities) * 0.2
        
        # Certain intents are more complex
        if extracted_query.intent in [QueryIntent.ANALYSIS, QueryIntent.TROUBLESHOOTING, QueryIntent.OPTIMIZATION]:
            complexity += 0.3
        
        # Time context adds complexity
        if extracted_query.time_context:
            complexity += 0.2
        
        # Multiple parameters add complexity
        complexity += len(extracted_query.query_parameters) * 0.1
        
        return min(complexity, 1.0)


@dataclass
class ConversationContext:
    """Context information for ongoing conversation."""
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    # Conversation history
    conversation_history: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # Current context state
    current_entities: List[SystemEntity] = field(default_factory=list)
    current_time_context: Optional[str] = None
    current_focus: Optional[str] = None  # "gpu_analysis", "container_monitoring", etc.
    
    # Referenced system data
    recent_data_references: Dict[str, Any] = field(default_factory=dict)
    system_state_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # User preferences
    preferences: UserPreferences = field(default_factory=UserPreferences)
    
    # Context metadata
    topic_thread: List[str] = field(default_factory=list)  # Track conversation topics
    pending_clarifications: List[str] = field(default_factory=list)
    
    def add_conversation_turn(self, turn: ConversationTurn):
        """Add a conversation turn and update context."""
        self.conversation_history.append(turn)
        self.last_activity = turn.timestamp
        
        # Update current context based on this turn
        self._update_current_context(turn)
        
        # Update user preferences
        self.preferences.update_preferences(turn)
        
        # Track topic evolution
        self._update_topic_thread(turn)
    
    def _update_current_context(self, turn: ConversationTurn):
        """Update current context based on conversation turn."""
        extracted_query = turn.extracted_query
        
        # Update current entities (weighted by recency)
        new_entities = []
        for entity in self.current_entities:
            # Keep existing entities with reduced weight
            new_entities.append(entity)
        
        # Add new entities
        for entity in extracted_query.entities:
            if entity not in new_entities:
                new_entities.append(entity)
        
        # Keep only most recent entities (limit to 5)
        self.current_entities = new_entities[-5:]
        
        # Update time context
        if extracted_query.time_context and extracted_query.time_context.relative_time:
            self.current_time_context = extracted_query.time_context.relative_time
        
        # Update focus based on intent and entities
        self.current_focus = self._determine_focus(extracted_query)
    
    def _determine_focus(self, extracted_query: ExtractedQuery) -> str:
        """Determine conversation focus from extracted query."""
        intent = extracted_query.intent
        entities = extracted_query.entities
        
        # Map intent and entities to focus areas
        if intent == QueryIntent.TROUBLESHOOTING:
            if SystemEntity.GPU in entities:
                return "gpu_troubleshooting"
            elif SystemEntity.CONTAINERS in entities:
                return "container_troubleshooting"
            else:
                return "system_troubleshooting"
        
        elif intent == QueryIntent.ANALYSIS:
            if SystemEntity.PERFORMANCE in entities:
                return "performance_analysis"
            elif SystemEntity.MEMORY in entities:
                return "memory_analysis"
            else:
                return "system_analysis"
        
        elif intent == QueryIntent.MONITORING:
            if SystemEntity.GPU in entities:
                return "gpu_monitoring"
            elif SystemEntity.CONTAINERS in entities:
                return "container_monitoring"
            else:
                return "system_monitoring"
        
        elif intent == QueryIntent.OPTIMIZATION:
            return "system_optimization"
        
        else:
            return "general_inquiry"
    
    def _update_topic_thread(self, turn: ConversationTurn):
        """Update topic thread tracking."""
        current_topic = self._extract_topic(turn)
        
        if current_topic:
            # Add to topic thread
            if not self.topic_thread or self.topic_thread[-1] != current_topic:
                self.topic_thread.append(current_topic)
                
                # Keep topic thread manageable
                if len(self.topic_thread) > 10:
                    self.topic_thread = self.topic_thread[-10:]
    
    def _extract_topic(self, turn: ConversationTurn) -> Optional[str]:
        """Extract topic from conversation turn."""
        extracted_query = turn.extracted_query
        
        # Determine topic from intent and entities
        primary_entity = extracted_query.entities[0] if extracted_query.entities else None
        
        if primary_entity:
            return f"{extracted_query.intent.value}_{primary_entity.value}"
        else:
            return extracted_query.intent.value
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context state."""
        return {
            "session_id": self.session_id,
            "conversation_turns": len(self.conversation_history),
            "current_entities": [e.value for e in self.current_entities],
            "current_focus": self.current_focus,
            "current_time_context": self.current_time_context,
            "topic_thread": self.topic_thread[-3:],  # Last 3 topics
            "technical_level": self.preferences.technical_level,
            "last_activity": self.last_activity.isoformat()
        }


class ContextManager:
    """
    Conversational context and state management for system consciousness.
    
    Manages conversation sessions, user preferences, and contextual understanding
    to enable natural, coherent interactions with the system intelligence.
    """
    
    def __init__(self, 
                 session_timeout_minutes: int = 60,
                 max_active_sessions: int = 100):
        """Initialize context manager."""
        self.session_timeout_minutes = session_timeout_minutes
        self.max_active_sessions = max_active_sessions
        
        # Active conversation contexts
        self.active_contexts: Dict[str, ConversationContext] = {}
        
        # Global user patterns
        self.global_patterns = defaultdict(int)
        
        logger.info(f"ContextManager initialized with {session_timeout_minutes}min timeout")
    
    def get_or_create_context(self, session_id: str, user_id: Optional[str] = None) -> ConversationContext:
        """Get existing context or create new one."""
        
        # Clean up expired sessions first
        self._cleanup_expired_sessions()
        
        if session_id in self.active_contexts:
            context = self.active_contexts[session_id]
            context.last_activity = datetime.now()
            return context
        
        # Create new context
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id
        )
        
        # Manage active session limit
        if len(self.active_contexts) >= self.max_active_sessions:
            self._evict_oldest_session()
        
        self.active_contexts[session_id] = context
        logger.info(f"Created new conversation context for session {session_id}")
        
        return context
    
    def add_conversation_turn(self, session_id: str, turn: ConversationTurn):
        """Add conversation turn to context."""
        context = self.get_or_create_context(session_id)
        context.add_conversation_turn(turn)
        
        # Update global patterns
        self._update_global_patterns(turn)
    
    def enrich_query_with_context(self, session_id: str, extracted_query: ExtractedQuery) -> ExtractedQuery:
        """Enrich extracted query with contextual information."""
        try:
            context = self.get_or_create_context(session_id)
            
            # Create enriched copy of extracted query
            enriched_query = ExtractedQuery(
                intent=extracted_query.intent,
                entities=extracted_query.entities.copy(),
                time_context=extracted_query.time_context,
                query_parameters=extracted_query.query_parameters.copy(),
                confidence=extracted_query.confidence,
                original_query=extracted_query.original_query,
                processing_notes=extracted_query.processing_notes.copy()
            )
            
            # Enrich entities with context
            enriched_query.entities = self._enrich_entities_with_context(
                extracted_query.entities, context
            )
            
            # Enrich time context with conversation history
            if not extracted_query.time_context and context.current_time_context:
                # Use context time if query doesn't have explicit time
                enriched_query.query_parameters['time_range'] = context.current_time_context
                enriched_query.processing_notes.append(f"Added time context from conversation: {context.current_time_context}")
            
            # Enrich query parameters with user preferences
            enriched_query.query_parameters = self._enrich_parameters_with_preferences(
                extracted_query.query_parameters, context.preferences
            )
            
            # Add contextual focus
            if context.current_focus:
                enriched_query.query_parameters['contextual_focus'] = context.current_focus
                enriched_query.processing_notes.append(f"Added contextual focus: {context.current_focus}")
            
            # Adjust confidence based on context
            context_confidence_boost = self._calculate_context_confidence_boost(extracted_query, context)
            enriched_query.confidence = min(extracted_query.confidence + context_confidence_boost, 1.0)
            
            return enriched_query
            
        except Exception as e:
            logger.error(f"Error enriching query with context: {e}")
            return extracted_query  # Return original query if enrichment fails
    
    def _enrich_entities_with_context(self, entities: List[SystemEntity], 
                                    context: ConversationContext) -> List[SystemEntity]:
        """Enrich entity list with contextual entities."""
        enriched_entities = entities.copy()
        
        # Add contextual entities if query is underspecified
        if len(entities) == 0 and context.current_entities:
            # Use most recent contextual entity
            enriched_entities.append(context.current_entities[-1])
        
        # Add related entities based on context
        for entity in entities:
            related_entities = self._get_related_entities(entity, context)
            for related in related_entities:
                if related not in enriched_entities:
                    enriched_entities.append(related)
        
        return enriched_entities
    
    def _get_related_entities(self, entity: SystemEntity, context: ConversationContext) -> List[SystemEntity]:
        """Get entities related to given entity based on context."""
        related = []
        
        # Define entity relationships
        relationships = {
            SystemEntity.GPU: [SystemEntity.TEMPERATURE, SystemEntity.PERFORMANCE, SystemEntity.UTILIZATION],
            SystemEntity.CONTAINERS: [SystemEntity.PERFORMANCE, SystemEntity.UTILIZATION, SystemEntity.HEALTH],
            SystemEntity.MEMORY: [SystemEntity.UTILIZATION, SystemEntity.PERFORMANCE],
            SystemEntity.CPU: [SystemEntity.UTILIZATION, SystemEntity.TEMPERATURE, SystemEntity.PERFORMANCE]
        }
        
        # Add related entities if they're in user's preferences
        if entity in relationships:
            for related_entity in relationships[entity]:
                # Check if user frequently queries this related entity
                if related_entity in context.preferences.preferred_entities:
                    if context.preferences.preferred_entities[related_entity] > 0.3:
                        related.append(related_entity)
        
        return related
    
    def _enrich_parameters_with_preferences(self, parameters: Dict[str, Any], 
                                          preferences: UserPreferences) -> Dict[str, Any]:
        """Enrich query parameters with user preferences."""
        enriched_params = parameters.copy()
        
        # Add preferred granularity if not specified
        if 'granularity' not in enriched_params and preferences.preferred_granularity:
            best_granularity = max(preferences.preferred_granularity.items(), key=lambda x: x[1])
            if best_granularity[1] > 0.3:  # Confidence threshold
                enriched_params['granularity'] = best_granularity[0]
        
        # Add preferred time range if not specified
        if 'time_range' not in enriched_params and preferences.preferred_time_ranges:
            best_time_range = max(preferences.preferred_time_ranges.items(), key=lambda x: x[1])
            if best_time_range[1] > 0.3:  # Confidence threshold
                enriched_params['time_range'] = best_time_range[0]
        
        # Adjust technical level
        enriched_params['technical_level'] = preferences.technical_level
        enriched_params['conversation_style'] = preferences.conversation_style
        
        return enriched_params
    
    def _calculate_context_confidence_boost(self, extracted_query: ExtractedQuery, 
                                          context: ConversationContext) -> float:
        """Calculate confidence boost from contextual information."""
        boost = 0.0
        
        # Boost for entity continuity
        matching_entities = set(extracted_query.entities) & set(context.current_entities)
        if matching_entities:
            boost += len(matching_entities) * 0.05
        
        # Boost for topic continuity
        current_topic = f"{extracted_query.intent.value}_{extracted_query.entities[0].value}" if extracted_query.entities else extracted_query.intent.value
        if current_topic in context.topic_thread:
            boost += 0.1
        
        # Boost for established preferences
        if extracted_query.entities:
            for entity in extracted_query.entities:
                if entity in context.preferences.preferred_entities:
                    if context.preferences.preferred_entities[entity] > 0.5:
                        boost += 0.05
        
        return min(boost, 0.3)  # Cap boost at 0.3
    
    def get_conversation_suggestions(self, session_id: str) -> List[str]:
        """Get conversation suggestions based on context."""
        context = self.get_or_create_context(session_id)
        suggestions = []
        
        # Suggestions based on current focus
        if context.current_focus:
            focus_suggestions = self._get_focus_based_suggestions(context.current_focus)
            suggestions.extend(focus_suggestions)
        
        # Suggestions based on recent entities
        if context.current_entities:
            entity_suggestions = self._get_entity_based_suggestions(context.current_entities[-1])
            suggestions.extend(entity_suggestions)
        
        # Suggestions based on conversation history
        history_suggestions = self._get_history_based_suggestions(context)
        suggestions.extend(history_suggestions)
        
        # Remove duplicates and limit
        unique_suggestions = list(dict.fromkeys(suggestions))[:5]
        
        return unique_suggestions
    
    def _get_focus_based_suggestions(self, focus: str) -> List[str]:
        """Get suggestions based on conversation focus."""
        focus_suggestions = {
            "gpu_monitoring": [
                "How is GPU temperature trending?",
                "Show me GPU utilization over time",
                "Check GPU memory usage"
            ],
            "container_troubleshooting": [
                "What container issues occurred recently?",
                "Show me container resource usage",
                "Check container health status"
            ],
            "system_analysis": [
                "Analyze system performance patterns",
                "Compare current vs historical performance",
                "Show me system bottlenecks"
            ]
        }
        
        return focus_suggestions.get(focus, [])
    
    def _get_entity_based_suggestions(self, entity: SystemEntity) -> List[str]:
        """Get suggestions based on current entity."""
        entity_suggestions = {
            SystemEntity.GPU: [
                "Check GPU thermal throttling",
                "Show me GPU process activity",
                "Compare GPU performance"
            ],
            SystemEntity.CONTAINERS: [
                "Show container resource usage",
                "Check container restart events",
                "Analyze container performance"
            ],
            SystemEntity.MEMORY: [
                "Show memory usage trends",
                "Check for memory pressure",
                "Analyze memory allocation"
            ]
        }
        
        return entity_suggestions.get(entity, [])
    
    def _get_history_based_suggestions(self, context: ConversationContext) -> List[str]:
        """Get suggestions based on conversation history."""
        suggestions = []
        
        # Look at recent queries for patterns
        if len(context.conversation_history) >= 2:
            recent_turns = list(context.conversation_history)[-2:]
            
            # If user asked about problems, suggest solutions
            if any(turn.extracted_query.intent == QueryIntent.TROUBLESHOOTING for turn in recent_turns):
                suggestions.append("How can I optimize system performance?")
                suggestions.append("Show me system health recommendations")
            
            # If user analyzed data, suggest deeper analysis
            if any(turn.extracted_query.intent == QueryIntent.ANALYSIS for turn in recent_turns):
                suggestions.append("Show me historical patterns")
                suggestions.append("Compare with previous time periods")
        
        return suggestions
    
    def _cleanup_expired_sessions(self):
        """Clean up expired conversation sessions."""
        now = datetime.now()
        expired_sessions = []
        
        for session_id, context in self.active_contexts.items():
            if now - context.last_activity > timedelta(minutes=self.session_timeout_minutes):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_contexts[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
    
    def _evict_oldest_session(self):
        """Evict oldest active session."""
        if not self.active_contexts:
            return
        
        oldest_session = min(self.active_contexts.keys(), 
                           key=lambda k: self.active_contexts[k].last_activity)
        del self.active_contexts[oldest_session]
        logger.info(f"Evicted oldest session: {oldest_session}")
    
    def _update_global_patterns(self, turn: ConversationTurn):
        """Update global usage patterns."""
        extracted_query = turn.extracted_query
        
        # Track intent patterns
        self.global_patterns[f"intent_{extracted_query.intent.value}"] += 1
        
        # Track entity patterns
        for entity in extracted_query.entities:
            self.global_patterns[f"entity_{entity.value}"] += 1
        
        # Track time context patterns
        if extracted_query.time_context and extracted_query.time_context.relative_time:
            self.global_patterns[f"time_{extracted_query.time_context.relative_time}"] += 1
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get context manager statistics."""
        return {
            "active_sessions": len(self.active_contexts),
            "session_timeout_minutes": self.session_timeout_minutes,
            "global_patterns": dict(self.global_patterns),
            "average_turns_per_session": sum(len(ctx.conversation_history) for ctx in self.active_contexts.values()) / max(len(self.active_contexts), 1)
        }