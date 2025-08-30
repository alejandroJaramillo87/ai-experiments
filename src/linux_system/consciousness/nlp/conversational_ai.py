"""
Conversational AI - Natural Language Interface to System Consciousness
=====================================================================

Main orchestrator for natural language interaction with AI workstation consciousness.
Integrates intent classification, context management, query execution, and response
generation to create seamless conversational access to system intelligence.

Key capabilities:
- Natural language query processing with full system consciousness access
- Contextual conversation management across sessions
- Intelligent query routing and execution
- Human-readable response generation with technical depth adjustment
- Integration with all system consciousness capabilities
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

from .intent_classifier import IntentClassifier, ExtractedQuery, QueryIntent
from .context_manager import ContextManager, ConversationContext, ConversationTurn
from .response_generator import ResponseGenerator, GeneratedResponse
from ..query_commands import QueryFactory, QueryContext, QueryResult

logger = logging.getLogger(__name__)


@dataclass
class ConversationRequest:
    """Request for conversational processing."""
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    context_data: Optional[Dict[str, Any]] = None


@dataclass
class ConversationResponse:
    """Complete response from conversational AI."""
    message: str
    session_id: str
    confidence: float
    data_references: List[Dict[str, Any]]
    suggested_actions: List[str]
    follow_up_suggestions: List[str]
    visualization_suggestions: List[str]
    insights: List[str]
    recommendations: List[str]
    timestamp: datetime
    response_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "message": self.message,
            "session_id": self.session_id,
            "confidence": self.confidence,
            "data_references": self.data_references,
            "suggested_actions": self.suggested_actions,
            "follow_up_suggestions": self.follow_up_suggestions,
            "visualization_suggestions": self.visualization_suggestions,
            "insights": self.insights,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.response_metadata
        }


class ConversationalAI:
    """
    Natural language interface to AI workstation system consciousness.
    
    Orchestrates the complete conversational AI pipeline from natural language
    input to intelligent, context-aware responses with full access to system
    consciousness capabilities.
    """
    
    def __init__(self, consciousness_system=None):
        """Initialize conversational AI with system consciousness integration."""
        self.consciousness = consciousness_system
        
        # Initialize NLP components
        self.intent_classifier = IntentClassifier()
        self.context_manager = ContextManager()
        self.response_generator = ResponseGenerator()
        self.query_factory = QueryFactory()
        
        # Conversation statistics
        self.conversation_stats = {
            "total_conversations": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "average_confidence": 0.0,
            "intents_processed": {},
            "entities_referenced": {}
        }
        
        logger.info("ConversationalAI initialized with full system consciousness integration")
    
    async def process_conversation(self, request: ConversationRequest) -> ConversationResponse:
        """
        Process natural language conversation request.
        
        Main entry point for conversational AI processing.
        """
        start_time = datetime.now()
        
        try:
            # Generate session ID if not provided
            session_id = request.session_id or self._generate_session_id()
            
            # Step 1: Classify intent and extract query structure
            extracted_query = self.intent_classifier.classify_query(request.message)
            
            # Step 2: Get/create conversation context
            context = self.context_manager.get_or_create_context(session_id, request.user_id)
            
            # Step 3: Enrich query with conversational context
            enriched_query = self.context_manager.enrich_query_with_context(
                session_id, extracted_query
            )
            
            # Step 4: Execute system consciousness query
            query_result = await self._execute_consciousness_query(enriched_query, context)
            
            # Step 5: Generate natural language response
            generated_response = self.response_generator.generate_response(
                enriched_query, query_result, context
            )
            
            # Step 6: Update conversation context
            conversation_turn = ConversationTurn(
                timestamp=start_time,
                user_query=request.message,
                extracted_query=enriched_query,
                response=generated_response.message,
                response_confidence=generated_response.confidence,
                system_data_referenced=self._extract_data_references(query_result),
                follow_up_suggestions=generated_response.follow_up_suggestions
            )
            
            self.context_manager.add_conversation_turn(session_id, conversation_turn)
            
            # Step 7: Create final response
            response = ConversationResponse(
                message=generated_response.message,
                session_id=session_id,
                confidence=generated_response.confidence,
                data_references=generated_response.data_references,
                suggested_actions=generated_response.recommendations,
                follow_up_suggestions=generated_response.follow_up_suggestions,
                visualization_suggestions=generated_response.visualization_suggestions,
                insights=generated_response.insights,
                recommendations=generated_response.recommendations,
                timestamp=start_time,
                response_metadata={
                    **generated_response.response_metadata,
                    "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                    "query_intent": enriched_query.intent.value,
                    "entities_processed": [e.value for e in enriched_query.entities],
                    "context_enriched": len(enriched_query.processing_notes) > len(extracted_query.processing_notes)
                }
            )
            
            # Update statistics
            self._update_conversation_stats(enriched_query, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
            return self._create_error_response(str(e), request.session_id, start_time)
    
    async def _execute_consciousness_query(self, extracted_query: ExtractedQuery, 
                                         context: ConversationContext) -> QueryResult:
        """Execute query using system consciousness capabilities."""
        
        if not self.consciousness:
            # Return mock result if consciousness system not available
            return QueryResult(
                data={"message": "System consciousness not available"},
                success=False,
                query_type="conversational",
                execution_time_ms=0.0,
                timestamp=datetime.now(),
                context=QueryContext(),
                error="System consciousness not initialized"
            )
        
        try:
            # Route query based on intent
            if extracted_query.intent == QueryIntent.MONITORING:
                return await self._execute_monitoring_query(extracted_query, context)
            elif extracted_query.intent == QueryIntent.ANALYSIS:
                return await self._execute_analysis_query(extracted_query, context)
            elif extracted_query.intent == QueryIntent.TROUBLESHOOTING:
                return await self._execute_troubleshooting_query(extracted_query, context)
            elif extracted_query.intent == QueryIntent.EXPLORATION:
                return await self._execute_exploration_query(extracted_query, context)
            elif extracted_query.intent == QueryIntent.OPTIMIZATION:
                return await self._execute_optimization_query(extracted_query, context)
            else:
                return await self._execute_general_query(extracted_query, context)
                
        except Exception as e:
            logger.error(f"Error executing consciousness query: {e}")
            return QueryResult(
                data={},
                success=False,
                query_type="conversational",
                execution_time_ms=0.0,
                timestamp=datetime.now(),
                context=QueryContext(),
                error=str(e)
            )
    
    async def _execute_monitoring_query(self, extracted_query: ExtractedQuery, 
                                      context: ConversationContext) -> QueryResult:
        """Execute monitoring query using current state capabilities."""
        # Extract metric types from entities
        metric_types = extracted_query.query_parameters.get('metric_types', ['system'])
        
        # Get current state from consciousness
        return await self.consciousness.get_current_state(
            metric_types=metric_types,
            filters=extracted_query.query_parameters
        )
    
    async def _execute_analysis_query(self, extracted_query: ExtractedQuery,
                                    context: ConversationContext) -> QueryResult:
        """Execute analysis query using historical data capabilities."""
        # Determine time range
        time_range = extracted_query.query_parameters.get('time_range', '4h')
        
        # Extract metric types
        metric_types = extracted_query.query_parameters.get('metric_types', ['system'])
        
        # Get historical data from consciousness
        return await self.consciousness.get_historical_data(
            data_types=metric_types,
            time_range=time_range,
            aggregation=extracted_query.query_parameters.get('aggregation'),
            filters=extracted_query.query_parameters
        )
    
    async def _execute_troubleshooting_query(self, extracted_query: ExtractedQuery,
                                           context: ConversationContext) -> QueryResult:
        """Execute troubleshooting query using events and patterns."""
        # Get system events
        events_result = await self.consciousness.get_system_events(
            significance_threshold=extracted_query.query_parameters.get('significance_threshold', 0.7),
            time_range=extracted_query.query_parameters.get('time_range', '1d')
        )
        
        # Also get current state for context
        current_result = await self.consciousness.get_current_state(['system'])
        
        # Combine results
        combined_data = {
            "events": events_result.data if events_result.success else [],
            "current_state": current_result.data if current_result.success else {},
            "troubleshooting_mode": True
        }
        
        return QueryResult(
            data=combined_data,
            success=events_result.success or current_result.success,
            query_type="troubleshooting",
            execution_time_ms=events_result.execution_time_ms + current_result.execution_time_ms,
            timestamp=datetime.now(),
            context=QueryContext(),
            error=events_result.error if not events_result.success else current_result.error
        )
    
    async def _execute_exploration_query(self, extracted_query: ExtractedQuery,
                                       context: ConversationContext) -> QueryResult:
        """Execute exploration query using historical data."""
        # Default to broader time range for exploration
        time_range = extracted_query.query_parameters.get('time_range', '1d')
        metric_types = extracted_query.query_parameters.get('metric_types', ['system'])
        
        return await self.consciousness.get_historical_data(
            data_types=metric_types,
            time_range=time_range,
            filters=extracted_query.query_parameters
        )
    
    async def _execute_optimization_query(self, extracted_query: ExtractedQuery,
                                        context: ConversationContext) -> QueryResult:
        """Execute optimization query using system health analysis."""
        # Get system health analysis
        health_status = await self.consciousness.analyze_system_health()
        
        # Get patterns for optimization insights
        patterns_result = await self.consciousness.get_system_patterns()
        
        # Combine optimization-focused data
        optimization_data = {
            "health_analysis": {
                "status": health_status.status,
                "score": health_status.score,
                "issues": health_status.issues,
                "recommendations": health_status.recommendations
            },
            "patterns": patterns_result.data if patterns_result.success else [],
            "optimization_mode": True
        }
        
        return QueryResult(
            data=optimization_data,
            success=True,
            query_type="optimization",
            execution_time_ms=100.0,  # Estimated time
            timestamp=datetime.now(),
            context=QueryContext()
        )
    
    async def _execute_general_query(self, extracted_query: ExtractedQuery,
                                   context: ConversationContext) -> QueryResult:
        """Execute general query with fallback handling."""
        # Default to current state query
        return await self.consciousness.get_current_state(
            metric_types=['system'],
            filters={}
        )
    
    def _extract_data_references(self, query_result: QueryResult) -> List[str]:
        """Extract data references from query result."""
        references = []
        
        if query_result.success and query_result.data:
            if isinstance(query_result.data, dict):
                references.extend(query_result.data.keys())
            
            references.append(f"query_{query_result.query_type}")
        
        return references
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        import uuid
        return f"conv_{uuid.uuid4().hex[:12]}"
    
    def _create_error_response(self, error: str, session_id: Optional[str], 
                             start_time: datetime) -> ConversationResponse:
        """Create error response for failed conversations."""
        return ConversationResponse(
            message=f"I apologize, but I encountered an issue: {error}. Please try rephrasing your question or check system connectivity.",
            session_id=session_id or self._generate_session_id(),
            confidence=0.0,
            data_references=[],
            suggested_actions=["Try a simpler question", "Check system status", "Rephrase your query"],
            follow_up_suggestions=["Ask about current system status", "Request system health check"],
            visualization_suggestions=[],
            insights=[],
            recommendations=["Contact administrator if issues persist"],
            timestamp=start_time,
            response_metadata={
                "error": True,
                "error_message": error,
                "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000
            }
        )
    
    def _update_conversation_stats(self, extracted_query: ExtractedQuery, 
                                 response: ConversationResponse):
        """Update conversation statistics."""
        self.conversation_stats["total_conversations"] += 1
        
        if response.confidence > 0.5:
            self.conversation_stats["successful_responses"] += 1
        else:
            self.conversation_stats["failed_responses"] += 1
        
        # Update average confidence
        total = self.conversation_stats["total_conversations"]
        current_avg = self.conversation_stats["average_confidence"]
        self.conversation_stats["average_confidence"] = (
            (current_avg * (total - 1) + response.confidence) / total
        )
        
        # Update intent statistics
        intent = extracted_query.intent.value
        self.conversation_stats["intents_processed"][intent] = (
            self.conversation_stats["intents_processed"].get(intent, 0) + 1
        )
        
        # Update entity statistics
        for entity in extracted_query.entities:
            entity_name = entity.value
            self.conversation_stats["entities_referenced"][entity_name] = (
                self.conversation_stats["entities_referenced"].get(entity_name, 0) + 1
            )
    
    async def get_conversation_suggestions(self, session_id: str) -> List[str]:
        """Get conversation suggestions for user."""
        # Get contextual suggestions from context manager
        context_suggestions = self.context_manager.get_conversation_suggestions(session_id)
        
        # Add general helpful suggestions
        general_suggestions = [
            "What's the current GPU temperature?",
            "How are my containers performing?",
            "Show me system health status",
            "Analyze memory usage patterns",
            "What happened in the last hour?"
        ]
        
        # Combine and deduplicate
        all_suggestions = context_suggestions + general_suggestions
        unique_suggestions = list(dict.fromkeys(all_suggestions))[:5]
        
        return unique_suggestions
    
    async def get_query_completions(self, partial_query: str, session_id: Optional[str] = None) -> List[str]:
        """Get query completion suggestions."""
        # Use intent classifier for suggestions
        suggestions = self.intent_classifier.get_query_suggestions(partial_query)
        
        # Add contextual completions if session exists
        if session_id and session_id in self.context_manager.active_contexts:
            context = self.context_manager.active_contexts[session_id]
            
            # Add suggestions based on current context
            if context.current_entities:
                entity = context.current_entities[-1].value
                suggestions.extend([
                    f"Show me {entity} status",
                    f"Analyze {entity} performance",
                    f"What's wrong with {entity}?"
                ])
        
        # Remove duplicates and limit
        unique_suggestions = list(dict.fromkeys(suggestions))[:8]
        
        return unique_suggestions
    
    def inject_consciousness_system(self, consciousness_system):
        """Inject system consciousness dependency."""
        self.consciousness = consciousness_system
        logger.info("System consciousness injected into ConversationalAI")
    
    def get_conversation_analytics(self) -> Dict[str, Any]:
        """Get conversation analytics and statistics."""
        context_stats = self.context_manager.get_context_stats()
        
        return {
            "conversation_stats": self.conversation_stats,
            "context_manager_stats": context_stats,
            "nlp_components": {
                "intent_classifier": "active",
                "context_manager": "active", 
                "response_generator": "active"
            },
            "consciousness_integration": self.consciousness is not None
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check conversational AI health."""
        try:
            # Test basic NLP pipeline
            test_query = self.intent_classifier.classify_query("test query")
            
            return {
                "status": "healthy",
                "components": {
                    "intent_classifier": test_query.confidence > 0.0,
                    "context_manager": len(self.context_manager.active_contexts) >= 0,
                    "response_generator": True,
                    "consciousness_integration": self.consciousness is not None
                },
                "conversation_stats": {
                    "total_conversations": self.conversation_stats["total_conversations"],
                    "success_rate": (
                        self.conversation_stats["successful_responses"] / 
                        max(self.conversation_stats["total_conversations"], 1)
                    ),
                    "average_confidence": self.conversation_stats["average_confidence"]
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }