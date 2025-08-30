"""
Natural Language Processing API - Thin Delegation Layer

Simple delegation layer that forwards natural language queries to the sophisticated
AI workstation consciousness system. All complex processing happens in src/ while
this layer handles only HTTP/API concerns.

The actual natural language intelligence lives in:
src/linux_system/ai_workstation/natural_language_intelligence/

This approach maintains proper separation of concerns with thin API layer
and sophisticated consciousness processing in the core system.
"""

import logging
from typing import Dict, Optional, Any

from .models import ChatResponse

logger = logging.getLogger(__name__)


class NaturalLanguageProcessor:
    """
    Thin natural language processing wrapper for AI workstation consciousness.
    
    Delegates all complex natural language understanding, query routing, contextual
    processing, and response synthesis to the AIWorkstationController's integrated
    natural language intelligence system.
    """
    
    def __init__(self, workstation_controller):
        """
        Initialize with reference to the AI workstation controller.
        
        Args:
            workstation_controller: AIWorkstationController instance with integrated
                                   natural language intelligence
        """
        self.controller = workstation_controller
        
        logger.info("Natural language processor initialized (thin delegation layer)")
    
    async def process_question(self, question: str, session_id: Optional[str] = None) -> ChatResponse:
        """
        Process a natural language question by delegating to consciousness system.
        
        Args:
            question: Natural language question from user
            session_id: Optional session ID for conversational continuity
            
        Returns:
            ChatResponse with natural language answer and supporting data
        """
        try:
            # Delegate to the sophisticated consciousness system
            result = await self.controller.process_natural_language_query(
                query=question,
                session_id=session_id,
                context=None
            )
            
            # Convert to ChatResponse format expected by API
            return ChatResponse(
                answer=result.get('answer', ''),
                confidence=result.get('confidence', 0.0),
                structured_data=result.get('structured_data', {}),
                visualizations=result.get('visualizations', []),
                follow_up_suggestions=result.get('follow_up_suggestions', []),
                timestamp=result.get('timestamp'),
                processing_time_ms=result.get('processing_time_ms', 0.0)
            )
            
        except Exception as e:
            logger.error(f"Natural language processing delegation failed: {e}")
            
            # Return error response in expected format
            return ChatResponse(
                answer=f"I encountered an error processing your question: {str(e)}. Please try rephrasing your question.",
                confidence=0.0,
                structured_data={'error': str(e), 'delegation_failed': True},
                visualizations=[],
                follow_up_suggestions=[
                    "Try asking about GPU performance",
                    "Ask about system status",
                    "Inquire about thermal management"
                ],
                timestamp=None,
                processing_time_ms=0.0
            )
    
    def get_processor_status(self) -> Dict[str, Any]:
        """Get status of the natural language processor"""
        if not self.controller:
            return {"status": "no_controller", "delegation_layer": "thin"}
        
        # Delegate status request to consciousness controller
        orchestrator_status = self.controller.get_natural_language_orchestrator_status()
        
        return {
            "status": "active_delegation",
            "delegation_layer": "thin",
            "controller_available": True,
            "orchestrator_status": orchestrator_status
        }