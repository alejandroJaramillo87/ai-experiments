"""
Natural Language Orchestrator

Main orchestrator for the AI Workstation Natural Language Intelligence system.
Coordinates intent understanding, query routing, contextual processing, and 
response synthesis to create true digital symbiosis between human language
and machine consciousness.

This orchestrator enables sophisticated natural language interaction with
the complete AI workstation consciousness infrastructure.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from .intent_understanding_engine import IntentUnderstandingEngine, QueryContext
from .query_intelligence_router import QueryIntelligenceRouter, RoutingDecision, IntelligenceSystem
from .contextual_query_processor import ContextualQueryProcessor, ProcessedQuery
from .response_synthesizer import ResponseSynthesizer, SynthesizedResponse

logger = logging.getLogger(__name__)


class NaturalLanguageOrchestrator:
    """
    Main orchestrator for natural language intelligence in AI workstation consciousness.
    
    Provides the primary interface between human natural language and the sophisticated
    consciousness systems (Container Consciousness, Hardware Specialization, 
    Multi-Model Oracle, Temporal Intelligence).
    
    Creates digital symbiosis by intelligently routing queries, gathering contextual
    insights, and synthesizing human-readable responses from machine consciousness.
    """
    
    def __init__(self, consciousness_controller=None):
        self.consciousness_controller = consciousness_controller
        
        # Initialize intelligence components
        self.intent_engine = IntentUnderstandingEngine()
        self.query_router = QueryIntelligenceRouter()
        self.context_processor = ContextualQueryProcessor()
        self.response_synthesizer = ResponseSynthesizer()
        
        # Session and context management
        self.session_contexts = {}
        self.query_history = {}
        self.performance_metrics = {
            'queries_processed': 0,
            'average_processing_time': 0.0,
            'success_rate': 0.0
        }
        
        logger.info("Natural Language Orchestrator initialized for AI workstation consciousness")
    
    async def process_natural_language_query(self, query: str, 
                                           session_id: Optional[str] = None,
                                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main entry point for processing natural language queries about the AI workstation.
        
        This method orchestrates the complete pipeline from natural language understanding
        to consciousness system consultation to human-readable response generation.
        
        Args:
            query: Natural language question from user
            session_id: Optional session ID for conversational continuity
            context: Optional additional context for query processing
            
        Returns:
            Complete response with natural language answer, confidence, visualizations,
            follow-up suggestions, and technical insights
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing natural language query: '{query[:100]}...'")
            
            # Phase 1: Intent Understanding
            session_context = self._get_session_context(session_id)
            query_context = self.intent_engine.understand_query(query, session_context)
            
            logger.info(f"Intent: {query_context.intent.value}, Confidence: {query_context.confidence:.2f}")
            
            # Phase 2: Intelligence Routing
            routing_decision = self.query_router.route_query(query_context)
            intelligence_queries = self.query_router.create_intelligence_queries(
                query_context, routing_decision
            )
            
            logger.info(f"Routed to {routing_decision.primary_system.value} + {len(routing_decision.secondary_systems)} secondary systems")
            
            # Phase 3: Contextual Processing
            processed_query = await self.context_processor.process_with_context(
                query_context, routing_decision, intelligence_queries, self.consciousness_controller
            )
            
            logger.info(f"Context depth: {processed_query.contextual_enrichment.context_depth}")
            
            # Phase 4: Consciousness System Consultation
            consciousness_results = await self._consult_consciousness_systems(
                processed_query, routing_decision
            )
            
            # Phase 5: Response Synthesis
            processing_time = (datetime.now() - start_time).total_seconds()
            synthesized_response = await self.response_synthesizer.synthesize_response(
                processed_query, consciousness_results, processing_time
            )
            
            # Update session context and metrics
            self._update_session_context(session_id, query_context, synthesized_response)
            self._update_performance_metrics(processing_time, True)
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s with confidence {synthesized_response.confidence:.2f}")
            
            # Return response in API-compatible format
            return {
                'answer': synthesized_response.primary_answer,
                'confidence': synthesized_response.confidence,
                'structured_data': {
                    'intent': query_context.intent.value,
                    'components': [c.value for c in query_context.components],
                    'routing_decision': {
                        'primary_system': routing_decision.primary_system.value,
                        'secondary_systems': [s.value for s in routing_decision.secondary_systems],
                        'execution_strategy': routing_decision.execution_strategy
                    },
                    'context_depth': processed_query.contextual_enrichment.context_depth,
                    'consciousness_results': consciousness_results
                },
                'visualizations': synthesized_response.visualizations,
                'follow_up_suggestions': synthesized_response.follow_up_suggestions,
                'timestamp': datetime.now(),
                'processing_time_ms': processing_time * 1000,
                'technical_insights': synthesized_response.technical_insights,
                'supporting_details': synthesized_response.supporting_details
            }
            
        except Exception as e:
            logger.error(f"Error processing natural language query: {e}")
            self._update_performance_metrics((datetime.now() - start_time).total_seconds(), False)
            
            # Return error response
            return {
                'answer': f"I encountered an issue processing your question: {str(e)}. Please try rephrasing or asking something more specific about your AI workstation.",
                'confidence': 0.0,
                'structured_data': {'error': str(e)},
                'visualizations': [],
                'follow_up_suggestions': [
                    "Try asking about GPU performance",
                    "Ask about thermal management", 
                    "Request system status",
                    "Inquire about AI service orchestration"
                ],
                'timestamp': datetime.now(),
                'processing_time_ms': (datetime.now() - start_time).total_seconds() * 1000
            }
    
    async def _consult_consciousness_systems(self, processed_query: ProcessedQuery,
                                           routing_decision: RoutingDecision) -> Dict[IntelligenceSystem, Dict[str, Any]]:
        """
        Consult the appropriate AI workstation consciousness systems based on routing decision.
        
        This method implements the execution strategy (sequential, parallel, hierarchical)
        to gather insights from Container Consciousness, Hardware Specialization,
        Multi-Model Oracle, and Temporal Intelligence systems.
        """
        consciousness_results = {}
        
        if not self.consciousness_controller:
            logger.warning("No consciousness controller available - returning mock results")
            return await self._generate_mock_consciousness_results(routing_decision)
        
        try:
            # Execute based on routing strategy
            if routing_decision.execution_strategy == 'sequential':
                consciousness_results = await self._execute_sequential_consultation(processed_query, routing_decision)
            elif routing_decision.execution_strategy == 'parallel':
                consciousness_results = await self._execute_parallel_consultation(processed_query, routing_decision)
            elif routing_decision.execution_strategy == 'hierarchical':
                consciousness_results = await self._execute_hierarchical_consultation(processed_query, routing_decision)
            else:
                # Default to sequential
                consciousness_results = await self._execute_sequential_consultation(processed_query, routing_decision)
                
        except Exception as e:
            logger.error(f"Error consulting consciousness systems: {e}")
            consciousness_results = await self._generate_fallback_results(routing_decision, e)
        
        return consciousness_results
    
    async def _execute_sequential_consultation(self, processed_query: ProcessedQuery,
                                             routing_decision: RoutingDecision) -> Dict[IntelligenceSystem, Dict[str, Any]]:
        """Execute consciousness system consultation sequentially, passing context between systems"""
        results = {}
        context_accumulator = {}
        
        # Start with primary system
        primary_result = await self._consult_single_system(
            routing_decision.primary_system, processed_query, context_accumulator
        )
        results[routing_decision.primary_system] = primary_result
        context_accumulator['primary_result'] = primary_result
        
        # Consult secondary systems with accumulated context
        for system in routing_decision.secondary_systems:
            secondary_result = await self._consult_single_system(
                system, processed_query, context_accumulator
            )
            results[system] = secondary_result
            context_accumulator[f'{system.value}_result'] = secondary_result
        
        return results
    
    async def _execute_parallel_consultation(self, processed_query: ProcessedQuery,
                                           routing_decision: RoutingDecision) -> Dict[IntelligenceSystem, Dict[str, Any]]:
        """Execute consciousness system consultation in parallel for faster response"""
        systems_to_consult = [routing_decision.primary_system] + routing_decision.secondary_systems
        
        # Create consultation tasks
        tasks = []
        for system in systems_to_consult:
            task = self._consult_single_system(system, processed_query, {})
            tasks.append((system, task))
        
        # Execute all consultations in parallel
        results = {}
        completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for (system, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"Error consulting {system.value}: {result}")
                results[system] = {'error': str(result)}
            else:
                results[system] = result
        
        return results
    
    async def _execute_hierarchical_consultation(self, processed_query: ProcessedQuery,
                                               routing_decision: RoutingDecision) -> Dict[IntelligenceSystem, Dict[str, Any]]:
        """Execute consciousness consultation hierarchically, with primary system guiding secondary queries"""
        results = {}
        
        # First, consult primary system
        primary_result = await self._consult_single_system(
            routing_decision.primary_system, processed_query, {}
        )
        results[routing_decision.primary_system] = primary_result
        
        # Use primary results to refine secondary system queries
        refined_context = {
            'primary_insights': primary_result,
            'refinement_needed': True
        }
        
        # Consult secondary systems with refined understanding
        secondary_tasks = []
        for system in routing_decision.secondary_systems:
            task = self._consult_single_system(system, processed_query, refined_context)
            secondary_tasks.append((system, task))
        
        # Execute secondary consultations
        completed_tasks = await asyncio.gather(*[task for _, task in secondary_tasks], return_exceptions=True)
        
        for (system, _), result in zip(secondary_tasks, completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"Error consulting {system.value}: {result}")
                results[system] = {'error': str(result)}
            else:
                results[system] = result
        
        return results
    
    async def _consult_single_system(self, system: IntelligenceSystem, 
                                   processed_query: ProcessedQuery,
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Consult a single consciousness system"""
        query_context = processed_query.original_context
        
        try:
            if system == IntelligenceSystem.HARDWARE_SPECIALIZATION:
                return await self._consult_hardware_specialization(query_context, context)
            elif system == IntelligenceSystem.CONTAINER_CONSCIOUSNESS:
                return await self._consult_container_consciousness(query_context, context)
            elif system == IntelligenceSystem.MULTI_MODEL_ORACLE:
                return await self._consult_multi_model_oracle(query_context, context)
            elif system == IntelligenceSystem.TEMPORAL_INTELLIGENCE:
                return await self._consult_temporal_intelligence(query_context, context)
            else:
                return {'error': f'Unknown system: {system.value}'}
                
        except Exception as e:
            logger.error(f"Error consulting {system.value}: {e}")
            return {'error': str(e)}
    
    async def _consult_hardware_specialization(self, query_context: QueryContext,
                                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Consult Hardware Specialization consciousness system"""
        insights = await self.consciousness_controller.get_performance_insights()
        hardware_data = insights.get('hardware_specialization', {})
        
        # Extract relevant hardware information based on query components
        result = {
            'system_type': 'hardware_specialization',
            'timestamp': datetime.now().isoformat()
        }
        
        for component in query_context.components:
            if component.value in ['rtx5090_gpu', 'gpu', 'graphics']:
                result['rtx5090_blackwall'] = hardware_data.get('rtx5090_blackwall', {})
            elif component.value in ['amd_zen5_cpu', 'cpu', 'processor']:
                result['amd_zen5'] = hardware_data.get('amd_zen5', {})
            elif component.value in ['thermal_system', 'thermal', 'cooling']:
                result['thermal_intelligence'] = hardware_data.get('thermal_intelligence', {})
        
        # Add general hardware status if no specific components
        if not any(comp.value in ['gpu', 'cpu', 'thermal'] for comp in query_context.components):
            result.update(hardware_data)
        
        return result
    
    async def _consult_container_consciousness(self, query_context: QueryContext,
                                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Consult Container Consciousness system"""
        insights = await self.consciousness_controller.get_performance_insights()
        container_data = insights.get('container_intelligence', {})
        
        return {
            'system_type': 'container_consciousness',
            'timestamp': datetime.now().isoformat(),
            'service_orchestration': container_data.get('service_orchestration', {}),
            'resource_flows': container_data.get('resource_flows', []),
            'ai_service_interactions': container_data.get('ai_service_interactions', []),
            'load_balancing_efficiency': container_data.get('load_balancing_efficiency', 0.0),
            'container_events': container_data.get('events', [])
        }
    
    async def _consult_multi_model_oracle(self, query_context: QueryContext,
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Consult Multi-Model Oracle system"""
        insights = await self.consciousness_controller.get_performance_insights()
        oracle_data = insights.get('multi_model_oracle', {})
        
        return {
            'system_type': 'multi_model_oracle',
            'timestamp': datetime.now().isoformat(),
            'predictions': oracle_data.get('predictions', []),
            'optimization_strategies': oracle_data.get('optimization_strategies', []),
            'performance_bottlenecks': oracle_data.get('performance_bottlenecks', []),
            'resource_planning': oracle_data.get('resource_planning', {}),
            'confidence_levels': oracle_data.get('confidence_levels', {})
        }
    
    async def _consult_temporal_intelligence(self, query_context: QueryContext,
                                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Consult Temporal Intelligence system"""
        insights = await self.consciousness_controller.get_performance_insights()
        temporal_data = insights.get('temporal_intelligence', {})
        
        result = {
            'system_type': 'temporal_intelligence',
            'timestamp': datetime.now().isoformat(),
            'patterns': temporal_data.get('patterns', []),
            'correlations': temporal_data.get('correlations', []),
            'causal_analysis': temporal_data.get('causal_analysis', {}),
            'trends': temporal_data.get('trends', {}),
            'events': temporal_data.get('events', [])
        }
        
        # Add time-range specific analysis if requested
        if query_context.time_range:
            start_time, end_time = query_context.time_range
            result['time_range_analysis'] = {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration_hours': (end_time - start_time).total_seconds() / 3600
            }
        
        return result
    
    async def _generate_mock_consciousness_results(self, routing_decision: RoutingDecision) -> Dict[IntelligenceSystem, Dict[str, Any]]:
        """Generate mock results when consciousness controller is not available"""
        results = {}
        systems = [routing_decision.primary_system] + routing_decision.secondary_systems
        
        for system in systems:
            if system == IntelligenceSystem.HARDWARE_SPECIALIZATION:
                results[system] = {
                    'rtx5090_blackwall': {'temperature': 75.0, 'utilization': 0.82, 'vram_usage': 18.5},
                    'amd_zen5': {'temperature': 65.0, 'utilization': 0.67, 'efficiency': 0.84},
                    'thermal_intelligence': {'cooling_efficiency': 0.89, 'fan_speeds': [1200, 1180, 1220]}
                }
            elif system == IntelligenceSystem.CONTAINER_CONSCIOUSNESS:
                results[system] = {
                    'service_orchestration': {'efficiency': 0.87},
                    'ai_service_interactions': [{'service': 'llama-gpu', 'status': 'running'}],
                    'load_balancing_efficiency': 0.78
                }
            elif system == IntelligenceSystem.MULTI_MODEL_ORACLE:
                results[system] = {
                    'predictions': [{'type': 'performance', 'confidence': 0.85}],
                    'optimization_strategies': [{'category': 'thermal', 'potential_gain': 0.12}],
                    'confidence_levels': {'overall': 0.82}
                }
            elif system == IntelligenceSystem.TEMPORAL_INTELLIGENCE:
                results[system] = {
                    'patterns': [{'type': 'usage_pattern', 'confidence': 0.78}],
                    'correlations': [{'components': ['gpu', 'thermal'], 'strength': 0.91}],
                    'causal_analysis': {'confidence': 0.75}
                }
        
        return results
    
    async def _generate_fallback_results(self, routing_decision: RoutingDecision, error: Exception) -> Dict[IntelligenceSystem, Dict[str, Any]]:
        """Generate fallback results when consciousness system consultation fails"""
        systems = [routing_decision.primary_system] + routing_decision.secondary_systems
        return {system: {'error': f'System consultation failed: {str(error)}'} for system in systems}
    
    def _get_session_context(self, session_id: Optional[str]) -> Dict[str, Any]:
        """Get or create session context for conversational continuity"""
        if not session_id:
            return {}
        
        if session_id not in self.session_contexts:
            self.session_contexts[session_id] = {
                'created_at': datetime.now(),
                'query_count': 0,
                'last_components': [],
                'last_intent': None,
                'context_history': []
            }
        
        return self.session_contexts[session_id]
    
    def _update_session_context(self, session_id: Optional[str], 
                              query_context: QueryContext,
                              response: SynthesizedResponse):
        """Update session context with latest interaction"""
        if not session_id:
            return
        
        context = self.session_contexts.get(session_id, {})
        context['query_count'] = context.get('query_count', 0) + 1
        context['last_components'] = query_context.components
        context['last_intent'] = query_context.intent
        context['last_interaction'] = datetime.now()
        
        # Keep limited history
        history_entry = {
            'query': query_context.original_query,
            'intent': query_context.intent.value,
            'confidence': response.confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        if 'context_history' not in context:
            context['context_history'] = []
        
        context['context_history'].append(history_entry)
        
        # Keep only last 10 interactions
        context['context_history'] = context['context_history'][-10:]
        
        self.session_contexts[session_id] = context
    
    def _update_performance_metrics(self, processing_time: float, success: bool):
        """Update orchestrator performance metrics"""
        self.performance_metrics['queries_processed'] += 1
        
        # Update average processing time
        current_avg = self.performance_metrics['average_processing_time']
        query_count = self.performance_metrics['queries_processed']
        self.performance_metrics['average_processing_time'] = (
            (current_avg * (query_count - 1) + processing_time) / query_count
        )
        
        # Update success rate
        if success:
            current_success_count = self.performance_metrics['success_rate'] * (query_count - 1)
            self.performance_metrics['success_rate'] = (current_success_count + 1) / query_count
        else:
            current_success_count = self.performance_metrics['success_rate'] * (query_count - 1)
            self.performance_metrics['success_rate'] = current_success_count / query_count
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get current orchestrator status and performance metrics"""
        return {
            'status': 'active',
            'consciousness_controller_available': self.consciousness_controller is not None,
            'active_sessions': len(self.session_contexts),
            'performance_metrics': self.performance_metrics,
            'components_initialized': {
                'intent_engine': self.intent_engine is not None,
                'query_router': self.query_router is not None,
                'context_processor': self.context_processor is not None,
                'response_synthesizer': self.response_synthesizer is not None
            }
        }