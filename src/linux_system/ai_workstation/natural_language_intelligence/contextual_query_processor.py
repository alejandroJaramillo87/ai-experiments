"""
Contextual Query Processor

Leverages the temporal intelligence system and consciousness infrastructure to provide
deep contextual understanding for natural language queries. Creates rich context by
analyzing historical patterns, causal relationships, and system state evolution.

This processor enables sophisticated query understanding that goes beyond simple
keyword matching by incorporating the full consciousness system's temporal knowledge.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .intent_understanding_engine import QueryContext, SystemComponent
from .query_intelligence_router import RoutingDecision, IntelligenceQuery

logger = logging.getLogger(__name__)


@dataclass
class ContextualEnrichment:
    """Rich contextual information extracted from consciousness systems"""
    historical_context: Dict[str, Any]
    causal_context: Dict[str, Any]
    predictive_context: Dict[str, Any]
    system_state_context: Dict[str, Any]
    correlation_context: Dict[str, Any]
    confidence: float
    context_depth: str  # 'shallow', 'moderate', 'deep'


@dataclass
class ProcessedQuery:
    """Query enriched with consciousness system context"""
    original_context: QueryContext
    routing_decision: RoutingDecision
    intelligence_queries: List[IntelligenceQuery]
    contextual_enrichment: ContextualEnrichment
    execution_priority: int
    estimated_processing_time: float


class ContextualQueryProcessor:
    """
    Processes queries with deep contextual understanding from consciousness systems.
    
    Leverages temporal intelligence, hardware specialization, container consciousness,
    and multi-model oracle to provide rich context for natural language understanding.
    """
    
    def __init__(self, consciousness_systems_available: bool = True):
        self.consciousness_available = consciousness_systems_available
        self.context_cache = {}  # Simple context caching
        self.temporal_context_depth = self._configure_temporal_depth()
        
        logger.info(f"Contextual Query Processor initialized. Consciousness systems: {consciousness_systems_available}")
    
    async def process_with_context(self, query_context: QueryContext, 
                                 routing_decision: RoutingDecision,
                                 intelligence_queries: List[IntelligenceQuery],
                                 consciousness_controller) -> ProcessedQuery:
        """
        Process query with rich contextual understanding from consciousness systems.
        
        Args:
            query_context: Original query understanding
            routing_decision: How to route through consciousness systems
            intelligence_queries: Structured queries for each system
            consciousness_controller: Access to AI workstation consciousness
            
        Returns:
            ProcessedQuery with enriched contextual information
        """
        # Gather contextual enrichment from consciousness systems
        enrichment = await self._gather_contextual_enrichment(
            query_context, routing_decision, consciousness_controller
        )
        
        # Estimate processing complexity and time
        processing_time = self._estimate_processing_time(routing_decision, enrichment)
        
        # Assign execution priority
        priority = self._assign_execution_priority(query_context, enrichment)
        
        return ProcessedQuery(
            original_context=query_context,
            routing_decision=routing_decision,
            intelligence_queries=intelligence_queries,
            contextual_enrichment=enrichment,
            execution_priority=priority,
            estimated_processing_time=processing_time
        )
    
    async def _gather_contextual_enrichment(self, query_context: QueryContext,
                                          routing_decision: RoutingDecision,
                                          consciousness_controller) -> ContextualEnrichment:
        """Gather rich contextual information from consciousness systems"""
        
        # Initialize context containers
        historical_context = {}
        causal_context = {}
        predictive_context = {}
        system_state_context = {}
        correlation_context = {}
        
        try:
            # Gather temporal/historical context
            if query_context.requires_historical_data or query_context.time_range:
                historical_context = await self._gather_historical_context(
                    query_context, consciousness_controller
                )
            
            # Gather causal analysis context
            if query_context.requires_causal_analysis:
                causal_context = await self._gather_causal_context(
                    query_context, consciousness_controller
                )
            
            # Gather predictive context
            if query_context.requires_prediction:
                predictive_context = await self._gather_predictive_context(
                    query_context, consciousness_controller
                )
            
            # Gather current system state context
            system_state_context = await self._gather_system_state_context(
                query_context, consciousness_controller
            )
            
            # Gather correlation context
            correlation_context = await self._gather_correlation_context(
                query_context, consciousness_controller
            )
            
            # Calculate overall context confidence and depth
            confidence = self._calculate_context_confidence(
                historical_context, causal_context, predictive_context, 
                system_state_context, correlation_context
            )
            
            depth = self._assess_context_depth(query_context, historical_context, causal_context)
            
        except Exception as e:
            logger.error(f"Error gathering contextual enrichment: {e}")
            # Provide fallback context
            confidence = 0.3
            depth = 'shallow'
        
        return ContextualEnrichment(
            historical_context=historical_context,
            causal_context=causal_context,
            predictive_context=predictive_context,
            system_state_context=system_state_context,
            correlation_context=correlation_context,
            confidence=confidence,
            context_depth=depth
        )
    
    async def _gather_historical_context(self, query_context: QueryContext,
                                       consciousness_controller) -> Dict[str, Any]:
        """Gather historical context from temporal intelligence system"""
        historical_context = {
            'time_range_analyzed': None,
            'historical_patterns': [],
            'trend_analysis': {},
            'significant_events': [],
            'baseline_metrics': {}
        }
        
        try:
            # Determine time range for historical analysis
            if query_context.time_range:
                start_time, end_time = query_context.time_range
            else:
                # Default to last 24 hours for context
                end_time = datetime.now()
                start_time = end_time - timedelta(days=1)
            
            historical_context['time_range_analyzed'] = {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            }
            
            # Get performance insights for historical context
            insights = await consciousness_controller.get_performance_insights()
            
            # Extract temporal intelligence data
            temporal_data = insights.get('temporal_intelligence', {})
            historical_context['historical_patterns'] = temporal_data.get('patterns', [])
            historical_context['trend_analysis'] = temporal_data.get('trends', {})
            historical_context['significant_events'] = temporal_data.get('events', [])
            
            # Get baseline metrics for components of interest
            for component in query_context.components:
                if component == SystemComponent.RTX5090_GPU:
                    gpu_data = insights.get('hardware_specialization', {}).get('rtx5090_blackwall', {})
                    historical_context['baseline_metrics']['gpu'] = {
                        'avg_temperature': gpu_data.get('avg_temperature', 75.0),
                        'avg_utilization': gpu_data.get('avg_utilization', 0.7),
                        'avg_vram_usage': gpu_data.get('avg_vram_usage', 16.0)
                    }
                elif component == SystemComponent.AMD_ZEN5_CPU:
                    cpu_data = insights.get('hardware_specialization', {}).get('amd_zen5', {})
                    historical_context['baseline_metrics']['cpu'] = {
                        'avg_temperature': cpu_data.get('avg_temperature', 65.0),
                        'avg_utilization': cpu_data.get('avg_utilization', 0.5),
                        'avg_efficiency': cpu_data.get('avg_efficiency', 0.8)
                    }
            
        except Exception as e:
            logger.error(f"Error gathering historical context: {e}")
            historical_context['error'] = str(e)
        
        return historical_context
    
    async def _gather_causal_context(self, query_context: QueryContext,
                                   consciousness_controller) -> Dict[str, Any]:
        """Gather causal analysis context from temporal intelligence"""
        causal_context = {
            'causal_chains': [],
            'root_causes': [],
            'contributing_factors': [],
            'causal_confidence': 0.0,
            'causal_timeline': []
        }
        
        try:
            insights = await consciousness_controller.get_performance_insights()
            
            # Extract causal analysis from temporal intelligence
            temporal_data = insights.get('temporal_intelligence', {})
            causal_analysis = temporal_data.get('causal_analysis', {})
            
            causal_context['causal_chains'] = causal_analysis.get('chains', [])
            causal_context['root_causes'] = causal_analysis.get('root_causes', [])
            causal_context['contributing_factors'] = causal_analysis.get('factors', [])
            causal_context['causal_confidence'] = causal_analysis.get('confidence', 0.0)
            
            # Build causal timeline if temporal keywords present
            if query_context.temporal_keywords:
                causal_context['causal_timeline'] = await self._build_causal_timeline(
                    query_context, consciousness_controller
                )
            
            # Add component-specific causal factors
            for component in query_context.components:
                component_causality = await self._analyze_component_causality(
                    component, consciousness_controller
                )
                causal_context[f'{component.value}_causality'] = component_causality
            
        except Exception as e:
            logger.error(f"Error gathering causal context: {e}")
            causal_context['error'] = str(e)
        
        return causal_context
    
    async def _gather_predictive_context(self, query_context: QueryContext,
                                       consciousness_controller) -> Dict[str, Any]:
        """Gather predictive context from multi-model oracle"""
        predictive_context = {
            'predictions': [],
            'confidence_levels': {},
            'prediction_horizon': 'short',
            'risk_factors': [],
            'optimization_opportunities': []
        }
        
        try:
            insights = await consciousness_controller.get_performance_insights()
            
            # Extract predictions from multi-model oracle
            oracle_data = insights.get('multi_model_oracle', {})
            predictive_context['predictions'] = oracle_data.get('predictions', [])
            predictive_context['confidence_levels'] = oracle_data.get('confidence', {})
            predictive_context['risk_factors'] = oracle_data.get('risks', [])
            predictive_context['optimization_opportunities'] = oracle_data.get('optimizations', [])
            
            # Determine prediction horizon based on query
            if any(keyword in query_context.original_query.lower() for keyword in ['long term', 'future', 'tomorrow']):
                predictive_context['prediction_horizon'] = 'long'
            elif any(keyword in query_context.original_query.lower() for keyword in ['soon', 'next', 'immediate']):
                predictive_context['prediction_horizon'] = 'short'
            else:
                predictive_context['prediction_horizon'] = 'medium'
            
            # Generate component-specific predictions
            for component in query_context.components:
                component_predictions = await self._generate_component_predictions(
                    component, consciousness_controller
                )
                predictive_context[f'{component.value}_predictions'] = component_predictions
            
        except Exception as e:
            logger.error(f"Error gathering predictive context: {e}")
            predictive_context['error'] = str(e)
        
        return predictive_context
    
    async def _gather_system_state_context(self, query_context: QueryContext,
                                         consciousness_controller) -> Dict[str, Any]:
        """Gather current system state context"""
        system_state_context = {
            'current_mode': 'unknown',
            'system_health': 'unknown',
            'performance_score': 0.0,
            'active_workloads': [],
            'resource_utilization': {},
            'component_states': {},
            'consciousness_status': {}
        }
        
        try:
            # Get current system status
            status = await consciousness_controller.get_workstation_status()
            
            system_state_context['current_mode'] = status.mode.value
            system_state_context['system_health'] = status.health_status.value
            system_state_context['performance_score'] = status.overall_performance_score
            system_state_context['active_workloads'] = status.active_workloads
            system_state_context['resource_utilization'] = status.resource_utilization
            
            # Get consciousness system status
            system_state_context['consciousness_status'] = {
                'container_consciousness': status.container_consciousness_active,
                'hardware_specialization': status.hardware_specialization_active,
                'multi_model_oracle': status.multi_model_oracle_active,
                'temporal_intelligence': status.temporal_intelligence_active
            }
            
            # Get component-specific states
            insights = await consciousness_controller.get_performance_insights()
            hardware_data = insights.get('hardware_specialization', {})
            
            for component in query_context.components:
                if component == SystemComponent.RTX5090_GPU:
                    system_state_context['component_states']['gpu'] = hardware_data.get('rtx5090_blackwall', {})
                elif component == SystemComponent.AMD_ZEN5_CPU:
                    system_state_context['component_states']['cpu'] = hardware_data.get('amd_zen5', {})
                elif component == SystemComponent.THERMAL_SYSTEM:
                    system_state_context['component_states']['thermal'] = hardware_data.get('thermal_intelligence', {})
            
        except Exception as e:
            logger.error(f"Error gathering system state context: {e}")
            system_state_context['error'] = str(e)
        
        return system_state_context
    
    async def _gather_correlation_context(self, query_context: QueryContext,
                                        consciousness_controller) -> Dict[str, Any]:
        """Gather correlation context between system components"""
        correlation_context = {
            'component_correlations': [],
            'performance_correlations': [],
            'temporal_correlations': [],
            'cross_system_interactions': []
        }
        
        try:
            insights = await consciousness_controller.get_performance_insights()
            
            # Extract correlations from temporal intelligence
            temporal_data = insights.get('temporal_intelligence', {})
            correlation_context['temporal_correlations'] = temporal_data.get('correlations', [])
            
            # Extract performance correlations
            performance_data = insights.get('performance_trends', {})
            correlation_context['performance_correlations'] = performance_data.get('correlations', [])
            
            # Analyze component interactions
            if len(query_context.components) > 1:
                correlations = await self._analyze_component_correlations(
                    query_context.components, consciousness_controller
                )
                correlation_context['component_correlations'] = correlations
            
            # Analyze cross-system interactions
            container_data = insights.get('container_intelligence', {})
            correlation_context['cross_system_interactions'] = container_data.get('interactions', [])
            
        except Exception as e:
            logger.error(f"Error gathering correlation context: {e}")
            correlation_context['error'] = str(e)
        
        return correlation_context
    
    async def _build_causal_timeline(self, query_context: QueryContext,
                                   consciousness_controller) -> List[Dict[str, Any]]:
        """Build a causal timeline for the query context"""
        timeline = []
        
        try:
            if query_context.time_range:
                start_time, end_time = query_context.time_range
            else:
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=4)
            
            # Get temporal intelligence data for the time range
            insights = await consciousness_controller.get_performance_insights()
            temporal_data = insights.get('temporal_intelligence', {})
            
            events = temporal_data.get('events', [])
            for event in events:
                if 'timestamp' in event:
                    event_time = datetime.fromisoformat(event['timestamp'])
                    if start_time <= event_time <= end_time:
                        timeline.append({
                            'timestamp': event['timestamp'],
                            'event_type': event.get('type', 'unknown'),
                            'description': event.get('description', ''),
                            'impact': event.get('impact', 'unknown'),
                            'components_affected': event.get('components', [])
                        })
            
            # Sort timeline by timestamp
            timeline.sort(key=lambda x: x['timestamp'])
            
        except Exception as e:
            logger.error(f"Error building causal timeline: {e}")
        
        return timeline
    
    async def _analyze_component_causality(self, component: SystemComponent,
                                         consciousness_controller) -> Dict[str, Any]:
        """Analyze causality patterns for a specific component"""
        causality = {
            'primary_influences': [],
            'secondary_influences': [],
            'causal_strength': 0.0
        }
        
        try:
            insights = await consciousness_controller.get_performance_insights()
            
            if component == SystemComponent.RTX5090_GPU:
                gpu_data = insights.get('hardware_specialization', {}).get('rtx5090_blackwall', {})
                causality['primary_influences'] = ['thermal_conditions', 'workload_intensity', 'memory_pressure']
                causality['secondary_influences'] = ['ambient_temperature', 'power_delivery', 'driver_optimization']
                causality['causal_strength'] = gpu_data.get('causal_confidence', 0.8)
                
            elif component == SystemComponent.AMD_ZEN5_CPU:
                cpu_data = insights.get('hardware_specialization', {}).get('amd_zen5', {})
                causality['primary_influences'] = ['thread_utilization', 'memory_bandwidth', 'thermal_throttling']
                causality['secondary_influences'] = ['boost_clocks', 'cache_efficiency', 'numa_topology']
                causality['causal_strength'] = cpu_data.get('causal_confidence', 0.7)
                
            elif component == SystemComponent.THERMAL_SYSTEM:
                thermal_data = insights.get('hardware_specialization', {}).get('thermal_intelligence', {})
                causality['primary_influences'] = ['component_heat_generation', 'fan_performance', 'airflow_patterns']
                causality['secondary_influences'] = ['ambient_conditions', 'case_design', 'thermal_paste_aging']
                causality['causal_strength'] = thermal_data.get('causal_confidence', 0.9)
            
        except Exception as e:
            logger.error(f"Error analyzing component causality: {e}")
        
        return causality
    
    async def _generate_component_predictions(self, component: SystemComponent,
                                            consciousness_controller) -> Dict[str, Any]:
        """Generate predictions for a specific component"""
        predictions = {
            'short_term': [],
            'medium_term': [],
            'confidence': 0.0
        }
        
        try:
            insights = await consciousness_controller.get_performance_insights()
            oracle_data = insights.get('multi_model_oracle', {})
            
            component_key = component.value.replace('_', '-')
            component_predictions = oracle_data.get(f'{component_key}_predictions', {})
            
            predictions['short_term'] = component_predictions.get('short_term', [])
            predictions['medium_term'] = component_predictions.get('medium_term', [])
            predictions['confidence'] = component_predictions.get('confidence', 0.6)
            
        except Exception as e:
            logger.error(f"Error generating component predictions: {e}")
        
        return predictions
    
    async def _analyze_component_correlations(self, components: List[SystemComponent],
                                            consciousness_controller) -> List[Dict[str, Any]]:
        """Analyze correlations between multiple components"""
        correlations = []
        
        try:
            insights = await consciousness_controller.get_performance_insights()
            
            # Look for known correlation patterns
            component_pairs = [
                (SystemComponent.RTX5090_GPU, SystemComponent.THERMAL_SYSTEM, 'thermal_coupling', 0.9),
                (SystemComponent.AMD_ZEN5_CPU, SystemComponent.THERMAL_SYSTEM, 'thermal_coupling', 0.8),
                (SystemComponent.RTX5090_GPU, SystemComponent.AI_SERVICES, 'workload_coupling', 0.7),
                (SystemComponent.AMD_ZEN5_CPU, SystemComponent.AI_SERVICES, 'workload_coupling', 0.6)
            ]
            
            for comp1, comp2, correlation_type, strength in component_pairs:
                if comp1 in components and comp2 in components:
                    correlations.append({
                        'component_1': comp1.value,
                        'component_2': comp2.value,
                        'correlation_type': correlation_type,
                        'strength': strength,
                        'description': f'{comp1.value} performance influences {comp2.value} behavior'
                    })
            
        except Exception as e:
            logger.error(f"Error analyzing component correlations: {e}")
        
        return correlations
    
    def _calculate_context_confidence(self, historical_context: Dict[str, Any],
                                    causal_context: Dict[str, Any],
                                    predictive_context: Dict[str, Any],
                                    system_state_context: Dict[str, Any],
                                    correlation_context: Dict[str, Any]) -> float:
        """Calculate overall confidence in contextual enrichment"""
        confidence_factors = []
        
        # Historical context confidence
        if historical_context and 'error' not in historical_context:
            if historical_context.get('historical_patterns'):
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        # Causal context confidence
        if causal_context and 'error' not in causal_context:
            causal_conf = causal_context.get('causal_confidence', 0.5)
            confidence_factors.append(causal_conf)
        else:
            confidence_factors.append(0.4)
        
        # System state context confidence (usually high since it's current data)
        if system_state_context and 'error' not in system_state_context:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)
        
        # Predictive context confidence
        if predictive_context and 'error' not in predictive_context:
            pred_conf = max(predictive_context.get('confidence_levels', {}).values()) if predictive_context.get('confidence_levels') else 0.6
            confidence_factors.append(pred_conf)
        else:
            confidence_factors.append(0.5)
        
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    
    def _assess_context_depth(self, query_context: QueryContext,
                            historical_context: Dict[str, Any],
                            causal_context: Dict[str, Any]) -> str:
        """Assess the depth of contextual understanding achieved"""
        depth_indicators = 0
        
        # Check for deep historical analysis
        if historical_context.get('historical_patterns') and len(historical_context['historical_patterns']) > 3:
            depth_indicators += 1
        
        # Check for causal analysis
        if causal_context.get('causal_chains') and causal_context.get('causal_confidence', 0) > 0.7:
            depth_indicators += 2
        
        # Check for temporal complexity
        if query_context.time_range or query_context.temporal_keywords:
            depth_indicators += 1
        
        # Check for multi-component analysis
        if len(query_context.components) > 2:
            depth_indicators += 1
        
        # Check for requirements complexity
        requirements = [
            query_context.requires_causal_analysis,
            query_context.requires_prediction,
            query_context.requires_optimization,
            query_context.requires_historical_data
        ]
        depth_indicators += sum(requirements)
        
        if depth_indicators >= 5:
            return 'deep'
        elif depth_indicators >= 3:
            return 'moderate'
        else:
            return 'shallow'
    
    def _estimate_processing_time(self, routing_decision: RoutingDecision,
                                enrichment: ContextualEnrichment) -> float:
        """Estimate processing time based on complexity"""
        base_time = 1.0  # seconds
        
        # Add time for multiple systems
        system_count = len(routing_decision.secondary_systems) + 1
        base_time += system_count * 0.5
        
        # Add time for complex execution strategies
        if routing_decision.execution_strategy == 'hierarchical':
            base_time += 1.0
        elif routing_decision.execution_strategy == 'sequential':
            base_time += 0.5
        
        # Add time for deep context
        if enrichment.context_depth == 'deep':
            base_time += 2.0
        elif enrichment.context_depth == 'moderate':
            base_time += 1.0
        
        return base_time
    
    def _assign_execution_priority(self, query_context: QueryContext,
                                 enrichment: ContextualEnrichment) -> int:
        """Assign execution priority (1-10, higher is more urgent)"""
        priority = 5  # Default medium priority
        
        # Urgent keywords boost priority
        urgent_keywords = ['critical', 'emergency', 'failure', 'crash', 'down']
        if any(keyword in query_context.original_query.lower() for keyword in urgent_keywords):
            priority += 3
        
        # System health issues boost priority
        if enrichment.system_state_context.get('system_health') in ['degraded', 'critical']:
            priority += 2
        
        # High confidence reduces priority (less urgent if we're confident)
        if enrichment.confidence > 0.8:
            priority -= 1
        
        # Complex analysis reduces priority (can wait)
        if enrichment.context_depth == 'deep':
            priority -= 1
        
        return max(1, min(10, priority))
    
    def _configure_temporal_depth(self) -> Dict[str, timedelta]:
        """Configure how far back to look for different types of context"""
        return {
            'recent': timedelta(hours=2),
            'short_term': timedelta(days=1),
            'medium_term': timedelta(days=7),
            'long_term': timedelta(days=30)
        }