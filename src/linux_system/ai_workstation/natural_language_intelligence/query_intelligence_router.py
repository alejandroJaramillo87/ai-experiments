"""
Query Intelligence Router

Advanced routing system that maps natural language queries to the appropriate
AI workstation consciousness systems. Creates digital symbiosis by intelligently
directing queries to Container Consciousness, Hardware Specialization, 
Multi-Model Oracle, and Temporal Intelligence systems based on intent and context.

This router enables sophisticated query distribution that leverages the full
consciousness infrastructure rather than simple template matching.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .intent_understanding_engine import QueryContext, QueryIntent, SystemComponent

logger = logging.getLogger(__name__)


class IntelligenceSystem(Enum):
    """AI Workstation consciousness systems available for query routing"""
    CONTAINER_CONSCIOUSNESS = "container_consciousness"
    HARDWARE_SPECIALIZATION = "hardware_specialization"  
    MULTI_MODEL_ORACLE = "multi_model_oracle"
    TEMPORAL_INTELLIGENCE = "temporal_intelligence"


@dataclass
class RoutingDecision:
    """Decision about how to route a query through the consciousness systems"""
    primary_system: IntelligenceSystem
    secondary_systems: List[IntelligenceSystem]
    routing_confidence: float
    execution_strategy: str  # 'sequential', 'parallel', 'hierarchical'
    context_requirements: Dict[str, Any]
    expected_data_types: List[str]


@dataclass
class IntelligenceQuery:
    """Structured query for consciousness systems"""
    query_context: QueryContext
    system_target: IntelligenceSystem
    specific_request: str
    parameters: Dict[str, Any]
    expected_format: str


class QueryIntelligenceRouter:
    """
    Routes natural language queries to appropriate AI workstation consciousness systems.
    
    Creates sophisticated mapping between human intent and machine intelligence,
    enabling true digital symbiosis through intelligent system orchestration.
    """
    
    def __init__(self):
        self.routing_rules = self._build_routing_rules()
        self.system_capabilities = self._define_system_capabilities()
        self.execution_strategies = self._define_execution_strategies()
        
        logger.info("Query Intelligence Router initialized with consciousness system mapping")
    
    def route_query(self, query_context: QueryContext) -> RoutingDecision:
        """
        Route a query to appropriate consciousness systems based on intent and context.
        
        Args:
            query_context: Rich context from intent understanding
            
        Returns:
            RoutingDecision with primary/secondary systems and execution strategy
        """
        # Analyze routing requirements
        routing_analysis = self._analyze_routing_requirements(query_context)
        
        # Determine primary system
        primary_system = self._select_primary_system(query_context, routing_analysis)
        
        # Determine secondary systems for comprehensive responses
        secondary_systems = self._select_secondary_systems(query_context, primary_system, routing_analysis)
        
        # Choose execution strategy
        strategy = self._select_execution_strategy(query_context, primary_system, secondary_systems)
        
        # Calculate routing confidence
        confidence = self._calculate_routing_confidence(query_context, primary_system, secondary_systems)
        
        # Determine context requirements
        context_reqs = self._determine_context_requirements(query_context, primary_system, secondary_systems)
        
        # Predict expected data types
        expected_data = self._predict_expected_data_types(query_context, primary_system, secondary_systems)
        
        return RoutingDecision(
            primary_system=primary_system,
            secondary_systems=secondary_systems,
            routing_confidence=confidence,
            execution_strategy=strategy,
            context_requirements=context_reqs,
            expected_data_types=expected_data
        )
    
    def create_intelligence_queries(self, query_context: QueryContext, 
                                  routing_decision: RoutingDecision) -> List[IntelligenceQuery]:
        """
        Create specific queries for each consciousness system based on routing decision.
        
        Args:
            query_context: Original query context
            routing_decision: How to route the query
            
        Returns:
            List of structured queries for consciousness systems
        """
        queries = []
        
        # Create primary system query
        primary_query = self._create_system_query(
            query_context, routing_decision.primary_system, is_primary=True
        )
        queries.append(primary_query)
        
        # Create secondary system queries
        for system in routing_decision.secondary_systems:
            secondary_query = self._create_system_query(
                query_context, system, is_primary=False
            )
            queries.append(secondary_query)
        
        return queries
    
    def _build_routing_rules(self) -> Dict[QueryIntent, Dict[str, Any]]:
        """Build sophisticated routing rules for each query intent"""
        return {
            QueryIntent.CAUSAL_ANALYSIS: {
                'primary_candidates': [IntelligenceSystem.TEMPORAL_INTELLIGENCE],
                'secondary_required': [IntelligenceSystem.HARDWARE_SPECIALIZATION],
                'execution_strategy': 'sequential',
                'requires_historical': True
            },
            QueryIntent.ROOT_CAUSE: {
                'primary_candidates': [IntelligenceSystem.TEMPORAL_INTELLIGENCE],
                'secondary_required': [IntelligenceSystem.HARDWARE_SPECIALIZATION, IntelligenceSystem.CONTAINER_CONSCIOUSNESS],
                'execution_strategy': 'hierarchical',
                'requires_historical': True
            },
            QueryIntent.SYSTEM_STATUS: {
                'primary_candidates': [IntelligenceSystem.HARDWARE_SPECIALIZATION],
                'secondary_required': [IntelligenceSystem.CONTAINER_CONSCIOUSNESS],
                'execution_strategy': 'parallel',
                'requires_current': True
            },
            QueryIntent.COMPONENT_STATUS: {
                'primary_candidates': [IntelligenceSystem.HARDWARE_SPECIALIZATION],
                'secondary_required': [],
                'execution_strategy': 'sequential',
                'requires_current': True
            },
            QueryIntent.PATTERN_DISCOVERY: {
                'primary_candidates': [IntelligenceSystem.TEMPORAL_INTELLIGENCE],
                'secondary_required': [IntelligenceSystem.MULTI_MODEL_ORACLE],
                'execution_strategy': 'sequential',
                'requires_historical': True,
                'requires_analysis': True
            },
            QueryIntent.BEHAVIORAL_ANALYSIS: {
                'primary_candidates': [IntelligenceSystem.TEMPORAL_INTELLIGENCE],
                'secondary_required': [IntelligenceSystem.CONTAINER_CONSCIOUSNESS, IntelligenceSystem.MULTI_MODEL_ORACLE],
                'execution_strategy': 'hierarchical',
                'requires_historical': True,
                'requires_analysis': True
            },
            QueryIntent.PREDICTIVE_QUERY: {
                'primary_candidates': [IntelligenceSystem.MULTI_MODEL_ORACLE],
                'secondary_required': [IntelligenceSystem.TEMPORAL_INTELLIGENCE, IntelligenceSystem.HARDWARE_SPECIALIZATION],
                'execution_strategy': 'hierarchical',
                'requires_prediction': True
            },
            QueryIntent.OPTIMIZATION_REQUEST: {
                'primary_candidates': [IntelligenceSystem.MULTI_MODEL_ORACLE],
                'secondary_required': [IntelligenceSystem.HARDWARE_SPECIALIZATION, IntelligenceSystem.CONTAINER_CONSCIOUSNESS],
                'execution_strategy': 'hierarchical',
                'requires_prediction': True,
                'requires_optimization': True
            },
            QueryIntent.RESOURCE_PLANNING: {
                'primary_candidates': [IntelligenceSystem.MULTI_MODEL_ORACLE],
                'secondary_required': [IntelligenceSystem.TEMPORAL_INTELLIGENCE, IntelligenceSystem.HARDWARE_SPECIALIZATION],
                'execution_strategy': 'sequential',
                'requires_prediction': True,
                'requires_current': True
            },
            QueryIntent.TEMPORAL_QUERY: {
                'primary_candidates': [IntelligenceSystem.TEMPORAL_INTELLIGENCE],
                'secondary_required': [],
                'execution_strategy': 'sequential',
                'requires_historical': True
            },
            QueryIntent.CONSCIOUSNESS_QUERY: {
                'primary_candidates': [IntelligenceSystem.MULTI_MODEL_ORACLE],
                'secondary_required': [IntelligenceSystem.TEMPORAL_INTELLIGENCE],
                'execution_strategy': 'hierarchical',
                'requires_consciousness': True
            }
        }
    
    def _define_system_capabilities(self) -> Dict[IntelligenceSystem, Dict[str, Any]]:
        """Define capabilities of each consciousness system"""
        return {
            IntelligenceSystem.CONTAINER_CONSCIOUSNESS: {
                'primary_expertise': ['service_orchestration', 'resource_flows', 'ai_service_interactions'],
                'data_types': ['service_metrics', 'container_events', 'orchestration_patterns'],
                'temporal_capability': 'medium',
                'predictive_capability': 'low',
                'causal_capability': 'medium'
            },
            IntelligenceSystem.HARDWARE_SPECIALIZATION: {
                'primary_expertise': ['gpu_performance', 'cpu_optimization', 'thermal_management'],
                'data_types': ['hardware_metrics', 'thermal_data', 'performance_indicators'],
                'temporal_capability': 'high',
                'predictive_capability': 'medium',
                'causal_capability': 'high'
            },
            IntelligenceSystem.MULTI_MODEL_ORACLE: {
                'primary_expertise': ['performance_prediction', 'optimization_strategies', 'resource_planning'],
                'data_types': ['predictions', 'strategies', 'optimization_plans'],
                'temporal_capability': 'high',
                'predictive_capability': 'high',
                'causal_capability': 'high'
            },
            IntelligenceSystem.TEMPORAL_INTELLIGENCE: {
                'primary_expertise': ['causal_analysis', 'pattern_detection', 'trend_analysis'],
                'data_types': ['time_series', 'correlations', 'causal_relationships'],
                'temporal_capability': 'high',
                'predictive_capability': 'medium',
                'causal_capability': 'high'
            }
        }
    
    def _define_execution_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Define how to execute multi-system queries"""
        return {
            'sequential': {
                'description': 'Execute systems one after another, passing context',
                'use_when': ['causal_analysis', 'detailed_investigation'],
                'context_flow': True
            },
            'parallel': {
                'description': 'Execute systems simultaneously, combine results',
                'use_when': ['status_checks', 'comprehensive_overview'],
                'context_flow': False
            },
            'hierarchical': {
                'description': 'Primary system guides secondary system queries',
                'use_when': ['complex_analysis', 'optimization_planning'],
                'context_flow': True
            }
        }
    
    def _analyze_routing_requirements(self, query_context: QueryContext) -> Dict[str, Any]:
        """Analyze what the query requires from consciousness systems"""
        requirements = {
            'temporal_depth': 'current',  # current, recent, historical, deep
            'analysis_complexity': 'simple',  # simple, moderate, complex, deep
            'prediction_horizon': None,  # None, short, medium, long
            'system_scope': 'component',  # component, system, ecosystem
            'response_urgency': 'normal'  # normal, high, critical
        }
        
        # Analyze temporal requirements
        if query_context.time_range:
            time_span = query_context.time_range[1] - query_context.time_range[0]
            if time_span.days > 7:
                requirements['temporal_depth'] = 'deep'
            elif time_span.days > 1:
                requirements['temporal_depth'] = 'historical'
            else:
                requirements['temporal_depth'] = 'recent'
        
        # Analyze complexity requirements
        if query_context.requires_causal_analysis and query_context.requires_prediction:
            requirements['analysis_complexity'] = 'deep'
        elif query_context.requires_optimization:
            requirements['analysis_complexity'] = 'complex'
        elif query_context.requires_historical_data:
            requirements['analysis_complexity'] = 'moderate'
        
        # Analyze prediction requirements
        if query_context.requires_prediction:
            if query_context.intent == QueryIntent.RESOURCE_PLANNING:
                requirements['prediction_horizon'] = 'long'
            else:
                requirements['prediction_horizon'] = 'medium'
        
        # Analyze system scope
        if len(query_context.components) > 2:
            requirements['system_scope'] = 'ecosystem'
        elif SystemComponent.SYSTEM_CONSCIOUSNESS in query_context.components:
            requirements['system_scope'] = 'system'
        
        return requirements
    
    def _select_primary_system(self, query_context: QueryContext, 
                             routing_analysis: Dict[str, Any]) -> IntelligenceSystem:
        """Select the primary consciousness system for the query"""
        intent_rules = self.routing_rules.get(query_context.intent, {})
        primary_candidates = intent_rules.get('primary_candidates', [])
        
        if not primary_candidates:
            # Fallback logic based on components and requirements
            if query_context.requires_prediction or query_context.requires_optimization:
                return IntelligenceSystem.MULTI_MODEL_ORACLE
            elif query_context.requires_causal_analysis or query_context.requires_historical_data:
                return IntelligenceSystem.TEMPORAL_INTELLIGENCE
            elif any(comp in [SystemComponent.RTX5090_GPU, SystemComponent.AMD_ZEN5_CPU, SystemComponent.THERMAL_SYSTEM] 
                    for comp in query_context.components):
                return IntelligenceSystem.HARDWARE_SPECIALIZATION
            elif any(comp in [SystemComponent.AI_SERVICES, SystemComponent.CONTAINER_CONSCIOUSNESS] 
                    for comp in query_context.components):
                return IntelligenceSystem.CONTAINER_CONSCIOUSNESS
            else:
                return IntelligenceSystem.MULTI_MODEL_ORACLE  # Default orchestrator
        
        # Select best candidate based on query context
        best_candidate = primary_candidates[0]  # Default to first candidate
        
        # Refine selection based on components
        for candidate in primary_candidates:
            if self._system_matches_components(candidate, query_context.components):
                best_candidate = candidate
                break
        
        return best_candidate
    
    def _select_secondary_systems(self, query_context: QueryContext, 
                                primary_system: IntelligenceSystem,
                                routing_analysis: Dict[str, Any]) -> List[IntelligenceSystem]:
        """Select secondary consciousness systems to provide comprehensive responses"""
        intent_rules = self.routing_rules.get(query_context.intent, {})
        secondary_required = intent_rules.get('secondary_required', [])
        
        # Add component-specific secondary systems
        component_systems = []
        for component in query_context.components:
            if component in [SystemComponent.RTX5090_GPU, SystemComponent.AMD_ZEN5_CPU, SystemComponent.THERMAL_SYSTEM]:
                if IntelligenceSystem.HARDWARE_SPECIALIZATION not in [primary_system] + secondary_required:
                    component_systems.append(IntelligenceSystem.HARDWARE_SPECIALIZATION)
            elif component in [SystemComponent.AI_SERVICES, SystemComponent.CONTAINER_CONSCIOUSNESS]:
                if IntelligenceSystem.CONTAINER_CONSCIOUSNESS not in [primary_system] + secondary_required:
                    component_systems.append(IntelligenceSystem.CONTAINER_CONSCIOUSNESS)
        
        # Combine and deduplicate
        all_secondary = list(set(secondary_required + component_systems))
        
        # Remove primary system from secondary list
        return [sys for sys in all_secondary if sys != primary_system]
    
    def _select_execution_strategy(self, query_context: QueryContext, 
                                 primary_system: IntelligenceSystem,
                                 secondary_systems: List[IntelligenceSystem]) -> str:
        """Select execution strategy based on query complexity and requirements"""
        intent_rules = self.routing_rules.get(query_context.intent, {})
        preferred_strategy = intent_rules.get('execution_strategy', 'sequential')
        
        # Override based on complexity
        if len(secondary_systems) > 2:
            return 'hierarchical'  # Complex queries need orchestration
        elif query_context.requires_causal_analysis and query_context.requires_prediction:
            return 'hierarchical'  # Complex analysis needs coordination
        elif len(secondary_systems) == 0:
            return 'sequential'  # Single system
        elif query_context.intent == QueryIntent.SYSTEM_STATUS:
            return 'parallel'  # Status checks can be parallel
        
        return preferred_strategy
    
    def _calculate_routing_confidence(self, query_context: QueryContext,
                                    primary_system: IntelligenceSystem,
                                    secondary_systems: List[IntelligenceSystem]) -> float:
        """Calculate confidence in routing decision"""
        base_confidence = query_context.confidence
        
        # Boost confidence if routing matches intent patterns well
        intent_rules = self.routing_rules.get(query_context.intent, {})
        if primary_system in intent_rules.get('primary_candidates', []):
            base_confidence += 0.1
        
        # Boost confidence if components match system capabilities
        if self._system_matches_components(primary_system, query_context.components):
            base_confidence += 0.1
        
        # Reduce confidence if too many secondary systems (complex routing)
        if len(secondary_systems) > 3:
            base_confidence -= 0.1
        
        return min(base_confidence, 1.0)
    
    def _determine_context_requirements(self, query_context: QueryContext,
                                      primary_system: IntelligenceSystem,
                                      secondary_systems: List[IntelligenceSystem]) -> Dict[str, Any]:
        """Determine what context each system needs"""
        requirements = {
            'temporal_context': query_context.time_range is not None,
            'component_focus': query_context.components,
            'technical_terms': query_context.technical_terms,
            'cross_system_correlation': len(secondary_systems) > 0
        }
        
        # Add specific requirements based on intent
        if query_context.requires_causal_analysis:
            requirements['causal_chain'] = True
        if query_context.requires_prediction:
            requirements['prediction_parameters'] = True
        if query_context.requires_optimization:
            requirements['optimization_constraints'] = True
        
        return requirements
    
    def _predict_expected_data_types(self, query_context: QueryContext,
                                   primary_system: IntelligenceSystem,
                                   secondary_systems: List[IntelligenceSystem]) -> List[str]:
        """Predict what types of data the query will return"""
        data_types = []
        
        # Get data types from system capabilities
        systems_to_check = [primary_system] + secondary_systems
        for system in systems_to_check:
            system_caps = self.system_capabilities.get(system, {})
            data_types.extend(system_caps.get('data_types', []))
        
        # Add intent-specific data types
        if query_context.intent in [QueryIntent.PREDICTIVE_QUERY, QueryIntent.RESOURCE_PLANNING]:
            data_types.append('predictions')
        if query_context.intent in [QueryIntent.CAUSAL_ANALYSIS, QueryIntent.ROOT_CAUSE]:
            data_types.append('causal_relationships')
        if query_context.intent == QueryIntent.PATTERN_DISCOVERY:
            data_types.append('patterns')
        
        return list(set(data_types))  # Remove duplicates
    
    def _create_system_query(self, query_context: QueryContext, 
                           system: IntelligenceSystem, is_primary: bool) -> IntelligenceQuery:
        """Create a structured query for a specific consciousness system"""
        # Generate system-specific request
        specific_request = self._generate_system_specific_request(query_context, system, is_primary)
        
        # Generate parameters for the system
        parameters = self._generate_system_parameters(query_context, system)
        
        # Determine expected format
        expected_format = 'detailed' if is_primary else 'contextual'
        
        return IntelligenceQuery(
            query_context=query_context,
            system_target=system,
            specific_request=specific_request,
            parameters=parameters,
            expected_format=expected_format
        )
    
    def _generate_system_specific_request(self, query_context: QueryContext,
                                        system: IntelligenceSystem, is_primary: bool) -> str:
        """Generate a specific request for each consciousness system"""
        intent = query_context.intent
        components = query_context.components
        
        if system == IntelligenceSystem.TEMPORAL_INTELLIGENCE:
            if intent == QueryIntent.CAUSAL_ANALYSIS:
                return f"Analyze causal relationships for {', '.join([c.value for c in components])}"
            elif intent == QueryIntent.PATTERN_DISCOVERY:
                return f"Identify patterns in {', '.join([c.value for c in components])} behavior"
            else:
                return f"Provide temporal analysis for {', '.join([c.value for c in components])}"
        
        elif system == IntelligenceSystem.HARDWARE_SPECIALIZATION:
            if intent == QueryIntent.SYSTEM_STATUS:
                return f"Report current status of {', '.join([c.value for c in components])}"
            elif intent == QueryIntent.CAUSAL_ANALYSIS:
                return f"Analyze hardware factors contributing to events in {', '.join([c.value for c in components])}"
            else:
                return f"Provide hardware analysis for {', '.join([c.value for c in components])}"
        
        elif system == IntelligenceSystem.CONTAINER_CONSCIOUSNESS:
            if intent == QueryIntent.SYSTEM_STATUS:
                return "Report AI service orchestration status and resource flows"
            elif intent == QueryIntent.BEHAVIORAL_ANALYSIS:
                return "Analyze AI service interaction patterns and behaviors"
            else:
                return "Provide container consciousness insights"
        
        elif system == IntelligenceSystem.MULTI_MODEL_ORACLE:
            if intent == QueryIntent.PREDICTIVE_QUERY:
                return f"Generate predictions for {', '.join([c.value for c in components])}"
            elif intent == QueryIntent.OPTIMIZATION_REQUEST:
                return f"Provide optimization strategies for {', '.join([c.value for c in components])}"
            else:
                return f"Provide intelligence orchestration for {', '.join([c.value for c in components])}"
        
        return f"Process query about {', '.join([c.value for c in components])}"
    
    def _generate_system_parameters(self, query_context: QueryContext, 
                                  system: IntelligenceSystem) -> Dict[str, Any]:
        """Generate parameters for system-specific queries"""
        parameters = {
            'components': [c.value for c in query_context.components],
            'confidence_threshold': 0.7,
            'max_results': 10
        }
        
        # Add temporal parameters if needed
        if query_context.time_range:
            parameters['time_range'] = {
                'start': query_context.time_range[0].isoformat(),
                'end': query_context.time_range[1].isoformat()
            }
        
        # Add system-specific parameters
        if system == IntelligenceSystem.TEMPORAL_INTELLIGENCE:
            parameters['include_correlations'] = True
            parameters['causal_analysis'] = query_context.requires_causal_analysis
        elif system == IntelligenceSystem.MULTI_MODEL_ORACLE:
            parameters['include_predictions'] = query_context.requires_prediction
            parameters['optimization_focus'] = query_context.requires_optimization
        
        return parameters
    
    def _system_matches_components(self, system: IntelligenceSystem, 
                                 components: List[SystemComponent]) -> bool:
        """Check if a consciousness system is well-suited for the target components"""
        system_caps = self.system_capabilities.get(system, {})
        expertise = system_caps.get('primary_expertise', [])
        
        for component in components:
            if component == SystemComponent.RTX5090_GPU and 'gpu_performance' in expertise:
                return True
            elif component == SystemComponent.AMD_ZEN5_CPU and 'cpu_optimization' in expertise:
                return True
            elif component == SystemComponent.THERMAL_SYSTEM and 'thermal_management' in expertise:
                return True
            elif component == SystemComponent.AI_SERVICES and 'service_orchestration' in expertise:
                return True
            elif component == SystemComponent.CONTAINER_CONSCIOUSNESS and 'ai_service_interactions' in expertise:
                return True
        
        return False