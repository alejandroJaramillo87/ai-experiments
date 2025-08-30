"""
Response Synthesizer

Converts sophisticated consciousness system insights back into natural language responses.
Creates human-readable answers from complex ML analysis, temporal intelligence, and 
predictive data while maintaining technical accuracy and contextual relevance.

This synthesizer enables true digital symbiosis by translating machine consciousness
insights into intuitive human communication.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .intent_understanding_engine import QueryContext, QueryIntent, SystemComponent
from .query_intelligence_router import RoutingDecision, IntelligenceSystem
from .contextual_query_processor import ContextualEnrichment, ProcessedQuery

logger = logging.getLogger(__name__)


@dataclass
class SynthesizedResponse:
    """Natural language response synthesized from consciousness systems"""
    primary_answer: str
    confidence: float
    supporting_details: List[str]
    visualizations: List[str]
    follow_up_suggestions: List[str]
    technical_insights: Dict[str, Any]
    processing_metadata: Dict[str, Any]


class ResponseSynthesizer:
    """
    Synthesizes natural language responses from AI workstation consciousness insights.
    
    Converts complex ML analysis, temporal patterns, causal relationships, and predictive
    data into clear, contextually relevant human communication.
    """
    
    def __init__(self):
        self.response_templates = self._build_response_templates()
        self.technical_translators = self._build_technical_translators()
        self.visualization_mappers = self._build_visualization_mappers()
        self.confidence_calibration = self._build_confidence_calibration()
        
        logger.info("Response Synthesizer initialized with consciousness translation capabilities")
    
    async def synthesize_response(self, processed_query: ProcessedQuery,
                                consciousness_results: Dict[IntelligenceSystem, Dict[str, Any]],
                                processing_time: float) -> SynthesizedResponse:
        """
        Synthesize natural language response from consciousness system results.
        
        Args:
            processed_query: Query with rich contextual understanding
            consciousness_results: Results from each consciousness system
            processing_time: Time taken to process the query
            
        Returns:
            SynthesizedResponse with natural language answer and supporting data
        """
        query_context = processed_query.original_context
        routing_decision = processed_query.routing_decision
        enrichment = processed_query.contextual_enrichment
        
        # Generate primary response based on intent and consciousness results
        primary_answer = await self._generate_primary_answer(
            query_context, consciousness_results, enrichment, routing_decision
        )
        
        # Calculate response confidence
        confidence = self._calculate_response_confidence(
            query_context, consciousness_results, enrichment
        )
        
        # Generate supporting details
        supporting_details = await self._generate_supporting_details(
            query_context, consciousness_results, enrichment
        )
        
        # Suggest relevant visualizations
        visualizations = self._suggest_visualizations(
            query_context, consciousness_results, routing_decision
        )
        
        # Generate contextual follow-up suggestions
        follow_ups = self._generate_follow_up_suggestions(
            query_context, consciousness_results, enrichment
        )
        
        # Extract technical insights for advanced users
        technical_insights = self._extract_technical_insights(
            consciousness_results, enrichment
        )
        
        # Create processing metadata
        processing_metadata = {
            'processing_time_ms': processing_time * 1000,
            'systems_consulted': [routing_decision.primary_system.value] + [s.value for s in routing_decision.secondary_systems],
            'context_depth': enrichment.context_depth,
            'consciousness_confidence': enrichment.confidence,
            'routing_strategy': routing_decision.execution_strategy,
            'timestamp': datetime.now().isoformat()
        }
        
        return SynthesizedResponse(
            primary_answer=primary_answer,
            confidence=confidence,
            supporting_details=supporting_details,
            visualizations=visualizations,
            follow_up_suggestions=follow_ups,
            technical_insights=technical_insights,
            processing_metadata=processing_metadata
        )
    
    async def _generate_primary_answer(self, query_context: QueryContext,
                                     consciousness_results: Dict[IntelligenceSystem, Dict[str, Any]],
                                     enrichment: ContextualEnrichment,
                                     routing_decision: RoutingDecision) -> str:
        """Generate the primary natural language answer"""
        
        intent = query_context.intent
        components = query_context.components
        primary_system = routing_decision.primary_system
        
        # Get the template generator for this intent
        template_generator = self.response_templates.get(
            intent, self._generate_general_response
        )
        
        # Get primary system results
        primary_results = consciousness_results.get(primary_system, {})
        
        # Generate intent-specific response
        response = template_generator(
            query_context, primary_results, consciousness_results, enrichment
        )
        
        # Add contextual enrichment if significant
        if enrichment.context_depth == 'deep' and enrichment.confidence > 0.7:
            contextual_addition = await self._add_contextual_insights(
                query_context, enrichment, consciousness_results
            )
            if contextual_addition:
                response += f"\n\n{contextual_addition}"
        
        return response.strip()
    
    def _generate_causal_analysis_response(self, query_context: QueryContext,
                                         primary_results: Dict[str, Any],
                                         consciousness_results: Dict[IntelligenceSystem, Dict[str, Any]],
                                         enrichment: ContextualEnrichment) -> str:
        """Generate response for causal analysis queries"""
        components_str = self._format_components_list(query_context.components)
        
        response = f"Based on my temporal intelligence analysis of your {components_str}"
        
        # Add time context if available
        if query_context.time_range:
            time_desc = self._format_time_range(query_context.time_range)
            response += f" {time_desc}"
        
        response += ":\n\n"
        
        # Add causal analysis results
        causal_context = enrichment.causal_context
        if causal_context.get('root_causes'):
            response += "**Root Cause Analysis:**\n"
            for cause in causal_context['root_causes'][:3]:  # Limit to top 3
                response += f"• {self._humanize_technical_term(cause)}\n"
            response += "\n"
        
        # Add causal chains
        if causal_context.get('causal_chains'):
            response += "**Causal Relationships:**\n"
            for chain in causal_context['causal_chains'][:2]:
                response += f"• {self._format_causal_chain(chain)}\n"
            response += "\n"
        
        # Add contributing factors
        if causal_context.get('contributing_factors'):
            response += "**Contributing Factors:**\n"
            for factor in causal_context['contributing_factors'][:3]:
                response += f"• {self._humanize_technical_term(factor)}\n"
        
        return response
    
    def _generate_status_response(self, query_context: QueryContext,
                                primary_results: Dict[str, Any],
                                consciousness_results: Dict[IntelligenceSystem, Dict[str, Any]],
                                enrichment: ContextualEnrichment) -> str:
        """Generate response for status queries"""
        system_state = enrichment.system_state_context
        
        health = system_state.get('system_health', 'unknown')
        score = system_state.get('performance_score', 0.0)
        mode = system_state.get('current_mode', 'unknown')
        
        response = f"Your AI workstation is currently in **{health}** condition, "
        response += f"operating in **{mode}** mode with an overall performance score of **{score:.1%}**.\n\n"
        
        # Add component-specific status
        component_states = system_state.get('component_states', {})
        if component_states:
            response += "**Component Status:**\n"
            for component_name, state_data in component_states.items():
                status_summary = self._summarize_component_status(component_name, state_data)
                response += f"• **{component_name.title()}**: {status_summary}\n"
            response += "\n"
        
        # Add active workloads if relevant
        active_workloads = system_state.get('active_workloads', [])
        if active_workloads:
            response += f"**Active AI Services**: {', '.join(active_workloads)}\n\n"
        
        # Add performance insights
        hardware_data = consciousness_results.get(IntelligenceSystem.HARDWARE_SPECIALIZATION, {})
        if hardware_data:
            insights = self._extract_performance_insights(hardware_data)
            if insights:
                response += f"**Performance Insights**: {insights}\n"
        
        return response
    
    def _generate_predictive_response(self, query_context: QueryContext,
                                    primary_results: Dict[str, Any],
                                    consciousness_results: Dict[IntelligenceSystem, Dict[str, Any]],
                                    enrichment: ContextualEnrichment) -> str:
        """Generate response for predictive queries"""
        predictive_context = enrichment.predictive_context
        
        confidence_levels = predictive_context.get('confidence_levels', {})
        avg_confidence = sum(confidence_levels.values()) / len(confidence_levels) if confidence_levels else 0.6
        
        response = f"Based on my multi-model oracle analysis (confidence: **{avg_confidence:.1%}**), here are my predictions:\n\n"
        
        # Add predictions
        predictions = predictive_context.get('predictions', [])
        if predictions:
            response += "**System Predictions:**\n"
            for prediction in predictions[:3]:
                response += f"• {self._format_prediction(prediction)}\n"
            response += "\n"
        
        # Add component-specific predictions
        for component in query_context.components:
            component_key = f"{component.value}_predictions"
            component_preds = predictive_context.get(component_key, {})
            if component_preds:
                short_term = component_preds.get('short_term', [])
                if short_term:
                    response += f"**{component.value.replace('_', ' ').title()} Forecast:**\n"
                    for pred in short_term[:2]:
                        response += f"• {self._format_prediction(pred)}\n"
                    response += "\n"
        
        # Add risk factors
        risk_factors = predictive_context.get('risk_factors', [])
        if risk_factors:
            response += "**Risk Factors to Monitor:**\n"
            for risk in risk_factors[:3]:
                response += f"⚠️ {self._humanize_technical_term(risk)}\n"
        
        return response
    
    def _generate_optimization_response(self, query_context: QueryContext,
                                      primary_results: Dict[str, Any],
                                      consciousness_results: Dict[IntelligenceSystem, Dict[str, Any]],
                                      enrichment: ContextualEnrichment) -> str:
        """Generate response for optimization requests"""
        components_str = self._format_components_list(query_context.components)
        
        response = f"Here are optimization opportunities I've identified for your {components_str}:\n\n"
        
        # Add optimization opportunities
        predictive_context = enrichment.predictive_context
        opportunities = predictive_context.get('optimization_opportunities', [])
        if opportunities:
            response += "**Optimization Opportunities:**\n"
            for opp in opportunities[:4]:
                formatted_opp = self._format_optimization_opportunity(opp)
                response += f"• {formatted_opp}\n"
            response += "\n"
        
        # Add oracle-specific optimization strategies
        oracle_results = consciousness_results.get(IntelligenceSystem.MULTI_MODEL_ORACLE, {})
        strategies = oracle_results.get('optimization_strategies', [])
        if strategies:
            response += "**Recommended Strategies:**\n"
            for strategy in strategies[:3]:
                response += f"• {self._format_strategy(strategy)}\n"
            response += "\n"
        
        # Add performance bottlenecks if identified
        bottlenecks = oracle_results.get('performance_bottlenecks', [])
        if bottlenecks:
            response += "**Performance Bottlenecks:**\n"
            for bottleneck in bottlenecks[:3]:
                response += f"• {self._humanize_technical_term(bottleneck)}\n"
        
        return response
    
    def _generate_pattern_discovery_response(self, query_context: QueryContext,
                                           primary_results: Dict[str, Any],
                                           consciousness_results: Dict[IntelligenceSystem, Dict[str, Any]],
                                           enrichment: ContextualEnrichment) -> str:
        """Generate response for pattern discovery queries"""
        components_str = self._format_components_list(query_context.components)
        
        response = f"I've identified the following patterns in your {components_str} behavior:\n\n"
        
        # Add historical patterns
        historical_context = enrichment.historical_context
        patterns = historical_context.get('historical_patterns', [])
        if patterns:
            response += "**Behavioral Patterns:**\n"
            for pattern in patterns[:4]:
                response += f"• {self._format_pattern(pattern)}\n"
            response += "\n"
        
        # Add correlations
        correlation_context = enrichment.correlation_context
        correlations = correlation_context.get('temporal_correlations', [])
        if correlations:
            response += "**System Correlations:**\n"
            for corr in correlations[:3]:
                response += f"• {self._format_correlation(corr)}\n"
            response += "\n"
        
        # Add trend analysis
        trends = historical_context.get('trend_analysis', {})
        if trends:
            response += "**Trend Analysis:**\n"
            for trend_name, trend_data in trends.items():
                response += f"• **{trend_name}**: {self._format_trend(trend_data)}\n"
        
        return response
    
    def _generate_general_response(self, query_context: QueryContext,
                                 primary_results: Dict[str, Any],
                                 consciousness_results: Dict[IntelligenceSystem, Dict[str, Any]],
                                 enrichment: ContextualEnrichment) -> str:
        """Generate response for general or unclear queries"""
        system_state = enrichment.system_state_context
        
        response = f"Based on my analysis of your AI workstation consciousness:\n\n"
        
        # Add system overview
        health = system_state.get('system_health', 'unknown')
        performance = system_state.get('performance_score', 0.0)
        response += f"**System Status**: {health} (Performance: {performance:.1%})\n\n"
        
        # Add consciousness status
        consciousness_status = system_state.get('consciousness_status', {})
        active_systems = [name for name, active in consciousness_status.items() if active]
        if active_systems:
            response += f"**Active Consciousness Systems**: {', '.join(active_systems)}\n\n"
        
        # Add general insights
        response += "For more specific insights, try asking about:\n"
        response += "• Specific components like 'GPU performance' or 'thermal management'\n"
        response += "• Recent events with 'what happened yesterday'\n"
        response += "• Optimization with 'how can I improve performance'\n"
        response += "• Patterns with 'what patterns do you see'"
        
        return response
    
    async def _add_contextual_insights(self, query_context: QueryContext,
                                     enrichment: ContextualEnrichment,
                                     consciousness_results: Dict[IntelligenceSystem, Dict[str, Any]]) -> Optional[str]:
        """Add rich contextual insights for deep analysis"""
        insights = []
        
        # Add causal timeline if significant
        causal_context = enrichment.causal_context
        causal_timeline = causal_context.get('causal_timeline', [])
        if len(causal_timeline) > 2:
            insights.append("**Timeline Analysis**: Multiple related events detected in the specified timeframe")
        
        # Add correlation insights
        correlation_context = enrichment.correlation_context
        component_correlations = correlation_context.get('component_correlations', [])
        if len(component_correlations) > 1:
            insights.append(f"**Cross-System Impact**: {len(component_correlations)} component interactions identified")
        
        # Add predictive insights
        predictive_context = enrichment.predictive_context
        if predictive_context.get('prediction_horizon') == 'long':
            insights.append("**Long-term Forecast**: Extended prediction analysis performed")
        
        return " | ".join(insights) if insights else None
    
    async def _generate_supporting_details(self, query_context: QueryContext,
                                         consciousness_results: Dict[IntelligenceSystem, Dict[str, Any]],
                                         enrichment: ContextualEnrichment) -> List[str]:
        """Generate supporting details for the response"""
        details = []
        
        # Add temporal context details
        if query_context.time_range:
            time_span = query_context.time_range[1] - query_context.time_range[0]
            details.append(f"Analysis covers {time_span.total_seconds() / 3600:.1f} hours of system data")
        
        # Add consciousness system details
        active_systems = len([s for s, results in consciousness_results.items() if results])
        details.append(f"Consulted {active_systems} consciousness systems for comprehensive analysis")
        
        # Add confidence details
        if enrichment.confidence > 0.8:
            details.append("High-confidence analysis based on strong data correlation")
        elif enrichment.confidence < 0.5:
            details.append("Moderate confidence - consider gathering more specific data")
        
        # Add context depth details
        if enrichment.context_depth == 'deep':
            details.append("Deep contextual analysis including causal relationships and predictions")
        
        return details[:4]  # Limit to 4 supporting details
    
    def _suggest_visualizations(self, query_context: QueryContext,
                              consciousness_results: Dict[IntelligenceSystem, Dict[str, Any]],
                              routing_decision: RoutingDecision) -> List[str]:
        """Suggest relevant visualizations based on query and results"""
        visualizations = []
        
        # Intent-based visualizations
        intent_viz = self.visualization_mappers.get(query_context.intent, [])
        visualizations.extend(intent_viz)
        
        # Component-based visualizations
        for component in query_context.components:
            component_viz = self._get_component_visualizations(component)
            visualizations.extend(component_viz)
        
        # System-based visualizations
        if IntelligenceSystem.TEMPORAL_INTELLIGENCE in consciousness_results:
            visualizations.append('temporal_analysis_timeline')
        if IntelligenceSystem.HARDWARE_SPECIALIZATION in consciousness_results:
            visualizations.append('hardware_performance_dashboard')
        if IntelligenceSystem.CONTAINER_CONSCIOUSNESS in consciousness_results:
            visualizations.append('container_orchestration_flow')
        
        return list(set(visualizations))[:5]  # Remove duplicates, limit to 5
    
    def _generate_follow_up_suggestions(self, query_context: QueryContext,
                                      consciousness_results: Dict[IntelligenceSystem, Dict[str, Any]],
                                      enrichment: ContextualEnrichment) -> List[str]:
        """Generate contextual follow-up suggestions"""
        suggestions = []
        
        # Intent-based follow-ups
        intent = query_context.intent
        if intent == QueryIntent.CAUSAL_ANALYSIS:
            suggestions.extend([
                "How can I prevent this from happening again?",
                "What other components were affected during this time?",
                "Show me the performance timeline for this period"
            ])
        elif intent == QueryIntent.SYSTEM_STATUS:
            suggestions.extend([
                "What optimization opportunities exist right now?",
                "How does this compare to yesterday's performance?",
                "What patterns do you see in recent system behavior?"
            ])
        elif intent == QueryIntent.PREDICTIVE_QUERY:
            suggestions.extend([
                "What factors most influence these predictions?",
                "How can I improve the predicted outcomes?",
                "Show me similar historical scenarios"
            ])
        elif intent == QueryIntent.OPTIMIZATION_REQUEST:
            suggestions.extend([
                "What's the expected impact of these optimizations?",
                "Are there any risks with these changes?",
                "When is the best time to apply these optimizations?"
            ])
        
        # Component-specific follow-ups
        for component in query_context.components:
            component_suggestions = self._get_component_followups(component, enrichment)
            suggestions.extend(component_suggestions)
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    def _extract_technical_insights(self, consciousness_results: Dict[IntelligenceSystem, Dict[str, Any]],
                                   enrichment: ContextualEnrichment) -> Dict[str, Any]:
        """Extract technical insights for advanced users"""
        insights = {
            'processing_complexity': enrichment.context_depth,
            'confidence_breakdown': {},
            'system_correlations': [],
            'raw_metrics': {}
        }
        
        # Extract confidence breakdown
        insights['confidence_breakdown'] = {
            'overall': enrichment.confidence,
            'causal': enrichment.causal_context.get('causal_confidence', 0.0),
            'predictive': max(enrichment.predictive_context.get('confidence_levels', {}).values()) if enrichment.predictive_context.get('confidence_levels') else 0.0
        }
        
        # Extract correlations
        correlation_context = enrichment.correlation_context
        insights['system_correlations'] = correlation_context.get('component_correlations', [])
        
        # Extract raw metrics
        for system, results in consciousness_results.items():
            insights['raw_metrics'][system.value] = self._extract_key_metrics(results)
        
        return insights
    
    def _build_response_templates(self) -> Dict[QueryIntent, callable]:
        """Build response generation templates for each intent"""
        return {
            QueryIntent.CAUSAL_ANALYSIS: self._generate_causal_analysis_response,
            QueryIntent.ROOT_CAUSE: self._generate_causal_analysis_response,
            QueryIntent.SYSTEM_STATUS: self._generate_status_response,
            QueryIntent.COMPONENT_STATUS: self._generate_status_response,
            QueryIntent.PREDICTIVE_QUERY: self._generate_predictive_response,
            QueryIntent.RESOURCE_PLANNING: self._generate_predictive_response,
            QueryIntent.OPTIMIZATION_REQUEST: self._generate_optimization_response,
            QueryIntent.PATTERN_DISCOVERY: self._generate_pattern_discovery_response,
            QueryIntent.BEHAVIORAL_ANALYSIS: self._generate_pattern_discovery_response,
        }
    
    def _build_technical_translators(self) -> Dict[str, str]:
        """Build translators for technical terms to human language"""
        return {
            'thermal_throttling': 'temperature-based performance reduction',
            'vram_pressure': 'GPU memory usage approaching limits',
            'tensor_core_utilization': 'AI processing unit efficiency',
            'numa_locality': 'memory access optimization',
            'cache_thrashing': 'inefficient memory access patterns',
            'workload_imbalance': 'uneven distribution of AI tasks',
            'inference_latency': 'AI model response time',
            'batch_optimization': 'processing efficiency improvement',
            'resource_contention': 'competition for system resources',
            'service_orchestration': 'AI service coordination and management'
        }
    
    def _build_visualization_mappers(self) -> Dict[QueryIntent, List[str]]:
        """Map query intents to relevant visualizations"""
        return {
            QueryIntent.CAUSAL_ANALYSIS: ['causal_network_graph', 'temporal_analysis_timeline'],
            QueryIntent.SYSTEM_STATUS: ['system_dashboard', 'performance_overview'],
            QueryIntent.PREDICTIVE_QUERY: ['prediction_forecast_chart', 'trend_analysis'],
            QueryIntent.PATTERN_DISCOVERY: ['pattern_heatmap', 'behavioral_timeline'],
            QueryIntent.OPTIMIZATION_REQUEST: ['optimization_impact_chart', 'resource_allocation_view']
        }
    
    def _build_confidence_calibration(self) -> Dict[str, Tuple[float, str]]:
        """Build confidence level calibration"""
        return {
            'very_high': (0.9, 'Very confident - strong data correlation and clear patterns'),
            'high': (0.75, 'High confidence - good data availability and clear analysis'),
            'moderate': (0.6, 'Moderate confidence - adequate data with some uncertainty'),
            'low': (0.4, 'Low confidence - limited data or unclear patterns'),
            'very_low': (0.2, 'Very low confidence - insufficient data for reliable analysis')
        }
    
    # Utility methods for formatting and translation
    
    def _format_components_list(self, components: List[SystemComponent]) -> str:
        """Format components list for natural language"""
        if len(components) == 0:
            return "system"
        elif len(components) == 1:
            return components[0].value.replace('_', ' ')
        elif len(components) == 2:
            return f"{components[0].value.replace('_', ' ')} and {components[1].value.replace('_', ' ')}"
        else:
            formatted = [comp.value.replace('_', ' ') for comp in components[:-1]]
            return f"{', '.join(formatted)}, and {components[-1].value.replace('_', ' ')}"
    
    def _format_time_range(self, time_range: Tuple[datetime, datetime]) -> str:
        """Format time range for natural language"""
        start, end = time_range
        now = datetime.now()
        
        if start.date() == now.date():
            return f"from {start.strftime('%I:%M %p')} to {end.strftime('%I:%M %p')} today"
        elif start.date() == (now.date() - timedelta(days=1)):
            return f"yesterday from {start.strftime('%I:%M %p')} to {end.strftime('%I:%M %p')}"
        else:
            return f"from {start.strftime('%B %d, %I:%M %p')} to {end.strftime('%B %d, %I:%M %p')}"
    
    def _humanize_technical_term(self, term: str) -> str:
        """Convert technical term to human-readable description"""
        if isinstance(term, dict) and 'description' in term:
            return term['description']
        
        term_str = str(term).lower()
        return self.technical_translators.get(term_str, term_str.replace('_', ' ').title())
    
    def _format_causal_chain(self, chain) -> str:
        """Format causal chain for human reading"""
        if isinstance(chain, dict):
            cause = chain.get('cause', 'Unknown cause')
            effect = chain.get('effect', 'Unknown effect')
            return f"{self._humanize_technical_term(cause)} led to {self._humanize_technical_term(effect)}"
        return str(chain)
    
    def _summarize_component_status(self, component_name: str, state_data: Dict[str, Any]) -> str:
        """Summarize component status in human terms"""
        if component_name == 'gpu':
            temp = state_data.get('temperature', 0)
            util = state_data.get('utilization', 0)
            return f"Running at {temp}°C with {util:.1%} utilization"
        elif component_name == 'cpu':
            temp = state_data.get('temperature', 0)
            util = state_data.get('average_utilization', 0)
            return f"Operating at {temp}°C with {util:.1%} average utilization"
        elif component_name == 'thermal':
            efficiency = state_data.get('cooling_efficiency', 0)
            return f"Cooling efficiency at {efficiency:.1%}"
        else:
            return "Status available"
    
    def _extract_performance_insights(self, hardware_data: Dict[str, Any]) -> str:
        """Extract key performance insights from hardware data"""
        insights = []
        
        gpu_data = hardware_data.get('rtx5090_blackwall', {})
        if gpu_data.get('tensor_core_utilization', 0) > 0.9:
            insights.append("GPU tensor cores highly utilized")
        
        cpu_data = hardware_data.get('amd_zen5', {})
        if cpu_data.get('numa_efficiency', 0) > 0.85:
            insights.append("Excellent CPU memory locality")
        
        thermal_data = hardware_data.get('thermal_intelligence', {})
        if thermal_data.get('cooling_efficiency', 0) > 0.8:
            insights.append("Thermal management performing well")
        
        return '; '.join(insights) if insights else "System performing within normal parameters"
    
    def _format_prediction(self, prediction) -> str:
        """Format prediction for human reading"""
        if isinstance(prediction, dict):
            desc = prediction.get('description', str(prediction))
            confidence = prediction.get('confidence', 0)
            return f"{desc} (confidence: {confidence:.1%})"
        return str(prediction)
    
    def _format_optimization_opportunity(self, opportunity) -> str:
        """Format optimization opportunity for human reading"""
        if isinstance(opportunity, dict):
            category = opportunity.get('category', 'System')
            gain = opportunity.get('potential_gain', 0)
            desc = opportunity.get('description', 'Optimization available')
            return f"**{category.title()}**: {desc} (potential gain: {gain:.1%})"
        return str(opportunity)
    
    def _format_strategy(self, strategy) -> str:
        """Format optimization strategy for human reading"""
        if isinstance(strategy, dict):
            return strategy.get('description', str(strategy))
        return str(strategy)
    
    def _format_pattern(self, pattern) -> str:
        """Format pattern for human reading"""
        if isinstance(pattern, dict):
            return pattern.get('description', str(pattern))
        return str(pattern)
    
    def _format_correlation(self, correlation) -> str:
        """Format correlation for human reading"""
        if isinstance(correlation, dict):
            comp1 = correlation.get('component_1', 'Component A')
            comp2 = correlation.get('component_2', 'Component B')
            strength = correlation.get('strength', 0)
            return f"{comp1} correlates with {comp2} (strength: {strength:.1%})"
        return str(correlation)
    
    def _format_trend(self, trend_data) -> str:
        """Format trend data for human reading"""
        if isinstance(trend_data, dict):
            direction = trend_data.get('direction', 'stable')
            magnitude = trend_data.get('magnitude', 0)
            return f"{direction} trend with {magnitude:.1%} change"
        return str(trend_data)
    
    def _get_component_visualizations(self, component: SystemComponent) -> List[str]:
        """Get visualizations relevant to a component"""
        viz_map = {
            SystemComponent.RTX5090_GPU: ['gpu_thermal_heatmap', 'gpu_utilization_chart', 'vram_usage_timeline'],
            SystemComponent.AMD_ZEN5_CPU: ['cpu_core_utilization', 'cpu_performance_chart', 'numa_efficiency_view'],
            SystemComponent.THERMAL_SYSTEM: ['thermal_heatmap', 'fan_performance_chart', 'cooling_efficiency_graph'],
            SystemComponent.AI_SERVICES: ['service_orchestration_flow', 'inference_performance_chart'],
            SystemComponent.CONTAINER_CONSCIOUSNESS: ['container_resource_flows', 'service_topology_graph']
        }
        return viz_map.get(component, [])
    
    def _get_component_followups(self, component: SystemComponent, enrichment: ContextualEnrichment) -> List[str]:
        """Get follow-up suggestions for a component"""
        followup_map = {
            SystemComponent.RTX5090_GPU: [
                "What's the GPU thermal pattern over time?",
                "How is VRAM utilization trending?"
            ],
            SystemComponent.AMD_ZEN5_CPU: [
                "Show me CPU core utilization patterns",
                "How is memory bandwidth being used?"
            ],
            SystemComponent.THERMAL_SYSTEM: [
                "What's the cooling system efficiency trend?",
                "Are there thermal bottlenecks?"
            ]
        }
        return followup_map.get(component, [])[:2]  # Limit to 2 per component
    
    def _extract_key_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from system results"""
        key_metrics = {}
        
        # Extract common metrics
        for key in ['temperature', 'utilization', 'efficiency', 'performance_score']:
            if key in results:
                key_metrics[key] = results[key]
        
        return key_metrics
    
    def _calculate_response_confidence(self, query_context: QueryContext,
                                     consciousness_results: Dict[IntelligenceSystem, Dict[str, Any]],
                                     enrichment: ContextualEnrichment) -> float:
        """Calculate overall response confidence"""
        confidence_factors = []
        
        # Base confidence from query understanding
        confidence_factors.append(query_context.confidence)
        
        # Confidence from contextual enrichment
        confidence_factors.append(enrichment.confidence)
        
        # Confidence from number of systems that provided results
        systems_with_results = len([s for s, r in consciousness_results.items() if r])
        system_confidence = min(systems_with_results / 4.0, 1.0)  # Max 4 systems
        confidence_factors.append(system_confidence)
        
        # Confidence from context depth
        depth_confidence = {'shallow': 0.6, 'moderate': 0.8, 'deep': 1.0}
        confidence_factors.append(depth_confidence.get(enrichment.context_depth, 0.7))
        
        return sum(confidence_factors) / len(confidence_factors)