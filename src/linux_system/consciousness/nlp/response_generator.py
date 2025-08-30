"""
Response Generator - Human-Readable System Intelligence Responses
================================================================

Generates natural, human-readable responses from system consciousness queries,
incorporating contextual understanding and technical depth appropriate for the user.

Key capabilities:
- Natural language response generation from system data
- Context-aware formatting and technical depth adjustment
- Data visualization and chart suggestions for D3.js frontend
- Follow-up question suggestions and conversation flow
- System insights and predictive recommendations
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .intent_classifier import QueryIntent, SystemEntity, ExtractedQuery
from .context_manager import ConversationContext, UserPreferences
from ..query_commands import QueryResult

logger = logging.getLogger(__name__)


class ResponseStyle(Enum):
    """Response formatting styles."""
    CONCISE = "concise"       # Brief, to-the-point responses
    DETAILED = "detailed"     # Comprehensive explanations
    BALANCED = "balanced"     # Mix of detail and brevity
    TECHNICAL = "technical"   # Technical depth with specifics
    EXECUTIVE = "executive"   # High-level summary focus


@dataclass
class GeneratedResponse:
    """Complete generated response with metadata."""
    message: str
    confidence: float
    technical_level: float
    data_references: List[Dict[str, Any]]
    visualization_suggestions: List[str]
    follow_up_suggestions: List[str]
    insights: List[str]
    recommendations: List[str]
    response_metadata: Dict[str, Any]


@dataclass
class DataInsight:
    """Structured insight extracted from system data."""
    insight_type: str  # "trend", "anomaly", "threshold", "correlation"
    description: str
    significance: float  # 0.0-1.0
    supporting_data: Dict[str, Any]
    recommendation: Optional[str] = None


class ResponseGenerator:
    """
    Generate human-readable responses from system consciousness data.
    
    Transforms technical system data into natural language responses
    appropriate for user context and preferences.
    """
    
    def __init__(self):
        """Initialize response generator with templates and patterns."""
        self.response_templates = self._build_response_templates()
        self.insight_generators = self._build_insight_generators()
        self.visualization_suggestions = self._build_visualization_suggestions()
        
    def generate_response(self, 
                         extracted_query: ExtractedQuery,
                         query_result: QueryResult,
                         context: Optional[ConversationContext] = None) -> GeneratedResponse:
        """
        Generate comprehensive response from query results.
        
        Main entry point for response generation.
        """
        try:
            # Determine response style from context/preferences
            response_style = self._determine_response_style(extracted_query, context)
            
            # Extract insights from data
            insights = self._extract_insights(query_result, extracted_query)
            
            # Generate main response message
            message = self._generate_main_message(
                extracted_query, query_result, response_style, insights
            )
            
            # Generate data references for frontend
            data_references = self._generate_data_references(query_result, extracted_query)
            
            # Generate visualization suggestions
            viz_suggestions = self._generate_visualization_suggestions(
                extracted_query, query_result, insights
            )
            
            # Generate follow-up suggestions
            follow_ups = self._generate_follow_up_suggestions(
                extracted_query, query_result, context, insights
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(insights, extracted_query)
            
            # Calculate response confidence
            confidence = self._calculate_response_confidence(
                query_result, extracted_query, insights
            )
            
            return GeneratedResponse(
                message=message,
                confidence=confidence,
                technical_level=self._get_technical_level(context),
                data_references=data_references,
                visualization_suggestions=viz_suggestions,
                follow_up_suggestions=follow_ups,
                insights=[insight.description for insight in insights],
                recommendations=recommendations,
                response_metadata={
                    "response_style": response_style.value,
                    "insights_count": len(insights),
                    "query_intent": extracted_query.intent.value,
                    "entities_addressed": [e.value for e in extracted_query.entities]
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_error_response(str(e))
    
    def _determine_response_style(self, extracted_query: ExtractedQuery, 
                                context: Optional[ConversationContext]) -> ResponseStyle:
        """Determine appropriate response style."""
        if context and context.preferences:
            # Use user preferences
            if context.preferences.conversation_style == "detailed":
                return ResponseStyle.DETAILED
            elif context.preferences.conversation_style == "concise":
                return ResponseStyle.CONCISE
            elif context.preferences.technical_level > 0.7:
                return ResponseStyle.TECHNICAL
            else:
                return ResponseStyle.BALANCED
        
        # Determine from query intent
        if extracted_query.intent == QueryIntent.TROUBLESHOOTING:
            return ResponseStyle.DETAILED  # Troubleshooting needs detail
        elif extracted_query.intent == QueryIntent.MONITORING:
            return ResponseStyle.CONCISE   # Monitoring is often quick checks
        elif extracted_query.intent == QueryIntent.ANALYSIS:
            return ResponseStyle.TECHNICAL # Analysis needs technical depth
        else:
            return ResponseStyle.BALANCED
    
    def _extract_insights(self, query_result: QueryResult, 
                         extracted_query: ExtractedQuery) -> List[DataInsight]:
        """Extract meaningful insights from query results."""
        insights = []
        
        if not query_result.success or not query_result.data:
            return insights
        
        # Extract insights based on query intent
        if extracted_query.intent == QueryIntent.MONITORING:
            insights.extend(self._extract_monitoring_insights(query_result, extracted_query))
        elif extracted_query.intent == QueryIntent.ANALYSIS:
            insights.extend(self._extract_analysis_insights(query_result, extracted_query))
        elif extracted_query.intent == QueryIntent.TROUBLESHOOTING:
            insights.extend(self._extract_troubleshooting_insights(query_result, extracted_query))
        
        # Sort insights by significance
        insights.sort(key=lambda x: x.significance, reverse=True)
        
        return insights[:5]  # Return top 5 insights
    
    def _extract_monitoring_insights(self, query_result: QueryResult, 
                                   extracted_query: ExtractedQuery) -> List[DataInsight]:
        """Extract insights for monitoring queries."""
        insights = []
        data = query_result.data
        
        # GPU monitoring insights
        if 'gpu' in data and isinstance(data['gpu'], dict):
            gpu_data = data['gpu']
            
            # Temperature insights
            if 'temperature' in gpu_data:
                temp = gpu_data['temperature']
                if temp > 80:
                    insights.append(DataInsight(
                        insight_type="threshold",
                        description=f"GPU temperature is elevated at {temp}°C",
                        significance=0.8,
                        supporting_data={"temperature": temp, "threshold": 80},
                        recommendation="Monitor thermal throttling and consider increasing fan speeds"
                    ))
                elif temp > 75:
                    insights.append(DataInsight(
                        insight_type="threshold",
                        description=f"GPU temperature is moderately high at {temp}°C",
                        significance=0.6,
                        supporting_data={"temperature": temp, "threshold": 75},
                        recommendation="Keep monitoring thermal performance"
                    ))
            
            # Utilization insights
            if 'utilization' in gpu_data:
                util_data = gpu_data['utilization']
                if isinstance(util_data, dict) and 'gpu' in util_data:
                    gpu_util = util_data['gpu']
                    if gpu_util > 95:
                        insights.append(DataInsight(
                            insight_type="threshold",
                            description=f"GPU utilization is very high at {gpu_util}%",
                            significance=0.9,
                            supporting_data={"utilization": gpu_util},
                            recommendation="System is under heavy GPU load - consider workload optimization"
                        ))
        
        # Container monitoring insights
        if 'containers' in data and isinstance(data['containers'], dict):
            container_data = data['containers']
            
            if 'containers' in container_data and isinstance(container_data['containers'], list):
                containers = container_data['containers']
                
                # Health status insights
                unhealthy_containers = [c for c in containers if c.get('health_status') != 'healthy']
                if unhealthy_containers:
                    insights.append(DataInsight(
                        insight_type="anomaly",
                        description=f"{len(unhealthy_containers)} containers are not healthy",
                        significance=0.9,
                        supporting_data={"unhealthy_containers": [c.get('id') for c in unhealthy_containers]},
                        recommendation="Check container logs and resource allocation"
                    ))
        
        # Memory monitoring insights
        if 'memory' in data and isinstance(data['memory'], dict):
            memory_data = data['memory']
            
            if 'total' in memory_data and 'used' in memory_data:
                total = memory_data['total']
                used = memory_data['used']
                usage_percent = (used / total) * 100
                
                if usage_percent > 90:
                    insights.append(DataInsight(
                        insight_type="threshold",
                        description=f"Memory usage is very high at {usage_percent:.1f}%",
                        significance=0.8,
                        supporting_data={"usage_percent": usage_percent, "used_gb": used / (1024**3)},
                        recommendation="Consider closing unnecessary applications or adding more RAM"
                    ))
        
        return insights
    
    def _extract_analysis_insights(self, query_result: QueryResult, 
                                 extracted_query: ExtractedQuery) -> List[DataInsight]:
        """Extract insights for analysis queries."""
        insights = []
        
        # For temporal data analysis
        if isinstance(query_result.data, list) and query_result.data:
            # Look for trends in temporal data
            insights.extend(self._analyze_temporal_trends(query_result.data))
        
        return insights
    
    def _extract_troubleshooting_insights(self, query_result: QueryResult, 
                                        extracted_query: ExtractedQuery) -> List[DataInsight]:
        """Extract insights for troubleshooting queries."""
        insights = []
        
        # Look for anomalies and issues in the data
        if hasattr(query_result.data, 'get') and query_result.data.get('events'):
            events = query_result.data['events']
            
            # Find critical events
            critical_events = [e for e in events if getattr(e, 'severity', 'info') == 'critical']
            if critical_events:
                insights.append(DataInsight(
                    insight_type="anomaly",
                    description=f"Found {len(critical_events)} critical system events",
                    significance=0.9,
                    supporting_data={"critical_events": len(critical_events)},
                    recommendation="Investigate critical events for root cause analysis"
                ))
        
        return insights
    
    def _analyze_temporal_trends(self, temporal_data: List[Any]) -> List[DataInsight]:
        """Analyze trends in temporal data."""
        insights = []
        
        try:
            # Simple trend analysis (would be more sophisticated in production)
            if len(temporal_data) >= 3:
                insights.append(DataInsight(
                    insight_type="trend",
                    description=f"Analyzing {len(temporal_data)} data points over time",
                    significance=0.5,
                    supporting_data={"data_points": len(temporal_data)},
                    recommendation="Review temporal patterns for optimization opportunities"
                ))
        except Exception as e:
            logger.error(f"Error analyzing temporal trends: {e}")
        
        return insights
    
    def _generate_main_message(self, extracted_query: ExtractedQuery, 
                              query_result: QueryResult,
                              response_style: ResponseStyle,
                              insights: List[DataInsight]) -> str:
        """Generate the main response message."""
        
        if not query_result.success:
            return self._generate_error_message(query_result.error, extracted_query)
        
        # Get base message template
        base_message = self._get_base_message(extracted_query, query_result)
        
        # Add insights based on style
        if response_style in [ResponseStyle.DETAILED, ResponseStyle.TECHNICAL] and insights:
            insight_text = self._format_insights_for_message(insights, response_style)
            base_message += f"\n\n{insight_text}"
        
        # Adjust message tone based on style
        if response_style == ResponseStyle.CONCISE:
            base_message = self._make_message_concise(base_message)
        elif response_style == ResponseStyle.TECHNICAL:
            base_message = self._add_technical_details(base_message, query_result)
        
        return base_message
    
    def _get_base_message(self, extracted_query: ExtractedQuery, 
                         query_result: QueryResult) -> str:
        """Get base message template for query."""
        
        intent = extracted_query.intent
        entities = extracted_query.entities
        
        if intent == QueryIntent.MONITORING:
            return self._generate_monitoring_message(entities, query_result)
        elif intent == QueryIntent.ANALYSIS:
            return self._generate_analysis_message(entities, query_result)
        elif intent == QueryIntent.TROUBLESHOOTING:
            return self._generate_troubleshooting_message(entities, query_result)
        elif intent == QueryIntent.EXPLORATION:
            return self._generate_exploration_message(entities, query_result)
        elif intent == QueryIntent.OPTIMIZATION:
            return self._generate_optimization_message(entities, query_result)
        else:
            return self._generate_generic_message(query_result)
    
    def _generate_monitoring_message(self, entities: List[SystemEntity], 
                                   query_result: QueryResult) -> str:
        """Generate monitoring response message."""
        data = query_result.data
        
        if not entities:
            return "Here's your current system status overview:"
        
        primary_entity = entities[0]
        
        if primary_entity == SystemEntity.GPU and 'gpu' in data:
            gpu_data = data['gpu']
            temp = gpu_data.get('temperature', 'unknown')
            util = gpu_data.get('utilization', {}).get('gpu', 'unknown')
            
            return f"Your RTX 5090 is currently at {temp}°C with {util}% utilization. "
        
        elif primary_entity == SystemEntity.CONTAINERS and 'containers' in data:
            container_data = data['containers']
            containers = container_data.get('containers', [])
            healthy_count = sum(1 for c in containers if c.get('health_status') == 'healthy')
            total_count = len(containers)
            
            return f"You have {healthy_count} of {total_count} containers running healthy. "
        
        elif primary_entity == SystemEntity.MEMORY and 'memory' in data:
            memory_data = data['memory']
            if 'total' in memory_data and 'used' in memory_data:
                total_gb = memory_data['total'] / (1024**3)
                used_gb = memory_data['used'] / (1024**3)
                usage_percent = (memory_data['used'] / memory_data['total']) * 100
                
                return f"Memory usage is {used_gb:.1f}GB of {total_gb:.1f}GB ({usage_percent:.1f}%). "
        
        return "Current system monitoring data retrieved successfully."
    
    def _generate_analysis_message(self, entities: List[SystemEntity], 
                                 query_result: QueryResult) -> str:
        """Generate analysis response message."""
        if isinstance(query_result.data, list):
            data_points = len(query_result.data)
            return f"Analysis complete. Processed {data_points} data points to identify patterns and trends."
        else:
            return "System analysis shows current state and key performance indicators."
    
    def _generate_troubleshooting_message(self, entities: List[SystemEntity], 
                                        query_result: QueryResult) -> str:
        """Generate troubleshooting response message."""
        return "I've analyzed the system for potential issues and patterns that might explain the behavior you're seeing."
    
    def _generate_exploration_message(self, entities: List[SystemEntity], 
                                    query_result: QueryResult) -> str:
        """Generate exploration response message."""
        return "Here's what I found in the historical system data:"
    
    def _generate_optimization_message(self, entities: List[SystemEntity], 
                                     query_result: QueryResult) -> str:
        """Generate optimization response message."""
        return "Based on system analysis, here are optimization recommendations:"
    
    def _generate_generic_message(self, query_result: QueryResult) -> str:
        """Generate generic response message."""
        return "I've processed your system query and found the following information:"
    
    def _format_insights_for_message(self, insights: List[DataInsight], 
                                   response_style: ResponseStyle) -> str:
        """Format insights for inclusion in message."""
        if not insights:
            return ""
        
        insight_text = "Key insights:\n"
        
        for insight in insights[:3]:  # Top 3 insights
            if response_style == ResponseStyle.TECHNICAL:
                insight_text += f"• {insight.description} (significance: {insight.significance:.1f})\n"
            else:
                insight_text += f"• {insight.description}\n"
        
        return insight_text.strip()
    
    def _generate_data_references(self, query_result: QueryResult, 
                                extracted_query: ExtractedQuery) -> List[Dict[str, Any]]:
        """Generate data references for frontend consumption."""
        references = []
        
        try:
            if query_result.success and query_result.data:
                # Create reference to the main data
                references.append({
                    "type": "primary_data",
                    "query_type": extracted_query.intent.value,
                    "entities": [e.value for e in extracted_query.entities],
                    "data_summary": self._summarize_data_for_reference(query_result.data),
                    "timestamp": query_result.timestamp.isoformat()
                })
                
                # Add specific data references based on content
                if isinstance(query_result.data, dict):
                    for key, value in query_result.data.items():
                        if key in ['gpu', 'containers', 'memory', 'processes']:
                            references.append({
                                "type": "component_data",
                                "component": key,
                                "data_type": type(value).__name__,
                                "has_data": bool(value)
                            })
        
        except Exception as e:
            logger.error(f"Error generating data references: {e}")
        
        return references
    
    def _summarize_data_for_reference(self, data: Any) -> Dict[str, Any]:
        """Create summary of data for reference."""
        summary = {
            "data_type": type(data).__name__,
            "is_empty": not bool(data)
        }
        
        if isinstance(data, dict):
            summary["keys"] = list(data.keys())[:5]  # First 5 keys
            summary["key_count"] = len(data)
        elif isinstance(data, list):
            summary["length"] = len(data)
            summary["has_items"] = len(data) > 0
        
        return summary
    
    def _generate_visualization_suggestions(self, extracted_query: ExtractedQuery,
                                          query_result: QueryResult,
                                          insights: List[DataInsight]) -> List[str]:
        """Generate D3.js visualization suggestions."""
        suggestions = []
        
        # Based on query intent
        if extracted_query.intent == QueryIntent.MONITORING:
            suggestions.extend([
                "real_time_metrics_dashboard",
                "gauge_charts_for_utilization",
                "status_indicator_grid"
            ])
        elif extracted_query.intent == QueryIntent.ANALYSIS:
            suggestions.extend([
                "time_series_line_charts",
                "correlation_heatmap",
                "trend_analysis_chart"
            ])
        elif extracted_query.intent == QueryIntent.TROUBLESHOOTING:
            suggestions.extend([
                "event_timeline",
                "anomaly_detection_chart",
                "cause_effect_flow_diagram"
            ])
        
        # Based on entities
        for entity in extracted_query.entities:
            if entity == SystemEntity.GPU:
                suggestions.extend([
                    "gpu_temperature_gauge",
                    "gpu_utilization_chart",
                    "thermal_management_timeline"
                ])
            elif entity == SystemEntity.CONTAINERS:
                suggestions.extend([
                    "container_resource_treemap",
                    "service_dependency_graph",
                    "container_health_matrix"
                ])
            elif entity == SystemEntity.MEMORY:
                suggestions.extend([
                    "memory_usage_stacked_area",
                    "memory_allocation_donut",
                    "memory_pressure_timeline"
                ])
        
        # Remove duplicates and limit
        unique_suggestions = list(dict.fromkeys(suggestions))[:5]
        
        return unique_suggestions
    
    def _generate_follow_up_suggestions(self, extracted_query: ExtractedQuery,
                                      query_result: QueryResult,
                                      context: Optional[ConversationContext],
                                      insights: List[DataInsight]) -> List[str]:
        """Generate follow-up question suggestions."""
        suggestions = []
        
        # Based on insights
        for insight in insights[:2]:  # Top 2 insights
            if insight.insight_type == "threshold":
                suggestions.append("How can I optimize this metric?")
            elif insight.insight_type == "anomaly":
                suggestions.append("What caused this anomaly?")
            elif insight.insight_type == "trend":
                suggestions.append("Show me historical patterns")
        
        # Based on query intent
        if extracted_query.intent == QueryIntent.MONITORING:
            suggestions.extend([
                "Show me historical trends for this metric",
                "Compare with yesterday's performance",
                "What are the optimization recommendations?"
            ])
        elif extracted_query.intent == QueryIntent.TROUBLESHOOTING:
            suggestions.extend([
                "How can I prevent this issue?",
                "Show me related system events",
                "What are the recommended fixes?"
            ])
        elif extracted_query.intent == QueryIntent.ANALYSIS:
            suggestions.extend([
                "Dive deeper into this pattern",
                "Compare with other time periods",
                "What factors influence this metric?"
            ])
        
        # Limit and remove duplicates
        unique_suggestions = list(dict.fromkeys(suggestions))[:4]
        
        return unique_suggestions
    
    def _generate_recommendations(self, insights: List[DataInsight], 
                                extracted_query: ExtractedQuery) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # From insights
        for insight in insights:
            if insight.recommendation:
                recommendations.append(insight.recommendation)
        
        # General recommendations based on intent
        if extracted_query.intent == QueryIntent.MONITORING:
            recommendations.append("Set up alerts for critical thresholds")
        elif extracted_query.intent == QueryIntent.OPTIMIZATION:
            recommendations.append("Monitor performance after implementing changes")
        elif extracted_query.intent == QueryIntent.TROUBLESHOOTING:
            recommendations.append("Document the resolution for future reference")
        
        # Limit recommendations
        return recommendations[:3]
    
    def _calculate_response_confidence(self, query_result: QueryResult,
                                     extracted_query: ExtractedQuery,
                                     insights: List[DataInsight]) -> float:
        """Calculate confidence in generated response."""
        confidence = 0.5  # Base confidence
        
        # Boost for successful query
        if query_result.success:
            confidence += 0.3
        
        # Boost for clear intent
        if extracted_query.confidence > 0.7:
            confidence += 0.2
        
        # Boost for meaningful insights
        if insights:
            avg_insight_significance = sum(i.significance for i in insights) / len(insights)
            confidence += avg_insight_significance * 0.2
        
        # Boost for data availability
        if query_result.data:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _get_technical_level(self, context: Optional[ConversationContext]) -> float:
        """Get technical level for response."""
        if context and context.preferences:
            return context.preferences.technical_level
        return 0.5  # Default medium technical level
    
    def _make_message_concise(self, message: str) -> str:
        """Make message more concise."""
        # Remove extra explanations and keep core facts
        sentences = message.split('. ')
        if len(sentences) > 2:
            return '. '.join(sentences[:2]) + '.'
        return message
    
    def _add_technical_details(self, message: str, query_result: QueryResult) -> str:
        """Add technical details to message."""
        technical_details = f"\n\nTechnical details: Query executed in {query_result.execution_time_ms:.1f}ms"
        
        if hasattr(query_result, 'metadata') and query_result.metadata:
            technical_details += f", returned {len(query_result.data) if hasattr(query_result.data, '__len__') else 'data'} items"
        
        return message + technical_details
    
    def _generate_error_message(self, error: str, extracted_query: ExtractedQuery) -> str:
        """Generate user-friendly error message."""
        base_message = "I encountered an issue while processing your request. "
        
        # Make error more user-friendly
        if "timeout" in error.lower():
            return base_message + "The query took longer than expected. Try a smaller time range or more specific criteria."
        elif "not found" in error.lower():
            return base_message + "The requested data wasn't found. Check that the system component is active."
        elif "permission" in error.lower():
            return base_message + "I don't have permission to access that information."
        else:
            return base_message + "Please try rephrasing your question or check system connectivity."
    
    def _generate_error_response(self, error: str) -> GeneratedResponse:
        """Generate error response."""
        return GeneratedResponse(
            message=f"I apologize, but I encountered an error while generating your response: {error}",
            confidence=0.0,
            technical_level=0.0,
            data_references=[],
            visualization_suggestions=[],
            follow_up_suggestions=["Try asking a simpler question", "Check system connectivity"],
            insights=[],
            recommendations=["Contact system administrator if the issue persists"],
            response_metadata={"error": True, "error_message": error}
        )
    
    def _build_response_templates(self) -> Dict[str, Dict[str, str]]:
        """Build response templates (placeholder for future expansion)."""
        return {
            # Would contain more sophisticated templates in production
            "monitoring": {
                "gpu": "GPU status: {temperature}°C, {utilization}% utilization",
                "containers": "{healthy_count} of {total_count} containers healthy",
                "memory": "Memory: {used_gb}GB of {total_gb}GB ({usage_percent}%)"
            }
        }
    
    def _build_insight_generators(self) -> Dict[str, Any]:
        """Build insight generators (placeholder for future expansion)."""
        return {
            # Would contain more sophisticated insight generation logic
        }
    
    def _build_visualization_suggestions(self) -> Dict[str, List[str]]:
        """Build visualization suggestions mapping."""
        return {
            "gpu": ["temperature_gauge", "utilization_chart", "thermal_timeline"],
            "containers": ["resource_treemap", "health_matrix", "dependency_graph"],
            "memory": ["usage_area_chart", "allocation_donut", "pressure_timeline"],
            "processes": ["process_tree", "resource_usage_bars", "activity_timeline"]
        }