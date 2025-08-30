"""
Container Resource Correlator

Provides sophisticated correlation intelligence between container behavior,
system performance, and service interactions. Analyzes multi-dimensional
relationships across AI services with predictive insights and optimization
recommendations specific to AI workstation environments.

Features:
- Cross-service resource correlation and impact analysis
- Service interaction pattern detection and prediction
- Performance correlation with system-wide metrics
- AI workload behavior pattern recognition
- Resource contention detection and resolution recommendations
- Predictive correlation modeling for optimization insights
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from statistics import mean, stdev
from concurrent.futures import ThreadPoolExecutor
import threading

from ...temporal.core.types import Significance, ComponentType
from .ai_container_detector import AIServiceState, ContainerHealthMetrics, AIServiceConfig


logger = logging.getLogger(__name__)


@dataclass
class CorrelationMetric:
    """Represents correlation between two metrics or services."""
    metric_a: str
    metric_b: str
    correlation_strength: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    sample_size: int
    time_lag: Optional[float] = None  # seconds
    significance: Significance = Significance.LOW


@dataclass
class ServiceInteractionPattern:
    """Represents detected interaction patterns between services."""
    pattern_id: str
    pattern_type: str  # competition, collaboration, dependency, cascade
    involved_services: List[str]
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    description: str
    detected_at: datetime
    historical_occurrences: int = 0
    impact_metrics: Dict[str, float] = field(default_factory=dict)
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class ResourceCorrelationInsight:
    """Insight derived from resource correlation analysis."""
    insight_type: str
    title: str
    description: str
    affected_services: List[str]
    confidence: float
    potential_impact: str
    optimization_opportunity: Optional[str] = None
    recommended_actions: List[str] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceCorrelationModel:
    """Model for predicting performance based on correlations."""
    model_type: str
    target_metric: str
    input_features: List[str]
    correlation_coefficients: Dict[str, float]
    accuracy: float
    last_trained: datetime
    prediction_horizon: timedelta


class ContainerResourceCorrelator:
    """
    Sophisticated correlation intelligence for AI container ecosystems.
    
    Provides deep analysis of service interactions, resource correlations,
    and performance patterns with predictive capabilities optimized for
    AI workstation environments with multiple concurrent inference services.
    """
    
    def __init__(self, correlation_window: timedelta = timedelta(hours=4)):
        self.correlation_window = correlation_window
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ContainerCorrelator")
        self.analysis_lock = threading.RLock()
        
        # Time-series data storage for correlation analysis
        self.service_metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.system_metrics_history: deque = deque(maxlen=1000)
        self.interaction_patterns: Dict[str, ServiceInteractionPattern] = {}
        
        # Correlation tracking
        self.correlation_cache: Dict[str, CorrelationMetric] = {}
        self.correlation_update_interval = timedelta(minutes=5)
        self.last_correlation_update = datetime.min
        
        # AI service configuration knowledge
        self.ai_service_configs = {
            'llama-cpu-0': {'type': 'cpu_inference', 'cores': '0-7', 'expected_memory_gb': 32},
            'llama-cpu-1': {'type': 'cpu_inference', 'cores': '8-15', 'expected_memory_gb': 32},
            'llama-cpu-2': {'type': 'cpu_inference', 'cores': '16-23', 'expected_memory_gb': 32},
            'llama-gpu': {'type': 'gpu_inference', 'gpu_access': True, 'expected_vram_gb': 20},
            'vllm-gpu': {'type': 'vllm_inference', 'gpu_access': True, 'expected_vram_gb': 15},
            'open-webui': {'type': 'interface', 'lightweight': True}
        }
        
        # Performance models
        self.performance_models: Dict[str, PerformanceCorrelationModel] = {}
        self.model_training_threshold = 50  # Minimum samples for model training
        
        # Correlation analysis thresholds
        self.correlation_thresholds = {
            'strong_correlation': 0.7,
            'moderate_correlation': 0.4,
            'weak_correlation': 0.2,
            'minimum_samples': 10,
            'confidence_threshold': 0.6
        }
        
    def analyze_service_correlations(
        self, service_states: Dict[str, AIServiceState], system_context: Dict[str, Any]
    ) -> List[ResourceCorrelationInsight]:
        """Analyze correlations between services and generate insights."""
        insights = []
        
        with self.analysis_lock:
            # Update time-series data
            self._update_metrics_history(service_states, system_context)
            
            # Perform correlation analysis if sufficient data
            if self._should_update_correlations():
                correlations = self._calculate_correlations()
                self.correlation_cache.update(correlations)
                self.last_correlation_update = datetime.now()
                
            # Generate insights from correlations
            insights.extend(self._generate_correlation_insights())
            
            # Detect service interaction patterns
            interaction_insights = self._detect_service_interactions(service_states)
            insights.extend(interaction_insights)
            
            # Analyze resource contention patterns
            contention_insights = self._analyze_resource_contention(service_states)
            insights.extend(contention_insights)
            
            # Generate predictive insights
            predictive_insights = self._generate_predictive_insights(service_states)
            insights.extend(predictive_insights)
            
        return insights
        
    def _update_metrics_history(
        self, service_states: Dict[str, AIServiceState], system_context: Dict[str, Any]
    ):
        """Update historical metrics for correlation analysis."""
        current_time = datetime.now()
        
        # Update service metrics
        for service_name, service_state in service_states.items():
            if service_state.container_metrics:
                metrics = service_state.container_metrics
                metric_point = {
                    'timestamp': current_time,
                    'cpu_usage': metrics.cpu_usage_percent,
                    'memory_usage_mb': metrics.memory_usage_mb,
                    'memory_usage_percent': (metrics.memory_usage_mb / metrics.memory_limit_mb * 100 
                                           if metrics.memory_limit_mb > 0 else 0),
                    'network_rx_mb': metrics.network_rx_bytes / (1024 * 1024),
                    'network_tx_mb': metrics.network_tx_bytes / (1024 * 1024),
                    'health_status': metrics.health_status,
                    'uptime_hours': metrics.uptime_seconds / 3600,
                    'restart_count': metrics.restart_count,
                    'inference_active': service_state.inference_active,
                    'model_loaded': service_state.model_loaded is not None,
                    'alert_count': len(service_state.resource_alerts)
                }
                
                self.service_metrics_history[service_name].append(metric_point)
                
        # Update system-wide metrics
        system_point = {
            'timestamp': current_time,
            'total_running_services': sum(
                1 for state in service_states.values() 
                if state.container_metrics and state.container_metrics.status == 'running'
            ),
            'total_cpu_usage': sum(
                state.container_metrics.cpu_usage_percent for state in service_states.values()
                if state.container_metrics
            ),
            'total_memory_usage_mb': sum(
                state.container_metrics.memory_usage_mb for state in service_states.values()
                if state.container_metrics
            ),
            'services_with_alerts': sum(
                1 for state in service_states.values() if state.resource_alerts
            ),
            'active_inference_services': sum(
                1 for state in service_states.values() if state.inference_active
            ),
            **system_context
        }
        
        self.system_metrics_history.append(system_point)
        
    def _should_update_correlations(self) -> bool:
        """Determine if correlations should be recalculated."""
        return (datetime.now() - self.last_correlation_update > self.correlation_update_interval and
                len(self.system_metrics_history) >= self.correlation_thresholds['minimum_samples'])
                
    def _calculate_correlations(self) -> Dict[str, CorrelationMetric]:
        """Calculate correlations between various metrics."""
        correlations = {}
        
        # Get recent data within correlation window
        cutoff_time = datetime.now() - self.correlation_window
        
        # Service-to-service correlations
        service_correlations = self._calculate_service_correlations(cutoff_time)
        correlations.update(service_correlations)
        
        # Service-to-system correlations
        system_correlations = self._calculate_system_correlations(cutoff_time)
        correlations.update(system_correlations)
        
        # Cross-resource correlations within services
        resource_correlations = self._calculate_resource_correlations(cutoff_time)
        correlations.update(resource_correlations)
        
        return correlations
        
    def _calculate_service_correlations(self, cutoff_time: datetime) -> Dict[str, CorrelationMetric]:
        """Calculate correlations between different services."""
        correlations = {}
        service_names = list(self.service_metrics_history.keys())
        
        # Compare each pair of services
        for i in range(len(service_names)):
            for j in range(i + 1, len(service_names)):
                service_a = service_names[i]
                service_b = service_names[j]
                
                # Get recent data for both services
                data_a = [point for point in self.service_metrics_history[service_a] 
                         if point['timestamp'] > cutoff_time]
                data_b = [point for point in self.service_metrics_history[service_b]
                         if point['timestamp'] > cutoff_time]
                         
                if len(data_a) < self.correlation_thresholds['minimum_samples'] or len(data_b) < 10:
                    continue
                    
                # Calculate correlations for key metrics
                metric_pairs = [
                    ('cpu_usage', 'cpu_usage'),
                    ('memory_usage_percent', 'memory_usage_percent'),
                    ('cpu_usage', 'memory_usage_percent'),
                    ('inference_active', 'cpu_usage'),
                    ('alert_count', 'cpu_usage')
                ]
                
                for metric_a_name, metric_b_name in metric_pairs:
                    correlation = self._calculate_metric_correlation(
                        data_a, data_b, metric_a_name, metric_b_name,
                        f"{service_a}.{metric_a_name}", f"{service_b}.{metric_b_name}"
                    )
                    if correlation:
                        correlations[f"{service_a}_{service_b}_{metric_a_name}_{metric_b_name}"] = correlation
                        
        return correlations
        
    def _calculate_system_correlations(self, cutoff_time: datetime) -> Dict[str, CorrelationMetric]:
        """Calculate correlations between services and system-wide metrics."""
        correlations = {}
        
        # Get recent system data
        system_data = [point for point in self.system_metrics_history 
                      if point['timestamp'] > cutoff_time]
                      
        if len(system_data) < self.correlation_thresholds['minimum_samples']:
            return correlations
            
        # Calculate service-to-system correlations
        for service_name, service_history in self.service_metrics_history.items():
            service_data = [point for point in service_history 
                           if point['timestamp'] > cutoff_time]
                           
            if len(service_data) < self.correlation_thresholds['minimum_samples']:
                continue
                
            # Key service-to-system correlation pairs
            correlation_pairs = [
                ('cpu_usage', 'total_cpu_usage'),
                ('memory_usage_mb', 'total_memory_usage_mb'),
                ('inference_active', 'active_inference_services'),
                ('alert_count', 'services_with_alerts')
            ]
            
            for service_metric, system_metric in correlation_pairs:
                correlation = self._calculate_metric_correlation(
                    service_data, system_data, service_metric, system_metric,
                    f"{service_name}.{service_metric}", f"system.{system_metric}"
                )
                if correlation:
                    correlations[f"{service_name}_system_{service_metric}_{system_metric}"] = correlation
                    
        return correlations
        
    def _calculate_resource_correlations(self, cutoff_time: datetime) -> Dict[str, CorrelationMetric]:
        """Calculate correlations between different resource metrics within services."""
        correlations = {}
        
        for service_name, service_history in self.service_metrics_history.items():
            service_data = [point for point in service_history 
                           if point['timestamp'] > cutoff_time]
                           
            if len(service_data) < self.correlation_thresholds['minimum_samples']:
                continue
                
            # Resource correlation pairs within the same service
            resource_pairs = [
                ('cpu_usage', 'memory_usage_percent'),
                ('cpu_usage', 'network_tx_mb'),
                ('memory_usage_percent', 'inference_active'),
                ('network_rx_mb', 'network_tx_mb'),
                ('uptime_hours', 'restart_count'),
                ('alert_count', 'cpu_usage')
            ]
            
            for metric_a, metric_b in resource_pairs:
                correlation = self._calculate_single_service_correlation(
                    service_data, metric_a, metric_b, 
                    f"{service_name}.{metric_a}", f"{service_name}.{metric_b}"
                )
                if correlation:
                    correlations[f"{service_name}_internal_{metric_a}_{metric_b}"] = correlation
                    
        return correlations
        
    def _calculate_metric_correlation(
        self, data_a: List[Dict], data_b: List[Dict], 
        metric_a: str, metric_b: str, 
        label_a: str, label_b: str
    ) -> Optional[CorrelationMetric]:
        """Calculate correlation between metrics from two different data series."""
        try:
            # Align data by timestamp (use closest matches)
            aligned_pairs = []
            for point_a in data_a:
                closest_point_b = min(
                    data_b, 
                    key=lambda p: abs((p['timestamp'] - point_a['timestamp']).total_seconds()),
                    default=None
                )
                if closest_point_b:
                    time_diff = abs((closest_point_b['timestamp'] - point_a['timestamp']).total_seconds())
                    if time_diff <= 60:  # Within 1 minute
                        aligned_pairs.append((
                            self._extract_metric_value(point_a, metric_a),
                            self._extract_metric_value(closest_point_b, metric_b)
                        ))
                        
            if len(aligned_pairs) < self.correlation_thresholds['minimum_samples']:
                return None
                
            values_a = [pair[0] for pair in aligned_pairs if pair[0] is not None and pair[1] is not None]
            values_b = [pair[1] for pair in aligned_pairs if pair[0] is not None and pair[1] is not None]
            
            if len(values_a) < self.correlation_thresholds['minimum_samples']:
                return None
                
            correlation_coeff = np.corrcoef(values_a, values_b)[0, 1]
            
            if np.isnan(correlation_coeff):
                return None
                
            # Calculate confidence based on sample size and correlation strength
            confidence = min(1.0, len(values_a) / 50.0) * min(1.0, abs(correlation_coeff) * 2)
            
            # Determine significance
            significance = Significance.LOW
            if abs(correlation_coeff) > self.correlation_thresholds['strong_correlation']:
                significance = Significance.HIGH
            elif abs(correlation_coeff) > self.correlation_thresholds['moderate_correlation']:
                significance = Significance.MEDIUM
                
            return CorrelationMetric(
                metric_a=label_a,
                metric_b=label_b,
                correlation_strength=float(correlation_coeff),
                confidence=confidence,
                sample_size=len(values_a),
                significance=significance
            )
            
        except Exception as e:
            logger.error(f"Error calculating correlation between {label_a} and {label_b}: {e}")
            return None
            
    def _calculate_single_service_correlation(
        self, service_data: List[Dict], metric_a: str, metric_b: str,
        label_a: str, label_b: str
    ) -> Optional[CorrelationMetric]:
        """Calculate correlation between two metrics within the same service."""
        try:
            values_a = []
            values_b = []
            
            for point in service_data:
                val_a = self._extract_metric_value(point, metric_a)
                val_b = self._extract_metric_value(point, metric_b)
                
                if val_a is not None and val_b is not None:
                    values_a.append(val_a)
                    values_b.append(val_b)
                    
            if len(values_a) < self.correlation_thresholds['minimum_samples']:
                return None
                
            correlation_coeff = np.corrcoef(values_a, values_b)[0, 1]
            
            if np.isnan(correlation_coeff):
                return None
                
            confidence = min(1.0, len(values_a) / 50.0) * min(1.0, abs(correlation_coeff) * 2)
            
            significance = Significance.LOW
            if abs(correlation_coeff) > self.correlation_thresholds['strong_correlation']:
                significance = Significance.HIGH
            elif abs(correlation_coeff) > self.correlation_thresholds['moderate_correlation']:
                significance = Significance.MEDIUM
                
            return CorrelationMetric(
                metric_a=label_a,
                metric_b=label_b,
                correlation_strength=float(correlation_coeff),
                confidence=confidence,
                sample_size=len(values_a),
                significance=significance
            )
            
        except Exception as e:
            logger.error(f"Error calculating single service correlation: {e}")
            return None
            
    def _extract_metric_value(self, data_point: Dict, metric_name: str) -> Optional[float]:
        """Extract numeric value for a metric, handling different data types."""
        value = data_point.get(metric_name)
        
        if value is None:
            return None
        elif isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, bool):
            return 1.0 if value else 0.0
        elif isinstance(value, str):
            # Try to convert string to number
            try:
                return float(value)
            except ValueError:
                # Handle categorical values
                if value == 'healthy':
                    return 1.0
                elif value == 'unhealthy':
                    return 0.0
                elif value == 'starting':
                    return 0.5
                else:
                    return None
        else:
            return None
            
    def _generate_correlation_insights(self) -> List[ResourceCorrelationInsight]:
        """Generate insights from calculated correlations."""
        insights = []
        
        # Analyze strong correlations for insights
        strong_correlations = [
            corr for corr in self.correlation_cache.values()
            if abs(corr.correlation_strength) > self.correlation_thresholds['moderate_correlation']
            and corr.confidence > self.correlation_thresholds['confidence_threshold']
        ]
        
        # Service competition insights
        competition_insights = self._analyze_service_competition(strong_correlations)
        insights.extend(competition_insights)
        
        # Resource utilization insights
        utilization_insights = self._analyze_resource_utilization(strong_correlations)
        insights.extend(utilization_insights)
        
        # Performance correlation insights
        performance_insights = self._analyze_performance_correlations(strong_correlations)
        insights.extend(performance_insights)
        
        return insights
        
    def _analyze_service_competition(
        self, correlations: List[CorrelationMetric]
    ) -> List[ResourceCorrelationInsight]:
        """Analyze correlations that indicate service competition."""
        insights = []
        
        # Look for negative correlations between services (competition)
        competition_correlations = [
            corr for corr in correlations
            if corr.correlation_strength < -0.4 and
            'cpu_usage' in corr.metric_a and 'cpu_usage' in corr.metric_b
        ]
        
        for corr in competition_correlations:
            # Extract service names
            service_a = corr.metric_a.split('.')[0]
            service_b = corr.metric_b.split('.')[0]
            
            # Determine service types for context
            type_a = self.ai_service_configs.get(service_a, {}).get('type', 'unknown')
            type_b = self.ai_service_configs.get(service_b, {}).get('type', 'unknown')
            
            # Generate appropriate recommendations
            recommendations = []
            if 'gpu' in type_a and 'gpu' in type_b:
                recommendations = [
                    "Implement GPU memory management to prevent contention",
                    "Consider model quantization to reduce VRAM requirements",
                    "Implement request queuing to serialize GPU access"
                ]
            elif 'cpu' in type_a and 'cpu' in type_b:
                recommendations = [
                    "Verify CPU core pinning is working correctly",
                    "Consider adjusting CPU resource limits",
                    "Monitor for memory bandwidth competition"
                ]
                
            insights.append(ResourceCorrelationInsight(
                insight_type='service_competition',
                title=f"Resource Competition: {service_a} ↔ {service_b}",
                description=f"Negative correlation ({corr.correlation_strength:.2f}) detected between {service_a} and {service_b} CPU usage, indicating resource competition",
                affected_services=[service_a, service_b],
                confidence=corr.confidence,
                potential_impact="Performance degradation due to resource contention",
                optimization_opportunity="Optimize resource allocation or scheduling",
                recommended_actions=recommendations,
                supporting_data={
                    'correlation_strength': corr.correlation_strength,
                    'sample_size': corr.sample_size,
                    'service_types': [type_a, type_b]
                }
            ))
            
        return insights
        
    def _analyze_resource_utilization(
        self, correlations: List[CorrelationMetric]
    ) -> List[ResourceCorrelationInsight]:
        """Analyze resource utilization correlation patterns."""
        insights = []
        
        # Look for strong positive correlations between CPU and memory
        cpu_memory_correlations = [
            corr for corr in correlations
            if corr.correlation_strength > 0.6 and
            'cpu_usage' in corr.metric_a and 'memory_usage' in corr.metric_b
        ]
        
        for corr in cpu_memory_correlations:
            service_name = corr.metric_a.split('.')[0]
            service_config = self.ai_service_configs.get(service_name, {})
            
            recommendations = [
                "Strong CPU-memory correlation indicates healthy resource scaling",
                "Monitor for memory leaks if correlation weakens over time",
                "Consider memory-optimized model configurations"
            ]
            
            if service_config.get('type') == 'gpu_inference':
                recommendations.append("GPU inference showing consistent CPU-memory scaling")
            elif service_config.get('type') == 'cpu_inference':
                recommendations.append("CPU inference showing expected resource utilization pattern")
                
            insights.append(ResourceCorrelationInsight(
                insight_type='resource_utilization',
                title=f"Healthy Resource Scaling: {service_name}",
                description=f"Strong positive correlation ({corr.correlation_strength:.2f}) between CPU and memory usage indicates healthy resource scaling",
                affected_services=[service_name],
                confidence=corr.confidence,
                potential_impact="Predictable resource utilization pattern",
                optimization_opportunity="Resource allocation is well-balanced",
                recommended_actions=recommendations,
                supporting_data={
                    'correlation_strength': corr.correlation_strength,
                    'service_type': service_config.get('type', 'unknown')
                }
            ))
            
        return insights
        
    def _analyze_performance_correlations(
        self, correlations: List[CorrelationMetric]
    ) -> List[ResourceCorrelationInsight]:
        """Analyze performance-related correlations."""
        insights = []
        
        # Look for correlations with inference activity
        inference_correlations = [
            corr for corr in correlations
            if 'inference_active' in corr.metric_a or 'inference_active' in corr.metric_b
        ]
        
        for corr in inference_correlations:
            if 'inference_active' in corr.metric_a:
                service_name = corr.metric_a.split('.')[0]
                performance_metric = corr.metric_b
            else:
                service_name = corr.metric_b.split('.')[0]
                performance_metric = corr.metric_a
                
            service_config = self.ai_service_configs.get(service_name, {})
            
            if corr.correlation_strength > 0.5:
                insights.append(ResourceCorrelationInsight(
                    insight_type='performance_correlation',
                    title=f"Inference Performance Correlation: {service_name}",
                    description=f"Strong correlation ({corr.correlation_strength:.2f}) between inference activity and {performance_metric}",
                    affected_services=[service_name],
                    confidence=corr.confidence,
                    potential_impact="Predictable performance characteristics during inference",
                    optimization_opportunity="Performance metrics are well-correlated with workload",
                    recommended_actions=[
                        "Performance scaling is working as expected",
                        "Monitor for changes in correlation strength",
                        "Use correlation for performance prediction"
                    ],
                    supporting_data={
                        'correlation_strength': corr.correlation_strength,
                        'performance_metric': performance_metric,
                        'service_type': service_config.get('type', 'unknown')
                    }
                ))
                
        return insights
        
    def _detect_service_interactions(
        self, service_states: Dict[str, AIServiceState]
    ) -> List[ResourceCorrelationInsight]:
        """Detect service interaction patterns."""
        insights = []
        
        # Detect load balancing patterns across CPU services
        cpu_services = ['llama-cpu-0', 'llama-cpu-1', 'llama-cpu-2']
        cpu_states = {name: state for name, state in service_states.items() if name in cpu_services}
        
        if len(cpu_states) >= 2:
            load_balancing_insight = self._analyze_load_balancing_pattern(cpu_states)
            if load_balancing_insight:
                insights.append(load_balancing_insight)
                
        # Detect GPU service interactions
        gpu_services = ['llama-gpu', 'vllm-gpu']
        gpu_states = {name: state for name, state in service_states.items() if name in gpu_services}
        
        if len(gpu_states) >= 2:
            gpu_interaction_insight = self._analyze_gpu_service_interactions(gpu_states)
            if gpu_interaction_insight:
                insights.append(gpu_interaction_insight)
                
        return insights
        
    def _analyze_load_balancing_pattern(
        self, cpu_states: Dict[str, AIServiceState]
    ) -> Optional[ResourceCorrelationInsight]:
        """Analyze load balancing patterns across CPU services."""
        cpu_usages = []
        active_services = []
        
        for service_name, state in cpu_states.items():
            if state.container_metrics and state.container_metrics.status == 'running':
                cpu_usages.append(state.container_metrics.cpu_usage_percent)
                active_services.append(service_name)
                
        if len(cpu_usages) < 2:
            return None
            
        # Calculate load distribution metrics
        avg_usage = mean(cpu_usages)
        usage_stdev = stdev(cpu_usages) if len(cpu_usages) > 1 else 0
        
        # Coefficient of variation (lower is better for load balancing)
        cv = usage_stdev / avg_usage if avg_usage > 0 else 0
        
        # Determine load balancing quality
        if cv < 0.2:  # Very even distribution
            balance_quality = "excellent"
            confidence = 0.9
        elif cv < 0.4:  # Reasonably even
            balance_quality = "good"
            confidence = 0.7
        elif cv < 0.6:  # Moderate imbalance
            balance_quality = "moderate"
            confidence = 0.6
        else:  # Poor distribution
            balance_quality = "poor"
            confidence = 0.8
            
        recommendations = []
        if balance_quality in ['poor', 'moderate']:
            recommendations = [
                "Investigate load balancer configuration",
                "Check for service health or performance issues",
                "Consider request routing optimization"
            ]
        else:
            recommendations = [
                "Load balancing is working effectively",
                "Monitor for changes in distribution patterns",
                "Maintain current load balancing configuration"
            ]
            
        return ResourceCorrelationInsight(
            insight_type='load_balancing',
            title=f"CPU Service Load Balancing: {balance_quality.title()}",
            description=f"Load distribution across {len(active_services)} CPU services shows {balance_quality} balancing (CV: {cv:.2f})",
            affected_services=active_services,
            confidence=confidence,
            potential_impact=f"Load balancing quality affects overall CPU inference performance",
            optimization_opportunity="Optimize load distribution" if balance_quality != "excellent" else None,
            recommended_actions=recommendations,
            supporting_data={
                'cpu_usages': cpu_usages,
                'coefficient_of_variation': cv,
                'average_usage': avg_usage,
                'balance_quality': balance_quality
            }
        )
        
    def _analyze_gpu_service_interactions(
        self, gpu_states: Dict[str, AIServiceState]
    ) -> Optional[ResourceCorrelationInsight]:
        """Analyze interactions between GPU services."""
        running_gpu_services = []
        total_gpu_activity = 0
        
        for service_name, state in gpu_states.items():
            if (state.container_metrics and 
                state.container_metrics.status == 'running' and 
                state.inference_active):
                running_gpu_services.append(service_name)
                total_gpu_activity += 1
                
        if len(running_gpu_services) < 2:
            return None
            
        # Multiple GPU services are active - potential contention
        confidence = 0.8
        potential_impact = "GPU memory and compute resource contention"
        
        recommendations = [
            "Monitor GPU memory usage to prevent OOM errors",
            "Consider request queuing to manage GPU access",
            "Implement dynamic model swapping if memory constrained",
            "Monitor CUDA context switching overhead"
        ]
        
        return ResourceCorrelationInsight(
            insight_type='gpu_contention',
            title="GPU Service Contention Detected",
            description=f"Multiple GPU services ({', '.join(running_gpu_services)}) are simultaneously active",
            affected_services=running_gpu_services,
            confidence=confidence,
            potential_impact=potential_impact,
            optimization_opportunity="Implement GPU resource management strategy",
            recommended_actions=recommendations,
            supporting_data={
                'concurrent_gpu_services': len(running_gpu_services),
                'service_names': running_gpu_services
            }
        )
        
    def _analyze_resource_contention(
        self, service_states: Dict[str, AIServiceState]
    ) -> List[ResourceCorrelationInsight]:
        """Analyze resource contention patterns across services."""
        insights = []
        
        # Check for high overall resource usage
        total_cpu_usage = 0
        total_memory_usage_mb = 0
        high_usage_services = []
        
        for service_name, state in service_states.items():
            if state.container_metrics:
                cpu_usage = state.container_metrics.cpu_usage_percent
                memory_usage = state.container_metrics.memory_usage_mb
                
                total_cpu_usage += cpu_usage
                total_memory_usage_mb += memory_usage
                
                # Check for individual high usage services
                if cpu_usage > 80 or memory_usage > 25000:  # > 25GB
                    high_usage_services.append({
                        'service': service_name,
                        'cpu': cpu_usage,
                        'memory_gb': memory_usage / 1024
                    })
                    
        # Generate contention insights
        if total_cpu_usage > 400:  # More than 400% total CPU usage
            insights.append(ResourceCorrelationInsight(
                insight_type='system_contention',
                title="High System CPU Contention",
                description=f"Total CPU usage across all services: {total_cpu_usage:.1f}%",
                affected_services=list(service_states.keys()),
                confidence=0.9,
                potential_impact="System-wide performance degradation possible",
                optimization_opportunity="Distribute workload or scale resources",
                recommended_actions=[
                    "Monitor system thermal state",
                    "Consider request rate limiting",
                    "Evaluate service priority and scheduling",
                    "Check for runaway processes"
                ],
                supporting_data={'total_cpu_usage': total_cpu_usage}
            ))
            
        if total_memory_usage_mb > 100000:  # More than 100GB total memory usage
            insights.append(ResourceCorrelationInsight(
                insight_type='memory_contention',
                title="High System Memory Utilization",
                description=f"Total memory usage across all services: {total_memory_usage_mb/1024:.1f}GB",
                affected_services=list(service_states.keys()),
                confidence=0.9,
                potential_impact="Potential memory pressure and swap usage",
                optimization_opportunity="Optimize memory allocation or consider model quantization",
                recommended_actions=[
                    "Monitor system swap usage",
                    "Consider model quantization for high-memory services",
                    "Implement memory limits and cleanup",
                    "Evaluate container memory constraints"
                ],
                supporting_data={'total_memory_usage_gb': total_memory_usage_mb/1024}
            ))
            
        # Individual service high usage insights
        if high_usage_services:
            insights.append(ResourceCorrelationInsight(
                insight_type='individual_high_usage',
                title="High Resource Usage Services Detected",
                description=f"{len(high_usage_services)} services showing high resource usage",
                affected_services=[service['service'] for service in high_usage_services],
                confidence=0.8,
                potential_impact="Individual services may be under stress or processing heavy workloads",
                optimization_opportunity="Optimize high-usage services or redistribute load",
                recommended_actions=[
                    "Investigate workload patterns for high-usage services",
                    "Consider load balancing or request distribution",
                    "Monitor for memory leaks or inefficient processing",
                    "Evaluate model complexity vs. performance requirements"
                ],
                supporting_data={'high_usage_services': high_usage_services}
            ))
            
        return insights
        
    def _generate_predictive_insights(
        self, service_states: Dict[str, AIServiceState]
    ) -> List[ResourceCorrelationInsight]:
        """Generate predictive insights based on correlation patterns."""
        insights = []
        
        # Train or update performance models if sufficient data
        if len(self.system_metrics_history) >= self.model_training_threshold:
            self._update_performance_models()
            
        # Generate predictions based on current state
        for service_name, state in service_states.items():
            if not state.container_metrics:
                continue
                
            predictions = self._generate_service_predictions(service_name, state)
            if predictions:
                insights.extend(predictions)
                
        return insights
        
    def _update_performance_models(self):
        """Update performance prediction models based on historical data."""
        # This is a simplified implementation - in practice, you might use
        # more sophisticated ML models like scikit-learn
        
        try:
            for service_name in self.service_metrics_history:
                service_data = list(self.service_metrics_history[service_name])
                
                if len(service_data) < self.model_training_threshold:
                    continue
                    
                # Create simple linear model for CPU usage prediction
                model = self._train_simple_linear_model(
                    service_data, target_metric='cpu_usage'
                )
                
                if model:
                    self.performance_models[f"{service_name}_cpu_prediction"] = model
                    
        except Exception as e:
            logger.error(f"Error updating performance models: {e}")
            
    def _train_simple_linear_model(
        self, data: List[Dict], target_metric: str
    ) -> Optional[PerformanceCorrelationModel]:
        """Train a simple linear correlation model."""
        try:
            # Extract features and target
            features = ['memory_usage_percent', 'inference_active', 'alert_count', 'uptime_hours']
            
            X = []
            y = []
            
            for point in data:
                target_value = self._extract_metric_value(point, target_metric)
                if target_value is None:
                    continue
                    
                feature_values = []
                valid_features = []
                
                for feature in features:
                    feature_value = self._extract_metric_value(point, feature)
                    if feature_value is not None:
                        feature_values.append(feature_value)
                        valid_features.append(feature)
                        
                if len(feature_values) == len(features):  # All features present
                    X.append(feature_values)
                    y.append(target_value)
                    
            if len(X) < 10:  # Not enough data
                return None
                
            # Calculate simple correlation coefficients
            correlation_coefficients = {}
            for i, feature in enumerate(features):
                feature_values = [x[i] for x in X]
                try:
                    corr = np.corrcoef(feature_values, y)[0, 1]
                    if not np.isnan(corr):
                        correlation_coefficients[feature] = float(corr)
                except:
                    correlation_coefficients[feature] = 0.0
                    
            # Simple accuracy estimate (R-squared approximation)
            accuracy = max(abs(coeff) for coeff in correlation_coefficients.values())
            
            return PerformanceCorrelationModel(
                model_type='simple_linear',
                target_metric=target_metric,
                input_features=features,
                correlation_coefficients=correlation_coefficients,
                accuracy=accuracy,
                last_trained=datetime.now(),
                prediction_horizon=timedelta(minutes=30)
            )
            
        except Exception as e:
            logger.error(f"Error training simple linear model: {e}")
            return None
            
    def _generate_service_predictions(
        self, service_name: str, state: AIServiceState
    ) -> List[ResourceCorrelationInsight]:
        """Generate predictions for a specific service."""
        insights = []
        
        model_key = f"{service_name}_cpu_prediction"
        model = self.performance_models.get(model_key)
        
        if not model or not state.container_metrics:
            return insights
            
        # Generate prediction based on current state
        current_features = {
            'memory_usage_percent': (state.container_metrics.memory_usage_mb / 
                                   state.container_metrics.memory_limit_mb * 100 
                                   if state.container_metrics.memory_limit_mb > 0 else 0),
            'inference_active': 1.0 if state.inference_active else 0.0,
            'alert_count': len(state.resource_alerts),
            'uptime_hours': state.container_metrics.uptime_seconds / 3600
        }
        
        # Simple prediction using correlation coefficients
        predicted_cpu = sum(
            current_features.get(feature, 0) * coeff
            for feature, coeff in model.correlation_coefficients.items()
        )
        
        current_cpu = state.container_metrics.cpu_usage_percent
        cpu_change_predicted = predicted_cpu - current_cpu
        
        # Generate insight if significant change is predicted
        if abs(cpu_change_predicted) > 20:  # > 20% CPU change predicted
            trend = "increase" if cpu_change_predicted > 0 else "decrease"
            
            recommendations = []
            if cpu_change_predicted > 0:
                recommendations = [
                    "Prepare for increased CPU load",
                    "Monitor thermal conditions",
                    "Consider preemptive load balancing"
                ]
            else:
                recommendations = [
                    "CPU usage may decrease",
                    "Opportunity for additional workload",
                    "Monitor for idle resource optimization"
                ]
                
            insights.append(ResourceCorrelationInsight(
                insight_type='performance_prediction',
                title=f"CPU Usage Prediction: {service_name}",
                description=f"Model predicts {abs(cpu_change_predicted):.1f}% CPU {trend} in next {model.prediction_horizon}",
                affected_services=[service_name],
                confidence=model.accuracy * 0.8,  # Reduce confidence for predictions
                potential_impact=f"CPU usage {trend} may affect service performance",
                optimization_opportunity=f"Proactive resource management based on predicted {trend}",
                recommended_actions=recommendations,
                supporting_data={
                    'predicted_cpu': predicted_cpu,
                    'current_cpu': current_cpu,
                    'predicted_change': cpu_change_predicted,
                    'model_accuracy': model.accuracy,
                    'prediction_horizon': str(model.prediction_horizon)
                }
            ))
            
        return insights
        
    def get_correlation_summary(self) -> Dict[str, Any]:
        """Get summary of current correlations and insights."""
        return {
            'total_correlations': len(self.correlation_cache),
            'strong_correlations': len([
                c for c in self.correlation_cache.values()
                if abs(c.correlation_strength) > self.correlation_thresholds['strong_correlation']
            ]),
            'services_monitored': len(self.service_metrics_history),
            'historical_data_points': len(self.system_metrics_history),
            'performance_models': len(self.performance_models),
            'last_analysis': self.last_correlation_update.isoformat(),
            'top_correlations': [
                {
                    'metrics': f"{c.metric_a} ↔ {c.metric_b}",
                    'strength': c.correlation_strength,
                    'confidence': c.confidence
                }
                for c in sorted(
                    self.correlation_cache.values(),
                    key=lambda x: abs(x.correlation_strength),
                    reverse=True
                )[:5]
            ]
        }
        
    def cleanup(self):
        """Cleanup resources."""
        try:
            self.thread_pool.shutdown(wait=True)
        except Exception:
            pass