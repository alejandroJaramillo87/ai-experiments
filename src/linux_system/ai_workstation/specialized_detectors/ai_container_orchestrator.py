"""
AI Container Orchestrator Detector - Docker Service Intelligence
==============================================================

Specialized detector for monitoring Docker container orchestration with focus on
AI inference services. Provides deep intelligence about container lifecycle,
resource allocation, and service health across the AI workstation infrastructure.

Key Capabilities:
- Docker service lifecycle monitoring (llama-cpu, llama-gpu, vLLM services)
- Container health correlation with resource usage patterns
- Model loading/unloading detection across GPU and CPU containers
- Inter-service dependency analysis and communication monitoring
- Resource contention detection and optimization recommendations
- Container restart event analysis with root cause identification
"""

import asyncio
import json
import logging
import docker
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path

from ..base_collector import BaseCollector
from ...temporal.types import SystemChange, ChangeType

logger = logging.getLogger(__name__)


@dataclass
class ContainerMetrics:
    """Detailed container performance metrics."""
    container_id: str
    name: str
    service_name: Optional[str]
    cpu_percent: float
    memory_usage_mb: float
    memory_limit_mb: float
    memory_percent: float
    network_rx_mb: float
    network_tx_mb: float
    block_read_mb: float
    block_write_mb: float
    pids: int
    status: str
    health_status: Optional[str]
    restart_count: int
    uptime_seconds: int
    ports: Dict[str, Any]
    labels: Dict[str, str]
    environment: Dict[str, str]
    timestamp: datetime


@dataclass
class ServiceDependency:
    """Container service dependency relationship."""
    service: str
    depends_on: str
    dependency_type: str  # 'network', 'volume', 'startup_order'
    health_required: bool
    strength: float  # 0.0-1.0, dependency strength


@dataclass
class ModelLoadEvent:
    """AI model loading/unloading event detection."""
    container_name: str
    service_type: str  # 'llama-cpu', 'llama-gpu', 'vllm'
    event_type: str  # 'model_load', 'model_unload', 'model_switch'
    model_name: Optional[str]
    model_size_gb: Optional[float]
    memory_delta_mb: float
    gpu_memory_delta_mb: float
    load_time_seconds: Optional[float]
    timestamp: datetime
    performance_impact: Dict[str, Any]


@dataclass
class ContainerResourcePattern:
    """Resource usage pattern analysis."""
    container_name: str
    pattern_type: str  # 'startup_spike', 'inference_load', 'memory_leak', 'idle_baseline'
    cpu_pattern: List[float]
    memory_pattern: List[float]
    duration_seconds: int
    frequency: str  # 'one_time', 'periodic', 'continuous'
    significance: float
    correlation_factors: List[str]


class AIContainerOrchestratorDetector:
    """
    Specialized detector for AI container orchestration intelligence.
    
    Monitors Docker-based AI inference services with deep understanding of
    containerized AI workload patterns, resource allocation strategies,
    and inter-service dependencies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize AI container orchestrator detector."""
        self.config = config or {}
        self.docker_client = None
        self.previous_metrics: Dict[str, ContainerMetrics] = {}
        self.service_dependencies: Dict[str, List[ServiceDependency]] = {}
        self.model_load_history: List[ModelLoadEvent] = []
        self.resource_patterns: Dict[str, List[ContainerResourcePattern]] = defaultdict(list)
        
        # AI service configuration mapping
        self.ai_services = {
            'llama-cpu-1': {'type': 'llama-cpu', 'cores': [0, 1, 2, 3, 4, 5, 6, 7], 'gpu': False},
            'llama-cpu-2': {'type': 'llama-cpu', 'cores': [8, 9, 10, 11, 12, 13, 14, 15], 'gpu': False},
            'llama-cpu-3': {'type': 'llama-cpu', 'cores': [16, 17, 18, 19, 20, 21, 22, 23], 'gpu': False},
            'llama-gpu': {'type': 'llama-gpu', 'cores': [], 'gpu': True},
            'vllm': {'type': 'vllm', 'cores': [], 'gpu': True}
        }
        
        # Performance thresholds for AI workloads
        self.thresholds = {
            'cpu_high': 80.0,
            'memory_high': 85.0,
            'memory_critical': 95.0,
            'restart_concern': 3,
            'model_load_timeout': 300,  # 5 minutes
            'inference_response_slow': 10.0,  # seconds
            'memory_leak_threshold': 1024  # MB growth over 1 hour
        }
        
        self._initialize_docker_client()
        logger.info("AIContainerOrchestratorDetector initialized")
    
    def _initialize_docker_client(self):
        """Initialize Docker API client with error handling."""
        try:
            self.docker_client = docker.from_env()
            # Test connection
            self.docker_client.ping()
            logger.info("Docker API connection established")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            self.docker_client = None
    
    async def collect_container_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive container metrics and intelligence."""
        if not self.docker_client:
            return {'error': 'Docker client not available'}
        
        try:
            containers = self.docker_client.containers.list(all=True)
            current_metrics = {}
            service_health = {}
            resource_analysis = {}
            
            for container in containers:
                metrics = await self._collect_single_container_metrics(container)
                if metrics:
                    current_metrics[metrics.name] = metrics
                    
                    # Service health assessment
                    health = self._assess_service_health(metrics)
                    service_health[metrics.name] = health
                    
                    # Resource pattern analysis
                    patterns = self._analyze_resource_patterns(metrics)
                    if patterns:
                        resource_analysis[metrics.name] = patterns
            
            # Detect model loading events
            model_events = self._detect_model_events(current_metrics)
            
            # Update service dependencies
            self._update_service_dependencies(current_metrics)
            
            # Analyze inter-service correlations
            correlations = self._analyze_service_correlations(current_metrics)
            
            # Generate optimization recommendations
            optimizations = self._generate_optimization_recommendations(
                current_metrics, service_health, resource_analysis
            )
            
            # Store for temporal analysis
            self.previous_metrics = current_metrics
            
            return {
                'container_metrics': {name: self._metrics_to_dict(metrics) 
                                    for name, metrics in current_metrics.items()},
                'service_health': service_health,
                'resource_patterns': resource_analysis,
                'model_events': [self._model_event_to_dict(event) for event in model_events],
                'service_dependencies': self._dependencies_to_dict(),
                'service_correlations': correlations,
                'optimization_recommendations': optimizations,
                'ai_service_summary': self._generate_ai_service_summary(current_metrics),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error collecting container metrics: {e}")
            return {'error': str(e)}
    
    async def _collect_single_container_metrics(self, container) -> Optional[ContainerMetrics]:
        """Collect detailed metrics for a single container."""
        try:
            # Get container stats
            stats = container.stats(stream=False)
            
            # Calculate CPU percentage
            cpu_percent = self._calculate_cpu_percent(stats)
            
            # Calculate memory metrics
            memory_stats = stats['memory_stats']
            memory_usage = memory_stats.get('usage', 0)
            memory_limit = memory_stats.get('limit', 0)
            memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0
            
            # Calculate network metrics
            network_stats = stats.get('networks', {})
            total_rx = sum(net.get('rx_bytes', 0) for net in network_stats.values())
            total_tx = sum(net.get('tx_bytes', 0) for net in network_stats.values())
            
            # Calculate block I/O metrics
            blkio_stats = stats.get('blkio_stats', {})
            block_read = sum(entry.get('value', 0) 
                           for entry in blkio_stats.get('io_service_bytes_recursive', [])
                           if entry.get('op') == 'Read')
            block_write = sum(entry.get('value', 0)
                            for entry in blkio_stats.get('io_service_bytes_recursive', [])
                            if entry.get('op') == 'Write')
            
            # Get container info
            container.reload()
            attrs = container.attrs
            
            # Calculate uptime
            started_at = datetime.fromisoformat(
                attrs['State']['StartedAt'].replace('Z', '+00:00')
            )
            uptime = (datetime.now(started_at.tzinfo) - started_at).total_seconds()
            
            return ContainerMetrics(
                container_id=container.id[:12],
                name=container.name,
                service_name=self._extract_service_name(container.name),
                cpu_percent=cpu_percent,
                memory_usage_mb=memory_usage / 1024 / 1024,
                memory_limit_mb=memory_limit / 1024 / 1024,
                memory_percent=memory_percent,
                network_rx_mb=total_rx / 1024 / 1024,
                network_tx_mb=total_tx / 1024 / 1024,
                block_read_mb=block_read / 1024 / 1024,
                block_write_mb=block_write / 1024 / 1024,
                pids=stats.get('pids_stats', {}).get('current', 0),
                status=container.status,
                health_status=attrs.get('State', {}).get('Health', {}).get('Status'),
                restart_count=attrs['RestartCount'],
                uptime_seconds=int(uptime),
                ports=attrs.get('NetworkSettings', {}).get('Ports', {}),
                labels=attrs.get('Config', {}).get('Labels', {}),
                environment={env.split('=', 1)[0]: env.split('=', 1)[1] if '=' in env else ''
                           for env in attrs.get('Config', {}).get('Env', [])},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics for container {container.name}: {e}")
            return None
    
    def _calculate_cpu_percent(self, stats: Dict[str, Any]) -> float:
        """Calculate CPU percentage from Docker stats."""
        try:
            cpu_stats = stats['cpu_stats']
            precpu_stats = stats['precpu_stats']
            
            cpu_delta = cpu_stats['cpu_usage']['total_usage'] - precpu_stats['cpu_usage']['total_usage']
            system_delta = cpu_stats['system_cpu_usage'] - precpu_stats['system_cpu_usage']
            
            if system_delta > 0 and cpu_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * len(cpu_stats['cpu_usage']['percpu_usage']) * 100
                return round(cpu_percent, 2)
            
            return 0.0
            
        except (KeyError, ZeroDivisionError):
            return 0.0
    
    def _extract_service_name(self, container_name: str) -> Optional[str]:
        """Extract service name from container name."""
        # Handle docker-compose naming convention
        parts = container_name.split('-')
        if len(parts) >= 2:
            return '-'.join(parts[:-1])  # Remove container number suffix
        return container_name
    
    def _assess_service_health(self, metrics: ContainerMetrics) -> Dict[str, Any]:
        """Assess individual service health based on metrics."""
        health_score = 100.0
        issues = []
        recommendations = []
        
        # CPU health
        if metrics.cpu_percent > self.thresholds['cpu_high']:
            health_score -= 20
            issues.append(f"High CPU usage: {metrics.cpu_percent}%")
            recommendations.append("Consider scaling or optimizing workload")
        
        # Memory health
        if metrics.memory_percent > self.thresholds['memory_critical']:
            health_score -= 30
            issues.append(f"Critical memory usage: {metrics.memory_percent}%")
            recommendations.append("Immediate memory optimization required")
        elif metrics.memory_percent > self.thresholds['memory_high']:
            health_score -= 15
            issues.append(f"High memory usage: {metrics.memory_percent}%")
            recommendations.append("Monitor memory usage trends")
        
        # Restart health
        if metrics.restart_count > self.thresholds['restart_concern']:
            health_score -= 25
            issues.append(f"Frequent restarts: {metrics.restart_count}")
            recommendations.append("Investigate restart causes")
        
        # Container status health
        if metrics.status != 'running':
            health_score -= 50
            issues.append(f"Container not running: {metrics.status}")
            recommendations.append("Check container logs and restart if needed")
        
        # Health check status
        if metrics.health_status == 'unhealthy':
            health_score -= 40
            issues.append("Container health check failing")
            recommendations.append("Review health check configuration")
        
        return {
            'health_score': max(0, health_score),
            'status': 'healthy' if health_score > 80 else 'warning' if health_score > 50 else 'critical',
            'issues': issues,
            'recommendations': recommendations,
            'service_type': self.ai_services.get(metrics.service_name, {}).get('type', 'unknown')
        }
    
    def _analyze_resource_patterns(self, metrics: ContainerMetrics) -> List[Dict[str, Any]]:
        """Analyze resource usage patterns for predictive intelligence."""
        patterns = []
        
        # Check for memory growth pattern (potential leak)
        if metrics.name in self.previous_metrics:
            prev = self.previous_metrics[metrics.name]
            time_delta = (metrics.timestamp - prev.timestamp).total_seconds() / 3600  # hours
            
            if time_delta > 0:
                memory_growth_rate = (metrics.memory_usage_mb - prev.memory_usage_mb) / time_delta
                
                if memory_growth_rate > self.thresholds['memory_leak_threshold']:
                    patterns.append({
                        'type': 'memory_leak_suspected',
                        'growth_rate_mb_per_hour': round(memory_growth_rate, 2),
                        'severity': 'high' if memory_growth_rate > 2048 else 'medium',
                        'recommendation': 'Monitor for memory leaks and consider restart'
                    })
        
        # AI service specific patterns
        service_type = self.ai_services.get(metrics.service_name, {}).get('type')
        if service_type in ['llama-cpu', 'llama-gpu', 'vllm']:
            # Inference load pattern detection
            if metrics.cpu_percent > 50 or metrics.memory_percent > 60:
                patterns.append({
                    'type': 'active_inference_load',
                    'cpu_utilization': metrics.cpu_percent,
                    'memory_utilization': metrics.memory_percent,
                    'service_type': service_type,
                    'recommendation': 'High utilization indicates active AI inference workload'
                })
        
        return patterns
    
    def _detect_model_events(self, current_metrics: Dict[str, ContainerMetrics]) -> List[ModelLoadEvent]:
        """Detect AI model loading/unloading events."""
        events = []
        
        for name, metrics in current_metrics.items():
            if name in self.previous_metrics and name in self.ai_services:
                prev = self.previous_metrics[name]
                service_config = self.ai_services[name]
                
                # Significant memory increase suggests model loading
                memory_delta = metrics.memory_usage_mb - prev.memory_usage_mb
                
                if abs(memory_delta) > 1024:  # 1GB threshold
                    event_type = 'model_load' if memory_delta > 0 else 'model_unload'
                    
                    events.append(ModelLoadEvent(
                        container_name=name,
                        service_type=service_config['type'],
                        event_type=event_type,
                        model_name=self._infer_model_name(metrics),
                        model_size_gb=abs(memory_delta) / 1024,
                        memory_delta_mb=memory_delta,
                        gpu_memory_delta_mb=0,  # TODO: Add GPU memory tracking
                        load_time_seconds=None,  # TODO: Track timing
                        timestamp=metrics.timestamp,
                        performance_impact={
                            'cpu_impact': metrics.cpu_percent - prev.cpu_percent,
                            'memory_percent_change': metrics.memory_percent - prev.memory_percent
                        }
                    ))
        
        # Store events for temporal analysis
        self.model_load_history.extend(events)
        
        return events
    
    def _infer_model_name(self, metrics: ContainerMetrics) -> Optional[str]:
        """Infer model name from container environment or logs."""
        # Check environment variables for model hints
        env_vars = metrics.environment
        
        model_hints = [
            env_vars.get('MODEL_NAME'),
            env_vars.get('MODEL_PATH'),
            env_vars.get('HUGGINGFACE_MODEL'),
            env_vars.get('MODEL_ID')
        ]
        
        for hint in model_hints:
            if hint:
                return hint.split('/')[-1] if '/' in hint else hint
        
        return None
    
    def _update_service_dependencies(self, current_metrics: Dict[str, ContainerMetrics]):
        """Update service dependency mapping based on current state."""
        # Analyze which services are running together
        running_services = [name for name, metrics in current_metrics.items() 
                          if metrics.status == 'running']
        
        # GPU services dependency analysis
        gpu_services = [name for name in running_services 
                       if self.ai_services.get(name, {}).get('gpu', False)]
        
        for gpu_service in gpu_services:
            if gpu_service not in self.service_dependencies:
                self.service_dependencies[gpu_service] = []
            
            # GPU services depend on GPU being available
            self.service_dependencies[gpu_service].append(
                ServiceDependency(
                    service=gpu_service,
                    depends_on='nvidia_runtime',
                    dependency_type='hardware',
                    health_required=True,
                    strength=1.0
                )
            )
    
    def _analyze_service_correlations(self, current_metrics: Dict[str, ContainerMetrics]) -> Dict[str, Any]:
        """Analyze correlations between different AI services."""
        correlations = {}
        
        # CPU vs GPU service load correlation
        cpu_services = [name for name in current_metrics.keys()
                       if self.ai_services.get(name, {}).get('type') == 'llama-cpu']
        gpu_services = [name for name in current_metrics.keys()
                       if self.ai_services.get(name, {}).get('gpu', False)]
        
        if cpu_services and gpu_services:
            cpu_load = sum(current_metrics[name].cpu_percent for name in cpu_services) / len(cpu_services)
            gpu_load = sum(current_metrics[name].cpu_percent for name in gpu_services) / len(gpu_services)
            
            correlations['cpu_gpu_load_balance'] = {
                'cpu_average_load': round(cpu_load, 2),
                'gpu_average_load': round(gpu_load, 2),
                'load_imbalance': abs(cpu_load - gpu_load),
                'recommendation': 'Consider load balancing' if abs(cpu_load - gpu_load) > 30 else 'Load balanced'
            }
        
        return correlations
    
    def _generate_optimization_recommendations(self, 
                                             current_metrics: Dict[str, ContainerMetrics],
                                             service_health: Dict[str, Any],
                                             resource_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate AI workstation optimization recommendations."""
        recommendations = []
        
        # Resource optimization
        high_memory_services = [name for name, metrics in current_metrics.items()
                              if metrics.memory_percent > 70]
        
        if high_memory_services:
            recommendations.append({
                'type': 'resource_optimization',
                'priority': 'high',
                'title': 'Memory Optimization Required',
                'description': f'Services with high memory usage: {", ".join(high_memory_services)}',
                'actions': [
                    'Consider increasing memory limits',
                    'Optimize model loading strategies',
                    'Implement model caching optimizations'
                ]
            })
        
        # Service health optimization
        unhealthy_services = [name for name, health in service_health.items()
                            if health['status'] in ['warning', 'critical']]
        
        if unhealthy_services:
            recommendations.append({
                'type': 'service_health',
                'priority': 'critical',
                'title': 'Service Health Issues Detected',
                'description': f'Unhealthy services: {", ".join(unhealthy_services)}',
                'actions': [
                    'Review service logs',
                    'Check resource allocation',
                    'Consider service restart if needed'
                ]
            })
        
        # Load balancing recommendations
        cpu_services = [name for name, metrics in current_metrics.items()
                       if self.ai_services.get(name, {}).get('type') == 'llama-cpu']
        
        if len(cpu_services) > 1:
            cpu_loads = [current_metrics[name].cpu_percent for name in cpu_services]
            load_variance = max(cpu_loads) - min(cpu_loads)
            
            if load_variance > 30:
                recommendations.append({
                    'type': 'load_balancing',
                    'priority': 'medium',
                    'title': 'CPU Service Load Imbalance',
                    'description': f'Load variance: {load_variance:.1f}%',
                    'actions': [
                        'Review request routing logic',
                        'Consider dynamic load balancing',
                        'Analyze core pinning effectiveness'
                    ]
                })
        
        return recommendations
    
    def _generate_ai_service_summary(self, current_metrics: Dict[str, ContainerMetrics]) -> Dict[str, Any]:
        """Generate high-level AI service summary."""
        summary = {
            'total_services': len(current_metrics),
            'running_services': len([m for m in current_metrics.values() if m.status == 'running']),
            'cpu_services': 0,
            'gpu_services': 0,
            'total_memory_usage_gb': 0,
            'average_cpu_utilization': 0,
            'service_types': {},
            'health_overview': {'healthy': 0, 'warning': 0, 'critical': 0}
        }
        
        if not current_metrics:
            return summary
        
        total_cpu = 0
        for name, metrics in current_metrics.items():
            service_config = self.ai_services.get(name, {})
            service_type = service_config.get('type', 'unknown')
            
            if service_type == 'llama-cpu':
                summary['cpu_services'] += 1
            elif service_type in ['llama-gpu', 'vllm']:
                summary['gpu_services'] += 1
            
            summary['total_memory_usage_gb'] += metrics.memory_usage_mb / 1024
            total_cpu += metrics.cpu_percent
            
            summary['service_types'][service_type] = summary['service_types'].get(service_type, 0) + 1
        
        summary['average_cpu_utilization'] = total_cpu / len(current_metrics)
        summary['total_memory_usage_gb'] = round(summary['total_memory_usage_gb'], 2)
        summary['average_cpu_utilization'] = round(summary['average_cpu_utilization'], 2)
        
        return summary
    
    def _metrics_to_dict(self, metrics: ContainerMetrics) -> Dict[str, Any]:
        """Convert ContainerMetrics to dictionary."""
        return {
            'container_id': metrics.container_id,
            'name': metrics.name,
            'service_name': metrics.service_name,
            'cpu_percent': metrics.cpu_percent,
            'memory_usage_mb': round(metrics.memory_usage_mb, 2),
            'memory_limit_mb': round(metrics.memory_limit_mb, 2),
            'memory_percent': round(metrics.memory_percent, 2),
            'network_rx_mb': round(metrics.network_rx_mb, 2),
            'network_tx_mb': round(metrics.network_tx_mb, 2),
            'block_read_mb': round(metrics.block_read_mb, 2),
            'block_write_mb': round(metrics.block_write_mb, 2),
            'pids': metrics.pids,
            'status': metrics.status,
            'health_status': metrics.health_status,
            'restart_count': metrics.restart_count,
            'uptime_seconds': metrics.uptime_seconds,
            'ports': metrics.ports,
            'labels': metrics.labels,
            'timestamp': metrics.timestamp.isoformat()
        }
    
    def _model_event_to_dict(self, event: ModelLoadEvent) -> Dict[str, Any]:
        """Convert ModelLoadEvent to dictionary."""
        return {
            'container_name': event.container_name,
            'service_type': event.service_type,
            'event_type': event.event_type,
            'model_name': event.model_name,
            'model_size_gb': event.model_size_gb,
            'memory_delta_mb': round(event.memory_delta_mb, 2),
            'gpu_memory_delta_mb': round(event.gpu_memory_delta_mb, 2),
            'load_time_seconds': event.load_time_seconds,
            'timestamp': event.timestamp.isoformat(),
            'performance_impact': event.performance_impact
        }
    
    def _dependencies_to_dict(self) -> Dict[str, Any]:
        """Convert service dependencies to dictionary."""
        return {
            service: [
                {
                    'depends_on': dep.depends_on,
                    'dependency_type': dep.dependency_type,
                    'health_required': dep.health_required,
                    'strength': dep.strength
                }
                for dep in deps
            ]
            for service, deps in self.service_dependencies.items()
        }
    
    async def detect_changes(self, old_data: Dict[str, Any], new_data: Dict[str, Any]) -> List[SystemChange]:
        """Detect changes in container orchestration state."""
        changes = []
        
        if 'container_metrics' not in old_data or 'container_metrics' not in new_data:
            return changes
        
        old_containers = old_data['container_metrics']
        new_containers = new_data['container_metrics']
        
        # Container lifecycle changes
        old_names = set(old_containers.keys())
        new_names = set(new_containers.keys())
        
        # New containers
        for name in new_names - old_names:
            changes.append(SystemChange(
                category='ai_containers',
                change_type=ChangeType.ADDED,
                entity_id=f'container:{name}',
                old_value=None,
                new_value=new_containers[name],
                significance=0.8,
                metadata={
                    'container_name': name,
                    'service_type': self.ai_services.get(name, {}).get('type', 'unknown'),
                    'change_type': 'container_started'
                },
                timestamp=datetime.now()
            ))
        
        # Removed containers
        for name in old_names - new_names:
            changes.append(SystemChange(
                category='ai_containers',
                change_type=ChangeType.REMOVED,
                entity_id=f'container:{name}',
                old_value=old_containers[name],
                new_value=None,
                significance=0.8,
                metadata={
                    'container_name': name,
                    'service_type': self.ai_services.get(name, {}).get('type', 'unknown'),
                    'change_type': 'container_stopped'
                },
                timestamp=datetime.now()
            ))
        
        # Modified containers (resource changes)
        for name in old_names & new_names:
            old_container = old_containers[name]
            new_container = new_containers[name]
            
            # Significant resource changes
            cpu_delta = abs(new_container['cpu_percent'] - old_container['cpu_percent'])
            memory_delta = abs(new_container['memory_percent'] - old_container['memory_percent'])
            
            if cpu_delta > 20:  # 20% CPU change threshold
                changes.append(SystemChange(
                    category='ai_containers',
                    change_type=ChangeType.MODIFIED,
                    entity_id=f'container_cpu:{name}',
                    old_value=old_container['cpu_percent'],
                    new_value=new_container['cpu_percent'],
                    significance=0.6,
                    metadata={
                        'container_name': name,
                        'service_type': self.ai_services.get(name, {}).get('type', 'unknown'),
                        'change_type': 'cpu_utilization_change',
                        'delta': new_container['cpu_percent'] - old_container['cpu_percent']
                    },
                    timestamp=datetime.now()
                ))
            
            if memory_delta > 15:  # 15% memory change threshold
                changes.append(SystemChange(
                    category='ai_containers',
                    change_type=ChangeType.MODIFIED,
                    entity_id=f'container_memory:{name}',
                    old_value=old_container['memory_percent'],
                    new_value=new_container['memory_percent'],
                    significance=0.7,
                    metadata={
                        'container_name': name,
                        'service_type': self.ai_services.get(name, {}).get('type', 'unknown'),
                        'change_type': 'memory_utilization_change',
                        'delta': new_container['memory_percent'] - old_container['memory_percent']
                    },
                    timestamp=datetime.now()
                ))
            
            # Container restart detection
            if new_container['restart_count'] > old_container['restart_count']:
                changes.append(SystemChange(
                    category='ai_containers',
                    change_type=ChangeType.THRESHOLD_CROSSED,
                    entity_id=f'container_restart:{name}',
                    old_value=old_container['restart_count'],
                    new_value=new_container['restart_count'],
                    significance=0.9,
                    metadata={
                        'container_name': name,
                        'service_type': self.ai_services.get(name, {}).get('type', 'unknown'),
                        'change_type': 'container_restart',
                        'restart_delta': new_container['restart_count'] - old_container['restart_count']
                    },
                    timestamp=datetime.now()
                ))
        
        # Model loading event changes
        if 'model_events' in new_data and new_data['model_events']:
            for event in new_data['model_events']:
                changes.append(SystemChange(
                    category='ai_containers',
                    change_type=ChangeType.ADDED if event['event_type'] == 'model_load' else ChangeType.REMOVED,
                    entity_id=f'model_event:{event["container_name"]}',
                    old_value=None,
                    new_value=event,
                    significance=0.8,
                    metadata={
                        'container_name': event['container_name'],
                        'service_type': event['service_type'],
                        'change_type': 'model_lifecycle_event',
                        'event_type': event['event_type'],
                        'model_name': event.get('model_name')
                    },
                    timestamp=datetime.now()
                ))
        
        return changes