"""
AI Container Orchestrator Detector

Monitors Docker-based AI service orchestration with specialized intelligence
for AI workstation container ecosystems. Provides real-time awareness of
container states, resource utilization, and AI service interactions.

Features:
- Docker API integration for container lifecycle monitoring
- AI service-specific state tracking and health correlation
- Model loading/unloading detection through log analysis
- Resource usage monitoring per container with core pinning awareness
- Service interaction and load balancing detection
"""

import docker
import json
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque

from ...temporal.core.change_detector import SystemChangeDetector, SystemState, SystemChange
from ...temporal.core.types import ChangeType, ComponentType, Significance


logger = logging.getLogger(__name__)


@dataclass
class AIServiceConfig:
    """Configuration for AI service containers."""
    name: str
    port: int
    cores: Optional[str] = None  # CPU core pinning like "0-7"
    memory_limit: Optional[str] = None  # Memory limit like "32GB" 
    gpu_access: bool = False
    expected_model: Optional[str] = None
    service_type: str = "cpu"  # cpu, gpu, vllm, interface


@dataclass 
class ContainerHealthMetrics:
    """Health and performance metrics for AI containers."""
    container_id: str
    name: str
    status: str  # running, exited, restarting, etc.
    health_status: Optional[str]  # healthy, unhealthy, starting
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_limit_mb: float
    network_rx_bytes: int
    network_tx_bytes: int
    restart_count: int
    uptime_seconds: float
    last_updated: datetime


@dataclass
class AIServiceState:
    """State information for AI service containers."""
    service_config: AIServiceConfig
    container_metrics: Optional[ContainerHealthMetrics]
    model_loaded: Optional[str]
    inference_active: bool
    request_count: int
    last_activity: Optional[datetime]
    health_trend: List[str]  # Recent health status history
    resource_alerts: List[str]


class AIContainerOrchestratorDetector(SystemChangeDetector):
    """
    Advanced detector for AI service container orchestration.
    
    Monitors Docker-based AI services with specialized intelligence for:
    - Container lifecycle and health monitoring
    - AI service resource utilization tracking
    - Model loading/unloading detection
    - Service interaction and load balancing analysis
    - Performance correlation with system state
    """
    
    def __init__(self, monitor_interval: float = 10.0):
        super().__init__()
        self.monitor_interval = monitor_interval
        self.docker_client = None
        self._initialize_docker_client()
        
        # AI Service Configuration (based on docker-compose.yaml)
        self.ai_services = {
            'llama-cpu-0': AIServiceConfig(
                name='llama-cpu-0', port=8001, cores='0-7', 
                memory_limit='32GB', service_type='cpu'
            ),
            'llama-cpu-1': AIServiceConfig(
                name='llama-cpu-1', port=8002, cores='8-15',
                memory_limit='32GB', service_type='cpu'
            ),
            'llama-cpu-2': AIServiceConfig(
                name='llama-cpu-2', port=8003, cores='16-23',
                memory_limit='32GB', service_type='cpu'
            ),
            'llama-gpu': AIServiceConfig(
                name='llama-gpu', port=8004, gpu_access=True,
                service_type='gpu', expected_model='Qwen3-Coder-30B'
            ),
            'vllm-gpu': AIServiceConfig(
                name='vllm-gpu', port=8005, gpu_access=True,
                service_type='vllm'
            ),
            'open-webui': AIServiceConfig(
                name='open-webui', port=3000, service_type='interface'
            )
        }
        
        # State tracking
        self.previous_states: Dict[str, AIServiceState] = {}
        self.service_states: Dict[str, AIServiceState] = {}
        self.container_logs_cache: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        
        # Performance monitoring
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.thread_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="AIContainer")
        
        # Change detection thresholds
        self.cpu_threshold = 20.0  # CPU usage change threshold
        self.memory_threshold = 1024  # Memory change threshold in MB
        self.health_change_significance = {
            'healthy_to_unhealthy': Significance.CRITICAL,
            'unhealthy_to_healthy': Significance.HIGH,
            'starting_to_healthy': Significance.MEDIUM,
            'container_restart': Significance.HIGH,
            'container_stopped': Significance.CRITICAL,
            'container_started': Significance.MEDIUM
        }
        
    def _initialize_docker_client(self):
        """Initialize Docker client with error handling."""
        try:
            self.docker_client = docker.from_env()
            # Test connection
            self.docker_client.ping()
            logger.info("Docker client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            self.docker_client = None
            
    def start_monitoring(self):
        """Start continuous container monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="AIContainerMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("AI container monitoring started")
        
    def stop_monitoring(self):
        """Stop continuous container monitoring."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        self.thread_pool.shutdown(wait=True)
        logger.info("AI container monitoring stopped")
        
    def _monitoring_loop(self):
        """Continuous monitoring loop for container state changes."""
        while self.monitoring_active:
            try:
                self._update_all_container_states()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in container monitoring loop: {e}")
                time.sleep(5.0)
                
    def _update_all_container_states(self):
        """Update state information for all AI service containers."""
        if not self.docker_client:
            return
            
        current_states = {}
        
        # Collect container information in parallel
        futures = []
        for service_name, config in self.ai_services.items():
            future = self.thread_pool.submit(
                self._collect_container_state, service_name, config
            )
            futures.append((service_name, future))
            
        # Gather results
        for service_name, future in futures:
            try:
                service_state = future.result(timeout=5.0)
                if service_state:
                    current_states[service_name] = service_state
            except Exception as e:
                logger.error(f"Error collecting state for {service_name}: {e}")
                
        # Update states atomically
        self.previous_states = self.service_states.copy()
        self.service_states = current_states
        
    def _collect_container_state(
        self, service_name: str, config: AIServiceConfig
    ) -> Optional[AIServiceState]:
        """Collect comprehensive state for a single AI service container."""
        try:
            # Find container by name
            containers = self.docker_client.containers.list(
                all=True, filters={'name': service_name}
            )
            
            if not containers:
                logger.debug(f"Container {service_name} not found")
                return AIServiceState(
                    service_config=config,
                    container_metrics=None,
                    model_loaded=None,
                    inference_active=False,
                    request_count=0,
                    last_activity=None,
                    health_trend=[],
                    resource_alerts=[]
                )
                
            container = containers[0]
            
            # Collect container metrics
            metrics = self._collect_container_metrics(container)
            
            # Analyze container logs for AI-specific events
            model_info = self._analyze_container_logs(container, service_name)
            
            # Build service state
            previous_state = self.service_states.get(service_name)
            health_trend = []
            if previous_state:
                health_trend = previous_state.health_trend.copy()
            
            if metrics and metrics.health_status:
                health_trend.append(metrics.health_status)
                # Keep only recent health status
                health_trend = health_trend[-10:]
                
            # Generate resource alerts
            resource_alerts = self._generate_resource_alerts(metrics, config)
            
            return AIServiceState(
                service_config=config,
                container_metrics=metrics,
                model_loaded=model_info.get('model_loaded'),
                inference_active=model_info.get('inference_active', False),
                request_count=model_info.get('request_count', 0),
                last_activity=model_info.get('last_activity'),
                health_trend=health_trend,
                resource_alerts=resource_alerts
            )
            
        except Exception as e:
            logger.error(f"Error collecting state for container {service_name}: {e}")
            return None
            
    def _collect_container_metrics(
        self, container: docker.models.containers.Container
    ) -> Optional[ContainerHealthMetrics]:
        """Collect detailed metrics from a container."""
        try:
            # Get container stats
            stats = container.stats(stream=False)
            
            # Calculate CPU usage
            cpu_usage = self._calculate_cpu_usage(stats)
            
            # Calculate memory usage
            memory_usage = stats['memory_stats'].get('usage', 0)
            memory_limit = stats['memory_stats'].get('limit', 0)
            memory_usage_mb = memory_usage / (1024 * 1024)
            memory_limit_mb = memory_limit / (1024 * 1024)
            
            # Network statistics
            networks = stats.get('networks', {})
            total_rx = sum(net.get('rx_bytes', 0) for net in networks.values())
            total_tx = sum(net.get('tx_bytes', 0) for net in networks.values())
            
            # Container info
            container.reload()
            health_status = None
            if hasattr(container.attrs['State'], 'Health'):
                health_status = container.attrs['State']['Health']['Status']
                
            # Calculate uptime
            started_at = container.attrs['State']['StartedAt']
            start_time = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
            uptime = (datetime.now(start_time.tzinfo) - start_time).total_seconds()
            
            return ContainerHealthMetrics(
                container_id=container.id,
                name=container.name,
                status=container.status,
                health_status=health_status,
                cpu_usage_percent=cpu_usage,
                memory_usage_mb=memory_usage_mb,
                memory_limit_mb=memory_limit_mb,
                network_rx_bytes=total_rx,
                network_tx_bytes=total_tx,
                restart_count=container.attrs['RestartCount'],
                uptime_seconds=uptime,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error collecting container metrics: {e}")
            return None
            
    def _calculate_cpu_usage(self, stats: Dict) -> float:
        """Calculate CPU usage percentage from container stats."""
        try:
            cpu_stats = stats.get('cpu_stats', {})
            precpu_stats = stats.get('precpu_stats', {})
            
            cpu_usage = cpu_stats.get('cpu_usage', {})
            precpu_usage = precpu_stats.get('cpu_usage', {})
            
            cpu_delta = cpu_usage.get('total_usage', 0) - precpu_usage.get('total_usage', 0)
            system_delta = cpu_stats.get('system_cpu_usage', 0) - precpu_stats.get('system_cpu_usage', 0)
            
            if system_delta > 0 and cpu_delta >= 0:
                cpu_count = cpu_stats.get('online_cpus', len(cpu_usage.get('percpu_usage', [])))
                if cpu_count > 0:
                    return (cpu_delta / system_delta) * cpu_count * 100.0
                    
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating CPU usage: {e}")
            return 0.0
            
    def _analyze_container_logs(
        self, container: docker.models.containers.Container, service_name: str
    ) -> Dict[str, Any]:
        """Analyze container logs for AI-specific events and patterns."""
        try:
            # Get recent logs
            logs = container.logs(tail=50, timestamps=True).decode('utf-8', errors='ignore')
            
            # Cache logs for trend analysis
            log_lines = logs.strip().split('\n')
            self.container_logs_cache[service_name].extend(log_lines)
            
            model_info = {
                'model_loaded': None,
                'inference_active': False,
                'request_count': 0,
                'last_activity': None
            }
            
            # Parse logs for AI-specific patterns
            for line in log_lines:
                if not line.strip():
                    continue
                    
                line_lower = line.lower()
                
                # Model loading detection
                if any(pattern in line_lower for pattern in [
                    'model loaded', 'loading model', 'model:', 'gguf'
                ]):
                    # Extract model name if possible
                    if '.gguf' in line:
                        model_name = self._extract_model_name(line)
                        if model_name:
                            model_info['model_loaded'] = model_name
                            
                # Inference activity detection
                if any(pattern in line_lower for pattern in [
                    'completion', 'generate', 'inference', 'request', 'POST /v1'
                ]):
                    model_info['inference_active'] = True
                    model_info['request_count'] += 1
                    model_info['last_activity'] = datetime.now()
                    
            return model_info
            
        except Exception as e:
            logger.error(f"Error analyzing logs for {service_name}: {e}")
            return {
                'model_loaded': None,
                'inference_active': False,
                'request_count': 0,
                'last_activity': None
            }
            
    def _extract_model_name(self, log_line: str) -> Optional[str]:
        """Extract model name from log line containing model information."""
        try:
            # Look for common model file patterns
            if '.gguf' in log_line:
                parts = log_line.split('/')
                for part in parts:
                    if '.gguf' in part:
                        return part.strip().split('.gguf')[0]
            return None
        except Exception:
            return None
            
    def _generate_resource_alerts(
        self, metrics: Optional[ContainerHealthMetrics], config: AIServiceConfig
    ) -> List[str]:
        """Generate resource-based alerts for container performance."""
        alerts = []
        
        if not metrics:
            alerts.append("Container metrics unavailable")
            return alerts
            
        # CPU usage alerts
        if metrics.cpu_usage_percent > 90:
            alerts.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
        elif metrics.cpu_usage_percent > 80:
            alerts.append(f"Elevated CPU usage: {metrics.cpu_usage_percent:.1f}%")
            
        # Memory usage alerts  
        if metrics.memory_limit_mb > 0:
            memory_percent = (metrics.memory_usage_mb / metrics.memory_limit_mb) * 100
            if memory_percent > 90:
                alerts.append(f"High memory usage: {memory_percent:.1f}%")
            elif memory_percent > 80:
                alerts.append(f"Elevated memory usage: {memory_percent:.1f}%")
                
        # Health status alerts
        if metrics.health_status == 'unhealthy':
            alerts.append("Container health check failing")
        elif metrics.health_status == 'starting':
            alerts.append("Container still starting up")
            
        # Restart alerts
        if metrics.restart_count > 0:
            alerts.append(f"Container has restarted {metrics.restart_count} times")
            
        return alerts
        
    def detect_changes(self, previous_state: SystemState) -> List[SystemChange]:
        """Detect changes in AI container orchestration state."""
        changes = []
        
        if not self.docker_client:
            return changes
            
        # Update container states if not already monitoring
        if not self.monitoring_active:
            self._update_all_container_states()
            
        # Detect changes for each service
        for service_name, current_state in self.service_states.items():
            previous_service_state = self.previous_states.get(service_name)
            
            service_changes = self._detect_service_changes(
                service_name, previous_service_state, current_state
            )
            changes.extend(service_changes)
            
        # Detect cross-service patterns
        orchestration_changes = self._detect_orchestration_changes()
        changes.extend(orchestration_changes)
        
        return changes
        
    def _detect_service_changes(
        self, 
        service_name: str,
        previous: Optional[AIServiceState], 
        current: AIServiceState
    ) -> List[SystemChange]:
        """Detect changes for a specific AI service."""
        changes = []
        
        # Container availability changes
        if not previous or not previous.container_metrics:
            if current.container_metrics and current.container_metrics.status == 'running':
                changes.append(SystemChange(
                    component=ComponentType.SERVICE,
                    change_type=ChangeType.SERVICE_START,
                    description=f"AI service {service_name} started",
                    details={
                        'service_name': service_name,
                        'service_type': current.service_config.service_type,
                        'port': current.service_config.port,
                        'container_id': current.container_metrics.container_id
                    },
                    significance=Significance.MEDIUM,
                    timestamp=datetime.now()
                ))
        elif previous.container_metrics and not current.container_metrics:
            changes.append(SystemChange(
                component=ComponentType.SERVICE,
                change_type=ChangeType.SERVICE_STOP,
                description=f"AI service {service_name} stopped",
                details={'service_name': service_name},
                significance=Significance.HIGH,
                timestamp=datetime.now()
            ))
            
        if not current.container_metrics:
            return changes
            
        # Health status changes
        if (previous and previous.container_metrics and 
            previous.container_metrics.health_status != current.container_metrics.health_status):
            
            health_change = f"{previous.container_metrics.health_status}_to_{current.container_metrics.health_status}"
            significance = self.health_change_significance.get(health_change, Significance.LOW)
            
            changes.append(SystemChange(
                component=ComponentType.SERVICE,
                change_type=ChangeType.STATE_CHANGE,
                description=f"AI service {service_name} health changed: {previous.container_metrics.health_status} → {current.container_metrics.health_status}",
                details={
                    'service_name': service_name,
                    'previous_health': previous.container_metrics.health_status,
                    'current_health': current.container_metrics.health_status
                },
                significance=significance,
                timestamp=datetime.now()
            ))
            
        # Resource usage changes
        if previous and previous.container_metrics:
            cpu_change = abs(current.container_metrics.cpu_usage_percent - 
                           previous.container_metrics.cpu_usage_percent)
            if cpu_change > self.cpu_threshold:
                changes.append(SystemChange(
                    component=ComponentType.SERVICE,
                    change_type=ChangeType.RESOURCE_CHANGE,
                    description=f"AI service {service_name} CPU usage changed significantly: {cpu_change:.1f}% change",
                    details={
                        'service_name': service_name,
                        'cpu_change': cpu_change,
                        'previous_cpu': previous.container_metrics.cpu_usage_percent,
                        'current_cpu': current.container_metrics.cpu_usage_percent
                    },
                    significance=Significance.MEDIUM if cpu_change > 50 else Significance.LOW,
                    timestamp=datetime.now()
                ))
                
            memory_change = abs(current.container_metrics.memory_usage_mb -
                              previous.container_metrics.memory_usage_mb)
            if memory_change > self.memory_threshold:
                changes.append(SystemChange(
                    component=ComponentType.SERVICE,
                    change_type=ChangeType.RESOURCE_CHANGE,
                    description=f"AI service {service_name} memory usage changed: {memory_change:.0f}MB change",
                    details={
                        'service_name': service_name,
                        'memory_change_mb': memory_change,
                        'previous_memory_mb': previous.container_metrics.memory_usage_mb,
                        'current_memory_mb': current.container_metrics.memory_usage_mb
                    },
                    significance=Significance.MEDIUM if memory_change > 2048 else Significance.LOW,
                    timestamp=datetime.now()
                ))
                
        # Model loading changes
        if (not previous or previous.model_loaded != current.model_loaded) and current.model_loaded:
            changes.append(SystemChange(
                component=ComponentType.SERVICE,
                change_type=ChangeType.STATE_CHANGE,
                description=f"AI service {service_name} loaded model: {current.model_loaded}",
                details={
                    'service_name': service_name,
                    'model_loaded': current.model_loaded,
                    'previous_model': previous.model_loaded if previous else None
                },
                significance=Significance.MEDIUM,
                timestamp=datetime.now()
            ))
            
        # Resource alerts
        if current.resource_alerts:
            for alert in current.resource_alerts:
                if not previous or alert not in previous.resource_alerts:
                    changes.append(SystemChange(
                        component=ComponentType.SERVICE,
                        change_type=ChangeType.ALERT,
                        description=f"AI service {service_name} resource alert: {alert}",
                        details={
                            'service_name': service_name,
                            'alert': alert
                        },
                        significance=Significance.MEDIUM,
                        timestamp=datetime.now()
                    ))
                    
        return changes
        
    def _detect_orchestration_changes(self) -> List[SystemChange]:
        """Detect changes in overall container orchestration patterns."""
        changes = []
        
        # Count running services
        running_services = [
            name for name, state in self.service_states.items()
            if state.container_metrics and state.container_metrics.status == 'running'
        ]
        
        # Detect service availability patterns
        if len(running_services) < 3:  # Expect at least 3 services running
            changes.append(SystemChange(
                component=ComponentType.SYSTEM,
                change_type=ChangeType.ALERT,
                description=f"Low AI service availability: only {len(running_services)} services running",
                details={
                    'running_services': running_services,
                    'expected_minimum': 3
                },
                significance=Significance.HIGH,
                timestamp=datetime.now()
            ))
            
        # Detect resource distribution patterns
        total_cpu_usage = sum(
            state.container_metrics.cpu_usage_percent
            for state in self.service_states.values()
            if state.container_metrics
        )
        
        if total_cpu_usage > 300:  # High total CPU usage across services
            changes.append(SystemChange(
                component=ComponentType.SYSTEM,
                change_type=ChangeType.RESOURCE_CHANGE,
                description=f"High aggregate AI service CPU usage: {total_cpu_usage:.1f}%",
                details={'total_cpu_usage': total_cpu_usage},
                significance=Significance.MEDIUM,
                timestamp=datetime.now()
            ))
            
        return changes
        
    def get_service_states(self) -> Dict[str, AIServiceState]:
        """Get current service states for external analysis."""
        return self.service_states.copy()
        
    def get_container_health_summary(self) -> Dict[str, Any]:
        """Get summary of container health across all AI services."""
        summary = {
            'total_services': len(self.ai_services),
            'running_services': 0,
            'healthy_services': 0,
            'services_with_alerts': 0,
            'total_cpu_usage': 0.0,
            'total_memory_usage_mb': 0.0,
            'services_detail': {}
        }
        
        for name, state in self.service_states.items():
            service_summary = {
                'status': 'unknown',
                'health': 'unknown',
                'cpu_usage': 0.0,
                'memory_usage_mb': 0.0,
                'alerts': []
            }
            
            if state.container_metrics:
                service_summary.update({
                    'status': state.container_metrics.status,
                    'health': state.container_metrics.health_status or 'unknown',
                    'cpu_usage': state.container_metrics.cpu_usage_percent,
                    'memory_usage_mb': state.container_metrics.memory_usage_mb,
                    'alerts': state.resource_alerts
                })
                
                if state.container_metrics.status == 'running':
                    summary['running_services'] += 1
                    
                if state.container_metrics.health_status == 'healthy':
                    summary['healthy_services'] += 1
                    
                if state.resource_alerts:
                    summary['services_with_alerts'] += 1
                    
                summary['total_cpu_usage'] += state.container_metrics.cpu_usage_percent
                summary['total_memory_usage_mb'] += state.container_metrics.memory_usage_mb
                
            summary['services_detail'][name] = service_summary
            
        return summary
        
    def __del__(self):
        """Cleanup on object destruction."""
        try:
            self.stop_monitoring()
        except Exception:
            pass