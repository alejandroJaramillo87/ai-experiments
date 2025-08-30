"""
AI Service Lifecycle Extractor

Converts container orchestration changes into AI-specific semantic events.
Provides intelligent interpretation of container behavior with specialized
knowledge of AI workstation patterns, model lifecycle, and service interactions.

Features:
- AI service lifecycle event extraction (model loading, inference patterns)
- Service interaction and load balancing detection
- Performance event analysis with AI workstation context
- Model lifecycle tracking and transition analysis
- Resource utilization pattern recognition
- Service health correlation and predictive insights
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter

from ...temporal.core.event_extractor import SystemEventExtractor, SystemEvent
from ...temporal.core.change_detector import SystemChange
from ...temporal.core.types import ChangeType, ComponentType, Significance, EventCategory


logger = logging.getLogger(__name__)


@dataclass
class AIServiceEvent:
    """Specialized event for AI service operations."""
    service_name: str
    service_type: str  # cpu, gpu, vllm, interface
    event_type: str  # model_loaded, inference_spike, health_degraded, etc.
    model_name: Optional[str] = None
    resource_impact: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    correlation_events: Optional[List[str]] = None


@dataclass
class ServiceInteractionPattern:
    """Pattern representing service interactions and load balancing."""
    pattern_type: str  # load_balancing, resource_competition, cascade_failure
    involved_services: List[str]
    confidence: float
    description: str
    impact_level: Significance
    suggested_actions: List[str]


class AIServiceLifecycleExtractor(SystemEventExtractor):
    """
    Extracts AI-specific semantic events from container orchestration changes.
    
    Transforms raw container state changes into meaningful AI workstation events
    with specialized understanding of model lifecycle, service interactions,
    and performance patterns specific to AI development environments.
    """
    
    def __init__(self):
        super().__init__()
        
        # AI service knowledge base
        self.service_types = {
            'llama-cpu-0': 'cpu_inference',
            'llama-cpu-1': 'cpu_inference', 
            'llama-cpu-2': 'cpu_inference',
            'llama-gpu': 'gpu_inference',
            'vllm-gpu': 'vllm_inference',
            'open-webui': 'user_interface'
        }
        
        # Model name patterns for recognition
        self.model_patterns = {
            'llama': r'(llama|Llama)[\w\-\.]*',
            'qwen': r'(qwen|Qwen)[\w\-\.]*',
            'mistral': r'(mistral|Mistral)[\w\-\.]*',
            'claude': r'(claude|Claude)[\w\-\.]*',
            'gpt': r'(gpt|GPT)[\w\-\.]*'
        }
        
        # Event pattern recognition
        self.event_patterns = {
            'model_loading': [
                r'loading model|model loaded|gguf.*loaded',
                r'initializing.*model|model.*initialization',
                r'loading.*checkpoint|checkpoint.*loaded'
            ],
            'inference_activity': [
                r'completion.*request|generate.*request',
                r'POST /v1/(chat/|)completions',
                r'inference.*started|processing.*request'
            ],
            'memory_events': [
                r'CUDA.*memory|GPU.*memory|VRAM',
                r'out of memory|OOM|memory.*exhausted',
                r'memory.*allocated|memory.*freed'
            ],
            'performance_events': [
                r'tokens/sec|tokens per second|throughput',
                r'latency.*ms|response.*time',
                r'batch.*size|sequence.*length'
            ]
        }
        
        # Service interaction tracking
        self.recent_events: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)
        self.service_health_history: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)
        self.resource_usage_trends: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        
        # Performance thresholds for event significance
        self.performance_thresholds = {
            'cpu_usage_high': 80.0,
            'cpu_usage_critical': 95.0,
            'memory_usage_high': 80.0,
            'memory_usage_critical': 95.0,
            'inference_rate_low': 0.1,  # requests per second
            'inference_rate_high': 10.0
        }
        
    def extract_events(self, changes: List[SystemChange]) -> List[SystemEvent]:
        """Extract AI-specific semantic events from container changes."""
        extracted_events = []
        
        # Group changes by service for correlation analysis
        service_changes = self._group_changes_by_service(changes)
        
        # Extract events for each service
        for service_name, service_changes_list in service_changes.items():
            service_events = self._extract_service_events(service_name, service_changes_list)
            extracted_events.extend(service_events)
            
        # Extract cross-service interaction patterns
        interaction_events = self._extract_service_interaction_events(changes)
        extracted_events.extend(interaction_events)
        
        # Extract system-wide AI workstation events
        workstation_events = self._extract_workstation_events(changes)
        extracted_events.extend(workstation_events)
        
        # Update internal tracking for future correlations
        self._update_service_tracking(changes)
        
        return extracted_events
        
    def _group_changes_by_service(self, changes: List[SystemChange]) -> Dict[str, List[SystemChange]]:
        """Group changes by AI service for correlation analysis."""
        service_changes = defaultdict(list)
        
        for change in changes:
            # Extract service name from change details
            service_name = None
            if hasattr(change, 'details') and isinstance(change.details, dict):
                service_name = change.details.get('service_name')
                
            if service_name and service_name in self.service_types:
                service_changes[service_name].append(change)
            else:
                # General system changes that might affect AI services
                service_changes['system'].append(change)
                
        return service_changes
        
    def _extract_service_events(
        self, service_name: str, changes: List[SystemChange]
    ) -> List[SystemEvent]:
        """Extract events specific to an AI service."""
        events = []
        
        for change in changes:
            # Service lifecycle events
            if change.change_type == ChangeType.SERVICE_START:
                event = self._create_service_lifecycle_event(
                    service_name, 'service_started', change
                )
                events.append(event)
                
            elif change.change_type == ChangeType.SERVICE_STOP:
                event = self._create_service_lifecycle_event(
                    service_name, 'service_stopped', change
                )
                events.append(event)
                
            # Model lifecycle events
            elif 'model' in change.description.lower():
                model_event = self._extract_model_lifecycle_event(service_name, change)
                if model_event:
                    events.append(model_event)
                    
            # Resource events
            elif change.change_type == ChangeType.RESOURCE_CHANGE:
                resource_event = self._extract_resource_event(service_name, change)
                if resource_event:
                    events.append(resource_event)
                    
            # Health events  
            elif 'health' in change.description.lower():
                health_event = self._extract_health_event(service_name, change)
                if health_event:
                    events.append(health_event)
                    
        return events
        
    def _create_service_lifecycle_event(
        self, service_name: str, event_type: str, change: SystemChange
    ) -> SystemEvent:
        """Create a service lifecycle event with AI context."""
        service_type = self.service_types.get(service_name, 'unknown')
        
        # Determine impact based on service type
        impact_description = self._get_service_impact_description(service_name, event_type)
        
        # Extract additional context
        context = self._extract_service_context(change)
        
        return SystemEvent(
            category=EventCategory.SERVICE_MANAGEMENT,
            title=f"AI Service {event_type.replace('_', ' ').title()}",
            description=f"{service_name} ({service_type}) {event_type.replace('_', ' ')}: {impact_description}",
            details={
                'service_name': service_name,
                'service_type': service_type,
                'event_type': event_type,
                'impact_description': impact_description,
                **context
            },
            significance=change.significance,
            timestamp=change.timestamp,
            duration=None,
            related_components=[ComponentType.SERVICE],
            causal_chain=[],
            predicted_effects=self._predict_service_lifecycle_effects(service_name, event_type)
        )
        
    def _extract_model_lifecycle_event(
        self, service_name: str, change: SystemChange
    ) -> Optional[SystemEvent]:
        """Extract model lifecycle events (loading, switching, etc.)."""
        description = change.description.lower()
        
        # Determine model event type
        event_type = None
        model_name = None
        
        if 'loaded model' in description or 'model loaded' in description:
            event_type = 'model_loaded'
            model_name = self._extract_model_name_from_description(change.description)
        elif 'loading model' in description:
            event_type = 'model_loading'
            model_name = self._extract_model_name_from_description(change.description)
        elif 'model switched' in description or 'switching model' in description:
            event_type = 'model_switched'
            model_name = self._extract_model_name_from_description(change.description)
            
        if not event_type:
            return None
            
        # Assess model impact
        model_impact = self._assess_model_impact(service_name, model_name, event_type)
        
        return SystemEvent(
            category=EventCategory.AI_MODEL_LIFECYCLE,
            title=f"AI Model {event_type.replace('_', ' ').title()}",
            description=f"{service_name} {event_type.replace('_', ' ')}: {model_name or 'Unknown Model'}",
            details={
                'service_name': service_name,
                'service_type': self.service_types.get(service_name, 'unknown'),
                'event_type': event_type,
                'model_name': model_name,
                'model_impact': model_impact,
                'estimated_vram_usage': model_impact.get('vram_usage_gb', 0),
                'estimated_ram_usage': model_impact.get('ram_usage_gb', 0)
            },
            significance=self._determine_model_event_significance(event_type, model_impact),
            timestamp=change.timestamp,
            duration=None,
            related_components=[ComponentType.SERVICE, ComponentType.AI_MODEL],
            causal_chain=[],
            predicted_effects=self._predict_model_lifecycle_effects(service_name, event_type, model_name)
        )
        
    def _extract_resource_event(
        self, service_name: str, change: SystemChange
    ) -> Optional[SystemEvent]:
        """Extract resource-related events with AI context."""
        details = change.details or {}
        
        # Determine resource event type
        event_type = 'resource_change'
        if 'cpu_change' in details:
            event_type = 'cpu_usage_change'
        elif 'memory_change' in details:
            event_type = 'memory_usage_change'
            
        # Assess resource impact significance
        resource_significance = self._assess_resource_change_significance(change, details)
        
        # Generate recommendations
        recommendations = self._generate_resource_recommendations(service_name, change, details)
        
        return SystemEvent(
            category=EventCategory.RESOURCE_OPTIMIZATION,
            title=f"AI Service Resource {event_type.replace('_', ' ').title()}",
            description=change.description,
            details={
                'service_name': service_name,
                'service_type': self.service_types.get(service_name, 'unknown'),
                'event_type': event_type,
                'resource_change': details,
                'recommendations': recommendations,
                'performance_impact': resource_significance.get('performance_impact', 'minimal')
            },
            significance=resource_significance.get('significance', change.significance),
            timestamp=change.timestamp,
            duration=None,
            related_components=[ComponentType.SERVICE, ComponentType.RESOURCE],
            causal_chain=[],
            predicted_effects=self._predict_resource_change_effects(service_name, details)
        )
        
    def _extract_health_event(
        self, service_name: str, change: SystemChange
    ) -> Optional[SystemEvent]:
        """Extract health-related events with AI service context."""
        details = change.details or {}
        
        previous_health = details.get('previous_health', 'unknown')
        current_health = details.get('current_health', 'unknown')
        
        # Assess health transition impact
        health_impact = self._assess_health_transition_impact(
            service_name, previous_health, current_health
        )
        
        return SystemEvent(
            category=EventCategory.HEALTH_MONITORING,
            title=f"AI Service Health Transition",
            description=f"{service_name} health changed: {previous_health} → {current_health}",
            details={
                'service_name': service_name,
                'service_type': self.service_types.get(service_name, 'unknown'),
                'previous_health': previous_health,
                'current_health': current_health,
                'health_impact': health_impact,
                'recovery_recommendations': health_impact.get('recommendations', [])
            },
            significance=health_impact.get('significance', change.significance),
            timestamp=change.timestamp,
            duration=None,
            related_components=[ComponentType.SERVICE, ComponentType.HEALTH],
            causal_chain=[],
            predicted_effects=self._predict_health_change_effects(service_name, current_health)
        )
        
    def _extract_service_interaction_events(
        self, changes: List[SystemChange]
    ) -> List[SystemEvent]:
        """Extract events related to service interactions and patterns."""
        events = []
        
        # Detect load balancing patterns
        load_balancing_events = self._detect_load_balancing_patterns(changes)
        events.extend(load_balancing_events)
        
        # Detect resource competition
        competition_events = self._detect_resource_competition_patterns(changes)
        events.extend(competition_events)
        
        # Detect cascade failures
        cascade_events = self._detect_cascade_failure_patterns(changes)
        events.extend(cascade_events)
        
        return events
        
    def _detect_load_balancing_patterns(
        self, changes: List[SystemChange]
    ) -> List[SystemEvent]:
        """Detect load balancing patterns across AI services."""
        events = []
        
        # Look for simultaneous resource changes across CPU services
        cpu_services = ['llama-cpu-0', 'llama-cpu-1', 'llama-cpu-2']
        cpu_changes = []
        
        for change in changes:
            if (change.details and 
                change.details.get('service_name') in cpu_services and
                change.change_type == ChangeType.RESOURCE_CHANGE):
                cpu_changes.append(change)
                
        if len(cpu_changes) >= 2:
            # Multiple CPU services showing resource changes - likely load balancing
            event = SystemEvent(
                category=EventCategory.WORKLOAD_OPTIMIZATION,
                title="AI Service Load Balancing Detected",
                description=f"Load balancing pattern detected across {len(cpu_changes)} CPU inference services",
                details={
                    'pattern_type': 'load_balancing',
                    'affected_services': [c.details.get('service_name') for c in cpu_changes],
                    'resource_changes': [c.details for c in cpu_changes],
                    'load_distribution_efficiency': self._calculate_load_distribution_efficiency(cpu_changes)
                },
                significance=Significance.MEDIUM,
                timestamp=datetime.now(),
                duration=None,
                related_components=[ComponentType.SERVICE, ComponentType.WORKLOAD],
                causal_chain=[],
                predicted_effects=[
                    "Improved inference throughput across CPU services",
                    "More balanced resource utilization",
                    "Reduced latency for concurrent requests"
                ]
            )
            events.append(event)
            
        return events
        
    def _detect_resource_competition_patterns(
        self, changes: List[SystemChange]
    ) -> List[SystemEvent]:
        """Detect resource competition between GPU services."""
        events = []
        
        # Look for GPU services showing high resource usage
        gpu_services = ['llama-gpu', 'vllm-gpu']
        gpu_resource_changes = []
        
        for change in changes:
            if (change.details and 
                change.details.get('service_name') in gpu_services and
                change.change_type == ChangeType.RESOURCE_CHANGE):
                gpu_resource_changes.append(change)
                
        if len(gpu_resource_changes) >= 2:
            # Multiple GPU services competing for resources
            event = SystemEvent(
                category=EventCategory.RESOURCE_CONTENTION,
                title="GPU Resource Competition Detected", 
                description="Multiple GPU services competing for RTX 5090 resources",
                details={
                    'pattern_type': 'resource_competition',
                    'competing_services': [c.details.get('service_name') for c in gpu_resource_changes],
                    'resource_changes': [c.details for c in gpu_resource_changes],
                    'recommended_resolution': self._recommend_gpu_competition_resolution(gpu_resource_changes)
                },
                significance=Significance.HIGH,
                timestamp=datetime.now(),
                duration=None,
                related_components=[ComponentType.SERVICE, ComponentType.GPU, ComponentType.RESOURCE],
                causal_chain=[],
                predicted_effects=[
                    "Potential GPU memory contention",
                    "Reduced inference performance for both services",
                    "Possible CUDA context switching overhead"
                ]
            )
            events.append(event)
            
        return events
        
    def _detect_cascade_failure_patterns(
        self, changes: List[SystemChange]
    ) -> List[SystemEvent]:
        """Detect cascade failure patterns across services."""
        events = []
        
        # Look for multiple service failures in short time window
        failure_changes = [
            change for change in changes
            if (change.change_type == ChangeType.SERVICE_STOP or
                change.significance == Significance.CRITICAL)
        ]
        
        if len(failure_changes) >= 2:
            # Multiple failures detected - potential cascade
            time_window = timedelta(minutes=5)
            failure_times = [change.timestamp for change in failure_changes]
            
            if max(failure_times) - min(failure_times) <= time_window:
                event = SystemEvent(
                    category=EventCategory.SYSTEM_FAILURE,
                    title="AI Service Cascade Failure Detected",
                    description=f"Multiple service failures detected within {time_window}",
                    details={
                        'pattern_type': 'cascade_failure',
                        'failed_services': [
                            change.details.get('service_name', 'unknown') 
                            for change in failure_changes if change.details
                        ],
                        'failure_sequence': [
                            {
                                'service': change.details.get('service_name', 'unknown'),
                                'timestamp': change.timestamp.isoformat(),
                                'description': change.description
                            }
                            for change in failure_changes if change.details
                        ],
                        'root_cause_analysis': self._analyze_cascade_root_cause(failure_changes)
                    },
                    significance=Significance.CRITICAL,
                    timestamp=min(failure_times),
                    duration=max(failure_times) - min(failure_times),
                    related_components=[ComponentType.SERVICE, ComponentType.SYSTEM],
                    causal_chain=[],
                    predicted_effects=[
                        "AI inference capability severely impacted",
                        "Potential system instability",
                        "Manual intervention required for recovery"
                    ]
                )
                events.append(event)
                
        return events
        
    def _extract_workstation_events(self, changes: List[SystemChange]) -> List[SystemEvent]:
        """Extract system-wide AI workstation events."""
        events = []
        
        # Analyze overall AI service ecosystem health
        service_availability = self._analyze_service_availability(changes)
        if service_availability['event_significant']:
            events.append(service_availability['event'])
            
        # Analyze resource distribution patterns
        resource_distribution = self._analyze_resource_distribution(changes)
        if resource_distribution['event_significant']:
            events.append(resource_distribution['event'])
            
        return events
        
    def _analyze_service_availability(self, changes: List[SystemChange]) -> Dict[str, Any]:
        """Analyze overall AI service ecosystem availability."""
        running_services = set()
        stopped_services = set()
        
        for change in changes:
            if change.details and 'service_name' in change.details:
                service_name = change.details['service_name']
                if change.change_type == ChangeType.SERVICE_START:
                    running_services.add(service_name)
                elif change.change_type == ChangeType.SERVICE_STOP:
                    stopped_services.add(service_name)
                    
        total_expected_services = len(self.service_types)
        availability_percentage = (len(running_services) / total_expected_services) * 100
        
        event_significant = availability_percentage < 80  # Less than 80% availability
        
        event = None
        if event_significant:
            event = SystemEvent(
                category=EventCategory.SYSTEM_HEALTH,
                title="AI Service Ecosystem Availability Alert",
                description=f"AI service availability at {availability_percentage:.1f}% ({len(running_services)}/{total_expected_services} services)",
                details={
                    'availability_percentage': availability_percentage,
                    'running_services': list(running_services),
                    'stopped_services': list(stopped_services),
                    'total_expected_services': total_expected_services,
                    'impact_assessment': self._assess_availability_impact(availability_percentage)
                },
                significance=Significance.HIGH if availability_percentage < 60 else Significance.MEDIUM,
                timestamp=datetime.now(),
                duration=None,
                related_components=[ComponentType.SERVICE, ComponentType.SYSTEM],
                causal_chain=[],
                predicted_effects=self._predict_availability_impact_effects(availability_percentage)
            )
            
        return {
            'event_significant': event_significant,
            'event': event,
            'availability_percentage': availability_percentage
        }
        
    def _analyze_resource_distribution(self, changes: List[SystemChange]) -> Dict[str, Any]:
        """Analyze resource distribution across AI services."""
        resource_changes = [
            change for change in changes 
            if change.change_type == ChangeType.RESOURCE_CHANGE
        ]
        
        if len(resource_changes) < 2:
            return {'event_significant': False, 'event': None}
            
        # Calculate total resource impact
        total_cpu_impact = 0
        total_memory_impact = 0
        
        for change in resource_changes:
            details = change.details or {}
            total_cpu_impact += abs(details.get('cpu_change', 0))
            total_memory_impact += abs(details.get('memory_change_mb', 0))
            
        # Determine if resource distribution is significant
        event_significant = (total_cpu_impact > 100 or total_memory_impact > 4096)  # 100% CPU or 4GB memory
        
        event = None
        if event_significant:
            event = SystemEvent(
                category=EventCategory.RESOURCE_OPTIMIZATION,
                title="AI Workstation Resource Distribution Analysis",
                description=f"Significant resource redistribution: {total_cpu_impact:.1f}% CPU, {total_memory_impact:.0f}MB memory",
                details={
                    'total_cpu_impact': total_cpu_impact,
                    'total_memory_impact_mb': total_memory_impact,
                    'affected_services': [
                        change.details.get('service_name') for change in resource_changes 
                        if change.details and 'service_name' in change.details
                    ],
                    'optimization_opportunities': self._identify_optimization_opportunities(resource_changes)
                },
                significance=Significance.MEDIUM,
                timestamp=datetime.now(),
                duration=None,
                related_components=[ComponentType.SERVICE, ComponentType.RESOURCE],
                causal_chain=[],
                predicted_effects=[
                    "Resource utilization pattern change",
                    "Potential performance impact on concurrent inference",
                    "Opportunity for workload optimization"
                ]
            )
            
        return {
            'event_significant': event_significant,
            'event': event,
            'resource_impact': {
                'cpu': total_cpu_impact,
                'memory_mb': total_memory_impact
            }
        }
        
    def _update_service_tracking(self, changes: List[SystemChange]):
        """Update internal tracking for future correlation analysis."""
        current_time = datetime.now()
        
        for change in changes:
            if change.details and 'service_name' in change.details:
                service_name = change.details['service_name']
                
                # Track recent events
                self.recent_events[service_name].append((current_time, change.description))
                # Keep only recent events (last hour)
                cutoff_time = current_time - timedelta(hours=1)
                self.recent_events[service_name] = [
                    (time, desc) for time, desc in self.recent_events[service_name]
                    if time > cutoff_time
                ]
                
                # Track health changes
                if 'health' in change.description.lower():
                    health_status = change.details.get('current_health', 'unknown')
                    self.service_health_history[service_name].append((current_time, health_status))
                    # Keep only recent health history (last 24 hours)
                    cutoff_time = current_time - timedelta(hours=24)
                    self.service_health_history[service_name] = [
                        (time, status) for time, status in self.service_health_history[service_name]
                        if time > cutoff_time
                    ]
                    
                # Track resource trends
                if change.change_type == ChangeType.RESOURCE_CHANGE:
                    cpu_usage = change.details.get('current_cpu', 0)
                    self.resource_usage_trends[service_name].append((current_time, cpu_usage))
                    # Keep only recent trends (last 4 hours)
                    cutoff_time = current_time - timedelta(hours=4)
                    self.resource_usage_trends[service_name] = [
                        (time, usage) for time, usage in self.resource_usage_trends[service_name]
                        if time > cutoff_time
                    ]
                    
    # Helper methods for event extraction and analysis
    
    def _get_service_impact_description(self, service_name: str, event_type: str) -> str:
        """Get impact description for service lifecycle events."""
        service_type = self.service_types.get(service_name, 'unknown')
        
        if event_type == 'service_started':
            if 'cpu' in service_type:
                return "CPU-based inference capability restored"
            elif 'gpu' in service_type:
                return "GPU-accelerated inference capability restored"
            elif 'interface' in service_type:
                return "User interface access restored"
        elif event_type == 'service_stopped':
            if 'cpu' in service_type:
                return "CPU-based inference capability lost"
            elif 'gpu' in service_type:
                return "GPU-accelerated inference capability lost"
            elif 'interface' in service_type:
                return "User interface access lost"
                
        return f"Service {event_type} impact on {service_type}"
        
    def _extract_service_context(self, change: SystemChange) -> Dict[str, Any]:
        """Extract additional context from service changes."""
        context = {}
        
        if change.details:
            context.update(change.details)
            
        # Add timing context
        context['event_timestamp'] = change.timestamp.isoformat()
        context['significance_level'] = change.significance.value
        
        return context
        
    def _extract_model_name_from_description(self, description: str) -> Optional[str]:
        """Extract model name from change description."""
        # Try to match known model patterns
        for model_family, pattern in self.model_patterns.items():
            match = re.search(pattern, description)
            if match:
                return match.group(0)
                
        # Try to extract .gguf filename
        gguf_match = re.search(r'(\w+[.\-]\w*)?\.gguf', description, re.IGNORECASE)
        if gguf_match:
            return gguf_match.group(0).replace('.gguf', '')
            
        return None
        
    def _assess_model_impact(self, service_name: str, model_name: Optional[str], event_type: str) -> Dict[str, Any]:
        """Assess the impact of model lifecycle events."""
        impact = {
            'vram_usage_gb': 0,
            'ram_usage_gb': 0,
            'compute_intensity': 'medium',
            'inference_capability': 'standard'
        }
        
        # Estimate resource usage based on model name and service type
        service_type = self.service_types.get(service_name, 'unknown')
        
        if model_name and 'gpu' in service_type:
            # GPU models typically use VRAM
            if any(size in model_name.lower() for size in ['30b', '33b']):
                impact['vram_usage_gb'] = 20  # Large model
                impact['compute_intensity'] = 'high'
            elif any(size in model_name.lower() for size in ['13b', '7b']):
                impact['vram_usage_gb'] = 8   # Medium model
            else:
                impact['vram_usage_gb'] = 4   # Default estimate
                
        elif model_name and 'cpu' in service_type:
            # CPU models use system RAM
            if any(size in model_name.lower() for size in ['30b', '33b']):
                impact['ram_usage_gb'] = 30  # Large model
                impact['compute_intensity'] = 'high'
            elif any(size in model_name.lower() for size in ['13b']):
                impact['ram_usage_gb'] = 15  # Medium model  
            elif any(size in model_name.lower() for size in ['7b']):
                impact['ram_usage_gb'] = 8   # Small model
            else:
                impact['ram_usage_gb'] = 10  # Default estimate
                
        return impact
        
    def _determine_model_event_significance(self, event_type: str, model_impact: Dict[str, Any]) -> Significance:
        """Determine significance level for model events."""
        if event_type == 'model_loaded':
            if model_impact.get('vram_usage_gb', 0) > 15 or model_impact.get('ram_usage_gb', 0) > 20:
                return Significance.HIGH  # Large model loaded
            else:
                return Significance.MEDIUM
        elif event_type == 'model_loading':
            return Significance.LOW  # Loading in progress
        elif event_type == 'model_switched':
            return Significance.MEDIUM  # Model switching
            
        return Significance.LOW
        
    def _predict_service_lifecycle_effects(self, service_name: str, event_type: str) -> List[str]:
        """Predict effects of service lifecycle events."""
        effects = []
        service_type = self.service_types.get(service_name, 'unknown')
        
        if event_type == 'service_started':
            if 'cpu' in service_type:
                effects.append("Increased CPU inference capacity")
                effects.append("Improved load balancing across CPU services")
            elif 'gpu' in service_type:
                effects.append("GPU-accelerated inference capability restored")
                effects.append("Potential reduction in CPU service load")
            elif 'interface' in service_type:
                effects.append("User access to AI services restored")
                
        elif event_type == 'service_stopped':
            if 'cpu' in service_type:
                effects.append("Reduced CPU inference capacity")
                effects.append("Increased load on remaining CPU services")
            elif 'gpu' in service_type:
                effects.append("Loss of GPU-accelerated inference")
                effects.append("Potential increase in CPU service load")
                
        return effects
        
    def _predict_model_lifecycle_effects(
        self, service_name: str, event_type: str, model_name: Optional[str]
    ) -> List[str]:
        """Predict effects of model lifecycle events."""
        effects = []
        service_type = self.service_types.get(service_name, 'unknown')
        
        if event_type == 'model_loaded':
            if 'gpu' in service_type:
                effects.append("GPU memory allocated for model parameters")
                effects.append("Service ready for high-performance inference")
            elif 'cpu' in service_type:
                effects.append("System memory allocated for model parameters")
                effects.append("CPU-based inference capability enabled")
                
            if model_name and any(size in model_name.lower() for size in ['30b', '33b']):
                effects.append("Large model capabilities: complex reasoning, extensive knowledge")
                effects.append("Higher latency but superior output quality")
                
        elif event_type == 'model_loading':
            effects.append("Service temporarily unavailable during model loading")
            effects.append("Resource allocation in progress")
            
        return effects
        
    def _assess_resource_change_significance(
        self, change: SystemChange, details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess significance of resource changes for AI services."""
        significance_info = {'significance': change.significance}
        
        cpu_change = abs(details.get('cpu_change', 0))
        memory_change = abs(details.get('memory_change_mb', 0))
        
        # Determine performance impact
        if cpu_change > 50 or memory_change > 2048:
            significance_info['performance_impact'] = 'significant'
            significance_info['significance'] = Significance.HIGH
        elif cpu_change > 20 or memory_change > 512:
            significance_info['performance_impact'] = 'moderate'
            significance_info['significance'] = Significance.MEDIUM
        else:
            significance_info['performance_impact'] = 'minimal'
            
        return significance_info
        
    def _generate_resource_recommendations(
        self, service_name: str, change: SystemChange, details: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for resource changes."""
        recommendations = []
        service_type = self.service_types.get(service_name, 'unknown')
        
        cpu_usage = details.get('current_cpu', 0)
        memory_usage_mb = details.get('current_memory_mb', 0)
        
        if cpu_usage > 90:
            if 'cpu' in service_type:
                recommendations.append("Consider distributing load to other CPU services")
                recommendations.append("Monitor for potential thermal throttling")
            recommendations.append("Evaluate model complexity vs performance requirements")
            
        if memory_usage_mb > 30000:  # > 30GB
            if 'cpu' in service_type:
                recommendations.append("High memory usage - consider model quantization")
                recommendations.append("Monitor system memory availability for other services")
                
        if cpu_usage < 10 and memory_usage_mb > 10000:
            recommendations.append("Low CPU with high memory usage - potential model loading/idle state")
            
        return recommendations
        
    def _predict_resource_change_effects(
        self, service_name: str, details: Dict[str, Any]
    ) -> List[str]:
        """Predict effects of resource changes."""
        effects = []
        
        cpu_change = details.get('cpu_change', 0)
        memory_change = details.get('memory_change_mb', 0)
        
        if cpu_change > 30:
            effects.append("Increased inference processing capability")
            effects.append("Higher power consumption and heat generation")
        elif cpu_change < -30:
            effects.append("Reduced inference processing load")
            effects.append("Lower power consumption")
            
        if memory_change > 1024:
            effects.append("Increased model or data caching")
            effects.append("Potential impact on system memory availability")
        elif memory_change < -1024:
            effects.append("Reduced memory footprint")
            effects.append("Memory available for other processes")
            
        return effects
        
    def _assess_health_transition_impact(
        self, service_name: str, previous_health: str, current_health: str
    ) -> Dict[str, Any]:
        """Assess impact of health transitions."""
        impact = {
            'significance': Significance.LOW,
            'recommendations': []
        }
        
        if previous_health == 'healthy' and current_health == 'unhealthy':
            impact['significance'] = Significance.HIGH
            impact['recommendations'] = [
                "Investigate service logs for error patterns",
                "Check resource availability and constraints",
                "Verify model loading and initialization status",
                "Consider service restart if issues persist"
            ]
            
        elif previous_health == 'unhealthy' and current_health == 'healthy':
            impact['significance'] = Significance.MEDIUM
            impact['recommendations'] = [
                "Monitor service stability for sustained health",
                "Review what resolved the health issue",
                "Update health check thresholds if needed"
            ]
            
        elif current_health == 'starting':
            impact['significance'] = Significance.LOW
            impact['recommendations'] = [
                "Allow additional time for service initialization",
                "Monitor model loading progress",
                "Check for resource availability during startup"
            ]
            
        return impact
        
    def _predict_health_change_effects(self, service_name: str, current_health: str) -> List[str]:
        """Predict effects of health changes."""
        effects = []
        service_type = self.service_types.get(service_name, 'unknown')
        
        if current_health == 'healthy':
            effects.append("Service available for inference requests")
            if 'cpu' in service_type:
                effects.append("CPU inference capability restored")
            elif 'gpu' in service_type:
                effects.append("GPU inference capability restored")
                
        elif current_health == 'unhealthy':
            effects.append("Service unavailable for inference requests")
            effects.append("Potential impact on overall AI system capability")
            if 'cpu' in service_type:
                effects.append("Reduced CPU inference capacity")
            elif 'gpu' in service_type:
                effects.append("Loss of GPU inference capability")
                
        elif current_health == 'starting':
            effects.append("Service initialization in progress")
            effects.append("Temporary unavailability during startup")
            
        return effects
        
    def _calculate_load_distribution_efficiency(self, cpu_changes: List[SystemChange]) -> float:
        """Calculate load distribution efficiency across CPU services."""
        if len(cpu_changes) < 2:
            return 0.0
            
        # Calculate variance in CPU usage changes
        cpu_deltas = []
        for change in cpu_changes:
            cpu_change = change.details.get('cpu_change', 0)
            cpu_deltas.append(abs(cpu_change))
            
        if not cpu_deltas:
            return 0.0
            
        mean_delta = sum(cpu_deltas) / len(cpu_deltas)
        variance = sum((delta - mean_delta) ** 2 for delta in cpu_deltas) / len(cpu_deltas)
        
        # Lower variance indicates better load distribution
        # Convert to efficiency score (higher is better)
        efficiency = max(0.0, 1.0 - (variance / (mean_delta ** 2)) if mean_delta > 0 else 0.0)
        
        return efficiency
        
    def _recommend_gpu_competition_resolution(self, gpu_changes: List[SystemChange]) -> Dict[str, Any]:
        """Recommend resolution for GPU resource competition."""
        recommendations = {
            'immediate_actions': [],
            'optimization_strategies': []
        }
        
        recommendations['immediate_actions'] = [
            "Monitor GPU memory usage to prevent OOM errors",
            "Consider model quantization to reduce VRAM requirements",
            "Implement request queuing to serialize GPU access"
        ]
        
        recommendations['optimization_strategies'] = [
            "Configure model-specific GPU memory limits",
            "Implement dynamic model swapping based on request patterns",
            "Consider load balancing between GPU and CPU services"
        ]
        
        return recommendations
        
    def _analyze_cascade_root_cause(self, failure_changes: List[SystemChange]) -> Dict[str, Any]:
        """Analyze root cause of cascade failures."""
        analysis = {
            'potential_root_causes': [],
            'failure_pattern': 'unknown',
            'recommended_investigation': []
        }
        
        # Analyze failure sequence
        failure_times = [change.timestamp for change in failure_changes]
        failure_services = [
            change.details.get('service_name', 'unknown') for change in failure_changes 
            if change.details
        ]
        
        # Determine failure pattern
        if 'llama-gpu' in failure_services or 'vllm-gpu' in failure_services:
            analysis['failure_pattern'] = 'gpu_initiated'
            analysis['potential_root_causes'] = [
                "GPU driver instability",
                "CUDA runtime error",
                "GPU thermal throttling",
                "Power supply instability"
            ]
        elif len([s for s in failure_services if 'cpu' in s]) >= 2:
            analysis['failure_pattern'] = 'cpu_services_cascade'
            analysis['potential_root_causes'] = [
                "System memory exhaustion",
                "CPU thermal throttling",
                "Network connectivity issues",
                "Container orchestration problems"
            ]
            
        analysis['recommended_investigation'] = [
            "Check system logs for hardware errors",
            "Review container orchestration logs",
            "Monitor thermal and power management",
            "Verify network connectivity between services"
        ]
        
        return analysis
        
    def _assess_availability_impact(self, availability_percentage: float) -> Dict[str, Any]:
        """Assess impact of service availability changes."""
        impact = {}
        
        if availability_percentage < 40:
            impact['severity'] = 'critical'
            impact['inference_capability'] = 'severely impacted'
            impact['user_impact'] = 'major service disruption'
        elif availability_percentage < 60:
            impact['severity'] = 'high'
            impact['inference_capability'] = 'significantly reduced'
            impact['user_impact'] = 'notable service degradation'
        elif availability_percentage < 80:
            impact['severity'] = 'medium'
            impact['inference_capability'] = 'moderately reduced'
            impact['user_impact'] = 'some service limitations'
        else:
            impact['severity'] = 'low'
            impact['inference_capability'] = 'minimally impacted'
            impact['user_impact'] = 'minor service disruption'
            
        return impact
        
    def _predict_availability_impact_effects(self, availability_percentage: float) -> List[str]:
        """Predict effects of availability changes."""
        effects = []
        
        if availability_percentage < 60:
            effects.append("Significant reduction in concurrent inference capacity")
            effects.append("Increased latency for inference requests")
            effects.append("Potential request queuing or rejection")
        elif availability_percentage < 80:
            effects.append("Reduced inference throughput")
            effects.append("Load concentration on available services")
            effects.append("Potential performance degradation")
        else:
            effects.append("Minimal impact on overall inference capability")
            effects.append("Automatic load rebalancing among available services")
            
        return effects
        
    def _identify_optimization_opportunities(self, resource_changes: List[SystemChange]) -> List[str]:
        """Identify optimization opportunities from resource changes."""
        opportunities = []
        
        # Analyze resource usage patterns
        high_cpu_services = []
        low_cpu_services = []
        high_memory_services = []
        
        for change in resource_changes:
            if not change.details:
                continue
                
            service_name = change.details.get('service_name')
            cpu_usage = change.details.get('current_cpu', 0)
            memory_usage_mb = change.details.get('current_memory_mb', 0)
            
            if cpu_usage > 80:
                high_cpu_services.append(service_name)
            elif cpu_usage < 20:
                low_cpu_services.append(service_name)
                
            if memory_usage_mb > 20000:  # > 20GB
                high_memory_services.append(service_name)
                
        # Generate optimization recommendations
        if high_cpu_services and low_cpu_services:
            opportunities.append(f"Load balancing opportunity: redistribute from {high_cpu_services} to {low_cpu_services}")
            
        if high_memory_services:
            opportunities.append(f"Memory optimization: consider model quantization for {high_memory_services}")
            
        if len(high_cpu_services) >= 2:
            opportunities.append("Multiple high-CPU services: consider request queuing or rate limiting")
            
        return opportunities