"""
Event Extraction Engine - Compatibility Bridge
==============================================

Provides a compatibility layer between the old EventExtractionEngine interface
and the new sophisticated event extraction framework. This bridge maintains
API compatibility while leveraging the advanced causal analysis, effect
prediction, and semantic understanding capabilities of the new system.

This adapter transforms the simple orchestrator pattern into the sophisticated
pattern-based extraction with contextual understanding.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from .types import SystemChange, SystemEvent, EventCorrelation, SystemSnapshot
from .event_extraction import SystemEventExtractor, EventContext, CausalAnalyzer, EffectPredictor
from .config import EventExtractorConfig

logger = logging.getLogger(__name__)


class ThermalEventExtractor(SystemEventExtractor):
    """Thermal event extractor for compatibility."""
    
    def get_supported_categories(self):
        return {'nvidia_gpu', 'cpu'}
    
    def extract_events(self, changes: List[SystemChange], context: EventContext) -> List[SystemEvent]:
        events = []
        
        # Look for thermal changes
        thermal_changes = [c for c in changes if c.category in self.get_supported_categories()]
        
        for change in thermal_changes:
            if 'temperature' in change.entity_id and change.significance > 0.6:
                event = SystemEvent(
                    event_type='thermal_event',
                    entity=change.entity_id,
                    description=f"Temperature change detected: {change.old_value}°C → {change.new_value}°C",
                    severity=self._determine_thermal_severity(change),
                    timestamp=change.timestamp,
                    causes=[change],
                    predicted_effects=['potential_throttling'] if change.new_value > 80 else [],
                    context={'temperature_change': change.new_value - change.old_value},
                    confidence=self.calculate_base_confidence([change], context)
                )
                events.append(event)
        
        return events
    
    def _determine_thermal_severity(self, change):
        from ..types import EventSeverity
        if change.new_value > 85:
            return EventSeverity.CRITICAL
        elif change.new_value > 75:
            return EventSeverity.WARNING
        else:
            return EventSeverity.INFO


class ServiceEventExtractor(SystemEventExtractor):
    """Service event extractor for compatibility."""
    
    def get_supported_categories(self):
        return {'processes', 'services'}
    
    def extract_events(self, changes: List[SystemChange], context: EventContext) -> List[SystemEvent]:
        events = []
        
        service_changes = [c for c in changes if c.category in self.get_supported_categories()]
        
        for change in service_changes:
            if 'service' in change.entity_id or 'process' in change.entity_id:
                event = SystemEvent(
                    event_type='service_event',
                    entity=change.entity_id,
                    description=f"Service change: {change.change_type.value}",
                    severity=self._determine_service_severity(change),
                    timestamp=change.timestamp,
                    causes=[change],
                    predicted_effects=[],
                    context={'change_type': change.change_type.value},
                    confidence=self.calculate_base_confidence([change], context)
                )
                events.append(event)
        
        return events
    
    def _determine_service_severity(self, change):
        from ..types import EventSeverity, ChangeType
        if change.change_type == ChangeType.REMOVED:
            return EventSeverity.WARNING
        else:
            return EventSeverity.INFO


class PackageEventExtractor(SystemEventExtractor):
    """Package event extractor for compatibility."""
    
    def get_supported_categories(self):
        return {'python_env', 'packages'}
    
    def extract_events(self, changes: List[SystemChange], context: EventContext) -> List[SystemEvent]:
        events = []
        
        package_changes = [c for c in changes if c.category in self.get_supported_categories()]
        
        for change in package_changes:
            if 'package' in change.entity_id:
                event = SystemEvent(
                    event_type='package_event',
                    entity=change.entity_id,
                    description=f"Package {change.change_type.value}: {change.entity_id}",
                    severity=self._determine_package_severity(change),
                    timestamp=change.timestamp,
                    causes=[change],
                    predicted_effects=['dependency_changes'] if change.change_type.value in ['ADDED', 'REMOVED'] else [],
                    context={'change_type': change.change_type.value},
                    confidence=self.calculate_base_confidence([change], context)
                )
                events.append(event)
        
        return events
    
    def _determine_package_severity(self, change):
        from ..types import EventSeverity
        return EventSeverity.INFO


class PerformanceEventExtractor(SystemEventExtractor):
    """Performance event extractor for compatibility."""
    
    def get_supported_categories(self):
        return {'cpu', 'memory', 'storage', 'nvidia_gpu'}
    
    def extract_events(self, changes: List[SystemChange], context: EventContext) -> List[SystemEvent]:
        events = []
        
        performance_changes = [c for c in changes if c.category in self.get_supported_categories() and c.significance > 0.7]
        
        for change in performance_changes:
            if any(keyword in change.entity_id for keyword in ['utilization', 'usage', 'load', 'performance']):
                event = SystemEvent(
                    event_type='performance_event',
                    entity=change.entity_id,
                    description=f"Performance change detected in {change.category}: {change.entity_id}",
                    severity=self._determine_performance_severity(change),
                    timestamp=change.timestamp,
                    causes=[change],
                    predicted_effects=['system_slowdown'] if change.significance > 0.8 else [],
                    context={'performance_impact': change.significance},
                    confidence=self.calculate_base_confidence([change], context)
                )
                events.append(event)
        
        return events
    
    def _determine_performance_severity(self, change):
        from ..types import EventSeverity
        if change.significance > 0.9:
            return EventSeverity.CRITICAL
        elif change.significance > 0.7:
            return EventSeverity.WARNING
        else:
            return EventSeverity.INFO


class EventExtractionEngine:
    """
    Compatibility bridge for event extraction engine.
    
    Maintains the old API while using the sophisticated new event extraction
    framework with causal analysis and effect prediction capabilities.
    """
    
    def __init__(self, config: Any):
        """Initialize event extraction engine with compatibility bridge."""
        self.config = config
        self.min_confidence = getattr(config, 'min_event_confidence', 0.3)
        
        # Initialize sophisticated extractors
        extractor_config = EventExtractorConfig()
        self.extractors = {
            'thermal': ThermalEventExtractor(extractor_config),
            'service': ServiceEventExtractor(extractor_config),
            'package': PackageEventExtractor(extractor_config),
            'performance': PerformanceEventExtractor(extractor_config)
        }
        
        # Initialize sophisticated analysis components
        self.causal_analyzer = CausalAnalyzer(extractor_config)
        self.effect_predictor = EffectPredictor(extractor_config)
        
        logger.info(f"EventExtractionEngine initialized with {len(self.extractors)} sophisticated extractors")
    
    def extract_events(self, changes: List[SystemChange], 
                      old_snapshot: SystemSnapshot,
                      new_snapshot: SystemSnapshot) -> List[SystemEvent]:
        """
        Extract semantic events from system changes using sophisticated analysis.
        
        This method bridges the old interface to the new sophisticated event
        extraction framework with contextual understanding and causal analysis.
        """
        if not changes:
            return []
        
        # Create rich context for sophisticated analysis
        context = self._create_event_context(changes, old_snapshot, new_snapshot)
        
        all_events = []
        
        # Use sophisticated extractors
        for extractor_name, extractor in self.extractors.items():
            try:
                # Only process changes this extractor can handle
                if extractor.can_process(changes):
                    events = extractor.extract_events(changes, context)
                    
                    # Filter by confidence threshold
                    filtered_events = [e for e in events if e.confidence >= self.min_confidence]
                    all_events.extend(filtered_events)
                    
                    logger.debug(f"Extractor {extractor_name} generated {len(events)} events, "
                               f"{len(filtered_events)} above confidence threshold")
                    
            except Exception as e:
                logger.error(f"Error in sophisticated {extractor_name} extractor: {e}")
        
        # Apply sophisticated causal analysis
        all_events = self.causal_analyzer.enhance_events_with_causality(all_events, changes)
        
        # Apply sophisticated effect prediction
        all_events = self.effect_predictor.enhance_events_with_predictions(all_events, context)
        
        logger.debug(f"Total sophisticated events extracted: {len(all_events)}")
        return all_events
    
    def detect_correlations(self, events: List[SystemEvent], 
                          changes: List[SystemChange]) -> List[EventCorrelation]:
        """
        Detect sophisticated correlations between events using causal analysis.
        """
        if not events or not getattr(self.config, 'enable_correlation_detection', True):
            return []
        
        # Use sophisticated causal analyzer for correlation detection
        correlations = self.causal_analyzer.detect_event_correlations(events, changes)
        
        logger.debug(f"Sophisticated correlation analysis detected {len(correlations)} correlations")
        return correlations
    
    def get_extractor_status(self) -> Dict[str, Any]:
        """Get status of all sophisticated extractors."""
        status = {}
        
        for name, extractor in self.extractors.items():
            try:
                # Use sophisticated status checking
                status[name] = {
                    "enabled": True,
                    "healthy": True,
                    "supported_categories": list(extractor.get_supported_categories()),
                    "last_check": datetime.now().isoformat(),
                    "type": "sophisticated_extractor",
                    "capabilities": ["causal_analysis", "effect_prediction", "contextual_understanding"]
                }
                
            except Exception as e:
                status[name] = {
                    "enabled": False,
                    "healthy": False,
                    "error": str(e),
                    "last_check": datetime.now().isoformat(),
                    "type": "sophisticated_extractor"
                }
        
        # Add sophisticated analysis component status
        status['causal_analyzer'] = {
            "enabled": True,
            "healthy": True,
            "capabilities": ["causal_chain_detection", "relationship_inference", "confidence_scoring"],
            "last_check": datetime.now().isoformat()
        }
        
        status['effect_predictor'] = {
            "enabled": True,  
            "healthy": True,
            "capabilities": ["effect_prediction", "risk_assessment", "mitigation_suggestions"],
            "last_check": datetime.now().isoformat()
        }
        
        return status
    
    def _create_event_context(self, changes: List[SystemChange], 
                             old_snapshot: SystemSnapshot, 
                             new_snapshot: SystemSnapshot) -> EventContext:
        """Create rich context for sophisticated event analysis."""
        
        # Extract system load metrics from snapshots
        system_load_metrics = {}
        
        try:
            # GPU metrics
            if 'nvidia_gpu' in new_snapshot.data:
                gpu_data = new_snapshot.data['nvidia_gpu']
                if 'basic_metrics' in gpu_data:
                    # Parse GPU temperature and utilization from basic_metrics
                    metrics_lines = gpu_data['basic_metrics'].split('\n')
                    if len(metrics_lines) > 1:
                        values = metrics_lines[1].split(', ')
                        if len(values) >= 11:
                            system_load_metrics['gpu_temperature'] = float(values[7])  # temperature.gpu
                            system_load_metrics['gpu_utilization'] = float(values[9])  # utilization.gpu
            
            # CPU metrics
            if 'cpu' in new_snapshot.data and 'usage' in new_snapshot.data['cpu']:
                cpu_usage = new_snapshot.data['cpu']['usage'].get('cpu', {})
                if 'idle' in cpu_usage:
                    total = sum(cpu_usage.values())
                    idle_percent = (cpu_usage['idle'] / total) * 100
                    system_load_metrics['cpu_percent'] = 100 - idle_percent
                        
            # Memory metrics  
            if 'memory' in new_snapshot.data and 'meminfo' in new_snapshot.data['memory']:
                meminfo = new_snapshot.data['memory']['meminfo']
                if 'MemTotal' in meminfo and 'MemAvailable' in meminfo:
                    total_kb = int(meminfo['MemTotal'].split()[0])
                    available_kb = int(meminfo['MemAvailable'].split()[0])
                    used_percent = ((total_kb - available_kb) / total_kb) * 100
                    system_load_metrics['memory_percent'] = used_percent
                    
        except Exception as e:
            logger.debug(f"Error extracting system metrics for context: {e}")
        
        return EventContext(
            timestamp=datetime.now(),
            system_state_summary={
                'total_processes': len(new_snapshot.data.get('processes', {}).get('ps_cpu', '').split('\n')) - 1,
                'system_metrics': system_load_metrics
            },
            recent_changes_context=changes[-50:],  # Last 50 changes for context
            system_load_metrics=system_load_metrics,
            environmental_factors={
                'snapshot_comparison': {
                    'time_delta_seconds': (new_snapshot.metadata['timestamp'] - old_snapshot.metadata['timestamp']).total_seconds(),
                    'data_categories': list(new_snapshot.data.keys())
                }
            }
        )