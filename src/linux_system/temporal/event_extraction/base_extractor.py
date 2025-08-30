"""
Base Event Extraction Framework
================================

Abstract base classes and core infrastructure for extracting semantic events
from system changes. Provides the foundation for specialized event extractors
that understand domain-specific patterns and relationships.
"""

import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque

from ..types import SystemChange, SystemEvent, EventSeverity, SystemDelta
from ..config import EventExtractorConfig


@dataclass
class EventContext:
    """Rich contextual information for events."""
    timestamp: datetime
    system_state_summary: Dict[str, Any] = field(default_factory=dict)
    recent_changes_context: List[SystemChange] = field(default_factory=list)
    concurrent_events: List[str] = field(default_factory=list)
    system_load_metrics: Dict[str, float] = field(default_factory=dict)
    environmental_factors: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def system_stress_level(self) -> float:
        """Calculate overall system stress level (0.0 to 1.0)."""
        stress_factors = []
        
        # CPU stress
        cpu_usage = self.system_load_metrics.get('cpu_percent', 0)
        stress_factors.append(min(cpu_usage / 90, 1.0))
        
        # Memory stress  
        memory_usage = self.system_load_metrics.get('memory_percent', 0)
        stress_factors.append(min(memory_usage / 85, 1.0))
        
        # GPU stress
        gpu_temp = self.system_load_metrics.get('gpu_temperature', 0)
        if gpu_temp > 0:
            stress_factors.append(min((gpu_temp - 50) / 40, 1.0))
        
        # Change rate stress
        change_rate = len(self.recent_changes_context) / 10  # Normalize by 10 changes
        stress_factors.append(min(change_rate, 1.0))
        
        return sum(stress_factors) / len(stress_factors) if stress_factors else 0.0
    
    @property
    def is_high_activity_period(self) -> bool:
        """Check if system is in high activity period."""
        return len(self.recent_changes_context) > 20 or len(self.concurrent_events) > 5


@dataclass
class EventPattern:
    """Pattern definition for event recognition."""
    name: str
    description: str
    required_changes: List[Dict[str, Any]]
    optional_changes: List[Dict[str, Any]] = field(default_factory=list)
    time_window_seconds: int = 300  # 5 minutes default
    confidence_threshold: float = 0.7
    severity_mapping: Dict[str, EventSeverity] = field(default_factory=dict)
    
    def matches(self, changes: List[SystemChange], context: EventContext) -> Tuple[bool, float]:
        """Check if changes match this pattern and return confidence."""
        pattern_score = 0.0
        total_weight = 0.0
        
        # Check required changes
        required_weight = 2.0
        for required_change_spec in self.required_changes:
            if self._find_matching_change(changes, required_change_spec):
                pattern_score += required_weight
            total_weight += required_weight
        
        # Check optional changes (boost confidence)
        optional_weight = 1.0
        for optional_change_spec in self.optional_changes:
            if self._find_matching_change(changes, optional_change_spec):
                pattern_score += optional_weight
                total_weight += optional_weight
        
        confidence = pattern_score / total_weight if total_weight > 0 else 0.0
        matches = confidence >= self.confidence_threshold
        
        return matches, confidence
    
    def _find_matching_change(self, changes: List[SystemChange], 
                             change_spec: Dict[str, Any]) -> bool:
        """Find change matching specification."""
        for change in changes:
            match = True
            
            # Check category
            if 'category' in change_spec and change.category != change_spec['category']:
                match = False
            
            # Check change type
            if 'change_type' in change_spec and change.change_type != change_spec['change_type']:
                match = False
            
            # Check entity pattern
            if 'entity_pattern' in change_spec:
                if not re.search(change_spec['entity_pattern'], change.entity_id):
                    match = False
            
            # Check significance threshold
            if 'min_significance' in change_spec:
                if change.significance < change_spec['min_significance']:
                    match = False
            
            # Check metadata conditions
            if 'metadata_conditions' in change_spec:
                for key, expected_value in change_spec['metadata_conditions'].items():
                    if change.metadata.get(key) != expected_value:
                        match = False
            
            if match:
                return True
        
        return False


class EventExtractorRegistry:
    """Registry for specialized event extractors."""
    
    def __init__(self):
        self.extractors: Dict[str, 'SystemEventExtractor'] = {}
        self.patterns: Dict[str, List[EventPattern]] = defaultdict(list)
    
    def register_extractor(self, name: str, extractor: 'SystemEventExtractor') -> None:
        """Register an event extractor."""
        self.extractors[name] = extractor
    
    def register_pattern(self, category: str, pattern: EventPattern) -> None:
        """Register an event pattern for category."""
        self.patterns[category].append(pattern)
    
    def get_extractors_for_changes(self, changes: List[SystemChange]) -> List['SystemEventExtractor']:
        """Get relevant extractors for changes."""
        relevant_extractors = []
        categories = set(change.category for change in changes)
        
        for extractor in self.extractors.values():
            if any(category in extractor.supported_categories for category in categories):
                relevant_extractors.append(extractor)
        
        return relevant_extractors
    
    def get_patterns_for_category(self, category: str) -> List[EventPattern]:
        """Get patterns for specific category."""
        return self.patterns.get(category, [])


class SystemEventExtractor(ABC):
    """Abstract base for system event extractors."""
    
    def __init__(self, config: EventExtractorConfig = None):
        self.config = config or EventExtractorConfig()
        self.supported_categories: Set[str] = set()
        self.event_history: deque = deque(maxlen=1000)  # Keep recent event history
        
    @abstractmethod
    def extract_events(self, changes: List[SystemChange], 
                      context: EventContext) -> List[SystemEvent]:
        """Extract events from changes with context."""
        pass
    
    @abstractmethod
    def get_supported_categories(self) -> Set[str]:
        """Get categories this extractor supports."""
        pass
    
    def can_process(self, changes: List[SystemChange]) -> bool:
        """Check if extractor can process these changes."""
        change_categories = set(change.category for change in changes)
        return bool(change_categories & self.get_supported_categories())
    
    def calculate_base_confidence(self, changes: List[SystemChange], 
                                 context: EventContext) -> float:
        """Calculate base confidence for event extraction."""
        confidence = 0.5  # Base confidence
        
        # Higher confidence with more relevant changes
        relevant_changes = [c for c in changes if c.category in self.supported_categories]
        if len(relevant_changes) > 1:
            confidence += min(len(relevant_changes) * 0.1, 0.3)
        
        # Higher confidence with higher significance changes
        avg_significance = sum(c.significance for c in relevant_changes) / len(relevant_changes)
        confidence += avg_significance * 0.2
        
        # Context-based adjustments
        if context.is_high_activity_period:
            confidence += 0.1  # More likely to be meaningful during high activity
        
        # Temporal consistency (changes happening close together)
        if len(relevant_changes) > 1:
            timestamps = [c.timestamp for c in relevant_changes]
            time_span = max(timestamps) - min(timestamps)
            if time_span < timedelta(minutes=5):
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def determine_event_severity(self, changes: List[SystemChange], 
                               context: EventContext) -> EventSeverity:
        """Determine event severity based on changes and context."""
        # Start with change significance
        max_significance = max(c.significance for c in changes) if changes else 0.0
        
        # System stress amplifies severity
        stress_multiplier = 1.0 + context.system_stress_level
        effective_significance = min(max_significance * stress_multiplier, 1.0)
        
        # Map to severity levels
        if effective_significance >= 0.8:
            return EventSeverity.CRITICAL
        elif effective_significance >= 0.6:
            return EventSeverity.WARNING
        else:
            return EventSeverity.INFO
    
    def extract_contextual_metadata(self, changes: List[SystemChange], 
                                   context: EventContext) -> Dict[str, Any]:
        """Extract contextual metadata for event."""
        metadata = {
            'extraction_timestamp': datetime.now().isoformat(),
            'change_count': len(changes),
            'categories_involved': list(set(c.category for c in changes)),
            'avg_significance': sum(c.significance for c in changes) / len(changes) if changes else 0.0,
            'system_stress_level': context.system_stress_level,
            'concurrent_event_count': len(context.concurrent_events)
        }
        
        # Add category-specific aggregations
        category_stats = defaultdict(list)
        for change in changes:
            category_stats[change.category].append(change.significance)
        
        metadata['category_significance'] = {
            category: sum(significances) / len(significances)
            for category, significances in category_stats.items()
        }
        
        return metadata
    
    def predict_effects(self, changes: List[SystemChange], 
                       context: EventContext) -> List[str]:
        """Predict likely effects of this event."""
        effects = []
        
        # Base effects from changes
        for change in changes:
            if change.category == 'nvidia_gpu':
                if change.entity_id.endswith('temperature'):
                    effects.extend(['thermal_stress', 'potential_throttling'])
                elif 'memory' in change.entity_id:
                    effects.extend(['memory_pressure', 'performance_impact'])
            
            elif change.category == 'processes':
                if change.change_type.name == 'ADDED':
                    effects.extend(['resource_consumption_increase'])
                elif change.change_type.name == 'REMOVED':
                    effects.extend(['resource_availability_increase'])
            
            elif change.category == 'python_env':
                if 'framework' in str(change.metadata):
                    effects.extend(['capability_change', 'compatibility_impact'])
        
        # Context-based effects
        if context.system_stress_level > 0.7:
            effects.append('system_instability_risk')
        
        if context.is_high_activity_period:
            effects.append('performance_degradation_likely')
        
        return list(set(effects))  # Remove duplicates
    
    def enrich_event_description(self, event_type: str, changes: List[SystemChange], 
                               context: EventContext) -> str:
        """Create rich, descriptive event description."""
        if not changes:
            return f"System event: {event_type}"
        
        primary_category = max(set(c.category for c in changes), 
                             key=lambda cat: sum(c.significance for c in changes if c.category == cat))
        
        # Category-specific descriptions
        if primary_category == 'nvidia_gpu':
            return self._describe_gpu_event(event_type, changes, context)
        elif primary_category == 'processes':
            return self._describe_process_event(event_type, changes, context)
        elif primary_category == 'python_env':
            return self._describe_python_env_event(event_type, changes, context)
        else:
            return f"System {primary_category} event: {event_type} ({len(changes)} changes)"
    
    def _describe_gpu_event(self, event_type: str, changes: List[SystemChange], 
                          context: EventContext) -> str:
        """Describe GPU-related event."""
        gpu_changes = [c for c in changes if c.category == 'nvidia_gpu']
        
        thermal_changes = [c for c in gpu_changes if 'temperature' in c.entity_id]
        memory_changes = [c for c in gpu_changes if 'memory' in c.entity_id]
        process_changes = [c for c in gpu_changes if 'process' in c.entity_id]
        
        if thermal_changes and memory_changes:
            return f"GPU thermal and memory event: temperature and memory pressure changes detected"
        elif thermal_changes:
            temp_change = thermal_changes[0]
            return f"GPU thermal event: {event_type} (temperature: {temp_change.new_value})"
        elif memory_changes:
            return f"GPU memory event: {event_type} (memory pressure detected)"
        elif process_changes:
            return f"GPU process event: {event_type} ({len(process_changes)} process changes)"
        else:
            return f"GPU system event: {event_type}"
    
    def _describe_process_event(self, event_type: str, changes: List[SystemChange], 
                              context: EventContext) -> str:
        """Describe process-related event."""
        process_changes = [c for c in changes if c.category == 'processes']
        
        spawns = [c for c in process_changes if c.change_type.name == 'ADDED']
        terminations = [c for c in process_changes if c.change_type.name == 'REMOVED']
        
        if spawns and terminations:
            return f"Process lifecycle event: {len(spawns)} spawned, {len(terminations)} terminated"
        elif spawns:
            return f"Process spawn event: {len(spawns)} new processes started"
        elif terminations:
            return f"Process termination event: {len(terminations)} processes ended"
        else:
            return f"Process resource event: {event_type}"
    
    def _describe_python_env_event(self, event_type: str, changes: List[SystemChange], 
                                 context: EventContext) -> str:
        """Describe Python environment event."""
        python_changes = [c for c in changes if c.category == 'python_env']
        
        package_changes = [c for c in python_changes if 'package:' in c.entity_id]
        env_changes = [c for c in python_changes if 'virtual_env:' in c.entity_id]
        
        if package_changes and env_changes:
            return f"Python environment setup event: environment and package changes"
        elif package_changes:
            installs = [c for c in package_changes if c.change_type.name == 'ADDED']
            if installs:
                return f"Python package installation event: {len(installs)} packages installed"
            else:
                return f"Python package event: {len(package_changes)} package changes"
        elif env_changes:
            return f"Python virtual environment event: {event_type}"
        else:
            return f"Python environment event: {event_type}"
    
    def _log_event_extraction(self, event: SystemEvent, changes: List[SystemChange]) -> None:
        """Log event extraction for debugging/monitoring."""
        # Add to history for pattern learning
        self.event_history.append({
            'timestamp': event.timestamp,
            'event_type': event.event_type,
            'severity': event.severity.value,
            'confidence': event.confidence,
            'change_count': len(changes),
            'categories': list(set(c.category for c in changes))
        })


class PatternBasedEventExtractor(SystemEventExtractor):
    """Event extractor using predefined patterns."""
    
    def __init__(self, config: EventExtractorConfig = None):
        super().__init__(config)
        self.patterns: List[EventPattern] = []
    
    def add_pattern(self, pattern: EventPattern) -> None:
        """Add event pattern."""
        self.patterns.append(pattern)
    
    def extract_events(self, changes: List[SystemChange], 
                      context: EventContext) -> List[SystemEvent]:
        """Extract events using pattern matching."""
        events = []
        
        for pattern in self.patterns:
            matches, confidence = pattern.matches(changes, context)
            
            if matches and confidence >= self.config.min_event_confidence:
                # Create event from pattern
                event = self._create_event_from_pattern(pattern, changes, context, confidence)
                events.append(event)
        
        return events
    
    def _create_event_from_pattern(self, pattern: EventPattern, changes: List[SystemChange], 
                                 context: EventContext, confidence: float) -> SystemEvent:
        """Create event from matched pattern."""
        # Determine primary entity
        primary_change = max(changes, key=lambda c: c.significance)
        entity = self._extract_primary_entity(primary_change)
        
        # Determine severity
        severity = self.determine_event_severity(changes, context)
        
        # Override with pattern-specific severity if configured
        if str(severity) in pattern.severity_mapping:
            severity = pattern.severity_mapping[str(severity)]
        
        # Create event
        event = SystemEvent(
            event_type=pattern.name,
            entity=entity,
            description=self.enrich_event_description(pattern.name, changes, context),
            severity=severity,
            timestamp=context.timestamp,
            causes=changes,
            predicted_effects=self.predict_effects(changes, context),
            context=self.extract_contextual_metadata(changes, context),
            confidence=confidence
        )
        
        self._log_event_extraction(event, changes)
        return event
    
    def _extract_primary_entity(self, change: SystemChange) -> str:
        """Extract primary entity from change."""
        if change.category == 'nvidia_gpu':
            return 'rtx_5090'
        elif change.category == 'processes':
            # Extract process name or PID
            if ':' in change.entity_id:
                return change.entity_id.split(':')[1]
            return change.entity_id
        elif change.category == 'python_env':
            # Extract package name or environment name
            if 'package:' in change.entity_id:
                parts = change.entity_id.split(':')
                return parts[-1] if len(parts) > 2 else change.entity_id
            return change.entity_id
        else:
            return change.entity_id
    
    def get_supported_categories(self) -> Set[str]:
        """Get categories supported by patterns."""
        categories = set()
        for pattern in self.patterns:
            for required_change in pattern.required_changes:
                if 'category' in required_change:
                    categories.add(required_change['category'])
        return categories