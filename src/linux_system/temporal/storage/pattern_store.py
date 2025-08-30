"""
Pattern Store
=============

Long-term storage and analysis of system behavioral patterns.
Learns recurring patterns, anomalies, and system evolution trends.
"""

import json
import hashlib
import threading
from pathlib import Path
from datetime import datetime, timedelta, time
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field

from ..types import SystemDelta, SystemChange, SystemEvent


@dataclass
class SystemPattern:
    """Represents a learned system pattern."""
    pattern_id: str
    pattern_type: str  # 'temporal', 'causal', 'periodic', 'anomaly'
    name: str
    description: str
    
    # Pattern signature
    signature: Dict[str, Any]  # Unique pattern characteristics
    template: Dict[str, Any]   # Pattern template for matching
    
    # Statistics
    occurrence_count: int = 0
    confidence: float = 0.0
    last_seen: Optional[datetime] = None
    first_seen: Optional[datetime] = None
    
    # Context
    typical_conditions: Dict[str, Any] = field(default_factory=dict)
    associated_events: List[str] = field(default_factory=list)
    time_patterns: List[str] = field(default_factory=list)  # 'morning', 'evening', 'weekend', etc.
    
    # Evolution tracking
    pattern_stability: float = 1.0  # How consistent the pattern is
    recent_variations: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'name': self.name,
            'description': self.description,
            'signature': self.signature,
            'template': self.template,
            'occurrence_count': self.occurrence_count,
            'confidence': self.confidence,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'first_seen': self.first_seen.isoformat() if self.first_seen else None,
            'typical_conditions': self.typical_conditions,
            'associated_events': self.associated_events,
            'time_patterns': self.time_patterns,
            'pattern_stability': self.pattern_stability,
            'recent_variations': self.recent_variations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemPattern':
        """Create from dictionary."""
        return cls(
            pattern_id=data['pattern_id'],
            pattern_type=data['pattern_type'],
            name=data['name'],
            description=data['description'],
            signature=data['signature'],
            template=data['template'],
            occurrence_count=data.get('occurrence_count', 0),
            confidence=data.get('confidence', 0.0),
            last_seen=datetime.fromisoformat(data['last_seen']) if data.get('last_seen') else None,
            first_seen=datetime.fromisoformat(data['first_seen']) if data.get('first_seen') else None,
            typical_conditions=data.get('typical_conditions', {}),
            associated_events=data.get('associated_events', []),
            time_patterns=data.get('time_patterns', []),
            pattern_stability=data.get('pattern_stability', 1.0),
            recent_variations=data.get('recent_variations', [])
        )


class PatternStore:
    """
    Storage and analysis system for system behavioral patterns.
    
    Learns from system deltas to identify recurring patterns,
    anomalies, and evolutionary trends in system behavior.
    """
    
    def __init__(self, retention_months: int = 12, storage_path: Optional[Path] = None):
        self.retention_months = retention_months
        self.storage_path = storage_path
        self._patterns: Dict[str, SystemPattern] = {}
        self._pattern_occurrences: Dict[str, List[datetime]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # Pattern detection thresholds
        self.min_occurrences_for_pattern = 3
        self.confidence_threshold = 0.7
        self.stability_threshold = 0.8
        
        # Load existing patterns if storage enabled
        if self.storage_path:
            self._load_existing_patterns()
    
    def learn_from(self, system_delta: SystemDelta) -> None:
        """
        Learn patterns from system delta.
        
        Args:
            system_delta: System delta to analyze for patterns
        """
        with self._lock:
            # Extract potential patterns from this delta
            potential_patterns = self._extract_patterns(system_delta)
            
            for pattern_candidate in potential_patterns:
                self._update_or_create_pattern(pattern_candidate, system_delta.timestamp)
            
            # Periodically analyze for new patterns
            if len(self._pattern_occurrences) % 100 == 0:  # Every 100 deltas
                self._detect_new_patterns()
            
            # Clean up old data
            self._cleanup_old_data()
    
    def _extract_patterns(self, system_delta: SystemDelta) -> List[Dict[str, Any]]:
        """Extract potential patterns from system delta."""
        patterns = []
        
        # Temporal patterns (time-based recurring activities)
        temporal_pattern = self._extract_temporal_pattern(system_delta)
        if temporal_pattern:
            patterns.append(temporal_pattern)
        
        # Event sequence patterns
        sequence_patterns = self._extract_sequence_patterns(system_delta)
        patterns.extend(sequence_patterns)
        
        # Category activity patterns
        category_patterns = self._extract_category_patterns(system_delta)
        patterns.extend(category_patterns)
        
        # Thermal patterns
        thermal_patterns = self._extract_thermal_patterns(system_delta)
        patterns.extend(thermal_patterns)
        
        # ML workload patterns
        ml_patterns = self._extract_ml_patterns(system_delta)
        patterns.extend(ml_patterns)
        
        return patterns
    
    def _extract_temporal_pattern(self, system_delta: SystemDelta) -> Optional[Dict[str, Any]]:
        """Extract temporal patterns (daily, weekly rhythms)."""
        timestamp = system_delta.timestamp
        
        # Skip if no significant activity
        if not system_delta.semantic_events or len(system_delta.raw_delta) < 2:
            return None
        
        hour = timestamp.hour
        weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
        
        # Categorize time periods
        time_category = None
        if 6 <= hour <= 11:
            time_category = 'morning_activity'
        elif 12 <= hour <= 17:
            time_category = 'afternoon_activity'
        elif 18 <= hour <= 23:
            time_category = 'evening_activity'
        elif 0 <= hour <= 5:
            time_category = 'night_activity'
        
        if time_category:
            event_types = [e.event_type for e in system_delta.semantic_events]
            categories = list(set(c.category for c in system_delta.raw_delta))
            
            pattern_signature = {
                'time_category': time_category,
                'hour': hour,
                'weekday': weekday,
                'event_types': sorted(set(event_types)),
                'categories': sorted(categories)
            }
            
            return {
                'pattern_type': 'temporal',
                'name': f"{time_category}_{weekday}",
                'signature': pattern_signature,
                'timestamp': timestamp
            }
        
        return None
    
    def _extract_sequence_patterns(self, system_delta: SystemDelta) -> List[Dict[str, Any]]:
        """Extract event sequence patterns."""
        patterns = []
        
        events = system_delta.semantic_events
        if len(events) < 2:
            return patterns
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        # Look for meaningful sequences
        for i in range(len(sorted_events) - 1):
            event1 = sorted_events[i]
            event2 = sorted_events[i + 1]
            
            # Check if events are close in time (within 5 minutes)
            time_diff = event2.timestamp - event1.timestamp
            if time_diff <= timedelta(minutes=5):
                sequence_signature = {
                    'event1_type': event1.event_type,
                    'event2_type': event2.event_type,
                    'time_diff_seconds': int(time_diff.total_seconds()),
                    'entity1': event1.entity,
                    'entity2': event2.entity
                }
                
                patterns.append({
                    'pattern_type': 'causal',
                    'name': f"{event1.event_type}_to_{event2.event_type}",
                    'signature': sequence_signature,
                    'timestamp': system_delta.timestamp
                })
        
        return patterns
    
    def _extract_category_patterns(self, system_delta: SystemDelta) -> List[Dict[str, Any]]:
        """Extract category activity patterns."""
        patterns = []
        
        # Group changes by category
        category_counts = Counter(c.category for c in system_delta.raw_delta)
        
        # Look for categories with significant activity
        for category, count in category_counts.items():
            if count >= 3:  # Significant activity in category
                # Calculate average significance
                category_changes = [c for c in system_delta.raw_delta if c.category == category]
                avg_significance = sum(c.significance for c in category_changes) / len(category_changes)
                
                if avg_significance > 0.5:  # Meaningful changes
                    change_types = [c.change_type.value for c in category_changes]
                    
                    patterns.append({
                        'pattern_type': 'category_activity',
                        'name': f"{category}_high_activity",
                        'signature': {
                            'category': category,
                            'change_count': count,
                            'avg_significance': avg_significance,
                            'change_types': sorted(set(change_types))
                        },
                        'timestamp': system_delta.timestamp
                    })
        
        return patterns
    
    def _extract_thermal_patterns(self, system_delta: SystemDelta) -> List[Dict[str, Any]]:
        """Extract GPU thermal patterns."""
        patterns = []
        
        # Look for thermal events
        thermal_events = [
            e for e in system_delta.semantic_events 
            if 'thermal' in e.event_type.lower()
        ]
        
        if thermal_events:
            # Look for associated changes
            gpu_changes = [c for c in system_delta.raw_delta if c.category == 'nvidia_gpu']
            process_changes = [c for c in system_delta.raw_delta if c.category == 'processes']
            
            pattern_signature = {
                'thermal_event_count': len(thermal_events),
                'gpu_change_count': len(gpu_changes),
                'process_change_count': len(process_changes),
                'hour': system_delta.timestamp.hour
            }
            
            patterns.append({
                'pattern_type': 'thermal',
                'name': 'thermal_activity_pattern',
                'signature': pattern_signature,
                'timestamp': system_delta.timestamp
            })
        
        return patterns
    
    def _extract_ml_patterns(self, system_delta: SystemDelta) -> List[Dict[str, Any]]:
        """Extract ML/AI workload patterns."""
        patterns = []
        
        # Look for ML-related activity
        ml_indicators = 0
        
        # Check for ML frameworks in process changes
        for change in system_delta.raw_delta:
            if change.category == 'python_env':
                if change.metadata.get('is_ml_framework', False):
                    ml_indicators += 1
            elif change.category == 'processes':
                if change.metadata.get('is_ml_framework', False):
                    ml_indicators += 1
        
        # Check for GPU thermal events (often associated with ML training)
        gpu_thermal_events = [
            e for e in system_delta.semantic_events
            if 'gpu' in e.event_type.lower() and 'thermal' in e.event_type.lower()
        ]
        
        if gpu_thermal_events:
            ml_indicators += len(gpu_thermal_events)
        
        # Check for memory pressure events
        memory_events = [
            e for e in system_delta.semantic_events
            if 'memory' in e.event_type.lower()
        ]
        
        if memory_events:
            ml_indicators += len(memory_events)
        
        if ml_indicators >= 2:  # Threshold for ML activity
            patterns.append({
                'pattern_type': 'ml_workload',
                'name': 'ml_training_session',
                'signature': {
                    'ml_indicators': ml_indicators,
                    'thermal_events': len(gpu_thermal_events),
                    'memory_events': len(memory_events),
                    'hour': system_delta.timestamp.hour,
                    'weekday': system_delta.timestamp.weekday()
                },
                'timestamp': system_delta.timestamp
            })
        
        return patterns
    
    def _update_or_create_pattern(self, pattern_candidate: Dict[str, Any], timestamp: datetime) -> None:
        """Update existing pattern or create new one."""
        # Create pattern ID from signature
        pattern_id = self._create_pattern_id(pattern_candidate['signature'])
        
        # Record occurrence
        self._pattern_occurrences[pattern_id].append(timestamp)
        
        if pattern_id in self._patterns:
            # Update existing pattern
            pattern = self._patterns[pattern_id]
            pattern.occurrence_count += 1
            pattern.last_seen = timestamp
            
            # Update confidence based on frequency
            pattern.confidence = self._calculate_pattern_confidence(pattern_id)
            
            # Update pattern stability
            pattern.pattern_stability = self._calculate_pattern_stability(pattern_id)
            
        else:
            # Check if we have enough occurrences to create pattern
            occurrence_count = len(self._pattern_occurrences[pattern_id])
            
            if occurrence_count >= self.min_occurrences_for_pattern:
                # Create new pattern
                pattern = SystemPattern(
                    pattern_id=pattern_id,
                    pattern_type=pattern_candidate['pattern_type'],
                    name=pattern_candidate['name'],
                    description=self._generate_pattern_description(pattern_candidate),
                    signature=pattern_candidate['signature'],
                    template=self._create_pattern_template(pattern_candidate),
                    occurrence_count=occurrence_count,
                    confidence=self._calculate_pattern_confidence(pattern_id),
                    first_seen=min(self._pattern_occurrences[pattern_id]),
                    last_seen=max(self._pattern_occurrences[pattern_id]),
                    typical_conditions=self._analyze_typical_conditions(pattern_id),
                    time_patterns=self._analyze_time_patterns(pattern_id)
                )
                
                self._patterns[pattern_id] = pattern
    
    def _create_pattern_id(self, signature: Dict[str, Any]) -> str:
        """Create unique pattern ID from signature."""
        # Sort signature keys for consistent hashing
        sorted_signature = json.dumps(signature, sort_keys=True)
        return hashlib.md5(sorted_signature.encode()).hexdigest()[:12]
    
    def _calculate_pattern_confidence(self, pattern_id: str) -> float:
        """Calculate pattern confidence based on occurrence frequency and regularity."""
        occurrences = self._pattern_occurrences[pattern_id]
        if len(occurrences) < 2:
            return 0.0
        
        # Base confidence from occurrence count
        base_confidence = min(len(occurrences) / 10, 0.7)  # Max 0.7 from count
        
        # Regularity bonus (consistent timing)
        regularity_bonus = self._calculate_regularity_bonus(occurrences)
        
        return min(base_confidence + regularity_bonus, 1.0)
    
    def _calculate_regularity_bonus(self, occurrences: List[datetime]) -> float:
        """Calculate bonus based on regularity of occurrences."""
        if len(occurrences) < 3:
            return 0.0
        
        # Calculate time intervals between occurrences
        intervals = []
        for i in range(1, len(occurrences)):
            interval = occurrences[i] - occurrences[i-1]
            intervals.append(interval.total_seconds())
        
        if not intervals:
            return 0.0
        
        # Calculate coefficient of variation (stability measure)
        mean_interval = sum(intervals) / len(intervals)
        if mean_interval == 0:
            return 0.0
        
        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = variance ** 0.5
        cv = std_dev / mean_interval
        
        # Convert to regularity bonus (lower CV = higher bonus)
        regularity_bonus = max(0, 0.3 * (1 - min(cv, 1)))
        
        return regularity_bonus
    
    def _calculate_pattern_stability(self, pattern_id: str) -> float:
        """Calculate how stable/consistent the pattern is."""
        occurrences = self._pattern_occurrences[pattern_id]
        
        if len(occurrences) < 3:
            return 1.0  # New patterns are considered stable initially
        
        # Analyze recent occurrences vs overall pattern
        recent_count = 5
        if len(occurrences) > recent_count:
            recent_occurrences = occurrences[-recent_count:]
            
            # Calculate recent regularity vs overall regularity
            recent_regularity = self._calculate_regularity_bonus(recent_occurrences)
            overall_regularity = self._calculate_regularity_bonus(occurrences)
            
            # Stability is how close recent behavior is to overall pattern
            if overall_regularity > 0:
                stability = 1 - abs(recent_regularity - overall_regularity) / overall_regularity
            else:
                stability = 1.0
            
            return max(0.0, min(1.0, stability))
        
        return 1.0
    
    def _generate_pattern_description(self, pattern_candidate: Dict[str, Any]) -> str:
        """Generate human-readable pattern description."""
        pattern_type = pattern_candidate['pattern_type']
        signature = pattern_candidate['signature']
        
        if pattern_type == 'temporal':
            time_cat = signature.get('time_category', 'unknown')
            weekday = signature.get('weekday', 0)
            weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_name = weekday_names[weekday] if 0 <= weekday <= 6 else 'Unknown'
            
            return f"Regular {time_cat.replace('_', ' ')} on {day_name}s"
        
        elif pattern_type == 'causal':
            event1 = signature.get('event1_type', 'event')
            event2 = signature.get('event2_type', 'event')
            return f"Causal sequence: {event1} typically leads to {event2}"
        
        elif pattern_type == 'category_activity':
            category = signature.get('category', 'unknown')
            return f"High activity periods in {category} category"
        
        elif pattern_type == 'thermal':
            return "GPU thermal activity pattern during intensive workloads"
        
        elif pattern_type == 'ml_workload':
            return "Machine learning training session pattern"
        
        return f"System pattern of type {pattern_type}"
    
    def _create_pattern_template(self, pattern_candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Create pattern template for matching future occurrences."""
        signature = pattern_candidate['signature']
        template = {}
        
        # Create fuzzy matching template
        for key, value in signature.items():
            if isinstance(value, (int, float)):
                # For numeric values, create ranges
                template[key] = {
                    'type': 'numeric',
                    'value': value,
                    'tolerance': abs(value * 0.2)  # 20% tolerance
                }
            elif isinstance(value, str):
                template[key] = {
                    'type': 'exact',
                    'value': value
                }
            elif isinstance(value, list):
                template[key] = {
                    'type': 'list_overlap',
                    'values': value,
                    'min_overlap': max(1, len(value) // 2)  # At least half overlap
                }
            else:
                template[key] = {
                    'type': 'exact',
                    'value': value
                }
        
        return template
    
    def _analyze_typical_conditions(self, pattern_id: str) -> Dict[str, Any]:
        """Analyze typical conditions when pattern occurs."""
        occurrences = self._pattern_occurrences[pattern_id]
        
        conditions = {
            'typical_hours': [],
            'typical_weekdays': [],
            'frequency': len(occurrences)
        }
        
        for occurrence in occurrences:
            conditions['typical_hours'].append(occurrence.hour)
            conditions['typical_weekdays'].append(occurrence.weekday())
        
        # Calculate most common hours and weekdays
        if conditions['typical_hours']:
            hour_counts = Counter(conditions['typical_hours'])
            conditions['most_common_hour'] = hour_counts.most_common(1)[0][0]
        
        if conditions['typical_weekdays']:
            weekday_counts = Counter(conditions['typical_weekdays'])
            conditions['most_common_weekday'] = weekday_counts.most_common(1)[0][0]
        
        return conditions
    
    def _analyze_time_patterns(self, pattern_id: str) -> List[str]:
        """Analyze time-based patterns."""
        occurrences = self._pattern_occurrences[pattern_id]
        time_patterns = []
        
        hours = [occ.hour for occ in occurrences]
        weekdays = [occ.weekday() for occ in occurrences]
        
        # Analyze hour patterns
        hour_counts = Counter(hours)
        most_common_hours = [h for h, c in hour_counts.most_common(3)]
        
        if any(6 <= h <= 11 for h in most_common_hours):
            time_patterns.append('morning_pattern')
        if any(12 <= h <= 17 for h in most_common_hours):
            time_patterns.append('afternoon_pattern')
        if any(18 <= h <= 23 for h in most_common_hours):
            time_patterns.append('evening_pattern')
        
        # Analyze weekday patterns
        weekday_counts = Counter(weekdays)
        weekend_occurrences = sum(weekday_counts[d] for d in [5, 6])  # Saturday, Sunday
        weekday_occurrences = sum(weekday_counts[d] for d in [0, 1, 2, 3, 4])  # Mon-Fri
        
        if weekend_occurrences > weekday_occurrences:
            time_patterns.append('weekend_pattern')
        elif weekday_occurrences > weekend_occurrences * 2:
            time_patterns.append('weekday_pattern')
        
        return time_patterns
    
    def _detect_new_patterns(self) -> None:
        """Detect new patterns from accumulated occurrence data."""
        # This would involve more sophisticated pattern detection algorithms
        # For now, patterns are detected incrementally as they occur
        pass
    
    def get_pattern(self, pattern_id: str) -> Optional[SystemPattern]:
        """Get specific pattern by ID."""
        with self._lock:
            return self._patterns.get(pattern_id)
    
    def get_patterns_by_type(self, pattern_type: str) -> List[SystemPattern]:
        """Get all patterns of specific type."""
        with self._lock:
            return [p for p in self._patterns.values() if p.pattern_type == pattern_type]
    
    def get_confident_patterns(self, min_confidence: float = None) -> List[SystemPattern]:
        """Get patterns above confidence threshold."""
        threshold = min_confidence or self.confidence_threshold
        
        with self._lock:
            return [p for p in self._patterns.values() if p.confidence >= threshold]
    
    def analyze_pattern(self, pattern_type: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze patterns of specific type within time range."""
        patterns = self.get_patterns_by_type(pattern_type)
        
        analysis = {
            'pattern_count': len(patterns),
            'active_patterns': 0,
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'most_frequent': None,
            'recent_activity': {}
        }
        
        for pattern in patterns:
            # Check if pattern was active in time range
            pattern_occurrences = [
                occ for occ in self._pattern_occurrences.get(pattern.pattern_id, [])
                if start_time <= occ <= end_time
            ]
            
            if pattern_occurrences:
                analysis['active_patterns'] += 1
                analysis['recent_activity'][pattern.pattern_id] = len(pattern_occurrences)
            
            # Confidence distribution
            if pattern.confidence >= 0.8:
                analysis['confidence_distribution']['high'] += 1
            elif pattern.confidence >= 0.5:
                analysis['confidence_distribution']['medium'] += 1
            else:
                analysis['confidence_distribution']['low'] += 1
        
        # Find most frequent pattern
        if patterns:
            most_frequent = max(patterns, key=lambda p: p.occurrence_count)
            analysis['most_frequent'] = {
                'pattern_id': most_frequent.pattern_id,
                'name': most_frequent.name,
                'occurrence_count': most_frequent.occurrence_count,
                'confidence': most_frequent.confidence
            }
        
        return analysis
    
    def extract_new_patterns(self) -> Dict[str, Any]:
        """Extract new patterns from recent data."""
        with self._lock:
            results = {
                'new_patterns': [],
                'updated_patterns': [],
                'total_patterns': len(self._patterns)
            }
            
            # This would implement more sophisticated pattern extraction
            # For now, return current state
            results['confident_patterns'] = len(self.get_confident_patterns())
            
            return results
    
    def get_pattern_count(self) -> int:
        """Get total number of stored patterns."""
        with self._lock:
            return len(self._patterns)
    
    def clear(self) -> None:
        """Clear all patterns."""
        with self._lock:
            self._patterns.clear()
            self._pattern_occurrences.clear()
    
    def _cleanup_old_data(self) -> None:
        """Clean up patterns and occurrences older than retention period."""
        cutoff_time = datetime.now() - timedelta(days=self.retention_months * 30)
        
        # Clean up old occurrences
        for pattern_id in list(self._pattern_occurrences.keys()):
            occurrences = self._pattern_occurrences[pattern_id]
            recent_occurrences = [occ for occ in occurrences if occ >= cutoff_time]
            
            if recent_occurrences:
                self._pattern_occurrences[pattern_id] = recent_occurrences
            else:
                # Remove pattern with no recent occurrences
                del self._pattern_occurrences[pattern_id]
                if pattern_id in self._patterns:
                    del self._patterns[pattern_id]
    
    def persist(self) -> None:
        """Persist patterns to disk."""
        if not self.storage_path:
            return
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            # Save patterns
            patterns_file = self.storage_path / 'patterns.json'
            with open(patterns_file, 'w') as f:
                patterns_data = {
                    pattern_id: pattern.to_dict()
                    for pattern_id, pattern in self._patterns.items()
                }
                json.dump(patterns_data, f, indent=2)
            
            # Save occurrences
            occurrences_file = self.storage_path / 'occurrences.json'
            with open(occurrences_file, 'w') as f:
                occurrences_data = {
                    pattern_id: [occ.isoformat() for occ in occurrences]
                    for pattern_id, occurrences in self._pattern_occurrences.items()
                }
                json.dump(occurrences_data, f, indent=2)
    
    def _load_existing_patterns(self) -> None:
        """Load existing patterns from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        # Load patterns
        patterns_file = self.storage_path / 'patterns.json'
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                
                for pattern_id, pattern_dict in patterns_data.items():
                    pattern = SystemPattern.from_dict(pattern_dict)
                    self._patterns[pattern_id] = pattern
                    
            except Exception as e:
                print(f"Error loading patterns: {e}")
        
        # Load occurrences
        occurrences_file = self.storage_path / 'occurrences.json'
        if occurrences_file.exists():
            try:
                with open(occurrences_file, 'r') as f:
                    occurrences_data = json.load(f)
                
                for pattern_id, occurrence_strings in occurrences_data.items():
                    occurrences = [datetime.fromisoformat(occ) for occ in occurrence_strings]
                    self._pattern_occurrences[pattern_id] = occurrences
                    
            except Exception as e:
                print(f"Error loading pattern occurrences: {e}")
    
    def backup_to(self, backup_path: Path) -> None:
        """Backup patterns to specified path."""
        backup_path.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            # Backup patterns
            patterns_file = backup_path / 'patterns.json'
            with open(patterns_file, 'w') as f:
                patterns_data = {
                    pattern_id: pattern.to_dict()
                    for pattern_id, pattern in self._patterns.items()
                }
                json.dump(patterns_data, f, indent=2)
            
            # Backup occurrences
            occurrences_file = backup_path / 'occurrences.json'
            with open(occurrences_file, 'w') as f:
                occurrences_data = {
                    pattern_id: [occ.isoformat() for occ in occurrences]
                    for pattern_id, occurrences in self._pattern_occurrences.items()
                }
                json.dump(occurrences_data, f, indent=2)
    
    def restore_from(self, backup_path: Path) -> None:
        """Restore patterns from backup path."""
        if not backup_path.exists():
            return
        
        with self._lock:
            self.clear()
            
            # Restore patterns
            patterns_file = backup_path / 'patterns.json'
            if patterns_file.exists():
                try:
                    with open(patterns_file, 'r') as f:
                        patterns_data = json.load(f)
                    
                    for pattern_id, pattern_dict in patterns_data.items():
                        pattern = SystemPattern.from_dict(pattern_dict)
                        self._patterns[pattern_id] = pattern
                        
                except Exception as e:
                    print(f"Error restoring patterns: {e}")
            
            # Restore occurrences
            occurrences_file = backup_path / 'occurrences.json'
            if occurrences_file.exists():
                try:
                    with open(occurrences_file, 'r') as f:
                        occurrences_data = json.load(f)
                    
                    for pattern_id, occurrence_strings in occurrences_data.items():
                        occurrences = [datetime.fromisoformat(occ) for occ in occurrence_strings]
                        self._pattern_occurrences[pattern_id] = occurrences
                        
                except Exception as e:
                    print(f"Error restoring pattern occurrences: {e}")