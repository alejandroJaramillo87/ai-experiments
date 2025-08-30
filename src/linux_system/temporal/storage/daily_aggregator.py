"""
Daily Aggregator
================

Aggregates temporal data into daily summaries for efficient long-term storage.
Compresses detailed system deltas while preserving key patterns and events.
"""

import json
import threading
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field

from ..types import SystemDelta, SystemChange, SystemEvent, EventSeverity


@dataclass
class DailySummary:
    """Summary of system activity for a single day."""
    date: date
    total_deltas: int
    total_events: int
    total_changes: int
    
    # Category statistics
    categories_active: Set[str] = field(default_factory=set)
    category_change_counts: Dict[str, int] = field(default_factory=dict)
    category_significance: Dict[str, float] = field(default_factory=dict)
    
    # Event statistics
    event_types: Dict[str, int] = field(default_factory=dict)
    severity_counts: Dict[str, int] = field(default_factory=dict)
    high_confidence_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Significant moments
    peak_activity_hour: Optional[int] = None
    most_significant_changes: List[Dict[str, Any]] = field(default_factory=list)
    notable_patterns: List[str] = field(default_factory=list)
    
    # System health indicators
    avg_system_stress: float = 0.0
    thermal_events: int = 0
    memory_pressure_events: int = 0
    process_churn: int = 0  # Processes spawned + terminated
    package_changes: int = 0
    
    # Time patterns
    hourly_activity: Dict[int, int] = field(default_factory=dict)  # hour -> event count
    activity_periods: List[Dict[str, Any]] = field(default_factory=list)  # High activity periods
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'date': self.date.isoformat(),
            'total_deltas': self.total_deltas,
            'total_events': self.total_events,
            'total_changes': self.total_changes,
            'categories_active': list(self.categories_active),
            'category_change_counts': self.category_change_counts,
            'category_significance': self.category_significance,
            'event_types': self.event_types,
            'severity_counts': self.severity_counts,
            'high_confidence_events': self.high_confidence_events,
            'peak_activity_hour': self.peak_activity_hour,
            'most_significant_changes': self.most_significant_changes,
            'notable_patterns': self.notable_patterns,
            'avg_system_stress': self.avg_system_stress,
            'thermal_events': self.thermal_events,
            'memory_pressure_events': self.memory_pressure_events,
            'process_churn': self.process_churn,
            'package_changes': self.package_changes,
            'hourly_activity': self.hourly_activity,
            'activity_periods': self.activity_periods
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DailySummary':
        """Create from dictionary."""
        return cls(
            date=date.fromisoformat(data['date']),
            total_deltas=data['total_deltas'],
            total_events=data['total_events'],
            total_changes=data['total_changes'],
            categories_active=set(data.get('categories_active', [])),
            category_change_counts=data.get('category_change_counts', {}),
            category_significance=data.get('category_significance', {}),
            event_types=data.get('event_types', {}),
            severity_counts=data.get('severity_counts', {}),
            high_confidence_events=data.get('high_confidence_events', []),
            peak_activity_hour=data.get('peak_activity_hour'),
            most_significant_changes=data.get('most_significant_changes', []),
            notable_patterns=data.get('notable_patterns', []),
            avg_system_stress=data.get('avg_system_stress', 0.0),
            thermal_events=data.get('thermal_events', 0),
            memory_pressure_events=data.get('memory_pressure_events', 0),
            process_churn=data.get('process_churn', 0),
            package_changes=data.get('package_changes', 0),
            hourly_activity=data.get('hourly_activity', {}),
            activity_periods=data.get('activity_periods', [])
        )


class DailyAggregator:
    """
    Aggregates system deltas into daily summaries.
    
    Provides efficient storage and retrieval of historical system patterns
    while maintaining key insights and trends.
    """
    
    def __init__(self, retention_days: int = 90, storage_path: Optional[Path] = None):
        self.retention_days = retention_days
        self.storage_path = storage_path
        self._summaries: Dict[date, DailySummary] = {}
        self._pending_deltas: Dict[date, List[SystemDelta]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # Load existing summaries if storage path provided
        if self.storage_path:
            self._load_existing_summaries()
    
    def update(self, system_delta: SystemDelta) -> None:
        """
        Update daily aggregation with new system delta.
        
        Args:
            system_delta: New system delta to process
        """
        delta_date = system_delta.timestamp.date()
        
        with self._lock:
            # Add to pending deltas for this date
            self._pending_deltas[delta_date].append(system_delta)
            
            # If this is from a previous day, trigger aggregation
            if delta_date < datetime.now().date():
                self._aggregate_date(delta_date)
            
            # Clean up old pending data
            self._cleanup_old_pending()
    
    def _aggregate_date(self, target_date: date) -> None:
        """Aggregate all deltas for a specific date."""
        if target_date not in self._pending_deltas:
            return
        
        deltas = self._pending_deltas[target_date]
        if not deltas:
            return
        
        # Create or update summary for this date
        summary = self._create_daily_summary(target_date, deltas)
        self._summaries[target_date] = summary
        
        # Clear pending deltas for this date
        del self._pending_deltas[target_date]
        
        # Persist if storage enabled
        if self.storage_path:
            self._save_summary(summary)
        
        # Clean up old summaries
        self._cleanup_old_summaries()
    
    def _create_daily_summary(self, target_date: date, deltas: List[SystemDelta]) -> DailySummary:
        """Create comprehensive daily summary from deltas."""
        # Initialize counters
        total_events = 0
        total_changes = 0
        category_changes = defaultdict(int)
        category_significance = defaultdict(list)
        event_types = defaultdict(int)
        severity_counts = defaultdict(int)
        hourly_activity = defaultdict(int)
        
        high_confidence_events = []
        significant_changes = []
        thermal_events = 0
        memory_pressure_events = 0
        process_churn = 0
        package_changes = 0
        
        system_stress_values = []
        
        # Process all deltas for this date
        for delta in deltas:
            total_changes += len(delta.raw_delta)
            total_events += len(delta.semantic_events)
            
            hour = delta.timestamp.hour
            hourly_activity[hour] += len(delta.semantic_events) + len(delta.raw_delta)
            
            # Calculate system stress for this delta
            stress = self._calculate_system_stress(delta)
            system_stress_values.append(stress)
            
            # Process changes
            for change in delta.raw_delta:
                category_changes[change.category] += 1
                category_significance[change.category].append(change.significance)
                
                # Track specific change types
                if change.category == 'processes':
                    if change.change_type.value in ['ADDED', 'REMOVED']:
                        process_churn += 1
                elif change.category == 'python_env':
                    package_changes += 1
                
                # Collect highly significant changes
                if change.significance >= 0.8:
                    significant_changes.append({
                        'timestamp': change.timestamp.isoformat(),
                        'category': change.category,
                        'entity_id': change.entity_id,
                        'change_type': change.change_type.value,
                        'significance': change.significance,
                        'description': self._describe_change(change)
                    })
            
            # Process events
            for event in delta.semantic_events:
                event_types[event.event_type] += 1
                severity_counts[event.severity.value] += 1
                
                # Track specific event types
                if event.event_type == 'gpu_thermal_event':
                    thermal_events += 1
                elif 'memory_pressure' in event.event_type:
                    memory_pressure_events += 1
                
                # Collect high confidence events
                if event.confidence >= 0.8:
                    high_confidence_events.append({
                        'timestamp': event.timestamp.isoformat(),
                        'event_type': event.event_type,
                        'entity': event.entity,
                        'description': event.description,
                        'severity': event.severity.value,
                        'confidence': event.confidence,
                        'predicted_effects': event.predicted_effects
                    })
        
        # Calculate aggregated metrics
        categories_active = set(category_changes.keys())
        
        # Average significance per category
        avg_category_significance = {}
        for category, significances in category_significance.items():
            avg_category_significance[category] = sum(significances) / len(significances)
        
        # Find peak activity hour
        peak_activity_hour = max(hourly_activity.keys(), key=hourly_activity.get) if hourly_activity else None
        
        # Calculate average system stress
        avg_system_stress = sum(system_stress_values) / len(system_stress_values) if system_stress_values else 0.0
        
        # Detect activity periods (consecutive hours with high activity)
        activity_periods = self._detect_activity_periods(hourly_activity)
        
        # Detect notable patterns
        notable_patterns = self._detect_daily_patterns(deltas)
        
        # Keep only top significant changes
        significant_changes.sort(key=lambda x: x['significance'], reverse=True)
        most_significant_changes = significant_changes[:10]  # Top 10
        
        # Keep only top confidence events
        high_confidence_events.sort(key=lambda x: x['confidence'], reverse=True)
        high_confidence_events = high_confidence_events[:20]  # Top 20
        
        return DailySummary(
            date=target_date,
            total_deltas=len(deltas),
            total_events=total_events,
            total_changes=total_changes,
            categories_active=categories_active,
            category_change_counts=dict(category_changes),
            category_significance=avg_category_significance,
            event_types=dict(event_types),
            severity_counts=dict(severity_counts),
            high_confidence_events=high_confidence_events,
            peak_activity_hour=peak_activity_hour,
            most_significant_changes=most_significant_changes,
            notable_patterns=notable_patterns,
            avg_system_stress=avg_system_stress,
            thermal_events=thermal_events,
            memory_pressure_events=memory_pressure_events,
            process_churn=process_churn,
            package_changes=package_changes,
            hourly_activity=dict(hourly_activity),
            activity_periods=activity_periods
        )
    
    def _calculate_system_stress(self, delta: SystemDelta) -> float:
        """Calculate system stress level for a delta."""
        stress_factors = []
        
        # High significance changes indicate stress
        if delta.raw_delta:
            avg_significance = sum(c.significance for c in delta.raw_delta) / len(delta.raw_delta)
            stress_factors.append(avg_significance)
        
        # High number of changes indicates stress
        change_count_stress = min(len(delta.raw_delta) / 20, 1.0)  # Normalize by expected max
        stress_factors.append(change_count_stress)
        
        # Critical events indicate high stress
        critical_events = [e for e in delta.semantic_events if e.severity == EventSeverity.CRITICAL]
        if critical_events:
            stress_factors.append(0.8)
        
        # Warning events indicate moderate stress
        warning_events = [e for e in delta.semantic_events if e.severity == EventSeverity.WARNING]
        if warning_events:
            stress_factors.append(0.5)
        
        return sum(stress_factors) / len(stress_factors) if stress_factors else 0.0
    
    def _describe_change(self, change: SystemChange) -> str:
        """Create human-readable description of change."""
        if change.category == 'nvidia_gpu':
            if 'temperature' in change.entity_id:
                return f"GPU temperature: {change.old_value}°C → {change.new_value}°C"
            elif 'memory' in change.entity_id:
                return f"GPU memory change in {change.entity_id}"
            elif 'process' in change.entity_id:
                return f"GPU process {change.change_type.value}: {change.entity_id}"
        
        elif change.category == 'processes':
            if change.change_type.value == 'ADDED':
                return f"Process started: {change.entity_id}"
            elif change.change_type.value == 'REMOVED':
                return f"Process terminated: {change.entity_id}"
            else:
                return f"Process {change.change_type.value}: {change.entity_id}"
        
        elif change.category == 'python_env':
            if 'package:' in change.entity_id:
                package_name = change.entity_id.split(':')[-1]
                return f"Python package {change.change_type.value}: {package_name}"
        
        return f"{change.category} {change.change_type.value}: {change.entity_id}"
    
    def _detect_activity_periods(self, hourly_activity: Dict[int, int]) -> List[Dict[str, Any]]:
        """Detect periods of high activity during the day."""
        if not hourly_activity:
            return []
        
        # Calculate activity threshold
        activities = list(hourly_activity.values())
        avg_activity = sum(activities) / len(activities)
        threshold = avg_activity * 1.5  # 50% above average
        
        periods = []
        current_period = None
        
        for hour in sorted(hourly_activity.keys()):
            activity = hourly_activity[hour]
            
            if activity >= threshold:
                if current_period is None:
                    current_period = {
                        'start_hour': hour,
                        'end_hour': hour,
                        'total_activity': activity,
                        'peak_hour': hour,
                        'peak_activity': activity
                    }
                else:
                    current_period['end_hour'] = hour
                    current_period['total_activity'] += activity
                    if activity > current_period['peak_activity']:
                        current_period['peak_hour'] = hour
                        current_period['peak_activity'] = activity
            else:
                if current_period is not None:
                    # End current period
                    current_period['duration_hours'] = current_period['end_hour'] - current_period['start_hour'] + 1
                    periods.append(current_period)
                    current_period = None
        
        # Handle period that extends to end of day
        if current_period is not None:
            current_period['duration_hours'] = current_period['end_hour'] - current_period['start_hour'] + 1
            periods.append(current_period)
        
        return periods
    
    def _detect_daily_patterns(self, deltas: List[SystemDelta]) -> List[str]:
        """Detect notable patterns in daily activity."""
        patterns = []
        
        if not deltas:
            return patterns
        
        # Pattern: Heavy ML workload day
        ml_events = []
        for delta in deltas:
            ml_events.extend([
                e for e in delta.semantic_events 
                if any(keyword in e.event_type.lower() for keyword in ['gpu_thermal', 'ml', 'training', 'memory_pressure'])
            ])
        
        if len(ml_events) > 10:
            patterns.append("heavy_ml_workload_day")
        
        # Pattern: Package management day
        package_changes = 0
        for delta in deltas:
            package_changes += len([
                c for c in delta.raw_delta
                if c.category == 'python_env' and 'package:' in c.entity_id
            ])
        
        if package_changes > 5:
            patterns.append("package_management_activity")
        
        # Pattern: System maintenance day
        service_events = []
        for delta in deltas:
            service_events.extend([
                e for e in delta.semantic_events
                if 'service' in e.event_type.lower() or 'restart' in e.event_type.lower()
            ])
        
        if len(service_events) > 3:
            patterns.append("system_maintenance_activity")
        
        # Pattern: High thermal activity day
        thermal_events = []
        for delta in deltas:
            thermal_events.extend([
                e for e in delta.semantic_events
                if 'thermal' in e.event_type.lower()
            ])
        
        if len(thermal_events) > 5:
            patterns.append("high_thermal_activity")
        
        # Pattern: Development activity day
        dev_processes = 0
        for delta in deltas:
            dev_processes += len([
                c for c in delta.raw_delta
                if c.category == 'processes' and 
                any(tool in c.entity_id.lower() for tool in ['code', 'vim', 'python', 'git'])
            ])
        
        if dev_processes > 20:
            patterns.append("development_activity")
        
        return patterns
    
    def get_summary(self, target_date: date) -> Optional[DailySummary]:
        """
        Get daily summary for specific date.
        
        Args:
            target_date: Date to retrieve summary for
            
        Returns:
            Daily summary or None if not available
        """
        with self._lock:
            # Check if we have pending data for this date
            if target_date in self._pending_deltas and target_date < datetime.now().date():
                self._aggregate_date(target_date)
            
            return self._summaries.get(target_date)
    
    def get_summaries_range(self, start_date: date, end_date: date) -> List[DailySummary]:
        """
        Get daily summaries for date range.
        
        Args:
            start_date: Range start date
            end_date: Range end date
            
        Returns:
            List of daily summaries
        """
        summaries = []
        current_date = start_date
        
        while current_date <= end_date:
            summary = self.get_summary(current_date)
            if summary:
                summaries.append(summary)
            current_date += timedelta(days=1)
        
        return summaries
    
    def get_recent_summaries(self, days: int = 7) -> List[DailySummary]:
        """
        Get recent daily summaries.
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            List of recent daily summaries
        """
        end_date = datetime.now().date() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=days - 1)
        
        return self.get_summaries_range(start_date, end_date)
    
    def get_summary_count(self) -> int:
        """Get total number of daily summaries."""
        with self._lock:
            return len(self._summaries)
    
    def get_total_events(self) -> int:
        """Get total number of events across all summaries."""
        with self._lock:
            return sum(summary.total_events for summary in self._summaries.values())
    
    def get_time_range(self) -> tuple[Optional[date], Optional[date]]:
        """
        Get time range of available summaries.
        
        Returns:
            Tuple of (oldest_date, newest_date)
        """
        with self._lock:
            if not self._summaries:
                return None, None
            
            dates = list(self._summaries.keys())
            return min(dates), max(dates)
    
    def aggregate_old_data(self) -> Dict[str, Any]:
        """Aggregate any pending old data."""
        with self._lock:
            results = {'aggregated_dates': [], 'total_deltas': 0}
            
            today = datetime.now().date()
            
            for target_date in list(self._pending_deltas.keys()):
                if target_date < today:
                    delta_count = len(self._pending_deltas[target_date])
                    self._aggregate_date(target_date)
                    results['aggregated_dates'].append(target_date.isoformat())
                    results['total_deltas'] += delta_count
            
            return results
    
    def clear(self) -> None:
        """Clear all daily summaries."""
        with self._lock:
            self._summaries.clear()
            self._pending_deltas.clear()
    
    def _cleanup_old_pending(self) -> None:
        """Clean up old pending deltas that should be aggregated."""
        cutoff_date = datetime.now().date() - timedelta(days=2)
        
        old_dates = [d for d in self._pending_deltas.keys() if d <= cutoff_date]
        for old_date in old_dates:
            self._aggregate_date(old_date)
    
    def _cleanup_old_summaries(self) -> None:
        """Remove summaries older than retention period."""
        cutoff_date = datetime.now().date() - timedelta(days=self.retention_days)
        
        old_dates = [d for d in self._summaries.keys() if d < cutoff_date]
        for old_date in old_dates:
            del self._summaries[old_date]
            
            # Remove persisted file if it exists
            if self.storage_path:
                summary_file = self.storage_path / f"{old_date.isoformat()}.json"
                if summary_file.exists():
                    summary_file.unlink()
    
    def _save_summary(self, summary: DailySummary) -> None:
        """Save summary to disk."""
        if not self.storage_path:
            return
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        summary_file = self.storage_path / f"{summary.date.isoformat()}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)
    
    def _load_existing_summaries(self) -> None:
        """Load existing summaries from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        for summary_file in self.storage_path.glob('*.json'):
            try:
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                
                summary = DailySummary.from_dict(data)
                self._summaries[summary.date] = summary
                
            except Exception as e:
                print(f"Error loading summary from {summary_file}: {e}")
    
    def persist(self) -> None:
        """Persist all summaries to disk."""
        if not self.storage_path:
            return
        
        with self._lock:
            for summary in self._summaries.values():
                self._save_summary(summary)
    
    def backup_to(self, backup_path: Path) -> None:
        """Backup daily summaries to specified path."""
        backup_path.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            for summary in self._summaries.values():
                backup_file = backup_path / f"{summary.date.isoformat()}.json"
                with open(backup_file, 'w') as f:
                    json.dump(summary.to_dict(), f, indent=2)
    
    def restore_from(self, backup_path: Path) -> None:
        """Restore daily summaries from backup path."""
        if not backup_path.exists():
            return
        
        with self._lock:
            self._summaries.clear()
            
            for summary_file in backup_path.glob('*.json'):
                try:
                    with open(summary_file, 'r') as f:
                        data = json.load(f)
                    
                    summary = DailySummary.from_dict(data)
                    self._summaries[summary.date] = summary
                    
                except Exception as e:
                    print(f"Error restoring summary from {summary_file}: {e}")