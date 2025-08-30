"""
Significance Calculation Engine
===============================

Advanced significance scoring for system changes with domain-specific
knowledge and adaptive learning capabilities.

The significance calculator determines how important a detected change is,
helping prioritize attention and reduce noise in system monitoring.
"""

import re
import statistics
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque

from ..types import SystemChange, ChangeType


class SignificanceCalculator:
    """
    Calculates significance scores for system changes with adaptive learning.
    
    Features:
    - Domain-specific scoring rules
    - Historical context awareness  
    - Adaptive thresholds based on system patterns
    - Performance impact assessment
    - Critical system component prioritization
    """
    
    def __init__(self):
        # Base scoring weights for different change types
        self.change_type_weights = {
            ChangeType.ADDED: 0.7,
            ChangeType.REMOVED: 0.8, 
            ChangeType.THRESHOLD_CROSSED: 0.9,
            ChangeType.ANOMALY_DETECTED: 0.95,
            ChangeType.STATE_TRANSITION: 0.8,
            ChangeType.MODIFIED: 0.5
        }
        
        # Critical system components that get higher significance
        self.critical_components = {
            'gpu', 'cuda', 'nvidia', 'memory', 'cpu', 'thermal', 'power',
            'kernel', 'driver', 'pytorch', 'tensorflow', 'transformers'
        }
        
        # Performance-related keywords that increase significance
        self.performance_keywords = {
            'performance', 'throughput', 'latency', 'bottleneck', 'throttle',
            'memory_leak', 'cpu_usage', 'gpu_usage', 'io_wait', 'swap'
        }
        
        # Error/problem keywords that significantly increase importance
        self.error_keywords = {
            'error', 'fail', 'crash', 'exception', 'timeout', 'deadlock',
            'segfault', 'oom', 'panic', 'abort', 'critical'
        }
        
        # Historical context for adaptive scoring
        self.change_history: deque = deque(maxlen=1000)  # Last 1000 changes
        self.category_baselines: Dict[str, float] = {}  # Average significance per category
        
        # Pattern recognition for recurring changes
        self.recurring_patterns: Dict[str, int] = defaultdict(int)  # Track frequency
        
    def calculate(self, change: SystemChange, historical_context: Optional[List[SystemChange]] = None) -> float:
        """
        Calculate comprehensive significance score for a system change.
        
        Args:
            change: The system change to score
            historical_context: Recent changes for context (optional)
            
        Returns:
            Significance score between 0.0 and 1.0
        """
        # Start with base score from change type
        base_score = self.change_type_weights.get(change.change_type, 0.5)
        
        # Apply various scoring factors
        score = base_score
        score += self._score_component_criticality(change)
        score += self._score_performance_impact(change)
        score += self._score_error_indicators(change)
        score += self._score_value_change_magnitude(change)
        score += self._score_temporal_context(change, historical_context)
        score += self._score_rarity(change)
        
        # Apply category-specific adjustments
        score = self._apply_category_adjustments(change, score)
        
        # Ensure score is in valid range
        final_score = max(0.0, min(1.0, score))
        
        # Update historical tracking
        self._update_historical_context(change, final_score)
        
        return final_score
    
    def _score_component_criticality(self, change: SystemChange) -> float:
        """Score based on criticality of affected component."""
        score_boost = 0.0
        
        # Check entity ID and metadata for critical component keywords
        text_to_check = f"{change.entity_id} {change.metadata}".lower()
        
        for critical_component in self.critical_components:
            if critical_component in text_to_check:
                score_boost += 0.2
                break  # Don't double-count
        
        # Special handling for GPU-related changes (RTX 5090 focus)
        if any(gpu_term in text_to_check for gpu_term in ['gpu', 'nvidia', 'cuda', 'rtx']):
            score_boost += 0.1  # Extra boost for GPU changes
        
        return min(0.3, score_boost)  # Cap the boost
    
    def _score_performance_impact(self, change: SystemChange) -> float:
        """Score based on potential performance impact."""
        score_boost = 0.0
        
        text_to_check = f"{change.entity_id} {change.metadata}".lower()
        
        # Check for performance-related keywords
        for perf_keyword in self.performance_keywords:
            if perf_keyword in text_to_check:
                score_boost += 0.15
                break
        
        # Special scoring for specific performance scenarios
        if change.category == 'nvidia_gpu':
            # Temperature changes affecting performance
            if 'temperature' in change.entity_id and isinstance(change.new_value, (int, float)):
                temp_value = float(change.new_value)
                if temp_value > 80:  # Approaching thermal limits
                    score_boost += 0.2
                elif temp_value > 70:  # Getting warm
                    score_boost += 0.1
            
            # Memory pressure
            if 'memory' in change.entity_id:
                score_boost += 0.1
        
        elif change.category == 'memory':
            # System memory pressure indicators
            if any(mem_term in change.entity_id.lower() 
                   for mem_term in ['available', 'free', 'swap']):
                score_boost += 0.1
        
        return min(0.25, score_boost)
    
    def _score_error_indicators(self, change: SystemChange) -> float:
        """Score based on error/problem indicators."""
        score_boost = 0.0
        
        text_to_check = f"{change.entity_id} {change.metadata}".lower()
        
        # Check for error keywords
        for error_keyword in self.error_keywords:
            if error_keyword in text_to_check:
                score_boost += 0.3  # Errors are very significant
                break
        
        # Check if change indicates a problem state
        if change.change_type == ChangeType.THRESHOLD_CROSSED:
            threshold_type = change.metadata.get('threshold_type', '').lower()
            if any(problem in threshold_type for problem in ['critical', 'error', 'limit']):
                score_boost += 0.2
        
        return min(0.4, score_boost)  # Errors can dominate significance
    
    def _score_value_change_magnitude(self, change: SystemChange) -> float:
        """Score based on magnitude of value change."""
        score_boost = 0.0
        
        # Only score magnitude for numeric changes
        if (isinstance(change.old_value, (int, float)) and 
            isinstance(change.new_value, (int, float))):
            
            old_val = float(change.old_value)
            new_val = float(change.new_value)
            
            # Avoid division by zero
            if old_val == 0:
                # New value appearing is significant
                score_boost += 0.1 if new_val > 0 else 0.0
            else:
                # Calculate percentage change
                percent_change = abs((new_val - old_val) / old_val)
                
                if percent_change > 1.0:  # > 100% change
                    score_boost += 0.2
                elif percent_change > 0.5:  # > 50% change
                    score_boost += 0.15
                elif percent_change > 0.2:  # > 20% change
                    score_boost += 0.1
                elif percent_change > 0.1:  # > 10% change
                    score_boost += 0.05
        
        return min(0.2, score_boost)
    
    def _score_temporal_context(self, change: SystemChange, 
                               historical_context: Optional[List[SystemChange]]) -> float:
        """Score based on temporal patterns and context."""
        score_boost = 0.0
        
        if not historical_context:
            return 0.0
        
        # Look for recent similar changes
        recent_threshold = timedelta(minutes=10)
        current_time = change.timestamp
        
        similar_recent_changes = [
            c for c in historical_context
            if (c.category == change.category and 
                c.entity_id == change.entity_id and
                abs((current_time - c.timestamp).total_seconds()) < recent_threshold.total_seconds())
        ]
        
        # Rapid successive changes are more significant
        if len(similar_recent_changes) > 2:
            score_boost += 0.1
        
        # First occurrence of a type of change is more significant
        similar_ever = [
            c for c in historical_context
            if (c.category == change.category and 
                c.entity_id == change.entity_id)
        ]
        
        if len(similar_ever) == 0:  # First time seeing this change
            score_boost += 0.15
        
        return min(0.2, score_boost)
    
    def _score_rarity(self, change: SystemChange) -> float:
        """Score based on how rare this type of change is."""
        # Create a pattern signature for this change
        pattern = f"{change.category}:{change.change_type.value}:{change.entity_id}"
        
        # Count how often we've seen this pattern
        pattern_count = self.recurring_patterns.get(pattern, 0)
        
        # Rare patterns are more significant
        if pattern_count == 0:
            return 0.1  # First occurrence
        elif pattern_count < 3:
            return 0.05  # Still relatively rare
        else:
            return 0.0  # Common pattern
    
    def _apply_category_adjustments(self, change: SystemChange, current_score: float) -> float:
        """Apply category-specific score adjustments."""
        adjustments = {
            'nvidia_gpu': 0.1,  # GPU changes are important for this system
            'processes': -0.05,  # Process changes are frequent, slightly less important
            'python_env': 0.05,  # Python env changes matter for ML work
            'security': 0.15,   # Security changes are very important
            'kernel': 0.2,      # Kernel changes are critical
            'storage': 0.05,    # Storage changes can indicate problems
        }
        
        adjustment = adjustments.get(change.category, 0.0)
        return current_score + adjustment
    
    def _update_historical_context(self, change: SystemChange, final_score: float):
        """Update historical tracking for adaptive scoring."""
        # Add to change history
        self.change_history.append((change, final_score))
        
        # Update category baseline
        category_scores = [
            score for (c, score) in self.change_history 
            if c.category == change.category
        ]
        
        if category_scores:
            self.category_baselines[change.category] = statistics.mean(category_scores)
        
        # Update pattern frequency
        pattern = f"{change.category}:{change.change_type.value}:{change.entity_id}"
        self.recurring_patterns[pattern] += 1
    
    def get_category_baseline(self, category: str) -> float:
        """Get the baseline significance for a category."""
        return self.category_baselines.get(category, 0.5)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scoring statistics for monitoring."""
        total_changes = len(self.change_history)
        
        if total_changes == 0:
            return {
                'total_changes_scored': 0,
                'average_significance': 0.0,
                'category_baselines': {},
                'most_common_patterns': []
            }
        
        all_scores = [score for (_, score) in self.change_history]
        
        # Most common patterns
        most_common = sorted(
            self.recurring_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'total_changes_scored': total_changes,
            'average_significance': statistics.mean(all_scores),
            'median_significance': statistics.median(all_scores),
            'high_significance_rate': len([s for s in all_scores if s > 0.8]) / total_changes,
            'category_baselines': dict(self.category_baselines),
            'most_common_patterns': most_common,
            'unique_patterns_seen': len(self.recurring_patterns)
        }