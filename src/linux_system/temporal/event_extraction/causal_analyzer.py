"""
Causal Analysis Engine
======================

Analyzes cause-effect relationships between system changes and events.
Identifies causal chains, feedback loops, and emergent system behaviors.
"""

import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter

from ..types import SystemChange, SystemEvent, ChangeType


@dataclass
class CausalRelationship:
    """Represents a causal relationship between changes or events."""
    cause_id: str
    effect_id: str  
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    time_delay: timedelta
    relationship_type: str  # 'direct', 'indirect', 'feedback', 'correlation'
    supporting_evidence: List[str] = field(default_factory=list)
    
    @property
    def is_strong_causal(self) -> bool:
        """Check if this is a strong causal relationship."""
        return self.strength >= 0.7 and self.confidence >= 0.8
    
    @property
    def is_immediate_causal(self) -> bool:
        """Check if this is an immediate causal relationship."""
        return self.time_delay < timedelta(seconds=30)


@dataclass
class CausalChain:
    """Represents a chain of causal relationships."""
    chain_id: str
    relationships: List[CausalRelationship]
    overall_confidence: float
    total_time_span: timedelta
    
    @property
    def length(self) -> int:
        """Get chain length."""
        return len(self.relationships)
    
    @property
    def root_cause_id(self) -> str:
        """Get root cause ID."""
        return self.relationships[0].cause_id if self.relationships else ""
    
    @property
    def final_effect_id(self) -> str:
        """Get final effect ID."""
        return self.relationships[-1].effect_id if self.relationships else ""


class CausalPatternLibrary:
    """Library of known causal patterns."""
    
    def __init__(self):
        self.patterns: Dict[str, Dict[str, Any]] = {
            'gpu_thermal_cascade': {
                'description': 'GPU process spawn → Memory increase → Thermal increase',
                'pattern_sequence': [
                    {'category': 'processes', 'type': 'gpu_process_spawn'},
                    {'category': 'nvidia_gpu', 'type': 'memory_increase'},
                    {'category': 'nvidia_gpu', 'type': 'thermal_increase'}
                ],
                'max_time_window': timedelta(minutes=10),
                'strength': 0.8,
                'confidence': 0.9
            },
            
            'ml_training_startup': {
                'description': 'Package install → Environment setup → GPU process → Resource usage',
                'pattern_sequence': [
                    {'category': 'python_env', 'type': 'package_install'},
                    {'category': 'python_env', 'type': 'env_activation'},
                    {'category': 'processes', 'type': 'ml_process_spawn'},
                    {'category': 'nvidia_gpu', 'type': 'resource_allocation'}
                ],
                'max_time_window': timedelta(minutes=30),
                'strength': 0.7,
                'confidence': 0.8
            },
            
            'resource_pressure_cascade': {
                'description': 'High memory usage → Process spawning → System instability',
                'pattern_sequence': [
                    {'category': 'nvidia_gpu', 'type': 'memory_pressure'},
                    {'category': 'processes', 'type': 'process_termination'},
                    {'category': 'processes', 'type': 'recovery_process_spawn'}
                ],
                'max_time_window': timedelta(minutes=5),
                'strength': 0.6,
                'confidence': 0.7
            },
            
            'development_workflow': {
                'description': 'Environment creation → Package installation → Development activity',
                'pattern_sequence': [
                    {'category': 'python_env', 'type': 'virtual_env_creation'},
                    {'category': 'python_env', 'type': 'package_installation_batch'},
                    {'category': 'processes', 'type': 'development_tool_launch'}
                ],
                'max_time_window': timedelta(hours=1),
                'strength': 0.5,
                'confidence': 0.6
            }
        }
    
    def get_pattern(self, pattern_name: str) -> Optional[Dict[str, Any]]:
        """Get causal pattern by name."""
        return self.patterns.get(pattern_name)
    
    def match_pattern(self, changes: List[SystemChange], 
                     time_window: timedelta = timedelta(hours=1)) -> List[Tuple[str, float]]:
        """Match changes against known patterns."""
        matches = []
        
        for pattern_name, pattern in self.patterns.items():
            confidence = self._calculate_pattern_match_confidence(
                changes, pattern, time_window
            )
            if confidence > 0.3:  # Minimum threshold
                matches.append((pattern_name, confidence))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def _calculate_pattern_match_confidence(self, changes: List[SystemChange], 
                                          pattern: Dict[str, Any],
                                          time_window: timedelta) -> float:
        """Calculate how well changes match a pattern."""
        sequence = pattern['pattern_sequence']
        max_window = pattern['max_time_window']
        
        # Filter changes within time window
        if changes:
            latest_time = max(c.timestamp for c in changes)
            recent_changes = [c for c in changes 
                            if latest_time - c.timestamp <= min(time_window, max_window)]
        else:
            recent_changes = changes
        
        # Check for sequence matches
        sequence_matches = 0
        total_sequence_steps = len(sequence)
        
        for step in sequence:
            matching_changes = [
                c for c in recent_changes 
                if c.category == step['category'] and 
                   self._change_matches_pattern_type(c, step['type'])
            ]
            if matching_changes:
                sequence_matches += 1
        
        # Calculate confidence
        sequence_confidence = sequence_matches / total_sequence_steps if total_sequence_steps > 0 else 0.0
        
        # Temporal proximity bonus
        if len(recent_changes) > 1:
            time_span = max(c.timestamp for c in recent_changes) - min(c.timestamp for c in recent_changes)
            if time_span <= max_window:
                temporal_bonus = 1.0 - (time_span.total_seconds() / max_window.total_seconds())
                sequence_confidence += temporal_bonus * 0.2
        
        return min(sequence_confidence, 1.0)
    
    def _change_matches_pattern_type(self, change: SystemChange, pattern_type: str) -> bool:
        """Check if change matches pattern type."""
        type_mappings = {
            'gpu_process_spawn': lambda c: 'gpu_process' in c.entity_id and c.change_type == ChangeType.ADDED,
            'memory_increase': lambda c: 'memory' in c.entity_id and c.change_type == ChangeType.MODIFIED,
            'thermal_increase': lambda c: 'temperature' in c.entity_id and c.change_type == ChangeType.THRESHOLD_CROSSED,
            'package_install': lambda c: 'package:' in c.entity_id and c.change_type == ChangeType.ADDED,
            'env_activation': lambda c: 'virtual_env:' in c.entity_id and 'activation' in c.entity_id,
            'ml_process_spawn': lambda c: c.category == 'processes' and c.change_type == ChangeType.ADDED,
            'resource_allocation': lambda c: c.category == 'nvidia_gpu' and c.change_type == ChangeType.MODIFIED,
            'memory_pressure': lambda c: 'memory' in c.entity_id and c.significance > 0.7,
            'process_termination': lambda c: c.category == 'processes' and c.change_type == ChangeType.REMOVED,
            'recovery_process_spawn': lambda c: c.category == 'processes' and c.change_type == ChangeType.ADDED,
            'virtual_env_creation': lambda c: 'virtual_env:' in c.entity_id and c.change_type == ChangeType.ADDED,
            'package_installation_batch': lambda c: 'package:' in c.entity_id and c.change_type == ChangeType.ADDED,
            'development_tool_launch': lambda c: c.category == 'processes' and c.change_type == ChangeType.ADDED
        }
        
        matcher = type_mappings.get(pattern_type)
        return matcher(change) if matcher else False


class CausalAnalyzer:
    """Analyzes causal relationships between system changes and events."""
    
    def __init__(self):
        self.pattern_library = CausalPatternLibrary()
        self.relationship_history: List[CausalRelationship] = []
        self.causal_strengths: Dict[Tuple[str, str], float] = defaultdict(float)  # (cause_type, effect_type) -> strength
        self.temporal_patterns: Dict[str, List[float]] = defaultdict(list)  # entity -> [time_deltas]
    
    def analyze_causal_relationships(self, changes: List[SystemChange], 
                                   context_events: List[SystemEvent] = None) -> List[CausalRelationship]:
        """Analyze causal relationships in changes."""
        relationships = []
        
        if len(changes) < 2:
            return relationships
        
        # Sort changes by timestamp
        sorted_changes = sorted(changes, key=lambda c: c.timestamp)
        
        # Analyze pairwise relationships
        for i in range(len(sorted_changes)):
            for j in range(i + 1, len(sorted_changes)):
                cause_change = sorted_changes[i]
                effect_change = sorted_changes[j]
                
                relationship = self._analyze_change_pair(cause_change, effect_change)
                if relationship and relationship.confidence > 0.3:
                    relationships.append(relationship)
        
        # Analyze pattern-based relationships
        pattern_relationships = self._analyze_pattern_based_causality(sorted_changes)
        relationships.extend(pattern_relationships)
        
        # Update learning models
        self._update_causal_learning(relationships)
        
        return relationships
    
    def _analyze_change_pair(self, cause: SystemChange, effect: SystemChange) -> Optional[CausalRelationship]:
        """Analyze potential causal relationship between two changes."""
        time_delay = effect.timestamp - cause.timestamp
        
        # Skip if effect comes before cause
        if time_delay < timedelta(0):
            return None
        
        # Skip if too much time has passed (>1 hour)
        if time_delay > timedelta(hours=1):
            return None
        
        # Calculate causal strength
        strength = self._calculate_causal_strength(cause, effect, time_delay)
        
        # Calculate confidence
        confidence = self._calculate_causal_confidence(cause, effect, time_delay)
        
        if strength > 0.3 and confidence > 0.3:
            relationship_type = self._determine_relationship_type(cause, effect, time_delay)
            
            return CausalRelationship(
                cause_id=f"{cause.category}:{cause.entity_id}",
                effect_id=f"{effect.category}:{effect.entity_id}",
                strength=strength,
                confidence=confidence,
                time_delay=time_delay,
                relationship_type=relationship_type,
                supporting_evidence=self._gather_supporting_evidence(cause, effect)
            )
        
        return None
    
    def _calculate_causal_strength(self, cause: SystemChange, effect: SystemChange, 
                                 time_delay: timedelta) -> float:
        """Calculate strength of causal relationship."""
        strength = 0.0
        
        # Same category relationships are stronger
        if cause.category == effect.category:
            strength += 0.3
        
        # Known strong relationships
        if (cause.category == 'processes' and effect.category == 'nvidia_gpu'):
            # Process spawn can cause GPU changes
            if cause.change_type == ChangeType.ADDED:
                strength += 0.4
        
        elif (cause.category == 'python_env' and effect.category == 'processes'):
            # Environment changes can cause process changes
            strength += 0.3
        
        elif (cause.category == 'nvidia_gpu' and effect.category == 'nvidia_gpu'):
            # GPU changes can cascade
            if 'memory' in cause.entity_id and 'temperature' in effect.entity_id:
                strength += 0.5  # Memory pressure → thermal increase
        
        # Significance-based strength
        combined_significance = (cause.significance + effect.significance) / 2
        strength += combined_significance * 0.3
        
        # Temporal proximity increases strength
        if time_delay < timedelta(seconds=30):
            strength += 0.2
        elif time_delay < timedelta(minutes=2):
            strength += 0.1
        
        # Historical pattern reinforcement
        cause_type = f"{cause.category}:{cause.change_type.value}"
        effect_type = f"{effect.category}:{effect.change_type.value}"
        historical_strength = self.causal_strengths.get((cause_type, effect_type), 0.0)
        strength += historical_strength * 0.2
        
        return min(strength, 1.0)
    
    def _calculate_causal_confidence(self, cause: SystemChange, effect: SystemChange, 
                                   time_delay: timedelta) -> float:
        """Calculate confidence in causal relationship."""
        confidence = 0.5  # Base confidence
        
        # Entity relationship confidence
        if cause.entity_id == effect.entity_id:
            confidence += 0.2  # Same entity changes are more likely causal
        elif self._entities_are_related(cause.entity_id, effect.entity_id):
            confidence += 0.15
        
        # Metadata correlation confidence
        if self._metadata_suggests_causality(cause, effect):
            confidence += 0.2
        
        # Temporal confidence (closer in time = higher confidence)
        if time_delay < timedelta(seconds=10):
            confidence += 0.2
        elif time_delay < timedelta(minutes=1):
            confidence += 0.1
        
        # Pattern recognition confidence
        cause_type = f"{cause.category}:{cause.change_type.value}"
        effect_type = f"{effect.category}:{effect.change_type.value}"
        
        # Check against known patterns
        pattern_matches = self.pattern_library.match_pattern([cause, effect], time_delay)
        if pattern_matches:
            best_match_confidence = pattern_matches[0][1]
            confidence += best_match_confidence * 0.3
        
        return min(confidence, 1.0)
    
    def _determine_relationship_type(self, cause: SystemChange, effect: SystemChange, 
                                   time_delay: timedelta) -> str:
        """Determine type of causal relationship."""
        if time_delay < timedelta(seconds=5):
            return 'direct'
        elif time_delay < timedelta(minutes=2):
            if cause.category == effect.category:
                return 'direct'
            else:
                return 'indirect'
        else:
            return 'indirect'
    
    def _entities_are_related(self, entity1: str, entity2: str) -> bool:
        """Check if entities are related."""
        # GPU entities
        gpu_entities = ['gpu:', 'nvidia:', 'rtx_5090']
        if any(gpu in entity1 for gpu in gpu_entities) and any(gpu in entity2 for gpu in gpu_entities):
            return True
        
        # Process entities
        if 'process:' in entity1 and 'process:' in entity2:
            # Same process or parent-child relationship
            return True
        
        # Python environment entities
        if ('package:' in entity1 or 'virtual_env:' in entity1) and \
           ('package:' in entity2 or 'virtual_env:' in entity2):
            return True
        
        return False
    
    def _metadata_suggests_causality(self, cause: SystemChange, effect: SystemChange) -> bool:
        """Check if metadata suggests causal relationship."""
        # Process PID relationships
        if cause.category == 'processes' and effect.category == 'processes':
            cause_pid = cause.metadata.get('pid')
            effect_ppid = effect.metadata.get('parent_pid')
            if cause_pid and effect_ppid and cause_pid == effect_ppid:
                return True
        
        # GPU process relationships
        if 'gpu_process' in str(cause.metadata) or 'gpu_process' in str(effect.metadata):
            return True
        
        # ML framework relationships
        if cause.metadata.get('is_ml_framework') or effect.metadata.get('is_ml_framework'):
            return True
        
        return False
    
    def _gather_supporting_evidence(self, cause: SystemChange, effect: SystemChange) -> List[str]:
        """Gather supporting evidence for causal relationship."""
        evidence = []
        
        # Temporal evidence
        time_diff = effect.timestamp - cause.timestamp
        if time_diff < timedelta(seconds=30):
            evidence.append(f"Occurred within {time_diff.total_seconds():.1f} seconds")
        
        # Significance evidence
        if cause.significance > 0.7:
            evidence.append("High significance cause event")
        
        # Category evidence
        if cause.category == effect.category:
            evidence.append(f"Both changes in {cause.category} category")
        
        # Metadata evidence
        if self._metadata_suggests_causality(cause, effect):
            evidence.append("Metadata indicates relationship")
        
        return evidence
    
    def _analyze_pattern_based_causality(self, changes: List[SystemChange]) -> List[CausalRelationship]:
        """Analyze causality based on known patterns."""
        relationships = []
        
        pattern_matches = self.pattern_library.match_pattern(changes)
        
        for pattern_name, confidence in pattern_matches:
            if confidence > 0.5:
                pattern = self.pattern_library.get_pattern(pattern_name)
                
                # Create relationships based on pattern sequence
                pattern_relationships = self._create_pattern_relationships(
                    changes, pattern, confidence
                )
                relationships.extend(pattern_relationships)
        
        return relationships
    
    def _create_pattern_relationships(self, changes: List[SystemChange], 
                                    pattern: Dict[str, Any], confidence: float) -> List[CausalRelationship]:
        """Create causal relationships based on pattern matching."""
        relationships = []
        sequence = pattern['pattern_sequence']
        
        # Find changes matching each step in sequence
        sequence_changes = []
        for step in sequence:
            matching_changes = [
                c for c in changes 
                if c.category == step['category'] and 
                   self.pattern_library._change_matches_pattern_type(c, step['type'])
            ]
            if matching_changes:
                # Take the most significant matching change
                best_match = max(matching_changes, key=lambda c: c.significance)
                sequence_changes.append(best_match)
        
        # Create relationships between consecutive sequence steps
        for i in range(len(sequence_changes) - 1):
            cause = sequence_changes[i]
            effect = sequence_changes[i + 1]
            
            time_delay = effect.timestamp - cause.timestamp
            
            relationship = CausalRelationship(
                cause_id=f"{cause.category}:{cause.entity_id}",
                effect_id=f"{effect.category}:{effect.entity_id}",
                strength=pattern['strength'],
                confidence=confidence * pattern['confidence'],
                time_delay=time_delay,
                relationship_type='pattern_based',
                supporting_evidence=[f"Matches pattern: {pattern['description']}"]
            )
            relationships.append(relationship)
        
        return relationships
    
    def _update_causal_learning(self, relationships: List[CausalRelationship]) -> None:
        """Update causal learning models."""
        for rel in relationships:
            # Update strength statistics
            cause_parts = rel.cause_id.split(':')
            effect_parts = rel.effect_id.split(':')
            
            if len(cause_parts) >= 2 and len(effect_parts) >= 2:
                cause_type = f"{cause_parts[0]}:{cause_parts[1] if len(cause_parts) > 2 else 'general'}"
                effect_type = f"{effect_parts[0]}:{effect_parts[1] if len(effect_parts) > 2 else 'general'}"
                
                # Exponential moving average
                current_strength = self.causal_strengths[(cause_type, effect_type)]
                self.causal_strengths[(cause_type, effect_type)] = \
                    current_strength * 0.9 + rel.strength * 0.1
            
            # Update temporal patterns
            entity = rel.effect_id
            self.temporal_patterns[entity].append(rel.time_delay.total_seconds())
            
            # Keep only recent temporal patterns
            if len(self.temporal_patterns[entity]) > 100:
                self.temporal_patterns[entity] = self.temporal_patterns[entity][-100:]
        
        # Store relationships for future analysis
        self.relationship_history.extend(relationships)
        if len(self.relationship_history) > 1000:
            self.relationship_history = self.relationship_history[-1000:]
    
    def build_causal_chains(self, relationships: List[CausalRelationship], 
                           max_chain_length: int = 5) -> List[CausalChain]:
        """Build causal chains from relationships."""
        chains = []
        
        # Group relationships by cause-effect connections
        cause_to_effects: Dict[str, List[CausalRelationship]] = defaultdict(list)
        effect_to_causes: Dict[str, List[CausalRelationship]] = defaultdict(list)
        
        for rel in relationships:
            cause_to_effects[rel.cause_id].append(rel)
            effect_to_causes[rel.effect_id].append(rel)
        
        # Find chain starts (causes with no incoming relationships or weak incoming)
        chain_starts = set()
        for rel in relationships:
            if rel.cause_id not in effect_to_causes or \
               all(r.confidence < 0.5 for r in effect_to_causes[rel.cause_id]):
                chain_starts.add(rel.cause_id)
        
        # Build chains from each start
        for start_id in chain_starts:
            chain = self._build_chain_from_start(
                start_id, cause_to_effects, [], max_chain_length
            )
            if chain and len(chain) > 1:  # Only chains with multiple steps
                chain_confidence = min(rel.confidence for rel in chain)
                total_time = chain[-1].time_delay if chain else timedelta(0)
                
                causal_chain = CausalChain(
                    chain_id=f"chain_{start_id}_{len(chains)}",
                    relationships=chain,
                    overall_confidence=chain_confidence,
                    total_time_span=total_time
                )
                chains.append(causal_chain)
        
        return chains
    
    def _build_chain_from_start(self, start_id: str, 
                               cause_to_effects: Dict[str, List[CausalRelationship]],
                               current_chain: List[CausalRelationship],
                               max_length: int) -> List[CausalRelationship]:
        """Recursively build causal chain from start."""
        if len(current_chain) >= max_length:
            return current_chain
        
        if start_id not in cause_to_effects:
            return current_chain
        
        # Find best next relationship
        next_relationships = cause_to_effects[start_id]
        if not next_relationships:
            return current_chain
        
        # Choose relationship with highest confidence
        best_next = max(next_relationships, key=lambda r: r.confidence)
        
        # Avoid cycles
        if any(rel.effect_id == best_next.effect_id for rel in current_chain):
            return current_chain
        
        # Continue chain
        new_chain = current_chain + [best_next]
        return self._build_chain_from_start(
            best_next.effect_id, cause_to_effects, new_chain, max_length
        )