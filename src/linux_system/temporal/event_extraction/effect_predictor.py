"""
Effect Prediction Engine
========================

Predicts likely future effects of system events and changes.
Uses pattern recognition, causal models, and system knowledge
to anticipate system behavior and potential issues.
"""

import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum

from ..types import SystemChange, SystemEvent, ChangeType, EventSeverity


class EffectType(Enum):
    """Types of predicted effects."""
    IMMEDIATE = "immediate"      # Effects within 1 minute
    SHORT_TERM = "short_term"    # Effects within 15 minutes  
    MEDIUM_TERM = "medium_term"  # Effects within 1 hour
    LONG_TERM = "long_term"      # Effects within 24 hours


class EffectCategory(Enum):
    """Categories of effects."""
    PERFORMANCE = "performance"
    STABILITY = "stability"
    RESOURCE = "resource"
    CAPABILITY = "capability"
    SECURITY = "security"
    MAINTENANCE = "maintenance"


@dataclass
class PredictedEffect:
    """A predicted future effect."""
    effect_id: str
    description: str
    category: EffectCategory
    effect_type: EffectType
    probability: float  # 0.0 to 1.0
    confidence: float   # 0.0 to 1.0
    severity: EventSeverity
    expected_timeframe: timedelta
    mitigation_suggestions: List[str] = field(default_factory=list)
    monitoring_recommendations: List[str] = field(default_factory=list)
    
    @property
    def is_high_risk(self) -> bool:
        """Check if this is a high-risk effect."""
        return (self.probability > 0.7 and self.confidence > 0.6 and 
                self.severity in [EventSeverity.WARNING, EventSeverity.CRITICAL])


@dataclass
class EffectPredictionModel:
    """Model for predicting specific types of effects."""
    name: str
    description: str
    trigger_patterns: List[Dict[str, Any]]
    effect_templates: List[Dict[str, Any]]
    confidence_factors: Dict[str, float]
    
    def matches_trigger(self, changes: List[SystemChange], context: Dict[str, Any]) -> Tuple[bool, float]:
        """Check if changes match trigger patterns."""
        total_score = 0.0
        total_weight = 0.0
        
        for pattern in self.trigger_patterns:
            weight = pattern.get('weight', 1.0)
            score = self._evaluate_pattern(changes, context, pattern)
            total_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return False, 0.0
        
        match_score = total_score / total_weight
        threshold = pattern.get('threshold', 0.6)
        
        return match_score >= threshold, match_score
    
    def _evaluate_pattern(self, changes: List[SystemChange], context: Dict[str, Any], 
                         pattern: Dict[str, Any]) -> float:
        """Evaluate single pattern against changes."""
        score = 0.0
        
        # Category matching
        if 'category' in pattern:
            matching_changes = [c for c in changes if c.category == pattern['category']]
            if matching_changes:
                score += 0.3
        
        # Change type matching
        if 'change_type' in pattern:
            matching_changes = [c for c in changes if c.change_type.value == pattern['change_type']]
            if matching_changes:
                score += 0.3
        
        # Significance threshold
        if 'min_significance' in pattern:
            significant_changes = [c for c in changes if c.significance >= pattern['min_significance']]
            if significant_changes:
                score += 0.2
        
        # Context conditions
        if 'context_conditions' in pattern:
            for key, expected_value in pattern['context_conditions'].items():
                if context.get(key) == expected_value:
                    score += 0.2
        
        return min(score, 1.0)


class EffectPredictionLibrary:
    """Library of effect prediction models."""
    
    def __init__(self):
        self.models: Dict[str, EffectPredictionModel] = {}
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default prediction models."""
        
        # GPU Thermal Throttling Model
        self.models['gpu_thermal_throttling'] = EffectPredictionModel(
            name="GPU Thermal Throttling",
            description="Predicts when GPU thermal throttling is likely",
            trigger_patterns=[
                {
                    'category': 'nvidia_gpu',
                    'change_type': 'THRESHOLD_CROSSED',
                    'min_significance': 0.6,
                    'weight': 2.0,
                    'threshold': 0.5
                },
                {
                    'category': 'processes',
                    'change_type': 'ADDED',
                    'context_conditions': {'is_gpu_process': True},
                    'weight': 1.5
                }
            ],
            effect_templates=[
                {
                    'description': 'GPU performance degradation due to thermal throttling',
                    'category': EffectCategory.PERFORMANCE,
                    'severity': EventSeverity.WARNING,
                    'probability_base': 0.7,
                    'timeframe_minutes': 5
                }
            ],
            confidence_factors={
                'high_temperature': 0.3,
                'sustained_load': 0.2,
                'multiple_processes': 0.2
            }
        )
        
        # Memory Pressure Model
        self.models['memory_pressure_cascade'] = EffectPredictionModel(
            name="Memory Pressure Cascade",
            description="Predicts memory pressure leading to process termination",
            trigger_patterns=[
                {
                    'category': 'nvidia_gpu',
                    'min_significance': 0.7,
                    'context_conditions': {'memory_usage_high': True},
                    'weight': 2.0,
                    'threshold': 0.6
                }
            ],
            effect_templates=[
                {
                    'description': 'Process termination due to GPU memory exhaustion',
                    'category': EffectCategory.STABILITY,
                    'severity': EventSeverity.CRITICAL,
                    'probability_base': 0.8,
                    'timeframe_minutes': 10
                },
                {
                    'description': 'System performance degradation',
                    'category': EffectCategory.PERFORMANCE,
                    'severity': EventSeverity.WARNING,
                    'probability_base': 0.6,
                    'timeframe_minutes': 15
                }
            ],
            confidence_factors={
                'memory_usage_trend': 0.4,
                'concurrent_processes': 0.3
            }
        )
        
        # ML Training Instability Model
        self.models['ml_training_instability'] = EffectPredictionModel(
            name="ML Training Instability",
            description="Predicts instability in ML training workflows",
            trigger_patterns=[
                {
                    'category': 'processes',
                    'change_type': 'ADDED',
                    'context_conditions': {'is_ml_process': True},
                    'weight': 1.5,
                    'threshold': 0.4
                },
                {
                    'category': 'nvidia_gpu',
                    'min_significance': 0.5,
                    'weight': 1.0
                }
            ],
            effect_templates=[
                {
                    'description': 'Training job interruption due to resource constraints',
                    'category': EffectCategory.STABILITY,
                    'severity': EventSeverity.WARNING,
                    'probability_base': 0.5,
                    'timeframe_minutes': 30
                }
            ],
            confidence_factors={
                'model_size': 0.3,
                'available_memory': 0.4
            }
        )
        
        # Environment Compatibility Issues Model
        self.models['environment_compatibility'] = EffectPredictionModel(
            name="Environment Compatibility Issues",
            description="Predicts compatibility issues from environment changes",
            trigger_patterns=[
                {
                    'category': 'python_env',
                    'change_type': 'MODIFIED',
                    'min_significance': 0.6,
                    'weight': 2.0,
                    'threshold': 0.5
                }
            ],
            effect_templates=[
                {
                    'description': 'Package compatibility conflicts',
                    'category': EffectCategory.CAPABILITY,
                    'severity': EventSeverity.WARNING,
                    'probability_base': 0.4,
                    'timeframe_minutes': 60
                },
                {
                    'description': 'Application runtime errors',
                    'category': EffectCategory.STABILITY,
                    'severity': EventSeverity.WARNING,
                    'probability_base': 0.3,
                    'timeframe_minutes': 120
                }
            ],
            confidence_factors={
                'major_version_change': 0.5,
                'dependency_complexity': 0.3
            }
        )
    
    def get_model(self, model_name: str) -> Optional[EffectPredictionModel]:
        """Get prediction model by name."""
        return self.models.get(model_name)
    
    def get_applicable_models(self, changes: List[SystemChange], 
                            context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Get models applicable to given changes."""
        applicable = []
        
        for model_name, model in self.models.items():
            matches, score = model.matches_trigger(changes, context)
            if matches:
                applicable.append((model_name, score))
        
        return sorted(applicable, key=lambda x: x[1], reverse=True)


class EffectPredictor:
    """Predicts future effects of system changes and events."""
    
    def __init__(self):
        self.prediction_library = EffectPredictionLibrary()
        self.historical_patterns: Dict[str, List[float]] = defaultdict(list)  # Effect -> probabilities
        self.prediction_accuracy: Dict[str, float] = defaultdict(lambda: 0.5)  # Model -> accuracy
        self.effect_history: List[Dict[str, Any]] = []
    
    def predict_effects(self, changes: List[SystemChange], 
                       events: List[SystemEvent] = None,
                       context: Dict[str, Any] = None) -> List[PredictedEffect]:
        """Predict effects from changes and events."""
        if not changes:
            return []
        
        context = context or {}
        events = events or []
        
        # Enrich context with change analysis
        enriched_context = self._enrich_context(changes, events, context)
        
        # Get applicable prediction models
        applicable_models = self.prediction_library.get_applicable_models(changes, enriched_context)
        
        predicted_effects = []
        
        # Generate predictions from each applicable model
        for model_name, trigger_score in applicable_models:
            model = self.prediction_library.get_model(model_name)
            if model:
                model_effects = self._generate_model_predictions(
                    model, changes, enriched_context, trigger_score
                )
                predicted_effects.extend(model_effects)
        
        # Add pattern-based predictions
        pattern_effects = self._generate_pattern_based_predictions(changes, enriched_context)
        predicted_effects.extend(pattern_effects)
        
        # Remove duplicates and rank by risk
        deduplicated_effects = self._deduplicate_effects(predicted_effects)
        ranked_effects = self._rank_effects_by_risk(deduplicated_effects)
        
        # Store for learning
        self._store_predictions(ranked_effects, changes, events)
        
        return ranked_effects
    
    def _enrich_context(self, changes: List[SystemChange], events: List[SystemEvent],
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich context with analysis of changes and events."""
        enriched = dict(context)
        
        # Change analysis
        enriched['change_count'] = len(changes)
        enriched['categories'] = list(set(c.category for c in changes))
        enriched['avg_significance'] = sum(c.significance for c in changes) / len(changes) if changes else 0.0
        enriched['max_significance'] = max(c.significance for c in changes) if changes else 0.0
        
        # GPU-specific context
        gpu_changes = [c for c in changes if c.category == 'nvidia_gpu']
        if gpu_changes:
            enriched['gpu_changes_count'] = len(gpu_changes)
            enriched['has_thermal_changes'] = any('temperature' in c.entity_id for c in gpu_changes)
            enriched['has_memory_changes'] = any('memory' in c.entity_id for c in gpu_changes)
            enriched['memory_usage_high'] = any(c.significance > 0.8 and 'memory' in c.entity_id for c in gpu_changes)
        
        # Process-specific context
        process_changes = [c for c in changes if c.category == 'processes']
        if process_changes:
            spawns = [c for c in process_changes if c.change_type == ChangeType.ADDED]
            terminations = [c for c in process_changes if c.change_type == ChangeType.REMOVED]
            
            enriched['process_spawns'] = len(spawns)
            enriched['process_terminations'] = len(terminations)
            enriched['is_gpu_process'] = any('gpu' in str(c.metadata) for c in process_changes)
            enriched['is_ml_process'] = any(c.metadata.get('is_ml_framework', False) for c in process_changes)
        
        # Python environment context
        python_changes = [c for c in changes if c.category == 'python_env']
        if python_changes:
            enriched['python_changes_count'] = len(python_changes)
            enriched['has_package_changes'] = any('package:' in c.entity_id for c in python_changes)
            enriched['has_ml_packages'] = any(c.metadata.get('is_ml_framework', False) for c in python_changes)
        
        # Temporal context
        if len(changes) > 1:
            timestamps = [c.timestamp for c in changes]
            time_span = max(timestamps) - min(timestamps)
            enriched['time_span_seconds'] = time_span.total_seconds()
            enriched['rapid_changes'] = time_span < timedelta(minutes=2)
        
        # Event context
        if events:
            enriched['concurrent_events'] = len(events)
            enriched['critical_events'] = len([e for e in events if e.severity == EventSeverity.CRITICAL])
        
        return enriched
    
    def _generate_model_predictions(self, model: EffectPredictionModel, 
                                  changes: List[SystemChange],
                                  context: Dict[str, Any], 
                                  trigger_score: float) -> List[PredictedEffect]:
        """Generate predictions from a specific model."""
        predictions = []
        
        for effect_template in model.effect_templates:
            # Base probability from template
            base_probability = effect_template['probability_base']
            
            # Adjust probability based on trigger score
            probability = base_probability * trigger_score
            
            # Apply confidence factors
            confidence_adjustment = 0.0
            for factor, weight in model.confidence_factors.items():
                if self._evaluate_confidence_factor(factor, changes, context):
                    confidence_adjustment += weight
            
            probability = min(probability + confidence_adjustment, 1.0)
            
            # Calculate confidence (how sure we are about the probability)
            confidence = self._calculate_prediction_confidence(model, trigger_score, context)
            
            # Adjust for historical accuracy
            model_accuracy = self.prediction_accuracy.get(model.name, 0.5)
            confidence *= model_accuracy
            
            if probability > 0.2:  # Only include meaningful predictions
                effect = PredictedEffect(
                    effect_id=f"{model.name}_{len(predictions)}",
                    description=effect_template['description'],
                    category=effect_template['category'],
                    effect_type=self._determine_effect_type(effect_template['timeframe_minutes']),
                    probability=probability,
                    confidence=confidence,
                    severity=effect_template['severity'],
                    expected_timeframe=timedelta(minutes=effect_template['timeframe_minutes']),
                    mitigation_suggestions=self._generate_mitigation_suggestions(
                        effect_template, changes, context
                    ),
                    monitoring_recommendations=self._generate_monitoring_recommendations(
                        effect_template, changes, context
                    )
                )
                predictions.append(effect)
        
        return predictions
    
    def _evaluate_confidence_factor(self, factor: str, changes: List[SystemChange], 
                                  context: Dict[str, Any]) -> bool:
        """Evaluate if confidence factor applies to current situation."""
        factor_evaluations = {
            'high_temperature': lambda: any(
                c.category == 'nvidia_gpu' and 'temperature' in c.entity_id and c.significance > 0.7
                for c in changes
            ),
            'sustained_load': lambda: context.get('rapid_changes', False),
            'multiple_processes': lambda: context.get('process_spawns', 0) > 1,
            'memory_usage_trend': lambda: context.get('has_memory_changes', False),
            'concurrent_processes': lambda: context.get('process_spawns', 0) > 0,
            'model_size': lambda: context.get('max_significance', 0) > 0.8,
            'available_memory': lambda: not context.get('memory_usage_high', False),
            'major_version_change': lambda: any(
                'major' in str(c.metadata) for c in changes if c.category == 'python_env'
            ),
            'dependency_complexity': lambda: context.get('python_changes_count', 0) > 3
        }
        
        evaluator = factor_evaluations.get(factor)
        return evaluator() if evaluator else False
    
    def _calculate_prediction_confidence(self, model: EffectPredictionModel, 
                                       trigger_score: float, context: Dict[str, Any]) -> float:
        """Calculate confidence in prediction."""
        confidence = 0.5  # Base confidence
        
        # Higher trigger score = higher confidence
        confidence += trigger_score * 0.3
        
        # More context information = higher confidence
        context_richness = len([v for v in context.values() if v]) / max(len(context), 1)
        confidence += context_richness * 0.2
        
        # Historical accuracy of model
        model_accuracy = self.prediction_accuracy.get(model.name, 0.5)
        confidence = confidence * (0.5 + model_accuracy * 0.5)
        
        return min(confidence, 1.0)
    
    def _determine_effect_type(self, timeframe_minutes: int) -> EffectType:
        """Determine effect type based on timeframe."""
        if timeframe_minutes <= 1:
            return EffectType.IMMEDIATE
        elif timeframe_minutes <= 15:
            return EffectType.SHORT_TERM
        elif timeframe_minutes <= 60:
            return EffectType.MEDIUM_TERM
        else:
            return EffectType.LONG_TERM
    
    def _generate_pattern_based_predictions(self, changes: List[SystemChange], 
                                          context: Dict[str, Any]) -> List[PredictedEffect]:
        """Generate predictions based on historical patterns."""
        predictions = []
        
        # GPU memory exhaustion pattern
        if (context.get('has_memory_changes', False) and 
            context.get('memory_usage_high', False)):
            
            predictions.append(PredictedEffect(
                effect_id="pattern_gpu_memory_exhaustion",
                description="GPU memory exhaustion leading to process termination",
                category=EffectCategory.STABILITY,
                effect_type=EffectType.SHORT_TERM,
                probability=0.6,
                confidence=0.7,
                severity=EventSeverity.WARNING,
                expected_timeframe=timedelta(minutes=10),
                mitigation_suggestions=[
                    "Monitor GPU memory usage closely",
                    "Consider reducing batch size or model complexity",
                    "Free up GPU memory by terminating unnecessary processes"
                ]
            ))
        
        # ML workflow interruption pattern
        if (context.get('is_ml_process', False) and 
            context.get('system_stress_level', 0) > 0.7):
            
            predictions.append(PredictedEffect(
                effect_id="pattern_ml_workflow_interruption",
                description="ML workflow interruption due to system stress",
                category=EffectCategory.PERFORMANCE,
                effect_type=EffectType.MEDIUM_TERM,
                probability=0.5,
                confidence=0.6,
                severity=EventSeverity.WARNING,
                expected_timeframe=timedelta(minutes=30)
            ))
        
        return predictions
    
    def _generate_mitigation_suggestions(self, effect_template: Dict[str, Any], 
                                       changes: List[SystemChange],
                                       context: Dict[str, Any]) -> List[str]:
        """Generate mitigation suggestions for predicted effect."""
        suggestions = []
        
        category = effect_template['category']
        
        if category == EffectCategory.PERFORMANCE:
            if context.get('has_thermal_changes', False):
                suggestions.extend([
                    "Improve cooling or reduce system load",
                    "Monitor GPU temperature closely"
                ])
            if context.get('memory_usage_high', False):
                suggestions.extend([
                    "Free up GPU memory",
                    "Optimize memory usage in applications"
                ])
        
        elif category == EffectCategory.STABILITY:
            suggestions.extend([
                "Monitor system resources continuously",
                "Prepare for graceful process termination",
                "Ensure data is saved frequently"
            ])
        
        elif category == EffectCategory.CAPABILITY:
            suggestions.extend([
                "Test compatibility before full deployment",
                "Keep backup of working configuration",
                "Review dependency requirements"
            ])
        
        return suggestions
    
    def _generate_monitoring_recommendations(self, effect_template: Dict[str, Any], 
                                           changes: List[SystemChange],
                                           context: Dict[str, Any]) -> List[str]:
        """Generate monitoring recommendations."""
        recommendations = []
        
        # Base monitoring based on categories involved
        if 'nvidia_gpu' in context.get('categories', []):
            recommendations.extend([
                "Monitor GPU temperature and memory usage",
                "Watch for thermal throttling events"
            ])
        
        if 'processes' in context.get('categories', []):
            recommendations.extend([
                "Monitor process memory usage",
                "Watch for unexpected process terminations"
            ])
        
        if 'python_env' in context.get('categories', []):
            recommendations.extend([
                "Monitor for import errors",
                "Check application startup success"
            ])
        
        # Severity-based monitoring
        severity = effect_template['severity']
        if severity == EventSeverity.CRITICAL:
            recommendations.extend([
                "Set up automated alerts",
                "Increase monitoring frequency"
            ])
        
        return recommendations
    
    def _deduplicate_effects(self, effects: List[PredictedEffect]) -> List[PredictedEffect]:
        """Remove duplicate predictions."""
        seen_descriptions = set()
        unique_effects = []
        
        for effect in effects:
            if effect.description not in seen_descriptions:
                seen_descriptions.add(effect.description)
                unique_effects.append(effect)
            else:
                # Merge with existing effect if duplicate
                existing = next(e for e in unique_effects if e.description == effect.description)
                existing.probability = max(existing.probability, effect.probability)
                existing.confidence = max(existing.confidence, effect.confidence)
        
        return unique_effects
    
    def _rank_effects_by_risk(self, effects: List[PredictedEffect]) -> List[PredictedEffect]:
        """Rank effects by risk level."""
        def risk_score(effect):
            severity_weight = {
                EventSeverity.CRITICAL: 3.0,
                EventSeverity.WARNING: 2.0,
                EventSeverity.INFO: 1.0
            }
            
            return (effect.probability * effect.confidence * 
                   severity_weight.get(effect.severity, 1.0))
        
        return sorted(effects, key=risk_score, reverse=True)
    
    def _store_predictions(self, effects: List[PredictedEffect], 
                         changes: List[SystemChange], events: List[SystemEvent]) -> None:
        """Store predictions for learning and accuracy tracking."""
        prediction_record = {
            'timestamp': datetime.now(),
            'predictions': [
                {
                    'effect_id': effect.effect_id,
                    'probability': effect.probability,
                    'confidence': effect.confidence,
                    'expected_timeframe': effect.expected_timeframe,
                    'category': effect.category.value,
                    'severity': effect.severity.value
                }
                for effect in effects
            ],
            'trigger_changes': len(changes),
            'trigger_events': len(events)
        }
        
        self.effect_history.append(prediction_record)
        
        # Keep history manageable
        if len(self.effect_history) > 1000:
            self.effect_history = self.effect_history[-1000:]