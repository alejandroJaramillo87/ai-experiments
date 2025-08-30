"""
AI Workload Predictor

Advanced machine learning system for predicting workload performance characteristics
and resource requirements in the AI workstation environment. Uses historical data
from the temporal intelligence system to provide accurate performance forecasts
and resource optimization recommendations.

This predictor specializes in:
- Multi-model inference performance prediction
- Hardware-specific resource requirement estimation  
- Thermal-aware performance forecasting
- Container orchestration optimization
- Cross-workload interference prediction
"""

import logging
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import pickle
from collections import defaultdict, deque

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available - ML predictions will use heuristic methods")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class PredictionConfidence(Enum):
    """Confidence levels for workload predictions"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class WorkloadCategory(Enum):
    """Categories of AI workloads for specialized prediction"""
    LLM_INFERENCE = "llm_inference"
    LLM_TRAINING = "llm_training"
    VISION_INFERENCE = "vision_inference"
    VISION_TRAINING = "vision_training"
    MULTIMODAL = "multimodal"
    RAG_OPERATIONS = "rag_operations"
    FINE_TUNING = "fine_tuning"
    BATCH_PROCESSING = "batch_processing"
    INTERACTIVE = "interactive"
    UNKNOWN = "unknown"


class HardwareProfile(Enum):
    """Hardware utilization profiles for prediction context"""
    GPU_INTENSIVE = "gpu_intensive"
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    BALANCED = "balanced"
    THERMAL_CONSTRAINED = "thermal_constrained"


@dataclass
class WorkloadFeatures:
    """Feature vector for workload prediction"""
    # Workload characteristics
    model_size_gb: float
    batch_size: int
    sequence_length: int
    precision_bits: int
    workload_category: WorkloadCategory
    
    # Historical performance
    avg_gpu_utilization: float
    avg_cpu_utilization: float
    avg_memory_usage_gb: float
    avg_vram_usage_gb: float
    
    # System context
    concurrent_workloads: int
    thermal_state: float  # 0.0 = cool, 1.0 = hot
    time_of_day_factor: float
    day_of_week_factor: float
    
    # Hardware state
    gpu_temperature: float
    cpu_temperature: float
    available_vram_gb: float
    available_ram_gb: float
    
    def to_vector(self) -> np.ndarray:
        """Convert features to ML-compatible vector"""
        return np.array([
            self.model_size_gb,
            self.batch_size,
            self.sequence_length,
            self.precision_bits,
            self.workload_category.value.__hash__() % 100,  # Hash enum to number
            self.avg_gpu_utilization,
            self.avg_cpu_utilization,
            self.avg_memory_usage_gb,
            self.avg_vram_usage_gb,
            self.concurrent_workloads,
            self.thermal_state,
            self.time_of_day_factor,
            self.day_of_week_factor,
            self.gpu_temperature,
            self.cpu_temperature,
            self.available_vram_gb,
            self.available_ram_gb
        ])


@dataclass
class PerformancePrediction:
    """Predicted performance characteristics for a workload"""
    # Performance metrics
    predicted_throughput: float  # tokens/second or samples/second
    predicted_latency: float    # milliseconds
    predicted_memory_peak: float  # GB
    predicted_vram_peak: float   # GB
    
    # Resource requirements
    recommended_gpu_allocation: float  # 0.0-1.0
    recommended_cpu_cores: int
    recommended_memory_gb: float
    recommended_batch_size: int
    
    # Thermal predictions
    predicted_gpu_temp_increase: float
    predicted_cpu_temp_increase: float
    thermal_sustainability_score: float  # 0.0-1.0
    
    # Confidence and metadata
    confidence: PredictionConfidence
    confidence_score: float  # 0.0-1.0
    prediction_timestamp: datetime
    model_version: str
    
    # Optimization suggestions
    optimization_suggestions: List[str] = field(default_factory=list)
    bottleneck_predictions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'predicted_throughput': self.predicted_throughput,
            'predicted_latency': self.predicted_latency,
            'predicted_memory_peak': self.predicted_memory_peak,
            'predicted_vram_peak': self.predicted_vram_peak,
            'recommended_gpu_allocation': self.recommended_gpu_allocation,
            'recommended_cpu_cores': self.recommended_cpu_cores,
            'recommended_memory_gb': self.recommended_memory_gb,
            'recommended_batch_size': self.recommended_batch_size,
            'predicted_gpu_temp_increase': self.predicted_gpu_temp_increase,
            'predicted_cpu_temp_increase': self.predicted_cpu_temp_increase,
            'thermal_sustainability_score': self.thermal_sustainability_score,
            'confidence': self.confidence.value,
            'confidence_score': self.confidence_score,
            'prediction_timestamp': self.prediction_timestamp.isoformat(),
            'model_version': self.model_version,
            'optimization_suggestions': self.optimization_suggestions,
            'bottleneck_predictions': self.bottleneck_predictions
        }


class WorkloadPatternAnalyzer:
    """Analyzes historical workload patterns for prediction training"""
    
    def __init__(self):
        self.pattern_cache = {}
        self.seasonal_patterns = defaultdict(list)
        self.workload_clusters = {}
        
    def analyze_temporal_patterns(self, workload_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in workload performance"""
        patterns = {
            'hourly_trends': defaultdict(list),
            'daily_trends': defaultdict(list),
            'workload_interference': defaultdict(list),
            'thermal_correlations': defaultdict(list)
        }
        
        for record in workload_history:
            timestamp = datetime.fromisoformat(record.get('timestamp', datetime.now().isoformat()))
            hour = timestamp.hour
            weekday = timestamp.weekday()
            
            performance = record.get('performance_metrics', {})
            if performance:
                patterns['hourly_trends'][hour].append(performance.get('throughput', 0))
                patterns['daily_trends'][weekday].append(performance.get('latency', 0))
                
                # Analyze thermal correlation
                thermal_data = record.get('thermal_state', {})
                if thermal_data:
                    gpu_temp = thermal_data.get('gpu_temperature', 0)
                    throughput = performance.get('throughput', 0)
                    patterns['thermal_correlations']['gpu_temp_vs_throughput'].append((gpu_temp, throughput))
        
        return patterns
    
    def identify_workload_clusters(self, features_list: List[WorkloadFeatures]) -> Dict[int, List[int]]:
        """Identify clusters of similar workloads for specialized prediction"""
        if not SKLEARN_AVAILABLE or len(features_list) < 10:
            return {0: list(range(len(features_list)))}  # Single cluster fallback
        
        try:
            # Convert features to vectors
            feature_vectors = np.array([f.to_vector() for f in features_list])
            
            # Normalize features
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(feature_vectors)
            
            # Perform clustering
            n_clusters = min(5, len(features_list) // 5)  # Adaptive cluster count
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(normalized_features)
            
            # Group by clusters
            clusters = defaultdict(list)
            for idx, label in enumerate(cluster_labels):
                clusters[label].append(idx)
            
            return dict(clusters)
            
        except Exception as e:
            logging.warning(f"Clustering failed: {e}")
            return {0: list(range(len(features_list)))}


class MLPredictor:
    """Machine learning predictor using multiple algorithms"""
    
    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.feature_importance = {}
        self.model_performance = {}
        
    def train(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Train models for multiple prediction targets"""
        if not SKLEARN_AVAILABLE:
            logging.warning("Scikit-learn not available - using heuristic predictions")
            return {"heuristic_mode": 1.0}
        
        performance_metrics = {}
        
        try:
            # Split data
            X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
            
            # Train models for each target
            for target_name, y in y_dict.items():
                if len(y) != len(X):
                    continue
                    
                y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train ensemble of models
                models = {
                    'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                    'gbr': GradientBoostingRegressor(random_state=42),
                    'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
                }
                
                target_models = {}
                target_performance = {}
                
                for model_name, model in models.items():
                    try:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        
                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        target_models[model_name] = model
                        target_performance[model_name] = {
                            'r2': r2,
                            'mse': mse,
                            'mae': mae
                        }
                        
                    except Exception as e:
                        logging.warning(f"Training {model_name} for {target_name} failed: {e}")
                
                if target_models:
                    self.models[target_name] = target_models
                    self.scalers[target_name] = scaler
                    self.model_performance[target_name] = target_performance
                    
                    # Calculate average performance
                    avg_r2 = np.mean([p['r2'] for p in target_performance.values()])
                    performance_metrics[target_name] = avg_r2
                    
        except Exception as e:
            logging.error(f"Training failed: {e}")
            return {"error": 0.0}
        
        self.is_trained = len(self.models) > 0
        return performance_metrics
    
    def predict(self, X: np.ndarray, target: str) -> Tuple[np.ndarray, float]:
        """Predict with confidence estimation"""
        if not self.is_trained or target not in self.models:
            # Heuristic fallback
            return self._heuristic_predict(X, target)
        
        try:
            # Scale features
            X_scaled = self.scalers[target].transform(X)
            
            # Get predictions from all models
            predictions = []
            for model_name, model in self.models[target].items():
                pred = model.predict(X_scaled)
                predictions.append(pred)
            
            # Ensemble prediction (average)
            ensemble_pred = np.mean(predictions, axis=0)
            
            # Confidence based on agreement
            pred_std = np.std(predictions, axis=0)
            confidence = np.exp(-pred_std.mean())  # Higher agreement = higher confidence
            
            return ensemble_pred, confidence
            
        except Exception as e:
            logging.warning(f"ML prediction failed: {e}")
            return self._heuristic_predict(X, target)
    
    def _heuristic_predict(self, X: np.ndarray, target: str) -> Tuple[np.ndarray, float]:
        """Fallback heuristic predictions when ML is unavailable"""
        n_samples = X.shape[0]
        
        # Simple heuristics based on feature patterns
        if target == 'throughput':
            # Throughput roughly inversely proportional to model size
            model_sizes = X[:, 0] if X.shape[1] > 0 else np.ones(n_samples)
            pred = 1000.0 / np.maximum(model_sizes, 1.0)  # tokens/second
            confidence = 0.3
            
        elif target == 'latency':
            # Latency increases with model size and batch size
            model_sizes = X[:, 0] if X.shape[1] > 0 else np.ones(n_samples)
            batch_sizes = X[:, 1] if X.shape[1] > 1 else np.ones(n_samples)
            pred = model_sizes * batch_sizes * 0.1
            confidence = 0.3
            
        elif target == 'memory_peak':
            # Memory roughly proportional to model size
            model_sizes = X[:, 0] if X.shape[1] > 0 else np.ones(n_samples)
            pred = model_sizes * 1.2  # Add overhead
            confidence = 0.4
            
        elif target == 'vram_peak':
            # VRAM similar to memory but GPU-focused
            model_sizes = X[:, 0] if X.shape[1] > 0 else np.ones(n_samples)
            pred = model_sizes * 1.1
            confidence = 0.4
            
        else:
            pred = np.zeros(n_samples)
            confidence = 0.1
        
        return pred, confidence


class AIWorkloadPredictor:
    """
    Advanced AI workload performance predictor for the RTX 5090 + AMD 9950X workstation.
    
    Uses machine learning to predict workload performance characteristics, resource
    requirements, and thermal behavior based on historical system data.
    """
    
    def __init__(self, temporal_storage_path: Optional[str] = None):
        self.logger = logging.getLogger(f"{__name__}.AIWorkloadPredictor")
        
        # Core components
        self.pattern_analyzer = WorkloadPatternAnalyzer()
        self.ml_predictor = MLPredictor()
        
        # Training data and models
        self.training_data = deque(maxlen=10000)  # Keep recent training data
        self.prediction_history = deque(maxlen=1000)
        self.performance_cache = {}
        
        # Hardware-specific parameters (RTX 5090 + AMD 9950X)
        self.hardware_profile = {
            'gpu_memory_gb': 32.0,
            'system_memory_gb': 128.0,
            'gpu_cores': 21760,  # CUDA cores
            'cpu_cores': 16,
            'cpu_threads': 32,
            'max_gpu_temp': 83.0,
            'max_cpu_temp': 90.0,
            'thermal_design_power_gpu': 575,  # Watts
            'thermal_design_power_cpu': 170   # Watts
        }
        
        # Prediction models state
        self.model_version = "1.0.0"
        self.last_training = None
        self.training_lock = threading.Lock()
        
        # Performance tracking
        self.prediction_accuracy = defaultdict(list)
        self.calibration_data = defaultdict(list)
        
        self.logger.info("AIWorkloadPredictor initialized for RTX 5090 + AMD 9950X")
    
    def add_training_data(self, workload_features: WorkloadFeatures, 
                         actual_performance: Dict[str, float]) -> None:
        """Add new training data from completed workloads"""
        training_record = {
            'timestamp': datetime.now(),
            'features': workload_features,
            'performance': actual_performance,
            'hardware_state': self._get_current_hardware_state()
        }
        
        self.training_data.append(training_record)
        
        # Retrain periodically
        if len(self.training_data) % 100 == 0:
            self._schedule_retraining()
    
    def predict_workload_performance(self, workload_features: WorkloadFeatures) -> PerformancePrediction:
        """Predict performance characteristics for a given workload"""
        try:
            # Check cache first
            cache_key = self._generate_cache_key(workload_features)
            if cache_key in self.performance_cache:
                cached_result = self.performance_cache[cache_key]
                if (datetime.now() - cached_result['timestamp']).seconds < 300:  # 5 minute cache
                    return cached_result['prediction']
            
            # Extract features
            feature_vector = workload_features.to_vector().reshape(1, -1)
            
            # Predict core performance metrics
            throughput_pred, throughput_conf = self.ml_predictor.predict(feature_vector, 'throughput')
            latency_pred, latency_conf = self.ml_predictor.predict(feature_vector, 'latency')
            memory_pred, memory_conf = self.ml_predictor.predict(feature_vector, 'memory_peak')
            vram_pred, vram_conf = self.ml_predictor.predict(feature_vector, 'vram_peak')
            
            # Hardware-specific adjustments
            adjusted_predictions = self._apply_hardware_adjustments(
                throughput_pred[0], latency_pred[0], memory_pred[0], vram_pred[0],
                workload_features
            )
            
            # Generate resource recommendations
            recommendations = self._generate_resource_recommendations(
                workload_features, adjusted_predictions
            )
            
            # Predict thermal impact
            thermal_predictions = self._predict_thermal_impact(
                workload_features, adjusted_predictions
            )
            
            # Calculate overall confidence
            overall_confidence = np.mean([throughput_conf, latency_conf, memory_conf, vram_conf])
            confidence_level = self._categorize_confidence(overall_confidence)
            
            # Generate optimization suggestions
            optimizations = self._generate_optimization_suggestions(
                workload_features, adjusted_predictions, recommendations
            )
            
            # Create prediction result
            prediction = PerformancePrediction(
                predicted_throughput=adjusted_predictions['throughput'],
                predicted_latency=adjusted_predictions['latency'],
                predicted_memory_peak=adjusted_predictions['memory_peak'],
                predicted_vram_peak=adjusted_predictions['vram_peak'],
                recommended_gpu_allocation=recommendations['gpu_allocation'],
                recommended_cpu_cores=recommendations['cpu_cores'],
                recommended_memory_gb=recommendations['memory_gb'],
                recommended_batch_size=recommendations['batch_size'],
                predicted_gpu_temp_increase=thermal_predictions['gpu_temp_increase'],
                predicted_cpu_temp_increase=thermal_predictions['cpu_temp_increase'],
                thermal_sustainability_score=thermal_predictions['sustainability_score'],
                confidence=confidence_level,
                confidence_score=overall_confidence,
                prediction_timestamp=datetime.now(),
                model_version=self.model_version,
                optimization_suggestions=optimizations['suggestions'],
                bottleneck_predictions=optimizations['bottlenecks']
            )
            
            # Cache result
            self.performance_cache[cache_key] = {
                'prediction': prediction,
                'timestamp': datetime.now()
            }
            
            # Track prediction for accuracy monitoring
            self.prediction_history.append({
                'features': workload_features,
                'prediction': prediction,
                'timestamp': datetime.now()
            })
            
            self.logger.info(f"Generated workload prediction with {confidence_level.value} confidence")
            return prediction
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return self._generate_fallback_prediction(workload_features)
    
    def retrain_models(self, force: bool = False) -> Dict[str, float]:
        """Retrain prediction models with accumulated data"""
        if not force and len(self.training_data) < 50:
            return {"insufficient_data": 0.0}
        
        with self.training_lock:
            try:
                self.logger.info(f"Retraining models with {len(self.training_data)} samples")
                
                # Prepare training data
                X_list = []
                y_dict = defaultdict(list)
                
                for record in self.training_data:
                    features = record['features']
                    performance = record['performance']
                    
                    X_list.append(features.to_vector())
                    
                    # Extract target variables
                    y_dict['throughput'].append(performance.get('throughput', 0.0))
                    y_dict['latency'].append(performance.get('latency', 100.0))
                    y_dict['memory_peak'].append(performance.get('memory_peak', 1.0))
                    y_dict['vram_peak'].append(performance.get('vram_peak', 1.0))
                
                if not X_list:
                    return {"no_data": 0.0}
                
                X = np.array(X_list)
                y_arrays = {k: np.array(v) for k, v in y_dict.items()}
                
                # Train models
                performance_metrics = self.ml_predictor.train(X, y_arrays)
                self.last_training = datetime.now()
                
                # Analyze patterns
                workload_history = [
                    {
                        'timestamp': record['timestamp'].isoformat(),
                        'performance_metrics': record['performance'],
                        'thermal_state': record.get('hardware_state', {})
                    }
                    for record in self.training_data
                ]
                
                patterns = self.pattern_analyzer.analyze_temporal_patterns(workload_history)
                
                # Update model version
                self.model_version = f"1.0.{int(time.time())}"
                
                self.logger.info(f"Retraining completed. Performance: {performance_metrics}")
                return performance_metrics
                
            except Exception as e:
                self.logger.error(f"Retraining failed: {e}")
                return {"error": 0.0}
    
    def validate_prediction_accuracy(self) -> Dict[str, float]:
        """Validate prediction accuracy against actual results"""
        if len(self.prediction_history) < 10:
            return {"insufficient_validation_data": 0.0}
        
        accuracy_metrics = {}
        
        try:
            # Find predictions that have actual results
            validated_predictions = []
            
            for pred_record in self.prediction_history:
                # Look for matching actual results in training data
                pred_timestamp = pred_record['timestamp']
                
                for train_record in self.training_data:
                    train_timestamp = train_record['timestamp']
                    
                    # If training record is within 1 hour of prediction
                    if abs((train_timestamp - pred_timestamp).total_seconds()) < 3600:
                        validated_predictions.append({
                            'predicted': pred_record['prediction'],
                            'actual': train_record['performance']
                        })
                        break
            
            if not validated_predictions:
                return {"no_matches_found": 0.0}
            
            # Calculate accuracy for each metric
            metrics_to_validate = ['throughput', 'latency', 'memory_peak', 'vram_peak']
            
            for metric in metrics_to_validate:
                predicted_values = []
                actual_values = []
                
                for val_record in validated_predictions:
                    if hasattr(val_record['predicted'], f'predicted_{metric}'):
                        predicted_val = getattr(val_record['predicted'], f'predicted_{metric}')
                        actual_val = val_record['actual'].get(metric)
                        
                        if predicted_val is not None and actual_val is not None:
                            predicted_values.append(predicted_val)
                            actual_values.append(actual_val)
                
                if len(predicted_values) >= 5:
                    predicted_array = np.array(predicted_values)
                    actual_array = np.array(actual_values)
                    
                    # Calculate MAPE (Mean Absolute Percentage Error)
                    mape = np.mean(np.abs((actual_array - predicted_array) / actual_array)) * 100
                    accuracy = max(0, 100 - mape)  # Convert to accuracy percentage
                    
                    accuracy_metrics[f'{metric}_accuracy'] = accuracy
            
            # Overall accuracy
            if accuracy_metrics:
                accuracy_metrics['overall_accuracy'] = np.mean(list(accuracy_metrics.values()))
            
            self.logger.info(f"Prediction validation: {accuracy_metrics}")
            return accuracy_metrics
            
        except Exception as e:
            self.logger.error(f"Accuracy validation failed: {e}")
            return {"validation_error": 0.0}
    
    def get_workload_insights(self) -> Dict[str, Any]:
        """Get insights about workload patterns and predictions"""
        insights = {
            'training_data_size': len(self.training_data),
            'prediction_history_size': len(self.prediction_history),
            'model_version': self.model_version,
            'last_training': self.last_training.isoformat() if self.last_training else None,
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'hardware_profile': self.hardware_profile.copy()
        }
        
        # Pattern analysis
        if len(self.training_data) > 10:
            workload_categories = defaultdict(int)
            performance_trends = defaultdict(list)
            
            for record in list(self.training_data)[-100:]:  # Last 100 records
                features = record['features']
                performance = record['performance']
                
                workload_categories[features.workload_category.value] += 1
                performance_trends['throughput'].append(performance.get('throughput', 0))
                performance_trends['latency'].append(performance.get('latency', 0))
            
            insights['workload_distribution'] = dict(workload_categories)
            insights['performance_trends'] = {
                metric: {
                    'mean': float(np.mean(values)) if values else 0,
                    'std': float(np.std(values)) if values else 0,
                    'trend': self._calculate_trend(values) if len(values) > 5 else 'insufficient_data'
                }
                for metric, values in performance_trends.items()
            }
        
        # Model performance
        accuracy_metrics = self.validate_prediction_accuracy()
        insights['prediction_accuracy'] = accuracy_metrics
        
        return insights
    
    def _get_current_hardware_state(self) -> Dict[str, Any]:
        """Get current hardware state for context"""
        return {
            'timestamp': datetime.now().isoformat(),
            'gpu_temperature': 45.0,  # Placeholder - should integrate with thermal detectors
            'cpu_temperature': 35.0,  # Placeholder
            'available_vram_gb': 28.0,  # Placeholder
            'available_ram_gb': 100.0,  # Placeholder
            'gpu_utilization': 0.2,    # Placeholder
            'cpu_utilization': 0.15    # Placeholder
        }
    
    def _apply_hardware_adjustments(self, throughput: float, latency: float, 
                                  memory: float, vram: float,
                                  features: WorkloadFeatures) -> Dict[str, float]:
        """Apply RTX 5090 + AMD 9950X specific adjustments"""
        
        # RTX 5090 throughput boost for supported operations
        if features.workload_category in [WorkloadCategory.LLM_INFERENCE, WorkloadCategory.VISION_INFERENCE]:
            # Blackwell architecture provides significant inference speedup
            throughput *= 1.35  # ~35% boost from Blackwell improvements
            
        # AMD 9950X CPU optimizations
        if features.workload_category in [WorkloadCategory.RAG_OPERATIONS, WorkloadCategory.BATCH_PROCESSING]:
            # Zen 5 provides better CPU performance
            throughput *= 1.15  # ~15% boost from Zen 5 improvements
            
        # Memory bandwidth adjustments for 128GB setup
        if memory > 64.0:  # Large memory workloads
            # High memory bandwidth setup optimization
            latency *= 0.95  # 5% latency reduction
            
        # VRAM optimization for 32GB RTX 5090
        if vram > 24.0:  # Large VRAM workloads that benefit from 32GB
            throughput *= 1.10  # 10% throughput boost from not hitting VRAM limits
            
        # Thermal throttling adjustments
        thermal_factor = 1.0
        if features.thermal_state > 0.7:  # High thermal state
            thermal_factor = 0.95  # 5% performance reduction under thermal stress
        elif features.thermal_state < 0.3:  # Cool state
            thermal_factor = 1.02  # 2% boost when cool
            
        throughput *= thermal_factor
        latency /= thermal_factor  # Lower thermal stress = lower latency
        
        return {
            'throughput': throughput,
            'latency': latency,
            'memory_peak': memory,
            'vram_peak': vram
        }
    
    def _generate_resource_recommendations(self, features: WorkloadFeatures,
                                         predictions: Dict[str, float]) -> Dict[str, Any]:
        """Generate resource allocation recommendations"""
        
        # GPU allocation based on workload type and VRAM requirements
        if predictions['vram_peak'] > 24.0:  # Utilize 32GB VRAM advantage
            gpu_allocation = 1.0  # Full GPU dedication
        elif predictions['vram_peak'] > 16.0:
            gpu_allocation = 0.8
        elif predictions['vram_peak'] > 8.0:
            gpu_allocation = 0.6
        else:
            gpu_allocation = 0.4
            
        # CPU cores based on workload parallelization
        if features.workload_category in [WorkloadCategory.BATCH_PROCESSING, WorkloadCategory.RAG_OPERATIONS]:
            cpu_cores = min(16, max(4, int(predictions['memory_peak'] / 8)))  # Scale with memory usage
        elif features.workload_category == WorkloadCategory.LLM_TRAINING:
            cpu_cores = 12  # Balance for training workloads
        else:
            cpu_cores = 6   # Conservative for inference
            
        # Memory allocation with 128GB availability
        memory_gb = min(120.0, max(8.0, predictions['memory_peak'] * 1.2))  # 20% overhead
        
        # Optimal batch size based on hardware capabilities
        if features.workload_category == WorkloadCategory.LLM_INFERENCE:
            # Optimize for RTX 5090's tensor cores
            if predictions['vram_peak'] < 16.0:
                batch_size = min(32, features.batch_size * 2)  # Can handle larger batches
            else:
                batch_size = features.batch_size
        else:
            batch_size = features.batch_size
            
        return {
            'gpu_allocation': gpu_allocation,
            'cpu_cores': cpu_cores,
            'memory_gb': memory_gb,
            'batch_size': batch_size
        }
    
    def _predict_thermal_impact(self, features: WorkloadFeatures,
                               predictions: Dict[str, float]) -> Dict[str, float]:
        """Predict thermal impact of workload"""
        
        # Base thermal impact estimation
        gpu_power_estimate = predictions['vram_peak'] / 32.0 * 575  # Scale with VRAM usage
        cpu_power_estimate = predictions['memory_peak'] / 128.0 * 170  # Scale with memory usage
        
        # Temperature increase estimation (simplified model)
        gpu_temp_increase = gpu_power_estimate / 575 * 25.0  # Max ~25°C increase at full load
        cpu_temp_increase = cpu_power_estimate / 170 * 20.0   # Max ~20°C increase at full load
        
        # Workload-specific adjustments
        if features.workload_category in [WorkloadCategory.LLM_TRAINING, WorkloadCategory.FINE_TUNING]:
            gpu_temp_increase *= 1.2  # Training generates more heat
            
        # Current thermal state influence
        base_temp_factor = 1.0 + features.thermal_state * 0.3  # Higher base temps = more impact
        gpu_temp_increase *= base_temp_factor
        cpu_temp_increase *= base_temp_factor
        
        # Sustainability score (how long can this workload run without throttling)
        max_sustainable_gpu_temp = self.hardware_profile['max_gpu_temp'] - 8  # Safety margin
        max_sustainable_cpu_temp = self.hardware_profile['max_cpu_temp'] - 10
        
        projected_gpu_temp = features.gpu_temperature + gpu_temp_increase
        projected_cpu_temp = features.cpu_temperature + cpu_temp_increase
        
        gpu_sustainability = max(0, 1.0 - (projected_gpu_temp - max_sustainable_gpu_temp) / 10.0)
        cpu_sustainability = max(0, 1.0 - (projected_cpu_temp - max_sustainable_cpu_temp) / 10.0)
        
        sustainability_score = min(gpu_sustainability, cpu_sustainability)
        
        return {
            'gpu_temp_increase': gpu_temp_increase,
            'cpu_temp_increase': cpu_temp_increase,
            'sustainability_score': sustainability_score
        }
    
    def _generate_optimization_suggestions(self, features: WorkloadFeatures,
                                         predictions: Dict[str, float],
                                         recommendations: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate optimization suggestions"""
        suggestions = []
        bottlenecks = []
        
        # VRAM optimization
        if predictions['vram_peak'] > 28.0:  # Close to 32GB limit
            suggestions.append("Consider model quantization to reduce VRAM usage")
            suggestions.append("Enable gradient checkpointing for training workloads")
            bottlenecks.append("VRAM capacity approaching limit")
            
        # Memory optimization
        if predictions['memory_peak'] > 100.0:  # Using >100GB of 128GB
            suggestions.append("Monitor memory usage - approaching system limits")
            bottlenecks.append("System memory usage very high")
            
        # Thermal optimization
        thermal_predictions = self._predict_thermal_impact(features, predictions)
        if thermal_predictions['sustainability_score'] < 0.7:
            suggestions.append("Consider reducing batch size for thermal sustainability")
            suggestions.append("Enable thermal throttling protection")
            bottlenecks.append("Thermal constraints may limit performance")
            
        # Performance optimization
        if features.workload_category == WorkloadCategory.LLM_INFERENCE:
            if predictions['throughput'] < 500:  # Low throughput
                suggestions.append("Enable tensor core optimization for Blackwell architecture")
                suggestions.append("Consider using FP16 precision for inference")
                
        # Concurrent workload optimization
        if features.concurrent_workloads > 2:
            suggestions.append("High concurrency detected - consider workload scheduling")
            bottlenecks.append("Resource contention from concurrent workloads")
            
        # Hardware-specific suggestions
        if features.workload_category in [WorkloadCategory.RAG_OPERATIONS, WorkloadCategory.BATCH_PROCESSING]:
            suggestions.append("Leverage AMD Zen 5 AOCL optimizations for CPU workloads")
            
        return {
            'suggestions': suggestions,
            'bottlenecks': bottlenecks
        }
    
    def _categorize_confidence(self, confidence_score: float) -> PredictionConfidence:
        """Categorize numerical confidence into confidence levels"""
        if confidence_score > 0.9:
            return PredictionConfidence.VERY_HIGH
        elif confidence_score > 0.75:
            return PredictionConfidence.HIGH
        elif confidence_score > 0.5:
            return PredictionConfidence.MEDIUM
        elif confidence_score > 0.25:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.VERY_LOW
    
    def _generate_cache_key(self, features: WorkloadFeatures) -> str:
        """Generate cache key for performance predictions"""
        key_components = [
            str(features.model_size_gb),
            str(features.batch_size),
            str(features.workload_category.value),
            str(int(features.thermal_state * 10)),  # Discretize thermal state
            str(features.concurrent_workloads)
        ]
        return "|".join(key_components)
    
    def _generate_fallback_prediction(self, features: WorkloadFeatures) -> PerformancePrediction:
        """Generate fallback prediction when ML prediction fails"""
        return PerformancePrediction(
            predicted_throughput=100.0,  # Conservative estimate
            predicted_latency=200.0,     # Conservative estimate
            predicted_memory_peak=features.model_size_gb * 1.5,
            predicted_vram_peak=features.model_size_gb * 1.2,
            recommended_gpu_allocation=0.5,
            recommended_cpu_cores=4,
            recommended_memory_gb=16.0,
            recommended_batch_size=features.batch_size,
            predicted_gpu_temp_increase=10.0,
            predicted_cpu_temp_increase=5.0,
            thermal_sustainability_score=0.8,
            confidence=PredictionConfidence.LOW,
            confidence_score=0.2,
            prediction_timestamp=datetime.now(),
            model_version=self.model_version,
            optimization_suggestions=["Fallback prediction - limited optimization available"],
            bottleneck_predictions=["Prediction system unavailable"]
        )
    
    def _schedule_retraining(self) -> None:
        """Schedule model retraining in background thread"""
        def retrain_worker():
            try:
                time.sleep(1)  # Brief delay to avoid blocking
                self.retrain_models()
            except Exception as e:
                self.logger.error(f"Background retraining failed: {e}")
                
        thread = threading.Thread(target=retrain_worker, daemon=True)
        thread.start()
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate performance cache hit rate"""
        # Simplified calculation - in practice would track hits/misses
        return 0.75  # Placeholder
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from time series values"""
        if len(values) < 2:
            return "insufficient_data"
            
        # Simple linear trend calculation
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"