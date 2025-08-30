"""
AI Model Lifecycle Detector - Intelligent Model Management Monitoring
=====================================================================

Specialized detector for monitoring AI model lifecycle across the AI workstation
infrastructure. Tracks model loading, unloading, switching, and performance across
llama-cpu, llama-gpu, and vLLM services with deep understanding of model management,
memory allocation patterns, and inference performance optimization.

Key Capabilities:
- Model loading/unloading detection across CPU and GPU services
- Memory allocation pattern analysis for large language models
- Model switching and warm-up time tracking
- Inference performance correlation with model characteristics
- Model versioning and update detection
- Resource utilization optimization per model type
- Cache effectiveness and model persistence analysis
- Cross-service model sharing and optimization opportunities
"""

import asyncio
import json
import logging
import subprocess
import psutil
import requests
import os
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path

from ..base_collector import BaseCollector
from ...temporal.types import SystemChange, ChangeType

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Comprehensive model metadata and characteristics."""
    model_id: str
    model_name: str
    model_family: str  # 'llama', 'gpt', 'mistral', etc.
    model_size: str   # '7B', '13B', '70B', etc.
    model_type: str   # 'base', 'instruct', 'chat', 'code', etc.
    precision: str    # 'fp16', 'fp32', 'int8', 'int4', etc.
    estimated_memory_gb: float
    context_length: int
    architecture: str  # 'transformer', 'mixture_of_experts', etc.
    quantization: Optional[str]
    optimization_flags: List[str]


@dataclass
class ModelInstance:
    """Active model instance running on a service."""
    model_metadata: ModelMetadata
    service_name: str
    service_type: str  # 'llama-cpu', 'llama-gpu', 'vllm'
    container_id: str
    load_timestamp: datetime
    memory_allocated_mb: float
    gpu_memory_allocated_mb: float
    load_time_seconds: float
    warmup_completed: bool
    status: str  # 'loading', 'ready', 'error', 'unloading'
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    resource_utilization: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformanceProfile:
    """Model performance characteristics and benchmarks."""
    model_id: str
    service_type: str
    tokens_per_second: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_requests_per_second: float
    memory_efficiency: float
    inference_accuracy: Optional[float]
    power_efficiency_score: float
    optimal_batch_size: int
    concurrent_request_capacity: int


@dataclass
class ModelLoadEvent:
    """Model lifecycle event tracking."""
    event_type: str  # 'load', 'unload', 'switch', 'warmup', 'error'
    model_metadata: ModelMetadata
    service_name: str
    timestamp: datetime
    duration_seconds: Optional[float]
    memory_delta_mb: float
    gpu_memory_delta_mb: float
    success: bool
    error_details: Optional[str]
    performance_impact: Dict[str, Any]
    resource_correlation: Dict[str, Any]


class AIModelLifecycleDetector:
    """
    Specialized detector for AI model lifecycle intelligence.
    
    Monitors model management across CPU and GPU inference services,
    providing insights into model performance, resource optimization,
    and lifecycle management efficiency.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize AI model lifecycle detector."""
        self.config = config or {}
        
        # AI service endpoints for model monitoring
        self.service_endpoints = {
            'llama-cpu-1': {'host': 'localhost', 'port': 8001, 'health_path': '/health'},
            'llama-cpu-2': {'host': 'localhost', 'port': 8002, 'health_path': '/health'},
            'llama-cpu-3': {'host': 'localhost', 'port': 8003, 'health_path': '/health'},
            'llama-gpu': {'host': 'localhost', 'port': 8004, 'health_path': '/health'},
            'vllm': {'host': 'localhost', 'port': 8000, 'health_path': '/health'}
        }
        
        # Model directories and paths for monitoring
        self.model_paths = [
            '/mnt/ai-data/models',
            '/opt/models',
            '/home/user/models',
            '/tmp/model_cache'
        ]
        
        # Known model patterns and characteristics
        self.model_patterns = {
            'llama': {
                'family': 'llama',
                'memory_per_billion': 2.0,  # GB per billion parameters (fp16)
                'context_lengths': [2048, 4096, 8192, 32768],
                'typical_sizes': ['7B', '13B', '30B', '65B', '70B']
            },
            'mistral': {
                'family': 'mistral',
                'memory_per_billion': 2.0,
                'context_lengths': [8192, 32768],
                'typical_sizes': ['7B', '8x7B', '8x22B']
            },
            'codellama': {
                'family': 'codellama',
                'memory_per_billion': 2.0,
                'context_lengths': [4096, 16384],
                'typical_sizes': ['7B', '13B', '34B']
            }
        }
        
        # Performance thresholds for model optimization
        self.thresholds = {
            'load_time_warning': 120,  # seconds
            'load_time_critical': 300,
            'memory_efficiency_low': 60,  # percentage
            'latency_high_ms': 5000,
            'throughput_low_rps': 0.1,
            'tokens_per_second_low': 10,
            'cache_hit_rate_low': 70
        }
        
        # Historical tracking
        self.active_models: Dict[str, ModelInstance] = {}
        self.model_history = deque(maxlen=100)
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.load_events: List[ModelLoadEvent] = []
        
        logger.info("AIModelLifecycleDetector initialized for AI workstation monitoring")
    
    async def collect_model_lifecycle_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive AI model lifecycle metrics."""
        try:
            # Discover active models across services
            active_models = await self._discover_active_models()
            
            # Analyze model performance across services
            performance_analysis = await self._analyze_model_performance()
            
            # Memory utilization analysis
            memory_analysis = await self._analyze_model_memory_usage()
            
            # Model lifecycle events
            lifecycle_events = await self._detect_lifecycle_events()
            
            # Cross-service optimization analysis
            optimization_analysis = await self._analyze_cross_service_optimization()
            
            # Model cache effectiveness
            cache_analysis = await self._analyze_model_caching()
            
            # Resource correlation analysis
            resource_correlation = await self._analyze_resource_correlation()
            
            # Generate recommendations
            recommendations = await self._generate_model_recommendations(
                active_models, performance_analysis, memory_analysis
            )
            
            return {
                'active_models': active_models,
                'model_performance_analysis': performance_analysis,
                'memory_utilization_analysis': memory_analysis,
                'lifecycle_events': lifecycle_events,
                'cross_service_optimization': optimization_analysis,
                'model_cache_analysis': cache_analysis,
                'resource_correlation': resource_correlation,
                'optimization_recommendations': recommendations,
                'service_health': await self._check_service_health(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error collecting model lifecycle metrics: {e}")
            return {'error': str(e)}
    
    async def _discover_active_models(self) -> Dict[str, Any]:
        """Discover active models across all AI services."""
        try:
            discovered_models = {}
            service_model_count = {}
            
            for service_name, endpoint_config in self.service_endpoints.items():
                models = await self._discover_service_models(service_name, endpoint_config)
                if models:
                    discovered_models[service_name] = models
                    service_model_count[service_name] = len(models)
            
            # Analyze model distribution across services
            total_models = sum(service_model_count.values())
            model_families = defaultdict(int)
            model_sizes = defaultdict(int)
            
            for service_models in discovered_models.values():
                for model in service_models:
                    if 'metadata' in model:
                        family = model['metadata'].get('model_family', 'unknown')
                        size = model['metadata'].get('model_size', 'unknown')
                        model_families[family] += 1
                        model_sizes[size] += 1
            
            return {
                'services_with_models': len(discovered_models),
                'total_active_models': total_models,
                'model_distribution': service_model_count,
                'model_families': dict(model_families),
                'model_sizes': dict(model_sizes),
                'detailed_models': discovered_models,
                'discovery_summary': {
                    'cpu_services_active': len([s for s in discovered_models.keys() if 'cpu' in s]),
                    'gpu_services_active': len([s for s in discovered_models.keys() if 'gpu' in s or s == 'vllm']),
                    'largest_model_detected': max(model_sizes.keys(), key=lambda x: self._estimate_model_params(x)) if model_sizes else 'none'
                }
            }
            
        except Exception as e:
            logger.error(f"Error discovering active models: {e}")
            return {'error': str(e)}
    
    async def _discover_service_models(self, service_name: str, endpoint_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover models for a specific service."""
        models = []
        
        try:
            # Try to get model information via API
            health_url = f"http://{endpoint_config['host']}:{endpoint_config['port']}{endpoint_config['health_path']}"
            models_url = f"http://{endpoint_config['host']}:{endpoint_config['port']}/v1/models"
            
            # Check service health first
            try:
                health_response = requests.get(health_url, timeout=5)
                if health_response.status_code != 200:
                    logger.warning(f"Service {service_name} health check failed")
                    return []
            except requests.RequestException:
                # Service might not be available, try alternative detection
                return await self._discover_models_via_process_analysis(service_name)
            
            # Try to get models list
            try:
                models_response = requests.get(models_url, timeout=10)
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    
                    if 'data' in models_data:
                        for model_info in models_data['data']:
                            model_metadata = self._parse_model_metadata(model_info)
                            models.append({
                                'model_id': model_info.get('id', 'unknown'),
                                'metadata': model_metadata,
                                'status': 'active',
                                'api_detected': True
                            })
                
            except (requests.RequestException, json.JSONDecodeError):
                # API not available, try process analysis
                process_models = await self._discover_models_via_process_analysis(service_name)
                models.extend(process_models)
            
        except Exception as e:
            logger.warning(f"Error discovering models for {service_name}: {e}")
        
        return models
    
    async def _discover_models_via_process_analysis(self, service_name: str) -> List[Dict[str, Any]]:
        """Discover models by analyzing container processes."""
        models = []
        
        try:
            # Look for container processes related to the service
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
                try:
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    
                    # Check if process belongs to this service
                    if (service_name.replace('-', '_') in cmdline or
                        service_name in proc.info['name'] or
                        any(service_name.split('-')[0] in part for part in cmdline.split())):
                        
                        # Look for model paths or model names in command line
                        model_indicators = self._extract_model_indicators_from_cmdline(cmdline)
                        
                        if model_indicators:
                            for indicator in model_indicators:
                                model_metadata = self._infer_model_metadata_from_path(indicator)
                                models.append({
                                    'model_id': indicator.split('/')[-1] if '/' in indicator else indicator,
                                    'metadata': model_metadata,
                                    'status': 'inferred',
                                    'api_detected': False,
                                    'process_memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                                    'detection_method': 'process_analysis'
                                })
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
        except Exception as e:
            logger.warning(f"Error in process analysis for {service_name}: {e}")
        
        return models
    
    def _extract_model_indicators_from_cmdline(self, cmdline: str) -> List[str]:
        """Extract model indicators from command line."""
        indicators = []
        
        # Common model path patterns
        model_path_patterns = [
            r'--model[_-]path[=\s]+([^\s]+)',
            r'--model[=\s]+([^\s]+)',
            r'--checkpoint[=\s]+([^\s]+)',
            r'/[^\s]*models/[^\s]*',
            r'huggingface\.co/([^\s/]+/[^\s/]+)',
            r'microsoft/([^\s/]+)',
            r'meta-llama/([^\s/]+)',
            r'mistralai/([^\s/]+)'
        ]
        
        for pattern in model_path_patterns:
            matches = re.findall(pattern, cmdline, re.IGNORECASE)
            indicators.extend(matches)
        
        # Remove duplicates
        return list(set(indicators))
    
    def _parse_model_metadata(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse model metadata from API response."""
        model_id = model_info.get('id', 'unknown')
        model_name = model_info.get('object', model_id)
        
        # Infer metadata from model name/id
        metadata = self._infer_model_metadata_from_name(model_name)
        
        return {
            'model_id': model_id,
            'model_name': model_name,
            'model_family': metadata['family'],
            'model_size': metadata['size'],
            'model_type': metadata['type'],
            'estimated_memory_gb': metadata['estimated_memory_gb'],
            'context_length': metadata['context_length'],
            'precision': metadata['precision']
        }
    
    def _infer_model_metadata_from_name(self, model_name: str) -> Dict[str, Any]:
        """Infer model metadata from model name."""
        metadata = {
            'family': 'unknown',
            'size': 'unknown',
            'type': 'base',
            'estimated_memory_gb': 0.0,
            'context_length': 2048,
            'precision': 'fp16'
        }
        
        name_lower = model_name.lower()
        
        # Detect model family
        for family in self.model_patterns.keys():
            if family in name_lower:
                metadata['family'] = family
                break
        
        # Detect model size
        size_patterns = [r'(\d+)b', r'(\d+\.?\d*)b', r'(\d+x\d+)b']
        for pattern in size_patterns:
            match = re.search(pattern, name_lower)
            if match:
                metadata['size'] = match.group(1) + 'B'
                # Estimate memory usage
                try:
                    if 'x' in match.group(1):  # MoE model
                        params = float(match.group(1).split('x')[1])
                    else:
                        params = float(match.group(1))
                    
                    # Estimate memory (fp16 = 2 bytes per parameter + overhead)
                    metadata['estimated_memory_gb'] = params * 2.2  # 2 bytes + 10% overhead
                except ValueError:
                    metadata['estimated_memory_gb'] = 7.0  # Default estimate
                break
        
        # Detect model type
        if any(keyword in name_lower for keyword in ['instruct', 'chat', 'conversation']):
            metadata['type'] = 'instruct'
        elif 'code' in name_lower:
            metadata['type'] = 'code'
        
        # Detect context length
        context_patterns = [r'(\d+)k', r'context[_-]?(\d+)']
        for pattern in context_patterns:
            match = re.search(pattern, name_lower)
            if match:
                try:
                    if 'k' in match.group(0):
                        metadata['context_length'] = int(match.group(1)) * 1024
                    else:
                        metadata['context_length'] = int(match.group(1))
                except ValueError:
                    pass
                break
        
        return metadata
    
    def _infer_model_metadata_from_path(self, path: str) -> Dict[str, Any]:
        """Infer model metadata from file path."""
        # Extract model name from path
        model_name = Path(path).name if '/' in path else path
        return self._infer_model_metadata_from_name(model_name)
    
    def _estimate_model_params(self, size_str: str) -> float:
        """Estimate model parameters from size string."""
        if not size_str or size_str == 'unknown':
            return 0.0
        
        size_str = size_str.lower().replace('b', '')
        
        try:
            if 'x' in size_str:  # MoE model
                parts = size_str.split('x')
                return float(parts[1]) if len(parts) > 1 else 0.0
            else:
                return float(size_str)
        except ValueError:
            return 0.0
    
    async def _analyze_model_performance(self) -> Dict[str, Any]:
        """Analyze model performance across services."""
        try:
            performance_analysis = {
                'service_performance': {},
                'model_benchmarks': {},
                'performance_trends': {},
                'bottleneck_analysis': []
            }
            
            # Analyze performance for each service
            for service_name, endpoint_config in self.service_endpoints.items():
                service_perf = await self._analyze_service_performance(service_name, endpoint_config)
                if service_perf:
                    performance_analysis['service_performance'][service_name] = service_perf
            
            # Generate performance trends
            if self.performance_history:
                for service_name, history in self.performance_history.items():
                    if len(history) > 1:
                        recent_metrics = list(history)[-5:]  # Last 5 measurements
                        avg_latency = sum(m.get('latency_ms', 0) for m in recent_metrics) / len(recent_metrics)
                        avg_throughput = sum(m.get('throughput_rps', 0) for m in recent_metrics) / len(recent_metrics)
                        
                        performance_analysis['performance_trends'][service_name] = {
                            'average_latency_ms': round(avg_latency, 2),
                            'average_throughput_rps': round(avg_throughput, 2),
                            'trend_quality': 'good' if avg_latency < 2000 and avg_throughput > 0.5 else 'poor'
                        }
            
            # Identify bottlenecks
            bottlenecks = []
            for service_name, perf in performance_analysis['service_performance'].items():
                if perf.get('latency_p95_ms', 0) > self.thresholds['latency_high_ms']:
                    bottlenecks.append({
                        'service': service_name,
                        'type': 'high_latency',
                        'value': perf['latency_p95_ms'],
                        'threshold': self.thresholds['latency_high_ms']
                    })
                
                if perf.get('throughput_rps', 0) < self.thresholds['throughput_low_rps']:
                    bottlenecks.append({
                        'service': service_name,
                        'type': 'low_throughput',
                        'value': perf['throughput_rps'],
                        'threshold': self.thresholds['throughput_low_rps']
                    })
            
            performance_analysis['bottleneck_analysis'] = bottlenecks
            
            return performance_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing model performance: {e}")
            return {'error': str(e)}
    
    async def _analyze_service_performance(self, service_name: str, endpoint_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze performance for a specific service."""
        try:
            # Try to get performance metrics via API
            metrics_url = f"http://{endpoint_config['host']}:{endpoint_config['port']}/metrics"
            stats_url = f"http://{endpoint_config['host']}:{endpoint_config['port']}/stats"
            
            performance_data = {
                'service_name': service_name,
                'status': 'unknown',
                'latency_metrics': {},
                'throughput_metrics': {},
                'resource_utilization': {}
            }
            
            # Try metrics endpoint
            try:
                metrics_response = requests.get(metrics_url, timeout=5)
                if metrics_response.status_code == 200:
                    metrics_text = metrics_response.text
                    performance_data['latency_metrics'] = self._parse_prometheus_metrics(metrics_text, 'latency')
                    performance_data['throughput_metrics'] = self._parse_prometheus_metrics(metrics_text, 'throughput')
                    performance_data['status'] = 'active'
            except requests.RequestException:
                pass
            
            # Try stats endpoint
            try:
                stats_response = requests.get(stats_url, timeout=5)
                if stats_response.status_code == 200:
                    stats_data = stats_response.json()
                    performance_data.update(self._parse_service_stats(stats_data))
                    performance_data['status'] = 'active'
            except (requests.RequestException, json.JSONDecodeError):
                pass
            
            # If no API metrics available, estimate from system resources
            if performance_data['status'] == 'unknown':
                system_metrics = self._estimate_performance_from_system(service_name)
                if system_metrics:
                    performance_data.update(system_metrics)
                    performance_data['status'] = 'estimated'
            
            return performance_data if performance_data['status'] != 'unknown' else None
            
        except Exception as e:
            logger.warning(f"Error analyzing performance for {service_name}: {e}")
            return None
    
    def _parse_prometheus_metrics(self, metrics_text: str, metric_type: str) -> Dict[str, float]:
        """Parse Prometheus metrics from response text."""
        parsed_metrics = {}
        
        if metric_type == 'latency':
            # Look for latency-related metrics
            latency_patterns = [
                r'request_duration_seconds_sum\s+([0-9.]+)',
                r'request_duration_seconds_count\s+([0-9.]+)',
                r'inference_latency_ms\s+([0-9.]+)',
                r'token_latency_ms\s+([0-9.]+)'
            ]
            
            for pattern in latency_patterns:
                matches = re.findall(pattern, metrics_text)
                if matches:
                    parsed_metrics[pattern.split('\\')[0]] = float(matches[0])
        
        elif metric_type == 'throughput':
            # Look for throughput-related metrics
            throughput_patterns = [
                r'requests_per_second\s+([0-9.]+)',
                r'tokens_per_second\s+([0-9.]+)',
                r'inference_throughput\s+([0-9.]+)'
            ]
            
            for pattern in throughput_patterns:
                matches = re.findall(pattern, metrics_text)
                if matches:
                    parsed_metrics[pattern.split('\\')[0]] = float(matches[0])
        
        return parsed_metrics
    
    def _parse_service_stats(self, stats_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse service statistics from API response."""
        parsed_stats = {}
        
        # Common stat fields
        stat_mappings = {
            'requests_processed': 'requests_processed',
            'avg_latency_ms': 'latency_avg_ms',
            'p95_latency_ms': 'latency_p95_ms',
            'p99_latency_ms': 'latency_p99_ms',
            'throughput_rps': 'throughput_rps',
            'tokens_per_second': 'tokens_per_second',
            'active_requests': 'active_requests',
            'queue_length': 'queue_length'
        }
        
        for api_field, internal_field in stat_mappings.items():
            if api_field in stats_data:
                parsed_stats[internal_field] = stats_data[api_field]
        
        return parsed_stats
    
    def _estimate_performance_from_system(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Estimate performance from system resource usage."""
        try:
            # Find processes related to the service
            service_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                try:
                    if (service_name.replace('-', '_') in proc.info['name'] or
                        any(service_name.split('-')[0] in part for part in proc.info['cmdline'] if proc.info['cmdline'])):
                        service_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not service_processes:
                return None
            
            # Calculate estimated metrics
            total_cpu = sum(p['cpu_percent'] for p in service_processes)
            total_memory_mb = sum(p['memory_info'].rss / 1024 / 1024 for p in service_processes)
            
            # Rough performance estimation based on resource usage
            estimated_metrics = {
                'estimated_from_system': True,
                'total_cpu_percent': round(total_cpu, 2),
                'total_memory_mb': round(total_memory_mb, 2),
                'process_count': len(service_processes),
                'estimated_throughput_rps': max(0.1, total_cpu / 100),  # Very rough estimate
                'estimated_latency_ms': max(100, (100 - total_cpu) * 50)  # Inverse relationship
            }
            
            return estimated_metrics
            
        except Exception as e:
            logger.warning(f"Error estimating performance for {service_name}: {e}")
            return None
    
    async def _analyze_model_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns for loaded models."""
        try:
            memory_analysis = {
                'total_model_memory_gb': 0.0,
                'memory_efficiency': 0.0,
                'memory_distribution': {},
                'memory_optimization_opportunities': []
            }
            
            # Analyze memory usage per service
            service_memory = {}
            total_memory_mb = 0
            
            for service_name in self.service_endpoints.keys():
                # Get memory usage for service processes
                service_mem = await self._get_service_memory_usage(service_name)
                if service_mem:
                    service_memory[service_name] = service_mem
                    total_memory_mb += service_mem['total_memory_mb']
            
            memory_analysis['memory_distribution'] = service_memory
            memory_analysis['total_model_memory_gb'] = total_memory_mb / 1024
            
            # Calculate memory efficiency
            system_memory = psutil.virtual_memory()
            memory_efficiency = (total_memory_mb / (system_memory.total / 1024 / 1024)) * 100
            memory_analysis['memory_efficiency'] = round(memory_efficiency, 2)
            
            # Identify optimization opportunities
            optimizations = []
            
            if memory_efficiency > 85:
                optimizations.append({
                    'type': 'memory_pressure',
                    'description': 'High memory utilization detected',
                    'recommendation': 'Consider model quantization or smaller models'
                })
            
            # Check for memory imbalance across services
            if service_memory:
                memory_values = [s['total_memory_mb'] for s in service_memory.values()]
                if max(memory_values) > 2 * min(memory_values):  # 2x imbalance
                    optimizations.append({
                        'type': 'memory_imbalance',
                        'description': 'Uneven memory distribution across services',
                        'recommendation': 'Consider redistributing models for better balance'
                    })
            
            memory_analysis['memory_optimization_opportunities'] = optimizations
            
            return memory_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing model memory usage: {e}")
            return {'error': str(e)}
    
    async def _get_service_memory_usage(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get memory usage for a specific service."""
        try:
            service_processes = []
            
            # Find processes for the service
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'memory_percent']):
                try:
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    
                    if (service_name.replace('-', '_') in cmdline or
                        service_name in proc.info['name'] or
                        any(service_name.split('-')[0] in part for part in cmdline.split())):
                        
                        service_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                            'memory_percent': proc.info['memory_percent']
                        })
                
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not service_processes:
                return None
            
            total_memory_mb = sum(p['memory_mb'] for p in service_processes)
            total_memory_percent = sum(p['memory_percent'] for p in service_processes)
            
            return {
                'service_name': service_name,
                'process_count': len(service_processes),
                'total_memory_mb': round(total_memory_mb, 2),
                'total_memory_percent': round(total_memory_percent, 2),
                'processes': service_processes
            }
            
        except Exception as e:
            logger.warning(f"Error getting memory usage for {service_name}: {e}")
            return None
    
    async def _detect_lifecycle_events(self) -> List[Dict[str, Any]]:
        """Detect model lifecycle events since last collection."""
        events = []
        
        try:
            # Compare current state with previous state to detect changes
            current_models = {}
            
            # Get current model state
            for service_name, endpoint_config in self.service_endpoints.items():
                service_models = await self._discover_service_models(service_name, endpoint_config)
                current_models[service_name] = service_models
            
            # Compare with previous state to detect events
            for service_name, models in current_models.items():
                previous_models = self.active_models.get(service_name, [])
                
                # Detect new model loads
                current_model_ids = {m['model_id'] for m in models}
                previous_model_ids = {m.model_metadata.model_id for m in previous_models if hasattr(m, 'model_metadata')}
                
                # New models loaded
                for model_id in current_model_ids - previous_model_ids:
                    model_info = next((m for m in models if m['model_id'] == model_id), None)
                    if model_info:
                        events.append({
                            'event_type': 'model_load',
                            'service_name': service_name,
                            'model_id': model_id,
                            'model_metadata': model_info.get('metadata', {}),
                            'timestamp': datetime.now().isoformat(),
                            'detection_method': model_info.get('detection_method', 'api')
                        })
                
                # Models unloaded
                for model_id in previous_model_ids - current_model_ids:
                    events.append({
                        'event_type': 'model_unload',
                        'service_name': service_name,
                        'model_id': model_id,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Update active models state
            self.active_models = current_models
            
            # Store events for historical analysis
            self.load_events.extend([ModelLoadEvent(
                event_type=e['event_type'],
                model_metadata=ModelMetadata(**e['model_metadata']) if e.get('model_metadata') else ModelMetadata(
                    model_id=e['model_id'],
                    model_name=e['model_id'],
                    model_family='unknown',
                    model_size='unknown',
                    model_type='unknown',
                    precision='unknown',
                    estimated_memory_gb=0.0,
                    context_length=0,
                    architecture='unknown'
                ),
                service_name=e['service_name'],
                timestamp=datetime.fromisoformat(e['timestamp']),
                duration_seconds=None,
                memory_delta_mb=0.0,
                gpu_memory_delta_mb=0.0,
                success=True,
                error_details=None,
                performance_impact={},
                resource_correlation={}
            ) for e in events])
            
        except Exception as e:
            logger.error(f"Error detecting lifecycle events: {e}")
        
        return events
    
    async def _analyze_cross_service_optimization(self) -> Dict[str, Any]:
        """Analyze optimization opportunities across services."""
        try:
            optimization_analysis = {
                'model_sharing_opportunities': [],
                'load_balancing_recommendations': [],
                'resource_optimization': [],
                'service_efficiency_comparison': {}
            }
            
            # Analyze model duplication across services
            all_models = {}
            for service_name, endpoint_config in self.service_endpoints.items():
                models = await self._discover_service_models(service_name, endpoint_config)
                for model in models:
                    model_id = model['model_id']
                    if model_id not in all_models:
                        all_models[model_id] = []
                    all_models[model_id].append(service_name)
            
            # Find duplicated models
            duplicated_models = {model_id: services for model_id, services in all_models.items() if len(services) > 1}
            
            for model_id, services in duplicated_models.items():
                optimization_analysis['model_sharing_opportunities'].append({
                    'model_id': model_id,
                    'services': services,
                    'optimization': 'Consider model sharing or specialized deployment'
                })
            
            # Service efficiency comparison
            for service_name in self.service_endpoints.keys():
                service_perf = await self._analyze_service_performance(service_name, self.service_endpoints[service_name])
                service_mem = await self._get_service_memory_usage(service_name)
                
                if service_perf and service_mem:
                    efficiency_score = self._calculate_service_efficiency(service_perf, service_mem)
                    optimization_analysis['service_efficiency_comparison'][service_name] = {
                        'efficiency_score': efficiency_score,
                        'memory_usage_mb': service_mem['total_memory_mb'],
                        'performance_score': service_perf.get('estimated_throughput_rps', 0)
                    }
            
            return optimization_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing cross-service optimization: {e}")
            return {'error': str(e)}
    
    def _calculate_service_efficiency(self, performance: Dict[str, Any], memory: Dict[str, Any]) -> float:
        """Calculate efficiency score for a service."""
        try:
            # Normalize metrics
            throughput = performance.get('estimated_throughput_rps', performance.get('throughput_rps', 0))
            memory_mb = memory.get('total_memory_mb', 0)
            
            if memory_mb == 0:
                return 0.0
            
            # Efficiency = throughput per GB of memory
            efficiency = (throughput / (memory_mb / 1024)) * 100
            return round(efficiency, 2)
            
        except Exception:
            return 0.0
    
    async def _analyze_model_caching(self) -> Dict[str, Any]:
        """Analyze model caching effectiveness."""
        try:
            cache_analysis = {
                'cache_effectiveness': 'analyzing',
                'model_persistence': {},
                'cache_optimization_opportunities': []
            }
            
            # Analyze model persistence across time
            if len(self.model_history) > 5:
                persistent_models = set()
                recent_history = list(self.model_history)[-5:]
                
                # Find models that appear in multiple history snapshots
                for snapshot in recent_history:
                    for service_name, models in snapshot.items():
                        for model in models:
                            model_id = model.get('model_id', 'unknown')
                            persistent_models.add((service_name, model_id))
                
                cache_analysis['model_persistence'] = {
                    'persistent_models_count': len(persistent_models),
                    'cache_hit_rate_estimate': len(persistent_models) / max(len(recent_history), 1) * 100
                }
                
                # Cache effectiveness assessment
                hit_rate = cache_analysis['model_persistence']['cache_hit_rate_estimate']
                if hit_rate > 80:
                    cache_analysis['cache_effectiveness'] = 'excellent'
                elif hit_rate > 60:
                    cache_analysis['cache_effectiveness'] = 'good'
                else:
                    cache_analysis['cache_effectiveness'] = 'poor'
                    cache_analysis['cache_optimization_opportunities'].append({
                        'type': 'low_cache_hit_rate',
                        'description': f'Cache hit rate is {hit_rate:.1f}%',
                        'recommendation': 'Consider implementing model persistence strategies'
                    })
            
            return cache_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing model caching: {e}")
            return {'error': str(e)}
    
    async def _analyze_resource_correlation(self) -> Dict[str, Any]:
        """Analyze correlation between models and resource usage."""
        try:
            correlation_analysis = {
                'model_resource_patterns': {},
                'resource_predictions': {},
                'optimization_insights': []
            }
            
            # Analyze resource patterns for each active model type
            model_resource_data = defaultdict(list)
            
            for service_name, endpoint_config in self.service_endpoints.items():
                models = await self._discover_service_models(service_name, endpoint_config)
                service_memory = await self._get_service_memory_usage(service_name)
                
                if models and service_memory:
                    for model in models:
                        model_family = model.get('metadata', {}).get('model_family', 'unknown')
                        model_size = model.get('metadata', {}).get('model_size', 'unknown')
                        
                        resource_data = {
                            'memory_mb': service_memory['total_memory_mb'],
                            'model_count': len(models),
                            'service_type': 'cpu' if 'cpu' in service_name else 'gpu'
                        }
                        
                        model_key = f"{model_family}-{model_size}"
                        model_resource_data[model_key].append(resource_data)
            
            # Calculate resource patterns
            for model_key, resource_list in model_resource_data.items():
                if resource_list:
                    avg_memory = sum(r['memory_mb'] for r in resource_list) / len(resource_list)
                    correlation_analysis['model_resource_patterns'][model_key] = {
                        'average_memory_mb': round(avg_memory, 2),
                        'sample_count': len(resource_list),
                        'service_distribution': {
                            'cpu': len([r for r in resource_list if r['service_type'] == 'cpu']),
                            'gpu': len([r for r in resource_list if r['service_type'] == 'gpu'])
                        }
                    }
            
            return correlation_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing resource correlation: {e}")
            return {'error': str(e)}
    
    async def _check_service_health(self) -> Dict[str, Any]:
        """Check health status of all AI services."""
        health_status = {}
        
        for service_name, endpoint_config in self.service_endpoints.items():
            try:
                health_url = f"http://{endpoint_config['host']}:{endpoint_config['port']}{endpoint_config['health_path']}"
                response = requests.get(health_url, timeout=5)
                
                health_status[service_name] = {
                    'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                    'response_code': response.status_code,
                    'response_time_ms': response.elapsed.total_seconds() * 1000
                }
                
            except requests.RequestException as e:
                health_status[service_name] = {
                    'status': 'unreachable',
                    'error': str(e)
                }
        
        return health_status
    
    async def _generate_model_recommendations(self, 
                                            active_models: Dict[str, Any],
                                            performance_analysis: Dict[str, Any],
                                            memory_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate model optimization recommendations."""
        recommendations = []
        
        # Memory optimization recommendations
        if memory_analysis.get('memory_efficiency', 0) > 85:
            recommendations.append({
                'category': 'memory_optimization',
                'priority': 'high',
                'title': 'High Memory Utilization',
                'description': f'Memory efficiency at {memory_analysis["memory_efficiency"]}%',
                'actions': [
                    'Consider quantization (int8/int4) for large models',
                    'Implement model streaming for very large models',
                    'Review model caching strategies'
                ]
            })
        
        # Performance optimization
        bottlenecks = performance_analysis.get('bottleneck_analysis', [])
        if bottlenecks:
            recommendations.append({
                'category': 'performance_optimization',
                'priority': 'medium',
                'title': 'Performance Bottlenecks Detected',
                'description': f'{len(bottlenecks)} bottlenecks found',
                'actions': [
                    'Optimize batch sizes for affected services',
                    'Consider model parallelization',
                    'Review hardware allocation'
                ]
            })
        
        # Model distribution optimization
        total_models = active_models.get('total_active_models', 0)
        if total_models > 8:
            recommendations.append({
                'category': 'resource_efficiency',
                'priority': 'low',
                'title': 'High Model Count',
                'description': f'{total_models} models active across services',
                'actions': [
                    'Consider model consolidation strategies',
                    'Implement dynamic model loading',
                    'Review model usage patterns'
                ]
            })
        
        return recommendations
    
    async def detect_changes(self, old_data: Dict[str, Any], new_data: Dict[str, Any]) -> List[SystemChange]:
        """Detect changes in AI model lifecycle state."""
        changes = []
        
        if 'active_models' not in old_data or 'active_models' not in new_data:
            return changes
        
        old_models = old_data['active_models']
        new_models = new_data['active_models']
        
        # Model count changes
        old_count = old_models.get('total_active_models', 0)
        new_count = new_models.get('total_active_models', 0)
        
        if old_count != new_count:
            changes.append(SystemChange(
                category='ai_model_lifecycle',
                change_type=ChangeType.MODIFIED if old_count > 0 else ChangeType.ADDED,
                entity_id='total_active_models',
                old_value=old_count,
                new_value=new_count,
                significance=0.7,
                metadata={
                    'change_type': 'model_count_change',
                    'model_delta': new_count - old_count,
                    'model_distribution': new_models.get('model_distribution', {})
                },
                timestamp=datetime.now()
            ))
        
        # Service model changes
        old_distribution = old_models.get('model_distribution', {})
        new_distribution = new_models.get('model_distribution', {})
        
        for service_name in set(old_distribution.keys()) | set(new_distribution.keys()):
            old_service_count = old_distribution.get(service_name, 0)
            new_service_count = new_distribution.get(service_name, 0)
            
            if old_service_count != new_service_count:
                changes.append(SystemChange(
                    category='ai_model_lifecycle',
                    change_type=ChangeType.MODIFIED,
                    entity_id=f'service_models:{service_name}',
                    old_value=old_service_count,
                    new_value=new_service_count,
                    significance=0.8,
                    metadata={
                        'change_type': 'service_model_change',
                        'service_name': service_name,
                        'model_delta': new_service_count - old_service_count
                    },
                    timestamp=datetime.now()
                ))
        
        # Memory utilization changes
        if ('memory_utilization_analysis' in old_data and 
            'memory_utilization_analysis' in new_data):
            
            old_memory = old_data['memory_utilization_analysis']
            new_memory = new_data['memory_utilization_analysis']
            
            old_total_gb = old_memory.get('total_model_memory_gb', 0)
            new_total_gb = new_memory.get('total_model_memory_gb', 0)
            
            memory_delta_gb = abs(new_total_gb - old_total_gb)
            if memory_delta_gb > 2.0:  # 2GB threshold
                changes.append(SystemChange(
                    category='ai_model_lifecycle',
                    change_type=ChangeType.MODIFIED,
                    entity_id='model_memory_usage',
                    old_value=old_total_gb,
                    new_value=new_total_gb,
                    significance=0.7,
                    metadata={
                        'change_type': 'model_memory_change',
                        'memory_delta_gb': new_total_gb - old_total_gb,
                        'memory_efficiency': new_memory.get('memory_efficiency', 0)
                    },
                    timestamp=datetime.now()
                ))
        
        # Lifecycle events
        if 'lifecycle_events' in new_data and new_data['lifecycle_events']:
            for event in new_data['lifecycle_events']:
                changes.append(SystemChange(
                    category='ai_model_lifecycle',
                    change_type=ChangeType.ADDED if event['event_type'] == 'model_load' else ChangeType.REMOVED,
                    entity_id=f'lifecycle_event:{event["model_id"]}',
                    old_value=None,
                    new_value=event,
                    significance=0.9,
                    metadata={
                        'change_type': 'model_lifecycle_event',
                        'event_type': event['event_type'],
                        'service_name': event['service_name'],
                        'model_id': event['model_id']
                    },
                    timestamp=datetime.now()
                ))
        
        return changes