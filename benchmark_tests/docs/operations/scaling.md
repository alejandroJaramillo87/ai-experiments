# Scaling

Horizontal and vertical scaling strategies for the AI Model Evaluation Framework.

## Scaling Architecture

### Scaling Dimensions

**Compute Scaling**:
- CPU resources for concurrent evaluation processing
- GPU resources for model inference acceleration
- Memory allocation for large-scale batch processing
- Storage capacity for evaluation data and results

**Performance Scaling**:
- Evaluation throughput (evaluations per second)
- Response latency (time per evaluation)
- Concurrent user capacity
- Data processing bandwidth

**Geographic Scaling**:
- Multi-region deployment for global accessibility
- Edge computing for reduced latency
- Data locality for compliance requirements
- Disaster recovery and high availability

### Scaling Patterns

**Horizontal Scaling** (Scale Out):
- Add more instances to distribute load
- Use load balancers for traffic distribution
- Implement service mesh for communication
- Enable auto-scaling based on metrics

**Vertical Scaling** (Scale Up):
- Increase CPU/GPU resources per instance
- Expand memory capacity
- Upgrade storage performance
- Optimize single-node performance

## Auto-Scaling Implementation

### Kubernetes Auto-Scaling

**Horizontal Pod Autoscaler (HPA)**:
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: benchmark-framework-hpa
  namespace: benchmark-framework
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: benchmark-framework
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: evaluation_queue_depth
      target:
        type: AverageValue
        averageValue: "10"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 600
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

**Vertical Pod Autoscaler (VPA)**:
```yaml
# k8s/vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: benchmark-framework-vpa
  namespace: benchmark-framework
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: benchmark-framework
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: benchmark-framework
      maxAllowed:
        cpu: "8"
        memory: "32Gi"
      minAllowed:
        cpu: "500m"
        memory: "4Gi"
      controlledResources: ["cpu", "memory"]
```

**Cluster Autoscaler**:
```yaml
# k8s/cluster-autoscaler.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      containers:
      - image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.21.0
        name: cluster-autoscaler
        resources:
          limits:
            cpu: 100m
            memory: 300Mi
          requests:
            cpu: 100m
            memory: 300Mi
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/benchmark-cluster
        - --balance-similar-node-groups
        - --skip-nodes-with-system-pods=false
        - --scale-down-delay-after-add=10m
        - --scale-down-unneeded-time=10m
```

### Custom Auto-Scaling Logic

**Intelligent Auto-Scaler**:
```python
# scaling/intelligent_autoscaler.py
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from kubernetes import client, config
import redis

@dataclass
class ScalingMetrics:
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float
    queue_depth: int
    response_latency_p95: float
    active_evaluations: int
    error_rate: float

@dataclass
class ScalingDecision:
    action: str  # 'scale_up', 'scale_down', 'no_change'
    target_replicas: int
    confidence: float
    reasoning: str

class IntelligentAutoScaler:
    def __init__(self, namespace='benchmark-framework'):
        config.load_incluster_config()
        self.apps_v1 = client.AppsV1Api()
        self.namespace = namespace
        self.redis_client = redis.Redis(host='redis', port=6379)
        
        # Scaling parameters
        self.min_replicas = 2
        self.max_replicas = 50
        self.target_cpu_utilization = 0.70
        self.target_memory_utilization = 0.80
        self.max_queue_depth = 20
        self.max_latency_ms = 5000
        
        # ML-based prediction model (simplified)
        self.scaling_history = []
        
    def collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics"""
        # Get metrics from Prometheus or monitoring system
        # This is a simplified version - in production, use proper metrics collection
        
        # CPU and memory from Kubernetes metrics
        cpu_usage = self._get_cpu_utilization()
        memory_usage = self._get_memory_utilization()
        
        # GPU metrics from nvidia-smi or GPU monitoring
        gpu_usage = self._get_gpu_utilization()
        
        # Application-specific metrics from Redis
        queue_depth = int(self.redis_client.get('evaluation_queue_depth') or 0)
        active_evals = int(self.redis_client.get('active_evaluations') or 0)
        
        # Performance metrics
        latency_key = 'latency_p95_last_5min'
        latency_p95 = float(self.redis_client.get(latency_key) or 0)
        
        error_key = 'error_rate_last_5min'
        error_rate = float(self.redis_client.get(error_key) or 0)
        
        return ScalingMetrics(
            cpu_utilization=cpu_usage,
            memory_utilization=memory_usage,
            gpu_utilization=gpu_usage,
            queue_depth=queue_depth,
            response_latency_p95=latency_p95,
            active_evaluations=active_evals,
            error_rate=error_rate
        )
    
    def make_scaling_decision(self, metrics: ScalingMetrics) -> ScalingDecision:
        """Make intelligent scaling decision based on multiple factors"""
        current_replicas = self._get_current_replicas()
        
        # Calculate scaling factors
        scaling_factors = self._calculate_scaling_factors(metrics)
        
        # Predict future load based on historical data
        predicted_load = self._predict_future_load()
        
        # Make decision based on multiple factors
        target_replicas = self._calculate_target_replicas(
            current_replicas, scaling_factors, predicted_load
        )
        
        # Determine action and confidence
        if target_replicas > current_replicas:
            action = 'scale_up'
            confidence = self._calculate_confidence(scaling_factors, 'up')
            reasoning = self._generate_reasoning(scaling_factors, 'up')
        elif target_replicas < current_replicas:
            action = 'scale_down'
            confidence = self._calculate_confidence(scaling_factors, 'down')
            reasoning = self._generate_reasoning(scaling_factors, 'down')
        else:
            action = 'no_change'
            confidence = 0.9
            reasoning = "Current capacity is optimal"
        
        return ScalingDecision(
            action=action,
            target_replicas=target_replicas,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _calculate_scaling_factors(self, metrics: ScalingMetrics) -> Dict[str, float]:
        """Calculate scaling factors for different metrics"""
        factors = {}
        
        # CPU factor
        if metrics.cpu_utilization > self.target_cpu_utilization:
            factors['cpu'] = metrics.cpu_utilization / self.target_cpu_utilization
        else:
            factors['cpu'] = 1.0
        
        # Memory factor
        if metrics.memory_utilization > self.target_memory_utilization:
            factors['memory'] = metrics.memory_utilization / self.target_memory_utilization
        else:
            factors['memory'] = 1.0
        
        # Queue depth factor
        if metrics.queue_depth > self.max_queue_depth:
            factors['queue'] = metrics.queue_depth / self.max_queue_depth
        else:
            factors['queue'] = 1.0
        
        # Latency factor
        if metrics.response_latency_p95 > self.max_latency_ms:
            factors['latency'] = metrics.response_latency_p95 / self.max_latency_ms
        else:
            factors['latency'] = 1.0
        
        # Error rate factor (scale up if errors increase)
        if metrics.error_rate > 0.05:  # 5% error rate threshold
            factors['errors'] = 1.5  # Aggressive scaling for errors
        else:
            factors['errors'] = 1.0
        
        return factors
    
    def _predict_future_load(self) -> float:
        """Predict future load based on historical patterns"""
        if len(self.scaling_history) < 10:
            return 1.0  # No prediction with insufficient data
        
        # Simple time series analysis (in production, use more sophisticated models)
        recent_metrics = self.scaling_history[-10:]
        loads = [h['total_load'] for h in recent_metrics]
        
        # Calculate trend
        if len(loads) >= 3:
            trend = np.polyfit(range(len(loads)), loads, 1)[0]
            return max(1.0, 1.0 + trend * 5)  # Project 5 time units ahead
        
        return 1.0
    
    def _calculate_target_replicas(self, current_replicas: int, 
                                 scaling_factors: Dict[str, float], 
                                 predicted_load: float) -> int:
        """Calculate target number of replicas"""
        # Weight different factors
        weights = {
            'cpu': 0.3,
            'memory': 0.2,
            'queue': 0.25,
            'latency': 0.2,
            'errors': 0.05
        }
        
        # Calculate weighted scaling factor
        weighted_factor = sum(
            weights.get(metric, 0) * factor 
            for metric, factor in scaling_factors.items()
        )
        
        # Apply prediction adjustment
        adjusted_factor = weighted_factor * predicted_load
        
        # Calculate target replicas
        target = int(current_replicas * adjusted_factor)
        
        # Apply bounds
        target = max(self.min_replicas, min(self.max_replicas, target))
        
        # Prevent thrashing - only change if significant difference
        if abs(target - current_replicas) <= 1 and adjusted_factor < 1.5:
            target = current_replicas
        
        return target
    
    def execute_scaling(self, decision: ScalingDecision) -> bool:
        """Execute scaling decision"""
        if decision.action == 'no_change':
            return True
        
        if decision.confidence < 0.7:
            print(f"Skipping scaling action due to low confidence: {decision.confidence}")
            return False
        
        try:
            # Get current deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name='benchmark-framework',
                namespace=self.namespace
            )
            
            # Update replica count
            deployment.spec.replicas = decision.target_replicas
            
            # Apply update
            self.apps_v1.patch_namespaced_deployment(
                name='benchmark-framework',
                namespace=self.namespace,
                body=deployment
            )
            
            print(f"Scaling executed: {decision.action} to {decision.target_replicas} replicas")
            print(f"Reasoning: {decision.reasoning}")
            
            # Log scaling event
            self._log_scaling_event(decision)
            
            return True
            
        except Exception as e:
            print(f"Scaling failed: {e}")
            return False
    
    def _log_scaling_event(self, decision: ScalingDecision):
        """Log scaling event for analysis"""
        event = {
            'timestamp': time.time(),
            'action': decision.action,
            'target_replicas': decision.target_replicas,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning
        }
        
        self.scaling_history.append(event)
        
        # Keep only last 1000 events
        if len(self.scaling_history) > 1000:
            self.scaling_history = self.scaling_history[-1000:]
```

## Load Balancing Strategies

### Advanced Load Balancing

**Nginx Advanced Configuration**:
```nginx
# nginx/advanced_load_balancer.conf
upstream benchmark_backend {
    # Weighted round-robin with health checks
    server benchmark-1:8080 weight=5 max_fails=3 fail_timeout=30s;
    server benchmark-2:8080 weight=5 max_fails=3 fail_timeout=30s;
    server benchmark-3:8080 weight=3 max_fails=3 fail_timeout=30s backup;
    
    # Connection pooling
    keepalive 64;
    keepalive_requests 100;
    keepalive_timeout 60s;
    
    # Health check configuration
    health_check interval=30s fails=3 passes=2 uri=/health;
}

# Intelligent routing based on request type
upstream gpu_heavy_backend {
    server gpu-node-1:8080 weight=3;
    server gpu-node-2:8080 weight=3;
    keepalive 32;
}

upstream cpu_heavy_backend {
    server cpu-node-1:8080 weight=3;
    server cpu-node-2:8080 weight=3;
    server cpu-node-3:8080 weight=3;
    keepalive 32;
}

# Rate limiting maps
map $binary_remote_addr $limit_key {
    default $binary_remote_addr;
    ~^192\.168\. "";  # No limit for internal IPs
}

limit_req_zone $limit_key zone=api_limit:10m rate=10r/s;
limit_req_zone $limit_key zone=eval_limit:10m rate=5r/s;

server {
    listen 80;
    server_name benchmark-api.example.com;
    
    # Request routing based on evaluation type
    location ~ ^/api/evaluate/(creativity|reasoning|integration) {
        limit_req zone=eval_limit burst=20 nodelay;
        
        # Route GPU-intensive evaluations to GPU nodes
        proxy_pass http://gpu_heavy_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Timeout settings for complex evaluations
        proxy_connect_timeout 30s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    location ~ ^/api/evaluate/(language|social|knowledge) {
        limit_req zone=eval_limit burst=30 nodelay;
        
        # Route CPU-intensive evaluations to CPU nodes
        proxy_pass http://cpu_heavy_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Standard timeout settings
        proxy_connect_timeout 10s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    location /api/ {
        limit_req zone=api_limit burst=50 nodelay;
        
        # General API requests
        proxy_pass http://benchmark_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    # Health check endpoint
    location /nginx-health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

### Intelligent Request Routing

**Smart Load Balancer**:
```python
# scaling/smart_load_balancer.py
import hashlib
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import requests

@dataclass
class BackendNode:
    id: str
    endpoint: str
    capabilities: List[str]
    current_load: float
    health_status: str
    last_health_check: float
    max_concurrent_requests: int
    current_requests: int

class SmartLoadBalancer:
    def __init__(self):
        self.nodes: Dict[str, BackendNode] = {}
        self.request_history = {}
        self.health_check_interval = 30  # seconds
        
    def register_node(self, node: BackendNode):
        """Register a backend node"""
        self.nodes[node.id] = node
        print(f"Registered node: {node.id} with capabilities: {node.capabilities}")
    
    def select_node(self, request_type: str, complexity: str = 'medium') -> Optional[BackendNode]:
        """Select optimal node for request"""
        # Filter nodes by capability
        capable_nodes = [
            node for node in self.nodes.values()
            if request_type in node.capabilities and node.health_status == 'healthy'
        ]
        
        if not capable_nodes:
            # Fallback to any healthy node
            capable_nodes = [
                node for node in self.nodes.values()
                if node.health_status == 'healthy'
            ]
        
        if not capable_nodes:
            return None
        
        # Score nodes based on multiple factors
        best_node = None
        best_score = float('inf')
        
        for node in capable_nodes:
            score = self._calculate_node_score(node, request_type, complexity)
            if score < best_score:
                best_score = score
                best_node = node
        
        return best_node
    
    def _calculate_node_score(self, node: BackendNode, request_type: str, complexity: str) -> float:
        """Calculate node suitability score (lower is better)"""
        # Base score from current load
        load_score = node.current_load * 100
        
        # Penalty for high utilization
        utilization = node.current_requests / node.max_concurrent_requests
        utilization_penalty = utilization ** 2 * 50
        
        # Bonus for specialized capabilities
        specialization_bonus = 0
        if f"{request_type}_optimized" in node.capabilities:
            specialization_bonus = -20
        
        # Complexity-based scoring
        complexity_factor = {
            'easy': 1.0,
            'medium': 1.5,
            'hard': 2.0
        }.get(complexity, 1.5)
        
        # Historical performance bonus
        historical_bonus = self._get_historical_performance_bonus(node.id, request_type)
        
        total_score = (load_score + utilization_penalty + 
                      specialization_bonus + historical_bonus) * complexity_factor
        
        return total_score
    
    def _get_historical_performance_bonus(self, node_id: str, request_type: str) -> float:
        """Get performance bonus based on historical success"""
        key = f"{node_id}:{request_type}"
        if key not in self.request_history:
            return 0
        
        history = self.request_history[key]
        success_rate = history.get('success_rate', 0.5)
        avg_latency = history.get('avg_latency', 1000)  # milliseconds
        
        # Bonus for high success rate and low latency
        success_bonus = (success_rate - 0.5) * 30  # Up to 15 point bonus
        latency_penalty = min(avg_latency / 100, 20)  # Up to 20 point penalty
        
        return success_bonus - latency_penalty
    
    def health_check_all_nodes(self):
        """Perform health checks on all nodes"""
        for node in self.nodes.values():
            if time.time() - node.last_health_check > self.health_check_interval:
                self._health_check_node(node)
    
    def _health_check_node(self, node: BackendNode):
        """Perform health check on a single node"""
        try:
            response = requests.get(f"{node.endpoint}/health", timeout=5)
            if response.status_code == 200:
                node.health_status = 'healthy'
                
                # Update load information if provided
                health_data = response.json()
                if 'load' in health_data:
                    node.current_load = health_data['load']
                if 'current_requests' in health_data:
                    node.current_requests = health_data['current_requests']
                    
            else:
                node.health_status = 'unhealthy'
                
        except Exception as e:
            node.health_status = 'unhealthy'
            print(f"Health check failed for node {node.id}: {e}")
        
        node.last_health_check = time.time()
    
    def update_request_history(self, node_id: str, request_type: str, 
                             success: bool, latency: float):
        """Update historical performance data"""
        key = f"{node_id}:{request_type}"
        
        if key not in self.request_history:
            self.request_history[key] = {
                'success_count': 0,
                'total_requests': 0,
                'total_latency': 0,
                'success_rate': 0,
                'avg_latency': 0
            }
        
        history = self.request_history[key]
        history['total_requests'] += 1
        history['total_latency'] += latency
        
        if success:
            history['success_count'] += 1
        
        history['success_rate'] = history['success_count'] / history['total_requests']
        history['avg_latency'] = history['total_latency'] / history['total_requests']
```

## Multi-Region Scaling

### Geographic Distribution

**Multi-Region Deployment**:
```yaml
# k8s/multi-region-deployment.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: region-config
data:
  us-east-1.yaml: |
    region: us-east-1
    replicas: 10
    gpu_nodes: 3
    cpu_nodes: 7
    data_locality: high
    
  us-west-2.yaml: |
    region: us-west-2
    replicas: 8
    gpu_nodes: 2
    cpu_nodes: 6
    data_locality: medium
    
  eu-west-1.yaml: |
    region: eu-west-1
    replicas: 6
    gpu_nodes: 2
    cpu_nodes: 4
    data_locality: low

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: benchmark-framework-us-east-1
  namespace: benchmark-framework
spec:
  replicas: 10
  selector:
    matchLabels:
      app: benchmark-framework
      region: us-east-1
  template:
    metadata:
      labels:
        app: benchmark-framework
        region: us-east-1
    spec:
      nodeSelector:
        topology.kubernetes.io/region: us-east-1
      containers:
      - name: benchmark-framework
        image: benchmark-framework:latest
        env:
        - name: REGION
          value: "us-east-1"
        - name: DATA_LOCALITY_PREFERENCE
          value: "high"
        resources:
          requests:
            cpu: 2
            memory: 8Gi
            nvidia.com/gpu: 1
          limits:
            cpu: 4
            memory: 16Gi
            nvidia.com/gpu: 1
```

**Global Load Balancing**:
```python
# scaling/global_load_balancer.py
import geoip2.database
from typing import Dict, List
import requests

class GlobalLoadBalancer:
    def __init__(self):
        self.regions = {
            'us-east-1': {
                'endpoint': 'https://us-east-1.benchmark-api.com',
                'capacity': 1000,
                'current_load': 0,
                'latency_baseline': 50,  # milliseconds
                'geographic_zones': ['US', 'CA', 'MX']
            },
            'us-west-2': {
                'endpoint': 'https://us-west-2.benchmark-api.com',
                'capacity': 800,
                'current_load': 0,
                'latency_baseline': 60,
                'geographic_zones': ['US', 'CA']
            },
            'eu-west-1': {
                'endpoint': 'https://eu-west-1.benchmark-api.com',
                'capacity': 600,
                'current_load': 0,
                'latency_baseline': 40,
                'geographic_zones': ['GB', 'DE', 'FR', 'IT', 'ES']
            }
        }
        
        # GeoIP database for location detection
        self.geoip_reader = geoip2.database.Reader('/usr/share/GeoIP/GeoLite2-Country.mmdb')
    
    def route_request(self, client_ip: str, request_type: str) -> str:
        """Route request to optimal region"""
        # Get client location
        client_country = self._get_client_country(client_ip)
        
        # Score regions for this request
        region_scores = {}
        for region_id, region_info in self.regions.items():
            score = self._calculate_region_score(
                region_info, client_country, request_type
            )
            region_scores[region_id] = score
        
        # Select best region
        best_region = min(region_scores.items(), key=lambda x: x[1])[0]
        
        return self.regions[best_region]['endpoint']
    
    def _get_client_country(self, client_ip: str) -> str:
        """Get client country from IP address"""
        try:
            response = self.geoip_reader.country(client_ip)
            return response.country.iso_code
        except:
            return 'US'  # Default fallback
    
    def _calculate_region_score(self, region_info: Dict, 
                               client_country: str, request_type: str) -> float:
        """Calculate region suitability score"""
        # Geographic proximity score
        if client_country in region_info['geographic_zones']:
            geo_score = 0  # Perfect match
        else:
            geo_score = 100  # Geographic penalty
        
        # Capacity utilization score
        utilization = region_info['current_load'] / region_info['capacity']
        capacity_score = utilization * 200
        
        # Base latency score
        latency_score = region_info['latency_baseline']
        
        # Request type optimization
        type_score = self._get_request_type_score(region_info, request_type)
        
        total_score = geo_score + capacity_score + latency_score + type_score
        
        return total_score
    
    def _get_request_type_score(self, region_info: Dict, request_type: str) -> float:
        """Get region optimization score for request type"""
        # Different regions might be optimized for different workloads
        optimization_map = {
            'us-east-1': {'creativity': 0, 'reasoning': 10, 'integration': 5},
            'us-west-2': {'language': 0, 'social': 5, 'knowledge': 10},
            'eu-west-1': {'creativity': 5, 'language': 0, 'social': 0}
        }
        
        region_id = None
        for rid, rinfo in self.regions.items():
            if rinfo == region_info:
                region_id = rid
                break
        
        if region_id and region_id in optimization_map:
            return optimization_map[region_id].get(request_type, 10)
        
        return 10  # Neutral score
```

## Performance Optimization at Scale

### Connection Pooling and Caching

**Distributed Caching Strategy**:
```python
# scaling/distributed_cache.py
import redis.sentinel
import hashlib
from typing import Any, Optional
import pickle

class DistributedCache:
    def __init__(self, sentinels: List[tuple], service_name: str = 'redis-cluster'):
        self.sentinel = redis.sentinel.Sentinel(sentinels)
        self.service_name = service_name
        
        # Get master and slave connections
        self.master = self.sentinel.master_for(
            service_name, socket_timeout=0.1, password='your_password'
        )
        self.slaves = [
            self.sentinel.slave_for(service_name, socket_timeout=0.1, password='your_password')
        ]
        
        # Cache configuration
        self.default_ttl = 3600  # 1 hour
        self.max_key_size = 1024 * 1024  # 1MB
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, using read replicas"""
        try:
            # Try to read from a random slave first
            import random
            if self.slaves:
                slave = random.choice(self.slaves)
                data = slave.get(key)
                if data:
                    return pickle.loads(data)
            
            # Fallback to master
            data = self.master.get(key)
            if data:
                return pickle.loads(data)
                
        except Exception as e:
            print(f"Cache get error: {e}")
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl or self.default_ttl
            serialized_value = pickle.dumps(value)
            
            # Check size limit
            if len(serialized_value) > self.max_key_size:
                print(f"Value too large for cache: {len(serialized_value)} bytes")
                return False
            
            # Write to master
            return self.master.setex(key, ttl, serialized_value)
            
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            return bool(self.master.delete(key))
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False
    
    def cached_evaluation(self, ttl: int = None):
        """Decorator for caching evaluation results"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Create cache key from function arguments
                cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate deterministic cache key"""
        key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
```

## Monitoring Scaling Performance

### Scaling Metrics Dashboard

**Prometheus Metrics for Scaling**:
```python
# scaling/metrics.py
from prometheus_client import Counter, Gauge, Histogram

# Scaling operation metrics
SCALING_OPERATIONS = Counter(
    'scaling_operations_total',
    'Total number of scaling operations',
    ['action', 'trigger', 'success']
)

CURRENT_REPLICAS = Gauge(
    'current_replicas',
    'Current number of replicas',
    ['service', 'region']
)

SCALING_LATENCY = Histogram(
    'scaling_operation_duration_seconds',
    'Time taken for scaling operations',
    ['action'],
    buckets=(1, 5, 10, 30, 60, 120, 300)
)

# Resource utilization metrics
RESOURCE_UTILIZATION = Gauge(
    'resource_utilization_ratio',
    'Resource utilization ratio',
    ['resource_type', 'region']
)

# Performance impact metrics
PERFORMANCE_DURING_SCALING = Histogram(
    'request_latency_during_scaling_seconds',
    'Request latency during scaling operations',
    ['scaling_phase'],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0)
)

QUEUE_DEPTH_DURING_SCALING = Gauge(
    'queue_depth_during_scaling',
    'Request queue depth during scaling',
    ['scaling_phase']
)
```

**Grafana Dashboard Configuration**:
```json
{
  "dashboard": {
    "title": "Scaling Performance Dashboard",
    "panels": [
      {
        "title": "Scaling Operations",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(scaling_operations_total[5m])",
            "legendFormat": "Scaling Ops/sec"
          }
        ]
      },
      {
        "title": "Current Replicas by Region",
        "type": "graph",
        "targets": [
          {
            "expr": "current_replicas",
            "legendFormat": "{{region}} - {{service}}"
          }
        ]
      },
      {
        "title": "Scaling Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, scaling_operation_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, scaling_operation_duration_seconds_bucket)",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Resource Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "resource_utilization_ratio{resource_type=\"cpu\"}",
            "legendFormat": "CPU - {{region}}"
          },
          {
            "expr": "resource_utilization_ratio{resource_type=\"memory\"}",
            "legendFormat": "Memory - {{region}}"
          },
          {
            "expr": "resource_utilization_ratio{resource_type=\"gpu\"}",
            "legendFormat": "GPU - {{region}}"
          }
        ]
      }
    ]
  }
}
```

## Cost Optimization

### Cost-Aware Scaling

**Cost-Optimized Auto-Scaler**:
```python
# scaling/cost_optimizer.py
from dataclasses import dataclass
from typing import Dict, List
import time

@dataclass
class NodeCost:
    instance_type: str
    hourly_cost: float
    cpu_cores: int
    memory_gb: int
    gpu_count: int
    network_performance: str

class CostOptimizedScaler:
    def __init__(self):
        # Cost data for different instance types
        self.instance_costs = {
            'm5.large': NodeCost('m5.large', 0.096, 2, 8, 0, 'up_to_10gbps'),
            'm5.xlarge': NodeCost('m5.xlarge', 0.192, 4, 16, 0, 'up_to_10gbps'),
            'c5.xlarge': NodeCost('c5.xlarge', 0.17, 4, 8, 0, 'up_to_10gbps'),
            'p3.2xlarge': NodeCost('p3.2xlarge', 3.06, 8, 61, 1, '25gbps'),
            'p3.8xlarge': NodeCost('p3.8xlarge', 12.24, 32, 244, 4, '25gbps'),
        }
        
        # Workload characteristics
        self.workload_requirements = {
            'cpu_intensive': {'cpu_weight': 0.7, 'memory_weight': 0.2, 'gpu_weight': 0.1},
            'gpu_intensive': {'cpu_weight': 0.2, 'memory_weight': 0.2, 'gpu_weight': 0.6},
            'memory_intensive': {'cpu_weight': 0.2, 'memory_weight': 0.7, 'gpu_weight': 0.1},
            'balanced': {'cpu_weight': 0.4, 'memory_weight': 0.4, 'gpu_weight': 0.2}
        }
    
    def optimize_node_mix(self, target_capacity: Dict[str, float], 
                         workload_type: str = 'balanced') -> List[Dict]:
        """Optimize node mix for cost efficiency"""
        requirements = self.workload_requirements[workload_type]
        
        # Calculate cost per unit of each resource type
        cost_efficiency = {}
        for instance_type, cost_info in self.instance_costs.items():
            weighted_performance = (
                cost_info.cpu_cores * requirements['cpu_weight'] +
                cost_info.memory_gb * requirements['memory_weight'] +
                cost_info.gpu_count * requirements['gpu_weight'] * 10  # GPU multiplier
            )
            
            if weighted_performance > 0:
                cost_efficiency[instance_type] = cost_info.hourly_cost / weighted_performance
        
        # Sort by cost efficiency
        sorted_instances = sorted(cost_efficiency.items(), key=lambda x: x[1])
        
        # Allocate instances to meet requirements
        allocation = []
        remaining_cpu = target_capacity.get('cpu_cores', 0)
        remaining_memory = target_capacity.get('memory_gb', 0)
        remaining_gpu = target_capacity.get('gpu_count', 0)
        
        for instance_type, efficiency in sorted_instances:
            if remaining_cpu <= 0 and remaining_memory <= 0 and remaining_gpu <= 0:
                break
            
            instance_info = self.instance_costs[instance_type]
            
            # Calculate how many instances of this type we need
            cpu_instances_needed = max(0, remaining_cpu / instance_info.cpu_cores)
            memory_instances_needed = max(0, remaining_memory / instance_info.memory_gb)
            gpu_instances_needed = max(0, remaining_gpu / max(1, instance_info.gpu_count))
            
            instances_needed = int(max(cpu_instances_needed, memory_instances_needed, gpu_instances_needed))
            
            if instances_needed > 0:
                allocation.append({
                    'instance_type': instance_type,
                    'count': instances_needed,
                    'hourly_cost': instance_info.hourly_cost * instances_needed,
                    'total_cpu': instance_info.cpu_cores * instances_needed,
                    'total_memory': instance_info.memory_gb * instances_needed,
                    'total_gpu': instance_info.gpu_count * instances_needed
                })
                
                # Update remaining requirements
                remaining_cpu -= instance_info.cpu_cores * instances_needed
                remaining_memory -= instance_info.memory_gb * instances_needed
                remaining_gpu -= instance_info.gpu_count * instances_needed
        
        return allocation
    
    def calculate_total_cost(self, allocation: List[Dict], hours: int = 24) -> float:
        """Calculate total cost for allocation"""
        return sum(item['hourly_cost'] for item in allocation) * hours
    
    def recommend_cost_savings(self, current_allocation: List[Dict]) -> Dict:
        """Recommend cost-saving optimizations"""
        recommendations = []
        
        # Analyze utilization patterns
        for item in current_allocation:
            if item.get('cpu_utilization', 0) < 0.5:
                recommendations.append({
                    'type': 'downsize',
                    'instance_type': item['instance_type'],
                    'reason': 'Low CPU utilization',
                    'potential_savings': item['hourly_cost'] * 0.5 * 24
                })
            
            if item.get('memory_utilization', 0) < 0.4:
                recommendations.append({
                    'type': 'optimize',
                    'instance_type': item['instance_type'],
                    'reason': 'Low memory utilization',
                    'suggestion': 'Consider memory-optimized instances'
                })
        
        return {
            'recommendations': recommendations,
            'total_potential_savings': sum(r.get('potential_savings', 0) for r in recommendations)
        }
```

## References

- [Performance](./performance.md)
- [Monitoring](./monitoring.md)
- [Deployment](./deployment.md)
- [Configuration](../engineering/configuration.md)