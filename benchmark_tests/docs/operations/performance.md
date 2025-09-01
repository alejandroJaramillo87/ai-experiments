# Performance

Performance optimization and tuning guidelines for the AI Model Evaluation Framework.

## Performance Baseline

### System Specifications

**Reference Hardware Configuration**:
- CPU: AMD Ryzen 9950X (16 cores, 32 threads)
- RAM: 128GB DDR5-5600
- GPU: NVIDIA RTX 5090 (24GB VRAM)
- Storage: Samsung 990 Pro 2TB NVMe
- Network: 10Gbps Ethernet

**Expected Performance Metrics**:
- Single evaluation latency: 50-200ms (depending on domain complexity)
- Concurrent evaluations: 20-50 parallel evaluations
- Throughput: 100-500 evaluations per minute
- GPU utilization: 70-90% under load
- Memory utilization: 60-80% of available RAM

## CPU Optimization

### Process and Threading Configuration

**Worker Process Optimization**:
```python
# config/performance.py
import multiprocessing
import os

def calculate_optimal_workers():
    """Calculate optimal number of worker processes"""
    cpu_count = multiprocessing.cpu_count()
    
    # For CPU-intensive tasks (reasoning, language analysis)
    cpu_workers = min(cpu_count, 16)
    
    # For I/O-bound tasks (file operations, network requests)
    io_workers = min(cpu_count * 2, 32)
    
    return {
        'evaluation_workers': cpu_workers,
        'io_workers': io_workers,
        'total_workers': cpu_workers + io_workers
    }

# Optimal configuration
WORKER_CONFIG = {
    'processes': calculate_optimal_workers()['evaluation_workers'],
    'threads_per_process': 2,
    'max_concurrent_evaluations': 20,
    'queue_size': 100
}
```

**Thread Pool Configuration**:
```python
# performance/thread_pool.py
import concurrent.futures
import threading
from queue import Queue

class OptimizedThreadPool:
    def __init__(self, max_workers=None):
        if max_workers is None:
            max_workers = min(32, multiprocessing.cpu_count() * 2)
        
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix='evaluation'
        )
        self.semaphore = threading.Semaphore(max_workers)
    
    def submit_evaluation(self, evaluation_func, *args, **kwargs):
        """Submit evaluation with resource limiting"""
        def wrapped_evaluation():
            with self.semaphore:
                return evaluation_func(*args, **kwargs)
        
        return self.executor.submit(wrapped_evaluation)
```

### CPU Affinity and NUMA Optimization

**CPU Affinity Configuration**:
```bash
#!/bin/bash
# scripts/cpu_optimization.sh

# Set CPU affinity for evaluation processes
PID=$(pgrep -f "benchmark_framework")
if [ ! -z "$PID" ]; then
    # Bind to cores 0-15 (first socket on dual-socket systems)
    taskset -cp 0-15 $PID
    echo "CPU affinity set for PID $PID"
fi

# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU frequency scaling for consistent performance
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true

# Optimize system interrupts
echo 2 | sudo tee /proc/irq/*/smp_affinity_list
```

**NUMA Optimization**:
```python
# performance/numa_optimization.py
import numa
import os

def optimize_numa_allocation():
    """Optimize NUMA memory allocation"""
    if numa.available():
        # Get NUMA node information
        numa_nodes = numa.get_max_node() + 1
        
        # Bind memory allocation to local NUMA node
        numa.set_preferred_node(0)
        
        # Set memory policy for evaluation processes
        numa.set_membind_nodes([0])
        
        return {
            'numa_enabled': True,
            'numa_nodes': numa_nodes,
            'preferred_node': 0
        }
    
    return {'numa_enabled': False}
```

## GPU Optimization

### CUDA Configuration

**GPU Memory Management**:
```python
# performance/gpu_optimization.py
import torch
import gc

class GPUMemoryManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_threshold = 0.85  # 85% memory usage threshold
    
    def optimize_memory_settings(self):
        """Optimize GPU memory settings"""
        if torch.cuda.is_available():
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.9)
            
            # Enable memory pooling
            torch.cuda.empty_cache()
            
            # Set CUDA memory pool configuration
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            
            return {
                'gpu_available': True,
                'memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                'memory_reserved': torch.cuda.memory_reserved() / 1024**3,
                'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def check_memory_usage(self):
        """Check GPU memory usage and trigger cleanup if needed"""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            
            if memory_used > self.memory_threshold:
                self.cleanup_memory()
                return True
        
        return False
```

**CUDA Stream Optimization**:
```python
# performance/cuda_streams.py
import torch

class CUDAStreamManager:
    def __init__(self, num_streams=4):
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.current_stream = 0
    
    def get_stream(self):
        """Get next available CUDA stream"""
        stream = self.streams[self.current_stream]
        self.current_stream = (self.current_stream + 1) % len(self.streams)
        return stream
    
    def evaluate_with_stream(self, evaluator, data):
        """Run evaluation on specific CUDA stream"""
        stream = self.get_stream()
        
        with torch.cuda.stream(stream):
            result = evaluator(data)
        
        # Synchronize stream before returning
        stream.synchronize()
        return result
```

### GPU Utilization Optimization

**Batch Processing Configuration**:
```python
# performance/batch_optimization.py
import torch
from typing import List, Any

class BatchOptimizer:
    def __init__(self, max_batch_size=64, adaptive_batching=True):
        self.max_batch_size = max_batch_size
        self.adaptive_batching = adaptive_batching
        self.performance_history = []
    
    def calculate_optimal_batch_size(self, data_size: int) -> int:
        """Calculate optimal batch size based on GPU memory and performance"""
        if not torch.cuda.is_available():
            return min(16, data_size)
        
        # Get available GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        available_memory = gpu_memory - allocated_memory
        
        # Estimate memory per sample (rough approximation)
        memory_per_sample = 50 * 1024 * 1024  # 50MB per sample estimate
        
        memory_based_batch_size = int(available_memory * 0.8 / memory_per_sample)
        optimal_batch_size = min(
            self.max_batch_size,
            memory_based_batch_size,
            data_size
        )
        
        return max(1, optimal_batch_size)
    
    def adaptive_batch_processing(self, data: List[Any], process_func):
        """Process data with adaptive batching"""
        results = []
        batch_size = self.calculate_optimal_batch_size(len(data))
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            try:
                start_time = time.time()
                batch_results = process_func(batch)
                processing_time = time.time() - start_time
                
                results.extend(batch_results)
                
                # Update performance history for adaptive adjustment
                if self.adaptive_batching:
                    self.performance_history.append({
                        'batch_size': len(batch),
                        'processing_time': processing_time,
                        'throughput': len(batch) / processing_time
                    })
                    
                    # Adjust batch size based on performance
                    if len(self.performance_history) > 10:
                        self._adjust_batch_size()
                
            except torch.cuda.OutOfMemoryError:
                # Reduce batch size and retry
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                self.max_batch_size = batch_size
                
                # Process smaller batches
                for j in range(0, len(batch), batch_size):
                    sub_batch = batch[j:j + batch_size]
                    sub_results = process_func(sub_batch)
                    results.extend(sub_results)
        
        return results
```

## Memory Optimization

### System Memory Management

**Memory Pool Configuration**:
```python
# performance/memory_pool.py
import psutil
from pymemcache.client.base import Client

class MemoryOptimizer:
    def __init__(self):
        self.total_memory = psutil.virtual_memory().total
        self.cache_client = Client(('localhost', 11211))
    
    def calculate_memory_allocation(self):
        """Calculate optimal memory allocation"""
        available_memory = psutil.virtual_memory().available
        
        allocation = {
            # Reserve 20% for system
            'system_reserved': self.total_memory * 0.2,
            
            # 50% for model loading and inference
            'model_memory': self.total_memory * 0.5,
            
            # 20% for evaluation data and intermediate results
            'evaluation_memory': self.total_memory * 0.2,
            
            # 10% for caching and buffers
            'cache_memory': self.total_memory * 0.1
        }
        
        return allocation
    
    def optimize_swap_usage(self):
        """Optimize system swap usage"""
        # Set swappiness to minimize swap usage
        try:
            with open('/proc/sys/vm/swappiness', 'w') as f:
                f.write('10')
            
            # Set dirty ratio for better I/O performance
            with open('/proc/sys/vm/dirty_ratio', 'w') as f:
                f.write('5')
                
            with open('/proc/sys/vm/dirty_background_ratio', 'w') as f:
                f.write('2')
                
        except PermissionError:
            print("Warning: Could not optimize swap settings (requires root)")
```

**Garbage Collection Optimization**:
```python
# performance/gc_optimization.py
import gc
import threading
import time

class GarbageCollectionOptimizer:
    def __init__(self, gc_threshold=1000):
        self.gc_threshold = gc_threshold
        self.gc_counter = 0
        self.gc_stats = []
        
        # Optimize GC thresholds
        gc.set_threshold(700, 10, 10)
    
    def start_gc_monitoring(self):
        """Start background GC monitoring thread"""
        def gc_monitor():
            while True:
                # Collect GC stats
                stats = gc.get_stats()
                self.gc_stats.append({
                    'timestamp': time.time(),
                    'collections': stats,
                    'objects': len(gc.get_objects())
                })
                
                # Trigger manual GC if needed
                if self.gc_counter > self.gc_threshold:
                    start_time = time.time()
                    collected = gc.collect()
                    gc_time = time.time() - start_time
                    
                    print(f"Manual GC: collected {collected} objects in {gc_time:.3f}s")
                    self.gc_counter = 0
                
                time.sleep(60)  # Check every minute
        
        thread = threading.Thread(target=gc_monitor, daemon=True)
        thread.start()
    
    def increment_gc_counter(self):
        """Increment GC counter for operations that create many objects"""
        self.gc_counter += 1
```

## I/O Optimization

### Disk I/O Performance

**Async I/O Configuration**:
```python
# performance/async_io.py
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor

class AsyncIOOptimizer:
    def __init__(self, max_workers=8):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)
    
    async def read_file_async(self, filepath):
        """Asynchronously read file"""
        async with self.semaphore:
            async with aiofiles.open(filepath, 'r') as f:
                return await f.read()
    
    async def write_results_async(self, results, output_path):
        """Asynchronously write evaluation results"""
        async with self.semaphore:
            async with aiofiles.open(output_path, 'w') as f:
                await f.write(json.dumps(results, indent=2))
    
    async def batch_file_operations(self, file_operations):
        """Execute batch file operations asynchronously"""
        tasks = [operation() for operation in file_operations]
        return await asyncio.gather(*tasks)
```

**File System Optimization**:
```bash
#!/bin/bash
# scripts/filesystem_optimization.sh

# Mount options for better performance
mount -o remount,noatime,nodiratime /

# Optimize I/O scheduler
echo deadline > /sys/block/nvme0n1/queue/scheduler

# Increase read-ahead buffer
echo 4096 > /sys/block/nvme0n1/queue/read_ahead_kb

# Optimize dirty page writeback
echo 5 > /proc/sys/vm/dirty_ratio
echo 2 > /proc/sys/vm/dirty_background_ratio
echo 500 > /proc/sys/vm/dirty_writeback_centisecs
echo 3000 > /proc/sys/vm/dirty_expire_centisecs
```

### Network I/O Optimization

**Connection Pool Management**:
```python
# performance/network_optimization.py
import aiohttp
import asyncio
from aiohttp import ClientSession, TCPConnector

class NetworkOptimizer:
    def __init__(self):
        self.connector = TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=20,  # Connections per host
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        self.session = None
    
    async def create_session(self):
        """Create optimized HTTP session"""
        self.session = ClientSession(
            connector=self.connector,
            timeout=aiohttp.ClientTimeout(total=300)
        )
        return self.session
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
```

## Caching Strategies

### Multi-Level Caching

**Cache Configuration**:
```python
# performance/caching.py
import redis
import hashlib
import pickle
from functools import wraps

class MultiLevelCache:
    def __init__(self):
        # Level 1: In-memory cache (LRU)
        self.memory_cache = {}
        self.memory_cache_size = 1000
        
        # Level 2: Redis cache
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=False,
            max_connections=20
        )
        
        # Level 3: Disk cache
        self.disk_cache_dir = "/tmp/benchmark_cache"
        os.makedirs(self.disk_cache_dir, exist_ok=True)
    
    def cache_key(self, func_name, args, kwargs):
        """Generate cache key from function and arguments"""
        key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_result(self, key):
        """Get result from multi-level cache"""
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check Redis cache
        redis_result = self.redis_client.get(key)
        if redis_result:
            result = pickle.loads(redis_result)
            # Store in memory cache for faster access
            self._store_in_memory_cache(key, result)
            return result
        
        # Check disk cache
        disk_path = os.path.join(self.disk_cache_dir, key)
        if os.path.exists(disk_path):
            with open(disk_path, 'rb') as f:
                result = pickle.load(f)
            
            # Store in higher-level caches
            self._store_in_memory_cache(key, result)
            self._store_in_redis_cache(key, result)
            return result
        
        return None
    
    def store_result(self, key, result, ttl=3600):
        """Store result in multi-level cache"""
        self._store_in_memory_cache(key, result)
        self._store_in_redis_cache(key, result, ttl)
        self._store_in_disk_cache(key, result)
    
    def cached_evaluation(self, ttl=3600):
        """Decorator for caching evaluation results"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = self.cache_key(func.__name__, args, kwargs)
                
                # Try to get cached result
                cached_result = self.get_cached_result(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Compute result and cache it
                result = func(*args, **kwargs)
                self.store_result(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
```

## Load Balancing and Scaling

### Horizontal Scaling Configuration

**Load Balancer Setup**:
```nginx
# nginx/load_balancer.conf
upstream benchmark_backend {
    least_conn;
    server benchmark-1:8080 weight=3 max_fails=3 fail_timeout=30s;
    server benchmark-2:8080 weight=3 max_fails=3 fail_timeout=30s;
    server benchmark-3:8080 weight=2 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

server {
    listen 80;
    location / {
        proxy_pass http://benchmark_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Connection pooling
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        
        # Timeouts
        proxy_connect_timeout 10s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

**Auto-scaling Configuration**:
```python
# performance/autoscaling.py
import psutil
import time
from kubernetes import client, config

class AutoScaler:
    def __init__(self, min_replicas=2, max_replicas=10):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        
        # Load Kubernetes config
        config.load_incluster_config()
        self.apps_v1 = client.AppsV1Api()
        
    def get_current_load(self):
        """Get current system load metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'load_average': psutil.getloadavg()[0]
        }
    
    def should_scale_up(self, load_metrics):
        """Determine if scaling up is needed"""
        return (load_metrics['cpu_percent'] > 80 or 
                load_metrics['memory_percent'] > 85 or
                load_metrics['load_average'] > psutil.cpu_count() * 0.8)
    
    def should_scale_down(self, load_metrics):
        """Determine if scaling down is possible"""
        return (load_metrics['cpu_percent'] < 30 and 
                load_metrics['memory_percent'] < 50 and
                load_metrics['load_average'] < psutil.cpu_count() * 0.3)
    
    def scale_deployment(self, replicas):
        """Scale Kubernetes deployment"""
        try:
            body = {'spec': {'replicas': replicas}}
            self.apps_v1.patch_namespaced_deployment_scale(
                name='benchmark-framework',
                namespace='default',
                body=body
            )
            return True
        except Exception as e:
            print(f"Scaling failed: {e}")
            return False
```

## Performance Monitoring

### Real-time Performance Metrics

```python
# performance/metrics_collector.py
import time
import threading
from collections import deque
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    timestamp: float
    evaluations_per_second: float
    avg_latency: float
    p95_latency: float
    memory_usage: float
    gpu_utilization: float
    queue_depth: int

class PerformanceCollector:
    def __init__(self, window_size=1000):
        self.metrics_history = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.running = False
        
    def start_collection(self):
        """Start background metrics collection"""
        self.running = True
        thread = threading.Thread(target=self._collect_metrics)
        thread.daemon = True
        thread.start()
    
    def record_evaluation(self, latency):
        """Record single evaluation metrics"""
        self.latencies.append(latency)
    
    def get_performance_summary(self):
        """Get current performance summary"""
        if not self.latencies:
            return None
        
        latencies = list(self.latencies)
        latencies.sort()
        
        return {
            'avg_latency': sum(latencies) / len(latencies),
            'p50_latency': latencies[len(latencies) // 2],
            'p95_latency': latencies[int(len(latencies) * 0.95)],
            'p99_latency': latencies[int(len(latencies) * 0.99)],
            'evaluations_per_second': len(latencies) / 60.0,  # Last minute
            'total_evaluations': len(latencies)
        }
```

## Performance Tuning Recommendations

### Hardware-Specific Optimizations

**GPU Tuning**:
```bash
#!/bin/bash
# scripts/gpu_tuning.sh

# Set GPU power limit to maximum
nvidia-smi -pl 450  # For RTX 5090

# Set GPU memory clock
nvidia-smi -mc 5600

# Set GPU core clock (be conservative)
nvidia-smi -cc 2500

# Enable persistence mode
nvidia-smi -pm 1

# Set compute mode to exclusive
nvidia-smi -c 3
```

**CPU Tuning**:
```bash
#!/bin/bash
# scripts/cpu_tuning.sh

# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU idle states for consistent latency
for i in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
    echo 1 > $i 2>/dev/null
done

# Set CPU affinity for network interrupts
echo 1 | sudo tee /proc/irq/*/smp_affinity
```

### Application-Level Optimizations

**Configuration Tuning**:
```yaml
# config/performance_tuning.yaml
evaluation:
  # Optimize batch sizes for different domains
  batch_sizes:
    reasoning: 32
    creativity: 16
    language: 64
    social: 24
    integration: 8
    knowledge: 48
  
  # Concurrent evaluation limits
  max_concurrent_evaluations: 20
  evaluation_timeout: 300
  
  # Memory management
  memory_threshold: 0.85
  gc_threshold: 1000
  
  # Caching configuration
  cache_enabled: true
  cache_ttl: 3600
  cache_size_limit: "2GB"

performance:
  # Worker configuration
  worker_processes: 16
  threads_per_worker: 2
  
  # I/O optimization
  async_io_enabled: true
  max_concurrent_io: 100
  
  # GPU optimization
  gpu_memory_fraction: 0.9
  cuda_streams: 4
```

## Troubleshooting Performance Issues

### Common Performance Problems

**Memory Leaks**:
```python
# performance/memory_profiling.py
import tracemalloc
import psutil
import gc

class MemoryProfiler:
    def __init__(self):
        self.baseline_memory = None
        
    def start_profiling(self):
        """Start memory profiling"""
        tracemalloc.start()
        self.baseline_memory = psutil.virtual_memory().used
        
    def check_memory_growth(self):
        """Check for memory growth"""
        current_memory = psutil.virtual_memory().used
        growth = current_memory - self.baseline_memory
        
        if growth > 1024 * 1024 * 1024:  # 1GB growth
            print(f"Warning: Memory growth detected: {growth / (1024**3):.2f} GB")
            
            # Get top memory consuming objects
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage: {current / (1024**2):.2f} MB")
            
            # Get memory statistics
            top_stats = tracemalloc.take_snapshot().statistics('lineno')
            for stat in top_stats[:10]:
                print(stat)
            
            return True
        return False
    
    def force_cleanup(self):
        """Force memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

**Performance Bottleneck Detection**:
```python
# performance/bottleneck_detector.py
import cProfile
import pstats
from functools import wraps

def profile_performance(func):
    """Decorator to profile function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        return result
    return wrapper
```

## References

- [Deployment](./deployment.md)
- [Monitoring](./monitoring.md)
- [Scaling](./scaling.md)
- [Configuration](../engineering/configuration.md)