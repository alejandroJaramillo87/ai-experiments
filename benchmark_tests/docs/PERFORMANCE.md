# Performance Guide

This guide covers performance optimization, monitoring, and tuning for the AI Model Evaluation Framework. Think of this as your **performance engineering manual** - techniques for maximizing throughput, minimizing latency, and efficient resource utilization.

## 🎯 **Performance Goals and Benchmarks**

### **Target Performance Metrics**
```
Evaluation Throughput: 50-200 evaluations/minute
Response Time: <5 seconds per evaluation
Memory Usage: <4GB for concurrent processing
CPU Utilization: 70-85% (sustainable load)
Network Latency: <100ms to model API
Storage I/O: <50ms per result write
```

### **Hardware-Specific Optimization**
**RTX 5090 + AMD Ryzen 9950X + 128GB RAM Configuration**:
```
CPU Cores: Utilize 12-14 cores (leave 2 for system)
Memory: Use up to 96GB (leave 32GB for OS/cache)
GPU: Reserve for local model inference (optional)
Storage: NVMe SSD for optimal I/O performance
Network: Dedicated 10Gbps+ connection to model APIs
```

## 🚀 **Optimization Strategies**

### **1. Concurrent Processing Optimization**

**Optimal Worker Configuration**:
```python
import psutil
import os

def calculate_optimal_workers():
    """Calculate optimal number of concurrent workers based on system specs."""
    
    # Get system specifications
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Conservative worker calculation
    # Rule: 1 worker per 2 physical cores, limited by memory
    cpu_workers = max(1, cpu_count // 2)
    memory_workers = max(1, int(memory_gb // 4))  # 4GB per worker
    
    optimal_workers = min(cpu_workers, memory_workers, 16)  # Cap at 16
    
    return {
        'recommended_workers': optimal_workers,
        'max_safe_workers': min(cpu_workers, 20),
        'memory_per_worker_gb': memory_gb / optimal_workers,
        'system_info': {
            'cpu_cores': cpu_count,
            'memory_gb': memory_gb
        }
    }

# Example usage
config = calculate_optimal_workers()
print(f"Recommended workers: {config['recommended_workers']}")
```

**Advanced Concurrent Configuration**:
```json
{
  "concurrency": {
    "workers": 12,
    "batch_size": 20,
    "queue_size": 100,
    "worker_timeout": 300,
    "rate_limiting": {
      "requests_per_second": 10,
      "burst_size": 15,
      "backoff_factor": 1.5
    },
    "resource_limits": {
      "memory_per_worker_mb": 3072,
      "cpu_affinity": "auto",
      "max_concurrent_api_calls": 8
    }
  }
}
```

**High-Performance Worker Implementation**:
```python
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

class HighPerformanceEvaluator:
    """High-performance concurrent evaluation engine."""
    
    def __init__(self, max_workers: int = 12):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.session = None
        self.semaphore = asyncio.Semaphore(max_workers)
        
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(
            limit=100,           # Total connection pool size
            limit_per_host=30,   # Connections per host
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        timeout = aiohttp.ClientTimeout(total=300, connect=30)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    async def evaluate_batch_async(self, 
                                  tests: List[Dict[str, Any]],
                                  model_endpoint: str) -> List[Dict[str, Any]]:
        """Asynchronously evaluate a batch of tests."""
        
        async def evaluate_single(test: Dict[str, Any]) -> Dict[str, Any]:
            async with self.semaphore:  # Limit concurrent requests
                try:
                    # Make API call
                    async with self.session.post(
                        model_endpoint,
                        json={
                            "model": test.get("model", "default"),
                            "prompt": test["prompt"],
                            "max_tokens": test.get("max_tokens", 500)
                        }
                    ) as response:
                        model_response = await response.json()
                    
                    # Run evaluation in thread pool (CPU-intensive)
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor,
                        self._evaluate_response,
                        model_response,
                        test
                    )
                    
                    return result
                    
                except Exception as e:
                    return {"test_id": test["test_id"], "error": str(e)}
        
        # Process all tests concurrently
        tasks = [evaluate_single(test) for test in tests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    def _evaluate_response(self, response: Dict, test: Dict) -> Dict[str, Any]:
        """CPU-intensive evaluation logic (runs in thread pool)."""
        # Actual evaluation logic here
        # This runs in a separate thread to avoid blocking async loop
        pass
```

---

### **2. Memory Optimization**

**Smart Caching Strategy**:
```python
import functools
import weakref
from typing import Dict, Any, Optional

class MemoryOptimizedCache:
    """Memory-efficient caching with automatic cleanup."""
    
    def __init__(self, max_size_mb: int = 1000):
        self.max_size_mb = max_size_mb
        self.cache = {}
        self.size_tracker = {}
        self.current_size_mb = 0
        
    def cache_result(self, key: str, result: Any) -> None:
        """Cache evaluation result with memory tracking."""
        import sys
        
        # Calculate approximate size
        result_size = sys.getsizeof(result) / (1024 * 1024)  # MB
        
        # Clean up if needed
        if self.current_size_mb + result_size > self.max_size_mb:
            self._cleanup_old_entries(result_size)
        
        self.cache[key] = result
        self.size_tracker[key] = result_size
        self.current_size_mb += result_size
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if available."""
        return self.cache.get(key)
    
    def _cleanup_old_entries(self, needed_space: float) -> None:
        """Clean up old cache entries to make space."""
        import heapq
        
        # Sort by last access time (implement LRU)
        entries_to_remove = []
        freed_space = 0
        
        for key, size in self.size_tracker.items():
            if freed_space >= needed_space:
                break
            entries_to_remove.append(key)
            freed_space += size
        
        for key in entries_to_remove:
            if key in self.cache:
                del self.cache[key]
                self.current_size_mb -= self.size_tracker[key]
                del self.size_tracker[key]

# Global cache instance
evaluation_cache = MemoryOptimizedCache(max_size_mb=2000)  # 2GB cache

@functools.lru_cache(maxsize=1000)
def cached_pattern_analysis(text_hash: str) -> Dict[str, Any]:
    """Cache expensive pattern analysis operations."""
    # Expensive analysis logic here
    pass
```

**Memory-Efficient Batch Processing**:
```python
class BatchProcessor:
    """Process large batches without memory buildup."""
    
    def __init__(self, batch_size: int = 50):
        self.batch_size = batch_size
        
    def process_large_dataset(self, 
                            test_iterator,
                            output_callback) -> None:
        """Process large datasets in memory-efficient batches."""
        
        batch = []
        batch_count = 0
        
        for test in test_iterator:
            batch.append(test)
            
            if len(batch) >= self.batch_size:
                # Process batch
                results = self._process_batch(batch)
                
                # Save results immediately
                output_callback(results, batch_count)
                
                # Clear memory
                del batch
                del results
                batch = []
                batch_count += 1
                
                # Force garbage collection periodically
                if batch_count % 10 == 0:
                    import gc
                    gc.collect()
        
        # Process remaining items
        if batch:
            results = self._process_batch(batch)
            output_callback(results, batch_count)
    
    def _process_batch(self, batch: List[Dict]) -> List[Dict]:
        """Process a single batch of tests."""
        # Batch processing logic
        results = []
        for test in batch:
            # Process individual test
            result = self._process_single_test(test)
            results.append(result)
        return results
```

---

### **3. I/O and Storage Optimization**

**Efficient Result Storage**:
```python
import json
import gzip
from pathlib import Path
from typing import Dict, Any, List
import asyncio
import aiofiles

class OptimizedResultStorage:
    """High-performance result storage with compression and batching."""
    
    def __init__(self, base_path: str, compression: bool = True):
        self.base_path = Path(base_path)
        self.compression = compression
        self.write_queue = asyncio.Queue()
        self.writer_task = None
        
    async def start_writer(self):
        """Start background writer task."""
        self.writer_task = asyncio.create_task(self._background_writer())
    
    async def stop_writer(self):
        """Stop background writer task."""
        if self.writer_task:
            self.write_queue.put_nowait(None)  # Sentinel to stop
            await self.writer_task
    
    async def store_result_async(self, result: Dict[str, Any]) -> None:
        """Queue result for async storage."""
        await self.write_queue.put(result)
    
    async def _background_writer(self):
        """Background task that writes queued results."""
        while True:
            try:
                result = await self.write_queue.get()
                if result is None:  # Sentinel to stop
                    break
                    
                await self._write_single_result(result)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error writing result: {e}")
    
    async def _write_single_result(self, result: Dict[str, Any]) -> None:
        """Write single result to storage."""
        test_id = result.get("test_id", "unknown")
        timestamp = result.get("timestamp", "")
        
        filename = f"{test_id}_{timestamp}.json"
        if self.compression:
            filename += ".gz"
            
        filepath = self.base_path / filename
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        data = json.dumps(result, indent=2).encode('utf-8')
        
        if self.compression:
            data = gzip.compress(data)
            
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(data)
    
    def store_batch_sync(self, results: List[Dict[str, Any]]) -> None:
        """Synchronously store batch of results (for compatibility)."""
        for result in results:
            test_id = result.get("test_id", "unknown")
            timestamp = result.get("timestamp", "")
            
            filename = f"{test_id}_{timestamp}.json"
            if self.compression:
                filename += ".gz"
                
            filepath = self.base_path / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            data = json.dumps(result, indent=2)
            
            if self.compression:
                with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                    f.write(data)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(data)
```

**Database Storage for High Volume**:
```python
import sqlite3
import json
from contextlib import contextmanager
from typing import Dict, Any, List

class DatabaseResultStorage:
    """High-performance database storage for evaluation results."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    overall_score REAL,
                    dimension_scores TEXT,  -- JSON
                    model_response TEXT,
                    execution_time_ms INTEGER,
                    success BOOLEAN,
                    metadata TEXT,  -- JSON
                    
                    -- Indexes for fast queries
                    INDEX idx_test_id (test_id),
                    INDEX idx_model_name (model_name),
                    INDEX idx_timestamp (timestamp),
                    INDEX idx_overall_score (overall_score)
                );
                
                CREATE TABLE IF NOT EXISTS batch_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_tests INTEGER,
                    success_count INTEGER,
                    average_score REAL,
                    execution_time_ms INTEGER,
                    metadata TEXT  -- JSON
                );
            ''')
    
    def store_results_batch(self, results: List[Dict[str, Any]]) -> None:
        """Store batch of results efficiently."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany('''
                INSERT INTO evaluation_results (
                    test_id, model_name, overall_score, dimension_scores,
                    model_response, execution_time_ms, success, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', [
                (
                    result.get("test_id"),
                    result.get("model_name"),
                    result.get("overall_score"),
                    json.dumps(result.get("dimension_scores", {})),
                    result.get("model_response", ""),
                    result.get("execution_time_ms", 0),
                    result.get("success", False),
                    json.dumps(result.get("metadata", {}))
                ) for result in results
            ])
    
    def query_results(self, 
                     model_name: str = None,
                     test_category: str = None,
                     min_score: float = None,
                     limit: int = 1000) -> List[Dict[str, Any]]:
        """Query results with efficient filtering."""
        
        query = "SELECT * FROM evaluation_results WHERE 1=1"
        params = []
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
            
        if min_score:
            query += " AND overall_score >= ?"
            params.append(min_score)
            
        if test_category:
            query += " AND test_id LIKE ?"
            params.append(f"{test_category}%")
            
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.execute(query, params)
            
            results = []
            for row in cursor:
                result = dict(row)
                result["dimension_scores"] = json.loads(result["dimension_scores"])
                result["metadata"] = json.loads(result["metadata"])
                results.append(result)
                
            return results
```

---

### **4. Network Optimization**

**Connection Pooling and Keep-Alive**:
```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class OptimizedAPIClient:
    """High-performance API client with connection pooling."""
    
    def __init__(self, base_url: str, max_connections: int = 20):
        self.base_url = base_url
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )
        
        # Configure adapters with connection pooling
        adapter = HTTPAdapter(
            pool_connections=max_connections,
            pool_maxsize=max_connections,
            max_retries=retry_strategy,
            pool_block=True
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set keep-alive and timeout
        self.session.headers.update({
            'Connection': 'keep-alive',
            'User-Agent': 'BenchmarkTests/1.0'
        })
        
    def make_evaluation_request(self, 
                               prompt: str,
                               model_name: str,
                               **kwargs) -> Dict[str, Any]:
        """Make optimized evaluation request."""
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 500),
            "temperature": kwargs.get("temperature", 0.7),
            **kwargs
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/completions",
                json=payload,
                timeout=(10, 60)  # (connect_timeout, read_timeout)
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise APIError(f"API request failed: {str(e)}")
```

**Load Balancing for Multiple Model Endpoints**:
```python
import random
import time
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class EndpointHealth:
    url: str
    response_time_ms: float = 0.0
    success_rate: float = 1.0
    last_check: float = 0.0
    consecutive_failures: int = 0
    
class LoadBalancedAPIClient:
    """Load balancer for multiple model API endpoints."""
    
    def __init__(self, endpoints: List[str]):
        self.endpoints = [
            EndpointHealth(url=url) for url in endpoints
        ]
        self.health_check_interval = 60  # seconds
        
    def get_best_endpoint(self) -> str:
        """Select best endpoint based on health metrics."""
        
        # Update health checks if needed
        current_time = time.time()
        for endpoint in self.endpoints:
            if current_time - endpoint.last_check > self.health_check_interval:
                self._check_endpoint_health(endpoint)
        
        # Filter healthy endpoints
        healthy_endpoints = [
            ep for ep in self.endpoints 
            if ep.consecutive_failures < 3 and ep.success_rate > 0.5
        ]
        
        if not healthy_endpoints:
            # All endpoints unhealthy, use original list
            healthy_endpoints = self.endpoints
        
        # Weight by performance (lower response time = higher weight)
        weights = []
        for endpoint in healthy_endpoints:
            # Inverse of response time (faster = higher weight)
            weight = 1.0 / (endpoint.response_time_ms + 1)
            # Multiply by success rate
            weight *= endpoint.success_rate
            weights.append(weight)
        
        # Weighted random selection
        if weights:
            return random.choices(healthy_endpoints, weights=weights)[0].url
        else:
            return random.choice(healthy_endpoints).url
    
    def _check_endpoint_health(self, endpoint: EndpointHealth):
        """Check health of individual endpoint."""
        start_time = time.time()
        
        try:
            response = requests.get(
                f"{endpoint.url}/health",
                timeout=5
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                endpoint.response_time_ms = response_time
                endpoint.success_rate = min(1.0, endpoint.success_rate + 0.1)
                endpoint.consecutive_failures = 0
            else:
                endpoint.consecutive_failures += 1
                endpoint.success_rate = max(0.0, endpoint.success_rate - 0.2)
                
        except Exception:
            endpoint.consecutive_failures += 1
            endpoint.success_rate = max(0.0, endpoint.success_rate - 0.3)
            endpoint.response_time_ms = 5000  # Penalty for failures
            
        endpoint.last_check = time.time()
```

## 📊 **Performance Monitoring**

### **Real-time Metrics Collection**

**metrics_collector.py**:
```python
import time
import psutil
import threading
from collections import deque, defaultdict
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class PerformanceSnapshot:
    """Single performance measurement snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    
    # Application-specific metrics
    active_workers: int = 0
    evaluations_per_second: float = 0.0
    average_response_time_ms: float = 0.0
    api_calls_per_second: float = 0.0
    cache_hit_rate: float = 0.0

class RealTimeMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self, history_size: int = 3600):  # 1 hour at 1s intervals
        self.history_size = history_size
        self.snapshots = deque(maxlen=history_size)
        self.running = False
        self.monitor_thread = None
        
        # Application metrics
        self.app_metrics = {
            'evaluations_count': 0,
            'api_calls_count': 0,
            'cache_hits': 0,
            'cache_requests': 0,
            'active_workers': 0
        }
        self.last_evaluation_count = 0
        self.last_api_calls_count = 0
        
        # System metrics baseline
        self.last_disk_io = psutil.disk_io_counters()
        self.last_network_io = psutil.net_io_counters()
        
    def start_monitoring(self, interval: float = 1.0):
        """Start real-time monitoring."""
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.running:
            try:
                snapshot = self._collect_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def _collect_snapshot(self) -> PerformanceSnapshot:
        """Collect single performance snapshot."""
        current_time = time.time()
        
        # System metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        # Disk I/O
        current_disk_io = psutil.disk_io_counters()
        disk_read_mb = (current_disk_io.read_bytes - self.last_disk_io.read_bytes) / (1024*1024)
        disk_write_mb = (current_disk_io.write_bytes - self.last_disk_io.write_bytes) / (1024*1024)
        self.last_disk_io = current_disk_io
        
        # Network I/O
        current_network_io = psutil.net_io_counters()
        net_sent_mb = (current_network_io.bytes_sent - self.last_network_io.bytes_sent) / (1024*1024)
        net_recv_mb = (current_network_io.bytes_recv - self.last_network_io.bytes_recv) / (1024*1024)
        self.last_network_io = current_network_io
        
        # Application metrics
        evaluations_delta = self.app_metrics['evaluations_count'] - self.last_evaluation_count
        api_calls_delta = self.app_metrics['api_calls_count'] - self.last_api_calls_count
        
        evaluations_per_second = evaluations_delta  # Per interval (1 second)
        api_calls_per_second = api_calls_delta
        
        self.last_evaluation_count = self.app_metrics['evaluations_count']
        self.last_api_calls_count = self.app_metrics['api_calls_count']
        
        # Cache hit rate
        cache_requests = self.app_metrics['cache_requests']
        cache_hit_rate = (self.app_metrics['cache_hits'] / cache_requests) if cache_requests > 0 else 0.0
        
        return PerformanceSnapshot(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_sent_mb=net_sent_mb,
            network_recv_mb=net_recv_mb,
            active_workers=self.app_metrics['active_workers'],
            evaluations_per_second=evaluations_per_second,
            api_calls_per_second=api_calls_per_second,
            cache_hit_rate=cache_hit_rate
        )
    
    def get_current_metrics(self) -> Optional[PerformanceSnapshot]:
        """Get most recent performance snapshot."""
        return self.snapshots[-1] if self.snapshots else None
    
    def get_metrics_summary(self, duration_minutes: int = 10) -> Dict[str, Any]:
        """Get performance summary over specified duration."""
        if not self.snapshots:
            return {}
        
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_snapshots = [
            s for s in self.snapshots 
            if s.timestamp >= cutoff_time
        ]
        
        if not recent_snapshots:
            return {}
        
        return {
            'duration_minutes': duration_minutes,
            'sample_count': len(recent_snapshots),
            'cpu_percent': {
                'avg': sum(s.cpu_percent for s in recent_snapshots) / len(recent_snapshots),
                'max': max(s.cpu_percent for s in recent_snapshots),
                'min': min(s.cpu_percent for s in recent_snapshots)
            },
            'memory_percent': {
                'avg': sum(s.memory_percent for s in recent_snapshots) / len(recent_snapshots),
                'max': max(s.memory_percent for s in recent_snapshots)
            },
            'evaluations_per_second': {
                'avg': sum(s.evaluations_per_second for s in recent_snapshots) / len(recent_snapshots),
                'max': max(s.evaluations_per_second for s in recent_snapshots),
                'total': sum(s.evaluations_per_second for s in recent_snapshots)
            },
            'cache_hit_rate': {
                'current': recent_snapshots[-1].cache_hit_rate,
                'avg': sum(s.cache_hit_rate for s in recent_snapshots) / len(recent_snapshots)
            }
        }

# Global monitor instance
performance_monitor = RealTimeMonitor()
```

### **Performance Alerting**

**alerts.py**:
```python
from typing import Callable, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class AlertRule:
    name: str
    condition: Callable[[PerformanceSnapshot], bool]
    severity: AlertSeverity
    message_template: str
    cooldown_seconds: int = 300  # 5 minutes
    last_triggered: float = 0.0

class AlertManager:
    """Performance alerting system."""
    
    def __init__(self):
        self.rules = []
        self.alert_handlers = []
        
    def add_rule(self, rule: AlertRule):
        """Add performance alert rule."""
        self.rules.append(rule)
    
    def add_handler(self, handler: Callable[[str, AlertSeverity, Dict], None]):
        """Add alert handler (email, slack, etc.)."""
        self.alert_handlers.append(handler)
    
    def check_alerts(self, snapshot: PerformanceSnapshot):
        """Check all alert rules against current snapshot."""
        current_time = time.time()
        
        for rule in self.rules:
            # Check cooldown
            if current_time - rule.last_triggered < rule.cooldown_seconds:
                continue
                
            # Check condition
            if rule.condition(snapshot):
                # Trigger alert
                message = rule.message_template.format(
                    cpu=snapshot.cpu_percent,
                    memory=snapshot.memory_percent,
                    evaluations_per_sec=snapshot.evaluations_per_second,
                    timestamp=snapshot.timestamp
                )
                
                # Send to all handlers
                for handler in self.alert_handlers:
                    handler(rule.name, rule.severity, {
                        'message': message,
                        'snapshot': asdict(snapshot),
                        'rule': rule.name
                    })
                
                rule.last_triggered = current_time

def setup_default_alerts(alert_manager: AlertManager):
    """Setup default performance alerts."""
    
    # High CPU usage
    alert_manager.add_rule(AlertRule(
        name="high_cpu_usage",
        condition=lambda s: s.cpu_percent > 90,
        severity=AlertSeverity.WARNING,
        message_template="High CPU usage: {cpu:.1f}% at {timestamp}",
        cooldown_seconds=300
    ))
    
    # High memory usage
    alert_manager.add_rule(AlertRule(
        name="high_memory_usage", 
        condition=lambda s: s.memory_percent > 85,
        severity=AlertSeverity.CRITICAL,
        message_template="High memory usage: {memory:.1f}% at {timestamp}",
        cooldown_seconds=180
    ))
    
    # Low evaluation rate
    alert_manager.add_rule(AlertRule(
        name="low_evaluation_rate",
        condition=lambda s: s.evaluations_per_second < 1.0 and s.active_workers > 0,
        severity=AlertSeverity.WARNING,
        message_template="Low evaluation rate: {evaluations_per_sec:.1f}/sec with active workers",
        cooldown_seconds=600
    ))
    
    # Low cache hit rate
    alert_manager.add_rule(AlertRule(
        name="low_cache_hit_rate",
        condition=lambda s: s.cache_hit_rate < 0.3,
        severity=AlertSeverity.INFO,
        message_template="Low cache hit rate: {cache_hit_rate:.1%}",
        cooldown_seconds=900
    ))

def email_alert_handler(rule_name: str, severity: AlertSeverity, data: Dict[str, Any]):
    """Email alert handler."""
    # Implement email sending logic
    print(f"EMAIL ALERT [{severity.value.upper()}] {rule_name}: {data['message']}")

def slack_alert_handler(rule_name: str, severity: AlertSeverity, data: Dict[str, Any]):
    """Slack alert handler."""
    # Implement Slack webhook logic
    print(f"SLACK ALERT [{severity.value.upper()}] {rule_name}: {data['message']}")
```

## ⚡ **Performance Tuning Techniques**

### **1. Evaluation-Specific Optimizations**

**Smart Evaluation Caching**:
```python
import hashlib
from functools import wraps

def cache_evaluation_result(cache_duration_seconds: int = 3600):
    """Cache evaluation results based on text content hash."""
    
    def decorator(evaluate_func):
        @wraps(evaluate_func)
        def wrapper(self, text: str, context: Dict[str, Any]) -> Any:
            # Create cache key from text content and evaluator version
            content_hash = hashlib.md5(
                f"{text}_{self.evaluator_name}_{self.version}".encode()
            ).hexdigest()
            
            # Check cache
            cached_result = evaluation_cache.get_cached_result(content_hash)
            if cached_result:
                return cached_result
            
            # Perform evaluation
            result = evaluate_func(self, text, context)
            
            # Cache result
            evaluation_cache.cache_result(content_hash, result)
            
            return result
        
        return wrapper
    return decorator

# Usage in evaluator
class OptimizedReasoningEvaluator(BaseEvaluator):
    @cache_evaluation_result(cache_duration_seconds=7200)  # 2 hours
    def evaluate(self, text: str, context: Dict[str, Any]) -> DomainEvaluationResult:
        # Actual evaluation logic
        pass
```

**Parallel Dimension Evaluation**:
```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class ParallelDimensionEvaluator(BaseEvaluator):
    """Evaluate multiple dimensions in parallel."""
    
    def __init__(self):
        super().__init__()
        self.dimension_evaluators = {
            'organization_quality': self._evaluate_organization,
            'technical_accuracy': self._evaluate_accuracy,
            'completeness': self._evaluate_completeness,
            'reliability': self._evaluate_reliability
        }
    
    def evaluate(self, text: str, context: Dict[str, Any]) -> DomainEvaluationResult:
        """Evaluate all dimensions in parallel."""
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all dimension evaluations
            future_to_dimension = {
                executor.submit(evaluator, text, context): dimension
                for dimension, evaluator in self.dimension_evaluators.items()
            }
            
            dimensions = []
            for future in as_completed(future_to_dimension):
                dimension_name = future_to_dimension[future]
                try:
                    dimension_result = future.result()
                    dimensions.append(dimension_result)
                except Exception as e:
                    print(f"Error evaluating {dimension_name}: {e}")
                    # Create error dimension
                    dimensions.append(EvaluationDimension(
                        name=dimension_name,
                        score=0.0,
                        confidence=0.0,
                        cultural_relevance=0.0,
                        evidence=[f"Error: {str(e)}"],
                        cultural_markers=[]
                    ))
        
        # Calculate overall score
        overall_score = sum(dim.score for dim in dimensions) / len(dimensions)
        
        return DomainEvaluationResult(
            domain=self.evaluator_name,
            evaluation_type="parallel",
            overall_score=overall_score,
            dimensions=dimensions,
            cultural_context=context.get("cultural_context", CulturalContext()),
            metadata={"parallel_evaluation": True}
        )
```

### **2. System-Level Optimizations**

**CPU Affinity and NUMA Optimization**:
```python
import os
import psutil

def optimize_cpu_affinity(worker_count: int):
    """Optimize CPU affinity for worker processes."""
    
    # Get available CPU cores
    cpu_count = psutil.cpu_count(logical=False)
    logical_cpu_count = psutil.cpu_count(logical=True)
    
    # Distribute workers across physical cores
    if worker_count <= cpu_count:
        # One worker per physical core
        for i in range(worker_count):
            cpu_list = [i * 2, i * 2 + 1]  # Both logical cores of physical core
            os.sched_setaffinity(0, cpu_list)
    else:
        # Distribute evenly across all cores
        cores_per_worker = max(1, logical_cpu_count // worker_count)
        for i in range(worker_count):
            start_core = i * cores_per_worker
            end_core = min(start_core + cores_per_worker, logical_cpu_count)
            cpu_list = list(range(start_core, end_core))
            os.sched_setaffinity(0, cpu_list)

def set_process_priority():
    """Set appropriate process priority for evaluation tasks."""
    try:
        # Set higher priority for evaluation process
        os.nice(-5)  # Higher priority (requires permissions)
    except PermissionError:
        # Fallback: normal priority
        pass

def optimize_memory_settings():
    """Optimize memory settings for large-scale evaluation."""
    import gc
    
    # Tune garbage collection
    gc.set_threshold(700, 10, 10)  # More aggressive GC
    
    # Set memory allocation strategy
    try:
        import mmap
        # Use memory mapping for large files when possible
        pass
    except ImportError:
        pass
```

---

This comprehensive performance guide provides enterprise-grade optimization techniques for running the AI Model Evaluation Framework at scale. Implement these optimizations incrementally and monitor their impact on your specific workload and hardware configuration.