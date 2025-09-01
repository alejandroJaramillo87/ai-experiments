# Monitoring

Comprehensive monitoring and observability for the AI Model Evaluation Framework.

## Monitoring Stack

### Core Components

**Metrics Collection**:
- Prometheus for metrics storage and alerting
- Node Exporter for system metrics
- NVIDIA GPU Exporter for GPU monitoring
- Custom application metrics

**Visualization**:
- Grafana dashboards for metrics visualization
- Real-time performance monitoring
- Historical trend analysis

**Logging**:
- Centralized logging with ELK Stack (Elasticsearch, Logstash, Kibana)
- Structured JSON logging
- Log aggregation and analysis

**Alerting**:
- Prometheus Alertmanager
- PagerDuty integration
- Slack/Email notifications

## Application Metrics

### Core Performance Metrics

**Evaluation Metrics**:
```python
# metrics/evaluation_metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Evaluation counters
EVALUATIONS_TOTAL = Counter(
    'evaluations_total',
    'Total number of evaluations performed',
    ['domain', 'difficulty', 'cultural_context']
)

EVALUATION_ERRORS = Counter(
    'evaluation_errors_total',
    'Total evaluation errors',
    ['error_type', 'domain']
)

# Evaluation latency
EVALUATION_DURATION = Histogram(
    'evaluation_duration_seconds',
    'Time spent on evaluations',
    ['domain', 'difficulty'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0)
)

# Queue metrics
EVALUATION_QUEUE_SIZE = Gauge(
    'evaluation_queue_size',
    'Number of evaluations waiting in queue'
)

# Resource utilization
MEMORY_USAGE = Gauge(
    'memory_usage_bytes',
    'Current memory usage'
)

GPU_UTILIZATION = Gauge(
    'gpu_utilization_percent',
    'GPU utilization percentage',
    ['gpu_id']
)
```

**Model Performance Metrics**:
```python
# metrics/model_metrics.py

MODEL_RESPONSE_TIME = Histogram(
    'model_response_time_seconds',
    'Model response generation time',
    ['model_name', 'model_type'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

MODEL_THROUGHPUT = Gauge(
    'model_throughput_tokens_per_second',
    'Model token generation throughput',
    ['model_name']
)

MODEL_ERRORS = Counter(
    'model_errors_total',
    'Model generation errors',
    ['model_name', 'error_type']
)

CULTURAL_AUTHENTICITY_SCORES = Histogram(
    'cultural_authenticity_scores',
    'Distribution of cultural authenticity scores',
    ['cultural_context'],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)
```

### Business Metrics

**Domain-Specific Performance**:
```python
# Track performance across cognitive domains
DOMAIN_PERFORMANCE = Histogram(
    'domain_performance_scores',
    'Performance scores by domain',
    ['domain', 'model_type', 'difficulty'],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

CROSS_CULTURAL_CONSISTENCY = Gauge(
    'cross_cultural_consistency_score',
    'Cross-cultural evaluation consistency',
    ['domain', 'model_name']
)
```

## System Monitoring

### Infrastructure Metrics

**Prometheus Configuration**:
```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'benchmark-framework'
    static_configs:
      - targets: ['benchmark-framework:8080']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'nvidia-gpu-exporter'
    static_configs:
      - targets: ['gpu-exporter:9445']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

**System Resource Monitoring**:
```bash
#!/bin/bash
# scripts/system_monitor.sh

while true; do
    # CPU usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    
    # Memory usage
    mem_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    
    # GPU usage
    gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    gpu_memory=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)
    
    # Disk usage
    disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    # Log metrics
    echo "$(date '+%Y-%m-%d %H:%M:%S') CPU: ${cpu_usage}% MEM: ${mem_usage}% GPU: ${gpu_usage}% DISK: ${disk_usage}%"
    
    sleep 30
done
```

### Application Health Monitoring

**Health Check Endpoints**:
```python
# health_checks.py
from flask import Blueprint, jsonify
import psutil
import torch
import redis
import time

health_bp = Blueprint('health', __name__)

@health_bp.route('/health')
def health_check():
    """Basic health check endpoint"""
    checks = {
        'status': 'healthy',
        'timestamp': time.time(),
        'version': get_app_version(),
        'uptime': get_uptime()
    }
    return jsonify(checks)

@health_bp.route('/ready')
def readiness_check():
    """Readiness check for load balancer"""
    checks = {}
    overall_status = True
    
    # Check database connection
    try:
        r = redis.Redis(host='redis', port=6379)
        r.ping()
        checks['redis'] = 'healthy'
    except Exception as e:
        checks['redis'] = f'unhealthy: {str(e)}'
        overall_status = False
    
    # Check GPU availability
    try:
        if torch.cuda.is_available():
            torch.cuda.current_device()
            checks['gpu'] = 'available'
        else:
            checks['gpu'] = 'unavailable'
    except Exception as e:
        checks['gpu'] = f'error: {str(e)}'
        overall_status = False
    
    # Check system resources
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    
    if cpu_percent > 90:
        checks['cpu'] = f'high_usage: {cpu_percent}%'
        overall_status = False
    else:
        checks['cpu'] = f'normal: {cpu_percent}%'
    
    if memory_percent > 85:
        checks['memory'] = f'high_usage: {memory_percent}%'
        overall_status = False
    else:
        checks['memory'] = f'normal: {memory_percent}%'
    
    status_code = 200 if overall_status else 503
    return jsonify({
        'status': 'ready' if overall_status else 'not_ready',
        'checks': checks
    }), status_code

@health_bp.route('/metrics')
def metrics_endpoint():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
```

## Alerting Rules

### Critical Alerts

**Prometheus Alert Rules**:
```yaml
# prometheus/rules/critical.yml
groups:
- name: critical_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(evaluation_errors_total[5m]) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High evaluation error rate detected"
      description: "Error rate is {{ $value }} errors per second"

  - alert: HighLatency
    expr: histogram_quantile(0.95, evaluation_duration_seconds_bucket) > 30
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High evaluation latency detected"
      description: "95th percentile latency is {{ $value }} seconds"

  - alert: SystemDown
    expr: up{job="benchmark-framework"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Benchmark framework is down"
      description: "The benchmark framework has been down for more than 1 minute"

  - alert: HighCPUUsage
    expr: 100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 90
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is {{ $value }}% on {{ $labels.instance }}"

  - alert: HighMemoryUsage
    expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is {{ $value }}% on {{ $labels.instance }}"

  - alert: GPUUtilizationHigh
    expr: nvidia_gpu_utilization > 95
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "High GPU utilization"
      description: "GPU {{ $labels.gpu_id }} utilization is {{ $value }}%"

  - alert: DiskSpaceLow
    expr: (1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100 > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Low disk space"
      description: "Disk usage is {{ $value }}% on {{ $labels.instance }}"
```

**Alertmanager Configuration**:
```yaml
# alertmanager/alertmanager.yml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@yourcompany.com'

templates:
  - '/etc/alertmanager/templates/*.tmpl'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default-receiver'
  routes:
  - match:
      severity: critical
    receiver: 'critical-receiver'
    continue: true

receivers:
- name: 'default-receiver'
  email_configs:
  - to: 'team@yourcompany.com'
    subject: '[ALERT] {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}

- name: 'critical-receiver'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#alerts'
    title: 'CRITICAL ALERT: {{ .GroupLabels.alertname }}'
    text: |
      {{ range .Alerts }}
      {{ .Annotations.summary }}
      {{ .Annotations.description }}
      {{ end }}
  pagerduty_configs:
  - routing_key: 'YOUR_PAGERDUTY_INTEGRATION_KEY'
    description: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'
```

## Grafana Dashboards

### Main Dashboard Configuration

```json
{
  "dashboard": {
    "title": "AI Evaluation Framework - Overview",
    "panels": [
      {
        "title": "Evaluations per Second",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(evaluations_total[5m])",
            "legendFormat": "Evaluations/sec"
          }
        ]
      },
      {
        "title": "Evaluation Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, evaluation_duration_seconds_bucket)",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, evaluation_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.99, evaluation_duration_seconds_bucket)",
            "legendFormat": "99th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(evaluation_errors_total[5m])",
            "legendFormat": "Errors/sec"
          }
        ]
      },
      {
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - (avg by (instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "CPU Usage %"
          },
          {
            "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
            "legendFormat": "Memory Usage %"
          },
          {
            "expr": "nvidia_gpu_utilization",
            "legendFormat": "GPU Usage %"
          }
        ]
      }
    ]
  }
}
```

### Cultural Performance Dashboard

```python
# grafana/cultural_dashboard.py
CULTURAL_DASHBOARD = {
    "title": "Cultural Authenticity Monitoring",
    "panels": [
        {
            "title": "Cultural Authenticity Scores by Context",
            "type": "heatmap",
            "targets": [
                {
                    "expr": "cultural_authenticity_scores_bucket",
                    "legendFormat": "{{cultural_context}}"
                }
            ]
        },
        {
            "title": "Cross-Cultural Consistency",
            "type": "stat",
            "targets": [
                {
                    "expr": "avg(cross_cultural_consistency_score)",
                    "legendFormat": "Avg Consistency"
                }
            ]
        }
    ]
}
```

## Log Management

### Structured Logging

**Application Logging Configuration**:
```python
# logging/config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'evaluation_id'):
            log_data['evaluation_id'] = record.evaluation_id
        if hasattr(record, 'domain'):
            log_data['domain'] = record.domain
        if hasattr(record, 'cultural_context'):
            log_data['cultural_context'] = record.cultural_context
            
        return json.dumps(log_data)

def setup_logging():
    logger = logging.getLogger('benchmark_framework')
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    return logger
```

**ELK Stack Configuration**:
```yaml
# elk/logstash.conf
input {
  beats {
    port => 5044
  }
}

filter {
  if [fields][service] == "benchmark-framework" {
    json {
      source => "message"
    }
    
    date {
      match => [ "timestamp", "ISO8601" ]
    }
    
    if [level] == "ERROR" {
      mutate {
        add_tag => [ "error" ]
      }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "benchmark-framework-%{+YYYY.MM.dd}"
  }
}
```

### Log Analysis Queries

**Common Kibana Queries**:
```
# Error analysis
level:ERROR AND timestamp:[now-1h TO now]

# High latency evaluations
message:"evaluation completed" AND duration:>30

# Cultural context analysis
cultural_context:* AND level:INFO

# Domain performance tracking
domain:reasoning AND message:"evaluation result"
```

## Performance Monitoring

### Real-time Performance Tracking

```python
# monitoring/performance_tracker.py
import time
import psutil
import threading
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PerformanceSnapshot:
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_utilization: float
    active_evaluations: int
    queue_size: int

class PerformanceTracker:
    def __init__(self):
        self.snapshots: List[PerformanceSnapshot] = []
        self.running = False
        
    def start_tracking(self):
        self.running = True
        thread = threading.Thread(target=self._track_performance)
        thread.daemon = True
        thread.start()
        
    def _track_performance(self):
        while self.running:
            try:
                snapshot = PerformanceSnapshot(
                    timestamp=time.time(),
                    cpu_percent=psutil.cpu_percent(),
                    memory_percent=psutil.virtual_memory().percent,
                    gpu_utilization=self._get_gpu_utilization(),
                    active_evaluations=self._get_active_evaluations(),
                    queue_size=self._get_queue_size()
                )
                
                self.snapshots.append(snapshot)
                
                # Keep only last 1000 snapshots
                if len(self.snapshots) > 1000:
                    self.snapshots = self.snapshots[-1000:]
                    
            except Exception as e:
                logging.error(f"Performance tracking error: {e}")
                
            time.sleep(10)
    
    def get_performance_summary(self, minutes=60):
        """Get performance summary for last N minutes"""
        cutoff_time = time.time() - (minutes * 60)
        recent_snapshots = [s for s in self.snapshots if s.timestamp > cutoff_time]
        
        if not recent_snapshots:
            return None
            
        return {
            'avg_cpu': sum(s.cpu_percent for s in recent_snapshots) / len(recent_snapshots),
            'max_cpu': max(s.cpu_percent for s in recent_snapshots),
            'avg_memory': sum(s.memory_percent for s in recent_snapshots) / len(recent_snapshots),
            'max_memory': max(s.memory_percent for s in recent_snapshots),
            'avg_gpu': sum(s.gpu_utilization for s in recent_snapshots) / len(recent_snapshots),
            'max_queue_size': max(s.queue_size for s in recent_snapshots),
            'total_snapshots': len(recent_snapshots)
        }
```

## Monitoring Automation

### Automated Monitoring Scripts

```bash
#!/bin/bash
# scripts/monitoring_automation.sh

# Check system health and send alerts if needed
check_system_health() {
    # Check disk space
    DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ $DISK_USAGE -gt 80 ]; then
        send_alert "High disk usage: ${DISK_USAGE}%"
    fi
    
    # Check memory usage
    MEM_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    if (( $(echo "$MEM_USAGE > 85" | bc -l) )); then
        send_alert "High memory usage: ${MEM_USAGE}%"
    fi
    
    # Check GPU temperature
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    if [ $GPU_TEMP -gt 80 ]; then
        send_alert "High GPU temperature: ${GPU_TEMP}°C"
    fi
}

send_alert() {
    local message="$1"
    echo "$(date): ALERT - $message"
    # Send to monitoring system
    curl -X POST -H "Content-Type: application/json" \
         -d "{\"text\":\"$message\"}" \
         "$SLACK_WEBHOOK_URL"
}

# Run health check every 5 minutes
while true; do
    check_system_health
    sleep 300
done
```

## References

- [Deployment](./deployment.md)
- [Performance](./performance.md)
- [Security](./security.md)
- [Maintenance](./maintenance.md)