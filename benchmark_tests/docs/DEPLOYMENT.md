# Deployment Guide

This guide covers deploying the AI Model Evaluation Framework in production environments. Think of this as your **DevOps and infrastructure manual** - everything needed for reliable, scalable deployments.

## 🎯 **Deployment Scenarios**

### **Local Development** 
- Single developer machine
- Interactive testing and experimentation
- Resource constraints: laptop/desktop hardware

### **Team Development**
- Shared development server
- Continuous integration pipelines
- Multiple developers, coordinated testing

### **Production Evaluation**
- High-availability model assessment
- Automated monitoring and alerting
- Scale: hundreds to thousands of evaluations

### **Research Infrastructure**
- Large-scale academic/research usage
- Batch processing of evaluation campaigns
- Scale: thousands to millions of evaluations

## 🖥️ **Hardware Requirements**

### **Minimum Requirements** (Development)
```
CPU: 4 cores, 2.5GHz+
Memory: 8GB RAM
Storage: 10GB available space
Network: Stable internet for model API calls
```

### **Recommended Requirements** (Production)
```
CPU: 8-16 cores, 3.0GHz+ (AMD Ryzen 9950X ideal)
Memory: 32-64GB RAM (128GB for large-scale)
Storage: 100GB+ SSD (fast I/O for concurrent processing)
Network: 1Gbps+ with low latency to model APIs
GPU: Optional, RTX 5090 for local model inference
```

### **Scaling Considerations**
```
Concurrent Workers: 1 worker per 2 CPU cores
Memory per Worker: 2-4GB depending on evaluator complexity
Disk I/O: 100+ IOPS for result storage
Network: 10Mbps+ per concurrent model API call
```

## 🐳 **Containerized Deployment**

### **Docker Setup**

**Dockerfile**:
```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 evaluator && \
    chown -R evaluator:evaluator /app
USER evaluator

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "import evaluator; print('OK')" || exit 1

# Default command
CMD ["python", "benchmark_runner.py", "--help"]
```

**docker-compose.yml** (Complete Stack):
```yaml
version: '3.8'

services:
  # Main evaluation service
  benchmark-evaluator:
    build: .
    image: benchmark-tests:latest
    container_name: benchmark-evaluator
    restart: unless-stopped
    environment:
      - MODEL_ENDPOINT=${MODEL_ENDPOINT}
      - MODEL_NAME=${MODEL_NAME}
      - LOG_LEVEL=INFO
      - MAX_WORKERS=8
      - RESULTS_DIR=/data/results
    volumes:
      - ./data:/data
      - ./config:/app/config
      - ./logs:/app/logs
    networks:
      - evaluation-network
    depends_on:
      - redis-cache
      - monitoring
  
  # Redis for caching and coordination
  redis-cache:
    image: redis:7-alpine
    container_name: benchmark-redis
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    networks:
      - evaluation-network
    ports:
      - "6379:6379"
  
  # Monitoring with Prometheus
  monitoring:
    image: prom/prometheus:latest
    container_name: benchmark-prometheus
    restart: unless-stopped
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - evaluation-network
    ports:
      - "9090:9090"
  
  # Visualization with Grafana
  grafana:
    image: grafana/grafana:latest
    container_name: benchmark-grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - ./monitoring/grafana:/etc/grafana/provisioning
      - grafana-data:/var/lib/grafana
    networks:
      - evaluation-network
    ports:
      - "3000:3000"
    depends_on:
      - monitoring

volumes:
  redis-data:
  prometheus-data:
  grafana-data:

networks:
  evaluation-network:
    driver: bridge
```

**Build and Deploy**:
```bash
# Build image
docker build -t benchmark-tests:latest .

# Create environment file
cat > .env << EOF
MODEL_ENDPOINT=http://your-model-server:8000/v1/completions
MODEL_NAME=production-model
GRAFANA_PASSWORD=secure-password-here
EOF

# Start services
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f benchmark-evaluator
```

## ☸️ **Kubernetes Deployment**

### **Kubernetes Manifests**

**namespace.yaml**:
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: benchmark-evaluation
  labels:
    name: benchmark-evaluation
```

**configmap.yaml**:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: benchmark-config
  namespace: benchmark-evaluation
data:
  config.json: |
    {
      "model": {
        "endpoint": "http://model-service:8000/v1/completions",
        "timeout": 60,
        "max_retries": 3
      },
      "evaluation": {
        "concurrent_workers": 8,
        "performance_monitoring": true
      },
      "logging": {
        "level": "INFO",
        "format": "json"
      }
    }
```

**deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: benchmark-evaluator
  namespace: benchmark-evaluation
spec:
  replicas: 3
  selector:
    matchLabels:
      app: benchmark-evaluator
  template:
    metadata:
      labels:
        app: benchmark-evaluator
    spec:
      containers:
      - name: evaluator
        image: benchmark-tests:latest
        imagePullPolicy: Always
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        env:
        - name: MODEL_ENDPOINT
          valueFrom:
            secretKeyRef:
              name: model-credentials
              key: endpoint
        - name: MODEL_API_KEY
          valueFrom:
            secretKeyRef:
              name: model-credentials
              key: api-key
        - name: MAX_WORKERS
          value: "8"
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: data-volume
          mountPath: /data
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "import evaluator; print('Ready')"
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "import evaluator; print('Alive')"
          initialDelaySeconds: 60
          periodSeconds: 30
      volumes:
      - name: config-volume
        configMap:
          name: benchmark-config
      - name: data-volume
        persistentVolumeClaim:
          claimName: benchmark-data-pvc
```

**service.yaml**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: benchmark-evaluator-service
  namespace: benchmark-evaluation
spec:
  selector:
    app: benchmark-evaluator
  ports:
    - protocol: TCP
      port: 8080
      targetPort: 8080
  type: ClusterIP
```

**pvc.yaml**:
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: benchmark-data-pvc
  namespace: benchmark-evaluation
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
```

### **Deploy to Kubernetes**
```bash
# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check deployment
kubectl get pods -n benchmark-evaluation
kubectl logs -f deployment/benchmark-evaluator -n benchmark-evaluation

# Scale deployment
kubectl scale deployment benchmark-evaluator --replicas=5 -n benchmark-evaluation
```

## 🔄 **CI/CD Integration**

### **GitHub Actions Workflow**

**.github/workflows/deploy.yml**:
```yaml
name: Deploy Benchmark Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        make test
    
    - name: Run security checks
      run: |
        bandit -r evaluator/
        safety check
    
    - name: Check code quality
      run: |
        flake8 evaluator/ --max-line-length=100
        black evaluator/ --check

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout
      uses: actions/checkout@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - name: Deploy to Kubernetes
      run: |
        echo "${{ secrets.KUBECONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
        kubectl set image deployment/benchmark-evaluator \
          evaluator=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:main \
          -n benchmark-evaluation
        kubectl rollout status deployment/benchmark-evaluator -n benchmark-evaluation
```

### **Jenkins Pipeline**

**Jenkinsfile**:
```groovy
pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = 'your-registry.com'
        IMAGE_NAME = 'benchmark-tests'
        KUBECONFIG = credentials('kubernetes-config')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Test') {
            steps {
                sh '''
                    python -m venv venv
                    . venv/bin/activate
                    pip install -r requirements.txt
                    pip install -r requirements-dev.txt
                    make test
                '''
            }
            post {
                always {
                    publishTestResults testResultsPattern: 'test-results.xml'
                    publishCoverageResults([
                        reportDir: 'htmlcov',
                        reportFiles: 'index.html'
                    ])
                }
            }
        }
        
        stage('Security Scan') {
            steps {
                sh '''
                    . venv/bin/activate
                    bandit -r evaluator/ -f json -o bandit-report.json
                    safety check --json --output safety-report.json
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: '*-report.json'
                }
            }
        }
        
        stage('Build Image') {
            steps {
                script {
                    def image = docker.build("${DOCKER_REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER}")
                    docker.withRegistry("https://${DOCKER_REGISTRY}", 'docker-registry-credentials') {
                        image.push()
                        image.push('latest')
                    }
                }
            }
        }
        
        stage('Deploy to Staging') {
            steps {
                sh '''
                    kubectl set image deployment/benchmark-evaluator \
                      evaluator=${DOCKER_REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER} \
                      -n benchmark-staging
                    kubectl rollout status deployment/benchmark-evaluator -n benchmark-staging
                '''
            }
        }
        
        stage('Integration Tests') {
            steps {
                sh '''
                    # Run integration tests against staging
                    python tests/integration/test_deployment.py --environment staging
                '''
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                input message: 'Deploy to Production?', ok: 'Deploy'
                sh '''
                    kubectl set image deployment/benchmark-evaluator \
                      evaluator=${DOCKER_REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER} \
                      -n benchmark-production
                    kubectl rollout status deployment/benchmark-evaluator -n benchmark-production
                '''
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        failure {
            emailext(
                subject: "Pipeline Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "The pipeline failed. Check ${env.BUILD_URL} for details.",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

## 📊 **Monitoring and Observability**

### **Application Metrics**

**metrics.py** (Add to evaluator/core/):
```python
import time
from functools import wraps
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collect application metrics for monitoring."""
    
    def __init__(self):
        self.metrics = {
            'evaluations_total': 0,
            'evaluations_success': 0,
            'evaluations_failed': 0,
            'evaluation_duration_seconds': [],
            'model_api_calls': 0,
            'model_api_failures': 0,
            'model_api_duration_seconds': []
        }
    
    def record_evaluation(self, success: bool, duration: float):
        """Record evaluation metrics."""
        self.metrics['evaluations_total'] += 1
        if success:
            self.metrics['evaluations_success'] += 1
        else:
            self.metrics['evaluations_failed'] += 1
        self.metrics['evaluation_duration_seconds'].append(duration)
    
    def record_api_call(self, success: bool, duration: float):
        """Record model API call metrics."""
        self.metrics['model_api_calls'] += 1
        if not success:
            self.metrics['model_api_failures'] += 1
        self.metrics['model_api_duration_seconds'].append(duration)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        metrics = self.metrics.copy()
        
        # Calculate averages
        if metrics['evaluation_duration_seconds']:
            metrics['avg_evaluation_duration'] = sum(metrics['evaluation_duration_seconds']) / len(metrics['evaluation_duration_seconds'])
        
        if metrics['model_api_duration_seconds']:
            metrics['avg_api_duration'] = sum(metrics['model_api_duration_seconds']) / len(metrics['model_api_duration_seconds'])
        
        # Calculate rates
        total_evals = metrics['evaluations_total']
        if total_evals > 0:
            metrics['success_rate'] = metrics['evaluations_success'] / total_evals
            metrics['failure_rate'] = metrics['evaluations_failed'] / total_evals
        
        return metrics

# Global metrics collector
metrics_collector = MetricsCollector()

def monitor_evaluation(func):
    """Decorator to monitor evaluation functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        success = False
        try:
            result = func(*args, **kwargs)
            success = True
            return result
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
        finally:
            duration = time.time() - start_time
            metrics_collector.record_evaluation(success, duration)
    return wrapper
```

### **Health Checks**

**health.py**:
```python
from typing import Dict, Any
import requests
import psutil
import logging

logger = logging.getLogger(__name__)

class HealthChecker:
    """Comprehensive health checking for the evaluation service."""
    
    def __init__(self, model_endpoint: str):
        self.model_endpoint = model_endpoint
    
    def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            'status': 'healthy',
            'checks': {},
            'timestamp': time.time()
        }
        
        # Check model API connectivity
        try:
            response = requests.get(
                f"{self.model_endpoint}/health", 
                timeout=10
            )
            health_status['checks']['model_api'] = {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'response_time_ms': response.elapsed.total_seconds() * 1000,
                'status_code': response.status_code
            }
        except Exception as e:
            health_status['checks']['model_api'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['status'] = 'unhealthy'
        
        # Check system resources
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent(interval=1)
            
            health_status['checks']['system_resources'] = {
                'status': 'healthy',
                'memory_usage_percent': memory.percent,
                'disk_usage_percent': disk.percent,
                'cpu_usage_percent': cpu_percent
            }
            
            # Alert if resources are critically low
            if memory.percent > 90 or disk.percent > 90 or cpu_percent > 95:
                health_status['checks']['system_resources']['status'] = 'warning'
                if health_status['status'] == 'healthy':
                    health_status['status'] = 'degraded'
                    
        except Exception as e:
            health_status['checks']['system_resources'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['status'] = 'unhealthy'
        
        # Check evaluator modules
        try:
            from evaluator.subjects import ReasoningEvaluator
            reasoning_evaluator = ReasoningEvaluator()
            # Quick validation test
            test_result = reasoning_evaluator.evaluate("Test", {})
            
            health_status['checks']['evaluators'] = {
                'status': 'healthy',
                'modules_loaded': True
            }
        except Exception as e:
            health_status['checks']['evaluators'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['status'] = 'unhealthy'
        
        return health_status
```

### **Prometheus Metrics**

**prometheus.yml**:
```yaml
global:
  scrape_interval: 15s
  
scrape_configs:
  - job_name: 'benchmark-evaluator'
    static_configs:
      - targets: ['benchmark-evaluator:8080']
    scrape_interval: 10s
    metrics_path: /metrics
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    scrape_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### **Grafana Dashboard Configuration**
```json
{
  "dashboard": {
    "title": "AI Model Evaluation Metrics",
    "panels": [
      {
        "title": "Evaluation Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(evaluations_total[5m])",
            "legendFormat": "Evaluations/sec"
          }
        ]
      },
      {
        "title": "Success Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "evaluations_success / evaluations_total * 100",
            "legendFormat": "Success %"
          }
        ]
      },
      {
        "title": "Average Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "avg(evaluation_duration_seconds)",
            "legendFormat": "Avg Duration"
          }
        ]
      },
      {
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "cpu_usage_percent",
            "legendFormat": "CPU %"
          },
          {
            "expr": "memory_usage_percent", 
            "legendFormat": "Memory %"
          }
        ]
      }
    ]
  }
}
```

## 🔒 **Security Considerations**

### **API Security**
```yaml
# API Gateway Configuration (nginx.conf)
server {
    listen 443 ssl http2;
    server_name evaluation-api.your-domain.com;
    
    ssl_certificate /etc/ssl/certs/your-domain.crt;
    ssl_certificate_key /etc/ssl/private/your-domain.key;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    location / {
        proxy_pass http://benchmark-evaluator-service:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Authentication
        auth_request /auth;
    }
    
    location = /auth {
        internal;
        proxy_pass http://auth-service/verify;
        proxy_pass_request_body off;
        proxy_set_header Content-Length "";
        proxy_set_header X-Original-URI $request_uri;
    }
}
```

### **Container Security**
```dockerfile
# Security-hardened Dockerfile
FROM python:3.9-slim

# Security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r evaluator && useradd -r -g evaluator -s /bin/false evaluator

# Set working directory and permissions
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
RUN chown -R evaluator:evaluator /app

# Switch to non-root user
USER evaluator

# Remove unnecessary packages
RUN pip uninstall -y pip setuptools

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "from health import HealthChecker; print('OK')" || exit 1

EXPOSE 8080
CMD ["python", "app.py"]
```

## 📋 **Production Checklist**

### **Pre-Deployment**
- [ ] All tests pass (unit, integration, security)
- [ ] Performance benchmarks meet requirements
- [ ] Security scan passes (no high/critical vulnerabilities)
- [ ] Configuration validated for production environment
- [ ] Monitoring and alerting configured
- [ ] Backup and recovery procedures tested
- [ ] Load testing completed
- [ ] Documentation updated

### **Deployment**
- [ ] Blue-green deployment strategy implemented
- [ ] Database migrations (if applicable) completed
- [ ] Configuration secrets properly managed
- [ ] Health checks passing
- [ ] Monitoring dashboards showing green status
- [ ] Load balancer configured correctly

### **Post-Deployment**
- [ ] Application metrics flowing to monitoring system
- [ ] Alerts configured and tested
- [ ] Log aggregation working correctly
- [ ] Performance monitoring baseline established
- [ ] Backup verification completed
- [ ] Runbook updated with deployment-specific details

---

This deployment guide provides enterprise-ready patterns for running the AI Model Evaluation Framework at scale. Adapt the configurations to match your specific infrastructure requirements and security policies.