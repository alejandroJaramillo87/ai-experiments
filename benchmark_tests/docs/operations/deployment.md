# Deployment

Production deployment procedures for the AI Model Evaluation Framework.

## System Requirements

### Hardware Specifications

**Minimum Requirements**:
- CPU: 8 cores, 3.0GHz
- RAM: 32GB DDR4
- Storage: 500GB SSD
- Network: 1Gbps connection

**Recommended Requirements**:
- CPU: 16+ cores, 3.5GHz (AMD Ryzen 9950X or equivalent)
- RAM: 128GB DDR5
- GPU: 24GB VRAM (RTX 5090 or equivalent)
- Storage: 2TB NVMe SSD (Samsung 990 Pro or equivalent)
- Network: 10Gbps connection

### Operating System Support

**Supported Platforms**:
- Ubuntu 22.04 LTS or 24.04 LTS (recommended)
- CentOS 8/Rocky Linux 8
- Red Hat Enterprise Linux 8/9
- Docker containers on any Linux host

**Required Software**:
- Python 3.9+
- CUDA 12.0+ (for GPU acceleration)
- Docker 24.0+ and Docker Compose 2.20+
- Git 2.30+

## Deployment Methods

### Method 1: Bare Metal Installation

**Environment Setup**:
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python development tools
sudo apt install -y python3.11 python3.11-dev python3.11-venv
sudo apt install -y build-essential git curl

# Install CUDA (for GPU support)
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-ubuntu2404-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-3
```

**Framework Installation**:
```bash
# Clone repository
git clone https://github.com/alejandroJaramillo87/ai-workstation.git
cd ai-workstation/benchmark_tests

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Verify installation
make test-installation
```

**Configuration**:
```bash
# Copy configuration template
cp config/production.template.yaml config/production.yaml

# Edit configuration for your environment
vim config/production.yaml

# Validate configuration
python -c "from config.loader import load_config; load_config('production')"
```

### Method 2: Docker Deployment

**Docker Compose Setup**:
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  benchmark-framework:
    build:
      context: .
      dockerfile: docker/Dockerfile.prod
    ports:
      - "8080:8080"
    environment:
      - ENVIRONMENT=production
      - GPU_ENABLED=true
    volumes:
      - ./data:/app/data:ro
      - ./models:/app/models:ro
      - ./logs:/app/logs
      - ./results:/app/results
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8080/health')"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - benchmark-framework
    restart: unless-stopped

volumes:
  redis_data:
```

**Production Docker Build**:
```dockerfile
# docker/Dockerfile.prod
FROM nvidia/cuda:12.3-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -r benchmark && useradd -r -g benchmark benchmark
RUN chown -R benchmark:benchmark /app
USER benchmark

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Default command
CMD ["python", "-m", "benchmark_tests.server", "--config", "production"]
```

### Method 3: Kubernetes Deployment

**Namespace and ConfigMap**:
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: benchmark-framework

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: benchmark-config
  namespace: benchmark-framework
data:
  production.yaml: |
    # Production configuration
    server:
      host: "0.0.0.0"
      port: 8080
      workers: 4
    
    evaluation:
      batch_size: 32
      timeout: 300
      concurrent_evaluations: 10
    
    storage:
      results_path: "/data/results"
      models_path: "/data/models"
      cache_enabled: true
      cache_ttl: 3600
```

**Deployment and Service**:
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: benchmark-framework
  namespace: benchmark-framework
spec:
  replicas: 3
  selector:
    matchLabels:
      app: benchmark-framework
  template:
    metadata:
      labels:
        app: benchmark-framework
    spec:
      containers:
      - name: benchmark-framework
        image: benchmark-framework:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: data-volume
          mountPath: /app/data
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: config-volume
        configMap:
          name: benchmark-config
      - name: data-volume
        persistentVolumeClaim:
          claimName: benchmark-data-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: benchmark-framework-service
  namespace: benchmark-framework
spec:
  selector:
    app: benchmark-framework
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

## Environment Configuration

### Production Configuration

**Core Settings**:
```yaml
# config/production.yaml
environment: production

server:
  host: "0.0.0.0"
  port: 8080
  workers: 8
  max_connections: 1000
  keep_alive_timeout: 65

evaluation:
  batch_size: 64
  timeout: 600
  concurrent_evaluations: 20
  max_retries: 3
  circuit_breaker_enabled: true

storage:
  results_path: "/data/results"
  models_path: "/data/models"
  cache_enabled: true
  cache_ttl: 7200
  backup_enabled: true
  backup_interval: "daily"

logging:
  level: "INFO"
  format: "json"
  file: "/var/log/benchmark/app.log"
  rotation: "daily"
  retention: "30d"

security:
  api_key_required: true
  rate_limiting: true
  max_requests_per_minute: 100
  cors_enabled: false

monitoring:
  metrics_enabled: true
  health_check_interval: 30
  performance_tracking: true
```

### Security Hardening

**SSL/TLS Configuration**:
```nginx
# nginx/nginx.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://benchmark-framework:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**API Authentication**:
```python
# config/auth.py
import os
from functools import wraps
from flask import request, jsonify

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != os.getenv('API_KEY'):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function
```

## Database Setup

### Redis Configuration

**Production Redis Settings**:
```conf
# redis/redis.conf
port 6379
bind 127.0.0.1
protected-mode yes
requirepass your_secure_password

# Memory management
maxmemory 8gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000

# Logging
loglevel notice
logfile /var/log/redis/redis.log
```

### Data Migration

**Initial Data Setup**:
```bash
# Import evaluation domains
python scripts/import_domains.py --source data/domains/ --environment production

# Load cultural context data
python scripts/load_cultural_data.py --source data/cultural/ --validate

# Initialize evaluation models
python scripts/initialize_evaluators.py --config production

# Verify data integrity
make verify-data-integrity
```

## Deployment Verification

### Health Checks

**System Health Validation**:
```bash
#!/bin/bash
# scripts/health_check.sh

# Check system resources
echo "=== System Resources ==="
free -h
df -h
nvidia-smi

# Check services
echo "=== Service Status ==="
systemctl status benchmark-framework
systemctl status redis
systemctl status nginx

# Check API endpoints
echo "=== API Health Check ==="
curl -f http://localhost:8080/health || exit 1
curl -f http://localhost:8080/ready || exit 1

# Check evaluation functionality
echo "=== Evaluation Test ==="
python -c "
from benchmark_tests import evaluate_sample
result = evaluate_sample('Test response', 'reasoning', 'easy')
assert result['status'] == 'success'
print('Evaluation system operational')
"

echo "All health checks passed"
```

### Performance Validation

**Load Testing**:
```bash
# Run load test with Apache Bench
ab -n 1000 -c 10 -H "X-API-Key: ${API_KEY}" \
   http://localhost:8080/api/v1/evaluate

# Monitor system during load
while true; do
    echo "$(date): $(free -h | grep Mem:)"
    echo "$(date): $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"
    sleep 5
done
```

## Deployment Checklist

### Pre-Deployment

- [ ] System requirements verified
- [ ] Dependencies installed and versions confirmed
- [ ] Configuration files reviewed and customized
- [ ] SSL certificates installed and configured
- [ ] Database connections tested
- [ ] Network connectivity verified
- [ ] Firewall rules configured
- [ ] Backup procedures tested

### Post-Deployment

- [ ] Health checks passing
- [ ] API endpoints responding correctly
- [ ] Evaluation functionality verified
- [ ] Performance within acceptable limits
- [ ] Monitoring systems operational
- [ ] Log files being generated correctly
- [ ] SSL certificate expiration monitoring configured
- [ ] Documentation updated with deployment specifics

## Rollback Procedures

### Emergency Rollback

**Docker Deployment Rollback**:
```bash
# Stop current deployment
docker-compose -f docker-compose.prod.yml down

# Revert to previous version
git checkout previous-stable-tag
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Verify rollback
./scripts/health_check.sh
```

**Kubernetes Rollback**:
```bash
# Check deployment history
kubectl rollout history deployment/benchmark-framework -n benchmark-framework

# Rollback to previous version
kubectl rollout undo deployment/benchmark-framework -n benchmark-framework

# Monitor rollback status
kubectl rollout status deployment/benchmark-framework -n benchmark-framework
```

## References

- [Configuration](../engineering/configuration.md)
- [Monitoring](./monitoring.md)
- [Performance](./performance.md)
- [Security](./security.md)