# Security

Security hardening, compliance, and threat mitigation for the AI Model Evaluation Framework.

## Security Architecture

### Defense in Depth Strategy

**Security Layers**:
1. Network security (firewalls, VPNs, network segmentation)
2. Application security (authentication, authorization, input validation)
3. Data security (encryption, access controls, data classification)
4. Infrastructure security (hardened OS, secure configurations)
5. Monitoring and incident response (SIEM, alerting, forensics)

**Security Principles**:
- Principle of least privilege
- Zero trust architecture
- Defense in depth
- Security by design
- Continuous monitoring

### Threat Model

**Identified Threats**:
- Unauthorized access to evaluation data
- Model inference attacks
- Data poisoning attempts
- Denial of service attacks
- Insider threats
- Supply chain attacks

**Risk Assessment**:
```python
# security/risk_assessment.py
from dataclasses import dataclass
from enum import Enum

class ThreatLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class SecurityThreat:
    name: str
    description: str
    likelihood: ThreatLevel
    impact: ThreatLevel
    mitigation_status: str
    mitigation_measures: list

THREAT_MATRIX = [
    SecurityThreat(
        name="Data Exfiltration",
        description="Unauthorized access to evaluation datasets or model outputs",
        likelihood=ThreatLevel.MEDIUM,
        impact=ThreatLevel.HIGH,
        mitigation_status="implemented",
        mitigation_measures=[
            "Data encryption at rest and in transit",
            "Access control and audit logging",
            "Network segmentation",
            "Regular access reviews"
        ]
    ),
    SecurityThreat(
        name="Model Inference Attacks",
        description="Attempts to extract model parameters or training data",
        likelihood=ThreatLevel.HIGH,
        impact=ThreatLevel.MEDIUM,
        mitigation_status="partial",
        mitigation_measures=[
            "Rate limiting on inference endpoints",
            "Query pattern analysis",
            "Response filtering",
            "Differential privacy techniques"
        ]
    ),
    SecurityThreat(
        name="Denial of Service",
        description="Resource exhaustion attacks on evaluation services",
        likelihood=ThreatLevel.HIGH,
        impact=ThreatLevel.MEDIUM,
        mitigation_status="implemented",
        mitigation_measures=[
            "Request rate limiting",
            "Resource quotas and limits",
            "Load balancing and auto-scaling",
            "DDoS protection services"
        ]
    )
]
```

## Authentication and Authorization

### Multi-Factor Authentication

**API Key Management**:
```python
# security/api_auth.py
import secrets
import hashlib
import time
from datetime import datetime, timedelta
from functools import wraps
import jwt

class APIKeyManager:
    def __init__(self, secret_key):
        self.secret_key = secret_key
        self.active_keys = {}
        self.revoked_keys = set()
    
    def generate_api_key(self, user_id, permissions, expiry_days=90):
        """Generate secure API key with metadata"""
        # Generate random key
        raw_key = secrets.token_urlsafe(32)
        
        # Create key metadata
        key_data = {
            'user_id': user_id,
            'permissions': permissions,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(days=expiry_days)).isoformat(),
            'key_hash': hashlib.sha256(raw_key.encode()).hexdigest()
        }
        
        # Store key metadata
        self.active_keys[raw_key] = key_data
        
        return raw_key, key_data
    
    def validate_api_key(self, api_key):
        """Validate API key and check permissions"""
        if not api_key or api_key in self.revoked_keys:
            return None
        
        key_data = self.active_keys.get(api_key)
        if not key_data:
            return None
        
        # Check expiration
        expires_at = datetime.fromisoformat(key_data['expires_at'])
        if datetime.now() > expires_at:
            self.revoke_key(api_key)
            return None
        
        return key_data
    
    def revoke_key(self, api_key):
        """Revoke API key"""
        if api_key in self.active_keys:
            del self.active_keys[api_key]
        self.revoked_keys.add(api_key)

def require_auth(required_permissions=None):
    """Decorator for API endpoint authentication"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from flask import request, jsonify
            
            # Extract API key from header
            api_key = request.headers.get('X-API-Key')
            if not api_key:
                return jsonify({'error': 'API key required'}), 401
            
            # Validate key
            key_manager = APIKeyManager(os.getenv('API_SECRET_KEY'))
            key_data = key_manager.validate_api_key(api_key)
            
            if not key_data:
                return jsonify({'error': 'Invalid API key'}), 401
            
            # Check permissions
            if required_permissions:
                user_permissions = key_data.get('permissions', [])
                if not any(perm in user_permissions for perm in required_permissions):
                    return jsonify({'error': 'Insufficient permissions'}), 403
            
            # Add user context to request
            request.user_id = key_data['user_id']
            request.permissions = key_data['permissions']
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

### JWT Token Authentication

**Token Management**:
```python
# security/jwt_auth.py
import jwt
import time
from datetime import datetime, timedelta

class JWTManager:
    def __init__(self, secret_key, algorithm='HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_blacklist = set()
    
    def generate_token(self, user_id, permissions, expires_in_hours=24):
        """Generate JWT token"""
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=expires_in_hours),
            'jti': secrets.token_hex(16)  # JWT ID for blacklisting
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def validate_token(self, token):
        """Validate and decode JWT token"""
        try:
            # Check if token is blacklisted
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            jti = payload.get('jti')
            
            if jti in self.token_blacklist:
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def revoke_token(self, token):
        """Add token to blacklist"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            jti = payload.get('jti')
            if jti:
                self.token_blacklist.add(jti)
        except jwt.InvalidTokenError:
            pass
```

### Role-Based Access Control

**Permission System**:
```python
# security/rbac.py
from enum import Enum
from dataclasses import dataclass
from typing import List, Set

class Permission(Enum):
    # Evaluation permissions
    EVALUATE_BASIC = "evaluate:basic"
    EVALUATE_ADVANCED = "evaluate:advanced"
    EVALUATE_ADMIN = "evaluate:admin"
    
    # Data permissions
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    
    # System permissions
    SYSTEM_MONITOR = "system:monitor"
    SYSTEM_ADMIN = "system:admin"
    
    # User management
    USER_READ = "user:read"
    USER_WRITE = "user:write"
    USER_ADMIN = "user:admin"

@dataclass
class Role:
    name: str
    permissions: Set[Permission]
    description: str

# Predefined roles
ROLES = {
    'user': Role(
        name='user',
        permissions={Permission.EVALUATE_BASIC, Permission.DATA_READ},
        description='Basic user with evaluation access'
    ),
    'researcher': Role(
        name='researcher',
        permissions={
            Permission.EVALUATE_BASIC,
            Permission.EVALUATE_ADVANCED,
            Permission.DATA_READ,
            Permission.DATA_WRITE,
            Permission.SYSTEM_MONITOR
        },
        description='Researcher with advanced evaluation capabilities'
    ),
    'admin': Role(
        name='admin',
        permissions=set(Permission),  # All permissions
        description='System administrator with full access'
    )
}

class RBACManager:
    def __init__(self):
        self.user_roles = {}
    
    def assign_role(self, user_id: str, role_name: str):
        """Assign role to user"""
        if role_name not in ROLES:
            raise ValueError(f"Unknown role: {role_name}")
        
        self.user_roles[user_id] = role_name
    
    def check_permission(self, user_id: str, required_permission: Permission) -> bool:
        """Check if user has required permission"""
        role_name = self.user_roles.get(user_id)
        if not role_name:
            return False
        
        role = ROLES.get(role_name)
        if not role:
            return False
        
        return required_permission in role.permissions
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for user"""
        role_name = self.user_roles.get(user_id)
        if not role_name:
            return set()
        
        role = ROLES.get(role_name)
        return role.permissions if role else set()
```

## Data Security

### Encryption

**Data Encryption at Rest**:
```python
# security/encryption.py
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class DataEncryption:
    def __init__(self, password: str):
        self.password = password.encode()
        self.salt = os.urandom(16)
        self.key = self._derive_key()
        self.cipher = Fernet(self.key)
    
    def _derive_key(self):
        """Derive encryption key from password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password))
        return key
    
    def encrypt_file(self, file_path: str, encrypted_path: str):
        """Encrypt file on disk"""
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        encrypted_data = self.cipher.encrypt(file_data)
        
        with open(encrypted_path, 'wb') as f:
            f.write(self.salt + encrypted_data)
    
    def decrypt_file(self, encrypted_path: str, decrypted_path: str):
        """Decrypt file from disk"""
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()
        
        # Extract salt and encrypted data
        salt = encrypted_data[:16]
        encrypted_content = encrypted_data[16:]
        
        # Re-derive key with stored salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password))
        cipher = Fernet(key)
        
        decrypted_data = cipher.decrypt(encrypted_content)
        
        with open(decrypted_path, 'wb') as f:
            f.write(decrypted_data)
    
    def encrypt_string(self, plaintext: str) -> str:
        """Encrypt string data"""
        encrypted_bytes = self.cipher.encrypt(plaintext.encode())
        return base64.urlsafe_b64encode(encrypted_bytes).decode()
    
    def decrypt_string(self, encrypted_text: str) -> str:
        """Decrypt string data"""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_text.encode())
        decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
        return decrypted_bytes.decode()
```

**TLS/SSL Configuration**:
```nginx
# security/ssl_config.conf
server {
    listen 443 ssl http2;
    server_name benchmark.example.com;

    # SSL Certificate Configuration
    ssl_certificate /etc/ssl/certs/benchmark.crt;
    ssl_certificate_key /etc/ssl/private/benchmark.key;

    # SSL Security Settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-SHA256:ECDHE-RSA-AES256-SHA384;
    ssl_prefer_server_ciphers off;
    
    # SSL Performance Optimizations
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_session_tickets off;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    
    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/ssl/certs/ca-certificates.crt;
    
    location / {
        proxy_pass http://benchmark-backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Secure Data Handling

**Sensitive Data Protection**:
```python
# security/data_protection.py
import re
import hashlib
from typing import Dict, Any

class DataSanitizer:
    def __init__(self):
        # Patterns for detecting sensitive information
        self.sensitive_patterns = {
            'api_key': r'(?i)(api[_-]?key|apikey)["\s]*[:=]["\s]*([a-zA-Z0-9\-_]{20,})',
            'password': r'(?i)(password|passwd|pwd)["\s]*[:=]["\s]*([^\s"\']{8,})',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?\d{1,3}[-.\s]?)?(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        }
    
    def sanitize_text(self, text: str) -> str:
        """Remove or mask sensitive information from text"""
        sanitized = text
        
        for pattern_name, pattern in self.sensitive_patterns.items():
            sanitized = re.sub(pattern, f'[REDACTED_{pattern_name.upper()}]', sanitized)
        
        return sanitized
    
    def sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize dictionary data"""
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = self.sanitize_text(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [self.sanitize_text(item) if isinstance(item, str) else item 
                                for item in value]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def hash_pii(self, text: str, salt: str = "") -> str:
        """Hash personally identifiable information"""
        combined = text + salt
        return hashlib.sha256(combined.encode()).hexdigest()
```

## Network Security

### Firewall Configuration

**iptables Rules**:
```bash
#!/bin/bash
# security/firewall_setup.sh

# Flush existing rules
iptables -F
iptables -X
iptables -Z

# Default policies
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback traffic
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow established and related connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Allow SSH (limit connections)
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --set
iptables -A INPUT -p tcp --dport 22 -m conntrack --ctstate NEW -m recent --update --seconds 60 --hitcount 4 -j DROP
iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Allow HTTP and HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Allow benchmark API (rate limited)
iptables -A INPUT -p tcp --dport 8080 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT

# Allow Redis (only from localhost)
iptables -A INPUT -s 127.0.0.1 -p tcp --dport 6379 -j ACCEPT

# Block common attack patterns
iptables -A INPUT -p tcp --tcp-flags ALL ALL -j DROP
iptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP
iptables -A INPUT -p tcp --tcp-flags SYN,FIN SYN,FIN -j DROP
iptables -A INPUT -p tcp --tcp-flags SYN,RST SYN,RST -j DROP

# Log dropped packets
iptables -A INPUT -j LOG --log-prefix "IPTables-Dropped: "

# Save rules
iptables-save > /etc/iptables/rules.v4
```

### DDoS Protection

**Rate Limiting**:
```python
# security/rate_limiting.py
import time
import redis
from functools import wraps
from flask import request, jsonify

class RateLimiter:
    def __init__(self, redis_client):
        self.redis_client = redis_client
    
    def is_rate_limited(self, key: str, limit: int, window: int) -> bool:
        """Check if key exceeds rate limit"""
        current_time = int(time.time())
        window_start = current_time - window
        
        # Clean old entries
        self.redis_client.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        current_requests = self.redis_client.zcard(key)
        
        if current_requests >= limit:
            return True
        
        # Add current request
        self.redis_client.zadd(key, {str(current_time): current_time})
        self.redis_client.expire(key, window)
        
        return False

def rate_limit(max_requests: int = 100, window: int = 3600):
    """Decorator for API rate limiting"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get client IP
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR') or request.remote_addr
            key = f"rate_limit:{client_ip}:{func.__name__}"
            
            # Check rate limit
            rate_limiter = RateLimiter(redis.Redis())
            if rate_limiter.is_rate_limited(key, max_requests, window):
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'retry_after': window
                }), 429
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

## Security Monitoring

### Intrusion Detection

**Log Analysis for Security Events**:
```python
# security/intrusion_detection.py
import re
import json
from datetime import datetime, timedelta
from collections import defaultdict

class IntrusionDetector:
    def __init__(self):
        self.attack_patterns = {
            'sql_injection': [
                r'(?i)(union.*select|select.*from|insert.*into|drop.*table)',
                r'(?i)(or\s+1\s*=\s*1|and\s+1\s*=\s*1)',
                r'(?i)(exec\(|eval\(|system\()'
            ],
            'xss_attack': [
                r'(?i)(<script|javascript:|vbscript:|onload=|onerror=)',
                r'(?i)(alert\(|prompt\(|confirm\()',
                r'(?i)(<iframe|<object|<embed)'
            ],
            'path_traversal': [
                r'\.\./',
                r'\.\.\\',
                r'%2e%2e%2f',
                r'/etc/passwd',
                r'/proc/self/environ'
            ],
            'brute_force': [
                r'(?i)(failed.*login|authentication.*failed)',
                r'(?i)(invalid.*password|wrong.*password)',
                r'(?i)(unauthorized.*access|access.*denied)'
            ]
        }
        
        self.suspicious_ips = set()
        self.failed_attempts = defaultdict(int)
    
    def analyze_request(self, request_data: dict) -> dict:
        """Analyze request for security threats"""
        threats_detected = []
        risk_score = 0
        
        # Extract request components
        url = request_data.get('url', '')
        headers = request_data.get('headers', {})
        body = request_data.get('body', '')
        ip_address = request_data.get('ip', '')
        
        # Check for attack patterns
        for attack_type, patterns in self.attack_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url) or re.search(pattern, body):
                    threats_detected.append({
                        'type': attack_type,
                        'pattern': pattern,
                        'location': 'url' if re.search(pattern, url) else 'body'
                    })
                    risk_score += 25
        
        # Check for suspicious headers
        suspicious_headers = [
            'x-forwarded-for', 'x-real-ip', 'x-originating-ip'
        ]
        for header in suspicious_headers:
            if header in headers and self._is_suspicious_ip(headers[header]):
                threats_detected.append({
                    'type': 'suspicious_ip',
                    'value': headers[header],
                    'location': 'headers'
                })
                risk_score += 15
        
        # Check request frequency for potential DoS
        if self._check_dos_pattern(ip_address):
            threats_detected.append({
                'type': 'potential_dos',
                'ip': ip_address,
                'location': 'source'
            })
            risk_score += 30
        
        return {
            'threats_detected': threats_detected,
            'risk_score': min(risk_score, 100),
            'action_recommended': 'block' if risk_score > 70 else 'monitor' if risk_score > 30 else 'allow'
        }
    
    def _is_suspicious_ip(self, ip: str) -> bool:
        """Check if IP is in suspicious list or patterns"""
        # Check against known malicious IP patterns
        malicious_patterns = [
            r'^10\.',  # Internal networks attempting external access
            r'^172\.16\.',  # Internal networks
            r'^192\.168\.'  # Internal networks
        ]
        
        for pattern in malicious_patterns:
            if re.match(pattern, ip):
                return True
        
        return ip in self.suspicious_ips
    
    def _check_dos_pattern(self, ip: str) -> bool:
        """Check for DoS attack patterns"""
        current_time = datetime.now()
        window = timedelta(minutes=5)
        
        # Simple frequency check (in production, use Redis for distributed tracking)
        key = f"requests_{ip}_{current_time.strftime('%Y%m%d_%H%M')}"
        
        # Increment request count
        self.failed_attempts[key] += 1
        
        # Check if requests exceed threshold
        return self.failed_attempts[key] > 100  # 100 requests per minute
```

### Security Alerting

**Alert Management System**:
```python
# security/alerting.py
import smtplib
import json
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from datetime import datetime
import requests

class SecurityAlertManager:
    def __init__(self, config):
        self.config = config
        self.alert_channels = {
            'email': self._send_email_alert,
            'slack': self._send_slack_alert,
            'webhook': self._send_webhook_alert
        }
    
    def send_alert(self, alert_type: str, severity: str, details: dict):
        """Send security alert through configured channels"""
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'severity': severity,
            'details': details,
            'source': 'benchmark-security-system'
        }
        
        # Determine which channels to use based on severity
        channels = self._get_channels_for_severity(severity)
        
        for channel in channels:
            try:
                self.alert_channels[channel](alert_data)
            except Exception as e:
                print(f"Failed to send alert via {channel}: {e}")
    
    def _get_channels_for_severity(self, severity: str) -> list:
        """Get alert channels based on severity level"""
        channel_config = {
            'low': ['email'],
            'medium': ['email', 'slack'],
            'high': ['email', 'slack', 'webhook'],
            'critical': ['email', 'slack', 'webhook']
        }
        return channel_config.get(severity, ['email'])
    
    def _send_email_alert(self, alert_data: dict):
        """Send email alert"""
        msg = MimeMultipart()
        msg['From'] = self.config['email']['from']
        msg['To'] = ', '.join(self.config['email']['to'])
        msg['Subject'] = f"[SECURITY ALERT] {alert_data['type']} - {alert_data['severity']}"
        
        body = f"""
Security Alert Report
{'='*50}

Alert Type: {alert_data['type']}
Severity: {alert_data['severity']}
Timestamp: {alert_data['timestamp']}
Source: {alert_data['source']}

Details:
{json.dumps(alert_data['details'], indent=2)}

Please investigate immediately.
"""
        
        msg.attach(MimeText(body, 'plain'))
        
        with smtplib.SMTP(self.config['email']['smtp_host'], self.config['email']['smtp_port']) as server:
            server.starttls()
            server.login(self.config['email']['username'], self.config['email']['password'])
            server.send_message(msg)
    
    def _send_slack_alert(self, alert_data: dict):
        """Send Slack alert"""
        webhook_url = self.config['slack']['webhook_url']
        
        color = {
            'low': '#36a64f',
            'medium': '#ff9500',
            'high': '#ff0000',
            'critical': '#8b0000'
        }.get(alert_data['severity'], '#36a64f')
        
        payload = {
            'attachments': [
                {
                    'color': color,
                    'title': f"Security Alert: {alert_data['type']}",
                    'fields': [
                        {
                            'title': 'Severity',
                            'value': alert_data['severity'].upper(),
                            'short': True
                        },
                        {
                            'title': 'Timestamp',
                            'value': alert_data['timestamp'],
                            'short': True
                        },
                        {
                            'title': 'Details',
                            'value': f"```{json.dumps(alert_data['details'], indent=2)}```",
                            'short': False
                        }
                    ]
                }
            ]
        }
        
        requests.post(webhook_url, json=payload)
    
    def _send_webhook_alert(self, alert_data: dict):
        """Send webhook alert to external security system"""
        webhook_url = self.config['webhook']['url']
        headers = self.config['webhook'].get('headers', {})
        
        requests.post(webhook_url, json=alert_data, headers=headers)
```

## Compliance and Auditing

### Audit Logging

**Comprehensive Audit System**:
```python
# security/audit_logging.py
import json
import logging
from datetime import datetime
from functools import wraps
from flask import request, g

class AuditLogger:
    def __init__(self, log_file='/var/log/benchmark/audit.log'):
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_access(self, user_id: str, resource: str, action: str, 
                   result: str, additional_data: dict = None):
        """Log access attempt"""
        audit_entry = {
            'event_type': 'access',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'resource': resource,
            'action': action,
            'result': result,
            'ip_address': getattr(request, 'remote_addr', 'unknown'),
            'user_agent': getattr(request, 'user_agent', {}).string if hasattr(request, 'user_agent') else 'unknown',
            'additional_data': additional_data or {}
        }
        
        self.logger.info(json.dumps(audit_entry))
    
    def log_data_access(self, user_id: str, data_type: str, operation: str, 
                        record_count: int = 1):
        """Log data access"""
        audit_entry = {
            'event_type': 'data_access',
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'data_type': data_type,
            'operation': operation,
            'record_count': record_count,
            'ip_address': getattr(request, 'remote_addr', 'unknown')
        }
        
        self.logger.info(json.dumps(audit_entry))
    
    def log_security_event(self, event_type: str, severity: str, details: dict):
        """Log security-related events"""
        audit_entry = {
            'event_type': 'security',
            'security_event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'details': details,
            'ip_address': getattr(request, 'remote_addr', 'unknown')
        }
        
        self.logger.warning(json.dumps(audit_entry))

def audit_access(resource: str, action: str):
    """Decorator for auditing resource access"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            audit_logger = AuditLogger()
            user_id = getattr(request, 'user_id', 'anonymous')
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful access
                audit_logger.log_access(
                    user_id=user_id,
                    resource=resource,
                    action=action,
                    result='success'
                )
                
                return result
                
            except Exception as e:
                # Log failed access
                audit_logger.log_access(
                    user_id=user_id,
                    resource=resource,
                    action=action,
                    result='failure',
                    additional_data={'error': str(e)}
                )
                raise
                
        return wrapper
    return decorator
```

### Compliance Reporting

**Automated Compliance Reports**:
```python
# security/compliance_reporting.py
import sqlite3
import json
from datetime import datetime, timedelta
from collections import defaultdict

class ComplianceReporter:
    def __init__(self, audit_db_path='/var/log/benchmark/audit.db'):
        self.db_path = audit_db_path
        self.init_database()
    
    def init_database(self):
        """Initialize audit database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                event_type TEXT,
                user_id TEXT,
                resource TEXT,
                action TEXT,
                result TEXT,
                ip_address TEXT,
                details TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def generate_access_report(self, days_back: int = 30) -> dict:
        """Generate access report for compliance"""
        conn = sqlite3.connect(self.db_path)
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        # Query access patterns
        cursor = conn.execute('''
            SELECT user_id, resource, action, result, COUNT(*) as count
            FROM audit_logs
            WHERE timestamp >= ? AND event_type = 'access'
            GROUP BY user_id, resource, action, result
            ORDER BY count DESC
        ''', (cutoff_date.isoformat(),))
        
        access_patterns = cursor.fetchall()
        
        # Query failed access attempts
        cursor = conn.execute('''
            SELECT user_id, resource, action, ip_address, COUNT(*) as failures
            FROM audit_logs
            WHERE timestamp >= ? AND event_type = 'access' AND result = 'failure'
            GROUP BY user_id, resource, action, ip_address
            HAVING failures >= 5
            ORDER BY failures DESC
        ''', (cutoff_date.isoformat(),))
        
        failed_attempts = cursor.fetchall()
        
        # Query data access
        cursor = conn.execute('''
            SELECT user_id, data_type, operation, SUM(record_count) as total_records
            FROM audit_logs
            WHERE timestamp >= ? AND event_type = 'data_access'
            GROUP BY user_id, data_type, operation
            ORDER BY total_records DESC
        ''', (cutoff_date.isoformat(),))
        
        data_access = cursor.fetchall()
        
        conn.close()
        
        return {
            'report_period': f'{cutoff_date.date()} to {datetime.now().date()}',
            'access_patterns': [
                {
                    'user_id': row[0],
                    'resource': row[1],
                    'action': row[2],
                    'result': row[3],
                    'count': row[4]
                }
                for row in access_patterns
            ],
            'failed_attempts': [
                {
                    'user_id': row[0],
                    'resource': row[1],
                    'action': row[2],
                    'ip_address': row[3],
                    'failures': row[4]
                }
                for row in failed_attempts
            ],
            'data_access': [
                {
                    'user_id': row[0],
                    'data_type': row[1],
                    'operation': row[2],
                    'total_records': row[3]
                }
                for row in data_access
            ]
        }
    
    def check_compliance_violations(self) -> list:
        """Check for potential compliance violations"""
        violations = []
        
        # Check for excessive failed login attempts
        failed_logins = self._check_failed_logins()
        if failed_logins:
            violations.append({
                'type': 'excessive_failed_logins',
                'severity': 'high',
                'details': failed_logins
            })
        
        # Check for data access outside business hours
        after_hours_access = self._check_after_hours_access()
        if after_hours_access:
            violations.append({
                'type': 'after_hours_data_access',
                'severity': 'medium',
                'details': after_hours_access
            })
        
        # Check for bulk data exports
        bulk_exports = self._check_bulk_exports()
        if bulk_exports:
            violations.append({
                'type': 'bulk_data_export',
                'severity': 'high',
                'details': bulk_exports
            })
        
        return violations
```

## Incident Response

### Security Incident Playbook

**Automated Incident Response**:
```python
# security/incident_response.py
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import subprocess

class IncidentSeverity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class SecurityIncident:
    incident_id: str
    incident_type: str
    severity: IncidentSeverity
    description: str
    affected_systems: list
    timestamp: datetime
    status: str = "open"

class IncidentResponseManager:
    def __init__(self):
        self.active_incidents = {}
        self.response_playbooks = {
            'data_breach': self._handle_data_breach,
            'ddos_attack': self._handle_ddos,
            'unauthorized_access': self._handle_unauthorized_access,
            'malware_detection': self._handle_malware
        }
    
    def create_incident(self, incident_type: str, severity: IncidentSeverity, 
                       description: str, affected_systems: list) -> str:
        """Create new security incident"""
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        incident = SecurityIncident(
            incident_id=incident_id,
            incident_type=incident_type,
            severity=severity,
            description=description,
            affected_systems=affected_systems,
            timestamp=datetime.now()
        )
        
        self.active_incidents[incident_id] = incident
        
        # Execute automated response
        self._execute_automated_response(incident)
        
        return incident_id
    
    def _execute_automated_response(self, incident: SecurityIncident):
        """Execute automated incident response"""
        playbook = self.response_playbooks.get(incident.incident_type)
        
        if playbook:
            try:
                playbook(incident)
            except Exception as e:
                print(f"Automated response failed for {incident.incident_id}: {e}")
        
        # Always send alert for incidents
        alert_manager = SecurityAlertManager({})
        alert_manager.send_alert(
            alert_type=incident.incident_type,
            severity=incident.severity.name.lower(),
            details={
                'incident_id': incident.incident_id,
                'description': incident.description,
                'affected_systems': incident.affected_systems
            }
        )
    
    def _handle_data_breach(self, incident: SecurityIncident):
        """Handle data breach incident"""
        # Immediate containment
        self._isolate_affected_systems(incident.affected_systems)
        
        # Stop data processing
        self._stop_evaluation_services()
        
        # Secure audit logs
        self._backup_audit_logs()
        
        # Notify stakeholders
        print(f"Data breach response initiated for {incident.incident_id}")
    
    def _handle_ddos(self, incident: SecurityIncident):
        """Handle DDoS attack"""
        # Enable aggressive rate limiting
        subprocess.run(['iptables', '-A', 'INPUT', '-p', 'tcp', '--dport', '80', 
                       '-m', 'limit', '--limit', '5/minute', '-j', 'ACCEPT'])
        
        # Block suspicious IPs
        self._block_suspicious_ips()
        
        # Scale up infrastructure if possible
        self._auto_scale_services()
        
        print(f"DDoS response initiated for {incident.incident_id}")
    
    def _handle_unauthorized_access(self, incident: SecurityIncident):
        """Handle unauthorized access attempt"""
        # Revoke potentially compromised tokens
        self._revoke_suspicious_tokens()
        
        # Enable additional authentication
        self._enable_strict_auth_mode()
        
        # Monitor for lateral movement
        self._enhance_monitoring()
        
        print(f"Unauthorized access response initiated for {incident.incident_id}")
    
    def _isolate_affected_systems(self, systems: list):
        """Isolate affected systems"""
        for system in systems:
            # Implementation depends on infrastructure
            print(f"Isolating system: {system}")
    
    def _stop_evaluation_services(self):
        """Stop evaluation services for containment"""
        subprocess.run(['systemctl', 'stop', 'benchmark-framework'])
        print("Evaluation services stopped for containment")
    
    def _backup_audit_logs(self):
        """Secure backup of audit logs"""
        subprocess.run([
            'tar', '-czf', 
            f'/secure/backup/audit_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.tar.gz',
            '/var/log/benchmark/audit.log'
        ])
        print("Audit logs backed up securely")
```

## Security Best Practices

### Development Security Guidelines

**Secure Coding Practices**:
1. Input validation and sanitization
2. Output encoding and escaping
3. Parameterized queries for database access
4. Proper error handling without information leakage
5. Secure session management
6. Regular dependency updates and security scanning

**Security Testing**:
```python
# security/security_testing.py
import requests
import json
from urllib.parse import urlencode

class SecurityTester:
    def __init__(self, base_url):
        self.base_url = base_url
        self.test_results = []
    
    def test_sql_injection(self):
        """Test for SQL injection vulnerabilities"""
        payloads = [
            "1' OR '1'='1",
            "1'; DROP TABLE users; --",
            "1' UNION SELECT NULL, version(); --"
        ]
        
        for payload in payloads:
            response = requests.post(
                f"{self.base_url}/api/evaluate",
                json={"input": payload}
            )
            
            # Check for SQL error messages
            if any(error in response.text.lower() for error in ['sql', 'database', 'mysql', 'postgres']):
                self.test_results.append({
                    'test': 'sql_injection',
                    'payload': payload,
                    'status': 'FAIL',
                    'response': response.text[:200]
                })
    
    def test_xss(self):
        """Test for XSS vulnerabilities"""
        payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>"
        ]
        
        for payload in payloads:
            response = requests.post(
                f"{self.base_url}/api/evaluate",
                json={"input": payload}
            )
            
            # Check if payload is reflected unescaped
            if payload in response.text:
                self.test_results.append({
                    'test': 'xss',
                    'payload': payload,
                    'status': 'FAIL',
                    'response': response.text[:200]
                })
    
    def run_all_tests(self):
        """Run all security tests"""
        self.test_sql_injection()
        self.test_xss()
        
        return self.test_results
```

## References

- [Monitoring](./monitoring.md)
- [Deployment](./deployment.md)
- [Maintenance](./maintenance.md)
- [Configuration](../engineering/configuration.md)