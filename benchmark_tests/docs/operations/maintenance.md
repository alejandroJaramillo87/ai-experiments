# Maintenance

Regular maintenance procedures and system administration tasks for the AI Model Evaluation Framework.

## Routine Maintenance Schedule

### Daily Tasks

**System Health Checks**:
```bash
#!/bin/bash
# scripts/daily_maintenance.sh

echo "=== Daily Maintenance Report - $(date) ==="

# Check system resources
echo "--- System Resources ---"
free -h
df -h
uptime

# Check GPU status
echo "--- GPU Status ---"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv

# Check service status
echo "--- Services Status ---"
systemctl status benchmark-framework --no-pager
systemctl status redis --no-pager
systemctl status nginx --no-pager

# Check recent errors in logs
echo "--- Recent Errors ---"
journalctl -u benchmark-framework --since "24 hours ago" --grep ERROR | tail -20

# Check disk usage for critical directories
echo "--- Critical Directory Usage ---"
du -sh /var/log/benchmark/
du -sh /data/results/
du -sh /data/models/
du -sh /tmp/

# Network connectivity test
echo "--- Network Connectivity ---"
ping -c 3 8.8.8.8 > /dev/null && echo "Internet: OK" || echo "Internet: FAILED"

echo "=== Daily Maintenance Complete ==="
```

**Log Rotation and Cleanup**:
```bash
#!/bin/bash
# scripts/log_cleanup.sh

# Rotate application logs
LOG_DIR="/var/log/benchmark"
RETENTION_DAYS=30

find $LOG_DIR -name "*.log" -type f -mtime +$RETENTION_DAYS -delete
find $LOG_DIR -name "*.log.*" -type f -mtime +$RETENTION_DAYS -delete

# Compress old logs
find $LOG_DIR -name "*.log" -type f -mtime +7 -exec gzip {} \;

# Clean temporary files
find /tmp/benchmark_* -type f -mtime +1 -delete

# Clean evaluation cache if too large
CACHE_DIR="/tmp/evaluation_cache"
if [ -d "$CACHE_DIR" ]; then
    CACHE_SIZE=$(du -s $CACHE_DIR | cut -f1)
    if [ $CACHE_SIZE -gt 10485760 ]; then  # 10GB in KB
        echo "Cache size ${CACHE_SIZE}KB exceeds limit, cleaning old entries"
        find $CACHE_DIR -type f -mtime +3 -delete
    fi
fi
```

### Weekly Tasks

**Database Maintenance**:
```bash
#!/bin/bash
# scripts/weekly_maintenance.sh

echo "=== Weekly Maintenance - $(date) ==="

# Redis maintenance
echo "--- Redis Maintenance ---"
redis-cli INFO memory
redis-cli BGSAVE
redis-cli MEMORY PURGE

# Check Redis key expiration
EXPIRED_KEYS=$(redis-cli --scan --pattern "*" | wc -l)
echo "Total Redis keys: $EXPIRED_KEYS"

# Analyze Redis memory usage
redis-cli --bigkeys

# System updates (if allowed)
echo "--- System Updates Check ---"
apt list --upgradable

# GPU driver check
echo "--- GPU Driver Status ---"
nvidia-smi -q -d PERFORMANCE

echo "=== Weekly Maintenance Complete ==="
```

**Performance Analysis**:
```python
# scripts/weekly_performance_analysis.py
import json
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class WeeklyPerformanceAnalyzer:
    def __init__(self, db_path="/var/log/benchmark/metrics.db"):
        self.db_path = db_path
        self.week_start = datetime.now() - timedelta(days=7)
    
    def analyze_weekly_performance(self):
        """Analyze performance metrics from the last week"""
        conn = sqlite3.connect(self.db_path)
        
        # Query performance metrics
        query = """
        SELECT timestamp, domain, avg_latency, throughput, error_rate
        FROM performance_metrics 
        WHERE timestamp >= ?
        ORDER BY timestamp
        """
        
        metrics = conn.execute(query, (self.week_start,)).fetchall()
        conn.close()
        
        # Analyze trends
        analysis = {
            'avg_latency_trend': self._calculate_trend([m[2] for m in metrics]),
            'throughput_trend': self._calculate_trend([m[3] for m in metrics]),
            'error_rate_trend': self._calculate_trend([m[4] for m in metrics]),
            'domain_performance': self._analyze_by_domain(metrics)
        }
        
        return analysis
    
    def generate_weekly_report(self):
        """Generate weekly performance report"""
        analysis = self.analyze_weekly_performance()
        
        report = f"""
Weekly Performance Report - {datetime.now().strftime('%Y-%m-%d')}
{'=' * 50}

Performance Trends:
- Latency: {'↑' if analysis['avg_latency_trend'] > 0 else '↓'} {abs(analysis['avg_latency_trend']):.2f}%
- Throughput: {'↑' if analysis['throughput_trend'] > 0 else '↓'} {abs(analysis['throughput_trend']):.2f}%
- Error Rate: {'↑' if analysis['error_rate_trend'] > 0 else '↓'} {abs(analysis['error_rate_trend']):.2f}%

Domain Performance:
"""
        for domain, perf in analysis['domain_performance'].items():
            report += f"- {domain}: {perf['avg_score']:.3f} (±{perf['std_dev']:.3f})\n"
        
        return report
```

### Monthly Tasks

**System Optimization Review**:
```bash
#!/bin/bash
# scripts/monthly_maintenance.sh

echo "=== Monthly Maintenance - $(date) ==="

# Full system backup verification
echo "--- Backup Verification ---"
python scripts/verify_backups.py

# Database optimization
echo "--- Database Optimization ---"
redis-cli --eval scripts/redis_optimize.lua

# Model cache cleanup and optimization
echo "--- Model Cache Optimization ---"
python scripts/optimize_model_cache.py

# Security audit
echo "--- Security Audit ---"
python scripts/security_audit.py

# Performance baseline update
echo "--- Performance Baseline Update ---"
python scripts/update_performance_baseline.py

# System package updates (scheduled maintenance window)
echo "--- System Updates ---"
if [ "$MAINTENANCE_WINDOW" = "true" ]; then
    apt update && apt upgrade -y
    nvidia-driver-update
fi

echo "=== Monthly Maintenance Complete ==="
```

## Database Maintenance

### Redis Maintenance

**Memory Optimization**:
```python
# maintenance/redis_maintenance.py
import redis
import json
from datetime import datetime, timedelta

class RedisMaintenanceManager:
    def __init__(self, host='localhost', port=6379):
        self.redis_client = redis.Redis(host=host, port=port, decode_responses=True)
    
    def analyze_memory_usage(self):
        """Analyze Redis memory usage patterns"""
        info = self.redis_client.info('memory')
        
        analysis = {
            'used_memory_human': info['used_memory_human'],
            'used_memory_peak_human': info['used_memory_peak_human'],
            'memory_fragmentation_ratio': info.get('mem_fragmentation_ratio', 0),
            'expired_keys': info.get('expired_keys', 0),
            'evicted_keys': info.get('evicted_keys', 0)
        }
        
        return analysis
    
    def cleanup_expired_keys(self):
        """Clean up expired keys and optimize memory"""
        # Get key pattern analysis
        patterns = {}
        for key in self.redis_client.scan_iter():
            pattern = key.split(':')[0] if ':' in key else 'misc'
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        # Clean up old evaluation results
        cutoff_date = datetime.now() - timedelta(days=7)
        cutoff_timestamp = cutoff_date.timestamp()
        
        deleted_count = 0
        for key in self.redis_client.scan_iter(match="evaluation:*"):
            try:
                # Check if key has timestamp in name or get its creation time
                parts = key.split(':')
                if len(parts) >= 3 and parts[2].isdigit():
                    key_timestamp = float(parts[2])
                    if key_timestamp < cutoff_timestamp:
                        self.redis_client.delete(key)
                        deleted_count += 1
            except (ValueError, IndexError):
                continue
        
        return {
            'patterns': patterns,
            'deleted_keys': deleted_count,
            'memory_after_cleanup': self.redis_client.info('memory')['used_memory_human']
        }
    
    def optimize_database(self):
        """Optimize Redis database"""
        # Force background save
        self.redis_client.bgsave()
        
        # Memory purge
        try:
            self.redis_client.execute_command('MEMORY PURGE')
        except redis.ResponseError:
            pass  # Command not available in all Redis versions
        
        # Defragmentation (Redis 4.0+)
        try:
            self.redis_client.execute_command('MEMORY DOCTOR')
        except redis.ResponseError:
            pass
```

**Backup and Recovery**:
```python
# maintenance/redis_backup.py
import os
import subprocess
import datetime
import boto3
from pathlib import Path

class RedisBackupManager:
    def __init__(self, backup_dir="/var/backups/redis"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # AWS S3 client for remote backups
        self.s3_client = boto3.client('s3') if os.getenv('AWS_ACCESS_KEY_ID') else None
    
    def create_backup(self):
        """Create Redis backup"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = self.backup_dir / f"redis_backup_{timestamp}.rdb"
        
        # Create Redis backup
        result = subprocess.run([
            'redis-cli', 'BGSAVE'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"Redis backup failed: {result.stderr}")
        
        # Wait for background save to complete
        subprocess.run(['redis-cli', 'LASTSAVE'], check=True)
        
        # Copy RDB file to backup location
        subprocess.run([
            'cp', '/var/lib/redis/dump.rdb', str(backup_file)
        ], check=True)
        
        # Compress backup
        subprocess.run(['gzip', str(backup_file)], check=True)
        backup_file = backup_file.with_suffix('.rdb.gz')
        
        # Upload to S3 if configured
        if self.s3_client:
            self._upload_to_s3(backup_file)
        
        return backup_file
    
    def restore_backup(self, backup_file):
        """Restore Redis from backup"""
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")
        
        # Stop Redis service
        subprocess.run(['systemctl', 'stop', 'redis'], check=True)
        
        try:
            # Decompress if needed
            if backup_file.suffix == '.gz':
                subprocess.run(['gunzip', str(backup_file)], check=True)
                backup_file = backup_file.with_suffix('')
            
            # Copy backup to Redis data directory
            subprocess.run([
                'cp', str(backup_file), '/var/lib/redis/dump.rdb'
            ], check=True)
            
            # Set proper permissions
            subprocess.run([
                'chown', 'redis:redis', '/var/lib/redis/dump.rdb'
            ], check=True)
            
            # Start Redis service
            subprocess.run(['systemctl', 'start', 'redis'], check=True)
            
        except Exception as e:
            # Restart Redis even if restore failed
            subprocess.run(['systemctl', 'start', 'redis'])
            raise e
    
    def cleanup_old_backups(self, keep_days=30):
        """Clean up backups older than specified days"""
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=keep_days)
        
        deleted_count = 0
        for backup_file in self.backup_dir.glob("redis_backup_*.rdb*"):
            if backup_file.stat().st_mtime < cutoff_date.timestamp():
                backup_file.unlink()
                deleted_count += 1
        
        return deleted_count
```

## Model Cache Management

### Cache Optimization

```python
# maintenance/cache_management.py
import os
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

class CacheManager:
    def __init__(self, cache_dir="/tmp/model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache policies
        self.max_cache_size = 50 * 1024 * 1024 * 1024  # 50GB
        self.default_ttl = timedelta(days=7)
    
    def analyze_cache_usage(self):
        """Analyze current cache usage"""
        total_size = 0
        file_count = 0
        oldest_file = None
        newest_file = None
        
        for cache_file in self.cache_dir.rglob("*"):
            if cache_file.is_file():
                file_size = cache_file.stat().st_size
                file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                
                total_size += file_size
                file_count += 1
                
                if oldest_file is None or file_mtime < oldest_file[1]:
                    oldest_file = (cache_file, file_mtime)
                
                if newest_file is None or file_mtime > newest_file[1]:
                    newest_file = (cache_file, file_mtime)
        
        return {
            'total_size_gb': total_size / (1024**3),
            'file_count': file_count,
            'oldest_file': oldest_file[0].name if oldest_file else None,
            'oldest_date': oldest_file[1].isoformat() if oldest_file else None,
            'newest_file': newest_file[0].name if newest_file else None,
            'newest_date': newest_file[1].isoformat() if newest_file else None,
            'utilization_percent': (total_size / self.max_cache_size) * 100
        }
    
    def cleanup_expired_cache(self):
        """Remove expired cache entries"""
        cutoff_time = datetime.now() - self.default_ttl
        
        removed_files = 0
        freed_space = 0
        
        for cache_file in self.cache_dir.rglob("*"):
            if cache_file.is_file():
                file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                
                if file_mtime < cutoff_time:
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    removed_files += 1
                    freed_space += file_size
        
        return {
            'removed_files': removed_files,
            'freed_space_gb': freed_space / (1024**3)
        }
    
    def optimize_cache_structure(self):
        """Optimize cache directory structure"""
        # Reorganize cache by access patterns
        access_log = self.cache_dir / "access.log"
        
        if access_log.exists():
            # Read access patterns
            with open(access_log, 'r') as f:
                access_data = [line.strip().split(',') for line in f]
            
            # Sort by access frequency
            file_access_count = {}
            for timestamp, filename in access_data:
                file_access_count[filename] = file_access_count.get(filename, 0) + 1
            
            # Create hot/cold cache directories
            hot_cache = self.cache_dir / "hot"
            cold_cache = self.cache_dir / "cold"
            hot_cache.mkdir(exist_ok=True)
            cold_cache.mkdir(exist_ok=True)
            
            # Move files based on access patterns
            threshold = 10  # Files accessed more than 10 times are "hot"
            
            for filename, count in file_access_count.items():
                source_path = self.cache_dir / filename
                if source_path.exists() and source_path.is_file():
                    if count > threshold:
                        dest_path = hot_cache / filename
                    else:
                        dest_path = cold_cache / filename
                    
                    if not dest_path.exists():
                        shutil.move(str(source_path), str(dest_path))
```

## Log Management

### Log Analysis and Cleanup

```python
# maintenance/log_management.py
import gzip
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

class LogManager:
    def __init__(self, log_dir="/var/log/benchmark"):
        self.log_dir = Path(log_dir)
        self.error_patterns = [
            r'ERROR.*evaluation failed',
            r'CRITICAL.*system overload',
            r'WARNING.*high latency',
            r'ERROR.*CUDA out of memory'
        ]
    
    def analyze_error_patterns(self, days_back=7):
        """Analyze error patterns in logs"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        error_counts = defaultdict(int)
        error_details = defaultdict(list)
        
        # Process log files
        for log_file in self.log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                continue
            
            # Handle compressed logs
            if log_file.suffix == '.gz':
                file_opener = gzip.open
            else:
                file_opener = open
            
            try:
                with file_opener(log_file, 'rt') as f:
                    for line_num, line in enumerate(f, 1):
                        for pattern in self.error_patterns:
                            if re.search(pattern, line):
                                error_counts[pattern] += 1
                                error_details[pattern].append({
                                    'file': log_file.name,
                                    'line': line_num,
                                    'content': line.strip(),
                                    'timestamp': self._extract_timestamp(line)
                                })
            
            except Exception as e:
                print(f"Error processing {log_file}: {e}")
        
        return dict(error_counts), dict(error_details)
    
    def rotate_logs(self, max_size_mb=100, keep_days=30):
        """Rotate and compress log files"""
        max_size_bytes = max_size_mb * 1024 * 1024
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        rotated_files = []
        deleted_files = []
        
        for log_file in self.log_dir.glob("*.log"):
            file_size = log_file.stat().st_size
            file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            
            # Rotate large files
            if file_size > max_size_bytes:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                rotated_name = f"{log_file.stem}_{timestamp}.log"
                rotated_path = log_file.parent / rotated_name
                
                log_file.rename(rotated_path)
                
                # Compress rotated file
                with open(rotated_path, 'rb') as f_in:
                    with gzip.open(f"{rotated_path}.gz", 'wb') as f_out:
                        f_out.write(f_in.read())
                
                rotated_path.unlink()
                rotated_files.append(f"{rotated_path}.gz")
                
                # Create new empty log file
                log_file.touch()
            
            # Delete old compressed logs
            for compressed_log in self.log_dir.glob(f"{log_file.stem}_*.log.gz"):
                if datetime.fromtimestamp(compressed_log.stat().st_mtime) < cutoff_date:
                    compressed_log.unlink()
                    deleted_files.append(compressed_log.name)
        
        return {
            'rotated_files': rotated_files,
            'deleted_files': deleted_files
        }
    
    def generate_log_summary(self, days=7):
        """Generate log summary report"""
        error_counts, error_details = self.analyze_error_patterns(days)
        
        summary = f"""
Log Analysis Summary - Last {days} Days
{'=' * 50}

Error Pattern Counts:
"""
        for pattern, count in error_counts.items():
            summary += f"- {pattern}: {count} occurrences\n"
        
        summary += "\nTop Error Details:\n"
        for pattern, details in error_details.items():
            if details:
                summary += f"\n{pattern}:\n"
                for detail in details[:5]:  # Show top 5
                    summary += f"  - {detail['timestamp']}: {detail['content'][:100]}...\n"
        
        return summary
```

## Security Maintenance

### Security Auditing

```python
# maintenance/security_audit.py
import os
import subprocess
import json
from pathlib import Path

class SecurityAuditor:
    def __init__(self):
        self.audit_results = {}
    
    def check_file_permissions(self):
        """Audit file permissions for security"""
        critical_files = [
            '/etc/benchmark/config.yaml',
            '/var/log/benchmark/',
            '/data/models/',
            '/etc/ssl/certs/',
        ]
        
        permission_issues = []
        
        for file_path in critical_files:
            path = Path(file_path)
            if path.exists():
                stat_info = path.stat()
                mode = oct(stat_info.st_mode)[-3:]
                
                # Check for overly permissive permissions
                if file_path.endswith('.yaml') or file_path.endswith('.conf'):
                    if mode != '600':  # Should be readable only by owner
                        permission_issues.append({
                            'file': file_path,
                            'current_mode': mode,
                            'recommended_mode': '600'
                        })
                
                elif path.is_dir():
                    if mode not in ['755', '750']:
                        permission_issues.append({
                            'file': file_path,
                            'current_mode': mode,
                            'recommended_mode': '755'
                        })
        
        return permission_issues
    
    def check_service_security(self):
        """Check service security configuration"""
        security_issues = []
        
        # Check Redis security
        try:
            redis_info = subprocess.check_output(
                ['redis-cli', 'CONFIG', 'GET', 'requirepass'],
                text=True
            )
            if 'requirepass' not in redis_info or redis_info.strip().endswith('""'):
                security_issues.append({
                    'service': 'redis',
                    'issue': 'No password authentication configured',
                    'severity': 'high'
                })
        except subprocess.CalledProcessError:
            pass
        
        # Check nginx SSL configuration
        nginx_conf = Path('/etc/nginx/sites-enabled/benchmark')
        if nginx_conf.exists():
            with open(nginx_conf, 'r') as f:
                config_content = f.read()
                
            if 'ssl_certificate' not in config_content:
                security_issues.append({
                    'service': 'nginx',
                    'issue': 'SSL/TLS not configured',
                    'severity': 'high'
                })
            
            if 'ssl_protocols' not in config_content or 'TLSv1.3' not in config_content:
                security_issues.append({
                    'service': 'nginx',
                    'issue': 'Weak SSL/TLS protocols',
                    'severity': 'medium'
                })
        
        return security_issues
    
    def check_system_updates(self):
        """Check for available security updates"""
        try:
            # Update package lists
            subprocess.run(['apt', 'update'], check=True, capture_output=True)
            
            # Check for upgradeable packages
            result = subprocess.run(
                ['apt', 'list', '--upgradable'],
                capture_output=True, text=True
            )
            
            upgradeable = [line for line in result.stdout.split('\n') 
                          if 'upgradable' in line]
            
            # Check for security updates specifically
            security_updates = []
            for package_line in upgradeable:
                if 'security' in package_line.lower():
                    security_updates.append(package_line.strip())
            
            return {
                'total_updates': len(upgradeable) - 1,  # Subtract header line
                'security_updates': security_updates
            }
        
        except subprocess.CalledProcessError as e:
            return {'error': f"Failed to check updates: {e}"}
    
    def generate_security_report(self):
        """Generate comprehensive security audit report"""
        permission_issues = self.check_file_permissions()
        service_issues = self.check_service_security()
        update_info = self.check_system_updates()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'permission_issues': permission_issues,
            'service_security_issues': service_issues,
            'system_updates': update_info,
            'overall_security_score': self._calculate_security_score(
                permission_issues, service_issues, update_info
            )
        }
        
        return report
    
    def _calculate_security_score(self, permission_issues, service_issues, update_info):
        """Calculate overall security score (0-100)"""
        base_score = 100
        
        # Deduct points for issues
        base_score -= len(permission_issues) * 5
        
        for issue in service_issues:
            if issue['severity'] == 'high':
                base_score -= 20
            elif issue['severity'] == 'medium':
                base_score -= 10
            else:
                base_score -= 5
        
        # Deduct points for pending security updates
        if isinstance(update_info, dict) and 'security_updates' in update_info:
            base_score -= len(update_info['security_updates']) * 3
        
        return max(0, base_score)
```

## Automated Maintenance

### Maintenance Scheduler

```python
# maintenance/scheduler.py
import schedule
import time
import threading
from datetime import datetime

class MaintenanceScheduler:
    def __init__(self):
        self.maintenance_tasks = {}
        self.running = False
    
    def add_task(self, name, func, schedule_type, **kwargs):
        """Add maintenance task to scheduler"""
        self.maintenance_tasks[name] = {
            'function': func,
            'schedule_type': schedule_type,
            'kwargs': kwargs,
            'last_run': None,
            'next_run': None
        }
        
        # Schedule task based on type
        if schedule_type == 'daily':
            schedule.every().day.at(kwargs.get('time', '02:00')).do(
                self._run_task, name
            )
        elif schedule_type == 'weekly':
            schedule.every().week.at(kwargs.get('time', '03:00')).do(
                self._run_task, name
            )
        elif schedule_type == 'monthly':
            schedule.every().month.do(self._run_task, name)
    
    def _run_task(self, task_name):
        """Execute maintenance task with error handling"""
        task = self.maintenance_tasks.get(task_name)
        if not task:
            return
        
        try:
            print(f"Starting maintenance task: {task_name}")
            task['last_run'] = datetime.now()
            
            result = task['function']()
            
            print(f"Completed maintenance task: {task_name}")
            if result:
                print(f"Task result: {result}")
                
        except Exception as e:
            print(f"Maintenance task failed: {task_name} - {e}")
            # Log error or send alert
    
    def start_scheduler(self):
        """Start maintenance scheduler in background thread"""
        self.running = True
        
        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        thread = threading.Thread(target=run_scheduler, daemon=True)
        thread.start()
        print("Maintenance scheduler started")
    
    def stop_scheduler(self):
        """Stop maintenance scheduler"""
        self.running = False
        print("Maintenance scheduler stopped")

# Initialize and configure scheduler
def setup_maintenance_scheduler():
    scheduler = MaintenanceScheduler()
    
    # Add daily tasks
    scheduler.add_task("daily_health_check", daily_health_check, "daily", time="02:00")
    scheduler.add_task("log_cleanup", log_cleanup_task, "daily", time="03:00")
    
    # Add weekly tasks
    scheduler.add_task("performance_analysis", weekly_performance_analysis, "weekly", time="04:00")
    scheduler.add_task("database_optimization", redis_optimization, "weekly", time="05:00")
    
    # Add monthly tasks
    scheduler.add_task("security_audit", monthly_security_audit, "monthly")
    scheduler.add_task("cache_optimization", cache_optimization, "monthly")
    
    scheduler.start_scheduler()
    return scheduler
```

## References

- [Monitoring](./monitoring.md)
- [Security](./security.md)
- [Performance](./performance.md)
- [Deployment](./deployment.md)