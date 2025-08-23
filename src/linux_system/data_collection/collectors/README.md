# SystemCollector Documentation

## Overview

The SystemCollector is a comprehensive, read-only data collection component designed to gather Ubuntu system intelligence for LLM optimization and AI workstation monitoring. It provides deep visibility into system performance, hardware utilization, security posture, and development environments.

## Key Features

### 🎯 LLM Optimization Focus
- **RTX 5090 GPU monitoring**: Real-time utilization, memory patterns, thermal states
- **Storage I/O analysis**: Model loading bottlenecks, cache performance, disk health
- **Python environment tracking**: Framework conflicts, CUDA availability, package security
- **NUMA topology analysis**: Memory allocation patterns for large model inference

### 🔒 Security & Safety
- **Read-only operations**: No system modifications, safe for production use
- **Permission validation**: Comprehensive file and command access checks
- **Resource limits**: Timeout controls, file size limits, process constraints
- **Error handling**: Graceful degradation when commands/files unavailable

### 📊 Comprehensive Data Coverage
- **Hardware**: CPU, GPU, memory, storage, network, sensors
- **Performance**: Real-time metrics, I/O patterns, resource utilization
- **Security**: User accounts, permissions, firewall, audit logs
- **Development**: Python environments, AI/ML frameworks, virtual environments

## Architecture

```
SystemCollector
├── collect() → Dict[str, Any]
├── Hardware Collection
│   ├── _collect_nvidia_gpu_data()      # RTX 5090 specific metrics
│   ├── _collect_cuda_environment()     # CUDA runtime & libraries
│   ├── _collect_cpu_data()             # CPU topology, frequencies, usage
│   ├── _collect_memory_data()          # NUMA, swap, allocation patterns
│   └── _collect_hardware_data()        # PCI, USB, sensors, DMI
├── Storage & I/O
│   └── _collect_comprehensive_storage_data()
│       ├── _collect_storage_hardware()     # NVMe, SCSI, controllers
│       ├── _collect_storage_performance()  # I/O metrics, schedulers
│       ├── _collect_storage_health()       # SMART data, temperatures
│       └── _collect_ai_specific_storage()  # Model files, cache analysis
├── Security & Access
│   └── _collect_security_data()
│       ├── _collect_user_security()        # Users, groups, sudo
│       ├── _collect_network_security()     # Connections, firewall
│       ├── _collect_permissions_data()     # File permissions, SUID
│       └── _collect_audit_logs()           # Authentication, security events
├── Development Environment
│   └── _collect_python_environment_data()
│       ├── _collect_python_installations() # All Python versions
│       ├── _collect_virtual_environments() # venv, conda, poetry
│       └── _collect_ai_ml_frameworks()     # torch, transformers, etc.
└── System Foundation
    ├── _collect_kernel_data()          # Version, modules, parameters
    ├── _collect_network_data()         # Interfaces, routing, statistics
    ├── _collect_process_data()         # Process tree, resource usage
    └── _collect_performance_data()     # I/O stats, VM statistics
```

## Data Structure

The SystemCollector returns a structured dictionary with metadata:

```python
{
    'metadata': {
        'collector': 'system',
        'timestamp': '2024-01-15T10:30:00',
        'collection_count': 42,
        'data_hash': 'a1b2c3d4e5f6...',
        'collection_duration_ms': 1250
    },
    'data': {
        'cpu': {...},           # CPU topology, usage, frequencies
        'memory': {...},        # Memory usage, NUMA topology
        'nvidia_gpu': {...},    # RTX 5090 comprehensive metrics
        'cuda': {...},          # CUDA environment and capabilities
        'storage': {...},       # Storage hardware, performance, health
        'security': {...},      # Security posture and audit data
        'python_env': {...},    # Python environments and AI frameworks
        'hardware': {...},      # PCI devices, sensors, system info
        'kernel': {...},        # Kernel version, modules, parameters
        'network': {...},       # Network interfaces and connections
        'processes': {...},     # Process information and resource usage
        'performance': {...},   # I/O and performance statistics
        'docker_nvidia': {...}  # Docker GPU container integration
    }
}
```

## Usage Examples

### Basic Collection

```python
from ubuntu_llm_system.data_collection.collectors import SystemCollector

# Initialize collector with 30-second interval
collector = SystemCollector(collection_interval=30)

# Check if collection should run
if collector.should_collect():
    # Collect comprehensive system data
    snapshot = collector.collect()
    
    # Access specific data sections
    gpu_data = snapshot['data']['nvidia_gpu']
    storage_data = snapshot['data']['storage']
    security_data = snapshot['data']['security']
    
    print(f"Collection completed in {snapshot['metadata']['collection_duration_ms']}ms")
```

### LLM Performance Analysis

```python
# Analyze GPU utilization for LLM inference
snapshot = collector.collect()
gpu_metrics = snapshot['data']['nvidia_gpu']['basic_metrics']

# Check storage performance for model loading
storage_perf = snapshot['data']['storage']['performance']['iostat_extended']

# Verify CUDA environment
cuda_available = snapshot['data']['python_env']['ai_frameworks']['torch_cuda']

# Analyze memory pressure
memory_info = snapshot['data']['memory']['meminfo']
numa_topology = snapshot['data']['cpu']['numa']
```

### Security Monitoring

```python
# Monitor system security posture
security_data = snapshot['data']['security']

# Check for suspicious file permissions
suid_files = security_data['permissions']['suid_files']
world_writable = security_data['permissions']['world_writable']

# Review authentication events
auth_logs = security_data['audit']['auth_log']
failed_logins = security_data['audit']['failed_logins']

# Firewall status
firewall_status = security_data['firewall']['ufw_status']
```

### Python Environment Analysis

```python
# Analyze Python development environment
python_env = snapshot['data']['python_env']

# Check AI framework versions
frameworks = python_env['ai_frameworks']
torch_version = frameworks.get('torch', 'Not installed')
transformers_version = frameworks.get('transformers', 'Not installed')

# Virtual environment discovery
venvs = python_env['virtual_envs']['virtual_environments']

# Package security vulnerabilities
security_issues = python_env['packages'].get('pip_security', '')
```

## Configuration

### Collection Intervals

```python
# High-frequency monitoring (every 10 seconds)
fast_collector = SystemCollector(collection_interval=10)

# Standard monitoring (every 30 seconds)
standard_collector = SystemCollector(collection_interval=30)

# Low-frequency monitoring (every 5 minutes)
slow_collector = SystemCollector(collection_interval=300)
```

### Safety Limits

The collector enforces several safety limits:

- **File Size Limit**: 50MB maximum file read
- **Command Timeout**: 30 seconds maximum execution
- **Process Limit**: Top 20 processes for NUMA maps
- **Sample Limit**: Limited log lines and output samples

## Error Handling

The SystemCollector gracefully handles various error conditions:

- **Missing Commands**: Continues collection if optional commands unavailable
- **Permission Denied**: Logs warnings but doesn't crash collection
- **File Not Found**: Safely handles missing system files
- **Timeout Errors**: Prevents hanging on slow system calls

## Performance Considerations

### Resource Usage
- **Memory**: Typically 10-50MB during collection
- **CPU**: Low impact, mostly I/O bound operations
- **Disk**: Read-only operations, no write overhead
- **Network**: Local system only, no external calls

### Collection Timing
- **Fast Items**: /proc filesystem reads (1-10ms)
- **Medium Items**: System commands (50-200ms)
- **Slow Items**: Storage analysis, security scans (500ms-2s)
- **Total Duration**: Usually 1-5 seconds for complete collection

## Integration Points

### With Other Collectors
- **LogCollector**: SystemCollector focuses on current state, LogCollector on historical patterns
- **ServiceCollector**: SystemCollector covers system services, ServiceCollector covers application services
- **UserCollector**: SystemCollector covers user security, UserCollector covers activity patterns

### With Analysis Systems
- **RAG Integration**: Real-time system state for contextual queries
- **Fine-tuning**: System state correlation with performance patterns
- **Alerting**: Threshold monitoring for critical system metrics

## Troubleshooting

### Common Issues

**Collection Returns Empty Data**
```python
# Check collector status
status = collector.get_status()
print(f"Last collection: {status['last_collection_time']}")
print(f"Collection count: {status['collection_count']}")
```

**Slow Collection Performance**
```python
# Enable timing analysis
snapshot = collector.collect()
duration = snapshot['metadata']['collection_duration_ms']
if duration > 5000:  # More than 5 seconds
    print("Collection is slow, check system load")
```

**Missing GPU Data**
- Verify NVIDIA drivers installed: `nvidia-smi`
- Check CUDA installation: `nvcc --version`
- Confirm permissions: User can access `/dev/nvidia*`

**Permission Errors**
- Some system files require root access
- Collector gracefully skips inaccessible files
- Check logs for specific permission warnings

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
collector = SystemCollector()
snapshot = collector.collect()  # Will show detailed debug info
```

## Security Considerations

### Data Privacy
- **No Sensitive Data**: Avoids reading password files, private keys
- **Local Only**: All operations stay on local system
- **Read-Only**: No system modifications or state changes

### Access Control
- **File Permissions**: Respects system file permissions
- **Command Validation**: Only allows predefined safe commands
- **Resource Limits**: Prevents resource exhaustion attacks

### Audit Trail
- **Collection Logging**: All collection attempts logged
- **Error Tracking**: Failed operations recorded
- **Timing Data**: Performance metrics for anomaly detection