import requests
import json
import time
import os
import textwrap

# 1. Configuration
API_URL = "http://127.0.0.1:8001/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json"
}
os.makedirs("test_results", exist_ok=True)

TEST_CASES = [
    {
        "name": "Test 1: Ultra-Complex Shell Pipeline with Process Substitution",
        "prompt": """
You are a Linux command-line expert. Generate a single, robust shell command that performs ALL of the following tasks in one pipeline:

**Task:**
1. Find all files in /var/log/, /home/*/.local/share/, and /tmp/ that:
   - End in .log, .txt, or .err
   - Were modified between 24-72 hours ago
   - Are larger than 1MB but smaller than 100MB
   - Contain EITHER "error", "fail", "critical" OR "exception" (case-insensitive)
   
2. For each matching file:
   - Extract lines containing timestamps in ISO 8601 or syslog format
   - Group errors by hour
   - Calculate the rate of errors per minute for each hour
   - Identify any hour where error rate exceeds 10 errors/minute
   
3. Generate a summary that includes:
   - Filename (basename only)
   - Total error count
   - Peak error hour and rate
   - SHA256 checksum of the file
   - File owner and permissions
   
4. Output must be:
   - Sorted by peak error rate (descending)
   - Formatted as TSV with headers
   - Include a final line with totals
   - Handle files with spaces, special chars, and unicode
   - Use process substitution to minimize intermediate files
   - Must work with dash/sh (POSIX compliant where possible)

**Requirements:**
- Single command using pipes, process substitution, and command grouping
- No temporary files
- Must handle edge cases (empty files, no matches, permission denied)
- Output only the command, no explanation
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False,
            "top_p": 0.9,
            "repeat_penalty": 1.15
        }
    },
    {
        "name": "Test 2: Complete Microservices Stack with Monitoring",
        "prompt": """
You are a DevOps specialist. Generate a complete production-ready microservices setup.

**Generate these files:**

1. **docker-compose.yml** with:
   - Python API service (FastAPI with async endpoints)
   - PostgreSQL with replication (primary + read replica)
   - Redis cluster (3 masters, 3 slaves)
   - Nginx reverse proxy with rate limiting
   - Prometheus + Grafana for monitoring
   - Jaeger for distributed tracing
   - ELK stack (Elasticsearch, Logstash, Kibana)
   - Health checks for all services
   - Resource limits and reservations
   - Custom networks with proper isolation
   - Volume management with backup considerations

2. **Dockerfile.api** (multi-stage build):
   - Build stage with poetry for dependencies
   - Runtime stage with distroless base
   - Non-root user with specific UIDs
   - Security scanning integration
   - Proper signal handling for graceful shutdown

3. **init-cluster.sh**:
   - Initialize PostgreSQL replication
   - Configure Redis cluster with proper sharding
   - Set up Prometheus scrape configs
   - Create Grafana dashboards via API
   - Configure log rotation
   - Set up SSL certificates with Let's Encrypt
   - Handle idempotency (can run multiple times safely)
   - Comprehensive error handling and rollback

4. **health-monitor.py**:
   - Async Python script monitoring all services
   - Custom health check endpoints
   - Automatic restart of unhealthy services
   - Slack/Discord webhook notifications
   - Metrics export to Prometheus
   - Circuit breaker pattern implementation

5. **backup-restore.sh**:
   - Automated backup of all stateful services
   - Point-in-time recovery for PostgreSQL
   - Redis snapshot management
   - Elasticsearch index snapshots
   - S3/MinIO integration for offsite backup
   - Restore verification and testing

Present each file in separate markdown code blocks with detailed inline comments.
""",
        "params": {
            "max_tokens": 4096,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 3: Advanced Log Correlation and Anomaly Detection",
        "prompt": """
You are a system monitoring agent performing advanced log analysis. Analyze the provided multi-source log data and produce a comprehensive JSON report with correlation and anomaly detection.

**Requirements:**
1. Parse logs from multiple sources (syslog, nginx, application, kernel, audit)
2. Correlate events across different log formats
3. Detect anomalies using statistical methods
4. Identify potential security incidents
5. Track service dependencies and cascade failures

**Output JSON structure:**
```json
{
  "analysis_metadata": {
    "start_time": "ISO8601",
    "end_time": "ISO8601",
    "total_events": int,
    "sources": ["list", "of", "sources"]
  },
  "events": [
    {
      "event_id": "UUID",
      "timestamp": "ISO8601",
      "source": "string",
      "severity": "critical|error|warning|info|debug",
      "category": "security|performance|availability|configuration",
      "summary": "string",
      "raw_log": "string",
      "extracted_fields": {},
      "correlated_events": ["event_id1", "event_id2"],
      "anomaly_score": float (0-1),
      "tags": ["list", "of", "tags"]
    }
  ],
  "correlations": [
    {
      "correlation_id": "UUID",
      "type": "temporal|causal|spatial",
      "confidence": float (0-1),
      "events": ["event_id1", "event_id2"],
      "pattern": "string description",
      "potential_root_cause": "string"
    }
  ],
  "anomalies": [
    {
      "anomaly_id": "UUID",
      "type": "statistical|behavioral|threshold",
      "severity": "high|medium|low",
      "description": "string",
      "affected_services": ["service1", "service2"],
      "recommended_action": "string",
      "related_events": ["event_id1", "event_id2"]
    }
  ],
  "security_incidents": [
    {
      "incident_id": "UUID",
      "type": "brute_force|privilege_escalation|data_exfiltration|dos",
      "confidence": float (0-1),
      "timeline": [
        {
          "timestamp": "ISO8601",
          "event": "string",
          "event_ids": ["event_id1"]
        }
      ],
      "indicators": ["IP addresses", "user accounts", "files"],
      "recommended_response": "string"
    }
  ],
  "service_health": {
    "service_name": {
      "status": "healthy|degraded|unhealthy|unknown",
      "error_rate": float,
      "response_time_p99": float,
      "dependencies": ["service1", "service2"],
      "recent_issues": ["description1", "description2"]
    }
  },
  "executive_summary": {
    "critical_findings": ["finding1", "finding2"],
    "trends": {
      "error_rate_trend": "increasing|stable|decreasing",
      "security_posture": "strong|moderate|weak",
      "system_stability": float (0-1)
    },
    "recommendations": ["action1", "action2"]
  }
}
```

---
**Multi-Source Log Data:**

Aug 03 13:30:01 prod-web-01 systemd[1]: Starting nginx.service...
Aug 03 13:30:02 prod-web-01 systemd[1]: Started nginx.service.
Aug 03 13:30:15 prod-web-01 nginx[2345]: 192.168.1.100 - - [03/Aug/2025:13:30:15 +0000] "GET /api/users HTTP/1.1" 200 1245 "-" "Mozilla/5.0"
Aug 03 13:30:16 prod-api-01 app[3456]: INFO: User authentication successful for user_id=12345
Aug 03 13:30:17 prod-web-01 nginx[2345]: 10.0.0.50 - - [03/Aug/2025:13:30:17 +0000] "POST /api/login HTTP/1.1" 401 89 "-" "python-requests/2.28.1"
Aug 03 13:30:18 prod-api-01 app[3456]: WARNING: Failed login attempt for username=admin from IP=10.0.0.50
Aug 03 13:30:19 prod-web-01 nginx[2345]: 10.0.0.50 - - [03/Aug/2025:13:30:19 +0000] "POST /api/login HTTP/1.1" 401 89 "-" "python-requests/2.28.1"
Aug 03 13:30:20 prod-api-01 app[3456]: WARNING: Failed login attempt for username=admin from IP=10.0.0.50
Aug 03 13:30:21 prod-web-01 nginx[2345]: 10.0.0.50 - - [03/Aug/2025:13:30:21 +0000] "POST /api/login HTTP/1.1" 401 89 "-" "python-requests/2.28.1"
Aug 03 13:30:22 prod-api-01 app[3456]: ERROR: Brute force detected - 3 failed attempts for username=admin from IP=10.0.0.50
Aug 03 13:30:23 prod-db-01 postgres[4567]: ERROR: connection limit exceeded for non-superusers
Aug 03 13:30:24 prod-api-01 app[3456]: ERROR: Database connection failed - psycopg2.OperationalError
Aug 03 13:30:25 prod-web-01 nginx[2345]: 192.168.1.100 - - [03/Aug/2025:13:30:25 +0000] "GET /api/users HTTP/1.1" 500 45 "-" "Mozilla/5.0"
Aug 03 13:30:26 prod-api-01 app[3456]: CRITICAL: Service degraded - database unreachable
Aug 03 13:30:27 prod-monitor-01 prometheus[5678]: ALERT ServiceDown service=postgresql instance=prod-db-01
Aug 03 13:30:30 prod-db-01 kernel: Out of memory: Kill process 4567 (postgres) score 850 or sacrifice child
Aug 03 13:30:31 prod-db-01 systemd[1]: postgresql.service: Main process exited, code=killed, status=9/KILL
Aug 03 13:30:32 prod-api-01 app[3456]: INFO: Circuit breaker activated for database connections
Aug 03 13:30:45 prod-web-01 nginx[2345]: 10.0.0.51 - - [03/Aug/2025:13:30:45 +0000] "GET /.git/config HTTP/1.1" 404 125 "-" "curl/7.68.0"
Aug 03 13:30:46 prod-web-01 nginx[2345]: 10.0.0.51 - - [03/Aug/2025:13:30:46 +0000] "GET /wp-admin/ HTTP/1.1" 404 125 "-" "curl/7.68.0"
Aug 03 13:30:47 prod-web-01 nginx[2345]: 10.0.0.51 - - [03/Aug/2025:13:30:47 +0000] "GET /phpmyadmin/ HTTP/1.1" 404 125 "-" "curl/7.68.0"
Aug 03 13:30:48 prod-fw-01 audit[6789]: SUSPICIOUS: Port scan detected from 10.0.0.51 - 50 ports in 3 seconds
Aug 03 13:31:00 prod-db-01 systemd[1]: Started postgresql.service.
Aug 03 13:31:01 prod-db-01 postgres[7890]: LOG: database system is ready to accept connections
Aug 03 13:31:02 prod-api-01 app[3456]: INFO: Database connection restored
Aug 03 13:31:03 prod-api-01 app[3456]: INFO: Processing 47 queued requests
Aug 03 13:31:15 prod-web-01 nginx[2345]: 192.168.1.100 - - [03/Aug/2025:13:31:15 +0000] "GET /api/users HTTP/1.1" 200 1245 "-" "Mozilla/5.0"
Aug 03 13:31:30 prod-storage-01 kernel: XFS (sda1): Corruption detected. Unmount and run xfs_repair
Aug 03 13:31:31 prod-storage-01 systemd[1]: Emergency mode activated
Aug 03 13:31:32 prod-api-01 app[3456]: ERROR: Storage service unreachable - Failed to upload user data
Aug 03 13:31:33 prod-api-01 app[3456]: ERROR: Fallback to local cache failed - Disk full
---

Output only the JSON object. No markdown formatting or explanations.
""",
        "params": {
            "max_tokens": 4096,
            "temperature": 0.0,
            "stream": False,
            "response_format": {"type": "json_object"}
        }
    },
    {
        "name": "Test 4: Comprehensive System Performance Analysis Script",
        "prompt": """
You are a Linux performance engineering expert. Generate a complete Python script that performs deep system performance analysis and optimization recommendations.

**Script Requirements:**

The script must analyze and report on:
1. **CPU Performance:**
   - Per-core utilization with thermal throttling detection
   - Context switch rates and their impact
   - Cache miss rates using perf events
   - Process CPU affinity optimization recommendations
   - Identify CPU-bound processes with thread analysis
   - Detect priority inversion issues

2. **Memory Analysis:**
   - Detailed memory fragmentation analysis
   - Page fault rates (minor vs major)
   - NUMA node usage and optimization
   - Memory leak detection using /proc analysis
   - Shared memory and huge pages utilization
   - OOM killer prediction based on trends
   - Memory bandwidth saturation detection

3. **I/O Performance:**
   - Block device queue depths and latencies
   - Filesystem cache effectiveness
   - inode and dentry cache pressure
   - Per-process I/O patterns (sequential vs random)
   - NVMe specific metrics and wear leveling
   - RAID array performance and degradation
   - Network filesystem (NFS/CIFS) bottlenecks

4. **Network Stack Analysis:**
   - TCP retransmission rates per connection
   - Socket buffer tuning recommendations
   - Connection state distribution analysis
   - Packet processing latencies
   - Hardware offload utilization (TSO, GRO, etc.)
   - eBPF-based packet drop analysis
   - Container network namespace overhead

5. **Advanced Features:**
   - Real-time kernel scheduling analysis
   - cgroup resource limit effectiveness
   - SystemTap/eBPF probe integration
   - Correlation with hardware events (MCE logs)
   - Time-series data collection for trend analysis
   - Machine learning-based anomaly detection
   - Automated remediation script generation

**Output Format:**
- Interactive ncurses-based dashboard
- JSON export for monitoring integration
- HTML report with D3.js visualizations
- Prometheus metrics endpoint
- Specific optimization scripts generated

The script must handle:
- Running with minimal performance impact
- Privilege escalation when needed
- Kernel version compatibility (4.x to 6.x)
- Container and VM awareness
- Multiple architecture support (x86_64, ARM64)

Include extensive error handling, logging, and the ability to run continuously as a daemon with configurable intervals.
""",
        "params": {
            "max_tokens": 6144,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 5: Automated Security Audit and Hardening Framework",
        "prompt": """
You are a Linux security expert. Create a comprehensive security audit and automated hardening framework.

**Generate these components:**

1. **security-audit.sh** - Master audit script that:
   - Performs CIS benchmark compliance checking
   - Scans for CVEs in installed packages
   - Analyzes SSH configurations and key strengths
   - Detects privilege escalation vectors
   - Audits sudo rules and PAM configurations
   - Checks for world-writable files and SUID/SGID binaries
   - Analyzes firewall rules and network exposure
   - Reviews SELinux/AppArmor policies
   - Scans for rootkits and backdoors
   - Audits container security (Docker/Podman)
   - Checks for weak SSL/TLS configurations
   - Analyzes systemd service hardening
   - Reviews kernel parameters and modules
   - Detects information disclosure risks
   - Performs supply chain security checks

2. **auto-harden.py** - Python script that:
   - Parses audit results and generates fixes
   - Implements compensating controls
   - Creates custom SELinux policies
   - Generates iptables/nftables rules
   - Configures fail2ban with custom jails
   - Sets up AIDE/Tripwire with optimal rules
   - Implements USB device control
   - Configures audit rules for compliance
   - Sets up centralized logging with encryption
   - Implements time-based access controls
   - Creates automated backup of changes
   - Provides rollback functionality

3. **threat-monitor.go** - Go-based real-time monitor:
   - Uses eBPF for syscall monitoring
   - Detects process injection attempts
   - Monitors file integrity in real-time
   - Tracks network connections with GeoIP
   - Implements UEBA (User Behavior Analytics)
   - Detects lateral movement patterns
   - Monitors for cryptomining activity
   - Implements kill chain detection
   - Integrates with SIEM systems
   - Provides GraphQL API for queries

4. **incident-response.yaml** - Ansible playbook that:
   - Automates incident containment
   - Performs memory acquisition
   - Collects forensic artifacts
   - Isolates affected systems
   - Implements network segmentation
   - Captures network traffic
   - Preserves evidence chain of custody
   - Generates timeline of events
   - Performs automated triage
   - Integrates with ticketing systems

5. **security-test.rs** - Rust-based penetration testing tool:
   - Performs authenticated security testing
   - Tests for common misconfigurations
   - Validates hardening effectiveness
   - Simulates APT techniques
   - Tests detection capabilities
   - Generates detailed reports
   - Provides remediation verification

Each component must include:
- Comprehensive error handling
- Detailed logging with syslog integration
- Support for air-gapped environments
- Integration with common security tools
- Performance impact minimization
- Multi-distro compatibility
""",
        "params": {
            "max_tokens": 8192,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 6: Distributed System Orchestration and Chaos Engineering",
        "prompt": """
You are a distributed systems architect. Design a complete orchestration and chaos engineering platform for testing system resilience.

**Create these components:**

1. **orchestrator.py** - Main orchestration engine:
   - Implements Raft consensus for HA
   - Service discovery with health checking
   - Dynamic load balancing algorithms
   - Circuit breaker pattern implementation
   - Distributed tracing integration
   - A/B testing and canary deployments
   - Feature flag management
   - Distributed lock management
   - Event sourcing for state management
   - CQRS pattern implementation
   - Saga pattern for transactions
   - Service mesh integration
   - GraphQL federation gateway
   - WebSocket-based real-time updates

2. **chaos-controller.go** - Chaos engineering controller:
   - Network partition simulation
   - Latency and packet loss injection
   - CPU and memory stress testing
   - Disk I/O throttling
   - Time drift simulation
   - Process killing strategies
   - Kernel panic simulation
   - DNS poisoning tests
   - Certificate expiration simulation
   - Resource exhaustion attacks
   - Cascading failure scenarios
   - Byzantine failure injection
   - Split-brain scenario testing
   - Data corruption simulation

3. **distributed-test.yaml** - Test scenario definitions:
   - Multi-region failure scenarios
   - Database replication lag testing
   - Message queue overflow conditions
   - Cache avalanche simulation
   - Thundering herd problems
   - Race condition detection
   - Deadlock scenario creation
   - Network topology changes
   - Storage failure patterns
   - Clock synchronization issues
   - Service dependency failures
   - Data consistency verification
   - Performance degradation curves
   - Recovery time objectives

4. **mesh-config.lua** - Service mesh configuration:
   - Envoy Lua scripting for traffic shaping
   - Custom load balancing algorithms
   - Request hedging strategies
   - Retry policies with jitter
   - Circuit breaker tuning
   - Outlier detection rules
   - Rate limiting with tokens
   - Header-based routing
   - Shadow traffic testing
   - Fault injection rules
   - Observability integration
   - Security policy enforcement
   - Multi-cluster federation
   - Traffic encryption policies

5. **analytics-engine.rs** - Real-time analytics:
   - Stream processing with backpressure
   - Complex event correlation
   - Anomaly detection algorithms
   - Predictive failure analysis
   - Capacity planning models
   - Cost optimization recommendations
   - Performance regression detection
   - SLO/SLA tracking and alerting
   - Distributed tracing analysis
   - Log aggregation and parsing
   - Metric correlation engine
   - Custom dashboard generation
   - Report automation system
   - ML model deployment pipeline

6. **recovery-agent.sh** - Automated recovery:
   - Intelligent restart strategies
   - Data consistency verification
   - Automated rollback procedures
   - State reconciliation logic
   - Backup restoration automation
   - Cross-region failover
   - DNS failover automation
   - Load rebalancing during recovery
   - Health check verification
   - Post-mortem generation
   - Runbook automation
   - Communication templates
   - Stakeholder notifications
   - Recovery metrics collection

Include:
- Kubernetes operator patterns
- Terraform infrastructure as code
- Prometheus rules and Grafana dashboards
- Documentation generation
- API specifications (OpenAPI, gRPC)
- Integration test suites
- Performance benchmarking harnesses
- Security scanning pipelines
- Compliance reporting tools
""",
        "params": {
            "max_tokens": 10240,
            "temperature": 0.1,
            "stream": False
        }
    }
]

def run_performance_test(test_case):
    """Sends a request to the server for a given test case and prints performance metrics."""
    
    name = test_case["name"]
    prompt = test_case["prompt"]
    params = test_case["params"]
    
    print(f"\n{'='*80}\n--- 🚀 RUNNING: {name} ---\n{'='*80}")
    
    # Use textwrap to print a more readable version of the prompt
    print("--- Prompt Snippet ---")
    print(textwrap.shorten(prompt, width=120, placeholder="..."))
    print("-" * 22)

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        **params
    }
    
    try:
        start_time = time.time()
        # A generous timeout is needed for long generation tasks on a CPU.
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=1800) # 30-minute timeout
        response.raise_for_status()
        end_time = time.time()

        response_data = response.json()
        
        print("\n--- ✅ SUCCESS: Server Responded ---")
        
        # --- Performance Metrics ---
        total_duration = end_time - start_time
        usage_data = response_data.get("usage", {})
        completion_tokens = usage_data.get("completion_tokens", 0)
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        total_tokens = usage_data.get("total_tokens", 0)
        
        print(f"\n--- 📊 Performance Metrics for '{name}' ---")
        print(f"Total Request Time: {total_duration:.2f} seconds")
        print(f"Prompt Tokens: {prompt_tokens} | Completion Tokens: {completion_tokens} | Total Tokens: {total_tokens}")
        
        if completion_tokens > 0 and total_duration > 0:
            tokens_per_second = completion_tokens / total_duration
            print(f"Tokens per Second (T/s): {tokens_per_second:.2f}")
        else:
            print("Tokens per Second (T/s): N/A (no completion tokens or duration)")

        # --- Save Response ---
        full_content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        filename_safe_name = name.lower().replace(":", "").replace("&", "and").replace(" ", "_")
        output_path = os.path.join("test_results", f"{filename_safe_name}_response.md")
        
        with open(output_path, "w", encoding="utf-8") as f:
            # If the response is expected to be JSON, try to format it nicely.
            if "json" in name.lower():
                try:
                    json_obj = json.loads(full_content)
                    f.write("```json\n")
                    f.write(json.dumps(json_obj, indent=2))
                    f.write("\n```")
                except json.JSONDecodeError:
                    f.write(full_content) # Write as-is if not valid JSON
            else:
                f.write(full_content)
                
        print(f"\n--- 📜 Full response saved to '{output_path}' ---")
        print(f"--- (Preview of first 300 chars) ---\n{full_content[:300]}...")

        # --- Performance Summary ---
        with open(os.path.join("test_results", "performance_summary.txt"), "a", encoding="utf-8") as f:
            f.write(f"\n{name}\n")
            f.write(f"Duration: {total_duration:.2f}s | ")
            f.write(f"Tokens: {total_tokens} | ")
            f.write(f"T/s: {tokens_per_second:.2f}\n")
            f.write("-" * 80)

    except requests.exceptions.Timeout:
        print(f"\n❌ ERROR: The request for '{name}' timed out. The model generation took too long.")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ ERROR: Could not connect to the server for '{name}'.")
        print(f"   Is the llama.cpp server running and accessible at {API_URL}?")
        print(f"   Details: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during '{name}': {e}")

if __name__ == "__main__":
    print("--- Starting Enhanced AI Agent CPU Performance Test Suite for Linux Devs ---")
    print(f"Targeting Server: {API_URL}")
    print(f"Testing {len(TEST_CASES)} complex scenarios...")
    
    # Clear performance summary file
    with open(os.path.join("test_results", "performance_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== PERFORMANCE SUMMARY ===\n")
    
    for i, test in enumerate(TEST_CASES):
        print(f"\n\n[{i+1}/{len(TEST_CASES)}] Preparing test...")
        run_performance_test(test)
        
    print(f"\n{'='*80}\n--- ✅ All tests completed. Check the 'test_results' directory for outputs. ---\n{'='*80}")