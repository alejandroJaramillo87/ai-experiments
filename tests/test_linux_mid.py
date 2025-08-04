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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
            "max_tokens": 8192,
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
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=300)
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
    print("--- Advanced Linux Agent Test Suite (30 Complex Challenges) ---")
    print(f"Targeting Server: {API_URL}")
    print(f"Total Tests: {len(TEST_CASES)}")
    print("\nTest Design Philosophy:")
    print("- Complex, multi-step problems requiring reasoning.")
    print("- Scenarios based on modern, production-level tooling (Ansible, eBPF, asyncio).")
    print("- Emphasis on automation, security, and robust error handling.")
    
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