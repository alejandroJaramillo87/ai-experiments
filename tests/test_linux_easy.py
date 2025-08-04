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
        "name": "Test 1: Multi-Source Log Analysis",
        "prompt": """
Write a bash command that:
1. Searches /var/log for .log files modified in last 24 hours
2. Counts occurrences of "error" or "fail" (case-insensitive)
3. Shows filename and count, sorted by highest count
4. Excludes files with 0 matches

Use grep, find, and sort. One-liner preferred.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 2: Container Stack Setup",
        "prompt": """
Create a docker-compose.yml with:
- nginx (port 80) as reverse proxy
- python app (port 5000) 
- redis for caching
- proper networking (frontend/backend)
- healthchecks for each service
- restart policies

Keep it concise but production-ready.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 3: Firewall Security Rules",
        "prompt": """
Write iptables rules that:
1. Allow SSH only from 192.168.1.0/24
2. Rate limit HTTP/HTTPS to 100 connections/minute
3. Drop packets from IPs with >50 failed connections
4. Log blocked attempts
5. Allow established connections

Include comments for each rule.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 4: System Health Monitor",
        "prompt": """
Create a bash script that checks:
- CPU usage (warn if >80%)
- Memory usage (warn if >90%)
- Disk usage for / (warn if >85%)
- Load average (warn if >4)

Output format: "METRIC: VALUE [WARNING]"
Include proper error handling.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 5: Security Quick Scan",
        "prompt": """
Write a script that detects:
1. SUID binaries in /tmp or /home
2. Users with UID 0 besides root
3. World-writable files in /etc
4. Failed SSH attempts (last 100 lines of auth.log)

Show results with severity (HIGH/MEDIUM/LOW).
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 6: Database Backup Automation",
        "prompt": """
Create a PostgreSQL backup script that:
1. Dumps database 'production' 
2. Compresses with gzip
3. Names file with date: backup_YYYYMMDD.sql.gz
4. Keeps only last 7 backups
5. Logs success/failure

Include error handling and exit codes.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 7: Service Manager Script",
        "prompt": """
Write a script that:
1. Checks if nginx, mysql, redis are running
2. Restarts any stopped services
3. Waits 5 seconds and verifies they started
4. Sends alert if restart failed
5. Logs all actions with timestamps

Use systemctl and proper exit codes.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 8: Network Diagnostics",
        "prompt": """
Create a network test script that:
1. Pings 3 hosts (8.8.8.8, 1.1.1.1, gateway)
2. Checks DNS resolution
3. Tests HTTP connectivity to google.com
4. Measures average latency
5. Reports any failures

Output should be clear and actionable.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 9: Log Rotation Setup",
        "prompt": """
Create a logrotate config for:
- /var/log/myapp/*.log
- Rotate daily, keep 14 days
- Compress after 1 day
- Size limit 100M
- Create new files as user:group 2048
- Run post-rotate script to reload app

Include all necessary directives.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 10: Process Monitor Daemon",
        "prompt": """
Write a monitoring script that:
1. Checks every 30 seconds if 'myapp' process exists
2. If not running, starts it with: /usr/bin/myapp
3. If crashes 3 times in 5 minutes, stop trying
4. Logs to syslog
5. Handles SIGTERM gracefully

Make it daemon-friendly.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 11: Backup Verification",
        "prompt": """
Create a script that:
1. Tests latest backup in /backup/
2. Restores to /tmp/verify/
3. Checks if key files exist
4. Compares file count with original
5. Reports integrity status

Clean up after verification.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 12: SSL Certificate Manager",
        "prompt": """
Write a script that:
1. Checks SSL certificates in /etc/ssl/certs/
2. Warns if expiring within 30 days
3. Shows: domain, expiry date, days left
4. Optionally renews with certbot
5. Reloads nginx after renewal

Format output as a table.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 13: Docker Cleanup Tool",
        "prompt": """
Create a Docker maintenance script:
1. Remove stopped containers older than 7 days
2. Remove unused images (no container)
3. Clean build cache
4. Remove unused volumes
5. Show space reclaimed
6. Keep images tagged 'latest' or 'prod'

Include dry-run option.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 14: Performance Baseline",
        "prompt": """
Write a script that captures:
1. Current CPU, memory, disk usage
2. Top 5 processes by CPU and memory
3. Network throughput (last minute)
4. Active connections count
5. System uptime and load

Output as JSON for monitoring integration.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 15: User Audit Script",
        "prompt": """
Create a user audit script showing:
1. Users who haven't logged in for 90 days
2. Users with password expiry <7 days
3. Users without home directories
4. Accounts with no password set
5. Sudo users list

Mark security concerns as [WARNING].
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 16: API Health Checker",
        "prompt": """
Write a script that monitors APIs:
1. Check https://api.example.com/health
2. Verify status code is 200
3. Check response time <2 seconds
4. Parse JSON for status: "ok"
5. Alert if 3 consecutive failures
6. Log response times

Use curl with proper timeouts.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 17: Git Repository Maintenance",
        "prompt": """
Create a git maintenance script:
1. Fetch all remotes
2. Prune deleted remote branches
3. Delete merged local branches
4. Run garbage collection
5. Show repository size before/after
6. Update hooks if needed

Handle bare and normal repos.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 18: System Update Manager",
        "prompt": """
Write an update script that:
1. Checks for security updates
2. Lists packages to be updated
3. Creates system snapshot/backup
4. Applies updates with logging
5. Reboots if kernel updated
6. Verifies system health after

Support apt and yum.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 19: Resource Limiter",
        "prompt": """
Create a script using cgroups or nice that:
1. Limits process CPU to 50%
2. Limits memory to 1GB
3. Sets I/O priority to low
4. Monitors actual usage
5. Kills if exceeds limits for 60s

Accept PID or process name as argument.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 20: Service Dependency Checker",
        "prompt": """
Write a script that:
1. Reads service dependencies from config
2. Checks each service status
3. Identifies broken dependencies  
4. Suggests start order
5. Detects circular dependencies

Config format: SERVICE:DEPENDS_ON1,DEPENDS_ON2
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 21: Automated Troubleshooter",
        "prompt": """
Create a troubleshooting script for web issues:
1. Check if nginx is running
2. Test if port 80/443 are listening
3. Verify DNS resolution
4. Check disk space
5. Review last 10 error log entries
6. Test database connectivity

Provide specific fix suggestions.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 22: Compliance Checker",
        "prompt": """
Write a basic compliance script checking:
1. SSH: no root login, key-only auth
2. Firewall is enabled
3. Passwords meet complexity requirements  
4. Unnecessary services disabled
5. File permissions on /etc/passwd, /etc/shadow

Output pass/fail with remediation steps.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 23: Load Balancer Health",
        "prompt": """
Create a script that tests load balancer:
1. Check each backend server health
2. Verify response times <500ms
3. Test session persistence
4. Monitor active connections
5. Alert if any backend down
6. Show traffic distribution

Output health percentage and details.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 24: Database Performance",
        "prompt": """
Write a MySQL/PostgreSQL script that:
1. Shows slow queries (>1 second)
2. Lists tables needing optimization
3. Checks for missing indexes
4. Shows connection pool usage
5. Identifies lock waits

Format as actionable report.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 25: Container Security Scan",
        "prompt": """
Create a container security checker:
1. List running containers
2. Check for containers running as root
3. Verify resource limits are set
4. Check for outdated base images
5. Scan for exposed sensitive ports
6. Review mounted volumes

Categorize findings by severity.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 26: Automated Deployment",
        "prompt": """
Write a deployment script that:
1. Pulls latest code from git
2. Runs tests
3. Builds application
4. Creates backup of current version
5. Deploys with zero downtime
6. Rolls back on failure

Include pre/post deployment hooks.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 27: System Hardening Script",
        "prompt": """
Create a hardening script that:
1. Disables unnecessary services
2. Configures firewall basics
3. Sets secure kernel parameters
4. Hardens SSH configuration
5. Sets up fail2ban
6. Creates audit rules

Show before/after security score.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 28: Monitoring Integration",
        "prompt": """
Write a script that sends metrics to monitoring:
1. Collect system metrics (CPU, RAM, disk)
2. Format for Prometheus/InfluxDB
3. Include custom application metrics
4. Handle connection failures
5. Buffer data if endpoint down
6. Rotate buffer files

Use curl to POST metrics.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 29: Incident Response Helper",
        "prompt": """
Create an incident response script:
1. Captures system state (ps, netstat, etc)
2. Preserves logs with timestamps
3. Creates memory dump of suspicious process
4. Isolates system (optional)
5. Generates investigation report
6. Notifies security team

Include evidence chain tracking.
""",
        "params": {
            "max_tokens": 2048,
            "temperature": 0.1,
            "stream": False
        }
    },
    {
        "name": "Test 30: Infrastructure Report",
        "prompt": """
Build a comprehensive report generator:
1. System inventory (HW/SW)
2. Service status overview
3. Resource utilization trends
4. Security posture summary
5. Recent changes/updates
6. Recommendations

Output as HTML with summary dashboard.
""",
        "params": {
            "max_tokens": 2048,
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
        
        if completion_tokens > 0 and total_duration > 0:
            tokens_per_second = completion_tokens / total_duration
            print(f"Tokens per Second (T/s): {tokens_per_second:.2f}")
        else:
            print("Tokens per Second (T/s): N/A")

        # --- Response Quality Check ---
        full_content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Check for repetitive patterns (signs of model breakdown)
        lines = full_content.split('\n')
        if len(lines) > 5:
            # Check for excessive repetition
            unique_lines = len(set(lines))
            repetition_ratio = unique_lines / len(lines)
            if repetition_ratio < 0.5:
                print("⚠️  WARNING: High repetition detected in response")
        
        # Check for incomplete response
        if full_content.count('```') % 2 != 0:
            print("⚠️  WARNING: Unclosed code block detected")
            
        # Save Response
        filename_safe_name = name.lower().replace(":", "").replace(" ", "_")
        output_path = os.path.join("test_results", f"{filename_safe_name}_response.txt")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_content)
                
        print(f"\n--- 📜 Response Preview ---")
        print(full_content[:400] + "..." if len(full_content) > 400 else full_content)
        print(f"\n--- Full response saved to '{output_path}' ---")

        # --- Performance Summary ---
        with open(os.path.join("test_results", "performance_summary.txt"), "a", encoding="utf-8") as f:
            f.write(f"\n{name}\n")
            f.write(f"Duration: {total_duration:.2f}s | ")
            f.write(f"Tokens: {total_tokens} | ")
            if completion_tokens > 0 and total_duration > 0:
                f.write(f"T/s: {tokens_per_second:.2f}")
                if repetition_ratio < 0.5:
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
    print("--- Optimized Linux Agent Test Suite (30 Practical Challenges) ---")
    print(f"Targeting Server: {API_URL}")
    print(f"Model: SmallThinker-4BA0.6B-Instruct (4-bit quantized)")
    print(f"Total Tests: {len(TEST_CASES)}")
    print("\nTest Design Philosophy:")
    print("- Practical complexity that challenges without overwhelming")
    print("- Real tasks a Linux admin would automate")
    print("- Token limits calibrated for 6B model capabilities")
    print("- Quality checks for response coherence")
    
    # Clear performance summary file
    with open(os.path.join("test_results", "performance_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== OPTIMIZED LINUX AGENT TEST SUITE - PERFORMANCE SUMMARY ===\n")
        f.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80)
    
    successful_tests = 0
    failed_tests = 0
    
    for i, test in enumerate(TEST_CASES):
        print(f"\n\n[{i+1}/{len(TEST_CASES)}] Starting test...")
        if run_performance_test(test):
            successful_tests += 1
        else:
            failed_tests += 1
        
        # Small delay between tests
        time.sleep(1)
    
    # Final Summary
    print(f"\n{'='*80}")
    print(f"--- 📈 FINAL SUMMARY ---")
    print(f"Total Tests: {len(TEST_CASES)}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(successful_tests/len(TEST_CASES))*100:.1f}%")
    print(f"\n--- ✅ Test suite completed. Check 'test_results' directory for all outputs. ---")
    print(f"{'='*80}")
    
    # Append summary to file
    with open(os.path.join("test_results", "performance_summary.txt"), "a", encoding="utf-8") as f:
        f.write(f"\n\nFINAL SUMMARY\n")
        f.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Tests: {len(TEST_CASES)}\n")
        f.write(f"Successful: {successful_tests}\n")
        f.write(f"Failed: {failed_tests}\n")
        f.write(f"Success Rate: {(successful_tests/len(TEST_CASES))*100:.1f}%\n")