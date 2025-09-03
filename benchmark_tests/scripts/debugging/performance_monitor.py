#!/usr/bin/env python3
"""
Performance Monitoring Script for Concurrent Execution Debug

Monitors benchmark_runner.py execution to identify performance bottlenecks
causing 300s+ timeouts in concurrent functional tests.
"""

import time
import psutil
import subprocess
import sys
import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import threading
import os


@dataclass
class PerformanceMetrics:
    """Performance metrics collected during test execution"""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    io_read_bytes: int
    io_write_bytes: int
    network_sent: int
    network_recv: int
    process_count: int
    thread_count: int


class PerformanceMonitor:
    """Monitors system performance during benchmark execution"""
    
    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self.metrics: List[PerformanceMetrics] = []
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None
        
    def start_monitoring(self):
        """Start performance monitoring in background thread"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.start_time = time.time()
        self.metrics.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"🔍 Performance monitoring started (interval: {self.sample_interval}s)")
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.monitoring:
            return
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        duration = time.time() - self.start_time if self.start_time else 0
        print(f"🔍 Performance monitoring stopped (duration: {duration:.1f}s, samples: {len(self.metrics)})")
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                io_counters = psutil.disk_io_counters()
                network = psutil.net_io_counters()
                
                # Count processes and threads
                process_count = len(psutil.pids())
                thread_count = sum(p.num_threads() for p in psutil.process_iter(['num_threads']) 
                                 if p.info['num_threads'])
                
                # Create metrics sample
                metrics = PerformanceMetrics(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_mb=memory.used / (1024 * 1024),
                    memory_percent=memory.percent,
                    io_read_bytes=io_counters.read_bytes if io_counters else 0,
                    io_write_bytes=io_counters.write_bytes if io_counters else 0,
                    network_sent=network.bytes_sent if network else 0,
                    network_recv=network.bytes_recv if network else 0,
                    process_count=process_count,
                    thread_count=thread_count
                )
                
                self.metrics.append(metrics)
                
            except Exception as e:
                print(f"⚠️  Monitoring error: {e}")
                
            time.sleep(self.sample_interval)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from collected metrics"""
        if not self.metrics:
            return {}
            
        # Calculate basic statistics
        cpu_values = [m.cpu_percent for m in self.metrics]
        memory_values = [m.memory_mb for m in self.metrics]
        
        return {
            "duration_seconds": self.metrics[-1].timestamp - self.metrics[0].timestamp,
            "sample_count": len(self.metrics),
            "cpu_percent": {
                "avg": sum(cpu_values) / len(cpu_values),
                "max": max(cpu_values),
                "min": min(cpu_values)
            },
            "memory_mb": {
                "avg": sum(memory_values) / len(memory_values),
                "max": max(memory_values),
                "min": min(memory_values)
            },
            "final_metrics": asdict(self.metrics[-1]) if self.metrics else None
        }
    
    def save_detailed_report(self, filepath: str):
        """Save detailed performance report to file"""
        report = {
            "monitoring_info": {
                "sample_interval": self.sample_interval,
                "start_time": self.start_time,
                "end_time": time.time(),
                "sample_count": len(self.metrics)
            },
            "summary_stats": self.get_summary_stats(),
            "detailed_metrics": [asdict(m) for m in self.metrics]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Detailed performance report saved: {filepath}")


def run_benchmark_with_monitoring(args: List[str], timeout: int = 300) -> Dict[str, Any]:
    """Run benchmark_runner.py with performance monitoring"""
    print(f"🚀 Starting monitored benchmark execution")
    print(f"   Command: python3 benchmark_runner.py {' '.join(args)}")
    print(f"   Timeout: {timeout}s")
    
    monitor = PerformanceMonitor(sample_interval=2.0)  # 2-second intervals
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Run benchmark command
        cmd = ["python3", "benchmark_runner.py"] + args
        start_time = time.time()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/home/alejandro/workspace/ai-workstation/benchmark_tests"
        )
        
        execution_time = time.time() - start_time
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Collect results
        performance_summary = monitor.get_summary_stats()
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "execution_time": execution_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "performance": performance_summary,
            "monitor": monitor  # For detailed analysis
        }
        
    except subprocess.TimeoutExpired:
        monitor.stop_monitoring()
        execution_time = time.time() - start_time
        
        return {
            "success": False,
            "returncode": -1,
            "execution_time": execution_time,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "performance": monitor.get_summary_stats(),
            "timeout": True,
            "monitor": monitor
        }
        
    except Exception as e:
        monitor.stop_monitoring()
        return {
            "success": False,
            "returncode": -2,
            "execution_time": 0,
            "stdout": "",
            "stderr": f"Execution error: {str(e)}",
            "performance": {},
            "error": str(e)
        }


def analyze_concurrent_timeout_issue():
    """Analyze the concurrent execution timeout issue step by step"""
    print("=" * 80)
    print("🔧 CONCURRENT EXECUTION TIMEOUT ANALYSIS")
    print("=" * 80)
    
    # Test scenarios to analyze
    scenarios = [
        {
            "name": "Single Basic Test (Baseline)",
            "args": ["--test-type", "base", "--test-id", "basic_01", 
                    "--endpoint", "http://localhost:8004", "--model", "llama3"],
            "timeout": 60,
            "expected_time": "< 30s"
        },
        {
            "name": "Two Sequential Tests", 
            "args": ["--test-type", "base", "--test-id", "basic_01,basic_02",
                    "--mode", "sequential", "--endpoint", "http://localhost:8004", "--model", "llama3"],
            "timeout": 120,
            "expected_time": "< 60s"
        },
        {
            "name": "Two Concurrent Tests (2 workers)",
            "args": ["--test-type", "base", "--test-id", "basic_01,basic_02", 
                    "--mode", "concurrent", "--workers", "2",
                    "--endpoint", "http://localhost:8004", "--model", "llama3"],
            "timeout": 180,
            "expected_time": "< 45s (should be faster than sequential)"
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n📋 Testing: {scenario['name']}")
        print(f"   Expected: {scenario['expected_time']}")
        print("-" * 60)
        
        result = run_benchmark_with_monitoring(scenario['args'], scenario['timeout'])
        results[scenario['name']] = result
        
        # Print immediate results
        if result['success']:
            print(f"✅ SUCCESS: {result['execution_time']:.1f}s")
        elif result.get('timeout'):
            print(f"⏰ TIMEOUT: {result['execution_time']:.1f}s (limit: {scenario['timeout']}s)")
        else:
            print(f"❌ FAILED: {result.get('stderr', 'Unknown error')}")
        
        # Print performance summary
        if result['performance']:
            perf = result['performance']
            print(f"   💻 CPU avg/max: {perf['cpu_percent']['avg']:.1f}%/{perf['cpu_percent']['max']:.1f}%")
            print(f"   🧠 Memory avg/max: {perf['memory_mb']['avg']:.0f}MB/{perf['memory_mb']['max']:.0f}MB")
        
        # Save detailed report for analysis
        if 'monitor' in result:
            report_file = f"/tmp/performance_report_{scenario['name'].lower().replace(' ', '_')}.json"
            result['monitor'].save_detailed_report(report_file)
        
        print()
    
    # Summary analysis
    print("=" * 80)
    print("📊 PERFORMANCE ANALYSIS SUMMARY")
    print("=" * 80)
    
    for name, result in results.items():
        status = "✅ SUCCESS" if result['success'] else ("⏰ TIMEOUT" if result.get('timeout') else "❌ FAILED")
        print(f"{status:12} {name:25} {result['execution_time']:6.1f}s")
    
    # Identify the issue
    if results.get("Single Basic Test (Baseline)", {}).get('success'):
        if results.get("Two Concurrent Tests (2 workers)", {}).get('timeout'):
            print("\n🚨 ISSUE IDENTIFIED: Concurrent execution is causing timeouts")
            print("   Single test works, but concurrent execution fails")
            print("   This suggests a concurrency implementation problem")
        else:
            print("\n✅ All tests passing - timeout issue may be intermittent")
    else:
        print("\n🚨 ISSUE: Even basic single test is failing")
        print("   This suggests a fundamental server or configuration problem")
    
    return results


if __name__ == "__main__":
    # Run the analysis
    analyze_concurrent_timeout_issue()