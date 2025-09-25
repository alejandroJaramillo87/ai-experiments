#!/usr/bin/env python3
"""
Llama.cpp inference performance benchmark utility.
Measures tokens per second across different optimization scenarios.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests

# Status indicators
STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_ERROR = "ERROR"

# Exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_INVALID_USAGE = 2

class BenchmarkError(Exception):
    """Raised when benchmark execution fails."""
    pass

class APIConnectionError(Exception):
    """Raised when API connection fails."""
    pass

class SystemMetricsError(Exception):
    """Raised when system metrics collection fails."""
    pass

# Test prompts designed to stress different optimization aspects
BENCHMARK_PROMPTS = {
    "memory_sequential": {
        "description": "Sequential memory access pattern",
        "prompt": "Count from 1 to 20, writing each number on a new line:",
        "max_tokens": 100,
        "expected_pattern": "numbers"
    },
    "cache_loops": {
        "description": "CPU cache efficiency with repetitive patterns",
        "prompt": "Write the word 'test' exactly 15 times separated by spaces:",
        "max_tokens": 80,
        "expected_pattern": "repetition"
    },
    "compute_arithmetic": {
        "description": "Basic compute operations",
        "prompt": "Calculate: 2+2=, 5+5=, 10+10=, 20+20=, 50+50=",
        "max_tokens": 50,
        "expected_pattern": "calculations"
    },
    "memory_structured": {
        "description": "Structured data generation for memory allocation",
        "prompt": "Create a JSON object with three fields: name (string), age (number), active (boolean). Use simple values:",
        "max_tokens": 100,
        "expected_pattern": "json"
    },
    "throughput_sustained": {
        "description": "Sustained generation for throughput testing",
        "prompt": "List 10 common programming languages, one per line:",
        "max_tokens": 150,
        "expected_pattern": "list"
    },
    "memory_bandwidth": {
        "description": "Large memory bandwidth test",
        "prompt": "Generate a comma-separated list of the first 30 even numbers:",
        "max_tokens": 200,
        "expected_pattern": "sequence"
    }
}


class LlamaBenchmark:
    def __init__(self, host: str = "localhost", port: int = 8001, timeout: int = 30, label: Optional[str] = None):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.results = []
        self.label = label
        
    def get_gpu_config(self) -> Dict[str, Any]:
        """Collect GPU configuration information.

        Returns:
            Dictionary containing GPU configuration parameters:
            - persistence_mode: NVIDIA persistence mode status
            - compute_mode: GPU compute mode setting
            - power_limit_w: Power limit in watts
            - gpu_clock_mhz: GPU clock frequency in MHz
            - mem_clock_mhz: Memory clock frequency in MHz
            - pcie_gen: PCIe generation
            - pcie_width: PCIe lane width

        Raises:
            SystemMetricsError: If GPU configuration cannot be retrieved
        """
        config = {}
        try:
            result = subprocess.run([
                "nvidia-smi",
                "--query-gpu=persistence_mode,compute_mode,power.limit,clocks.applications.graphics,clocks.applications.memory,pcie.link.gen.current,pcie.link.width.current",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                values = result.stdout.strip().split(", ")
                config = {
                    "persistence_mode": values[0] if len(values) > 0 else None,
                    "compute_mode": values[1] if len(values) > 1 else None,
                    "power_limit_w": float(values[2]) if len(values) > 2 and values[2] != "[N/A]" else None,
                    "gpu_clock_mhz": float(values[3]) if len(values) > 3 and values[3] != "[N/A]" else None,
                    "mem_clock_mhz": float(values[4]) if len(values) > 4 and values[4] != "[N/A]" else None,
                    "pcie_gen": int(values[5]) if len(values) > 5 and values[5] != "[N/A]" else None,
                    "pcie_width": int(values[6]) if len(values) > 6 and values[6] != "[N/A]" else None
                }
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            config["error"] = str(e)
        return config

    def get_gpu_runtime_metrics(self) -> Dict[str, Any]:
        """Get current GPU runtime metrics.

        Returns:
            Dictionary containing runtime GPU metrics:
            - gpu_utilization_percent: GPU utilization percentage
            - vram_used_mb: VRAM usage in megabytes
            - vram_total_mb: Total VRAM in megabytes
            - temperature_c: GPU temperature in Celsius
            - power_draw_w: Current power draw in watts

        Raises:
            SystemMetricsError: If runtime metrics cannot be retrieved
        """
        metrics = {}
        try:
            result = subprocess.run([
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                values = result.stdout.strip().split(", ")
                metrics = {
                    "gpu_utilization_percent": float(values[0]) if len(values) > 0 else None,
                    "vram_used_mb": float(values[1]) if len(values) > 1 else None,
                    "vram_total_mb": float(values[2]) if len(values) > 2 else None,
                    "temperature_c": float(values[3]) if len(values) > 3 else None,
                    "power_draw_w": float(values[4]) if len(values) > 4 else None
                }
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            metrics["error"] = str(e)
        return metrics

    def wait_for_api(self, max_attempts: int = 30) -> bool:
        """Wait for API to become available.

        Args:
            max_attempts: Maximum number of connection attempts

        Returns:
            True if API responds successfully, False if timeout

        Raises:
            APIConnectionError: If API connection fails after max attempts
        """
        print(f"API connection check: {self.base_url}")

        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(f"{self.base_url}/v1/models", timeout=2)
                if response.status_code == 200:
                    print(f"API status: {STATUS_OK} (connected after {attempt} attempts)")
                    return True
            except (requests.RequestException, requests.Timeout):
                pass

            if attempt % 10 == 0:  # Progress update every 10 attempts
                print(f"API connection: {STATUS_WARN} (attempt {attempt}/{max_attempts})")

            time.sleep(2)

        print(f"API status: {STATUS_ERROR} (timeout after {max_attempts} attempts)")
        return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Collect system configuration information.

        Returns:
            Dictionary containing system information:
            - timestamp: Benchmark execution timestamp
            - label: Optional benchmark label
            - api_url: API endpoint URL
            - model: Model identifier
            - gpu_config: GPU configuration parameters
            - hugepages_total: Total huge pages configured
            - hugepages_free: Available huge pages
            - hugepage_size_kb: Huge page size in KB

        Raises:
            SystemMetricsError: If system information cannot be collected
        """
        info = {
            "timestamp": datetime.now().isoformat(),
            "label": self.label,
            "api_url": self.base_url
        }
        
        # Get model info
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                info["model"] = data.get("data", [{}])[0].get("id", "unknown")
        except (requests.RequestException, requests.Timeout, KeyError):
            info["model"] = "unknown"

        # Get GPU configuration
        gpu_config = self.get_gpu_config()
        if gpu_config:
            info["gpu_config"] = gpu_config

        # Get memory info
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("HugePages_Total:"):
                        info["hugepages_total"] = int(line.split()[1])
                    elif line.startswith("HugePages_Free:"):
                        info["hugepages_free"] = int(line.split()[1])
                    elif line.startswith("Hugepagesize:"):
                        info["hugepage_size_kb"] = int(line.split()[1])
        except (FileNotFoundError, ValueError, IndexError):
            pass
        
        return info
    
    def run_single_test(self, prompt_key: str, prompt_data: Dict[str, Any], run_num: int) -> Dict[str, Any]:
        """Execute a single benchmark test.

        Args:
            prompt_key: Identifier for the test prompt
            prompt_data: Dictionary containing prompt configuration
            run_num: Run number for tracking

        Returns:
            Dictionary containing test results:
            - prompt: Test prompt identifier
            - run: Run number
            - success: Whether test completed successfully
            - total_time: Total execution time in seconds
            - tokens_per_second: Calculated throughput
            - response: Generated response text

        Raises:
            BenchmarkError: If test execution fails
        """
        request_data = {
            "model": "model",
            "messages": [{"role": "user", "content": prompt_data["prompt"]}],
            "max_tokens": prompt_data["max_tokens"],
            "temperature": 0.1,
            "stream": False
        }
        
        try:
            start_time = time.perf_counter()
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=request_data,
                timeout=self.timeout
            )
            end_time = time.perf_counter()
            
            if response.status_code != 200:
                return {
                    "prompt": prompt_key,
                    "run": run_num,
                    "success": False,
                    "error": f"HTTP {response.status_code}"
                }
            
            data = response.json()
            total_time = end_time - start_time
            
            # Extract metrics
            usage = data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            
            tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
            
            response_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            return {
                "prompt": prompt_key,
                "run": run_num,
                "success": True,
                "total_time": round(total_time, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tokens_per_second": round(tokens_per_second, 2),
                "response_length": len(response_content),
                "response": response_content
            }
            
        except requests.Timeout:
            return {
                "prompt": prompt_key,
                "run": run_num,
                "success": False,
                "error": "request timeout"
            }
        except (requests.RequestException, KeyError, ValueError) as e:
            return {
                "prompt": prompt_key,
                "run": run_num,
                "success": False,
                "error": str(e)
            }
    
    def run_benchmark(self, num_runs: int = 5, prompts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute complete benchmark suite.

        Args:
            num_runs: Number of test runs per prompt
            prompts: List of prompt keys to test (all if None)

        Returns:
            Dictionary containing complete benchmark results:
            - system_info: System configuration information
            - prompts: Per-prompt results and statistics
            - summary: Overall performance summary

        Raises:
            BenchmarkError: If benchmark execution fails
        """
        if prompts is None:
            prompts = list(BENCHMARK_PROMPTS.keys())
        
        print("Llama.cpp performance benchmark")
        print()
        
        # Collect system info
        system_info = self.get_system_info()
        print(f"Model: {system_info.get('model', 'unknown')}")
        if "hugepages_total" in system_info:
            used = system_info["hugepages_total"] - system_info.get("hugepages_free", 0)
            size_mb = system_info.get("hugepage_size_kb", 0) / 1024
            print(f"Huge pages: {STATUS_OK} ({used}/{system_info['hugepages_total']} pages, {size_mb:.0f}MB each)")
        print(f"Benchmark configuration: {num_runs} runs per prompt")
        print(f"Timestamp: {system_info['timestamp']}")
        print()
        
        all_results = {
            "system_info": system_info,
            "prompts": {},
            "summary": {}
        }
        
        for prompt_key in prompts:
            if prompt_key not in BENCHMARK_PROMPTS:
                print(f"Prompt validation: {STATUS_ERROR} (unknown prompt: {prompt_key})")
                continue

            prompt_data = BENCHMARK_PROMPTS[prompt_key]
            print(f"Testing prompt: {prompt_key}")
            print(f"  Description: {prompt_data['description']}")

            results = []

            # Warmup run
            print(f"  Warmup: ", end="")
            warmup_result = self.run_single_test(prompt_key, prompt_data, 0)
            warmup_status = STATUS_OK if warmup_result["success"] else STATUS_WARN
            print(f"{warmup_status}")

            # Actual runs
            successful_runs = 0
            for run in range(1, num_runs + 1):
                result = self.run_single_test(prompt_key, prompt_data, run)
                results.append(result)
                if result["success"]:
                    successful_runs += 1

            run_status = STATUS_OK if successful_runs == num_runs else STATUS_WARN if successful_runs > 0 else STATUS_ERROR
            print(f"  Test runs: {run_status} ({successful_runs}/{num_runs} successful)")
            
            # Calculate statistics for successful runs
            successful_test_results = [r for r in results if r["success"]]
            if successful_test_results:
                tokens_per_sec = [r["tokens_per_second"] for r in successful_test_results]
                avg_tokens_per_sec = round(statistics.mean(tokens_per_sec), 2)
                all_results["prompts"][prompt_key] = {
                    "prompt_text": prompt_data["prompt"],
                    "description": prompt_data["description"],
                    "results": results,
                    "stats": {
                        "success_rate": len(successful_test_results) / len(results),
                        "avg_tokens_per_second": avg_tokens_per_sec,
                        "median_tokens_per_second": round(statistics.median(tokens_per_sec), 2),
                        "min_tokens_per_second": round(min(tokens_per_sec), 2),
                        "max_tokens_per_second": round(max(tokens_per_sec), 2),
                        "stdev_tokens_per_second": round(statistics.stdev(tokens_per_sec), 2) if len(tokens_per_sec) > 1 else 0
                    }
                }
                print(f"  Performance: {STATUS_OK} ({avg_tokens_per_sec} tokens/sec average)")
            else:
                all_results["prompts"][prompt_key] = {
                    "prompt_text": prompt_data["prompt"],
                    "description": prompt_data["description"],
                    "results": results,
                    "stats": {"success_rate": 0}
                }
                print(f"  Performance: {STATUS_ERROR} (all test runs failed)")
        
        # Overall summary
        all_tokens_per_sec = []
        for prompt_key in all_results["prompts"]:
            stats = all_results["prompts"][prompt_key]["stats"]
            if "avg_tokens_per_second" in stats:
                all_tokens_per_sec.append(stats["avg_tokens_per_second"])

        if all_tokens_per_sec:
            overall_avg = round(statistics.mean(all_tokens_per_sec), 2)
            all_results["summary"] = {
                "overall_avg_tokens_per_second": overall_avg,
                "overall_median_tokens_per_second": round(statistics.median(all_tokens_per_sec), 2),
                "overall_min_tokens_per_second": round(min(all_tokens_per_sec), 2),
                "overall_max_tokens_per_second": round(max(all_tokens_per_sec), 2)
            }

            print()
            print(f"Benchmark summary: {STATUS_OK} ({overall_avg} tokens/sec overall average)")

            # Add GPU runtime metrics
            gpu_metrics = self.get_gpu_runtime_metrics()
            if gpu_metrics and not gpu_metrics.get("error"):
                all_results["summary"]["gpu_runtime"] = gpu_metrics
                print(f"GPU metrics: {STATUS_OK} (collected)")
            elif gpu_metrics.get("error"):
                print(f"GPU metrics: {STATUS_WARN} (collection failed)")
        else:
            print(f"Benchmark summary: {STATUS_ERROR} (no successful test runs)")
        
        return all_results
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print comprehensive benchmark results summary.

        Args:
            results: Complete benchmark results dictionary
        """
        print()
        print("Benchmark results summary:")
        print()

        # Print label if present
        if "system_info" in results and results["system_info"].get("label"):
            print(f"Benchmark label: {results['system_info']['label']}")
        
        # Per-prompt results
        print("Per-prompt performance:")
        print(f"{'Prompt':<20} {'Tokens/sec':>12} {'Success':>10}")
        
        for prompt_key in results["prompts"]:
            stats = results["prompts"][prompt_key]["stats"]
            avg_tps = stats.get("avg_tokens_per_second", 0)
            success_rate = stats.get("success_rate", 0) * 100
            print(f"{prompt_key:<20} {avg_tps:>12.2f} {success_rate:>9.0f}%")
        
        # Overall summary
        if "summary" in results and results["summary"]:
            print()
            print("Overall performance statistics:")
            print(f"Average: {results['summary']['overall_avg_tokens_per_second']:.2f} tokens/sec")
            print(f"Median: {results['summary']['overall_median_tokens_per_second']:.2f} tokens/sec")
            print(f"Range: {results['summary']['overall_min_tokens_per_second']:.2f}-{results['summary']['overall_max_tokens_per_second']:.2f} tokens/sec")

            # GPU runtime metrics
            if "gpu_runtime" in results["summary"]:
                gpu = results["summary"]["gpu_runtime"]
                print()
                print("GPU runtime metrics:")
                if "gpu_utilization_percent" in gpu:
                    print(f"GPU utilization: {gpu['gpu_utilization_percent']:.1f}%")
                if "vram_used_mb" in gpu and "vram_total_mb" in gpu:
                    print(f"VRAM usage: {gpu['vram_used_mb']:.0f}/{gpu['vram_total_mb']:.0f}MB")
                if "temperature_c" in gpu:
                    print(f"Temperature: {gpu['temperature_c']:.0f}C")
                if "power_draw_w" in gpu:
                    print(f"Power draw: {gpu['power_draw_w']:.0f}W")

        # GPU configuration
        if "system_info" in results and "gpu_config" in results["system_info"]:
            gpu_cfg = results["system_info"]["gpu_config"]
            if gpu_cfg and not gpu_cfg.get("error"):
                print()
                print("GPU configuration:")
                if "persistence_mode" in gpu_cfg:
                    print(f"Persistence mode: {gpu_cfg['persistence_mode']}")
                if "compute_mode" in gpu_cfg:
                    print(f"Compute mode: {gpu_cfg['compute_mode']}")
                if "power_limit_w" in gpu_cfg:
                    print(f"Power limit: {gpu_cfg['power_limit_w']:.0f}W")
                if "gpu_clock_mhz" in gpu_cfg and gpu_cfg["gpu_clock_mhz"]:
                    print(f"GPU clock: {gpu_cfg['gpu_clock_mhz']:.0f}MHz")
                if "mem_clock_mhz" in gpu_cfg and gpu_cfg["mem_clock_mhz"]:
                    print(f"Memory clock: {gpu_cfg['mem_clock_mhz']:.0f}MHz")
                if "pcie_gen" in gpu_cfg and "pcie_width" in gpu_cfg:
                    print(f"PCIe link: Gen{gpu_cfg['pcie_gen']} x{gpu_cfg['pcie_width']}")
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save benchmark results to JSON file.

        Args:
            results: Complete benchmark results dictionary
            filename: Output filename (auto-generated if None)

        Returns:
            Path to saved results file

        Raises:
            BenchmarkError: If file cannot be saved
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        try:
            # Ensure parent directory exists
            filepath = Path(filename)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results: {STATUS_OK} (saved to {filename})")
            return filename
        except (IOError, OSError) as e:
            raise BenchmarkError(f"Failed to save results to {filename}: {e}")


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Llama.cpp inference performance benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --label baseline
  python benchmark.py --label optimized --runs 10 --output baseline_results.json
  python benchmark.py --prompts memory_sequential,compute_arithmetic --port 8004
        """
    )

    parser.add_argument(
        "--host",
        default="localhost",
        help="API host address (default: localhost)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="API port number (default: 8001)"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of test runs per prompt (default: 5)"
    )

    parser.add_argument(
        "--prompts",
        help="Comma-separated list of prompts to test (default: all)"
    )

    parser.add_argument(
        "--output",
        help="Output JSON filename (default: auto-generated)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)"
    )

    parser.add_argument(
        "--label",
        help="Benchmark label (e.g., baseline, optimized)"
    )

    return parser

def main() -> int:
    """Main function to execute benchmark.

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    parser = create_parser()
    args = parser.parse_args()

    try:
        # Parse prompts if specified
        prompts = None
        if args.prompts:
            prompts = [p.strip() for p in args.prompts.split(",")]

        # Create benchmark instance
        benchmark = LlamaBenchmark(
            host=args.host,
            port=args.port,
            timeout=args.timeout,
            label=args.label
        )

        # Wait for API
        if not benchmark.wait_for_api():
            print(f"API connection: {STATUS_ERROR} (not responding)", file=sys.stderr)
            print("  Solution: Verify API service is running and accessible", file=sys.stderr)
            return EXIT_FAILURE

        # Run benchmark
        results = benchmark.run_benchmark(num_runs=args.runs, prompts=prompts)

        # Print summary
        benchmark.print_summary(results)

        # Save results
        benchmark.save_results(results, args.output)

        return EXIT_SUCCESS

    except BenchmarkError as e:
        print(f"Benchmark execution: {STATUS_ERROR} ({e})", file=sys.stderr)
        return EXIT_FAILURE
    except APIConnectionError as e:
        print(f"API connection: {STATUS_ERROR} ({e})", file=sys.stderr)
        return EXIT_FAILURE
    except Exception as e:
        print(f"Unexpected error: {STATUS_ERROR} ({e})", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())