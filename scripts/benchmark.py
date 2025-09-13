#!/usr/bin/env python3
"""
Performance benchmark for llama.cpp inference optimization testing.
Focuses on measuring tokens/second with reliable prompts that models can complete.
"""

import json
import time
import sys
import argparse
import statistics
from datetime import datetime
from typing import Dict, List, Any
import requests
import subprocess

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
    def __init__(self, host: str = "localhost", port: int = 8001, timeout: int = 30):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.results = []
        
    def wait_for_api(self, max_attempts: int = 30) -> bool:
        """Wait for API to be ready."""
        print(f"Waiting for API at {self.base_url}...", end="")
        for _ in range(max_attempts):
            try:
                response = requests.get(f"{self.base_url}/v1/models", timeout=2)
                if response.status_code == 200:
                    print(" Ready!")
                    return True
            except:
                pass
            print(".", end="", flush=True)
            time.sleep(2)
        print(" Timeout!")
        return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        info = {
            "timestamp": datetime.now().isoformat(),
            "api_url": self.base_url
        }
        
        # Get model info
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                data = response.json()
                info["model"] = data.get("data", [{}])[0].get("id", "unknown")
        except:
            info["model"] = "unknown"
        
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
        except:
            pass
        
        return info
    
    def run_single_test(self, prompt_key: str, prompt_data: Dict, run_num: int) -> Dict[str, Any]:
        """Run a single benchmark test."""
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
                "error": "timeout"
            }
        except Exception as e:
            return {
                "prompt": prompt_key,
                "run": run_num,
                "success": False,
                "error": str(e)
            }
    
    def run_benchmark(self, num_runs: int = 5, prompts: List[str] = None):
        """Run the complete benchmark suite."""
        if prompts is None:
            prompts = list(BENCHMARK_PROMPTS.keys())
        
        print("\n" + "="*60)
        print("  LLAMA.CPP PERFORMANCE BENCHMARK")
        print("="*60)
        
        # Collect system info
        system_info = self.get_system_info()
        print(f"\nModel: {system_info.get('model', 'unknown')}")
        if "hugepages_total" in system_info:
            used = system_info["hugepages_total"] - system_info.get("hugepages_free", 0)
            size_mb = system_info.get("hugepage_size_kb", 0) / 1024
            print(f"Huge Pages: {used}/{system_info['hugepages_total']} ({size_mb:.0f} MB each)")
        print(f"Timestamp: {system_info['timestamp']}")
        print(f"Runs per prompt: {num_runs}")
        
        print("\n" + "-"*60)
        
        all_results = {
            "system_info": system_info,
            "prompts": {},
            "summary": {}
        }
        
        for prompt_key in prompts:
            if prompt_key not in BENCHMARK_PROMPTS:
                print(f"Unknown prompt: {prompt_key}")
                continue
                
            prompt_data = BENCHMARK_PROMPTS[prompt_key]
            print(f"\nTesting: {prompt_key}")
            print(f"  {prompt_data['description']}")
            print(f"  Progress: ", end="")
            
            results = []
            
            # Warmup run
            print("W", end="", flush=True)
            self.run_single_test(prompt_key, prompt_data, 0)
            
            # Actual runs
            for run in range(1, num_runs + 1):
                result = self.run_single_test(prompt_key, prompt_data, run)
                results.append(result)
                if result["success"]:
                    print(".", end="", flush=True)
                else:
                    print("X", end="", flush=True)
            
            # Calculate statistics for successful runs
            successful_runs = [r for r in results if r["success"]]
            if successful_runs:
                tokens_per_sec = [r["tokens_per_second"] for r in successful_runs]
                all_results["prompts"][prompt_key] = {
                    "prompt_text": prompt_data["prompt"],
                    "description": prompt_data["description"],
                    "results": results,
                    "stats": {
                        "success_rate": len(successful_runs) / len(results),
                        "avg_tokens_per_second": round(statistics.mean(tokens_per_sec), 2),
                        "median_tokens_per_second": round(statistics.median(tokens_per_sec), 2),
                        "min_tokens_per_second": round(min(tokens_per_sec), 2),
                        "max_tokens_per_second": round(max(tokens_per_sec), 2),
                        "stdev_tokens_per_second": round(statistics.stdev(tokens_per_sec), 2) if len(tokens_per_sec) > 1 else 0
                    }
                }
                print(f" Avg: {all_results['prompts'][prompt_key]['stats']['avg_tokens_per_second']} tok/s")
            else:
                all_results["prompts"][prompt_key] = {
                    "results": results,
                    "stats": {"success_rate": 0}
                }
                print(" All failed!")
        
        # Overall summary
        all_tokens_per_sec = []
        for prompt_key in all_results["prompts"]:
            stats = all_results["prompts"][prompt_key]["stats"]
            if "avg_tokens_per_second" in stats:
                all_tokens_per_sec.append(stats["avg_tokens_per_second"])
        
        if all_tokens_per_sec:
            all_results["summary"] = {
                "overall_avg_tokens_per_second": round(statistics.mean(all_tokens_per_sec), 2),
                "overall_median_tokens_per_second": round(statistics.median(all_tokens_per_sec), 2),
                "overall_min_tokens_per_second": round(min(all_tokens_per_sec), 2),
                "overall_max_tokens_per_second": round(max(all_tokens_per_sec), 2)
            }
        
        return all_results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print human-readable summary."""
        print("\n" + "="*60)
        print("  RESULTS SUMMARY")
        print("="*60)
        
        # Per-prompt results
        print("\nPer-Prompt Performance:")
        print("-"*40)
        print(f"{'Prompt':<20} {'Avg tok/s':>12} {'Success':>10}")
        print("-"*40)
        
        for prompt_key in results["prompts"]:
            stats = results["prompts"][prompt_key]["stats"]
            avg_tps = stats.get("avg_tokens_per_second", 0)
            success_rate = stats.get("success_rate", 0) * 100
            print(f"{prompt_key:<20} {avg_tps:>12.2f} {success_rate:>9.0f}%")
        
        # Overall summary
        if "summary" in results and results["summary"]:
            print("\n" + "-"*40)
            print("Overall Statistics:")
            print(f"  Average: {results['summary']['overall_avg_tokens_per_second']:.2f} tokens/sec")
            print(f"  Median:  {results['summary']['overall_median_tokens_per_second']:.2f} tokens/sec")
            print(f"  Min:     {results['summary']['overall_min_tokens_per_second']:.2f} tokens/sec")
            print(f"  Max:     {results['summary']['overall_max_tokens_per_second']:.2f} tokens/sec")
        
        print("="*60)
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
        return filename


def main():
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp inference performance")
    parser.add_argument("--host", default="localhost", help="API host")
    parser.add_argument("--port", type=int, default=8001, help="API port")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per prompt")
    parser.add_argument("--prompts", help="Comma-separated list of prompts to test")
    parser.add_argument("--output", help="Output JSON filename")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    # Parse prompts if specified
    prompts = None
    if args.prompts:
        prompts = args.prompts.split(",")
    
    # Create benchmark instance
    benchmark = LlamaBenchmark(host=args.host, port=args.port, timeout=args.timeout)
    
    # Wait for API
    if not benchmark.wait_for_api():
        print("Error: API not responding")
        sys.exit(1)
    
    # Run benchmark
    results = benchmark.run_benchmark(num_runs=args.runs, prompts=prompts)
    
    # Print summary
    benchmark.print_summary(results)
    
    # Save results
    benchmark.save_results(results, args.output)


if __name__ == "__main__":
    main()