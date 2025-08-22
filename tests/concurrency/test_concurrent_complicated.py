# test_concurrent_enhanced.py

import asyncio
import aiohttp
import json
import time
import os
import random
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import statistics

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8004/v1/chat/completions"  # Change port: 8005 for vLLM, 8004 for llama.cpp
HEADERS = {"Content-Type": "application/json"}
OUTPUT_DIR = "test_results_concurrency_enhanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- HELPER FUNCTIONS ---

def calculate_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Calculate latency percentiles."""
    if not latencies:
        return {"p50": 0, "p90": 0, "p99": 0, "p99.9": 0, "mean": 0, "min": 0, "max": 0}
    
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    
    return {
        "p50": sorted_latencies[int(n * 0.50)] if n > 0 else 0,
        "p90": sorted_latencies[int(n * 0.90)] if n > 0 else 0,
        "p99": sorted_latencies[int(n * 0.99)] if n > 0 else 0,
        "p99.9": sorted_latencies[int(n * 0.999)] if n >= 1000 else sorted_latencies[-1] if n > 0 else 0,
        "mean": statistics.mean(latencies),
        "min": min(latencies),
        "max": max(latencies)
    }

async def send_request_streaming(session, payload, index):
    """Send request with streaming to measure time-to-first-token."""
    payload["stream"] = True
    start_time = time.time()
    time_to_first_token = None
    tokens_received = []
    
    try:
        async with session.post(API_URL, headers=HEADERS, json=payload, timeout=600) as response:
            if response.status != 200:
                return {
                    "status": "failed",
                    "index": index,
                    "error": f"HTTP Status {response.status}"
                }
            
            completion_text = ""
            token_count = 0
            
            async for line in response.content:
                if time_to_first_token is None:
                    time_to_first_token = time.time() - start_time
                
                line = line.decode('utf-8').strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                chunk = delta["content"]
                                completion_text += chunk
                                token_count += len(chunk.split())  # Rough estimate
                                tokens_received.append(time.time() - start_time)
                    except json.JSONDecodeError:
                        continue
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            return {
                "status": "success",
                "index": index,
                "duration": total_duration,
                "time_to_first_token": time_to_first_token or total_duration,
                "completion_tokens": token_count,
                "token_latencies": tokens_received,
                "throughput_per_token": token_count / total_duration if total_duration > 0 else 0
            }
            
    except Exception as e:
        return {"status": "failed", "index": index, "error": str(e)}

async def send_request(session, payload, index, use_streaming=False):
    """Send a single API request and return performance data."""
    if use_streaming:
        return await send_request_streaming(session, payload, index)
    
    start_time = time.time()
    try:
        async with session.post(API_URL, headers=HEADERS, json=payload, timeout=600) as response:
            response_data = await response.json()
            end_time = time.time()

            if response.status != 200:
                return {
                    "status": "failed",
                    "index": index,
                    "error": response_data.get("message", f"HTTP Status {response.status}")
                }

            usage = response_data.get("usage", {})
            total_duration = end_time - start_time
            completion_tokens = usage.get("completion_tokens", 0)
            
            return {
                "status": "success",
                "index": index,
                "duration": total_duration,
                "completion_tokens": completion_tokens,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "tokens_per_second": completion_tokens / total_duration if total_duration > 0 else 0
            }
    except Exception as e:
        return {"status": "failed", "index": index, "error": str(e)}

# --- TEST SCENARIOS ---

async def run_saturation_test():
    """Test different concurrency levels to find the saturation point."""
    print(f"\n{'='*80}\n--- 🔍 SATURATION TEST ---\n{'='*80}")
    print("Finding the optimal concurrency level by testing various loads...")
    
    concurrency_levels = [1, 2, 4, 8, 16, 32, 64, 128]
    results = []
    
    prompt = "Explain quantum computing in simple terms."
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.5
    }
    
    for level in concurrency_levels:
        print(f"\nTesting concurrency level: {level}")
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            tasks = [send_request(session, payload, i) for i in range(level)]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        successful = [r for r in responses if isinstance(r, dict) and r.get("status") == "success"]
        latencies = [r["duration"] for r in successful]
        total_tokens = sum(r.get("completion_tokens", 0) for r in successful)
        
        if latencies:
            percentiles = calculate_percentiles(latencies)
            aggregate_tps = total_tokens / duration if duration > 0 else 0
            
            result = {
                "concurrency": level,
                "duration": duration,
                "success_rate": len(successful) / level,
                "aggregate_tps": aggregate_tps,
                "throughput_per_request": aggregate_tps / level,
                **percentiles
            }
            results.append(result)
            
            print(f"  Success Rate: {result['success_rate']:.1%}")
            print(f"  Aggregate T/s: {aggregate_tps:.2f}")
            print(f"  P50 Latency: {percentiles['p50']:.2f}s")
            print(f"  P99 Latency: {percentiles['p99']:.2f}s")
    
    # Save results
    with open(os.path.join(OUTPUT_DIR, "saturation_test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results

async def run_time_to_first_token_test():
    """Test time-to-first-token under concurrent load."""
    print(f"\n{'='*80}\n--- ⏱️ TIME-TO-FIRST-TOKEN TEST ---\n{'='*80}")
    print("Testing streaming response times under concurrent load...")
    
    num_requests = 20
    prompt = "Write a detailed explanation of machine learning."
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.5
    }
    
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, payload, i, use_streaming=True) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
    ttft_values = [r["time_to_first_token"] for r in successful if "time_to_first_token" in r]
    
    if ttft_values:
        ttft_stats = calculate_percentiles(ttft_values)
        print(f"\n--- Time-to-First-Token Statistics ---")
        print(f"  Mean: {ttft_stats['mean']:.3f}s")
        print(f"  P50: {ttft_stats['p50']:.3f}s")
        print(f"  P90: {ttft_stats['p90']:.3f}s")
        print(f"  P99: {ttft_stats['p99']:.3f}s")
        
        # Save results
        with open(os.path.join(OUTPUT_DIR, "ttft_results.json"), "w") as f:
            json.dump({"stats": ttft_stats, "raw_values": ttft_values}, f, indent=2)
    
    return ttft_stats if ttft_values else None

async def run_staggered_arrival_test():
    """Test with realistic arrival patterns (Poisson distribution)."""
    print(f"\n{'='*80}\n--- 📊 STAGGERED ARRIVAL TEST ---\n{'='*80}")
    print("Simulating realistic request arrival patterns...")
    
    num_requests = 100
    arrival_rate = 10  # requests per second
    duration_seconds = num_requests / arrival_rate
    
    prompt = "What are the benefits of renewable energy?"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.5
    }
    
    async def staggered_sender(session, results_list):
        """Send requests with Poisson arrival pattern."""
        for i in range(num_requests):
            # Schedule request
            task = asyncio.create_task(send_request(session, payload, i))
            results_list.append(task)
            
            # Wait for next arrival (exponential distribution for Poisson process)
            inter_arrival_time = random.expovariate(arrival_rate)
            await asyncio.sleep(inter_arrival_time)
    
    results_list = []
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        # Start sending with staggered arrivals
        sender_task = asyncio.create_task(staggered_sender(session, results_list))
        
        # Wait for all requests to complete
        await sender_task
        results = await asyncio.gather(*results_list, return_exceptions=True)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    successful = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
    latencies = [r["duration"] for r in successful]
    total_tokens = sum(r.get("completion_tokens", 0) for r in successful)
    
    if latencies:
        percentiles = calculate_percentiles(latencies)
        aggregate_tps = total_tokens / total_duration if total_duration > 0 else 0
        
        print(f"\n--- Staggered Arrival Results ---")
        print(f"  Total Duration: {total_duration:.2f}s")
        print(f"  Success Rate: {len(successful)/num_requests:.1%}")
        print(f"  Aggregate T/s: {aggregate_tps:.2f}")
        print(f"  P50 Latency: {percentiles['p50']:.2f}s")
        print(f"  P99 Latency: {percentiles['p99']:.2f}s")
        
        # Save results
        with open(os.path.join(OUTPUT_DIR, "staggered_arrival_results.json"), "w") as f:
            json.dump({
                "duration": total_duration,
                "success_rate": len(successful)/num_requests,
                "aggregate_tps": aggregate_tps,
                "percentiles": percentiles
            }, f, indent=2)

async def run_long_context_stress_test():
    """Test with long context under concurrent load."""
    print(f"\n{'='*80}\n--- 📚 LONG CONTEXT STRESS TEST ---\n{'='*80}")
    print("Testing performance with long input contexts...")
    
    num_requests = 10
    
    # Create a long context prompt (~4000 tokens)
    long_text = "The history of artificial intelligence is fascinating. " * 500
    prompt = f"Summarize the following text in 3 paragraphs:\n\n{long_text}"
    
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.3
    }
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, payload, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    successful = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
    latencies = [r["duration"] for r in successful]
    total_tokens = sum(r.get("completion_tokens", 0) for r in successful)
    
    if latencies:
        percentiles = calculate_percentiles(latencies)
        aggregate_tps = total_tokens / duration if duration > 0 else 0
        
        print(f"\n--- Long Context Results ---")
        print(f"  Requests: {num_requests}")
        print(f"  Success Rate: {len(successful)/num_requests:.1%}")
        print(f"  Total Duration: {duration:.2f}s")
        print(f"  Aggregate T/s: {aggregate_tps:.2f}")
        print(f"  Mean Latency: {percentiles['mean']:.2f}s")
        print(f"  P99 Latency: {percentiles['p99']:.2f}s")

async def run_fairness_test():
    """Test fairness: 1 long request mixed with many short ones."""
    print(f"\n{'='*80}\n--- ⚖️ FAIRNESS TEST ---\n{'='*80}")
    print("Testing if long requests block short ones...")
    
    # One very long request
    long_payload = {
        "messages": [{"role": "user", "content": "Write a comprehensive 10-paragraph essay about the future of space exploration."}],
        "max_tokens": 2000,
        "temperature": 0.7
    }
    
    # Many short requests
    short_payload = {
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 20,
        "temperature": 0.1
    }
    
    async with aiohttp.ClientSession() as session:
        # Send long request first
        long_task = asyncio.create_task(send_request(session, long_payload, 0))
        
        # Wait 100ms, then send short requests
        await asyncio.sleep(0.1)
        
        short_tasks = [
            asyncio.create_task(send_request(session, short_payload, i+1))
            for i in range(20)
        ]
        
        # Gather all results
        all_tasks = [long_task] + short_tasks
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
    
    long_result = results[0]
    short_results = results[1:]
    
    short_successful = [r for r in short_results if isinstance(r, dict) and r.get("status") == "success"]
    short_latencies = [r["duration"] for r in short_successful]
    
    print(f"\n--- Fairness Results ---")
    if isinstance(long_result, dict) and long_result.get("status") == "success":
        print(f"  Long Request Duration: {long_result['duration']:.2f}s")
        print(f"  Long Request Tokens: {long_result.get('completion_tokens', 0)}")
    
    if short_latencies:
        short_stats = calculate_percentiles(short_latencies)
        print(f"  Short Requests Mean Latency: {short_stats['mean']:.2f}s")
        print(f"  Short Requests P50: {short_stats['p50']:.2f}s")
        print(f"  Short Requests P99: {short_stats['p99']:.2f}s")
        
        # Check if short requests were blocked
        if short_stats['mean'] > 2.0:
            print("  ⚠️ Warning: Short requests appear to be blocked by long request")
        else:
            print("  ✅ Good: Short requests processed fairly alongside long request")

async def run_ramp_test():
    """Gradually increase load to find breaking point."""
    print(f"\n{'='*80}\n--- 📈 RAMP TEST ---\n{'='*80}")
    print("Gradually increasing load to find system limits...")
    
    max_concurrent = 100
    step_size = 10
    step_duration = 5  # seconds per step
    
    payload = {
        "messages": [{"role": "user", "content": "Explain the water cycle."}],
        "max_tokens": 100,
        "temperature": 0.5
    }
    
    results_by_level = {}
    
    async def send_continuous_requests(session, level, duration, results_list):
        """Send requests continuously for a given duration."""
        end_time = time.time() + duration
        request_count = 0
        
        while time.time() < end_time:
            # Maintain 'level' concurrent requests
            active_tasks = [t for t in results_list if not t.done()]
            
            if len(active_tasks) < level:
                for _ in range(level - len(active_tasks)):
                    task = asyncio.create_task(send_request(session, payload, request_count))
                    results_list.append(task)
                    request_count += 1
            
            await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning
    
    async with aiohttp.ClientSession() as session:
        for current_level in range(step_size, max_concurrent + 1, step_size):
            print(f"\nRamping to {current_level} concurrent requests...")
            
            results_list = []
            step_start = time.time()
            
            # Run at this level for step_duration seconds
            await send_continuous_requests(session, current_level, step_duration, results_list)
            
            # Wait for remaining requests to complete
            completed_results = await asyncio.gather(*results_list, return_exceptions=True)
            
            step_duration_actual = time.time() - step_start
            
            # Analyze results
            successful = [r for r in completed_results if isinstance(r, dict) and r.get("status") == "success"]
            
            if successful:
                latencies = [r["duration"] for r in successful]
                tokens = sum(r.get("completion_tokens", 0) for r in successful)
                percentiles = calculate_percentiles(latencies)
                
                results_by_level[current_level] = {
                    "success_rate": len(successful) / len(completed_results),
                    "total_requests": len(completed_results),
                    "duration": step_duration_actual,
                    "aggregate_tps": tokens / step_duration_actual,
                    "p50_latency": percentiles["p50"],
                    "p99_latency": percentiles["p99"]
                }
                
                print(f"  Completed: {len(completed_results)} requests")
                print(f"  Success Rate: {results_by_level[current_level]['success_rate']:.1%}")
                print(f"  T/s: {results_by_level[current_level]['aggregate_tps']:.2f}")
                print(f"  P99: {percentiles['p99']:.2f}s")
                
                # Stop if success rate drops below 90% or P99 > 30s
                if results_by_level[current_level]['success_rate'] < 0.9 or percentiles['p99'] > 30:
                    print(f"\n⚠️ System saturation detected at {current_level} concurrent requests")
                    break
    
    # Save ramp test results
    with open(os.path.join(OUTPUT_DIR, "ramp_test_results.json"), "w") as f:
        json.dump(results_by_level, f, indent=2)
    
    return results_by_level

async def run_mixed_workload_advanced():
    """Advanced mixed workload with various request types."""
    print(f"\n{'='*80}\n--- 🎯 ADVANCED MIXED WORKLOAD ---\n{'='*80}")
    print("Testing with diverse request types and sizes...")
    
    request_types = [
        {
            "name": "trivial",
            "prompt": "Hi",
            "max_tokens": 5,
            "weight": 20
        },
        {
            "name": "short",
            "prompt": "Explain gravity in one sentence.",
            "max_tokens": 50,
            "weight": 30
        },
        {
            "name": "medium",
            "prompt": "Write a short paragraph about climate change.",
            "max_tokens": 200,
            "weight": 30
        },
        {
            "name": "long",
            "prompt": "Write a detailed technical explanation of how neural networks work.",
            "max_tokens": 1000,
            "weight": 15
        },
        {
            "name": "very_long",
            "prompt": "Write a comprehensive guide to machine learning for beginners.",
            "max_tokens": 2000,
            "weight": 5
        }
    ]
    
    # Create weighted request distribution
    num_requests = 100
    requests_to_send = []
    
    for req_type in request_types:
        count = int(num_requests * req_type["weight"] / 100)
        for _ in range(count):
            requests_to_send.append({
                "messages": [{"role": "user", "content": req_type["prompt"]}],
                "max_tokens": req_type["max_tokens"],
                "temperature": 0.5,
                "metadata": {"type": req_type["name"]}
            })
    
    # Shuffle for realistic mix
    random.shuffle(requests_to_send)
    
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, req, i) for i, req in enumerate(requests_to_send)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Analyze results by request type
    successful = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
    total_tokens = sum(r.get("completion_tokens", 0) for r in successful)
    aggregate_tps = total_tokens / total_duration if total_duration > 0 else 0
    
    all_latencies = [r["duration"] for r in successful]
    overall_stats = calculate_percentiles(all_latencies)
    
    print(f"\n--- Mixed Workload Results ---")
    print(f"  Total Requests: {len(requests_to_send)}")
    print(f"  Success Rate: {len(successful)/len(requests_to_send):.1%}")
    print(f"  Total Duration: {total_duration:.2f}s")
    print(f"  Aggregate T/s: {aggregate_tps:.2f}")
    print(f"  Overall P50 Latency: {overall_stats['p50']:.2f}s")
    print(f"  Overall P99 Latency: {overall_stats['p99']:.2f}s")

# --- MAIN TEST SUITE ---

async def main():
    """Run the complete concurrent testing suite."""
    print(f"{'='*80}")
    print(f"🚀 ENHANCED CONCURRENT TESTING SUITE")
    print(f"{'='*80}")
    print(f"Target Server: {API_URL}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # Overall summary
    summary = {
        "server": API_URL,
        "start_time": datetime.now().isoformat(),
        "tests": {}
    }
    
    # Run all tests
    print("\n" + "="*80)
    print("STARTING TEST SEQUENCE")
    print("="*80)
    
    # 1. Saturation Test
    print("\n[1/7] Running Saturation Test...")
    saturation_results = await run_saturation_test()
    summary["tests"]["saturation"] = saturation_results
    await asyncio.sleep(2)
    
    # 2. Time-to-First-Token Test
    print("\n[2/7] Running Time-to-First-Token Test...")
    ttft_results = await run_time_to_first_token_test()
    summary["tests"]["ttft"] = ttft_results
    await asyncio.sleep(2)
    
    # 3. Staggered Arrival Test
    print("\n[3/7] Running Staggered Arrival Test...")
    await run_staggered_arrival_test()
    await asyncio.sleep(2)
    
    # 4. Long Context Stress Test
    print("\n[4/7] Running Long Context Stress Test...")
    await run_long_context_stress_test()
    await asyncio.sleep(2)
    
    # 5. Fairness Test
    print("\n[5/7] Running Fairness Test...")
    await run_fairness_test()
    await asyncio.sleep(2)
    
    # 6. Mixed Workload Test
    print("\n[6/7] Running Advanced Mixed Workload Test...")
    await run_mixed_workload_advanced()
    await asyncio.sleep(2)
    
    # 7. Ramp Test (last as it finds breaking point)
    print("\n[7/7] Running Ramp Test...")
    ramp_results = await run_ramp_test()
    summary["tests"]["ramp"] = ramp_results
    
    # Save overall summary
    summary["end_time"] = datetime.now().isoformat()
    with open(os.path.join(OUTPUT_DIR, "complete_test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✅ ALL TESTS COMPLETED")
    print(f"{'='*80}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print key insights
    print(f"\n{'='*80}")
    print("📊 KEY INSIGHTS")
    print(f"{'='*80}")
    
    if saturation_results:
        # Find optimal concurrency
        best_throughput = max(saturation_results, key=lambda x: x["aggregate_tps"])
        best_latency = min(saturation_results, key=lambda x: x["p99"])
        print(f"\n🎯 Optimal Concurrency:")
        print(f"  Best Throughput: {best_throughput['concurrency']} concurrent requests ({best_throughput['aggregate_tps']:.2f} T/s)")
        print(f"  Best P99 Latency: {best_latency['concurrency']} concurrent requests ({best_latency['p99']:.2f}s)")
    
    if ramp_results:
        max_stable = max(k for k, v in ramp_results.items() if v["success_rate"] >= 0.95)
        print(f"\n💪 Maximum Stable Concurrency: {max_stable} requests")
        print(f"  Throughput at max: {ramp_results[max_stable]['aggregate_tps']:.2f} T/s")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    asyncio.run(main())