# test_concurrency.py

import asyncio
import aiohttp
import json
import time
import os
import random
import textwrap

# --- CONFIGURATION ---
# IMPORTANT: Point this to the server you want to test.
# vLLM is typically on port 8005, llama.cpp on 8004.
API_URL = "http://127.0.0.1:8005/v1/chat/completions" # Change port for llama.cpp
HEADERS = {"Content-Type": "application/json"}
OUTPUT_DIR = "test_results_concurrency"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- TEST CASE DEFINITION ---
# These workloads are designed to test high-throughput scenarios.

WORKLOADS = [
    {
        "name": "Scenario 2a: High Volume of Short Prompts",
        "description": "Simulates many users asking simple, quick questions. Tests raw request handling and scheduling.",
        "num_requests": 100,  # Number of concurrent requests to send
        "prompts": [
            {
                "content": "Explain the concept of photosynthesis in three sentences.",
                "params": {"max_tokens": 150, "temperature": 0.5}
            }
        ]
    },
    {
        "name": "Scenario 2b: Mixed Workload (Short & Long Prompts)",
        "description": "Simulates a more realistic API with a mix of quick questions and requests for longer content.",
        "num_requests": 50, # Fewer requests due to longer completions
        "prompts": [
            {
                "content": "What is the capital of France?",
                "params": {"max_tokens": 50, "temperature": 0.2}
            },
            {
                "content": "Write a short, 5-paragraph story about a robot who discovers music.",
                "params": {"max_tokens": 1024, "temperature": 0.8}
            }
        ]
    }
]

async def send_request(session, payload, index):
    """Asynchronously sends a single API request and returns performance data."""
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
            return {
                "status": "success",
                "index": index,
                "duration": end_time - start_time,
                "completion_tokens": usage.get("completion_tokens", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0)
            }
    except Exception as e:
        return {"status": "failed", "index": index, "error": str(e)}

async def run_concurrency_test(workload):
    """Runs a full concurrency test for a given workload."""
    name = workload["name"]
    description = workload["description"]
    num_requests = workload["num_requests"]
    
    print(f"\n{'='*80}\n--- 🚀 RUNNING: {name} ---\n{'='*80}")
    print(f"Description: {description}")
    print(f"Sending {num_requests} concurrent requests to {API_URL}...\n")

    # Prepare all payloads
    payloads = []
    for _ in range(num_requests):
        prompt_config = random.choice(workload["prompts"])
        payload = {
            "messages": [{"role": "user", "content": prompt_config["content"]}],
            **prompt_config["params"]
        }
        payloads.append(payload)

    # --- Execute all requests concurrently ---
    start_time_total = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, payload, i) for i, payload in enumerate(payloads)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time_total = time.time()

    # --- Process Results ---
    successful_requests = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
    failed_requests = [r for r in results if not (isinstance(r, dict) and r.get("status") == "success")]

    total_duration = end_time_total - start_time_total
    total_completion_tokens = sum(r.get("completion_tokens", 0) for r in successful_requests)
    
    aggregate_tps = 0
    if total_completion_tokens > 0 and total_duration > 0:
        aggregate_tps = total_completion_tokens / total_duration

    # --- Print Summary ---
    print("--- ✅ All requests completed. ---")
    print(f"\n--- 📊 Aggregate Performance Metrics ---")
    print(f"Total Time to Complete All {num_requests} Requests: {total_duration:.2f} seconds")
    print(f"Total Completion Tokens Generated: {total_completion_tokens}")
    print(f"Successful Requests: {len(successful_requests)}/{num_requests}")
    if failed_requests:
        print(f"Failed Requests: {len(failed_requests)}/{num_requests}")
    print("-" * 40)
    print(f"Aggregate Tokens per Second (T/s): {aggregate_tps:.2f}")
    print("-" * 40)
    
    # --- Save Summary to File ---
    filename_safe_name = name.lower().replace(":", "").replace(" ", "_").replace("(", "").replace(")", "")
    summary_path = os.path.join(OUTPUT_DIR, f"{filename_safe_name}_summary.txt")
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"--- Test Summary for: {name} ---\n\n")
        f.write(f"Total Duration: {total_duration:.2f}s\n")
        f.write(f"Concurrent Requests: {num_requests}\n")
        f.write(f"Successful / Failed: {len(successful_requests)} / {len(failed_requests)}\n")
        f.write(f"Total Completion Tokens: {total_completion_tokens}\n")
        f.write(f"Aggregate Tokens per Second: {aggregate_tps:.2f} T/s\n\n")
        if failed_requests:
            f.write("--- Errors ---\n")
            for i, res in enumerate(failed_requests[:10]): # Log first 10 errors
                 f.write(f"Error {i+1}: {res}\n")

    print(f"--- Detailed summary saved to '{summary_path}' ---")
    return {"name": name, "tps": aggregate_tps, "duration": total_duration, "success_rate": len(successful_requests)/num_requests}


if __name__ == "__main__":
    print(f"--- High-Concurrency Throughput Test Suite ---")
    print(f"This script tests the server's ability to handle many requests at once.")
    print(f"vLLM is expected to perform significantly better here than llama.cpp.\n")
    
    summary_file = os.path.join(OUTPUT_DIR, "_final_throughput_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=== HIGH-CONCURRENCY THROUGHPUT TEST - FINAL SUMMARY ===\n")
        f.write(f"Test Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target Server: {API_URL}\n")
        f.write("=" * 60 + "\n")
    
    overall_results = []
    start_time_suite = time.time()

    # Must run the async main loop
    async def main():
        for workload in WORKLOADS:
            result = await run_concurrency_test(workload)
            overall_results.append(result)
            time.sleep(2) # Give server a moment to breathe between heavy workloads

    asyncio.run(main())

    end_time_suite = time.time()
    total_suite_duration = end_time_suite - start_time_suite

    # Final Summary
    final_summary_text = (
        f"\n\n{'='*60}\n"
        f"--- 📈 FINAL THROUGHPUT SUMMARY ---\n"
        f"Total Suite Duration: {total_suite_duration:.2f} seconds\n\n"
    )
    for res in overall_results:
        final_summary_text += f"  - {res['name']}:\n"
        final_summary_text += f"    Aggregate T/s: {res['tps']:.2f}\n"
        final_summary_text += f"    Success Rate: {res['success_rate']:.1%}\n"

    final_summary_text += (
        f"\n--- ✅ Test suite completed. Check '{OUTPUT_DIR}' for detailed logs. ---\n"
        f"{'='*60}"
    )
    print(final_summary_text)

    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(final_summary_text)