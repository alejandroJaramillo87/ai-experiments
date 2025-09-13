# Performance Benchmark Guide

## Overview

The `scripts/benchmark.py` tool measures inference performance for llama.cpp containers, focusing on tokens per second as the primary metric for optimization validation.

## Quick Start

```bash
# Basic benchmark with default settings
python scripts/benchmark.py

# Custom configuration
python scripts/benchmark.py --port 8001 --runs 5 --output results.json

# Test specific prompts
python scripts/benchmark.py --prompts memory_sequential,cache_loops
```

## Available Test Prompts

| Prompt Key | Description | Tests |
|------------|-------------|-------|
| `memory_sequential` | Sequential memory access | Memory bandwidth, TLB efficiency |
| `cache_loops` | Repetitive patterns | CPU cache efficiency |
| `compute_arithmetic` | Basic calculations | Raw compute performance |
| `memory_structured` | JSON generation | Memory allocation patterns |
| `throughput_sustained` | List generation | Sustained generation speed |
| `memory_bandwidth` | Long sequences | Large memory transfers |

## Command-Line Options

```bash
python scripts/benchmark.py [OPTIONS]

Options:
  --host HOST          API host (default: localhost)
  --port PORT          API port (default: 8001)
  --runs N             Number of runs per prompt (default: 5)
  --prompts LIST       Comma-separated prompt keys to test
  --output FILE        Output JSON filename
  --timeout SECONDS    Request timeout (default: 30)
```

## Output Format

### Terminal Output
```
============================================================
  LLAMA.CPP PERFORMANCE BENCHMARK
============================================================
Model: Qwen3-30B-IQ4_XS.gguf
Huge Pages: 7813/46080 (2 MB each)
Timestamp: 2025-09-12T20:30:17
Runs per prompt: 5

Per-Prompt Performance:
----------------------------------------
Prompt                  Avg tok/s    Success
----------------------------------------
memory_sequential           32.00       100%
cache_loops                 31.94       100%
```

### JSON Output
```json
{
  "system_info": {
    "model": "model_path",
    "hugepages_total": 46080,
    "hugepages_free": 38267
  },
  "prompts": {
    "memory_sequential": {
      "prompt_text": "Count from 1 to 20...",
      "results": [...],
      "stats": {
        "avg_tokens_per_second": 32.0,
        "success_rate": 1.0
      }
    }
  },
  "summary": {
    "overall_avg_tokens_per_second": 31.95
  }
}
```

## Interpreting Results

### Key Metrics

1. **Tokens per Second**: Primary performance indicator
   - Higher is better
   - Typical range: 20-50 tok/s for CPU inference
   - Affected by: CPU speed, memory bandwidth, optimizations

2. **Success Rate**: Completion percentage
   - Should be 100% for healthy system
   - < 100% indicates timeout or error issues

3. **Standard Deviation**: Consistency measure
   - < 5% is excellent
   - > 10% suggests system instability

### Performance Baselines

| Model Size | Expected tok/s (CPU) | With Optimizations |
|------------|---------------------|-------------------|
| 7B params | 40-60 | 50-70 |
| 13B params | 25-40 | 35-50 |
| 30B params | 15-25 | 25-35 |

## Using for Optimization Testing

### Before/After Comparison

1. **Baseline Measurement**:
```bash
python scripts/benchmark.py --runs 10 --output baseline.json
```

2. **Apply Optimization** (e.g., enable huge pages)

3. **Measure Impact**:
```bash
python scripts/benchmark.py --runs 10 --output optimized.json
```

4. **Compare Results**:
```python
import json

with open('baseline.json') as f:
    baseline = json.load(f)
with open('optimized.json') as f:
    optimized = json.load(f)

improvement = (
    optimized['summary']['overall_avg_tokens_per_second'] / 
    baseline['summary']['overall_avg_tokens_per_second'] - 1
) * 100

print(f"Performance improvement: {improvement:.1f}%")
```

## Best Practices

1. **Warmup**: Script includes automatic warmup run
2. **Multiple Runs**: Use at least 5 runs for statistical significance
3. **System State**: Ensure consistent system state between tests
4. **Model Loading**: Wait for full model load before benchmarking
5. **Isolation**: Minimize other system activity during tests

## Troubleshooting

### "API not responding"
- Check container is running: `docker ps`
- Verify port: `curl http://localhost:8001/v1/models`

### High Variance in Results
- Check CPU governor: `cpupower frequency-info`
- Disable CPU frequency scaling
- Check for thermal throttling

### Timeouts
- Increase `--timeout` parameter
- Reduce prompt complexity
- Check model size vs available RAM