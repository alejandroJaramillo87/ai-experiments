---
name: benchmarking
description: The reproducible benchmark protocol for this workstation — which entry points to run, warm-up policy, how results are recorded and compared, and what counts as a regression. Use before reporting any latency/throughput/memory number.
---
# Benchmarking protocol

Reference for producing comparable performance numbers on the RTX 5090 / Ryzen 9950X
stack. Follow `.styles/TEST_OUTPUT_STYLE.md` for all output.

## Entry points

There is no `make benchmark` target — run the scripts directly against a running service:

- CPU: `./scripts/benchmark/benchmark_cpu.sh -n <name> [-l basic|detailed|deep] [-p N]`
  (default container `llama-cpu`, 20 prompts, `basic` = `perf stat` core events).
- GPU: `./scripts/benchmark/benchmark_gpu.sh -n <name> [-l basic|deep] [-p N]`
  (default container `llama-gpu`, `basic` = nvidia-smi polling, `deep` = Nsight Systems).
- Perplexity/quality: `./scripts/benchmark/test_perplexity.sh`.
- The load generator underneath both is `scripts/benchmark/server-bench.py`.

Preconditions: the target service must be healthy first — check `make status` (and
`make health`) before starting. A benchmark against a cold or unhealthy service is invalid.

## Warm-up and sampling

1. Discard the first run (warm-up): caches, JIT, and clocks are not yet settled.
2. Take multiple samples (>= 3 prompts-batches) at a fixed prompt count and dataset.
3. Report **median and spread**, never a single best number.
4. Hold the workload constant across comparisons: same model, same `-p` prompt count,
   same profiling level, same dataset under `scripts/benchmark/datasets/`.

## Recording and comparison

- Name each run with `-n <name>` so its output files are traceable to a configuration.
- Record the fixed conditions alongside the number: model, quantization, prompt count,
  profiling level, driver/CUDA versions, and the exact command line.
- Compare only like-for-like runs (same workload, same service).

## Regression criterion

A change is a regression when the median of the new run is more than **5%** slower than the
recorded baseline for the same workload, outside the run-to-run spread. Flag it explicitly;
do not average a regression away against unrelated wins.
