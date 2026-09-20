# Optimizations

System optimizations for LLM inference on AMD Ryzen 9950X + RTX 5090: latency-focused
(llama.cpp on CPU/GPU) and throughput-focused (vLLM on GPU). The firmware and the host
operating system are tuned separately, and the parked ideas are kept apart from what is
applied.

## Directories

<!-- inventory:dirs:start -->

| Directory | Holds |
|-----------|-------|
| [bios/](bios/README.md) | The firmware settings measured for LLM inference on the Gigabyte X870E board, and the raw CPU benchmark results before and after them: `cpu_benchmark_results_no_bios.json` and `cpu_benchmark_results_final_bios.json` are the captured data behind the figures the parent README states. |
| [ideation/](ideation/README.md) | Parked proposals: recorded menus of optimization work that is not scheduled, dated at authoring and edited only as items graduate into the living pages each names. |
| [os/](os/README.md) | The host operating system tuning applied for containerized inference: governor, huge pages, swap, CPU pinning, and what each was measured to do. |

<!-- inventory:dirs:end -->

## Pages

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [system-optimizations.md](system-optimizations.md) | System optimizations for LLM inference on AMD Ryzen 9950X + RTX 5090: latency-focused (llama.cpp on CPU/GPU) and throughput-focused (vLLM on GPU). |

<!-- inventory:pages:end -->

## Related documentation

- [../inference/README.md](../inference/README.md) — the engines and the parameters these
  settings were measured with
- [../sandbox/README.md](../sandbox/README.md) — the containers that carry the CPU
  pinning and memory locking
