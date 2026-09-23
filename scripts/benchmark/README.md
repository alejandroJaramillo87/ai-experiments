# scripts/benchmark

Latency and throughput measurement against the live inference services. Nothing here
runs in a sandbox: every script needs the GPU or a running service, and the two profiling
wrappers have no `make` target by design, so a benchmark is always started deliberately.

`benchmark_cpu.sh` and `benchmark_gpu.sh` drive `server-bench.py` at several profiling
depths; `test_perplexity.sh` runs llama.cpp's perplexity over the wikitext-2 slice it
downloads into `datasets/`, which is git-kept and otherwise ignored. A benchmark is a GPU
consumer, so the one-job-at-a-time rule in `AGENTS.md` applies, and the protocol for
reporting a number is `.claude/skills/benchmarking/SKILL.md`.

## Scripts

<!-- inventory:scripts:start -->

| Script | Purpose |
|--------|---------|
| `benchmark_cpu.sh` | CPU benchmark with multi-level profiling. |
| `benchmark_gpu.sh` | GPU benchmark with multi-level profiling. |
| `server-bench.py` | Throughput benchmark for the llama.cpp HTTP server, driven by the two benchmark scripts. |
| `test_perplexity.sh` | Run perplexity tests on Docker containers. |

<!-- inventory:scripts:end -->

## Where the real docs live

- [../../docs/optimizations/README.md](../../docs/optimizations/README.md) — the measured results these scripts produced
- [../../docs/inference/README.md](../../docs/inference/README.md) — what the services being measured are tuned for
