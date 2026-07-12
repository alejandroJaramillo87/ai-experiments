# AI Infrastructure Base Repository

Production-ready AI infrastructure for an RTX 5090 + AMD Ryzen 9950X workstation:
containerized LLM inference optimized for both latency (llama.cpp) and throughput (vLLM).

## Operating rules

These are standing instructions for anyone (human or agent) making changes in this repo.

- **Pin, then fix, then update — in that order.** Pin every image tag, model revision, and
  dependency to an exact version before changing anything. Get the pinned state building and
  passing. Only then bump a version, one at a time, re-running the acceptance check after each
  bump. Never combine a version bump with a functional change in the same step.
- **Compose files and Dockerfiles are the source of truth; documentation follows them.** When a
  doc disagrees with `docker-compose*.yml` or a `Dockerfile`, the compose/Dockerfile is correct
  and the doc is the bug. Fix the doc to match — never adjust the running config to match a stale
  doc.
- **Every roadmap task ends by running its stated acceptance check.** A task is not complete
  until its acceptance check has been executed and observed to pass. Report the exact command and
  its output; do not mark a task done on the strength of a code change alone.
- **One GPU consumer at a time; parallelize the rest across worktrees.** Independent, non-GPU
  tasks may run concurrently in separate git worktrees (one agent session each). Anything that
  touches the GPU, a running inference service, or the benchmark suite is serialized — exactly
  one such job at a time, regardless of how many sessions or worktrees are active. Never run two
  benchmarks or two GPU jobs at once; results and VRAM will collide.

## Hardware

- GPU: RTX 5090 (32GB VRAM, Blackwell)
- CPU: AMD Ryzen 9950X (16 cores, Zen 5)
- Memory: 128GB DDR5-6000
- Models: `/mnt/ai-data/models/`

## Services

| Service    | Port | Purpose                       |
|------------|------|-------------------------------|
| llama-cpu  | 8001 | Low-latency CPU inference     |
| llama-gpu  | 8002 | GPU llama.cpp inference       |
| vllm-gpu   | 8003 | High-throughput batch serving |
| open-webui | 3000 | Web interface                 |

## Validated optimizations

- Huge pages: 26% CPU-inference improvement (`docker/llama-cpu/hugepage_mmap_wrapper.cpp`)
- BIOS: FCLK 2100MHz, C-states disabled
- CPU allocation: 12 dedicated cores per single model instance

## Common workflows

```bash
make up          # Start all services
make status      # Check health
make logs        # View logs
make benchmark   # Run performance tests
make shell-gpu   # Access GPU container
```

## Tooling conventions

- Use the `gh` CLI for all GitHub operations (pull requests, issues, checks, releases)
  instead of raw git remotes or the web API.
- Prefer `make` targets (`make status`, `make logs`) over raw docker commands.

## Code navigation

Prefer the LSP tools (definitions, references, diagnostics) for symbol-level questions; they
return less context than grep sweeps. Use grep/glob for exact strings — error messages,
config keys, version pins.

## Core principles

- Unix/Linux philosophy: simple, composable, text-based
- No emojis or decorative elements
- Clear technical writing; reproducible benchmarks

## Style guides (read the relevant one before writing that file type)

- `.styles/BASH_OUTPUT_STYLE.md`   — bash script output formatting
- `.styles/COMMENT_STYLE.md`       — comment conventions (all languages)
- `.styles/CPP_STYLE.md`           — C++ standards (llama-zen-turbo libs)
- `.styles/DOCUMENTATION_STYLE.md` — docs writing standards
- `.styles/PYTHON_OUTPUT_STYLE.md`  — Python CLI output formatting
- `.styles/TEST_OUTPUT_STYLE.md`   — test program/runner output format
