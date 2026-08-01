# AI Infrastructure Base Repository

Production-ready AI infrastructure for an RTX 5090 + AMD Ryzen 9950X workstation:
containerized LLM inference optimized for both latency (llama.cpp) and throughput (vLLM).

## Execution environment (agents)

**A contained agent session (`make agent-sandbox`) has no access to the host.** Inside it there
is no host Docker daemon, no GPU, no `nvidia-smi`, no `/mnt/ai-data`, and no running inference
services (ports 8001-8003 and 3000). Only the git worktree at `/work` (read-write) and network
egress to an allowlist (via the proxy) are available. A contained session is flagged
authoritatively by the sandbox banner seeded into `~/.claude/CLAUDE.md`, so do not spend a turn
rediscovering these limits.

Do not attempt these from a contained session — none can work:

- `docker` / `docker compose`, and the `make` targets that drive services or docker (`make up`,
  `down`, `status`, `logs`, `shell-*`, `clean`, `prune`, `agent-sandbox-clean`).
- `nvidia-smi`, any GPU work, and the benchmark scripts.
- Healthcheck curls against the service ports, and any read under `/mnt/ai-data`.

Instead:

- When a task requires a host-side command, output the exact command for the human to run,
  then stop and wait for the human to report the output back.
- Human-reported output is the authoritative evidence for acceptance checks, healthchecks,
  and benchmarks. Record the reported command and output verbatim when marking work done.
- Never claim to have run a host-side command, and never fabricate its output.
- Repo-local work remains directly executable in the sandbox: editing files, grep, git,
  `gh`, and unit tests that need no GPU or running service.

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
  until its acceptance check has been executed and observed to pass. For host-side checks the
  human executes the command and reports the output back; the agent records the exact command
  and its reported output. Do not mark a task done on the strength of a code change alone.
- **One GPU consumer at a time; parallelize the rest across worktrees.** Independent, non-GPU
  tasks may run concurrently in separate git worktrees (one agent session each). Anything that
  touches the GPU, a running inference service, or the benchmark suite is serialized — exactly
  one such job at a time, regardless of how many sessions or worktrees are active. Never run two
  benchmarks or two GPU jobs at once; results and VRAM will collide.
- **`make quality` is mandatory before every push, for humans and agents alike.** The gate runs
  the ruff format check, ruff lint, mypy, the uv lockfile check, and shellcheck, in that order,
  failing on the first error, then chains the rust-harness gate (`rust-harness/Makefile`:
  cargo fmt, clippy, lockfile check, rustdoc lints, tests) when that directory exists. That
  rust gate also needs a `python3` on `PATH` that can import numpy and gguf: the npz writer's
  unit test (`rust-harness/runtime/src/npz.rs`) shells out to it to re-read a written shard
  with `np.load`, `harness/tests/probe_identity.rs` takes an independent SHA-256 from its
  hashlib, and the cross-language contract tests run the real Python exporters and readers
  from the parent tree (see `schemas/README.md`). Those contract tests report `[SKIP]` and
  pass when a dependency or the parent tree is absent, so a green rust gate on a machine
  without gguf is a weaker result than it looks — check the output for `[SKIP]` lines rather
  than trusting the exit code alone. No step needs the network. The sandbox caveats all
  trace to one cause — an unsynced virtual environment — and were re-measured on
  2026-08-01. `uv run --no-sync` resolves whatever `.venv` happens to hold rather than
  the pins (ruff 0.15.8 against a 0.15.21 pin, mypy 1.19.1 against 2.3.0), and the
  pinned `types-pyyaml` is absent altogether, so `mypy` reports three
  missing-yaml-stub errors in `training/` that a synced workstation does not see. A warm
  `.mypy_cache` masks even that: the gate's `uv run --no-sync mypy` passes incrementally
  and fails under `--no-incremental`, so a sandbox run stops at mypy or sails past it
  depending on cache state. The other four steps — both ruff steps, `uv lock --check`,
  and shellcheck — pass. Treat a clean sandbox run as evidence, not proof, and re-run the
  gate on the workstation before trusting it. `make test` likewise cannot sync, because
  it reaches the pytorch cu130 index; run `uv run --no-sync pytest` instead.

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
| training   | -    | Fine-tuning / pretraining (Phase 0 D1; separate compose, no port) |

## Validated optimizations

- Huge pages: 26% CPU-inference improvement (`docker/llama-cpu/hugepage_mmap_wrapper.cpp`)
- BIOS: FCLK 2100MHz, C-states disabled
- CPU allocation: 12 dedicated cores per single model instance

## Common workflows

Host-side commands — run by the human on the workstation, not from an agent sandbox:

```bash
make up          # Start all services
make status      # Check health
make logs        # View logs
./scripts/benchmark/benchmark_gpu.sh   # Run performance tests (no make wrapper)
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
