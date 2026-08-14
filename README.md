# AI experiments workstation

Local AI infrastructure for an RTX 5090 and AMD Ryzen 9950X workstation: inference, training,
interpretability, observability, and a Rust inference harness. Inference is tuned for both low
latency (llama.cpp) and high throughput (vLLM). Every tier is containerized and pinned, and the
Python environment is managed with uv.

The work here is experimental in the literal sense: capabilities are built, then run on this
hardware and measured, and the difference between those two states is tracked rather than
assumed. The status table below is the summary of where each tier stands.

This file is the human entry point. Agents working in this repository read
[AGENTS.md](AGENTS.md), which carries the operating rules and the map of which document owns
which topic.

## Table of contents

- [Status](#status-7)
- [Not yet real](#not-yet-real)
- [Quick start](#quick-start)
- [Services](#services)
- [Hardware](#hardware)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [License](#license)

## Status (7)

Two independent questions. **State** is what a reader can do with a tier today. **Evidence** is
how strong the proof is, and it is deliberately separate: a tier can run every day and still have
no acceptance check on file.

- **In use** — running now, whatever the verification runbook says about it.
- **Built, not exercised** — code and sandbox tests exist; never run on this hardware.
- **Planned** — not built.

| Capability | State | Evidence |
|------------|-------|----------|
| Inference (llama.cpp CPU and GPU, vLLM) | In use | `docs/inference/README.md` and the per-service overviews in `docs/sandbox/`; no acceptance entry exists |
| OS and system optimizations | In use | `docs/optimizations/README.md` implementation status, with measured results; the BIOS soak tests HS.0 through HS.5 have never run |
| Agent sandbox | In use | `docs/agents/README.md`; no acceptance entry exists |
| Interpretability: harvest, SAE training, SAE evaluation | In use | `docs/verification/interp/interp.md` entries I.9, I.9b, and I.11, run on this hardware 2026-08-08 with output recorded |
| Training | Built, not exercised | `docs/verification/training/training-phase0.md` entries P0.1 through P0.3 run on hardware; P0.4 onward, and all of phases 1 and 2, pending |
| Rust harness (CPU backend) | Built, not exercised | `docs/verification/rust-harness/rust-harness.md`; 17 entries, none run. The sandbox quality campaign is complete |
| Observability | Built, not exercised | `docs/verification/observability/observability.md`; entries OBS-V1 through OBS-V8 pending |

## Not yet real

This repository has more built surface than proven surface, and the gap is large enough that
hiding it would make the rest untrustworthy. The verification runbook holds 15 entries marked
`Status: VERIFIED` against 71 marked `Status: PENDING`. Three defects are open in the ledger.

Three specific gaps a reader would otherwise discover by being misled:

- **Inference has no acceptance entry.** It is the tier in daily use and the one with measured
  benchmark numbers, but no file in `docs/verification/` covers it. Its evidence is measurement,
  not an observed acceptance check.
- **The step-zero verifications kept no output.** All eight entries in
  `docs/verification/step-zero-environment.md` are marked verified, and each Observed block reads
  only that the command was run on 2026-08-01 without the output being recorded. They are the
  weakest verified entries in the tree.
- **The Gemma 4 conformance audit was never executed.** Its 50 host checks remain unrun, blocked
  on absent weights, and freezing the record did not retire them.

The runbook and the ledger own these facts: [docs/verification/README.md](docs/verification/README.md)
for the protocol and the phase order, [docs/verification/bugs/known-bugs.md](docs/verification/bugs/known-bugs.md)
for the open defects. Neither is summarized here, because a count stated twice will disagree with
itself.

## Quick start

```bash
# 1. Clone the repository
$ git clone https://github.com/user/ai-experiments
$ cd ai-experiments

# 2. Install the GPU stack (if needed)
$ scripts/setup/setup_nvidia.sh
$ scripts/setup/setup_cuda.sh
$ scripts/setup/setup_cudnn.sh

# 3. Configure models
$ cp env.example .env
# Edit .env with model paths

# 4. Provision the Python environment
$ make install

# 5. Start services
$ make up          # All services
$ make gpu-up      # GPU only
$ make cpu-up      # CPU only

# 6. Verify
$ make status      # Check health
$ make logs-gpu    # Monitor GPU
```

Before pushing, run the quality gate. `make quality` is the portable form and runs identically in
a contained agent sandbox; `make quality-full` is the workstation superset that also runs the test
suite and the hardware smoke tier. See
[docs/testing/quality-gate.md](docs/testing/quality-gate.md).

## Services

| Service | Port | Purpose |
|---------|------|---------|
| llama-gpu | 8002 | Primary GPU inference, latency-focused |
| llama-cpu | 8001 | CPU inference on 16 dedicated cores, latency-focused |
| vllm-gpu | 8003 | High-throughput batch inference |
| open-webui | 3000 | Web interface for the inference services |
| training | none | Fine-tuning and pretraining; separate compose file |

`docker-compose.inference.yaml` is the source of truth for ports, profiles, and cpusets; the other
tiers live in `docker-compose.training.yaml`, `docker-compose.interp.yaml`,
`docker-compose.agents.yaml`, and `docker-compose.rust-harness.yaml`. Per-service detail is in
`docs/sandbox/`.

Only one GPU consumer runs at a time. Training, interpretability work, and benchmarks are
serialized against each other and against inference, regardless of how many worktrees are open.

## Hardware

| Component | Specification |
|-----------|---------------|
| CPU | AMD Ryzen 9950X, 16 cores and 32 threads |
| GPU | NVIDIA RTX 5090, 32GB VRAM, SM 12.0 Blackwell |
| Memory | 128GB DDR5-6000, 90GB allocated to huge pages |
| Storage | Samsung 990 PRO 2TB for models, Samsung 990 EVO Plus 1TB for the OS |
| Motherboard | Gigabyte X870E Aorus Elite WiFi |

Driver 580.65.06, CUDA 13.0.88, cuDNN 9.13.0.50-1. Full detail, including the settings that were
measured and the claims that were not, is in
[docs/hardware/README.md](docs/hardware/README.md).

## Repository layout

Each top-level source directory carries its own README covering purpose, entry points, and traps.
They are indexed by the table at the top of [docs/README.md](docs/README.md), which is the better
starting point than a tree.

- `docker/` — container definitions, one directory per service
- `training/`, `interp/`, `rust-harness/` — the three non-inference tiers
- `scripts/` — setup, update, documentation gates, and utilities
- `tests/` — the pytest suite; see `tests/README.md` for its tiers and markers
- `docs/` — everything else, indexed by `docs/README.md`

## Documentation

[docs/README.md](docs/README.md) is the documentation index and lists every document by category.
It is more complete than any summary that could live here, and it is the file to update when a
document is added.

The conventions that govern all of it are in `.styles/`, and `make quality` enforces the
mechanical half: every path a document cites must exist, every `make` target it names must
resolve, and every heading that states a count must match its section.

## License

MIT
