---
title: Environment
---

# Environment

The tier runs in two environments, and which one a verb needs decides where its `make`
target runs it.

## The host environment

The repository's Python environment, managed with `uv`, runs the verbs that load no model:
`probe-export`, `probe-eval`, `steer-scorecard`, the `organism` stages, and `runs`. It also
runs the CPU verbs `probe-train` and `steer-extract` when the workbench is not wanted. The
dependency groups decide what is installed:

```bash
uv sync --only-group sandbox --only-group dev
uv sync
```

The first line installs the `sandbox` group, which pins numpy, scikit-learn, safetensors, and
xxhash without torch, and is enough for every CPU verb and for `--help` on every verb. The
second installs everything, including the `interp` group with torch, which the host needs
only for verbs that run outside the workbench and load a model. A verb invoked on the host
without its dependencies fails at the lazy import inside its handler, after parsing, with the
missing module named.

## The workbench

The workbench is the container that holds the GPU stack: PyTorch for CUDA 13 on the RTX
5090, TransformerLens, nnsight, and SAELens, built once and verified at build time by the
smoke test. Every verb that runs a model on the GPU runs there: `probe-capture`,
`harvest`, `sae-train`, `sae-eval`, and by default `probe-train`, `steer-extract`, and
`probe-monitor`. [Workbench](workbench.md) covers building and starting it, and
[Smoke test](smoke-test.md) covers proving it.

Inside the container the data trees are mounted at container paths, and a verb that runs
there takes the container spelling:

| Host | Container | Mounted |
|------|-----------|---------|
| The repository | `/app` | Read-write |
| `/mnt/ai-data/models` | `/app/models` | Read-only |
| `/mnt/ai-data/interp` | `/mnt/ai-data/interp` | Read-write |
| `/mnt/ai-data/training` | `/mnt/ai-data/training` | Read-write |
| `/mnt/ai-data/logs` | `/mnt/ai-data/logs` | Read-write |

The models tree is the one path that changes spelling, which is why `probe-capture --model`
takes `/app/models/...` and refuses the host path with that correction.

## Which environment a verb needs

| Verb | Loads a model | Needs the GPU | Needs a service up | Runs in |
|------|---------------|---------------|--------------------|---------|
| `probe-capture`, `harvest`, `sae-train`, `sae-eval` | Yes | Yes | No | Workbench, after the preflight |
| `probe-train`, `steer-extract`, `probe-export`, `gc` | No | No | No | Either |
| `probe-monitor`, `runs` | No | No | No | Either, while inference is up |
| `probe-eval`, `steer-scorecard`, `organism generate`, `organism verify` | No | No | Yes | Host |
| `organism seal`, `organism check-seal` | No | No | No | Host |
| `smoke` | Yes | Yes | No | Workbench |

## The one-GPU rule

The workstation has one GPU, and the tier enforces one consumer at a time. Every target in
the first row runs `scripts/interp/preflight.sh` before starting, which asks the inference
compose file which services are running and refuses while `llama-gpu` or `vllm-gpu` is up,
naming the service and the command that stops it. `llama-cpu` and the serving harness are
permitted. A failing compose invocation is refused too, rather than read as nothing running.
The harvest additionally takes the repository-wide GPU lock that training runs hold, so the
two cannot overlap even across worktrees.
