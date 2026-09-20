# scripts/interp

Host-side helpers for the interp tier: the GPU preflight the make targets run first, the
isolated SAEBench environment, the refusal-rate measurement and the activation-magnitude
probe. None runs in a sandbox; each needs the GPU, a running service or the workbench
container.

`preflight.sh` is the gate: it refuses to start while `llama-gpu` or `vllm-gpu` is running,
and `make interp-up`, `make interp-shell`, `make harvest`, `make sae-train`, `make sae-eval`,
`make probe-capture` and `make steer-extract` all call it before doing anything. The other
three are run by hand from the runbook entries that cite them.

## Scripts

<!-- inventory:scripts:start -->

| Script | Purpose |
|--------|---------|
| `activation_magnitude_probe.py` | Measure residual-stream magnitude to decide whether fp16 can store a capture. |
| `preflight.sh` | Interpretability concurrency preflight (interp Phase 0 I1). |
| `refusal_rate.sh` | Refusal rate under steering (rust-harness runbook RH.9e). |
| `saebench_env.sh` | SAEBench isolated environment (interp Phase 4 P4-D3). |

<!-- inventory:scripts:end -->

## Where the real docs live

- [../../interp/README.md](../../interp/README.md) — the tier these scripts serve
- [../../docs/interp/README.md](../../docs/interp/README.md) — the interp documentation
- [../../docs/verification/interp/interp.md](../../docs/verification/interp/interp.md) — the entries that run the hand tools
