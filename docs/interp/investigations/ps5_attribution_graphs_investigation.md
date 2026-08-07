# Interpretability PS5 attribution graphs: implementation investigation

**Frozen record — 2026-07-14. Drift against the current repository is expected and
correct; this document is never updated.**

Investigation of interp Phase 2 deliverable PS5 (attribution graphs). The planning
document's design is taken as intent; the repository is the source of truth. PS5 is
tagged orchestrate and depends on PS1 (merged) and PS2 (viewer needs imported sources;
host execution pending per `ps2_gemma_scope_import_investigation.md`). It runs in the
interactive concurrency mode already defined in `platform.md`. Most of the plumbing PS5
assumes landed with PS1; what remains is verification against the pinned checkout, two
small repository-side additions, the runbook section, and the host-side commissioning
mini-project that discharges roadmap acceptance criterion 3.

## Table of contents

- [What PS5 delivers](#what-ps5-delivers)
- [Design-versus-repository deviations](#design-versus-repository-deviations)
- [Orchestration already in place](#orchestration-already-in-place)
- [The experiments log gap](#the-experiments-log-gap)
- [Graph persistence question](#graph-persistence-question)
- [Repository changes](#repository-changes)
- [Upstream facts to re-verify at implementation time](#upstream-facts-to-re-verify-at-implementation-time)
- [Acceptance checks and host-side split](#acceptance-checks-and-host-side-split)
- [Risks](#risks)
- [Open questions](#open-questions)

## What PS5 delivers

circuit-tracer attribution graphs generated locally on Gemma-2-2B with the 8x/k64
transcoder configuration (verified inside the 32GB VRAM envelope; the 32x configs need
40GB+ and are out of scope), served by the upstream Neuronpedia graph service (FastAPI
wrapper around `safety-research/circuit-tracer`, container port 5004), and rendered in
the local webapp's graph viewer with node inspection and pinning. The acceptance vehicle
is a commissioning mini-project: generate a graph for a chosen prompt, form a mechanistic
hypothesis from it, test the hypothesis with prompt interventions through the ordinary
serving stack, and record a post-mortem including a falsification attempt.

Other models (Qwen3-4B, Llama-3.2-1B) and larger transcoder configs are attempted only
when a project pulls them. The hosted S3-upload path is not configured; graphs stay in
the local stack. The upstream graph-crm service (5005) remains undeployed.

## Design-versus-repository deviations

| Design says | Repository reality | Resolution |
|-------------|--------------------|------------|
| GPU graph server via `make np-graph-up`, one GPU tenant enforced by the PS1 preflight | `Makefile` `np-graph-up` exists, runs `scripts/interp/preflight.sh`, composes upstream `compose.graph.dev.yaml` + `compose.graph.gpu.yaml` + the override | Consistent; verify host-side that it actually brings the service up at pin `e5488b2a` |
| Transcoder weights through the shared HF cache | `neuronpedia.override.yaml` already mounts `/mnt/ai-data/interp/hf-cache/hub` into the graph container | Consistent; first run downloads tens of GB into the cache — record actual size |
| Post-mortem lands in `docs/interpretability/experiments/` per the Phase 1 log convention | The directory and convention do not exist; the interp tier shipped infrastructure only (see the M3 removal, commit `bdd2521`) | Bootstrap the convention as part of PS5: a `README.md` stating the post-mortem standard, then the mini-project write-up as its first entry |
| Graph server VRAM well under 32GB for Gemma-2-2B 8x/k64 | Unverifiable from the sandbox; upstream demonstrates 15GB-class GPUs | Host-side measurement at acceptance time; record peak VRAM in the runbook |
| Localhost mode keeps graphs in the local stack | Where the graph service persists generated graphs (Postgres, container filesystem, named volume, or S3) is not captured in-repo | Host-side verification; if graphs land outside the Postgres bind mount they do not survive `np-down` and the override needs one more bind mount |

One repository quirk to verify rather than "fix": `np-graph-up` invokes compose with
`--project-directory .` and `ENV_FILE=.env.localhost` (no `../` prefix), unlike the
inference and autointerp targets, and additionally passes `--env-file apps/graph/.env`.
This mirrors upstream's own graph make targets at the pin; confirm `apps/graph/.env`
exists in the checkout and carries the model/transcoder selection knobs before assuming
the target works.

## Orchestration already in place

PS1 landed the harness PS5 sits on; no new make targets are required:

- `make np-graph-up` (`Makefile`, Interpretability Platform section): preflight, then
  upstream compose files plus the override, port `127.0.0.1:5004`.
- `scripts/interp/preflight.sh`: refuses to start while llama-gpu, vllm-gpu, or comfyui
  is running — and the PS4 preflight symmetrically refuses autointerp while the graph
  server is up, so the one-GPU-tenant policy is already structural in both directions.
- `neuronpedia.override.yaml`: loopback port remap and HF-cache mount for the graph
  service.
- `platform.md` concurrency matrix: PS5 is already listed under interactive platform
  work.

What PS1 did not cover: `scripts/interp/np_smoke.sh` checks torch sm_120 capability only
inside the inference container. The graph image builds on the same CUDA 12.1-era base
family and carries the same risk (a torch pin predating sm_120 wheels fails only at
runtime). Extending the smoke check to run the identical capability probe against
`neuronpedia-graph-1` when it is running is a small, in-pattern addition.

## The experiments log gap

The planning document treats the Phase 1 experiments log convention as a given
(`docs/interpretability/experiments/`), but the repository never created it — the interp
tier was cut to infrastructure only. PS5's acceptance cannot be discharged without a
place for the post-mortem, so PS5 bootstraps the convention minimally:

- `docs/interpretability/experiments/README.md`: entry naming
  (`YYYY-MM-DD-short-slug.md`), required sections (question, setup, what was observed,
  hypothesis, falsification attempt, verdict, artifacts), and the standard that negative
  results are recorded identically.
- The commissioning mini-project write-up becomes the first entry and must cite the
  locally generated graph (prompt, model, transcoder config, and how to regenerate it).

This is a documentation convention, not tooling; no code is pulled forward.

## Graph persistence question

The override bind-mounts Postgres and the HF cache, nothing else. If the graph service
stores generated graphs in Postgres (webapp-mediated), persistence across
`np-down`/`np-up` is already handled. If it writes JSON artifacts to the container
filesystem or an upstream named volume, graphs are lost on teardown and the override
gains one bind mount under `/mnt/ai-data/interp/neuronpedia/`. This is the only open
item that could add a repository change beyond the two listed below; resolve it during
host-side verification before the mini-project, since losing the acceptance graph to a
`np-down` would be an avoidable re-run.

## Repository changes

| Path | Change | Status |
|------|--------|--------|
| `scripts/interp/np_smoke.sh` | Run the sm_120 torch capability check against the graph container when it is running, identical to the inference-container check | Landed with this investigation |
| `docs/interpretability/experiments/README.md` | Bootstrap the experiments log convention (post-mortem standard) | Landed with this investigation |
| `docs/interpretability/platform.md` | "Attribution graphs" runbook section: bring-up order, operational facts, and a measurements table to fill from the first host run | Landed with this investigation; measurements pending host run |
| `docker/interp/neuronpedia.override.yaml` | Only if verification shows graphs persist outside Postgres: one bind mount for graph artifacts | Deferred to host-side verification |
| `docs/interpretability/experiments/<date>-<slug>.md` | The commissioning mini-project post-mortem (host-side, acceptance) | Host-side |

No new make targets, no patches expected: unlike PS4 there is no known upstream gap
shaped like the OpenRouter URL. The patch mechanism from PS4 is available if
verification finds one.

## Upstream facts to re-verify at implementation time

Against the checkout at pin `e5488b2a`, on the host (the sandbox cannot read
`/mnt/ai-data/interp/neuronpedia/src`):

- `apps/graph/.env` exists and its knobs: which model ids and transcoder configurations
  the wrapper exposes, and how Gemma-2-2B 8x/k64 is selected (env var, request field, or
  hardcoded default).
- Whether the graph service needs `HF_TOKEN` in its environment for the gated Gemma
  weights, and whether `scripts/interp/np_env.sh` already delivers it on the graph
  service's env path (it was verified for the inference path in PS1).
- torch version in the graph image and sm_120 support; if the pin predates sm_120
  wheels, override via build arg or a one-line patch recorded beside the pin, same
  procedure as the inference image.
- Where generated graphs persist (see above), and whether generation is driven from the
  webapp UI, a webapp API route, or the graph service's own API — the runbook documents
  whichever is real.
- Whether the webapp graph viewer requires PS2's imported sources for node inspection
  (feature pages behind the nodes) or renders transcoder features standalone. The
  design's "depends on PS2 (viewer)" is taken as intent; confirm the failure mode if
  sources are missing.

## Acceptance checks and host-side split

All execution host-side; the sandbox contributes orchestration and records reported
output verbatim.

1. Interactive mode bring-up: GPU inference services and the PS3 instance down,
   `make np-up && make np-graph-up && make np-smoke` green, including the graph
   container sm_120 check.
2. An attribution graph generated locally on Gemma-2-2B (8x/k64) renders in the local
   viewer with working node inspection and pinning; graph survives
   `make np-down && make np-up` (or the persistence gap is closed first).
3. The commissioning mini-project post-mortem exists in
   `docs/interpretability/experiments/`, cites the graph, and includes a falsification
   attempt (roadmap criterion 3).
4. Runbook records measured transcoder cache size, first-run and warm generation
   timings, and peak VRAM.

## Risks

- **Transcoder download size** (tens of GB) counts against the interp storage budget
  and the disk-free watchdog floor; measure and record on first run.
- **First-run latency** (weight load plus compile) can look like a hang; the runbook
  notes expected first-run versus warm timings once measured.
- **Graph persistence** may require an override change discovered only at verification
  time; sequenced before the mini-project so the acceptance graph is not lost.
- **PS2 is still pending host execution.** If the viewer needs imported sources, PS5's
  acceptance queues behind the PS2 import; the graph server bring-up and measurements
  can proceed independently.
- **torch/sm_120 in the graph image**: same class of risk PS1 handled for the inference
  image; the smoke-check extension turns it from a runtime surprise into a named
  failure.

## Open questions

1. Prompt domain for the commissioning mini-project — any small, single-fact behavior
   of Gemma-2-2B works; picking one the operator cares about makes the post-mortem more
   useful.
2. Whether to close the persistence gap preemptively (add the bind mount now, harmless
   if unused) or only after verification shows graphs live outside Postgres.
3. Whether PS5 runs before PS4's acceptance batch — the design prefers PS5 first (the
   crisper acceptance test) unless a mini-project is pulling for feature explanations;
   both are unblocked by the same pending PS2 import.
