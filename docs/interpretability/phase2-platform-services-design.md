# Interpretability Phase 2 product design: platform services

This document is the implementation-ready product design for every Phase 2 deliverable in the interpretability roadmap ("Phase 2: Platform services"). It follows the same contract as the training-tier product designs: each section can be picked up in a fresh session and implemented without re-deriving context. It records the decisions already made, the interfaces to build, the repository and ecosystem facts the design depends on, and the tests that define done.

The roadmap defines *what* Phase 2 delivers; this document defines *how*. Where the two disagree, this document wins and the roadmap should be corrected.

Phase 2 deploys the self-hosted Neuronpedia platform — webapp, Postgres, activation-inference server, autointerp server, and attribution-graph server — against local models and local serving. It is a capability gate, not a calendar entry: it deploys when browsing features, steering interactively, or reading attribution graphs becomes the bottleneck in Phase 1 mini-project work, and the phase is complete when its acceptance criteria have been met by real project use. Every deliverable here is tagged [orchestrate]: this phase writes no interpretability code, only deployment harness, configuration, and one small vendored patch.

## Table of contents

- [Roadmap contract](#roadmap-contract)
- [Cross-cutting conventions](#cross-cutting-conventions)
- [Ecosystem facts (verified 2026-07)](#ecosystem-facts-verified-2026-07)
- [Build order](#build-order)
- [PS1: Neuronpedia deployment harness](#ps1-neuronpedia-deployment-harness)
- [PS2: Database and Gemma Scope import](#ps2-database-and-gemma-scope-import)
- [PS3: Activation inference and steering](#ps3-activation-inference-and-steering)
- [PS4: Autointerp through local serving](#ps4-autointerp-through-local-serving)
- [PS5: Attribution graphs](#ps5-attribution-graphs)
- [Phase 2 exit checklist](#phase-2-exit-checklist)

## Roadmap contract

Phase 2 consumes these interfaces as given and does not reimplement them. If an interface named here differs from what actually shipped, update this list and adapt; do not fork.

| Dependency | What Phase 2 consumes |
| :-- | :-- |
| Interp Phase 0 workbench | The `docker/interp/` workbench image and `make interp-shell` exist (or land alongside PS1). Phase 2 does not require the workbench at runtime — the platform is a separate stack — but the commissioning mini-project in PS5 runs from it. |
| Interp Phase 1 experiments log | `docs/interpretability/experiments/` convention. The PS5 acceptance requires an attribution graph to support a recorded mini-project, written up in this log. |
| Inference services | Real ports from docker-compose.yaml: llama-cpu 8001, llama-gpu 8002, vllm-gpu 8003 (documentation claiming 8004/8005 is stale; the compose file is the source of truth). PS4 drives the vllm-gpu OpenAI-compatible endpoint. Structured output on that endpoint (inference roadmap task 2.4) is a nice-to-have for autointerp scoring, not a prerequisite. |
| Storage layout | `/mnt/ai-data/interp/` as defined in the roadmap's storage section; this phase adds `neuronpedia/` under it. The existing disk-free watchdog floor applies. |
| Concurrency policy | Interactive platform work runs with GPU inference services quiesced; serving-dependent work (PS4) runs with vllm-gpu up and interp GPU work idle; llama-cpu may stay up on its dedicated cores throughout. |

## Cross-cutting conventions

Inherited unchanged from the repository: code and configuration in git, heavy data under /mnt/ai-data; Python style per .claude/PYTHON_STYLE.md and friends; no emojis, plain technical prose; every new CLI or lifecycle action gets a Makefile target, here under a new `##@ Interpretability Platform` section; secrets via an untracked `.env.interp` (HF_TOKEN for gated Gemma weights, following the training tier's `.env.training` convention).

Phase 2 additions:

- **Adopt upstream orchestration; do not port it.** Neuronpedia is a monorepo with its own Makefile and Docker Compose files. Rewriting its seven services into this repository's docker-compose.yaml would be a maintenance treadmill against an actively developed upstream. Instead, this repository holds a pinned checkout, a compose *override* file carrying only our deltas (ports, volumes, networks), and wrapper make targets. This amends the roadmap's "under the interp compose profile" wording: the workbench stays in our compose under the `interp` profile; the Neuronpedia stack runs from upstream's compose plus our override. Correct the roadmap.
- **Pin discipline.** The Neuronpedia checkout is pinned to an exact git SHA recorded in `docker/interp/neuronpedia-pin.txt` with a dated comment, same convention as the llama.cpp pin (inference roadmap 1.1). Bumping the pin means re-running the PS1 smoke checks before merging.
- **Port block.** Neuronpedia's webapp defaults to 3000, which collides with open-webui. The platform gets: webapp 3001 (host) → 3000 (container); postgres 5432; inference 5002; autointerp 5003; graph 5004. All host ports bind to 127.0.0.1 only, matching the inference services' pattern. The upstream `graph-crm` service (port 5005) is not deployed; nothing in this phase needs it.
- **Localhost trust model.** Neuronpedia's localhost mode disables sign-in (`NEXT_PUBLIC_ENABLE_SIGNIN=false`) and authenticates inter-service calls with shared static secrets (`x-secret-key`). That is acceptable only because every port binds to loopback. Do not "fix" the dummy secrets; do fail loudly (a PS1 smoke check) if any port mapping loses its 127.0.0.1 prefix.
- **Storage.** Everything heavy lives under `/mnt/ai-data/interp/neuronpedia/`: `src/` (the pinned checkout — it is multi-gigabyte with images and node_modules after builds, so it lives on NVMe, not in this repository), `postgres/` (bind-mounted database volume, replacing the upstream named volume), and the shared Hugging Face cache mount (`USE_LOCAL_HF_CACHE=1` pointed at the existing HF cache location so Gemma weights download once).
- **GPU images run under the CUDA 13 host driver.** Upstream GPU images build on CUDA 12.1 runtime bases. That is fine — the driver is backward-compatible — but it means no sm_120-specific kernels are compiled here and torch inside those images must be a version whose prebuilt wheels support sm_120. Verify at PS1 smoke time (`torch.cuda.get_device_capability()` inside the inference container); if upstream's torch pin predates sm_120 support, override the torch version via build arg or a one-line Dockerfile patch and record it with the pin.

## Ecosystem facts (verified 2026-07)

Re-verify against the pinned SHA at implementation time; upstream moves quickly.

- Neuronpedia (github.com/hijohnnylin/neuronpedia, MIT) self-hosts via `docker/compose.yaml` plus per-service overlays (`compose.inference.gpu.yaml`, `compose.autointerp.gpu.yaml`, `compose.graph.gpu.yaml`, `compose.hf-cache.yaml`) driven by upstream make targets (`webapp-localhost-*`, `inference-localhost-dev-gpu MODEL_SOURCESET=...`, `autointerp-localhost-dev-gpu`, `graph-localhost-dev-gpu`).
- Services and container ports: webapp (Next.js, node:22-alpine) 3000; db-init (Prisma migrations, run-once); postgres (pgvector/pgvector:pg15, user/pass/db all `postgres`) 5432; inference (Python 3.10, FastAPI) 5002; autointerp (FastAPI, wraps EleutherAI Delphi) 5003; graph (FastAPI, wraps safety-research/circuit-tracer, MIT) 5004.
- Root env file `.env.localhost` carries the knobs this design touches: `PORT`, `NEXT_PUBLIC_URL`, `NEXTAUTH_URL` (all follow the webapp port remap), `USE_LOCALHOST_INFERENCE=true`, `USE_LOCALHOST_AUTOINTERP` (flip to true), the three `*_SERVER_SECRET`s, and `OPENAI_API_KEY` (used only for explanation-search embeddings; see PS2 risk).
- Inference instances are configured per model/source-set via `.env.inference.<model>.<sourceset>` files; `gemma-2-2b-it.gemmascope-res-16k` ships as a preset. **Localhost mode runs exactly one inference instance at a time.**
- SAE sources import through the webapp admin panel at `/admin`: pick model → source set → per-layer download from the public S3 dataset dumps. Imports take tens of minutes per source set and **cannot be resumed if interrupted**. Per-source-set database size is not published; treat it as unknown until the first import (PS2 measures it).
- Autointerp is Delphi-based; the explainer LLM is set **per request** (`model` plus `openrouter_key` fields), and the bundled client targets OpenRouter's API URL. No verified env var redirects it to an arbitrary OpenAI-compatible base URL — PS4 owns this gap.
- Graph server VRAM: Gemma-2-2B with the 8x/k64 transcoder config runs in well under 32GB (upstream demonstrates it on 15GB-class GPUs; the 32x configs need 40GB+ and are out of envelope). Neuronpedia's graph app documents Gemma-2-2B and Qwen3-4B; upstream circuit-tracer additionally lists Llama-3.2-1B — whether the Neuronpedia wrapper exposes Llama-3.2-1B is unverified, so this design commits only to Gemma-2-2B.
- Practical minima: ≥16GB RAM for the stack (trivial on 128GB), images are linux/amd64, gated Gemma weights need HF_TOKEN.

## Build order

Dependencies among deliverables, not calendar:

PS1 harness ── PS2 import ──┬── PS3 inference + steering
                            ├── PS4 autointerp (also needs vllm-gpu up)
                            └── PS5 attribution graphs

PS1 is pure plumbing and everything sits on it. PS2 before PS3 because the steering UI needs imported sources to browse. PS4 and PS5 are independent of each other; do PS4 first only if an active mini-project is pulling for feature explanations, otherwise PS5 first (it has the crisper acceptance test). Each of PS3–PS5 corresponds to one roadmap acceptance criterion.

## PS1: Neuronpedia deployment harness

**Tag:** orchestrate. **Depends on:** nothing.

### Purpose

One command brings the platform up in a known state from a pinned upstream revision, with this repository's ports, storage, and network conventions applied as a thin override — and one command tears it down.

### Decisions

1. **Pinned checkout at `/mnt/ai-data/interp/neuronpedia/src`,** cloned and checked out to the SHA in `docker/interp/neuronpedia-pin.txt` by a `make np-clone` target (idempotent: fetch + checkout if the directory exists). The checkout is not a git submodule: it is heavy, mutable (builds write into it), and lives outside the repository by the storage convention.
2. **Override file `docker/interp/neuronpedia.override.yaml`** checked into this repository, passed with `-f` after upstream's compose files. It carries only: host port remaps (3001/5432/5002/5003/5004, all 127.0.0.1-bound), the postgres bind mount to `/mnt/ai-data/interp/neuronpedia/postgres`, the HF-cache mount, and an external network attachment joining the autointerp container to this repository's compose network so `http://vllm-gpu:8003/v1` resolves directly (PS4). Everything else stays upstream's.
3. **Env file `docker/interp/neuronpedia.env`** checked in, seeding `.env.localhost` deltas: `NEXT_PUBLIC_URL=http://localhost:3001`, `NEXTAUTH_URL=http://localhost:3001`, `USE_LOCALHOST_AUTOINTERP=true`. Secrets (HF_TOKEN, optional OPENAI_API_KEY) come from the untracked `.env.interp` and are merged at invocation, never committed.
4. **Make targets** in the `##@ Interpretability Platform` section: `np-clone`, `np-build`, `np-up` (webapp + postgres + db-init), `np-inference-up MODEL_SOURCESET=gemma-2-2b-it.gemmascope-res-16k`, `np-autointerp-up`, `np-graph-up`, `np-down`, `np-status`. Targets shell into the pinned checkout's compose with the override; they do not duplicate upstream logic. `np-up` refuses to start (clear message, nonzero exit) if `make status` shows llama-gpu or vllm-gpu running with a GPU-attached platform service requested — the concurrency policy made structural. `np-down` is always safe.
5. **Smoke check script** `scripts/interp/np_smoke.sh`: asserts every mapped port is loopback-bound (`ss -tlnp` grep), webapp answers on 3001, postgres accepts connections, and — when the GPU inference instance is up — the container sees the RTX 5090 with a torch build that supports sm_120.

### Interfaces

- `make np-clone && make np-build && make np-up` from a clean machine state.
- `docker/interp/neuronpedia-pin.txt`, `docker/interp/neuronpedia.override.yaml`, `docker/interp/neuronpedia.env`, `scripts/interp/np_smoke.sh`.
- A runbook page `docs/interpretability/platform.md`: port map, start/stop matrix per concurrency mode, pin-bump procedure, and the import runbook from PS2.

### Risks

- Upstream compose layout churn: the override references upstream service names; a pin bump can rename them. Mitigation: the pin-bump procedure requires `np-smoke` green before merge.
- Two compose projects sharing one network: the external-network attachment requires the inference stack's network to exist even when its services are down; document `docker network` behavior in the runbook (a stopped stack keeps its network unless `down` removed it — the runbook's start matrix covers the order).

### Acceptance

- From a machine with neither stack running: `make np-up` then `np-smoke` passes; `make np-down` returns to clean state.
- All five host ports verified loopback-only.
- `make np-up` with llama-gpu running and a GPU platform service requested fails with the policy message, not a CUDA OOM.

## PS2: Database and Gemma Scope import

**Tag:** orchestrate. **Depends on:** PS1.

### Purpose

A populated local database: Gemma Scope residual-stream sources for Gemma-2-2B imported, browsable, and persistent across `np-down`.

### Decisions

1. **Import scope: `gemmascope-res-16k` for gemma-2-2b(-it), all layers.** This is the reference dictionary set the roadmap names and the preset the inference server ships. Wider sets (65k, attention/MLP sites) are pulled later by projects that need them, per the research-pulls-infrastructure rule.
2. **Import through the admin panel** (`localhost:3001/admin`), per layer, as upstream intends. No custom import tooling: the admin flow is the supported path and this phase is orchestrate-only.
3. **Non-resumable imports get an operational rule, not code:** import one layer at a time, verify it appears in the dashboard, then proceed. The runbook records per-layer wall-clock and resulting Postgres size from the first real import (currently unpublished upstream; measuring it is part of this deliverable's output).
4. **Database persistence via the PS1 bind mount;** back up by stopping postgres and snapshotting the directory. Given imports are re-downloadable, retention policy is "don't be precious": the database is regenerable at the cost of import hours, so no automated backup job. Record the measured size against the 1–2TB interp storage budget.
5. **Explanation-search embeddings** call OpenAI via `OPENAI_API_KEY`. Decision: leave unset by default; feature search degrades gracefully to activation search. Revisit only if a project pulls semantic explanation search, and prefer routing to a local embedding endpoint if upstream supports a base-URL override by then.

### Acceptance

- Feature dashboards and activation search work against the imported Gemma Scope sources on localhost:3001 (first half of the roadmap's criterion 1).
- Sources survive `make np-down && make np-up`.
- The runbook records measured import time and database size per layer.

## PS3: Activation inference and steering

**Tag:** orchestrate. **Depends on:** PS1, PS2.

### Purpose

The platform's live half: the local inference server computing activations and steered generations for Gemma-2-2B, driving feature testing and the steering UI.

### Decisions

1. **One instance, GPU build, `gemma-2-2b-it.gemmascope-res-16k`.** Localhost mode supports a single inference instance; the instruct model is the right default because the steering UI's output is judged on chat behavior (and upstream's `NEXT_PUBLIC_STEER_FORCE_ALLOW_INSTRUCT_MODELS` already allowlists it). Switching model/source-set = `np-down` the instance, `np-inference-up` with a different `MODEL_SOURCESET`; the runbook lists available presets.
2. **Interactive concurrency mode:** the inference instance owns the GPU while up, so it runs only with llama-gpu/vllm-gpu quiesced (enforced by the PS1 preflight). llama-cpu may stay up.
3. **Weights via the shared HF cache mount** with HF_TOKEN from `.env.interp`; the model downloads once, ever.
4. **No custom steering tooling in this phase.** The steering UI exercises the platform; the production write-back path (repeng, GGUF control vectors, scorecards) is Phase 3 and is explicitly not pulled forward here.

### Acceptance

- A feature from an imported source shows live max-activating behavior on user-supplied text (activation testing round-trips webapp → inference server → webapp).
- The steering UI produces visibly steered generations for a known Gemma Scope feature at nonzero strength versus strength zero, on localhost (completes the roadmap's criterion 1).

## PS4: Autointerp through local serving

**Tag:** orchestrate. **Depends on:** PS1, PS2; vllm-gpu up (serving-dependent mode).

### Purpose

Delphi-based feature explanation running as a local batch workload against the vllm-gpu OpenAI-compatible endpoint — no external API spend, exercising the inference-to-interpretability integration the roadmap promises.

### Decisions

1. **Explainer endpoint is vllm-gpu:** `http://vllm-gpu:8003/v1` over the shared network from PS1, reachable because the autointerp container joins the inference compose network. The explainer model is whatever vllm-gpu currently serves; the request's `model` field must match the served model id (`/v1/models` is the source of truth).
2. **The OpenRouter-shaped hole is closed with a minimal vendored patch.** The autointerp route constructs its client with a hardcoded OpenRouter URL and a per-request key. Implementation order: (a) re-verify at the pinned SHA whether a base-URL override now exists (env var or request field) — upstream moves and this is the kind of gap that gets fixed; (b) if not, patch the client construction to honor an `OPENROUTER_BASE_URL` env var, defaulting to upstream's URL so the patch is upstreamable, and carry it as a single-commit branch on the pinned checkout recorded next to the pin; (c) open the upstream PR — the patch dies when it merges. The per-request `openrouter_key` becomes a dummy string (vLLM ignores it). Do not fork more than this one construction site.
3. **Serving-dependent concurrency mode, made structural:** `make np-autointerp-up` preflights that vllm-gpu is healthy (its /health endpoint) and that no GPU platform service (PS3 inference, PS5 graph) is up. Autointerp itself is CPU-side except Delphi's embedding scorer (CUDA-only); explanation and the non-embedding scorers are the Phase 2 surface — the embedding scorer is exercised only if the GPU is otherwise idle, and its result is a nice-to-have.
4. **Batch shape.** Autointerp cost is tens of millions of explainer tokens for thousands of features; Phase 2's acceptance batch is deliberately small (one source/layer, on the order of tens of features) to prove the path, not to explain the dictionary. Full-dictionary runs are project-pulled, sized with the vllm-gpu throughput numbers, and run overnight like benchmarks.
5. **Explanations land in the local database** and render on the feature dashboards, closing the loop visibly.

### Risks

- The vendored patch is the phase's only code-shaped risk: if the client library resists a clean base-URL override (e.g. OpenRouter-specific response parsing), the fallback is a 20-line reverse proxy container that speaks OpenRouter's chat-completions shape to the client and forwards to vLLM — still orchestration, no Delphi changes. Decide after (a); do not build the proxy speculatively.
- Explainer quality depends on the served model; a weak local model produces weak explanations. That is acceptable for Phase 2 (the criterion is the path, not explanation quality) and is a routing decision for later phases.

### Acceptance

- An autointerp batch over at least 20 features of one imported Gemma Scope source completes using only local inference (roadmap criterion 2), with vllm-gpu request logs as evidence that no external endpoint was touched.
- Generated explanations are visible on the corresponding feature pages.
- The vendored patch, if needed, exists as one commit referenced from the pin file, with the upstream PR link.

## PS5: Attribution graphs

**Tag:** orchestrate. **Depends on:** PS1, PS2 (viewer); interactive concurrency mode.

### Purpose

circuit-tracer attribution graphs generated locally on Gemma-2-2B and rendered in the Neuronpedia graph viewer, supporting hypothesis work in mini-projects.

### Decisions

1. **Model commitment: Gemma-2-2B with the 8x/k64 transcoder configuration** — verified inside the 32GB envelope. Larger configs (32x, 40GB+) are out of envelope and out of scope; other models (Qwen3-4B, Llama-3.2-1B) are attempted only when a project pulls them, and the Llama support question in the Neuronpedia wrapper is resolved then, not now.
2. **GPU graph server via `make np-graph-up`,** interactive mode (GPU inference services down, PS3 instance down — one GPU tenant at a time, enforced by the PS1 preflight). Transcoder weights come through the shared HF cache.
3. **Generation through the graph server's API / webapp UI,** viewed in the webapp's graph viewer with pinning and annotation. Local mode keeps graphs in the local stack; the hosted S3-upload path is not configured.
4. **Commissioning mini-project (the acceptance vehicle):** reprise the Phase 1 menu item "form a hypothesis from an attribution graph and verify it by prompting alone," but on a locally generated graph: pick a prompt, generate the graph, form a mechanistic hypothesis from it, test the hypothesis with prompt interventions through the ordinary serving stack, and write the post-mortem in `docs/interpretability/experiments/` including a falsification attempt, per the standards for results.

### Risks

- Transcoder/CLT artifact downloads are large (tens of GB); they land in the HF cache and count against the interp storage budget — record actual size in the runbook.
- Graph generation latency on first run (weight load + compile) can look like a hang; the runbook notes expected first-run versus warm timings once measured.

### Acceptance

- An attribution graph generated locally on Gemma-2-2B renders in the local viewer with working node inspection and pinning.
- The commissioning mini-project post-mortem exists in the experiments log and cites the graph (roadmap criterion 3).

## Phase 2 exit checklist

Restating the roadmap's acceptance criteria against the deliverables that discharge them:

- [ ] Feature dashboards, activation search, and the steering UI work against imported Gemma Scope sources on localhost (PS2 + PS3).
- [ ] An autointerp batch completes using only local inference, explanations visible on feature pages (PS4).
- [ ] A locally generated attribution graph renders in the viewer and supports a recorded mini-project with post-mortem (PS5).
- [ ] Operational: `np-up`/`np-down` round-trip cleanly; all ports loopback-bound; concurrency preflights refuse GPU contention; pin, override, env template, patch (if any), and runbook are in git; measured import sizes and times recorded (PS1 + PS2).

Phase 2 remains "deployed, not done" until the three roadmap criteria have been met by real project use, per the capability-gate rule.

*Last updated: 2026-07-10*
