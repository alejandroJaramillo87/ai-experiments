# Interpretability PS4 autointerp through local serving: implementation investigation

Investigation and repository-side implementation of interp Phase 2 deliverable PS4
(autointerp through local serving). The planning document's design is taken as intent;
the repository is the source of truth. PS4 is tagged orchestrate and depends on PS1
(merged) and PS2 (imported sources to explain; procedure investigated in
`ps2_gemma_scope_import_investigation.md`, execution pending on the host). The repository
side of PS4 — preflight, make wiring, env delta, patch-carry mechanism, runbook — lands
now; the acceptance batch runs host-side once PS2's import exists.

## Table of contents

- [What PS4 delivers](#what-ps4-delivers)
- [Design-versus-repository deviations](#design-versus-repository-deviations)
- [Serving-dependent preflight](#serving-dependent-preflight)
- [Explainer routing and the OpenRouter-shaped hole](#explainer-routing-and-the-openrouter-shaped-hole)
- [Batch shape](#batch-shape)
- [Repository changes](#repository-changes)
- [Upstream facts to re-verify at implementation time](#upstream-facts-to-re-verify-at-implementation-time)
- [Acceptance checks and host-side split](#acceptance-checks-and-host-side-split)
- [Risks](#risks)
- [Open questions](#open-questions)

## What PS4 delivers

Delphi-based feature explanation running as a local batch workload against the vllm-gpu
OpenAI-compatible endpoint (`http://vllm-gpu:8003/v1` over the shared `ai-network`), with
no external API spend. Explanations land in the local database and render on the feature
dashboards. This exercises the inference-to-interpretability integration the roadmap
promises and is roadmap acceptance criterion 2.

## Design-versus-repository deviations

PS1 landed most of the plumbing PS4 assumes. Verified against the repository:

| Design says | Repository reality | Resolution |
|-------------|--------------------|------------|
| Autointerp container joins the inference compose network | `neuronpedia.override.yaml:48` already attaches autointerp to `ai-network` (external, `AI_NETWORK_NAME`) | Consistent; `http://vllm-gpu:8003/v1` resolves once vllm-gpu's stack has created the network |
| `make np-autointerp-up` preflights that vllm-gpu is healthy and no GPU platform service is up | The target existed but ran no preflight in CPU mode (only `NP_GPU=1` ran the standard GPU-idle preflight) | New `scripts/interp/np_autointerp_preflight.sh` wired as the default path; `NP_GPU=1` keeps the standard preflight for the embedding-scorer mode |
| Vendored patch carried "as a single-commit branch on the pinned checkout" | The checkout is disposable: `make np-clone` runs `checkout -f` and discards local state, so a branch on the checkout does not survive the harness | Deviation: the patch is carried as a `.patch` file in `docker/interp/patches/`, applied by `np-clone` after checkout — same idempotence guarantee, survives re-clone, reviewed in this repository |
| `USE_LOCALHOST_AUTOINTERP=true` | Already set in `docker/interp/neuronpedia.env-deltas` (PS1) | No change |
| Serving-dependent mode in the concurrency matrix | `platform.md` matrix already lists the mode (vllm-gpu up, autointerp CPU) | Preflight now makes the documented mode structural |

## Serving-dependent preflight

`scripts/interp/np_autointerp_preflight.sh`, the inverse of the interactive preflight:

1. vllm-gpu healthy: `curl -sf http://127.0.0.1:8003/health` (loopback host port per
   `docker-compose.yaml`).
2. No other inference-stack GPU consumer running: llama-gpu, comfyui (llama-cpu
   permitted, as everywhere).
3. No Neuronpedia GPU service running: the PS3 inference instance or PS5 graph server,
   detected via the compose project label (`com.docker.compose.project=neuronpedia`,
   service `inference` or `graph`) — matching how `make np-status` finds them.

`make np-autointerp-up` runs it by default; `NP_GPU=1` (embedding-scorer mode, GPU
attached to autointerp itself) runs the standard GPU-idle preflight instead. The two
modes are mutually exclusive by construction: the scorer mode requires the GPU free,
which means vllm-gpu down, which means no local explainer. The embedding scorer is a
nice-to-have exercised only when the GPU is otherwise idle, per the planning decision.

## Explainer routing and the OpenRouter-shaped hole

The autointerp route constructs its client with a hardcoded OpenRouter URL and a
per-request `openrouter_key`. Implementation order per the planning decision:

1. Re-verify at pin `e5488b2a` whether a base-URL override now exists (env var or request
   field). Upstream moves fast and this is the kind of gap that gets fixed. This
   verification requires the checkout, which lives on the host
   (`/mnt/ai-data/interp/neuronpedia/src`) — it is a host-side step; the sandbox cannot
   read it.
2. If no override exists: patch the single client-construction site to honor
   `OPENROUTER_BASE_URL`, defaulting to upstream's URL so the patch is upstreamable.
   The patch lands as `docker/interp/patches/0001-autointerp-openrouter-base-url.patch`,
   applied automatically by `make np-clone` (mechanism merged now; see the directory
   README for the rules).
3. Open the upstream PR; the patch is deleted when the PR lands in a pin bump.

Repository side landed ahead of the verification because it is harmless either way:
`OPENROUTER_BASE_URL=http://vllm-gpu:8003/v1` is in `neuronpedia.env-deltas` (ignored
until something honors it), and the patch directory is a no-op while empty. The
per-request `openrouter_key` becomes a dummy string; vLLM ignores it. Do not fork more
than the one construction site.

## Batch shape

Autointerp cost at full scale is tens of millions of explainer tokens for thousands of
features. The Phase 2 acceptance batch is deliberately small: one imported Gemma Scope
source/layer, at least 20 features — proving the path, not explaining the dictionary.
Full-dictionary runs are project-pulled, sized with the vllm-gpu throughput numbers from
the benchmark suite, and run overnight like benchmarks. Explainer quality tracks the
served model and is explicitly acceptable at whatever vllm-gpu currently serves; the
criterion is the path.

## Repository changes

| Path | Change |
|------|--------|
| `scripts/interp/np_autointerp_preflight.sh` | New serving-dependent preflight (vllm-gpu healthy, all other GPU consumers down) |
| `Makefile` | `np-autointerp-up` default path runs the serving preflight; `np-clone` applies `docker/interp/patches/*.patch` after checkout |
| `docker/interp/neuronpedia.env-deltas` | `OPENROUTER_BASE_URL=http://vllm-gpu:8003/v1` delta |
| `docker/interp/patches/README.md` | Patch-carry rules; directory applied by `np-clone` |
| `docs/interpretability/platform.md` | "Autointerp through local serving" runbook section; patches row in the layout table; pin-bump step re-verifying patches |
| `docs/interpretability/ps4_autointerp_investigation.md` | This document |

Deferred to host-side implementation time: the upstream base-URL verification and, if
needed, the patch file itself plus the upstream PR.

## Upstream facts to re-verify at implementation time

Against the checkout at pin `e5488b2a`, on the host:

- Whether the autointerp client construction already honors a base-URL env var or
  request field; if so, adopt it and adjust the env-delta key name to match — the
  `OPENROUTER_BASE_URL` name is this repository's proposal, not a verified upstream knob.
- How autointerp receives environment from compose: the make target passes
  `.env.localhost` via `ENV_FILE` and `--env-file`; confirm the delta key actually
  reaches the container process (PS1 verified this path for `USE_LOCALHOST_AUTOINTERP`).
- The exact request fields for an explanation batch from the webapp (model id,
  `openrouter_key`, feature selection), and whether the batch is driven from the feature
  dashboard UI or an admin/API route.
- Delphi scorer split: explanation and non-embedding scorers CPU-side; embedding scorer
  CUDA-only. Confirm the CPU-mode container degrades gracefully when the scorer is
  requested without a GPU.
- Served-model id matching: the request's `model` field must equal an id from
  `http://127.0.0.1:8003/v1/models`.

## Acceptance checks and host-side split

All execution host-side; sandbox contributed the orchestration and records reported
output verbatim.

1. Batch over at least 20 features of one imported Gemma Scope source completes using
   only local inference. Evidence: vllm-gpu request logs
   (`docker compose logs vllm-gpu`) covering the batch window, showing the explainer
   traffic; no external endpoint configured or reachable.
2. Generated explanations visible on the corresponding feature pages on
   `localhost:3001`.
3. The vendored patch, if needed, exists in `docker/interp/patches/` with the upstream
   PR link in its header; `make np-clone && make np-build` green with it applied.

## Risks

- **The vendored patch is the phase's only code-shaped risk.** If the client library
  resists a clean base-URL override (OpenRouter-specific response parsing, for example),
  the fallback is a small reverse-proxy container speaking OpenRouter's chat-completions
  shape and forwarding to vLLM — still orchestration, no Delphi changes. Decide after
  the verification step; do not build the proxy speculatively.
- **Env-var delivery.** If `.env.localhost` deltas do not reach the autointerp process
  environment, the override has to move into the compose override's `environment` block
  for the autointerp service — a one-line change to `neuronpedia.override.yaml`.
- **Network liveness ordering.** `ai-network` exists only while the inference stack is
  up (or not downed); the start order in the runbook (vllm-gpu first) makes the
  preflight's health check double as the network-existence check.
- **Explainer quality.** Tracks the served model; acceptable for Phase 2 by design.

## Open questions

1. Which model vllm-gpu should serve for the acceptance batch — whatever is currently
   configured is acceptable per the design; a stronger instruct model produces more
   legible explanations if a choice is available.
2. Whether to exercise the `NP_GPU=1` embedding-scorer mode at all in Phase 2 (requires
   a separate GPU-idle session; the result is a nice-to-have).
