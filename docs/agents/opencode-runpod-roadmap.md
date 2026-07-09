# OpenCode and RunPod integration roadmap

This document is an implementation roadmap for the "OpenCode and RunPod integration" work. It is written to be executed task by task with Claude Code: each task states the files involved, the change to make, and the acceptance criteria that define done. Findings and version references are current as of July 2026; re-verify versions, prices, and endpoint URLs at implementation time.

Goal: run OpenCode (the open-source terminal coding agent, github.com/anomalyco/opencode) as a daily coding agent on the workstation, backed by two tiers of inference:

- Local tier: the existing containers in this repo (llama-cpu on 8001, llama-gpu on 8002, vllm-gpu on 8003), which already expose OpenAI-compatible endpoints on 127.0.0.1.
- Cloud tier: RunPod, for models that do not fit the RTX 5090's 32GB VRAM, using the serverless vLLM worker's OpenAI-compatible endpoint.

Plus a usage-tracking layer covering both tiers, since neither OpenCode nor RunPod gives a complete picture on its own.

Ordering principle: account and safety rails first, then local integration (zero marginal cost, validates the OpenCode config mechanics), then cloud integration, then tracking, then automation. Every phase ends with a verifiable acceptance check.

## Background findings (July 2026)

- OpenCode is at v1.17.x with near-daily releases. Install via `curl -fsSL https://opencode.ai/install | bash`, npm, or brew. Config lives in `~/.config/opencode/opencode.json` (global) merged with a project-root `opencode.json`; `{env:VAR}` substitution is supported.
- Custom OpenAI-compatible providers are configured with a `provider` block using `"npm": "@ai-sdk/openai-compatible"` and a `baseURL`. The model key must exactly match the model ID the backend reports on `/v1/models`.
- OpenCode has no built-in per-session cost display (open feature request, anomalyco/opencode#13003). Token usage is persisted in local storage and readable by community tools (opencode-tokenscope, opencode-token-monitor plugin, opencode-usage on PyPI).
- RunPod's serverless vLLM worker (runpod-workers/worker-vllm) exposes an OpenAI-compatible endpoint at `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1`, authenticated with the RunPod API key. Per-second billing, scale-to-zero. On-demand pods are an alternative (`https://<pod-id>-<port>.proxy.runpod.net/v1`) when a long interactive session makes cold starts annoying.
- Reference prices (Secure Cloud, verify before committing): RTX 5090 ~$0.99/hr, H100 ~$2.89-3.29/hr, H200 ~$4.39/hr, B200 ~$5.89/hr; Community Cloud ~20-40% cheaper. Network volumes $0.07/GB/mo. Serverless bills per second with Flex (cold) and Active (warm) worker pricing.
- RunPod spend is queryable programmatically via the GraphQL API (`myself { currentSpendPerHr spendLimit clientLifetimeSpend }`); no fine-grained REST billing export was verified.
- Tool calling on self-hosted vLLM requires launching with `--enable-auto-tool-choice --tool-call-parser <parser>` appropriate to the model family; agent quality depends heavily on the model's tool-calling ability.

## Phase 1: RunPod account and safety rails

No repo files; account-level work.

Tasks:

- Create the RunPod account and enable two-factor auth.
- Create an API key scoped to what is needed (serverless invoke + pod management); store it outside the repo (e.g. `~/.config/runpod/credentials` or the shell environment as `RUNPOD_API_KEY`). Never commit it.
- Set a hard spend limit and a low auto-reload threshold in the billing console; enable budget alert emails. Start with a small credit load (e.g. $25) rather than a card on file with high auto-reload.
- Record account decisions (datacenter region, Secure vs Community Cloud default, spend limit) in `docs/agents/runpod.md` as they are made.

Acceptance:

- `curl -s https://api.runpod.io/graphql -H "Authorization: Bearer $RUNPOD_API_KEY" -d '{"query":"query { myself { clientLifetimeSpend spendLimit } }"}' -H 'content-type: application/json'` returns the account's spend fields.
- A spend limit is visible in the dashboard and documented.

## Phase 2: OpenCode against the local containers

Files: new `opencode.json` at the repo root (project config, committed); global config `~/.config/opencode/opencode.json` (not committed, documented).

Tasks:

- Install OpenCode via the curl script; pin the installed version in the doc when done.
- Add a project `opencode.json` defining the three local providers, e.g.:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llama-gpu": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama.cpp GPU (local)",
      "options": { "baseURL": "http://127.0.0.1:8002/v1" },
      "models": { "<served-model-id>": { "limit": { "context": 32768 } } }
    }
  }
}
```

  Repeat for llama-cpu (8001) and vllm-gpu (8003). Query each service's `/v1/models` to get the exact model IDs; the key must match verbatim. Set `limit.context` to the value each container is actually launched with, not the model's theoretical maximum.
- Confirm the vllm-gpu entrypoint passes `--enable-auto-tool-choice` and a `--tool-call-parser` matching the served model; add them if missing (files: `docker/vllm-gpu/`, `docker-compose.yaml`). For llama.cpp, verify `llama-server` is recent enough that tool calls work through `/v1/chat/completions` (it is in current builds; the pin task in the inference roadmap governs the version).
- Test: `make up`, then `opencode run "list the files in this repo and summarize the Makefile"` against each provider (`opencode run -m llama-gpu/<model> ...`). Exercise at least one tool call (file read) and one edit in a scratch directory.
- Document the setup, including which local model handles agentic tool calling acceptably, in `docs/agents/opencode.md`.

Acceptance:

- `opencode run` completes a multi-step tool-using task against llama-gpu and vllm-gpu without provider errors.
- The committed `opencode.json` contains no secrets and works on a fresh clone with services up.

## Phase 3: RunPod cloud tier

Files: `opencode.json`, `docs/agents/runpod.md`.

Tasks:

- Decide the first cloud model: something materially beyond local capacity (e.g. a 70B+ class open-weight coder that needs more than 32GB VRAM). Record the choice and rationale.
- Deploy it via the RunPod Hub vLLM quick-deploy (worker-vllm): select model, GPU type, set `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, and tool-calling env/args for the model family. Start with Flex workers, max workers 1, scale-to-zero enabled.
- If model weights exceed what bakes reasonably into the worker, attach a network volume for the HF cache so cold starts do not re-download.
- Add the provider to `opencode.json`:

```json
"runpod": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "RunPod vLLM",
  "options": {
    "baseURL": "https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1",
    "apiKey": "{env:RUNPOD_API_KEY}"
  },
  "models": { "<served-model-id>": { "limit": { "context": 65536 } } }
}
```

- Measure cold-start latency for the first request after scale-to-zero. If it is unacceptable for interactive use, evaluate: (a) an Active/min-1 worker during work hours, (b) an on-demand pod started and stopped via the `runpod` CLI or Python SDK for planned sessions, exposed through the pod proxy URL. Record measured cold start, tokens/s, and effective $/hr for each option.
- Test the same multi-step agentic task from Phase 2 against the RunPod provider.

Acceptance:

- `opencode run -m runpod/<model> "<task>"` completes a tool-using task end to end.
- Cold-start and throughput numbers plus a chosen serving mode (serverless flex, warm worker, or pod) are documented with costs in `docs/agents/runpod.md`.
- The endpoint scales to zero (or the pod is stopped) after the test; dashboard shows no idle spend.

## Phase 4: Usage tracking

Files: new `scripts/usage/opencode_usage.py` (or shell), `docs/agents/usage-tracking.md`, Makefile.

Neither side alone answers "what did this week of agent use cost": OpenCode records tokens but not RunPod dollars (time-billed, not token-billed); RunPod reports dollars but not which sessions consumed them.

Tasks:

- OpenCode side: adopt one community token reporter — evaluate opencode-tokenscope and the opencode-token-monitor plugin; prefer whichever reads the local session storage without patching OpenCode. For local providers, token counts are the whole story (marginal cost is electricity); no prices needed.
- RunPod side: write a small script that queries the GraphQL API for `currentSpendPerHr`, `clientLifetimeSpend`, and spend details, and prints a delta since last invocation (state file in the repo's scratch/ or `~/.local/state`). Python with plain `requests`; follow `.claude/PYTHON_STYLE.md`.
- Combine: a `make agent-usage` target that prints OpenCode token totals per provider (from the chosen reporter) alongside RunPod lifetime/period spend.
- Re-check upstream monthly: if OpenCode ships native usage reporting (issue #13003), drop the community tool and simplify.

Acceptance:

- `make agent-usage` runs offline-safe (local tokens always print; RunPod section degrades gracefully without network or key) and shows: tokens per provider, RunPod spend to date.
- After one real RunPod session, the reported spend delta matches the dashboard to within a billing-granularity rounding error.

## Phase 5: Hardening and automation

Files: Makefile, `docs/agents/README.md`, `.gitignore`.

Tasks:

- Makefile targets: `agent-usage` (Phase 4), optionally `runpod-up`/`runpod-down` wrapping pod start/stop via the RunPod SDK if the pod path was chosen in Phase 3.
- `.gitignore`: ensure OpenCode local state and any usage state files are excluded; verify `auth.json` (`~/.local/share/opencode/`) never lands in the repo.
- Write `docs/agents/README.md` tying together opencode.md, runpod.md, and usage-tracking.md, including a fresh-machine bootstrap checklist (install OpenCode, export RUNPOD_API_KEY, make up, verify providers).
- Update CLAUDE.md's service/workflow sections to mention the agent tier once it is real.

Acceptance:

- A fresh clone plus the documented bootstrap steps yields a working `opencode run` against local and RunPod providers.
- `git grep -iE 'runpod.{0,40}(key|token)'` finds no committed secrets.

## Deferred / watch list

- OpenCode Zen gateway: an alternative cloud tier with per-token pricing and built-in usage dashboards; revisit if RunPod self-hosting proves more expensive or higher-friction than expected for frontier-model access.
- OpenCode server mode (`opencode serve`) for driving the agent programmatically from other tooling in this repo.
- SkyPilot/dstack over RunPod for reproducible pod provisioning if manual deploys become repetitive.
- Native OpenCode usage reporting (anomalyco/opencode#13003) — replaces the Phase 4 community tool when it lands.
