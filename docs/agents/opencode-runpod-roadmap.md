# OpenCode and RunPod integration roadmap

This document is an implementation roadmap for the "OpenCode and RunPod integration" work. It is written to be executed task by task with Claude Code: each task states the files involved, the change to make, and the acceptance criteria that define done. Findings and version references are current as of July 2026; re-verify versions, prices, and endpoint URLs at implementation time.

Goal: run OpenCode (the open-source terminal coding agent, github.com/anomalyco/opencode) as a daily coding agent on the workstation, backed by open-source frontier models served on RunPod. The core deliverable is the configuration: a committed `opencode.json` that points OpenCode at a RunPod serverless vLLM endpoint hosting a frontier open-weight model, plus the account safety rails and usage tracking around it.

Ordering principle: account and safety rails first, then endpoint deployment, then OpenCode provider configuration, then tracking, then automation. Every phase ends with a verifiable acceptance check.

## Background findings (July 2026)

- OpenCode is at v1.17.x with near-daily releases. Install via `curl -fsSL https://opencode.ai/install | bash`, npm, or brew. Config lives in `~/.config/opencode/opencode.json` (global) merged with a project-root `opencode.json`; `{env:VAR}` substitution is supported.
- Custom OpenAI-compatible providers are configured with a provider block using `"npm": "@ai-sdk/openai-compatible"` and a `baseURL`. The model key must exactly match the model ID the backend reports on `/v1/models`.
- OpenCode has no built-in per-session cost display (open feature request, anomalyco/opencode#13003). Token usage is persisted in local storage and readable by community tools (opencode-tokenscope, opencode-token-monitor plugin, opencode-usage on PyPI).
- RunPod's serverless vLLM worker (runpod-workers/worker-vllm) exposes an OpenAI-compatible endpoint at `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1`, authenticated with the RunPod API key. Per-second billing, scale-to-zero. On-demand pods are an alternative (`https://<pod-id>-<port>.proxy.runpod.net/v1`) when a long interactive session makes cold starts annoying.
- Reference prices (Secure Cloud, verify before committing): RTX 5090 ~$0.99/hr, H100 ~$2.89-3.29/hr, H200 ~$4.39/hr, B200 ~$5.89/hr; Community Cloud ~20-40% cheaper. Network volumes $0.07/GB/mo. Serverless bills per second with Flex (cold) and Active (warm) worker pricing.
- RunPod spend is queryable programmatically via the GraphQL API (`myself { currentSpendPerHr spendLimit clientLifetimeSpend }`); no fine-grained REST billing export was verified.
- Tool calling on vLLM requires launching with `--enable-auto-tool-choice --tool-call-parser <parser>` appropriate to the model family; agent quality depends heavily on the model's tool-calling ability.

## Phase 1: RunPod account and safety rails

No repo files; account-level work.

Tasks:

- Create the RunPod account and enable two-factor auth.
- Create an API key scoped to what is needed (serverless invoke + pod management); store it outside the repo (e.g. `~/.config/runpod/credentials` or the shell environment as `RUNPOD_API_KEY`). Never commit it.
- Set a hard spend limit and a low auto-reload threshold in the billing console; enable budget alert emails. Start with a small credit load (e.g. $25) rather than a card on file with high auto-reload.
- Record account decisions (datacenter region, Secure vs Community Cloud default, spend limit) in docs/agents/runpod.md as they are made.

Acceptance:

- `curl -s https://api.runpod.io/graphql -H "Authorization: Bearer $RUNPOD_API_KEY" -d '{"query":"query { myself { clientLifetimeSpend spendLimit } }"}' -H 'content-type: application/json'` returns the account's spend fields.
- A spend limit is visible in the dashboard and documented.

## Phase 2: Deploy the frontier model endpoint

Files: docs/agents/runpod.md.

Tasks:

- Decide the first frontier open-weight model (e.g. a 70B+ class open-weight coder). Weigh tool-calling quality first, then context length and $/hr on the GPU class it needs. Record the choice and rationale in docs/agents/runpod.md.
- Deploy it via the RunPod Hub vLLM quick-deploy (worker-vllm): select model, GPU type, set `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, and the tool-calling env/args for the model family (`--enable-auto-tool-choice --tool-call-parser <parser>`). Start with Flex workers, max workers 1, scale-to-zero enabled.
- If model weights exceed what bakes reasonably into the worker, attach a network volume for the HF cache so cold starts do not re-download.

Acceptance:

- `curl` against `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1/models` with the API key returns the served model ID.
- The endpoint scales to zero after the test; dashboard shows no idle spend.

## Phase 3: OpenCode provider configuration

Files: new `opencode.json` at the repo root (project config, committed); global config `~/.config/opencode/opencode.json` (not committed, documented); docs/agents/opencode.md.

Tasks:

- Install OpenCode via the curl script; pin the installed version in the doc when done.
- Add a project `opencode.json` defining the RunPod provider:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "runpod": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "RunPod vLLM",
      "options": {
        "baseURL": "https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1",
        "apiKey": "{env:RUNPOD_API_KEY}"
      },
      "models": { "<served-model-id>": { "limit": { "context": 65536 } } }
    }
  }
}
```

Query the endpoint's `/v1/models` to get the exact model ID; the key must match verbatim. Set `limit.context` to the `MAX_MODEL_LEN` the worker is actually launched with, not the model's theoretical maximum.

- Test: `opencode run -m runpod/<model> "list the files in this repo and summarize the Makefile"`. Exercise at least one tool call (file read) and one edit in a scratch directory.
- Measure cold-start latency for the first request after scale-to-zero. If it is unacceptable for interactive use, evaluate: (a) an Active/min-1 worker during work hours, (b) an on-demand pod started and stopped via the runpod CLI or Python SDK for planned sessions, exposed through the pod proxy URL. Record measured cold start, tokens/s, and effective $/hr for each option.
- Document the setup, including how well the chosen model handles agentic tool calling, in docs/agents/opencode.md.

Acceptance:

- `opencode run -m runpod/<model> "<task>"` completes a multi-step tool-using task end to end without provider errors.
- Cold-start and throughput numbers plus a chosen serving mode (serverless flex, warm worker, or pod) are documented with costs in docs/agents/runpod.md.
- The committed `opencode.json` contains no secrets and works on a fresh clone with `RUNPOD_API_KEY` exported.

## Phase 4: Usage tracking

Files: new scripts/usage/opencode_usage.py (or shell), docs/agents/usage-tracking.md, Makefile.

Neither side alone answers "what did this week of agent use cost": OpenCode records tokens but not RunPod dollars (time-billed, not token-billed); RunPod reports dollars but not which sessions consumed them.

Tasks:

- OpenCode side: adopt one community token reporter — evaluate opencode-tokenscope and the opencode-token-monitor plugin; prefer whichever reads the local session storage without patching OpenCode.
- RunPod side: write a small script that queries the GraphQL API for `currentSpendPerHr`, `clientLifetimeSpend`, and spend details, and prints a delta since last invocation (state file in the repo's scratch/ or `~/.local/state`). Python with plain requests.
- Combine: a `make agent-usage` target that prints OpenCode token totals (from the chosen reporter) alongside RunPod lifetime/period spend.
- Re-check upstream monthly: if OpenCode ships native usage reporting (issue #13003), drop the community tool and simplify.

Acceptance:

- `make agent-usage` runs offline-safe (token totals always print; RunPod section degrades gracefully without network or key) and shows: tokens per provider, RunPod spend to date.
- After one real RunPod session, the reported spend delta matches the dashboard to within a billing-granularity rounding error.

## Phase 5: Hardening and automation

Files: Makefile, docs/agents/README.md, .gitignore.

Tasks:

- Makefile targets: agent-usage (Phase 4), optionally runpod-up/runpod-down wrapping pod start/stop via the RunPod SDK if the pod path was chosen in Phase 3.
- .gitignore: ensure OpenCode local state and any usage state files are excluded; verify auth.json (`~/.local/share/opencode/`) never lands in the repo.
- Write docs/agents/README.md tying together opencode.md, runpod.md, and usage-tracking.md, including a fresh-machine bootstrap checklist (install OpenCode, export RUNPOD_API_KEY, verify the provider).
- Update CLAUDE.md's workflow sections to mention the agent tier once it is real.

Acceptance:

- A fresh clone plus the documented bootstrap steps yields a working `opencode run` against the RunPod provider.
- `git grep -iE 'runpod.{0,40}(key|token)'` finds no committed secrets.

## Deferred / watch list

- Local inference containers in this repo as a secondary OpenCode provider tier (zero marginal cost); revisit once the RunPod configuration is stable and daily-driven.
- OpenCode Zen gateway: an alternative cloud tier with per-token pricing and built-in usage dashboards; revisit if RunPod self-hosting proves more expensive or higher-friction than expected for frontier-model access.
- OpenCode server mode (`opencode serve`) for driving the agent programmatically from other tooling in this repo.
- SkyPilot/dstack over RunPod for reproducible pod provisioning if manual deploys become repetitive.
- Native OpenCode usage reporting (anomalyco/opencode#13003) — replaces the Phase 4 community tool when it lands.
