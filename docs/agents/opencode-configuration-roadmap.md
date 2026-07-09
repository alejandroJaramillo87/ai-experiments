# OpenCode configuration roadmap

This document is an implementation roadmap for configuring OpenCode itself: rules, model routing, agents, commands, skills, plugins, and permissions. It assumes the integration roadmap (opencode-runpod-roadmap.md) is done or in flight: OpenCode installed, local providers (llama-cpu 8001, llama-gpu 8002, vllm-gpu 8003) configured, and a RunPod serverless vLLM endpoint serving a frontier open-weight model. Findings are current as of July 2026 (OpenCode v1.17.x; extensibility details verified against the anomalyco/opencode docs source tree — re-verify at implementation time, the project ships near-daily releases).

Goal: make OpenCode an effective executor of the planned roadmaps — inference cleanup (tasks 1.1-1.8 plus features), training tier (Phase 0 deliverables D1-D10, then phases 1-4), interpretability tier (time-boxed, falsification-first sprints), and the blog shell cleanup — while keeping cost and cycle time down. Two facts drive every design decision below:

1. RunPod serverless bills wall-clock GPU seconds, not tokens. Every unnecessary agent iteration, oversized context, or verbose reply is money and latency. Cost control therefore means: route work to the cheapest model that can do it, cap iterations, keep contexts small, and never leave workers warm by accident.
2. The planned work is roadmap-shaped: discrete tasks with stated files, steps, and acceptance criteria. The configuration should exploit that structure — commands that take a task and drive it to its acceptance check beat open-ended chat.

Ordering principle: rules and permissions first (they make every later session better and safer), then model routing (the cost lever), then agents, then commands and skills (the throughput levers), then plugins (observability and guardrails), then workflow automation, then security hardening — placed last in numbering but its containment decision (Phase 7) should be read before automating anything in Phase 6, since `--auto` mode is only acceptable inside containment. Each phase is usable on its own; later phases compound.

## Background: the extensibility surface (July 2026)

- Rules: `AGENTS.md` at project root plus global `~/.config/opencode/AGENTS.md`; `CLAUDE.md` is a compat fallback used only when no AGENTS.md exists. The `instructions` config key takes an array of paths/globs merged in addition to AGENTS.md.
- Agents: markdown files in `.opencode/agents/` with frontmatter (`description` required; `mode: primary|subagent|all`, `model`, `temperature`, `prompt`, `permission`, `steps` to cap iterations; extra keys pass through as model options, e.g. `reasoningEffort`). Subagents inherit the caller's model unless pinned. Built-ins: Build, Plan (primary); General, Explore, Scout (subagents); hidden `compaction`/`title`/`summary` system agents.
- Commands: markdown in `.opencode/commands/` with `description`, `agent`, `model`, `subtask` frontmatter; templates support `$ARGUMENTS`, positional `$1..`, shell injection `` !`cmd` ``, and file inclusion `@path`.
- Skills: `SKILL.md` per directory under `.opencode/skills/<name>/`; Claude-compat discovery of `.claude/skills/` also works. Loaded on demand via the skill tool; gated by `permission.skill`.
- Plugins: JS/TS in `.opencode/plugins/` or npm via the `plugin` config key. Hooks include `tool.execute.before/after`, `chat.params`, `permission.ask`, `event` (bus events like `session.idle`), `experimental.session.compacting`, `experimental.provider.small_model`.
- Permissions: `permission` key with `allow/ask/deny` per tool, pattern-matched bash commands, per-agent overrides, `doom_loop` detection. Defaults are permissive.
- Context: `compaction: {auto, prune, reserved}`; `small_model` routes title/summary work to a cheap model.
- Cost knobs: per-agent `steps`, per-model `options.reasoningEffort`/variants, `small_model`, provider `setCacheKey`.

## Phase 1: Rules, permissions, and hygiene

Files: `AGENTS.md` (repo root, committed), `opencode.json` (extend the one from the integration roadmap), `~/.config/opencode/opencode.json` (documented, not committed).

Tasks:

- Create `AGENTS.md` from the current CLAUDE.md, rewritten for an executor rather than a describer: hardware envelope, service ports, the pin-then-fix-then-update ordering rule, "compose and Dockerfiles are the source of truth, docs follow", and the standing instruction that every roadmap task ends by running its stated acceptance check. Keep CLAUDE.md for Claude Code sessions until content converges, then make one a pointer to the other.
- Wire the existing style guides in via `instructions` instead of duplicating them: `"instructions": [".claude/BASH_OUTPUT_STYLE.md", ".claude/PYTHON_STYLE.md", ".claude/COMMENT_STYLE.md", ".claude/DOCUMENTATION_STYLE.md"]`. One source of truth serves both Claude Code and OpenCode.
- Set a permission baseline in `opencode.json`. Defaults are permissive; tighten where this machine is unforgiving:

```json
"permission": {
  "bash": {
    "*": "allow",
    "rm -rf *": "deny",
    "docker system prune*": "ask",
    "docker volume rm*": "ask",
    "git push*": "ask",
    "nvidia-smi -r*": "deny",
    "* /mnt/ai-data/models*": "ask"
  },
  "external_directory": "ask",
  "doom_loop": "deny"
}
```

  The model directory guard matters: `/mnt/ai-data/models` holds hundreds of GB that re-download at real cost and time. `doom_loop: deny` (rather than the default ask) kills the repeated-identical-call failure mode that burns serverless seconds unattended.
- Set `"share": "disabled"` — sessions will contain infrastructure details and there is no collaboration need.
- Enable LSP and formatters only for languages in play: `"lsp": {"python": {}, "typescript": {"disabled": true}}` equivalent per actual key names, plus ruff/black-compatible formatter entries matching pyproject. LSP is off by default; enabling Python buys the agent go-to-definition and diagnostics across the training/interp Python code for free tokens.
- Set compaction: `"compaction": {"auto": true, "prune": true, "reserved": 12000}`. Prune matters here: benchmark and docker build output is huge and worthless after the step that consumed it.

Acceptance:

- A fresh OpenCode session in the repo shows AGENTS.md and the four style guides loaded (visible in the system context), refuses `rm -rf` under `/mnt/ai-data`, and asks before `git push`.
- `share` is disabled; `/share` refuses.

## Phase 2: Model routing for cost

Files: `opencode.json`.

The routing principle: frontier open-weight on RunPod for work that needs judgment across many files (roadmap task execution, design, debugging); local GPU model for everything mechanical (title/summary, doc sweeps, exploration, log triage); local is also the fallback when the endpoint is cold and the task is small. As of July 2026 the frontier open-weight candidates with strong tool calling are the Kimi K2, DeepSeek V4, Qwen3-Coder, and GLM families — re-verify current leaderboards and vLLM tool-parser support at implementation time; the choice was made in the integration roadmap Phase 3 and this document only routes to it.

Tasks:

- Set `"model": "runpod/<frontier-model>"` as default primary and `"small_model": "llama-gpu/<local-model>"`. The small model handles titles, summaries, and compaction — none of that should ever touch a billed endpoint.
- Set realistic `limit.context` per model, and cap `limit.output` where the provider supports it; open-weight coders will happily produce long diffs, and on serverless, output tokens are wall-clock.
- Define model variants for the RunPod model where the family supports reasoning-effort control (e.g. `variants` with `options.reasoningEffort` low/high, or thinking-budget equivalents). Default to the low variant; the high variant is a deliberate per-task escalation via the TUI variant cycle, not the resting state.
- Set provider `options.timeout` on the RunPod provider above the measured cold-start (integration roadmap Phase 3) so the first request of a session does not spuriously fail, and `setCacheKey: true` if the serving stack honors prefix caching (vLLM automatic prefix caching makes repeated system-prompt prefixes nearly free — verify it is enabled in the worker config, since AGENTS.md plus style guides form a large stable prefix resent every request).
- Document the routing table (which class of task uses which model, and why) in `docs/agents/opencode.md`.

Acceptance:

- A session's title and compaction demonstrably run against the local endpoint (visible in llama-gpu logs) while the primary conversation runs against RunPod.
- Switching variants in the TUI changes reasoning effort on the RunPod model without editing config.

## Phase 3: Agents

Files: `.opencode/agents/*.md` (committed).

Keep the built-in Build and Plan as primaries — Plan already gives a read-only mode for cheap roadmap discussion. Add subagents that encode the workstreams' recurring roles, each pinned to the cheapest adequate model so the frontier primary can delegate without multiplying its own cost:

- `explorer.md` — mode: subagent, model: llama-gpu local, permission read-only. Fan-out codebase/doc searching so raw file contents never enter the frontier context; the primary receives conclusions. This is the single biggest context-and-cost saver for the doc-reconciliation tasks (inference roadmap 1.3) and the training-repo layout work.
- `skeptic.md` — mode: subagent, model: the RunPod frontier model (judgment work), `steps` capped low (e.g. 15), temperature 0. Encodes the interp roadmap's rule 4: given a claimed result or finished task, attempt to falsify it — rerun the acceptance check from scratch, look for the way the claim could be wrong, report CONFIRMED or REFUTED with evidence. Invoked at the end of every roadmap task before commit (the `/task` command in Phase 4 does this automatically).
- `bench-runner.md` — mode: subagent, model: local, permission allowing docker/make but denying edits. Runs the benchmark suite or a smoke test and returns the parsed result table only. Benchmarks produce megabytes of output that must not land in the primary context.
- `doc-writer.md` — mode: subagent, model: local (escalate manually if quality disappoints), prompt referencing DOCUMENTATION_STYLE.md. Handles the high-volume, low-judgment writing: doc sweeps, README updates, blog shell boilerplate. Frontier tokens should buy engineering judgment, not markdown formatting.
- Set `permission.task` on Build so it can spawn all of these; consider `hidden: true` on none — @-mention invocation is useful during manual sessions.

Acceptance:

- From a Build session, `@explorer` answers a multi-file question about the repo while llama-gpu logs (not RunPod) show the traffic.
- `@skeptic` given a deliberately broken "completed" task (acceptance check that fails) returns REFUTED.

## Phase 4: Commands and skills

Files: `.opencode/commands/*.md`, `.opencode/skills/*/SKILL.md` (all committed).

Commands encode the SDLC loop the roadmaps imply; skills hold the reference knowledge the commands need but that should not sit in every context.

Commands:

- `/task <roadmap-file> <task-id>` — the workhorse. Template includes the named roadmap section via `@` file inclusion, instructs: restate the task's files/steps/acceptance, implement, run the acceptance check verbatim, then invoke `@skeptic` on the result, then stop for review before commit. Runs on the frontier model. This turns each roadmap item into one bounded, self-verifying session.
- `/audit-docs <path>` — agent: doc-writer, subtask: true. Sweeps a docs subtree against the Dockerfiles/compose truth (inference task 1.3 and its recurrences) and returns a drift list. Cheap model, isolated context.
- `/bench <service>` — agent: bench-runner. Wraps `` !`make status` `` for preconditions, runs the suite, returns the table. Used as the closing step of every performance-relevant task ("every feature task ends with a benchmark artifact").
- `/postmortem` — for interp sprints: template prompts for what was tried, what was learned per unit time, continue-or-pivot with pivot as default, and appends to the research log. Local model; this is structured writing, not reasoning.
- `/pins` — quick audit: shell-injects greps for `:latest`, `:main`, unpinned clones across docker/, reports violations of the pinning rules from inference tasks 1.1/1.2.
- `/cost` — shell-injects the `make agent-usage` output from the integration roadmap Phase 4 so spend is one keystroke away inside a session.

Skills (loaded on demand, so they cost nothing until relevant):

- `benchmarking/SKILL.md` — the exact benchmark protocol: which make targets, warm-up policy, how results are recorded and compared, what counts as a regression. Sourced from docs/optimizations/.
- `docker-debug/SKILL.md` — the container debugging runbook for this stack: healthcheck endpoints per service, common CUDA/driver failure signatures, log locations, safe restart procedure.
- `training-runs/SKILL.md` — once training Phase 0 lands: how to launch/resume runs, checkpoint layout, the memory-fit calculator (D4), the train-and-serve round trip (D7). Grows with the tier.
- `runpod-ops/SKILL.md` — endpoint management: checking worker state, forcing scale-down, reading the spend query, cold-start expectations. The skeptic and primary both use it before/after cloud sessions.
- Place them under `.opencode/skills/`; note that `.claude/skills/` is also discovered, so if Claude Code sessions need the same knowledge, put them there once instead — decide one location, do not duplicate.

Acceptance:

- `/task docs/agents/opencode-runpod-roadmap.md 4` (or any real pending task) drives implement → acceptance-check → skeptic without manual re-prompting.
- Skills appear in the available-skills block but are not loaded into context at session start (verify via context inspection); invoking a task that needs the benchmark protocol pulls the skill in.

## Phase 5: Plugins — guardrails and observability

Files: `.opencode/plugins/*.ts`, `.opencode/package.json`, `opencode.json` (`plugin` key).

Tasks:

- `guard.ts` — `tool.execute.before` hook: block reads of `*.env*` and any write under `/mnt/ai-data/models`; belt-and-suspenders over the Phase 1 permissions (permissions gate tools, the hook also catches bash indirection the pattern matcher misses).
- `usage.ts` — `tool.execute.after`/`event` hooks accumulating per-session token counts by provider, written to the state file the `make agent-usage` script (integration roadmap Phase 4) reads. Prefer adapting an existing community plugin (opencode-token-monitor was the candidate) over writing from scratch; wrap rather than fork.
- `runpod-idle.ts` — on `session.idle`, query the RunPod GraphQL API; if any worker is warm or `currentSpendPerHr > 0` with no active session, emit a toast/notification. This is the "never pay for a forgotten warm worker" guardrail, and it is the plugin with the clearest dollar payoff — build it first.
- Optional, evaluate before building: a `chat.params` hook that drops `reasoningEffort` for messages under a size threshold. Only if the variant-cycling discipline from Phase 2 proves too manual in practice.
- Keep plugins dependency-light (bun installs them at startup; startup time is inner-loop time).

Acceptance:

- Simulated idle with a warm endpoint produces the warning; `make agent-usage` reflects a just-finished session's tokens.
- An attempted `.env` read via a bash indirection (e.g. `cat $(echo .env)`) is blocked by the hook.

## Phase 6: Workflow automation and fast SDLC

Files: Makefile, `docs/agents/opencode.md`, git configuration.

Tasks:

- Parallel sessions via git worktrees: document (and optionally wrap in `make worktree-<name>`) the pattern of one OpenCode session per worktree for independent roadmap tasks — the inference cleanup tasks 1.1-1.4 are mutually independent and parallelize cleanly. Serialize anything touching the GPU or the benchmark suite (the doc's concurrency policies already say one GPU consumer at a time; encode that in AGENTS.md).
- Non-interactive batch mode for mechanical sweeps: `opencode run --auto` invocations for the fully-specified, low-risk tasks (pin audits, doc drift fixes) using the local model, run from a make target. Auto mode auto-approves everything not denied — the Phase 1 deny rules are the safety floor, so review them before first use.
- Session export for the research log: `opencode export <sessionID>` after interp sprints, stored under the interp tier's log directory; the `/postmortem` command references the export.
- Escalation ladder, documented as the standing SDLC in `docs/agents/opencode.md`: (1) mechanical and fully specified → `opencode run --auto`, local model; (2) roadmap task → `/task`, frontier low variant, subagents for exploration and verification; (3) design/debugging that resists (1) and (2) → interactive frontier session, high variant, human in the loop. Start every piece of work at the lowest rung that could plausibly succeed.
- Revisit cadence: monthly, check OpenCode releases for native usage reporting, skills/plugin API changes, and whether any custom piece here (usage plugin, skeptic pattern) has been subsumed upstream; delete what upstream now provides.

Acceptance:

- Two independent roadmap tasks complete in parallel worktrees without merge conflicts or GPU contention.
- A pin-audit run via `opencode run --auto` on the local model completes with zero RunPod spend for that run (verified via the spend delta).

## Phase 7: Security hardening

Files: new `docker/opencode/Dockerfile.opencode` and compose service (or Docker Sandbox usage doc), `opencode.json`, `docs/agents/opencode-security.md`, Makefile.

Threat model findings (July 2026) that shape this phase:

- OpenCode ships no sandbox on Linux (an experimental macOS-only seatbelt PR exists; sandboxing is an open issue upstream), and its permission system has known enforcement gaps — bash side effects can escape `external_directory` restrictions (upstream issues #18396, #24429). Conclusion: the Phase 1 permission rules are UX guardrails, not a security boundary. The boundary must be OS-level isolation.
- The CVE history is real: CVE-2026-22812 (critical, pre-1.0.216) was an unauthenticated localhost HTTP server with permissive CORS allowing drive-by shell execution from any website; CVE-2026-22813 was web-UI XSS escalating to command execution. Both are fixed in current versions, but the pattern — a local HTTP API bridging to a shell — means version currency and server lockdown are standing requirements, not one-time fixes.
- Plugins are the weakest link: npm plugins are auto-installed at startup and run in-process with full user privileges, with no signing, review, or sandboxing. Project-local `opencode.json`, `.opencode/plugins/`, and `.opencode/skills/` load automatically — opening OpenCode inside a cloned third-party repo can execute attacker-controlled plugin code or inject attacker-controlled prompts.
- Prompt injection arrives via webfetch/websearch output, MCP tool results, and untrusted repo files; OpenCode does no output sanitization.

So yes: the answer to "does secure mean running it in a container" is essentially yes — containment first, then configuration, then hygiene.

Tasks — containment (the actual boundary):

- Adopt Docker Sandboxes as the default way to run agent sessions: Docker supports OpenCode as a first-class agent (`docker sandbox create opencode` / `docker sandbox run`), giving each session a microVM with its own kernel, private Docker daemon, and an egress-controlling network proxy. Evaluate it first because it is the strongest off-the-shelf option and this machine already runs Docker.
- Fallback if Docker Sandboxes does not fit the workflow (e.g. GPU or compose access friction): build our own `docker/opencode/` image following the devcontainer pattern — non-root user, OpenCode pinned via npm (`opencode-ai@<version>`), iptables/ipset egress allowlist permitting only the RunPod API, the local service ports, and the package registries; bind-mount only the repo worktree. Never mount `~/.ssh`, `~/.aws`, or `~/.local/share/opencode/auth.json` into it; inject the single scoped key it needs via environment.
- Decide the tier split and document it: interactive sessions in this trusted repo may run on the host (with the config hardening below); anything running `--auto`, anything touching untrusted input (webfetch-heavy research, third-party repo checkouts), and any future `opencode serve` usage runs contained, no exceptions. `--auto` mode auto-approves everything not denied, so it inherits none of the ask-prompts that make host sessions tolerable.

Tasks — configuration hardening (`opencode.json`, best-effort layer):

- `"autoupdate": false` inside the pinned sandbox image; on the host, leave updates on or check deliberately and promptly — given the CVE history, running stale is the greater risk. Subscribe to the repo's GitHub security advisories either way.
- Set `OPENCODE_SERVER_PASSWORD` even for localhost use (blunts CSRF/local-process abuse of the API); never pass `--hostname 0.0.0.0` — the server has no TLS, so any remote exposure goes through an SSH tunnel or reverse proxy, and preferably never leaves the sandbox.
- Tighten Phase 1 permissions with the injection surface in mind: `"webfetch": {"*": "ask", "https://docs.runpod.io/*": "allow", "https://github.com/*": "allow"}` (allowlist grows deliberately); add bash deny patterns for `cat *.env*`, `* ~/.ssh/*`, `* ~/.aws/*` — the default `.env` read denial covers the read tool only, not bash.
- Verify `~/.local/share/opencode/auth.json` (and any MCP token store found beside it) is mode 600; fix and note it in the security doc.
- Plugin policy: exact-version pins in the `plugin` key (`pkg@1.2.3`), read the source of any community plugin before adopting (they are small), prefer vendoring reviewed code into `.opencode/plugins/` over npm references. The Phase 5 plugins we author ourselves are fine by construction.

Tasks — hygiene and process:

- Before opening OpenCode in any repo not authored here (including submodule updates pulled from upstream): inspect `opencode.json`, `.opencode/`, `.claude/skills/`, and `.agents/` for unexpected plugins, commands, or skills. Wrap this in a `make agent-preflight` target (a short script: list those paths, diff against git history) so it actually happens.
- RunPod: replace the Phase 1 account key with scoped keys — one endpoint-scoped key for inference calls (the only one OpenCode sees), a separate read-only key for the usage script's GraphQL spend queries, the account key nowhere on disk.
- Write `docs/agents/opencode-security.md` recording the threat model, the tier split, and a quarterly review checklist: OpenCode advisories since last check, permission-gap issues status (has Linux sandboxing landed upstream?), plugin pin bumps, key rotation.

Acceptance:

- A contained session cannot read `~/.ssh` or reach a non-allowlisted domain (test both explicitly); the host `auth.json` is absent inside it.
- `opencode --version` in the sandbox image matches the pin; the host install is at or above the latest release with security fixes.
- An `--auto` run executes only inside containment (verify the make target refuses to run it on the host).
- `make agent-preflight` flags a planted unexpected `.opencode/plugins/` file in a test checkout.
- RunPod dashboard shows only scoped keys; the account-wide key is deleted.

## Explicitly deferred

- OpenCode server mode (`opencode serve`) driving scheduled unattended work — revisit only after the `--auto` batch pattern proves trustworthy under the permission floor.
- Custom provider auth plugins, remote `.well-known/opencode` org config — single-user setup, no need.
- Self-hosted response-caching gateway (Helicone-style) in front of RunPod — only if the spend reports show significant repeated-prefix cost that vLLM prefix caching is not already absorbing.
- Fine-tuning a local model on this repo's conventions as a doc-writer replacement — a natural training-tier mini-project later, not configuration work now.
