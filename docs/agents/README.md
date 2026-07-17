# Agent SDLC: the escalation ladder

The standing workflow for doing work in this repo with OpenCode or Claude Code. The tools are
kept at parity (see `docs/tooling-parity.md`); this doc is the once-documented ladder plus a
per-tool invocation appendix.

**Sandboxed execution.** Agent sessions run in a sandbox with no access to the host: no host
Docker daemon, no GPU, no running services, no `/mnt/ai-data`. The `make` targets in this
document (`make agent-*`, `make worktree-*`, service and benchmark targets) are host-side
commands the human runs on the workstation. An agent that needs one prepares the exact
command, stops, and waits for the human to report the output; the reported output is the
acceptance evidence.

## The ladder

**Start every piece of work at the lowest rung that could plausibly succeed.** Escalate only
when a rung fails. Tiers are reasoning-effort levels on the single frontier model — there is no
second, cheaper model (see the model policy in `docs/tooling-parity.md`).

| Rung | Use when | How | Effort |
|------|----------|-----|--------|
| 1. Batch | Mechanical & fully specified: pin audits, doc-drift fixes, formatting sweeps | `make agent-run` (`opencode run --auto` / `claude -p`) | low |
| 2. Roadmap task | A defined roadmap item with files/steps/acceptance | `/task <roadmap> <id>`; explorer + skeptic subagents | medium |
| 3. Design / debugging | Open-ended work that resists rungs 1–2 | Interactive frontier session, human in the loop | high |

Rung 1 is non-interactive and auto-approves everything not denied; the Phase 1 permission deny
rules and the Phase 4 `guard.sh` PreToolUse hook are the safety floor and fire regardless of
permission mode. Review the deny rules (`.claude/settings.json`, `opencode.json`) before the
first unattended run.

## Running agents: user guide

Step-by-step recipes for the `make` workflows. Every target below prints `[ERROR]` with a
solution when a required variable or secret is missing.

### Host batch run (rung 1, Claude Code only)

```bash
$ make agent-run PROMPT="Audit docker/ for unpinned image tags" EFFORT=low ALLOW='Read Grep'
```

Runs `claude -p` on the host at the given reasoning effort and appends token usage to
`.agent-state/usage.jsonl`. Only Claude Code runs on the host: `AGENT=opencode` refuses by
design, because OpenCode batch mode (`opencode run --auto`) auto-approves every action not
explicitly denied. The refusal message redirects to the sandbox command below.

### Sandboxed session (either tool)

```bash
$ make agent-sandbox AGENT=claude NAME=mytask                  # interactive
$ make agent-sandbox AGENT=opencode NAME=mytask PROMPT="..."   # batch, contained
```

Creates the worktree `../ai-experiments-<NAME>` automatically if it does not exist, then
starts a contained session (network-restricted container, egress proxy) in that worktree.
Each tool receives exactly one secret:

- `AGENT=claude` requires `CLAUDE_CODE_OAUTH_TOKEN` — generate once with
  `claude setup-token` on the host, then export it.
- `AGENT=opencode` requires an endpoint-scoped `RUNPOD_API_KEY`.

`CLAUDE_THEME` (default `dark`) and `EFFORT` (default `low`) preseed the Claude Code
theme and default reasoning effort each session, because the sandbox home is ephemeral
tmpfs. See the persistent settings section in
[agent-sandbox.md](../sandbox/agent-sandbox.md).

### Worktree lifecycle

```bash
$ make worktree-mytask        # create ../ai-experiments-mytask on branch task/mytask
$ make worktree-ls            # list worktrees
$ make worktree-clean mytask  # remove when merged
```

Creating a worktree provisions its `.venv` automatically with the `dev` and `sandbox`
dependency groups (`uv sync --frozen --only-group dev --only-group sandbox`), so
`make quality` and the non-GPU `make test` suite run without a manual sync. The step is
best-effort: if it cannot resolve (a stale `uv.lock`, no network, or `uv` absent), it
prints a `[WARN]` with the manual `make install` command and worktree creation still
succeeds. Run `make install` inside the worktree when it will do GPU or training work,
which additionally installs torch (cu130) and the CUDA-compiled groups.

Removal accepts the bare name, the `task/<name>` branch, a worktree path, `NAME=<name>`,
or the suffix form `make worktree-clean-<name>` — all resolve to the same worktree. A bare
`make worktree-clean` (no name) errors with usage instead of matching the creation
pattern. Delete the task branch after removal: `git branch -D task/mytask`. If the
branch is kept, recreating the worktree with `make worktree-mytask` reuses it.

### Session export

```bash
$ make agent-export AGENT=claude ID=<session-id>
```

Writes the transcript to `logs/sessions/<id>.json`. The session ID is the CLI's conversation
identifier: for Claude Code, the UUID shown by `claude --resume` (transcript also at
`~/.claude/projects/<slug>/<id>.jsonl`); for OpenCode, the session ID from
`opencode sessions`.

### Usage and maintenance

| Command | Purpose |
|---------|---------|
| `make agent-usage` | Merge token usage (OpenCode ledger) and RunPod spend into one report |
| `make agent-preflight` | `command-sync` + `agent-parity` + supply-chain scan of both tool dirs |
| `make agent-parity` | Diff shared agent bodies; flag forked or broken commands |
| `make command-sync` | Symlink `.claude/commands` into `.opencode/commands`; prunes links whose source was deleted |
| `make agent-versions` | Compare Dockerfile CLI pins against the npm registry (check-only) |

### Troubleshooting

- `make worktree-clean` used to create a worktree named `clean` by matching the
  `worktree-%` creation pattern; a bare invocation now errors with usage. Removal takes a
  name in any form: `make worktree-clean <name>`, `NAME=<name>`, `task/<name>`, a
  worktree path, or `make worktree-clean-<name>`.
- `make test` runs the deterministic fixture tests under `tests/` (see
  `tests/agents/README.md`). Exit code 5 means pytest collected zero tests — a pytest
  convention, not a Makefile failure.
- `[ERROR] CLAUDE_CODE_OAUTH_TOKEN not set`: run `claude setup-token` on the host and
  export the token before `make agent-sandbox AGENT=claude`.
- Dangling symlinks in `.opencode/commands` (source command deleted): run
  `make command-sync` to prune them; `make agent-parity` reports them as broken.

## Worktree parallelism

Independent, non-GPU tasks parallelize cleanly — run one agent session per git worktree:

```bash
make worktree-<name>        # git worktree add ../ai-experiments-<name> -b task/<name>
make worktree-ls
make worktree-clean <name>  # remove when merged
```

What parallelizes vs what serializes is governed by the AGENTS.md operating rule (the source of
truth): **one GPU consumer at a time.** Independent doc/config/code tasks that never touch the
GPU, a running service, or the benchmark suite may run concurrently across worktrees; anything
that does is serialized to exactly one job at a time — no matter how many worktrees are open.

## Per-tool invocation appendix

| Action | OpenCode | Claude Code |
|--------|----------|-------------|
| Rung 1 batch | `opencode run --auto "<prompt>"` | `claude -p "<prompt>" --output-format json --effort low` |
| Rung 2 task | `/task <roadmap> <id>` | `/task <roadmap> <id>` |
| Rung 3 interactive | `opencode` (frontier, high effort) | `claude` (frontier, `--effort high`) |
| Session export | `opencode export <id>` | `claude -p --resume <id> --output-format json` |
| Both via make | `make agent-run AGENT=opencode PROMPT=...` | `make agent-run AGENT=claude PROMPT=... EFFORT=low` |

Session exports land under `logs/sessions/<id>.json` (`make agent-export`); interactive Claude
transcripts are also at `~/.claude/projects/<slug>/<id>.jsonl`. `/postmortem` cites the export.

## Monthly revisit cadence

Once a month, check both tools' changelogs (OpenCode releases, Claude Code releases) for:

- Native usage/cost reporting (may subsume `.opencode/plugin/usage.ts` and `make agent-usage`).
- Skills / plugin / hook API changes (may simplify the adapters in `scripts/agents/` +
  `.opencode/plugin/`).
- Any custom piece here — the usage plugin, the skeptic pattern, the guard scripts — being
  provided upstream. **Delete what upstream now provides**; do not maintain a private copy of a
  feature the tool ships.

Precedent already set: **Claude Code subsumed usage tracking** (built-in `/usage`, so `usage.ts`
is OpenCode-only) and **the explorer role** (built-in `Explore`, so the custom `explorer` agent
exists mainly for context isolation, not capability). Expect more of this and prune accordingly.
