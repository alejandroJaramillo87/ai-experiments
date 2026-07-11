# Agent SDLC: the escalation ladder

The standing workflow for doing work in this repo with OpenCode or Claude Code. The tools are
kept at parity (see `docs/tooling-parity.md`); this doc is the once-documented ladder plus a
per-tool invocation appendix.

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

## Worktree parallelism

Independent, non-GPU tasks parallelize cleanly — run one agent session per git worktree:

```bash
make worktree-<name>        # git worktree add ../ai-experiments-<name> -b task/<name>
make worktree-ls
make worktree-clean-<name>  # remove when merged
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
